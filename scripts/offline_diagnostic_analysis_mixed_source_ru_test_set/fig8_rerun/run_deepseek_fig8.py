"""Rebuild Fig. 8 with name-only matching and actual predicted categories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
OFFLINE_DIAGNOSTIC_DIR = THIS_DIR.parent
DISTANCE_ANALYSIS_DIR = OFFLINE_DIAGNOSTIC_DIR / "distance_analysis"
REVIEWED_DIR = OFFLINE_DIAGNOSTIC_DIR / "distance" / "review"
CONFIG_PATH = OFFLINE_DIAGNOSTIC_DIR / "distance" / "llm_config.json"
SOURCE_CACHE_PATH = DISTANCE_ANALYSIS_DIR / "match_cache.jsonl"
LOCAL_CACHE_PATH = THIS_DIR / "deepseek_match_cache.jsonl"
ORIGINAL_SUMMARY_PATH = OFFLINE_DIAGNOSTIC_DIR / "offline_diagnostic_analysis_mixed_source_ru_test_set_analysis" / "offline_diagnostic_analysis_mixed_source_ru_test_set_confusion_summary.json"

sys.path.insert(0, str(DISTANCE_ANALYSIS_DIR))

from offline_diagnostic_analysis import deepseek_matcher, io, matching, validation as val
from offline_diagnostic_analysis.io import AlignedRecord
from offline_diagnostic_analysis.models import HazardObject, Record
from offline_diagnostic_analysis.rule_matcher import MatchPair


CATEGORY_ORDER = ("GROUND", "PIT", "OVERHEAD", "SAFE")
PLOT_CATEGORY_ORDER = ("SAFE", "OVERHEAD", "PIT", "GROUND")
PLOT_CATEGORY_DISPLAY = {
    "SAFE": "Safe",
    "OVERHEAD": "Upper-body\nHazard",
    "PIT": "Pitfall\nHazard",
    "GROUND": "Common\nObstacle",
}
METHOD_FILES = {
    "ours": "qwen25vl7b_reviewed.json",
    "fewshot": "qwen25vl7bfs_reviewed.json",
}
EXPECTED_RECORDS = 863
EXPECTED_UNIQUE_IMAGES = 694


@dataclass(frozen=True)
class RunPaths:
    output_root: Path
    cache_path: Path


def resolve_run_paths(match_mode: str) -> RunPaths:
    if match_mode == "strict":
        return RunPaths(THIS_DIR, LOCAL_CACHE_PATH)
    if match_mode == "loose":
        root = THIS_DIR / "loose"
        return RunPaths(root, root / "deepseek_match_cache.jsonl")
    raise ValueError("match_mode must be 'strict' or 'loose'")


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_all_records(
    gt: Sequence[Record],
    ours: Sequence[Record],
    fewshot: Sequence[Record],
    *,
    expected_records: int = EXPECTED_RECORDS,
) -> tuple[AlignedRecord, ...]:
    aligned = io.align_records(gt, ours, fewshot)
    if len(aligned) != expected_records:
        raise ValueError(
            f"expected {expected_records} positionally aligned records, got {len(aligned)}"
        )
    return aligned


def select_unique_images(
    aligned: Sequence[AlignedRecord],
) -> tuple[tuple[AlignedRecord, ...], dict[str, int]]:
    """Keep one position per image, preferring a hazardous GT record."""
    selected_by_image: dict[str, AlignedRecord] = {}
    group_sizes: Counter[str] = Counter()
    hazard_preferred_groups: set[str] = set()
    for records in aligned:
        gt_record = records[0]
        group_sizes[gt_record.image] += 1
        existing = selected_by_image.get(gt_record.image)
        if existing is None:
            selected_by_image[gt_record.image] = records
            continue
        if not existing[0].objects and gt_record.objects:
            selected_by_image[gt_record.image] = records
            hazard_preferred_groups.add(gt_record.image)

    selected_indices = {
        records[0].record_index for records in selected_by_image.values()
    }
    selected = tuple(
        records for records in aligned if records[0].record_index in selected_indices
    )
    duplicate_groups = sum(size > 1 for size in group_sizes.values())
    grouped_gt: dict[str, list[Record]] = {}
    for records in aligned:
        grouped_gt.setdefault(records[0].image, []).append(records[0])
    mixed_groups = sum(
        len(records) > 1
        and any(record.objects for record in records)
        and any(not record.objects for record in records)
        for records in grouped_gt.values()
    )
    return selected, {
        "input_records": len(aligned),
        "selected_unique_images": len(selected),
        "removed_duplicate_positions": len(aligned) - len(selected),
        "duplicate_image_groups": duplicate_groups,
        "hazard_preferred_groups": len(hazard_preferred_groups),
        "mixed_safe_hazard_duplicate_groups": mixed_groups,
    }


def validate_one_to_one(
    pairs: Sequence[MatchPair], *, gt_count: int, pred_count: int
) -> None:
    gt_indices = [pair.gt_index for pair in pairs]
    pred_indices = [pair.pred_index for pair in pairs]
    if len(gt_indices) != len(set(gt_indices)) or len(pred_indices) != len(set(pred_indices)):
        raise ValueError("matched pairs must be one-to-one")
    if any(not 0 <= index < gt_count for index in gt_indices):
        raise ValueError("matched GT index out of range")
    if any(not 0 <= index < pred_count for index in pred_indices):
        raise ValueError("matched prediction index out of range")


def _categories(obj: HazardObject) -> tuple[str, ...]:
    categories = tuple(category for category in obj.categories if category in CATEGORY_ORDER[:-1])
    if not categories:
        raise ValueError(f"object {obj.name!r} has no supported hazard category")
    return categories


def _add_fractional_event(
    matrix: np.ndarray, actual_categories: Sequence[str], predicted_categories: Sequence[str]
) -> None:
    weight = 1.0 / (len(actual_categories) * len(predicted_categories))
    for actual in actual_categories:
        for predicted in predicted_categories:
            matrix[CATEGORY_ORDER.index(actual), CATEGORY_ORDER.index(predicted)] += weight


def accumulate_record(
    matrix: np.ndarray,
    gt_objects: Sequence[HazardObject],
    pred_objects: Sequence[HazardObject],
    pairs: Sequence[MatchPair],
) -> dict[str, int | float]:
    """Add one record using one unit of mass per matched/unmatched object event."""
    validate_one_to_one(pairs, gt_count=len(gt_objects), pred_count=len(pred_objects))
    matched_gt = {pair.gt_index for pair in pairs}
    matched_pred = {pair.pred_index for pair in pairs}

    if not gt_objects and not pred_objects:
        matrix[CATEGORY_ORDER.index("SAFE"), CATEGORY_ORDER.index("SAFE")] += 1.0
        return {
            "matched": 0,
            "unmatched_gt": 0,
            "unmatched_pred": 0,
            "both_empty": 1,
            "event_mass": 1.0,
        }

    for pair in pairs:
        _add_fractional_event(
            matrix,
            _categories(gt_objects[pair.gt_index]),
            _categories(pred_objects[pair.pred_index]),
        )
    for index, obj in enumerate(gt_objects):
        if index not in matched_gt:
            _add_fractional_event(matrix, _categories(obj), ("SAFE",))
    for index, obj in enumerate(pred_objects):
        if index not in matched_pred:
            _add_fractional_event(matrix, ("SAFE",), _categories(obj))

    matched = len(pairs)
    unmatched_gt = len(gt_objects) - matched
    unmatched_pred = len(pred_objects) - matched
    event_mass = float(matched + unmatched_gt + unmatched_pred)
    return {
        "matched": matched,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
        "both_empty": 0,
        "event_mass": event_mass,
    }


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    ) * 100.0


def reorder_matrix_for_plot(matrix: np.ndarray) -> np.ndarray:
    indices = [CATEGORY_ORDER.index(category) for category in PLOT_CATEGORY_ORDER]
    return matrix[np.ix_(indices, indices)]


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CATEGORY_ORDER])
        for label, row in zip(CATEGORY_ORDER, matrix, strict=True):
            writer.writerow([label, *(f"{value:.6f}" for value in row)])


def plot_heatmap(matrix: np.ndarray, method: str, path: Path) -> None:
    del method
    display_matrix = reorder_matrix_for_plot(matrix)
    percentages = row_normalize(display_matrix)
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(percentages, cmap=plt.cm.Blues, vmin=0, vmax=100)
    labels = [PLOT_CATEGORY_DISPLAY[category] for category in PLOT_CATEGORY_ORDER]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Ground-truth Category")
    ax.set_title("Hazard Category Confusion Matrix")
    for row in range(len(PLOT_CATEGORY_ORDER)):
        for col in range(len(PLOT_CATEGORY_ORDER)):
            value = percentages[row, col]
            mass = int(display_matrix[row, col])
            color = "white" if value >= 50 else "black"
            ax.text(
                col,
                row,
                f"{value:.1f}%\n({mass})",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Percentage [%]")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _object_payload(obj: HazardObject) -> dict[str, Any]:
    return {"name": obj.name, "categories": list(obj.categories)}


def _pair_payload(
    pair: MatchPair, gt_objects: Sequence[HazardObject], pred_objects: Sequence[HazardObject]
) -> dict[str, Any]:
    return {
        "gt_index": pair.gt_index,
        "gt_name": gt_objects[pair.gt_index].name,
        "gt_categories": list(gt_objects[pair.gt_index].categories),
        "pred_index": pair.pred_index,
        "pred_name": pred_objects[pair.pred_index].name,
        "pred_categories": list(pred_objects[pair.pred_index].categories),
        "source": pair.source,
        "reason": pair.reason,
        "confidence": pair.confidence,
    }


def _decision_payload(decision: deepseek_matcher.DeepSeekDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "status": decision.status,
        "model": decision.model,
        "cache_key": decision.cache_key,
        "from_cache": decision.from_cache,
        "error": decision.error,
    }


def _sum_counters(counters: Iterable[dict[str, int | float]]) -> dict[str, int | float]:
    total: Counter[str] = Counter()
    for counter in counters:
        total.update(counter)
    return dict(total)


def run_method(
    method: str,
    aligned: Sequence[AlignedRecord],
    config: deepseek_matcher.LLMConfig,
    *,
    match_mode: str,
    run_paths: RunPaths,
) -> dict[str, Any]:
    out_dir = run_paths.output_root / method
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((len(CATEGORY_ORDER), len(CATEGORY_ORDER)), dtype=float)
    record_summaries: list[dict[str, int | float]] = []
    details: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    api_failures: list[dict[str, Any]] = []
    deepseek_decisions = 0
    cache_hits = 0
    api_successes = 0
    safe_safe_count = 0
    review_items: list[dict[str, Any]] = []
    detections: dict[tuple[int, int], dict[str, Any]] = {}

    for gt_record, ours_record, fewshot_record in aligned:
        pred_record = ours_record if method == "ours" else fewshot_record
        selected_matcher = partial(
            deepseek_matcher.match_with_deepseek,
            profile=match_mode,
        )
        result = matching.match_record(
            gt_record,
            pred_record,
            method,
            selected_matcher,
            config,
            run_paths.cache_path,
        )
        validate_one_to_one(
            result.matched_pairs,
            gt_count=len(gt_record.objects),
            pred_count=len(pred_record.objects),
        )
        summary = accumulate_record(
            matrix, gt_record.objects, pred_record.objects, result.matched_pairs
        )
        record_summaries.append(summary)
        source_counts.update(pair.source for pair in result.matched_pairs)
        if summary["both_empty"]:
            safe_safe_count += 1

        decision = result.deepseek_decision
        if decision is not None:
            deepseek_decisions += 1
            if decision.from_cache:
                cache_hits += 1
            elif decision.status == "matched":
                api_successes += 1
            else:
                api_failures.append(
                    {
                        "record_index": gt_record.record_index,
                        "image": gt_record.image,
                        "cache_key": decision.cache_key,
                        "error": decision.error,
                    }
                )

        details.append(
            {
                "record_index": gt_record.record_index,
                "image": gt_record.image,
                "gt_objects": [_object_payload(obj) for obj in gt_record.objects],
                "pred_objects": [_object_payload(obj) for obj in pred_record.objects],
                "pairs": [
                    _pair_payload(pair, gt_record.objects, pred_record.objects)
                    for pair in result.matched_pairs
                ],
                "unmatched_gt_indices": list(result.unresolved_gt_indices),
                "unmatched_pred_indices": list(result.unresolved_pred_indices),
                "deepseek_decision": _decision_payload(decision),
                "counting_summary": summary,
            }
        )

        gt_objs = gt_record.objects
        pred_objs = pred_record.objects
        pair_map = {p.gt_index: p for p in result.matched_pairs}

        for pair in result.matched_pairs:
            if pair.source == "deepseek":
                review_items.append(
                    {
                        "record_index": gt_record.record_index,
                        "image": gt_record.image,
                        "type": "deepseek_match",
                        "gt_index": pair.gt_index,
                        "gt_name": gt_objs[pair.gt_index].name,
                        "gt_categories": list(gt_objs[pair.gt_index].categories),
                        "pred_index": pair.pred_index,
                        "pred_name": pred_objs[pair.pred_index].name,
                        "pred_categories": list(pred_objs[pair.pred_index].categories),
                        "source": pair.source,
                        "reason": pair.reason,
                        "confidence": pair.confidence,
                    }
                )

        _elevated = {"PIT", "OVERHEAD"}
        for i, obj in enumerate(gt_objs):
            if set(obj.categories) & _elevated:
                pair = pair_map.get(i)
                review_items.append(
                    {
                        "record_index": gt_record.record_index,
                        "image": gt_record.image,
                        "type": "pit_overhead",
                        "gt_index": i,
                        "gt_name": obj.name,
                        "gt_categories": list(obj.categories),
                        "detected": pair is not None,
                        "match_source": pair.source if pair else "unresolved",
                    }
                )

        if result.unresolved_gt_indices and result.unresolved_pred_indices:
            review_items.append(
                {
                    "record_index": gt_record.record_index,
                    "image": gt_record.image,
                    "type": "competing_candidates",
                    "unresolved_gt": [
                        {
                            "gt_index": i,
                            "gt_name": gt_objs[i].name,
                            "gt_categories": list(gt_objs[i].categories),
                        }
                        for i in result.unresolved_gt_indices
                    ],
                    "unresolved_pred": [
                        {
                            "pred_index": i,
                            "pred_name": pred_objs[i].name,
                            "pred_categories": list(pred_objs[i].categories),
                        }
                        for i in result.unresolved_pred_indices
                    ],
                }
            )

        if decision is not None and decision.status == "unresolved":
            review_items.append(
                {
                    "record_index": gt_record.record_index,
                    "image": gt_record.image,
                    "type": "deepseek_failure",
                    "error": decision.error,
                }
            )

        for i, obj in enumerate(gt_objs):
            pair = pair_map.get(i)
            detections[(gt_record.record_index, i)] = {
                "image": gt_record.image,
                "gt_name": obj.name,
                "gt_categories": list(obj.categories),
                "detected": pair is not None,
                "match_source": pair.source if pair else None,
                "pred_index": pair.pred_index if pair else None,
                "pred_name": pred_objs[pair.pred_index].name if pair else None,
            }

    totals = _sum_counters(record_summaries)
    expected_mass = float(totals["event_mass"])
    matrix_sum = float(matrix.sum())
    if not np.isclose(matrix_sum, expected_mass):
        raise ValueError(f"{method}: matrix sum {matrix_sum} != event mass {expected_mass}")

    percentages = row_normalize(matrix)
    payload_base = {
        "category_order": list(CATEGORY_ORDER),
        "counting_unit": (
            "one unit of object-event mass per matched pair, unmatched GT object, "
            "unmatched predicted object, or selected unique both-empty record; "
            "multi-category events "
            "split that unit uniformly across the category Cartesian product"
        ),
    }
    json_dump(out_dir / "count_matrix.json", {**payload_base, "matrix": matrix.tolist()})
    json_dump(
        out_dir / "row_normalized_percentages.json",
        {**payload_base, "matrix": percentages.tolist()},
    )
    write_matrix_csv(out_dir / "count_matrix.csv", matrix)
    write_matrix_csv(out_dir / "row_normalized_percentages.csv", percentages)
    with (out_dir / "match_details.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n")

    off_diagonal_hazard_mass = float(
        sum(
            matrix[row, col]
            for row in range(3)
            for col in range(3)
            if row != col
        )
    )
    audit = {
        "method": method,
        "match_mode": match_mode,
        "records": len(aligned),
        "gt_objects": sum(len(records[0].objects) for records in aligned),
        "pred_objects": sum(
            len(records[1].objects if method == "ours" else records[2].objects)
            for records in aligned
        ),
        "matching_uses_names_only": True,
        "category_and_distance_used_for_matching": False,
        "one_to_one_validated": True,
        "record_totals": totals,
        "match_source_counts": dict(source_counts),
        "deepseek_decisions": deepseek_decisions,
        "deepseek_cache_hits": cache_hits,
        "deepseek_api_successes": api_successes,
        "deepseek_api_failures": len(api_failures),
        "api_failures": api_failures,
        "matrix_sum": matrix_sum,
        "expected_event_mass": expected_mass,
        "matrix_sum_traceable": bool(np.isclose(matrix_sum, expected_mass)),
        "actual_category_confusion_mass": off_diagonal_hazard_mass,
    }
    json_dump(out_dir / "audit_summary.json", audit)
    plot_heatmap(matrix, method, out_dir / "heatmap.png")
    return {
        "audit": audit,
        "matrix": matrix.tolist(),
        "percentages": percentages.tolist(),
        "safe_safe_count": safe_safe_count,
        "review_items": review_items,
        "detections": detections,
    }


def write_comparison(
    results: dict[str, dict[str, Any]],
    *,
    match_mode: str,
    output_root: Path,
) -> None:
    lines = [
        "# New Fig. 8 vs. known original output",
        "",
        "The known original summary at `offline_diagnostic_analysis_mixed_source_ru_test_set_analysis/offline_diagnostic_analysis_mixed_source_ru_test_set_confusion_summary.json` "
        "contains one matrix in SAFE/OVERHEAD/PIT/GROUND order and labels it `loose`. "
        "The new rerun evaluates both reviewed methods independently with rule-first + "
        f"cached DeepSeek name-only one-to-one `{match_mode}` matching.",
        "",
    ]
    if ORIGINAL_SUMMARY_PATH.exists():
        original = json.loads(ORIGINAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        old_matrix = np.asarray(original["matrix_count"], dtype=float)
        lines.extend(
            [
                f"- Known original matrix total: {old_matrix.sum():.2f}",
                f"- Known original records: {original.get('record_count')}",
                f"- Known original match mode: {original.get('match_mode')}",
            ]
        )
    else:
        lines.append("- No machine-readable original summary was found.")
    for method, result in results.items():
        audit = result["audit"]
        matrix = np.asarray(result["matrix"], dtype=float)
        lines.extend(
            [
                f"- New `{method}` matrix total: {matrix.sum():.2f}",
                f"- New `{method}` actual hazard-category off-diagonal mass: "
                f"{audit['actual_category_confusion_mass']:.2f}",
                f"- New `{method}` matched/rule/DeepSeek: "
                f"{audit['record_totals']['matched']}/"
                f"{audit['match_source_counts'].get('rule', 0)}/"
                f"{audit['match_source_counts'].get('deepseek', 0)}",
            ]
        )
    lines.extend(
        [
            "",
            "The totals are not expected to match the original exactly because the new "
            "run preserves positional duplicates, counts both reviewed methods separately, "
            "uses conservative fractional mass for multi-category objects, and preserves "
            "the actual predicted category for matched objects.",
            "",
        ]
    )
    (output_root / "comparison_original.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_strict_loose_comparison() -> None:
    loose_root = THIS_DIR / "loose"
    if not loose_root.exists():
        return
    lines = [
        "# Strict vs. controlled-loose new Fig. 8",
        "",
        "Both runs use the same 694 unique-image records, names-only one-to-one "
        "matching, actual predicted categories, and SAFE/SAFE counting. The only "
        "difference is the DeepSeek semantic matching profile.",
        "",
        "| Method | Profile | Matched pairs | GROUND recall | PIT recall | OVERHEAD recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    added_counts: dict[str, int] = {}
    for method in METHOD_FILES:
        strict_rows = [
            json.loads(line)
            for line in (THIS_DIR / method / "match_details.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        loose_rows = [
            json.loads(line)
            for line in (loose_root / method / "match_details.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        strict_details = {row["record_index"]: row for row in strict_rows}
        loose_details = {row["record_index"]: row for row in loose_rows}
        added: list[dict[str, Any]] = []
        for record_index, loose_detail in loose_details.items():
            strict_pairs = {
                (pair["gt_index"], pair["pred_index"])
                for pair in strict_details[record_index]["pairs"]
            }
            for pair in loose_detail["pairs"]:
                if (pair["gt_index"], pair["pred_index"]) not in strict_pairs:
                    added.append(
                        {
                            "record_index": record_index,
                            "image": loose_detail["image"],
                            **pair,
                        }
                    )
        with (loose_root / f"added_loose_matches_{method}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for item in added:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        added_counts[method] = len(added)

        for profile, root in (("strict", THIS_DIR), ("loose", loose_root)):
            audit = json.loads(
                (root / method / "audit_summary.json").read_text(encoding="utf-8")
            )
            matrix = np.asarray(
                json.loads(
                    (root / method / "count_matrix.json").read_text(encoding="utf-8")
                )["matrix"],
                dtype=float,
            )
            recalls = [
                0.0 if matrix[row].sum() == 0 else matrix[row, row] / matrix[row].sum()
                for row in range(3)
            ]
            lines.append(
                f"| {method} | {profile} | {audit['record_totals']['matched']} | "
                f"{recalls[0] * 100:.2f}% | {recalls[1] * 100:.2f}% | "
                f"{recalls[2] * 100:.2f}% |"
            )
    lines.extend(
        [
            "",
            *[
                f"- `{method}` loose-only added matches requiring review: {count}."
                for method, count in added_counts.items()
            ],
            "",
            "The controlled-loose result is the selected formal replacement for "
            "the original Fig. 8. The strict result remains a matching-policy "
            "sensitivity baseline, and the two `added_loose_matches_*.jsonl` files "
            "are retained as audit artifacts.",
            "",
        ]
    )
    (loose_root / "comparison_strict_loose.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHOD_FILES), default=list(METHOD_FILES)
    )
    parser.add_argument(
        "--match-mode", choices=deepseek_matcher.MATCHER_PROFILES, default="strict"
    )
    args = parser.parse_args()
    run_paths = resolve_run_paths(args.match_mode)
    run_paths.output_root.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "gt": REVIEWED_DIR / "testset_reviewed.json",
        "ours": REVIEWED_DIR / METHOD_FILES["ours"],
        "fewshot": REVIEWED_DIR / METHOD_FILES["fewshot"],
        "config": CONFIG_PATH,
    }
    hashes_before = {name: file_sha256(path) for name, path in input_paths.items()}

    gt_records = io.load_records(input_paths["gt"])
    ours_records = io.load_records(input_paths["ours"])
    fewshot_records = io.load_records(input_paths["fewshot"])
    aligned_all = align_all_records(gt_records, ours_records, fewshot_records)
    aligned, selection_audit = select_unique_images(aligned_all)
    if len(aligned) != EXPECTED_UNIQUE_IMAGES:
        raise ValueError(
            f"expected {EXPECTED_UNIQUE_IMAGES} selected unique images, got {len(aligned)}"
        )
    config = deepseek_matcher.load_llm_config(CONFIG_PATH)

    if (
        args.match_mode == "strict"
        and not run_paths.cache_path.exists()
        and SOURCE_CACHE_PATH.exists()
    ):
        shutil.copy2(SOURCE_CACHE_PATH, run_paths.cache_path)

    results = {
        method: run_method(
            method,
            aligned,
            config,
            match_mode=args.match_mode,
            run_paths=run_paths,
        )
        for method in args.methods
    }

    disagreements: list[dict[str, Any]] = []
    if "ours" in results and "fewshot" in results:
        ours_det = results["ours"]["detections"]
        fewshot_det = results["fewshot"]["detections"]
        for key, ours_info in ours_det.items():
            fewshot_info = fewshot_det.get(key)
            if fewshot_info is None:
                continue
            if ours_info["detected"] != fewshot_info["detected"]:
                disagreements.append(
                    {
                        "record_index": key[0],
                        "image": ours_info["image"],
                        "type": "detection_disagreement",
                        "gt_index": key[1],
                        "gt_name": ours_info["gt_name"],
                        "gt_categories": ours_info["gt_categories"],
                        "ours_detected": ours_info["detected"],
                        "ours_pred_index": ours_info["pred_index"],
                        "ours_pred_name": ours_info["pred_name"],
                        "fewshot_detected": fewshot_info["detected"],
                        "fewshot_pred_index": fewshot_info["pred_index"],
                        "fewshot_pred_name": fewshot_info["pred_name"],
                    }
                )

    for method in results:
        full_queue = results[method]["review_items"] + disagreements
        queue_path = run_paths.output_root / method / "manual_review_queue.jsonl"
        with queue_path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in full_queue:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    safe_safe_counts = {
        "description": (
            "Selected unique-image records where both GT and prediction have "
            "zero hazard objects; each contributes one SAFE/SAFE matrix event"
        ),
        "total_records": len(aligned),
        "safe_safe_count": {
            method: results[method]["safe_safe_count"]
            for method in results
        },
    }
    json_dump(run_paths.output_root / "safe_safe_counts.json", safe_safe_counts)

    hashes_after = {name: file_sha256(path) for name, path in input_paths.items()}
    hashes_unchanged = hashes_before == hashes_after
    if not hashes_unchanged:
        raise RuntimeError("one or more read-only inputs changed during the run")

    duplicate_positions = len(aligned_all) - len(aligned)
    api_failures = sum(result["audit"]["deepseek_api_failures"] for result in results.values())
    prompt = deepseek_matcher.build_semantic_match_prompt(
        ["alpha"], ["beta"], profile=args.match_mode
    ).lower()
    forbidden = [
        token for token in ("category", "distance", "alert", "ssi")
        if token in prompt
    ]
    prompt_leak = {
        "passed": not forbidden,
        "details": {"forbidden_tokens": forbidden},
    }
    validation = {
        "match_mode": args.match_mode,
        "raw_records_per_source": len(aligned_all),
        "records_per_source": len(aligned),
        "expected_records": EXPECTED_RECORDS,
        "expected_unique_images": EXPECTED_UNIQUE_IMAGES,
        "unique_image_selection": selection_audit,
        "position_and_image_alignment_validated": True,
        "duplicate_record_positions_removed": duplicate_positions,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "input_hashes_unchanged": hashes_unchanged,
        "shared_validation": {
            "no_match_prompt_leak": prompt_leak,
        },
        "safe_safe_counts": safe_safe_counts,
        "cross_method_detection_disagreements": len(disagreements),
        "methods": {
            method: {
                "one_to_one_validated": result["audit"]["one_to_one_validated"],
                "matrix_sum_traceable": result["audit"]["matrix_sum_traceable"],
                "deepseek_api_failures": result["audit"]["deepseek_api_failures"],
                "actual_category_confusion_mass": result["audit"][
                    "actual_category_confusion_mass"
                ],
            }
            for method, result in results.items()
        },
        "all_api_failures_explicit": True,
        "total_api_failures": api_failures,
    }
    json_dump(run_paths.output_root / "validation_summary.json", validation)
    write_comparison(
        results,
        match_mode=args.match_mode,
        output_root=run_paths.output_root,
    )
    if args.match_mode == "loose":
        write_strict_loose_comparison()

    if api_failures:
        raise RuntimeError(
            f"run completed with {api_failures} explicit DeepSeek API failure(s); "
            "see per-method audit_summary.json"
        )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
