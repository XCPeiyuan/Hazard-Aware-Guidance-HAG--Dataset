"""Recalculate frozen pre-audit structured-field metrics for the Structured-field Evaluation on an Independently Annotated Subset table.

This script reuses the existing object matching stored in
evaluation_results_pre_audit.json. It does not rerun matching or modify GT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


STRUCTURED_FIELD_EVAL_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = STRUCTURED_FIELD_EVAL_DIR / "evaluation_results_pre_audit.json"
OUTPUT_PATH = STRUCTURED_FIELD_EVAL_DIR / "field_f1_recalculation_pre_audit.json"

CATEGORY_LABELS = ("GROUND", "PIT", "OVERHEAD")
DIRECTION_LABELS = (
    "right",
    "right_ahead",
    "slightly_right_ahead",
    "ahead",
    "slightly_left_ahead",
    "left_ahead",
    "left",
)


def visual_prompt_distance_bin(value: object) -> str | None:
    """Manuscript visual-prompt thresholds: <10, 10-15, >15 steps."""
    if not isinstance(value, (int, float)):
        return None
    if value < 10:
        return "near"
    if value <= 15:
        return "medium"
    return "far"


def p007_current_distance_bin(value: object) -> str | None:
    """Current Offline Diagnostic Analysis on the Mixed-source R+U Test Set post-hoc thresholds: <=5, 6-8, >=9 steps."""
    if not isinstance(value, (int, float)):
        return None
    if value <= 5:
        return "near"
    if value <= 8:
        return "medium"
    return "far"


def valid_label(value: object, labels: tuple[str, ...]) -> str | None:
    return value if isinstance(value, str) and value in labels else None


def valid_distance_bin(
    value: object, bin_fn: Callable[[object], str | None]
) -> str | None:
    return bin_fn(value)


def new_counts(labels: tuple[str, ...]) -> dict:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "per_label": {
            label: {"tp": 0, "fp": 0, "fn": 0}
            for label in labels
        },
    }


def add_match(
    counts: dict,
    gt_label: str | None,
    pred_label: str | None,
    role: str,
) -> None:
    if gt_label is None:
        return
    if gt_label == pred_label:
        counts["tp"] += 1
        counts["per_label"][gt_label]["tp"] += 1
        return
    if role == "required":
        counts["fn"] += 1
        counts["per_label"][gt_label]["fn"] += 1
    if pred_label is not None:
        counts["fp"] += 1
        counts["per_label"][pred_label]["fp"] += 1


def add_unmatched_gt(counts: dict, gt_label: str | None, role: str) -> None:
    if gt_label is not None and role == "required":
        counts["fn"] += 1
        counts["per_label"][gt_label]["fn"] += 1


def add_unmatched_pred(counts: dict, pred_label: str | None) -> None:
    if pred_label is not None:
        counts["fp"] += 1
        counts["per_label"][pred_label]["fp"] += 1


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def finish_counts(counts: dict) -> dict:
    per_label_f1 = {
        label: f1(**label_counts)
        for label, label_counts in counts["per_label"].items()
    }
    supported = [
        label
        for label, label_counts in counts["per_label"].items()
        if label_counts["tp"] + label_counts["fn"] > 0
    ]
    counts["micro_f1"] = f1(counts["tp"], counts["fp"], counts["fn"])
    counts["macro_f1_supported_gt"] = (
        sum(per_label_f1[label] for label in supported) / len(supported)
        if supported
        else 0.0
    )
    counts["per_label_f1"] = per_label_f1
    return counts


def field_metrics(
    per_image: dict,
    field: str,
    labels: tuple[str, ...],
    transform: Callable[[object], str | None],
) -> dict:
    counts = new_counts(labels)
    for image_result in per_image.values():
        if "matched_pairs" not in image_result:
            continue
        for pair in image_result["matched_pairs"]:
            add_match(
                counts,
                transform(pair["gt"].get(field)),
                transform(pair["pred"].get(field)),
                pair["gt"].get("evaluation_role", ""),
            )
        for _, gt in image_result["unmatched_gt"]:
            add_unmatched_gt(
                counts,
                transform(gt.get(field)),
                gt.get("evaluation_role", ""),
            )
        for _, pred in image_result["unmatched_pred"]:
            add_unmatched_pred(counts, transform(pred.get(field)))
    return finish_counts(counts)


def hazard_metrics(per_image: dict) -> dict:
    tp = 0
    fp = 0
    fn = 0
    for image_result in per_image.values():
        if "matched_pairs" not in image_result:
            continue
        tp += len(image_result["matched_pairs"])
        fp += len(image_result["unmatched_pred"])
        fn += sum(
            1
            for _, gt in image_result["unmatched_gt"]
            if gt.get("evaluation_role") == "required"
        )
    return {"tp": tp, "fp": fp, "fn": fn, "micro_f1": f1(tp, fp, fn)}


def main() -> None:
    source = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    output = {
        "source": INPUT_PATH.name,
        "definitions": {
            "reporting_status": (
                "Main table reports hazard/category F1, distance MAE, and "
                "direction exact/adjacent. Distance-bin and direction-bin F1 "
                "are retained for audit only per advisor decision."
            ),
            "hazard_f1": (
                "Role-aware name-compatible object detection F1: matched "
                "required/optional GT is TP; unmatched prediction is FP; "
                "only unmatched required GT is FN."
            ),
            "field_micro_f1": (
                "Correct matched field: TP; wrong matched field: FP+FN; "
                "missing matched field: FN; unmatched prediction: FP. "
                "FN applies only to required GT; unmatched optional GT is "
                "not penalized."
            ),
            "direction_exact": "Exact / (exact + adjacent + mirror + mismatch).",
            "direction_adjacent": (
                "(exact + adjacent) / (exact + adjacent + mirror + mismatch)."
            ),
        },
        "methods": {},
    }

    for method, result in source["results"].items():
        aggregate = result["aggregate"]
        per_image = result["per_image"]
        direction_denominator = (
            aggregate["direction_exact"]
            + aggregate["direction_adjacent"]
            + aggregate["direction_mirror"]
            + aggregate["direction_mismatch"]
        )
        expected_exact = aggregate["direction_exact"] / direction_denominator
        expected_adjacent = (
            aggregate["direction_exact"] + aggregate["direction_adjacent"]
        ) / direction_denominator
        assert abs(expected_exact - aggregate["direction_accuracy"]) < 1e-12
        assert (
            abs(expected_adjacent - aggregate["direction_adjacent_accuracy"])
            < 1e-12
        )
        role_aware_hazard = hazard_metrics(per_image)
        method_output = {
            "hazard": role_aware_hazard,
            "legacy_hazard_f1_all_active_gt_denominator": aggregate["f1"],
            "category": field_metrics(
                per_image,
                "hazard_category",
                CATEGORY_LABELS,
                lambda value: valid_label(value, CATEGORY_LABELS),
            ),
            "direction_bin": field_metrics(
                per_image,
                "direction_bin",
                DIRECTION_LABELS,
                lambda value: valid_label(value, DIRECTION_LABELS),
            ),
            "distance_bin_visual_prompt": field_metrics(
                per_image,
                "distance_steps",
                ("near", "medium", "far"),
                lambda value: valid_distance_bin(value, visual_prompt_distance_bin),
            ),
            "distance_bin_p007_current": field_metrics(
                per_image,
                "distance_steps",
                ("near", "medium", "far"),
                lambda value: valid_distance_bin(value, p007_current_distance_bin),
            ),
            "distance_mae": aggregate["distance_mae"],
            "direction_exact": aggregate["direction_accuracy"],
            "direction_adjacent": aggregate["direction_adjacent_accuracy"],
            "direction_counts": {
                "exact": aggregate["direction_exact"],
                "adjacent_only": aggregate["direction_adjacent"],
                "mirror": aggregate["direction_mirror"],
                "mismatch": aggregate["direction_mismatch"],
                "unavailable": aggregate["direction_unavailable"],
            },
        }
        output["methods"][method] = method_output

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Main table (distance-bin and direction-bin F1 are audit-only)")
    print("Method | Hazard F1 | Category F1 | Dist MAE | "
          "Dir Adjacent | Dir Exact")
    for method, values in output["methods"].items():
        print(
            f"{method} | {values['hazard']['micro_f1']:.3f} | "
            f"{values['category']['micro_f1']:.3f} | "
            f"{values['distance_mae']:.2f} | "
            f"{values['direction_adjacent']:.3f} | "
            f"{values['direction_exact']:.3f}"
        )


if __name__ == "__main__":
    main()
