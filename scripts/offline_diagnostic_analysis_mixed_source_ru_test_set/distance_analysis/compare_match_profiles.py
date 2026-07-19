"""Compare strict vs loose match profiles: diff object_results.jsonl between the two modes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_object_results(
    strict_results: list[dict],
    loose_results: list[dict],
) -> list[dict]:
    """Return list of change entries between strict and loose object results."""
    strict_map: dict[tuple[int, str, int], dict] = {}
    loose_map: dict[tuple[int, str, int], dict] = {}

    for row in strict_results:
        key = (row["record_index"], row["method"], row["gt_object_index"])
        strict_map[key] = row
    for row in loose_results:
        key = (row["record_index"], row["method"], row["gt_object_index"])
        loose_map[key] = row

    keys = sorted(set(strict_map) | set(loose_map))
    changes: list[dict] = []

    for key in keys:
        strict_row = strict_map.get(key)
        loose_row = loose_map.get(key)
        if strict_row is None or loose_row is None:
            continue

        strict_detected = strict_row["detected"]
        loose_detected = loose_row["detected"]

        if strict_detected == loose_detected:
            if strict_detected and strict_row.get("matched_pred_index") != loose_row.get("matched_pred_index"):
                changes.append({
                    "record_index": key[0],
                    "method": key[1],
                    "gt_object_index": key[2],
                    "gt_object_name": strict_row["gt_object_name"],
                    "change_type": "changed_prediction_pair",
                    "strict_matched_pred_index": strict_row.get("matched_pred_index"),
                    "strict_matched_pred_name": strict_row.get("matched_pred_name"),
                    "loose_matched_pred_index": loose_row.get("matched_pred_index"),
                    "loose_matched_pred_name": loose_row.get("matched_pred_name"),
                })
        elif not strict_detected and loose_detected:
            changes.append({
                "record_index": key[0],
                "method": key[1],
                "gt_object_index": key[2],
                "gt_object_name": loose_row["gt_object_name"],
                "change_type": "newly_detected_in_loose",
                "loose_matched_pred_index": loose_row.get("matched_pred_index"),
                "loose_matched_pred_name": loose_row.get("matched_pred_name"),
                "loose_match_source": loose_row.get("match_source"),
                "loose_match_reason": loose_row.get("match_reason"),
            })
        elif strict_detected and not loose_detected:
            changes.append({
                "record_index": key[0],
                "method": key[1],
                "gt_object_index": key[2],
                "gt_object_name": strict_row["gt_object_name"],
                "change_type": "lost_detection_in_loose",
                "strict_matched_pred_index": strict_row.get("matched_pred_index"),
                "strict_matched_pred_name": strict_row.get("matched_pred_name"),
                "strict_match_source": strict_row.get("match_source"),
                "strict_match_reason": strict_row.get("match_reason"),
            })

    return sorted(
        changes,
        key=lambda c: (c["change_type"], c["record_index"], c["method"], c["gt_object_index"]),
    )


def write_comparison(
    strict_dir: Path,
    loose_dir: Path,
) -> None:
    """Compare object_results.jsonl and metrics from strict_dir vs loose_dir.

    Writes three files to loose_dir:
        - loose_matching_changes.jsonl
        - strict_loose_comparison.json
        - strict_loose_comparison.md
    """
    strict_results_path = strict_dir / "object_results.jsonl"
    loose_results_path = loose_dir / "object_results.jsonl"

    strict_results: list[dict] = []
    if strict_results_path.exists():
        for line in strict_results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                strict_results.append(json.loads(line))

    loose_results: list[dict] = []
    if loose_results_path.exists():
        for line in loose_results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                loose_results.append(json.loads(line))

    changes = compare_object_results(strict_results, loose_results)

    with (loose_dir / "loose_matching_changes.jsonl").open("w", encoding="utf-8") as f:
        for change in changes:
            f.write(json.dumps(change, ensure_ascii=False) + "\n")

    newly_detected = [c for c in changes if c["change_type"] == "newly_detected_in_loose"]
    lost_detection = [c for c in changes if c["change_type"] == "lost_detection_in_loose"]
    changed_pair = [c for c in changes if c["change_type"] == "changed_prediction_pair"]

    strict_metrics: dict[str, Any] = {}
    strict_metrics_path = strict_dir / "metrics_main.json"
    if strict_metrics_path.exists():
        strict_metrics = json.loads(strict_metrics_path.read_text(encoding="utf-8"))

    loose_metrics: dict[str, Any] = {}
    loose_metrics_path = loose_dir / "metrics_main.json"
    if loose_metrics_path.exists():
        loose_metrics = json.loads(loose_metrics_path.read_text(encoding="utf-8"))

    comparison: dict[str, Any] = {
        "newly_detected_in_loose_count": len(newly_detected),
        "lost_detection_in_loose_count": len(lost_detection),
        "changed_prediction_pair_count": len(changed_pair),
        "total_changes": len(changes),
        "strict_metrics_summary": {},
        "loose_metrics_summary": {},
        "recall_delta": {},
    }

    for method in ("ours", "fewshot"):
        strict_overall = strict_metrics.get(method, {}).get("overall", {})
        loose_overall = loose_metrics.get(method, {}).get("overall", {})
        comparison["strict_metrics_summary"][method] = strict_overall
        comparison["loose_metrics_summary"][method] = loose_overall
        strict_recall = strict_overall.get("recall")
        loose_recall = loose_overall.get("recall")
        if strict_recall is not None and loose_recall is not None:
            comparison["recall_delta"][method] = loose_recall - strict_recall

    with (loose_dir / "strict_loose_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    md_lines: list[str] = [
        "# Strict vs Loose Match Profile Comparison",
        "",
        "## Summary",
        f"- Total changes: {len(changes)}",
        f"- Newly detected in loose: {len(newly_detected)}",
        f"- Lost detection in loose: {len(lost_detection)}",
        f"- Changed prediction pair: {len(changed_pair)}",
        "",
        "## Recall Comparison",
        "",
        "| Method | Strict Recall | Loose Recall | Delta |",
        "|--------|--------------|-------------|-------|",
    ]
    for method in ("ours", "fewshot"):
        strict_recall = strict_metrics.get(method, {}).get("overall", {}).get("recall")
        loose_recall = loose_metrics.get(method, {}).get("overall", {}).get("recall")
        sr_str = f"{strict_recall:.4f}" if isinstance(strict_recall, (int, float)) else "N/A"
        lr_str = f"{loose_recall:.4f}" if isinstance(loose_recall, (int, float)) else "N/A"
        if isinstance(strict_recall, (int, float)) and isinstance(loose_recall, (int, float)):
            delta = loose_recall - strict_recall
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"
        md_lines.append(f"| {method} | {sr_str} | {lr_str} | {delta_str} |")

    if newly_detected:
        md_lines.append("")
        md_lines.append("## Newly Detected in Loose")
        md_lines.append("")
        md_lines.append("| Record | Method | GT Index | GT Name | Loose Pred | Loose Source | Loose Reason |")
        md_lines.append("|--------|--------|----------|---------|------------|-------------|-------------|")
        for c in newly_detected[:50]:
            md_lines.append(
                f"| {c['record_index']} | {c['method']} | {c['gt_object_index']} | "
                f"{c['gt_object_name']} | {c.get('loose_matched_pred_name', 'N/A')} | "
                f"{c.get('loose_match_source', 'N/A')} | {c.get('loose_match_reason', 'N/A')} |"
            )
        if len(newly_detected) > 50:
            md_lines.append("")
            md_lines.append(f"...and {len(newly_detected) - 50} more.")

    if lost_detection:
        md_lines.append("")
        md_lines.append("## Lost Detection in Loose")
        md_lines.append("")
        md_lines.append("| Record | Method | GT Index | GT Name | Strict Pred | Strict Source | Strict Reason |")
        md_lines.append("|--------|--------|----------|---------|-------------|--------------|--------------|")
        for c in lost_detection[:50]:
            md_lines.append(
                f"| {c['record_index']} | {c['method']} | {c['gt_object_index']} | "
                f"{c['gt_object_name']} | {c.get('strict_matched_pred_name', 'N/A')} | "
                f"{c.get('strict_match_source', 'N/A')} | {c.get('strict_match_reason', 'N/A')} |"
            )
        if len(lost_detection) > 50:
            md_lines.append("")
            md_lines.append(f"...and {len(lost_detection) - 50} more.")

    if changed_pair:
        md_lines.append("")
        md_lines.append("## Changed Prediction Pair (Detected in Both)")
        md_lines.append("")
        md_lines.append("| Record | Method | GT Index | GT Name | Strict Pred | Loose Pred |")
        md_lines.append("|--------|--------|----------|---------|-------------|-----------|")
        for c in changed_pair[:50]:
            md_lines.append(
                f"| {c['record_index']} | {c['method']} | {c['gt_object_index']} | "
                f"{c['gt_object_name']} | {c.get('strict_matched_pred_name', 'N/A')} | "
                f"{c.get('loose_matched_pred_name', 'N/A')} |"
            )
        if len(changed_pair) > 50:
            md_lines.append("")
            md_lines.append(f"...and {len(changed_pair) - 50} more.")

    md_lines.append("")

    with (loose_dir / "strict_loose_comparison.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


if __name__ == "__main__":
    import argparse

    THIS_DIR = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare strict vs loose match profiles")
    parser.add_argument(
        "--strict-dir",
        type=Path,
        default=THIS_DIR,
        help="Directory containing strict mode outputs",
    )
    parser.add_argument(
        "--loose-dir",
        type=Path,
        default=THIS_DIR.parent / "distance_analysis_loose",
        help="Directory containing loose mode outputs",
    )
    args = parser.parse_args()
    write_comparison(args.strict_dir, args.loose_dir)
    print(f"Comparison written to {args.loose_dir}")
