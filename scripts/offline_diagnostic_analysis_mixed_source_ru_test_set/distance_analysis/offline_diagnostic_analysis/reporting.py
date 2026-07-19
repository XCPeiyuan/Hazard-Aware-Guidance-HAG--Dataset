"""Reporting: tables, failure cases, validation report."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .matching import ObjectResult


def write_category_distance_csv(metrics: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, str]] = []
    for method in sorted(metrics):
        if method == "comparison":
            continue
        m = metrics[method]
        by_cat_bin = m.get("by_category_distance", {})
        for cat in sorted(by_cat_bin):
            for bin_name in ("Near", "Medium", "Far", "Unknown"):
                cell = by_cat_bin[cat].get(bin_name)
                if cell is None:
                    continue
                recall_str = f"{cell['recall']:.3f}" if cell["recall"] is not None else "null"
                rows.append({
                    "method": method,
                    "category": cat,
                    "distance_bin": bin_name,
                    "detected": str(cell["detected"]),
                    "missed": str(cell["missed"]),
                    "total": str(cell["total"]),
                    "recall": recall_str,
                })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "category", "distance_bin", "detected", "missed", "total", "recall"])
        writer.writeheader()
        writer.writerows(rows)


def select_failure_cases(object_results: Sequence[ObjectResult], path: Path) -> None:
    missed = [r for r in object_results if not r.detected]
    # Prioritize PIT/OVERHEAD misses
    pit_overhead = [r for r in missed if "PIT" in r.gt_categories or "OVERHEAD" in r.gt_categories]
    other = [r for r in missed if r not in pit_overhead]

    selected = pit_overhead + other
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "record_index": r.record_index,
                    "image": r.image,
                    "method": r.method,
                    "gt_object_name": r.gt_object_name,
                    "gt_categories": list(r.gt_categories),
                    "gt_distance_steps": r.gt_distance_steps,
                    "gt_distance_bin": r.gt_distance_bin,
                    "match_source": r.match_source,
                }
                for r in selected
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_validation_report(context: dict[str, Any], path: Path) -> None:
    checks = context["checks"]
    final_ready = all(check["passed"] for check in checks.values())
    lines = [
        "# Offline Diagnostic Analysis on the Mixed-source R+U Test Set Distance-Aware Object Recall - Validation Report",
        "",
        f"**Final-ready: {'YES' if final_ready else 'NO'}**",
        "",
        "## Experiment Scope",
        "",
        "- GT source: reviewed test-set JSON (SSI -> Qwen3-14B alignment -> human review)",
        "- Methods: ours (qwen25vl7b), fewshot (qwen25vl7bfs)",
        "- Matching: deterministic name rules + DeepSeek V4 Flash for unresolved names",
        "- Category and distance are never used in match prompts",
        "- Distance bins: Near <= 5, Medium 6-8, Far >= 9. These are empirical post-hoc bins for this analysis.",
        "- Claim boundary: mention recall across GT categories and distances; not predicted-distance accuracy, collision reduction, or closed-loop safety.",
        "",
        "## Input Validation",
        "",
        f"- Records per source: {context.get('total_records', 'N/A')}",
        f"- GT hazard records: {context.get('gt_hazard_records', 'N/A')}",
        f"- GT hazard objects: {context.get('gt_hazard_objects', 'N/A')}",
        "",
        "## Matching Statistics",
        "",
        f"- Rule matches: {context.get('rule_matches', 'N/A')}",
        f"- DeepSeek matches: {context.get('deepseek_matches', 'N/A')}",
        f"- Unresolved: {context.get('unresolved', 'N/A')}",
        f"- Manual overrides: {context.get('overrides', 'N/A')}",
        "",
        "## Verification",
        "",
    ]
    for name, check in checks.items():
        lines.append(f"- {name}: {'YES' if check['passed'] else 'NO'}")
        lines.append(
            "  - Details: "
            + json.dumps(check["details"], ensure_ascii=False, sort_keys=True)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
