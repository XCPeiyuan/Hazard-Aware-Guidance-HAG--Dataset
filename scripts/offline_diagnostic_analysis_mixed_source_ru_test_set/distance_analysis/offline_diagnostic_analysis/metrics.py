"""Main and sensitivity metrics aggregation for distance-aware object recall."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .matching import ObjectResult


def distance_bin(distance_steps: int | None) -> str:
    if distance_steps is None:
        return "Unknown"
    if distance_steps <= 5:
        return "Near"
    if distance_steps <= 8:
        return "Medium"
    return "Far"


def _recall_cell(detected: int, missed: int, total: int) -> dict[str, Any]:
    if total == 0:
        return {"detected": 0, "missed": 0, "total": 0, "recall": None}
    return {"detected": detected, "missed": missed, "total": total, "recall": detected / total}


def aggregate_main(
    object_results: Sequence[ObjectResult],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for method in sorted({r.method for r in object_results}):
        results = [r for r in object_results if r.method == method]
        total_gt = len(results)
        detected = sum(1 for r in results if r.detected)
        missed = total_gt - detected

        method_metrics: dict[str, Any] = {
            "overall": _recall_cell(detected, missed, total_gt),
            "by_category": {},
            "by_distance_bin": {},
            "by_category_distance": {},
        }

        # by_category
        cat_results: dict[str, dict[str, int]] = defaultdict(lambda: {"detected": 0, "missed": 0})
        for r in results:
            for cat in r.gt_categories:
                cat_results[cat]["detected" if r.detected else "missed"] += 1
        for cat, counts in sorted(cat_results.items()):
            total = counts["detected"] + counts["missed"]
            method_metrics["by_category"][cat] = _recall_cell(counts["detected"], counts["missed"], total)

        # by_distance_bin (exclude Unknown)
        bin_results: dict[str, dict[str, int]] = defaultdict(lambda: {"detected": 0, "missed": 0})
        for r in results:
            b = distance_bin(r.gt_distance_steps)
            if b != "Unknown":
                bin_results[b]["detected" if r.detected else "missed"] += 1
        for b in ("Near", "Medium", "Far"):
            if b in bin_results:
                c = bin_results[b]
                total = c["detected"] + c["missed"]
                method_metrics["by_distance_bin"][b] = _recall_cell(c["detected"], c["missed"], total)

        # by_category_distance
        cat_bin_map: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"detected": 0, "missed": 0})
        for r in results:
            b = distance_bin(r.gt_distance_steps)
            for cat in r.gt_categories:
                cat_bin_map[(cat, b)]["detected" if r.detected else "missed"] += 1
        for (cat, b), counts in sorted(cat_bin_map.items()):
            total = counts["detected"] + counts["missed"]
            if cat not in method_metrics["by_category_distance"]:
                method_metrics["by_category_distance"][cat] = {}
            method_metrics["by_category_distance"][cat][b] = _recall_cell(counts["detected"], counts["missed"], total)

        metrics[method] = method_metrics

    # comparison (ours - fewshot)
    if "ours" in metrics and "fewshot" in metrics:
        comparison = compare_methods(metrics, "ours", "fewshot")
        metrics["comparison"] = comparison

    return metrics


def aggregate_direct_numeric_sensitivity(
    object_results: Sequence[ObjectResult],
    provenance_map: dict[tuple[int, int], str],
) -> dict[str, Any]:
    direct_results = [
        r for r in object_results
        if provenance_map.get((r.record_index, r.gt_object_index)) == "direct_numeric"
    ]
    if not direct_results:
        return {}
    return aggregate_main(direct_results)


def compare_methods(
    metrics: dict[str, Any],
    ours_label: str = "ours",
    baseline_label: str = "fewshot",
) -> dict[str, Any]:
    ours = metrics.get(ours_label, {})
    base = metrics.get(baseline_label, {})
    comparison: dict[str, Any] = {}

    ours_overall = ours.get("overall", {}).get("recall")
    base_overall = base.get("overall", {}).get("recall")
    if ours_overall is not None and base_overall is not None:
        comparison["overall"] = {
            "ours": ours_overall,
            "fewshot": base_overall,
            "delta": ours_overall - base_overall,
        }

    ours_cat = ours.get("by_category", {})
    base_cat = base.get("by_category", {})
    comparison["by_category"] = {}
    for cat in sorted(set(ours_cat) | set(base_cat)):
        o_r = ours_cat.get(cat, {}).get("recall")
        f_r = base_cat.get(cat, {}).get("recall")
        if o_r is not None and f_r is not None:
            comparison["by_category"][cat] = {"ours": o_r, "fewshot": f_r, "delta": o_r - f_r}

    ours_bin = ours.get("by_distance_bin", {})
    base_bin = base.get("by_distance_bin", {})
    comparison["by_distance_bin"] = {}
    for b in ("Near", "Medium", "Far"):
        o_r = ours_bin.get(b, {}).get("recall")
        f_r = base_bin.get(b, {}).get("recall")
        if o_r is not None and f_r is not None:
            comparison["by_distance_bin"][b] = {"ours": o_r, "fewshot": f_r, "delta": o_r - f_r}

    return comparison
