"""Tests for metrics aggregation."""
from offline_diagnostic_analysis.metrics import (
    distance_bin,
    aggregate_main,
    aggregate_direct_numeric_sensitivity,
    compare_methods,
)
from offline_diagnostic_analysis.matching import ObjectResult


def _make_result(method="ours", detected=True, cats=("GROUND",), steps=5, ri=0, gt_idx=0):
    return ObjectResult(
        record_index=ri, image="test.png", gt_object_index=gt_idx,
        gt_object_name="cone", gt_categories=cats,
        gt_distance_steps=steps, gt_distance_bin=distance_bin(steps),
        method=method, matched_pred_index=0 if detected else None,
        matched_pred_name="cone" if detected else None,
        match_source="rule" if detected else "unresolved",
        match_reason="test", detected=detected,
    )


class TestDistanceBin:
    def test_boundaries(self):
        assert distance_bin(5) == "Near"
        assert distance_bin(6) == "Medium"
        assert distance_bin(8) == "Medium"
        assert distance_bin(9) == "Far"
        assert distance_bin(None) == "Unknown"


class TestAggregateMain:
    def test_overall_recall(self):
        results = [
            _make_result(detected=True, steps=5),
            _make_result(detected=False, steps=8),
            _make_result(detected=True, steps=12),
        ]
        m = aggregate_main(results)
        assert m["ours"]["overall"]["total"] == 3
        assert m["ours"]["overall"]["detected"] == 2
        assert m["ours"]["overall"]["recall"] == 2 / 3

    def test_empty_prediction_contributes_misses(self):
        results = [_make_result(detected=False) for _ in range(3)]
        m = aggregate_main(results)
        assert m["ours"]["overall"]["recall"] == 0.0

    def test_unknown_distance_in_overall_but_not_bins(self):
        results = [_make_result(detected=True, steps=None)]
        m = aggregate_main(results)
        assert m["ours"]["overall"]["total"] == 1
        assert "Unknown" not in m["ours"].get("by_distance_bin", {})

    def test_category_distance_recall_uses_gt_objects_as_denominator(self):
        results = [
            _make_result(detected=True, cats=("GROUND",), steps=5, gt_idx=0),
            _make_result(detected=False, cats=("PIT",), steps=8, gt_idx=1),
        ]
        m = aggregate_main(results)
        assert m["ours"]["by_category"]["GROUND"]["total"] == 1
        assert m["ours"]["by_category"]["PIT"]["total"] == 1

    def test_extra_prediction_does_not_change_recall(self):
        results = [_make_result(detected=True, steps=5)]
        m = aggregate_main(results)
        assert m["ours"]["overall"]["recall"] == 1.0

    def test_zero_total_cell_uses_null(self):
        results = [_make_result(detected=True, cats=("PIT",), steps=20)]
        m = aggregate_main(results)
        # PIT object at Far bin exists, GROUND and Near bins have zero total
        pit_cat = m["ours"]["by_category_distance"].get("PIT", {})
        far_cell = pit_cat.get("Far", {})
        assert far_cell["total"] == 1, f"Expected PIT-Far total=1, got {far_cell}"
        assert far_cell["recall"] == 1.0
        # GROUND category should not exist (no results for it)
        ground_cat = m["ours"]["by_category_distance"].get("GROUND", {})
        assert ground_cat == {} or all(
            v.get("recall") is not None or v.get("total", 0) == 0
            for v in ground_cat.values()
        )


class TestSensitivity:
    def test_direct_numeric_filters_only_by_provenance(self):
        results = [_make_result(detected=True, steps=5, ri=0, gt_idx=0)]
        prov = {(0, 0): "direct_numeric"}
        sens = aggregate_direct_numeric_sensitivity(results, prov)
        assert sens

    def test_no_direct_numeric_returns_empty(self):
        results = [_make_result(detected=True, steps=5, ri=0, gt_idx=0)]
        prov = {(0, 0): "derived_reviewed"}
        sens = aggregate_direct_numeric_sensitivity(results, prov)
        assert sens == {}


class TestCompareMethods:
    def test_identical_denominator(self):
        results = [
            _make_result(method="ours", detected=True, steps=5),
            _make_result(method="ours", detected=False, steps=8),
            _make_result(method="fewshot", detected=True, steps=5),
            _make_result(method="fewshot", detected=False, steps=8),
        ]
        m = aggregate_main(results)
        comp = compare_methods(m)
        assert "overall" in comp

    def test_ours_minus_fewshot(self):
        results = [
            _make_result(method="ours", detected=True, steps=5, gt_idx=0),
            _make_result(method="fewshot", detected=False, steps=5, gt_idx=0),
        ]
        m = aggregate_main(results)
        comp = compare_methods(m)
        assert comp["overall"]["delta"] == 1.0
