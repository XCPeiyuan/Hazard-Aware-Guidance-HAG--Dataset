import json

from offline_diagnostic_analysis.validation import (
    build_input_manifest,
    validate_metrics_recomputable,
    validate_no_match_prompt_leak,
    validate_one_to_one,
    validate_zero_total_null,
    verify_input_manifest,
)


def test_input_manifest_detects_source_change(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"value":1}', encoding="utf-8")
    manifest_path = tmp_path / "baseline" / "input_manifest.json"

    build_input_manifest({"gt": source}, manifest_path)
    unchanged = verify_input_manifest(manifest_path, {"gt": source})
    source.write_text('{"value":2}', encoding="utf-8")
    changed = verify_input_manifest(manifest_path, {"gt": source})

    assert unchanged["passed"] is True
    assert changed["passed"] is False
    assert changed["details"]["gt"]["status"] == "CHANGED"


def test_validation_checks_are_computed_not_hardcoded():
    object_rows = [
        {"record_index": 0, "method": "ours", "gt_object_index": 0, "matched_pred_index": 0, "detected": True},
        {"record_index": 0, "method": "ours", "gt_object_index": 1, "matched_pred_index": None, "detected": False},
    ]
    metrics = {
        "ours": {
            "overall": {"detected": 1, "missed": 1, "total": 2, "recall": 0.5},
            "by_category": {},
            "by_distance_bin": {},
            "by_category_distance": {"PIT": {"Near": {"detected": 0, "missed": 0, "total": 0, "recall": None}}},
        }
    }

    assert validate_one_to_one(object_rows)["passed"] is True
    assert validate_metrics_recomputable(object_rows, metrics)["passed"] is True
    assert validate_zero_total_null(metrics)["passed"] is True
    assert validate_no_match_prompt_leak()["passed"] is True


def test_one_to_one_validation_rejects_duplicate_prediction_use():
    rows = [
        {"record_index": 0, "method": "ours", "gt_object_index": 0, "matched_pred_index": 0, "detected": True},
        {"record_index": 0, "method": "ours", "gt_object_index": 1, "matched_pred_index": 0, "detected": True},
    ]
    result = validate_one_to_one(rows)
    assert result["passed"] is False
    assert result["details"]["duplicate_pred_pairs"]
