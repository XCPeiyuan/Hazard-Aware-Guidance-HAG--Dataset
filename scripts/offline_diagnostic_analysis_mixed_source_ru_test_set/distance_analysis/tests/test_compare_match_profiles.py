import json
from pathlib import Path

from compare_match_profiles import compare_object_results, write_comparison


def _row(method: str, detected: bool, pred_name: str | None) -> dict:
    return {
        "record_index": 7,
        "image": "image.png",
        "gt_object_index": 0,
        "gt_object_name": "fallen branch",
        "gt_categories": ["GROUND"],
        "gt_distance_steps": 4,
        "gt_distance_bin": "Near",
        "method": method,
        "matched_pred_index": 0 if detected else None,
        "matched_pred_name": pred_name,
        "match_source": "deepseek" if detected else "unresolved",
        "match_reason": "same_real_world_obstacle" if detected else "no match found",
        "detected": detected,
    }


def test_compare_object_results_reports_new_detection() -> None:
    changes = compare_object_results(
        [_row("ours", False, None)],
        [_row("ours", True, "tree limb")],
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == "newly_detected_in_loose"
    assert changes[0]["loose_matched_pred_name"] == "tree limb"


def test_write_comparison_creates_audit_files(tmp_path: Path) -> None:
    strict_dir = tmp_path / "strict"
    loose_dir = tmp_path / "loose"
    strict_dir.mkdir()
    loose_dir.mkdir()
    (strict_dir / "object_results.jsonl").write_text(
        json.dumps(_row("ours", False, None)) + "\n", encoding="utf-8"
    )
    (loose_dir / "object_results.jsonl").write_text(
        json.dumps(_row("ours", True, "tree limb")) + "\n", encoding="utf-8"
    )
    metrics = {
        "ours": {
            "overall": {"detected": 0, "missed": 1, "total": 1, "recall": 0.0},
            "by_category": {},
            "by_distance_bin": {},
            "by_category_distance": {},
        }
    }
    (strict_dir / "metrics_main.json").write_text(json.dumps(metrics), encoding="utf-8")
    metrics["ours"]["overall"]["detected"] = 1
    metrics["ours"]["overall"]["missed"] = 0
    metrics["ours"]["overall"]["recall"] = 1.0
    (loose_dir / "metrics_main.json").write_text(json.dumps(metrics), encoding="utf-8")

    write_comparison(strict_dir, loose_dir)

    assert (loose_dir / "strict_loose_comparison.md").exists()
    rows = (loose_dir / "loose_matching_changes.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(rows) == 1
