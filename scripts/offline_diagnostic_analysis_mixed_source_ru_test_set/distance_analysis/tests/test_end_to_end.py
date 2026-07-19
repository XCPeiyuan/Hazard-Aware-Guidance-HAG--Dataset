"""End-to-end tests using tiny fixtures."""
import json
from pathlib import Path

from offline_diagnostic_analysis.io import load_records, align_records, select_gt_hazard_records
from offline_diagnostic_analysis.provenance import build_gt_distance_provenance
from offline_diagnostic_analysis.matching import build_object_results, load_overrides
from offline_diagnostic_analysis.metrics import aggregate_main, aggregate_direct_numeric_sensitivity
from offline_diagnostic_analysis.reporting import write_category_distance_csv, select_failure_cases, write_validation_report


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_full_pipeline_on_tiny_fixtures(tmp_path):
    gt = load_records(FIXTURE_DIR / "tiny_gt.json")
    ours = load_records(FIXTURE_DIR / "tiny_ours.json")
    fewshot = load_records(FIXTURE_DIR / "tiny_fewshot.json")

    aligned = align_records(gt, ours, fewshot)
    hazard = select_gt_hazard_records(aligned)
    assert len(hazard) > 0

    # Provenance
    gt_hazard_recs = [a[0] for a in hazard]
    prov_rows = build_gt_distance_provenance(gt_hazard_recs)
    prov_map = {(r["record_index"], r["object_index"]): r["provenance"] for r in prov_rows}
    assert len(prov_rows) > 0

    # Write provenance
    with (tmp_path / "gt_distance_provenance.jsonl").open("w", encoding="utf-8") as f:
        for row in prov_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Matching
    overrides_path = tmp_path / "overrides.jsonl"
    overrides_path.write_text("", encoding="utf-8")
    overrides = load_overrides(overrides_path)
    assert len(overrides) == 0

    results, unmatched, queue = build_object_results(hazard, ["ours", "fewshot"], overrides)

    # Write review queue
    review_path = tmp_path / "manual_review_queue.jsonl"
    with review_path.open("w", encoding="utf-8") as f:
        for item in queue:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write mock match cache
    cache_path = tmp_path / "match_cache.jsonl"
    cache_path.write_text(
        json.dumps({"cache_key": "test", "model": "deepseek-v4-flash", "status": "matched", "request": {"gt_names": [], "pred_names": []}, "parsed_pairs": [], "raw_response": "{}"}) + "\n",
        encoding="utf-8",
    )

    # Write object results
    with (tmp_path / "object_results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "record_index": r.record_index, "image": r.image,
                "gt_object_index": r.gt_object_index, "gt_object_name": r.gt_object_name,
                "gt_categories": list(r.gt_categories),
                "gt_distance_steps": r.gt_distance_steps,
                "gt_distance_bin": r.gt_distance_bin, "method": r.method,
                "matched_pred_index": r.matched_pred_index,
                "matched_pred_name": r.matched_pred_name,
                "match_source": r.match_source, "match_reason": r.match_reason,
                "detected": r.detected,
            }, ensure_ascii=False) + "\n")

    # Metrics
    main = aggregate_main(results)
    with (tmp_path / "metrics_main.json").open("w", encoding="utf-8") as f:
        json.dump(main, f, indent=2, ensure_ascii=False)

    sens = aggregate_direct_numeric_sensitivity(results, prov_map)
    with (tmp_path / "metrics_sensitivity_direct_numeric.json").open("w", encoding="utf-8") as f:
        json.dump(sens, f, indent=2, ensure_ascii=False)

    # Reports
    write_category_distance_csv(main, tmp_path / "category_distance_table.csv")
    select_failure_cases(results, tmp_path / "failure_cases.json")

    context = {"total_records": 0, "gt_hazard_records": 0, "gt_hazard_objects": 0,
               "rule_matches": 0, "deepseek_matches": 0, "unresolved": 0, "overrides": 0,
               "checks": {
                   "All tests pass": {"passed": True, "details": {"count": 1}},
                   "No unresolved DeepSeek API failures": {"passed": False, "details": {"failure_count": 1}},
               }}
    write_validation_report(context, tmp_path / "validation_report.md")
    report = (tmp_path / "validation_report.md").read_text(encoding="utf-8")
    assert "Final-ready: NO" in report
    assert "PENDING" not in report

    # Verify all required outputs exist
    expected = [
        "object_results.jsonl", "metrics_main.json",
        "metrics_sensitivity_direct_numeric.json", "category_distance_table.csv",
        "failure_cases.json", "validation_report.md",
        "manual_review_queue.jsonl", "match_cache.jsonl",
        "gt_distance_provenance.jsonl",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), f"Missing output: {name}"
