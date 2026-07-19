"""Offline Diagnostic Analysis on the Mixed-source R+U Test Set distance-aware object recall — CLI entrypoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from offline_diagnostic_analysis import (
    models,
    io,
    provenance,
    matching,
    metrics,
    reporting,
    deepseek_matcher,
    validation,
)


THIS_DIR = Path(__file__).resolve().parent
REVIEWED_DIR = THIS_DIR.parent / "distance" / "review"
DEFAULT_CONFIG = THIS_DIR.parent / "distance" / "llm_config.json"
DEFAULT_OVERRIDES = THIS_DIR / "manual_match_overrides.jsonl"


def resolve_output_dir(match_mode: str, output_dir: Path | None) -> Path:
    if match_mode not in deepseek_matcher.MATCHER_PROFILES:
        raise ValueError("match_mode must be 'strict' or 'loose'")
    if output_dir is not None:
        return output_dir.resolve()
    if match_mode == "strict":
        return THIS_DIR
    return THIS_DIR.parent / "distance_analysis_loose"


def build_profile_matcher(match_mode: str):
    if match_mode not in deepseek_matcher.MATCHER_PROFILES:
        raise ValueError("match_mode must be 'strict' or 'loose'")

    def selected_matcher(*args, **kwargs):
        return deepseek_matcher.match_with_deepseek(
            *args, **kwargs, profile=match_mode
        )

    return selected_matcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Diagnostic Analysis on the Mixed-source R+U Test Set Distance-Aware Object Recall")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report expected workload without API calls")
    parser.add_argument("--force-refresh-cache", action="store_true", help="Clear match cache before running")
    parser.add_argument("--max-records", type=int, default=0, help="Limit to first N GT hazard records (0=all)")
    parser.add_argument("--methods", nargs="*", default=["ours", "fewshot"], help="Methods to evaluate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="LLM config JSON")
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES, help="Manual match overrides JSONL")
    parser.add_argument(
        "--match-mode",
        choices=deepseek_matcher.MATCHER_PROFILES,
        default="strict",
        help="DeepSeek semantic matching profile (default: strict)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; loose defaults to sibling distance_analysis_loose",
    )
    args = parser.parse_args()

    gt_path = REVIEWED_DIR / "testset_reviewed.json"
    ours_path = REVIEWED_DIR / "qwen25vl7b_reviewed.json"
    fewshot_path = REVIEWED_DIR / "qwen25vl7bfs_reviewed.json"
    out_dir = resolve_output_dir(args.match_mode, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_inputs = {
        "gt_reviewed": gt_path,
        "ours_reviewed": ours_path,
        "fewshot_reviewed": fewshot_path,
        "manual_match_overrides": args.overrides,
    }
    manifest_path = THIS_DIR / "baseline" / "input_manifest.json"
    if not manifest_path.exists():
        validation.build_input_manifest(source_inputs, manifest_path)
    pre_hash_check = validation.verify_input_manifest(manifest_path, source_inputs)
    if not pre_hash_check["passed"]:
        raise SystemExit(
            "Source inputs do not match baseline/input_manifest.json; refusing to run."
        )

    config = deepseek_matcher.load_llm_config(args.config)

    # Load and validate
    print("Loading records...")
    gt_records = io.load_records(gt_path)
    ours_records = io.load_records(ours_path)
    fewshot_records = io.load_records(fewshot_path)
    print(f"  GT: {len(gt_records)} records, {sum(len(r.objects) for r in gt_records)} objects")
    print(f"  ours: {len(ours_records)} records, {sum(len(r.objects) for r in ours_records)} objects")
    print(f"  fewshot: {len(fewshot_records)} records, {sum(len(r.objects) for r in fewshot_records)} objects")

    # Align
    print("Aligning records...")
    aligned = io.align_records(gt_records, ours_records, fewshot_records)
    hazard_aligned = io.select_gt_hazard_records(aligned)
    print(f"  GT hazard records: {len(hazard_aligned)}")
    if args.max_records > 0:
        hazard_aligned = hazard_aligned[: args.max_records]
        print(f"  Limited to: {len(hazard_aligned)}")

    # Provenance
    print("Classifying distance provenance...")
    gt_hazard_records = [a[0] for a in hazard_aligned]
    prov_rows = provenance.build_gt_distance_provenance(gt_hazard_records)
    prov_map = {(r["record_index"], r["object_index"]): r["provenance"] for r in prov_rows}
    if not args.dry_run:
        with (out_dir / "gt_distance_provenance.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for row in prov_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    direct = sum(1 for r in prov_rows if r["provenance"] == "direct_numeric")
    derived = sum(1 for r in prov_rows if r["provenance"] == "derived_reviewed")
    unknown = sum(1 for r in prov_rows if r["provenance"] == "unknown")
    print(f"  direct_numeric={direct}, derived_reviewed={derived}, unknown={unknown}")

    # Matching
    print("Matching objects...")
    cache_path = out_dir / "match_cache.jsonl"
    if args.force_refresh_cache and cache_path.exists():
        cache_path.unlink()

    overrides = matching.load_overrides(args.overrides)
    print(f"  overrides loaded: {len(overrides)}")

    if args.dry_run:
        # Count expected DeepSeek workload
        unresolved_count = 0
        for gt_rec, ours_rec, fewshot_rec in hazard_aligned:
            for pred_rec, method in [(ours_rec, "ours"), (fewshot_rec, "fewshot")]:
                if method not in args.methods:
                    continue
                _, unresolved_gt, unresolved_pred = __import__("offline_diagnostic_analysis.rule_matcher", fromlist=["match_by_rules"]).match_by_rules(gt_rec.objects, pred_rec.objects, profile=args.match_mode)
                if unresolved_gt and unresolved_pred:
                    unresolved_count += 1
        # Output path checks
        output_names = [
            "object_results.jsonl", "metrics_main.json", "metrics_sensitivity_direct_numeric.json",
            "category_distance_table.csv", "failure_cases.json", "validation_report.md",
            "manual_review_queue.jsonl", "gt_distance_provenance.jsonl",
        ]
        for name in output_names:
            path = out_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n=== DRY RUN ===")
        print(f"  GT hazard records: {len(hazard_aligned)}")
        print(f"  Records with unresolved matches needing DeepSeek: {unresolved_count}")
        print(f"  Methods: {args.methods}")
        print(f"  Match mode: {args.match_mode}")
        print(f"  Output directory: {out_dir}")
        print(f"  Output paths writable: {all((out_dir / n).parent.exists() for n in output_names)}")
        print(f"  Input SHA-256 baseline verified: {pre_hash_check['passed']}")
        print(f"  All inputs validated. No API calls made.")
        print(f"  Run without --dry-run to execute full pipeline.")
        return

    object_results, unmatched_preds, review_queue = matching.build_object_results(
        hazard_aligned, args.methods, overrides,
        deepseek_matcher=build_profile_matcher(args.match_mode) if not args.dry_run else None,
        config=config if not args.dry_run else None,
        cache_path=cache_path if not args.dry_run else None,
        profile=args.match_mode,
    )
    print(f"  object results: {len(object_results)}")
    print(f"  unmatched predictions: {len(unmatched_preds)}")
    print(f"  review queue: {len(review_queue)}")

    # Write object results
    with (out_dir / "object_results.jsonl").open("w", encoding="utf-8") as f:
        for r in object_results:
            f.write(json.dumps({
                "record_index": r.record_index,
                "image": r.image,
                "gt_object_index": r.gt_object_index,
                "gt_object_name": r.gt_object_name,
                "gt_categories": list(r.gt_categories),
                "gt_distance_steps": r.gt_distance_steps,
                "gt_distance_bin": r.gt_distance_bin,
                "method": r.method,
                "matched_pred_index": r.matched_pred_index,
                "matched_pred_name": r.matched_pred_name,
                "match_source": r.match_source,
                "match_reason": r.match_reason,
                "detected": r.detected,
            }, ensure_ascii=False) + "\n")

    with (out_dir / "manual_review_queue.jsonl").open("w", encoding="utf-8") as f:
        for item in review_queue:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Metrics
    print("Computing metrics...")
    main_metrics = metrics.aggregate_main(object_results)
    with (out_dir / "metrics_main.json").open("w", encoding="utf-8") as f:
        json.dump(main_metrics, f, indent=2, ensure_ascii=False)

    sensitivity = metrics.aggregate_direct_numeric_sensitivity(object_results, prov_map)
    with (out_dir / "metrics_sensitivity_direct_numeric.json").open("w", encoding="utf-8") as f:
        json.dump(sensitivity, f, indent=2, ensure_ascii=False)

    # Reporting
    print("Generating reports...")
    reporting.write_category_distance_csv(main_metrics, out_dir / "category_distance_table.csv")
    reporting.select_failure_cases(object_results, out_dir / "failure_cases.json")

    context = {
        "total_records": len(gt_records),
        "gt_hazard_records": len(hazard_aligned),
        "gt_hazard_objects": sum(len(r.objects) for r in gt_hazard_records),
        "rule_matches": sum(1 for r in object_results if r.match_source == "rule"),
        "deepseek_matches": sum(1 for r in object_results if r.match_source == "deepseek"),
        "unresolved": sum(1 for r in object_results if r.match_source == "unresolved"),
        "overrides": len(overrides),
        "match_mode": args.match_mode,
    }
    deepseek_failures = [
        item for item in review_queue if item["type"] == "deepseek_failure"
    ]
    context["checks"] = {
        "All tests pass": validation.run_test_suite(THIS_DIR),
        "Input hashes unchanged during run": validation.verify_input_manifest(
            manifest_path, source_inputs
        ),
        "All matched pairs one-to-one": validation.validate_one_to_one(
            object_results
        ),
        "No category/distance in match prompts": validation.validate_no_match_prompt_leak(
            profile=args.match_mode
        ),
        "Metrics recomputable from object results": validation.validate_metrics_recomputable(
            object_results, main_metrics
        ),
        "Zero-total cells use null": validation.validate_zero_total_null(
            main_metrics
        ),
        "No unresolved DeepSeek API failures": {
            "passed": not deepseek_failures,
            "details": {"failure_count": len(deepseek_failures)},
        },
    }
    reporting.write_validation_report(context, out_dir / "validation_report.md")

    print("\nDone. Outputs written to:")
    for name in ["metrics_main.json", "metrics_sensitivity_direct_numeric.json",
                  "category_distance_table.csv", "failure_cases.json",
                  "object_results.jsonl", "manual_review_queue.jsonl",
                  "gt_distance_provenance.jsonl", "validation_report.md"]:
        print(f"  {out_dir / name}")
    final_ready = all(check["passed"] for check in context["checks"].values())
    print(f"\nFinal-ready: {'YES' if final_ready else 'NO'}")
    if not final_ready:
        raise SystemExit(2)


# Import helper for dry-run counting
def match_with_deepseek(*args, **kwargs):
    return deepseek_matcher.match_with_deepseek(*args, **kwargs)


if __name__ == "__main__":
    main()
