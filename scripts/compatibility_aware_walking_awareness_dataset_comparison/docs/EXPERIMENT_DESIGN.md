# WalkVLM-New Experimental Design

## Primary Baseline: Image-Only to Alter

- Input: image plus a fixed image-only navigation prompt.
- Target: source WAD `alter` field.
- Purpose: the fairest comparison with an image-only R+U model.
- Test input: the same image-only prompt.

`alter` is the released WAD walking-reminder field name and is not renamed to `alert`.

## Auxiliary Variant: Image-Only to Full JSON

- Input: image plus a fixed image-only structured-annotation prompt.
- Target fields: `weather_condition`, `area_type`, `danger_level`, `traffic_flow_rating`, `summary`, and `alter`.
- Evaluation: extract only `alter` and use the existing test/evaluation pipeline.

This tests whether auxiliary WAD annotations improve learning. It changes the output objective and is not a clean data-only comparison with R+U. It is also not a reproduction of the original WalkVLM CoT pipeline. The source `summary` is supervised output, never input.

## Dataset Controls

- Source: `../WalkVLM/wad_dataset_v2/train_v2.json` (JSONL despite `.json`).
- Source records: 10,146 total; 8,581 `alter` records and 1,565 QA records.
- Include only records with non-empty `alter` and `frame_path`; exclude all QA records.
- Both variants use the exact same selected images.
- Fixed seed: 42; sample count must equal R+U.
- Verify each selected `frame_path/0.jpg` exists.
- Never include `summary` in an input prompt.

## Test Protocol

The existing validated WAD test JSON is copied into `test/` without reconversion. Both variants use image-only prompts. Convert full-JSON outputs with `extract_alter_from_full_json_output.py` before evaluation.

The original our-test image JSON was not found in the WalkVLM workspace and must be copied from the main experiment unchanged; it must not be reconstructed from basename-only result files.

## Reviewer-Facing Interpretation

- Compare R+U against alter-only as the primary fair baseline.
- Report full-JSON only as an auxiliary multitask ablation.
- The earlier summary-conditioned WAD run is compatibility analysis, not the primary baseline.
