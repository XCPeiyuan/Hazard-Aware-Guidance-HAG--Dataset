# Experiment Log

- Objective: build matched-sample image-only WAD training variants.
- Training: not started.
- Inference: not started.
- GPU tasks: none.
- Primary variant: `image-only -> alter`.
- Auxiliary variant: `image-only -> full WAD JSON`.
- Test data: copied from the existing validated WAD test conversion.
- Build result: 10,146 source records; 8,581 eligible alter records; 1,565 QA records excluded; 1,000 samples selected with seed 42.
- Validation result: both variants contain the same 1,000 images, no input summary, and valid full-JSON targets.
- Data audit: 1,000 unique selected train images, 1,007 unique WAD test images, zero train/test overlap, and zero empty full-JSON fields.
- Environment note: an earlier inefficient all-image scan remains in network-disk I/O wait; it does not affect `data/fair_variants` and was not terminated.

## 2026-06-06 WAD Variant Evaluation Code Preparation

- Objective: evaluate WAD alter-only and WAD full-JSON variants on Our non-SAFE and WAD test sets.
- Evaluation script: `scripts/eval_wad_training_variants.py`.
- External foreground commands: `RUN_EVALUATION.md`.
- Path behavior: project inputs are resolved dynamically from the script location; external commands use `<REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New`.
- Preflight: passed for all four conditions; evaluation counts are 254, 254, 1007, and 1007.
- Full-JSON extraction failures retained as explicit failures: 2 on Our non-SAFE and 4 on WAD test.
- BERTScore-only smoke: completed before the user redirected execution outside; no LAVE or full evaluation was started.
- Optimization: BERTScore scorer is loaded once and reused across all four conditions.
- Status: code ready for external smoke and full foreground runs; results pending.

## 2026-06-06 WAD Variant Full Evaluation Completed

- Result directory: `results/wad_variants_full_20260606_042930`.
- Evaluated condition-samples: 2,522; all judge outputs parsed successfully.
- Our non-SAFE (n=254): alter-only BERTScore-F1 0.8489 / judge 3.6890; full-JSON BERTScore-F1 0.8506 / judge 4.0197.
- WAD test (n=1,007): alter-only BERTScore-F1 0.9021 / judge 5.5164; full-JSON BERTScore-F1 0.8998 / judge 6.1480.
- Paired bootstrap, full-JSON minus alter-only: Our BERT +0.0017 [-0.0000, +0.0034], Our judge +0.3307 [-0.0354, +0.7087], WAD BERT -0.0023 [-0.0040, -0.0007], WAD judge +0.6316 [+0.4528, +0.8064].
- Full-JSON extraction failures: 2 Our + 4 WAD; all retained and assigned judge score 0.
- Interpretation: alter-only remains the primary fair WAD baseline; full-JSON is an auxiliary multitask ablation with significantly stronger judge performance on WAD test but no stable gain on Our test.

