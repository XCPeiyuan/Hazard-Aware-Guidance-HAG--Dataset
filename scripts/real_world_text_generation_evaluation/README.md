# Real-world Text Generation Evaluation

This directory contains the original scripts used to generate and evaluate
hazard-aware text outputs on the Real-world test workflow. The scripts are
preserved without source changes.

| Script | Role |
| --- | --- |
| `evaluation.py` | Computes SAFE classification accuracy, BERTScore, METEOR, and text-metric plots. |
| `eval_safe_and_danger.py` | Reports SAFE/Danger classification statistics. |
| `review-danger-correct.py` | Selects records whose reference and prediction are both non-SAFE for review. |
| `eval_lave_qwen.py` | Uses Qwen3-14B for LAVE-style reference-text evaluation. |
| `summarize_lave_results.py` | Aggregates LAVE score files across model settings. |
| `simple_check.py` | Checks the three-line image/output/blank-line record format. |
| `inference/` | Original inference scripts that generate the text files consumed above. |

Run a script from its own directory when preserving its original relative-path
behavior. The scripts intentionally retain the historical local paths, model
identifiers, GPU settings, prompts, and output conventions. Dataset files,
model weights, generated outputs, and private credentials are not included.

For the major-revision WAD comparison, use
`../compatibility_aware_walking_awareness_dataset_comparison/`. For the later
structured-field and object-level diagnostic analyses, use their dedicated
directories under `../`.
