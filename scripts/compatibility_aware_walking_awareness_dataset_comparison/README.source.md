# Compatibility-aware Walking Awareness Dataset Comparison WalkVLM/WAD Reviewer-Response Package

This folder contains the conclusions, raw text outputs, evaluation results, and lightweight code/configuration needed to reproduce the final reviewer-facing WalkVLM/WAD comparison.

## Final Experimental Position

- **Primary WAD baseline:** Qwen2.5-VL trained on 1,000 WAD samples with image-only input and direct `alter` supervision.
- **Primary comparison:** R+U-trained Qwen2.5-VL versus WAD alter-only Qwen2.5-VL.
- **Auxiliary ablation:** WAD image-only to full JSON, with only the generated `alter` field evaluated.
- **Our test protocol:** evaluate only the 254 non-SAFE references, consistent with the hazard-guidance main experiment.
- **WAD test protocol:** evaluate all 1,007 references.
- **Metrics:** BERTScore-F1 and a Qwen3-14B LAVE-style semantic/actionability judge.

The released WAD field is named `alter`; it is not renamed to `alert`.

## Directory Contents

- `CONCLUSIONS_AND_PAPER_MATERIAL.md`: final interpretation, tables, caveats, and paper-ready text.
- `REPRODUCE.md`: complete reproduction procedure and external dependencies.
- `code/scripts/`: data construction, validation, inference, extraction, and evaluation code.
- `code/configs/`: LLaMA-Factory LoRA training and merge configurations.
- `data/fair_variants/`: exact selected 1,000-sample training datasets and selection manifest.
- `test/`: test JSON files and references.
- `outputs/`: all four model inference outputs plus full-JSON raw outputs.
- `results/wad_variants_full_20260606_042930/`: complete final metric outputs and per-sample judge results.
- `comparison_context/previous_compatibility_aware_walking_awareness_dataset_comparison_final_results/`: prior R+U evaluation outputs used for reviewer-facing comparison context.
- `docs/`: experimental design, commands, and experiment log.
- `MANIFEST.tsv` and `SHA256SUMS.txt`: package inventory and checksums.

## Not Included

Large artifacts are intentionally excluded:

- Base Qwen2.5-VL-7B-Instruct weights.
- Merged model weights.
- LoRA checkpoints.
- Original WAD image/video files.
- Our-test image files.
- Qwen3-14B judge weights and RoBERTa-large BERTScore weights.

These are listed with expected paths in `REPRODUCE.md`.
