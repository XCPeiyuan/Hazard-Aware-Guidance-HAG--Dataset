# Reproduction Guide

## Scope

This package reproduces data construction, LoRA training configuration, inference text outputs, and evaluation logic. It does not bundle large model weights or image datasets.

## Expected External Runtime Root

```bash
<REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes
```

Place this package at:

```bash
<REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New/Compatibility-aware Walking Awareness Dataset Comparison
```

The original runnable project layout is expected at:

```bash
<REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New
```

## Required External Assets

- Original WAD source JSONL: `<REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM/wad_dataset_v2/train_v2.json`.
- WAD frame images referenced by the source dataset.
- Our-test images: `<REMOTE_WORKSPACE_ROOT>/dataset/images`.
- Base model: `<REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM/Qwen2.5-VL-7B-Instruct`.
- LLaMA-Factory with `llamafactory-cli`.
- Python packages used by the scripts: PyTorch, Transformers, Pillow, bert-score, tqdm.
- Judge model: `Qwen/Qwen3-14B`.
- BERTScore model: `roberta-large`.

## Exact Included Training Data

The exact selected training data is already included under `data/fair_variants/`. Rebuilding from WAD should reproduce it with seed 42 and sample count 1,000, but using the included files is the strongest exact-reproduction path.

## Restore Runnable Files

Copy package files into a clean `WalkVLM-New` project while preserving the structure:

```bash
cd <REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New
cp -r Compatibility-aware Walking Awareness Dataset Comparison/code/scripts/* scripts/
cp -r Compatibility-aware Walking Awareness Dataset Comparison/code/configs/* configs/
cp -r Compatibility-aware Walking Awareness Dataset Comparison/data/fair_variants data/
cp -r Compatibility-aware Walking Awareness Dataset Comparison/test/* test/
```

Do not overwrite existing results or model artifacts unless intentionally reconstructing a clean environment.

## Validate Data

```bash
cd <REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New
python scripts/validate_training_variants.py --data-dir data/fair_variants
python scripts/audit_generated_data.py --data-dir data/fair_variants --test-json test/walkvlm_qwen25vl_test_alter.json
```

## Train and Merge

```bash
CUDA_VISIBLE_DEVICES=<GPU> llamafactory-cli train <REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New/configs/train_wad_alter_only_lora.yaml
llamafactory-cli export <REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New/configs/merge_wad_alter_only_lora.yaml

CUDA_VISIBLE_DEVICES=<GPU> llamafactory-cli train <REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New/configs/train_wad_full_json_lora.yaml
llamafactory-cli export <REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New/configs/merge_wad_full_json_lora.yaml
```

## Inference

Run both models on `test/our_test.json` and `test/walkvlm_qwen25vl_test_alter.json`. The included `outputs/` directory contains the exact outputs used by the reported evaluation, so retraining/inference is not required to reproduce metrics.

## Reproduce Metrics from Included Outputs

Copy the included evaluation script into the runnable project or invoke it from the package with `--input-dir` pointing at a layout containing `test/` and `outputs/`.

Foreground full evaluation command:

```bash
cd <REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New
EXP=results/wad_variants_reproduced_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_wad_training_variants.py \
  --input-dir <REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New \
  --output-dir "$EXP" \
  --judge-model Qwen/Qwen3-14B \
  --judge-device-map auto \
  --judge-dtype bf16 \
  --lave-batch-size 4
```

BERTScore is deterministic for fixed software/model weights. The LLM judge uses deterministic greedy decoding, but small differences may still occur across Transformers/PyTorch/model revisions.

## Integrity

Run from this package directory:

```bash
sha256sum -c SHA256SUMS.txt
```

## Limitations of Exact Reproduction

- Exact training reproduction depends on LLaMA-Factory, Transformers, PyTorch, CUDA, and model revisions.
- The original environment lockfile was not captured.
- Model weights/checkpoints are excluded due to size.
- The Qwen3-14B judge and RoBERTa weights are external.
- Literature discussion is not reproducible from experiment files and must be verified separately.
