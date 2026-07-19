# WAD Training Variants Evaluation

All commands below run in the foreground and use the external runtime path.

## 1. Enter project

```bash
cd <REMOTE_WORKSPACE_ROOT>/root_ginger_expert/codes/WalkVLM-New
```

## 2. Preflight only

```bash
python scripts/eval_wad_training_variants.py \
  --output-dir results/wad_variants_preflight \
  --skip-bertscore \
  --skip-lave
```

Expected evaluation counts:

- `our_test_non_safe / WAD alter-only`: 254
- `our_test_non_safe / WAD full-JSON`: 254, including 2 explicit extraction failures
- `wad_test / WAD alter-only`: 1007
- `wad_test / WAD full-JSON`: 1007, including 4 explicit extraction failures

## 3. Smoke test

Use GPUs with enough combined free memory for Qwen3-14B. `CUDA_VISIBLE_DEVICES=2,3` makes those physical GPUs visible as logical `cuda:0,1`.

```bash
EXP=results/wad_variants_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_wad_training_variants.py \
  --output-dir "$EXP" \
  --limit 2 \
  --judge-model Qwen/Qwen3-14B \
  --judge-device-map auto \
  --judge-dtype bf16 \
  --lave-batch-size 4
```

Confirm that `aggregate_metrics.csv`, `bertscore_details.csv`, and `lave_scores.jsonl` are generated before the full run.

## 4. Full evaluation

```bash
EXP=results/wad_variants_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_wad_training_variants.py \
  --output-dir "$EXP" \
  --judge-model Qwen/Qwen3-14B \
  --judge-device-map auto \
  --judge-dtype bf16 \
  --lave-batch-size 4
```

The full run evaluates 2,522 condition-samples. Our test automatically excludes SAFE references. Full-JSON extraction failures remain in evaluation as `[ALTER_EXTRACTION_FAILED]`; do not delete them.

## 5. BERTScore-only run

Use this if the judge evaluation must be run separately:

```bash
EXP=results/wad_variants_bertscore_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EXP"
python scripts/eval_wad_training_variants.py \
  --output-dir "$EXP" \
  --skip-lave \
  --bertscore-model roberta-large
```

## Output files

- `preflight.json`: input alignment and failure counts
- `aggregate_metrics.csv`: aggregate BERTScore and judge metrics
- `bertscore_details.csv`: per-sample BERTScore
- `lave_scores.jsonl`: per-sample Qwen3-14B judge records
- `lave_performance.json`: judge throughput
- `WAD_Training_Variants_Report.md`: generated summary and LaTeX table
