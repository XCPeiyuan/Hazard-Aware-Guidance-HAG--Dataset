# Distance-bin Object Recall

`run_analysis.py` matches reference and predicted object names, then reports recall by category and reference-distance bin: Near (≤5 steps), Medium (6–8), and Far (≥9).

Matching uses deterministic name rules followed by cached DeepSeek name matching. It is one-to-one; category and distance are not supplied to the name-matching prompt. The `strict` and `loose` profiles use separate outputs/caches.

## Inputs

The script expects these files under the sibling `distance/review/` directory:

- `testset_reviewed.json`
- `qwen25vl7b_reviewed.json`
- `qwen25vl7bfs_reviewed.json`

These data are not bundled here. The default API configuration path is `../distance/llm_config.json`; it can be overridden with `--config`. Consult `offline_diagnostic_analysis/deepseek_matcher.py` for its configuration schema.

After supplying the inputs and configuration, run from this directory:

```bash
python run_analysis.py --dry-run --match-mode loose
python run_analysis.py --match-mode loose --output-dir PATH_TO_OUTPUT
```

`--dry-run` validates inputs and estimates matching workload; it may write validation metadata. `compare_match_profiles.py` compares completed strict/loose runs using its configured locations. This is an offline object-mention analysis.
