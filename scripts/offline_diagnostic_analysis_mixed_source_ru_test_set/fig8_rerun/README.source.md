# DeepSeek-based Fig. 8 rerun

This directory contains a reproducible rerun of Fig. 8 using the reviewed
Offline Diagnostic Analysis on the Mixed-source R+U Test Set data. It does not modify reviewed inputs or distance-analysis outputs.

## Run

```powershell
& 'python' `
  '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\new_fig_8\run_deepseek_fig8.py' --match-mode strict

& 'python' `
  '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\new_fig_8\run_deepseek_fig8.py' --match-mode loose

& 'python' -m pytest `
  '实验数据\Offline Diagnostic Analysis on the Mixed-source R+U Test Set\new_fig_8\test_run_deepseek_fig8.py' -q -p no:cacheprovider
```

Strict outputs remain at the top level and reuse the existing redacted strict
cache. Controlled-loose outputs and their separate cache are written under
`loose/`. The script reads `llm_config.json` without printing or writing the
API key.

## Method

- All 863 records are aligned by list position and exact image value.
- They are reduced to 694 unique images. A duplicate group keeps its hazardous
  GT record when present; otherwise it keeps the first SAFE record.
- Object matching is deterministic rule-first matching followed by cached
  DeepSeek matching for unresolved names.
- Both strict and controlled-loose profiles use names only and enforce a
  one-to-one mapping. Category and distance do not influence matching.
- Rows are actual GT categories and columns are actual predicted categories.
  Matched prediction categories are never overwritten with GT categories.
- Unmatched GT objects contribute to the SAFE column. Unmatched predictions
  contribute to the SAFE row.
- A selected unique both-empty record contributes one SAFE/SAFE event and is
  also tracked in `safe_safe_counts.json`.
- Multi-category object events split one unit uniformly across the relevant
  category Cartesian product.
- API failures are explicit and cause a non-zero exit after audit outputs are
  written.

## Outputs

Each method directory contains matrices, CSV files, a heatmap, per-record match
details, an audit summary, and a manual-review queue.

Top-level strict outputs include `validation_summary.json`,
`comparison_original.md`, `safe_safe_counts.json`, and
`deepseek_match_cache.jsonl`.

The `loose/` directory additionally contains:

- `comparison_strict_loose.md`: strict versus controlled-loose metrics.
- `added_loose_matches_ours.jsonl` and
  `added_loose_matches_fewshot.jsonl`: only matches added by the loose profile,
  for targeted human review.

## Interpretation

The controlled-loose result under `loose/` is the selected formal replacement
for the original Fig. 8. The strict result remains available as a
matching-policy sensitivity baseline. The loose-only added-match files are
retained as audit artifacts.
