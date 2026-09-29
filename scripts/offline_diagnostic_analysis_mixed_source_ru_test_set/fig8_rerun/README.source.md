# Object-level Category Accounting

`run_deepseek_fig8.py` generates object-level category matrices using the mixed-source evaluation inputs and the sibling distance-analysis matching code.

## Inputs and execution

Supply the reviewed reference and prediction JSON files under `../distance/review/` and an API configuration at `../distance/llm_config.json`. The script retains checks and path constants for the original evaluation cohort; inspect these before adapting it to other data. The reviewed data and matching caches are not included here.

From this directory:

```bash
python run_deepseek_fig8.py --match-mode loose
```

## Accounting

Matching is one-to-one and based on object names. Matched categories retain their actual reference/predicted values. Unmatched reference objects enter the SAFE column and unmatched predictions enter the SAFE row. These are accounting buckets, not counts of safe images. A both-empty record contributes one SAFE/SAFE event. Multi-category matches split their weight across the category combinations.

The controlled-loose profile supplies the reported category matrices; the strict profile supports matching-policy comparison. Results are separate from WAD's Hazard F1 and Category F1 evaluation.
