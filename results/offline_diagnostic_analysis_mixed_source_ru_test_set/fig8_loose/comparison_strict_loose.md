# Strict vs. controlled-loose new Fig. 8

Both runs use the same 694 unique-image records, names-only one-to-one matching, actual predicted categories, and SAFE/SAFE counting. The only difference is the DeepSeek semantic matching profile.

| Method | Profile | Matched pairs | GROUND recall | PIT recall | OVERHEAD recall |
|---|---:|---:|---:|---:|---:|
| ours | strict | 431 | 40.38% | 41.94% | 20.24% |
| ours | loose | 611 | 53.05% | 66.13% | 51.19% |
| fewshot | strict | 196 | 21.83% | 2.42% | 0.00% |
| fewshot | loose | 371 | 38.01% | 21.77% | 0.00% |

- `ours` loose-only added matches requiring review: 180.
- `fewshot` loose-only added matches requiring review: 175.

The controlled-loose result is the selected formal replacement for the original Fig. 8. The strict result remains a matching-policy sensitivity baseline, and the two `added_loose_matches_*.jsonl` files are retained as audit artifacts.
