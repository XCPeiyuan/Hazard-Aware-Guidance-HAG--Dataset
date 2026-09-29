# WAD Structured Comparison Results

| Test set | Model | BERTScore F1 | Hazard F1 | Category F1 |
| --- | --- | ---: | ---: | ---: |
| Real-world | R+U fine-tuned | 0.933 | 0.644 | 0.630 |
| Real-world | WAD alter-only | 0.849 | 0.293 | 0.284 |
| WAD | R+U fine-tuned | 0.858 | 0.166 | 0.171 |
| WAD | WAD alter-only | 0.902 | 0.253 | 0.270 |

The Real-world comparison uses the 254 hazard-reference samples selected for the comparison. WAD Hazard F1 uses 1,007 samples; Category F1 uses 961, excluding 46 references with unresolved categories for both models. Parsed SAFE predictions remain in the evaluation.

`structured_metrics.json` and `structured_metrics.csv` contain unrounded F1 values and aggregate TP/FP/FN counts. BERTScore values in the table are those reported in the manuscript; the new files concern structured-field scoring.

Hazard matching uses name compatibility and one-to-one assignment. Category F1 counts matching category intersections as true positives and missing/extra categories as false negatives/positives. These are object-level metrics, distinct from image-level hazardous-scene classification.

WAD references describe reminder-mentioned objects, not an exhaustive inventory of visible hazards. Its evaluable category labels comprise 1,079 Common obstacle, six Pitfall hazard, and two Upper-body hazard labels. Accordingly, these scores measure agreement with the supplied references.

See the [code and protocol](../../scripts/wad_structured_comparison/README.md). Dataset records and raw predictions are not included here.
