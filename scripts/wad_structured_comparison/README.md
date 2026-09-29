# WAD Structured-field Comparison

This additional experiment evaluates the R+U-trained and WAD alter-only models using object-level Hazard F1 and detection-aware Category F1. Results are in [results/wad_structured_comparison](../../results/wad_structured_comparison/README.md).

## Included code

- `data_metrics.py`: the experimental scorer with the selected word-boundary name matcher. Workspace-specific input preparation is omitted; matching, assignment, cohort selection, and counting are retained.
- `score.py`: file-based entry point for already extracted records.
- `extraction.py`: extraction prompts, schema validation, retry/cache handling, and request construction.
- `local_server.py`: local model-serving component used for extraction. This is a component, not a complete orchestration package.

## Scoring

Install the scoring dependencies from `requirements.txt`. With the reference and both prediction JSON arrays prepared:

```bash
python score.py --reference reference.json --model-a ru.json --model-b wad.json --output scores.json
```

Records identify `dataset_id` and `sample_id`, with `status`, `scene_text_state`, `objects`, `hazard_eligible`, and `category_eligible`. Object records contain `name`, `categories`, and `category_status`. The internal category identifiers `GROUND`, `PIT`, and `OVERHEAD` correspond to Common obstacle, Pitfall hazard, and Upper-body hazard. See `data_metrics.py` for the exact input validation and counting rules.

The same eligible sample set is used for both models within each dataset and metric. Successfully parsed SAFE predictions remain eligible and contribute missed reference objects. A parse failure is not treated as a SAFE prediction.

## Extraction protocol

For tagged R+U predictions, extract only the ALERT span; GUIDE does not add objects. Untagged WAD predictions use their reminder text. Prediction extraction has no image access. WAD reference extraction may use the image to resolve text-mentioned objects and their categories, but does not add image-only objects. Unresolved categories are retained as unresolved rather than guessed.

The local extraction server requires the model checkpoint and its runtime dependencies separately. Its command-line options are defined in `local_server.py`. Reference annotations, prediction texts, private review records, and model weights are not included in this code directory. The released components do not reproduce the extraction review stage without those inputs.
