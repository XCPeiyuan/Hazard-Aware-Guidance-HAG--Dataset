# Historical Experiment Code Snapshots

This directory preserves earlier experiment scripts without source changes.
They remain available for traceability and for reproducing the historical
workflow, but they are not the recommended entry points for the major-revision
analysis.

All Python files in this directory retain their original names, defaults,
paths, prompts, model settings, and output conventions.

| Script | Historical role | Related current implementation |
| --- | --- | --- |
| `build_category_labels.py` | Uses Qwen3 to derive hazard categories from `ALERT` text. | `../offline_diagnostic_analysis_mixed_source_ru_test_set/fig8_rerun/` |
| `build_type_confusion_heatmap.py` | Builds a category confusion heatmap from category JSON files. | `../offline_diagnostic_analysis_mixed_source_ru_test_set/fig8_rerun/` |
| `PRO_build_type_confusion_heatmap.py` | Earlier LLM-assisted object matching and category heatmap. | `../offline_diagnostic_analysis_mixed_source_ru_test_set/fig8_rerun/` |
| `PRO_build_type_confusion_heatmap_number.py` | Earlier category heatmap variant with count labels. | `../offline_diagnostic_analysis_mixed_source_ru_test_set/fig8_rerun/` |
| `inference_txt_based-WalkVLM-ourset.py` | Runs a local WalkVLM checkpoint on the project test format. | `../compatibility_aware_walking_awareness_dataset_comparison/` |

The related current implementations are not byte-for-byte replacements. They
are the major-revision analysis paths used for the corresponding later
experiments. Do not place model weights, raw images, generated outputs, caches,
or private configuration files in this directory.
