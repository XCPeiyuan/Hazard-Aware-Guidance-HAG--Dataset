# Hazard-Aware Guidance (HAG) Dataset

This repository is the planned release page for the dataset and code package associated with the paper:

**MLLM-Assisted Dataset Construction for Hazard-Aware Guidance for Individuals with Visual Impairments**

![.](Images/Fig_1b.png)

## Associated Hugging Face Releases

- **Model weights:** [HAG-Qwen2.5-VL-7B-RU](https://huggingface.co/xcyuan/HAG-Qwen2.5-VL-7B-RU)
- **Dataset (Coming soon):** [Hazard-Aware-Guidance-HAG_Dataset](https://huggingface.co/datasets/xcyuan/Hazard-Aware-Guidance-HAG_Dataset)

Model weights and the dataset release will be hosted separately on Hugging Face;
they are not included in this GitHub repository. This repository contains the
release documentation, code, configuration files, and compact result materials.

## Status

The release package is currently under preparation. No dataset files are currently published in this repository.

The planned public release will be prepared after paper acceptance, subject to third-party dataset licenses, base-model license constraints, and privacy/consent screening for self-collected media.

## Repository contents

This repository contains research code, configuration files, Unity scripts,
documentation, and selected result summaries related to hazard-aware guidance
from first-person outdoor images.

### `Fine-tuning_data/`

Files related to Qwen2.5-VL fine-tuning with LLaMA-Factory:

- `data/dataset_info.json`: dataset index configuration.
- `train/qwen2_5vl_lora_sft.yaml`: supervised fine-tuning configuration.
- `merge/qwen2_5vl_lora_sft.yaml`: LoRA adapter merge configuration.
- `README.md`: directory-specific documentation.

### `scripts/`

Research and evaluation scripts organized by workflow:

- `annotation_output_parser/`: parser for structured `ALERT`, `GUIDE`, and
  `SAFE` model outputs.
- `annotation_pipeline/`: annotation-pipeline script.
- `compatibility_aware_walking_awareness_dataset_comparison/`: WAD comparison
  scripts, configurations, experiment notes, and evaluation helpers.
- `dataset_distribution_and_preprocessing/`: dataset distribution and
  preprocessing utilities.
- `inference_efficiency_analysis/`: workstation and localhost inference
  efficiency scripts.
- `offline_diagnostic_analysis_mixed_source_ru_test_set/`: distance-aware
  diagnostic analysis, matching, metrics, validation, and tests.
- `pilot_human_evaluation_materials/`: pilot human-evaluation environment
  builder.
- `real_world_text_generation_evaluation/`: inference and text-generation
  evaluation scripts.
- `structured_field_evaluation_independently_annotated_subset/`: structured
  field evaluation code and annotation tool.
- `old_version/`: historical experiment scripts.
- `README.md`: overview of the script directories.

### `Unity_scripts/`

Unity-related scripts and plugins:

- `CameraCaptureAnnotator/`: captures rendered images and Structured Scene
  Information (SSI).
- `Unity_ConstructionPitGeneratorPlugin/`: procedural construction-pit
  generator.
- `Unity_ManholeGeneratorPlugin/`: procedural manhole and open-manhole hazard
  generator.
- `README.md`: overview of the Unity scripts.

### `results/`

Selected result files and summaries from several experiment groups:

- compatibility-aware WAD comparison;
- inference-efficiency analysis;
- offline diagnostic analysis on the mixed-source R+U test set; and
- structured-field evaluation on an independently annotated subset.

The directory also contains `README.md`, which lists the included result
groups.

### `docs/`

Documentation files currently present in the repository:

- `release_plan.md`: release-plan note.
- `huggingface_model_card/README.md`: model-card source document.
- `huggingface_model_card/README_zh-CN.md`: Chinese model-card source
  document.
- `huggingface_model_card/LICENSE`: license file stored with the model-card
  materials.

## Directory-specific documentation

More detailed descriptions are available in the README files under:

- [`Fine-tuning_data/`](Fine-tuning_data/)
- [`scripts/`](scripts/)
- [`scripts/annotation_output_parser/`](scripts/annotation_output_parser/)
- [`scripts/compatibility_aware_walking_awareness_dataset_comparison/`](scripts/compatibility_aware_walking_awareness_dataset_comparison/)
- [`scripts/dataset_distribution_and_preprocessing/`](scripts/dataset_distribution_and_preprocessing/)
- [`scripts/inference_efficiency_analysis/`](scripts/inference_efficiency_analysis/)
- [`scripts/offline_diagnostic_analysis_mixed_source_ru_test_set/`](scripts/offline_diagnostic_analysis_mixed_source_ru_test_set/)
- [`scripts/real_world_text_generation_evaluation/`](scripts/real_world_text_generation_evaluation/)
- [`scripts/structured_field_evaluation_independently_annotated_subset/`](scripts/structured_field_evaluation_independently_annotated_subset/)
- [`Unity_scripts/`](Unity_scripts/)
- [`Unity_scripts/CameraCaptureAnnotator/`](Unity_scripts/CameraCaptureAnnotator/)
- [`Unity_scripts/Unity_ConstructionPitGeneratorPlugin/`](Unity_scripts/Unity_ConstructionPitGeneratorPlugin/)
- [`Unity_scripts/Unity_ManholeGeneratorPlugin/`](Unity_scripts/Unity_ManholeGeneratorPlugin/)
- [`results/`](results/)

This document will be updated as the repository moves toward a final version.

## Citation

Citation information will be added after publication.

## License

License terms are not finalized yet. Please do not redistribute or reuse unreleased materials until the formal release license is posted.
