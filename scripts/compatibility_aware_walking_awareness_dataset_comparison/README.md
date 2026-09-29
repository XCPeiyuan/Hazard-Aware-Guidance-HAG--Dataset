# Compatibility-aware WAD Comparison

This directory contains the earlier WAD training and text-evaluation workflow.

- `code/scripts/`: training-data construction, validation, inference, and text evaluation.
- `code/configs/`: LLaMA-Factory training and adapter-merge configurations.
- `docs/`: original experiment notes.

The main comparison uses image-only input and the WAD `alter` field as the training target. The full-JSON variant is an auxiliary experiment, not the selected baseline in Table 8. The field name `alter` follows WAD's source format.

For the additional Hazard F1 and Category F1 evaluation, see [wad_structured_comparison](../wad_structured_comparison/README.md).

WAD images, training JSON files, and model weights must be supplied separately. Configure paths before running the scripts. Original experiment notes may refer to files from the full experiment workspace that are not included in this repository.
