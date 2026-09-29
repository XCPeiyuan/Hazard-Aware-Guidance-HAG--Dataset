# Independently Annotated Structured-field Evaluation

This directory contains a Flask-based annotation tool under `annotation_tool/` and the helper `evaluation/recalculate_field_f1.py`.

The helper recalculates field scores from existing matched evaluation records; it is not an image-to-prediction pipeline. Configure its input and output paths before running it. Images and full annotation workspaces are not included.

Selected summaries are in `../../results/structured_field_evaluation_independently_annotated_subset/`. Their `pre_audit` filenames identify the stored evaluation snapshot.

The additional check of automatic step estimates against manual references is documented separately in [automatic_step_distance_check](../automatic_step_distance_check/README.md).
