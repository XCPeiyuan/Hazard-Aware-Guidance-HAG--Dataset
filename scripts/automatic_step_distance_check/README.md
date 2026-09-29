# Automatic Step-distance Check

This experiment compares a reconstructed mask/depth procedure with manual step-count references on an independent 77-image evaluation set. It does not refit the distance mapping or evaluate generated guidance.

The retained implementation uses Grounding DINO, SAM2.1, and Depth Anything V2. A candidate mask must have bounding-box IoU ≥0.50 with the manually annotated object box; the eligible mask with the highest SAM score is selected. Bounding-box mask fallback is disabled. The median normalized depth within the mask is converted using the fixed mapping `round(3.876 + 5.197 * ln(255 / d))`. Image-level depth-consistency filters are not applied in this diagnostic check.

## Run

Install PyTorch, Transformers, NumPy, Pillow, and SciPy compatible with the retained implementation. The recorded run used PyTorch 2.11.0+cu128 and Transformers 5.10.2; model revision identifiers are in `model_revisions.json`.

```bash
python run_distance_check.py --annotations annotations.json --images PATH_TO_IMAGES --output NEW_OUTPUT_DIRECTORY
```

CUDA is required by this runner. Supply the independent annotations and images separately. The JSON has an `annotations` mapping keyed by image filename, with image dimensions and hazard records containing `id`, `object_name`, `hazard_category`, `evaluation_role`, `distance_steps`, and normalized `bbox` (`x`, `y`, `width`, `height`). The runner checks the 77-image input size. The output directory must be new.

`implementation/` retains the modules used in the distance experiment, separately from the evolving SSI utility elsewhere in the repository. Only input/output setup was adapted for this public runner. Outputs include masks, depth arrays, per-object records, and aggregate errors. The published summary is under [results/automatic_step_distance_check](../../results/automatic_step_distance_check/README.md).
