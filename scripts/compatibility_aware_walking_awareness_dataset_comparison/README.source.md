# WAD Experiment Settings

The selected baseline fine-tunes Qwen2.5-VL-7B with image-only input on 1,000 WAD training samples, using direct `alter` supervision. A separate full-JSON training variant is retained for comparison.

The text comparison uses 254 hazard-reference samples from the Real-world test set and 1,007 WAD test samples. This directory contains the training/inference helpers and original text-metric evaluation code. It does not contain the full training data, raw images, or all generated outputs.

See the [directory overview](README.md) and the [additional structured comparison](../wad_structured_comparison/README.md) for the current result organization.
