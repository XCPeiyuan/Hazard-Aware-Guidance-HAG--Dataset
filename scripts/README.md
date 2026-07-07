# Scripts

This directory will contain scripts for preparing, parsing, and evaluating the hazard-aware guidance dataset.

## Status

The scripts are currently under preparation. They will be released after paper acceptance, subject to dependency review, privacy screening, third-party dataset-license constraints, and base-model license constraints.

No executable release scripts are currently provided in this placeholder directory.

## Planned Scope

The planned release may include:

- annotation-generation helper scripts
- prompt-formatting utilities
- output parsing utilities
- structured-field validation scripts
- evaluation scripts for hazard category, distance, direction, and guidance quality
- dataset split and checksum utilities

## Not Included

The following materials are not planned for direct inclusion in this directory:

- private API keys or credentials
- raw third-party real-world images
- self-collected media before privacy/consent screening
- implementation-specific Unity project scripts that depend on local assets and environment settings
- large model weights or large dataset archives

## Expected Layout

```text
scripts/
├── annotation/
├── parsing/
├── evaluation/
└── data_checks/
