# LLaMA-Factory Fine-Tuning Files

This directory contains the LLaMA-Factory configuration files used to
fine-tune and merge the HAG Qwen2.5-VL model.

## Directory layout

```text
.
├── data/
│   └── dataset index files
├── train/
│   └── qwen2_5vl_lora_sft.yaml
└── merge/
    └── qwen2_5vl_lora_sft.yaml
```

Place the dataset index files in `data/`. Before training, place the training
data and image files in locations available on your machine, then update the
corresponding dataset, image, base-model, output, and adapter paths in the YAML
configuration files. Do not commit private images, third-party data that cannot
be redistributed, credentials, or machine-specific absolute paths.

## Requirements

Install a compatible LLaMA-Factory environment and ensure that
`llamafactory-cli` is available in your shell. The base model used for this
setting is `Qwen/Qwen2.5-VL-7B-Instruct`.

The configuration represents supervised fine-tuning (SFT) with Low-Rank
Adaptation (LoRA). Ensure that your hardware and installed PyTorch/CUDA stack
support the precision and batch settings specified in the YAML files.

## Fine-tune the model

Run the following command from this directory:

```bash
llamafactory-cli train train/qwen2_5vl_lora_sft.yaml
```

The LoRA adapter is written to the output directory configured in
`train/qwen2_5vl_lora_sft.yaml`.

## Merge the LoRA adapter

After fine-tuning, update `merge/qwen2_5vl_lora_sft.yaml` so that its adapter
path points to the LoRA adapter produced by training. Then run:

```bash
llamafactory-cli export merge/qwen2_5vl_lora_sft.yaml
```

The merged checkpoint is written to the export directory specified in
`merge/qwen2_5vl_lora_sft.yaml`.

## Reproducibility notes

For a result comparable to the reported HAG R+U setting, keep the base model,
dataset composition, prompt format, LoRA hyperparameters, preprocessing,
generation settings, and evaluation protocol consistent with the associated
release documentation. The released merged model is available at
[`xcyuan/HAG-Qwen2.5-VL-7B-RU`](https://huggingface.co/xcyuan/HAG-Qwen2.5-VL-7B-RU).

This documentation does not by itself redistribute the underlying training
images or any data subject to third-party license, privacy, or consent
restrictions.
