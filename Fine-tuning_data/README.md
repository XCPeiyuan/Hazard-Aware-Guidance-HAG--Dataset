# Fine-tuning and Adapter Merge Configurations

These LLaMA-Factory configurations describe LoRA supervised fine-tuning of `Qwen/Qwen2.5-VL-7B-Instruct` on the Real-world + Unity (R+U) training data.

- `data/dataset_info.json`: ShareGPT-style dataset definitions for `HAG_dataset_R_train.json` and `HAG_dataset_U_train.json`.
- `train/qwen2_5vl_lora_sft.yaml`: training settings.
- `merge/qwen2_5vl_lora_sft.yaml`: adapter-merge/export settings.

The training configuration sets LoRA rank 8, `lora_target: all`, three epochs, learning rate `1e-4`, per-device batch size 2, gradient accumulation 4, sequence length 2,048, and BF16. Global batch size also depends on the number of training devices.

## Use

Install LLaMA-Factory and its dependencies. Configure its dataset directory to use the supplied dataset definitions, and place the corresponding training JSON files there. Resolve the image paths in those files and configure the base-model, output, adapter, and export paths. Training data and weights are not included here.

From this directory, with `llamafactory-cli` available:

```bash
llamafactory-cli train train/qwen2_5vl_lora_sft.yaml
llamafactory-cli export merge/qwen2_5vl_lora_sft.yaml
```

Run export only after setting the adapter path to the trained adapter. The [HAG model page](https://huggingface.co/xcyuan/HAG-Qwen2.5-VL-7B-RU) provides the model release.
