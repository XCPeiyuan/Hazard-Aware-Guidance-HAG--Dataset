---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen2_5_vl
- vision-language-model
- hazard-aware-guidance
- assistive-navigation
- image-text-to-text
- bf16
base_model: Qwen/Qwen2.5-VL-7B-Instruct
base_model_relation: finetune
---

# HAG-Qwen2.5-VL-7B-RU

[English](README.md) | [简体中文](README_zh-CN.md)

A Qwen2.5-VL-7B-Instruct model fine-tuned with LoRA on mixed Real-world and Unity-based (R+U) data for the study **MLLM-Assisted Dataset Construction for Hazard-Aware Guidance for Individuals with Visual Impairments**.

## Model and output

The [model page](https://huggingface.co/xcyuan/HAG-Qwen2.5-VL-7B-RU) is the release location for the merged checkpoint and trained LoRA adapter. This GitHub directory contains model-card sources, not weights.

Input is a static first-person outdoor image. The intended English output is either `<SAFE/>` or:

```text
<ALERT>hazard description, relative direction, and approximate distance in steps</ALERT>
<GUIDE>avoidance guidance</GUIDE>
```

`<SAFE/>` is a predicted-safe label, not a guarantee that the scene is safe.

## Configuration and inference

The supplied [training configuration](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/Fine-tuning_data/train/qwen2_5vl_lora_sft.yaml) uses rank-8 LoRA (`lora_target: all`), three epochs, learning rate `1e-4`, per-device batch size 2, gradient accumulation 4, sequence length 2,048, and BF16. Global batch size depends on the number of training devices.

See the [inference scripts](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/scripts/real_world_text_generation_evaluation/inference/README.md) for prompts and inference processing. Configure the model and image paths for your environment. Generation settings depend on the experiment; the repeated latency benchmark uses `max_new_tokens=1024`.

## Evaluation

The manuscript reports image-level hazardous-scene precision **0.852**, recall **0.929**, and F1 **0.889** for the 7B R+U setting over the 426-record Real-world test set. These are scene-classification metrics, distinct from object-level Hazard F1.

The additional [WAD comparison](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/results/wad_structured_comparison/README.md) reports object-level Hazard F1 and detection-aware Category F1. In the repeated [workstation benchmark](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/results/repeated_workstation_latency/README.md), the 7B model averages **1.03 s** for predicted-safe outputs and **2.02 s** for predicted-hazard outputs, with peak allocated GPU memory of **16.05 GiB**. Measurements pool 800 timed observations for 7B from two replicas and two rounds over the same 200 benchmark records, with four concurrent single-GPU replicas across the 3B/7B models.

## Scope and availability

Evaluation concerns static daytime outdoor images. Distances are approximate step counts. Offline scores and workstation latency do not establish real-world navigation safety or mobile deployability.

Dataset materials are planned for release after acceptance, subject to licensing and privacy constraints. See the [repository overview](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/README.md) for planned contents and the reserved identifier. Third-party real-world images are not redistributed.

## License and citation

The model-card materials include an [Apache-2.0 license](LICENSE). This does not grant rights to third-party datasets or assets. Citation metadata will be added when available.
