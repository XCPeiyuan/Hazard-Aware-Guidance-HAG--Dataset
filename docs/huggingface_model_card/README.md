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

HAG-Qwen2.5-VL-7B-RU is a BF16 merged checkpoint of
`Qwen/Qwen2.5-VL-7B-Instruct` fine-tuned on the mixed-source Real-world and
Unity-based (R+U) dataset from the study *MLLM-Assisted Dataset Construction
for Hazard-Aware Guidance for Individuals with Visual Impairments*.

The model takes a static first-person outdoor image and is intended to produce
structured, English hazard-aware guidance. Its output schema uses three custom
tokens:

```text
<SAFE/>
```

or

```text
<ALERT>hazard description, relative direction, and approximate distance in steps</ALERT>
<GUIDE>concrete avoidance guidance</GUIDE>
```

`<SAFE/>` denotes that no hazard is present in the current scene. `<ALERT>`
describes the hazard, and `<GUIDE>` provides an avoidance instruction.

## Inference prompts

The following Chinese and English prompts define the fixed output contract used
for inference. The model must return either exactly `<SAFE/>`, or exactly one
`<ALERT>...</ALERT><GUIDE>...</GUIDE>` pair, with no additional content.

Chinese:

```json
{
  "content": "<image>你是一名协助盲人出行的虚拟志愿者。盲人正朝正前方行走。请根据场景判断前方是否安全：如果安全，只输出：<SAFE/>；如果存在危险，只输出（格式固定、简洁明了、不得添加其他内容）：<ALERT>描述障碍物、方向和大致距离</ALERT><GUIDE>给出明确避让的动作指令</GUIDE>",
  "role": "user"
}
```

English:

```json
{
  "content": "<image>You are a virtual assistant helping a blind person travel. The blind person is walking straight ahead. Please determine whether the path ahead is safe: If it is safe, output only: <SAFE/>; If there is hazard, output only (fixed format, concise, no additional content): <ALERT>Describe the obstacle, its direction, and approximate distance</ALERT><GUIDE>Provide clear instructions for avoidance actions</GUIDE>",
  "role": "user"
}
```

## Model details

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Checkpoint type | Merged full model, BF16 safetensors |
| Fine-tuning data | Mixed-source Real-world + Unity-based synthetic dataset (R+U) |
| Fine-tuning stage | Supervised fine-tuning (SFT) |
| Fine-tuning method | LoRA, merged into the released checkpoint |
| LoRA rank | 8 |
| LoRA target modules | All trainable layers |
| Training epochs | 3 |
| Per-device batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Learning rate | 1.0e-4 |
| Learning-rate schedule | Cosine decay |
| Warmup ratio | 0.1 |
| Maximum sequence length | 2,048 |
| Training precision | BF16 |
| Training framework | LLaMA-Factory-based configuration |

No class rebalancing or R/U mixing-ratio tuning was performed. The reported
R+U setting uses the constructed dataset composition directly.

## Intended use

This checkpoint is a research artifact for studying structured hazard-aware
text generation from static first-person outdoor images. It may be used for
research, reproduction of the reported model setting, and development or
evaluation of assistive-navigation research prototypes.

The checkpoint is not a complete assistive-navigation system. It does not
provide audio delivery, wearable feedback, temporal reasoning, dynamic-scene
understanding, collision avoidance, or validated real-time mobile deployment.

## Limitations and safety notice

Do not use this model as the sole basis for navigation, collision avoidance,
medical decisions, emergency decisions, or other safety-critical decisions.
It can miss hazards, generate false alarms, confuse spatial relationships, or
produce incomplete or unsuitable guidance, especially in cluttered scenes.

The model was studied on single-frame daytime outdoor scenes. Its step-based
distances are coarse approximations; reliable calibration residuals were not
reported for the Real-world source. The reported evaluation is offline and does
not demonstrate real-world walking safety, collision reduction, target-user
usability, or mobile/edge deployability.

## Installation

The released configuration records `transformers==4.56.2`. Install a compatible
PyTorch build for your hardware, then install the following packages:

```bash
pip install "transformers==4.56.2" accelerate pillow
```

## Minimal inference example

This example uses the published model identifier
`xcyuan/HAG-Qwen2.5-VL-7B-RU`. It reproduces the essential inference path of
the original experiment script: `AutoModelForImageTextToText`, a PIL RGB image,
a user message whose content order is text followed by image, `torch_dtype="auto"`,
`device_map="auto"`, and `max_new_tokens=128`.

Choose `language="zh"` for the Chinese prompt or `language="en"` for the
English prompt. In the original serialized training/inference records, the
prompt starts with `<image>`. In the Transformers message API below, the image
is instead supplied by the separate `{"type": "image", "image": image}`
content block; therefore, the text prompt deliberately omits the literal
`<image>` prefix. Do not add it again when using this message representation.

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "xcyuan/HAG-Qwen2.5-VL-7B-RU"

PROMPTS = {
    "zh": (
        "你是一名协助盲人出行的虚拟志愿者。盲人正朝正前方行走。"
        "请根据场景判断前方是否安全：如果安全，只输出：<SAFE/>；"
        "如果存在危险，只输出（格式固定、简洁明了、不得添加其他内容）："
        "<ALERT>描述障碍物、方向和大致距离</ALERT>"
        "<GUIDE>给出明确避让的动作指令</GUIDE>"
    ),
    "en": (
        "You are a virtual assistant helping a blind person travel. "
        "The blind person is walking straight ahead. Please determine whether "
        "the path ahead is safe: If it is safe, output only: <SAFE/>; "
        "If there is hazard, output only (fixed format, concise, no additional "
        "content): <ALERT>Describe the obstacle, its direction, and approximate "
        "distance</ALERT><GUIDE>Provide clear instructions for avoidance actions</GUIDE>"
    ),
}

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

def infer_one_image(image_path: str, language: str = "en") -> str:
    if language not in PROMPTS:
        raise ValueError("language must be 'zh' or 'en'")

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPTS[language]},
                {"type": "image", "image": image},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Preserve the original script's prompt-token trimming and decoding behavior.
    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

print(infer_one_image("/absolute/path/to/image.jpg", language="en"))
print(infer_one_image("/absolute/path/to/image.jpg", language="zh"))
```

The checkpoint's released generation configuration uses temperature `1e-6` and
repetition penalty `1.05`. The paper's workstation latency benchmark used
`max_new_tokens=128`, BF16 inference, Qwen2.5-VL dynamic-resolution
preprocessing, and no 8-bit or 4-bit quantization. The original inference
script used the checkpoint's generation configuration rather than overriding
these values in the `generate()` call.

## Reported evaluation context

The following results are reported for the 7B R+U fine-tuned configuration on
the 426 reviewed Real-world test set:

| Metric | Value |
| --- | ---: |
| ALERT BERTScore F1 | 0.932 |
| ALERT METEOR | 0.527 |
| GUIDE BERTScore F1 | 0.942 |
| GUIDE METEOR | 0.644 |
| Hazard Precision | 0.852 |
| Hazard Recall | 0.929 |
| Hazard F1 | 0.889 |
| ALERT LAVE score | 7.564 |
| GUIDE LAVE score | 7.648 |
| Overall LAVE score | 7.827 |

BERTScore, METEOR, and LAVE are reference-text-based measures. In particular,
the LAVE evaluator does not receive the input image and is not image-aware
verification of scene correctness. These numbers are study-specific results,
not guarantees of performance or safety in other environments.

In a workstation-side benchmark on an NVIDIA RTX A6000 GPU over 200 timed
images, the 7B model had a peak GPU memory use of 16.05 GB. Average end-to-end
latency was 1.63 s for outputs exactly equal to `<SAFE/>` and 2.65 s for
predicted-hazard outputs. These are non-streaming full-output measurements and
do not establish consumer-GPU, mobile, or edge performance.

## Reproducibility materials and release boundaries

This Hugging Face repository contains the merged model checkpoint and its
runtime configuration. The companion code/data release is intended to provide
prompt templates, annotation-generation scripts, evaluation scripts,
parser/evaluation utilities, Unity synthetic images and annotations,
structured annotation files, and reviewed evaluation labels, subject to
privacy, consent, and third-party license constraints.

Third-party Real-world images are not redistributed. Self-collected media may
be released only after privacy and consent screening, with masking where
needed. Implementation-specific Unity project scripts are not a complete
release commitment because they depend on project-specific assets and
environment settings. The model and associated materials are released after
acceptance, subject to applicable base-model, dataset, privacy, and consent
constraints.

## License

This release is distributed under Apache-2.0. The base model is
`Qwen/Qwen2.5-VL-7B-Instruct`, which is also labeled Apache-2.0. Users remain
responsible for complying with all applicable third-party dataset, privacy,
consent, and local-law requirements.

## Citation

Citation metadata will be added after the paper's final bibliographic metadata
is available.

## Acknowledgment

This model is derived from `Qwen/Qwen2.5-VL-7B-Instruct`. Please cite the
Qwen2.5-VL work when using the base model, in addition to citing the associated
paper when its final metadata becomes available.
