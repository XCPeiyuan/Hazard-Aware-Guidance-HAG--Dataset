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

HAG-Qwen2.5-VL-7B-RU 是 `Qwen/Qwen2.5-VL-7B-Instruct` 的 BF16 合并完整模型，
在论文 *MLLM-Assisted Dataset Construction for Hazard-Aware Guidance for
Individuals with Visual Impairments* 所使用的混合来源 Real-world 与 Unity 合成
数据集（R+U）上完成微调。

该模型接收静态第一人称室外图像，目标是生成结构化英文危险感知引导。输出模式
使用三个自定义 token：

```text
<SAFE/>
```

或：

```text
<ALERT>危险描述、相对方向和以步数表示的近似距离</ALERT>
<GUIDE>具体的规避引导</GUIDE>
```

`<SAFE/>` 表示当前场景中不存在危险。`<ALERT>` 描述危险，`<GUIDE>` 给出
规避指令。

## 推理 prompt

下列中文和英文 prompt 定义推理时使用的固定输出约定。模型必须仅返回精确的
`<SAFE/>`，或精确的一组 `<ALERT>...</ALERT><GUIDE>...</GUIDE>`，不得添加其他
内容。

中文：

```json
{
  "content": "<image>你是一名协助盲人出行的虚拟志愿者。盲人正朝正前方行走。请根据场景判断前方是否安全：如果安全，只输出：<SAFE/>；如果存在危险，只输出（格式固定、简洁明了、不得添加其他内容）：<ALERT>描述障碍物、方向和大致距离</ALERT><GUIDE>给出明确避让的动作指令</GUIDE>",
  "role": "user"
}
```

英文：

```json
{
  "content": "<image>You are a virtual assistant helping a blind person travel. The blind person is walking straight ahead. Please determine whether the path ahead is safe: If it is safe, output only: <SAFE/>; If there is hazard, output only (fixed format, concise, no additional content): <ALERT>Describe the obstacle, its direction, and approximate distance</ALERT><GUIDE>Provide clear instructions for avoidance actions</GUIDE>",
  "role": "user"
}
```

## 模型信息

| 项目 | 取值 |
| --- | --- |
| 基座模型 | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 检查点类型 | 合并完整模型，BF16 safetensors |
| 微调数据 | 混合来源 Real-world + Unity 合成数据集（R+U） |
| 微调阶段 | 监督微调（SFT） |
| 微调方法 | LoRA，已合并至发布的检查点 |
| LoRA rank | 8 |
| LoRA 目标模块 | 全部可训练层 |
| 训练轮数 | 3 |
| 单设备 batch size | 2 |
| 梯度累积步数 | 4 |
| 有效 batch size | 8 |
| 学习率 | 1.0e-4 |
| 学习率调度 | 余弦衰减 |
| warmup 比例 | 0.1 |
| 最大序列长度 | 2,048 |
| 训练精度 | BF16 |
| 训练框架 | 基于 LLaMA-Factory 的配置 |

训练中未进行类别重平衡或 R/U 混合比例调参。报告的 R+U 设置直接使用构造完成的
数据集组成。

## 预期用途

本检查点是研究用工件，用于研究从静态第一人称室外图像生成结构化危险感知文本。
它可用于研究、复现报告的模型设置，以及开发或评测辅助导航研究原型。

本检查点不是完整的辅助导航系统。它不提供音频传达、可穿戴反馈、时间推理、动态
场景理解、碰撞规避或经验证的实时移动端部署。

## 局限与安全提示

不得将本模型作为导航、碰撞规避、医疗决策、紧急决策或其他安全关键决策的唯一
依据。它可能漏检危险、产生误报、混淆空间关系，或生成不完整、不适当的引导，
尤其是在杂乱场景中。

本模型研究的场景为单帧、白天、室外图像。以步数表示的距离是粗略近似；论文未对
Real-world 来源报告可靠的标定残差。报告的评测是离线评测，不证明真实步行安全、
碰撞减少、目标用户可用性或移动端/边缘端可部署性。

## 安装

发布的配置记录为 `transformers==4.56.2`。请先为你的硬件安装兼容的 PyTorch，
再安装以下包：

```bash
pip install "transformers==4.56.2" accelerate pillow
```

## 最小推理示例

该示例使用已发布的模型标识符 `xcyuan/HAG-Qwen2.5-VL-7B-RU`，并复刻原实验脚本的
关键推理路径：`AutoModelForImageTextToText`、PIL RGB 图像、内容顺序为文本后图像的
user message、`torch_dtype="auto"`、`device_map="auto"` 和 `max_new_tokens=128`。

选择 `language="zh"` 使用中文 prompt，或选择 `language="en"` 使用英文 prompt。
原始序列化训练/推理记录中的 prompt 以 `<image>` 开头。下面的 Transformers message
API 则通过独立的 `{"type": "image", "image": image}` 内容块提供图像，因此文本
prompt 有意省略了字面量 `<image>` 前缀；使用这种 message 表示时，不要再次添加它。

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

    # 保留原脚本的输入 token 裁剪与解码行为。
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

检查点发布的生成配置使用温度 `1e-6` 和 repetition penalty `1.05`。论文中的工作站
延迟基准使用 `max_new_tokens=128`、BF16 推理、Qwen2.5-VL 动态分辨率预处理，且未
使用 8-bit 或 4-bit 量化。原始推理脚本使用检查点的生成配置，而未在 `generate()`
调用中覆盖这些值。

## 报告的评测条件

下列结果针对 426 条经审查的 Real-world 测试记录上报告的 7B R+U 微调配置：

| 指标 | 数值 |
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

BERTScore、METEOR 和 LAVE 都是基于参考文本的度量。特别是，LAVE 评估器不接收
输入图像，因此不是对场景正确性的图像感知验证。这些数字是特定研究设置下的结果，
不构成其他环境中性能或安全性的保证。

在 NVIDIA RTX A6000 GPU 上、针对 200 张计时图像的工作站侧基准中，7B 模型的峰值
GPU 显存使用为 16.05 GB。对于输出恰为 `<SAFE/>` 的情形，平均端到端延迟为 1.63 s；
对于预测为危险的输出，平均端到端延迟为 2.65 s。这些是非流式的完整输出测量，不
证明消费级 GPU、移动端或边缘端的性能。

## 可复现材料与发布边界

本 Hugging Face 仓库包含合并后的模型检查点及其运行配置。配套的代码/数据发布计划
提供 prompt 模板、标注生成脚本、评测脚本、解析/评测工具、Unity 合成图像及标注、
结构化标注文件和经审查的评测标签，但均受隐私、同意与第三方许可证约束。

第三方 Real-world 图像不会被重新分发。自行采集的媒体仅会在完成隐私与同意审查后、
并在需要时进行遮罩后发布。项目专用的 Unity 工程脚本并非完整发布承诺，因为它们
依赖项目专用资产和环境设置。模型及相关材料将在论文接受后发布，并受适用的基座
模型、数据集、隐私和同意约束。

## 许可证

本发布采用 Apache-2.0。基座模型 `Qwen/Qwen2.5-VL-7B-Instruct` 也标记为
Apache-2.0。用户仍有责任遵守适用的第三方数据集、隐私、同意和当地法律要求。

## 引用

论文最终书目信息确定后，将补充引用元数据。

## 致谢

本模型派生自 `Qwen/Qwen2.5-VL-7B-Instruct`。使用本模型时，请引用 Qwen2.5-VL
工作；当相关论文的最终书目信息可用时，也请引用该论文。
