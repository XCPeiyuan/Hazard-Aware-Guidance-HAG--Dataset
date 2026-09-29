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

[English](README.md) | 简体中文

本模型基于 Qwen2.5-VL-7B-Instruct，在 Real-world 与 Unity 合成数据组成的 R+U 数据上进行 LoRA 微调，对应论文 **MLLM-Assisted Dataset Construction for Hazard-Aware Guidance for Individuals with Visual Impairments**。

## 模型与输出

[模型页面](https://huggingface.co/xcyuan/HAG-Qwen2.5-VL-7B-RU)为合并权重及训练所得 LoRA adapter 的发布地址。本 GitHub 目录仅存放模型说明，不包含权重。

输入为静态第一人称户外图像。预期输出为英文 `<ALERT>...</ALERT><GUIDE>...</GUIDE>`，分别描述障碍物、相对方向、近似步数和避让建议；预测安全时输出 `<SAFE/>`。该标签不代表场景安全保证。

## 配置与使用

[训练配置](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/Fine-tuning_data/train/qwen2_5vl_lora_sft.yaml)使用 rank-8 LoRA、`lora_target: all`、3 个 epoch、学习率 `1e-4`、每设备 batch size 2、梯度累积 4、序列长度 2,048 和 BF16。全局 batch size 还取决于训练设备数量。

提示词及推理处理见[推理脚本说明](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/scripts/real_world_text_generation_evaluation/inference/README.md)，运行前需配置模型和图像路径。不同实验采用各自的生成参数；重复延迟实验的 `max_new_tokens` 为 1024。

## 实验结果

论文在 426 条 Real-world 测试记录上报告的 7B R+U 图像级危险场景分类 Precision、Recall、F1 分别为 **0.852、0.929、0.889**。这些指标与对象级 Hazard F1 不同。

新增[WAD 比较](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/results/wad_structured_comparison/README.md)报告对象级 Hazard F1 和考虑检测结果的 Category F1。[重复工作站实验](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/results/repeated_workstation_latency/README.md)中，7B 的预测安全/预测危险分支平均端到端延迟分别为 **1.03 s / 2.02 s**，峰值已分配显存为 **16.05 GiB**。3B、7B 各两个单卡副本并发运行，每个副本对同一组 200 条记录测量两轮，7B 共计 800 次计时观测。

## 范围与数据

研究针对静态白天户外图像，距离以近似步数表示。离线指标及工作站延迟不能证明真实导航安全或移动端可部署性。

数据材料计划在论文接受后公开，并遵守许可和隐私约束。计划内容与预留标识符见[仓库主页](https://github.com/XCPeiyuan/Hazard-Aware-Guidance-HAG--Dataset/blob/main/README.md)。第三方真实图像不重新分发。

## 许可与引用

模型说明目录附有 [Apache-2.0 许可证](LICENSE)，不代表授权使用第三方数据或素材。引用信息将在确定后补充。
