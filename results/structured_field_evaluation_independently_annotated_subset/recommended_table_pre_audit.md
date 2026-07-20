# Structured-field Evaluation on an Independently Annotated Subset 建议主表：冻结 GT / pre-audit 结果

> **当前状态：论文主结果。** 后续使用前先阅读
> `[main]Structured-field Evaluation on an Independently Annotated Subset_current_decision_and_file_map.md`。本文件不使用查看模型结果后产生的
> GT、方向或同义词修正。

## 主表

**Table X. Surface-form-robust structured evaluation on the independently annotated subset.**

| Method | Hazard F1 ↑ | Category F1 ↑ | Dist. MAE ↓ | Dir. Adjacent ↑ | Dir. Exact ↑ |
|---|---:|---:|---:|---:|---:|
| Prompt-only (ZS) | 0.750 | 0.731 | 3.84 | **0.857** | **0.584** |
| Prompt-only (FS) | 0.742 | 0.732 | 2.48 | 0.578 | 0.398 |
| **Ours** | **0.771** | **0.761** | **1.37** | **0.857** | 0.481 |

## 固定计算口径

- 数据源固定为 `evaluation_results_pre_audit.json`，不使用实验结果产生后追加的 GT、方向或同义词修正。
- `Hazard F1` 是名称兼容匹配产生的 role-aware 障碍物检测 micro-F1。
- `Category F1` 为 detection-aware micro-F1：正确字段记 TP；required GT 的错误字段记 FP+FN；required GT 的字段缺失记 FN；未匹配预测记 FP；未匹配 required GT 记 FN。
- 上述 FN 只适用于 `required` GT；未匹配 `optional` GT 不扣分。匹配成功的 `optional` GT 可以贡献 TP。
- 根据老师意见，`Distance-bin F1` 和 `Direction-bin F1` 不进入论文主表。距离使用更直接的 `Dist. MAE`；方向使用 `Dir. Adjacent` 和 `Dir. Exact`。
- 这些保留指标仍基于结构化字段，不依赖 ALERT/GUIDE 的表面措辞，因此仍可用于回应 reviewer 对 surface-form bias 的核心担忧。
- `Dir. Exact = exact / (exact + adjacent + mirror + mismatch)`。
- `Dir. Adjacent = (exact + adjacent) / (exact + adjacent + mirror + mismatch)`；该列含 exact，不是 adjacent-only。
- `not_mentioned` 和 `uncertain` 不进入 Dir. Exact/Adjacent 的分母，但在 detection-aware Direction-bin F1 中作为字段缺失受到惩罚。

## 方向指标核验

| Method | Exact | Adjacent-only | Mismatch | Unavailable | Exact denominator | Dir. Adjacent | Dir. Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prompt-only (ZS) | 45 | 21 | 11 | 1 | 77 | `(45+21)/77 = 0.857` | `45/77 = 0.584` |
| Prompt-only (FS) | 33 | 15 | 35 | 9 | 83 | `(33+15)/83 = 0.578` | `33/83 = 0.398` |
| **Ours** | 37 | 29 | 11 | 19 | 77 | `(37+29)/77 = 0.857` | `37/77 = 0.481` |

此前出现的 `Dir. Adjacent = 0.854 / 0.600 / 0.871` 和
`Dir. Exact = 0.607 / 0.453 / 0.506` 来自 post-audit
`evaluation_results.json`。这些值应用了后续方向修正、GT 补标及同义词扩展，
不能与本表冻结的 pre-audit 字段 F1 混用。

旧 pre-audit `F1 = 0.536 / 0.582 / 0.608` 也不应继续作为主表 F1：
旧实现虽然声明 optional 未匹配不计 FN，却实际用全部 required+optional GT
作为 recall 分母，与标注角色定义冲突。本表已按 role-aware 口径重算。

## 可复现文件

- 重算脚本：`evaluation/recalculate_field_f1.py`
- 完整计数及非主表 Distance-bin/Direction-bin F1 审计结果：`field_f1_recalculation_pre_audit.json`
- 当前决策与完整文件索引：`[main]Structured-field Evaluation on an Independently Annotated Subset_current_decision_and_file_map.md`
