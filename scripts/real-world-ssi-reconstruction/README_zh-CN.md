# Real-world SSI Reconstruction

[English](README.md) | 简体中文

此工具将 RGB 图像与 YOLO 标注转换为结构化场景信息（SSI），使用 Grounding DINO、SAM2、Depth Anything V2 和 SegFormer 完成本地视觉处理，再调用配置的 MiniMax 服务进行类别判断。图像按顺序处理，输出 SSI、深度图、掩码和质量检查记录。

## 实现范围

当前工具配置的分类模型为 `MiniMax-M3`。它是持续维护的实现，不等同于论文实验使用的全部模型服务与参数快照。请确认自己的接口支持所需的图像输入。论文新增的固定距离映射核查使用[单独留档的实现](../automatic_step_distance_check/README.md)。

## 安装与输入

在本目录运行 `setup_windows.bat` 安装依赖。需要 Python 3.11；本地视觉模型可使用 CUDA。依赖版本见 `requirements.txt`。

输入目录包含 `images/`、`labels/` 以及 `classes.txt`，或使用 `dataset.yaml` 指定类别。标签行为 `class_id x_center y_center width height`，坐标归一化至 `[0, 1]`。图像与标签的相对路径和文件名应对应。

复制 `config.example.json` 为 `config.json`，填写自己的服务地址、模型和 API key。该文件含凭据，不要上传。

## 运行

```text
run_windows.bat "PATH_TO_DATASET" "PATH_TO_OUTPUT" "PATH_TO_CONFIG" cuda
```

输出目录不得位于输入数据目录内。直接调用 Python 时可先检查输入：

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --dataset-root "PATH_TO_DATASET" --output-root "PATH_TO_OUTPUT" --dry-run
```

移除 `--dry-run` 后执行处理；可通过 `--minimax-config` 指定凭据文件，通过 `--device` 选择 `cuda`、`cpu` 或 `auto`。默认复用有效缓存；更改实验设置时应使用新的输出目录。

## 关键参数

- 有效障碍物数量必须小于 15；白名单类别不计入此限制。
- 天空区域比例下限为 0.005。
- 近场异常规则结合极近像素的面积、集中程度及 RGB/深度边缘相关性，具体实现见 `config.py` 和质量检查代码。
- `run_pipeline.py` 默认启用边界框掩码回退，会覆盖基础配置。使用 `--no-allow-bbox-fallback` 可禁用；比较实验时必须保持此设置一致。

输出包括 `ssi/`、`reports/`、`depth_8bit/`、`depth_raw/`、`masks/`、`overlays/` 和缓存。详细参数、输入布局和恢复方式见[英文说明](README.md)。自动生成的结果需要质量检查，不构成真实导航安全保证。
