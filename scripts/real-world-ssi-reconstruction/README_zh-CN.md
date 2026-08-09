# Real-world SSI Reconstruction

[English](README.md) | 简体中文

将真实街景的 RGB 图片与 YOLO 标注转换为结构化场景信息（SSI）的单线程流水线。项目组合 Grounding DINO、SAM2、Depth Anything V2、SegFormer 与 MiniMax M3，输出逐图 SSI JSON、深度图、掩码、叠加图和质量审计记录。

本仓库只包含可复现源码与 Windows 启动脚本，不包含数据集、模型权重、API Key、缓存、运行结果或测试材料。

## 处理流程

1. 读取 YOLO 图片、标签与类别名称。
2. 过滤有效障碍物数量过多的场景。
3. 使用 Grounding DINO + SAM2 将 YOLO 框细化为障碍物掩码。
4. 使用 Depth Anything V2 估计相对深度。
5. 使用 SegFormer 检查天空区域，并执行保守的近景深度伪影检查。
6. 将原图和障碍物叠加图提交给 MiniMax M3，生成风险类别。
7. 写出 SSI、可视化结果、逐图 QC 与批次汇总。

图片按顺序单线程处理。本地视觉模型可使用 GPU，但不会同时处理多张图片；MiniMax 请求同样逐图发送，因此此版本适合重视稳定恢复与结果审查的长批次任务。

## 从 gpt-o4-mini 迁移到 MiniMax M3

项目早期分类阶段使用的是 OpenAI `gpt-o4-mini`（也常写作 `o4-mini`）。当前开源版本已将该阶段替换为 `MiniMax-M3`，其余本地视觉流程和 SSI 数据契约保持不变。

这是针对本项目的工程选择，主要原因是：

- 分类器需要同时理解原始 RGB 图片和带有编号、掩码及距离信息的叠加图；本项目部署中使用 MiniMax M3 完成了这一双图分类流程。
- M3 输出会经过严格 JSON 校验，只接受与输入障碍物 ID 完全对应的四类结果：`Common Obstacle`、`Pitfall Hazard`、`Upper-body Hazard` 和 `Other`。
- 请求启用了自适应 thinking：`{"thinking": {"type": "adaptive"}}`，用于处理边界模糊或需要结合空间关系判断的障碍物。
- MiniMax 可通过 OpenAI SDK 兼容接口接入，因此迁移只替换分类服务，不需要重写缓存、质量筛选和 SSI 导出层。
- 在项目实际使用的服务可达性和配额条件下，MiniMax M3 更便于持续运行长批次。

这段迁移说明不表示 MiniMax M3 在所有任务上都优于 OpenAI 模型。模型、接口能力与套餐可能变化；请以 [MiniMax OpenAI 兼容接口文档](https://platform.minimax.io/docs/api-reference/text-openai-api)和你所使用账户的实际可用模型为准。

## 环境要求

- Windows 10/11
- Python 3.11
- 推荐使用支持 CUDA 的 NVIDIA GPU；CPU 可以运行，但深度、检测和分割阶段会明显更慢
- 足够的磁盘空间保存 Hugging Face 模型和逐图中间结果
- 可访问 Hugging Face 与 MiniMax API 的网络
- 有效的 MiniMax API Key，并且账户可调用 `MiniMax-M3`

依赖版本已固定在 `requirements.txt`。首次运行会下载以下模型：

- `IDEA-Research/grounding-dino-tiny`
- `facebook/sam2.1-hiera-tiny`
- `depth-anything/Depth-Anything-V2-Small-hf`
- `nvidia/segformer-b0-finetuned-ade-512-512`

## 安装

克隆仓库后，在项目目录双击：

```text
setup_windows.bat
```

脚本会在项目目录创建 `.venv` 并安装固定版本依赖。安装结束时会显示 CUDA 是否可用以及 GPU 名称。

也可以手动安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

验证 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 配置 MiniMax

复制示例配置：

```powershell
Copy-Item config.example.json config.json
```

编辑 `config.json`：

```json
{
  "base_url": "https://api.minimaxi.com/v1",
  "api_key": "YOUR_REAL_MINIMAX_API_KEY",
  "model": "MiniMax-M3"
}
```

`config.json` 已加入 `.gitignore`，不要把真实 API Key 提交到 GitHub。流水线不会把 API Key 写入 SSI、QC、缓存或批次报告。

## 输入数据格式

支持简单 YOLO 目录：

```text
dataset-root/
├─ images/
│  ├─ img_00001.jpg
│  └─ img_00002.jpg
├─ labels/
│  ├─ img_00001.txt
│  └─ img_00002.txt
└─ classes.txt
```

每个标签文件使用标准 YOLO 行格式：

```text
class_id x_center y_center width height
```

坐标和宽高必须归一化到 `[0, 1]`。`classes.txt` 每行对应一个类别名称，行号从 0 开始。

也支持含 `dataset.yaml` 的标准布局：

```text
dataset-root/
├─ dataset.yaml
├─ images/train/...
├─ images/val/...
├─ labels/train/...
└─ labels/val/...
```

图片与标签必须具有相同的相对路径和文件名 stem。

## 运行

最简单的方式是运行：

```text
run_windows.bat "数据集目录" "输出目录"
```

也可以显式指定 MiniMax 配置和设备：

```text
run_windows.bat "数据集目录" "输出目录" "MiniMax配置文件" cuda
```

第四个参数可设为 `cuda`、`cpu` 或 `auto`；批处理脚本默认使用 `cuda`。

直接运行 Python：

```powershell
.\.venv\Scripts\python.exe run_pipeline.py `
  --dataset-root "数据集目录" `
  --output-root "输出目录" `
  --minimax-config ".\config.json" `
  --device cuda
```

仅检查数据集，不加载模型、下载权重或调用 MiniMax：

```powershell
.\.venv\Scripts\python.exe run_pipeline.py `
  --dataset-root "数据集目录" `
  --output-root "输出目录" `
  --dry-run
```

查看全部参数：

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --help
```

## 进度与中断恢复

正常运行时，终端面板会持续显示：

- 总体完成数量和百分比
- 当前图片 ID 与障碍物数量
- 当前所在阶段及阶段耗时
- 成功、风险、警告、筛除和失败计数
- 平均处理速度、已运行时间和预计剩余时间
- GPU 利用率与显存占用（可读取时）
- 最近完成图片及单图耗时

可以随时按 `Ctrl+C` 停止。逐图缓存和正式 SSI 使用原子写入；再次使用相同输入、输出目录和参数运行时，默认 `--resume` 会复用已完成且仍有效的阶段。中断时正在处理的那一张图可能需要重新计算，但已完整写出的图片不会丢失。

如需关闭恢复：

```text
--no-resume
```

如需忽略既有结果并重新计算：

```text
--overwrite
```

## 输入过滤和质量规则

### 障碍物数量

默认只处理“有效障碍物数量小于 15”的场景，也就是最多 14 个需要风险判断的对象。以下白名单类别不计入数量过滤，并直接写为 `Other`，不会交给 MiniMax 强行归入三种风险类别：

- `braille_block`
- `crosswalk`
- `traffic_light`
- `signal_button`
- `signal_red`
- `signal_blue`
- `white_line`

### 天空不足

SegFormer 预测的天空面积低于全图 `0.5%` 时，样本直接记为失败并停止后续 MiniMax 分类，以避免为明显不符合输入要求的样本消耗 API token。

### 近景深度伪影

只有同时满足以下条件才标记 `near_artifact`：

1. 8-bit 深度值不低于 `250` 的极近像素非空；
2. 极近像素面积不超过全图 `1%`；
3. 其中最大四邻域连通区域占极近像素至少 `80%`；
4. RGB 边缘与深度边缘的相关性低于 `0.20`。

前三项只建立“小而高度集中的极近区域”候选，第四项用于确认该结构缺少 RGB 边缘支持。这样可减少正常近景物体被误判为深度伪影。

这些阈值集中在 `config.py`，如需改变实验协议，请修改配置并使用新的输出目录，避免与旧缓存混合。

## 输出目录

典型输出包括：

```text
output-root/
├─ ssi/                 # 正式 SSI JSON
├─ reports/             # 逐图 QC 与 batch_summary.json
├─ depth_8bit/          # SSI 使用的 8-bit 深度图
├─ depth_raw/           # 可选的原始浮点深度
├─ masks/               # 障碍物掩码
├─ overlays/            # 人工审查叠加图
├─ api_responses/       # 已验证、脱敏的类别映射副本
└─ cache/               # 恢复运行所需的逐阶段缓存
```

失败或被筛除的图片不会留下正式 SSI，但会保留 QC 记录和原因码，方便定位问题。

## 主要配置

`config.py` 集中保存模型 ID、质量阈值和输出开关。发布版默认值包括：

- 设备：`auto`（`run_windows.bat` 显式传入 `cuda`）
- 运行阶段：`full`
- 恢复：开启
- MiniMax 模型：`MiniMax-M3`
- MiniMax 图片细节：`high`
- MiniMax 总尝试次数：3
- MiniMax 单次超时：120 秒
- 有效障碍物上限：严格小于 15
- 天空最低比例：0.005
- 质量模式：`reject`

## 注意事项

- 首次运行下载模型会花费较长时间，之后从 `model_cache` 复用。
- MiniMax 网络等待期间 GPU 利用率可能接近 0%，这是单线程版本的正常行为。
- 进程不会把所有图片或所有中间数组长期保存在内存中；每张图完成后进入下一张，缓存写入磁盘。
- 输出属于自动生成标注，建议抽样人工审查后再用于训练或正式研究结果。
- 本项目不是道路安全系统，不能替代真实环境中的安全评估。
