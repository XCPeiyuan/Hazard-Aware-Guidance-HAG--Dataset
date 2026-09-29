# Real-world SSI Reconstruction

English | [简体中文](README_zh-CN.md)

A single-thread pipeline that converts real-world street-scene RGB images and YOLO annotations into Structured Scene Information (SSI). It combines Grounding DINO, SAM2, Depth Anything V2, SegFormer, and MiniMax M3 to produce per-image SSI JSON, depth maps, masks, review overlays, and quality-control records.

This directory contains source code and Windows launch scripts. Dataset files and model weights must be supplied or downloaded separately.

## Pipeline

1. Read YOLO images, labels, and class names.
2. Filter scenes that contain too many effective obstacles.
3. Refine YOLO boxes into obstacle masks with Grounding DINO and SAM2.
4. Estimate relative depth with Depth Anything V2.
5. Check sky coverage with SegFormer and apply a conservative near-field depth-artifact rule.
6. Send the original image and the numbered obstacle overlay to MiniMax M3 for risk categorization.
7. Write SSI, review visuals, per-image QC, and a batch summary.

Images are processed sequentially. Local vision inference can use the GPU, but this version never processes multiple images at the same time. MiniMax requests are also sent one image at a time. This design favors stable recovery and auditable results during long batch runs.

## Implementation scope

This utility currently configures `MiniMax-M3` for category assignment. It is a maintained implementation, not an exact snapshot of the model services and settings used in the manuscript experiments. Confirm that your configured endpoint supports the required image inputs.

For the fixed automatic-distance experiment, use [automatic_step_distance_check](../automatic_step_distance_check/README.md), which retains the implementation used for that check.

## Requirements

- Windows 10 or Windows 11
- Python 3.11
- An NVIDIA CUDA GPU is strongly recommended. CPU execution is supported but is substantially slower during detection, segmentation, and depth estimation.
- Enough disk space for Hugging Face models and per-image intermediate artifacts
- Network access to Hugging Face and the MiniMax API
- A valid MiniMax API key with access to `MiniMax-M3`

Runtime versions are pinned in `requirements.txt`. The first run downloads these models:

- `IDEA-Research/grounding-dino-tiny`
- `facebook/sam2.1-hiera-tiny`
- `depth-anything/Depth-Anything-V2-Small-hf`
- `nvidia/segformer-b0-finetuned-ade-512-512`

## Installation

From this directory, run:

```text
setup_windows.bat
```

The script creates `.venv` inside the project directory and installs the pinned runtime dependencies. When installation finishes, it reports whether CUDA is available and prints the detected GPU name.

Manual installation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify GPU access:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Configure MiniMax

Copy the example configuration:

```powershell
Copy-Item config.example.json config.json
```

Edit `config.json`:

```json
{
  "base_url": "https://api.minimaxi.com/v1",
  "api_key": "YOUR_REAL_MINIMAX_API_KEY",
  "model": "MiniMax-M3"
}
```

Keep `config.json` private; it contains your API credentials.

## Input Dataset Format

The simple YOLO layout is supported:

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

Each label line uses the standard YOLO format:

```text
class_id x_center y_center width height
```

Coordinates, width, and height must be normalized to `[0, 1]`. Each line in `classes.txt` defines one class name; line numbering starts at class ID 0.

A standard layout with `dataset.yaml` is also supported:

```text
dataset-root/
├─ dataset.yaml
├─ images/train/...
├─ images/val/...
├─ labels/train/...
└─ labels/val/...
```

Every image and label must have the same relative path and filename stem.

## Run

The simplest command is:

```text
run_windows.bat "PATH_TO_DATASET" "PATH_TO_OUTPUT"
```

You can explicitly provide the MiniMax configuration and local inference device:

```text
run_windows.bat "PATH_TO_DATASET" "PATH_TO_OUTPUT" "PATH_TO_MINIMAX_CONFIG" cuda
```

The fourth argument can be `cuda`, `cpu`, or `auto`. The batch script defaults to `cuda`.

Run the Python entry point directly:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py `
  --dataset-root "PATH_TO_DATASET" `
  --output-root "PATH_TO_OUTPUT" `
  --minimax-config ".\config.json" `
  --device cuda
```

Validate and enumerate the dataset without loading models, downloading weights, or calling MiniMax:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py `
  --dataset-root "PATH_TO_DATASET" `
  --output-root "PATH_TO_OUTPUT" `
  --dry-run
```

Show all command-line options:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --help
```

## Progress and Resuming an Interrupted Run

During a normal run, the console dashboard displays:

- Overall completion count and percentage
- Current image ID and object count
- Current stage and stage elapsed time
- Safe, hazard, warning, filtered, and failed counts
- Average processing speed, elapsed time, and estimated time remaining
- GPU utilization and VRAM usage when available
- Recently completed images and per-image elapsed time

Press `Ctrl+C` at any time to stop. Per-image caches and formal SSI files are written atomically. Run the same command again with the same input, output directory, and parameters to resume. The default `--resume` behavior reuses completed stages whose caches are still valid.

The image being processed at the moment of interruption may need to be recomputed. Images whose artifacts were fully written are preserved.

Disable cache reuse:

```text
--no-resume
```

Ignore existing artifacts and recompute:

```text
--overwrite
```

## Input Filtering and Quality Rules

### Object Count

By default, the pipeline processes only scenes with fewer than 15 effective obstacles. In other words, a scene may contain at most 14 objects that require risk classification.

The following deterministic whitelist classes do not count toward the input filter. They are assigned directly to `Other` and are not forced into one of the three risk categories by MiniMax:

- `braille_block`
- `crosswalk`
- `traffic_light`
- `signal_button`
- `signal_red`
- `signal_blue`
- `white_line`

### Insufficient Sky Area

If SegFormer predicts sky over less than `0.5%` of the image, the sample is marked as failed and MiniMax classification is skipped. This prevents API tokens from being spent on scenes that clearly fail the input-quality requirement.

### Near-field Depth Artifacts

A sample is marked with `near_artifact` only when all four conditions are true:

1. At least one extreme-near pixel has an 8-bit depth value of `250` or greater.
2. Extreme-near pixels cover no more than `1%` of the image.
3. The largest four-connected component contains at least `80%` of all extreme-near pixels.
4. RGB-edge and depth-edge correlation is below `0.20`.

The first three conditions identify a small, highly concentrated near-field candidate. The fourth condition checks whether that depth structure lacks matching RGB-edge support. The fourth condition requires evidence of disagreement between RGB and depth edges.

All thresholds are centralized in `config.py`. If you change the experimental protocol, use a new output directory to avoid mixing results with caches created under older settings.

## Output Layout

A typical output directory contains:

```text
output-root/
├─ ssi/                 # Formal SSI JSON
├─ reports/             # Per-image QC and batch_summary.json
├─ depth_8bit/          # 8-bit depth maps used by SSI
├─ depth_raw/           # Optional raw floating-point depth
├─ masks/               # Per-obstacle masks
├─ overlays/            # Human-review overlays
├─ api_responses/       # Validated, sanitized category mappings
└─ cache/               # Per-stage caches used for resuming
```

Failed or filtered images do not keep a formal SSI file, but they retain QC records and reason codes for auditing.

The `run_pipeline.py` entry point enables bounding-box mask fallback by default, overriding the base configuration. Pass `--no-allow-bbox-fallback` to disable it. Fallback behavior must be kept consistent when comparing runs.

## Main Configuration

`config.py` centralizes model IDs, quality thresholds, and artifact switches. Important release defaults include:

- Device: `auto` (`run_windows.bat` explicitly passes `cuda`)
- Run stage: `full`
- Resume: enabled
- MiniMax model: `MiniMax-M3`
- MiniMax image detail: `high`
- MiniMax total attempts: 3
- MiniMax timeout per request: 120 seconds
- Effective-object threshold: strictly fewer than 15
- Minimum sky fraction: 0.005
- Quality mode: `reject`

## Notes

- The first model download can take a long time. Later runs reuse `model_cache`.
- GPU utilization may fall close to zero while the single-thread pipeline waits for MiniMax. This is expected.
- The process does not retain every image or every intermediate array in memory. It completes one image, writes caches to disk, and then advances to the next image.
- These are automatically generated annotations. Review a representative sample before using them for training or reporting research results.
- This project is not a road-safety system and must not replace real-world safety assessment.
