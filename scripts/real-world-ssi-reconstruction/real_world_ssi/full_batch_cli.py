"""VIDVIP 全量 GPU 批处理入口：执行 CUDA 预检并启动详细进度面板。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import config

from .pipeline import run_batch
from .progress import ConsoleProgressReporter


def _cuda_device_name() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError):
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full VIDVIP SSI pipeline")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device_name = _cuda_device_name()
    if device_name is None:
        print("ERROR: CUDA is unavailable; full batch was not started.")
        return 2

    cfg = replace(
        config.build_config(),
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        device="cuda",
        resume=True,
        overwrite=False,
        run_stage="full",
        allow_bbox_prompt_fallback=True,
    )
    image_count = sum(1 for _ in (args.dataset_root / "images").glob("*.jpg"))
    print("VIDVIP full-batch preflight")
    print(f"Images: {image_count:,}")
    print(f"Output: {args.output_root}")
    print(f"GPU: {device_name}")
    print("Resume: enabled")
    print(f"Object threshold: fewer than {cfg.max_object_count}")

    reporter = ConsoleProgressReporter(args.output_root)
    summary = run_batch(cfg, progress=reporter)
    print(summary.to_chinese_text())
    return 1 if cfg.fail_batch_if_any_error and summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
