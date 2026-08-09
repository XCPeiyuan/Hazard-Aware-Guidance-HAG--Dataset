"""Portable command-line entry point for the single-thread SSI pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import config
from real_world_ssi.pipeline import run_batch
from real_world_ssi.progress import ConsoleProgressReporter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO-format real-world dataset to SSI (single-thread)."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="YOLO dataset root containing images, labels, and classes.txt.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory; it must not be inside the dataset root.",
    )
    parser.add_argument(
        "--minimax-config",
        type=Path,
        default=None,
        help="Credential JSON path (default: config.json beside this script).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Local model device (default: auto).",
    )
    parser.add_argument(
        "--run-stage",
        choices=("full", "local_only", "classify_only", "export_only"),
        default="full",
        help="Pipeline stage to execute (default: full).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse valid per-image caches (default: enabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing artifacts instead of reusing them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and enumerate inputs without loading models or calling MiniMax.",
    )
    parser.add_argument(
        "--allow-bbox-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a YOLO box mask if grounded mask extraction fails (default: enabled).",
    )
    return parser


def _cuda_device_name() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except (ImportError, RuntimeError):
        return None


def _image_count(dataset_root: Path) -> int:
    images_root = dataset_root / "images"
    if not images_root.is_dir():
        return 0
    return sum(
        1
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    minimax_config = (
        args.minimax_config
        if args.minimax_config is not None
        else Path(__file__).resolve().parent / "config.json"
    )
    cfg = replace(
        config.build_config(),
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        minimax_config_path=minimax_config,
        device=args.device,
        run_stage=args.run_stage,
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        allow_bbox_prompt_fallback=args.allow_bbox_fallback,
    ).validate()

    cuda_name = _cuda_device_name()
    if cfg.device == "cuda" and cuda_name is None and not cfg.dry_run:
        print("ERROR: --device cuda was requested, but CUDA is unavailable.")
        return 2

    print("Single-thread SSI preflight")
    print(f"Images: {_image_count(cfg.dataset_root):,}")
    print(f"Input: {cfg.dataset_root.resolve()}")
    print(f"Output: {cfg.output_root.resolve()}")
    print(f"Device: {cfg.device} ({cuda_name or 'CUDA unavailable'})")
    print(f"Stage: {cfg.run_stage}")
    print(f"Resume: {'enabled' if cfg.resume else 'disabled'}")
    print(f"Dry run: {'enabled' if cfg.dry_run else 'disabled'}")
    print(f"Effective-object threshold: fewer than {cfg.max_object_count}")

    reporter = None if cfg.dry_run else ConsoleProgressReporter(cfg.output_root)
    summary = run_batch(cfg, progress=reporter)
    print(summary.to_chinese_text())
    print(f"Filtered: {summary.filtered}")
    return 1 if cfg.fail_batch_if_any_error and summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
