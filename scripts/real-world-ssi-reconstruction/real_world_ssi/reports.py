"""保存 Real-world SSI 的逐图审查产物与批次报告。

本模块不作模型推理，只把 Task 10 已确认的内存结果写成公开格式。raw depth 是相对
深度 NPZ，8-bit depth/mask/overlay 是 PNG，QC 与汇总是 UTF-8 JSON/Markdown；
全部复用 ``cache.atomic_write_file``，并在进入磁盘前统一脱敏。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image

from .cache import (
    artifact_path,
    atomic_write_file,
    atomic_write_json,
    atomic_write_text,
    redact_sensitive,
    safe_sample_parts,
)
from .schemas import BatchSummary, DepthResult, ImageSample, ObstacleMask


def write_qc_record(
    output_root: Path,
    sample_id: str,
    record: Mapping[str, Any],
) -> Path:
    """写 ``reports/<sample_id>.qc.json``，并强制使用已验证的 sample ID。

    ``record`` 可包含阶段诊断，但不能覆盖顶层 ``sample_id``。NaN、任意对象等不满足
    JSON 契约的值会在原子替换前失败，因此已有 QC 不会被半写文件破坏。
    """
    safe_sample_parts(sample_id)
    if not isinstance(record, Mapping):
        raise TypeError("QC record must be a mapping")
    path = artifact_path(Path(output_root) / "reports", sample_id, suffix=".qc.json")
    document = {**dict(record), "sample_id": sample_id}
    atomic_write_json(path, document)
    return path


def _outcome_document(outcome: Any) -> dict[str, Any]:
    return {
        "sample_id": outcome.sample_id,
        "status": outcome.status,
        "ssi_path": None if outcome.ssi_path is None else str(outcome.ssi_path),
        "qc_path": str(outcome.qc_path),
        "reason_codes": list(outcome.reason_codes),
    }


def write_batch_summary(
    output_root: Path,
    summary: BatchSummary,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """同时写稳定的机器可读 JSON 与中文人工审查 Markdown 汇总。"""
    if not isinstance(summary, BatchSummary):
        raise TypeError("summary must be a BatchSummary")
    if diagnostics is not None and not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping")
    for name, value in (
        ("succeeded", summary.succeeded),
        ("rejected", summary.rejected),
        ("failed", summary.failed),
        ("filtered", summary.filtered),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"summary {name} must be a non-negative integer")

    report_root = Path(output_root) / "reports"
    json_path = report_root / "batch_summary.json"
    markdown_path = report_root / "batch_summary.md"
    document = redact_sensitive(
        {
            "succeeded": summary.succeeded,
            "rejected": summary.rejected,
            "failed": summary.failed,
            "filtered": summary.filtered,
            "outcomes": [_outcome_document(item) for item in summary.outcomes],
            "diagnostics": dict(diagnostics or {}),
        }
    )
    atomic_write_json(json_path, document)

    # Markdown 只复述已经脱敏的 JSON 值，不直接插入异常对象或原始 API 正文。
    markdown_lines = [
        "# Real-world SSI 批次审查汇总",
        "",
        f"- 成功: {summary.succeeded}",
        f"- 筛除: {summary.rejected}",
        f"- 失败: {summary.failed}",
        "",
        "## 逐图结果",
        "",
        "| sample_id | status | reason_codes |",
        "|---|---|---|",
    ]
    markdown_lines.insert(5, f"- filtered: {summary.filtered}")
    for outcome in document["outcomes"]:
        sample_id = str(outcome["sample_id"]).replace("|", "\\|")
        status = str(outcome["status"]).replace("|", "\\|")
        reason_codes = ", ".join(str(item) for item in outcome["reason_codes"])
        markdown_lines.append(f"| {sample_id} | {status} | {reason_codes} |")
    if document["diagnostics"]:
        markdown_lines.extend(
            [
                "",
                "## 批次诊断（已脱敏）",
                "",
                "```json",
                json.dumps(
                    document["diagnostics"],
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                    sort_keys=True,
                ),
                "```",
            ]
        )
    atomic_write_text(markdown_path, "\n".join(markdown_lines) + "\n")
    return json_path, markdown_path


def _validated_depth(result: DepthResult) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, DepthResult):
        raise TypeError("depth result must be a DepthResult")
    raw = np.asarray(result.raw)
    depth_8bit = np.asarray(result.depth_8bit)
    if raw.dtype != np.float32 or raw.ndim != 2 or raw.size == 0:
        raise ValueError("raw depth artifact must be a non-empty float32 2D array")
    if not np.isfinite(raw).all():
        raise ValueError("raw depth artifact must contain only finite values")
    if depth_8bit.dtype != np.uint8 or depth_8bit.shape != raw.shape:
        raise ValueError("8-bit depth artifact must be uint8 with raw depth shape")
    return raw, depth_8bit


def _write_png(path: Path, image: Image.Image) -> None:
    """让 Pillow 写入匿名临时文件句柄，避免依赖临时文件的 ``.tmp`` 后缀推格式。"""
    atomic_write_file(path, lambda stream: image.save(stream, format="PNG"))


def write_depth_artifacts(
    output_root: Path, sample_id: str, result: DepthResult
) -> tuple[Path, Path]:
    """保存相对 raw depth NPZ 与论文公式使用的 8-bit PNG。"""
    safe_sample_parts(sample_id)
    raw, depth_8bit = _validated_depth(result)
    raw_path = artifact_path(
        Path(output_root) / "depth_raw", sample_id, suffix=".npz"
    )
    png_path = artifact_path(
        Path(output_root) / "depth_8bit", sample_id, suffix=".png"
    )

    def write_raw(stream: BinaryIO) -> None:
        np.savez_compressed(stream, raw=raw)

    atomic_write_file(raw_path, write_raw)
    _write_png(png_path, Image.fromarray(depth_8bit, mode="L"))
    return raw_path, png_path


def write_mask_artifacts(
    output_root: Path,
    sample: ImageSample,
    masks: tuple[ObstacleMask, ...],
) -> tuple[Path, ...]:
    """按样本原尺寸逐 annotation index 保存 0/255 二值 PNG。

    Task 10 必须传入 ``ImageSample``，不能只给 sample ID；这样每个 mask 都在落盘前
    与 ``(height, width)`` 严格核对，避免一批混合 shape 的文件被标成同一张图片。
    """
    if not isinstance(sample, ImageSample):
        raise TypeError("sample must be an ImageSample")
    sample_id = sample.sample_id
    safe_sample_parts(sample_id)
    if type(sample.width) is not int or sample.width < 1:
        raise ValueError("sample width must be a positive integer")
    if type(sample.height) is not int or sample.height < 1:
        raise ValueError("sample height must be a positive integer")
    indexes = tuple(mask.annotation_index for mask in masks)
    if len(indexes) != len(set(indexes)):
        raise ValueError("mask annotation_index values must be unique")
    paths: list[Path] = []
    for mask in masks:
        if type(mask.annotation_index) is not int or mask.annotation_index < 0:
            raise ValueError("mask annotation_index must be a non-negative integer")
        values = np.asarray(mask.mask)
        if values.dtype != np.bool_ or values.shape != (sample.height, sample.width):
            raise ValueError(
                "mask artifact must be bool with the exact sample image shape"
            )
        path = artifact_path(
            Path(output_root) / "masks",
            sample_id,
            suffix=f".annotation-{mask.annotation_index}.png",
        )
        _write_png(path, Image.fromarray(values.astype(np.uint8) * 255, mode="L"))
        paths.append(path)
    return tuple(paths)


def write_overlay_artifact(
    output_root: Path, sample_id: str, overlay: Image.Image
) -> Path:
    """保存 ``render_overlay`` 的 RGB PNG；模式转换不会改动调用方对象。"""
    safe_sample_parts(sample_id)
    if not isinstance(overlay, Image.Image) or overlay.width < 1 or overlay.height < 1:
        raise ValueError("overlay must be a non-empty PIL image")
    path = artifact_path(Path(output_root) / "overlays", sample_id, suffix=".png")
    _write_png(path, overlay.convert("RGB"))
    return path
