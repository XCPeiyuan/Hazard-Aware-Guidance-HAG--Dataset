"""论文质量筛选：SegFormer 天空掩码与集中近景伪影检测。

本模块位于深度估计之后、类别分配和 SSI 写出之前。它只把 8-bit 相对深度用于
论文规定的质量指标，不把像素值解释为米制距离；天空服务则把 SegFormer 的低分辨率
语义 logits 恢复到 PIL 原图的 ``(height, width)`` 后再生成布尔掩码。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import sobel

from .model_registry import ModelRegistry
from .schemas import (
    PipelineConfig,
    QualityDecision,
    QualityMetrics,
    QualityReason,
)


class ConsistencyFilterError(RuntimeError):
    """质量图、SegFormer 输出或标签映射不满足可审计契约。"""


def _two_dimensional_array(
    values: Any,
    *,
    name: str,
    expected_dtype: np.dtype[Any],
) -> NDArray[Any]:
    """验证非空二维图及精确 dtype；不隐式转换或压缩 batch/channel 轴。"""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ConsistencyFilterError(
            f"{name} must be a non-empty two-dimensional map"
        ) from exc
    if array.ndim != 2 or array.size == 0:
        raise ConsistencyFilterError(f"{name} must be a non-empty two-dimensional map")
    if array.dtype != expected_dtype:
        raise ConsistencyFilterError(
            f"{name} must have dtype {expected_dtype.name}"
        )
    return array


def _largest_component_size(mask: NDArray[np.bool_]) -> int:
    """返回四邻域最大连通区域像素数；迭代遍历避免大图递归栈溢出。"""
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=np.bool_)
    largest = 0

    # 论文指标只需要最大区域面积，不需要保留标签图；逐个种子 flood-fill 可避免
    # 为公共脚本额外引入 SciPy/OpenCV 依赖。
    for start_y, start_x in np.argwhere(mask):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue

        visited[y, x] = True
        stack = [(y, x)]
        component_size = 0
        while stack:
            current_y, current_x = stack.pop()
            component_size += 1
            for neighbor_y, neighbor_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and mask[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    stack.append((neighbor_y, neighbor_x))
        largest = max(largest, component_size)
    return largest


def near_artifact_metrics(
    depth_8bit: NDArray[Any], near_threshold: int
) -> tuple[float, float]:
    """计算极近像素占比和其中最大四邻域连通区域的集中度。

    ``depth_8bit >= near_threshold`` 是重建默认值规定的极近像素。第一个返回值以
    整张图像素数为分母；第二个返回值以极近像素数为分母。没有极近像素时两者均为
    零，因此空掩码本身不会被判为近景伪影。
    """
    depth = _two_dimensional_array(
        depth_8bit,
        name="8-bit depth",
        expected_dtype=np.dtype(np.uint8),
    )
    near_mask = np.asarray(depth >= near_threshold, dtype=np.bool_)

    near_pixels = int(np.count_nonzero(near_mask))
    near_fraction = near_pixels / int(near_mask.size)
    if near_pixels == 0:
        return 0.0, 0.0

    largest_component_pixels = _largest_component_size(near_mask)
    near_concentration = largest_component_pixels / near_pixels
    return float(near_fraction), float(near_concentration)


def depth_rgb_edge_correlation(
    image: Image.Image, depth_8bit: NDArray[Any]
) -> float:
    """Compare RGB and depth structural edges in the same pixel coordinate system."""
    depth = _two_dimensional_array(
        depth_8bit,
        name="8-bit depth",
        expected_dtype=np.dtype(np.uint8),
    )
    if not isinstance(image, Image.Image):
        raise ConsistencyFilterError("image must be a PIL image")
    rgb = np.asarray(image.convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ConsistencyFilterError("image must have three RGB channels")
    if rgb.shape[:2] != depth.shape:
        raise ConsistencyFilterError("image and depth must have the same shape")

    grayscale = (
        0.299 * rgb[..., 0].astype(np.float64)
        + 0.587 * rgb[..., 1].astype(np.float64)
        + 0.114 * rgb[..., 2].astype(np.float64)
    )
    depth_float = depth.astype(np.float64)
    rgb_gradient = np.hypot(
        sobel(grayscale, axis=0, mode="reflect"),
        sobel(grayscale, axis=1, mode="reflect"),
    ).reshape(-1)
    depth_gradient = np.hypot(
        sobel(depth_float, axis=0, mode="reflect"),
        sobel(depth_float, axis=1, mode="reflect"),
    ).reshape(-1)

    rgb_centered = rgb_gradient - float(rgb_gradient.mean())
    depth_centered = depth_gradient - float(depth_gradient.mean())
    rgb_norm = float(np.linalg.norm(rgb_centered))
    depth_norm = float(np.linalg.norm(depth_centered))
    if rgb_norm == 0.0 or depth_norm == 0.0:
        return 1.0
    correlation = float(
        np.dot(rgb_centered, depth_centered) / (rgb_norm * depth_norm)
    )
    return float(np.clip(correlation, -1.0, 1.0))

def decide_quality(
    metrics: QualityMetrics, config: PipelineConfig
) -> QualityDecision:
    """应用论文阈值和 ``reject / warn / off`` 模式，保留全部发现原因。

    阈值等号属于论文规则：极近面积必须非空且 ``<=`` 上限，集中度 ``>=`` 下限；
    天空仅在严格小于下限时缺失。``off`` 不改变指标和原因，只把状态保持为 passed，
    以便后续 QC 仍能审计本来会命中的分支。
    """
    # 配置对象通常已由 build_config() 验证，但该函数也是公开的纯判定入口；必须在
    # “没有原因则 passed”的早退之前独立拒绝未知模式。
    if config.filter_mode not in {"reject", "warn", "off"}:
        raise ConsistencyFilterError(
            "filter_mode must be one of: reject, warn, off"
        )

    # 三个比例将进入 QC/JSON。NaN 和无穷虽能参与 Python 比较，却会绕过判定并产生
    # 非标准 JSON；因此先统一验证全部字段，再执行任何阈值分支。
    for name, value in (
        ("near_fraction", metrics.near_fraction),
        ("near_concentration", metrics.near_concentration),
        ("sky_fraction", metrics.sky_fraction),
    ):
        try:
            valid = math.isfinite(value) and 0.0 <= value <= 1.0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ConsistencyFilterError(
                f"{name} must be finite and within [0, 1]"
            )

    if not isinstance(metrics.near_candidate, (bool, np.bool_)):
        raise ConsistencyFilterError("near_candidate must be a boolean")
    try:
        correlation_valid = (
            math.isfinite(metrics.depth_rgb_edge_correlation)
            and -1.0 <= metrics.depth_rgb_edge_correlation <= 1.0
        )
    except (TypeError, ValueError, OverflowError):
        correlation_valid = False
    if not correlation_valid:
        raise ConsistencyFilterError(
            "depth_rgb_edge_correlation must be finite and within [-1, 1]"
        )

    expected_near_candidate = bool(
        metrics.near_fraction > 0.0
        and metrics.near_fraction <= config.max_small_near_area_ratio
        and metrics.near_concentration >= config.min_near_concentration
    )
    if bool(metrics.near_candidate) != expected_near_candidate:
        raise ConsistencyFilterError(
            "near_candidate does not match near-area metrics"
        )

    reasons: list[QualityReason] = []
    if (
        metrics.near_candidate
        and metrics.depth_rgb_edge_correlation
        < config.min_depth_rgb_edge_correlation
    ):
        reasons.append(
            QualityReason(
                code="near_artifact",
                message_zh="极近像素面积较小且高度集中，疑似近景深度伪影。",
            )
        )
    if metrics.sky_fraction < config.min_sky_area_ratio:
        reasons.append(
            QualityReason(
                code="sky_missing",
                message_zh="SegFormer 天空区域低于论文重建阈值。",
            )
        )

    if not reasons or config.filter_mode == "off":
        status = "passed"
    elif config.filter_mode == "warn":
        status = "warning"
    elif config.filter_mode == "reject":
        status = "rejected"
    return QualityDecision(
        status=status,
        metrics=metrics,
        reasons=tuple(reasons),
    )


def evaluate_quality(
    image: Image.Image,
    depth_8bit: NDArray[Any],
    sky_mask: NDArray[np.bool_],
    config: PipelineConfig,
) -> QualityDecision:
    """Compute quality metrics and apply the conservative quality policy."""
    depth = _two_dimensional_array(
        depth_8bit,
        name="8-bit depth",
        expected_dtype=np.dtype(np.uint8),
    )
    sky = _two_dimensional_array(
        sky_mask,
        name="sky mask",
        expected_dtype=np.dtype(np.bool_),
    )
    if depth.shape != sky.shape:
        raise ConsistencyFilterError("depth and sky mask must have the same shape")

    near_fraction, near_concentration = near_artifact_metrics(
        depth, near_threshold=config.near_depth_threshold
    )
    near_candidate = bool(
        near_fraction > 0.0
        and near_fraction <= config.max_small_near_area_ratio
        and near_concentration >= config.min_near_concentration
    )
    sky_fraction = float(np.count_nonzero(sky) / sky.size)
    correlation = depth_rgb_edge_correlation(image, depth)
    return decide_quality(
        QualityMetrics(
            near_fraction=near_fraction,
            near_concentration=near_concentration,
            sky_fraction=sky_fraction,
            near_candidate=near_candidate,
            depth_rgb_edge_correlation=correlation,
        ),
        config,
    )

def _canonical_class_id(class_id: Any) -> int:
    """验证并规范化 Transformers ``id2label`` 的单个 key。"""
    error_message = (
        "SegFormer sky class ID must be a non-negative integer "
        "or canonical decimal string"
    )
    if isinstance(class_id, bool):
        raise ConsistencyFilterError(error_message)
    if isinstance(class_id, Integral):
        normalized_id = int(class_id)
    elif isinstance(class_id, str) and re.fullmatch(
        r"(?:0|[1-9][0-9]*)", class_id, flags=re.ASCII
    ):
        normalized_id = int(class_id)
    else:
        raise ConsistencyFilterError(error_message)
    if normalized_id < 0:
        raise ConsistencyFilterError(error_message)
    return normalized_id


def _exact_sky_class_id(id2label: Any) -> int:
    """从 Transformers 配置映射中查找唯一的大小写不敏感精确 ``sky`` 标签。"""
    if not isinstance(id2label, Mapping):
        raise ConsistencyFilterError("SegFormer config.id2label must be a mapping")

    matches: list[int] = []
    for class_id, label in id2label.items():
        # Transformers 从 JSON 加载后通常给出 int key；测试或其他配置来源也可给出
        # 规范十进制字符串。所有条目都先验证，不能让非 sky 条目的异常 key 潜伏到 QC。
        normalized_id = _canonical_class_id(class_id)
        if isinstance(label, str) and label.casefold() == "sky":
            matches.append(normalized_id)
    if len(matches) != 1:
        raise ConsistencyFilterError(
            "SegFormer config.id2label must contain exactly one exact label 'sky'"
        )
    return matches[0]


class SegFormerSkyService:
    """通过集中 ``ModelRegistry`` 执行一次 SegFormer 天空语义分割。"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def predict_sky_mask(self, image: Image.Image) -> NDArray[np.bool_]:
        """返回 PIL 原图 ``(height, width)`` 的布尔天空掩码。

        Transformers 5.10.2 的 SegFormer 输出是 ``(batch, classes, H/4, W/4)``
        logits。依照本机 image processor 的官方后处理源码，先用 bilinear 和
        ``align_corners=False`` 恢复原尺寸，再在类别轴 argmax。
        """
        model = self.registry.sky_model
        model_config = getattr(model, "config", None)
        sky_class_id = _exact_sky_class_id(
            getattr(model_config, "id2label", None)
        )

        processor = self.registry.sky_processor
        inputs = processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.registry.device)
        with torch.inference_mode():
            outputs = model(**inputs)

        logits = getattr(outputs, "logits", None)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 4
            or logits.shape[0] != 1
            or any(dimension <= 0 for dimension in logits.shape[1:])
        ):
            raise ConsistencyFilterError(
                "SegFormer logits must have positive shape "
                "(1, classes, height, width)"
            )
        if not logits.is_floating_point():
            raise ConsistencyFilterError("SegFormer logits must use a floating dtype")
        if not bool(torch.isfinite(logits).all().item()):
            raise ConsistencyFilterError("SegFormer logits must all be finite")
        if not 0 <= sky_class_id < logits.shape[1]:
            raise ConsistencyFilterError(
                "SegFormer sky class ID is outside the logits class dimension"
            )

        resized_logits = torch_functional.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        class_map = resized_logits[0].argmax(dim=0)
        return (
            (class_map == sky_class_id)
            .detach()
            .cpu()
            .numpy()
            .astype(np.bool_, copy=False)
        )
