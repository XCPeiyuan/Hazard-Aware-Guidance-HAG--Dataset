"""Depth Anything V2 相对深度适配与论文步数映射。

本模块位于真实图像 SSI 重建的深度阶段：模型先产生保持空间方向的浮点
相对深度，再按整图 min-max 归一化成 SSI 所需的 8 位图。默认 Transformers
checkpoint 的输出在适配器中明确声明为“数值越大越近”，不会依据模型名猜测
极性，也不会把相对深度或由它计算的步数描述为米制距离。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from numpy.typing import NDArray
from PIL import Image

from .geometry import round_half_up
from .model_registry import ModelRegistry
from .schemas import DepthResult


class DepthEstimationError(RuntimeError):
    """深度模型输出或论文映射输入不满足可审计数值契约。"""


def _numeric_2d_array(values: Any, *, name: str) -> NDArray[np.float64]:
    """转换二维数值图；不在此处静默压缩 batch/channel 维。"""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DepthEstimationError(f"{name} must be a numeric two-dimensional map") from exc
    if array.ndim != 2 or array.size == 0:
        raise DepthEstimationError(f"{name} must be a non-empty two-dimensional map")
    return array


def scale_depth_to_uint8(raw: NDArray[Any]) -> NDArray[np.uint8]:
    """把“数值越大越近”的整图浮点相对深度缩放到 ``[0, 255]``。

    论文重建规定使用整张图的最小值和最大值，而不是逐 mask 缩放。函数先在
    float64 中完成 min-max，再用 ``np.rint`` 做像素量化；因此不会把 NumPy
    数组直接截断为整数。NaN、无穷或常量图无法形成可信的相对尺度，直接失败。
    """
    depth = _numeric_2d_array(raw, name="raw depth")
    if not np.isfinite(depth).all():
        raise DepthEstimationError("raw depth values must all be finite")

    minimum = float(depth.min())
    maximum = float(depth.max())
    if maximum == minimum:
        raise DepthEstimationError("raw depth map must not be constant")

    # 先验证所有 float64 中间量；极端有限端点的差可能已无法用 float64 表示。
    depth_range = maximum - minimum
    if not math.isfinite(depth_range) or depth_range <= 0.0:
        raise DepthEstimationError("raw depth range must be finite and representable")
    try:
        scale_factor = 255.0 / depth_range
    except OverflowError as exc:
        raise DepthEstimationError(
            "raw depth range must support finite scaling"
        ) from exc
    if not math.isfinite(scale_factor):
        raise DepthEstimationError("raw depth range must support finite scaling")

    # 极性约定是 larger-is-nearer，所以这里保持单调递增，不做 255-value 反转。
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scaled = (depth - minimum) * scale_factor
    except FloatingPointError as exc:
        raise DepthEstimationError("raw depth scaling must remain finite") from exc
    if not np.isfinite(scaled).all():
        raise DepthEstimationError("raw depth scaling must remain finite")
    return np.rint(scaled).astype(np.uint8)


def representative_depth(
    depth_8bit: NDArray[Any], mask: NDArray[np.bool_]
) -> float:
    """返回单个非空 mask 内有限 8 位相对深度的中位数。

    输入坐标均为图片左上角原点、形状为 ``(height, width)``。审计或测试数据
    即使暂时以浮点数组传入，也只让有限值参与中位数；mask 为空或没有任何
    有限值时失败，避免把缺测像素伪装成零距离。
    """
    depth = _numeric_2d_array(depth_8bit, name="8-bit depth")
    mask_array = np.asarray(mask, dtype=np.bool_)
    if mask_array.shape != depth.shape:
        raise DepthEstimationError("depth and mask must have the same shape")

    selected = depth[mask_array]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        raise DepthEstimationError("mask must contain at least one finite depth value")
    return float(np.median(finite))


def depth_to_steps(representative_d: float) -> int:
    """按论文公式把代表性 8 位相对深度映射为非负步数近似。

    公式为 ``3.876 + 5.197 * ln(255 / d)``。``d`` 的公共有效域为有限
    ``(0, 255]``；计算使用 Python 的双精度浮点（IEEE-754 binary64），最终舍入
    复用几何模块的 ``round_half_up``，避免 Python ``round`` 在 ``.5`` 处使用
    银行家舍入。
    """
    try:
        depth = float(representative_d)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DepthEstimationError("representative depth must be a finite positive number") from exc
    if not math.isfinite(depth) or not 0.0 < depth <= 255.0:
        raise DepthEstimationError(
            "representative depth must be in the finite 8-bit domain (0, 255]"
        )

    try:
        # 对数差与 ln(255 / d) 代数等价，并避免合法极小 d 在除法阶段先溢出。
        unrounded = 3.876 + 5.197 * (math.log(255.0) - math.log(depth))
    except (OverflowError, ValueError) as exc:
        raise DepthEstimationError("paper depth formula is undefined for this value") from exc
    if not math.isfinite(unrounded):
        raise DepthEstimationError("paper depth formula produced a non-finite result")
    return round_half_up(unrounded)


class DepthAnythingV2Service:
    """使用批次级 registry 执行一次 Depth Anything V2 相对深度推理。"""

    # 官方 Depth Anything V2 demo 把默认相对模型原始输出称为 disparity；本适配器
    # 因而显式采用 larger-is-nearer。若未来接入其他后端，必须创建另一个明确声明
    # 极性的适配器，不能从 checkpoint 文件名或模型 ID 推断是否需要反转。
    OUTPUT_POLARITY = "larger_is_nearer"

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def predict(self, image: Image.Image) -> DepthResult:
        """返回原图 ``(height, width)`` 的 float32 原始图和 uint8 SSI 深度图。"""
        processor = self.registry.depth_processor
        inputs = processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.registry.device)

        with torch.inference_mode():
            outputs = self.registry.depth_model(**inputs)

        predicted = getattr(outputs, "predicted_depth", None)
        if not isinstance(predicted, torch.Tensor):
            raise DepthEstimationError(
                "Transformers depth model must return tensor predicted_depth"
            )
        # 本机 Transformers 5.10.2 的 DepthEstimatorOutput 是 (batch, H, W)。
        # 本服务一次只接收一张 PIL 图片；拒绝额外 batch/channel，避免 squeeze
        # 掩盖接口漂移并意外交换空间轴。
        if predicted.ndim != 3 or predicted.shape[0] != 1:
            raise DepthEstimationError(
                "Transformers predicted_depth must have shape (1, height, width)"
            )

        # Hugging Face checkpoint 模型卡给出的原尺寸恢复顺序是 image.size[::-1]，
        # 即 (height, width)。bicubic 与 align_corners=False 均属于 Task 5 固定契约。
        resized = torch_functional.interpolate(
            predicted.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        raw = resized.detach().cpu().numpy().astype(np.float32, copy=False)
        depth_8bit = scale_depth_to_uint8(raw)
        return DepthResult(
            raw=raw,
            depth_8bit=depth_8bit,
            diagnostics={
                "backend": "transformers",
                "polarity": self.OUTPUT_POLARITY,
                "model_id": self.registry.config.depth_model,
            },
        )
