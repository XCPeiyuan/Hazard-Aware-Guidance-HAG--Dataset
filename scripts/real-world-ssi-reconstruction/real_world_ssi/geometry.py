"""真实场景 SSI 重建使用的确定性几何规则。

本模块只处理内存中的二维几何，不加载模型，也不访问网络。掩码和 YOLO
bbox 均采用左闭右开的连续坐标；掩码质心仍保留图片左上角为原点的内部坐标。
只有最终像素位置会按现有 SSI 契约翻转为 Unity 左下角原点。
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .schemas import BBoxXYXY


def round_half_up(value: float) -> int:
    """以半值远离零的统一规则把标量舍入为整数。

    重建流程中的像素坐标和后续距离步数必须复用同一规则，避免 Python
    ``round()`` 的银行家舍入让恰好位于 ``.5`` 的结果依赖奇偶性。
    """
    fractional, integral = math.modf(value)
    # 不能先做 value ± 0.5：紧邻半值但尚未到达半值的二进制浮点数，可能在
    # 该中间运算中舍入到整数边界。直接比较小数部分才能保留输入的真实侧别。
    if value >= 0:
        if fractional >= 0.5:
            integral += 1.0
    elif fractional <= -0.5:
        integral -= 1.0
    # 对 NaN/±inf 仍由 float 到 int 的标准转换抛出 ValueError/OverflowError，
    # 与原实现的非有限输入行为保持一致。
    return int(integral)


def bbox_iou(a: BBoxXYXY, b: BBoxXYXY) -> float:
    """返回两个左闭右开连续 bbox 的交并比；退化 bbox 的 IoU 为零。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def mask_centroid(mask: NDArray[np.bool_]) -> tuple[float, float]:
    """以图片左上角为原点返回非空掩码 True 像素中心索引的 ``(x, y)``。"""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("mask must not be empty")
    return float(xs.mean()), float(ys.mean())


def direction_degrees(cx: float, image_width: int) -> float:
    """把横向质心映射到论文方向角：最左 ``+90``，中心 ``0``，最右 ``-90``。"""
    if image_width < 1:
        raise ValueError("image_width must be positive")
    if image_width == 1:
        return 0.0
    return 90.0 - 180.0 * cx / (image_width - 1)


def unity_pixel_location(cx: float, cy: float, image_height: int) -> tuple[int, int]:
    """把左上角原点质心舍入并转换成现有 SSI 的 Unity 左下角像素坐标。"""
    if image_height < 1:
        raise ValueError("image_height must be positive")
    px = round_half_up(cx)
    py_top = round_half_up(cy)
    return px, image_height - 1 - py_top
