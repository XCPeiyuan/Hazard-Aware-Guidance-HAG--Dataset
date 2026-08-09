"""Grounded SAM 2 候选掩码的确定性选择与单组件清理。

候选选择只使用分数和几何关系，不在此处加载任何模型。离散 True 像素被
解释为连续单位方格 ``[x, x+1) × [y, y+1)``，因此能与浮点 YOLO bbox
使用同一套左闭右开面积语义，并保留边界阈值的可审计性。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes, label

from .geometry import bbox_iou
from .schemas import BBoxXYXY, MaskCandidate


def _binary_mask_bbox(mask: NDArray[np.bool_]) -> BBoxXYXY | None:
    """返回覆盖 True 像素单位方格的最小半开 bbox；空掩码返回 ``None``。"""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def select_mask_candidate(
    candidates: tuple[MaskCandidate, ...],
    annotated_bbox: BBoxXYXY,
    min_iou: float,
) -> MaskCandidate:
    """筛除 bbox IoU 不达标者，再按 brief 的稳定优先级返回唯一候选。

    排序键依次为 SAM 分数、mask-bbox IoU、Grounding 分数和候选 ID；前三项
    均降序，ID 升序。阈值使用闭边界，所以 IoU 恰好等于 ``min_iou`` 可通过。
    """
    eligible: list[tuple[MaskCandidate, float]] = []
    for candidate in candidates:
        candidate_bbox = _binary_mask_bbox(candidate.mask)
        if candidate_bbox is None:
            continue
        iou = bbox_iou(candidate_bbox, annotated_bbox)
        if iou >= min_iou:
            eligible.append((candidate, iou))

    if not eligible:
        raise ValueError("no mask candidate meets the required IoU")

    eligible.sort(
        key=lambda item: (
            -item[0].sam_score,
            -item[1],
            -item[0].grounding_score,
            item[0].candidate_id,
        )
    )
    return eligible[0][0]


def _pixel_bbox_intersection_weights(
    shape: tuple[int, int], bbox: BBoxXYXY
) -> NDArray[np.float64]:
    """计算每个像素单位方格与连续浮点 bbox 的真实几何交叠面积。"""
    height, width = shape
    x1, y1, x2, y2 = bbox
    pixel_x = np.arange(width, dtype=np.float64)
    pixel_y = np.arange(height, dtype=np.float64)
    x_overlap = np.maximum(
        0.0, np.minimum(pixel_x + 1.0, x2) - np.maximum(pixel_x, x1)
    )
    y_overlap = np.maximum(
        0.0, np.minimum(pixel_y + 1.0, y2) - np.maximum(pixel_y, y1)
    )
    return y_overlap[:, None] * x_overlap[None, :]


def clean_single_component(
    mask: NDArray[np.bool_], bbox: BBoxXYXY
) -> NDArray[np.bool_]:
    """填洞后仅保留与标注 bbox 真实交叠面积最大的四连通组件。

    组件交叠面积并列时，采用 ``scipy.ndimage.label`` 的最小正标签。默认
    四连通标签按行扫描，使该重建默认稳定地偏向更上、再更左的组件。
    """
    filled = np.asarray(binary_fill_holes(mask), dtype=np.bool_)
    labeled, component_count = label(filled)
    if component_count == 0:
        raise ValueError("cleaned mask must not be empty")

    weights = _pixel_bbox_intersection_weights(filled.shape, bbox)
    best_label = 1
    best_intersection = float(weights[labeled == best_label].sum())
    for component_label in range(2, component_count + 1):
        intersection = float(weights[labeled == component_label].sum())
        # 严格大于才替换，显式保留并列时的最小 scipy 标签。
        if intersection > best_intersection:
            best_label = component_label
            best_intersection = intersection

    cleaned = np.asarray(labeled == best_label, dtype=np.bool_)
    if not cleaned.any():
        raise ValueError("cleaned mask must not be empty")
    return cleaned
