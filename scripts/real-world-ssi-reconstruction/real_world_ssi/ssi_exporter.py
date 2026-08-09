"""把确定性的逐障碍物结果导出为现有标注程序可读取的规范 SSI。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Protocol

from .category_policy import SSI_CATEGORIES
from .geometry import direction_degrees, unity_pixel_location
from .schemas import CategoryAssignment, ImageSample, ObstacleMask, YoloObject


_CANONICAL_CATEGORIES = frozenset(
    SSI_CATEGORIES
)


class SSIExportError(ValueError):
    """导出输入无法形成可信、完整 SSI 时抛出的领域错误。"""


class _HasAnnotationIndex(Protocol):
    annotation_index: int


def _validated_indexes(
    values: Sequence[_HasAnnotationIndex], *, collection_name: str
) -> tuple[int, ...]:
    """先保留原顺序检查 ID，再建索引；禁止字典推导静默吞掉重复项。"""
    indexes = tuple(item.annotation_index for item in values)
    if any(type(index) is not int or index < 0 for index in indexes):
        raise ValueError(
            f"{collection_name} annotation_index must be a non-negative integer"
        )
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"{collection_name} annotation_index values must be unique")
    return indexes


def _validated_distance_indexes(distance_steps: Mapping[int, int]) -> tuple[int, ...]:
    if not isinstance(distance_steps, Mapping):
        raise TypeError("distance_steps must be a mapping keyed by annotation_index")
    indexes = tuple(distance_steps)
    # Mapping 可自定义 len/迭代语义；二者不一致时不能把 safe 空集当成可信输入。
    if len(distance_steps) != len(indexes):
        raise SSIExportError(
            "distance_steps length must match its iterated annotation_index count"
        )
    if any(type(index) is not int or index < 0 for index in indexes):
        raise ValueError(
            "distance_steps annotation_index must be a non-negative integer"
        )
    if len(indexes) != len(set(indexes)):
        raise ValueError("distance_steps annotation_index values must be unique")
    return indexes


def _validate_sample(sample: ImageSample) -> None:
    if not isinstance(sample, ImageSample):
        raise TypeError("sample must be an ImageSample")
    if type(sample.width) is not int or sample.width < 1:
        raise ValueError("sample width must be a positive integer")
    if type(sample.height) is not int or sample.height < 1:
        raise ValueError("sample height must be a positive integer")


def _finite_coordinate(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _validated_centroid(mask: ObstacleMask, sample: ImageSample) -> tuple[float, float]:
    centroid = mask.centroid_top_left
    if not isinstance(centroid, (tuple, list)) or len(centroid) != 2:
        raise TypeError("mask centroid_top_left must contain exactly two coordinates")
    cx = _finite_coordinate(centroid[0], field_name="mask centroid x")
    cy = _finite_coordinate(centroid[1], field_name="mask centroid y")
    # 现有 parser 只接受 [-90, 90]，所以越界质心不能被导出成看似合法的 SSI。
    if not 0.0 <= cx <= sample.width - 1:
        raise ValueError("mask centroid x must lie within the image")
    if not 0.0 <= cy <= sample.height - 1:
        raise ValueError("mask centroid y must lie within the image")
    return cx, cy


def _validated_steps(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("distance_steps values must be non-negative integers")
    try:
        finite_for_parser = math.isfinite(float(value))
    except OverflowError as exc:
        raise ValueError("distance_steps values must be finite") from exc
    if not finite_for_parser:
        raise ValueError("distance_steps values must be finite")
    return value


def _validated_name(obj: YoloObject) -> str:
    if type(obj.name) is not str or not obj.name:
        raise ValueError("hazard name must be a nonempty string")
    return obj.name


def _validated_category(assignment: CategoryAssignment) -> str:
    category = assignment.category
    if type(category) is not str or category not in _CANONICAL_CATEGORIES:
        raise ValueError("category must use the exact canonical hazard enum")
    return category


def build_ssi_document(
    sample: ImageSample,
    masks: tuple[ObstacleMask, ...],
    distance_steps: Mapping[int, int],
    assignments: tuple[CategoryAssignment, ...],
) -> dict[str, object]:
    """按原 YOLO 行序连接四组结果，并构造规范化 SSI JSON 对象。"""
    _validate_sample(sample)
    object_indexes = _validated_indexes(sample.objects, collection_name="objects")
    mask_indexes = _validated_indexes(masks, collection_name="masks")
    distance_indexes = _validated_distance_indexes(distance_steps)
    assignment_indexes = _validated_indexes(
        assignments, collection_name="assignments"
    )

    expected = set(object_indexes)
    if any(
        set(indexes) != expected
        for indexes in (mask_indexes, distance_indexes, assignment_indexes)
    ):
        raise ValueError(
            "objects, masks, distance_steps, and assignments annotation_index "
            "sets must exactly match"
        )

    if not object_indexes:
        return {"safety_status": "safe", "hazards": []}

    masks_by_index = {item.annotation_index: item for item in masks}
    assignments_by_index = {item.annotation_index: item for item in assignments}
    hazards: list[dict[str, object]] = []
    for obj in sample.objects:
        index = obj.annotation_index
        cx, cy = _validated_centroid(masks_by_index[index], sample)
        direction = float(direction_degrees(cx, sample.width))
        if not math.isfinite(direction):
            raise ValueError("direction_degrees must be finite")
        pixel_x, pixel_y = unity_pixel_location(cx, cy, sample.height)
        hazards.append(
            {
                "name": _validated_name(obj),
                "category": _validated_category(assignments_by_index[index]),
                "direction_degrees": direction,
                "distance_value": _validated_steps(distance_steps[index]),
                "distance_unit": "steps",
                "pixel_location": [pixel_x, pixel_y],
            }
        )

    return {"safety_status": "hazard", "hazards": hazards}


def write_ssi_atomic(document: Mapping[str, object], output_path: Path) -> None:
    """在目标同级写完、同步后原子替换；任一步失败均清理临时文件。"""
    if not isinstance(document, Mapping):
        raise TypeError("SSI document must be a mapping")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
