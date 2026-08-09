"""生成供用户逐障碍物审查的 Real-world SSI overlay。

内部几何仍采用左上角原点；图中同时显示最终 Unity 左下角像素坐标，防止人工审查时
混淆两个坐标系。函数只返回新 PIL 图片，不自行决定文件路径；Task 10 可先把同一张
overlay 交给分类器，再通过 ``reports.write_overlay_artifact`` 原子保存。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import direction_degrees, unity_pixel_location
from .schemas import CategoryAssignment, ImageSample, ObstacleMask


_COLORS = (
    (230, 57, 70),
    (29, 78, 216),
    (42, 157, 143),
    (255, 159, 28),
    (131, 56, 236),
)


def _font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """优先使用带 Unicode 的常见字体；发布环境缺失时退回 Pillow 默认字体。"""
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, 11)
        except OSError:
            continue
    return ImageFont.load_default()


def _unique_by_index(values: tuple[object, ...], *, name: str) -> dict[int, object]:
    result: dict[int, object] = {}
    for value in values:
        index = getattr(value, "annotation_index", None)
        if type(index) is not int or index < 0:
            raise ValueError(f"{name} annotation_index must be a non-negative integer")
        if index in result:
            raise ValueError(f"{name} annotation_index values must be unique")
        result[index] = value
    return result


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validated_mapping_keys(
    value: Mapping[object, object], *, name: str
) -> tuple[int, ...]:
    """一次性冻结并验证外部 Mapping 的 key 视图。

    bool 是 int 子类且会与 0/1 集合相等；自定义 Mapping 还可能让 ``len`` 与迭代
    数量不一致或重复产出 key。进入 exact-ID 比较前必须逐项拒绝这些歧义。
    """
    keys = tuple(value)
    if (
        len(value) != len(keys)
        or any(type(key) is not int or key < 0 for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError(
            f"{name} mapping keys must be unique non-negative integers "
            "and match mapping length"
        )
    return keys  # type: ignore[return-value] -- 上方已逐项收窄为精确 int。


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    *,
    fill: tuple[int, int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    """极旧 Pillow 字体不支持某字符时仍显示可审计的 Unicode 转义文本。"""
    try:
        draw.text(position, text, fill=fill, font=font, stroke_width=2, stroke_fill="black")
    except UnicodeEncodeError:
        escaped = text.encode("ascii", errors="backslashreplace").decode("ascii")
        draw.text(
            position,
            escaped,
            fill=fill,
            font=font,
            stroke_width=2,
            stroke_fill="black",
        )


def render_overlay(
    image: Image.Image,
    sample: ImageSample,
    masks: tuple[ObstacleMask, ...],
    representative_depths: Mapping[int, float],
    distance_steps: Mapping[int, int],
    assignments: tuple[CategoryAssignment, ...] = (),
) -> Image.Image:
    """绘制 bbox、选中 mask 与完整逐物体审查字段并返回 RGB 图片。

    ``assignments=()`` 是 ``local_only`` 的明确状态，此时类别显示为 pending；一旦
    提供类别，ID 集必须与 YOLO 物体完全相等，禁止部分分类看起来像完整结果。
    返回图的 ``info['audit_fields']`` 保存同一份结构化字段，使测试和下游报告无需 OCR。
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if not isinstance(sample, ImageSample):
        raise TypeError("sample must be an ImageSample")
    if image.size != (sample.width, sample.height):
        raise ValueError("image size must match sample dimensions")
    if not isinstance(representative_depths, Mapping) or not isinstance(
        distance_steps, Mapping
    ):
        raise TypeError("depths and distance_steps must be mappings")
    depth_keys = _validated_mapping_keys(
        representative_depths, name="representative depth"
    )
    step_keys = _validated_mapping_keys(distance_steps, name="distance step")

    object_by_id = _unique_by_index(sample.objects, name="objects")
    mask_by_id = _unique_by_index(masks, name="masks")
    assignment_by_id = _unique_by_index(assignments, name="assignments")
    expected_ids = set(object_by_id)
    if set(mask_by_id) != expected_ids:
        raise ValueError("mask annotation indexes must exactly match objects")
    if set(depth_keys) != expected_ids:
        raise ValueError("representative depth indexes must exactly match objects")
    if set(step_keys) != expected_ids:
        raise ValueError("distance step indexes must exactly match objects")
    if assignment_by_id and set(assignment_by_id) != expected_ids:
        raise ValueError("assignment indexes must be empty or exactly match objects")

    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    font = _font()
    audit_fields: list[dict[str, object]] = []

    for sequence, obj in enumerate(sample.objects):
        index = obj.annotation_index
        mask = mask_by_id[index]
        assert isinstance(mask, ObstacleMask)  # 由上方 tuple 契约与索引构造保证。
        values = np.asarray(mask.mask)
        if values.dtype != np.bool_ or values.shape != (sample.height, sample.width):
            raise ValueError("each selected mask must be bool with sample image shape")
        if not values.any():
            raise ValueError("selected masks must not be empty")

        centroid = mask.centroid_top_left
        if not isinstance(centroid, (tuple, list)) or len(centroid) != 2:
            raise ValueError("mask centroid must contain x and y")
        cx = _finite_number(centroid[0], name="centroid x")
        cy = _finite_number(centroid[1], name="centroid y")
        if not 0.0 <= cx <= sample.width - 1 or not 0.0 <= cy <= sample.height - 1:
            raise ValueError("mask centroid must lie within the image")
        representative = _finite_number(
            representative_depths[index], name="representative depth"
        )
        steps = distance_steps[index]
        if type(steps) is not int or steps < 0:
            raise ValueError("distance steps must be non-negative integers")

        color = _COLORS[sequence % len(_COLORS)]
        alpha = Image.fromarray(values.astype(np.uint8) * 92, mode="L")
        tint = Image.new("RGBA", canvas.size, (*color, 0))
        tint.putalpha(alpha)
        canvas.alpha_composite(tint)
        draw = ImageDraw.Draw(canvas, mode="RGBA")

        x1, y1, x2, y2 = obj.bbox_xyxy
        if not all(math.isfinite(float(value)) for value in (x1, y1, x2, y2)):
            raise ValueError("bbox coordinates must be finite")
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=2)
        draw.line((cx - 3, cy, cx + 3, cy), fill=(*color, 255), width=1)
        draw.line((cx, cy - 3, cx, cy + 3), fill=(*color, 255), width=1)

        unity = unity_pixel_location(cx, cy, sample.height)
        direction = float(direction_degrees(cx, sample.width))
        assignment = assignment_by_id.get(index)
        category = (
            None
            if assignment is None
            else getattr(assignment, "category", None)
        )
        fallback = mask.source == "bbox_fallback"
        audit_fields.append(
            {
                "annotation_index": index,
                "class_name": obj.name,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "centroid_top_left": [cx, cy],
                "unity_pixel_coordinate": [unity[0], unity[1]],
                "direction_degrees": direction,
                "representative_depth": representative,
                "steps": steps,
                "category": category,
                "fallback": fallback,
            }
        )

        category_text = "pending" if category is None else str(category)
        fallback_text = " [BBOX FALLBACK]" if fallback else ""
        label = (
            f"#{index} {obj.name}{fallback_text} | centroid_tl=({cx:.1f},{cy:.1f}) "
            f"| unity=({unity[0]},{unity[1]}) | dir={direction:.1f} deg "
            f"| depth={representative:.1f} | steps={steps} | category={category_text}"
        )
        _draw_text(
            draw,
            (max(0.0, float(x1)), max(0.0, float(y1) - 13.0)),
            label,
            fill=(*color, 255),
            font=font,
        )

    result = canvas.convert("RGB")
    result.info["audit_fields"] = audit_fields
    return result
