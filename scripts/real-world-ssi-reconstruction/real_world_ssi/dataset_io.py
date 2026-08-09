"""发现并严格校验用于 real-world SSI 重建的 YOLO 输入数据。

本模块位于论文重建流水线的最前端，只把磁盘输入转换为不可变的
``ImageSample``。这里不会加载视觉模型，也不会把缺失或损坏的标签误判为安全图；
只有“标签文件真实存在且内容为空”才产生空的障碍物元组。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from PIL import Image, UnidentifiedImageError

from real_world_ssi.category_policy import is_non_risk_class
from real_world_ssi.schemas import (
    BBoxXYXY,
    DiscoveryFiltered,
    ImageSample,
    PipelineConfig,
    YoloObject,
)


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
STANDARD_SPLITS = ("train", "val", "test")


class DatasetInputError(ValueError):
    """表示数据集布局、图片、类别或 YOLO 标签不满足输入契约。"""


@dataclass(frozen=True)
class DiscoveryFailure:
    """已经安全定位 sample_id、但单图输入无法形成 ``ImageSample`` 的记录。

    数据集根目录、类别表、split 布局和重复 stem 仍属于全局错误并直接抛出；本记录只用于
    missing label、单图解码和单图 YOLO 内容错误，使批次可为坏图写 QC 后继续。
    """

    sample_id: str
    image_path: Path
    label_path: Path
    reason_code: str
    message: str


DiscoveryItem: TypeAlias = ImageSample | DiscoveryFailure | DiscoveryFiltered


def _validated_class_names(raw_names: Any, source: Path) -> dict[int, str]:
    """把 YAML/list 或 classes.txt 的中间值收敛成严格的 ID 映射。"""
    if isinstance(raw_names, list):
        items = enumerate(raw_names)
    elif isinstance(raw_names, dict):
        converted: list[tuple[int, Any]] = []
        seen_ids: set[int] = set()
        for raw_id, name in raw_names.items():
            # bool 是 int 的子类，但不应被悄悄解释为类别 0/1。
            if isinstance(raw_id, bool):
                raise DatasetInputError(f"Invalid class ID in {source}: {raw_id!r}")
            try:
                class_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise DatasetInputError(
                    f"Invalid class ID in {source}: {raw_id!r}"
                ) from exc
            if str(raw_id).strip() != str(class_id) or class_id < 0:
                raise DatasetInputError(f"Invalid class ID in {source}: {raw_id!r}")
            if class_id in seen_ids:
                raise DatasetInputError(f"Duplicate class ID {class_id} in {source}")
            seen_ids.add(class_id)
            converted.append((class_id, name))
        items = converted
    else:
        raise DatasetInputError(f"Class names in {source} must be a list or mapping")

    result: dict[int, str] = {}
    for class_id, raw_name in items:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise DatasetInputError(f"Class name {class_id} in {source} is empty or invalid")
        result[class_id] = raw_name.strip()
    if not result:
        raise DatasetInputError(f"No class names found in {source}")
    return dict(sorted(result.items()))


def load_class_names(dataset_root: Path) -> dict[int, str]:
    """读取类别 ID 到名称的映射，优先使用根目录的 ``dataset.yaml``。

    YAML 的 ``names`` 可以是列表，也可以是整数或数字字符串键的映射。若没有
    YAML，则逐行读取 ``classes.txt``；两者都缺失时立即失败，不猜测类别名称。
    """
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / "dataset.yaml"
    classes_path = dataset_root / "classes.txt"

    if yaml_path.is_file():
        try:
            document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise DatasetInputError(f"Cannot read class names from {yaml_path}: {exc}") from exc
        if not isinstance(document, dict) or "names" not in document:
            raise DatasetInputError(f"Missing names in {yaml_path}")
        return _validated_class_names(document["names"], yaml_path)

    if classes_path.is_file():
        try:
            text = classes_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DatasetInputError(
                f"Cannot read class names from {classes_path}: {exc}"
            ) from exc
        # splitlines() 不会把正常的末尾换行当作额外类别，但会保留中间空行供严格校验。
        return _validated_class_names(text.splitlines(), classes_path)

    raise DatasetInputError(
        f"Dataset root {dataset_root} has neither dataset.yaml nor classes.txt"
    )


def normalized_xywh_to_xyxy(
    values: tuple[float, float, float, float], width: int, height: int
) -> BBoxXYXY:
    """把 YOLO 归一化 ``cx, cy, w, h`` 转为左上角原点的像素 ``xyxy``。

    重建输入契约要求中心坐标位于闭区间 ``[0, 1]``、宽高位于 ``(0, 1]``，
    且换算后的框不能越过图片边界。这里保留浮点坐标，不提前取整。
    """
    if width <= 0 or height <= 0:
        raise DatasetInputError("Image dimensions must be positive")

    cx, cy, bbox_width, bbox_height = values
    if not all(math.isfinite(value) for value in values):
        raise DatasetInputError("YOLO coordinates must be finite")
    if not (
        0 <= cx <= 1
        and 0 <= cy <= 1
        and 0 < bbox_width <= 1
        and 0 < bbox_height <= 1
    ):
        raise DatasetInputError("YOLO coordinates are outside normalized bounds")

    x1 = (cx - bbox_width / 2) * width
    y1 = (cy - bbox_height / 2) * height
    x2 = (cx + bbox_width / 2) * width
    y2 = (cy + bbox_height / 2) * height
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise DatasetInputError("YOLO bbox extends outside the image")
    return x1, y1, x2, y2


def parse_yolo_label(
    label_path: Path,
    class_names: dict[int, str],
    image_width: int,
    image_height: int,
) -> tuple[YoloObject, ...]:
    """解析单个 YOLO 标签，保留零基物理行号作为 ``annotation_index``。

    空白行被忽略，但不会重排后续对象的索引。任一非空行列数错误、数值非法、
    类别未知或 bbox 越界都会使整张图片输入失败。
    """
    label_path = Path(label_path)
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetInputError(f"Cannot read YOLO label {label_path}: {exc}") from exc

    objects: list[YoloObject] = []
    for line_number, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        columns = line.split()
        if len(columns) != 5:
            raise DatasetInputError(
                f"Invalid YOLO column count at {label_path}:{line_number + 1}"
            )

        try:
            class_id = int(columns[0])
        except ValueError as exc:
            raise DatasetInputError(
                f"Invalid class ID at {label_path}:{line_number + 1}"
            ) from exc
        if columns[0].strip() != str(class_id) or class_id not in class_names:
            raise DatasetInputError(
                f"Unknown class ID {columns[0]!r} at {label_path}:{line_number + 1}"
            )

        try:
            values = tuple(float(value) for value in columns[1:])
        except ValueError as exc:
            raise DatasetInputError(
                f"Invalid YOLO number at {label_path}:{line_number + 1}"
            ) from exc
        bbox = normalized_xywh_to_xyxy(
            values,  # type: ignore[arg-type] -- 列数已在上方严格锁定为四个坐标。
            image_width,
            image_height,
        )
        objects.append(
            YoloObject(
                annotation_index=line_number,
                class_id=class_id,
                name=class_names[class_id],
                bbox_xyxy=bbox,
            )
        )
    return tuple(objects)


def _discover_image_paths(config: PipelineConfig) -> tuple[Path, tuple[Path, ...]]:
    """根据已确认布局返回全局 images 根目录与本次需要扫描的子目录。"""
    root = config.dataset_root
    images_root = root / "images"
    if not images_root.is_dir():
        raise DatasetInputError(f"Missing images directory: {images_root}")

    if (root / "dataset.yaml").is_file():
        if config.dataset_split == "auto":
            scan_roots = tuple(
                images_root / split
                for split in STANDARD_SPLITS
                if (images_root / split).is_dir()
            )
            if not scan_roots:
                raise DatasetInputError(
                    f"Standard layout has no train, val, or test split under {images_root}"
                )
            return images_root, scan_roots
        if config.dataset_split not in STANDARD_SPLITS:
            raise DatasetInputError(
                "Standard layout dataset_split must be auto, train, val, or test"
            )
        split_root = images_root / config.dataset_split
        if not split_root.is_dir():
            raise DatasetInputError(
                f"Requested dataset split {config.dataset_split!r} is missing"
            )
        return images_root, (split_root,)

    if config.dataset_split != "auto":
        raise DatasetInputError("Simple layout requires dataset_split='auto'")
    return images_root, (images_root,)


def discover_dataset_items(config: PipelineConfig) -> tuple[DiscoveryItem, ...]:
    """发现全部图片，并把单图输入错误保留为可继续处理的失败项。

    标准布局扫描 ``images/{train,val,test}``，简单布局扫描 ``images/``。标签必须
    位于 ``labels/`` 下的相同相对路径。Pillow 在此只打开文件并读取尺寸，不执行
    后续流水线所需的完整 RGB 解码。根目录、类别表、split 和重复 stem 是批次级契约，
    仍会立即抛错；只有可明确归属 sample_id 的单图错误被转换为 ``DiscoveryFailure``。
    """
    root = Path(config.dataset_root)
    if not root.is_dir():
        raise DatasetInputError(f"Dataset root does not exist: {root}")

    class_names = load_class_names(root)
    images_root, scan_roots = _discover_image_paths(config)
    labels_root = root / "labels"

    image_paths = [
        path
        for scan_root in scan_roots
        for path in scan_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    image_paths.sort(key=lambda path: path.relative_to(images_root).as_posix())

    # brief 明确按 basename stem 判重；即使 sample_id 因 split 前缀不同也不能掩盖冲突。
    seen_stems: dict[str, Path] = {}
    for image_path in image_paths:
        stem_key = image_path.stem.casefold()
        if stem_key in seen_stems:
            raise DatasetInputError(
                f"Duplicate image stem {image_path.stem!r}: "
                f"{seen_stems[stem_key]} and {image_path}"
            )
        seen_stems[stem_key] = image_path

    items: list[DiscoveryItem] = []
    for image_path in image_paths:
        relative_image = image_path.relative_to(images_root)
        sample_id = relative_image.with_suffix("").as_posix()
        label_path = (labels_root / relative_image).with_suffix(".txt")
        try:
            if not label_path.is_file():
                raise DatasetInputError(
                    f"Missing label for image {relative_image.as_posix()}"
                )
            with Image.open(image_path) as image:
                width, height = image.size
            if width <= 0 or height <= 0:
                raise DatasetInputError(
                    f"Image dimensions must be positive: {image_path}"
                )
            objects = parse_yolo_label(label_path, class_names, width, height)
            whitelisted_object_count = sum(
                is_non_risk_class(item.name, config.non_risk_class_names)
                for item in objects
            )
            effective_object_count = len(objects) - whitelisted_object_count
            if effective_object_count >= config.max_object_count:
                items.append(
                    DiscoveryFiltered(
                        sample_id=sample_id,
                        image_path=image_path,
                        label_path=label_path,
                        object_count=effective_object_count,
                        max_object_count_exclusive=config.max_object_count,
                        total_object_count=len(objects),
                        whitelisted_object_count=whitelisted_object_count,
                    )
                )
                continue
            items.append(
                ImageSample(
                    sample_id=sample_id,
                    image_path=image_path,
                    label_path=label_path,
                    width=width,
                    height=height,
                    objects=objects,
                )
            )
        except (OSError, UnidentifiedImageError) as exc:
            error = DatasetInputError(
                f"Cannot decode image header {image_path}: {exc}"
            )
            items.append(
                DiscoveryFailure(
                    sample_id, image_path, label_path, "input_invalid", str(error)
                )
            )
        except DatasetInputError as exc:
            items.append(
                DiscoveryFailure(
                    sample_id, image_path, label_path, "input_invalid", str(exc)
                )
            )

    return tuple(items)


def discover_dataset(config: PipelineConfig) -> tuple[ImageSample, ...]:
    """保留 Task 2 的严格公共接口：任一单图失败时抛出首个输入错误。

    Task 10 的批次编排使用 ``discover_dataset_items``；既有调用方和单元测试继续通过此
    函数获得“全有或全无”的 ``ImageSample`` 元组，不发生静默语义变化。
    """
    items = discover_dataset_items(config)
    for item in items:
        if isinstance(item, DiscoveryFailure):
            raise DatasetInputError(item.message)

    return tuple(item for item in items if isinstance(item, ImageSample))
