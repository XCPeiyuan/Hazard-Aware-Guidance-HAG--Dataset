"""Real-world SSI 各阶段共享的不可变数据契约。

NumPy 数组只存在于进程内或缓存记录；面向 SSI JSON 的诊断字段必须由
``JsonValue`` 约束为可序列化值。类别文本逐字采用论文的三个规范名称。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import NoneType
from typing import Literal, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray


BBoxXYXY: TypeAlias = tuple[float, float, float, float]
HazardCategory: TypeAlias = Literal[
    "Common Obstacle",
    "Pitfall Hazard",
    "Upper-body Hazard",
]

# Python 3.11 的 PEP 604 联合类型需要类型对象，因此用 NoneType 表示 JSON null。
JsonScalar: TypeAlias = NoneType | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class PipelineConfig:
    """整条重建流水线的单一、不可变配置快照。"""

    dataset_root: Path
    output_root: Path
    dataset_split: str
    device: str
    run_stage: str
    resume: bool
    overwrite: bool
    dry_run: bool
    fail_batch_if_any_error: bool
    auto_download_models: bool
    model_cache_dir: Path
    grounding_dino_model: str
    sam2_model: str
    depth_model: str
    sky_model: str
    openai_model: str
    openai_image_detail: str
    openai_max_retries: int
    openai_timeout_seconds: int
    grounding_box_threshold: float
    grounding_text_threshold: float
    mask_bbox_iou_threshold: float
    allow_bbox_prompt_fallback: bool
    near_depth_threshold: int
    max_small_near_area_ratio: float
    min_near_concentration: float
    min_sky_area_ratio: float
    filter_mode: str
    save_raw_depth: bool
    save_masks: bool
    save_overlays: bool
    save_api_responses: bool

    def validate(self) -> "PipelineConfig":
        """验证阶段枚举、目录隔离和论文比例阈值。"""
        if self.filter_mode not in {"reject", "warn", "off"}:
            raise ValueError("FILTER_MODE must be reject, warn, or off")
        if self.run_stage not in {"full", "local_only", "classify_only", "export_only"}:
            raise ValueError("RUN_STAGE is unsupported")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("DEVICE must be auto, cpu, or cuda")

        # 输出落在输入树中会污染 YOLO 数据集，并可能在恢复运行时递归摄取产物。
        if self.output_root.resolve().is_relative_to(self.dataset_root.resolve()):
            raise ValueError("OUTPUT_ROOT must not be inside DATASET_ROOT")

        # 这些量均为比例或置信度，论文契约统一使用闭区间 [0, 1]。
        for name, value in {
            "GROUNDING_BOX_THRESHOLD": self.grounding_box_threshold,
            "GROUNDING_TEXT_THRESHOLD": self.grounding_text_threshold,
            "MASK_BBOX_IOU_THRESHOLD": self.mask_bbox_iou_threshold,
            "MAX_SMALL_NEAR_AREA_RATIO": self.max_small_near_area_ratio,
            "MIN_NEAR_CONCENTRATION": self.min_near_concentration,
            "MIN_SKY_AREA_RATIO": self.min_sky_area_ratio,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        return self


@dataclass(frozen=True)
class YoloObject:
    """保持原 YOLO 行序号的单个障碍物记录。"""

    annotation_index: int
    class_id: int
    name: str
    bbox_xyxy: BBoxXYXY


@dataclass(frozen=True)
class ImageSample:
    """完成路径与尺寸解析后的单个输入样本。"""

    sample_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    objects: tuple[YoloObject, ...]


@dataclass(frozen=True)
class MaskCandidate:
    """Grounding DINO/SAM2 选择前的内存候选掩码。"""

    candidate_id: str
    mask: NDArray[np.bool_]
    sam_score: float
    grounding_score: float


@dataclass(frozen=True)
class ObstacleMask:
    """与 YOLO 标注逐项对应的最终障碍物掩码。"""

    annotation_index: int
    mask: NDArray[np.bool_]
    centroid_top_left: tuple[float, float]
    source: Literal["grounded_sam2", "bbox_fallback"]
    diagnostics: Mapping[str, JsonValue]


@dataclass(frozen=True)
class DepthResult:
    """保留原始深度与 SSI 使用的 8 位深度。"""

    raw: NDArray[np.float32]
    depth_8bit: NDArray[np.uint8]
    diagnostics: Mapping[str, JsonValue]


@dataclass(frozen=True)
class QualityMetrics:
    """论文筛除规则所需的三个归一化质量指标。"""

    near_fraction: float
    near_concentration: float
    sky_fraction: float


@dataclass(frozen=True)
class QualityReason:
    """稳定机器码与中文审查说明组成的质量原因。"""

    code: Literal["near_artifact", "sky_missing"]
    message_zh: str


@dataclass(frozen=True)
class QualityDecision:
    """筛除模式应用后的样本级质量结论。"""

    status: Literal["passed", "warning", "rejected"]
    metrics: QualityMetrics
    reasons: tuple[QualityReason, ...]


@dataclass(frozen=True)
class CategoryAssignment:
    """以原标注序号关联论文规范风险类别。"""

    annotation_index: int
    category: HazardCategory


@dataclass(frozen=True)
class SampleOutcome:
    """仅含路径、状态和代码的 JSON 安全样本结果。"""

    sample_id: str
    status: Literal["safe", "hazard", "warning", "rejected", "failed", "category_pending"]
    ssi_path: Path | None
    qc_path: Path
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class BatchSummary:
    """批次完成后的计数摘要。"""

    outcomes: tuple[SampleOutcome, ...]
    succeeded: int
    rejected: int
    failed: int

    def to_chinese_text(self) -> str:
        """生成 brief 规定的稳定中文命令行摘要。"""
        return f"成功 {self.succeeded}，筛除 {self.rejected}，失败 {self.failed}"
