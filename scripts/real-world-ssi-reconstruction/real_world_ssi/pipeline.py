"""Real-world Source 从逐图视觉阶段到正式 SSI 的编排状态机。

本模块只连接 Tasks 1--9 已提交的接口，不实现任何新的模型算法。每张图片都有独立的
状态历史和 QC；失败只清理该图片的正式 SSI，绝不删除输入图片、YOLO 标签或其他样本。
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cache import StageCache, artifact_path, atomic_write_json, sample_fingerprint
from .category_policy import CATEGORY_POLICY_VERSION, is_non_risk_class
from .consistency_filter import (
    ConsistencyFilterError,
    decide_quality,
    evaluate_quality,
)
from .dataset_io import DiscoveryFailure, discover_dataset_items
from .depth_estimation import depth_to_steps, representative_depth
from .interfaces import (
    AtomicCacheAwareHazardClassifier,
    CacheIdentityProvider,
    DepthService,
    HazardClassifier,
    MaskService,
    SkyService,
)
from .minimax_config import load_minimax_credentials
from .progress import ProgressEvent, ProgressReporter, ProgressStage
from .reports import (
    write_batch_summary,
    write_depth_artifacts,
    write_mask_artifacts,
    write_overlay_artifact,
    write_qc_record,
)
from .schemas import (
    BatchSummary,
    CategoryAssignment,
    DepthResult,
    DiscoveryFiltered,
    ImageSample,
    ObstacleMask,
    PipelineConfig,
    QualityDecision,
    QualityMetrics,
    SampleOutcome,
    YoloObject,
)
from .ssi_exporter import build_ssi_document, write_ssi_atomic
from .visualization import render_overlay


@dataclass(frozen=True)
class PipelineServices:
    """编排层唯一接受的服务束；测试可整体注入 fake，避免隐式全局依赖。"""

    masks: MaskService
    depth: DepthService
    sky: SkyService
    classifier: HazardClassifier | None


class _MissingStageCache(RuntimeError):
    """分段运行缺少有效上游缓存；统一映射为稳定 QC reason code。"""


class _FormalSSICleanupError(RuntimeError):
    """旧正式 SSI 无法安全撤下；该样本必须失败，但批次仍继续。"""


class _UnavailableLocalService:
    """classify/export 默认服务占位符；构造它不会导入或加载本地模型。"""

    def extract(self, image, objects):  # pragma: no cover - 违反模式边界才会到达
        raise RuntimeError("local mask service is unavailable in this stage")

    def predict(self, image):  # pragma: no cover - 违反模式边界才会到达
        raise RuntimeError("local depth service is unavailable in this stage")

    def predict_sky_mask(self, image):  # pragma: no cover - 违反模式边界才会到达
        raise RuntimeError("local sky service is unavailable in this stage")


def _quality_payload(decision: QualityDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "metrics": {
            "near_fraction": decision.metrics.near_fraction,
            "near_concentration": decision.metrics.near_concentration,
            "sky_fraction": decision.metrics.sky_fraction,
            "near_candidate": decision.metrics.near_candidate,
            "depth_rgb_edge_correlation": decision.metrics.depth_rgb_edge_correlation,
        },
        "reasons": [
            {"code": reason.code, "message_zh": reason.message_zh}
            for reason in decision.reasons
        ],
    }


def _quality_from_payload(
    payload: dict[str, Any], config: PipelineConfig
) -> QualityDecision:
    """只信任缓存中的有限 metrics，并用当前配置重新执行既有质量判定。

    cached status/reasons 属于冗余审计字段，不能作为决策依据；否则恶意或旧缓存可以写出
    ``passed + sky_fraction=0`` 绕过当前筛选规则。
    """
    try:
        if set(payload) != {"status", "metrics", "reasons"}:
            raise ValueError
        status = payload["status"]
        metrics = payload["metrics"]
        reasons = payload["reasons"]
        if type(status) is not str or status not in {
            "passed",
            "warning",
            "rejected",
        }:
            raise ValueError
        if (
            not isinstance(metrics, dict)
            or set(metrics)
            != {
                "near_fraction",
                "near_concentration",
                "sky_fraction",
                "near_candidate",
                "depth_rgb_edge_correlation",
            }
            or type(reasons) is not list
        ):
            raise ValueError
        for reason in reasons:
            if (
                not isinstance(reason, dict)
                or set(reason) != {"code", "message_zh"}
                or type(reason["code"]) is not str
                or reason["code"] not in {"near_artifact", "sky_missing"}
                or type(reason["message_zh"]) is not str
                or not reason["message_zh"]
            ):
                raise ValueError

        def finite_fraction(name: str) -> float:
            value = metrics[name]
            if isinstance(value, bool) or type(value) not in {int, float}:
                raise ValueError
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError
            return numeric

        near_candidate = metrics["near_candidate"]
        if type(near_candidate) is not bool:
            raise ValueError
        correlation = metrics["depth_rgb_edge_correlation"]
        if isinstance(correlation, bool) or type(correlation) not in {int, float}:
            raise ValueError
        correlation = float(correlation)
        if not math.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
            raise ValueError
        return decide_quality(
            QualityMetrics(
                near_fraction=finite_fraction("near_fraction"),
                near_concentration=finite_fraction("near_concentration"),
                sky_fraction=finite_fraction("sky_fraction"),
                near_candidate=near_candidate,
                depth_rgb_edge_correlation=correlation,
            ),
            config,
        )
    except (KeyError, TypeError, ValueError, OverflowError, ConsistencyFilterError) as exc:
        raise _MissingStageCache("quality cache payload is invalid") from exc


def _save_masks(
    cache: StageCache, sample: ImageSample, config: PipelineConfig,
    masks: tuple[ObstacleMask, ...],
) -> None:
    arrays = {f"mask_{item.annotation_index}": item.mask for item in masks}
    metadata = {
        "items": [
            {
                "annotation_index": item.annotation_index,
                "centroid_top_left": list(item.centroid_top_left),
                "source": item.source,
                "diagnostics": dict(item.diagnostics),
            }
            for item in masks
        ]
    }
    cache.save_npz(
        sample.sample_id, "mask", sample_fingerprint("mask", sample, config),
        arrays, metadata=metadata,
    )


def _load_masks(
    cache: StageCache, sample: ImageSample, config: PipelineConfig,
) -> tuple[ObstacleMask, ...] | None:
    loaded = cache.load_npz(
        sample.sample_id, "mask", sample_fingerprint("mask", sample, config)
    )
    if loaded is None:
        return None
    arrays, metadata = loaded
    try:
        items = metadata["items"]
        if not isinstance(items, list):
            raise ValueError
        result = tuple(
            ObstacleMask(
                annotation_index=item["annotation_index"],
                mask=arrays[f"mask_{item['annotation_index']}"],
                centroid_top_left=tuple(item["centroid_top_left"]),
                source=item["source"],
                diagnostics=item["diagnostics"],
            )
            for item in items
        )
        if {item.annotation_index for item in result} != {
            item.annotation_index for item in sample.objects
        }:
            raise ValueError
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise _MissingStageCache("mask cache payload is invalid") from exc


def _save_depth(
    cache: StageCache, sample: ImageSample, config: PipelineConfig, result: DepthResult
) -> None:
    cache.save_npz(
        sample.sample_id, "depth", sample_fingerprint("depth", sample, config),
        {"raw": result.raw, "depth_8bit": result.depth_8bit},
        metadata={"diagnostics": dict(result.diagnostics)},
    )


def _load_depth(
    cache: StageCache, sample: ImageSample, config: PipelineConfig,
) -> DepthResult | None:
    loaded = cache.load_npz(
        sample.sample_id, "depth", sample_fingerprint("depth", sample, config)
    )
    if loaded is None:
        return None
    arrays, metadata = loaded
    try:
        diagnostics = metadata["diagnostics"]
        if not isinstance(diagnostics, dict):
            raise ValueError
        return DepthResult(arrays["raw"], arrays["depth_8bit"], diagnostics)
    except (KeyError, TypeError, ValueError) as exc:
        raise _MissingStageCache("depth cache payload is invalid") from exc


def _save_sky(
    cache: StageCache, sample: ImageSample, config: PipelineConfig,
    sky_mask: np.ndarray,
) -> None:
    cache.save_npz(
        sample.sample_id, "sky", sample_fingerprint("sky", sample, config),
        {"sky_mask": sky_mask},
    )


def _load_sky(
    cache: StageCache, sample: ImageSample, config: PipelineConfig,
) -> np.ndarray | None:
    loaded = cache.load_npz(
        sample.sample_id, "sky", sample_fingerprint("sky", sample, config)
    )
    if loaded is None:
        return None
    arrays, _metadata = loaded
    try:
        return arrays["sky_mask"]
    except KeyError as exc:
        raise _MissingStageCache("sky cache payload is invalid") from exc


def _save_categories(
    cache: StageCache, sample: ImageSample, config: PipelineConfig,
    assignments: tuple[CategoryAssignment, ...],
    classifier_identity: str,
) -> None:
    cache.save_json(
        sample.sample_id, "category",
        sample_fingerprint(
            "category", sample, config, classifier_identity=classifier_identity
        ),
        {
            "classifier_identity": classifier_identity,
            "assignments": [
                {"annotation_index": item.annotation_index, "category": item.category}
                for item in assignments
            ],
        },
        cache_identity=classifier_identity,
    )


def _load_categories(
    cache: StageCache, sample: ImageSample, config: PipelineConfig,
    classifier_identity: str | None,
    *,
    accept_stored_identity: bool = False,
) -> tuple[tuple[CategoryAssignment, ...], str] | None:
    # 没有原子分类接口的第三方分类器可以运行，但绝不读取可能属于旧实现的类别缓存。
    if not accept_stored_identity and classifier_identity is None:
        return None
    record = cache.load_json_record(sample.sample_id, "category")
    if record is None:
        return None
    payload, stored_fingerprint, manifest_identity = record
    payload_identity = payload.get("classifier_identity")
    if (
        type(payload_identity) is not str
        or not payload_identity
        or manifest_identity != payload_identity
    ):
        # manifest/payload 对同一语义身份自相矛盾，属于不可信完整性错误而非正常 stale miss。
        cache.quarantine_entry(sample.sample_id, "category")
        return None
    effective_identity = payload_identity
    if not accept_stored_identity and classifier_identity != payload_identity:
        return None
    expected_fingerprint = sample_fingerprint(
        "category", sample, config, classifier_identity=effective_identity
    )
    if stored_fingerprint != expected_fingerprint:
        return None
    try:
        if set(payload) != {"assignments", "classifier_identity"}:
            raise ValueError
        values = payload["assignments"]
        if not isinstance(values, list):
            raise ValueError
        assignments = tuple(
            CategoryAssignment(item["annotation_index"], item["category"])
            for item in values
        )
        return assignments, effective_identity
    except (KeyError, TypeError, ValueError) as exc:
        raise _MissingStageCache("category cache payload is invalid") from exc


def _classifier_cache_identity(
    classifier: HazardClassifier,
    sample: ImageSample,
    objects: tuple[YoloObject, ...],
    masks: tuple[ObstacleMask, ...],
) -> str | None:
    """读取可选身份接口；旧第三方实现缺失时返回 None 并关闭类别复用。"""
    if not isinstance(classifier, CacheIdentityProvider):
        return None
    identity = classifier.cache_identity(sample.sample_id, objects, masks)
    return _normalized_classifier_identity(identity)


def _normalized_classifier_identity(identity: object) -> str:
    """保留内置摘要；第三方原始身份统一哈希，禁止路径/token进入缓存。"""
    if type(identity) is not str or not identity or len(identity) > 512:
        raise ValueError("classifier cache identity must be a nonempty short string")
    prefix, separator, digest = identity.partition(":")
    if (
        separator
        and prefix in {"minimax", "manual"}
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        return identity
    # 第三方实现可能错误地把路径、token 等原文当作身份；只持久化稳定摘要。
    return "external:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _supports_atomic_classification(classifier: HazardClassifier) -> bool:
    return isinstance(classifier, AtomicCacheAwareHazardClassifier)


def _partition_category_targets(
    sample: ImageSample,
    masks: tuple[ObstacleMask, ...],
    config: PipelineConfig,
) -> tuple[
    tuple[YoloObject, ...],
    tuple[ObstacleMask, ...],
    tuple[CategoryAssignment, ...],
]:
    masks_by_id = {item.annotation_index: item for item in masks}
    model_objects: list[YoloObject] = []
    model_masks: list[ObstacleMask] = []
    deterministic: list[CategoryAssignment] = []
    for obj in sample.objects:
        mask = masks_by_id[obj.annotation_index]
        if is_non_risk_class(obj.name, config.non_risk_class_names):
            deterministic.append(CategoryAssignment(obj.annotation_index, "Other"))
        else:
            model_objects.append(obj)
            model_masks.append(mask)
    return tuple(model_objects), tuple(model_masks), tuple(deterministic)


def _merge_category_assignments(
    sample: ImageSample,
    *groups: tuple[CategoryAssignment, ...],
) -> tuple[CategoryAssignment, ...]:
    by_id: dict[int, CategoryAssignment] = {}
    for group in groups:
        for assignment in group:
            if assignment.annotation_index in by_id:
                raise ValueError("category assignments contain duplicate annotation_index")
            by_id[assignment.annotation_index] = assignment
    expected = {obj.annotation_index for obj in sample.objects}
    if set(by_id) != expected:
        raise ValueError("category assignments must exactly match sample objects")
    return tuple(by_id[obj.annotation_index] for obj in sample.objects)


def _classify_with_bound_identity(
    classifier: HazardClassifier,
    sample: ImageSample,
    image: Image.Image,
    overlay_path: Path,
    objects: tuple[YoloObject, ...],
    masks: tuple[ObstacleMask, ...],
) -> tuple[tuple[CategoryAssignment, ...], str]:
    """优先消费原子结果；旧第三方 classify 结果只能发布为不可复用身份。"""
    if isinstance(classifier, AtomicCacheAwareHazardClassifier):
        result = classifier.classify_with_cache_identity(
            sample.sample_id, image, overlay_path, objects, masks
        )
        if type(result) is not tuple or len(result) != 2:
            raise TypeError(
                "classify_with_cache_identity must return (assignments, identity)"
            )
        assignments, raw_identity = result
        if type(assignments) is not tuple:
            raise TypeError("classifier assignments must be a tuple")
        return assignments, _normalized_classifier_identity(raw_identity)

    assignments = classifier.classify(
        sample.sample_id, image, overlay_path, objects, masks
    )
    return assignments, _unidentified_result_identity(assignments)


def _unidentified_result_identity(
    assignments: tuple[CategoryAssignment, ...],
) -> str:
    """未知分类器的结果仍可供 export_only 使用，但该身份永不用于下次分类复用。"""
    serialized = json.dumps(
        [
            {"annotation_index": item.annotation_index, "category": item.category}
            for item in assignments
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "unidentified-result:" + hashlib.sha256(serialized).hexdigest()


def _required(value: Any, stage: str) -> Any:
    if value is None:
        raise _MissingStageCache(f"required {stage} cache is missing or stale")
    return value


def _validated_depth_for_sample(
    result: DepthResult, sample: ImageSample
) -> DepthResult:
    """绑定实时/缓存 depth 与样本尺寸，并保留 Task 5 的 dtype/有限性契约。"""
    if not isinstance(result, DepthResult):
        raise TypeError("depth service must return DepthResult")
    raw = np.asarray(result.raw)
    depth_8bit = np.asarray(result.depth_8bit)
    expected_shape = (sample.height, sample.width)
    if (
        raw.dtype != np.float32
        or raw.shape != expected_shape
        or not np.isfinite(raw).all()
    ):
        raise ValueError("raw depth must be finite float32 with sample image shape")
    if depth_8bit.dtype != np.uint8 or depth_8bit.shape != expected_shape:
        raise ValueError("8-bit depth must be uint8 with sample image shape")
    return result


def _validated_sky_for_sample(sky_mask: Any, sample: ImageSample) -> np.ndarray:
    """绑定实时/缓存 sky mask 与样本尺寸；禁止错误同形数组把空标签图写成 safe。"""
    values = np.asarray(sky_mask)
    if values.dtype != np.bool_ or values.shape != (sample.height, sample.width):
        raise ValueError("sky mask must be bool with sample image shape")
    return values


def _remove_formal_ssi_best_effort(path: Path) -> bool:
    """仅撤下一个明确 formal SSI 文件；任何 OS 错误转为 False，不覆盖原异常。"""
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _write_qc_best_effort(
    output_root: Path, sample_id: str, record: dict[str, Any]
) -> Path:
    """尽最大努力写逐图 QC；报告目录故障也不能阻止后续图片。"""
    planned = artifact_path(output_root / "reports", sample_id, suffix=".qc.json")
    try:
        return write_qc_record(output_root, sample_id, record)
    except Exception:
        return planned


def _discovery_failure_outcome(
    item: DiscoveryFailure, config: PipelineConfig
) -> SampleOutcome:
    """把可定位的单图发现错误收敛为 failed QC，并安全撤下该样本旧 SSI。"""
    ssi_path = artifact_path(config.output_root / "ssi", item.sample_id, suffix=".json")
    reasons = [item.reason_code]
    if not _remove_formal_ssi_best_effort(ssi_path):
        reasons.append("ssi_cleanup_failed")
    reason_codes = tuple(reasons)
    qc_path = _write_qc_best_effort(
        config.output_root,
        item.sample_id,
        _qc_document("failed", ["discovered"], reason_codes, None, []),
    )
    return SampleOutcome(item.sample_id, "failed", None, qc_path, reason_codes)


def _discovery_filtered_outcome(
    item: DiscoveryFiltered, config: PipelineConfig
) -> SampleOutcome:
    """将输入过滤记录写成不触发模型/API 的独立 QC 结果。"""
    ssi_path = artifact_path(config.output_root / "ssi", item.sample_id, suffix=".json")
    reasons = [item.reason_code]
    if not _remove_formal_ssi_best_effort(ssi_path):
        reasons.append("ssi_cleanup_failed")
    reason_codes = tuple(reasons)
    qc_document = _qc_document(
        "filtered", ["discovered", "input_filtered"], reason_codes, None, []
    )
    qc_document["input_filter"] = {
        "object_count": item.object_count,
        "max_object_count_exclusive": item.max_object_count_exclusive,
        "reason_code": item.reason_code,
    }
    if item.total_object_count is not None:
        qc_document["input_filter"].update(
            {
                "total_object_count": item.total_object_count,
                "whitelisted_object_count": item.whitelisted_object_count,
            }
        )
    qc_path = _write_qc_best_effort(
        config.output_root, item.sample_id, qc_document
    )
    return SampleOutcome(item.sample_id, "filtered", None, qc_path, reason_codes)


def _qc_document(
    status: str,
    state_history: list[str],
    reason_codes: tuple[str, ...],
    decision: QualityDecision | None,
    cache_hits: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "state_history": state_history,
        "reason_codes": list(reason_codes),
        "quality": None if decision is None else _quality_payload(decision),
        "cache_hits": cache_hits,
    }


def _emit_progress(
    reporter: ProgressReporter | None, event: ProgressEvent
) -> None:
    if reporter is None:
        return
    try:
        reporter.emit(event)
    except Exception:
        return


def process_sample(
    sample: ImageSample,
    services: PipelineServices,
    config: PipelineConfig,
    *,
    progress: ProgressReporter | None = None,
    progress_index: int = 0,
    progress_total: int = 0,
) -> SampleOutcome:
    """用注入服务和不可变配置执行一张图片，返回 ``SampleOutcome``。

    ``sample`` 的 bbox/mask 内部采用 top-left 像素坐标，正式 SSI 才转换为 Unity
    bottom-left。函数可能读取/写入 ``config.output_root`` 缓存、审查工件、QC 和 SSI，
    也可能调用 ``services``；任何阶段异常都收敛为逐图 failed QC。正式 SSI 只有完整
    通过导出时才存在；失败会撤下该样本旧 SSI，但永不触碰输入文件和其他样本产物。
    """
    config.validate()
    if not isinstance(sample, ImageSample) or not isinstance(services, PipelineServices):
        raise TypeError("sample and services must use the public pipeline contracts")

    cache = StageCache(config.output_root / "cache")
    ssi_path = artifact_path(config.output_root / "ssi", sample.sample_id, suffix=".json")
    state_history = ["discovered"]
    cache_hits: list[str] = []
    decision: QualityDecision | None = None
    required_local = config.run_stage in {"classify_only", "export_only"}
    use_optional_cache = config.resume and not config.overwrite
    stage_started_at = time.monotonic()

    def emit_stage_start(stage: ProgressStage) -> None:
        nonlocal stage_started_at
        stage_started_at = time.monotonic()
        _emit_progress(progress, ProgressEvent(
            kind="stage_start", timestamp=stage_started_at,
            index=progress_index, total=progress_total, sample_id=sample.sample_id,
            object_count=len(sample.objects), stage=stage,
        ))

    def emit_stage_end(stage: ProgressStage, *, cached: bool = False) -> None:
        now = time.monotonic()
        _emit_progress(progress, ProgressEvent(
            kind="stage_end", timestamp=now,
            index=progress_index, total=progress_total, sample_id=sample.sample_id,
            object_count=len(sample.objects), stage=stage, cached=cached,
            elapsed_seconds=max(now - stage_started_at, 0.0),
        ))

    emit_stage_start(ProgressStage.READ_IMAGE)
    try:
        # 清理属于状态机的一部分：Windows sharing violation 等错误必须变成单图 failed，
        # 不能在 try 外终止整个批次，也不能在 except 中让第二次 unlink 覆盖原异常。
        if not _remove_formal_ssi_best_effort(ssi_path):
            raise _FormalSSICleanupError("formal SSI could not be removed")
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        if image.size != (sample.width, sample.height):
            raise ValueError("sample dimensions do not match decoded image")
        emit_stage_end(ProgressStage.READ_IMAGE)
        emit_stage_start(ProgressStage.MASK)

        # 1. mask：空标签图片明确跳过模型，但仍进入 mask_ready 状态。
        if not sample.objects:
            masks: tuple[ObstacleMask, ...] = ()
        elif required_local:
            masks = _required(_load_masks(cache, sample, config), "mask")
            cache_hits.append("mask")
        else:
            masks = _load_masks(cache, sample, config) if use_optional_cache else None
            if masks is None:
                masks = services.masks.extract(image, sample.objects)
                _save_masks(cache, sample, config, masks)
            else:
                cache_hits.append("mask")
        state_history.append("mask_ready")
        emit_stage_end(ProgressStage.MASK, cached="mask" in cache_hits)
        emit_stage_start(ProgressStage.DEPTH)

        # 2. depth/sky：即使 YOLO 标签为空也必须执行整图质量检查。
        if required_local:
            depth = _required(_load_depth(cache, sample, config), "depth")
            cache_hits.append("depth")
        else:
            depth = _load_depth(cache, sample, config) if use_optional_cache else None
            if depth is None:
                depth = services.depth.predict(image)
                _validated_depth_for_sample(depth, sample)
                _save_depth(cache, sample, config, depth)
            else:
                _validated_depth_for_sample(depth, sample)
                cache_hits.append("depth")
        depth = _validated_depth_for_sample(depth, sample)
        emit_stage_end(ProgressStage.DEPTH, cached="depth" in cache_hits)
        emit_stage_start(ProgressStage.SKY_QUALITY)

        if required_local:
            sky_mask = _required(_load_sky(cache, sample, config), "sky")
            cache_hits.append("sky")
        else:
            sky_mask = _load_sky(cache, sample, config) if use_optional_cache else None
            if sky_mask is None:
                sky_mask = services.sky.predict_sky_mask(image)
                sky_mask = _validated_sky_for_sample(sky_mask, sample)
                _save_sky(cache, sample, config, sky_mask)
            else:
                sky_mask = _validated_sky_for_sample(sky_mask, sample)
                cache_hits.append("sky")
        sky_mask = _validated_sky_for_sample(sky_mask, sample)
        state_history.append("depth_ready")

        quality_fp = sample_fingerprint("quality", sample, config)
        quality_payload = (
            cache.load_json(sample.sample_id, "quality", quality_fp)
            if (required_local or use_optional_cache) else None
        )
        if quality_payload is None:
            if required_local:
                raise _MissingStageCache("required quality cache is missing or stale")
            decision = evaluate_quality(image, depth.depth_8bit, sky_mask, config)
            cache.save_json(
                sample.sample_id, "quality", quality_fp, _quality_payload(decision)
            )
        else:
            decision = _quality_from_payload(quality_payload, config)
            cache_hits.append("quality")
        emit_stage_end(
            ProgressStage.SKY_QUALITY,
            cached="sky" in cache_hits and "quality" in cache_hits,
        )

        reason_codes = tuple(reason.code for reason in decision.reasons)
        if "sky_missing" in reason_codes:
            state_history.append("failed")
            qc_path = write_qc_record(
                config.output_root,
                sample.sample_id,
                _qc_document(
                    "failed", state_history, reason_codes, decision, cache_hits
                ),
            )
            return SampleOutcome(
                sample.sample_id, "failed", None, qc_path, reason_codes
            )
        if decision.status == "rejected":
            state_history.append("rejected")
            qc_path = write_qc_record(
                config.output_root, sample.sample_id,
                _qc_document("rejected", state_history, reason_codes, decision, cache_hits),
            )
            return SampleOutcome(sample.sample_id, "rejected", None, qc_path, reason_codes)
        state_history.append("quality_passed")
        emit_stage_start(
            ProgressStage.EXPORT if config.run_stage == "local_only"
            else ProgressStage.MINIMAX
        )

        representative_depths = {
            item.annotation_index: representative_depth(depth.depth_8bit, item.mask)
            for item in masks
        }
        distance_steps = {
            index: depth_to_steps(value)
            for index, value in representative_depths.items()
        }

        # 本地阶段只产出可复核视觉结果；类别和正式 SSI 明确留给后续阶段。
        overlay = render_overlay(
            image, sample, masks, representative_depths, distance_steps
        )
        # overlay 是分类输入之一；关闭持久保存时，只在实际分类调用期间创建临时 PNG。
        overlay_path = artifact_path(
            config.output_root / "overlays", sample.sample_id, suffix=".png"
        )
        if config.save_overlays:
            overlay_path = write_overlay_artifact(
                config.output_root, sample.sample_id, overlay
            )
        if config.save_masks and masks:
            write_mask_artifacts(config.output_root, sample, masks)
        raw_depth_path, _depth_png_path = write_depth_artifacts(
            config.output_root, sample.sample_id, depth
        )
        if not config.save_raw_depth:
            # 8-bit 图是公式输入和必要审计产物；开关只控制额外的 raw depth NPZ。
            raw_depth_path.unlink(missing_ok=True)

        if config.run_stage == "local_only":
            emit_stage_end(ProgressStage.EXPORT)
            emit_stage_start(ProgressStage.FINALIZE)
            qc_path = write_qc_record(
                config.output_root, sample.sample_id,
                _qc_document("category_pending", state_history, reason_codes, decision, cache_hits),
            )
            emit_stage_end(ProgressStage.FINALIZE)
            return SampleOutcome(
                sample.sample_id, "category_pending", None, qc_path, reason_codes
            )

        model_objects, model_masks, deterministic_assignments = (
            _partition_category_targets(sample, masks, config)
        )

        # 3. category：export_only 只读；full/classify_only 可命中缓存或调用注入分类器。
        category_was_computed = False
        if config.run_stage == "export_only":
            cached_category = _required(
                _load_categories(
                    cache,
                    sample,
                    config,
                    None,
                    accept_stored_identity=True,
                ),
                "category",
            )
            assignments, category_identity = cached_category
            cache_hits.append("category")
        else:
            current_identity = (
                "empty-category-v1"
                if not sample.objects
                else (
                    f"{CATEGORY_POLICY_VERSION}-no-model-targets"
                    if not model_objects
                    else (
                        _classifier_cache_identity(
                            services.classifier, sample, model_objects, model_masks
                        )
                        if (
                            services.classifier is not None
                            and _supports_atomic_classification(services.classifier)
                        )
                        else None
                    )
                )
            )
            cached_category = (
                _load_categories(
                    cache, sample, config, current_identity
                )
                if use_optional_cache
                else None
            )
            category_was_computed = cached_category is None
            if cached_category is None:
                if not sample.objects:
                    assignments = ()
                    category_identity = current_identity
                elif not model_objects:
                    assignments = deterministic_assignments
                    category_identity = current_identity
                elif services.classifier is None:
                    raise RuntimeError("classifier service is required for this stage")
                else:
                    temporary_overlay: Path | None = None
                    try:
                        if not config.save_overlays:
                            with tempfile.NamedTemporaryFile(
                                mode="w+b", suffix=".png", delete=False
                            ) as stream:
                                temporary_overlay = Path(stream.name)
                                overlay.save(stream, format="PNG")
                            overlay_path = temporary_overlay
                        assignments, category_identity = _classify_with_bound_identity(
                            services.classifier,
                            sample,
                            image,
                            overlay_path,
                            model_objects,
                            model_masks,
                        )
                        assignments = _merge_category_assignments(
                            sample, deterministic_assignments, assignments
                        )
                    finally:
                        if temporary_overlay is not None:
                            temporary_overlay.unlink(missing_ok=True)
            else:
                assignments, category_identity = cached_category
                cache_hits.append("category")
        # 先复用正式导出器验证 ID 集、枚举和几何完整性，再发布类别缓存。这样分类器
        # 即使违反 Protocol，也不会留下可被 export_only 误认的半完整 category cache。
        emit_stage_end(ProgressStage.MINIMAX, cached="category" in cache_hits)
        emit_stage_start(ProgressStage.EXPORT)
        validated_document = build_ssi_document(
            sample, masks, distance_steps, assignments
        )
        if config.save_overlays:
            final_overlay = render_overlay(
                image, sample, masks, representative_depths, distance_steps, assignments
            )
            write_overlay_artifact(
                config.output_root, sample.sample_id, final_overlay
            )
        if config.run_stage != "export_only" and category_was_computed:
            _save_categories(
                cache, sample, config, assignments, category_identity
            )
        # Protocol 当前只返回已验证 assignments，不暴露供应商 raw body。开关因此只
        # 控制一份诚实、脱敏、可公开审计的分类结果副本；无论类别来自在线、人工适配器
        # 或 category cache，均在 SSI 契约验证后独立写入。恢复所需 category cache 始终
        # 保留，绝不受此展示开关影响。
        if config.save_api_responses and sample.objects:
            atomic_write_json(
                artifact_path(
                    config.output_root / "api_responses",
                    sample.sample_id,
                    suffix=".json",
                ),
                {
                    "source": "validated_category_assignments",
                    "assignments": [
                        {
                            "annotation_index": item.annotation_index,
                            "category": item.category,
                        }
                        for item in assignments
                    ],
                },
            )
        state_history.append("category_ready")

        if config.run_stage == "classify_only":
            emit_stage_end(ProgressStage.EXPORT)
            emit_stage_start(ProgressStage.FINALIZE)
            qc_path = write_qc_record(
                config.output_root, sample.sample_id,
                _qc_document("category_pending", state_history, reason_codes, decision, cache_hits),
            )
            emit_stage_end(ProgressStage.FINALIZE)
            return SampleOutcome(
                sample.sample_id, "category_pending", None, qc_path, reason_codes
            )

        write_ssi_atomic(validated_document, ssi_path)
        cache.save_json(
            sample.sample_id,
            "ssi",
            sample_fingerprint(
                "ssi",
                sample,
                config,
                classifier_identity=category_identity,
            ),
            validated_document,
        )
        state_history.append("ssi_written")
        status = "warning" if decision.status == "warning" else (
            "safe" if not sample.objects else "hazard"
        )
        emit_stage_end(ProgressStage.EXPORT)
        emit_stage_start(ProgressStage.FINALIZE)
        qc_path = write_qc_record(
            config.output_root, sample.sample_id,
            _qc_document(status, state_history, reason_codes, decision, cache_hits),
        )
        emit_stage_end(ProgressStage.FINALIZE)
        return SampleOutcome(sample.sample_id, status, ssi_path, qc_path, reason_codes)
    except Exception as exc:
        # 失败清理只针对本样本的正式输出；异常正文不写报告，避免 token/路径信息泄漏。
        cleanup_ok = _remove_formal_ssi_best_effort(ssi_path)
        if isinstance(exc, _MissingStageCache):
            reason = "cache_missing"
        elif isinstance(exc, _FormalSSICleanupError):
            reason = "ssi_cleanup_failed"
        else:
            reason = type(exc).__name__
        reasons = [reason]
        if not cleanup_ok and reason != "ssi_cleanup_failed":
            reasons.append("ssi_cleanup_failed")
        reason_codes = tuple(reasons)
        qc_path = _write_qc_best_effort(
            config.output_root, sample.sample_id,
            _qc_document("failed", state_history, reason_codes, decision, cache_hits),
        )
        return SampleOutcome(sample.sample_id, "failed", None, qc_path, reason_codes)


def _build_default_services(config: PipelineConfig) -> PipelineServices:
    """按运行模式构造默认服务；只有需要默认分类器时读取 MiniMax 配置。

    classify_only/export_only 的占位服务不会导入 Transformers，也不会加载 checkpoint。
    调用方注入 ``PipelineServices`` 时 ``run_batch`` 完全跳过本函数，因而不会读取配置文件。
    """
    classifier: HazardClassifier | None = None
    if config.run_stage in {"full", "classify_only"}:
        credentials = load_minimax_credentials(
            config.minimax_config_path,
            expected_model=config.minimax_model,
        )
        from openai import OpenAI

        from .hazard_classifier import MiniMaxHazardClassifier

        client = OpenAI(
            api_key=credentials.api_key,
            base_url=credentials.base_url,
            timeout=config.minimax_timeout_seconds,
            max_retries=0,
        )
        classifier = MiniMaxHazardClassifier(config, client=client)

    if config.run_stage in {"classify_only", "export_only"}:
        unavailable = _UnavailableLocalService()
        return PipelineServices(unavailable, unavailable, unavailable, classifier)

    from .consistency_filter import SegFormerSkyService
    from .depth_estimation import DepthAnythingV2Service
    from .device import select_device
    from .mask_extraction import GroundedSam2MaskService
    from .model_registry import ModelRegistry

    registry = ModelRegistry(config, select_device(config.device))
    return PipelineServices(
        GroundedSam2MaskService(registry, config),
        DepthAnythingV2Service(registry),
        SegFormerSkyService(registry),
        classifier,
    )


def _print_batch_progress(
    completed: int, total: int, sample_id: str, status: str
) -> None:
    if total <= 0:
        return
    filled = min(10, int(completed / total * 10))
    if completed >= total:
        filled = 10
    bar = "#" * filled + "-" * (10 - filled)
    percentage = completed / total * 100
    print(
        f"Progress: [{bar}] {percentage:.1f}% "
        f"({completed}/{total}) {sample_id} -> {status}",
        end="\r",
        flush=True,
    )


def run_batch(
    config: PipelineConfig,
    services: PipelineServices | None = None,
    *,
    progress: ProgressReporter | None = None,
) -> BatchSummary:
    """Discover and process all configured samples with optional detailed progress."""
    config.validate()
    items = discover_dataset_items(config)
    total_items = len(items)
    _emit_progress(progress, ProgressEvent(
        kind="batch_start", timestamp=time.monotonic(), index=0, total=total_items,
    ))
    try:
        if config.dry_run:
            outcomes_list = []
            for index, item in enumerate(items, start=1):
                sample_started_at = time.monotonic()
                object_count = len(item.objects) if isinstance(item, ImageSample) else getattr(item, "object_count", 0)
                _emit_progress(progress, ProgressEvent.sample_started(
                    timestamp=sample_started_at, index=index, total=total_items,
                    sample_id=item.sample_id, object_count=object_count,
                ))
                outcome = SampleOutcome(
                    item.sample_id,
                    "failed" if isinstance(item, DiscoveryFailure) else "filtered" if isinstance(item, DiscoveryFiltered) else "category_pending",
                    None,
                    artifact_path(config.output_root / "reports", item.sample_id, suffix=".qc.json"),
                    (item.reason_code,) if isinstance(item, (DiscoveryFailure, DiscoveryFiltered)) else ("dry_run",),
                )
                outcomes_list.append(outcome)
                elapsed = time.monotonic() - sample_started_at
                _emit_progress(progress, ProgressEvent(
                    kind="sample_end", timestamp=time.monotonic(), index=index, total=total_items,
                    sample_id=outcome.sample_id, object_count=object_count,
                    stage=ProgressStage.FINALIZE, status=outcome.status, elapsed_seconds=elapsed,
                ))
                if progress is None:
                    _print_batch_progress(index, total_items, outcome.sample_id, outcome.status)
            outcomes = tuple(outcomes_list)
            if outcomes and progress is None:
                print(flush=True)
            summary = BatchSummary(
                outcomes=outcomes,
                succeeded=sum(item.status == "category_pending" for item in outcomes),
                rejected=0,
                failed=sum(item.status == "failed" for item in outcomes),
                filtered=sum(item.status == "filtered" for item in outcomes),
            )
            _emit_progress(progress, ProgressEvent(
                kind="batch_end", timestamp=time.monotonic(), index=total_items, total=total_items,
            ))
            return summary

        has_eligible_samples = any(isinstance(item, ImageSample) for item in items)
        active_services = services if services is not None else _build_default_services(config) if has_eligible_samples else None
        outcomes_list = []
        for index, item in enumerate(items, start=1):
            sample_started_at = time.monotonic()
            object_count = len(item.objects) if isinstance(item, ImageSample) else getattr(item, "object_count", 0)
            _emit_progress(progress, ProgressEvent.sample_started(
                timestamp=sample_started_at, index=index, total=total_items,
                sample_id=item.sample_id, object_count=object_count,
            ))
            if isinstance(item, (DiscoveryFailure, DiscoveryFiltered)):
                filter_started_at = time.monotonic()
                _emit_progress(progress, ProgressEvent(
                    kind="stage_start", timestamp=filter_started_at, index=index, total=total_items,
                    sample_id=item.sample_id, object_count=object_count, stage=ProgressStage.INPUT_FILTER,
                ))
                outcome = _discovery_failure_outcome(item, config) if isinstance(item, DiscoveryFailure) else _discovery_filtered_outcome(item, config)
                filter_finished_at = time.monotonic()
                _emit_progress(progress, ProgressEvent(
                    kind="stage_end", timestamp=filter_finished_at, index=index, total=total_items,
                    sample_id=item.sample_id, object_count=object_count, stage=ProgressStage.INPUT_FILTER,
                    elapsed_seconds=filter_finished_at - filter_started_at,
                ))
            else:
                outcome = process_sample(
                    item, active_services, config, progress=progress,
                    progress_index=index, progress_total=total_items,
                )
            outcomes_list.append(outcome)
            elapsed = time.monotonic() - sample_started_at
            _emit_progress(progress, ProgressEvent(
                kind="sample_end", timestamp=time.monotonic(), index=index, total=total_items,
                sample_id=outcome.sample_id, object_count=object_count,
                stage=ProgressStage.FINALIZE, status=outcome.status, elapsed_seconds=elapsed,
            ))
            if progress is None:
                _print_batch_progress(index, total_items, outcome.sample_id, outcome.status)
        outcomes = tuple(outcomes_list)
        if outcomes and progress is None:
            print(flush=True)
        summary = BatchSummary(
            outcomes=outcomes,
            succeeded=sum(item.status in {"safe", "hazard", "warning", "category_pending"} for item in outcomes),
            rejected=sum(item.status == "rejected" for item in outcomes),
            failed=sum(item.status == "failed" for item in outcomes),
            filtered=sum(item.status == "filtered" for item in outcomes),
        )
        write_batch_summary(config.output_root, summary)
        _emit_progress(progress, ProgressEvent(
            kind="batch_end", timestamp=time.monotonic(), index=total_items, total=total_items,
        ))
        return summary
    finally:
        if progress is not None:
            try:
                progress.close()
            except Exception:
                pass
