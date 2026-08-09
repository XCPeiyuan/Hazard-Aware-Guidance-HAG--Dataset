"""障碍物严格三分类适配器。

MiniMax 适配器只负责构造一次多图 Chat Completions 请求和验证结构化结果；调用方
必须显式注入已构造的客户端，因此本模块不会读取 API key，也不会隐式创建
网络客户端。人工文件适配器复用同一套 ID/枚举验证，供离线测试和审查使用。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from openai import APIConnectionError, APIStatusError
from PIL import Image

from real_world_ssi.schemas import (
    CategoryAssignment,
    HazardCategory,
    ObstacleMask,
    PipelineConfig,
    YoloObject,
)


HAZARD_CATEGORIES: tuple[HazardCategory, ...] = (
    "Common Obstacle",
    "Pitfall Hazard",
    "Upper-body Hazard",
)

# Structured Outputs 只约束响应外形；下方应用层验证仍会证明 ID 集合与输入完全一致。
HAZARD_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "annotation_index": {"type": "integer"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "Common Obstacle",
                            "Pitfall Hazard",
                            "Upper-body Hazard",
                        ],
                    },
                },
                "required": ["annotation_index", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}

_MAX_BACKOFF_SECONDS = 2.0
_MINIMAX_ADAPTER_IDENTITY_VERSION = "minimax-m3-hazard-adapter-v1-mask-area"
_MANUAL_ADAPTER_IDENTITY_VERSION = "manual-hazard-adapter-v1"


class HazardClassificationError(RuntimeError):
    """分类输入、API 响应或人工映射不满足严格契约。"""


class _PayloadValidationError(ValueError):
    """内部可重试错误；消息只能使用固定文本，不能回显响应正文。"""


class _ChatCompletionsEndpoint(Protocol):
    def create(self, **kwargs: object) -> object:
        """提交一个 Chat Completions 请求。"""


class _ChatNamespace(Protocol):
    completions: _ChatCompletionsEndpoint


class _ChatClient(Protocol):
    chat: _ChatNamespace


def _field(value: object, name: str, default: object = None) -> object:
    """同时读取 SDK Pydantic 对象和等价映射，便于使用无网络 fake。"""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _strict_input_ids(
    objects: tuple[YoloObject, ...],
    masks: tuple[ObstacleMask, ...],
) -> tuple[int, ...]:
    """验证对象和掩码都具有唯一、完全相同的非负标注序号。"""
    object_ids = tuple(item.annotation_index for item in objects)
    mask_ids = tuple(item.annotation_index for item in masks)

    for ids in (object_ids, mask_ids):
        if any(type(value) is not int or value < 0 for value in ids):
            raise HazardClassificationError(
                "annotation_index must be a non-negative integer"
            )
        if len(set(ids)) != len(ids):
            raise HazardClassificationError("annotation_index values must be unique")
    if set(object_ids) != set(mask_ids):
        raise HazardClassificationError(
            "object and mask annotation_index sets must exactly match"
        )
    return object_ids


def _stable_identity(prefix: str, payload: object) -> str:
    """生成不含密钥、图像或路径的可审计语义身份摘要。"""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(serialized).hexdigest()}"


def _safe_exception_type(exc: BaseException) -> str:
    """保留可审计类型；拒绝异常类名本身携带超长或非标识符内容。"""
    name = type(exc).__name__
    return name if len(name) <= 64 and name.isidentifier() else "Exception"


def _png_data_url(
    image: Image.Image,
    *,
    source: Literal["original image", "overlay image"],
) -> str:
    """把图像编码为 PNG data URL，仅在错误中保留安全来源和异常类型。"""
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as exc:
        raise HazardClassificationError(
            f"{source} encoding failed ({_safe_exception_type(exc)})"
        ) from None
    return f"data:image/png;base64,{payload}"


def _overlay_data_url(overlay_path: Path) -> str:
    """读取审查叠加图并规范编码为 PNG，避免依赖扩展名猜测 MIME。"""
    try:
        with Image.open(overlay_path) as overlay:
            overlay.load()
            return _png_data_url(overlay, source="overlay image")
    except HazardClassificationError:
        raise
    except Exception as exc:
        raise HazardClassificationError(
            "overlay image open/load failed "
            f"({_safe_exception_type(exc)})"
        ) from None


def _prompt_text(
    sample_id: str,
    objects: tuple[YoloObject, ...],
    masks: tuple[ObstacleMask, ...],
) -> str:
    """生成只含审查元数据的提示；像素由两个独立图像内容项承载。"""
    masks_by_id = {item.annotation_index: item for item in masks}
    rows = []
    for item in objects:
        mask = masks_by_id[item.annotation_index]
        rows.append(
            "- annotation_index="
            f"{item.annotation_index}; class_id={item.class_id}; name={item.name!r}; "
            f"bbox_xyxy={item.bbox_xyxy}; mask_source={mask.source}; "
            f"centroid_top_left={mask.centroid_top_left}; "
            f"mask_area_pixels={int(np.count_nonzero(mask.mask))}"
        )
    object_block = "\n".join(rows)
    return (
        "Classify every listed walking hazard into exactly one canonical category: "
        "Common Obstacle, Pitfall Hazard, or Upper-body Hazard. "
        "Use the original image for scene context and the overlay for indexed regions. "
        "Return every annotation_index exactly once and do not invent IDs. "
        "Return only a JSON object with an assignments array; do not use Markdown.\n"
        f"sample_id={sample_id}\n{object_block}"
    )


def _request_kwargs(
    config: PipelineConfig,
    sample_id: str,
    image: Image.Image,
    overlay_path: Path,
    objects: tuple[YoloObject, ...],
    masks: tuple[ObstacleMask, ...],
) -> dict[str, object]:
    """按 SDK 2.30.0 构造 MiniMax 单消息、双图像 Chat 请求。"""
    return {
        "model": config.minimax_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _prompt_text(sample_id, objects, masks),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _png_data_url(
                                image,
                                source="original image",
                            ),
                            "detail": config.minimax_image_detail,
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _overlay_data_url(overlay_path),
                            "detail": config.minimax_image_detail,
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_completion_tokens": 1024,
        "extra_body": {"thinking": {"type": "adaptive"}},
        "timeout": config.minimax_timeout_seconds,
    }


def _contains_refusal(response: object) -> bool:
    """在读取消息正文前检查首个 Chat Completion 消息是否拒绝。"""
    message = _first_message(response)
    refusal = _field(message, "refusal")
    return isinstance(refusal, str) and bool(refusal.strip())


def _first_message(response: object) -> object:
    """提取首个 Chat Completion 消息，并拒绝缺失或畸形 choices。"""
    choices = _field(response, "choices", ())
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or not choices
    ):
        raise _PayloadValidationError("response choices must contain a message")
    message = _field(choices[0], "message")
    if message is None:
        raise _PayloadValidationError("response choices must contain a message")
    return message


def _strip_leading_thinking_block(content: str) -> str:
    """兼容 M3 偶发返回的单个前置 thinking 块，同时拒绝未闭合内容。"""
    stripped = content.lstrip()
    if not stripped.startswith("<think>"):
        return stripped
    closing_index = stripped.find("</think>")
    if closing_index < 0:
        raise _PayloadValidationError("response thinking block is not closed")
    return stripped[closing_index + len("</think>"):].lstrip()


def _validate_assignments(
    payload: object,
    expected_ids: tuple[int, ...],
) -> tuple[CategoryAssignment, ...]:
    """执行 schema 之外的精确键、精确 ID 集与精确枚举验证。"""
    if not isinstance(payload, dict) or set(payload) != {"assignments"}:
        raise _PayloadValidationError("structured response does not match hazard schema")
    records = payload["assignments"]
    if not isinstance(records, list):
        raise _PayloadValidationError("structured response does not match hazard schema")

    by_id: dict[int, HazardCategory] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "annotation_index",
            "category",
        }:
            raise _PayloadValidationError(
                "structured response does not match hazard schema"
            )
        annotation_index = record["annotation_index"]
        category = record["category"]
        if type(annotation_index) is not int or annotation_index < 0:
            raise _PayloadValidationError(
                "annotation_index must be a non-negative integer"
            )
        if annotation_index in by_id:
            raise _PayloadValidationError("annotation_index values must be unique")
        if type(category) is not str or category not in HAZARD_CATEGORIES:
            raise _PayloadValidationError("category must use the exact canonical enum")
        by_id[annotation_index] = category

    if set(by_id) != set(expected_ids):
        raise _PayloadValidationError(
            "annotation_index set must exactly match input objects"
        )
    return tuple(CategoryAssignment(index, by_id[index]) for index in expected_ids)


def _parse_response(
    response: object,
    expected_ids: tuple[int, ...],
) -> tuple[CategoryAssignment, ...]:
    """解析首个消息正文，但不把原始文本带入异常。"""
    content = _field(_first_message(response), "content")
    if not isinstance(content, str):
        raise _PayloadValidationError("response message content must be JSON text")
    content = _strip_leading_thinking_block(content)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        raise _PayloadValidationError(
            "response message content is malformed JSON"
        ) from None
    return _validate_assignments(payload, expected_ids)


def _retryable_client_error(exc: BaseException) -> bool:
    """仅重试连接/超时和 SDK 标记为临时的 HTTP 状态。"""
    if isinstance(exc, (TimeoutError, ConnectionError, APIConnectionError)):
        return True
    if not isinstance(exc, APIStatusError):
        return False
    status_code = exc.status_code
    return status_code in {408, 409, 429} or status_code >= 500


def _retry_delay(failed_attempt: int) -> float:
    """第 1 次失败等待 0.25 秒，并在 2 秒处封顶。"""
    return min(0.25 * (2 ** (failed_attempt - 1)), _MAX_BACKOFF_SECONDS)


class MiniMaxHazardClassifier:
    """使用注入的 MiniMax Chat client 为一张图中的全部障碍物分类。"""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        client: _ChatClient,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        # SDK 默认会在一次 ``create`` 内部再次重试；关闭该层后，配置值才是端到端
        # 总尝试次数。极简 fake 可不实现 ``with_options``，仍保持纯边界注入。
        with_options = getattr(client, "with_options", None)
        self._client = (
            with_options(max_retries=0) if callable(with_options) else client
        )
        self._sleep = sleep

    def cache_identity(
        self,
        sample_id: str,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> str:
        """绑定适配器提示词/schema 版本及会改变分类语义的 MiniMax 配置。"""
        del sample_id
        _strict_input_ids(objects, masks)
        return _stable_identity(
            "minimax",
            {
                "adapter_version": _MINIMAX_ADAPTER_IDENTITY_VERSION,
                "model": self._config.minimax_model,
                "image_detail": self._config.minimax_image_detail,
                "schema": HAZARD_SCHEMA,
            },
        )

    def classify(
        self,
        sample_id: str,
        image: Image.Image,
        overlay_path: Path,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> tuple[CategoryAssignment, ...]:
        """发送一个请求；拒绝立即失败，临时/API/解析错误按总次数重试。"""
        expected_ids = _strict_input_ids(objects, masks)
        if self._config.minimax_image_detail != "high":
            raise HazardClassificationError("MiniMax image detail must be 'high'")
        total_attempts = self._config.minimax_max_retries
        if type(total_attempts) is not int or total_attempts < 1:
            raise HazardClassificationError(
                "minimax_max_retries must be a positive total attempt count"
            )
        request = _request_kwargs(
            self._config,
            sample_id,
            image,
            overlay_path,
            objects,
            masks,
        )

        last_payload_reason = "classification response was invalid"
        for attempt in range(1, total_attempts + 1):
            try:
                response = self._client.chat.completions.create(**request)
            except Exception as exc:
                if not _retryable_client_error(exc):
                    # 不传播 SDK 异常正文，防止异常中夹带 API key 或 data URL。
                    raise HazardClassificationError(
                        "MiniMax request failed at client boundary "
                        f"({_safe_exception_type(exc)})"
                    ) from None
                last_payload_reason = (
                    "MiniMax request had a transient client error "
                    f"({_safe_exception_type(exc)})"
                )
            else:
                # Refusal 必须先于消息正文访问并且不能重试。
                if _contains_refusal(response):
                    raise HazardClassificationError(
                        "MiniMax response contained a refusal; classification stopped"
                    )
                try:
                    return _parse_response(response, expected_ids)
                except _PayloadValidationError as exc:
                    # 内部验证消息均为固定文本，不包含原始 JSON 或 Base64 正文。
                    last_payload_reason = str(exc)

            if attempt < total_attempts:
                self._sleep(_retry_delay(attempt))

        raise HazardClassificationError(
            f"{last_payload_reason}; exhausted {total_attempts} total attempts"
        )

    def classify_with_cache_identity(
        self,
        sample_id: str,
        image: Image.Image,
        overlay_path: Path,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> tuple[tuple[CategoryAssignment, ...], str]:
        """一次 MiniMax 分类完成后附上稳定、无外部可变读取的适配器身份。"""
        assignments = self.classify(
            sample_id, image, overlay_path, objects, masks
        )
        return assignments, self.cache_identity(sample_id, objects, masks)


class ManualFileHazardClassifier:
    """从 UTF-8 JSON 文件读取人工类别，不访问环境变量或任何网络。"""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def _validated_file_assignments(
        self,
        sample_id: str,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> tuple[tuple[CategoryAssignment, ...], bytes]:
        """一次读取文件 bytes，验证目标 sample 映射后返回规范结果与原始内容。"""
        expected_ids = _strict_input_ids(objects, masks)
        try:
            raw_bytes = self._path.read_bytes()
            document: Any = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HazardClassificationError(
                "manual category file read/parse failed "
                f"({_safe_exception_type(exc)})"
            ) from None
        if not isinstance(document, dict) or sample_id not in document:
            raise HazardClassificationError(
                "manual category file does not contain the requested sample"
            )
        mapping = document[sample_id]
        if not isinstance(mapping, dict):
            raise HazardClassificationError(
                "manual annotation_index mapping must be a JSON object"
            )

        records: list[dict[str, object]] = []
        for raw_index, category in mapping.items():
            if (
                not isinstance(raw_index, str)
                or not raw_index.isdecimal()
                or str(int(raw_index)) != raw_index
            ):
                raise HazardClassificationError(
                    "manual annotation_index keys must be canonical non-negative integers"
                )
            records.append(
                {"annotation_index": int(raw_index), "category": category}
            )
        try:
            assignments = _validate_assignments(
                {"assignments": records}, expected_ids
            )
        except _PayloadValidationError as exc:
            raise HazardClassificationError(str(exc)) from None
        return assignments, raw_bytes

    def cache_identity(
        self,
        sample_id: str,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> str:
        """身份同时绑定已验证映射与实际文件 bytes，任何人工修改都会失效。"""
        assignments, raw_bytes = self._validated_file_assignments(
            sample_id, objects, masks
        )
        return self._identity_for_snapshot(assignments, raw_bytes)

    @staticmethod
    def _identity_for_snapshot(
        assignments: tuple[CategoryAssignment, ...], raw_bytes: bytes
    ) -> str:
        """只使用产生 assignments 的同一份已验证文件快照生成身份。"""
        return _stable_identity(
            "manual",
            {
                "adapter_version": _MANUAL_ADAPTER_IDENTITY_VERSION,
                "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "assignments": [
                    {
                        "annotation_index": item.annotation_index,
                        "category": item.category,
                    }
                    for item in assignments
                ],
            },
        )

    def classify_with_cache_identity(
        self,
        sample_id: str,
        image: Image.Image,
        overlay_path: Path,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> tuple[tuple[CategoryAssignment, ...], str]:
        """单次 read_bytes 同时产生 assignments 与 identity，消除 A→B→A 竞态。"""
        del image, overlay_path
        assignments, raw_bytes = self._validated_file_assignments(
            sample_id, objects, masks
        )
        return assignments, self._identity_for_snapshot(assignments, raw_bytes)

    def classify(
        self,
        sample_id: str,
        image: Image.Image,
        overlay_path: Path,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> tuple[CategoryAssignment, ...]:
        """按样本读取映射，并执行与 MiniMax 输出相同的精确契约。"""
        del image, overlay_path  # 人工适配器特意保持完全离线且不重复读取图像。
        assignments, _raw_bytes = self._validated_file_assignments(
            sample_id, objects, masks
        )
        return assignments
