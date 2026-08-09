"""Real-world SSI 分阶段缓存、原子写入与敏感值过滤。

本模块位于各视觉/API 阶段和 Task 10 状态机之间。论文没有规定缓存格式，故这里
采用可公开审计的 JSON manifest 与压缩 NPZ；绝不反序列化 pickle。所有缓存都视为
不可信输入：路径、manifest、数组键、shape、dtype、维数、有限性、校验和与指纹
必须全部通过，任一失败就隔离而不是交给后续 SSI 阶段。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .category_policy import CATEGORY_POLICY_VERSION
from .schemas import ImageSample, PipelineConfig


JsonObject: TypeAlias = dict[str, Any]

# “代码版本”是缓存语义版本，不取 Git 工作树状态。未来修改序列化或阶段依赖时必须
# 显式递增，使旧缓存自然 miss，而不是尝试猜测旧文件是否仍可复用。
CACHE_SCHEMA_VERSION = "real-world-ssi-cache-v1"
_KNOWN_STAGES = frozenset({"mask", "depth", "sky", "quality", "category", "ssi"})
_SAFE_STAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z", flags=re.ASCII)
_SAFE_ARRAY_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z", flags=re.ASCII)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z", flags=re.ASCII)
_SECRET_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]+", flags=re.ASCII
)
_IMAGE_DATA_URL = re.compile(
    r"data\s*:\s*image/[A-Za-z0-9.+-]+(?:\s*;[^,]*)?\s*;\s*base64\s*,",
    flags=re.ASCII | re.IGNORECASE,
)
_SECRET_FIELD_NAMES = frozenset({"minimax_api_key", "api_key"})
_ALLOWED_ARRAY_DTYPES = frozenset(
    {np.dtype(np.bool_), np.dtype(np.uint8), np.dtype(np.float32)}
)
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_ARRAY_COUNT = 64

# NPZ 来自可被用户替换的恢复目录，不能把 ZIP 中声明的尺寸直接交给 NumPy。
# 下列上限允许单张 8192x8192 float32 深度图（约 256 MiB），也允许至少
# 15 张 4096x4096 bool 障碍物掩码（约 252M elements / 240 MiB）。element
# 上限负责限制对象数量级，bytes 上限仍是内存预算主约束；多 GiB 伪造输入会被拒绝。
_MAX_NPZ_MEMBER_ELEMENTS = 100_000_000
_MAX_NPZ_TOTAL_ELEMENTS = 300_000_000
_MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES = 320 * 1024 * 1024
_MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_NPZ_COMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 64 * 1024


def _validated_stage(stage: str) -> str:
    """只接受本流水线已定义的单段阶段名，拒绝路径和未知缓存命名。"""
    if (
        type(stage) is not str
        or not _SAFE_STAGE.fullmatch(stage)
        or stage not in _KNOWN_STAGES
    ):
        raise ValueError("stage must be a known safe stage name")
    return stage


def safe_sample_parts(sample_id: str) -> tuple[str, ...]:
    """把规范 POSIX sample ID 拆成安全路径段。

    ``discover_dataset`` 会生成 ``train/foo`` 形式的 ID，因此允许正斜杠层级；反斜杠、
    盘符、绝对路径、空段、点段和控制字符一律拒绝，确保 Windows/Unix 均不能穿越。
    """
    if type(sample_id) is not str or not sample_id or "\\" in sample_id:
        raise ValueError("sample_id must be a safe relative POSIX path")

    # 必须先检查原字符串；PurePosixPath 会把 ``a//b``、``a/./b`` 和末尾斜杠
    # 静默规范化，若先构造 Path 就再也无法区分用户输入是否包含危险/歧义段。
    raw_parts = tuple(sample_id.split("/"))
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("sample_id must be a safe relative POSIX path")
    path = PurePosixPath(sample_id)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or "/".join(parts) != sample_id
        or any(":" in part for part in parts)
        or any(any(ord(character) < 32 for character in part) for part in parts)
    ):
        raise ValueError("sample_id must be a safe relative POSIX path")
    return parts


def artifact_path(root: Path, sample_id: str, *, suffix: str) -> Path:
    """在给定根目录下生成 sample 对应路径，不允许 ``suffix`` 引入子目录。"""
    if (
        type(suffix) is not str
        or not suffix
        or "/" in suffix
        or "\\" in suffix
        or "\x00" in suffix
    ):
        raise ValueError("artifact suffix must be a safe filename suffix")
    parts = safe_sample_parts(sample_id)
    return Path(root).joinpath(*parts[:-1], f"{parts[-1]}{suffix}")


def _redact_string(value: str) -> str:
    """过滤环境 key、任意 sk-token 和内嵌 data:image Base64 正文。"""
    # data URL 可能含 MIME 参数，且 Base64 正文允许折行。检测到图片 data URL 时
    # 宁可替换整个字符串，也不保留 prefix/suffix 造成正文切片泄漏。
    if _IMAGE_DATA_URL.search(value):
        return "[REDACTED_BASE64_IMAGE]"
    result = value
    environment_secret = os.environ.get("MINIMAX_API_KEY")
    if environment_secret:
        result = result.replace(environment_secret, "[REDACTED]")
    return _SECRET_TOKEN.sub("[REDACTED]", result)


def _secret_field_name(key: object) -> bool:
    """识别不依赖环境变量的显式 API key 字段名。"""
    return isinstance(key, str) and key.casefold().replace("-", "_") in _SECRET_FIELD_NAMES


def redact_sensitive(value: Any) -> Any:
    """递归返回可写报告的脱敏副本，不修改调用方对象。

    字典 key 也经过过滤，避免异常对象把 token 放在字段名中。未知 Python 对象保持
    原值并在严格 JSON 序列化时失败，不能靠 ``str(obj)`` 泄露其自定义表示。
    """
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Mapping):
        result: dict[object, Any] = {}
        for key, item in value.items():
            redacted_key = _redact_string(key) if isinstance(key, str) else key
            result[redacted_key] = (
                "[REDACTED]" if _secret_field_name(key) else redact_sensitive(item)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item) for item in value]
    return value


def _prepare_file(path: Path, writer: Callable[[BinaryIO], None]) -> Path:
    """在目标同目录写完并 fsync 临时文件，但尚不替换活动路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        prepared_path = temporary_path
        temporary_path = None
        return prepared_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_file(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    """在目标同目录完整写入并 fsync 后以 ``os.replace`` 原子发布。

    同目录临时文件保证替换不跨文件系统。writer、flush 或 replace 任一步失败都会删除
    临时文件，原目标保持原样；这是 cache manifest、报告和图片共用的唯一写入原语。
    """
    target = Path(path)
    prepared_path = _prepare_file(target, writer)
    try:
        os.replace(prepared_path, target)
        prepared_path = None  # type: ignore[assignment] -- replace 后无需清理。
    finally:
        if prepared_path is not None:
            prepared_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    """以 UTF-8/LF 原子写文本；编码在创建临时文件前完成。"""
    if type(text) is not str:
        raise TypeError("text must be a string")
    encoded = text.encode("utf-8")
    atomic_write_file(path, lambda stream: stream.write(encoded))


def _serialized_json(value: Any) -> str:
    """返回与公开 manifest/report 格式完全一致的严格、已脱敏 JSON 文本。"""
    return json.dumps(
        redact_sensitive(value),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def atomic_write_json(path: Path, value: Any) -> None:
    """脱敏后按严格 JSON 原子写入；NaN/Infinity 和未知对象直接失败。"""
    atomic_write_text(path, _serialized_json(value))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> bool:
    """验证已解析 JSON 仍不含 Python json 默认接受的非有限浮点。"""
    if value is None or isinstance(value, (bool, str)):
        return True
    if type(value) is int:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(type(key) is str and _json_safe(item) for key, item in value.items())
    return False


def _validated_array(name: str, value: Any) -> NDArray[Any]:
    """收敛公开缓存允许的数组域：bool/uint8/float32、非空二维或三维、全有限。"""
    if type(name) is not str or not _SAFE_ARRAY_KEY.fullmatch(name):
        raise ValueError("array keys must be safe ASCII identifiers")
    if not isinstance(value, np.ndarray):
        raise TypeError("cache arrays must be NumPy arrays")
    array = np.asarray(value)
    if array.dtype not in _ALLOWED_ARRAY_DTYPES:
        raise ValueError("cache array dtype must be bool, uint8, or float32")
    if array.ndim not in {2, 3} or array.size == 0:
        raise ValueError("cache arrays must be non-empty and two- or three-dimensional")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("floating cache arrays must contain only finite values")
    return array


def _array_manifest(arrays: Mapping[str, NDArray[Any]]) -> dict[str, JsonObject]:
    return {
        name: {
            "shape": [int(dimension) for dimension in array.shape],
            "dtype": array.dtype.name,
            "ndim": array.ndim,
        }
        for name, array in sorted(arrays.items())
    }


def _validated_npz_descriptors(
    descriptors: object,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """在打开 ZIP 前验证 manifest 尺寸，并计算可信的资源预算。"""
    if (
        not isinstance(descriptors, dict)
        or not descriptors
        or len(descriptors) > _MAX_ARRAY_COUNT
    ):
        raise ValueError("NPZ descriptors are invalid")

    validated: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    total_elements = 0
    total_bytes = 0
    for name, descriptor in descriptors.items():
        if (
            type(name) is not str
            or not _SAFE_ARRAY_KEY.fullmatch(name)
            or not isinstance(descriptor, dict)
            or set(descriptor) != {"shape", "dtype", "ndim"}
            or type(descriptor.get("dtype")) is not str
            or type(descriptor.get("ndim")) is not int
            or descriptor["ndim"] not in {2, 3}
            or type(descriptor.get("shape")) is not list
            or len(descriptor["shape"]) != descriptor["ndim"]
            or any(
                type(dimension) is not int or dimension <= 0
                for dimension in descriptor["shape"]
            )
        ):
            raise ValueError("NPZ descriptor is invalid")
        try:
            dtype = np.dtype(descriptor["dtype"])
        except TypeError as exc:
            raise ValueError("NPZ descriptor dtype is invalid") from exc
        if dtype not in _ALLOWED_ARRAY_DTYPES or dtype.name != descriptor["dtype"]:
            raise ValueError("NPZ descriptor dtype is invalid")

        shape = tuple(descriptor["shape"])
        elements = math.prod(shape)
        declared_bytes = elements * dtype.itemsize
        if (
            elements > _MAX_NPZ_MEMBER_ELEMENTS
            or declared_bytes > _MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES
        ):
            raise ValueError("NPZ member exceeds resource limits")
        total_elements += elements
        total_bytes += declared_bytes
        if (
            total_elements > _MAX_NPZ_TOTAL_ELEMENTS
            or total_bytes > _MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES
        ):
            raise ValueError("NPZ archive exceeds resource limits")
        validated[name] = (shape, dtype)
    return validated


def _read_npy_header_bounded(
    stream: BinaryIO,
) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    """先验证长度字段，再把有上限的 header 副本交给 NumPy 解析。

    不能直接调用 ``read_array_header_*`` 读取原始流：其实现会先信任攻击者提供的
    header_length，并立即发起同等尺寸的 ``read``。这里每次读取最多 64 KiB。
    """

    def read_exact(size: int, field: str) -> bytes:
        value = stream.read(size)
        if len(value) != size:
            raise ValueError(f"truncated NPY {field}")
        return value

    magic = read_exact(6, "magic")
    if magic != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    version_bytes = read_exact(2, "version")
    version = (version_bytes[0], version_bytes[1])
    if version == (1, 0):
        length_format = "<H"
        parser = np.lib.format.read_array_header_1_0
    elif version == (2, 0):
        length_format = "<I"
        parser = np.lib.format.read_array_header_2_0
    else:
        # 本项目生成器只会写 v1/v2；拒绝未知版本比调用私有 NumPy API 更可审计。
        raise ValueError("unsupported NPY header version")
    length_size = struct.calcsize(length_format)
    length_bytes = read_exact(length_size, "header length")
    header_length = struct.unpack(length_format, length_bytes)[0]
    if header_length > _MAX_NPY_HEADER_BYTES:
        raise ValueError("NPY header exceeds resource limit")
    header = read_exact(header_length, "header")

    # NumPy 仍负责 header 字典、shape、order 与 dtype 的规范解析；但它只看到已验证的
    # 小型内存副本，因此无法再向底层 ZIP 流请求攻击者声明的超大长度。
    bounded = io.BytesIO(magic + version_bytes + length_bytes + header)
    parsed_version = np.lib.format.read_magic(bounded)
    shape, fortran_order, dtype = parser(
        bounded, max_header_size=_MAX_NPY_HEADER_BYTES
    )
    if parsed_version != version:
        raise ValueError("NPY version changed during bounded parsing")
    consumed = 6 + 2 + length_size + header_length
    return tuple(shape), fortran_order, np.dtype(dtype), consumed


def _preflight_npz_archive(
    array_path: Path,
    descriptors: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
) -> None:
    """核对 ZIP 目录与每个 NPY header，成功前绝不调用 ``np.load``。"""
    if array_path.stat().st_size > _MAX_NPZ_COMPRESSED_BYTES:
        raise ValueError("NPZ compressed file exceeds resource limit")

    expected_members = {f"{name}.npy": value for name, value in descriptors.items()}
    with zipfile.ZipFile(array_path, mode="r") as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if (
            len(members) != len(expected_members)
            or len(names) != len(set(names))
            or set(names) != set(expected_members)
        ):
            raise ValueError("NPZ members do not match manifest")

        total_uncompressed = 0
        for member in members:
            if (
                member.is_dir()
                or member.flag_bits & 0x1
                or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or member.file_size > _MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES
                or member.compress_size > _MAX_NPZ_COMPRESSED_BYTES
            ):
                raise ValueError("NPZ member metadata is unsafe")
            total_uncompressed += member.file_size
            if total_uncompressed > _MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError("NPZ archive exceeds resource limits")

            expected_shape, expected_dtype = expected_members[member.filename]
            with archive.open(member, mode="r") as stream:
                shape, fortran_order, dtype, header_bytes = _read_npy_header_bounded(
                    stream
                )
            if (
                type(fortran_order) is not bool
                or shape != expected_shape
                or dtype != expected_dtype
                or member.file_size != header_bytes + math.prod(shape) * dtype.itemsize
            ):
                raise ValueError("NPY header does not match manifest")


def _manifest_base(sample_id: str, stage: str, fingerprint: str, kind: str) -> JsonObject:
    if type(fingerprint) is not str or not fingerprint:
        raise ValueError("fingerprint must be a nonempty string")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "stage": stage,
        "fingerprint": fingerprint,
        "kind": kind,
    }


class StageCache:
    """以 ``stage/sample_id`` 组织的可恢复缓存。

    JSON cache 返回原 payload；NPZ cache 返回 ``(arrays, metadata)``。指纹不相等只
    表示正常 miss，不隔离；结构、校验和或数值契约异常才视为损坏并 quarantine。
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def quarantine_entry(self, sample_id: str, stage: str) -> None:
        """隔离一个已被上层业务契约判定为不可信的完整缓存条目。"""
        _validated_stage(stage)
        safe_sample_parts(sample_id)
        self._quarantine(sample_id, stage)

    def entry_path(self, sample_id: str, stage: str) -> Path:
        """返回 JSON manifest 路径，并在拼接前验证全部用户可控片段。"""
        validated_stage = _validated_stage(stage)
        return artifact_path(
            self.root / validated_stage,
            sample_id,
            suffix=".json",
        )

    def array_path(self, sample_id: str, stage: str) -> Path:
        """返回 NPZ 数据路径；manifest 和数组同 basename，便于人工审查。"""
        validated_stage = _validated_stage(stage)
        return artifact_path(
            self.root / validated_stage,
            sample_id,
            suffix=".npz",
        )

    def quarantine_path(self, sample_id: str, stage: str) -> Path:
        """返回损坏 manifest 的确定性隔离位置。"""
        return self.entry_path(sample_id, stage).with_suffix(".json.quarantine")

    def _planned_quarantine_targets(
        self, sample_id: str, stage: str
    ) -> tuple[Path, Path]:
        """为 JSON/NPZ 对选择同一个不覆盖旧证据的确定性后缀。"""
        canonical = (
            self.quarantine_path(sample_id, stage),
            self.array_path(sample_id, stage).with_suffix(".npz.quarantine"),
        )
        index = 0
        while True:
            targets = (
                canonical
                if index == 0
                else tuple(Path(f"{path}.{index}") for path in canonical)
            )
            if not any(path.exists() for path in targets):
                return targets  # type: ignore[return-value] -- 长度由 canonical 固定为 2。
            index += 1

    @staticmethod
    def _copy_quarantine_exclusive(source: Path, target: Path) -> None:
        """以 ``xb`` 排他创建目标并 fsync；目标已存在时绝不覆盖。"""
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_stream, target.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())

    def _quarantine(
        self,
        sample_id: str,
        stage: str,
        *,
        move_manifest: bool = True,
        move_array: bool = True,
    ) -> None:
        """成对规划并安全隔离 manifest/NPZ，永不覆盖既有审计证据。

        两个源文件先全部复制和 fsync 成功，再移除活动缓存名；如果任一复制失败，仅
        清理由本次创建的目标并保留全部源文件。这里不会递归删除或触碰缓存外文件。
        """
        sources = (
            self.entry_path(sample_id, stage),
            self.array_path(sample_id, stage),
        )
        existing = (
            move_manifest and sources[0].is_file(),
            move_array and sources[1].is_file(),
        )
        if not any(existing):
            return

        # 排他创建可抵御规划后出现同名证据的竞态；若发生，只选择下一个共同后缀。
        while True:
            targets = self._planned_quarantine_targets(sample_id, stage)
            created: list[Path] = []
            try:
                for source, target, should_move in zip(sources, targets, existing):
                    if not should_move:
                        continue
                    self._copy_quarantine_exclusive(source, target)
                    created.append(target)
            except FileExistsError:
                for target in created:
                    target.unlink(missing_ok=True)
                continue
            except OSError:
                for target in created:
                    target.unlink(missing_ok=True)
                raise
            break

        # 只有整对可用源都已有完整隔离副本后，才移除活动缓存名完成“移动”。
        for source, should_move in zip(sources, existing):
            if should_move:
                source.unlink()

    def _read_manifest(self, sample_id: str, stage: str) -> JsonObject | None:
        path = self.entry_path(sample_id, stage)
        if not path.is_file():
            # manifest 不存在而同 basename NPZ 存在时，它无法被任何可信指纹引用；
            # 必须隔离 orphan，不能永久留在 active cache 路径伪装成可恢复结果。
            if self.array_path(sample_id, stage).is_file():
                self._quarantine(sample_id, stage)
            return None
        try:
            if path.stat().st_size > _MAX_MANIFEST_BYTES:
                raise ValueError("cache manifest is too large")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not _json_safe(value):
                raise ValueError("cache manifest must be a finite JSON object")
            return value
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            self._quarantine(sample_id, stage)
            return None

    @staticmethod
    def _base_is_valid(
        manifest: Mapping[str, Any],
        sample_id: str,
        stage: str,
        *,
        kind: str,
    ) -> bool:
        return (
            manifest.get("schema_version") == CACHE_SCHEMA_VERSION
            and manifest.get("sample_id") == sample_id
            and manifest.get("stage") == stage
            and manifest.get("kind") == kind
            and type(manifest.get("fingerprint")) is str
            and bool(manifest.get("fingerprint"))
        )

    def save_json(
        self,
        sample_id: str,
        stage: str,
        fingerprint: str,
        payload: Mapping[str, Any],
        *,
        cache_identity: str | None = None,
    ) -> Path:
        """原子保存 JSON payload；写入前统一执行敏感值过滤。"""
        _validated_stage(stage)
        safe_sample_parts(sample_id)
        if not isinstance(payload, Mapping):
            raise TypeError("cache JSON payload must be a mapping")
        manifest = _manifest_base(sample_id, stage, fingerprint, "json")
        if cache_identity is not None:
            if type(cache_identity) is not str or not cache_identity:
                raise ValueError("cache_identity must be a nonempty string")
            manifest["cache_identity"] = cache_identity
        manifest["payload"] = redact_sensitive(dict(payload))
        path = self.entry_path(sample_id, stage)
        atomic_write_json(path, manifest)
        # 同 basename 旧 NPZ 已不属于当前 JSON 条目，避免人工误判为有效缓存。
        self.array_path(sample_id, stage).unlink(missing_ok=True)
        return path

    def load_json_record(
        self, sample_id: str, stage: str
    ) -> tuple[JsonObject, str, str | None] | None:
        """严格验证 JSON cache，并返回 payload、manifest 指纹及可选身份。

        ``export_only`` 没有分类器可提供当前身份，因此必须先读取这个经过结构验证的
        record，再由编排层使用其中的身份重算当前上游指纹；本方法本身不猜测业务语义。
        """
        _validated_stage(stage)
        safe_sample_parts(sample_id)
        manifest = self._read_manifest(sample_id, stage)
        if manifest is None:
            return None
        if manifest.get("kind") != "json":
            # 错误 loader 不负责审判另一种合法 cache kind；留给 load_npz 严格校验。
            if self._base_is_valid(manifest, sample_id, stage, kind="npz"):
                return None
            self._quarantine(sample_id, stage)
            return None
        if not self._base_is_valid(manifest, sample_id, stage, kind="json"):
            self._quarantine(sample_id, stage)
            return None
        # JSON kind 不消费 NPZ；同 basename 数组必为 orphan。只隔离 stray NPZ，不能
        # 把仍可命中的 JSON manifest 一起移动，并且此检查必须先于 fingerprint miss。
        if self.array_path(sample_id, stage).is_file():
            self._quarantine(
                sample_id, stage, move_manifest=False, move_array=True
            )
        expected_keys = {
            "schema_version",
            "sample_id",
            "stage",
            "fingerprint",
            "kind",
            "payload",
        }
        if "cache_identity" in manifest:
            if (
                type(manifest["cache_identity"]) is not str
                or not manifest["cache_identity"]
            ):
                self._quarantine(sample_id, stage)
                return None
            expected_keys.add("cache_identity")
        if set(manifest) != expected_keys:
            self._quarantine(sample_id, stage)
            return None
        payload = manifest.get("payload")
        if not isinstance(payload, dict) or not _json_safe(payload):
            self._quarantine(sample_id, stage)
            return None
        return payload, manifest["fingerprint"], manifest.get("cache_identity")

    def load_json(
        self, sample_id: str, stage: str, expected_fingerprint: str
    ) -> JsonObject | None:
        """只在结构和指纹均匹配时返回 payload，否则返回 cache miss。"""
        record = self.load_json_record(sample_id, stage)
        if record is None:
            return None
        payload, fingerprint, _cache_identity = record
        return payload if fingerprint == expected_fingerprint else None

    def save_npz(
        self,
        sample_id: str,
        stage: str,
        fingerprint: str,
        arrays: Mapping[str, NDArray[Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """完整 preflight 后发布压缩 NPZ，再发布含 SHA-256 的 manifest。

        两个文件不能跨文件原子替换，因此 manifest 的 checksum 是恢复边界：若进程在
        两次 replace 之间中断，旧 manifest 与新 NPZ 不匹配，下一次读取会隔离而不会
        将新旧阶段拼接。NPZ writer 始终只写已验证数组。
        """
        _validated_stage(stage)
        safe_sample_parts(sample_id)
        # fingerprint 属于 manifest 基础契约，必须在创建任何 NPZ 临时/活动文件前验证。
        manifest = _manifest_base(sample_id, stage, fingerprint, "npz")
        if not isinstance(arrays, Mapping) or not arrays:
            raise ValueError("NPZ cache requires at least one named array")
        if len(arrays) > _MAX_ARRAY_COUNT:
            raise ValueError("NPZ cache contains too many arrays")
        validated = {
            name: _validated_array(name, value) for name, value in arrays.items()
        }
        descriptor_document = _array_manifest(validated)
        # 写路径与读路径共用完全相同的 shape/dtype/elements/bytes 预算；该检查位于
        # 临时文件和 active artifacts 之前，超限请求不会触碰已存在的有效缓存。
        validated_descriptors = _validated_npz_descriptors(descriptor_document)
        safe_metadata = redact_sensitive(dict(metadata or {}))
        # 在覆盖任何文件前确认 metadata 能被严格 JSON 序列化。
        json.dumps(safe_metadata, allow_nan=False)

        array_path = self.array_path(sample_id, stage)
        manifest_path = self.entry_path(sample_id, stage)

        def write_archive(stream: BinaryIO) -> None:
            np.savez_compressed(stream, **validated)

        prepared_array: Path | None = None
        prepared_manifest: Path | None = None
        try:
            # 先在同目录临时文件中完成压缩和 checksum；活动 NPZ 此时完全未触碰。
            prepared_array = _prepare_file(array_path, write_archive)
            # 自己生成的 ZIP 也必须经过与 load 相同的 central-directory/NPY-header/
            # 正文长度检查；只有确定下一次可安全读取后，才允许进入 replace 阶段。
            _preflight_npz_archive(prepared_array, validated_descriptors)
            manifest.update(
                {
                    "array_sha256": _sha256_path(prepared_array),
                    "arrays": descriptor_document,
                    "metadata": safe_metadata,
                }
            )
            manifest_bytes = _serialized_json(manifest).encode("utf-8")
            prepared_manifest = _prepare_file(
                manifest_path, lambda stream: stream.write(manifest_bytes)
            )

            # 两个公开格式文件均已完整 preflight/fsync 后，才进入活动路径替换阶段。
            os.replace(prepared_array, array_path)
            prepared_array = None
            os.replace(prepared_manifest, manifest_path)
            prepared_manifest = None
        finally:
            if prepared_array is not None:
                prepared_array.unlink(missing_ok=True)
            if prepared_manifest is not None:
                prepared_manifest.unlink(missing_ok=True)
        return manifest_path, array_path

    def load_npz(
        self, sample_id: str, stage: str, expected_fingerprint: str
    ) -> tuple[dict[str, NDArray[Any]], JsonObject] | None:
        """在 ``allow_pickle=False`` 下读取并逐数组核验 manifest 契约。"""
        _validated_stage(stage)
        safe_sample_parts(sample_id)
        manifest = self._read_manifest(sample_id, stage)
        if manifest is None:
            return None
        if manifest.get("kind") != "npz":
            # 合法 JSON cache 由 load_json 管理；错误 loader 不隔离它。
            if self._base_is_valid(manifest, sample_id, stage, kind="json"):
                return None
            self._quarantine(sample_id, stage)
            return None
        if not self._base_is_valid(manifest, sample_id, stage, kind="npz"):
            self._quarantine(sample_id, stage)
            return None
        array_path = self.array_path(sample_id, stage)
        # 配对完整性优先于 fingerprint：缺 NPZ 的 manifest 是 orphan，即使调用方查询
        # 另一个 fingerprint 也必须隔离，不能伪装成普通 stale miss。
        if not array_path.is_file():
            self._quarantine(
                sample_id, stage, move_manifest=True, move_array=False
            )
            return None
        if manifest["fingerprint"] != expected_fingerprint:
            return None

        expected_keys = {
            "schema_version",
            "sample_id",
            "stage",
            "fingerprint",
            "kind",
            "array_sha256",
            "arrays",
            "metadata",
        }
        descriptors = manifest.get("arrays")
        metadata = manifest.get("metadata")
        try:
            if (
                set(manifest) != expected_keys
                or type(manifest.get("array_sha256")) is not str
                or not _SHA256_HEX.fullmatch(manifest["array_sha256"])
                or not isinstance(metadata, dict)
                or not array_path.is_file()
                or _sha256_path(array_path) != manifest["array_sha256"]
            ):
                raise ValueError("NPZ manifest or checksum is invalid")

            # descriptor 与 ZIP/NPY header 的资源预检必须先于 np.load；后者即使只访问
            # ``archive[name]`` 也可能按恶意 shape 申请巨量内存。
            validated_descriptors = _validated_npz_descriptors(descriptors)
            _preflight_npz_archive(array_path, validated_descriptors)

            loaded: dict[str, NDArray[Any]] = {}
            with np.load(array_path, allow_pickle=False) as archive:
                if set(archive.files) != set(validated_descriptors):
                    raise ValueError("NPZ keys do not match manifest")
                for name in archive.files:
                    expected_shape, expected_dtype = validated_descriptors[name]
                    array = _validated_array(name, archive[name])
                    if (
                        expected_shape != array.shape
                        or expected_dtype != array.dtype
                    ):
                        raise ValueError("NPZ array does not match manifest")
                    loaded[name] = array.copy()
            return loaded, dict(metadata)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            EOFError,
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
        ):
            self._quarantine(sample_id, stage)
            return None


def _file_hash(path: Path, *, field_name: str) -> str:
    actual_path = Path(path)
    try:
        return _sha256_path(actual_path)
    except OSError as exc:
        # 只暴露字段名和路径，便于 Task 10 QC 定位；绝不拼接文件正文或异常详情。
        raise ValueError(
            f"cannot hash sample {field_name} at {actual_path}"
        ) from exc


def _class_mapping_path(config: PipelineConfig) -> Path:
    """返回发现阶段实际优先使用的类别文件，缺失时保留可定位的 classes.txt 路径。"""
    dataset_yaml = Path(config.dataset_root) / "dataset.yaml"
    if dataset_yaml.is_file():
        return dataset_yaml
    return Path(config.dataset_root) / "classes.txt"


def _sample_objects(sample: ImageSample) -> list[JsonObject]:
    return [
        {
            "annotation_index": obj.annotation_index,
            "class_id": obj.class_id,
            "class_name": obj.name,
            "bbox_xyxy": list(obj.bbox_xyxy),
        }
        for obj in sample.objects
    ]


def _digest_payload(stage: str, payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        {"schema_version": CACHE_SCHEMA_VERSION, "stage": stage, **payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def sample_fingerprint(
    stage: str,
    sample: ImageSample,
    config: PipelineConfig,
    *,
    classifier_identity: str | None = None,
) -> str:
    """计算只包含本阶段及其上游语义依赖的 SHA-256 指纹。

    失效图由 Task 10 直接复用：

    1. ``depth``/``sky`` 是整图预测，只依赖图片和各自模型；label 或 mask 阈值变化
       不会无故重跑模型。
    2. ``mask`` 纳入图片、标签、类别/框、两模型 ID 与候选阈值。
    3. ``quality`` 纳入 depth/sky 与筛选阈值；``category`` 纳入 mask/depth 和 GPT
       语义设置；``ssi`` 再纳入全部上游，保证任何必要阶段变更都阻止旧结果复用。
    """
    validated_stage = _validated_stage(stage)
    if not isinstance(sample, ImageSample):
        raise TypeError("sample must be an ImageSample")
    if not isinstance(config, PipelineConfig):
        raise TypeError("config must be a PipelineConfig")
    if classifier_identity is not None and (
        type(classifier_identity) is not str or not classifier_identity
    ):
        raise ValueError("classifier_identity must be a nonempty string")

    image_hash = _file_hash(sample.image_path, field_name="image")
    if validated_stage == "depth":
        return _digest_payload(
            "depth", {"image_sha256": image_hash, "model_id": config.depth_model}
        )
    if validated_stage == "sky":
        return _digest_payload(
            "sky", {"image_sha256": image_hash, "model_id": config.sky_model}
        )

    if validated_stage == "mask":
        return _digest_payload(
            "mask",
            {
                "image_sha256": image_hash,
                "label_sha256": _file_hash(sample.label_path, field_name="label"),
                "class_mapping_sha256": _file_hash(
                    _class_mapping_path(config), field_name="class mapping file"
                ),
                "sample_dimensions": [sample.width, sample.height],
                "class_mapping_and_objects": _sample_objects(sample),
                "model_ids": [config.grounding_dino_model, config.sam2_model],
                "grounding_box_threshold": config.grounding_box_threshold,
                "grounding_text_threshold": config.grounding_text_threshold,
                "mask_bbox_iou_threshold": config.mask_bbox_iou_threshold,
                "allow_bbox_prompt_fallback": config.allow_bbox_prompt_fallback,
            },
        )
    if validated_stage == "quality":
        return _digest_payload(
            "quality",
            {
                "depth_fingerprint": sample_fingerprint("depth", sample, config),
                "sky_fingerprint": sample_fingerprint("sky", sample, config),
                "near_depth_threshold": config.near_depth_threshold,
                "max_small_near_area_ratio": config.max_small_near_area_ratio,
                "min_near_concentration": config.min_near_concentration,
                "near_artifact_policy_version": "near-artifact-structural-v1",
                "min_depth_rgb_edge_correlation": config.min_depth_rgb_edge_correlation,
                "min_sky_area_ratio": config.min_sky_area_ratio,
                "filter_mode": config.filter_mode,
            },
        )
    if validated_stage == "category":
        return _digest_payload(
            "category",
            {
                "mask_fingerprint": sample_fingerprint("mask", sample, config),
                "depth_fingerprint": sample_fingerprint("depth", sample, config),
                "model_id": config.minimax_model,
                "image_detail": config.minimax_image_detail,
                "category_policy_version": CATEGORY_POLICY_VERSION,
                "non_risk_class_names": list(config.non_risk_class_names),
                "classifier_identity": (
                    classifier_identity
                    if classifier_identity is not None
                    else "unbound-category-v1"
                ),
            },
        )
    return _digest_payload(
        "ssi",
        {
            "sample_dimensions": [sample.width, sample.height],
            "mask_fingerprint": sample_fingerprint("mask", sample, config),
            "depth_fingerprint": sample_fingerprint("depth", sample, config),
            "quality_fingerprint": sample_fingerprint("quality", sample, config),
            "category_fingerprint": sample_fingerprint(
                "category",
                sample,
                config,
                classifier_identity=classifier_identity,
            ),
            "ssi_contract": "normalized-ssi-v1",
        },
    )
