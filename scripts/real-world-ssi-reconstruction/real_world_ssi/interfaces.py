"""后续模型阶段依赖的四个最小服务协议。"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from real_world_ssi.schemas import (
    CategoryAssignment,
    DepthResult,
    ObstacleMask,
    YoloObject,
)


class MaskService(Protocol):
    """从图像和 YOLO 障碍物提取一一对应掩码。"""

    def extract(
        self, image: Image.Image, objects: tuple[YoloObject, ...]
    ) -> tuple[ObstacleMask, ...]:
        raise NotImplementedError


class DepthService(Protocol):
    """预测原始与 8 位 SSI 深度。"""

    def predict(self, image: Image.Image) -> DepthResult:
        raise NotImplementedError


class SkyService(Protocol):
    """预测质量控制所需的布尔天空掩码。"""

    def predict_sky_mask(self, image: Image.Image) -> NDArray[np.bool_]:
        raise NotImplementedError


class HazardClassifier(Protocol):
    """最初公开的最小分类契约；旧第三方只需实现此方法。"""

    def classify(
        self,
        sample_id: str,
        image: Image.Image,
        overlay_path: Path,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> tuple[CategoryAssignment, ...]:
        raise NotImplementedError


@runtime_checkable
class CacheIdentityProvider(Protocol):
    """可选扩展：提供当前分类语义输入的稳定缓存身份。"""

    def cache_identity(
        self,
        sample_id: str,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> str:
        """返回与分类语义输入绑定的稳定身份；编排层据此决定能否复用类别缓存。"""
        raise NotImplementedError


@runtime_checkable
class AtomicCacheAwareHazardClassifier(
    HazardClassifier, CacheIdentityProvider, Protocol
):
    """可选扩展：从同一快照原子返回分类结果与身份。"""

    def classify_with_cache_identity(
        self,
        sample_id: str,
        image: Image.Image,
        overlay_path: Path,
        objects: tuple[YoloObject, ...],
        masks: tuple[ObstacleMask, ...],
    ) -> tuple[tuple[CategoryAssignment, ...], str]:
        """从同一次语义输入快照返回分类结果及其缓存身份。"""
        raise NotImplementedError
