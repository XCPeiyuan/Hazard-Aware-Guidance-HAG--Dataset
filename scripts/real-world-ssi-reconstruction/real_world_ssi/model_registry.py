"""Transformers 模型的批次级懒加载注册表。

导入本模块不会导入 Transformers，也不会解析或下载 checkpoint。每个属性只在首次访问时
调用一次 ``from_pretrained``；处理器留在 CPU，模型则统一迁移到流水线已选择的设备并进入
推理模式。这样离线模式的行为由同一个 ``local_files_only`` 开关明确约束。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

from .schemas import PipelineConfig


T = TypeVar("T")


class ModelRegistry:
    """按需创建并复用 Grounding DINO、SAM 2 与深度模型组件。"""

    def __init__(self, config: PipelineConfig, device: str):
        self.config = config
        self.device = device
        self._models: dict[str, object] = {}

    def _once(self, key: str, loader: Callable[[], T]) -> T:
        """缓存一次成功的加载；加载抛错时不留下半初始化对象。"""
        if key not in self._models:
            self._models[key] = loader()
        return cast(T, self._models[key])

    def _pretrained_kwargs(self) -> dict[str, object]:
        # ``AUTO_DOWNLOAD_MODELS=False`` 必须落到 Transformers 的强制本地查询，
        # 不能仅依赖环境变量或调用方恰好断网。
        return {
            "cache_dir": self.config.model_cache_dir,
            "local_files_only": not self.config.auto_download_models,
        }

    @property
    def grounding_processor(self) -> Any:
        def load() -> object:
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(
                self.config.grounding_dino_model, **self._pretrained_kwargs()
            )

        return self._once("grounding_processor", load)

    @property
    def grounding_model(self) -> Any:
        def load() -> object:
            from transformers import AutoModelForZeroShotObjectDetection

            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.grounding_dino_model, **self._pretrained_kwargs()
            )
            return model.to(self.device).eval()

        return self._once("grounding_model", load)

    @property
    def sam2_processor(self) -> Any:
        def load() -> object:
            from transformers import Sam2Processor

            return Sam2Processor.from_pretrained(
                self.config.sam2_model, **self._pretrained_kwargs()
            )

        return self._once("sam2_processor", load)

    @property
    def sam2_model(self) -> Any:
        def load() -> object:
            from transformers import Sam2Model

            model = Sam2Model.from_pretrained(
                self.config.sam2_model, **self._pretrained_kwargs()
            )
            return model.to(self.device).eval()

        return self._once("sam2_model", load)

    @property
    def depth_processor(self) -> Any:
        """加载 Depth Anything V2 的图像处理器并在本批次内复用。"""
        def load() -> object:
            # Transformers 5.10.2 的 AutoProcessor 会从 checkpoint 的
            # preprocessor_config 回退到 AutoImageProcessor；使用 brief 指定的
            # AutoProcessor 可保持 registry 对各视觉阶段一致的公共入口。
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(
                self.config.depth_model, **self._pretrained_kwargs()
            )

        return self._once("depth_processor", load)

    @property
    def depth_model(self) -> Any:
        """加载 Depth Anything V2 推理模型，迁移设备并进入 eval 模式。"""
        def load() -> object:
            from transformers import AutoModelForDepthEstimation

            model = AutoModelForDepthEstimation.from_pretrained(
                self.config.depth_model, **self._pretrained_kwargs()
            )
            return model.to(self.device).eval()

        return self._once("depth_model", load)

    @property
    def sky_processor(self) -> Any:
        """按需加载 SegFormer 官方图像处理入口并在本批次复用。"""
        def load() -> object:
            # Transformers 5.10.2 的 SegFormer forward 文档和处理器源码均使用
            # AutoImageProcessor；这里不猜测不存在的 AutoProcessor 专用映射。
            from transformers import AutoImageProcessor

            return AutoImageProcessor.from_pretrained(
                self.config.sky_model, **self._pretrained_kwargs()
            )

        return self._once("sky_processor", load)

    @property
    def sky_model(self) -> Any:
        """按需加载 SegFormer 语义分割模型，迁移设备并进入 eval 模式。"""
        def load() -> object:
            from transformers import SegformerForSemanticSegmentation

            model = SegformerForSemanticSegmentation.from_pretrained(
                self.config.sky_model, **self._pretrained_kwargs()
            )
            return model.to(self.device).eval()

        return self._once("sky_model", load)
