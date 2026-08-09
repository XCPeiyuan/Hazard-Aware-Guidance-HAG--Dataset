"""类别文本 Grounding DINO + SAM 2 的像素级障碍物 mask 适配器。"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from .geometry import mask_centroid
from .mask_postprocess import clean_single_component, select_mask_candidate
from .model_registry import ModelRegistry
from .schemas import MaskCandidate, ObstacleMask, PipelineConfig, YoloObject


class MaskExtractionError(RuntimeError):
    """某个 YOLO 标注无法得到满足重建约束的非空 mask。"""


def _to_cpu_numpy(value: Any) -> NDArray[Any]:
    """在候选排序前切断设备和框架差异，只保留 CPU NumPy 数值。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class GroundedSam2MaskService:
    """逐个 YOLO 物体执行文本定位、SAM 2 分割和确定性候选清理。"""

    def __init__(self, registry: ModelRegistry, config: PipelineConfig):
        self.registry = registry
        self.config = config

    def _grounding_candidates(
        self, image: Image.Image, object_name: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Grounding DINO 的短类别提示以句点结束；每个物体独立推理，避免多个类别
        # 的 token 匹配结果在候选归属上产生新的、未在论文中定义的语义。
        prompt = f"{object_name.rstrip('.')}."
        processor = self.registry.grounding_processor
        inputs = processor(images=image, text=[prompt], return_tensors="pt")
        inputs = inputs.to(self.registry.device)

        # CPU 与 CUDA 共用 inference_mode；此处不使用 autocast，因此 CPU 路径不会
        # 意外进入 CUDA 专用上下文或改变 fake/真实后端的数值类型。
        with torch.inference_mode():
            outputs = self.registry.grounding_model(**inputs)

        processed = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=self.config.grounding_box_threshold,
            text_threshold=self.config.grounding_text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = _to_cpu_numpy(processed["boxes"]).astype(np.float64, copy=False)
        scores = _to_cpu_numpy(processed["scores"]).astype(np.float64, copy=False)
        if boxes.size == 0:
            boxes = np.empty((0, 4), dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (boxes.shape[0],):
            raise MaskExtractionError("Grounding DINO returned incompatible boxes/scores")
        if not np.isfinite(boxes).all():
            raise MaskExtractionError("Grounding DINO boxes must be finite")
        if not np.isfinite(scores).all():
            raise MaskExtractionError("Grounding DINO scores must be finite")
        return boxes, scores

    def _sam_candidates(
        self,
        image: Image.Image,
        boxes: NDArray[np.float64],
        grounding_scores: NDArray[np.float64],
        *,
        prefix: str,
    ) -> tuple[MaskCandidate, ...]:
        if boxes.shape[0] == 0:
            return ()

        processor = self.registry.sam2_processor
        # 本机 Transformers 5.10.2 的 Sam2Processor 契约是
        # [batch][prompt_box][xyxy]；本服务每次只处理一张图片。
        inputs = processor(
            images=image,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        )
        inputs = inputs.to(self.registry.device)
        with torch.inference_mode():
            outputs = self.registry.sam2_model(**inputs)

        processed_masks = processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"]
        )
        # post_process_masks 按 batch 返回；单图内部保持
        # [prompt_box][multimask][height][width]，与 iou_scores 一一对应。
        masks = _to_cpu_numpy(processed_masks[0]).astype(np.bool_, copy=False)
        sam_scores = _to_cpu_numpy(outputs.iou_scores)[0].astype(
            np.float64, copy=False
        )
        expected_mask_shape = (*sam_scores.shape, image.height, image.width)
        if masks.shape != expected_mask_shape:
            raise MaskExtractionError(
                "SAM 2 returned incompatible mask and IoU score shapes"
            )
        if sam_scores.shape[0] != grounding_scores.shape[0]:
            raise MaskExtractionError(
                "SAM 2 prompt count does not match Grounding DINO candidates"
            )
        if not np.isfinite(sam_scores).all():
            raise MaskExtractionError("SAM 2 IoU scores must be finite")

        candidates: list[MaskCandidate] = []
        for box_index in range(sam_scores.shape[0]):
            for mask_index in range(sam_scores.shape[1]):
                candidates.append(
                    MaskCandidate(
                        candidate_id=(
                            f"{prefix}:{box_index:06d}:{mask_index:06d}"
                        ),
                        mask=masks[box_index, mask_index],
                        sam_score=float(sam_scores[box_index, mask_index]),
                        grounding_score=float(grounding_scores[box_index]),
                    )
                )
        return tuple(candidates)

    def _select_and_clean(
        self,
        candidates: tuple[MaskCandidate, ...],
        yolo_object: YoloObject,
        source: Literal["grounded_sam2", "bbox_fallback"],
    ) -> ObstacleMask:
        try:
            selected = select_mask_candidate(
                candidates,
                yolo_object.bbox_xyxy,
                self.config.mask_bbox_iou_threshold,
            )
        except ValueError as exc:
            raise MaskExtractionError(
                f"annotation {yolo_object.annotation_index}: no mask candidate meets IoU threshold"
            ) from exc

        cleaned = clean_single_component(selected.mask, yolo_object.bbox_xyxy)
        return ObstacleMask(
            annotation_index=yolo_object.annotation_index,
            mask=cleaned,
            centroid_top_left=mask_centroid(cleaned),
            source=source,
            diagnostics={
                "candidate_id": selected.candidate_id,
                "sam_score": selected.sam_score,
                "grounding_score": selected.grounding_score,
            },
        )

    def _extract_one(self, image: Image.Image, yolo_object: YoloObject) -> ObstacleMask:
        boxes, grounding_scores = self._grounding_candidates(image, yolo_object.name)
        candidates = self._sam_candidates(
            image, boxes, grounding_scores, prefix="grounded"
        )
        try:
            return self._select_and_clean(candidates, yolo_object, "grounded_sam2")
        except MaskExtractionError:
            if not self.config.allow_bbox_prompt_fallback:
                raise

            fallback_box = np.asarray([yolo_object.bbox_xyxy], dtype=np.float64)
            fallback_candidates = self._sam_candidates(
                image,
                fallback_box,
                np.zeros((1,), dtype=np.float64),
                prefix="bbox_fallback",
            )
            try:
                return self._select_and_clean(
                    fallback_candidates, yolo_object, "bbox_fallback"
                )
            except MaskExtractionError as fallback_error:
                raise MaskExtractionError(
                    f"annotation {yolo_object.annotation_index}: grounded and bbox fallback masks failed IoU"
                ) from fallback_error

    def extract(
        self, image: Image.Image, objects: tuple[YoloObject, ...]
    ) -> tuple[ObstacleMask, ...]:
        """按原 YOLO 行序返回一一对应的障碍物 mask；任一失败即停止样本。"""
        return tuple(self._extract_one(image, item) for item in objects)
