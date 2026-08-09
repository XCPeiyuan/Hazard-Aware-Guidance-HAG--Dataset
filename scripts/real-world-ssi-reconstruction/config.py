"""Real-world SSI 重建流水线的集中配置。

论文复现实验的所有公开默认值都集中在此处；后续阶段只能通过
``build_config`` 获取冻结配置，避免各模型服务各自复制并漂移参数。
"""

from pathlib import Path

from real_world_ssi.category_policy import DEFAULT_NON_RISK_CLASS_NAMES
from real_world_ssi.schemas import PipelineConfig


# 所有路径默认值相对本功能目录，而不是相对调用者当前工作目录。
FEATURE_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = FEATURE_ROOT / "input"
OUTPUT_ROOT = FEATURE_ROOT / "output"
DATASET_SPLIT = "auto"
DEVICE = "auto"
RUN_STAGE = "full"
RESUME = True
OVERWRITE = False
DRY_RUN = False
FAIL_BATCH_IF_ANY_ERROR = True
AUTO_DOWNLOAD_MODELS = True
MODEL_CACHE_DIR = FEATURE_ROOT / "model_cache"

# 模型标识和 API 默认值属于论文重建配置；Task 1 只记录，不加载模型。
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL = "facebook/sam2.1-hiera-tiny"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SKY_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
MINIMAX_CONFIG_PATH = FEATURE_ROOT / "config.json"
MINIMAX_MODEL = "MiniMax-M3"
MINIMAX_IMAGE_DETAIL = "high"
MINIMAX_MAX_RETRIES = 3
MINIMAX_TIMEOUT_SECONDS = 120

# 检测、掩码和 SSI 质量控制阈值必须与任务 brief 的论文默认值一致。
GROUNDING_BOX_THRESHOLD = 0.25
GROUNDING_TEXT_THRESHOLD = 0.25
MASK_BBOX_IOU_THRESHOLD = 0.50
ALLOW_BBOX_PROMPT_FALLBACK = False
NEAR_DEPTH_THRESHOLD = 250
MAX_SMALL_NEAR_AREA_RATIO = 0.01
MIN_NEAR_CONCENTRATION = 0.80
MIN_DEPTH_RGB_EDGE_CORRELATION = 0.20
MIN_SKY_AREA_RATIO = 0.005
MAX_OBJECT_COUNT = 15
FILTER_MODE = "reject"
NON_RISK_CLASS_NAMES = DEFAULT_NON_RISK_CLASS_NAMES

# 中间产物默认全部保留，以支持论文结果复核和失败样本审计。
SAVE_RAW_DEPTH = True
SAVE_MASKS = True
SAVE_OVERLAYS = True
SAVE_API_RESPONSES = True


def build_config() -> PipelineConfig:
    """逐项映射公开常量并返回已验证的不可变配置。

    返回值冻结本批次的路径、模式、模型 ID 和阈值；函数只做对象构造与配置校验，
    不创建目录、不读取 API key，也不加载或下载任何模型。
    """
    return PipelineConfig(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        dataset_split=DATASET_SPLIT,
        device=DEVICE,
        run_stage=RUN_STAGE,
        resume=RESUME,
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
        fail_batch_if_any_error=FAIL_BATCH_IF_ANY_ERROR,
        auto_download_models=AUTO_DOWNLOAD_MODELS,
        model_cache_dir=MODEL_CACHE_DIR,
        grounding_dino_model=GROUNDING_DINO_MODEL,
        sam2_model=SAM2_MODEL,
        depth_model=DEPTH_MODEL,
        sky_model=SKY_MODEL,
        minimax_config_path=MINIMAX_CONFIG_PATH,
        minimax_model=MINIMAX_MODEL,
        minimax_image_detail=MINIMAX_IMAGE_DETAIL,
        minimax_max_retries=MINIMAX_MAX_RETRIES,
        minimax_timeout_seconds=MINIMAX_TIMEOUT_SECONDS,
        grounding_box_threshold=GROUNDING_BOX_THRESHOLD,
        grounding_text_threshold=GROUNDING_TEXT_THRESHOLD,
        mask_bbox_iou_threshold=MASK_BBOX_IOU_THRESHOLD,
        allow_bbox_prompt_fallback=ALLOW_BBOX_PROMPT_FALLBACK,
        near_depth_threshold=NEAR_DEPTH_THRESHOLD,
        max_small_near_area_ratio=MAX_SMALL_NEAR_AREA_RATIO,
        min_near_concentration=MIN_NEAR_CONCENTRATION,
        min_depth_rgb_edge_correlation=MIN_DEPTH_RGB_EDGE_CORRELATION,
        min_sky_area_ratio=MIN_SKY_AREA_RATIO,
        max_object_count=MAX_OBJECT_COUNT,
        filter_mode=FILTER_MODE,
        save_raw_depth=SAVE_RAW_DEPTH,
        save_masks=SAVE_MASKS,
        save_overlays=SAVE_OVERLAYS,
        save_api_responses=SAVE_API_RESPONSES,
        non_risk_class_names=NON_RISK_CLASS_NAMES,
    ).validate()
