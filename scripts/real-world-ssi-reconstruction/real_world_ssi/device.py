"""轻量设备选择；除 PyTorch 外不加载任何模型或外部 API SDK。"""

import torch


def select_device(requested: str, cuda_available: bool | None = None) -> str:
    """解析 ``cpu/cuda/auto`` 并返回本批次使用的 PyTorch 设备字符串。

    ``cuda_available`` 只用于测试注入；为 ``None`` 才查询当前机器。显式请求已知不可用
    的 CUDA 会失败，未知 ``requested`` 也会失败；本函数不加载模型、不下载权重。
    """
    if requested in {"cpu", "cuda"}:
        if requested == "cuda" and cuda_available is False:
            raise RuntimeError("DEVICE=cuda but CUDA is unavailable")
        return requested
    if requested != "auto":
        raise ValueError("DEVICE must be auto, cpu, or cuda")

    # 注入 cuda_available 可让测试不依赖执行机器；None 才查询真实 PyTorch 状态。
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    return "cuda" if available else "cpu"
