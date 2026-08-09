"""MiniMax 运行时凭据加载边界；校验连接配置且绝不序列化 API key。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


_MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimaxi.com/anthropic"
_MINIMAX_OPENAI_BASE_URL = "https://api.minimaxi.com/v1"


class MiniMaxConfigurationError(ValueError):
    """用户提供的 MiniMax 配置无法被安全使用。"""


@dataclass(frozen=True)
class MiniMaxCredentials:
    """已验证的 MiniMax 连接信息；调用方不得序列化 api_key。"""

    api_key: str
    base_url: str
    model: str


def load_minimax_credentials(
    path: Path,
    *,
    expected_model: str,
) -> MiniMaxCredentials:
    """Load the user-owned config and normalize it for MiniMax's OpenAI SDK API."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MiniMaxConfigurationError(
            "MiniMax config could not be read as a JSON object"
        ) from None

    if not isinstance(document, dict):
        raise MiniMaxConfigurationError("MiniMax config must be a JSON object")

    api_key = document.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise MiniMaxConfigurationError("MiniMax config requires a non-empty api_key")

    model = document.get("model")
    if not isinstance(model, str) or model != expected_model:
        raise MiniMaxConfigurationError(
            "MiniMax config model does not match the configured model"
        )

    base_url = document.get("base_url")
    if not isinstance(base_url, str):
        raise MiniMaxConfigurationError("MiniMax config requires a base_url")
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url == _MINIMAX_ANTHROPIC_BASE_URL:
        normalized_base_url = _MINIMAX_OPENAI_BASE_URL
    if normalized_base_url != _MINIMAX_OPENAI_BASE_URL:
        raise MiniMaxConfigurationError("MiniMax config has an unsupported base_url")

    return MiniMaxCredentials(
        api_key=api_key,
        base_url=normalized_base_url,
        model=model,
    )
