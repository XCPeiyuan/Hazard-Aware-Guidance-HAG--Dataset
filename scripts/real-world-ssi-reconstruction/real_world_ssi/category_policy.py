"""Canonical SSI categories and deterministic non-risk class policy."""

from __future__ import annotations
 # 类别策略模块。

DEFAULT_NON_RISK_CLASS_NAMES: tuple[str, ...] = (
    "braille_block",
    "crosswalk",
    "traffic_light",
    "signal_button",
    "signal_red",
    "signal_blue",
    "white_line",
)

MODEL_HAZARD_CATEGORIES: tuple[str, ...] = (
    "Common Obstacle",
    "Pitfall Hazard",
    "Upper-body Hazard",
)

SSI_CATEGORIES: tuple[str, ...] = MODEL_HAZARD_CATEGORIES + ("Other",)

CATEGORY_POLICY_VERSION = "category-policy-other-v2"


def is_non_risk_class(
    name: str, configured_names: tuple[str, ...] = DEFAULT_NON_RISK_CLASS_NAMES
) -> bool:
    """Return whether an exact dataset class name bypasses risk classification."""
    return type(name) is str and name in configured_names
