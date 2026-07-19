from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .rule_matcher import MatchPair


_PAIR_REASON = "same_real_world_obstacle"
_PAIR_CONFIDENCE = "medium"
MATCHER_VERSION = "v2"
MATCHER_SCHEMA_VERSION = "p0-07-name-only-v2"
LOOSE_MATCHER_SCHEMA_VERSION = "p0-07-name-only-controlled-loose-v1"
MATCHER_PROFILES = ("strict", "loose")


@dataclass(frozen=True, repr=False)
class LLMConfig:
    base_url: str
    api_key: str
    model: str

    def __post_init__(self) -> None:
        for field_name in ("base_url", "api_key", "model"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

    def __repr__(self) -> str:
        return (
            "LLMConfig("
            f"base_url={self.base_url!r}, api_key='[REDACTED]', "
            f"model={self.model!r})"
        )


@dataclass(frozen=True)
class DeepSeekDecision:
    pairs: tuple[MatchPair, ...]
    status: str
    model: str
    cache_key: str
    raw_response: str | None
    error: str | None
    from_cache: bool


def load_llm_config(path: Path) -> LLMConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unable to load valid LLM config JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM config must be a JSON object")
    missing = [
        field
        for field in ("base_url", "api_key", "model")
        if field not in payload
    ]
    if missing:
        raise ValueError(f"LLM config missing required fields: {missing}")
    try:
        return LLMConfig(
            base_url=payload["base_url"],
            api_key=payload["api_key"],
            model=payload["model"],
        )
    except ValueError as exc:
        raise ValueError("LLM config fields must be non-empty strings") from exc


def _validated_names(names: Sequence[str], label: str) -> list[str]:
    if isinstance(names, (str, bytes)) or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError(f"{label} must contain only non-empty strings")
    return list(names)


def _validated_profile(profile: str) -> str:
    if profile not in MATCHER_PROFILES:
        raise ValueError(f"profile must be one of {MATCHER_PROFILES}")
    return profile


def _schema_version(profile: str) -> str:
    return (
        MATCHER_SCHEMA_VERSION
        if _validated_profile(profile) == "strict"
        else LOOSE_MATCHER_SCHEMA_VERSION
    )


def build_semantic_match_prompt(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    *,
    profile: str = "strict",
) -> str:
    gt = _validated_names(gt_names, "gt_names")
    pred = _validated_names(pred_names, "pred_names")
    profile = _validated_profile(profile)
    if profile == "strict":
        instructions = [
            "Match two indexed names only when they denote the same real-world obstacle.",
            "Use names only. Enforce a strict one-to-one mapping.",
            "Do not infer beyond the supplied names.",
        ]
    else:
        instructions = [
            "Match indexed names using a controlled loose criterion for the same real-world obstacle.",
            "Use names only. Enforce a one-to-one mapping.",
            "Accept a plausible coarse semantic correspondence when wording differs,",
            "including close synonyms, partial descriptions, or one name being broader or narrower.",
            "Prefer a match when the names plausibly describe the same obstacle and no stronger competing match exists.",
            "The same broad hazard type is not enough: do not match unrelated objects unless the names overlap semantically.",
        ]
    lines = [
        *instructions,
        'Return JSON only with exactly this shape: {"pairs":[{"gt_index":0,"pred_index":0}]}',
        "Use an empty pairs array when no match is justified.",
        "Ground truth names:",
        *(f"GT[{index}]: {name}" for index, name in enumerate(gt)),
        "Predicted names:",
        *(f"PRED[{index}]: {name}" for index, name in enumerate(pred)),
    ]
    return "\n".join(lines)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_pairs(
    raw_text: str, gt_count: int, pred_count: int
) -> list[MatchPair]:
    if (
        isinstance(gt_count, bool)
        or not isinstance(gt_count, int)
        or gt_count < 0
        or isinstance(pred_count, bool)
        or not isinstance(pred_count, int)
        or pred_count < 0
    ):
        raise ValueError("counts must be non-negative integers")
    if not isinstance(raw_text, str):
        raise ValueError("response must be text")
    try:
        payload = json.loads(raw_text, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("response must be strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"pairs"}:
        raise ValueError("response must be an object containing exactly pairs")
    raw_pairs = payload["pairs"]
    if not isinstance(raw_pairs, list):
        raise ValueError("pairs must be an array")

    seen_gt: set[int] = set()
    seen_pred: set[int] = set()
    result: list[MatchPair] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict) or set(raw_pair) != {
            "gt_index",
            "pred_index",
        }:
            raise ValueError("each pair must contain exact index keys")
        gt_index = raw_pair["gt_index"]
        pred_index = raw_pair["pred_index"]
        if (
            isinstance(gt_index, bool)
            or not isinstance(gt_index, int)
            or isinstance(pred_index, bool)
            or not isinstance(pred_index, int)
        ):
            raise ValueError("pair indices must be integers")
        if not 0 <= gt_index < gt_count or not 0 <= pred_index < pred_count:
            raise ValueError("pair index out of range")
        if gt_index in seen_gt or pred_index in seen_pred:
            raise ValueError("pairs must be one-to-one without duplicates")
        seen_gt.add(gt_index)
        seen_pred.add(pred_index)
        result.append(
            MatchPair(
                gt_index=gt_index,
                pred_index=pred_index,
                source="deepseek",
                reason=_PAIR_REASON,
                confidence=_PAIR_CONFIDENCE,
            )
        )
    return sorted(result, key=lambda pair: (pair.gt_index, pair.pred_index))


def build_cache_key(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    *,
    profile: str = "strict",
) -> str:
    profile = _validated_profile(profile)
    request = {
        "matcher_version": MATCHER_VERSION,
        "matcher_schema_version": _schema_version(profile),
        "gt_names": _validated_names(gt_names, "gt_names"),
        "pred_names": _validated_names(pred_names, "pred_names"),
    }
    canonical = json.dumps(
        request, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _redact(value: str | None, api_key: str) -> str | None:
    if value is None:
        return None
    return value.replace(api_key, "[REDACTED]")


def _cache_entry(
    *,
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    decision: DeepSeekDecision,
    api_key: str,
    profile: str,
) -> dict[str, Any]:
    return {
        "cache_key": decision.cache_key,
        "matcher_version": MATCHER_VERSION,
        "matcher_schema_version": _schema_version(profile),
        "matcher_profile": profile,
        "request": {
            "gt_names": [_redact(name, api_key) for name in gt_names],
            "pred_names": [_redact(name, api_key) for name in pred_names],
        },
        "raw_response": _redact(decision.raw_response, api_key),
        "parsed_pairs": [
            {"gt_index": pair.gt_index, "pred_index": pair.pred_index}
            for pair in decision.pairs
        ],
        "model": decision.model,
        "status": decision.status,
        "error": _redact(decision.error, api_key),
    }


def _append_cache(
    cache_path: Path,
    *,
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    decision: DeepSeekDecision,
    api_key: str,
    profile: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    entry = _cache_entry(
        gt_names=gt_names,
        pred_names=pred_names,
        decision=decision,
        api_key=api_key,
        profile=profile,
    )
    with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def _cached_decision(
    cache_path: Path,
    *,
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    config: LLMConfig,
    cache_key: str,
    profile: str,
) -> DeepSeekDecision | None:
    if not cache_path.exists():
        return None
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line, object_pairs_hook=_strict_object)
            if (
                not isinstance(entry, dict)
                or entry.get("cache_key") != cache_key
                or entry.get("matcher_version") != MATCHER_VERSION
                or entry.get("matcher_schema_version") != _schema_version(profile)
                or (
                    entry.get("matcher_profile") not in (None, "strict")
                    if profile == "strict"
                    else entry.get("matcher_profile") != profile
                )
                or entry.get("model") != config.model
                or entry.get("status") != "matched"
                or entry.get("request")
                != {
                    "gt_names": list(gt_names),
                    "pred_names": list(pred_names),
                }
            ):
                continue
            raw_pairs = entry.get("parsed_pairs")
            pairs = parse_pairs(
                json.dumps({"pairs": raw_pairs}, separators=(",", ":")),
                len(gt_names),
                len(pred_names),
            )
            raw_response = entry.get("raw_response")
            if not isinstance(raw_response, str):
                continue
            return DeepSeekDecision(
                pairs=tuple(pairs),
                status="matched",
                model=config.model,
                cache_key=cache_key,
                raw_response=raw_response,
                error=None,
                from_cache=True,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("API response does not contain message text") from exc
    if not isinstance(content, str):
        raise ValueError("API response message content must be text")
    return content


def match_with_deepseek(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    config: LLMConfig,
    cache_path: Path,
    client: Any = None,
    *,
    max_attempts: int = 3,
    profile: str = "strict",
) -> DeepSeekDecision:
    gt = _validated_names(gt_names, "gt_names")
    pred = _validated_names(pred_names, "pred_names")
    profile = _validated_profile(profile)
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise ValueError("max_attempts must be a positive integer")
    cache_key = build_cache_key(gt, pred, profile=profile)
    cached = _cached_decision(
        cache_path,
        gt_names=gt,
        pred_names=pred,
        config=config,
        cache_key=cache_key,
        profile=profile,
    )
    if cached is not None:
        return cached

    prompt = build_semantic_match_prompt(gt, pred, profile=profile)
    last_error: str | None = None
    last_raw: str | None = None
    try:
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        for _ in range(max_attempts):
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                last_raw = _response_text(response)
                pairs = parse_pairs(last_raw, len(gt), len(pred))
                decision = DeepSeekDecision(
                    pairs=tuple(pairs),
                    status="matched",
                    model=config.model,
                    cache_key=cache_key,
                    raw_response=_redact(last_raw, config.api_key),
                    error=None,
                    from_cache=False,
                )
                _append_cache(
                    cache_path,
                    gt_names=gt,
                    pred_names=pred,
                    decision=decision,
                    api_key=config.api_key,
                    profile=profile,
                )
                return decision
            except Exception as exc:
                last_error = _redact(
                    f"{type(exc).__name__}: {exc}", config.api_key
                )
    except Exception as exc:
        last_error = _redact(f"{type(exc).__name__}: {exc}", config.api_key)

    decision = DeepSeekDecision(
        pairs=(),
        status="unresolved",
        model=config.model,
        cache_key=cache_key,
        raw_response=_redact(last_raw, config.api_key),
        error=last_error,
        from_cache=False,
    )
    _append_cache(
        cache_path,
        gt_names=gt,
        pred_names=pred,
        decision=decision,
        api_key=config.api_key,
        profile=profile,
    )
    return decision
