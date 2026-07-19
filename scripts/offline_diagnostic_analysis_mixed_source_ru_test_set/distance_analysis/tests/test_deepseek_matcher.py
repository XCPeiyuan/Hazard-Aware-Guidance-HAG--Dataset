import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Callable

import pytest

from offline_diagnostic_analysis.deepseek_matcher import (
    MATCHER_VERSION,
    MATCHER_SCHEMA_VERSION,
    DeepSeekDecision,
    LLMConfig,
    build_cache_key,
    build_semantic_match_prompt,
    load_llm_config,
    match_with_deepseek,
    parse_pairs,
)
from offline_diagnostic_analysis.rule_matcher import MatchPair


class _FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = type("Message", (), {"content": outcome})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(outcomes)})()


def _config(api_key: str = "test-secret-key") -> LLMConfig:
    return LLMConfig(
        base_url="https://example.invalid/v1",
        api_key=api_key,
        model="deepseek-v4-flash",
    )


@pytest.fixture
def task_path(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[[str], Path]:
    def make_path(suffix: str) -> Path:
        return tmp_path / suffix

    return make_path


def test_load_llm_config_accepts_required_non_empty_strings(
    task_path: Callable[[str], Path],
) -> None:
    path = task_path("llm_config.json")
    path.write_text(
        json.dumps(
            {
                "base_url": " https://example.invalid/v1 ",
                "api_key": " secret ",
                "model": " deepseek-v4-flash ",
                "unused": "allowed",
            }
        ),
        encoding="utf-8",
    )

    config = load_llm_config(path)

    assert config == _config("secret")
    with pytest.raises(FrozenInstanceError):
        config.model = "other"  # type: ignore[misc]


def test_llm_config_repr_redacts_api_key() -> None:
    config = _config()

    assert config.api_key not in repr(config)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"base_url": "url", "api_key": "key"},
        {"base_url": "url", "api_key": "key", "model": ""},
        {"base_url": "url", "api_key": 7, "model": "model"},
        [],
    ],
)
def test_load_llm_config_rejects_missing_or_invalid_fields_without_key_leak(
    task_path: Callable[[str], Path], payload: object
) -> None:
    path = task_path("llm_config.json")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_llm_config(path)

    assert "secret" not in str(exc_info.value)


def test_prompt_contains_only_indexed_names_and_strict_matching_instructions() -> None:
    prompt = build_semantic_match_prompt(
        ["open manhole", "fallen branch"],
        ["manhole opening", "tree limb"],
    )
    lower = prompt.lower()

    assert "GT[0]: open manhole" in prompt
    assert "GT[1]: fallen branch" in prompt
    assert "PRED[0]: manhole opening" in prompt
    assert "PRED[1]: tree limb" in prompt
    assert "same real-world obstacle" in lower
    assert "one-to-one" in lower
    assert "names only" in lower
    assert "json only" in lower
    for excluded in ("category", "distance", "alert", "ssi"):
        assert excluded not in lower


def test_loose_prompt_is_names_only_and_controlled_not_category_based() -> None:
    prompt = build_semantic_match_prompt(
        ["fallen tree limb"],
        ["branch"],
        profile="loose",
    )
    lower = prompt.lower()

    assert "coarse semantic correspondence" in lower
    assert "broader or narrower" in lower
    assert "same broad hazard type is not enough" in lower
    assert "one-to-one" in lower
    assert "names only" in lower
    for excluded in ("category", "distance", "alert", "ssi"):
        assert excluded not in lower


def test_loose_cache_key_is_distinct_from_strict() -> None:
    strict = build_cache_key(["fallen tree limb"], ["branch"])
    loose = build_cache_key(
        ["fallen tree limb"],
        ["branch"],
        profile="loose",
    )

    assert strict != loose


def test_unknown_matching_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="profile"):
        build_semantic_match_prompt(["a"], ["b"], profile="unknown")


def test_parse_pairs_returns_deterministic_deepseek_match_pairs() -> None:
    result = parse_pairs(
        '{"pairs":[{"gt_index":1,"pred_index":0},{"gt_index":0,"pred_index":1}]}',
        gt_count=2,
        pred_count=2,
    )

    assert result == [
        MatchPair(0, 1, "deepseek", "same_real_world_obstacle", "medium"),
        MatchPair(1, 0, "deepseek", "same_real_world_obstacle", "medium"),
    ]


@pytest.mark.parametrize(
    "raw_text",
    [
        '{"pairs":[{"gt_index":1,"pred_index":0}]}',
        '{"pairs":[{"gt_index":0,"pred_index":1}]}',
        '{"pairs":[{"gt_index":0,"pred_index":0},{"gt_index":0,"pred_index":1}]}',
        '{"pairs":[{"gt_index":0,"pred_index":0},{"gt_index":1,"pred_index":0}]}',
        '{"pairs":[{"gt_index":true,"pred_index":0}]}',
        '{"pairs":[{"gt_index":0,"pred_index":false}]}',
        '{"pairs":[{"gt_index":0,"pred_index":0,"reason":"guess"}]}',
        '{"pairs":[],"note":"extra"}',
        'Here is JSON: {"pairs":[]}',
        '{"pairs":[]}\nThanks',
        '{"pairs":[],"pairs":[]}',
        '{"pairs":[{"gt_index":0,"gt_index":0,"pred_index":0}]}',
        "[]",
        '{"pairs":"none"}',
    ],
)
def test_parse_pairs_rejects_invalid_or_non_strict_output(raw_text: str) -> None:
    with pytest.raises(ValueError):
        parse_pairs(raw_text, gt_count=1, pred_count=1)


def test_cache_key_is_stable_and_order_sensitive() -> None:
    first = build_cache_key(["open manhole", "branch"], ["opening", "limb"])
    same = build_cache_key(["open manhole", "branch"], ["opening", "limb"])
    reordered_gt = build_cache_key(["branch", "open manhole"], ["opening", "limb"])
    reordered_pred = build_cache_key(["open manhole", "branch"], ["limb", "opening"])

    assert first == same
    assert first != reordered_gt
    assert first != reordered_pred


def test_cache_key_and_entry_are_versioned() -> None:
    current = build_cache_key(["cone"], ["cone"])
    assert isinstance(MATCHER_VERSION, str)
    assert MATCHER_VERSION
    assert isinstance(MATCHER_SCHEMA_VERSION, str)
    assert MATCHER_SCHEMA_VERSION

    legacy_request = json.dumps(
        {"gt_names": ["cone"], "pred_names": ["cone"]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    import hashlib

    assert current != hashlib.sha256(legacy_request.encode("utf-8")).hexdigest()

    schema_only = json.dumps(
        {
            "matcher_schema_version": MATCHER_SCHEMA_VERSION,
            "gt_names": ["cone"],
            "pred_names": ["cone"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert current != hashlib.sha256(schema_only.encode("utf-8")).hexdigest()


def test_valid_cache_hit_avoids_api_call_and_preserves_audit_fields(
    task_path: Callable[[str], Path],
) -> None:
    cache_path = task_path("match_cache.jsonl")
    first_client = _FakeClient(
        ['{"pairs":[{"gt_index":0,"pred_index":0}]}']
    )

    first = match_with_deepseek(
        ["open manhole"],
        ["manhole opening"],
        _config(),
        cache_path,
        client=first_client,
    )
    unused_client = _FakeClient([RuntimeError("must not be called")])
    cached = match_with_deepseek(
        ["open manhole"],
        ["manhole opening"],
        _config(),
        cache_path,
        client=unused_client,
    )

    assert first.status == "matched"
    assert first.from_cache is False
    assert cached.pairs == first.pairs
    assert cached.from_cache is True
    assert unused_client.chat.completions.calls == []
    entry = json.loads(cache_path.read_text(encoding="utf-8").strip())
    assert entry["request"] == {
        "gt_names": ["open manhole"],
        "pred_names": ["manhole opening"],
    }
    assert entry["raw_response"] == (
        '{"pairs":[{"gt_index":0,"pred_index":0}]}'
    )
    assert entry["parsed_pairs"] == [{"gt_index": 0, "pred_index": 0}]
    assert entry["model"] == "deepseek-v4-flash"
    assert entry["matcher_version"] == MATCHER_VERSION
    assert entry["matcher_schema_version"] == MATCHER_SCHEMA_VERSION
    assert entry["status"] == "matched"
    assert "api_key" not in entry
    assert "test-secret-key" not in cache_path.read_text(encoding="utf-8")


def test_match_uses_temperature_zero_and_retries_invalid_output(
    task_path: Callable[[str], Path],
) -> None:
    client = _FakeClient(
        [
            "not json",
            '{"pairs":[{"gt_index":0,"pred_index":0}]}',
        ]
    )

    decision = match_with_deepseek(
        ["open manhole"],
        ["manhole opening"],
        _config(),
        task_path("cache.jsonl"),
        client=client,
        max_attempts=2,
    )

    assert decision.pairs == (
        MatchPair(0, 0, "deepseek", "same_real_world_obstacle", "medium"),
    )
    assert len(client.chat.completions.calls) == 2
    assert all(
        call["temperature"] == 0 for call in client.chat.completions.calls
    )
    request_text = json.dumps(client.chat.completions.calls)
    assert "test-secret-key" not in request_text


def test_final_api_failure_returns_unresolved_and_redacts_secret(
    task_path: Callable[[str], Path],
) -> None:
    secret = "highly-sensitive-key"
    client = _FakeClient(
        [
            RuntimeError(f"authorization failed for {secret}"),
            RuntimeError(f"authorization failed for {secret}"),
        ]
    )
    cache_path = task_path("cache.jsonl")

    decision = match_with_deepseek(
        ["open manhole"],
        ["manhole opening"],
        _config(secret),
        cache_path,
        client=client,
        max_attempts=2,
    )

    assert decision == DeepSeekDecision(
        pairs=(),
        status="unresolved",
        model="deepseek-v4-flash",
        cache_key=build_cache_key(["open manhole"], ["manhole opening"]),
        raw_response=None,
        error="RuntimeError: authorization failed for [REDACTED]",
        from_cache=False,
    )
    assert len(client.chat.completions.calls) == 2
    assert secret not in cache_path.read_text(encoding="utf-8")
    failure_entry = json.loads(cache_path.read_text(encoding="utf-8").strip())
    assert failure_entry["status"] == "unresolved"
    assert failure_entry["parsed_pairs"] == []
    assert failure_entry["error"].endswith("[REDACTED]")


def test_deepseek_decision_is_frozen() -> None:
    decision = DeepSeekDecision(
        pairs=(),
        status="matched",
        model="deepseek-v4-flash",
        cache_key="key",
        raw_response='{"pairs":[]}',
        error=None,
        from_cache=False,
    )

    with pytest.raises(FrozenInstanceError):
        decision.status = "unresolved"  # type: ignore[misc]


def test_failure_cache_entry_does_not_prevent_later_retry(
    task_path: Callable[[str], Path],
) -> None:
    cache_path = task_path("cache.jsonl")
    failed_client = _FakeClient([RuntimeError("temporary outage")])
    successful_client = _FakeClient(['{"pairs":[]}'])

    failed = match_with_deepseek(
        ["open manhole"],
        ["manhole opening"],
        _config(),
        cache_path,
        client=failed_client,
        max_attempts=1,
    )
    retried = match_with_deepseek(
        ["open manhole"],
        ["manhole opening"],
        _config(),
        cache_path,
        client=successful_client,
        max_attempts=1,
    )

    assert failed.status == "unresolved"
    assert retried.status == "matched"
    assert retried.pairs == ()
    assert len(successful_client.chat.completions.calls) == 1


def test_invalid_max_attempts_fails_before_api_call(
    task_path: Callable[[str], Path],
) -> None:
    client = _FakeClient(['{"pairs":[]}'])

    with pytest.raises(ValueError, match="max_attempts"):
        match_with_deepseek(
            ["a"],
            ["b"],
            _config(),
            task_path("cache.jsonl"),
            client=client,
            max_attempts=0,
        )

    assert client.chat.completions.calls == []


def test_stale_cache_version_entry_is_not_reused(
    task_path: Callable[[str], Path],
) -> None:
    cache_path = task_path("match_cache.jsonl")
    cache_key = build_cache_key(["a"], ["b"])
    stale_entry = json.dumps(
        {
            "cache_key": cache_key,
            "matcher_version": "v1",
            "matcher_schema_version": MATCHER_SCHEMA_VERSION,
            "request": {"gt_names": ["a"], "pred_names": ["b"]},
            "raw_response": '{"pairs":[{"gt_index":0,"pred_index":0}]}',
            "parsed_pairs": [{"gt_index": 0, "pred_index": 0}],
            "model": "deepseek-v4-flash",
            "status": "matched",
        },
        ensure_ascii=False,
    )
    cache_path.write_text(stale_entry + "\n", encoding="utf-8")

    client = _FakeClient(['{"pairs":[]}'])
    decision = match_with_deepseek(
        ["a"], ["b"], _config(), cache_path, client=client, max_attempts=1,
    )
    assert decision.status == "matched"
    assert decision.from_cache is False
    assert len(client.chat.completions.calls) == 1
