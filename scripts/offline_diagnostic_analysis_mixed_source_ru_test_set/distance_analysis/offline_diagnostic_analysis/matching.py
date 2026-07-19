"""Hybrid matching pipeline: rules → DeepSeek → overrides → object results."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_NEAR_THRESHOLD = 5
_MEDIUM_THRESHOLD = 8
_ELEVATED_CATEGORIES = frozenset({"PIT", "OVERHEAD"})
_OVERRIDE_FIELDS = frozenset(
    {"record_index", "method", "gt_index", "pred_index", "reason"}
)

from .io import AlignedRecord
from .models import ALLOWED_CATEGORIES, HazardObject, Record
from .rule_matcher import MatchPair, match_by_rules
from .deepseek_matcher import (
    LLMConfig,
    DeepSeekDecision,
    match_with_deepseek,
)


@dataclass(frozen=True)
class ManualOverride:
    record_index: int
    method: str
    gt_index: int
    pred_index: int | None
    reason: str


@dataclass(frozen=True)
class ObjectResult:
    record_index: int
    image: str
    gt_object_index: int
    gt_object_name: str
    gt_categories: tuple[str, ...]
    gt_distance_steps: int | None
    gt_distance_bin: str
    method: str
    matched_pred_index: int | None
    matched_pred_name: str | None
    match_source: str
    match_reason: str
    detected: bool


@dataclass(frozen=True)
class RecordMatchResult:
    record_index: int
    image: str
    method: str
    gt_count: int
    pred_count: int
    matched_pairs: tuple[MatchPair, ...]
    unresolved_gt_indices: tuple[int, ...]
    unresolved_pred_indices: tuple[int, ...]
    deepseek_decision: DeepSeekDecision | None


def _distance_bin(steps: int | None) -> str:
    if steps is None:
        return "Unknown"
    if steps <= _NEAR_THRESHOLD:
        return "Near"
    if steps <= _MEDIUM_THRESHOLD:
        return "Medium"
    return "Far"


def load_overrides(path: Path) -> tuple[ManualOverride, ...]:
    if not path.exists():
        return ()
    overrides: list[ManualOverride] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"override line {lineno} invalid JSON: {exc}")
            if not isinstance(obj, dict):
                raise ValueError(f"override line {lineno} must be an object")
            if set(obj) != _OVERRIDE_FIELDS:
                raise ValueError(
                    f"override line {lineno} fields must be exactly "
                    f"{sorted(_OVERRIDE_FIELDS)}"
                )
            ri = obj["record_index"]
            method = obj["method"]
            gt = obj["gt_index"]
            pred = obj["pred_index"]
            reason = obj["reason"]
            for field_name, value in (("record_index", ri), ("gt_index", gt)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"override line {lineno}: {field_name} must be a "
                        "non-negative integer"
                    )
            if method not in {"ours", "fewshot"}:
                raise ValueError(
                    f"override line {lineno}: method must be 'ours' or 'fewshot'"
                )
            if pred is not None and (
                isinstance(pred, bool) or not isinstance(pred, int) or pred < 0
            ):
                raise ValueError(
                    f"override line {lineno}: pred_index must be null or a "
                    "non-negative integer"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"override line {lineno}: reason must be a non-empty string"
                )
            overrides.append(ManualOverride(ri, method, gt, pred, reason.strip()))
    return tuple(overrides)


def match_record(
    gt_record: Record,
    pred_record: Record,
    method: str,
    deepseek_matcher: Any | None,
    config: LLMConfig | None,
    cache_path: Path | None,
    *,
    profile: str = "strict",
) -> RecordMatchResult:
    gt_objects = gt_record.objects
    pred_objects = pred_record.objects

    rule_pairs, unresolved_gt, unresolved_pred = match_by_rules(gt_objects, pred_objects, profile=profile)

    deepseek_decision: DeepSeekDecision | None = None
    deepseek_pairs: list[MatchPair] = []

    if unresolved_gt and unresolved_pred and deepseek_matcher is not None and config is not None and cache_path is not None:
        gt_names = [gt_objects[i].name for i in unresolved_gt]
        pred_names = [pred_objects[i].name for i in unresolved_pred]
        deepseek_decision = deepseek_matcher(
            gt_names=gt_names,
            pred_names=pred_names,
            config=config,
            cache_path=cache_path,
        )
        if deepseek_decision.status == "matched":
            for pair in deepseek_decision.pairs:
                deepseek_pairs.append(
                    MatchPair(
                        gt_index=unresolved_gt[pair.gt_index],
                        pred_index=unresolved_pred[pair.pred_index],
                        source="deepseek",
                        reason=pair.reason,
                        confidence=pair.confidence,
                    )
                )

    all_pairs = sorted(
        list(rule_pairs) + deepseek_pairs,
        key=lambda p: (p.gt_index, p.pred_index),
    )
    all_matched_gt = {p.gt_index for p in all_pairs}
    all_matched_pred = {p.pred_index for p in all_pairs}

    final_unresolved_gt = tuple(
        i for i in range(len(gt_objects)) if i not in all_matched_gt
    )
    final_unresolved_pred = tuple(
        i for i in range(len(pred_objects)) if i not in all_matched_pred
    )

    return RecordMatchResult(
        record_index=gt_record.record_index,
        image=gt_record.image,
        method=method,
        gt_count=len(gt_objects),
        pred_count=len(pred_objects),
        matched_pairs=tuple(all_pairs),
        unresolved_gt_indices=final_unresolved_gt,
        unresolved_pred_indices=final_unresolved_pred,
        deepseek_decision=deepseek_decision,
    )


def apply_overrides(
    result: RecordMatchResult,
    overrides: Sequence[ManualOverride],
) -> RecordMatchResult:
    pairs = list(result.matched_pairs)
    matched_gt = {p.gt_index for p in pairs}
    matched_pred = {p.pred_index for p in pairs}
    unresolved_gt = list(result.unresolved_gt_indices)
    unresolved_pred = list(result.unresolved_pred_indices)

    for override in overrides:
        if (
            override.record_index != result.record_index
            or override.method != result.method
        ):
            continue
        if override.gt_index >= result.gt_count or (
            override.pred_index is not None and override.pred_index >= result.pred_count
        ):
            raise ValueError(
                f"override out of bounds for record {result.record_index} "
                f"method {result.method}: gt={override.gt_index}, "
                f"pred={override.pred_index}"
            )

        pairs = [p for p in pairs if p.gt_index != override.gt_index]

        if override.pred_index is not None:
            pairs = [p for p in pairs if p.pred_index != override.pred_index]
            pairs.append(
                MatchPair(
                    gt_index=override.gt_index,
                    pred_index=override.pred_index,
                    source="manual_override",
                    reason=override.reason,
                    confidence="high",
                )
            )
            logger.info(
                "override applied: record=%s method=%s gt=%d pred=%s reason=%s",
                override.record_index, override.method,
                override.gt_index, override.pred_index, override.reason,
            )

    pairs = sorted(pairs, key=lambda p: (p.gt_index, p.pred_index))
    matched_gt = {p.gt_index for p in pairs}
    matched_pred = {p.pred_index for p in pairs}
    if len(matched_gt) != len(pairs) or len(matched_pred) != len(pairs):
        raise ValueError("manual overrides produced non-one-to-one matches")

    return RecordMatchResult(
        record_index=result.record_index,
        image=result.image,
        method=result.method,
        gt_count=result.gt_count,
        pred_count=result.pred_count,
        matched_pairs=tuple(pairs),
        unresolved_gt_indices=tuple(
            i for i in range(result.gt_count) if i not in matched_gt
        ),
        unresolved_pred_indices=tuple(
            i for i in range(result.pred_count) if i not in matched_pred
        ),
        deepseek_decision=result.deepseek_decision,
    )


def build_object_results(
    aligned: Sequence[AlignedRecord],
    methods: Sequence[str],
    overrides: Sequence[ManualOverride],
    deepseek_matcher: Any | None = None,
    config: LLMConfig | None = None,
    cache_path: Path | None = None,
    *,
    profile: str = "strict",
) -> tuple[list[ObjectResult], list[dict], list[dict]]:
    results: list[ObjectResult] = []
    unmatched_preds: list[dict] = []
    review_queue: list[dict] = []

    for gt_record, ours_record, fewshot_record in aligned:
        record_method_rows: dict[str, dict[int, ObjectResult]] = {}
        for method in methods:
            pred_record = ours_record if method == "ours" else fewshot_record
            method_overrides = [
                o for o in overrides
                if o.record_index == gt_record.record_index and o.method == method
            ]
            match_result = match_record(
                gt_record, pred_record, method,
                deepseek_matcher, config, cache_path,
                profile=profile,
            )
            match_result = apply_overrides(match_result, method_overrides)

            gt_objects = gt_record.objects
            pred_objects = pred_record.objects

            pair_map = {p.gt_index: p for p in match_result.matched_pairs}
            used_pred = {p.pred_index for p in match_result.matched_pairs}

            for gt_idx in range(len(gt_objects)):
                gt_obj = gt_objects[gt_idx]
                pair = pair_map.get(gt_idx)
                pred_obj = pred_objects[pair.pred_index] if pair is not None else None
                results.append(
                    ObjectResult(
                        record_index=gt_record.record_index,
                        image=gt_record.image,
                        gt_object_index=gt_idx,
                        gt_object_name=gt_obj.name,
                        gt_categories=gt_obj.categories,
                        gt_distance_steps=gt_obj.distance_steps,
                        gt_distance_bin=_distance_bin(gt_obj.distance_steps),
                        method=method,
                        matched_pred_index=pair.pred_index if pair else None,
                        matched_pred_name=pred_obj.name if pred_obj else None,
                        match_source=pair.source if pair else "unresolved",
                        match_reason=pair.reason if pair else "no match found",
                        detected=pair is not None,
                    )
                )
            record_method_rows[method] = {
                row.gt_object_index: row
                for row in results
                if row.record_index == gt_record.record_index and row.method == method
            }

            # Record unmatched predictions
            for pred_idx in range(len(pred_objects)):
                if pred_idx not in used_pred:
                    pred_obj = pred_objects[pred_idx]
                    unmatched_preds.append({
                        "record_index": gt_record.record_index,
                        "image": gt_record.image,
                        "method": method,
                        "pred_index": pred_idx,
                        "pred_name": pred_obj.name,
                        "pred_categories": list(pred_obj.categories),
                        "pred_distance_steps": pred_obj.distance_steps,
                    })

            # Build review queue items
            if match_result.deepseek_decision:
                for pair in (
                    p for p in match_result.matched_pairs if p.source == "deepseek"
                ):
                    review_queue.append({
                        "record_index": gt_record.record_index,
                        "image": gt_record.image,
                        "method": method,
                        "type": "deepseek_match",
                        "gt_index": pair.gt_index,
                        "gt_name": gt_objects[pair.gt_index].name,
                        "pred_index": pair.pred_index,
                        "pred_name": pred_objects[pair.pred_index].name,
                    })
                if match_result.deepseek_decision.status == "unresolved":
                    review_queue.append({
                        "record_index": gt_record.record_index,
                        "image": gt_record.image,
                        "method": method,
                        "type": "deepseek_failure",
                        "error": match_result.deepseek_decision.error,
                        "unresolved_gt": [
                            {"gt_index": i, "gt_name": gt_objects[i].name}
                            for i in match_result.unresolved_gt_indices
                        ],
                        "unresolved_pred": [
                            {"pred_index": i, "pred_name": pred_objects[i].name}
                            for i in match_result.unresolved_pred_indices
                        ],
                    })

            for gt_idx in range(len(gt_objects)):
                cats_set = set(gt_objects[gt_idx].categories)
                if cats_set & _ELEVATED_CATEGORIES:
                    pair = pair_map.get(gt_idx)
                    review_queue.append({
                        "record_index": gt_record.record_index,
                        "image": gt_record.image,
                        "method": method,
                        "type": "pit_overhead",
                        "gt_index": gt_idx,
                        "gt_name": gt_objects[gt_idx].name,
                        "gt_categories": list(gt_objects[gt_idx].categories),
                        "detected": pair is not None,
                        "match_source": pair.source if pair else "unresolved",
                    })

            # Competing candidates: records with unresolved GT AND unresolved pred
            if match_result.unresolved_gt_indices and match_result.unresolved_pred_indices:
                review_queue.append({
                    "record_index": gt_record.record_index,
                    "image": gt_record.image,
                    "method": method,
                    "type": "competing_candidates",
                    "unresolved_gt": list(match_result.unresolved_gt_indices),
                    "unresolved_pred": list(match_result.unresolved_pred_indices),
                })

            # Rule match sample: deterministic sample of rule-only matches
            rule_pairs = [p for p in match_result.matched_pairs if p.source == "rule"]
            if rule_pairs:
                review_queue.append({
                    "record_index": gt_record.record_index,
                    "image": gt_record.image,
                    "method": method,
                    "type": "rule_match_sample",
                    "rule_matches": [
                        {"gt_index": p.gt_index, "pred_index": p.pred_index,
                         "reason": p.reason}
                        for p in rule_pairs
                    ],
                })

        if {"ours", "fewshot"} <= set(record_method_rows):
            for gt_idx, ours_row in record_method_rows["ours"].items():
                fewshot_row = record_method_rows["fewshot"][gt_idx]
                if ours_row.detected == fewshot_row.detected:
                    continue
                review_queue.append({
                    "record_index": gt_record.record_index,
                    "image": gt_record.image,
                    "type": "detection_disagreement",
                    "gt_index": gt_idx,
                    "gt_name": ours_row.gt_object_name,
                    "ours_detected": ours_row.detected,
                    "ours_pred_index": ours_row.matched_pred_index,
                    "ours_pred_name": ours_row.matched_pred_name,
                    "fewshot_detected": fewshot_row.detected,
                    "fewshot_pred_index": fewshot_row.matched_pred_index,
                    "fewshot_pred_name": fewshot_row.matched_pred_name,
                })

    return results, unmatched_preds, review_queue


def build_disagreement_queue(
    object_results: list[ObjectResult],
) -> list[dict]:
    """Generate ours/few-shot disagreement review queue.

    Returns one entry per (record_index, gt_object_index) where ours and
    fewshot detection outcomes differ for the same GT object.
    """
    from collections import defaultdict

    grouped: dict[tuple[int, int], dict[str, ObjectResult]] = defaultdict(dict)
    for r in object_results:
        grouped[(r.record_index, r.gt_object_index)][r.method] = r

    disagreements: list[dict] = []
    for (record_index, gt_idx), method_map in sorted(grouped.items()):
        ours_row = method_map.get("ours")
        fewshot_row = method_map.get("fewshot")
        if ours_row is None or fewshot_row is None:
            continue
        if ours_row.detected == fewshot_row.detected:
            continue
        disagreements.append(
            {
                "record_index": record_index,
                "image": ours_row.image,
                "type": "detection_disagreement",
                "gt_index": gt_idx,
                "gt_name": ours_row.gt_object_name,
                "ours_detected": ours_row.detected,
                "ours_pred_index": ours_row.matched_pred_index,
                "ours_pred_name": ours_row.matched_pred_name,
                "fewshot_detected": fewshot_row.detected,
                "fewshot_pred_index": fewshot_row.matched_pred_index,
                "fewshot_pred_name": fewshot_row.matched_pred_name,
            }
        )
    return disagreements
