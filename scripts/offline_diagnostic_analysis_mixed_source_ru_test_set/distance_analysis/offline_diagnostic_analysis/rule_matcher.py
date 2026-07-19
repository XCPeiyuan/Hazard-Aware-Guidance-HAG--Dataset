from dataclasses import dataclass
from typing import Sequence

from .models import HazardObject
from .normalize import are_controlled_synonyms, normalize_name


@dataclass(frozen=True)
class MatchPair:
    gt_index: int
    pred_index: int
    source: str
    reason: str
    confidence: str


def _candidate_reason(left: str, right: str, *, profile: str = "strict") -> str | None:
    if left and left == right:
        return "exact_normalized_name"
    if are_controlled_synonyms(left, right, profile=profile):
        return "controlled_synonym"
    return None


def _deterministic_maximum_pairs(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    *,
    profile: str = "strict",
) -> list[tuple[int, int, str]]:
    unmatched_gt = set(range(len(gt_names)))
    unmatched_pred = set(range(len(pred_names)))
    pairs: list[tuple[int, int, str]] = []

    # Exact normalized names are stronger evidence than controlled synonyms.
    # Within each evidence tier, sorted greedy matching is maximum-cardinality
    # because approved synonym groups are disjoint.
    for reason in ("exact_normalized_name", "controlled_synonym"):
        for gt_index in sorted(unmatched_gt):
            candidates = [
                pred_index
                for pred_index in sorted(unmatched_pred)
                if _candidate_reason(gt_names[gt_index], pred_names[pred_index], profile=profile)
                == reason
            ]
            if not candidates:
                continue
            pred_index = candidates[0]
            pairs.append((gt_index, pred_index, reason))
            unmatched_gt.remove(gt_index)
            unmatched_pred.remove(pred_index)

    return pairs


def match_by_rules(
    gt_objects: Sequence[HazardObject],
    pred_objects: Sequence[HazardObject],
    *,
    profile: str = "strict",
) -> tuple[list[MatchPair], list[int], list[int]]:
    """Match only deterministic high-confidence one-to-one name pairs."""
    gt_names = [normalize_name(obj.name) for obj in gt_objects]
    pred_names = [normalize_name(obj.name) for obj in pred_objects]
    unique_pairs = _deterministic_maximum_pairs(gt_names, pred_names, profile=profile)
    pairs = [
        MatchPair(
            gt_index=gt_index,
            pred_index=pred_index,
            source="rule",
            reason=reason,
            confidence="high",
        )
        for gt_index, pred_index, reason in unique_pairs
    ]
    resolved_gt = {pair.gt_index for pair in pairs}
    resolved_pred = {pair.pred_index for pair in pairs}

    return (
        sorted(pairs, key=lambda pair: (pair.gt_index, pair.pred_index)),
        [index for index in range(len(gt_objects)) if index not in resolved_gt],
        [index for index in range(len(pred_objects)) if index not in resolved_pred],
    )
