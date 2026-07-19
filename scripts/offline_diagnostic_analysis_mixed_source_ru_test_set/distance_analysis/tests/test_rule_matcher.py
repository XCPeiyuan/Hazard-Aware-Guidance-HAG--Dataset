from dataclasses import FrozenInstanceError

import pytest

from offline_diagnostic_analysis.models import HazardObject
from offline_diagnostic_analysis.normalize import normalize_name
from offline_diagnostic_analysis.rule_matcher import MatchPair, match_by_rules


def _object(
    name: str,
    categories: tuple[str, ...] = ("GROUND",),
    distance_steps: int | None = 3,
) -> HazardObject:
    return HazardObject(
        name=name,
        categories=categories,
        distance_steps=distance_steps,
        distance_raw=None,
    )


def test_normalize_name_handles_case_punctuation_and_whitespace() -> None:
    assert normalize_name("  Traffic-CONE,  ") == "traffic cone"


@pytest.mark.parametrize(
    ("plural", "singular"),
    [
        ("bicycles", "bicycle"),
        ("cars", "car"),
        ("barriers", "barrier"),
        ("cones", "cone"),
        ("boxes", "box"),
        ("buses", "bus"),
    ],
)
def test_normalize_name_cautiously_singularizes_supported_hazard_plurals(
    plural: str, singular: str
) -> None:
    assert normalize_name(plural) == singular


@pytest.mark.parametrize(
    "name",
    ["species", "news", "status", "glass", "gas", "bus"],
)
def test_normalize_name_preserves_invariant_or_false_collision_nouns(
    name: str,
) -> None:
    assert normalize_name(name) == name


@pytest.mark.parametrize(
    ("invariant_name", "unsafe_collision"),
    [
        ("species", "specy"),
        ("news", "new"),
        ("status", "statu"),
        ("glass", "glas"),
        ("gas", "ga"),
        ("bus", "bu"),
    ],
)
def test_false_collision_nouns_do_not_match_unsafe_singular_forms(
    invariant_name: str, unsafe_collision: str
) -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object(invariant_name)],
        [_object(unsafe_collision)],
    )

    assert pairs == []
    assert unresolved_gt == [0]
    assert unresolved_pred == [0]


def test_glasses_does_not_match_glass() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("glasses")],
        [_object("glass")],
    )

    assert pairs == []
    assert unresolved_gt == [0]
    assert unresolved_pred == [0]


def test_exact_case_and_punctuation_match() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("Traffic Cone")],
        [_object("traffic-cone!")],
    )

    assert pairs == [
        MatchPair(0, 0, "rule", "exact_normalized_name", "high")
    ]
    assert unresolved_gt == []
    assert unresolved_pred == []


def test_simple_singular_plural_match() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("bicycles")],
        [_object("bicycle")],
    )

    assert pairs == [
        MatchPair(0, 0, "rule", "exact_normalized_name", "high")
    ]
    assert unresolved_gt == []
    assert unresolved_pred == []


@pytest.mark.parametrize(
    ("gt_name", "pred_name"),
    [("bike", "bicycle"), ("car", "vehicle")],
)
def test_controlled_synonym_match(gt_name: str, pred_name: str) -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object(gt_name)],
        [_object(pred_name)],
    )

    assert pairs == [
        MatchPair(0, 0, "rule", "controlled_synonym", "high")
    ]
    assert unresolved_gt == []
    assert unresolved_pred == []


def test_broad_synonym_is_not_matched() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("barrier")],
        [_object("obstacle")],
    )

    assert pairs == []
    assert unresolved_gt == [0]
    assert unresolved_pred == [0]


def test_empty_normalized_names_are_not_matched() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("!!!")],
        [_object("---")],
    )

    assert pairs == []
    assert unresolved_gt == [0]
    assert unresolved_pred == [0]


def test_category_disagreement_does_not_block_name_match() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("cone", ("GROUND",), 2)],
        [_object("cone", ("OVERHEAD",), 99)],
    )

    assert pairs == [
        MatchPair(0, 0, "rule", "exact_normalized_name", "high")
    ]
    assert unresolved_gt == []
    assert unresolved_pred == []


def test_does_not_match_only_because_categories_agree() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("barrier", ("GROUND",))],
        [_object("cone", ("GROUND",))],
    )

    assert pairs == []
    assert unresolved_gt == [0]
    assert unresolved_pred == [0]


def test_duplicate_predictions_match_maximum_provable_exact_pairs() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("cone")],
        [_object("cone"), _object("cone")],
    )

    assert pairs == [
        MatchPair(0, 0, "rule", "exact_normalized_name", "high")
    ]
    assert unresolved_gt == []
    assert unresolved_pred == [1]


def test_duplicate_gt_matches_maximum_provable_exact_pairs() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("cone"), _object("cone")],
        [_object("cone")],
    )

    assert pairs == [
        MatchPair(0, 0, "rule", "exact_normalized_name", "high")
    ]
    assert unresolved_gt == [1]
    assert unresolved_pred == []


def test_duplicate_synonym_candidates_match_maximum_provable_pairs() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("bike")],
        [_object("bicycle"), _object("bicycles")],
    )

    assert pairs == [
        MatchPair(0, 0, "rule", "controlled_synonym", "high")
    ]
    assert unresolved_gt == []
    assert unresolved_pred == [1]


def test_exact_match_has_deterministic_priority_over_synonym_candidate() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("car")],
        [_object("car"), _object("vehicle")],
    )

    assert pairs == [
        MatchPair(0, 0, "rule", "exact_normalized_name", "high")
    ]
    assert unresolved_gt == []
    assert unresolved_pred == [1]


def test_exact_match_has_deterministic_priority_with_competing_gt_synonym() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("car"), _object("vehicle")],
        [_object("car")],
    )

    assert pairs == [
        MatchPair(0, 0, "rule", "exact_normalized_name", "high")
    ]
    assert unresolved_gt == [1]
    assert unresolved_pred == []


def test_unapproved_safe_sounding_phrase_remains_unresolved() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("construction cones")],
        [_object("cones")],
    )

    assert pairs == []
    assert unresolved_gt == [0]
    assert unresolved_pred == [0]


@pytest.mark.parametrize(
    ("gt_objects", "pred_objects", "expected_gt", "expected_pred"),
    [
        ([], [], [], []),
        ([_object("cone")], [], [0], []),
        ([], [_object("cone")], [], [0]),
    ],
)
def test_empty_sides(
    gt_objects: list[HazardObject],
    pred_objects: list[HazardObject],
    expected_gt: list[int],
    expected_pred: list[int],
) -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        gt_objects, pred_objects
    )

    assert pairs == []
    assert unresolved_gt == expected_gt
    assert unresolved_pred == expected_pred


def test_results_have_deterministic_index_ordering() -> None:
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("vehicle"), _object("cone"), _object("unknown")],
        [_object("cone"), _object("car"), _object("other")],
    )

    assert pairs == [
        MatchPair(0, 1, "rule", "controlled_synonym", "high"),
        MatchPair(1, 0, "rule", "exact_normalized_name", "high"),
    ]
    assert unresolved_gt == [2]
    assert unresolved_pred == [2]


def test_match_pair_is_frozen() -> None:
    pair = MatchPair(0, 0, "rule", "exact_normalized_name", "high")

    with pytest.raises(FrozenInstanceError):
        pair.gt_index = 1  # type: ignore[misc]


def test_match_by_rules_does_not_mutate_inputs() -> None:
    gt_objects = [_object("cones"), _object("bike")]
    pred_objects = [_object("cone"), _object("bicycle")]
    original_gt = list(gt_objects)
    original_pred = list(pred_objects)

    match_by_rules(gt_objects, pred_objects)

    assert gt_objects == original_gt
    assert pred_objects == original_pred


def test_competing_gt_candidates_preserve_at_least_one_match() -> None:
    """When two GT objects normalize to the same target as one prediction,
    the rule matcher must produce exactly one match, not zero."""
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("cones"), _object("cone of traffic")],
        [_object("cone")],
    )
    assert len(pairs) == 1
    assert pairs[0].gt_index == 0
    assert pairs[0].pred_index == 0
    assert pairs[0].reason == "exact_normalized_name"
    assert len(unresolved_gt) == 1


def test_synonym_competing_candidates_preserve_max_one_to_one() -> None:
    """When GT has both an exact match and a synonym match for the same
    prediction, exact match wins and the synonym remains unresolved."""
    pairs, unresolved_gt, unresolved_pred = match_by_rules(
        [_object("bike"), _object("bicycles")],
        [_object("bicycle")],
    )
    assert len(pairs) == 1
    assert pairs[0].reason == "exact_normalized_name"
    assert 1 in unresolved_gt or 0 in unresolved_gt
    assert len(unresolved_gt) == 1
