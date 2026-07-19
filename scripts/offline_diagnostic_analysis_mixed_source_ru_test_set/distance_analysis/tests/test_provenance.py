from typing import get_args

import pytest

from offline_diagnostic_analysis.models import HazardObject, Record
from offline_diagnostic_analysis.provenance import (
    DistanceProvenance,
    build_gt_distance_provenance,
    classify_distance_provenance,
)


def _hazard(
    distance_steps: int | None,
    distance_raw: str | None,
) -> HazardObject:
    return HazardObject(
        name="traffic cone",
        categories=("GROUND",),
        distance_steps=distance_steps,
        distance_raw=distance_raw,
    )


def test_distance_provenance_has_only_the_auditable_classes() -> None:
    assert get_args(DistanceProvenance) == (
        "direct_numeric",
        "derived_reviewed",
        "unknown",
    )


@pytest.mark.parametrize(
    ("distance_steps", "raw"),
    [
        (3, "about 3 steps ahead"),
        (3, "roughly 1.8 meters to the left"),
        (5, "3 metres ahead"),
        (3, "approximately 6 feet to the right"),
        (4, "around 4 steps ahead"),
        (1, "1 foot away"),
        (4, "between 3 and 5 steps ahead"),
        (5, "4 to 5 steps ahead"),
        (7, "3-5 metres ahead"),
        (5, "from 4 to 5 steps ahead"),
        (5, "from 2.4 to 3.6 meters ahead"),
        (3, "from 5 to 7 feet ahead"),
    ],
)
def test_absolute_numeric_distance_phrases_are_direct_numeric(
    distance_steps: int, raw: str
) -> None:
    assert (
        classify_distance_provenance(_hazard(distance_steps, raw))
        == "direct_numeric"
    )


@pytest.mark.parametrize(
    ("distance_steps", "raw"),
    [
        (4, "about 3 steps ahead"),
        (2, "roughly 1.8 meters ahead"),
        (4, "approximately 6 feet ahead"),
        (5, "between 3 and 5 steps ahead"),
        (3, "3 steps ahead and 5 steps to the right"),
        (3, "3 steps and 1.8 meters ahead"),
        (3, "between 3 steps and 5 meters"),
        (5, "3 or 5 steps ahead"),
        (4, "5-3 steps ahead"),
        (4, "from 4 to 5 steps ahead"),
    ],
)
def test_mismatch_or_multiple_numeric_phrases_are_derived_reviewed(
    distance_steps: int, raw: str
) -> None:
    assert (
        classify_distance_provenance(_hazard(distance_steps, raw))
        == "derived_reviewed"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "about 3 steps behind the barrier",
        "2 meters in front of the doorway",
        "4 feet below the branch",
        "adjacent to the curb",
        "nearby",
        "around the same distance as the cone",
        "same as the barrier",
        "co-located with the bicycle",
        "about 1 step to its left",
        "objects around the pit are 3 steps away",
        "about 5 steps ahead of the van",
        "about 2 steps to the left of the sign",
        "about 2 steps to the right of the sign",
        "about 2 steps left of the sign",
        "about 2 steps right of the sign",
        "about 3 steps from the doorway",
        "within 3 steps of the curb",
        "near the cone, about 3 steps away",
        "about 2 steps in-front-of the doorway",
        "3 steps above the ground",
        "3 steps off the ground",
        "3 steps above the curb",
        "3 steps below the platform",
    ],
)
def test_relational_markers_are_derived_even_with_numeric_units(raw: str) -> None:
    assert classify_distance_provenance(_hazard(3, raw)) == "derived_reviewed"


@pytest.mark.parametrize(
    "raw",
    [
        "walk 3 steps ahead",
        "move 3 steps ahead",
        "turn and go 3 steps ahead",
        "3 steps wide",
        "width is 3 steps",
        "3 steps tall",
        "height is 3 steps",
        "3 steps high",
        "3 steps long",
        "length is 3 steps",
        "3 steps deep",
        "depth is 3 steps",
    ],
)
def test_navigation_actions_and_dimensions_are_derived_reviewed(raw: str) -> None:
    assert classify_distance_provenance(_hazard(3, raw)) == "derived_reviewed"


@pytest.mark.parametrize(
    "raw",
    [
        "at least 3 steps ahead",
        "at most 3 steps ahead",
        "less than 3 steps ahead",
        "more than 3 steps ahead",
        "no less than 3 steps ahead",
        "no more than 3 steps ahead",
        "up to 3 steps ahead",
        "3 steps or farther",
        "3 steps or further",
        "3 steps or closer",
        "3 steps or nearer",
        "minimum 3 steps ahead",
        "maximum 3 steps ahead",
        "likely 3 steps ahead",
        "unlikely 3 steps ahead",
        "apparently 3 steps ahead",
        "seemingly 3 steps ahead",
        "uncertainties remain around 3 steps ahead",
        "over 3 steps ahead",
        "greater than 3 steps ahead",
        ">= 3 steps ahead",
        "> 3 steps ahead",
        "3 steps or more",
        "under 3 steps ahead",
        "<= 3 steps ahead",
        "< 3 steps ahead",
        "3 steps or less",
    ],
)
def test_non_exact_bounds_and_evidence_qualifiers_are_derived_reviewed(
    raw: str,
) -> None:
    assert classify_distance_provenance(_hazard(3, raw)) == "derived_reviewed"


@pytest.mark.parametrize(
    "raw",
    [
        "3 steps ahead",
        "3 steps to the left ahead",
        "3 steps to the right ahead",
    ],
)
def test_ordinary_object_direction_is_not_relational(raw: str) -> None:
    assert classify_distance_provenance(_hazard(3, raw)) == "direct_numeric"


@pytest.mark.parametrize(
    "raw",
    [
        "uncertain, about 3 steps ahead",
        "possibly 2 meters ahead",
        "maybe 4 metres to the left",
        "might be 6 feet away",
        "could be 1 foot ahead",
        "unknown, approximately 5 steps ahead",
        "unclear whether it is 7 meters ahead",
        "uncertainty remains at 3 steps ahead",
        "perhaps 2 meters ahead",
        "probable distance is 4 metres",
        "probably 6 feet away",
        "estimated 1 foot ahead",
        "estimate: 5 steps ahead",
        "guess of 7 meters ahead",
        "guessed 8 steps ahead",
        "ambiguous but 9 feet away",
    ],
)
def test_numeric_uncertainty_markers_take_precedence_over_units(raw: str) -> None:
    assert classify_distance_provenance(_hazard(3, raw)) == "derived_reviewed"


@pytest.mark.parametrize(
    "raw",
    [
        "turn left at 3 o'clock",
        "take 2 turns to the right",
        "object number 4 on the left",
        "several steps ahead",
        "distance is uncertain",
    ],
)
def test_non_distance_numbers_and_uncertain_phrases_are_derived(raw: str) -> None:
    assert classify_distance_provenance(_hazard(3, raw)) == "derived_reviewed"


def test_non_null_distance_without_raw_is_derived_reviewed() -> None:
    assert classify_distance_provenance(_hazard(3, None)) == "derived_reviewed"


@pytest.mark.parametrize("raw", [None, "about 3 steps ahead"])
def test_null_distance_is_unknown_regardless_of_raw(raw: str | None) -> None:
    assert classify_distance_provenance(_hazard(None, raw)) == "unknown"


def test_build_gt_distance_provenance_emits_one_auditable_dict_per_object() -> None:
    records = (
        Record(
            record_index=7,
            image="scene.png",
            objects=(
                HazardObject(
                    name="cone",
                    categories=("GROUND", "PIT"),
                    distance_steps=3,
                    distance_raw="about 3 steps ahead",
                ),
                HazardObject(
                    name="branch",
                    categories=("OVERHEAD",),
                    distance_steps=None,
                    distance_raw=None,
                ),
            ),
        ),
    )

    result = build_gt_distance_provenance(records)

    assert result == (
        {
            "record_index": 7,
            "image": "scene.png",
            "object_index": 0,
            "name": "cone",
            "categories": ("GROUND", "PIT"),
            "distance_steps": 3,
            "distance_raw": "about 3 steps ahead",
            "provenance": "direct_numeric",
        },
        {
            "record_index": 7,
            "image": "scene.png",
            "object_index": 1,
            "name": "branch",
            "categories": ("OVERHEAD",),
            "distance_steps": None,
            "distance_raw": None,
            "provenance": "unknown",
        },
    )


def test_build_gt_distance_provenance_does_not_mutate_records() -> None:
    records = (
        Record(
            record_index=0,
            image="scene.png",
            objects=(_hazard(3, "about 3 steps ahead"),),
        ),
    )
    before = repr(records)

    build_gt_distance_provenance(records)

    assert repr(records) == before
