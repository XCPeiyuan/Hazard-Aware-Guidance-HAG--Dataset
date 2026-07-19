import math
import re
from typing import Literal, Sequence

from .models import HazardObject, Record


DistanceProvenance = Literal["direct_numeric", "derived_reviewed", "unknown"]

_NUMERIC_DISTANCE = re.compile(
    r"\b(?:between\s+)?(?P<low>\d+(?:\.\d+)?)\s*"
    r"(?:(?:-|to|and)\s*(?P<high>\d+(?:\.\d+)?)\s*)?"
    r"(?P<unit>steps?|meters?|metres?|feet|foot)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_FROM_RANGE = re.compile(
    r"\bfrom\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\s+"
    r"(?:steps?|meters?|metres?|feet|foot)\b",
    re.IGNORECASE,
)
_RELATIONAL_MARKER = re.compile(
    r"\b(?:"
    r"behind|ahead\s+of|in(?:\s+|-)front(?:\s+|-)of|above|below|"
    r"off\s+the\s+ground|adjacent|nearby|near|"
    r"(?:to\s+the\s+)?(?:left|right)\s+of|from|within|"
    r"same\s+as|same\s+distance|co[- ]located|next\s+to|beside|"
    r"under(?:neath)?|relative\s+to|to\s+its?"
    r")\b",
    re.IGNORECASE,
)
_NON_DISTANCE_MARKER = re.compile(
    r"\b(?:"
    r"walk|move|turn|go|wide|width|tall|height|high|long|length|deep|depth"
    r")\b",
    re.IGNORECASE,
)
_NON_EXACT_MARKER = re.compile(
    r"(?:\b(?:"
    r"at\s+least|at\s+most|(?:no\s+)?less\s+than|(?:no\s+)?more\s+than|"
    r"greater\s+than|over|up\s+to|farther|further|closer|nearer|"
    r"minimum|maximum|or\s+more|or\s+less"
    r")\b|[<>]=?)",
    re.IGNORECASE,
)


def _parsed_distance_steps(raw: str) -> int | None:
    matches = list(_NUMERIC_DISTANCE.finditer(raw))
    if len(matches) != 1:
        return None

    match = matches[0]
    low = float(match.group("low"))
    high_text = match.group("high")
    expected_number_count = 1 if high_text is None else 2
    if len(_NUMBER.findall(raw)) != expected_number_count:
        return None
    if high_text is not None and float(high_text) < low:
        return None

    value = low if high_text is None else (low + float(high_text)) / 2
    unit = match.group("unit").lower()

    if unit.startswith("meter") or unit.startswith("metre"):
        value /= 0.6
    elif unit in {"foot", "feet"}:
        value = value * 0.3048 / 0.6

    return math.floor(value + 0.5)
_RELATIONAL_AROUND = re.compile(r"\baround\b(?!\s*\d)", re.IGNORECASE)
_UNCERTAINTY_MARKER = re.compile(
    r"\b(?:"
    r"uncertain|uncertaint(?:y|ies)|possibly|maybe|might\s+be|could\s+be|"
    r"unknown|unclear|perhaps|probable|probably|estimated|estimate|"
    r"guess|guessed|ambiguous|likely|unlikely|apparently|seemingly"
    r")\b",
    re.IGNORECASE,
)


def classify_distance_provenance(obj: HazardObject) -> DistanceProvenance:
    if obj.distance_steps is None:
        return "unknown"
    if obj.distance_raw is None:
        return "derived_reviewed"
    if _UNCERTAINTY_MARKER.search(obj.distance_raw) or _NON_EXACT_MARKER.search(
        obj.distance_raw
    ):
        return "derived_reviewed"
    relational_text = _FROM_RANGE.sub("", obj.distance_raw)
    if (
        _RELATIONAL_MARKER.search(relational_text)
        or _RELATIONAL_AROUND.search(obj.distance_raw)
        or _NON_DISTANCE_MARKER.search(obj.distance_raw)
    ):
        return "derived_reviewed"
    if _parsed_distance_steps(obj.distance_raw) == obj.distance_steps:
        return "direct_numeric"
    return "derived_reviewed"


def build_gt_distance_provenance(
    gt_records: Sequence[Record],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for record in gt_records:
        for object_index, obj in enumerate(record.objects):
            rows.append(
                {
                    "record_index": record.record_index,
                    "image": record.image,
                    "object_index": object_index,
                    "name": obj.name,
                    "categories": obj.categories,
                    "distance_steps": obj.distance_steps,
                    "distance_raw": obj.distance_raw,
                    "provenance": classify_distance_provenance(obj),
                }
            )
    return tuple(rows)
