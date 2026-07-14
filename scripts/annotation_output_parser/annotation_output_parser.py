"""Deterministic parser for the public ALERT/GUIDE annotation schema."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass


class OutputParseWarning(UserWarning):
    """Warn that model-output formatting required recovery or was unusable."""


@dataclass(frozen=True)
class _FieldOccurrence:
    name: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class _JsonObject:
    pairs: tuple[tuple[str, object], ...]


_STRICT_FIELD_PATTERN = re.compile(
    r"<(ALERT|GUIDE)>(.*?)</\1>",
    re.DOTALL,
)
_TAG_OPEN_PATTERN = re.compile(r"<\s*(ALERT|GUIDE)\s*>", re.IGNORECASE)
_TAG_CLOSE_PATTERN = re.compile(
    r"<\s*/\s*(?:ALERT|GUIDE)"
    r"(?=\s|>|$|<\s*(?:ALERT|GUIDE)\b)\s*>?",
    re.IGNORECASE,
)
_TAG_LIKE_PATTERN = re.compile(
    r"<\s*/?\s*(?:ALERT|GUIDE)\b",
    re.IGNORECASE,
)
_PLAIN_FIELD_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:[-*+]\s*)?(?:\*\*)?"
    r"(ALERT|GUIDE)(?:\*\*)?\s*[:：=](?:\*\*)?[ \t]*"
)
_WRAPPING_FENCE_PATTERN = re.compile(
    r"\A\s*```[^\r\n]*\r?\n(.*?)\r?\n```\s*\Z",
    re.DOTALL,
)
_SAFE_PAIRED_PATTERN = re.compile(
    r"(?<!<)<\s*SAFE\s*>\s*<\s*/\s*SAFE\s*>(?!>)",
    re.IGNORECASE,
)
_SAFE_SELF_CLOSING_PATTERN = re.compile(
    r"(?<!<)<\s*SAFE\s*/\s*>(?!>)",
    re.IGNORECASE,
)
_SAFE_ANGLE_PATTERN = re.compile(
    r"(?<![\w<])<+\s*SAFE\s*/?\s*>+(?!>)",
    re.IGNORECASE,
)
_SAFE_LINE_PATTERN = re.compile(r"^[ \t]*SAFE[ \t]*\r?$", re.IGNORECASE | re.MULTILINE)


def _normalize_content(value: str) -> str:
    return " ".join(value.split())


def _remove_wrapping_fence(text: str) -> str:
    match = _WRAPPING_FENCE_PATTERN.fullmatch(text)
    return match.group(1) if match else text


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    characters = list(text)
    for start, end in spans:
        characters[start:end] = " " * (end - start)
    return "".join(characters)


def _strict_fields(
    text: str,
    field_start_boundaries: list[int],
) -> list[_FieldOccurrence] | None:
    occurrences: list[_FieldOccurrence] = []
    for match in _STRICT_FIELD_PATTERN.finditer(text):
        if _TAG_LIKE_PATTERN.search(match.group(2)):
            return None
        content_start, content_end = match.span(2)
        if any(
            content_start <= boundary < content_end
            for boundary in field_start_boundaries
        ):
            return None
        occurrences.append(
            _FieldOccurrence(
                name=match.group(1),
                value=_normalize_content(match.group(2)),
                start=match.start(),
                end=match.end(),
            )
        )
    if not occurrences:
        return None

    residual = _mask_spans(
        text,
        [(occurrence.start, occurrence.end) for occurrence in occurrences],
    )
    if _TAG_LIKE_PATTERN.search(residual):
        return None
    return occurrences


def _matching_close_pattern(field_name: str) -> re.Pattern[str]:
    return re.compile(
        rf"<\s*/\s*{re.escape(field_name)}"
        rf"(?=\s|>|$|<\s*(?:ALERT|GUIDE)\b)\s*>?",
        re.IGNORECASE,
    )


def _tolerant_tag_fields(
    text: str,
    marker_boundaries: list[int],
    field_start_boundaries: list[int],
) -> tuple[list[_FieldOccurrence], bool]:
    occurrences: list[_FieldOccurrence] = []
    recovered_formatting = False
    cursor = 0

    while opener := _TAG_OPEN_PATTERN.search(text, cursor):
        field_name = opener.group(1).upper()
        closing = _matching_close_pattern(field_name).search(text, opener.end())
        later_field_starts = [
            boundary
            for boundary in field_start_boundaries
            if boundary >= opener.end()
        ]
        next_field_start = min(later_field_starts, default=None)

        if closing is not None and (
            next_field_start is None or closing.start() < next_field_start
        ):
            content_end = closing.start()
            occurrence_end = closing.end()
            expected_open = f"<{field_name}>"
            expected_close = f"</{field_name}>"
            if opener.group(0) != expected_open or closing.group(0) != expected_close:
                recovered_formatting = True
            cursor = closing.end()
        elif closing is not None and next_field_start is not None:
            content_end = next_field_start
            occurrence_end = next_field_start
            recovered_formatting = True
            cursor = next_field_start
        else:
            later_markers = [
                boundary
                for boundary in marker_boundaries
                if boundary >= opener.end()
            ]
            content_end = min(later_markers, default=len(text))
            occurrence_end = content_end
            recovered_formatting = True
            cursor = content_end if content_end > opener.start() else opener.end()

        raw_content = text[opener.end() : content_end]
        if _TAG_LIKE_PATTERN.search(raw_content):
            recovered_formatting = True

        occurrences.append(
            _FieldOccurrence(
                name=field_name,
                value=_normalize_content(raw_content),
                start=opener.start(),
                end=occurrence_end,
            )
        )

    return occurrences, recovered_formatting


def _plain_fields(
    text: str,
    marker_boundaries: list[int] | None = None,
) -> list[_FieldOccurrence]:
    matches = list(_PLAIN_FIELD_PATTERN.finditer(text))
    marker_starts = sorted(marker_boundaries or [])
    occurrences: list[_FieldOccurrence] = []
    for index, match in enumerate(matches):
        content_boundaries = [
            boundary for boundary in marker_starts if boundary >= match.end()
        ]
        if index + 1 < len(matches):
            content_boundaries.append(matches[index + 1].start())
        content_end = min(content_boundaries, default=len(text))
        occurrences.append(
            _FieldOccurrence(
                name=match.group(1).upper(),
                value=_normalize_content(text[match.end() : content_end]),
                start=match.start(),
                end=content_end,
            )
        )
    return occurrences


def _preserve_json_object(pairs: list[tuple[str, object]]) -> _JsonObject:
    return _JsonObject(tuple(pairs))


def _json_object(text: str) -> _JsonObject | None:
    try:
        value = json.loads(text, object_pairs_hook=_preserve_json_object)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, _JsonObject) else None


def _json_fields(
    value: _JsonObject,
) -> tuple[list[_FieldOccurrence], bool, bool]:
    occurrences: list[_FieldOccurrence] = []
    invalid_field_value = False
    duplicate_field = False
    seen_fields: set[str] = set()
    for key, field_value in value.pairs:
        lowered_key = key.lower()
        if lowered_key not in {"alert", "guide"}:
            continue
        duplicate_field = duplicate_field or lowered_key in seen_fields
        seen_fields.add(lowered_key)
        if isinstance(field_value, str):
            occurrences.append(
                _FieldOccurrence(
                    name=key.upper(),
                    value=_normalize_content(field_value),
                    start=0,
                    end=0,
                )
            )
        else:
            invalid_field_value = True
    return occurrences, invalid_field_value, duplicate_field


def _collect_values(
    occurrences: list[_FieldOccurrence],
) -> tuple[str, str, bool, bool]:
    values: dict[str, list[str]] = {"ALERT": [], "GUIDE": []}
    duplicate = False
    empty_field = False

    for occurrence in occurrences:
        if not occurrence.value:
            empty_field = True
            continue
        field_values = values[occurrence.name]
        if field_values:
            duplicate = True
        if occurrence.value not in field_values:
            field_values.append(occurrence.value)

    return (
        " ".join(values["ALERT"]),
        " ".join(values["GUIDE"]),
        duplicate,
        empty_field,
    )


def _find_safe_markers(
    text: str,
    excluded_spans: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], bool]:
    visible = _mask_spans(text, excluded_spans)
    markers: list[tuple[int, int]] = []
    malformed = False

    def add_matches(pattern: re.Pattern[str], is_malformed: bool) -> None:
        nonlocal visible, malformed
        for match in pattern.finditer(visible):
            markers.append((match.start(), match.end()))
            malformed = malformed or is_malformed
        if markers:
            visible = _mask_spans(visible, markers)

    add_matches(_SAFE_PAIRED_PATTERN, False)
    add_matches(_SAFE_SELF_CLOSING_PATTERN, False)
    add_matches(_SAFE_ANGLE_PATTERN, True)
    add_matches(_SAFE_LINE_PATTERN, False)
    markers.sort()
    return markers, malformed


def _marker_starts_line(text: str, start: int) -> bool:
    line_start = max(text.rfind("\n", 0, start), text.rfind("\r", 0, start)) + 1
    return not text[line_start:start].strip()


def _collect_marker_boundaries(
    text: str,
    detection_enabled: bool,
) -> tuple[list[int], list[int]]:
    tag_open_starts = [match.start() for match in _TAG_OPEN_PATTERN.finditer(text)]
    tag_close_starts = [match.start() for match in _TAG_CLOSE_PATTERN.finditer(text)]
    plain_starts = [match.start() for match in _PLAIN_FIELD_PATTERN.finditer(text)]
    safe_boundary_starts: list[int] = []
    if detection_enabled:
        safe_markers, _ = _find_safe_markers(text, [])
        safe_boundary_starts = [
            start
            for start, _ in safe_markers
            if _marker_starts_line(text, start)
        ]

    field_start_boundaries = sorted(set([*tag_open_starts, *plain_starts]))
    marker_boundaries = sorted(
        set([*field_start_boundaries, *tag_close_starts, *safe_boundary_starts])
    )
    return marker_boundaries, field_start_boundaries


def _json_is_safe(value: _JsonObject | None) -> bool:
    if value is None:
        return False
    for key, item in value.pairs:
        lowered_key = key.lower()
        if lowered_key == "safe" and item is True:
            return True
        if (
            lowered_key == "status"
            and isinstance(item, str)
            and item.lower() == "safe"
        ):
            return True
    return False


def _validated_detection_switch(detect_hazard: object) -> bool:
    if isinstance(detect_hazard, bool):
        return detect_hazard
    if type(detect_hazard) is int and detect_hazard in (0, 1):
        return bool(detect_hazard)
    raise TypeError("detect_hazard must be a bool or the integer 0 or 1")


def parse_output(
    text: str,
    detect_hazard: bool | int = False,
) -> tuple[str, str] | tuple[str, str, bool | None]:
    """Parse ALERT/GUIDE fields and optionally classify explicit SAFE output."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    detection_enabled = _validated_detection_switch(detect_hazard)

    original = text
    working = _remove_wrapping_fence(text)
    format_problem = False
    json_value: _JsonObject | None = None
    invalid_json_field = False
    marker_boundaries, field_start_boundaries = _collect_marker_boundaries(
        working,
        detection_enabled,
    )

    tag_occurrences = _strict_fields(working, field_start_boundaries)
    if tag_occurrences is None:
        tag_occurrences, tolerant_recovery = _tolerant_tag_fields(
            working,
            marker_boundaries,
            field_start_boundaries,
        )
        format_problem = format_problem or tolerant_recovery

    tag_spans = [
        (occurrence.start, occurrence.end) for occurrence in tag_occurrences
    ]
    plain_occurrences = _plain_fields(
        _mask_spans(working, tag_spans),
        marker_boundaries,
    )
    if plain_occurrences:
        format_problem = True
    occurrences = sorted(
        [*tag_occurrences, *plain_occurrences],
        key=lambda occurrence: occurrence.start,
    )
    field_spans = [
        *tag_spans,
        *[(occurrence.start, occurrence.end) for occurrence in plain_occurrences],
    ]
    used_json = False
    if not occurrences:
        json_value = _json_object(working.strip())
        if json_value is not None:
            occurrences, invalid_json_field, duplicate_json_field = _json_fields(
                json_value
            )
            used_json = True
            format_problem = True
            format_problem = (
                format_problem or invalid_json_field or duplicate_json_field
            )

    alert, guide, duplicate, empty_field = _collect_values(occurrences)
    format_problem = format_problem or duplicate or empty_field

    safe_markers: list[tuple[int, int]] = []
    malformed_safe = False
    safe_found = False
    if detection_enabled:
        if json_value is None and not used_json:
            json_value = _json_object(working.strip())
        json_safe = _json_is_safe(json_value)
        if json_value is None:
            safe_markers, malformed_safe = _find_safe_markers(working, [])
        safe_found = json_safe or bool(safe_markers)
        format_problem = format_problem or malformed_safe

    if field_spans or safe_markers:
        residual_spans = field_spans + safe_markers
        if _mask_spans(working, residual_spans).strip():
            format_problem = True

    fields_present = bool(alert or guide)
    if safe_found and fields_present:
        format_problem = True

    if not fields_present and (not safe_found or invalid_json_field):
        fallback_alert = "" if not original.strip() else original
        warnings.warn(
            "model output did not contain a recoverable ALERT, GUIDE, or SAFE field",
            OutputParseWarning,
            stacklevel=2,
        )
        if detection_enabled:
            return fallback_alert, "", None
        return fallback_alert, ""

    if format_problem:
        warnings.warn(
            "model output formatting required deterministic recovery",
            OutputParseWarning,
            stacklevel=2,
        )

    if detection_enabled:
        hazard = True if fields_present else False
        return alert, guide, hazard
    return alert, guide


if __name__ == "__main__":
    x = "<ALERT>前方有车辆</ALERT><GUIDE>向左绕行</GUIDE>"
    alert, guide, hazard = parse_output(x, detect_hazard=True)
    print(f"alert: {alert}")
    print(f"guide: {guide}")
    print(f"hazard: {hazard}")
