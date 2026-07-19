import json
from pathlib import Path
from typing import Sequence

from .models import HazardObject, Record


AlignedRecord = tuple[Record, Record, Record]
REQUIRED_RECORD_KEYS = {"image", "objects"}
REQUIRED_OBJECT_KEYS = {"name", "categories", "distance_steps", "distance_raw"}


def _require_exact_keys(
    value: dict,
    required: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        unknown = sorted(actual - required)
        raise ValueError(f"{context} keys invalid: missing={missing}, unknown={unknown}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_records(path: Path) -> tuple[Record, ...]:
    raw_records = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(raw_records, list):
        raise ValueError("records JSON root must be a list")

    records: list[Record] = []
    for record_index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"record {record_index} must be an object")
        _require_exact_keys(raw_record, REQUIRED_RECORD_KEYS, f"record {record_index}")

        image = raw_record["image"]
        raw_objects = raw_record["objects"]
        if not isinstance(raw_objects, list):
            raise ValueError(f"record {record_index} objects must be a list")

        objects: list[HazardObject] = []
        for object_index, raw_object in enumerate(raw_objects):
            if not isinstance(raw_object, dict):
                raise ValueError(
                    f"record {record_index} object {object_index} must be an object"
                )
            _require_exact_keys(
                raw_object,
                REQUIRED_OBJECT_KEYS,
                f"record {record_index} object {object_index}",
            )
            raw_categories = raw_object["categories"]
            if not isinstance(raw_categories, list):
                raise ValueError(
                    f"record {record_index} object {object_index} categories must be a list"
                )
            try:
                obj = HazardObject(
                    name=raw_object["name"],
                    categories=tuple(raw_categories),
                    distance_steps=raw_object["distance_steps"],
                    distance_raw=raw_object["distance_raw"],
                )
            except ValueError as exc:
                raise ValueError(
                    f"record {record_index} object {object_index}: {exc}"
                ) from exc
            objects.append(obj)

        try:
            records.append(
                Record(record_index=record_index, image=image, objects=tuple(objects))
            )
        except ValueError as exc:
            raise ValueError(f"record {record_index}: {exc}") from exc

    return tuple(records)


def align_records(
    gt: Sequence[Record],
    ours: Sequence[Record],
    fewshot: Sequence[Record],
) -> tuple[AlignedRecord, ...]:
    if len(gt) != len(ours) or len(gt) != len(fewshot):
        raise ValueError("record source lengths must match")

    aligned: list[AlignedRecord] = []
    for position, records in enumerate(zip(gt, ours, fewshot, strict=True)):
        if any(record.record_index != position for record in records):
            raise ValueError(f"record positions do not match at position {position}")
        if len({record.image for record in records}) != 1:
            raise ValueError(f"record image values do not match at position {position}")
        aligned.append(records)
    return tuple(aligned)


def select_gt_hazard_records(
    aligned: Sequence[AlignedRecord],
) -> tuple[AlignedRecord, ...]:
    return tuple(records for records in aligned if records[0].objects)
