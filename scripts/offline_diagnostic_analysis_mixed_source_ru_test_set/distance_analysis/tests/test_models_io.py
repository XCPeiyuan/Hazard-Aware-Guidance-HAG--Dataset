import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from offline_diagnostic_analysis.io import align_records, load_records, select_gt_hazard_records
from offline_diagnostic_analysis.models import ALLOWED_CATEGORIES, HazardObject, Record


FIXTURES = Path(__file__).parent / "fixtures"


def _load_inline(monkeypatch: pytest.MonkeyPatch, records: list[dict]):
    monkeypatch.setattr(
        Path, "read_text", lambda self, encoding: json.dumps(records)
    )
    return load_records(Path("inline.json"))


def _load_raw(monkeypatch: pytest.MonkeyPatch, raw_json: str):
    monkeypatch.setattr(Path, "read_text", lambda self, encoding: raw_json)
    return load_records(Path("inline.json"))


def _valid_record(**object_overrides: object) -> dict:
    obj = {
        "name": "cone",
        "categories": ["GROUND"],
        "distance_steps": 3,
        "distance_raw": "about 3 steps ahead",
    }
    obj.update(object_overrides)
    return {"image": "image.png", "objects": [obj]}


def test_load_records_preserves_positions_empty_predictions_and_unknown_distance() -> None:
    records = load_records(FIXTURES / "tiny_ours.json")
    gt_records = load_records(FIXTURES / "tiny_gt.json")

    assert [record.record_index for record in records] == [0, 1, 2, 3]
    assert records[1].objects == ()
    assert gt_records[3].objects[0].distance_steps is None
    assert gt_records[3].objects[0].distance_raw is None


def test_models_are_frozen_and_use_tuples() -> None:
    record = load_records(FIXTURES / "tiny_gt.json")[1]

    assert ALLOWED_CATEGORIES == {"GROUND", "PIT", "OVERHEAD"}
    assert isinstance(ALLOWED_CATEGORIES, frozenset)
    assert isinstance(record, Record)
    assert isinstance(record.objects, tuple)
    assert isinstance(record.objects[0], HazardObject)
    assert isinstance(record.objects[0].categories, tuple)
    with pytest.raises(FrozenInstanceError):
        record.image = "changed.png"  # type: ignore[misc]


@pytest.mark.parametrize(
    "categories",
    [["OTHER"], ["GROUND", "GROUND"], [], ["GROUND", 7]],
)
def test_load_records_rejects_invalid_categories(
    monkeypatch: pytest.MonkeyPatch, categories: list[object]
) -> None:
    with pytest.raises(ValueError, match="categories"):
        _load_inline(monkeypatch, [_valid_record(categories=categories)])


@pytest.mark.parametrize("distance_steps", [-1, 1.5, True, "3"])
def test_load_records_rejects_invalid_distance_steps(
    monkeypatch: pytest.MonkeyPatch, distance_steps: object
) -> None:
    with pytest.raises(ValueError, match="distance_steps"):
        _load_inline(monkeypatch, [_valid_record(distance_steps=distance_steps)])


@pytest.mark.parametrize(
    "record",
    [
        {"image": "", "objects": []},
        {"image": "   ", "objects": []},
        {"image": 5, "objects": []},
        _valid_record(name=""),
        _valid_record(name="\t "),
        _valid_record(name=5),
        _valid_record(distance_raw=""),
        _valid_record(distance_raw=" \t "),
        _valid_record(distance_raw=5),
        {"image": "image.png", "objects": "not-a-list"},
    ],
)
def test_load_records_rejects_other_invalid_fields(
    monkeypatch: pytest.MonkeyPatch, record: dict
) -> None:
    with pytest.raises(ValueError):
        _load_inline(monkeypatch, [record])


@pytest.mark.parametrize(
    "record",
    [
        {"objects": []},
        {"image": "image.png"},
        {"image": "image.png", "objects": [], "extra": "unexpected"},
    ],
)
def test_load_records_rejects_missing_and_extra_record_keys(
    monkeypatch: pytest.MonkeyPatch, record: dict
) -> None:
    with pytest.raises(ValueError, match="record 0 keys"):
        _load_inline(monkeypatch, [record])


@pytest.mark.parametrize(
    "object_update",
    [
        {"remove": "name"},
        {"remove": "categories"},
        {"remove": "distance_steps"},
        {"remove": "distance_raw"},
        {"extra": "unexpected"},
    ],
)
def test_load_records_rejects_missing_and_extra_object_keys(
    monkeypatch: pytest.MonkeyPatch, object_update: dict
) -> None:
    record = _valid_record()
    obj = record["objects"][0]
    if "remove" in object_update:
        del obj[object_update["remove"]]
    else:
        obj.update(object_update)

    with pytest.raises(ValueError, match="record 0 object 0 keys"):
        _load_inline(monkeypatch, [record])


@pytest.mark.parametrize(
    "raw_json",
    [
        '[{"image":"first.png","image":"second.png","objects":[]}]',
        (
            '[{"image":"image.png","objects":[{"name":"first","name":"second",'
            '"categories":["GROUND"],"distance_steps":1,"distance_raw":null}]}]'
        ),
    ],
)
def test_load_records_rejects_duplicate_json_keys(
    monkeypatch: pytest.MonkeyPatch, raw_json: str
) -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_raw(monkeypatch, raw_json)


def test_load_records_allows_duplicate_safe_images_and_distance_raw_mixed_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {"image": "duplicate.png", "objects": []},
        {"image": "duplicate.png", "objects": []},
        _valid_record(distance_steps=2, distance_raw=None),
        _valid_record(distance_steps=None, distance_raw="several steps ahead"),
    ]

    loaded = _load_inline(monkeypatch, records)

    assert [record.image for record in loaded[:2]] == ["duplicate.png", "duplicate.png"]
    assert loaded[2].objects[0].distance_steps == 2
    assert loaded[2].objects[0].distance_raw is None
    assert loaded[3].objects[0].distance_steps is None
    assert loaded[3].objects[0].distance_raw == "several steps ahead"


def test_align_records_allows_duplicate_images_and_aligns_them_positionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_safe_records = [
        {"image": "duplicate.png", "objects": []},
        {"image": "duplicate.png", "objects": []},
    ]
    gt = _load_inline(monkeypatch, duplicate_safe_records)
    ours = _load_inline(monkeypatch, duplicate_safe_records)
    fewshot = _load_inline(monkeypatch, duplicate_safe_records)

    aligned = align_records(gt, ours, fewshot)

    assert [row[0].record_index for row in aligned] == [0, 1]
    assert aligned == ((gt[0], ours[0], fewshot[0]), (gt[1], ours[1], fewshot[1]))


def test_align_records_aligns_three_sources_by_position_and_image() -> None:
    gt = load_records(FIXTURES / "tiny_gt.json")
    ours = load_records(FIXTURES / "tiny_ours.json")
    fewshot = load_records(FIXTURES / "tiny_fewshot.json")

    aligned = align_records(gt, ours, fewshot)

    assert len(aligned) == 4
    assert aligned[2] == (gt[2], ours[2], fewshot[2])
    assert aligned[0][0].objects == ()
    assert aligned[0][1].objects[0].name == "shadow"
    assert aligned[0][2].objects[0].name == "false positive curb"
    assert aligned[2][0].objects[0].name == "low branch"
    assert aligned[2][1].objects[0].name == "traffic cone"
    assert aligned[2][2].objects[0].name == "fewshot branch prediction"
    assert {record.image for record in aligned[2]} == {"multi_object.png"}
    assert {record.record_index for record in aligned[2]} == {2}


def test_align_records_rejects_length_and_image_mismatches() -> None:
    gt = load_records(FIXTURES / "tiny_gt.json")
    ours = load_records(FIXTURES / "tiny_ours.json")
    fewshot = load_records(FIXTURES / "tiny_fewshot.json")
    changed = Record(record_index=1, image="different.png", objects=ours[1].objects)
    wrong_position = Record(record_index=2, image=ours[1].image, objects=ours[1].objects)

    with pytest.raises(ValueError, match="length"):
        align_records(gt, ours[:-1], fewshot)
    with pytest.raises(ValueError, match="image"):
        align_records(gt, (ours[0], changed, *ours[2:]), fewshot)
    with pytest.raises(ValueError, match="positions"):
        align_records(gt, (ours[0], wrong_position, *ours[2:]), fewshot)


def test_select_gt_hazard_records_excludes_gt_safe_and_keeps_empty_predictions() -> None:
    aligned = align_records(
        load_records(FIXTURES / "tiny_gt.json"),
        load_records(FIXTURES / "tiny_ours.json"),
        load_records(FIXTURES / "tiny_fewshot.json"),
    )

    selected = select_gt_hazard_records(aligned)

    assert [records[0].record_index for records in selected] == [1, 2, 3]
    assert selected[0][1].objects == ()
    assert selected[0][2].objects == ()
    assert selected[1][2].objects[0].name == "fewshot branch prediction"
    assert selected[2][2].objects == ()
    assert all(records[0].objects for records in selected)
