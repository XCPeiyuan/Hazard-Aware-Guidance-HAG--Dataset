from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


THIS_DIR = Path(__file__).resolve().parent
DISTANCE_ANALYSIS_DIR = THIS_DIR.parent / "distance_analysis"
sys.path.insert(0, str(DISTANCE_ANALYSIS_DIR))
sys.path.insert(0, str(THIS_DIR))

from offline_diagnostic_analysis.models import HazardObject, Record
from offline_diagnostic_analysis.rule_matcher import MatchPair
from run_deepseek_fig8 import (
    CATEGORY_ORDER,
    accumulate_record,
    align_all_records,
    select_unique_images,
    resolve_run_paths,
    reorder_matrix_for_plot,
    row_normalize,
    validate_one_to_one,
)


def obj(name: str, *categories: str) -> HazardObject:
    return HazardObject(name, tuple(categories), None, None)


def pair(gt_index: int, pred_index: int) -> MatchPair:
    return MatchPair(gt_index, pred_index, "test", "test", "high")


def test_matched_objects_preserve_actual_category_confusion() -> None:
    matrix = np.zeros((4, 4), dtype=float)

    summary = accumulate_record(
        matrix,
        (obj("branch", "GROUND"),),
        (obj("branch", "OVERHEAD"),),
        (pair(0, 0),),
    )

    assert matrix[CATEGORY_ORDER.index("GROUND"), CATEGORY_ORDER.index("OVERHEAD")] == 1
    assert matrix[CATEGORY_ORDER.index("GROUND"), CATEGORY_ORDER.index("GROUND")] == 0
    assert summary == {
        "matched": 1,
        "unmatched_gt": 0,
        "unmatched_pred": 0,
        "both_empty": 0,
        "event_mass": 1.0,
    }


def test_multicategory_match_distributes_one_object_event_without_inflation() -> None:
    matrix = np.zeros((4, 4), dtype=float)

    accumulate_record(
        matrix,
        (obj("opening", "GROUND", "PIT"),),
        (obj("opening", "PIT", "OVERHEAD"),),
        (pair(0, 0),),
    )

    assert matrix.sum() == pytest.approx(1.0)
    assert matrix[CATEGORY_ORDER.index("GROUND"), CATEGORY_ORDER.index("PIT")] == pytest.approx(0.25)
    assert matrix[CATEGORY_ORDER.index("PIT"), CATEGORY_ORDER.index("OVERHEAD")] == pytest.approx(0.25)


def test_safe_boundaries_and_both_empty_record_contributes_safe_safe() -> None:
    matrix = np.zeros((4, 4), dtype=float)

    summary_gt_only = accumulate_record(matrix, (obj("hole", "PIT"),), (), ())
    summary_pred_only = accumulate_record(matrix, (), (obj("beam", "OVERHEAD"),), ())
    summary_both_empty = accumulate_record(matrix, (), (), ())

    assert matrix[CATEGORY_ORDER.index("PIT"), CATEGORY_ORDER.index("SAFE")] == 1
    assert matrix[CATEGORY_ORDER.index("SAFE"), CATEGORY_ORDER.index("OVERHEAD")] == 1
    assert matrix[CATEGORY_ORDER.index("SAFE"), CATEGORY_ORDER.index("SAFE")] == 1
    assert matrix.sum() == pytest.approx(3.0)
    assert summary_both_empty == {
        "matched": 0,
        "unmatched_gt": 0,
        "unmatched_pred": 0,
        "both_empty": 1,
        "event_mass": 1.0,
    }


def test_validate_one_to_one_rejects_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        validate_one_to_one((pair(0, 0), pair(1, 0)), gt_count=2, pred_count=1)


def test_alignment_is_positional_and_checks_image_while_preserving_duplicates() -> None:
    gt = (
        Record(0, "duplicate.jpg", ()),
        Record(1, "duplicate.jpg", (obj("hole", "PIT"),)),
    )
    ours = (
        Record(0, "duplicate.jpg", ()),
        Record(1, "duplicate.jpg", (obj("hole", "GROUND"),)),
    )
    fewshot = (
        Record(0, "duplicate.jpg", ()),
        Record(1, "duplicate.jpg", ()),
    )

    aligned = align_all_records(gt, ours, fewshot, expected_records=2)

    assert len(aligned) == 2
    assert aligned[0][0].record_index == 0
    assert aligned[1][0].record_index == 1

    bad_ours = (ours[0], Record(1, "wrong.jpg", ours[1].objects))
    with pytest.raises(ValueError, match="image"):
        align_all_records(gt, bad_ours, fewshot, expected_records=2)


def test_unique_image_selection_prefers_hazard_gt_over_duplicate_safe() -> None:
    gt = (
        Record(0, "duplicate.jpg", ()),
        Record(1, "duplicate.jpg", (obj("hole", "PIT"),)),
        Record(2, "safe.jpg", ()),
        Record(3, "safe.jpg", ()),
    )
    ours = tuple(Record(row.record_index, row.image, ()) for row in gt)
    fewshot = tuple(Record(row.record_index, row.image, ()) for row in gt)
    aligned = tuple(zip(gt, ours, fewshot, strict=True))

    selected, audit = select_unique_images(aligned)

    assert [rows[0].record_index for rows in selected] == [1, 2]
    assert audit == {
        "input_records": 4,
        "selected_unique_images": 2,
        "removed_duplicate_positions": 2,
        "duplicate_image_groups": 2,
        "hazard_preferred_groups": 1,
        "mixed_safe_hazard_duplicate_groups": 1,
    }


def test_row_normalize_uses_zero_for_empty_rows() -> None:
    matrix = np.array([[2.0, 2.0], [0.0, 0.0]])

    percentages = row_normalize(matrix)

    assert percentages.tolist() == [[50.0, 50.0], [0.0, 0.0]]


def test_plot_matrix_uses_original_fig8_category_order() -> None:
    matrix = np.arange(16, dtype=float).reshape(4, 4)

    reordered = reorder_matrix_for_plot(matrix)

    # Source order: GROUND, PIT, OVERHEAD, SAFE.
    # Plot order: SAFE, OVERHEAD, PIT, GROUND.
    assert reordered.tolist() == matrix[np.ix_([3, 2, 1, 0], [3, 2, 1, 0])].tolist()


def test_loose_run_paths_are_isolated_from_strict_outputs_and_cache() -> None:
    strict = resolve_run_paths("strict")
    loose = resolve_run_paths("loose")

    assert strict.output_root == THIS_DIR
    assert strict.cache_path == THIS_DIR / "deepseek_match_cache.jsonl"
    assert loose.output_root == THIS_DIR / "loose"
    assert loose.cache_path == THIS_DIR / "loose" / "deepseek_match_cache.jsonl"
    assert strict.cache_path != loose.cache_path
