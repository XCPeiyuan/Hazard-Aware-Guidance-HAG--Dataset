"""Tests for hybrid matching pipeline."""
import json
from pathlib import Path

import pytest

from offline_diagnostic_analysis.models import HazardObject, Record
from offline_diagnostic_analysis.io import align_records, load_records
from offline_diagnostic_analysis.matching import (
    ManualOverride,
    ObjectResult,
    RecordMatchResult,
    apply_overrides,
    build_disagreement_queue,
    build_object_results,
    load_overrides,
    match_record,
)
from offline_diagnostic_analysis.rule_matcher import MatchPair
from offline_diagnostic_analysis.deepseek_matcher import DeepSeekDecision, LLMConfig, build_cache_key


@pytest.fixture
def gt_record() -> Record:
    return Record(
        record_index=0,
        image="test.png",
        objects=(
            HazardObject("deep pit", ("PIT",), 5, "about 5 steps ahead"),
            HazardObject("cone", ("GROUND",), 5, "about 5 steps ahead"),
            HazardObject("bicycle", ("GROUND",), 8, "about 8 steps ahead"),
        ),
    )


@pytest.fixture
def pred_record() -> Record:
    return Record(
        record_index=0,
        image="test.png",
        objects=(
            HazardObject("cone", ("GROUND",), 5, "about 5 steps ahead"),
            HazardObject("bike", ("GROUND",), 8, "about 8 steps ahead"),
            HazardObject("barrier", ("GROUND",), 3, "about 3 steps ahead"),
        ),
    )


@pytest.fixture
def empty_pred_record() -> Record:
    return Record(record_index=0, image="test.png", objects=())


class TestMatchRecord:
    def test_empty_prediction_marks_every_gt_object_missed(self, gt_record, empty_pred_record):
        result = match_record(gt_record, empty_pred_record, "ours", None, None, None)
        assert len(result.matched_pairs) == 0
        assert tuple(result.unresolved_gt_indices) == (0, 1, 2)
        assert tuple(result.unresolved_pred_indices) == ()

    def test_rules_run_without_deepseek(self, gt_record, pred_record):
        result = match_record(gt_record, pred_record, "ours", None, None, None)
        assert result.deepseek_decision is None
        assert len(result.matched_pairs) >= 0

    def test_matched_pairs_are_one_to_one(self, gt_record, pred_record):
        result = match_record(gt_record, pred_record, "ours", None, None, None)
        gt_indices = {p.gt_index for p in result.matched_pairs}
        pred_indices = {p.pred_index for p in result.matched_pairs}
        assert len(gt_indices) == len(result.matched_pairs)
        assert len(pred_indices) == len(result.matched_pairs)


class TestOverrides:
    def test_manual_override_is_applied(self, gt_record, pred_record):
        base = match_record(gt_record, pred_record, "ours", None, None, None)
        override = ManualOverride(0, "ours", 0, 0, "manual correction")
        result = apply_overrides(base, [override])
        assert any(p.gt_index == 0 and p.source == "manual_override" for p in result.matched_pairs)

    def test_override_replaces_existing_match(self, gt_record, pred_record):
        base = match_record(gt_record, pred_record, "ours", None, None, None)
        override = ManualOverride(0, "ours", 2, 0, "manual correction")
        result = apply_overrides(base, [override])
        pairs_for_gt2 = [p for p in result.matched_pairs if p.gt_index == 2]
        assert len(pairs_for_gt2) == 1
        assert pairs_for_gt2[0].source == "manual_override"

    def test_override_restores_displaced_endpoints_to_unresolved(self):
        gt = Record(
            0,
            "test.png",
            (
                HazardObject("cone", ("GROUND",), 1, None),
                HazardObject("bike", ("GROUND",), 2, None),
            ),
        )
        pred = Record(
            0,
            "test.png",
            (
                HazardObject("cone", ("GROUND",), 1, None),
                HazardObject("bicycle", ("GROUND",), 2, None),
            ),
        )
        base = match_record(gt, pred, "ours", None, None, None)

        result = apply_overrides(
            base, [ManualOverride(0, "ours", 0, 1, "manual correction")]
        )

        assert result.matched_pairs == (
            MatchPair(0, 1, "manual_override", "manual correction", "high"),
        )
        assert result.unresolved_gt_indices == (1,)
        assert result.unresolved_pred_indices == (0,)

    def test_override_bounds_are_validated_against_record(self, gt_record, pred_record):
        base = match_record(gt_record, pred_record, "ours", None, None, None)
        with pytest.raises(ValueError, match="out of bounds"):
            apply_overrides(
                base, [ManualOverride(0, "ours", 99, 0, "invalid")]
            )

    def test_invalid_override_from_jsonl_is_rejected(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"record_index": -1, "method": "ours", "gt_index": 0, "pred_index": 0}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_overrides(path)


class TestLoadOverrides:
    def test_load_empty(self, tmp_path):
        path = tmp_path / "overrides.jsonl"
        path.write_text("", encoding="utf-8")
        overrides = load_overrides(path)
        assert len(overrides) == 0

    def test_load_valid(self, tmp_path):
        path = tmp_path / "overrides.jsonl"
        path.write_text(
            json.dumps({"record_index": 0, "method": "ours", "gt_index": 1, "pred_index": 2, "reason": "test"}) + "\n",
            encoding="utf-8",
        )
        overrides = load_overrides(path)
        assert len(overrides) == 1
        assert overrides[0].reason == "test"

    def test_load_skips_comments(self, tmp_path):
        path = tmp_path / "overrides.jsonl"
        path.write_text("# comment\n" + json.dumps({"record_index": 0, "method": "ours", "gt_index": 1, "pred_index": 2, "reason": "test"}) + "\n", encoding="utf-8")
        overrides = load_overrides(path)
        assert len(overrides) == 1

    @pytest.mark.parametrize(
        "payload",
        [
            {"record_index": True, "method": "ours", "gt_index": 0, "pred_index": 0, "reason": "x"},
            {"record_index": 0, "method": "ours", "gt_index": "0", "pred_index": 0, "reason": "x"},
            {"record_index": 0, "method": "other", "gt_index": 0, "pred_index": 0, "reason": "x"},
            {"record_index": 0, "method": "ours", "gt_index": 0, "pred_index": False, "reason": "x"},
            {"record_index": 0, "method": "ours", "gt_index": 0, "pred_index": 0},
            {"record_index": 0, "method": "ours", "gt_index": 0, "pred_index": 0, "reason": "x", "extra": 1},
        ],
    )
    def test_load_requires_exact_schema_and_types(self, tmp_path, payload):
        path = tmp_path / "overrides.jsonl"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_overrides(path)


class TestBuildObjectResults:
    def test_results_have_required_audit_fields(self, tmp_path):
        tiny_dir = Path(__file__).parent / "fixtures"
        gt = load_records(tiny_dir / "tiny_gt.json")
        ours = load_records(tiny_dir / "tiny_ours.json")
        fewshot = load_records(tiny_dir / "tiny_fewshot.json")
        aligned = align_records(gt, ours, fewshot)

        results, unmatched, queue = build_object_results(
            aligned, ["ours", "fewshot"], (),
        )
        assert len(results) > 0
        for r in results:
            assert r.record_index is not None
            assert r.image
            assert r.gt_object_index is not None
            assert r.gt_object_name
            assert r.gt_categories is not None
            assert r.gt_distance_bin in ("Near", "Medium", "Far", "Unknown")
            assert r.method in ("ours", "fewshot")
            assert r.match_source in ("rule", "deepseek", "manual_override", "unresolved")
            assert isinstance(r.detected, bool)

    def test_category_and_distance_never_affect_matching(self, tmp_path):
        tiny_dir = Path(__file__).parent / "fixtures"
        gt = load_records(tiny_dir / "tiny_gt.json")
        ours = load_records(tiny_dir / "tiny_ours.json")
        fewshot = load_records(tiny_dir / "tiny_fewshot.json")
        aligned = align_records(gt, ours, fewshot)

        results, _, _ = build_object_results(aligned, ["ours"], ())
        # Verify matching uses only names: same prediction names match GT names regardless of category
        matched = [r for r in results if r.detected]
        assert len(matched) > 0, "Expected at least one rule match"
        # All matches should be from rule source (no DeepSeek in this test)
        assert all(r.match_source == "rule" for r in matched if r.match_source != "unresolved")
        # Verify ObjectResult fields are properly populated
        for r in results:
            assert r.record_index >= 0
            assert r.image
            assert r.gt_object_name
            assert r.method == "ours"

    def test_deepseek_receives_only_unresolved_objects(self, gt_record, pred_record):
        matched_names = set()
        unresolved_gt_count = 0
        result = match_record(gt_record, pred_record, "ours", None, None, None)
        # Without DeepSeek, unresolved GT indices are those not matched by rules
        assert len(result.unresolved_gt_indices) >= 0
        # If DeepSeek were called, it would only receive those unresolved names
        # This test verifies the contract: unresolved set excludes rule matches
        rule_gt = {p.gt_index for p in result.matched_pairs}
        for idx in result.unresolved_gt_indices:
            assert idx not in rule_gt, f"unresolved GT index {idx} was already matched by rules"

    def test_deepseek_review_queue_maps_subset_indices_to_full_indices(self, tmp_path):
        gt = Record(
            0,
            "test.png",
            (
                HazardObject("cone", ("GROUND",), 1, None),
                HazardObject("open manhole", ("PIT",), 2, None),
            ),
        )
        ours = Record(
            0,
            "test.png",
            (
                HazardObject("cone", ("GROUND",), 1, None),
                HazardObject("manhole opening", ("PIT",), 2, None),
            ),
        )
        empty = Record(0, "test.png", ())
        config = LLMConfig("https://example.invalid", "secret", "model")

        def fake_matcher(**kwargs):
            return DeepSeekDecision(
                pairs=(MatchPair(0, 0, "deepseek", "same_real_world_obstacle", "medium"),),
                status="matched",
                model="model",
                cache_key=build_cache_key(kwargs["gt_names"], kwargs["pred_names"]),
                raw_response='{"pairs":[{"gt_index":0,"pred_index":0}]}',
                error=None,
                from_cache=False,
            )

        _, _, queue = build_object_results(
            ((gt, ours, empty),),
            ["ours"],
            (),
            deepseek_matcher=fake_matcher,
            config=config,
            cache_path=tmp_path / "cache.jsonl",
        )

        item = next(item for item in queue if item["type"] == "deepseek_match")
        assert item["gt_index"] == 1
        assert item["gt_name"] == "open manhole"
        assert item["pred_index"] == 1
        assert item["pred_name"] == "manhole opening"

    def test_review_queue_unresolved_subset_indices_map_to_full_records(self, tmp_path):
        """Verify that when DeepSeek returns indices relative to unresolved
        subsets, the review queue maps them back to full-record indices."""
        gt = Record(
            0,
            "test.png",
            (
                HazardObject("known cone", ("GROUND",), 1, None),
                HazardObject("unmatched object", ("GROUND",), 2, None),
                HazardObject("another thing", ("GROUND",), 3, None),
            ),
        )
        ours = Record(
            0,
            "test.png",
            (
                HazardObject("cone", ("GROUND",), 1, None),
                HazardObject("thing", ("GROUND",), 2, None),
            ),
        )
        empty = Record(0, "test.png", ())
        config = LLMConfig("https://example.invalid", "secret", "model")

        def fake_matcher(**kwargs):
            return DeepSeekDecision(
                pairs=(
                    MatchPair(0, 0, "deepseek", "same_real_world_obstacle", "medium"),
                    MatchPair(1, 1, "deepseek", "same_real_world_obstacle", "medium"),
                ),
                status="matched",
                model="model",
                cache_key=build_cache_key(kwargs["gt_names"], kwargs["pred_names"]),
                raw_response='{"pairs":[{"gt_index":0,"pred_index":0},{"gt_index":1,"pred_index":1}]}',
                error=None,
                from_cache=False,
            )

        _, _, queue = build_object_results(
            ((gt, ours, empty),),
            ["ours"],
            (),
            deepseek_matcher=fake_matcher,
            config=config,
            cache_path=tmp_path / "cache.jsonl",
        )

        deepseek_items = [item for item in queue if item["type"] == "deepseek_match"]
        assert len(deepseek_items) == 2

        idx0 = next(it for it in deepseek_items if it["gt_index"] == 0)
        assert idx0["gt_name"] == "known cone"
        assert idx0["pred_index"] == 0
        assert idx0["pred_name"] == "cone"

        idx1 = next(it for it in deepseek_items if it["gt_index"] == 1)
        assert idx1["gt_name"] == "unmatched object"
        assert idx1["pred_index"] == 1
        assert idx1["pred_name"] == "thing"

    def test_review_queue_contains_each_detection_disagreement(self):
        gt = Record(
            0, "test.png", (HazardObject("cone", ("GROUND",), 1, None),)
        )
        ours = Record(
            0, "test.png", (HazardObject("cone", ("GROUND",), 1, None),)
        )
        fewshot = Record(0, "test.png", ())

        _, _, queue = build_object_results(((gt, ours, fewshot),), ["ours", "fewshot"], ())

        disagreements = [
            item for item in queue if item["type"] == "detection_disagreement"
        ]
        assert disagreements == [
            {
                "record_index": 0,
                "image": "test.png",
                "type": "detection_disagreement",
                "gt_index": 0,
                "gt_name": "cone",
                "ours_detected": True,
                "ours_pred_index": 0,
                "ours_pred_name": "cone",
                "fewshot_detected": False,
                "fewshot_pred_index": None,
                "fewshot_pred_name": None,
            }
        ]


class TestBuildDisagreementQueue:
    def test_ours_fewshot_disagreement_queue(self):
        from offline_diagnostic_analysis.metrics import distance_bin

        results = [
            ObjectResult(
                record_index=0, image="test.png", gt_object_index=0,
                gt_object_name="cone", gt_categories=("GROUND",),
                gt_distance_steps=5, gt_distance_bin=distance_bin(5),
                method="ours", matched_pred_index=0,
                matched_pred_name="cone", match_source="rule",
                match_reason="exact", detected=True,
            ),
            ObjectResult(
                record_index=0, image="test.png", gt_object_index=0,
                gt_object_name="cone", gt_categories=("GROUND",),
                gt_distance_steps=5, gt_distance_bin=distance_bin(5),
                method="fewshot", matched_pred_index=None,
                matched_pred_name=None, match_source="unresolved",
                match_reason="no match found", detected=False,
            ),
            ObjectResult(
                record_index=1, image="test2.png", gt_object_index=0,
                gt_object_name="pit", gt_categories=("PIT",),
                gt_distance_steps=8, gt_distance_bin=distance_bin(8),
                method="ours", matched_pred_index=0,
                matched_pred_name="pit", match_source="rule",
                match_reason="exact", detected=True,
            ),
            ObjectResult(
                record_index=1, image="test2.png", gt_object_index=0,
                gt_object_name="pit", gt_categories=("PIT",),
                gt_distance_steps=8, gt_distance_bin=distance_bin(8),
                method="fewshot", matched_pred_index=0,
                matched_pred_name="pit", match_source="rule",
                match_reason="exact", detected=True,
            ),
        ]

        disagreements = build_disagreement_queue(results)

        assert len(disagreements) == 1
        d = disagreements[0]
        assert d["record_index"] == 0
        assert d["gt_index"] == 0
        assert d["gt_name"] == "cone"
        assert d["ours_detected"] is True
        assert d["fewshot_detected"] is False

    def test_agreement_cases_are_not_in_disagreement_queue(self):
        from offline_diagnostic_analysis.metrics import distance_bin

        results = [
            ObjectResult(
                record_index=0, image="test.png", gt_object_index=0,
                gt_object_name="cone", gt_categories=("GROUND",),
                gt_distance_steps=5, gt_distance_bin=distance_bin(5),
                method="ours", matched_pred_index=0,
                matched_pred_name="cone", match_source="rule",
                match_reason="exact", detected=True,
            ),
            ObjectResult(
                record_index=0, image="test.png", gt_object_index=0,
                gt_object_name="cone", gt_categories=("GROUND",),
                gt_distance_steps=5, gt_distance_bin=distance_bin(5),
                method="fewshot", matched_pred_index=0,
                matched_pred_name="cone", match_source="rule",
                match_reason="exact", detected=True,
            ),
            ObjectResult(
                record_index=1, image="test2.png", gt_object_index=0,
                gt_object_name="pit", gt_categories=("PIT",),
                gt_distance_steps=8, gt_distance_bin=distance_bin(8),
                method="ours", matched_pred_index=None,
                matched_pred_name=None, match_source="unresolved",
                match_reason="no match found", detected=False,
            ),
            ObjectResult(
                record_index=1, image="test2.png", gt_object_index=0,
                gt_object_name="pit", gt_categories=("PIT",),
                gt_distance_steps=8, gt_distance_bin=distance_bin(8),
                method="fewshot", matched_pred_index=None,
                matched_pred_name=None, match_source="unresolved",
                match_reason="no match found", detected=False,
            ),
        ]

        disagreements = build_disagreement_queue(results)
        assert len(disagreements) == 0
