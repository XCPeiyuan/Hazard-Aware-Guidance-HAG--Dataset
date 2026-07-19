from pathlib import Path

import pytest

import run_analysis


def test_resolve_output_dir_preserves_strict_default_and_isolates_loose() -> None:
    assert run_analysis.resolve_output_dir("strict", None) == run_analysis.THIS_DIR
    assert run_analysis.resolve_output_dir("loose", None) == (
        run_analysis.THIS_DIR.parent / "distance_analysis_loose"
    )


def test_resolve_output_dir_accepts_explicit_path(tmp_path: Path) -> None:
    assert run_analysis.resolve_output_dir("loose", tmp_path) == tmp_path.resolve()


def test_unknown_match_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="match_mode"):
        run_analysis.resolve_output_dir("unknown", None)


def test_profile_matcher_forwards_selected_profile(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_matcher(*args, **kwargs):
        captured.update(kwargs)
        return "decision"

    monkeypatch.setattr(run_analysis.deepseek_matcher, "match_with_deepseek", fake_matcher)

    matcher = run_analysis.build_profile_matcher("loose")
    result = matcher(
        gt_names=["branch"],
        pred_names=["tree limb"],
        config=object(),
        cache_path=tmp_path / "cache.jsonl",
    )

    assert result == "decision"
    assert captured["profile"] == "loose"
