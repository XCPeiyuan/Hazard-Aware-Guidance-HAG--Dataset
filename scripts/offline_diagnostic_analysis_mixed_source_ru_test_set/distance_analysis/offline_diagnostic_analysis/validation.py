from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .deepseek_matcher import build_semantic_match_prompt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_input_manifest(
    sources: Mapping[str, Path], manifest_path: Path
) -> dict[str, Any]:
    manifest = {
        "algorithm": "sha256",
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in sorted(sources.items())
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_input_manifest(
    manifest_path: Path, sources: Mapping[str, Path]
) -> dict[str, Any]:
    try:
        baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "details": {"manifest": f"INVALID: {exc}"}}

    details: dict[str, Any] = {}
    expected_sources = baseline.get("sources", {})
    passed = set(expected_sources) == set(sources)
    for name, path in sorted(sources.items()):
        expected = expected_sources.get(name, {}).get("sha256")
        actual = _sha256(path)
        status = "UNCHANGED" if expected == actual else "CHANGED"
        details[name] = {"status": status, "expected": expected, "actual": actual}
        passed = passed and status == "UNCHANGED"
    return {"passed": passed, "details": details}


def _field(row: Any, name: str) -> Any:
    return row.get(name) if isinstance(row, dict) else getattr(row, name)


def validate_one_to_one(rows: Sequence[Any]) -> dict[str, Any]:
    gt_seen: set[tuple[int, str, int]] = set()
    pred_seen: set[tuple[int, str, int]] = set()
    duplicate_gt: list[tuple[int, str, int]] = []
    duplicate_pred: list[tuple[int, str, int]] = []
    for row in rows:
        if not _field(row, "detected"):
            continue
        gt_key = (
            _field(row, "record_index"),
            _field(row, "method"),
            _field(row, "gt_object_index"),
        )
        pred_key = (
            _field(row, "record_index"),
            _field(row, "method"),
            _field(row, "matched_pred_index"),
        )
        if gt_key in gt_seen:
            duplicate_gt.append(gt_key)
        if pred_key in pred_seen:
            duplicate_pred.append(pred_key)
        gt_seen.add(gt_key)
        pred_seen.add(pred_key)
    return {
        "passed": not duplicate_gt and not duplicate_pred,
        "details": {
            "matched_pairs": len(gt_seen),
            "duplicate_gt_pairs": duplicate_gt,
            "duplicate_pred_pairs": duplicate_pred,
        },
    }


def _validate_cells(value: Any, failures: list[str], path: str = "metrics") -> None:
    if isinstance(value, dict):
        if {"detected", "missed", "total", "recall"} <= set(value):
            detected = value["detected"]
            missed = value["missed"]
            total = value["total"]
            expected = None if total == 0 else detected / total
            if detected + missed != total or value["recall"] != expected:
                failures.append(path)
        for key, child in value.items():
            _validate_cells(child, failures, f"{path}.{key}")


def validate_metrics_recomputable(
    rows: Sequence[Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    _validate_cells(metrics, failures)
    for method in sorted({_field(row, "method") for row in rows}):
        method_rows = [row for row in rows if _field(row, "method") == method]
        detected = sum(bool(_field(row, "detected")) for row in method_rows)
        expected = {
            "detected": detected,
            "missed": len(method_rows) - detected,
            "total": len(method_rows),
            "recall": detected / len(method_rows) if method_rows else None,
        }
        if metrics.get(method, {}).get("overall") != expected:
            failures.append(f"{method}.overall")
    return {"passed": not failures, "details": {"failures": failures}}


def validate_no_match_prompt_leak(profile: str = "strict") -> dict[str, Any]:
    prompt = build_semantic_match_prompt(
        ["alpha"], ["beta"], profile=profile
    ).lower()
    forbidden = [
        token for token in ("category", "distance", "alert", "ssi")
        if token in prompt
    ]
    return {"passed": not forbidden, "details": {"forbidden_tokens": forbidden}}


def validate_zero_total_null(metrics: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []

    def visit(value: Any, path: str = "metrics") -> None:
        if not isinstance(value, dict):
            return
        if value.get("total") == 0 and value.get("recall", "missing") is not None:
            failures.append(path)
        for key, child in value.items():
            visit(child, f"{path}.{key}")

    visit(metrics)
    return {"passed": not failures, "details": {"failures": failures}}


def run_test_suite(root: Path) -> dict[str, Any]:
    basetemp = root / f".pytest_basetemp_validation_{uuid.uuid4().hex}"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests",
        "--basetemp",
        str(basetemp),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return {
        "passed": completed.returncode == 0,
        "details": {
            "command": " ".join(command),
            "exit_code": completed.returncode,
            "output_tail": output[-2000:],
        },
    }
