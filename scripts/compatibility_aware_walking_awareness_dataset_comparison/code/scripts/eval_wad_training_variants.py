#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate matched-sample WAD image-only training variants.

Compares alter-only supervision with auxiliary full-JSON supervision on the
Our non-SAFE and WAD test sets using BERTScore and a Qwen/LAVE-style judge.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parents[1]
LOCAL_DEPS = RESULTS_DIR.parent / "WalkVLM/Results/.deps_bertscore_slim"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

SAFE_TEXT = "The path ahead is safe."
SAFE_GUIDE_TEXT = "Continue walking carefully."

def build_conditions(input_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "dataset": "our_test_non_safe",
            "model": "WAD alter-only",
            "short_model": "alter_only",
            "reference": input_dir / "test/our_test_reference.txt",
            "prediction": input_dir / "outputs/alter_only_on_our_test.txt",
            "reference_schema": "alert_guide",
            "prediction_schema": "sentence",
        },
        {
            "dataset": "our_test_non_safe",
            "model": "WAD full-JSON",
            "short_model": "full_json",
            "reference": input_dir / "test/our_test_reference.txt",
            "prediction": input_dir / "outputs/full_json_alter_on_our_test.txt",
            "reference_schema": "alert_guide",
            "prediction_schema": "sentence",
        },
        {
            "dataset": "wad_test",
            "model": "WAD alter-only",
            "short_model": "alter_only",
            "reference": input_dir / "test/walkvlm_test_reference.txt",
            "prediction": input_dir / "outputs/alter_only_on_wad_test.txt",
            "reference_schema": "sentence",
            "prediction_schema": "sentence",
        },
        {
            "dataset": "wad_test",
            "model": "WAD full-JSON",
            "short_model": "full_json",
            "reference": input_dir / "test/walkvlm_test_reference.txt",
            "prediction": input_dir / "outputs/full_json_alter_on_wad_test.txt",
            "reference_schema": "sentence",
            "prediction_schema": "sentence",
        },
    ]


DEFAULT_INPUT_DIR = RESULTS_DIR
CONDITIONS = build_conditions(DEFAULT_INPUT_DIR)


@dataclass(frozen=True)
class Sample:
    dataset: str
    model: str
    short_model: str
    sample_id: str
    sample_index: int
    image_occurrence: int
    image: str
    reference_raw: str
    prediction_raw: str
    reference_schema: str
    prediction_schema: str


def parse_triplet_txt(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        image = lines[i].strip()
        i += 1
        text = lines[i].strip() if i < len(lines) else ""
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if image:
            items.append((image, text))
    return items


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S | re.I)
    return " ".join(match.group(1).split()) if match else ""


def is_safe(text: str) -> bool:
    return bool(re.fullmatch(r"\s*<SAFE\s*/>\s*", text, flags=re.I))


def structured_parts(text: str) -> tuple[str, str]:
    if is_safe(text):
        return SAFE_TEXT, SAFE_GUIDE_TEXT
    alert = extract_tag(text, "ALERT")
    guide = extract_tag(text, "GUIDE")
    return alert, guide


def normalize_for_bertscore(text: str, schema: str) -> str:
    if schema == "alert_guide":
        alert, guide = structured_parts(text)
        merged = " ".join(part for part in (alert, guide) if part).strip()
        return merged or " ".join(text.split())
    return " ".join(text.split())


def candidate_for_lave(text: str, schema: str) -> dict[str, str]:
    if schema == "alert_guide":
        alert, guide = structured_parts(text)
        full = " ".join(part for part in (alert, guide) if part).strip()
        return {"alert": alert, "guide": guide, "full": full or " ".join(text.split())}
    full = " ".join(text.split())
    return {"alert": full, "guide": full, "full": full}


def load_condition_samples(condition: dict[str, Any]) -> list[Sample]:
    ref_items = parse_triplet_txt(condition["reference"])
    pred_items = parse_triplet_txt(condition["prediction"])

    ref_map: dict[str, str] = {}
    for image, text in ref_items:
        if image in ref_map and ref_map[image] != text:
            raise ValueError(f"Conflicting duplicate reference for {image!r}")
        ref_map[image] = text
    pred_map: dict[str, str] = {}
    for image, text in pred_items:
        if image in pred_map:
            raise ValueError(f"Duplicate prediction key for {image!r}")
        pred_map[image] = text
    if set(ref_map) != set(pred_map):
        raise ValueError(
            f"Image-key mismatch for {condition['dataset']} / {condition['model']}: "
            f"reference_only={len(set(ref_map) - set(pred_map))}, "
            f"prediction_only={len(set(pred_map) - set(ref_map))}"
        )

    samples = []
    for index, (image, pred_text) in enumerate(pred_items):
        samples.append(Sample(
            dataset=condition["dataset"],
            model=condition["model"],
            short_model=condition["short_model"],
            sample_id=f"{condition['dataset']}::{index:06d}",
            sample_index=index,
            image_occurrence=1,
            image=image,
            reference_raw=ref_map[image],
            prediction_raw=pred_text,
            reference_schema=condition["reference_schema"],
            prediction_schema=condition["prediction_schema"],
        ))
    return samples


def preflight(output_dir: Path, include_safe_our_test: bool = False) -> dict[str, Any]:
    rows = []
    for condition in CONDITIONS:
        for key in ("reference", "prediction"):
            path = condition[key]
            if not path.exists():
                raise FileNotFoundError(path)
        samples = load_condition_samples(condition)
        evaluated_samples = samples
        if condition["dataset"] == "our_test_non_safe" and not include_safe_our_test:
            evaluated_samples = [sample for sample in samples if not is_safe(sample.reference_raw)]
        image_counts = Counter(sample.image for sample in samples)
        rows.append({
            "dataset": condition["dataset"],
            "model": condition["model"],
            "reference": str(condition["reference"]),
            "prediction": str(condition["prediction"]),
            "source_n": len(samples),
            "evaluation_n": len(evaluated_samples),
            "unique_image_labels": len(image_counts),
            "duplicate_image_label_extra_records": sum(count - 1 for count in image_counts.values()),
            "alter_extraction_failed_count": sum(sample.prediction_raw == "[ALTER_EXTRACTION_FAILED]" for sample in samples),
            "first_image": samples[0].image if samples else "",
            "last_image": samples[-1].image if samples else "",
        })
    payload = {"conditions": rows, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (output_dir / "preflight.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_all_samples(limit: int | None, include_safe_our_test: bool = False) -> list[Sample]:
    all_samples: list[Sample] = []
    for condition in CONDITIONS:
        samples = load_condition_samples(condition)
        if condition["dataset"] == "our_test_non_safe" and not include_safe_our_test:
            samples = [sample for sample in samples if not is_safe(sample.reference_raw)]
        if limit is not None:
            samples = samples[:limit]
        all_samples.extend(samples)
    return all_samples


def compute_bertscore(samples: list[Sample], output_dir: Path, model_type: str | None) -> dict[tuple[str, str], dict[str, float]]:
    try:
        from bert_score import BERTScorer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "bert_score is not importable. Install it into Results/.deps_bertscore or the active environment."
        ) from exc

    grouped: dict[tuple[str, str], list[Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.dataset, sample.model), []).append(sample)

    scorer_kwargs: dict[str, Any] = {"lang": "en", "rescale_with_baseline": False}
    if model_type:
        scorer_kwargs["model_type"] = model_type
    scorer = BERTScorer(**scorer_kwargs)

    aggregates: dict[tuple[str, str], dict[str, float]] = {}
    detail_rows = []
    for key, group in grouped.items():
        refs = [normalize_for_bertscore(s.reference_raw, s.reference_schema) for s in group]
        preds = [normalize_for_bertscore(s.prediction_raw, s.prediction_schema) for s in group]
        precision, recall, f1 = scorer.score(preds, refs, verbose=True)
        p_vals = [float(x) for x in precision.tolist()]
        r_vals = [float(x) for x in recall.tolist()]
        f_vals = [float(x) for x in f1.tolist()]
        aggregates[key] = {
            "bertscore_precision": statistics.fmean(p_vals),
            "bertscore_recall": statistics.fmean(r_vals),
            "bertscore_f1": statistics.fmean(f_vals),
        }
        for sample, p, r, f in zip(group, p_vals, r_vals, f_vals):
            detail_rows.append({
                "dataset": sample.dataset,
                "model": sample.model,
                "sample_id": sample.sample_id,
                "sample_index": sample.sample_index,
                "image_occurrence": sample.image_occurrence,
                "image": sample.image,
                "reference_is_safe": is_safe(sample.reference_raw) if sample.reference_schema == "alert_guide" else "",
                "bertscore_precision": p,
                "bertscore_recall": r,
                "bertscore_f1": f,
            })

    with (output_dir / "bertscore_details.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    serializable = {f"{dataset}::{model}": metrics for (dataset, model), metrics in aggregates.items()}
    (output_dir / "bertscore_aggregate.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return aggregates


def build_lave_prompt(sample: Sample) -> str:
    ref = candidate_for_lave(sample.reference_raw, sample.reference_schema)
    pred = candidate_for_lave(sample.prediction_raw, sample.prediction_schema)

    if sample.dataset == "our_test_non_safe":
        return f"""
You are an expert evaluator for walking guidance for blind pedestrians.

You will compare a human reference with a model output. The model cannot see the image; judge only the text.
For this dataset, the reference uses a structured schema with a hazard description and a navigation guide.
Some candidate outputs may be a single natural-language sentence rather than separated fields; in that case, use the full sentence as evidence for both hazard and guide scoring.

Return ONE LINE of valid JSON with integer 0-10 scores:
{{"hazard_score": int, "guide_score": int, "format_score": int, "overall_score": int, "comment": str}}

Scoring rules:
- hazard_score: Does the candidate identify the same safety-relevant hazards, including rough direction/distance when present?
- guide_score: Is the suggested walking action safe, specific, and executable for a blind pedestrian?
- format_score: Is the candidate understandable, concise, and free of irrelevant explanations? Do not require ALERT/GUIDE tags.
- overall_score: Overall semantic and safety usefulness, prioritizing hazard and guide correctness.

[REFERENCE_ALERT]
{ref['alert']}

[REFERENCE_GUIDE]
{ref['guide']}

[CANDIDATE_ALERT_EVIDENCE]
{pred['alert']}

[CANDIDATE_GUIDE_EVIDENCE]
{pred['guide']}

[CANDIDATE_FULL_OUTPUT]
{pred['full']}

Respond with ONE LINE of JSON only.
""".strip()

    return f"""
You are an expert evaluator for walking guidance for blind pedestrians.

You will compare a human reference walking reminder with a model output. The model cannot see the image; judge only the text.
This dataset uses single-sentence walking reminders. If the candidate has ALERT/GUIDE structure, evaluate the merged meaning of both fields.

Return ONE LINE of valid JSON with integer 0-10 scores:
{{"hazard_score": int, "guide_score": int, "format_score": int, "overall_score": int, "comment": str}}

Scoring rules:
- hazard_score: Does the candidate cover the main hazard or safe-state described by the reference?
- guide_score: Is the candidate walking advice safe, actionable, and compatible with the reference?
- format_score: Is the candidate understandable, concise, and clean? Do not require any specific tag format.
- overall_score: Overall semantic and safety usefulness, prioritizing hazard and guide correctness.

[REFERENCE_FULL_OUTPUT]
{ref['full']}

[CANDIDATE_FULL_OUTPUT]
{pred['full']}

Respond with ONE LINE of JSON only.
""".strip()


def load_judge(model_name: str, dtype: str, device_map_mode: str):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "auto": "auto",
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported judge dtype: {dtype}")
    if device_map_mode == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("--judge-device-map single requires CUDA.")
        device_map: str | dict[str, str] = {"": "cuda:0"}
    else:
        device_map = "auto"

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype_map[dtype], device_map=device_map)
    model.eval()
    load_seconds = time.perf_counter() - load_started
    print(f"judge load time: {load_seconds:.2f}s")
    print(f"judge input device: {model.device}")
    print(f"judge device map: {getattr(model, 'hf_device_map', None)}")
    print(f"judge dtype: {getattr(model, 'dtype', None)}")
    return tokenizer, model, load_seconds


def call_judge_batch(tokenizer, model, prompts: list[str], max_new_tokens: int) -> tuple[list[str], dict[str, float]]:
    import torch

    rendered = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rendered.append(text)

    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs.input_ids.shape[1]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    generated = outputs[:, input_length:]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    generated_tokens = int((generated != pad_id).sum().item())
    metrics = {
        "batch_size": len(prompts),
        "input_tokens_padded": int(inputs.input_ids.numel()),
        "generated_tokens": generated_tokens,
        "generation_seconds": generation_seconds,
        "tokens_per_second": generated_tokens / generation_seconds if generation_seconds else 0.0,
        "samples_per_second": len(prompts) / generation_seconds if generation_seconds else 0.0,
    }
    return tokenizer.batch_decode(generated, skip_special_tokens=True), metrics


def parse_judge_json(raw: str) -> dict[str, Any] | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except Exception:
        return None
    required = ["hazard_score", "guide_score", "format_score", "overall_score", "comment"]
    if not all(k in data for k in required):
        return None
    for key in required[:-1]:
        try:
            value = int(data[key])
        except Exception:
            return None
        if not 0 <= value <= 10:
            return None
        data[key] = value
    data["comment"] = str(data["comment"])
    return data


def run_lave(
    samples: list[Sample],
    output_dir: Path,
    model_name: str,
    max_new_tokens: int,
    max_retry: int,
    judge_dtype: str,
    judge_device_map: str,
    batch_size: int,
) -> dict[tuple[str, str], dict[str, float]]:
    from tqdm import tqdm

    if batch_size < 1:
        raise ValueError("LAVE batch size must be at least 1.")
    tokenizer, model, load_seconds = load_judge(model_name, judge_dtype, judge_device_map)
    jsonl_path = output_dir / "lave_scores.jsonl"
    grouped_scores: dict[tuple[str, str], list[dict[str, Any]]] = {}
    failed: dict[tuple[str, str], int] = {}
    performance_batches: list[dict[str, Any]] = []

    with jsonl_path.open("w", encoding="utf-8") as f:
        progress = tqdm(total=len(samples), desc="LAVE", ncols=100)
        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start:batch_start + batch_size]
            prompts = [build_lave_prompt(sample) for sample in batch]
            raws, batch_metrics = call_judge_batch(tokenizer, model, prompts, max_new_tokens=max_new_tokens)
            batch_metrics.update({"batch_start": batch_start, "retry": 0})
            performance_batches.append(batch_metrics)
            print(
                f"batch {batch_start // batch_size + 1}: generation={batch_metrics['generation_seconds']:.2f}s, "
                f"samples={len(batch)}, tokens={batch_metrics['generated_tokens']}, "
                f"throughput={batch_metrics['tokens_per_second']:.2f} tok/s"
            )
            parsed_batch = [parse_judge_json(raw) for raw in raws]

            # Retry only invalid outputs; valid batch members are never regenerated.
            for _ in range(1, max_retry):
                failed_indices = [i for i, parsed in enumerate(parsed_batch) if parsed is None]
                if not failed_indices:
                    break
                retry_raws, retry_metrics = call_judge_batch(
                    tokenizer,
                    model,
                    [prompts[i] for i in failed_indices],
                    max_new_tokens=max_new_tokens * 2,
                )
                retry_metrics.update({"batch_start": batch_start, "retry": 1})
                performance_batches.append(retry_metrics)
                for i, raw in zip(failed_indices, retry_raws):
                    raws[i] = raw
                    parsed_batch[i] = parse_judge_json(raw)

            for sample, raw, parsed in zip(batch, raws, parsed_batch):
                key = (sample.dataset, sample.model)
                if parsed is None:
                    failed[key] = failed.get(key, 0) + 1
                    row = {
                        "dataset": sample.dataset,
                        "model": sample.model,
                        "sample_id": sample.sample_id,
                        "sample_index": sample.sample_index,
                        "image_occurrence": sample.image_occurrence,
                        "image": sample.image,
                        "reference_is_safe": is_safe(sample.reference_raw) if sample.reference_schema == "alert_guide" else None,
                        "ok": False,
                        "raw": raw,
                    }
                else:
                    grouped_scores.setdefault(key, []).append(parsed)
                    row = {
                        "dataset": sample.dataset,
                        "model": sample.model,
                        "sample_id": sample.sample_id,
                        "sample_index": sample.sample_index,
                        "image_occurrence": sample.image_occurrence,
                        "image": sample.image,
                        "reference_is_safe": is_safe(sample.reference_raw) if sample.reference_schema == "alert_guide" else None,
                        "ok": True,
                        "scores": parsed,
                        "raw": raw,
                    }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            progress.update(len(batch))
        progress.close()

    total_generation_seconds = sum(row["generation_seconds"] for row in performance_batches)
    total_generated_tokens = sum(row["generated_tokens"] for row in performance_batches)
    performance = {
        "judge_load_seconds": load_seconds,
        "total_generation_seconds": total_generation_seconds,
        "total_generated_tokens": total_generated_tokens,
        "overall_tokens_per_second": total_generated_tokens / total_generation_seconds if total_generation_seconds else 0.0,
        "evaluated_samples": len(samples),
        "generation_seconds_per_sample": total_generation_seconds / len(samples) if samples else 0.0,
        "batches": performance_batches,
    }
    (output_dir / "lave_performance.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in performance.items() if key != "batches"}, indent=2))

    aggregates: dict[tuple[str, str], dict[str, float]] = {}
    for condition in CONDITIONS:
        key = (condition["dataset"], condition["model"])
        scores = grouped_scores.get(key, [])
        if scores:
            aggregates[key] = {
                "lave_hazard": statistics.fmean(s["hazard_score"] for s in scores),
                "lave_guide": statistics.fmean(s["guide_score"] for s in scores),
                "lave_format": statistics.fmean(s["format_score"] for s in scores),
                "lave_overall": statistics.fmean(s["overall_score"] for s in scores),
                "failed_lave_parse_count": float(failed.get(key, 0)),
            }
        else:
            aggregates[key] = {
                "lave_hazard": float("nan"),
                "lave_guide": float("nan"),
                "lave_format": float("nan"),
                "lave_overall": float("nan"),
                "failed_lave_parse_count": float(failed.get(key, 0)),
            }
    serializable = {f"{dataset}::{model}": metrics for (dataset, model), metrics in aggregates.items()}
    (output_dir / "lave_aggregate.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return aggregates


def write_aggregate_tables(output_dir: Path, samples: list[Sample], bert: dict[tuple[str, str], dict[str, float]], lave: dict[tuple[str, str], dict[str, float]] | None) -> list[dict[str, Any]]:
    rows = []
    count_by_key: dict[tuple[str, str], int] = {}
    for sample in samples:
        count_by_key[(sample.dataset, sample.model)] = count_by_key.get((sample.dataset, sample.model), 0) + 1
    for condition in CONDITIONS:
        key = (condition["dataset"], condition["model"])
        row = {
            "dataset": key[0],
            "model": key[1],
            "n": count_by_key.get(key, 0),
            "bertscore_precision": bert.get(key, {}).get("bertscore_precision", ""),
            "bertscore_recall": bert.get(key, {}).get("bertscore_recall", ""),
            "bertscore_f1": bert.get(key, {}).get("bertscore_f1", ""),
            "lave_hazard": "" if lave is None else lave.get(key, {}).get("lave_hazard", ""),
            "lave_guide": "" if lave is None else lave.get(key, {}).get("lave_guide", ""),
            "lave_format": "" if lave is None else lave.get(key, {}).get("lave_format", ""),
            "lave_overall": "" if lave is None else lave.get(key, {}).get("lave_overall", ""),
            "failed_lave_parse_count": "" if lave is None else lave.get(key, {}).get("failed_lave_parse_count", ""),
        }
        rows.append(row)

    with (output_dir / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def generate_report(output_dir: Path, rows: list[dict[str, Any]], limit: int | None, skip_lave: bool) -> None:
    row_map = {(r["dataset"], r["model"]): r for r in rows}
    ru_our = row_map.get(("our_test_non_safe", "WAD alter-only"), {})
    wad_our = row_map.get(("our_test_non_safe", "WAD full-JSON"), {})
    ru_wad = row_map.get(("wad_test", "WAD alter-only"), {})
    wad_wad = row_map.get(("wad_test", "WAD full-JSON"), {})

    latex = rf"""
\begin{{table}}[t]
\centering
\small
\caption{{Compatibility-aware comparison with WalkVLM/WAD. Both models use the same Qwen2.5-VL backbone and are evaluated under an image-only inference protocol.}}
\label{{tab:walkvlm_comparison}}
\begin{{tabular}}{{lcccc}}
\toprule
Training data & Our test BERTScore-F1 & Our test LLM judge & WAD test BERTScore-F1 & WAD test LLM judge \\
\midrule
WAD alter-only & {fmt(ru_our.get('bertscore_f1'))} & {fmt(ru_our.get('lave_overall'))} & {fmt(ru_wad.get('bertscore_f1'))} & {fmt(ru_wad.get('lave_overall'))} \\
WAD full-JSON & {fmt(wad_our.get('bertscore_f1'))} & {fmt(wad_our.get('lave_overall'))} & {fmt(wad_wad.get('bertscore_f1'))} & {fmt(wad_wad.get('lave_overall'))} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()

    chinese = f"""
# WAD 图像输入训练变体评估结果概要

本实验比较两个使用相同 1,000 个 WAD 样本训练的 Qwen2.5-VL 模型：主实验 alter-only supervision，以及辅助 full-JSON multitask supervision。两者均采用 image-only 输入，并分别在本文非-SAFE测试子集和 WAD 测试集上评估。

由于 WalkVLM/WAD 的原始任务是基于图像与 scene summary 的单句 walking reminder 生成，而本文任务是直接由图像生成结构化 ALERT/GUIDE，二者的数据 schema、训练输入和输出目标均不完全一致。因此该实验应作为面向审稿意见的兼容性诊断比较，而不是完全公平的 apples-to-apples benchmark。

{'当前为 smoke/subset 结果，不应直接写入论文主表。' if limit else '当前结果默认排除 Our test 中的 SAFE references，仅评估 254 条非-SAFE 样本；WAD test 使用全部 1,007 条样本。'}

输出文件：

- `aggregate_metrics.csv`: 汇总指标
- `aggregate_metrics.json`: 汇总指标 JSON
- `bertscore_details.csv`: 逐样本 BERTScore
- `lave_scores.jsonl`: 逐样本 LAVE-style Qwen3-14B judge 输出
- `bertscore_aggregate.json`: BERTScore 汇总
- `lave_aggregate.json`: LAVE 汇总{'（本次跳过 LAVE 时不存在）' if skip_lave else ''}
""".strip()

    paragraph = """
We compared two image-only Qwen2.5-VL variants trained on the same 1,000 WAD samples: direct alter-only supervision and auxiliary full-JSON multitask supervision. Both models were evaluated on the 254 non-SAFE samples from our test set and all 1,007 WAD test samples under an image-only inference protocol. We did not provide WAD scene summaries at test time, because such summaries are not available in our target deployment setting and would constitute privileged information. Since WAD uses summary-conditioned single-sentence walking reminders whereas our method directly predicts structured ALERT/GUIDE outputs, we report semantic BERTScore and a LAVE-style Qwen judge score rather than n-gram overlap metrics. These results are therefore intended as a diagnostic cross-dataset comparison rather than a fully matched benchmark; they show how the two fine-tuning datasets transfer across different annotation schemas and task formulations.
""".strip()

    report = f"""{chinese}

## Aggregate Metrics

| Dataset | Model | n | BERTScore-F1 | LAVE Hazard | LAVE Guide | LAVE Format | LAVE Overall | Failed LAVE Parses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['dataset']} | {row['model']} | {row['n']} | {fmt(row['bertscore_f1'])} | "
            f"{fmt(row['lave_hazard'])} | {fmt(row['lave_guide'])} | {fmt(row['lave_format'])} | "
            f"{fmt(row['lave_overall'])} | {fmt(row['failed_lave_parse_count'], 0)} |\n"
        )
    report += f"""

## English LaTeX Table

```latex
{latex}
```

## Paper-Ready Paragraph

{paragraph}

## Caveat Statement

N-gram metrics such as BLEU/ROUGE are not used as primary evidence because schema-shifted safety guidance allows multiple valid phrasings and requires semantic/actionability evaluation. The comparison should be interpreted with the WAD summary-conditioned training objective and the ALERT/GUIDE schema mismatch in mind.
"""
    (output_dir / "WAD_Training_Variants_Report.md").write_text(report, encoding="utf-8")


def main() -> int:
    global CONDITIONS
    parser = argparse.ArgumentParser(description="Evaluate matched-sample WAD training variants.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per condition for smoke tests.")
    parser.add_argument("--include-safe-our-test", action="store_true", help="Include SAFE references on our test; excluded by default for the hazard-guidance comparison.")
    parser.add_argument("--skip-lave", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retry", type=int, default=2)
    parser.add_argument("--judge-dtype", default="auto", choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--judge-device-map", default="single", choices=["single", "auto"], help="Use single to avoid slow cross-GPU token generation on systems without NVLink.")
    parser.add_argument("--lave-batch-size", type=int, default=4)
    args = parser.parse_args()
    CONDITIONS = build_conditions(args.input_dir.resolve())

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")

    preflight_payload = preflight(output_dir, args.include_safe_our_test)
    print(json.dumps(preflight_payload, ensure_ascii=False, indent=2))

    samples = load_all_samples(args.limit, args.include_safe_our_test)
    if not samples:
        raise RuntimeError("No samples loaded.")

    bert: dict[tuple[str, str], dict[str, float]] = {}
    if not args.skip_bertscore:
        bert = compute_bertscore(samples, output_dir, args.bertscore_model)

    lave = None
    if not args.skip_lave:
        lave = run_lave(
            samples,
            output_dir,
            args.judge_model,
            args.max_new_tokens,
            args.max_retry,
            args.judge_dtype,
            args.judge_device_map,
            args.lave_batch_size,
        )

    rows = write_aggregate_tables(output_dir, samples, bert, lave)
    generate_report(output_dir, rows, args.limit, args.skip_lave)
    print(f"Wrote report: {output_dir / 'WAD_Training_Variants_Report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
