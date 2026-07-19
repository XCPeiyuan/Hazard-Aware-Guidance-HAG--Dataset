#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-file workstation benchmark script for Inference Efficiency Analysis.

Design constraints:
1. Read default paths and prompt from path.txt next to this script.
2. Follow the inference style shown in inference_txt_based.py only.
3. Use images_root + image_name at runtime.
4. Run safe and hazard groups sequentially, finishing one group before the next.
5. Print detailed progress to the terminal and save machine-readable summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
PATH_TXT = SCRIPT_DIR / "path.txt"
SAFE_TEXTS = {"</safe>", "<safe/>", "<safe />", "safe"}


def parse_path_txt(path: Path) -> Dict[str, str]:
    key_map = {
        "模型路径": "model_path",
        "json路径": "dataset_json",
        "图片路径": "images_root",
        "prompt": "prompt",
        "输出路径": "output_path",
    }
    config: Dict[str, str] = {}
    if not path.exists():
        return config

    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip().strip('"')
        if not line:
            continue
        if "：" in line:
            raw_key, raw_value = line.split("：", 1)
        elif ":" in line:
            raw_key, raw_value = line.split(":", 1)
        else:
            continue

        key = raw_key.strip()
        value = raw_value.strip().strip('"')
        mapped = key_map.get(key)
        if mapped and value:
            config[mapped] = value
    return config


def parse_args() -> argparse.Namespace:
    defaults = parse_path_txt(PATH_TXT)
    default_output_root = defaults.get("output_path", "python文件同目录")
    if default_output_root in {"python文件同目录", "Python文件同目录"}:
        default_output_dir = SCRIPT_DIR / "server_latency_benchmark"
    else:
        default_output_dir = Path(default_output_root) / "server_latency_benchmark"

    parser = argparse.ArgumentParser(description="Run Inference Efficiency Analysis workstation latency benchmark from a single script.")
    parser.add_argument("--model-path", default=defaults.get("model_path"), help="Model path.")
    parser.add_argument("--dataset-json", default=defaults.get("dataset_json"), help="Dataset JSON path.")
    parser.add_argument("--images-root", default=defaults.get("images_root"), help="Image root directory.")
    parser.add_argument("--prompt-text", default=defaults.get("prompt"), help="Prompt text.")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="Output directory.")
    parser.add_argument("--safe-count", type=int, default=100, help="Number of safe samples.")
    parser.add_argument("--hazard-count", type=int, default=100, help="Number of hazard samples.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation max_new_tokens.")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Warmup runs before measurement.")
    parser.add_argument("--repeats", type=int, default=1, help="Formal repeats per image.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Use only local files.")
    parser.add_argument("--no-local-files-only", dest="local_files_only", action="store_false", help="Allow remote lookup.")
    parser.add_argument("--group-order", nargs="+", default=["safe", "hazard"], help="Execution order.")
    parser.add_argument("--skip-missing-images", action="store_true", default=True, help="Skip missing images.")
    parser.add_argument("--fail-on-missing-images", dest="skip_missing_images", action="store_false", help="Stop if an image is missing.")
    parser.add_argument("--model-label", default=None, help="Optional label for paper tables.")
    parser.add_argument("--precision-label", default="auto", help="Precision label for paper tables.")
    parser.add_argument("--quantization-label", default="none", help="Quantization label for paper tables.")
    parser.add_argument("--input-setting-label", default=None, help="Optional input setting label for paper tables.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.model_path:
        missing.append("model_path")
    if not args.dataset_json:
        missing.append("dataset_json")
    if not args.images_root:
        missing.append("images_root")
    if not args.prompt_text:
        missing.append("prompt")
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}. Fill them in path.txt or pass CLI args.")


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Dataset JSON must be a list.")
    return data


def normalize_safe_text(value: str) -> str:
    return value.strip().lower()


def extract_assistant_text(record: Dict[str, Any]) -> Optional[str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).strip().lower() != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def extract_image_name(record: Dict[str, Any]) -> Optional[str]:
    images = record.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str) and first.strip():
            return Path(first.strip()).name
    return None


def build_samples(dataset_json: Path, images_root: Path, safe_count: int, hazard_count: int, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    records = load_json_list(dataset_json)

    safe_samples: List[Dict[str, Any]] = []
    hazard_samples: List[Dict[str, Any]] = []
    skipped = 0

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            skipped += 1
            continue

        assistant_text = extract_assistant_text(record)
        image_name = extract_image_name(record)
        if not assistant_text or not image_name:
            skipped += 1
            continue

        sample = {
            "index": index,
            "image_name": image_name,
            "image_path": str((images_root / image_name).resolve()),
            "assistant_text": assistant_text,
        }

        normalized = normalize_safe_text(assistant_text)
        if normalized in SAFE_TEXTS:
            safe_samples.append(sample)
        else:
            hazard_samples.append(sample)

    rng.shuffle(safe_samples)
    rng.shuffle(hazard_samples)

    print(f"Dataset parsed: safe={len(safe_samples)} hazard={len(hazard_samples)} skipped={skipped}")
    return {
        "safe": safe_samples[:safe_count],
        "hazard": hazard_samples[:hazard_count],
    }


def load_model(model_path: str, local_files_only: bool):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    model.eval()
    return model, processor


def detect_hardware_label() -> str:
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        return " | ".join(gpu_names)
    return "CPU"


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def run_inference_once(model, processor, image_path: Path, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    end_to_end_start = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generate_start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generate_end = time.perf_counter()

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text_list = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    end_to_end_end = time.perf_counter()

    return {
        "end_to_end_latency_sec": end_to_end_end - end_to_end_start,
        "generate_only_latency_sec": generate_end - generate_start,
        "generated_tokens": int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0,
        "response_text": output_text_list[0].strip() if output_text_list else "",
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    end_to_end_latencies = [row["end_to_end_latency_sec"] for row in rows]
    generate_only_latencies = [row["generate_only_latency_sec"] for row in rows]
    tokens = [row["generated_tokens"] for row in rows]
    return {
        "count": len(rows),
        "avg_end_to_end_latency_sec": statistics.fmean(end_to_end_latencies) if end_to_end_latencies else None,
        "std_end_to_end_latency_sec": statistics.stdev(end_to_end_latencies) if len(end_to_end_latencies) > 1 else 0.0 if end_to_end_latencies else None,
        "min_end_to_end_latency_sec": min(end_to_end_latencies) if end_to_end_latencies else None,
        "max_end_to_end_latency_sec": max(end_to_end_latencies) if end_to_end_latencies else None,
        "p50_end_to_end_latency_sec": percentile(end_to_end_latencies, 0.50),
        "p95_end_to_end_latency_sec": percentile(end_to_end_latencies, 0.95),
        "avg_generate_only_latency_sec": statistics.fmean(generate_only_latencies) if generate_only_latencies else None,
        "std_generate_only_latency_sec": statistics.stdev(generate_only_latencies) if len(generate_only_latencies) > 1 else 0.0 if generate_only_latencies else None,
        "min_generate_only_latency_sec": min(generate_only_latencies) if generate_only_latencies else None,
        "max_generate_only_latency_sec": max(generate_only_latencies) if generate_only_latencies else None,
        "p50_generate_only_latency_sec": percentile(generate_only_latencies, 0.50),
        "p95_generate_only_latency_sec": percentile(generate_only_latencies, 0.95),
        "avg_generated_tokens": statistics.fmean(tokens) if tokens else None,
    }


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    fieldnames = [
        "group",
        "model_label",
        "hardware_label",
        "precision_label",
        "quantization_label",
        "input_setting_label",
        "count",
        "avg_end_to_end_latency_sec",
        "std_end_to_end_latency_sec",
        "min_end_to_end_latency_sec",
        "max_end_to_end_latency_sec",
        "p50_end_to_end_latency_sec",
        "p95_end_to_end_latency_sec",
        "avg_generate_only_latency_sec",
        "std_generate_only_latency_sec",
        "min_generate_only_latency_sec",
        "max_generate_only_latency_sec",
        "p50_generate_only_latency_sec",
        "p95_generate_only_latency_sec",
        "avg_generated_tokens",
        "peak_gpu_memory_allocated_gb",
        "peak_gpu_memory_reserved_gb",
        "max_new_tokens",
        "repeats",
    ]
    rows: List[Dict[str, Any]] = []
    common = {
        "model_label": summary["model_label"],
        "hardware_label": summary["hardware_label"],
        "precision_label": summary["precision_label"],
        "quantization_label": summary["quantization_label"],
        "input_setting_label": summary["input_setting_label"],
        "peak_gpu_memory_allocated_gb": summary["peak_gpu_memory_allocated_gb"],
        "peak_gpu_memory_reserved_gb": summary["peak_gpu_memory_reserved_gb"],
        "max_new_tokens": summary["max_new_tokens"],
        "repeats": summary["repeats"],
    }
    for group_name, group_summary in summary["groups"].items():
        row = {"group": group_name, **common, **group_summary}
        rows.append(row)
    rows.append({"group": "overall", **common, **summary["overall"]})

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    dataset_json = Path(args.dataset_json).resolve()
    images_root = Path(args.images_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = build_samples(
        dataset_json=dataset_json,
        images_root=images_root,
        safe_count=args.safe_count,
        hazard_count=args.hazard_count,
        seed=args.seed,
    )

    print("Loading model...")
    model, processor = load_model(args.model_path, args.local_files_only)
    hardware_label = detect_hardware_label()
    model_label = args.model_label or Path(args.model_path).name

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    warmup_candidates: List[Dict[str, Any]] = []
    for group_name in args.group_order:
        warmup_candidates.extend(groups.get(group_name, [])[:1])
    warmup_candidates = warmup_candidates[: args.warmup_runs]

    if warmup_candidates:
        print(f"Running {len(warmup_candidates)} warmup runs...")
        for sample in warmup_candidates:
            image_path = Path(sample["image_path"])
            if not image_path.is_file():
                continue
            _ = run_inference_once(model, processor, image_path, args.prompt_text, args.max_new_tokens)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    per_run_rows: List[Dict[str, Any]] = []
    group_summaries: Dict[str, Any] = {}

    for group_name in args.group_order:
        samples = groups.get(group_name, [])
        print(f"\n===== Start group: {group_name} | samples={len(samples)} =====")
        group_rows: List[Dict[str, Any]] = []

        for sample_index, sample in enumerate(samples, start=1):
            image_path = Path(sample["image_path"])
            if not image_path.is_file():
                message = f"Missing image: {image_path}"
                if args.skip_missing_images:
                    print(f"[SKIP] {message}")
                    continue
                raise FileNotFoundError(message)

            for repeat_index in range(args.repeats):
                try:
                    result = run_inference_once(
                        model=model,
                        processor=processor,
                        image_path=image_path,
                        prompt=args.prompt_text,
                        max_new_tokens=args.max_new_tokens,
                    )
                    error = None
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "latency_sec": None,
                        "generated_tokens": None,
                        "response_text": "",
                    }
                    error = f"{type(exc).__name__}: {exc}"

                row = {
                    "group": group_name,
                    "sample_index": sample_index,
                    "repeat_index": repeat_index + 1,
                    "dataset_index": sample["index"],
                    "image_name": sample["image_name"],
                    "image_path": str(image_path),
                    "assistant_text": sample["assistant_text"],
                    "end_to_end_latency_sec": result["end_to_end_latency_sec"],
                    "generate_only_latency_sec": result["generate_only_latency_sec"],
                    "generated_tokens": result["generated_tokens"],
                    "response_text": result["response_text"],
                    "error": error,
                }
                per_run_rows.append(row)
                if result["end_to_end_latency_sec"] is not None:
                    group_rows.append(row)
                    print(
                        f"[{group_name}] {sample_index}/{len(samples)} repeat={repeat_index + 1}/{args.repeats} "
                        f"image={sample['image_name']} e2e={result['end_to_end_latency_sec']:.4f}s "
                        f"gen={result['generate_only_latency_sec']:.4f}s "
                        f"tokens={result['generated_tokens']}"
                    )
                else:
                    print(
                        f"[{group_name}] {sample_index}/{len(samples)} repeat={repeat_index + 1}/{args.repeats} "
                        f"image={sample['image_name']} ERROR={error}"
                    )

        group_summaries[group_name] = summarize_rows(group_rows)
        print(f"===== Finish group: {group_name} =====")
        print(json.dumps(group_summaries[group_name], ensure_ascii=False, indent=2))

    valid_rows = [row for row in per_run_rows if row["end_to_end_latency_sec"] is not None]
    summary = {
        "model_label": model_label,
        "model_path": args.model_path,
        "hardware_label": hardware_label,
        "precision_label": args.precision_label,
        "quantization_label": args.quantization_label,
        "input_setting_label": args.input_setting_label,
        "dataset_json": str(dataset_json),
        "images_root": str(images_root),
        "safe_count": args.safe_count,
        "hazard_count": args.hazard_count,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "peak_gpu_memory_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024 ** 3) if torch.cuda.is_available() else None,
        "peak_gpu_memory_reserved_gb": float(torch.cuda.max_memory_reserved()) / (1024 ** 3) if torch.cuda.is_available() else None,
        "groups": group_summaries,
        "overall": summarize_rows(valid_rows),
    }

    save_json(output_dir / "summary.json", summary)
    save_jsonl(output_dir / "per_run.jsonl", per_run_rows)
    save_summary_csv(output_dir / "summary.csv", summary)

    print("\n===== Overall summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
