#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference Efficiency Analysis mobile / localhost benchmark template.

Purpose:
- Send every image in a folder to a localhost endpoint.
- Repeat each image multiple times.
- Print detailed runtime info to the terminal.
- Save machine-readable logs for later paper writing.

Important:
- This is a template. You still need to adapt `build_payload()` and
  `extract_text_from_response()` to match your phone-side software API.
- Only Python standard library is used.
"""

from __future__ import annotations

import base64
import csv
import json
import mimetypes
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# =========================
# TODO: fill these settings
# =========================
SERVER_URL = "http://127.0.0.1:8000/infer"
IMAGE_DIR = r"/path/to/mobile_eval_images"
OUTPUT_DIR = r"./mobile_localhost_benchmark"
PROMPT = (
    "You are a virtual assistant helping a blind person travel. "
    "The blind person is walking straight ahead. Please determine whether the path ahead is safe: "
    "If it is safe, output only: <SAFE/>; "
    "If there is hazard, output only (fixed format, concise, no additional content): "
    "<ALERT>Describe the obstacle, its direction, and approximate distance</ALERT>"
    "<GUIDE>Provide clear instructions for avoidance actions</GUIDE>"
)
REPEAT_COUNT = 3
TIMEOUT_SEC = 120.0

# Optional headers. Edit if your localhost service requires auth or custom content type.
HEADERS = {
    "Content-Type": "application/json",
}


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def encode_image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_payload(image_path: Path, prompt: str) -> Dict[str, Any]:
    """
    TODO:
    Replace this payload structure with the exact localhost API schema
    used by your phone-side software.

    Common placeholders already prepared:
    - image_path.name
    - base64 image bytes
    - mime type
    - prompt
    """
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    image_b64 = encode_image_base64(image_path)

    # Example payload only. Replace with your actual API schema.
    return {
        "image_name": image_path.name,
        "image_mime_type": mime_type,
        "image_base64": image_b64,
        "prompt": prompt,
        "max_new_tokens": 128,
    }


def extract_text_from_response(response_json: Any) -> Optional[str]:
    """
    TODO:
    Replace this with the exact response parsing logic of your localhost API.
    """
    if isinstance(response_json, dict):
        for key_path in (
            ("output", "text"),
            ("text",),
            ("response",),
            ("result",),
        ):
            current = response_json
            ok = True
            for key in key_path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    ok = False
                    break
            if ok and isinstance(current, str):
                return current.strip()
    return None


def is_safe_output(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    normalized = text.strip().lower()
    return normalized in {"</safe>", "<safe/>", "<safe />", "safe"}


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    if low == high:
        return values[low]
    weight = pos - low
    return values[low] * (1 - weight) + values[high] * weight


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if row["latency_sec"] is not None]
    latencies = [row["latency_sec"] for row in valid_rows]
    safe_rows = [row for row in valid_rows if row["predicted_group"] == "safe"]
    hazard_rows = [row for row in valid_rows if row["predicted_group"] == "hazard"]

    def small_summary(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        subset_latencies = [row["latency_sec"] for row in subset]
        return {
            "count": len(subset),
            "avg_latency_sec": statistics.fmean(subset_latencies) if subset_latencies else None,
            "std_latency_sec": statistics.stdev(subset_latencies) if len(subset_latencies) > 1 else 0.0 if subset_latencies else None,
            "min_latency_sec": min(subset_latencies) if subset_latencies else None,
            "max_latency_sec": max(subset_latencies) if subset_latencies else None,
            "p50_latency_sec": percentile(subset_latencies, 0.50),
            "p95_latency_sec": percentile(subset_latencies, 0.95),
        }

    return {
        "overall": small_summary(valid_rows),
        "predicted_safe": small_summary(safe_rows),
        "predicted_hazard": small_summary(hazard_rows),
    }


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "image_name",
        "repeat_index",
        "latency_sec",
        "http_status",
        "predicted_group",
        "response_text_preview",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    image_dir = Path(IMAGE_DIR).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise SystemExit(f"IMAGE_DIR does not exist: {image_dir}")

    images = list(iter_images(image_dir))
    if not images:
        raise SystemExit(f"No valid images found in: {image_dir}")

    print(f"Server URL: {SERVER_URL}")
    print(f"Image dir: {image_dir}")
    print(f"Repeat count: {REPEAT_COUNT}")
    print(f"Image count: {len(images)}")

    rows: List[Dict[str, Any]] = []
    for image_path in images:
        for repeat_index in range(1, REPEAT_COUNT + 1):
            payload = build_payload(image_path, PROMPT)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                url=SERVER_URL,
                data=body,
                headers=HEADERS,
                method="POST",
            )

            http_status = None
            response_json = None
            response_text = None
            error = None

            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
                    http_status = getattr(response, "status", None)
                    raw = response.read().decode("utf-8")
                    try:
                        response_json = json.loads(raw)
                    except json.JSONDecodeError:
                        response_json = {"raw_text": raw}
                    response_text = extract_text_from_response(response_json)
            except urllib.error.HTTPError as exc:
                http_status = exc.code
                error = f"HTTPError: {exc}"
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            end = time.perf_counter()

            predicted_group = None
            safe_flag = is_safe_output(response_text)
            if safe_flag is True:
                predicted_group = "safe"
            elif safe_flag is False:
                predicted_group = "hazard"

            preview = ""
            if response_text:
                preview = response_text.replace("\n", " ")[:180]

            row = {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "repeat_index": repeat_index,
                "latency_sec": end - start if error is None else None,
                "http_status": http_status,
                "predicted_group": predicted_group,
                "response_text_preview": preview,
                "response_json": response_json,
                "error": error,
            }
            rows.append(row)

            print(
                f"image={image_path.name} repeat={repeat_index}/{REPEAT_COUNT} "
                f"latency={row['latency_sec'] if row['latency_sec'] is not None else 'ERROR'} "
                f"http={http_status} predicted_group={predicted_group} error={error}"
            )
            if preview:
                print(f"  preview={preview}")

    summary = {
        "server_url": SERVER_URL,
        "image_dir": str(image_dir),
        "repeat_count": REPEAT_COUNT,
        "timeout_sec": TIMEOUT_SEC,
        **summarize(rows),
    }

    save_json(output_dir / "summary.json", summary)
    save_jsonl(output_dir / "per_run.jsonl", rows)
    save_csv(output_dir / "per_run.csv", rows)

    print("\n===== Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
