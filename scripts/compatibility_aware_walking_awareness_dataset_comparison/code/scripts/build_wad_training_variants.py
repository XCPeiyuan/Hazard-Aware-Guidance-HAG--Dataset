#!/usr/bin/env python3
"""Build matched-sample, image-only WAD training variants for LLaMA-Factory."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


ALTER_PROMPT = """<image>
You are a navigation assistant for a visually impaired pedestrian.
Based only on the image, provide one concise real-time walking instruction or warning."""

FULL_JSON_PROMPT = """<image>
You are a navigation assistant for a visually impaired pedestrian.
Based only on the image, predict the walking-assistance annotation as one valid JSON object.
Use exactly these keys: weather_condition, area_type, danger_level, traffic_flow_rating, summary, alter.
The alter field must contain one concise real-time walking instruction or warning.
Return JSON only, without Markdown fences or additional explanation."""

OUTPUT_FIELDS = (
    "weather_condition",
    "area_type",
    "danger_level",
    "traffic_flow_rating",
    "summary",
    "alter",
)


def make_record(image: Path, prompt: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "images": [str(image)],
    }


def load_records(path: Path) -> tuple[list[dict], str]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return records, "jsonl"
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array or JSONL records: {path}")
    return value, "json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../WalkVLM/wad_dataset_v2/train_v2.json"))
    parser.add_argument("--image-root", type=Path, default=Path("../WalkVLM/wad_dataset_v2/src_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-count", type=int, required=True, help="Use the same count as the R+U training set.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, source_format = load_records(source)

    eligible = []
    rejected = Counter()
    for source_index, row in enumerate(records):
        if not isinstance(row, dict):
            rejected["not_object"] += 1
            continue
        if not str(row.get("alter", "")).strip():
            rejected["missing_alter"] += 1
            continue
        frame_path = str(row.get("frame_path", "")).strip()
        if not frame_path:
            rejected["missing_frame_path"] += 1
            continue
        image = image_root / frame_path / "0.jpg"
        eligible.append((source_index, row, image))

    if args.sample_count > len(eligible):
        raise ValueError(f"Requested {args.sample_count} samples, but only {len(eligible)} alter records are eligible.")

    rng = random.Random(args.seed)
    shuffled = eligible[:]
    rng.shuffle(shuffled)
    selected = []
    for candidate in shuffled:
        if candidate[2].exists():
            selected.append(candidate)
            if len(selected) == args.sample_count:
                break
        else:
            rejected["missing_image"] += 1
    if len(selected) < args.sample_count:
        raise ValueError(f"Only {len(selected)} selected alter records have existing images; requested {args.sample_count}.")
    selected.sort(key=lambda item: item[0])

    alter_only = []
    full_json = []
    manifest = []
    for source_index, row, image in selected:
        alter = " ".join(str(row["alter"]).split())
        target = {field: row.get(field, "") for field in OUTPUT_FIELDS}
        target = {key: " ".join(str(value).split()) for key, value in target.items()}
        alter_only.append(make_record(image, ALTER_PROMPT, alter))
        full_json.append(make_record(image, FULL_JSON_PROMPT, json.dumps(target, ensure_ascii=False, separators=(",", ":"))))
        manifest.append({
            "source_index": source_index,
            "video": row.get("video", ""),
            "frame_path": row.get("frame_path", ""),
            "image": str(image),
        })

    (output_dir / "wad_alter_only_train.json").write_text(json.dumps(alter_only, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "wad_full_json_train.json").write_text(json.dumps(full_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "selected_samples_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "source": str(source),
        "image_root": str(image_root),
        "source_records": len(records),
        "source_format": source_format,
        "eligible_alter_records": len(eligible),
        "selected_records": len(selected),
        "sample_count": args.sample_count,
        "seed": args.seed,
        "rejected": dict(rejected),
        "qa_records_included": 0,
        "input_summary_included": False,
        "variants": ["image-only -> alter", "image-only -> full annotation JSON"],
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
