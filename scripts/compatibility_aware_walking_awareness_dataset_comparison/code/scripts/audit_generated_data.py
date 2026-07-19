#!/usr/bin/env python3
"""Audit generated WAD variants, field coverage, and WAD-test overlap."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FIELDS = ("weather_condition", "area_type", "danger_level", "traffic_flow_rating", "summary", "alter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/fair_variants"))
    parser.add_argument("--test-json", type=Path, default=Path("test/walkvlm_qwen25vl_test_alter.json"))
    args = parser.parse_args()
    alter = json.loads((args.data_dir / "wad_alter_only_train.json").read_text(encoding="utf-8"))
    full = json.loads((args.data_dir / "wad_full_json_train.json").read_text(encoding="utf-8"))
    test = json.loads(args.test_json.read_text(encoding="utf-8"))
    train_images = [row["images"][0] for row in alter]
    test_images = [row["images"][0] for row in test]
    targets = [json.loads(row["messages"][1]["content"]) for row in full]
    report = {
        "train_records_per_variant": len(alter),
        "test_records": len(test),
        "unique_train_images": len(set(train_images)),
        "unique_test_images": len(set(test_images)),
        "train_test_image_overlap": len(set(train_images) & set(test_images)),
        "empty_field_counts": {field: sum(not str(row[field]).strip() for row in targets) for field in FIELDS},
        "categorical_distributions": {
            field: dict(Counter(str(row[field]) for row in targets).most_common())
            for field in ("weather_condition", "area_type", "danger_level", "traffic_flow_rating")
        },
    }
    output = args.data_dir / "data_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["train_test_image_overlap"]:
        raise RuntimeError("Selected WAD train images overlap WAD test images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

