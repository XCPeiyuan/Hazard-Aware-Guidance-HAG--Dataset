#!/usr/bin/env python3
"""Validate that both WAD variants use identical images and expected targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = {"weather_condition", "area_type", "danger_level", "traffic_flow_rating", "summary", "alter"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    alter = json.loads((args.data_dir / "wad_alter_only_train.json").read_text(encoding="utf-8"))
    full = json.loads((args.data_dir / "wad_full_json_train.json").read_text(encoding="utf-8"))
    assert len(alter) == len(full) > 0
    for index, (a, f) in enumerate(zip(alter, full)):
        assert a["images"] == f["images"], index
        assert "Scene summary:" not in a["messages"][0]["content"], index
        assert "Scene summary:" not in f["messages"][0]["content"], index
        target = json.loads(f["messages"][1]["content"])
        assert set(target) == FIELDS, (index, set(target))
        assert target["alter"] == a["messages"][1]["content"], index
        assert Path(a["images"][0]).exists(), a["images"][0]
    print(json.dumps({"validated_samples_per_variant": len(alter), "same_images": True, "summary_in_input": False, "full_json_fields": sorted(FIELDS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

