#!/usr/bin/env python3
"""Extract alter strings from full-JSON model output txt for existing evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_txt", type=Path)
    parser.add_argument("output_txt", type=Path)
    args = parser.parse_args()
    lines = args.input_txt.read_text(encoding="utf-8").splitlines()
    output = []
    failures = []
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        image = lines[i].strip()
        raw = lines[i + 1].strip() if i + 1 < len(lines) else ""
        i += 2
        parsed = extract_json(raw)
        alter = str(parsed.get("alter", "")).strip() if parsed else ""
        if not alter:
            failures.append(image)
            alter = "[ALTER_EXTRACTION_FAILED]"
        output.extend([image, " ".join(alter.split()), ""])
    args.output_txt.write_text("\n".join(output) + "\n", encoding="utf-8")
    report = {"records": len(output) // 3, "failed": len(failures), "failed_images": failures}
    args.output_txt.with_suffix(args.output_txt.suffix + ".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

