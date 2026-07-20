#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import shutil


FILE1_DEFAULT = "/carrot/zhup/[main]inference/Prompt-only/gpt4o/r/review-danger-correct.txt"
FILE2_DEFAULT = "/carrot/zhup/[main]inference/VR/qwen25vl7b/vr-r/review-danger-correct.txt"
DATASET_IMG_DIR_DEFAULT = "/carrot/zhup/dataset/images/"


def _get_progress(iterable, total: int, desc: str):
    """Try to use tqdm; fallback to a simple generator with periodic prints."""
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        def gen():
            for i, x in enumerate(iterable, 1):
                if i == 1 or i == total or i % 50 == 0:
                    print(f"{desc}: {i}/{total}", file=sys.stderr)
                yield x
        return gen()


def parse_review_file(path: Path) -> Dict[str, str]:
    """
    Parse blocks:
      line1: image filename (e.g., img_000007.png)
      line2: text (e.g., <ALERT>...</ALERT><GUIDE>...</GUIDE>)
      line3: blank
    Return dict: {img_name: content_line2}
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    # Split by blank lines (one or more empty lines)
    blocks = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
    out: Dict[str, str] = {}

    for b in blocks:
        lines = [ln.rstrip("\n") for ln in b.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            # skip malformed block
            continue
        img = lines[0].strip()
        content = lines[1].strip()

        # Basic sanity check: looks like an image file
        if not re.search(r"\.(png|jpg|jpeg|webp|bmp)$", img, re.IGNORECASE):
            # still allow, but keep conservative: skip if it doesn't look like an image name
            continue

        out[img] = content

    return out


def ensure_group_dir(base_dir: Path) -> Path:
    """
    Create Human_Env/Group_001 (or next available).
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Find next available Group_XXX
    for i in range(1, 1000):
        group = base_dir / f"Group_{i:03d}"
        if not group.exists():
            group.mkdir(parents=True, exist_ok=False)
            return group

    raise RuntimeError("Too many Group_XXX folders (>=999). Please clean up Human_Env.")


def safe_copy_image(src: Path, dst: Path) -> Tuple[bool, str]:
    """
    Copy file content only (no metadata), using shutil.copyfile.
    Return (ok, message).
    """
    if not src.exists():
        return False, f"Missing image: {src}"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)  # copies data only, not permissions/timestamps like copy2
        return True, "OK"
    except Exception as e:
        return False, f"Copy failed: {src} -> {dst} ({e})"


def write_readme(
    readme_path: Path,
    file1: Path,
    file2: Path,
    d1: Dict[str, str],
    d2: Dict[str, str],
    img_list: List[str],
    missing_in_dataset: List[str],
    dataset_img_dir: Path,
) -> None:
    lines: List[str] = []
    lines.append("Human_Env build summary")
    lines.append("")
    lines.append(f"File 1: {file1}")
    lines.append(f"File 2: {file2}")
    lines.append(f"Dataset image dir: {dataset_img_dir}")
    lines.append("")
    lines.append(f"Count in File1: {len(d1)}")
    lines.append(f"Count in File2: {len(d2)}")
    lines.append(f"Intersection count (img_list): {len(img_list)}")
    lines.append("")

    lines.append("Matched image names (img_list):")
    for img in img_list:
        lines.append(f"  - {img}")
    lines.append("")

    if missing_in_dataset:
        lines.append("Warnings: images missing under dataset/images:")
        for img in missing_in_dataset:
            lines.append(f"  - {img}")
        lines.append("")
    else:
        lines.append("All matched images were found in dataset/images.")
        lines.append("")

    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    # Inputs (allow override via CLI args)
    file1 = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(FILE1_DEFAULT)
    file2 = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path(FILE2_DEFAULT)
    dataset_img_dir = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path(DATASET_IMG_DIR_DEFAULT)

    # Parse
    d1 = parse_review_file(file1)
    d2 = parse_review_file(file2)

    # Intersection list (sorted for reproducibility)
    img_list = sorted(set(d1.keys()) & set(d2.keys()))

    # Create Human_Env/Group_XXX under script directory
    script_dir = Path(__file__).resolve().parent
    human_env_dir = script_dir / "Human_Env"
    group_dir = ensure_group_dir(human_env_dir)

    # Pre-check dataset images existence
    missing_in_dataset: List[str] = []
    for img in img_list:
        if not (dataset_img_dir / img).exists():
            missing_in_dataset.append(img)

    # Write readme
    readme_path = group_dir / "readme.txt"
    write_readme(
        readme_path=readme_path,
        file1=file1,
        file2=file2,
        d1=d1,
        d2=d2,
        img_list=img_list,
        missing_in_dataset=missing_in_dataset,
        dataset_img_dir=dataset_img_dir,
    )

    # Build per-image folders
    iterator = _get_progress(img_list, total=len(img_list), desc=f"Building {group_dir.name}")

    for img in iterator:
        img_stem = Path(img).stem  # img_000007
        per_dir = group_dir / img_stem
        per_dir.mkdir(parents=True, exist_ok=True)

        # Copy image into folder
        src_img = dataset_img_dir / img
        dst_img = per_dir / img
        ok, msg = safe_copy_image(src_img, dst_img)

        # Write annotation.txt
        ann_path = per_dir / "annotation.txt"
        # If somehow missing from dict, fall back to empty string
        content1 = d1.get(img, "")
        content2 = d2.get(img, "")

        ann_lines: List[str] = []
        ann_lines.append(img)
        ann_lines.append(str(file1))
        ann_lines.append(content1)
        ann_lines.append(str(file2))
        ann_lines.append(content2)
        ann_lines.append("")  # trailing newline

        if not ok:
            # record warning in annotation too
            ann_lines.append(f"[WARN] {msg}")
            ann_lines.append("")

        ann_path.write_text("\n".join(ann_lines), encoding="utf-8")

    print(f"Done. Output: {group_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

