#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
格式检查器：用于检测 train_V_en.txt 格式是否严格遵守
  - 一行图片名
  - 一行标注
  - 一行空行
若有格式错误，输出对应行号与错误类型
最后统计每种问题数量
"""

import re

# ========== 配置 ==========
INPUT_TXT = "output_en.txt"
# =========================

IMAGE_PATTERN = re.compile(r".+\.(png|jpg|jpeg|bmp|gif)$", re.I)


def is_image_name(line: str) -> bool:
    return bool(IMAGE_PATTERN.fullmatch(line.strip()))


def is_annotation(line: str) -> bool:
    """必须是 SAFE 或 含 ALERT"""
    line = line.strip()
    if line == "<SAFE/>":
        return True
    if "<ALERT>" in line and "</ALERT>" in line:
        return True
    return False


def is_empty(line: str) -> bool:
    return line.strip() == ""


def main():
    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    n = len(lines)
    idx = 0  # 0-based

    error_counts = {
        "ImageNameError": 0,
        "AnnotationMissingError": 0,
        "AlertFormatError": 0,
        "MissingEmptyLineError": 0,
        "UnexpectedExtraLineError": 0,
        "IncompleteBlockError": 0,
    }

    while idx < n:
        line1_idx = idx + 1  # convert to 1-based for report

        # ===== 第1行：图片名 =====
        if idx >= n:
            print(f"[ERROR] Line {line1_idx}: IncompleteBlockError (missing image name)")
            error_counts["IncompleteBlockError"] += 1
            break

        img_line = lines[idx]
        if not is_image_name(img_line):
            print(f"[ERROR] Line {line1_idx}: ImageNameError -> '{img_line}'")
            error_counts["ImageNameError"] += 1
        idx += 1

        # ===== 第2行：标注 =====
        if idx >= n:
            print(f"[ERROR] Line {line1_idx+1}: IncompleteBlockError (missing annotation)")
            error_counts["IncompleteBlockError"] += 1
            break

        annotation = lines[idx]
        if annotation.strip() == "":
            print(f"[ERROR] Line {line1_idx+1}: AnnotationMissingError (empty annotation)")
            error_counts["AnnotationMissingError"] += 1
        elif not is_annotation(annotation):
            print(f"[ERROR] Line {line1_idx+1}: AlertFormatError -> '{annotation}'")
            error_counts["AlertFormatError"] += 1
        idx += 1

        # ===== 第3行：空行 =====
        if idx >= n:
            print(f"[ERROR] Line {line1_idx+2}: IncompleteBlockError (missing empty line)")
            error_counts["IncompleteBlockError"] += 1
            break

        empty_line = lines[idx]
        if not is_empty(empty_line):
            print(f"[ERROR] Line {line1_idx+2}: MissingEmptyLineError -> '{empty_line}'")
            error_counts["MissingEmptyLineError"] += 1
        idx += 1

        # ===== 检查是否有多余的非空行（理论上不会有）=====
        # 此逻辑保持为可扩张，将来如果你严格需要无多余空行，可以开启
        # if idx < n and not is_image_name(lines[idx]) and not is_empty(lines[idx]):
        #     print(f"[ERROR] Line {idx+1}: UnexpectedExtraLineError -> '{lines[idx]}'")
        #     error_counts["UnexpectedExtraLineError"] += 1
        #     idx += 1

    # ===== 输出统计 =====
    print("\n========== FORMAT CHECK SUMMARY ==========")
    total_errors = sum(error_counts.values())
    print(f"Total errors: {total_errors}")
    for k, v in error_counts.items():
        print(f"{k}: {v}")
    print("==========================================")



if __name__ == "__main__":
    main()

