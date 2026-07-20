#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
提取 Danger 判断正确的样本，保存到一个 txt 用于人工 review。
输入格式与之前完全一致（三行一组：图片名 / 文本 / 空行）。
输出：review-danger-correct.txt
"""

import os

# ========= 用户可修改区域 =========
REF_TXT = "test_R_en.txt"  
PRED_TXT_DIR = "/carrot/zhup/[main]inference/Prompt-only/qwen25vl32b/r-0shot"
PRED_TXT_NAME = "output_en.txt"
# =================================

OUTPUT_TXT = os.path.join(PRED_TXT_DIR, "review-danger-correct.txt")


def parse_triplet_txt(path):
    """解析三行一组格式"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i, n = 0, len(lines)
    while i < n:
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break

        img = lines[i].strip()
        i += 1

        content = ""
        if i < n:
            content = lines[i].strip()
            i += 1

        while i < n and lines[i].strip() == "":
            i += 1

        if img:
            items.append((img, content))
    return items


def is_safe(text: str) -> bool:
    return text.strip() == "<SAFE/>"


def main():
    ref_list = parse_triplet_txt(REF_TXT)
    pred_list = parse_triplet_txt(os.path.join(PRED_TXT_DIR, PRED_TXT_NAME))

    N = min(len(ref_list), len(pred_list))
    count = 0

    with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
        for i in range(N):
            ref_name, ref_text = ref_list[i]
            pred_name, pred_text = pred_list[i]

            # 必须：图片名一致
            if ref_name != pred_name:
                continue

            ref_safe = is_safe(ref_text)
            pred_safe = is_safe(pred_text)

            # 只选 Danger判断正确（非SAFE→非SAFE）
            if (not ref_safe) and (not pred_safe):
                out.write(ref_name + "\n")
                out.write(pred_text + "\n\n")
                count += 1

    print(f"已提取 {count} 条 Danger 判断正确的样本。")
    print(f"输出文件：{OUTPUT_TXT}")


if __name__ == "__main__":
    main()

