#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多文件合并的三类 hazard 共现矩阵（不含 SAFE）。
python cooccurrence_matrix.py \
  --inputs train_R_en_category.json test_R_en_category.json

输入：一个或多个 *_category.json
输出：
- cooccurrence_counts.json  (合并后的计数/比例)
- cooccurrence_heatmap.png  (3x3 蓝白渐变热力图，只显示百分比)

说明：
- SAFE 单独统计：objects为空的图像数
- 共现矩阵只考虑三类：GROUND / PIT / OVERHEAD
- 图像级统计：一张图只要出现该类别，就认为包含
"""

import os
import json
import argparse
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


HAZ_CATS = ["GROUND", "PIT", "OVERHEAD"]
CAT_DISPLAY = {
    "GROUND": "Common Obstacle",
    "PIT": "Pitfall Hazard",
    "OVERHEAD": "Upper-body Hazard",
}


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list JSON.")
    return data


def extract_image_hazard_set(item: Dict[str, Any]) -> List[str]:
    """
    从单条记录中提取该图像包含的 hazard 类别集合（图像级）。
    返回去重后的类别列表（只在 HAZ_CATS 内），可能为空。
    """
    objs = item.get("objects", [])
    haz_set = set()
    for obj in objs:
        cats = obj.get("categories", [])
        for c in cats:
            if not isinstance(c, str):
                continue
            c2 = c.strip().upper()
            if c2 in HAZ_CATS:
                haz_set.add(c2)
    return list(haz_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more *_category.json paths"
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: directory of first input."
    )
    args = parser.parse_args()

    input_paths = args.inputs
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(input_paths[0]))
    os.makedirs(out_dir, exist_ok=True)

    # ===== 合并统计容器 =====
    # cooccur_count[i,j] = 同时包含 HAZ_CATS[i] 和 HAZ_CATS[j] 的图像数
    cooccur_count = np.zeros((3, 3), dtype=np.int64)
    single_count = np.zeros(3, dtype=np.int64)  # 每类出现的图像数（图像级）
    safe_only_count = 0
    total_images = 0

    # ===== 遍历所有输入文件 =====
    for path in input_paths:
        data = load_json_list(path)
        print(f"Loaded {len(data)} images from {path}")

        for item in data:
            total_images += 1
            haz_list = extract_image_hazard_set(item)

            if len(haz_list) == 0:
                # 纯 SAFE
                safe_only_count += 1
                continue

            # 图像级出现统计
            for c in haz_list:
                single_count[HAZ_CATS.index(c)] += 1

            # 共现统计（包含自己与自己）
            for i, ci in enumerate(HAZ_CATS):
                for j, cj in enumerate(HAZ_CATS):
                    if (ci in haz_list) and (cj in haz_list):
                        cooccur_count[i, j] += 1

    # ===== 计算比例（以“包含行类别的图像数”为分母）=====
    # row-normalized: P(cj | ci)
    row_sums = single_count.reshape(3, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cooccur_percent = (cooccur_count / row_sums) * 100.0
        cooccur_percent = np.nan_to_num(cooccur_percent)

    # ===== 保存合并后的 json =====
    out_json_path = os.path.join(out_dir, "cooccurrence_counts.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "inputs": input_paths,
            "total_images": int(total_images),
            "safe_only_images": int(safe_only_count),
            "hazard_image_counts": {
                "GROUND": int(single_count[0]),
                "PIT": int(single_count[1]),
                "OVERHEAD": int(single_count[2]),
            },
            "cooccurrence_count_matrix": cooccur_count.tolist(),
            "cooccurrence_percent_row_norm": cooccur_percent.tolist(),
            "cat_order": HAZ_CATS
        }, f, ensure_ascii=False, indent=2)

    # ===== 画 3x3 热力图（蓝白渐变 + 百分比）=====
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cooccur_percent, cmap=plt.cm.Blues, vmin=0, vmax=100)

    labels = [CAT_DISPLAY[c] for c in HAZ_CATS]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cooccur_percent[i, j]:.1f}%",
                    ha="center", va="center", fontsize=11)

    ax.set_xlabel("Also contains ...")
    ax.set_ylabel("Contains ...")
    ax.set_title("Hazard Type Co-occurrence Matrix (Row-normalized %)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage (%)")

    heatmap_path = os.path.join(out_dir, "cooccurrence_heatmap.png")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nDone.")
    print(f"Saved JSON: {out_json_path}")
    print(f"Saved heatmap: {heatmap_path}")
    print("\nSummary:")
    print(f"Total images: {total_images}")
    print(f"SAFE-only images: {safe_only_count}")
    for i, c in enumerate(HAZ_CATS):
        print(f"{c} images: {single_count[i]}")


if __name__ == "__main__":
    main()

