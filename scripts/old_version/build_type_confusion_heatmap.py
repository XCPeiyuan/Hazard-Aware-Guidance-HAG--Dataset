#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 GT 与 Pred 的 category.json 生成 4x4 类别混淆热力图（方案A：仅按类别数量，不对齐物体名称）。

类别映射：
SAFE -> Safe
OVERHEAD -> Upper-body Hazard
PIT -> Pitfall Hazard
GROUND -> Common Obstacle

逻辑：
- 展开 GT / Pred 成多重集合（每个类别出现一次算一次）
- 对 GT 中每个 hazard 类 g：
    若 Pred 中有同类未用 -> g→g
    否则 -> g→SAFE
- Pred 中剩余 hazard 类 -> SAFE→p
- 若 GT 空且 Pred 空 -> SAFE→SAFE (1 次)
"""

import os
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ========== 配置 ==========
file_path = "/carrot/zhup/[main]inference/VR/qwen25vl7b/vr-r/"

PRED_JSON = file_path + "output_en_category.json"
REF_JSON = "test_R_en_category.json"

OUT_DIR = file_path  # 直接使用文件夹路径

OUT_PNG = OUT_DIR + "type_confusion_heatmap.png"
OUT_PDF = OUT_DIR + "type_confusion_heatmap.pdf"
OUT_COUNTS_JSON = OUT_DIR + "type_confusion_counts.json"
OUT_NORM_JSON = OUT_DIR + "type_confusion_row_norm.json"

# ==========================


# 内部类别名
CATS_INTERNAL = ["SAFE", "OVERHEAD", "PIT", "GROUND"]

# 展示名（按你要求）
CATS_DISPLAY = {
    "SAFE": "Safe",
    "OVERHEAD": "Upper-body Hazard",
    "PIT": "Pitfall Hazard",
    "GROUND": "Common Obstacle",
}

ALLOWED = set(CATS_INTERNAL)


def load_category_json(path: str) -> Dict[str, List[Dict]]:
    """
    读取 category.json，返回 image -> objects 列表的 dict
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    d = {}
    for item in data:
        img = item.get("image")
        objs = item.get("objects", [])
        if img:
            d[img] = objs
    return d


def expand_categories(objects: List[Dict]) -> List[str]:
    """
    将 objects 展开成多重类别列表：
    - 每个 object 的 categories 里的每个类别都算一次
    - 忽略未知类别
    """
    out = []
    if not objects:
        return out
    for obj in objects:
        cats = obj.get("categories", [])
        if not isinstance(cats, list):
            continue
        for c in cats:
            if isinstance(c, str):
                c2 = c.strip().upper()
                if c2 in ALLOWED and c2 != "SAFE":  # SAFE 不会在 objects 里出现，但保险起见排掉
                    out.append(c2)
    return out


def match_multiset(gt_list: List[str], pred_list: List[str]) -> Counter:
    """
    方案 A 的匹配：
    - gt_list / pred_list 都是 hazard 类别的多重集合（SAFE 不在这里）
    返回 pair 计数：Counter({(gt_cat, pred_cat): n, ...})
    """
    pairs = Counter()

    gt_counter = Counter(gt_list)
    pred_counter = Counter(pred_list)

    # 1) 逐个 GT hazard 匹配
    for g_cat, g_cnt in gt_counter.items():
        for _ in range(g_cnt):
            if pred_counter[g_cat] > 0:
                pairs[(g_cat, g_cat)] += 1
                pred_counter[g_cat] -= 1
            else:
                pairs[(g_cat, "SAFE")] += 1

    # 2) Pred 剩余 -> SAFE->pred
    for p_cat, p_cnt in pred_counter.items():
        for _ in range(p_cnt):
            pairs[("SAFE", p_cat)] += 1

    return pairs


def main():
    pred_dict = load_category_json(PRED_JSON)
    ref_dict = load_category_json(REF_JSON)

    # 混淆计数矩阵（GT 行, Pred 列）
    counts = defaultdict(int)

    missing_in_ref = 0
    total_images = 0

    for img, pred_objs in pred_dict.items():
        if img not in ref_dict:
            missing_in_ref += 1
            continue

        total_images += 1
        gt_objs = ref_dict[img]

        gt_hazards = expand_categories(gt_objs)
        pred_hazards = expand_categories(pred_objs)

        # 情况 A：GT SAFE（空），Pred SAFE（空）
        if len(gt_hazards) == 0 and len(pred_hazards) == 0:
            counts[("SAFE", "SAFE")] += 1
            continue

        # 情况 B：GT SAFE（空），Pred 非空
        if len(gt_hazards) == 0 and len(pred_hazards) > 0:
            for p in pred_hazards:
                counts[("SAFE", p)] += 1
            continue

        # 情况 C：GT 非空，Pred SAFE（空）
        if len(gt_hazards) > 0 and len(pred_hazards) == 0:
            for g in gt_hazards:
                counts[(g, "SAFE")] += 1
            continue

        # 情况 D：GT 非空，Pred 非空 -> 正常匹配
        pairs = match_multiset(gt_hazards, pred_hazards)
        for (g, p), n in pairs.items():
            counts[(g, p)] += n

    # 构造 4x4 矩阵
    mat = np.zeros((4, 4), dtype=int)
    for i, g in enumerate(CATS_INTERNAL):
        for j, p in enumerate(CATS_INTERNAL):
            mat[i, j] = counts.get((g, p), 0)

    # 行归一化（每行按 GT 类别比例）
    mat_row_sum = mat.sum(axis=1, keepdims=True)
    mat_norm = np.divide(
        mat, 
        mat_row_sum,
        out=np.zeros_like(mat, dtype=float),
        where=mat_row_sum != 0
    )

    # 保存 counts / norm 为 json 方便复现
    counts_out = {
        g: {p: int(mat[i, j]) for j, p in enumerate(CATS_INTERNAL)}
        for i, g in enumerate(CATS_INTERNAL)
    }
    norm_out = {
        g: {p: float(mat_norm[i, j]) for j, p in enumerate(CATS_INTERNAL)}
        for i, g in enumerate(CATS_INTERNAL)
    }

    with open(OUT_COUNTS_JSON, "w", encoding="utf-8") as f:
        json.dump(counts_out, f, ensure_ascii=False, indent=2)
    with open(OUT_NORM_JSON, "w", encoding="utf-8") as f:
        json.dump(norm_out, f, ensure_ascii=False, indent=2)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)

    # 使用白→浅蓝→深蓝渐变
    cmap = cm.get_cmap('Blues')
    im = ax.imshow(mat_norm, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # 设置轴标签
    xlabels = [CATS_DISPLAY[c] for c in CATS_INTERNAL]
    ylabels = [CATS_DISPLAY[c] for c in CATS_INTERNAL]
    ax.set_xticks(range(4), xlabels, rotation=30, ha="right")
    ax.set_yticks(range(4), ylabels)

    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Ground-truth Category")
    ax.set_title("Hazard Category Confusion Matrix")

    # 每格显示百分比（不显示原始计数）
    for i in range(4):
        for j in range(4):
            pct = mat_norm[i, j] * 100.0
            ax.text(
                j, i, f"{pct:.1f}%",
                ha="center", va="center",
                fontsize=14,
                fontweight="bold"
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Done.")
    print(f"- Total aligned images: {total_images}")
    print(f"- Missing in reference: {missing_in_ref}")
    print(f"- Counts saved to: {OUT_COUNTS_JSON}")
    print(f"- Row-normalized saved to: {OUT_NORM_JSON}")
    print(f"- Heatmap saved to: {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()

