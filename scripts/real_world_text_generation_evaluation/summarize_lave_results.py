#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对多个 LAVE JSON 结果进行汇总，生成适合论文使用的统计表：
  - 各模型 hazard/guide/format/overall 的 mean, std, median
  - overall_score >= 8 的比例（高质量）
  - overall_score >= 6 的比例（可接受）

输入：多个 JSON 文件，每个文件格式为 LAVE 评估输出的列表：
[
  {
    "image": "...",
    "ref_alert": "...",
    "ref_guide": "...",
    "pred_alert": "...",
    "pred_guide": "...",
    "scores": {
      "hazard_score": int,
      "guide_score": int,
      "format_score": int,
      "overall_score": int,
      "comment": "..."
    }
  },
  ...
]

使用前请在 MODEL_FILES 中填好模型名称和对应 JSON 路径。
"""

import os
import json
from typing import Dict

import pandas as pd

# ========== 配置区：在这里填你的模型结果 ==========

# 例子：请改成你自己的路径
MODEL_FILES: Dict[str, str] = {
    # "模型名字": "对应的 LAVE JSON 路径"
    "Qwen2.5-VL-3B (R)": "/carrot/zhup/[main]inference/R/qwen25vl3b/r-r/review-danger-correct_scores.json",
    "Qwen2.5-VL-3B (U)": "/carrot/zhup/[main]inference/V/qwen25vl3b/v-r/review-danger-correct_scores.json",
    "Qwen2.5-VL-3B (U+R)": "/carrot/zhup/[main]inference/VR/qwen25vl3b/vr-r/review-danger-correct_scores.json",
    "Qwen2.5-VL-7B (U+R)": "/carrot/zhup/[main]inference/VR/qwen25vl7b/vr-r/review-danger-correct_scores.json",
    "Qwen2-VL-2B (U+R)": "/carrot/zhup/[main]inference/VR/qwen2vl2b/vr-r/review-danger-correct_scores.json",
    "Qwen2.5-VL-7B (Few-shot)": "/carrot/zhup/[main]inference/Prompt-only/qwen25vl7b/r/review-danger-correct_scores.json",
    "Qwen2.5-VL-32B (U+R)": "/carrot/zhup/[main]inference/VR/qwen25vl32b/vr-r/review-danger-correct_scores.json",
    "Qwen2.5-VL-32B (Few-shot)": "/carrot/zhup/[main]inference/Prompt-only/qwen25vl32b/r/review-danger-correct_scores.json",
    "Qwen2.5-VL-72B (Few-shot)": "/carrot/zhup/[main]inference/Prompt-only/qwen25vl72b/r/review-danger-correct_scores.json",
    "GPT-4o (Few-shot)": "/carrot/zhup/[main]inference/Prompt-only/gpt4o/r/review-danger-correct_scores.json",
    "Qwen2.5-VL-7B (0-shot)": "/carrot/zhup/[main]inference/Prompt-only/qwen25vl7b/r-0shot/review-danger-correct_scores.json",
    "Qwen2.5-VL-32B (0-shot)": "/carrot/zhup/[main]inference/Prompt-only/qwen25vl32b/r-0shot/review-danger-correct_scores.json"
}

# “高质量”与“可接受”的阈值
GOOD_THRESHOLD = 8
OK_THRESHOLD = 6

# 输出 CSV 文件路径
OUTPUT_CSV = "lave_summary_all_models.csv"

# =================================================


def load_one_model_json(model_name: str, json_path: str) -> pd.DataFrame:
    """读取单个模型的 LAVE JSON，返回 DataFrame."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        s = item.get("scores", {})
        rows.append({
            "model": model_name,
            "image": item.get("image"),
            "hazard": s.get("hazard_score"),
            "guide": s.get("guide_score"),
            "format": s.get("format_score"),
            # overall 不从原数据读取，后续统一按公式重新计算
            "overall": None,
        })

    df = pd.DataFrame(rows)
    # 保险起见，把分数列转成数值
    for col in ["hazard", "guide", "format", "overall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 重新计算 overall: S = 0.5*Hazard + 0.4*Guide + 0.1*Format
    df["overall"] = 0.5 * df["hazard"] + 0.4 * df["guide"] + 0.1 * df["format"]

    return df


def main():
    # 1. 读入所有模型的数据
    all_dfs = []
    for model_name, path in MODEL_FILES.items():
        if not os.path.isfile(path):
            print(f"[WARN] JSON file not found for model {model_name}: {path}")
            continue
        df = load_one_model_json(model_name, path)
        print(f"[INFO] Loaded {len(df)} samples for model {model_name}")
        all_dfs.append(df)

    if not all_dfs:
        print("No valid JSON files loaded. Please check MODEL_FILES paths.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # 2. 计算聚合统计
    group = df_all.groupby("model")

    # 平均值 / 标准差 / 中位数
    agg_stats = group[["hazard", "guide", "format", "overall"]].agg(["mean", "std", "median"])

    # 展平 MultiIndex 列名，例如 hazard_mean, hazard_std, ...
    agg_stats.columns = [f"{metric}_{stat}" for metric, stat in agg_stats.columns]
    agg_stats = agg_stats.reset_index()

    # 3. 计算 through-rate（overall >= T）
    def rate_overall_at(threshold: int) -> pd.Series:
        return group["overall"].apply(
            lambda x: (x >= threshold).mean() * 100.0
        )

    rate_good = rate_overall_at(GOOD_THRESHOLD)   # overall >= GOOD_THRESHOLD
    rate_ok = rate_overall_at(OK_THRESHOLD)       # overall >= OK_THRESHOLD

    agg_stats["overall_ge_%d_pct" % GOOD_THRESHOLD] = rate_good.values
    agg_stats["overall_ge_%d_pct" % OK_THRESHOLD] = rate_ok.values

    # 4. 输出汇总（可以根据需要调整小数位）
    # 4.1 保存 CSV
    agg_stats.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[INFO] Summary saved to CSV: {OUTPUT_CSV}\n")

    # 4.2 打印 Markdown 表（可以直接复制到 Markdown / Jupyter）
    #    只保留需要在论文展示的列
    cols_for_table = [
        "model",
        "hazard_mean", "hazard_std",
        "guide_mean", "guide_std",
        "format_mean", "format_std",
        "overall_mean", "overall_std",
        f"overall_ge_{GOOD_THRESHOLD}_pct",
        f"overall_ge_{OK_THRESHOLD}_pct",
    ]

    # 有些列可能不存在（防止报错），做个过滤
    cols_for_table = [c for c in cols_for_table if c in agg_stats.columns]
    table_df = agg_stats[cols_for_table].copy()

    # 四舍五入
    for col in table_df.columns:
        if col != "model":
            table_df[col] = table_df[col].astype(float).round(3)

    print("========== Markdown Table ==========")
    try:
        print(table_df.to_markdown(index=False))
    except Exception:
        # 某些环境可能没有 tabulate 依赖，可以退化为普通 print
        print(table_df.to_string(index=False))
    print("====================================\n")

    # 4.3 打印 LaTeX 表格（可以直接丢 Overleaf）
    print("============== LaTeX Table ==============")
    print(
        table_df.to_latex(
            index=False,
            float_format="%.3f",
            escape=False
        )
    )
    print("=========================================\n")


if __name__ == "__main__":
    main()

