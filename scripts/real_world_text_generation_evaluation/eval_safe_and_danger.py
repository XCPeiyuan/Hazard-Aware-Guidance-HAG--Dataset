#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
补充分析脚本：
只计算：
  1) Danger 的召回率（有危险时能否识别出来）
  2) SAFE 的精确率（说 SAFE 时是不是大多真安全）

输入格式与原脚本完全一致（三行一组：图片名 / 文本 / 空行）。
输出：
  - detail_safe_metrics.csv：逐样本 SAFE/DANGER 分类情况
  - summary_safe_metrics.json：整体统计与指标

依赖：
  pip install pandas
"""

import os
import json
import math
from typing import List, Tuple

import pandas as pd

# ========== 可配置区域（按需修改，与原脚本保持一致） ==========
# 参考与预测文件（三行一组：图片名 / 文本 / 空行）
REF_TXT = "test_R_en.txt"  # 参考
PRED_TXT_DIR = "/carrot/zhup/[main]inference/Prompt-only/qwen25vl7b/r-0shot"  # 预测文件所在目录
PRED_TXT_NAME = "output_en.txt"  # 预测文件名

# 结果输出到与 output.txt 同路径下的 eval-safe 文件夹中
SAFE_OUTPUT_DIR = os.path.join(PRED_TXT_DIR, "eval-safe")

DETAIL_CSV = os.path.join(SAFE_OUTPUT_DIR, "detail_safe_metrics.csv")
SUMMARY_JSON = os.path.join(SAFE_OUTPUT_DIR, "summary_safe_metrics.json")
# =====================================================


def parse_triplet_txt(path: str) -> List[Tuple[str, str]]:
    """
    解析“三行一组”（图片名 / 文本 / 空行）格式
    返回 [(img_name, content), ...]
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i, n = 0, len(lines)
    while i < n:
        # 跳过空行
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

        # 跳过分隔空行
        while i < n and lines[i].strip() == "":
            i += 1

        if img:
            items.append((img, content))
    return items


def is_safe(text: str) -> bool:
    """判断文本是否为严格的 <SAFE/>"""
    return text.strip() == "<SAFE/>"


def main():
    os.makedirs(SAFE_OUTPUT_DIR, exist_ok=True)

    ref_list = parse_triplet_txt(REF_TXT)
    pred_list = parse_triplet_txt(os.path.join(PRED_TXT_DIR, PRED_TXT_NAME))

    if len(ref_list) != len(pred_list):
        print(f"[警告] 参考与预测样本数不一致：ref={len(ref_list)} vs pred={len(pred_list)}")

    N = min(len(ref_list), len(pred_list))
    if N == 0:
        print("没有可评估的样本。")
        return

    # 混淆矩阵统计
    # 对 SAFE 视角：
    #   - TP_safe: ref_safe=True & pred_safe=True
    #   - FP_safe: ref_safe=False & pred_safe=True
    #   - FN_safe: ref_safe=True & pred_safe=False
    #   - TN_safe: ref_safe=False & pred_safe=False
    tp_safe = fp_safe = fn_safe = tn_safe = 0

    # 对 Danger 视角（把“危险”当正类）：
    #   - TP_danger: ref_non_safe=True & pred_non_safe=True
    #   - FN_danger: ref_non_safe=True & pred_safe=True
    #   - FP_danger: ref_safe=True & pred_non_safe=True
    #   - TN_danger: ref_safe=True & pred_safe=True
    tp_danger = fp_danger = fn_danger = tn_danger = 0

    # 名称不匹配计数
    name_mismatch = 0

    rows = []

    for i in range(N):
        ref_name, ref_text = ref_list[i]
        pred_name, pred_text = pred_list[i]

        if ref_name != pred_name:
            name_mismatch += 1

        ref_is_safe = is_safe(ref_text)
        pred_is_safe = is_safe(pred_text)

        # SAFE 视角的混淆矩阵
        if ref_is_safe and pred_is_safe:
            tp_safe += 1
            tn_danger += 1   # 对 danger 视角来说，这是 TN
            cls_label = "TP_SAFE"
        elif (not ref_is_safe) and pred_is_safe:
            fp_safe += 1
            fn_danger += 1   # 参考危险但预测 SAFE -> danger FN
            cls_label = "FP_SAFE_FN_DANGER"
        elif ref_is_safe and (not pred_is_safe):
            fn_safe += 1
            fp_danger += 1   # 参考 SAFE 但预测危险 -> danger FP
            cls_label = "FN_SAFE_FP_DANGER"
        else:  # (not ref_is_safe) and (not pred_is_safe)
            tn_safe += 1
            tp_danger += 1   # 对 danger 视角来说，这是 TP
            cls_label = "TN_SAFE_TP_DANGER"

        rows.append({
            "image": ref_name,
            "match_name": (ref_name == pred_name),
            "ref_is_safe": ref_is_safe,
            "pred_is_safe": pred_is_safe,
            "cls_label": cls_label,
        })

    # 计算指标
    # SAFE 精确率
    denom_safe_prec = tp_safe + fp_safe
    if denom_safe_prec > 0:
        safe_precision = tp_safe / denom_safe_prec
    else:
        safe_precision = math.nan

    # Danger 召回率
    denom_danger_rec = tp_danger + fn_danger
    if denom_danger_rec > 0:
        danger_recall = tp_danger / denom_danger_rec
    else:
        danger_recall = math.nan

    # 一些额外指标（可选）
    total_correct = tp_safe + tn_safe  # SAFE 视角下的准确分类
    cls_accuracy = total_correct / N if N > 0 else math.nan

    # 参考/预测 SAFE 样本数量
    ref_safe_count = sum(1 for r in rows if r["ref_is_safe"])
    pred_safe_count = sum(1 for r in rows if r["pred_is_safe"])

    # 保存明细 CSV
    df = pd.DataFrame(rows)
    df.to_csv(DETAIL_CSV, index=False, encoding="utf-8")

    # 汇总 JSON
    summary = {
        "samples_total": int(N),
        "name_mismatch": int(name_mismatch),

        # SAFE 视角混淆矩阵
        "SAFE_confusion_matrix": {
            "TP_safe": int(tp_safe),
            "FP_safe": int(fp_safe),
            "FN_safe": int(fn_safe),
            "TN_safe": int(tn_safe),
        },

        # Danger 视角混淆矩阵
        "Danger_confusion_matrix": {
            "TP_danger": int(tp_danger),
            "FP_danger": int(fp_danger),
            "FN_danger": int(fn_danger),
            "TN_danger": int(tn_danger),
        },

        # 关注的两个核心指标
        "SAFE_precision": float(safe_precision),
        "Danger_recall": float(danger_recall),

        # 额外信息（可选）
        "overall_cls_accuracy": float(cls_accuracy),
        "ref_safe_count": int(ref_safe_count),
        "pred_safe_count": int(pred_safe_count),

        "detail_csv": DETAIL_CSV,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印结果
    print("==== SAFE / DANGER Evaluation (补充分析) ====")
    print(f"Total samples: {N}")
    print(f"Image name mismatch count: {name_mismatch}")
    print(f"Ref SAFE count: {ref_safe_count} | Pred SAFE count: {pred_safe_count}")
    print("")
    print("SAFE confusion matrix:")
    print(f"  TP_safe: {tp_safe}, FP_safe: {fp_safe}, FN_safe: {fn_safe}, TN_safe: {tn_safe}")
    print("Danger confusion matrix (treat 'dangerous' as positive):")
    print(f"  TP_danger: {tp_danger}, FP_danger: {fp_danger}, FN_danger: {fn_danger}, TN_danger: {tn_danger}")
    print("")
    print(f"SAFE precision: {safe_precision:.4f}" if not math.isnan(safe_precision) else "SAFE precision: NaN (no predicted SAFE samples)")
    print(f"Danger recall: {danger_recall:.4f}" if not math.isnan(danger_recall) else "Danger recall: NaN (no reference dangerous samples)")
    print(f"Overall classification accuracy (SAFE vs DANGER): {cls_accuracy:.4f}")
    print("")
    print(f"明细已保存：{DETAIL_CSV}")
    print(f"摘要已保存：{SUMMARY_JSON}")


if __name__ == "__main__":
    main()

