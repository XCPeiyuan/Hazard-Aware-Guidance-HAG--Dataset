#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估脚本（无 CIDEr 版）：
1) <SAFE/> 二分类准确率
2) 非 SAFE 样本：分别对 <ALERT> 与 <GUIDE> 计算 BERTScore(F1) 与 METEOR
3) 输出：detail_metrics.csv、summary.json、箱线图（PNG+PDF）

依赖：
  pip install evaluate bert-score matplotlib pandas
"""

import os
import re
import json
import math
import warnings
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

#Ckeck list - Qwen2vl-2b,Qwen2.5VL-32B
# ========== 可配置区域（按需修改） ==========
# 参考与预测文件（三行一组：图片名 / 文本 / 空行）
REF_TXT = "test_R_en.txt"  # 参考
PRED_TXT_DIR = "/carrot/zhup/[main]inference/Prompt-only/qwen25vl32b/r-0shot"  # 预测文件所在目录
PRED_TXT_NAME = "output_en.txt"  # 预测文件名
OUTPUT_DIR = "/carrot/zhup/[main]inference/Prompt-only/qwen25vl32b/r-0shot/eval"  # 评估输出目录

# 输出文件命名
CSV_PATH = os.path.join(OUTPUT_DIR, "detail_metrics.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")
BOXPLOT_ALERT_PNG = os.path.join(OUTPUT_DIR, "boxplot_ALERT.png")
BOXPLOT_GUIDE_PNG = os.path.join(OUTPUT_DIR, "boxplot_GUIDE.png")
BOXPLOT_ALERT_PDF = os.path.join(OUTPUT_DIR, "boxplot_ALERT.pdf")
BOXPLOT_GUIDE_PDF = os.path.join(OUTPUT_DIR, "boxplot_GUIDE.pdf")

# 图表参数
FIGSIZE = (7.5, 5.0)   # 单张图尺寸（英寸）
DPI = 300              # PNG 分辨率
FONTSIZE_BASE = 12     # 全局字体大小
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 尝试加载 evaluate
try:
    import evaluate
    _EVAL_OK = True
except Exception as e:
    _EVAL_OK = False
    warnings.warn(f"[evaluate] 未安装或加载失败，将无法计算 BERTScore/METEOR：{e}")

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
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n: break

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

def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S | re.I)
    return m.group(1).strip() if m else ""

def calc_bertscore(cands: List[str], refs: List[str]) -> List[float]:
    if not _EVAL_OK:
        return [math.nan] * len(cands)
    try:
        bert = evaluate.load("bertscore")
        res = bert.compute(predictions=cands, references=refs, lang="en")
        return [float(x) for x in res["f1"]]
    except Exception as e:
        warnings.warn(f"[BERTScore] 计算失败：{e}")
        return [math.nan] * len(cands)

def calc_meteor(cands: List[str], refs: List[str]) -> List[float]:
    if not _EVAL_OK:
        return [math.nan] * len(cands)
    try:
        mtr = evaluate.load("meteor")
        scores = []
        for c, r in zip(cands, refs):
            s = mtr.compute(predictions=[c], references=[r])["meteor"]
            scores.append(float(s))
        return scores
    except Exception as e:
        warnings.warn(f"[METEOR] 计算失败：{e}")
        return [math.nan] * len(cands)

def beautified_boxplot(values_list, labels, title, save_png, save_pdf):
    """
    更美观的箱线图（论文级）：
    - 更大的字体
    - 带均值点（●）与中位数线
    - 轻网格、须线、离群点样式
    - 紧凑布局，导出 PNG+PDF
    """
    plt.rcParams.update({
        "font.size": FONTSIZE_BASE,
        "axes.titlesize": FONTSIZE_BASE + 2,
        "axes.labelsize": FONTSIZE_BASE,
        "xtick.labelsize": FONTSIZE_BASE,
        "ytick.labelsize": FONTSIZE_BASE,
        "legend.fontsize": FONTSIZE_BASE,
    })

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)

    # 去 NaN
    data = [pd.Series(v).dropna().values for v in values_list]

    box = ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=False,
        whis=1.5,
        patch_artist=True,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=1.6),
        meanprops=dict(marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=4, alpha=0.6),
    )

    # 适度填充不同灰度以提升可视化层次（不花哨）
    fills = ["#e6e6e6", "#f2f2f2"]
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(fills[i % len(fills)])

    ax.set_title(title, pad=10)
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, axis="y")
    ax.set_axisbelow(True)

    # y 轴边界留白
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    ax.set_ylim(ymin - 0.02 * span, ymax + 0.02 * span)

    fig.tight_layout()
    fig.savefig(save_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(save_pdf, bbox_inches="tight")
    plt.close(fig)

def main():
    ref_list = parse_triplet_txt(REF_TXT)
    pred_list = parse_triplet_txt(os.path.join(PRED_TXT_DIR, PRED_TXT_NAME))

    if len(ref_list) != len(pred_list):
        warnings.warn(f"参考与预测样本数不一致：ref={len(ref_list)} vs pred={len(pred_list)}")

    N = min(len(ref_list), len(pred_list))
    if N == 0:
        print("没有可评估的样本。")
        return

    safe_correct = 0
    name_mismatch = 0

    # 为计算文本指标准备容器
    alert_refs, alert_preds = [], []
    guide_refs, guide_preds = [], []

    rows = []

    for i in range(N):
        ref_name, ref_text = ref_list[i]
        pred_name, pred_text = pred_list[i]

        if ref_name != pred_name:
            name_mismatch += 1

        ref_is_safe = is_safe(ref_text)
        pred_is_safe = is_safe(pred_text)
        if ref_is_safe == pred_is_safe:
            safe_correct += 1

        row = {
            "image": ref_name,
            "match_name": (ref_name == pred_name),
            "ref_is_safe": ref_is_safe,
            "pred_is_safe": pred_is_safe,
            "ref_alert": "",
            "pred_alert": "",
            "ref_guide": "",
            "pred_guide": "",
            "alert_bertscore": math.nan,
            "alert_meteor": math.nan,
            "guide_bertscore": math.nan,
            "guide_meteor": math.nan,
        }

        # 仅对双方均为非 SAFE 的样本计算文本
        if (not ref_is_safe) and (not pred_is_safe):
            ra = extract_tag(ref_text, "ALERT")
            rg = extract_tag(ref_text, "GUIDE")
            pa = extract_tag(pred_text, "ALERT")
            pg = extract_tag(pred_text, "GUIDE")

            row["ref_alert"] = ra
            row["ref_guide"] = rg
            row["pred_alert"] = pa
            row["pred_guide"] = pg

            alert_refs.append(ra)
            alert_preds.append(pa)
            guide_refs.append(rg)
            guide_preds.append(pg)

        rows.append(row)

    cls_acc = safe_correct / N

    # 计算文本指标（只有 BERTScore 与 METEOR）
    if len(alert_refs) > 0:
        alert_bert = calc_bertscore(alert_preds, alert_refs)
        alert_mtr = calc_meteor(alert_preds, alert_refs)
    else:
        alert_bert, alert_mtr = [], []

    if len(guide_refs) > 0:
        guide_bert = calc_bertscore(guide_preds, guide_refs)
        guide_mtr = calc_meteor(guide_preds, guide_refs)
    else:
        guide_bert, guide_mtr = [], []

    # 回填
    ai = gi = 0
    for r in rows:
        if (not r["ref_is_safe"]) and (not r["pred_is_safe"]):
            r["alert_bertscore"] = alert_bert[ai] if ai < len(alert_bert) else math.nan
            r["alert_meteor"] = alert_mtr[ai] if ai < len(alert_mtr) else math.nan
            ai += 1

            r["guide_bertscore"] = guide_bert[gi] if gi < len(guide_bert) else math.nan
            r["guide_meteor"] = guide_mtr[gi] if gi < len(guide_mtr) else math.nan
            gi += 1

    # 保存 CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    # 摘要
    def avg(col):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.mean()) if len(s) > 0 else math.nan

    summary = {
        "samples_total": int(N),
        "name_mismatch": int(name_mismatch),
        "safe_cls_accuracy": float(cls_acc),
        "ref_safe_count": int(df["ref_is_safe"].sum()),
        "pred_safe_count": int(df["pred_is_safe"].sum()),
        "both_non_safe_for_text_metrics": int(((~df["ref_is_safe"]) & (~df["pred_is_safe"])).sum()),
        "ALERT_avg": {
            "BERTScore_F1": avg("alert_bertscore"),
            "METEOR": avg("alert_meteor"),
        },
        "GUIDE_avg": {
            "BERTScore_F1": avg("guide_bertscore"),
            "METEOR": avg("guide_meteor"),
        },
        "detail_csv": CSV_PATH,
        "boxplot_ALERT_png": BOXPLOT_ALERT_PNG,
        "boxplot_GUIDE_png": BOXPLOT_GUIDE_PNG,
        "boxplot_ALERT_pdf": BOXPLOT_ALERT_PDF,
        "boxplot_GUIDE_pdf": BOXPLOT_GUIDE_PDF,
    }

    # 打印摘要
    print("==== Evaluation Summary ====")
    print(f"Total samples: {summary['samples_total']}")
    print(f"Image name mismatch count: {summary['name_mismatch']}")
    print(f"<SAFE/> classification accuracy: {summary['safe_cls_accuracy']:.4f}")
    print(f"Ref SAFE count: {summary['ref_safe_count']} | Pred SAFE count: {summary['pred_safe_count']}")
    print(f"Both non-SAFE (for text metrics): {summary['both_non_safe_for_text_metrics']}")
    print("ALERT avg -> BERTScore(F1): {B:.4f}, METEOR: {M:.4f}".format(
        B=(summary["ALERT_avg"]["BERTScore_F1"] if not math.isnan(summary["ALERT_avg"]["BERTScore_F1"]) else float('nan')),
        M=(summary["ALERT_avg"]["METEOR"] if not math.isnan(summary["ALERT_avg"]["METEOR"]) else float('nan')),
    ))
    print("GUIDE  avg -> BERTScore(F1): {B:.4f}, METEOR: {M:.4f}".format(
        B=(summary["GUIDE_avg"]["BERTScore_F1"] if not math.isnan(summary["GUIDE_avg"]["BERTScore_F1"]) else float('nan')),
        M=(summary["GUIDE_avg"]["METEOR"] if not math.isnan(summary["GUIDE_avg"]["METEOR"]) else float('nan')),
    ))

    # 保存摘要
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 画箱线图（分两张：ALERT / GUIDE）
    beautified_boxplot(
        values_list=[df["alert_bertscore"], df["alert_meteor"]],
        labels=["BERTScore(F1)", "METEOR"],
        title="ALERT Text Similarity Metrics",
        save_png=BOXPLOT_ALERT_PNG,
        save_pdf=BOXPLOT_ALERT_PDF,
    )
    beautified_boxplot(
        values_list=[df["guide_bertscore"], df["guide_meteor"]],
        labels=["BERTScore(F1)", "METEOR"],
        title="GUIDE Text Similarity Metrics",
        save_png=BOXPLOT_GUIDE_PNG,
        save_pdf=BOXPLOT_GUIDE_PDF,
    )

    print(f"明细已保存：{CSV_PATH}")
    print(f"摘要已保存：{SUMMARY_JSON}")
    print(f"箱线图已保存：{BOXPLOT_ALERT_PNG} / {BOXPLOT_GUIDE_PNG}（同时导出 PDF）")

if __name__ == "__main__":
    main()

