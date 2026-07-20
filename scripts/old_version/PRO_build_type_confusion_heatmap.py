#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于物体级匹配（LLM）计算类别混淆矩阵，并绘制热力图（百分比、蓝白渐变）。

输入：
- PRED_JSON: 模型生成的 category.json
- REF_JSON : 参考 category.json

输出：
- confusion_matrix_percent.json （可选）
- heatmap_category_confusion.png
保存到 PRED_JSON 同目录。
"""

import os
import json
import re
from typing import Dict, List, Any, Tuple, Optional

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ================== 配置区 ==================
PRED_JSON = "/carrot/zhup/[main]inference/V/qwen25vl7b/v-vr/output_en_category.json"
REF_JSON = "./test_VR_en_category.json"

MODEL_NAME = "Qwen/Qwen3-14B"
MAX_NEW_TOKENS = 1024
MAX_RETRY_MATCH = 5

# 类别顺序（行=GT, 列=Pred）
CAT_ORDER = ["SAFE", "OVERHEAD", "PIT", "GROUND"]
CAT_DISPLAY = {
    "SAFE": "Safe",
    "OVERHEAD": "Upper-body Hazard",
    "PIT": "Pitfall Hazard",
    "GROUND": "Common Obstacle",
}

THINK_END_TOKEN_ID = 151668  # 兼容；non-thinking 模式一般不会输出 think，但留着无害
# ==========================================


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"{path} is not a list json."
    return data


def build_image_dict(data_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    转成 image -> objects 列表 的 dict
    objects: [{"name":..., "categories":[...]}]
    """
    d = {}
    for item in data_list:
        img = item.get("image")
        objs = item.get("objects", [])
        if img is None:
            continue
        if not isinstance(objs, list):
            objs = []
        d[img] = objs
    return d


def normalize_categories(cats: List[str]) -> List[str]:
    """清洗类别字符串到标准集合"""
    allowed = set(CAT_ORDER) - {"SAFE"}
    out = []
    for c in cats:
        if not isinstance(c, str):
            continue
        c2 = c.strip().upper()
        if c2 in allowed:
            out.append(c2)
    # 去重但保序
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            res.append(x); seen.add(x)
    return res


def load_model_and_tokenizer():
    print(f"Loading matcher LLM: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    print("Matcher model loaded.")
    return tokenizer, model


def build_match_prompt(gt_objs: List[Dict[str, Any]],
                       pred_objs: List[Dict[str, Any]]) -> str:
    """
    让 LLM 做物体级配对。
    输出格式：
    {
      "pairs": [
        {"gt_index": 0, "pred_index": 1},
        ...
      ]
    }
    规则：
    - 一个 gt_index 最多匹配一个 pred_index
    - 一个 pred_index 最多匹配一个 gt_index
    - 只返回你认为最可能对应的配对
    """
    def fmt_objs(objs):
        # 只给 name + categories
        out = []
        for i, o in enumerate(objs):
            name = str(o.get("name", "")).strip()
            cats = normalize_categories(o.get("categories", []))
            out.append({"index": i, "name": name, "categories": cats})
        return out

    gt_clean = fmt_objs(gt_objs)
    pred_clean = fmt_objs(pred_objs)

    prompt = f"""
You are an expert matcher for obstacle mentions.

You will be given two lists of objects described in text:
1) Ground Truth (GT) objects
2) Model Predicted (PRED) objects

Each object has:
- index
- name (short phrase)
- categories (one or more of OVERHEAD / PIT / GROUND)

Your task:
Match GT objects to PRED objects that refer to the same real-world obstacle.
Use ONLY the object names and context implied by categories.
Do NOT invent new objects. Do NOT match one object to multiple objects.

If a GT object and a PRED object have the **same or highly similar names**,  
you MUST treat them as referring to the **same physical obstacle**,  
even if their categories are different.

Name similarity takes priority over category similarity.

Therefore:
- If the names match (or clearly refer to the same object),  
  **you must match them** and include the pair (gt_index, pred_index)  
  in the output.
- Category differences should NOT prevent matching.
- You do NOT judge category correctness; you only judge whether two names  
  refer to the same obstacle.

In summary:
**Name similarity overrides category differences.**
If names match, ALWAYS match the objects.

Output a SINGLE valid JSON object with key "pairs":
- "pairs" is a list of {{ "gt_index": int, "pred_index": int }}
- One gt_index can appear at most once.
- One pred_index can appear at most once.
- If no objects match, output an empty list.

Example:
{{
  "pairs": [
    {{"gt_index": 0, "pred_index": 1}},
    {{"gt_index": 2, "pred_index": 0}}
  ]
}}

Now match these two lists:

[GT_OBJECTS]
{json.dumps(gt_clean, ensure_ascii=False)}

[PRED_OBJECTS]
{json.dumps(pred_clean, ensure_ascii=False)}

Return JSON only, one line, no extra text.
"""
    return prompt.strip()


def call_match_llm(tokenizer, model, prompt: str) -> str:
    """
    non-thinking 调用：enable_thinking=False
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    out_ids = gen_ids[0][len(inputs.input_ids[0]):].tolist()

    # 如果模型意外输出 thinking token，也做个切分
    try:
        idx = len(out_ids) - 1 - out_ids[::-1].index(THINK_END_TOKEN_ID)
        content_ids = out_ids[idx:]
    except ValueError:
        content_ids = out_ids

    return tokenizer.decode(content_ids, skip_special_tokens=True).strip()


def try_parse_pairs(raw: str, gt_len: int, pred_len: int) -> Optional[List[Tuple[int, int]]]:
    """
    解析 LLM 输出 pairs，并做结构和范围校验。
    返回 [(gt_i, pred_i), ...] 或 None
    """
    s = raw.find("{")
    e = raw.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    js = raw[s:e+1]

    try:
        data = json.loads(js)
    except Exception:
        return None
    if not isinstance(data, dict) or "pairs" not in data:
        return None
    pairs = data["pairs"]
    if not isinstance(pairs, list):
        return None

    used_gt = set()
    used_pred = set()
    out_pairs = []

    for p in pairs:
        if not isinstance(p, dict):
            return None
        gi = p.get("gt_index")
        pi = p.get("pred_index")
        if not isinstance(gi, int) or not isinstance(pi, int):
            return None
        if gi < 0 or gi >= gt_len or pi < 0 or pi >= pred_len:
            return None
        if gi in used_gt or pi in used_pred:
            return None
        used_gt.add(gi); used_pred.add(pi)
        out_pairs.append((gi, pi))

    return out_pairs


def categories_of_obj(obj: Dict[str, Any]) -> List[str]:
    """获取单个 object 的 hazard 类别列表（不含 SAFE）"""
    cats = normalize_categories(obj.get("categories", []))
    return cats


def add_confusion_counts(
    confusion: np.ndarray,
    gt_objs: List[Dict[str, Any]],
    pred_objs: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]]
):
    """
    根据 pairs 计数：
    - 匹配到的：GT类别 -> Pred类别（逐类别）
    - 未匹配 GT：GT类别 -> SAFE
    - 未匹配 Pred：SAFE -> Pred类别
    注意：对象可能多标签，逐标签展开计数。
    """
    gt_matched = set(gi for gi, _ in pairs)
    pred_matched = set(pi for _, pi in pairs)

    # 1) matched pairs
    for gi, pi in pairs:
        gt_cats = categories_of_obj(gt_objs[gi])
        pred_cats = categories_of_obj(pred_objs[pi])

        # 如果 pred_cats 为空，视为 SAFE
        if not pred_cats:
            pred_cats = ["SAFE"]

        if not gt_cats:
            gt_cats = ["SAFE"]

        # ★ 核心改动：匹配成功后，Pred 类别一律按 GT 类别算
        pred_cats = gt_cats

        for gc in gt_cats:
            for pc in pred_cats:
                r = CAT_ORDER.index(gc if gc in CAT_ORDER else "SAFE")
                c = CAT_ORDER.index(pc if pc in CAT_ORDER else "SAFE")
                confusion[r, c] += 1


    # 2) unmatched GT -> SAFE
    for gi, obj in enumerate(gt_objs):
        if gi in gt_matched:
            continue
        gt_cats = categories_of_obj(obj)
        if not gt_cats:
            gt_cats = ["SAFE"]
        for gc in gt_cats:
            r = CAT_ORDER.index(gc if gc in CAT_ORDER else "SAFE")
            c = CAT_ORDER.index("SAFE")
            confusion[r, c] += 1

    # 3) unmatched Pred : SAFE -> pred_cat
    for pi, obj in enumerate(pred_objs):
        if pi in pred_matched:
            continue
        pred_cats = categories_of_obj(obj)
        if not pred_cats:
            pred_cats = ["SAFE"]
        for pc in pred_cats:
            r = CAT_ORDER.index("SAFE")
            c = CAT_ORDER.index(pc if pc in CAT_ORDER else "SAFE")
            confusion[r, c] += 1


def plot_heatmap_percent(confusion: np.ndarray, save_path: str):
    """
    画百分比混淆热力图（行归一化），蓝白渐变，只显示百分比。
    """
    # 行归一化：每个 GT 类别作为分母
    row_sums = confusion.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        percent = np.divide(confusion, row_sums) * 100.0
        percent = np.nan_to_num(percent)
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(percent, cmap=plt.cm.Blues, vmin=0, vmax=100)

    # x/y tick labels
    labels = [CAT_DISPLAY[c] for c in CAT_ORDER]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    # 写百分比（不写计数）
    for i in range(percent.shape[0]):
        for j in range(percent.shape[1]):
            ax.text(j, i, f"{percent[i, j]:.1f}%", ha="center", va="center", fontsize=10)

    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Ground-truth Category")
    ax.set_title("Hazard Category Confusion Matrix")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage (%)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # 1) load json
    pred_list = load_json_list(PRED_JSON)
    ref_list = load_json_list(REF_JSON)

    pred_dict = build_image_dict(pred_list)
    ref_dict = build_image_dict(ref_list)

    # 2) load matcher LLM
    tokenizer, model = load_model_and_tokenizer()

    # 3) init confusion counts
    confusion = np.zeros((len(CAT_ORDER), len(CAT_ORDER)), dtype=np.int64)

    images = list(pred_dict.keys())
    print(f"Total pred samples: {len(images)}")

    # 4) per-image matching + counting
    for idx, img in enumerate(images, start=1):
        pred_objs = pred_dict.get(img, [])
        gt_objs = ref_dict.get(img, [])

        # 如果参考里没有该图，跳过
        if img not in ref_dict:
            print(f"[WARN] {img} not found in reference, skip.")
            continue

        # 两边都无物体 => SAFE->SAFE 计一次
        if (not gt_objs) and (not pred_objs):
            confusion[0, 0] += 1
            continue

        # 只有一边有物体 => 无需 LLM 匹配
        if not gt_objs:
            # SAFE -> pred_cat
            for obj in pred_objs:
                for pc in categories_of_obj(obj) or ["SAFE"]:
                    confusion[0, CAT_ORDER.index(pc if pc in CAT_ORDER else "SAFE")] += 1
            continue
        if not pred_objs:
            # gt_cat -> SAFE
            for obj in gt_objs:
                for gc in categories_of_obj(obj) or ["SAFE"]:
                    confusion[CAT_ORDER.index(gc if gc in CAT_ORDER else "SAFE"), 0] += 1
            continue

        # 需要 LLM 匹配
        prompt = build_match_prompt(gt_objs, pred_objs)

        pairs = None
        for attempt in range(1, MAX_RETRY_MATCH + 1):
            raw = call_match_llm(tokenizer, model, prompt)
            print(f"\n[{idx}/{len(images)}] {img} attempt {attempt}")
            print("---- MATCH RAW OUTPUT ----")
            print(raw)
            print("--------------------------")

            pairs = try_parse_pairs(raw, len(gt_objs), len(pred_objs))
            if pairs is not None:
                break
            print("  [WARN] Invalid pairs JSON, retrying...")

        if pairs is None:
            print(f"  [ERROR] Matching failed for {img}, fallback to no pairs.")
            pairs = []

        add_confusion_counts(confusion, gt_objs, pred_objs, pairs)

    # 5) save confusion percent json (optional)
    out_dir = os.path.dirname(PRED_JSON)
    os.makedirs(out_dir, exist_ok=True)

    row_sums = confusion.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        percent = np.divide(confusion, row_sums) * 100.0
        percent = np.nan_to_num(percent)

    percent_json_path = os.path.join(out_dir, "confusion_matrix_percent.json")
    with open(percent_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "cat_order": CAT_ORDER,
            "matrix_count": confusion.tolist(),
            "matrix_percent_row_norm": percent.tolist()
        }, f, ensure_ascii=False, indent=2)

    # 6) plot heatmap
    heatmap_path = os.path.join(out_dir, "heatmap_category_confusion.png")
    plot_heatmap_percent(confusion, heatmap_path)

    print("\nDone.")
    print(f"Saved percent json: {percent_json_path}")
    print(f"Saved heatmap png : {heatmap_path}")


if __name__ == "__main__":
    main()

