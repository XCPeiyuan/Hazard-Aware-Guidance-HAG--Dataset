#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对 txt 标注文件（例如 train_V_en.txt）进行类别标注：
- 输入格式：三行一组
    1) 图片名
    2) 标注文本（<SAFE/> 或 <ALERT>...</ALERT><GUIDE>...</GUIDE>）
    3) 空行
- 对于 <SAFE/>：直接标记为 {"objects": []}
- 对于非 SAFE：
    只取 ALERT 内的内容，交给 Qwen3-14B (thinking 模式) 分类：
    每个提及的物体 -> ["GROUND", "PIT", "OVERHEAD"] 多标签分类

输出：
- 一个 JSON 文件，例如 train_V_en_category.json
  形如：
  [
    {
      "image": "IMG_20250829_171931552.png",
      "objects": [
        {"name": "bicycle", "categories": ["GROUND"]}
      ]
    },
    ...
  ]
"""

import os
import re
import json
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ========== 配置区域 ==========
INPUT_TXT = "test_R_en.txt"
OUTPUT_JSON = "test_R_en_category.json"

MODEL_NAME = "Qwen/Qwen3-14B"
MAX_NEW_TOKENS = 2048
MAX_RETRY = 5

# Qwen3-A3B 系列的 </think> token id，官方示例中是 151668
THINK_END_TOKEN_ID = 151668
# =============================


def parse_triplet_txt(path: str) -> List[Tuple[str, str]]:
    """
    解析三行一组：
      1) 图片名
      2) 文本
      3) 空行
    返回 [(image_name, content), ...]
    """
    items: List[Tuple[str, str]] = []
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


def extract_alert(text: str) -> str:
    """从标注文本中抽取 <ALERT>... </ALERT> 的内容"""
    m = re.search(r"<ALERT>(.*?)</ALERT>", text, flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def build_prompt(alert_text: str) -> str:
    """
    构造给 LLM 的英文 prompt：
    输入：ALERT 文本
    输出：要求 LLM 返回 {"objects":[...]} 的 JSON
    """
    prompt = f"""
You are an expert annotator for blind pedestrian navigation.

You will be given the ALERT text describing obstacles in the scene.
Your task is to extract every mentioned object that is an obstacle or relevant element,
and assign one or more categories to each object.

There are exactly three possible categories:

1. "GROUND":
   - Ground-level obstacles such as road barriers, vehicles, bicycles, pedestrians, boxes, cones, poles on the ground, etc.
   - Includes steps and stairs. Stairs belong to "GROUND" (not "PIT").

2. "PIT":
   - Pits, holes, gaps, deep drops, missing ground, trenches, or any vertical drop that could cause falling.
   - Anything that can make the blind person fall down or step into a hole.

3. "OVERHEAD":
   - Overhead obstacles such as low signs, low branches, beams, scaffolding bars, or anything hanging in the air
     that may hit the upper body or head of the blind person.

Important rules:

- Only use these three category strings: "GROUND", "PIT", "OVERHEAD".
- Do NOT invent objects that are not mentioned in the ALERT text.
- For each object mentioned in the ALERT text, create one entry with:
    - "name": a short phrase identifying the object (e.g., "bicycle", "barrier and cones", "overhead scaffolding beam")
    - "categories": a list of one or more of ["GROUND", "PIT", "OVERHEAD"].
- If an object is clearly not a hazard (pure background), you can ignore it.
- If there are no hazard objects in the ALERT text, return an empty list for "objects".
- Note that for objects like billboards, flags, and telegraph poles, you need to determine whether they are GROUND or OVERHEAD based on their description. Only objects explicitly stated to be suspended should be classified as OVERHEAD; The 'steps' corresponding to the number of steps taken are not objects; the word 'steps' corresponding to stairs is; For formats like 'A and B', A and B should be separated, rather than categorized as a single object.

Output format requirements:

- You MUST output a single valid JSON object.
- The JSON must have exactly one top-level key: "objects".
- "objects" must be a list of object entries.
- Do NOT output any explanation, comments, or extra text outside the JSON.

Example of the required JSON format:

{{
  "objects": [
    {{"name": "bicycle", "categories": ["GROUND"]}},
    {{"name": "open manhole", "categories": ["PIT"]}},
    {{"name": "overhead scaffolding beam", "categories": ["OVERHEAD"]}}
  ]
}}

Now process the following ALERT text and output only the JSON:

[ALERT_TEXT]
{alert_text}
"""
    return prompt.strip()


def load_model_and_tokenizer(model_name: str):
    """加载 Qwen3 模型和 tokenizer"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model


def call_model_once(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """
    调用一次 Qwen3 (thinking 模式)，返回 content 字符串
    （已经去掉 thinking 部分）
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # 开启思维模式
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # 分离 thinking 与最终内容
    try:
        # 从末尾向前找到 </think> 的位置（THINK_END_TOKEN_ID）
        index = len(output_ids) - 1 - output_ids[::-1].index(THINK_END_TOKEN_ID)
        thinking_ids = output_ids[:index]
        content_ids = output_ids[index:]
    except ValueError:
        # 没有找到 </think>，则全部视为 content
        thinking_ids = []
        content_ids = output_ids

    # thinking_content 可以忽略，不需要使用
    _ = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip("\n")

    content = tokenizer.decode(content_ids, skip_special_tokens=True).strip("\n")
    return content


def try_parse_category_json(text: str) -> Dict[str, Any] | None:
    """
    从模型输出中尝试解析 JSON，并进行结构验证：
    - 顶层必须是 {"objects": [...]}
    - "objects" 是 list
    - 每个元素有 "name"(str) 和 "categories"(list[str])
    - categories 必须是 ["GROUND", "PIT", "OVERHEAD"] 中的子集
    """
    # 去掉可能的代码块标记等
    # 找第一个 '{' 和最后一个 '}'，截取中间部分
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start:end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    if "objects" not in data:
        return None

    objs = data["objects"]
    if not isinstance(objs, list):
        return None

    allowed_cats = {"GROUND", "PIT", "OVERHEAD"}

    cleaned_objs = []
    for obj in objs:
        if not isinstance(obj, dict):
            return None
        name = obj.get("name")
        cats = obj.get("categories")
        if not isinstance(name, str):
            return None
        if not isinstance(cats, list):
            return None

        # 清洗 categories
        new_cats = []
        for c in cats:
            if isinstance(c, str):
                c2 = c.strip().upper()
                if c2 in allowed_cats:
                    new_cats.append(c2)
        # 可以允许 new_cats 为空（表示这个物体没分类好），也可以直接丢弃这个对象
        # 这里选择：如果为空则丢弃
        if not new_cats:
            # 忽略这个对象
            continue

        cleaned_objs.append({
            "name": name.strip(),
            "categories": new_cats
        })

    # 允许 objects 为空列表
    data["objects"] = cleaned_objs
    return data


def main():
    # 1) 读取 txt
    items = parse_triplet_txt(INPUT_TXT)
    print(f"Loaded {len(items)} samples from {INPUT_TXT}")

    # 2) 加载模型
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

    results = []

    for idx, (img, content) in enumerate(items, start=1):
        text = content.strip()
        print(f"\n[{idx}/{len(items)}] Processing {img}")

        # SAFE 情况：直接空 objects
        if text == "<SAFE/>":
            print("  -> SAFE sample, no hazards. Mark objects = [].")
            results.append({
                "image": img,
                "objects": []
            })
            continue

        # 提取 ALERT 文本
        alert_text = extract_alert(text)
        if not alert_text:
            print("  [WARN] No <ALERT> found, treat as no objects.")
            results.append({
                "image": img,
                "objects": []
            })
            continue

        prompt = build_prompt(alert_text)

        data = None
        for attempt in range(1, MAX_RETRY + 1):
            print(f"  Attempt {attempt}/{MAX_RETRY} ...")
            raw_output = call_model_once(tokenizer, model, prompt)
            print("  Raw model output:")
            print("  --------------------------")
            print(raw_output)
            print("  --------------------------")

            data = try_parse_category_json(raw_output)
            if data is not None:
                print(f"  -> Parsed JSON successfully with {len(data['objects'])} objects.")
                break
            else:
                print("  [WARN] Failed to parse valid JSON, will retry.")

        if data is None:
            print("  [ERROR] Could not get valid JSON after retries, fallback to empty objects.")
            data = {"objects": []}

        results.append({
            "image": img,
            "objects": data["objects"]
        })

    # 3) 保存 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved {len(results)} records to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

