#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import List, Dict, Any

def has_cjk(text: str) -> bool:
    """判断文本中是否包含中文（CJK）字符。"""
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def get_assistant_content(messages: List[Dict[str, Any]]) -> str:
    """从 messages 中取 role=='assistant' 的 content。若有多个，仅取第一个。"""
    for m in messages:
        if m.get("role") == "assistant":
            return str(m.get("content", "")).strip()
    return ""

def get_image_basename(images: List[str]) -> str:
    """从 images 数组中取第一项的文件名（去掉路径）。"""
    if not images:
        return ""
    return os.path.basename(str(images[0]).strip())

def main():
    parser = argparse.ArgumentParser(description="Split VIP dataset into EN/ZH txt lists.")
    parser.add_argument("--input", default="VIP_dataset_V_full_train.json", help="输入 JSON 文件路径")
    parser.add_argument("--out_en", default="train_V_en.txt", help="英文输出文件名")
    parser.add_argument("--out_zh", default="train_V_zh.txt", help="中文输出文件名")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    en_lines: List[str] = []
    zh_lines: List[str] = []

    # 统计计数
    en_total = zh_total = 0
    en_safe = zh_safe = 0
    en_alert = zh_alert = 0

    for item in data:
        messages = item.get("messages", [])
        images = item.get("images", [])

        content = get_assistant_content(messages).strip()
        imgname = get_image_basename(images)

        if not imgname or not content:
            continue

        block = f"{imgname}\n{content}\n\n"

        # 情况1：SAFE —— 同时写入中英文两个文件并分别计数
        if content == "<SAFE/>":
            en_lines.append(block)
            zh_lines.append(block)
            en_total += 1
            zh_total += 1
            en_safe += 1
            zh_safe += 1
            continue

        # 情况2：ALERT/GUIDE —— 依据 assistant 文本是否含中文来分流
        if has_cjk(content):
            zh_lines.append(block)
            zh_total += 1
            zh_alert += 1
        else:
            en_lines.append(block)
            en_total += 1
            en_alert += 1

    # 写文件
    with open(args.out_en, "w", encoding="utf-8", newline="\n") as f_en:
        f_en.writelines(en_lines)
    with open(args.out_zh, "w", encoding="utf-8", newline="\n") as f_zh:
        f_zh.writelines(zh_lines)

    # 打印统计
    print("==== Summary ====")
    print(f"English -> total: {en_total}, SAFE: {en_safe}, ALERT+GUIDE: {en_alert}, output: {args.out_en}")
    print(f"Chinese -> total: {zh_total}, SAFE: {zh_safe}, ALERT+GUIDE: {zh_alert}, output: {args.out_zh}")

if __name__ == "__main__":
    main()

