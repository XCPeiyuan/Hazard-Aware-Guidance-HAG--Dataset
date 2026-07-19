#!/usr/bin/env python3
"""Batch dataset-json inference for the merged WalkVLM Qwen2.5-VL model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on transformers version
    Qwen2_5_VLForConditionalGeneration = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models/Qwen2.5-VL-7B-WAD-Alter-Only"
DEFAULT_DATASET_JSON = ROOT / "test/walkvlm_qwen25vl_test_alter.json"
DEFAULT_PROMPT = """<image>\nYou are a navigation assistant for a visually impaired pedestrian.\nBased only on the image, provide one concise real-time walking instruction or warning."""


def load_dataset_images(dataset_json: Path) -> list[Path]:
    with dataset_json.open("r", encoding="utf-8") as f:
        records = json.load(f)

    image_paths: list[Path] = []
    for record in records:
        images = record.get("images") or []
        if images:
            image_paths.append(Path(images[0]).expanduser())

    return image_paths


def format_image_label(image_path: Path, dataset_json: Path, label_format: str) -> str:
    if label_format == "auto":
        label_format = "basename" if dataset_json.stem == "our_test" else "parent_basename"
    if label_format == "basename":
        return image_path.name
    return f"{image_path.parent.name}/{image_path.name}"


def build_messages(image_path: Path, prompt: str) -> list[dict]:
    text = prompt.replace("<image>", "").strip()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": text},
            ],
        }
    ]


def load_model(model_dir: Path, device: str):
    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )

    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Qwen2_5_VLForConditionalGeneration is unavailable in this transformers install.")

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    load_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if device.startswith("cuda"):
        load_kwargs["device_map"] = {"": device}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def infer(processor, model, image_path: Path, prompt: str, device: str, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = build_messages(image_path, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = inputs.to(device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    output = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return " ".join(output.split())


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch WalkVLM image inference for a dataset JSON.")
    parser.add_argument("dataset_json", type=Path, nargs="?", default=DEFAULT_DATASET_JSON, help="LLaMA-Factory dataset JSON containing image paths.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--label-format",
        choices=("auto", "basename", "parent_basename"),
        default="auto",
        help="Output image label format. Auto uses basename for our_test.json and parent/basename otherwise.",
    )
    args = parser.parse_args()

    dataset_json = args.dataset_json.expanduser()
    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model}")

    output_path = args.output or (dataset_json.parent / "output.txt")
    images = load_dataset_images(dataset_json)

    print(f"model: {args.model}")
    print(f"pid: {os.getpid()}")
    print(f"device: {args.device}")
    print(f"dataset_json: {dataset_json}")
    print(f"images: {len(images)}")
    print(f"output: {output_path}")
    print(f"label format: {args.label_format}")
    print("prompt source: fixed image-only prompt; dataset user content is ignored")

    processor, model = load_model(args.model, args.device)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, image_path in enumerate(images, start=1):
            try:
                result = infer(processor, model, image_path, DEFAULT_PROMPT, args.device, args.max_new_tokens)
            except Exception as exc:
                result = f"error: {type(exc).__name__}: {exc}"
                result = " ".join(result.split())

            image_label = format_image_label(image_path, dataset_json, args.label_format)
            f.write(f"{image_label}\n{result}\n\n")
            f.flush()
            print(f"[{idx}/{len(images)}] {image_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
