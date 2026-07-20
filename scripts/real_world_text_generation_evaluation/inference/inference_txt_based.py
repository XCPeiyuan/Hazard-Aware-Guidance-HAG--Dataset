# dataset inference code, processing for evaluation
import os
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_model(model_name: str):
    """
    加载模型和processor
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True
        # 如果你想强行用bf16 + flash_attn2，可以换成下面注释块
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def run_inference_on_image(model, processor, image_path: str, prompt: str, max_new_tokens: int = 128) -> str:
    """
    对单张图片进行推理，返回文本输出
    """
    # 读图
    image = Image.open(image_path).convert("RGB")

    # 把输入整理成Qwen3-VL需要的messages格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    # 处理成模型输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # 把tensor放到模型同一设备上 (GPU / CPU / 多卡自动分配)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 裁掉prompt本身，只保留新生成的内容
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # decode成文字
    output_text_list = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(output_text_list)

    # batch只有1张图，所以直接return第0个
    return output_text_list[0].strip()


def parse_image_names_from_triplet_txt(txt_path: str):
    """
    从 test_V_en.txt/test_V_zh.txt 这种“三行一条（图片名 / 文本 / 空行）”格式的文件中
    依次解析出图片文件名列表（只取每个样本的第一行）。
    """
    names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        # 跳过空行
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break

        # 第1行：图片名
        img_name = lines[i].strip()
        names.append(img_name)
        i += 1

        # 第2行：文本（跳过）
        if i < n:
            i += 1

        # 后续空行（跳过到下一个块）
        while i < n and lines[i].strip() == "":
            i += 1

    return names


def write_results_to_file(results, output_path: str):
    """
    results: list of (image_name, text_output)
    写入 output.txt
      第一行: 图片名
      第二行: 输出内容
      第三行: 空行(为了可读性)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for img_name, text_out in results:
            f.write(f"{img_name}\n")
            f.write(f"{text_out}\n")
            f.write("\n")  # 空行分隔


def main():
    # ===== 你可以在这里改参数 =====
    model_name = "/carrot/zhup/[main]model-final/(R)Real-Source/Qwen2.5-VL-7B-R-en"

    # 从哪个列表文件读取图片名（此处用英文版，亦可换中文）
    image_list_txt = "test_VR_en.txt"

    # 图片所在根目录（从列表文件中读出的文件名会拼到这个目录下查找）
    images_root = "/carrot/zhup/dataset/images"

    # 给模型的文字指令（prompt）
    prompt = (
        "You are a virtual assistant helping a blind person travel. The blind person is walking straight ahead. Please determine whether the path ahead is safe: If it is safe, output only: <SAFE/>; If there is hazard, output only (fixed format, concise, no additional content): <ALERT>Describe the obstacle, its direction, and approximate distance</ALERT><GUIDE>Provide clear instructions for avoidance actions</GUIDE>"
    )
    # 输出文件
    output_txt = "/carrot/zhup/[main]inference/R/qwen25vl7b/r-vr/output_en.txt"
    # ===== 参数到此结束 =====

    # 1. 加载模型
    print("Loading model...")
    model, processor = load_model(model_name)

    # 2. 从列表文件读取图片名并在 images_root 中查找
    img_names = parse_image_names_from_triplet_txt(image_list_txt)
    if not img_names:
        print(f"未在列表文件中解析到任何图片名: {image_list_txt}")
        return

    image_files = []
    missing = []
    for nm in img_names:
        full_path = os.path.join(images_root, nm)
        if os.path.isfile(full_path):
            image_files.append(full_path)
        else:
            missing.append(nm)

    if missing:
        print(f"[警告] 有 {len(missing)} 张图片在目录中未找到（将被跳过）:")
        for m in missing[:20]:
            print("  -", m)
        if len(missing) > 20:
            print("  ...(略)")

    if not image_files:
        print(f"在 {images_root} 中未找到任何有效图片。")
        return

    # 3. 循环推理 + 进度条
    results = []
    print(f"Start inference on {len(image_files)} images (from list: {image_list_txt}) ...")
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            text_out = run_inference_on_image(
                model=model,
                processor=processor,
                image_path=img_path,
                prompt=prompt,
                max_new_tokens=128,
            )
        except Exception as e:
            text_out = f"[ERROR] 推理失败: {e}"

        img_name_only = os.path.basename(img_path)
        results.append((img_name_only, text_out))

    # 4. 写入结果
    write_results_to_file(results, output_txt)
    print(f"完成。结果已写入 {output_txt}")


if __name__ == "__main__":
    main()

