# dataset inference code, processing for evaluation, using prompt engineering without fine-tuned model
import os
from tqdm import tqdm
from PIL import Image

import base64
from openai import OpenAI
client = OpenAI(api_key="your_api_key")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def run_inference_on_image(model, image_path: str, prompt: str) -> str:
    """
    对单张图片进行推理，返回文本输出
    """
    print('Processing...')

    image_base64 = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ],
            }
        ]
    )

    # batch只有1张图，所以直接return第0个
    text_out = response.choices[0].message.content
    print('Output--',text_out)
    return text_out



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
    model_name = "gpt-4o"

    # 从哪个列表文件读取图片名（此处用英文版，亦可换中文）
    image_list_txt = "test_R_en.txt"

    # 图片所在根目录（从列表文件中读出的文件名会拼到这个目录下查找）
    images_root = "/carrot/zhup/dataset/images"

    # 给模型的文字指令（prompt）
    prompt = (
        "You are a virtual assistant helping a blind person travel. The blind person is walking straight ahead. Please determine whether the path ahead is safe: If it is safe, output only: <SAFE/>; If there is hazard, output only (fixed format, concise, no additional content): <ALERT>Describe the obstacle, its direction, and approximate distance</ALERT><GUIDE>Provide clear instructions for avoidance actions</GUIDE>. You can refer to the following examples format for output: '<ALERT>Bicycle about 8 steps ahead to the right. Red barrier with multiple cones and signs about 13 steps ahead to the left.</ALERT><GUIDE>Walk straight, avoid the bicycle obstacle to the right, then continue along the sidewalk.</GUIDE>','<ALERT>Barrier across the tactile paving about 6 steps ahead. Boxes stacked about 1 step to the right of the barrier.</ALERT><GUIDE>Walk along the left edge of the tactile paving for 7 steps to bypass the boxes and barrier ahead.</GUIDE>'"
    )
    # 输出文件
    output_txt = "/carrot/zhup/[main]inference/Prompt-only/gpt4o/r/output_en.txt"
    # ===== 参数到此结束 =====

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
                model=model_name,
                image_path=img_path,
                prompt=prompt
            )
            # 打印成功结果
            tqdm.write(f"{os.path.basename(img_path)}: {text_out}")
        except Exception as e:
            tqdm.write(f"[ERROR] {img_path} 推理失败: {e}")
            text_out = f"[ERROR] 推理失败: {e}"

        img_name_only = os.path.basename(img_path)
        results.append((img_name_only, text_out))


    # 4. 写入结果
    write_results_to_file(results, output_txt)
    print(f"完成。结果已写入 {output_txt}")


if __name__ == "__main__":
    main()

