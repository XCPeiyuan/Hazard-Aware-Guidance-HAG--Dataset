# 此代码仅为展示demo使用，并非在标注中使用
import os
import json
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont

INPUT_DIR  = r"C:\Users\99681\OneDrive\桌面\test"          # input，包含json和对应图片
OUTPUT_DIR = r"C:\Users\99681\OneDrive\桌面\test-point"     # output，生成demo
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

POINT_RADIUS = 6
POINT_COLOR = (255, 0, 0)       # 红点
TEXT_COLOR  = (255, 255, 255)   # 白字
BOX_COLOR   = (0, 0, 0, 160)    # 半透明黑底
LINE_SPACING = 2                # 文本行距
FONT_SIZE = 18                  # 文本字号
FONT_PATH: Optional[str] = None # None 使用默认字体

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def find_image_for_json(json_path: str, json_data: dict) -> Optional[str]:
    folder = os.path.dirname(json_path)
    imgname = json_data.get("imageName")
    if isinstance(imgname, str):
        candidate = os.path.join(folder, imgname)
        if os.path.isfile(candidate):
            return candidate
        root, ext = os.path.splitext(candidate)
        if ext == "":
            for e in IMAGE_EXTS:
                p = root + e
                if os.path.isfile(p):
                    return p

    prefix = os.path.splitext(os.path.basename(json_path))[0]
    for e in IMAGE_EXTS:
        p = os.path.join(folder, prefix + e)
        if os.path.isfile(p):
            return p

    return None

def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        if FONT_PATH and os.path.isfile(FONT_PATH):
            return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        pass
    return ImageFont.load_default()

def flip_unity_y_to_image_y(y_unity: float, image_height: int) -> float:
    #左下角为(0,0)，因此纵轴需要换算一下
    return image_height - y_unity

def draw_dot(draw: ImageDraw.ImageDraw, center: Tuple[float, float], r: int, color):
    x, y = center
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=None)

def get_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    #PS:兼容 Pillow 新旧版本的文字尺寸计算
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return w, h
    except AttributeError:
        return draw.textsize(text, font=font)

def draw_label_box(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: Tuple[float, float],
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    text_color=TEXT_COLOR,
    box_color=BOX_COLOR,
    padding: int = 4
):
    widths = []
    heights = []
    for line in lines:
        w, h = get_text_size(draw, line, font)
        widths.append(w)
        heights.append(h)
    box_w = max(widths) + padding * 2
    box_h = sum(heights) + padding * 2 + LINE_SPACING * (len(lines) - 1)

    x, y = origin
    W, H = img.size

    if x + box_w > W:
        x = max(0, W - box_w)
    if y + box_h > H:
        y = max(0, H - box_h)

    overlay = Image.new("RGBA", (box_w, box_h), box_color)
    img.paste(overlay, (int(x), int(y)), overlay)

    tx = x + padding
    ty = y + padding
    for i, line in enumerate(lines):
        draw.text((tx, ty), line, fill=text_color, font=font)
        ty += heights[i] + LINE_SPACING

def annotate_one(json_path: str, out_dir: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_path = find_image_for_json(json_path, data)
    if not img_path or not os.path.isfile(img_path):
        print(f"[WARN] 找不到匹配图片: {json_path}")
        return

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = load_font(FONT_SIZE)

    models = data.get("models", [])
    if not isinstance(models, list):
        print(f"[WARN] 'models' 非列表: {json_path}")
        return

    for m in models:
        try:
            name = str(m.get("name", ""))

            category = str(m.get("category", ""))

            dist = m.get("horizontal_distance", None)
            dist_str = "N/A"
            if isinstance(dist, (int, float)):
                dist_str = f"{dist:.2f} m"

            direction_val = m.get("direction", None)
            if isinstance(direction_val, (int, float)):
                direction_str = f"{direction_val:.2f}"
            else:
                direction_str = "N/A"

            pos = m.get("image_position", None)
            if (
                not isinstance(pos, (list, tuple)) or
                len(pos) < 2 or
                pos[0] is None or pos[1] is None
            ):
                continue

            x_unity, y_unity = float(pos[0]), float(pos[1])
            if x_unity < 0 or y_unity < 0:
                continue

            x_img = x_unity
            y_img = flip_unity_y_to_image_y(y_unity, H)

            x_img = max(0, min(W - 1, x_img))
            y_img = max(0, min(H - 1, y_img))

            draw_dot(draw, (x_img, y_img), POINT_RADIUS, POINT_COLOR)

            lines = [name, dist_str]
            if category:
                lines.append(category)
            lines.append(f"direction: {direction_str}")

            draw_label_box(
                img, draw,
                (x_img + 8, y_img - 8 - FONT_SIZE),  
                lines, font
            )
        except Exception as e:
            print(f"[ERROR] 绘制某个 model 失败：{e}")

    # 保存
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    img.save(out_path)
    print(f"[OK] {os.path.basename(json_path)} -> {os.path.basename(out_path)}")

def main():
    ensure_dir(OUTPUT_DIR)
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".json")]
    files.sort()
    if not files:
        print("[INFO] 未在输入目录找到任何 .json 文件。")
        return

    for jf in files:
        annotate_one(os.path.join(INPUT_DIR, jf), OUTPUT_DIR)

if __name__ == "__main__":
    main()
