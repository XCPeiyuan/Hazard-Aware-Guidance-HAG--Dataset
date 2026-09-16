#python annotate_pointA_images.py "C:/Users/99681/OneDrive/桌面/test" "C:/Users/99681/OneDrive/桌面/test-pointA" --marker-color 255 0 0 --radius 5 --font-size 14
#原json文件的坐标是左下角为00的

#Road_Track Wall Curbstone Road_Angle C:/Users/99681/OneDrive/桌面/test 

import os
import argparse
import json
from PIL import Image, ImageDraw, ImageFont


def annotate_images(input_dir, output_dir, marker_color=(255, 0, 0), marker_radius=5, font_size=14):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Try to load a default truetype font; fallback to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Iterate through JSON files in input_dir
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.json'):
            continue
        base = os.path.splitext(fname)[0]
        json_path = os.path.join(input_dir, fname)

        # Determine image path (common extensions)
        found_image = None
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = os.path.join(input_dir, base + ext)
            if os.path.exists(img_path):
                found_image = img_path
                break
        if not found_image:
            print(f"Warning: no image found for {json_path}")
            continue

        # Load JSON and image
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        image = Image.open(found_image).convert('RGB')
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size

        # Annotate each model point
        for model in data.get('models', []):
            pt = model.get('pointA_image')
            name = model.get('name', '')
            if not pt or len(pt) < 2:
                continue
            x, y_unity = pt[0], pt[1]

            # Unity uses bottom-left origin, PIL uses top-left. Convert Y coordinate.
            y = img_height - y_unity

            # Draw circle at (x, y)
            left_up = (x - marker_radius, y - marker_radius)
            right_down = (x + marker_radius, y + marker_radius)
            draw.ellipse([left_up, right_down], outline=marker_color, width=2)

            # Draw label text slightly above/right of marker
            text_pos = (x + marker_radius + 2, y - marker_radius - 2)
            draw.text(text_pos, name, fill=marker_color, font=font)

        # Save annotated image
        output_path = os.path.join(output_dir, os.path.basename(found_image))
        image.save(output_path)
        print(f"Annotated image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Annotate images by marking pointA_image coordinates from matching JSON files'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing paired .json and image files'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to save annotated images'
    )
    parser.add_argument(
        '--marker-color',
        nargs=3,
        type=int,
        default=[255, 0, 0],
        help='RGB color for marker and text (default: 255 0 0)'
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Radius of the circular marker in pixels'
    )
    parser.add_argument(
        '--font-size',
        type=int,
        default=14,
        help='Font size for text labels'
    )
    args = parser.parse_args()

    annotate_images(
        args.input_dir,
        args.output_dir,
        tuple(args.marker_color),
        args.radius,
        args.font_size
    )

if __name__ == '__main__':
    main()
