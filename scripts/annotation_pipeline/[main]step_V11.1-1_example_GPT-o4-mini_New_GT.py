# step1-4,V11-lite 逐步标注 + 新GT注入 + ALERT二次润色
# 仅保留标注部分；移除了无关项；在改动处增加中文注释
#标注开始前47.48美元，标注910张图片。
#标注结束后 29.03美元。
#平均每张 0.02美元。

#注意，此代码在prompt中使用了总结自问卷的例子

import os
import base64
import io
from PIL import Image
import torch
import json
from openai import OpenAI
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import shutil

DEBUG = True
DEBUG_plus = False
Visual_prompt_ablation_study = False

MODEL = "o4-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your_api_key_here!( •̀ ω •́ )✧"))

def backup_files(mode):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if mode == "all":
        files_to_check = ["output.json", "log.txt", "log_all.txt"]
    else:
        files_to_check = []

    backup_folder = os.path.join(current_dir, "backup")
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for filename in files_to_check:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name, ext = os.path.splitext(filename)
            new_filename = f"{base_name}_{timestamp}{ext}"
            new_file_path = os.path.join(backup_folder, new_filename)
            shutil.move(file_path, new_file_path)
            print(f"已备份 {filename} -> {new_file_path}")

def encode_image(image_path, max_width=1920, max_height=1080, format="JPEG", quality=80):
    try:
        img = Image.open(image_path)
    except IOError:
        return "Error"
    img = img.convert("RGB")
    w, h = img.size
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        nw, nh = int(w * scale), int(h * scale)
        print(f"Downscaling {os.path.basename(image_path)} from {w}x{h} to {nw}x{nh}")
        img = img.resize((nw, nh), Image.LANCZOS)
    else:
        print(f"No downscaling needed for {os.path.basename(image_path)}: {w}x{h}")
    buf = io.BytesIO()
    img.save(buf, format=format, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_json_with_direction(json_path):

    #解析 JSON，构造 history 文本( •̀ ω •́ )✧
    #读取 JSON，返回 img_size, models以及 history_str（GT）
    #models: [name, distance, (x,y), category, direction]
    #👇
    #例：障碍物名称: Carts , 距离: 4.92 m, 分类: Common_Obstacle , 方向: -11.8
 
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    camera_params = data.get("cameraParameters", {})
    img_size = (camera_params.get("imageWidth"), camera_params.get("imageHeight"))

    models = []
    pieces = []  # 用于拼接 history_str
    for m in data.get("models", []):
        name = m.get("name")
        dist = round(m.get("horizontal_distance", 0), 2)
        img_pos = tuple(int(round(v)) for v in m.get("image_position", [0, 0]))
        category = m.get("category")
        direction = round(m.get("direction", 0), 1)
        models.append([name, dist, img_pos, category, direction])

        #history（GT-prompt）
        pieces.append(
            f"障碍物名称: {name} , 距离: {dist} m, 分类: {category} , 方向: {direction}"
        )

    history_str = " ".join(pieces)
    return img_size, models, history_str

def process_image_via_api(image_path, step, history_str):
    base64_image = encode_image(image_path)
    if base64_image == "Error":
        return "IMG Error"
    history = history_str if history_str is not None else ""
    if DEBUG:
        print(f"[DEBUG] step={step}, history:{history}")
    #step == 3 ALERT润色（调用时可传入VP图或原图）
    if step == "2":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text":
                 """SSI障碍描述和盲道状态如下：
                 """ + history + """

                 参照图片，检查并润色上述场景描述。你需要自行判断优先提醒哪些障碍物，最终不超过3个障碍物且少于约45字。如果多条描述对应相邻的多个障碍物，请将其整合。在信息完整的前提下保持简洁，使视觉障碍者能快速、全面地理解信息。

                 格式如下：方向+距离（大约多少步）+障碍描述，例如：……方向约x步有……

                 图片中圆形标记的颜色含义如下：
                 1. 绿色代表普通坐落在地面上的障碍物，分类为"Common Obstacle"。
                 2. 红色代表深坑等具有高低差、容易导致盲人坠落受伤的障碍，分类为"Pitfall Hazard"。
                 3. 蓝色代表盲人不易察觉、容易造成头部碰撞的悬空障碍物，分类为"Upper-body Hazard"。

                 图片中圆形标记内的数字含义如下：
                 1. 数字"1"代表障碍物距离较近，需要优先提醒。
                 2. 数字"2"代表障碍物在中等距离，可以次要提醒。
                 3. 数字"3"代表障碍物在较远距离，可以视情况提醒。

                 ALERT示例模板：
                 例子1
                 适用情况：悬空障碍物
                 危险类别：Upper-body Hazard
                 输出模板：{方向}约x步有悬空的{障碍物}，小心头部

                 例子2
                 适用情况：卡车尾部等突出障碍物
                 危险类别：Upper-body Hazard
                 输出模板：{方向}约x步有突出的{障碍物}，小心磕碰

                 例子3
                 适用情况：深坑或其他具有高低差的危险
                 危险类别：Pitfall Hazard
                 输出模板：{方向}约x步有{障碍物}，当心跌落

                 例子4
                 适用情况：施工场地危险
                 危险类别：Pitfall Hazard
                 输出模板：已经进入施工场地，{方向}约x步有{障碍物}

                 例子5
                 适用情况：向上或向下楼梯
                 危险类别：Common Obstacle
                 输出模板：{方向}约x步有{向上/向下}的楼梯

                 例子6
                 适用情况：车辆或其他地面普通障碍物
                 危险类别：Common Obstacle
                 输出模板：{方向}约x步有{障碍物}

                 最终必须保证视觉障碍者能根据描述了解障碍物的位置，并尽量使用常用语。不要进行其他输出，不要换行，只输出一句话。可以使用逗号分句，但不能使用分号进行分条或使用括号注释。"""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    #step == 2_new 使用VP图和SSI生成障碍物描述
    elif step == "2_new":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": """
                    结构化场景信息：
                    """ + history + """

                    你是使用中文帮助盲人出行的虚拟志愿者。当前图片由盲人的第一人称视角拍摄，盲人正在向正前方行走，前方存在会对盲人造成阻碍甚至危险的障碍物或情况。主要障碍物已在图中标出。

                    图片中圆形标记的颜色含义如下：
                    1. 绿色代表普通坐落在地面上的障碍物，分类为"Common Obstacle"。
                    2. 红色代表深坑等具有高低差、容易导致盲人坠落受伤的障碍，分类为"Pitfall Hazard"。
                    3. 蓝色代表盲人不易察觉、容易造成头部碰撞的悬空障碍物，分类为"Upper-body Hazard"。

                    图片中圆形标记内的数字含义如下：
                    1. 数字"1"代表障碍物距离较近，需要优先提醒。
                    2. 数字"2"代表障碍物在中等距离，可以次要提醒。
                    3. 数字"3"代表障碍物在较远距离，可以视情况提醒。

                    ALERT示例模板：
                    1. 适用情况：悬空障碍物；危险类别：Upper-body Hazard；输出模板：{方向}约x步有悬空的{障碍物}，小心头部
                    2. 适用情况：卡车尾部等突出障碍物；危险类别：Upper-body Hazard；输出模板：{方向}约x步有突出的{障碍物}，小心磕碰
                    3. 适用情况：深坑或其他具有高低差的危险；危险类别：Pitfall Hazard；输出模板：{方向}约x步有{障碍物}，当心跌落
                    4. 适用情况：施工场地危险；危险类别：Pitfall Hazard；输出模板：已经进入施工场地，{方向}约x步有{障碍物}
                    5. 适用情况：向上或向下楼梯；危险类别：Common Obstacle；输出模板：{方向}约x步有{向上/向下}的楼梯
                    6. 适用情况：车辆或其他地面普通障碍物；危险类别：Common Obstacle；输出模板：{方向}约x步有{障碍物}

                    结合结构化场景信息和图片，简要描述障碍物的大致位置（正前方/正前方略偏左/正前方略偏右/左前方/右前方/左侧/右侧）、距离（步数，约0.6米为1步）和障碍物详情。
                    格式为：方向+距离（大约多少步）+障碍描述，例如：……方向约x步有……
                    可能存在多个障碍物，因此允许多次使用此格式。

                    请严格遵循以下原则：
                    1. 使用简洁语言，避免任何多余描述。
                    2. 不要使用“前方有多个障碍”等模糊表述，不要解释。
                    3. 所有距离与方位必须结合图片和结构化信息判断。
                    4. 优先考虑正前方障碍物，视情况次要考虑侧边障碍物。
                    5. 不要照搬示例，需根据实际场景调整。
                    6. 只输出一句话，不要换行，不要添加双引号。
                """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    #step == 2 描述当前整体环境和左右两侧，只给GUIDE使用
    elif step == "environment":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": """你正在根据一张以第一人称视角拍摄的图片描述当前环境。
                    请用一句话简要描述当前整体场景，并说明图片左侧和右侧分别有什么。
                    要求：
                    1. 只输出一句话。
                    2. 先简要说明当前环境或整体场景。
                    3. 再说明图片左侧和右侧可见的内容。
                    4. 使用简洁、常用的语言。
                    5. 不要解释，不要列举，不要添加其他句子。
                    输出格式：你正处于……区域，左侧是……，右侧是……"""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    #step == 1 盲道判断（使用原图）
    elif step == "3":
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": """你是一个帮助盲人出行的虚拟志愿者。
                     请根据下列图片描述判断当前场景中是否存在“盲道”（专为盲人设置的步行通道），以及盲道的位置。
                     请严格遵守以下规则，输出结果必须完全匹配下列格式，不得添加其他文字或标点符号，不要进行其他输出，不要换行：
                     1.如果场景中存在盲道，且该盲道从图片底部正中央向前延伸，说明盲人正沿盲道行走，请输出："盲人正在盲道上行走"
                     2.如果场景中存在盲道，但不符合第1种情况，请判断盲道在图片中的位置判断其相对于盲人的位置（例如在画面的左侧、右侧等），并输出格式为："存在盲道，盲人未在盲道上行走，盲道在盲人的[左侧/右侧/左前方/右前方]" （其中，请根据实际情况选择最合适的描述）
                     3.如果场景中没有盲道，请输出："当前场景中不存在盲道"。
                 """}
            ]
        }]
    #step == 4 GUIDE生成，有Visual Prompt
    elif step == "4":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "场景、盲道状态和ALERT如下：" + history + """
                 你是帮助盲人出行的虚拟志愿者。请结合上述完整场景描述、盲道状态、ALERT和图片，为盲人提供简洁的避障指引。主要障碍物已在图中标出。

                 图片中圆形标记的颜色含义如下：
                 1. 绿色代表普通坐落在地面上的障碍物，分类为"Common Obstacle"。
                 2. 红色代表具有高低差的危险，分类为"Pitfall Hazard"。
                 3. 蓝色代表容易造成头部碰撞的障碍物，分类为"Upper-body Hazard"。

                 GUIDE示例模板：
                 1. 向{方向}移动x步。
                 2. 先向{方向1}移动x步，然后向{方向2}移动y步。

                 要求：
                 1.只输出指引本身，不要解释，不要换行。
                 2.避让策略需要避免盲人在躲避障碍后陷入更危险的境地。
                 3.避让路径优先向人行道进行避让，例如：若两条避让路径中一条为向人行道方向避让，一条为向车道方向上避让，则优先选择向人行道方向避让的路径。
                 4.在保证2与3的前提下，优先向更空旷的方向进行避让。
                 5.若当前盲人脚下是车道，而避让指引是上人行道行走，则之后就不需要“穿越障碍物后回到原先路线”的指引。
                 6.若没有绕行障碍物后回到原路径的需求（即不需要在避让障碍物后回到原先的路线），则不需要在向某方向避让后输出“向前直行多少步”的指引，避免迷惑盲人。
                 7.不要进行让盲人转身多少度的指引，盲人对多少度没有概念，这么做反而会迷惑盲人。
                 8.用步数(取约0.6m为一步)来指代推荐盲人避让的距离。"""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    #step == 4 GUIDE生成，但是没有Visual Prompt
    elif step == "5":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "场景、盲道状态和ALERT如下：" + history + """
                 你是帮助盲人出行的虚拟志愿者。请结合上述完整场景描述、盲道状态、ALERT和原始图片，为盲人提供简洁的避障指引。

                 GUIDE示例模板：
                 1. 向{方向}移动x步。
                 2. 先向{方向1}移动x步，然后向{方向2}移动y步。

                 要求：
                 1.只输出指引本身，不要解释，不要换行。
                 2.避让策略需要避免盲人在躲避障碍后陷入更危险的境地。
                 3.避让路径优先向人行道进行避让，例如：若两条避让路径中一条为向人行道方向避让，一条为向车道方向上避让，则优先选择向人行道方向避让的路径。
                 4.在保证2与3的前提下，优先向更空旷的方向进行避让。
                 5.若当前盲人脚下是车道，而避让指引是上人行道行走，则之后就不需要“穿越障碍物后回到原先路线”的指引。
                 6.若没有绕行障碍物后回到原路径的需求（即不需要在避让障碍物后回到原先的路线），则不需要在向某方向避让后输出“向前直行多少步”的指引，避免迷惑盲人。
                 7.不要进行让盲人转身多少度的指引，盲人对多少度没有概念，这么做反而会迷惑盲人。
                 8.用步数(取约0.6m为一步)来指代推荐盲人避让的距离。
                 """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    #ALERT消融评估，调用时传入原图
    elif step == "judge_alert":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": """你是帮助盲人出行的虚拟志愿者。请结合原始图片比较以下两个危险提醒候选，选择更安全、更准确且更简洁的一项。
                    仅输出1或2，不要输出原文、解释或其他内容。
                    候选和相关信息如下：
                    """ + history},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    #GUIDE消融评估，调用时传入原图
    elif step == "judge_guide":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": """你是帮助盲人出行的虚拟志愿者。请结合原始图片比较以下两个避障指引候选，选择更安全、更清晰且更容易执行的一项。
                    仅输出1或2，不要输出原文、解释或其他内容。
                    候选和相关信息如下：
                    """ + history},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    #LAVE评估哪一个更好
    elif step == "6":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": """\n你是帮助盲人出行的虚拟志愿者，当前图片是由盲人的第一人称视角拍摄的图片，盲人正在手持盲杖向正前方行走，前方存在会对盲人造成阻碍甚至危险的障碍物/场景。
                 以下为两位志愿者针对该场景提供的避障指导，你的任务是结合图片对两者进行评估，然后输出你认为内容更正确、格式更合适的一方的编号。仅输出1、2两个数字编号的其中一个，不要输出原文，也不要进行任何其他输出。避障指导如下：
                 """+ history},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    elif step == "7":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "请用一段话为盲人描述这张图片，不要换行。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    else:
        raise ValueError(f"Invalid step: {step}")

    try:
        response = client.chat.completions.create(model=MODEL, messages=messages)
        output_text = response.choices[0].message.content or ""
        if DEBUG:
            with open("log_all.txt", "a", encoding="utf-8") as log_all:
                log_all.write(f"\n步骤【{step}】\n历史:{history}\n输出:{output_text}\n")
    except Exception as e:
        print(f"API call error: {e}")
        output_text = "API Error"

    #清理
    if step in ("3","2","2_new","environment","4","7","5"):
        cleaned = (output_text.strip().replace('\n',''))
        if DEBUG: print("Step:",step," [DEBUG]cleaned_output:", cleaned)
        return cleaned
    elif step in ("6","judge_alert","judge_guide"):
        cleaned = (output_text.strip().replace('\n','').replace('，','').replace('。','').replace(',','').replace('.',''))
        if DEBUG: print("Step:",step," [DEBUG]cleaned_output:", cleaned)
        return cleaned
    else:
        if DEBUG: print("Step:",step," [DEBUG]output_text:", output_text)
        return output_text

#log.txt写入json
def write_2_json():
    with open('log.txt', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    data = []
    for i in range(0, len(lines), 2):
        image_filename = lines[i]
        assistant_content = lines[i+1]
        entry = {
            "messages": [
                {"content": "<image>你是一名协助盲人出行的虚拟志愿者。盲人正朝正前方行走。请根据场景判断前方是否安全：如果安全，只输出：<SAFE/>；如果存在危险，只输出（格式固定、简洁明了、不得添加其他内容）：<ALERT>描述障碍物、方向和大致距离</ALERT><GUIDE>给出明确避让的动作指令</GUIDE>", "role": "user"},
                {"content": assistant_content, "role": "assistant"}
            ],
            "images": ["images/" + image_filename]
        }
        data.append(entry)
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


#step 2_new → step 2 → step 4
def anno_old(vp_folder, json_folder):
    print("- 调用API -")
    log_file = "log.txt"
    allowed_ext = (".jpg", ".jpeg", ".png")

    #消融研究用计数
    #ALERT标注，是否使用Visual Prompt
    abs_alert_vp = 0
    abs_alert_wovp = 0
    #GUIDE标注，是否使用Visual Prompt
    abs_guide_vp = 0
    abs_guide_wovp = 0
    

    image_files = sorted([f for f in os.listdir(vp_folder) if f.lower().endswith(allowed_ext)])

    with open(log_file, "w", encoding="utf-8") as log:
        for image_file in tqdm(image_files, desc="Processing images", unit="image"):
            if DEBUG:
                with open("log_all.txt", "a", encoding="utf-8") as log_all:
                    log_all.write(f"\n========================\n图片：{image_file}\n")

            image_path = os.path.join(vp_folder, image_file)
            original_image_path = os.path.join(json_folder, image_file)

            base = os.path.splitext(image_file)[0]
            json_path = os.path.join(json_folder, base + ".json")
            if not os.path.exists(json_path):
                tqdm.write(f"[SKIP] 未找到同名 JSON：{json_path}")
                continue

            img_size, models, history_gt = parse_json_with_direction(json_path)

            tqdm.write(f"Processing {image_file}")
            print(f"Processing {image_file} ...")

            # 去掉“危险/安全”的外部判定部分，先直接走危险分支流水线
            result_tag = "危险"

            if "危险" in result_tag:
                #step == 1 使用原图判断盲道
                blind_alley = process_image_via_api(original_image_path, "3", None)

                #step == 2 使用原图描述当前环境和左右两侧，只给GUIDE使用
                environment_description = process_image_via_api(original_image_path, "environment", None)

                #step == 2_new 只输入已经画好VP的图片和SSI，不输入原图
                result_alert_raw = process_image_via_api(image_path, "2_new", history_gt)
                if DEBUG:
                    print("[DEBUG] 未润色ALERT: result_alert_raw:", result_alert_raw)

                #是否进行消融研究
                if Visual_prompt_ablation_study:
                    #step == 3 分别使用VP图和原图润色ALERT
                    result_alert_vp = process_image_via_api(
                        image_path,
                        "2",
                        result_alert_raw + "；盲道情况：" + blind_alley
                    )
                    result_alert_wovp = process_image_via_api(
                        original_image_path,
                        "2",
                        result_alert_raw + "；盲道情况：" + blind_alley
                    )

                    #使用原图评估两个ALERT，1为使用VP，2为不使用VP
                    judge_alert_num = process_image_via_api(
                        original_image_path,
                        "judge_alert",
                        "SSI信息：" + history_gt +
                        "\nSSI障碍描述：" + result_alert_raw +
                        "\n盲道情况：" + blind_alley +
                        "\n1：" + result_alert_vp +
                        "\n2：" + result_alert_wovp
                    )
                    if DEBUG:
                        print("[DEBUG] ALERT Judge评估：", judge_alert_num)

                    if judge_alert_num == "1":
                        result_alert_polished = result_alert_vp
                        print("【ALERT-1】使用VP者更优")
                        abs_alert_vp += 1
                    elif judge_alert_num == "2":
                        result_alert_polished = result_alert_wovp
                        print("【ALERT-2】不使用VP者更优")
                        abs_alert_wovp += 1
                    else:
                        raise ValueError(f"judge_alert 输出必须严格为1或2，实际为：{judge_alert_num!r}")

                    #环境描述接上SSI障碍描述，形成完整scene description
                    scene_description = environment_description + " " + result_alert_raw

                    #step == 4 分别使用VP图和原图生成GUIDE
                    result_guide_vp = process_image_via_api(
                        image_path,
                        "4",
                        scene_description + "；盲道情况：" + blind_alley + "；ALERT：" + result_alert_polished
                    )
                    if DEBUG:
                        print("[DEBUG] GUIDE(visual prompt)：result_guide:", result_guide_vp)

                    result_guide_wovp = process_image_via_api(
                        original_image_path,
                        "5",
                        scene_description + "；盲道情况：" + blind_alley + "；ALERT：" + result_alert_polished
                    )
                    if DEBUG:
                        print("[DEBUG] GUIDE：result_guide:", result_guide_wovp)

                    #使用原图评估两个GUIDE，1为使用VP，2为不使用VP
                    judge_guide_num = process_image_via_api(
                        original_image_path,
                        "judge_guide",
                        "SSI信息：" + history_gt +
                        "\n完整场景描述：" + scene_description +
                        "\n盲道情况：" + blind_alley +
                        "\n已选ALERT：" + result_alert_polished +
                        "\n1：" + result_guide_vp +
                        "\n2：" + result_guide_wovp
                    )
                    if DEBUG:
                        print("[DEBUG] GUIDE Judge评估：", judge_guide_num)

                    if judge_guide_num == "1":
                        result_guide = result_guide_vp
                        print("【GUIDE-1】使用VP者更优")
                        abs_guide_vp += 1
                    elif judge_guide_num == "2":
                        result_guide = result_guide_wovp
                        print("【GUIDE-2】不使用VP者更优")
                        abs_guide_wovp +=1
                    else:
                        raise ValueError(f"judge_guide 输出必须严格为1或2，实际为：{judge_guide_num!r}")

                else:
                    #step == 3 正常模式下使用VP图润色ALERT
                    result_alert_polished = process_image_via_api(
                        image_path,
                        "2",
                        result_alert_raw + "；盲道情况：" + blind_alley
                    )
                    if DEBUG:
                        print("[DEBUG] 润色ALERT：result_alert_polished:", result_alert_polished)

                    #环境描述接上SSI障碍描述，形成完整scene description
                    scene_description = environment_description + " " + result_alert_raw

                    #step == 4 正常模式下使用VP图生成GUIDE
                    result_guide = process_image_via_api(
                        image_path,
                        "4",
                        scene_description + "；盲道情况：" + blind_alley + "；ALERT：" + result_alert_polished
                    )
                    if DEBUG:
                        print("[DEBUG] GUIDE(visual prompt)：result_guide:", result_guide)

                #写入log
                log.write(f"{image_file}\n<ALERT>{result_alert_polished}</ALERT><GUIDE>{result_guide}</GUIDE>\n\n")
                tqdm.write(f"Result for {image_file}: <ALERT>{result_alert_polished}</ALERT><GUIDE>{result_guide}</GUIDE>")
                continue

            elif "安全" in result_tag:
                #需要时可改成用 step==7 的描述
                result_safe = process_image_via_api(image_path, "7", None)
                log.write(f"{image_file}\n<SAFE>{result_safe}\n\n")
                tqdm.write(f"Result for {image_file}: <SAFE>{result_safe}")
                continue

            tqdm.write("⚠result 错误")
            log.write(f"{image_file}\n⚠{result_tag}\n\n")

        #消融研究最后计数    
        if Visual_prompt_ablation_study:
            print("消融研究-VP in ALERT(w:w/o):",abs_alert_vp,":",abs_alert_wovp)
            print("消融研究-VP in GUIDE(w:w/o):",abs_guide_vp,":",abs_guide_wovp)


def main():
    vp_folder   = r"D:\Dataset\Visual_prompt"      #Visual prompt 
    json_folder = r"D:\Dataset\Unity-based_sources"     #JSON+ original image

    if not os.path.exists(vp_folder):
        raise FileNotFoundError(f"⚠路径不存在：{vp_folder}")
    if not os.path.exists(json_folder):
        raise FileNotFoundError(f"⚠路径不存在：{json_folder}")

    backup_files("all")
    anno_old(vp_folder, json_folder)
    write_2_json()

if __name__ == "__main__":
    main()
