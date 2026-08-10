import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from tqdm import tqdm
import time

def insert_spaces(string, nSpace):
    if nSpace == 0:
        return string
    return (" " * nSpace).join(string)

def rotate_coordinates(poly, width, height, angle_degrees):
    """将多边形坐标根据图像旋转角度进行变换（如果需要）"""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    poly = poly.reshape(-1, 2)
    new_poly = np.zeros_like(poly)
    
    cx, cy = width / 2, height / 2
    
    for i, (x, y) in enumerate(poly):
        x_c = x - cx
        y_c = y - cy
        x_new = x_c * cos_a + y_c * sin_a
        y_new = -x_c * sin_a + y_c * cos_a
        new_poly[i] = [x_new + cx, y_new + cy]
    
    return new_poly

def draw_glyph2(font, text, polygon, line_angle, vertAng=15, scale=1, width=512, height=512, add_space=True, color=(255, 255, 255, 255)):
    enlarge_polygon = polygon * scale
    rect = cv2.minAreaRect(enlarge_polygon)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # 使用 np.intp 代替 np.int0
    w, h = rect[1]
    
    # 直接使用 line["angle"]，单位为度
    angle = -line_angle  # 负号因为 PIL 旋转方向与 OpenCV 相反

    # 强制水平排版
    vert = False
    if abs(line_angle) > 0.1:  # angle != 0
        canvas_width = (width + height) * scale  
        canvas_height = (height + width) * scale  
        x_offset = (height / 2) * scale  
        y_offset = (width / 2) * scale  
    else:  # angle = 0
        canvas_width = width * scale    
        canvas_height = height * scale 
        x_offset = 0
        y_offset = 0

    img = Image.new('RGBA', (int(canvas_width), int(canvas_height)), (0, 0, 0, 0))

    # 推断字体大小
    image4ratio = Image.new("RGB", img.size, "white")
    draw = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
    text_w = min(w, h) * (_tw / _th)
    if text_w <= max(w, h):
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_space = insert_spaces(text, i)
                _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = insert_spaces(text, i - 1)
        font_size = min(w, h) * 0.80
    else:
        shrink = 0.85
        font_size = min(w, h) / (text_w / max(w, h)) * shrink
        
    font_size = max(1, int(font_size))        

    new_font = font.font_variant(size=int(font_size))

    left, top, right, bottom = new_font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    center = rect[0]
    draw_x = center[0] + x_offset - text_width // 2
    draw_y = center[1] + y_offset - text_height // 2 - top
    draw.text((draw_x, draw_y), text, font=new_font, fill=color)

    rotated_layer = layer.rotate(angle, expand=1, center=(center[0] + x_offset, center[1] + y_offset))
    crop_width = width * scale
    crop_height = height * scale
    crop_x = (rotated_layer.width - crop_width) // 2
    crop_y = (rotated_layer.height - crop_height) // 2
    cropped_layer = rotated_layer.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    return cropped_layer

def render_ocr_results(json_path, output_path="output.png", glyph_scale=1, eligen=False):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"读取 JSON 文件失败: {e}")
        return
    
   

    if "result" in json_data:
        json_data = json_data["result"]


    width = json_data["width"]
    height = json_data["height"]
    canvas = Image.new('RGB', (width, height), 'white')
    
    
    # print(json_data.key())


    try:
        font_path = "/juicefs-algorithm/data/IPT/wang_pang/workspace/font/arialunicodems.ttf"
        font = ImageFont.truetype(font_path, size=60)
    except Exception as e:
        print(f"字体加载失败: {e}，使用默认字体")
        font = ImageFont.load_default()
        
    # print(json_data.key())
    

    for idx, line in enumerate(json_data["lines"]):
        # text = line["text"]
        text = line.get("text", "")
        # print(idx,text)
        poly = np.array(line["position"]).reshape(4, 2)
        line_angle = line.get("angle", 0)  # 使用 line["angle"]
        # line_angle=line['angle']

        # color = json_data["attribute_map"]["color"][0]
        # if color.startswith('#'):
        #     color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) + (255,)
        # else:
        color_rgb = (0, 0, 0, 255)  # 黑色文字


        rendered_img = draw_glyph2(
            font=font,
            text=text,
            polygon=poly,
            line_angle=line_angle,
            vertAng=15,
            scale=glyph_scale,
            width=width,
            height=height,
            add_space=True,
            color=color_rgb,
            )
        if eligen:
            rendered_img = Image.new('RGBA', rendered_img.size, (0, 0, 0, 0))



        canvas.paste(rendered_img, (0, 0), rendered_img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)