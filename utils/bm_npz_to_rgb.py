import csv
import numpy as np
from tqdm import tqdm
from collections import Counter
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from PIL import Image

file_path = 'data/train/bm/origin_anno/20250417_train_noF2F_addSamplePers_selectJY_fixBg.txt'
origin_root = '/juicefs-algorithm/lts_data/IPT/pengcheng_yu/bm'
tgt_root = '/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/train/bm'
item_num = 10000
dataset_root = 'data/train/bm'

rows = [
    ["image", "prompt", "edit_image", "category"],
]

with open(file_path, 'r', encoding='utf-8') as f:
    first_items = [line.strip().split()[0] for line in f]
    if item_num:
        first_items = random.sample(first_items, min(item_num, len(first_items)))

def process_file(file_path):
    data = np.load(file_path)
    bm = (data['bm']+255.5)//2 # [0, 255]
    bm = bm.astype(np.uint8).transpose(1, 2, 0) # [H, W, 2]
    bm_rgb = np.zeros((bm.shape[0], bm.shape[1], 3), dtype=np.uint8)
    bm_rgb[:, :, 0] = bm[:, :, 0]
    bm_rgb[:, :, 1] = bm[:, :, 1]
    bm_rgb[:, :, 2] = 0
    gt_save_path = file_path.replace(origin_root, f'{tgt_root}/gt').replace('.npz', '.png')
    os.makedirs(os.path.dirname(gt_save_path), exist_ok=True)
    Image.fromarray(bm_rgb).save(gt_save_path)

    img = (data['img']).astype(np.uint8) # [H, W, 3]
    img_save_path = file_path.replace(origin_root, f'{tgt_root}/img').replace('.npz', '.png')
    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
    Image.fromarray(img).save(img_save_path)

    return [gt_save_path, "输入的图像是一张扭曲的文本图像，请将这张图像矫正为一张平整的图片。请输出一张图片，代表两个方向上的位移场，平整图片每个像素施加对应位置的位移后就是原扭曲图片。位移场图片的第一个通道是水平方向，第二个通道是垂直方向，第三个通道为 0。", img_save_path, "lens_blur"]

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, item) for item in first_items]
    for f in tqdm(as_completed(futures), total=len(futures)):
        rows.append(f.result())

with open(f"{dataset_root}/metadata.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)