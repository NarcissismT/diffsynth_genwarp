import csv
from glob import glob
import os
import json
from tqdm import tqdm

dataset_root = 'data/test/53'

if 'train' in dataset_root:
    rows = [
        ["image", "prompt", "edit_image", "category"],
    ]
    gts = glob(f'{dataset_root}/**/gt/*.*p*g', recursive=True)
elif 'test' in dataset_root:
    rows = [
        ["image", "prompt"],
    ]
    gts = glob(f'{dataset_root}/*.*p*g', recursive=True)

for gt in tqdm(gts):
    try:
        if 'train' in dataset_root:
            ocr_path = gt.replace('/gt/', '/ocr/')
            ocr_path = os.path.splitext(ocr_path)[0] + '.json'
        elif 'test' in dataset_root:
            ocr_path = dataset_root + '/ocr/' + os.path.basename(os.path.splitext(gt)[0]) + '.json'

        with open(ocr_path, 'r') as f:
            ocr = json.load(f)
        ocr_txt = ', '.join([item['text'] for item in ocr['result']['lines']])

        gt = gt.replace(f'{dataset_root}/','')

        if 'train' in dataset_root:
            rows.append([gt, f"这张图片包含文本：“{ocr_txt}”，请将这张图片变清晰、去除模糊，保持文本细节的正确", gt.replace('/gt/','/img/'), "lens_blur"])
        elif 'test' in dataset_root:
            rows.append([gt, f"这张图片包含文本：“{ocr_txt}”，请将这张图片变清晰、去除模糊，保持文本细节的正确"])
    except:
        pass

with open(f"{dataset_root}/metadata_ocr.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)