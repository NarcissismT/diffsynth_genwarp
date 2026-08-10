import numpy as np
from tqdm import tqdm
from collections import Counter
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

file_path = 'data/train/bm/origin_anno/20250417_train_noF2F_addSamplePers_selectJY_fixBg.txt'
item_num = 1000

with open(file_path, 'r', encoding='utf-8') as f:
    first_items = [line.strip().split()[0] for line in f]
    first_items = random.sample(first_items, min(item_num, len(first_items)))

def process_file(file_path):
    data = np.load(file_path)
    bm_max = round(np.max(data['bm']), 2)
    bm_min = round(np.min(data['bm']), 2)
    img_height = data['img'].shape[0]
    img_width = data['img'].shape[1]
    return bm_max, bm_min, img_height, img_width

bm_max_list = []
bm_min_list = []
img_height_list = []
img_width_list = []

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, fp) for fp in first_items]
    for future in tqdm(as_completed(futures), total=len(futures)):
        bm_max, bm_min, img_height, img_width = future.result()
        bm_max_list.append(bm_max)
        bm_min_list.append(bm_min)
        img_height_list.append(img_height)
        img_width_list.append(img_width)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].hist(bm_max_list, bins=30, color='skyblue', edgecolor='black')
axs[0, 0].set_title('bm_max Distribution')
axs[0, 0].set_xlabel('bm_max')
axs[0, 0].set_ylabel('Frequency')

axs[0, 1].hist(bm_min_list, bins=30, color='salmon', edgecolor='black')
axs[0, 1].set_title('bm_min Distribution')
axs[0, 1].set_xlabel('bm_min')
axs[0, 1].set_ylabel('Frequency')

axs[1, 0].hist(img_height_list, bins=30, color='lightgreen', edgecolor='black')
axs[1, 0].set_title('img_height Distribution')
axs[1, 0].set_xlabel('img_height')
axs[1, 0].set_ylabel('Frequency')

axs[1, 1].hist(img_width_list, bins=30, color='orange', edgecolor='black')
axs[1, 1].set_title('img_width Distribution')
axs[1, 1].set_xlabel('img_width')
axs[1, 1].set_ylabel('Frequency')

plt.tight_layout()
os.makedirs('data/train/bm/analysis', exist_ok=True)
plt.savefig(f'data/train/bm/analysis/bm_npz_distributions_{item_num}.png')
