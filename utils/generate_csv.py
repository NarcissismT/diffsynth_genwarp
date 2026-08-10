import csv
from glob import glob

dataset_root = 'data/train/ipad_scan'
# selected_dataset = ['53_2', 'ipad_scan_shadow']
selected_dataset = []

selected_dataset = sorted(selected_dataset)

rows = [
    ["image", "prompt", "edit_image", "category"],
]

gts = glob(f'{dataset_root}/**/gt/*.*', recursive=True)

for gt in gts:
    if not any([s in gt for s in selected_dataset]) and len(selected_dataset) > 0:
        continue
    gt = gt.replace(f'{dataset_root}/','')
    rows.append([gt, "将这张图片变清晰、去除模糊", gt.replace('gt','img'), "lens_blur"])

postfix = f"_{','.join(selected_dataset)}" if len(selected_dataset) > 0 else ""
with open(f"{dataset_root}/metadata.csv{postfix}", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)