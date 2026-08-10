import json
import os
import re
import requests
from natsort import natsorted
from glob import glob


p_hz = re.compile('[\u4e00-\u9fa5]')
p_sz = re.compile('[0-9]')
p_zm = re.compile('[a-zA-Z]')


dataset_root = "data/test/53"

url='http://10.48.226.13:30080/temp_service/temp-temp-ocr-generic-lu-2/ai/internal/v2/recognize'

if 'train' in dataset_root:
    gts = natsorted(glob(f'{dataset_root}/**/gt/*.*p*g', recursive=True))
elif 'test' in dataset_root:
    gts = natsorted(glob(f'{dataset_root}/*.*p*g', recursive=True))

for i, gt in enumerate(gts):
    if 'train' in dataset_root:
        ocr_path = gt.replace('/gt/', '/ocr/')
        ocr_path = os.path.splitext(ocr_path)[0] + '.json'
    elif 'test' in dataset_root:
        ocr_path = dataset_root + '/ocr/' + os.path.basename(os.path.splitext(gt)[0]) + '.json'


    os.makedirs(os.path.dirname(ocr_path), exist_ok=True)

    try:
        print(f"Processing {i+1}/{len(gts)}: {gt}")
        print(ocr_path)
        with open(gt, "rb") as f:
            img_data = f.read()
            res = requests.post(url=url, data=img_data, headers={'Content-Type': 'application/octet-stream', 'accept': 'application/json'})
            if res.status_code == 200:
                content = res.json()
                with open(ocr_path, 'w') as fjs:
                    json.dump(content, fjs, indent=4, ensure_ascii=False)
            else:
                print(f"Error processing {gt}: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"Exception for {gt}: {e}")
        pass


