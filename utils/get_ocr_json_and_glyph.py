import os
import json
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from ocr_visual_v4_new_api import render_ocr_results  # 复用你的render函数
from glob import glob

# OCR_URL = "http://10.30.4.40:80/icr/ctpn_recognize"
# OCR_HEADER = {"Host": "recognize-document-3d1"}


# url='http://10.48.226.13:30080/temp_service/temp-ocr-api-2/ai/internal/v2/recognize'
url='http://10.48.226.13:30080/temp_service/temp-ocr-api-25/ai/internal/v2/recognize'

#http://10.2.10.63:30080/temp_service/temp-ocrapi-123/ai/internal/v2/recognize


def get_ocr(img_path, ocr_json_path):
    """调用OCR服务并保存JSON"""
    if os.path.exists(ocr_json_path):
        return ocr_json_path  # 已存在直接返回

    with open(img_path, "rb") as f:
        # response = requests.post(OCR_URL, headers=OCR_HEADER, data=f)
        response = requests.post(url=url, data=f, headers={'Content-Type': 'application/octet-stream', 'accept': 'application/json'})
    if response.status_code == 200:
        ocr_result = response.json()
        os.makedirs(os.path.dirname(ocr_json_path), exist_ok=True)
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_result, f, ensure_ascii=False)
        return ocr_json_path
    else:
        print(f"[ERROR] OCR failed {img_path}, code={response.status_code}")
        return None

def run_ocr_pipeline(img_paths, img_dir, out_dir, max_workers=8):
    os.makedirs(out_dir, exist_ok=True)
    json_dir = os.path.join(out_dir, "json")
    glyph_dir = os.path.join(out_dir, "glyph")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(glyph_dir, exist_ok=True)

    start_time = time.time()

    # Step1: 并发 OCR 请求
    ocr_futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for img_path in img_paths:
            ocr_json_path = os.path.splitext(img_path.replace('/img/', '/ocr_json/'))[0] + ".json"
            future = executor.submit(get_ocr, img_path, ocr_json_path)
            ocr_futures[future] = (img_path, ocr_json_path)

        ocr_results = []
        for future in tqdm(as_completed(ocr_futures), total=len(ocr_futures), desc="OCR"):
            result = future.result()
            if result:
                ocr_results.append(result)
                
    # Step2: 多进程并行渲染 Glyph
    glyph_tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ocr_json_path in ocr_results:
            glyph_path = os.path.splitext(ocr_json_path.replace('/ocr_json/', '/glyph/'))[0] + ".png"
            if os.path.exists(glyph_path):
                continue
            future = executor.submit(render_ocr_results, ocr_json_path, glyph_path)
            glyph_tasks.append(future)

        for future in tqdm(as_completed(glyph_tasks), total=len(glyph_tasks), desc="Glyph"):
            future.result()  # 等待完成，抛出异常
                
    # Step3: 多进程并行渲染 EliGen
    eligen_tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ocr_json_path in ocr_results:
            eligen_path = os.path.splitext(ocr_json_path.replace('/ocr_json/', '/eligen/'))[0] + ".png"
            if os.path.exists(eligen_path):
                continue
            future = executor.submit(render_ocr_results, ocr_json_path, eligen_path, eligen=True)
            eligen_tasks.append(future)

        for future in tqdm(as_completed(eligen_tasks), total=len(eligen_tasks), desc="EliGen"):
            future.result()  # 等待完成，抛出异常

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n全部完成，总耗时 {elapsed:.2f} 秒，平均 {elapsed/len(img_paths):.2f} 秒/张")

if __name__ == "__main__":
    # 输入文件夹
    img_dir = "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/train/ipad_scan_shadow"
    out_dir = f"{img_dir}/ocr_img"

    # 收集所有图片
    img_paths = glob(f'{img_dir}/**/img/*.*p*g', recursive=True)

    
    run_ocr_pipeline(img_paths, img_dir, out_dir, max_workers=1) 
