#!/usr/bin/env python3
"""
VAE Roundtrip 预处理脚本
========================
对训练集 corrected_gt 图像做 VAE encode-decode，
输出 corrected_vae 图像，消除与扩散模型输出的 domain gap。

需要在 Docker 容器内运行（registry.intsig.net/junle_liu/diffsynth:v3）。

用法：
  python utils/generate_corrected_vae.py \\
    --input_csv  /path/to/metadata_with_flow.csv \\
    --output_csv /path/to/metadata_with_flow_vae.csv \\
    --vae_img_dir /path/to/corrected_vae_imgs/ \\
    --batch_size 8 \\
    --total_divide 1 \\
    --divide_index 0
"""

import os
import sys

_sys_usr = [p for p in sys.path if p.startswith("/usr/") or p == ""]
_sys_other = [p for p in sys.path if p not in set(_sys_usr)]
sys.path = _sys_usr + _sys_other
sys.path.insert(0, "/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp")
sys.path.insert(0, "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio")

import csv
import argparse
import torch
from PIL import Image
from tqdm import tqdm

_MODEL_BASE = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit"


def load_vae(device):
    """直接加载 QwenImageVAE，绕过 diffsynth/__init__.py 的完整依赖链。"""
    import importlib.util
    from safetensors.torch import load_file

    # 直接加载 qwen_image_vae.py，不触发 diffsynth.__init__
    _vae_path = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/diffsynth/models/qwen_image_vae.py"
    spec = importlib.util.spec_from_file_location("qwen_image_vae", _vae_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    vae = mod.QwenImageVAE()
    sd = load_file(f"{_MODEL_BASE}/vae/diffusion_pytorch_model.safetensors")
    converter = mod.QwenImageVAEStateDictConverter()
    sd = converter.from_diffusers(sd)
    vae.load_state_dict(sd)
    vae = vae.to(device=device, dtype=torch.bfloat16).eval()
    return vae


def pil_to_tensor(img: Image.Image, device) -> torch.Tensor:
    """PIL → (1,3,H,W) bfloat16，值域 [-1,1]"""
    import numpy as np
    arr = np.array(img.convert("RGB"), dtype=np.float32) * (2.0 / 255.0) - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=torch.bfloat16)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1,3,H,W) → PIL，值域 [-1,1]"""
    import numpy as np
    arr = ((t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() / 2 + 0.5)
           .clip(0, 1) * 255).astype("uint8")
    return Image.fromarray(arr)


def vae_roundtrip_batch(vae, imgs: list, device) -> list:
    """批量 VAE roundtrip，返回 PIL 列表。"""
    results = []
    for img in imgs:
        with torch.no_grad():
            t = pil_to_tensor(img.resize((1024, 1024), Image.LANCZOS), device)
            latent = vae.encode(t)
            out = vae.decode(latent)
        results.append(tensor_to_pil(out))
    return results


def merge_csvs(part_csvs: list, output_csv: str):
    """合并多个分片 CSV。"""
    import glob as _glob
    files = []
    for pattern in part_csvs:
        files.extend(_glob.glob(pattern))
    files = sorted(files)
    all_rows, fieldnames = [], None
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            all_rows.extend(list(reader))
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"合并完成：{len(files)} 个分片，共 {len(all_rows)} 条 → {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",    type=str, default=None)
    parser.add_argument("--output_csv",   type=str, required=True)
    parser.add_argument("--vae_img_dir",  type=str, default=None,
                        help="corrected_vae 图片保存目录")
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--total_divide", type=int, default=1)
    parser.add_argument("--divide_index", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--merge_csvs",   type=str, nargs="+", default=None,
                        help="合并多个分片 CSV，与 --output_csv 一起使用")
    args = parser.parse_args()

    # 合并模式
    if args.merge_csvs:
        merge_csvs(args.merge_csvs, args.output_csv)
        return

    assert args.input_csv and args.vae_img_dir, \
        "生成模式需要 --input_csv 和 --vae_img_dir"

    os.makedirs(args.vae_img_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("加载 VAE...")
    vae = load_vae(device)

    # 读 CSV
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        fieldnames = reader.fieldnames
    print(f"总样本数: {len(all_rows)}")

    # 分片
    my_rows = [r for i, r in enumerate(all_rows) if i % args.total_divide == args.divide_index]
    print(f"本进程: {len(my_rows)} 条（分片 {args.divide_index}/{args.total_divide}）")

    results = []
    skipped = errors = 0

    for batch_start in tqdm(range(0, len(my_rows), args.batch_size),
                             total=(len(my_rows) + args.batch_size - 1) // args.batch_size,
                             desc="VAE roundtrip"):
        batch = my_rows[batch_start: batch_start + args.batch_size]

        todo = []
        for row in batch:
            # 输出路径与 corrected_gt 路径结构对应
            rel = os.path.relpath(row["image"], os.path.dirname(args.input_csv))
            vae_path = os.path.join(args.vae_img_dir,
                                    rel.replace(".png", ".jpg").replace(".jpeg", ".jpg"))
            row_out = dict(row)
            row_out["corrected_vae_path"] = vae_path
            results.append(row_out)

            if args.skip_existing and os.path.exists(vae_path):
                skipped += 1
            else:
                todo.append((row, vae_path))

        if not todo:
            continue

        imgs = []
        for row, _ in todo:
            try:
                imgs.append(Image.open(row["image"]).convert("RGB"))
            except Exception as e:
                print(f"\n读图出错 {row['image']}: {e}")
                errors += 1
                imgs.append(None)

        valid = [(img, vp) for img, (_, vp) in zip(imgs, todo) if img is not None]
        if not valid:
            continue

        try:
            vae_imgs = vae_roundtrip_batch(vae, [v[0] for v in valid], device)
            for vae_img, (_, vae_path) in zip(vae_imgs, valid):
                os.makedirs(os.path.dirname(vae_path), exist_ok=True)
                vae_img.save(vae_path, quality=95)
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors += len(valid)

    # 写输出 CSV
    out_csv = args.output_csv
    if args.total_divide > 1:
        base, ext = os.path.splitext(out_csv)
        out_csv = f"{base}_part{args.divide_index}{ext}"

    new_fieldnames = list(fieldnames) + ["corrected_vae_path"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n完成：处理 {len(my_rows)-skipped-errors}，跳过 {skipped}，出错 {errors}")
    print(f"输出 CSV: {out_csv}")


if __name__ == "__main__":
    main()
