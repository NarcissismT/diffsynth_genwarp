#!/usr/bin/env python3
"""
离线生成 Flow GT 标签脚本
========================
对训练集每对 (warped, corrected) 用 RAFT 计算反向光流，
存为 .npy 文件，并生成新的含 flow_gt_path 字段的 CSV。

支持：
  - 多卡并行：每张 GPU 处理一个分片（--gpu_ids 指定）
  - batch size：同一 GPU 内批量推理（--batch_size）
  - 断点续跑：已存在的 .npy 自动跳过

单卡用法：
  python utils/generate_flow_labels.py \\
    --input_csv /juicefs-algorithm/data/IPT/yuang_feng/DATA/upwarp_img_1in10_z/1in10_z_metadata.csv \\
    --output_csv /path/to/metadata_with_flow.csv \\
    --flow_dir /path/to/flow_labels/ \\
    --gpu_ids 0 \\
    --batch_size 4

多卡并行（手动启动，每张卡一个进程）：
  CUDA_VISIBLE_DEVICES=0 python utils/generate_flow_labels.py \\
    --input_csv ... --output_csv .../output_part0.csv \\
    --flow_dir ... --gpu_ids 0 --total_divide 4 --divide_index 0 --batch_size 4 &

  CUDA_VISIBLE_DEVICES=1 python utils/generate_flow_labels.py \\
    --input_csv ... --output_csv .../output_part1.csv \\
    --flow_dir ... --gpu_ids 1 --total_divide 4 --divide_index 1 --batch_size 4 &

  # 等全部完成后合并 CSV：
  python utils/generate_flow_labels.py --merge_csvs .../output_part*.csv --output_csv .../merged.csv
"""

import os
import sys
import argparse
import csv
import glob
import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UPSTREAM = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
# 把当前目录（''）和本地 diffsynth_genwarp 路径从 sys.path 中移到末尾，
# 确保 import diffsynth 优先找到上游 DiffSynth-Studio，而非本地的 diffsynth/ 目录
sys.path = [p for p in sys.path if p not in ('', _HERE)] + ['', _HERE]
sys.path.insert(0, _UPSTREAM)
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from utils.flow_utils import load_raft_model, estimate_flow_batch


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 主模式
    parser.add_argument("--input_csv",  type=str, default=None,
                        help="原始训练 CSV（含 image/corrected 和 edit_image/warped 列）")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="输出 CSV，新增 flow_gt_path 列")
    parser.add_argument("--flow_dir",   type=str, default=None,
                        help="flow .npy 文件存放根目录")

    # 设备
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0],
                        help="使用的 GPU 编号列表，如 --gpu_ids 0 1 2 3")

    # 推理参数
    parser.add_argument("--raft_model_size", type=str, default="large",
                        choices=["large", "small"])
    parser.add_argument("--batch_size", type=int, default=4,
                        help="每次送入 RAFT 的图片数（同一 GPU 内 batch 推理）")
    parser.add_argument("--num_flow_updates", type=int, default=20)

    # 分片（配合多卡手动并行）
    parser.add_argument("--total_divide", type=int, default=1,
                        help="总分片数（等于启动的进程数）")
    parser.add_argument("--divide_index", type=int, default=0,
                        help="当前进程负责的分片编号（0-based）")

    # 断点续跑
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="已存在的 .npy 跳过重算（默认开启）")

    # VAE domain gap 消除
    parser.add_argument("--use_vae_roundtrip", action="store_true", default=False,
                        help="对 corrected_gt 做 VAE encode-decode，消除与扩散输出的 domain gap（推荐开启）")

    # CSV 合并模式（所有分片完成后运行一次）
    parser.add_argument("--merge_csvs", type=str, nargs="+", default=None,
                        help="合并多个分片 CSV 为一个，与 --output_csv 一起使用")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# CSV 合并工具
# ---------------------------------------------------------------------------

def merge_csvs(part_csvs: list, output_csv: str):
    """把多个分片 CSV 合并为一个。"""
    all_rows = []
    fieldnames = None
    for path in sorted(part_csvs):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            all_rows.extend(list(reader))
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"合并完成：共 {len(all_rows)} 条 → {output_csv}")


# ---------------------------------------------------------------------------
# 单 GPU 工作函数
# ---------------------------------------------------------------------------

def load_vae(device: torch.device):
    """加载 QwenImageVAE，用于对 corrected_gt 做 encode-decode，消除 domain gap。"""
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
    _MODEL_BASE = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit"
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=[ModelConfig(f"{_MODEL_BASE}/vae/diffusion_pytorch_model.safetensors")],
        tokenizer_config=ModelConfig(_MODEL_BASE + "/tokenizer"),
        processor_config=ModelConfig(_MODEL_BASE + "/processor"),
    )
    vae = pipe.vae.to(device=device, dtype=torch.bfloat16).eval()
    return vae, pipe


def vae_roundtrip(vae, pipe, img: "Image.Image", device: torch.device) -> "Image.Image":
    """corrected_gt → VAE encode → decode → corrected_vae（模拟扩散模型输出分布）。"""
    with torch.no_grad():
        t = pipe.preprocess_image(img).to(device=device, dtype=torch.bfloat16)
        latent = vae.encode(t)
        out = vae.decode(latent)
    return pipe.vae_output_to_image(out)


def process_on_device(rows: list, flow_dir: str, input_csv_dir: str,
                      device: torch.device, raft_model_size: str,
                      batch_size: int, num_flow_updates: int,
                      skip_existing: bool, use_vae_roundtrip: bool = True) -> list:
    """
    在指定 device 上处理 rows 列表，返回带 flow_gt_path 的 rows。
    """
    raft_model = load_raft_model(device, model_size=raft_model_size)

    # 加载 VAE（用于 corrected_gt → encode-decode，消除 domain gap）
    vae = pipe_vae = None
    if use_vae_roundtrip:
        print(f"[{device}] 加载 VAE 用于 domain gap 消除...")
        vae, pipe_vae = load_vae(device)

    results = []
    skipped = errors = 0

    # 按 batch_size 分批处理
    for batch_start in tqdm(range(0, len(rows), batch_size),
                             desc=f"[{device}] 生成 flow GT",
                             total=(len(rows) + batch_size - 1) // batch_size):
        batch_rows = rows[batch_start: batch_start + batch_size]

        # 为每条数据计算 flow_path
        flow_paths = []
        for row in batch_rows:
            corrected_path = row["image"]
            rel = os.path.relpath(corrected_path, input_csv_dir)
            flow_path = os.path.join(
                flow_dir,
                rel.replace(".png", ".npy")
                   .replace(".jpg", ".npy")
                   .replace(".jpeg", ".npy")
                   .replace(".bmp", ".npy"),
            )
            flow_paths.append(flow_path)
            row_out = dict(row)
            row_out["flow_gt_path"] = flow_path
            results.append(row_out)

        # 找出需要计算的（跳过已存在的）
        if skip_existing:
            todo_indices = [i for i, fp in enumerate(flow_paths) if not os.path.exists(fp)]
        else:
            todo_indices = list(range(len(batch_rows)))

        if not todo_indices:
            skipped += len(batch_rows)
            continue

        # 加载图像
        imgs_src, imgs_dst, valid_paths = [], [], []
        for i in todo_indices:
            row = batch_rows[i]
            try:
                # 优先用 corrected_vae_path（已做过 VAE roundtrip 的图），
                # 否则用 corrected_gt（原始高质量图，或在线做 VAE roundtrip）
                corrected_path = row.get("corrected_vae_path") or row["image"]
                corrected = Image.open(corrected_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
                warped    = Image.open(row["edit_image"]).convert("RGB").resize((1024, 1024), Image.LANCZOS)
                # 如果没有预计算的 vae 图且开启了在线 roundtrip，则在线处理
                if use_vae_roundtrip and vae is not None and not row.get("corrected_vae_path"):
                    corrected = vae_roundtrip(vae, pipe_vae, corrected, device)
                imgs_src.append(corrected)
                imgs_dst.append(warped)
                valid_paths.append(flow_paths[i])
            except Exception as e:
                print(f"\n读图出错 {row['image']}: {e}")
                errors += 1

        if not imgs_src:
            continue

        # 批量 RAFT 推理
        try:
            flows = estimate_flow_batch(
                raft_model, imgs_src, imgs_dst, device,
                num_flow_updates=num_flow_updates,
            )
            for flow, fp in zip(flows, valid_paths):
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                np.save(fp, flow.squeeze(0).cpu().numpy())  # (2, 1024, 1024)
        except Exception as e:
            print(f"\nRAFT 推理出错（batch size {len(imgs_src)}）: {e}")
            errors += len(imgs_src)

        skipped += len(batch_rows) - len(todo_indices)

    print(f"[{device}] 完成：处理 {len(rows)-skipped-errors}，跳过 {skipped}，出错 {errors}")
    return results


# ---------------------------------------------------------------------------
# 多卡并行主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- 合并模式 ---
    if args.merge_csvs:
        assert args.output_csv, "--merge_csvs 需要同时指定 --output_csv"
        # 支持 glob 模式（shell 不展开时）
        files = []
        for pattern in args.merge_csvs:
            files.extend(glob.glob(pattern))
        merge_csvs(files, args.output_csv)
        return

    # --- 生成模式 ---
    assert args.input_csv and args.output_csv and args.flow_dir, \
        "生成模式需要 --input_csv, --output_csv, --flow_dir"
    os.makedirs(args.flow_dir, exist_ok=True)

    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        fieldnames = reader.fieldnames
    print(f"总样本数: {len(all_rows)}")

    # 先按 total_divide 做进程级分片
    my_rows = [r for i, r in enumerate(all_rows) if i % args.total_divide == args.divide_index]
    print(f"本进程分片: {len(my_rows)} 条（{args.divide_index}/{args.total_divide}）")

    input_csv_dir = os.path.dirname(args.input_csv)

    if len(args.gpu_ids) == 1:
        # 单卡
        device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        results = process_on_device(
            my_rows, args.flow_dir, input_csv_dir,
            device, args.raft_model_size,
            args.batch_size, args.num_flow_updates, args.skip_existing,
            use_vae_roundtrip=args.use_vae_roundtrip,
        )
    else:
        # 多卡：在本进程内用多线程，每张卡一个线程
        import threading

        n_gpus = len(args.gpu_ids)
        # 按 GPU 数再次均分
        chunks = [my_rows[i::n_gpus] for i in range(n_gpus)]
        all_results = [None] * n_gpus

        exceptions = [None] * n_gpus

        def worker(gpu_idx, chunk, out_slot):
            try:
                device = torch.device(f"cuda:{args.gpu_ids[gpu_idx]}")
                all_results[out_slot] = process_on_device(
                    chunk, args.flow_dir, input_csv_dir,
                    device, args.raft_model_size,
                    args.batch_size, args.num_flow_updates, args.skip_existing,
                    use_vae_roundtrip=args.use_vae_roundtrip,
                )
            except Exception as e:
                import traceback
                exceptions[out_slot] = e
                traceback.print_exc()

        threads = [
            threading.Thread(target=worker, args=(i, chunks[i], i))
            for i in range(n_gpus)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 检查是否有 worker 出错
        for i, exc in enumerate(exceptions):
            if exc is not None:
                raise RuntimeError(f"GPU {args.gpu_ids[i]} worker 出错: {exc}")

        # 合并结果，按原始顺序还原（跳过 None 的 slot）
        results_dict = {}
        for slot_results in all_results:
            if slot_results is None:
                continue
            for row in slot_results:
                results_dict[row["image"]] = row
        results = [results_dict[r["image"]] for r in my_rows if r["image"] in results_dict]

    # 写输出 CSV
    out_csv = args.output_csv
    if args.total_divide > 1:
        base, ext = os.path.splitext(out_csv)
        out_csv = f"{base}_part{args.divide_index}{ext}"

    new_fieldnames = list(fieldnames) + ["flow_gt_path"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"输出 CSV: {out_csv}")


if __name__ == "__main__":
    main()
