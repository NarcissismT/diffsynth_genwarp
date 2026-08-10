#!/usr/bin/env python3
"""
预生成验证集的 corrected_low（真扩散输出）— V4.4 验证集 step 2/3
==================================================================
这是 A3 方案的核心：验证集 main 图必须用【Qwen 50 步扩散生成的 corrected_low】，
而不是 GT corrected，才能真实暴露训练-推理域差（训练用 GT、推理用扩散输出）。
否则验证集和训练集同分布，测不出域差过拟合。

复用推理脚本 qwen_image_flow_v4_4.py 的 load_pipeline（同一条扩散链路、同一 LoRA、
同一 resize_mode/img_size），对 metadata_val.csv 每个样本：
    warped(edit_image) → Qwen num_inference_steps 步扩散 → corrected_low（PIL）
存到 <out_dir>/{idx:04d}.png，并写 val_index.csv 记录 idx↔原始行的映射 + 难度。

【与训练对齐的关键】
  - resize_mode/img_size 必须和验证 forward 一致（默认 stretch 512）
  - 固定 seed（逐样本确定性），保证可复现、跨 ckpt 验证看的是同一批 corrected_low

跑法（一次性，需要一张 GPU，纯推理；约 num_val × 单图50步 的时间，300 张约十几分钟）：
  cd versions/20260603-v4_4-hardweight
  CUDA_VISIBLE_DEVICES=0 \
  /juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python gen_val_corrected_low.py \
      --val_csv /juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_val.csv \
      --out_dir /juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/val_corrected_low \
      --lora_path /juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors \
      --img_size 512 --resize_mode stretch --infer_steps 50 --dit_target_layers 35
"""
import argparse
import os
import sys
import types

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# 复用推理脚本的 pipeline 加载与预处理（同一条扩散链路）
from qwen_image_flow_v4_4 import (
    load_pipeline, preprocess_image, parse_layer_list,
)


def build_args_for_pipeline(a):
    """构造 load_pipeline 需要的 args 命名空间（复用推理默认值）。"""
    ns = types.SimpleNamespace(
        lora_path=a.lora_path,
        dit_target_layers=a.dit_target_layers,
        k_source="warped",
        diff_out_ch=96,
        flow_iters=4,
        ckpt_path="",   # 预生成不需要 FlowHead 权重（只用 pipe 生成 corrected_low）
    )
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lora_path", required=True)
    ap.add_argument("--prompt", default=(
        "Flatten this warped or curled document image to a flat, undistorted "
        "version. Preserve all text, lines, and content accurately."))
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--resize_mode", default="stretch")
    ap.add_argument("--infer_steps", type=int, default=50)
    ap.add_argument("--dit_target_layers", default="35")
    ap.add_argument("--seed_base", type=int, default=12345,
                    help="逐样本 seed = seed_base + idx，确定性可复现")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只生成前 N 个（调试用）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.val_csv)
    if args.limit > 0:
        df = df.iloc[:args.limit].copy()
    print(f"验证集 {len(df)} 行，生成真扩散 corrected_low → {args.out_dir}")

    # 加载推理 pipeline（同一 LoRA、同一扩散链路）
    pipe_args = build_args_for_pipeline(args)
    pipe, _flow_model, target_layers = load_pipeline(pipe_args, device)

    index_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="gen corrected_low"):
        out_png = os.path.join(args.out_dir, f"{idx:05d}.png")
        # warped = edit_image
        warped = Image.open(row["edit_image"]).convert("RGB")
        img_input, infer_w, infer_h = preprocess_image(
            warped, args.img_size, args.resize_mode)

        if os.path.exists(out_png):
            corrected_low = Image.open(out_png).convert("RGB")
        else:
            cur_prompt = row.get("prompt", args.prompt)
            if not isinstance(cur_prompt, str) or not cur_prompt.strip():
                cur_prompt = args.prompt
            result = pipe(
                prompt=cur_prompt,
                edit_image=img_input,
                seed=args.seed_base + int(idx),
                num_inference_steps=args.infer_steps,
                width=infer_w,
                height=infer_h,
            )
            corrected_low = result[0]   # (corrected_low, q, k)
            # 存成与 img_size 一致的尺寸（验证 forward 会再 resize 到 img_size，这里对齐）
            corrected_low.save(out_png)

        index_rows.append({
            "val_idx": idx,
            "corrected_low_path": out_png,
            "image": row["image"],
            "edit_image": row["edit_image"],
            "flow_gt_path": row["flow_gt_path"],
            "prompt": row.get("prompt", args.prompt),
            "category": row.get("category", ""),
            "zero_epe": row.get("zero_epe", -1),
            "difficulty": row.get("difficulty", ""),
        })

    index_csv = os.path.join(args.out_dir, "val_index.csv")
    pd.DataFrame(index_rows).to_csv(index_csv, index=False)
    print(f"\n✅ 生成完成 {len(index_rows)} 张 → {args.out_dir}")
    print(f"✅ 索引表 → {index_csv}")
    print(f"\n下一步: 训练脚本加 --val_index_csv {index_csv} 开启内嵌验证")


if __name__ == "__main__":
    main()
