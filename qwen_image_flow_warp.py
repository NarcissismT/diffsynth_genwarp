#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段光流 Warp 推理脚本
========================
解决扩散模型 VAE 信息瓶颈导致的文字细节丢失问题。

流程：
  Stage 1 - 扩散模型（低分辨率 1024）生成几何矫正结果
  Stage 2 - RAFT 估计 corrected_1024 → warped_1024 的反向光流
  Stage 3 - 将光流上采样到原始分辨率
  Stage 4 - 用上采样光流对原始高清 warped 图做像素重采样

输出文件（每张输入图片生成 4 个文件）：
  a_*.jpg  原始 warped 图（resize 到推理尺寸）
  b_*.jpg  扩散模型矫正结果（1024 分辨率）
  c_*.jpg  光流 warp 高清矫正结果（原始分辨率，文字细节完整保留）
  d_*.jpg  对比图（原始 | 扩散 | 光流warp 横向拼接）

用法示例见 scripts/flow_warp_sample.sh
"""

import os
import sys

# 必须在所有其他 import 之前执行：
# SLURM/pyxis 会通过 PYTHONPATH 把宿主机 miniconda3 路径注入 sys.path，
# 导致宿主机 Python 3.8 的 .so 文件（sklearn 等）在容器 Python 3.10 里崩溃。
# 只保留容器标准路径（/usr/）和空字符串（当前目录），彻底清除宿主机注入路径。
# 宿主机上直接运行时 sys.path 本来就只有 /usr/ 路径，行为不变。
sys.path = [p for p in sys.path if p.startswith("/usr/") or p == ""]

import torch
import glob
import argparse
import random
import csv
from tqdm import tqdm
from PIL import Image

import torchvision

# 完整版 diffsynth 必须在本地不完整版之前被找到
_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM_DIFFSYNTH = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
sys.path.insert(0, _HERE)             # 用于 utils/ 导入
sys.path.insert(0, _UPSTREAM_DIFFSYNTH)  # 优先于本地 diffsynth/

import diffsynth
print("Using diffsynth from:", diffsynth.__file__)

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict

from utils.flow_utils import load_raft_model, estimate_flow, upscale_flow, warp_image_with_flow


# ---------------------------------------------------------------------------
# 模型路径常量（与 qwen_image_batch_full_lora.py 保持一致）
# ---------------------------------------------------------------------------
_MODEL_BASE = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit"

DIT_SHARDS = [
    f"{_MODEL_BASE}/transformer/diffusion_pytorch_model-{i:05d}-of-00009.safetensors"
    for i in range(1, 10)
]
TEXT_ENCODER_SHARDS = [
    f"{_MODEL_BASE}/text_encoder/model-{i:05d}-of-00004.safetensors"
    for i in range(1, 5)
]
VAE_PATH = f"{_MODEL_BASE}/vae/diffusion_pytorch_model.safetensors"
TOKENIZER_PATH = f"{_MODEL_BASE}/tokenizer"
PROCESSOR_PATH = f"{_MODEL_BASE}/processor"


# ---------------------------------------------------------------------------
# 图像预处理（与原脚本保持一致的 resize 逻辑）
# ---------------------------------------------------------------------------

def preprocess_image(img: Image.Image, img_size: int, resize_mode: str, short_side_pixel: int):
    """
    将输入图缩放至推理尺寸。返回 (processed_img, new_w, new_h)。
    resize_mode 支持：stretch / crop / scale_to_short_side
    """
    width, height = img.size

    if resize_mode == "crop":
        short_side = min(width, height)
        scale = short_side_pixel / short_side
        new_h = round(height * scale)
        new_w = round(width * scale)
        img_resized = torchvision.transforms.functional.resize(
            img, (new_h, new_w),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        img_out = torchvision.transforms.functional.center_crop(img_resized, (img_size, img_size))
        return img_out, img_size, img_size

    elif resize_mode == "stretch":
        img_out = img.resize((img_size, img_size), Image.LANCZOS)
        return img_out, img_size, img_size

    elif resize_mode == "scale_to_short_side":
        short_side = min(width, height)
        scale = img_size / short_side
        new_h = int(height * scale) // 16 * 16
        new_w = int(width * scale) // 16 * 16
        img_out = torchvision.transforms.functional.resize(
            img, (new_h, new_w),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return img_out, new_w, new_h

    else:
        raise ValueError(f"未知的 resize_mode: {resize_mode}")


# ---------------------------------------------------------------------------
# 主处理函数
# ---------------------------------------------------------------------------

def process_images(
    input_dir: str,
    output_dir: str,
    dit_path: str,
    lora_path: str,
    prompt: str,
    batch_size: int = 1,
    prompt_dict: str = None,
    short_side_pixel: int = 2048,
    img_size: int = 1024,
    enable_dit_fp8_computation: bool = False,
    total_divide: int = 1,
    divide_index: int = 0,
    infer_steps: int = 50,
    resize_mode: str = "stretch",
    raft_model_size: str = "large",
    num_flow_updates: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 加载扩散模型
    # ------------------------------------------------------------------
    print("正在加载扩散模型...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(DIT_SHARDS),
            ModelConfig(TEXT_ENCODER_SHARDS),
            ModelConfig(VAE_PATH),
        ],
        tokenizer_config=ModelConfig(TOKENIZER_PATH),
        processor_config=ModelConfig(PROCESSOR_PATH),
    )

    if dit_path:
        print(f"加载 DiT 权重: {dit_path}")
        pipe.dit.load_state_dict(load_state_dict(dit_path))

    if lora_path:
        print(f"加载 LoRA 权重: {lora_path}")
        pipe.load_lora(pipe.dit, lora_path)

    pipe.enable_vram_management(enable_dit_fp8_computation=enable_dit_fp8_computation)

    # ------------------------------------------------------------------
    # 加载 RAFT 光流模型（在扩散模型之后加载，避免显存峰值过高）
    # ------------------------------------------------------------------
    print(f"正在加载 RAFT-{raft_model_size} 光流模型...")
    raft_model = load_raft_model(device, model_size=raft_model_size)

    # ------------------------------------------------------------------
    # 可选的 per-image prompt 字典
    # ------------------------------------------------------------------
    prompt_map = None
    if prompt_dict:
        prompt_map = {}
        with open(prompt_dict, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompt_map[row["image"]] = row["prompt"]

    # ------------------------------------------------------------------
    # 收集图像文件
    # ------------------------------------------------------------------
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    if os.path.isfile(input_dir):
        image_files = [input_dir]
    else:
        for root, _, _ in os.walk(input_dir):
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(root, ext)))
                image_files.extend(glob.glob(os.path.join(root, ext.upper())))

    print(f"找到 {len(image_files)} 张图像")

    # ------------------------------------------------------------------
    # 批量推理
    # ------------------------------------------------------------------
    total_batches = (len(image_files) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(tqdm(range(0, len(image_files), batch_size), total=total_batches, desc="处理图片")):
        if total_divide > 1 and (batch_idx % total_divide) != divide_index:
            continue

        for img_path in image_files[i : i + batch_size]:
            # input_dir 是文件时 relpath 会算出 "."，改用文件名
            if os.path.isfile(input_dir):
                rel_path = os.path.basename(img_path)
            else:
                rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 跳过已处理（以 b_ 结果文件判断）
            done_marker = output_path.replace(".jpg", "_b.jpg")
            if os.path.exists(done_marker):
                print(f"跳过已处理: {rel_path}")
                continue

            try:
                orig_img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = orig_img.size
                print(f"处理: {rel_path}  原始尺寸: {orig_w}×{orig_h}")

                # ---- Stage 1: 图像预处理 + 扩散推理 ----
                img_input, infer_w, infer_h = preprocess_image(
                    orig_img, img_size, resize_mode, short_side_pixel
                )
                print(f"  推理尺寸: {infer_w}×{infer_h}")

                cur_prompt = prompt
                if prompt_map:
                    cur_prompt = prompt_map.get(os.path.basename(img_path), prompt)

                corrected_low = pipe(
                    prompt=cur_prompt,
                    edit_image=img_input,
                    seed=random.randint(0, 2**32 - 1),
                    num_inference_steps=infer_steps,
                    width=infer_w,
                    height=infer_h,
                )
                print(f"  扩散推理完成")

                # ---- Stage 2: RAFT 估计反向光流（corrected→warped）----
                # 传入顺序：img_src=corrected, img_dst=warped
                # 得到的是"对于矫正图每个像素，应去 warped 图哪里采样"
                flow_low = estimate_flow(
                    raft_model,
                    img_src=corrected_low,
                    img_dst=img_input,
                    device=device,
                    num_flow_updates=num_flow_updates,
                )
                print(f"  光流估计完成，flow shape: {tuple(flow_low.shape)}")

                # ---- Stage 3: 将光流上采样到原始高清分辨率 ----
                flow_hires = upscale_flow(flow_low, orig_h, orig_w)

                # ---- Stage 4: 用高清 warped 图按光流采样 ----
                result_hires = warp_image_with_flow(orig_img, flow_hires)
                print(f"  光流 warp 完成，输出尺寸: {result_hires.size}")

                # ---- 保存四张结果图 ----
                base = output_path.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace(".bmp", "")

                path_a = base + "_a.jpg"   # 推理输入（resize 后的 warped）
                path_b = base + "_b.jpg"   # 扩散输出（低分辨率矫正）
                path_c = base + "_c.jpg"   # 光流 warp 高清矫正结果
                path_d = base + "_d.jpg"   # 对比横拼图

                img_input.save(path_a)
                corrected_low.save(path_b)
                result_hires.save(path_c)

                # 对比图：原始高清 | 扩散结果（上采样对齐高度）| 光流warp高清
                corrected_low_resized = corrected_low.resize((orig_w, orig_h), Image.LANCZOS)
                compare = Image.new("RGB", (orig_w * 3, orig_h))
                compare.paste(orig_img, (0, 0))
                compare.paste(corrected_low_resized, (orig_w, 0))
                compare.paste(result_hires, (orig_w * 2, 0))
                compare.save(path_d)

                print(f"  已保存: a={os.path.basename(path_a)}, b={os.path.basename(path_b)}, "
                      f"c={os.path.basename(path_c)}, d={os.path.basename(path_d)}")

            except Exception as e:
                import traceback
                print(f"处理 {rel_path} 时出错: {e}")
                traceback.print_exc()
                with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                    f.write(f"{rel_path}: {e}\n")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="两阶段光流 Warp 推理：扩散模型估计几何变换，RAFT 光流保留高清细节",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必需参数
    parser.add_argument("--input_dir",  type=str, required=True,  help="输入图片目录")
    parser.add_argument("--output_dir", type=str, required=True,  help="输出目录")

    # 模型路径
    parser.add_argument("--dit_path",   type=str, default=None,   help="自定义 DiT 权重路径（可选）")
    parser.add_argument("--lora_path",  type=str, default=None,   help="LoRA 权重路径（可选）")

    # Prompt
    parser.add_argument("--prompt",      type=str,
                        default="Flatten this warped or curled document image to a flat, undistorted version.",
                        help="矫正任务描述")
    parser.add_argument("--prompt_dict", type=str, default=None,  help="per-image prompt CSV 路径（列: image,prompt）")

    # 推理参数
    parser.add_argument("--batch_size",   type=int,   default=1,        help="批大小")
    parser.add_argument("--img_size",     type=int,   default=1024,     help="扩散推理分辨率（短边或正方形）")
    parser.add_argument("--short_side_pixel", type=int, default=2048,   help="crop 模式下短边目标像素")
    parser.add_argument("--resize_mode", type=str,   default="stretch",
                        choices=["stretch", "crop", "scale_to_short_side"],
                        help="图像预处理方式")
    parser.add_argument("--infer_steps",  type=int,   default=50,       help="扩散去噪步数")
    parser.add_argument("--enable_dit_fp8_computation", action="store_true",
                        help="启用 DiT FP8 计算（节省显存，略微降低精度）")

    # RAFT 参数
    parser.add_argument("--raft_model_size",  type=str, default="large",
                        choices=["large", "small"],
                        help="RAFT 模型规模（large 精度高，small 速度快）")
    parser.add_argument("--num_flow_updates", type=int, default=20,
                        help="RAFT 光流迭代精化次数（越大越精确，越慢）")

    # 分片并行
    parser.add_argument("--total_divide",  type=int, default=1, help="并行分片总数")
    parser.add_argument("--divide_index",  type=int, default=0, help="当前分片编号")

    args = parser.parse_args()

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dit_path=args.dit_path,
        lora_path=args.lora_path,
        prompt=args.prompt,
        batch_size=args.batch_size,
        prompt_dict=args.prompt_dict,
        short_side_pixel=args.short_side_pixel,
        img_size=args.img_size,
        enable_dit_fp8_computation=args.enable_dit_fp8_computation,
        total_divide=args.total_divide,
        divide_index=args.divide_index,
        infer_steps=args.infer_steps,
        resize_mode=args.resize_mode,
        raft_model_size=args.raft_model_size,
        num_flow_updates=args.num_flow_updates,
    )
