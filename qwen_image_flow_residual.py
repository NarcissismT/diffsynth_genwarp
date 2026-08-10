#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion Flow Residual 推理脚本
================================
在二阶段 pipeline 基础上，引入 FlowHead：

流程：
  Stage 1 - 扩散模型（1024）输出矫正像素图 corrected_low + 坐标残差 ΔG
  Stage 2 - RAFT 估计粗光流 G_0（corrected_low → warped，1024 分辨率）
  Stage 3 - 叠加残差：G_refined = G_0 + ΔG（上采样到 1024）
  Stage 4 - 将 G_refined 上采样到原始高清分辨率
  Stage 5 - 用 G_hires 对原始高清图做像素重采样

优势：
  ΔG 不经过 VAE decode，避免 VAE 感知压缩引入的坐标误差
  文字、表格线清晰度完整保留

输出文件（每张输入图片生成 4 个文件）：
  a_*.jpg  原始 warped 图（resize 到推理尺寸）
  b_*.jpg  扩散模型矫正结果（1024 分辨率，仅供参考）
  c_*.jpg  Flow Residual warp 高清矫正结果（原始分辨率）
  d_*.jpg  对比图（原始 | 扩散 | flow residual 横向拼接）
"""

import os
import sys

# 清除宿主机 miniconda3 路径，避免在 SLURM/pyxis 容器内加载错误的包
_sys_usr = [p for p in sys.path if p.startswith("/usr/") or p == ""]
_sys_other = [p for p in sys.path if p not in set(_sys_usr)]
sys.path = _sys_usr + _sys_other

import torch
import glob
import argparse
import random
import csv
from tqdm import tqdm
from PIL import Image

import torchvision

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM_DIFFSYNTH = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
sys.path.insert(0, _HERE)
sys.path.insert(0, _UPSTREAM_DIFFSYNTH)

import diffsynth
print("Using diffsynth from:", diffsynth.__file__)

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowHead
from diffsynth import load_state_dict

from utils.flow_utils import load_raft_model, estimate_flow, upscale_flow, warp_image_with_flow


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


def preprocess_image(img: Image.Image, img_size: int, resize_mode: str, short_side_pixel: int):
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


def process_images(
    input_dir: str,
    output_dir: str,
    lora_path: str,
    prompt: str,
    flow_head_path: str = None,
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

    # 加载扩散模型
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

    # 初始化 FlowHead
    pipe.flow_head = FlowHead(latent_channels=16, out_channels=2).to(device)

    if lora_path and os.path.exists(lora_path):
        from safetensors import safe_open
        # 检查 checkpoint 是否包含 FlowHead 权重（联合训练产生的 checkpoint）
        with safe_open(lora_path, framework="pt") as f:
            all_keys = list(f.keys())
        flow_keys = [k for k in all_keys if "flow_head" in k]

        if flow_keys:
            # 联合训练 checkpoint：同时包含 LoRA + FlowHead
            print(f"加载联合训练 checkpoint（LoRA + FlowHead）: {lora_path}")
            from safetensors.torch import load_file
            state_dict = load_file(lora_path, device=str(device))
            # 拆分并加载 LoRA
            lora_sd = {k: v for k, v in state_dict.items() if "flow_head" not in k}
            pipe.load_lora(pipe.dit, lora_path)
            # 拆分并加载 FlowHead（去掉 "pipe.flow_head." 前缀）
            flow_sd = {k.replace("pipe.flow_head.", ""): v
                       for k, v in state_dict.items() if "flow_head" in k}
            pipe.flow_head.load_state_dict(flow_sd)
            print(f"  LoRA keys: {len(lora_sd)}，FlowHead keys: {len(flow_sd)}")
        else:
            # 旧格式：只有 LoRA
            print(f"加载 LoRA 权重: {lora_path}")
            pipe.load_lora(pipe.dit, lora_path)
            # 单独加载 FlowHead
            if flow_head_path and os.path.exists(flow_head_path):
                print(f"加载 FlowHead 权重: {flow_head_path}")
                pipe.flow_head.load_state_dict(
                    torch.load(flow_head_path, map_location=device))
            else:
                print("FlowHead 使用随机初始化")
    else:
        print("FlowHead 使用随机初始化（未提供 checkpoint）")

    pipe.enable_vram_management(enable_dit_fp8_computation=enable_dit_fp8_computation)

    # 加载 RAFT
    print(f"正在加载 RAFT-{raft_model_size} 光流模型...")
    raft_model = load_raft_model(device, model_size=raft_model_size)

    # per-image prompt 字典
    prompt_map = None
    if prompt_dict:
        prompt_map = {}
        with open(prompt_dict, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompt_map[row["image"]] = row["prompt"]

    # 收集图像文件
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

    total_batches = (len(image_files) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(tqdm(range(0, len(image_files), batch_size), total=total_batches, desc="处理图片")):
        if total_divide > 1 and (batch_idx % total_divide) != divide_index:
            continue

        for img_path in image_files[i : i + batch_size]:
            if os.path.isfile(input_dir):
                rel_path = os.path.basename(img_path)
            else:
                rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            done_marker = output_path.replace(".jpg", "_b.jpg").replace(".png", "_b.jpg").replace(".jpeg", "_b.jpg").replace(".bmp", "_b.jpg")
            if os.path.exists(done_marker):
                print(f"跳过已处理: {rel_path}")
                continue

            try:
                orig_img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = orig_img.size
                print(f"处理: {rel_path}  原始尺寸: {orig_w}×{orig_h}")

                # Stage 1：图像预处理 + 扩散推理
                img_input, infer_w, infer_h = preprocess_image(
                    orig_img, img_size, resize_mode, short_side_pixel
                )
                print(f"  推理尺寸: {infer_w}×{infer_h}")

                cur_prompt = prompt
                if prompt_map:
                    cur_prompt = prompt_map.get(os.path.basename(img_path), prompt)

                # pipe 同时返回矫正像素图 + ΔG（因为 flow_head 已挂载）
                result = pipe(
                    prompt=cur_prompt,
                    edit_image=img_input,
                    seed=random.randint(0, 2**32 - 1),
                    num_inference_steps=infer_steps,
                    width=infer_w,
                    height=infer_h,
                )
                corrected_low, delta_g = result
                # delta_g shape: (1, 2, H/8, W/8) = (1, 2, 128, 128) for 1024 input
                print(f"  扩散推理完成，ΔG shape: {tuple(delta_g.shape)}")

                # Stage 2：RAFT 计算粗光流 G_0
                G_0 = estimate_flow(
                    raft_model,
                    img_src=corrected_low,
                    img_dst=img_input,
                    device=device,
                    num_flow_updates=num_flow_updates,
                )
                print(f"  G_0 shape: {tuple(G_0.shape)}, abs_mean: {G_0.abs().mean():.2f}px")

                # Stage 3：叠加残差
                # delta_g 上采样到 1024 分辨率
                delta_g_1024 = upscale_flow(delta_g.to(device), infer_h, infer_w)
                G_refined = G_0 + delta_g_1024
                print(f"  G_refined abs_mean: {G_refined.abs().mean():.2f}px")

                # Stage 4：上采样到原始高清分辨率
                G_hires = upscale_flow(G_refined, orig_h, orig_w)

                # Stage 5：warp 原始高清图
                result_hires = warp_image_with_flow(orig_img, G_hires)
                print(f"  warp 完成，输出尺寸: {result_hires.size}")

                # 保存四张结果图
                base = os.path.splitext(output_path)[0]
                path_a = base + "_a.jpg"
                path_b = base + "_b.jpg"
                path_c = base + "_c.jpg"
                path_d = base + "_d.jpg"

                img_input.save(path_a)
                corrected_low.save(path_b)
                result_hires.save(path_c)

                corrected_resized = corrected_low.resize((orig_w, orig_h), Image.LANCZOS)
                compare = Image.new("RGB", (orig_w * 3, orig_h))
                compare.paste(orig_img, (0, 0))
                compare.paste(corrected_resized, (orig_w, 0))
                compare.paste(result_hires, (orig_w * 2, 0))
                compare.save(path_d)

                print(f"  已保存: {os.path.basename(path_c)}")

            except Exception as e:
                import traceback
                print(f"处理 {rel_path} 时出错: {e}")
                traceback.print_exc()
                with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                    f.write(f"{rel_path}: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diffusion Flow Residual 推理：扩散输出 ΔG + RAFT 粗光流 → 高清 warp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir",   type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--lora_path",   type=str, default=None)
    parser.add_argument("--flow_head_path", type=str, default=None,
                        help="FlowHead 权重路径（.pth），不传则使用随机初始化")
    parser.add_argument("--prompt", type=str,
                        default="Flatten this warped or curled document image to a flat, undistorted version. Preserve all text, lines, and content accurately.")
    parser.add_argument("--prompt_dict",  type=str, default=None)
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--img_size",     type=int, default=1024)
    parser.add_argument("--short_side_pixel", type=int, default=2048)
    parser.add_argument("--resize_mode",  type=str, default="stretch",
                        choices=["stretch", "crop", "scale_to_short_side"])
    parser.add_argument("--infer_steps",  type=int, default=50)
    parser.add_argument("--enable_dit_fp8_computation", action="store_true")
    parser.add_argument("--raft_model_size",   type=str, default="large", choices=["large", "small"])
    parser.add_argument("--num_flow_updates",  type=int, default=20)
    parser.add_argument("--total_divide", type=int, default=1)
    parser.add_argument("--divide_index", type=int, default=0)

    args = parser.parse_args()
    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lora_path=args.lora_path,
        flow_head_path=args.flow_head_path,
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
