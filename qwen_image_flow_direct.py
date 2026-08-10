#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow Direct 推理脚本
====================
使用联合训练的 checkpoint（LoRA + FlowHead），无需 RAFT。

流程：
  Stage 1 - 扩散模型推理，得到矫正像素图 + FlowHead 输出的完整光流 ΔG
  Stage 2 - ΔG 上采样到原始高清分辨率
  Stage 3 - 用光流对原始高清图做像素重采样

输出文件：
  a_*.jpg  原始 warped 图（resize 到推理尺寸）
  b_*.jpg  扩散模型矫正结果（仅供参考）
  c_*.jpg  FlowHead warp 高清矫正结果（原始分辨率）
  d_*.jpg  对比图（原始 | 扩散 | FlowHead warp 横向拼接）
"""

import os
import sys

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
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowHead
from diffsynth import load_state_dict
from utils.flow_utils import upscale_flow, warp_image_with_flow

_MODEL_BASE = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit"
DIT_SHARDS = [
    f"{_MODEL_BASE}/transformer/diffusion_pytorch_model-{i:05d}-of-00009.safetensors"
    for i in range(1, 10)
]
TEXT_ENCODER_SHARDS = [
    f"{_MODEL_BASE}/text_encoder/model-{i:05d}-of-00004.safetensors"
    for i in range(1, 5)
]
VAE_PATH       = f"{_MODEL_BASE}/vae/diffusion_pytorch_model.safetensors"
TOKENIZER_PATH = f"{_MODEL_BASE}/tokenizer"
PROCESSOR_PATH = f"{_MODEL_BASE}/processor"


def preprocess_image(img, img_size, resize_mode, short_side_pixel):
    width, height = img.size
    if resize_mode == "crop":
        short_side = min(width, height)
        scale = short_side_pixel / short_side
        new_h, new_w = round(height * scale), round(width * scale)
        img = torchvision.transforms.functional.resize(
            img, (new_h, new_w),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        return torchvision.transforms.functional.center_crop(img, (img_size, img_size)), img_size, img_size
    elif resize_mode == "stretch":
        return img.resize((img_size, img_size), Image.LANCZOS), img_size, img_size
    elif resize_mode == "scale_to_short_side":
        short_side = min(width, height)
        scale = img_size / short_side
        new_h = int(height * scale) // 16 * 16
        new_w = int(width  * scale) // 16 * 16
        return torchvision.transforms.functional.resize(
            img, (new_h, new_w),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR), new_w, new_h
    else:
        raise ValueError(f"未知的 resize_mode: {resize_mode}")


def load_pipeline(checkpoint_path: str, device):
    """加载 pipeline，自动识别联合训练 checkpoint（LoRA + FlowHead）。"""
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

    pipe.flow_head = FlowHead(latent_channels=16, out_channels=2).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(checkpoint_path, framework="pt") as f:
            all_keys = list(f.keys())
        flow_keys = [k for k in all_keys if "flow_head" in k]

        if flow_keys:
            print(f"加载联合训练 checkpoint（LoRA + FlowHead）: {checkpoint_path}")
            state_dict = load_file(checkpoint_path, device=str(device))
            pipe.load_lora(pipe.dit, checkpoint_path)
            flow_sd = {k.replace("pipe.flow_head.", ""): v
                       for k, v in state_dict.items() if "flow_head" in k}
            pipe.flow_head.load_state_dict(flow_sd)
            print(f"  FlowHead keys: {len(flow_sd)}")
        else:
            print(f"加载 LoRA checkpoint: {checkpoint_path}")
            pipe.load_lora(pipe.dit, checkpoint_path)
            print("  FlowHead 使用随机初始化")
    else:
        print("未提供 checkpoint，使用基础模型 + 随机 FlowHead")

    pipe.enable_vram_management()
    return pipe


def process_images(
    input_dir, output_dir, checkpoint_path, prompt,
    batch_size=1, prompt_dict=None,
    short_side_pixel=2048, img_size=1024,
    enable_dit_fp8_computation=False,
    total_divide=1, divide_index=0,
    infer_steps=50, resize_mode="stretch",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = load_pipeline(checkpoint_path, device)

    prompt_map = None
    if prompt_dict:
        prompt_map = {}
        with open(prompt_dict, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompt_map[row["image"]] = row["prompt"]

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = [input_dir] if os.path.isfile(input_dir) else []
    if not image_files:
        for root, _, _ in os.walk(input_dir):
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(root, ext)))
                image_files.extend(glob.glob(os.path.join(root, ext.upper())))

    print(f"找到 {len(image_files)} 张图像")
    total_batches = (len(image_files) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(tqdm(range(0, len(image_files), batch_size),
                                        total=total_batches, desc="处理图片")):
        if total_divide > 1 and (batch_idx % total_divide) != divide_index:
            continue

        for img_path in image_files[i: i + batch_size]:
            rel_path = os.path.basename(img_path) if os.path.isfile(input_dir) \
                       else os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            base = os.path.splitext(output_path)[0]
            done_marker = base + "_b.jpg"
            if os.path.exists(done_marker):
                print(f"跳过已处理: {rel_path}")
                continue

            try:
                orig_img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = orig_img.size
                print(f"处理: {rel_path}  原始尺寸: {orig_w}×{orig_h}")

                img_input, infer_w, infer_h = preprocess_image(
                    orig_img, img_size, resize_mode, short_side_pixel)

                cur_prompt = prompt_map.get(os.path.basename(img_path), prompt) \
                             if prompt_map else prompt

                # 扩散推理 → 同时输出矫正图 + FlowHead 光流
                result = pipe(
                    prompt=cur_prompt,
                    edit_image=img_input,
                    seed=random.randint(0, 2**32 - 1),
                    num_inference_steps=infer_steps,
                    width=infer_w,
                    height=infer_h,
                )
                corrected_low, delta_g = result
                print(f"  扩散推理完成，delta_g shape: {tuple(delta_g.shape)}, "
                      f"abs_mean: {delta_g.abs().mean():.2f}px")

                # 上采样光流到原始高清分辨率
                G_hires = upscale_flow(delta_g.to(device), orig_h, orig_w)

                # warp 原始高清图
                result_hires = warp_image_with_flow(orig_img, G_hires)
                print(f"  warp 完成，输出尺寸: {result_hires.size}")

                # 保存结果
                img_input.save(base + "_a.jpg")
                corrected_low.save(base + "_b.jpg")
                result_hires.save(base + "_c.jpg")

                corrected_resized = corrected_low.resize((orig_w, orig_h), Image.LANCZOS)
                compare = Image.new("RGB", (orig_w * 3, orig_h))
                compare.paste(orig_img,          (0,        0))
                compare.paste(corrected_resized, (orig_w,   0))
                compare.paste(result_hires,      (orig_w*2, 0))
                compare.save(base + "_d.jpg")

                print(f"  已保存: {os.path.basename(base)}_[a-d].jpg")

            except Exception as e:
                import traceback
                print(f"处理 {rel_path} 时出错: {e}")
                traceback.print_exc()
                with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                    f.write(f"{rel_path}: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir",   type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="联合训练 checkpoint 路径（包含 LoRA + FlowHead）")
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
    parser.add_argument("--total_divide", type=int, default=1)
    parser.add_argument("--divide_index", type=int, default=0)
    args = parser.parse_args()

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
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
    )
