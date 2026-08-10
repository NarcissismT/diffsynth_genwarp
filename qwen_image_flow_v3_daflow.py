#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow V3 DA-Flow 推理脚本
========================
使用 DiT LoRA + FlowHead V3（DA-Flow 风格混合光流网络）。

与 V2 的差异：
  V2：FlowHead V2 只看像素图（corrected + warped）
  V3：FlowHead V3 额外使用 DiT 中间层 Q/K 特征（DA-Flow Section 4.4）
      - 扩散特征对退化/折痕不敏感，提供几何语义感知
      - CNN 特征提供局部纹理细节
      - 两者混合，光流精度更高

流程：
  Stage 1 - 扩散模型推理，得到 corrected_low + DiT Q/K 特征
  Stage 2 - FlowHead V3 输入 (corrected_low, warped_1024, q_feats, k_feats)，输出光流
  Stage 3 - 光流上采样到原始高清分辨率
  Stage 4 - warp 原始高清图

输出文件：
  a_*.jpg  resize 后的 warped 图
  b_*.jpg  扩散模型矫正结果（仅供参考）
  c_*.jpg  FlowHead V3 warp 高清结果（最终输出）
  d_*.jpg  对比图（原始 | 扩散 | V3 warp）
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
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM_DIFFSYNTH = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
sys.path.insert(0, _HERE)
sys.path.insert(0, _UPSTREAM_DIFFSYNTH)

import diffsynth
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from utils.flow_head_v3_daflow import FlowHeadV3DAFlow
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


def pil_to_tensor(img: Image.Image, device) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) * (2.0 / 255.0) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def load_pipeline(checkpoint_path: str, device,
                  flow_head_v3_path: str = None,
                  num_dit_layers: int = 4):
    """
    加载 pipeline 和 FlowHead V3。
    支持两种 checkpoint 格式：
      1. 联合训练 checkpoint（含 LoRA + flow_head_v3. 前缀的 keys）
      2. 分开的 lora_path + flow_head_v3_path
    """
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

    # 初始化 FlowHead V3
    flow_model = FlowHeadV3DAFlow(iters=12, num_dit_layers=num_dit_layers).to(device).eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(checkpoint_path, framework="pt") as f:
            all_keys = list(f.keys())
        v3_keys = [k for k in all_keys if k.startswith("flow_head_v3.")]

        if v3_keys:
            # 联合训练 checkpoint
            print(f"加载联合训练 checkpoint（LoRA + FlowHead V3）: {checkpoint_path}")
            sd = load_file(checkpoint_path, device=str(device))
            pipe.load_lora(pipe.dit, checkpoint_path)
            v3_sd = {k.replace("flow_head_v3.", ""): v
                     for k, v in sd.items() if k.startswith("flow_head_v3.")}
            missing, _ = flow_model.load_state_dict(v3_sd, strict=False)
            print(f"  FlowHead V3 keys: {len(v3_sd)}")
            if missing:
                print(f"  缺失 keys（BN buffer，使用默认值）: {len(missing)} 个")
        else:
            print(f"加载 LoRA: {checkpoint_path}")
            pipe.load_lora(pipe.dit, checkpoint_path)
            if flow_head_v3_path and os.path.exists(flow_head_v3_path):
                flow_model.load_state_dict(
                    torch.load(flow_head_v3_path, map_location=device))
                print(f"FlowHead V3 权重: {flow_head_v3_path}")
            else:
                print("警告：FlowHead V3 使用随机初始化")
    else:
        print("警告：未提供 checkpoint，FlowHead V3 随机初始化")

    # 启用 DA-Flow 特征提取
    pipe.extract_daflow_features = True
    pipe.daflow_num_layers = num_dit_layers

    pipe.enable_vram_management()
    return pipe, flow_model


def process_images(
    input_dir, output_dir, lora_path, flow_head_v3_path, prompt,
    batch_size=1, prompt_dict=None,
    short_side_pixel=2048, img_size=1024,
    enable_dit_fp8_computation=False,
    total_divide=1, divide_index=0,
    infer_steps=50, resize_mode="stretch",
    flow_iters=12, num_dit_layers=4,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("正在加载模型...")
    pipe, flow_model = load_pipeline(
        lora_path, device, flow_head_v3_path, num_dit_layers)

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
            if os.path.exists(base + "_b.jpg"):
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

                # Stage 1：扩散推理，同时返回 corrected_low + Q/K 特征
                result = pipe(
                    prompt=cur_prompt,
                    edit_image=img_input,
                    seed=random.randint(0, 2**32 - 1),
                    num_inference_steps=infer_steps,
                    width=infer_w,
                    height=infer_h,
                )
                corrected_low, q_features, k_features = result
                print(f"  扩散推理完成，提取 {len(q_features)} 层 Q/K 特征")

                # Stage 2：FlowHead V3 预测光流
                corrected_t = pil_to_tensor(corrected_low, device)
                warped_t    = pil_to_tensor(img_input,     device)

                with torch.no_grad():
                    flow_low = flow_model(
                        corrected_t, warped_t,
                        q_features, k_features,
                        iters=flow_iters,
                    )
                print(f"  光流估计完成，abs_mean: {flow_low.abs().mean():.2f}px")

                # Stage 3：上采样
                flow_hires = upscale_flow(flow_low, orig_h, orig_w)

                # Stage 4：warp 原始高清图
                result_hires = warp_image_with_flow(orig_img, flow_hires)
                print(f"  warp 完成，输出尺寸: {result_hires.size}")

                # 保存
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
    parser.add_argument("--input_dir",          type=str, required=True)
    parser.add_argument("--output_dir",          type=str, required=True)
    parser.add_argument("--lora_path",            type=str, default=None,
                        help="LoRA / 联合训练 checkpoint 路径")
    parser.add_argument("--flow_head_v3_path",    type=str, default=None,
                        help="FlowHead V3 权重路径（单独保存时使用）")
    parser.add_argument("--prompt", type=str,
                        default="Flatten this warped or curled document image to a flat, undistorted version. Preserve all text, lines, and content accurately.")
    parser.add_argument("--prompt_dict",          type=str, default=None)
    parser.add_argument("--batch_size",           type=int, default=1)
    parser.add_argument("--img_size",             type=int, default=1024)
    parser.add_argument("--short_side_pixel",     type=int, default=2048)
    parser.add_argument("--resize_mode",          type=str, default="stretch",
                        choices=["stretch", "crop", "scale_to_short_side"])
    parser.add_argument("--infer_steps",          type=int, default=50)
    parser.add_argument("--flow_iters",           type=int, default=12)
    parser.add_argument("--num_dit_layers",       type=int, default=4,
                        help="提取 DiT top-L 层的 Q/K 特征（默认 4，与 DA-Flow 一致）")
    parser.add_argument("--enable_dit_fp8_computation", action="store_true")
    parser.add_argument("--total_divide",         type=int, default=1)
    parser.add_argument("--divide_index",         type=int, default=0)
    args = parser.parse_args()

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lora_path=args.lora_path,
        flow_head_v3_path=args.flow_head_v3_path,
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
        flow_iters=args.flow_iters,
        num_dit_layers=args.num_dit_layers,
    )
