#!/usr/bin/env python3
"""
FlowHead V4 定量评测脚本
========================
支持两种模式：

模式 A（合成数据，有 flow_gt）：从训练 CSV 末尾采样，计算：
  - EPE（End-Point Error）：flow 像素级 L2 误差，越低越好
  - Warp L1：grid_sample(warped, flow_pred) 与 rectified_gt 的 L1 距离，越低越好

模式 B（真实数据，无 flow_gt）：从真实图目录读取，只计算 Warp L1
  并保存对比图（原始 | warp 结果）供肉眼对比。
  使用 --real_val_dir 指定真实图目录（例如 silver_bullet dewarp 验证集）。

使用方法：
  # 模式 A（合成数据）
  python eval_flow_v4_quant.py \
      --ckpt_path /path/to/step-XXXXX.safetensors \
      --dit_target_layers "35"

  # 模式 B（真实数据）
  python eval_flow_v4_quant.py \
      --ckpt_path /path/to/step-XXXXX.safetensors \
      --dit_target_layers "35" \
      --real_val_dir /juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/dewarp \
      --vis_output_dir /path/to/vis_output

使用 scripts/eval_exp_compare.sh 可以批量跑 6 个实验并输出对比表格。
"""

import os
import sys
import csv
import json
import argparse
import random
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
sys.path.insert(0, _HERE)
_PROJECT_ROOT = "/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp"
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _UPSTREAM)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.trainers.utils import DiffusionTrainingModule
from utils.flow_head_v4_layer_probe import FlowHeadV4LayerProbe

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


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def parse_layer_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def pil_to_tensor(img: Image.Image, device) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32) * (2.0 / 255.0) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def warp_batch(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    B, _, H, W = flow.shape
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=flow.device),
        torch.linspace(-1, 1, W, device=flow.device),
        indexing="ij",
    )
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = base + torch.stack([
        flow[:, 0] * 2.0 / (W - 1),
        flow[:, 1] * 2.0 / (H - 1),
    ], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


# ---------------------------------------------------------------------------
# 验证集采样：每个 category 取末尾 N 张
# ---------------------------------------------------------------------------

def build_val_subset(csv_path: str, per_category: int = 100,
                     val_csv: str = None) -> list:
    """
    返回 list of dict，每条含 image/edit_image/flow_gt_path 三个字段。
    若指定 val_csv 则直接加载；否则从 csv_path 每 category 取末尾 per_category 张。
    """
    if val_csv and os.path.exists(val_csv):
        with open(val_csv) as f:
            return list(csv.DictReader(f))

    by_cat = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("flow_gt_path") and os.path.exists(row["flow_gt_path"]):
                by_cat[row["category"]].append(row)

    subset = []
    for cat, rows in sorted(by_cat.items()):
        taken = rows[-per_category:]
        subset.extend(taken)
        print(f"  {cat}: 取 {len(taken)} 张（共 {len(rows)} 张）")
    return subset


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

class _LoRAHelper(DiffusionTrainingModule):
    """只借用 add_lora_to_model 和 mapping_lora_state_dict 工具方法。"""
    def forward(self, *a, **kw): pass


def load_models(args, device):
    print("加载 Qwen pipeline...")
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

    # 加载 LoRA
    if args.lora_checkpoint and os.path.exists(args.lora_checkpoint):
        helper = _LoRAHelper()
        lora_target_modules = [
            "to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj",
            "to_out.0", "to_add_out", "img_mlp.net.2", "img_mod.1",
            "txt_mlp.net.2", "txt_mod.1",
        ]
        dit_with_lora = helper.add_lora_to_model(
            pipe.dit, target_modules=lora_target_modules, lora_rank=32)
        sd = load_state_dict(args.lora_checkpoint)
        sd = helper.mapping_lora_state_dict(sd)
        dit_with_lora.load_state_dict(sd, strict=False)
        pipe.dit = dit_with_lora
        print(f"LoRA 加载完成: {args.lora_checkpoint}")

    pipe.eval()
    pipe.enable_vram_management()

    # 加载 FlowHead V4
    target_layers = parse_layer_list(args.dit_target_layers)
    num_layers = len(target_layers)
    flow_model = FlowHeadV4LayerProbe(
        iters=args.flow_iters,
        num_dit_layers=num_layers,
        diff_out_ch=args.diff_out_ch,
    ).to(device).eval()

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        from safetensors.torch import load_file
        sd = load_file(args.ckpt_path, device=str(device))
        # 训练时 ModelLogger 已用 --remove_prefix_in_ckpt "flow_head." 剥过顶层前缀，
        # 直接加载即可。不再做 replace，否则会把内层 FlowPredHead 的 flow_head. 也剥掉。
        missing, unexpected = flow_model.load_state_dict(sd, strict=False)
        critical_missing = [k for k in missing
                            if not any(b in k for b in ['running_mean','running_var','num_batches_tracked'])]
        print(f"FlowHead V4 加载: {len(sd)} key，缺失 {len(missing)}（参数 {len(critical_missing)}, 其余 BN buffer），多余 {len(unexpected)}")
        if critical_missing: print(f"  ! 关键缺失：{critical_missing[:5]}")
        if unexpected: print(f"  ! 多余 ckpt key：{unexpected[:5]}")
    else:
        print("警告：未提供 ckpt_path，FlowHead V4 使用随机权重（EPE 无意义）")

    return pipe, flow_model, target_layers


# ---------------------------------------------------------------------------
# 单步 DiT 特征提取（与训练脚本逻辑完全一致）
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_qk_features(pipe, inputs: dict, target_layers: list,
                         max_target_layer: int, device):
    dit_raw = pipe.dit.module if hasattr(pipe.dit, 'module') else pipe.dit

    timestep_id = torch.randint(400, 600, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype, device=device)

    latents = pipe.scheduler.add_noise(
        inputs["input_latents"], inputs["noise"], timestep)

    h_val = inputs["height"]
    w_val = inputs["width"]
    img_shapes = [(latents.shape[0], latents.shape[2] // 2, latents.shape[3] // 2)]
    prompt_emb      = inputs["prompt_emb"]
    prompt_emb_mask = inputs["prompt_emb_mask"]
    txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()

    image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)",
                      H=h_val // 16, W=w_val // 16, P=2, Q=2)
    image = dit_raw.img_in(image)
    text  = dit_raw.txt_in(dit_raw.txt_norm(prompt_emb))
    conditioning     = dit_raw.time_text_embed(timestep, image.dtype)
    image_rotary_emb = dit_raw.pos_embed(img_shapes, txt_seq_lens, device=device)

    q_features, k_features = [], []
    for block_idx, block in enumerate(dit_raw.transformer_blocks):
        if block_idx in target_layers:
            img_normed = block.img_norm1(image)
            img_mod_attn, _ = block.img_mod(conditioning).chunk(2, dim=-1)
            img_modulated, _ = block._modulate(img_normed, img_mod_attn)
            q_features.append(block.attn.to_q(img_modulated).float())
            k_features.append(block.attn.to_k(img_modulated).float())
        text, image = block(image=image, text=text, temb=conditioning,
                             image_rotary_emb=image_rotary_emb)
        if block_idx == max_target_layer:
            break

    return q_features, k_features


# ---------------------------------------------------------------------------
# 主评测循环
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== FlowHead V4 定量评测 ===")
    print(f"ckpt: {args.ckpt_path}")
    print(f"层:   {args.dit_target_layers}")

    # 验证集
    print("\n构建验证集...")
    val_data = build_val_subset(
        args.csv_path, per_category=args.per_category, val_csv=args.val_csv)
    print(f"验证集共 {len(val_data)} 张")

    # 加载模型
    pipe, flow_model, target_layers = load_models(args, device)
    max_target_layer = max(target_layers)
    pipe.scheduler.set_timesteps(1000, training=True)

    epe_list, warp_l1_list = [], []

    for row in tqdm(val_data, desc="评测"):
        try:
            warped_pil    = Image.open(row["edit_image"]).convert("RGB")
            rectified_pil = Image.open(row["image"]).convert("RGB")
            flow_gt = torch.from_numpy(
                np.load(row["flow_gt_path"]).astype(np.float32)).unsqueeze(0).to(device)

            # resize 到 max_pixels（与训练一致）
            max_px = args.max_pixels
            W, H = warped_pil.size
            if H * W > max_px:
                scale = (max_px / (H * W)) ** 0.5
                new_H = int(H * scale) // 16 * 16
                new_W = int(W * scale) // 16 * 16
                warped_pil    = warped_pil.resize((new_W, new_H), Image.LANCZOS)
                rectified_pil = rectified_pil.resize((new_W, new_H), Image.LANCZOS)
            else:
                new_H, new_W = H // 16 * 16, W // 16 * 16
                warped_pil    = warped_pil.resize((new_W, new_H), Image.LANCZOS)
                rectified_pil = rectified_pil.resize((new_W, new_H), Image.LANCZOS)

            warped_t    = pil_to_tensor(warped_pil, device)
            rectified_t = pil_to_tensor(rectified_pil, device)

            # 用 pipeline units 准备 inputs（与训练 forward_preprocess 完全一致）
            prompt = row.get("prompt", "Flatten this warped document.")
            inputs_posi = {"prompt": prompt}
            inputs_nega = {"negative_prompt": ""}
            inputs_shared = {
                "input_image": warped_pil,   # warped 作为输入，与训练一致
                "edit_image":  warped_pil,
                "height": new_H, "width": new_W,
                "cfg_scale": 1,
                "rand_device": device,
                "use_gradient_checkpointing": False,
                "use_gradient_checkpointing_offload": False,
            }
            with torch.no_grad():
                for unit in pipe.units:
                    inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                        unit, pipe, inputs_shared, inputs_posi, inputs_nega)
            inputs_fake = {**inputs_shared, **inputs_posi,
                           "noise": torch.randn_like(inputs_shared["input_latents"])}

            pipe.load_models_to_device(["dit"])
            q_feats, k_feats = extract_qk_features(
                pipe, inputs_fake, target_layers, max_target_layer, device)

            # FlowHead V4 推理
            with torch.no_grad():
                flow_pred = flow_model(
                    rectified_t.float(), warped_t.float(),
                    q_feats, k_feats, iters=args.flow_iters)

            # 缩放 flow_gt 到预测分辨率
            _, _, fh, fw = flow_pred.shape
            scale = float(fh) / flow_gt.shape[-2]
            flow_gt_r = F.interpolate(flow_gt, size=(fh, fw),
                                       mode="bilinear", align_corners=True) * scale

            # EPE
            epe = torch.norm(flow_pred - flow_gt_r, dim=1).mean().item()
            epe_list.append(epe)

            # Warp L1
            warp_result = warp_batch(
                F.interpolate(warped_t, size=(fh, fw), mode="bilinear", align_corners=True),
                flow_pred,
            )
            rect_r = F.interpolate(rectified_t, size=(fh, fw),
                                    mode="bilinear", align_corners=True)
            warp_l1 = F.l1_loss(warp_result, rect_r).item()
            warp_l1_list.append(warp_l1)

        except Exception as e:
            print(f"跳过 {row.get('edit_image', '')}: {e}")
            continue

    mean_epe    = float(np.mean(epe_list))    if epe_list    else float("nan")
    mean_warp   = float(np.mean(warp_l1_list)) if warp_l1_list else float("nan")

    print(f"\n{'='*50}")
    print(f"实验:      {args.exp_name or args.dit_target_layers}")
    print(f"样本数:    {len(epe_list)}")
    print(f"EPE  (↓):  {mean_epe:.4f}")
    print(f"WarpL1 (↓):{mean_warp:.4f}")
    print(f"{'='*50}")

    # 输出 JSON 便于汇总
    result = {
        "exp_name":     args.exp_name or args.dit_target_layers,
        "dit_layers":   args.dit_target_layers,
        "ckpt_path":    args.ckpt_path,
        "n_samples":    len(epe_list),
        "mean_epe":     round(mean_epe, 4),
        "mean_warp_l1": round(mean_warp, 4),
        "mode":         "synthetic",
    }
    if args.result_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
        with open(args.result_json, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.result_json}")

    return result


# ---------------------------------------------------------------------------
# 汇总：从多个 result_json 打印对比表格
# ---------------------------------------------------------------------------

def print_comparison_table(json_paths: list):
    results = []
    for p in json_paths:
        if os.path.exists(p):
            with open(p) as f:
                results.append(json.load(f))
        else:
            print(f"警告：找不到 {p}")

    if not results:
        print("没有可汇总的结果")
        return

    # 分合成/真实分别排序
    synthetic = [r for r in results if r.get("mode") != "real"]
    real      = [r for r in results if r.get("mode") == "real"]

    for group, label in [(synthetic, "合成数据（有 flow_gt）"), (real, "真实数据（无 flow_gt）")]:
        if not group:
            continue
        group.sort(key=lambda r: r.get("mean_epe", r.get("mean_warp_l1", 9999)))
        header = f"{'实验名':<32} {'层配置':<22} {'样本数':>6} {'EPE ↓':>10} {'WarpL1 ↓':>12}"
        print(f"\n[{label}]")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for r in group:
            epe_str  = f"{r['mean_epe']:>10.4f}" if r.get("mean_epe") is not None else f"{'N/A':>10}"
            warp_str = f"{r['mean_warp_l1']:>12.4f}"
            print(f"{r['exp_name']:<32} {r['dit_layers']:<22} {r['n_samples']:>6} "
                  f"{epe_str} {warp_str}")
        print("=" * len(header))
        best = group[0]
        print(f"最优: {best['exp_name']}")


# ---------------------------------------------------------------------------
# 模式 B：真实图推理 + 保存对比图（无 flow_gt，视觉评估）
# ---------------------------------------------------------------------------

def evaluate_real(args):
    """
    正确的两阶段推理（与 qwen_image_flow_v4.py 完全一致）：
      Stage 1: Qwen 完整扩散推理 → corrected_low + Q/K 特征
      Stage 2: flow_model(corrected_low, warped, q_feats, k_feats) → flow
      Stage 3: warp(warped, flow) → 最终矫正图

    保存三合一对比图：原始 warped | 扩散 corrected_low | FlowHead V4 warp 结果
    """
    import glob
    import random
    # 直接复用 qwen_image_flow_v4 的 pipeline 加载和 patch 逻辑
    from qwen_image_flow_v4 import (
        load_pipeline as load_pipeline_v4,
        preprocess_image as preprocess_v4,
        pil_to_tensor as pil_to_tensor_v4,
    )
    from utils.flow_utils import upscale_flow, warp_image_with_flow

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 真实图评测（模式 B，正确两阶段推理）===")
    print(f"输入目录: {args.real_val_dir}")

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
    img_files = []
    for ext in exts:
        img_files.extend(glob.glob(os.path.join(args.real_val_dir, ext)))
    img_files = sorted(img_files)
    if args.max_real_samples > 0:
        img_files = img_files[:args.max_real_samples]
    print(f"找到 {len(img_files)} 张图（最多处理 {args.max_real_samples}）")

    os.makedirs(args.vis_output_dir, exist_ok=True)

    # lora_path 未设置时 fallback 到 lora_checkpoint
    if not args.lora_path:
        args.lora_path = args.lora_checkpoint

    # 使用与推理脚本相同的 load_pipeline（含正确的 pipeline patch）
    pipe, flow_model, target_layers = load_pipeline_v4(args, device)

    PROMPT = ("Apply geometric correction to the input image to eliminate distortions "
              "such as skew, curl, folds, or non-frontal perspective, producing a flat, "
              "front-facing image. Preserve all original text and content.")

    warp_l1_list = []

    for img_path in tqdm(img_files, desc="真实图推理"):
        try:
            orig_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = orig_img.size

            # resize 到推理尺寸
            img_input, infer_w, infer_h = preprocess_v4(
                orig_img, args.img_size, args.resize_mode)

            # Stage 1：Qwen 完整扩散推理 + 提取 Q/K 特征
            result = pipe(
                prompt=PROMPT,
                edit_image=img_input,
                seed=random.randint(0, 2**32 - 1),
                num_inference_steps=args.infer_steps,
                width=infer_w,
                height=infer_h,
            )
            corrected_low, q_feats, k_feats = result

            if not q_feats:
                print(f"  警告：Q/K 特征为空，跳过 {os.path.basename(img_path)}")
                continue

            # Stage 2：FlowHead V4 预测光流
            corrected_t = pil_to_tensor_v4(corrected_low, device)
            warped_t    = pil_to_tensor_v4(img_input,     device)

            with torch.no_grad():
                flow_low = flow_model(
                    corrected_t.float(), warped_t.float(),
                    q_feats, k_feats, iters=args.flow_iters,
                )

            # Stage 3：上采样 + 高清 warp
            flow_hires   = upscale_flow(flow_low, orig_h, orig_w)
            result_hires = warp_image_with_flow(orig_img, flow_hires)

            # 计算推理分辨率下的 warp L1（corrected_low vs warp结果，衡量几何对齐）
            fh, fw = flow_low.shape[-2], flow_low.shape[-1]
            corr_r = F.interpolate(corrected_t, size=(fh, fw), mode="bilinear", align_corners=True)
            warp_r = warp_batch(
                F.interpolate(warped_t, size=(fh, fw), mode="bilinear", align_corners=True),
                flow_low)
            warp_l1 = F.l1_loss(warp_r, corr_r).item()
            warp_l1_list.append(warp_l1)

            # 保存三合一对比图：原始 | 扩散结果 | FlowHead V4 warp
            corrected_resized = corrected_low.resize((orig_w, orig_h), Image.LANCZOS)
            canvas = Image.new("RGB", (orig_w * 3, orig_h))
            canvas.paste(orig_img,          (0,          0))
            canvas.paste(corrected_resized, (orig_w,     0))
            canvas.paste(result_hires,      (orig_w * 2, 0))

            out_name = os.path.splitext(os.path.basename(img_path))[0] + "_compare.jpg"
            canvas.save(os.path.join(args.vis_output_dir, out_name), quality=90)

        except Exception as e:
            import traceback
            print(f"跳过 {img_path}: {e}")
            traceback.print_exc()
            continue

    mean_warp = float(np.mean(warp_l1_list)) if warp_l1_list else float("nan")
    print(f"\n{'='*50}")
    print(f"实验:        {args.exp_name or args.dit_target_layers}")
    print(f"真实图数:    {len(warp_l1_list)}")
    print(f"WarpL1（corrected_low vs flow结果，越低几何对齐越好）: {mean_warp:.4f}")
    print(f"对比图（原始|扩散|V4 warp）保存至: {args.vis_output_dir}")
    print(f"{'='*50}")

    result = {
        "exp_name":        args.exp_name or args.dit_target_layers,
        "dit_layers":      args.dit_target_layers,
        "ckpt_path":       args.ckpt_path,
        "n_samples":       len(warp_l1_list),
        "mean_epe":        None,
        "mean_warp_l1":    round(mean_warp, 4),
        "mode":            "real",
        "vis_output_dir":  args.vis_output_dir,
    }
    if args.result_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
        with open(args.result_json, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.result_json}")

    return result


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # 数据（合成）
    parser.add_argument("--csv_path", type=str,
                        default="/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_with_flow.csv")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="指定单独验证集 CSV，不传则从 csv_path 末尾采样")
    parser.add_argument("--per_category", type=int, default=100,
                        help="每个 category 取末尾多少张（默认100，共约600张）")
    # 数据（真实）
    parser.add_argument("--real_val_dir", type=str, default=None,
                        help="真实图验证集目录（无 flow_gt），与 --csv_path 二选一")
    parser.add_argument("--max_real_samples", type=int, default=200,
                        help="真实验证集最多处理多少张（默认200）")
    parser.add_argument("--vis_output_dir", type=str,
                        default="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/eval_vis",
                        help="真实图对比图输出目录")
    # 通用
    parser.add_argument("--max_pixels", type=int, default=1048576)
    # 模型
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dit_target_layers", type=str, required=True,
                        help="例如 \"35\" 或 \"23,35,47\"")
    parser.add_argument("--lora_checkpoint", type=str,
                        default="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors",
                        help="合成数据模式用的 LoRA（load_models 内使用）")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="真实图模式用的 LoRA（load_pipeline_v4 内使用），不传则与 lora_checkpoint 相同")
    parser.add_argument("--diff_out_ch", type=int, default=96)
    parser.add_argument("--flow_iters", type=int, default=12)
    # 真实图推理额外参数（evaluate_real 用）
    parser.add_argument("--img_size",    type=int, default=1024)
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--resize_mode", type=str, default="stretch",
                        choices=["stretch", "crop", "scale_to_short_side"])
    # 输出
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--result_json", type=str, default=None)
    # 汇总模式
    parser.add_argument("--compare", type=str, default=None,
                        help="逗号分隔的 result_json 路径，打印对比表格")
    args = parser.parse_args()

    if args.compare:
        json_paths = [p.strip() for p in args.compare.split(",") if p.strip()]
        print_comparison_table(json_paths)
    elif args.real_val_dir:
        evaluate_real(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
