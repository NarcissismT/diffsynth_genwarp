#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow V4.2 Path B 推理脚本（双图跨段 Q/K）
==========================================
与 V4.1 推理的关键区别：特征提取改为 DA-Flow 风格的双图跨段 Q/K。
  V4.1：只喂 warped 一张图，Q/K 同图 → 无跨图位移信号
  V4.2：corrected_low + warped 拼接喂 DiT（利用 Qwen 原生跨图 attention），
        Q 切 corrected_low 段、K 切 warped 段 → 真·跨图

两阶段推理：
  Stage 1 - Qwen 扩散完整推理（50步）→ corrected_low
  Stage 2 - 把 corrected_low(main,加噪) + warped(edit,干净) 拼接喂 DiT，
            跨段提取 Q/K → FlowHeadV4_3(corrected_low, warped, q, k) → flow_low
  Stage 3 - flow 上采样到原始高清分辨率
  Stage 4 - 用高清 flow 对原始高清 warped 图做像素重采样 → 最终输出

输出文件（每张输入图片生成 4 个）：
  a_*.jpg  resize 后的 warped 图（推理输入）
  b_*.jpg  扩散模型矫正结果（corrected_low，仅供参考）
  c_*.jpg  FlowHead V4.2 warp 高清结果（最终输出，文字来自原图像素）
  d_*.jpg  对比图（原始 | 扩散 | V4.2 warp）

参数必须与训练一致：--dit_target_layers、--k_source、--img_size（512）

用法：
  bash scripts/flow_v4_2_sample.sh
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
_UPSTREAM = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
# 主项目根目录（flow_utils.py 在这里），无论从哪里调用都需要加
_PROJECT_ROOT = "/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp"
sys.path.insert(0, _HERE)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _UPSTREAM)

import diffsynth
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.trainers.utils import DiffusionTrainingModule

from utils.flow_head_v4_3 import FlowHeadV4_3
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


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------

def preprocess_image(img: Image.Image, img_size: int, resize_mode: str,
                     short_side_pixel: int = 2048):
    width, height = img.size
    if resize_mode == "crop":
        short_side = min(width, height)
        scale = short_side_pixel / short_side
        new_h, new_w = round(height * scale), round(width * scale)
        img = torchvision.transforms.functional.resize(
            img, (new_h, new_w),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        return (torchvision.transforms.functional.center_crop(img, (img_size, img_size)),
                img_size, img_size)
    elif resize_mode == "stretch":
        return img.resize((img_size, img_size), Image.LANCZOS), img_size, img_size
    elif resize_mode == "scale_to_short_side":
        short_side = min(width, height)
        scale = img_size / short_side
        new_h = int(height * scale) // 16 * 16
        new_w = int(width  * scale) // 16 * 16
        return (torchvision.transforms.functional.resize(
            img, (new_h, new_w),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            new_w, new_h)
    else:
        raise ValueError(f"未知的 resize_mode: {resize_mode}")


def pil_to_tensor(img: Image.Image, device) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) * (2.0 / 255.0) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def parse_layer_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def visualize_flow(flow: torch.Tensor, save_path_prefix: str):
    """
    保存 flow 的 6 张诊断图（diagnose.md Step 3）：
      _flow_dx.png         dx 分量灰度图（带色阶）
      _flow_dy.png         dy 分量灰度图
      _flow_mag.png        flow magnitude (sqrt(dx²+dy²))
      _flow_quiver.png     箭头图（每 16 px 采样）
      _flow_jacobian.png   Jacobian determinant
      _flow_fold.png       fold 区域（J ≤ 0）

    flow: (1, 2, H, W) 像素单位
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    flow_np = flow[0].detach().cpu().numpy()  # (2, H, W)
    dx, dy = flow_np[0], flow_np[1]
    H, W = dx.shape
    mag = np.sqrt(dx ** 2 + dy ** 2)

    # ---- dx ----
    plt.figure(figsize=(6, 6))
    plt.imshow(dx, cmap='RdBu_r', vmin=-np.abs(flow_np).max(), vmax=np.abs(flow_np).max())
    plt.colorbar(label='dx (px)')
    plt.title(f"dx: mean={dx.mean():+.2f}, std={dx.std():.2f}, "
              f"min={dx.min():+.2f}, max={dx.max():+.2f}")
    plt.savefig(save_path_prefix + "_flow_dx.png", dpi=80, bbox_inches='tight')
    plt.close()

    # ---- dy ----
    plt.figure(figsize=(6, 6))
    plt.imshow(dy, cmap='RdBu_r', vmin=-np.abs(flow_np).max(), vmax=np.abs(flow_np).max())
    plt.colorbar(label='dy (px)')
    plt.title(f"dy: mean={dy.mean():+.2f}, std={dy.std():.2f}, "
              f"min={dy.min():+.2f}, max={dy.max():+.2f}")
    plt.savefig(save_path_prefix + "_flow_dy.png", dpi=80, bbox_inches='tight')
    plt.close()

    # ---- magnitude ----
    plt.figure(figsize=(6, 6))
    plt.imshow(mag, cmap='viridis')
    plt.colorbar(label='|flow| (px)')
    plt.title(f"|flow|: mean={mag.mean():.2f}, max={mag.max():.2f}")
    plt.savefig(save_path_prefix + "_flow_mag.png", dpi=80, bbox_inches='tight')
    plt.close()

    # ---- quiver（箭头图）----
    step = max(1, H // 32)
    Y, X = np.mgrid[0:H:step, 0:W:step]
    plt.figure(figsize=(6, 6))
    plt.imshow(mag, cmap='gray', alpha=0.4)
    plt.quiver(X, Y, dx[::step, ::step], -dy[::step, ::step],   # -dy 因为图像 y 轴向下
               color='red', scale_units='xy', scale=1, width=0.002)
    plt.title("flow quiver (red arrows = displacement)")
    plt.savefig(save_path_prefix + "_flow_quiver.png", dpi=80, bbox_inches='tight')
    plt.close()

    # ---- Jacobian determinant ----
    # J = det([[1 + ∂u/∂x, ∂u/∂y], [∂v/∂x, 1 + ∂v/∂y]])
    du_dx = np.gradient(dx, axis=1)
    du_dy = np.gradient(dx, axis=0)
    dv_dx = np.gradient(dy, axis=1)
    dv_dy = np.gradient(dy, axis=0)
    J = (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx

    plt.figure(figsize=(6, 6))
    plt.imshow(J, cmap='RdBu_r', vmin=0, vmax=2)
    plt.colorbar(label='det(J)')
    plt.title(f"Jacobian: mean={J.mean():.3f}, "
              f"min={J.min():.3f}, max={J.max():.3f}, "
              f"fold_ratio={(J <= 0).mean()*100:.2f}%")
    plt.savefig(save_path_prefix + "_flow_jacobian.png", dpi=80, bbox_inches='tight')
    plt.close()

    # ---- fold mask（J <= 0 区域）----
    fold_mask = (J <= 0).astype(np.float32)
    plt.figure(figsize=(6, 6))
    plt.imshow(fold_mask, cmap='hot')
    plt.title(f"fold mask (red = J <= 0): "
              f"{fold_mask.sum():.0f} px ({fold_mask.mean()*100:.2f}%)")
    plt.savefig(save_path_prefix + "_flow_fold.png", dpi=80, bbox_inches='tight')
    plt.close()

    return {
        "dx_mean": float(dx.mean()), "dx_std": float(dx.std()),
        "dy_mean": float(dy.mean()), "dy_std": float(dy.std()),
        "mag_mean": float(mag.mean()), "mag_max": float(mag.max()),
        "jacobian_mean": float(J.mean()),
        "jacobian_min": float(J.min()),
        "fold_ratio": float((J <= 0).mean()),
    }


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

class _LoRAHelper(DiffusionTrainingModule):
    def forward(self, *a, **kw): pass


def load_pipeline(args, device):
    """加载 Qwen pipeline 和 FlowHead V4。"""
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

    # 加载 LoRA（DiT 特征质量提升）
    if args.lora_path and os.path.exists(args.lora_path):
        helper = _LoRAHelper()
        lora_modules = [
            "to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj",
            "to_out.0", "to_add_out", "img_mlp.net.2", "img_mod.1",
            "txt_mlp.net.2", "txt_mod.1",
        ]
        dit_with_lora = helper.add_lora_to_model(
            pipe.dit, target_modules=lora_modules, lora_rank=32)
        sd = load_state_dict(args.lora_path)
        sd = helper.mapping_lora_state_dict(sd)
        dit_with_lora.load_state_dict(sd, strict=False)
        pipe.dit = dit_with_lora
        print(f"LoRA 加载完成: {args.lora_path}")

    # 启用 DA-Flow 特征提取模式（在 __call__ 返回 corrected_low + q/k 特征）
    target_layers = parse_layer_list(args.dit_target_layers)
    num_layers = len(target_layers)

    # upstream pipeline 的 extract_daflow_features 只支持"最后 N 层"
    # 需要改为支持任意 target_layers，因此用 monkey-patch 覆盖
    pipe._v4_target_layers = set(target_layers)
    pipe._v4_max_layer = max(target_layers)
    pipe._v4_k_source = getattr(args, 'k_source', 'warped')   # warped(正式) / corrected(sham)
    pipe.extract_daflow_features = True   # 开启 __call__ 中的特征返回分支
    pipe.daflow_num_layers = num_layers   # 兼容 upstream 的参数

    # Monkey-patch pipeline __call__ 特征提取部分，支持任意层
    _patch_pipeline_for_v4(pipe)

    pipe.enable_vram_management()

    # 加载 FlowHead V4
    flow_model = FlowHeadV4_3(
        iters=args.flow_iters,
        num_dit_layers=num_layers,
        diff_out_ch=args.diff_out_ch,
    ).to(device).eval()

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        from safetensors.torch import load_file
        sd = load_file(args.ckpt_path, device=str(device))
        # 训练时 ModelLogger 用 --remove_prefix_in_ckpt "flow_head." 已经剥过一次顶层前缀。
        # 顶层 self.flow_head.context_encoder.* → ckpt key 是 context_encoder.*
        # 内层 self.flow_head.flow_head.* (FlowPredHead) → ckpt key 是 flow_head.*
        # 推理时直接加载，不要再 replace("flow_head.", "")，否则会把内层 flow_head 也剥掉。
        missing, unexpected = flow_model.load_state_dict(sd, strict=False)
        # 过滤掉 BatchNorm buffer（running_mean/var/num_batches_tracked），
        # 这些是训练时自然累积的统计量，未保存属于已知问题（不影响 eval）
        critical_missing = [k for k in missing
                            if not any(b in k for b in
                                       ['running_mean', 'running_var', 'num_batches_tracked'])]
        print(f"FlowHead V4 加载: {len(sd)} key，"
              f"缺失 {len(missing)}（其中 {len(critical_missing)} 个是参数，"
              f"其余为 BN buffer），多余 {len(unexpected)}")
        if critical_missing:
            print(f"  ! 关键参数缺失（前 5 个）：{critical_missing[:5]}")
        if unexpected:
            print(f"  ! 多余 ckpt key（前 5 个）：{unexpected[:5]}")
    else:
        print("警告：未提供 --ckpt_path，FlowHead V4 使用随机权重（效果无意义）")

    return pipe, flow_model, target_layers


def _patch_pipeline_for_v4(pipe):
    """
    Monkey-patch QwenImagePipeline.__call__ 里的特征提取部分，
    将"取最后 N 层"替换为"取 pipe._v4_target_layers 指定的任意层"。

    upstream 代码：
        target_indices = list(range(num_blocks - self.daflow_num_layers, num_blocks))

    patch 后：
        target_indices = sorted(self._v4_target_layers)
        并在 self._v4_max_layer 处提前退出
    """
    original_call = pipe.__class__.__call__

    def patched_call(self, *args, **kwargs):
        if not getattr(self, 'extract_daflow_features', False):
            return original_call(self, *args, **kwargs)

        # ============================================================
        # V4.2 Path B：双图拼接 + 跨段 Q/K（与 train_flow_head_v4_2_pathB.py 一致）
        # ============================================================
        # 训练时：main=corrected(GT,加噪), edit=warped(干净)；拼接后跨段切 Q/K。
        # 推理时没有 GT，main=corrected_low(Qwen 50步生成)，edit=warped。
        #   → 训练-推理不对称（corrected_low 有生成损耗），但提取流程完全对齐。
        # ============================================================
        # Stage 1：关掉特征提取，跑完整扩散得到 corrected_low
        self.extract_daflow_features = False
        image = original_call(self, *args, **kwargs)   # corrected_low (PIL)
        self.extract_daflow_features = True

        height = kwargs.get('height', 1024)
        width  = kwargs.get('width',  1024)
        edit_image = kwargs.get('edit_image', None)
        prompt     = kwargs.get('prompt', '')

        if edit_image is None:
            return image, [], []

        from einops import rearrange as _r

        if not self.scheduler.training:
            self.scheduler.set_timesteps(1000, training=True)

        # ---- Step 1：复用 pipeline units 准备 inputs ----
        # main(input_image) = corrected_low（Qwen 生成），edit_image = warped。
        # InputImageEmbedder → corrected_low 的 input_latents（main）
        # EditImageEmbedder  → warped 的 edit_latents
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": image,        # corrected_low → input_latents（main）
            "edit_image":  edit_image,   # warped → edit_latents
            "height": height, "width": width,
            "cfg_scale": 1, "rand_device": self.device,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
        }
        with torch.no_grad():
            for unit in self.units:
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                    unit, self, inputs_shared, inputs_posi, inputs_nega)

        input_latents   = inputs_shared["input_latents"]    # corrected_low latent
        edit_latents    = inputs_shared.get("edit_latents")
        noise           = inputs_shared.get("noise")
        if noise is None:
            noise = torch.randn_like(input_latents)
        prompt_emb      = inputs_posi["prompt_emb"]
        prompt_emb_mask = inputs_posi["prompt_emb_mask"]

        if edit_latents is None:
            # 无 edit_latents 则无法做跨图，退化为空特征（FlowHead 仅 CNN 分支）
            return image, [], []

        # ---- Step 2：main 加噪到中等 timestep（与训练 [400,600) 取中点 500）----
        timestep = self.scheduler.timesteps[500].to(
            dtype=self.torch_dtype, device=self.device).reshape(1)
        main_noisy = self.scheduler.add_noise(input_latents, noise, timestep)

        # ---- Step 3：双图拼接 + DiT 单步前向，跨段提取 Q/K ----
        self.load_models_to_device(['dit'])
        dit_raw = self.dit.module if hasattr(self.dit, 'module') else self.dit

        h_val, w_val = height, width
        img_shapes = [
            (main_noisy.shape[0],   main_noisy.shape[2] // 2,   main_noisy.shape[3] // 2),
            (edit_latents.shape[0], edit_latents.shape[2] // 2, edit_latents.shape[3] // 2),
        ]
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()

        # main(corrected_low) token
        image_tokens = _r(main_noisy, "B C (H P) (W Q) -> B (H W) (C P Q)",
                          H=h_val // 16, W=w_val // 16, P=2, Q=2)
        seq_main = image_tokens.shape[1]

        # edit(warped) token
        edit_tok = _r(edit_latents, "B C (H P) (W Q) -> B (H W) (C P Q)",
                      H=edit_latents.shape[2] // 2, W=edit_latents.shape[3] // 2, P=2, Q=2)

        assert edit_tok.shape[1] == seq_main, (
            f"corrected_low/warped token 数不一致: {seq_main} vs {edit_tok.shape[1]}，"
            f"请确认两图同分辨率。")

        # ★ 拼接 [corrected_low段 ; warped段]
        image_tokens = torch.cat([image_tokens, edit_tok], dim=1)
        image_tokens = dit_raw.img_in(image_tokens)
        text_tokens  = dit_raw.txt_in(dit_raw.txt_norm(prompt_emb))
        # ★ timestep / 1000（对齐 model_fn:706 与训练）
        conditioning     = dit_raw.time_text_embed(timestep / 1000, image_tokens.dtype)
        image_rotary_emb = dit_raw.pos_embed(img_shapes, txt_seq_lens,
                                              device=main_noisy.device)

        q_features, k_features = [], []
        target_layers = self._v4_target_layers
        max_layer = self._v4_max_layer
        k_source = getattr(self, '_v4_k_source', 'warped')
        with torch.no_grad():
            for block_idx, block in enumerate(dit_raw.transformer_blocks):
                if block_idx in target_layers:
                    img_normed = block.img_norm1(image_tokens)
                    img_mod_attn, _ = block.img_mod(conditioning).chunk(2, dim=-1)
                    img_modulated, _ = block._modulate(img_normed, img_mod_attn)
                    q_all = block.attn.to_q(img_modulated).float()
                    k_all = block.attn.to_k(img_modulated).float()
                    q_features.append(q_all[:, :seq_main])             # Q ← corrected_low 段
                    if k_source == "warped":
                        k_features.append(k_all[:, seq_main:])         # K ← warped 段（正式）
                    else:
                        k_features.append(k_all[:, :seq_main])         # K ← corrected 段（sham）
                text_tokens, image_tokens = block(
                    image=image_tokens, text=text_tokens, temb=conditioning,
                    image_rotary_emb=image_rotary_emb)
                if block_idx == max_layer:
                    break

        return image, q_features, k_features

    import types
    pipe.__call__ = types.MethodType(patched_call, pipe)


# ---------------------------------------------------------------------------
# 主推理循环
# ---------------------------------------------------------------------------

def process_images(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe, flow_model, target_layers = load_pipeline(args, device)

    prompt_map = None
    if args.prompt_dict:
        prompt_map = {}
        with open(args.prompt_dict, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompt_map[row["image"]] = row["prompt"]

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = [args.input_dir] if os.path.isfile(args.input_dir) else []
    if not image_files:
        for root, _, _ in os.walk(args.input_dir):
            for ext in exts:
                image_files.extend(glob.glob(os.path.join(root, ext)))
    image_files = sorted(image_files)
    print(f"找到 {len(image_files)} 张图像")

    for img_path in tqdm(image_files, desc="V4 推理"):
        rel_path = os.path.basename(img_path) if os.path.isfile(args.input_dir) \
                   else os.path.relpath(img_path, args.input_dir)
        out_base = os.path.splitext(os.path.join(args.output_dir, rel_path))[0]
        os.makedirs(os.path.dirname(out_base), exist_ok=True)

        if os.path.exists(out_base + "_c.jpg"):
            print(f"跳过（已存在）: {rel_path}")
            continue

        try:
            orig_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = orig_img.size

            img_input, infer_w, infer_h = preprocess_image(
                orig_img, args.img_size, args.resize_mode)

            cur_prompt = (prompt_map.get(os.path.basename(img_path), args.prompt)
                          if prompt_map else args.prompt)

            # ---- Stage 1：Qwen 完整扩散推理 + 提取 Q/K 特征 ----
            result = pipe(
                prompt=cur_prompt,
                edit_image=img_input,
                seed=random.randint(0, 2**32 - 1),
                num_inference_steps=args.infer_steps,
                width=infer_w,
                height=infer_h,
            )
            corrected_low, q_features, k_features = result
            print(f"  扩散推理完成，提取 {len(q_features)} 层 Q/K 特征，"
                  f"层索引: {sorted(target_layers)}")

            if not q_features:
                print(f"  警告：Q/K 特征为空，跳过 {rel_path}")
                continue

            # ---- Stage 2：FlowHead V4 预测光流 ----
            # GT reference sanity check（diagnose.md Step 4）：
            # 若指定 --use_gt_corrected_dir，则优先用 GT 替代 corrected_low
            corrected_pil = corrected_low
            corrected_source = "corrected_low (Qwen 扩散输出)"
            if args.use_gt_corrected_dir:
                gt_candidates = [
                    os.path.join(args.use_gt_corrected_dir, os.path.basename(img_path)),
                    os.path.join(args.use_gt_corrected_dir,
                                 os.path.splitext(os.path.basename(img_path))[0] + ".png"),
                    os.path.join(args.use_gt_corrected_dir,
                                 os.path.splitext(os.path.basename(img_path))[0] + ".jpg"),
                ]
                for gtp in gt_candidates:
                    if os.path.exists(gtp):
                        gt_pil = Image.open(gtp).convert("RGB").resize(
                            (infer_w, infer_h), Image.LANCZOS)
                        corrected_pil = gt_pil
                        corrected_source = f"GT rectified ({gtp})"
                        break
            print(f"  corrected 输入: {corrected_source}")

            corrected_t = pil_to_tensor(corrected_pil, device)
            warped_t    = pil_to_tensor(img_input,    device)

            with torch.no_grad():
                flow_low = flow_model(
                    corrected_t.float(), warped_t.float(),
                    q_features, k_features,
                    iters=args.flow_iters,
                )
            print(f"  光流估计完成，abs_mean: {flow_low.abs().mean():.2f}px, "
                  f"max: {flow_low.abs().max():.2f}px")

            # ---- 可选：保存 flow 诊断图（diagnose.md Step 3）----
            if args.save_flow_vis:
                stats = visualize_flow(flow_low, out_base)
                print(f"  flow 诊断图已保存，"
                      f"|flow|.mean={stats['mag_mean']:.2f}, "
                      f"J.min={stats['jacobian_min']:.3f}, "
                      f"fold_ratio={stats['fold_ratio']*100:.2f}%")

            # ---- Stage 3：上采样光流到原始分辨率 ----
            flow_hires = upscale_flow(flow_low, orig_h, orig_w)

            # ---- Stage 4：高清 warp ----
            result_hires = warp_image_with_flow(orig_img, flow_hires)
            print(f"  warp 完成，输出尺寸: {result_hires.size}")

            # ---- 保存 ----
            img_input.save(out_base + "_a.jpg")
            corrected_low.save(out_base + "_b.jpg")
            result_hires.save(out_base + "_c.jpg", quality=95)

            # 三合一对比图
            corrected_resized = corrected_low.resize((orig_w, orig_h), Image.LANCZOS)
            compare = Image.new("RGB", (orig_w * 3, orig_h))
            compare.paste(orig_img,          (0,          0))
            compare.paste(corrected_resized, (orig_w,     0))
            compare.paste(result_hires,      (orig_w * 2, 0))
            compare.save(out_base + "_d.jpg", quality=90)
            print(f"  已保存: {os.path.basename(out_base)}_[a-d].jpg")

        except Exception as e:
            import traceback
            print(f"错误 [{rel_path}]: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlowHead V4.2 Path B 推理：Qwen 扩散 → 双图跨段Q/K → FlowHead 光流 → 高清 warp")

    parser.add_argument("--input_dir",   type=str, required=True,
                        help="输入图像目录或单张图像路径")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="输出目录")
    # FlowHead V4.2 ckpt
    parser.add_argument("--ckpt_path",   type=str, required=True,
                        help="FlowHead V4.2 checkpoint 路径（safetensors）")
    parser.add_argument("--dit_target_layers", type=str, default="35",
                        help="DiT 目标层（0-based，逗号分隔），与训练时一致。"
                             "例：单层=\"35\"，三层=\"23,35,47\"")
    parser.add_argument("--k_source", type=str, default="warped",
                        choices=["warped", "corrected"],
                        help="K 特征来源，必须与训练时一致。"
                             "warped=跨图(正式 path B)；corrected=sham 对照")
    parser.add_argument("--diff_out_ch", type=int, default=96)
    parser.add_argument("--flow_iters",  type=int, default=12,
                        help="FlowHead 迭代次数。训练用 4，推理也用 4。")
    # LoRA
    parser.add_argument("--lora_path",   type=str,
                        default="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/"
                                "result/20250929-1_1in10_w_unwarp/step-668000.safetensors",
                        help="DiT LoRA checkpoint（提升特征质量）")
    # 推理参数
    parser.add_argument("--img_size",    type=int, default=512,
                        help="必须与训练分辨率一致（训练用 512）。"
                             "用 1024 会因 CorrBlock 搜索半径不匹配导致 flow 错乱。")
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--resize_mode", type=str, default="stretch",
                        choices=["stretch", "crop", "scale_to_short_side"])
    # 诊断
    parser.add_argument("--save_flow_vis", action="store_true",
                        help="保存 flow 的 6 张诊断图（dx/dy heatmap, magnitude, "
                             "quiver, Jacobian, fold mask）。diagnose.md Step 3")
    parser.add_argument("--use_gt_corrected_dir", type=str, default=None,
                        help="GT reference sanity check（diagnose.md Step 4）。"
                             "若提供该目录，每张输入图会优先去这里找同名 GT 矫正图，"
                             "用 GT 替代 Qwen corrected_low 作为 FlowHead 的 corrected 输入。"
                             "若 GT 模式正常但 corrected_low 模式崩坏，"
                             "说明问题是 corrected_low 的 domain gap。")
    # Prompt
    parser.add_argument("--prompt", type=str,
                        default="Flatten this warped or curled document image to a flat, "
                                "undistorted version. Preserve all text, lines, and content accurately.")
    parser.add_argument("--prompt_dict", type=str, default=None,
                        help="CSV 文件，覆盖每张图的 prompt（列：image,prompt）")
    args = parser.parse_args()
    process_images(args)


if __name__ == "__main__":
    main()
