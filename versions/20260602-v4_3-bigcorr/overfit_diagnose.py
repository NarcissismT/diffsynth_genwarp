#!/usr/bin/env python3
"""
V4.2 Path B 分层过拟合诊断
============================
目标：判断 V4.2 的 <5px 卡在 40-50% 到底是【容量瓶颈】还是【数据/训练问题】。

方法：取固定 18 个分层样本（低/中/高位移各 6 个，见 overfit_samples.csv），
关数据增强，在这个小集上反复训练到收敛，看每层能否过拟合到接近 100% @<5px。

判据（决定后续方向）：
  - 全部层都能过拟合到 ~100% <5px → 容量够，瓶颈在数据/训练动力学 → 走工程档
  - 卡在 40-60% 上不去          → 容量/特征是真瓶颈 → 走架构档
  - 低/中层 OK 但高位移层卡住    → 高位移/亚像素精度受限（CorrBlock radius/单尺度）

单卡运行（过拟合诊断不需要 DDP）。复用 V4.2 训练模块的完整前向（双图跨段 Q/K + loss），
保证与正式训练逻辑完全一致。

用法见 scripts/overfit_diagnose.sh
"""

import os
import sys
import csv
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = "/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp"
_UPSTREAM = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
sys.path.insert(0, _HERE)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _UPSTREAM)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 复用 V4.2 训练模块（双图跨段 Q/K + 完整 loss，与正式训练完全一致）
from train_flow_head_v4_3 import (
    QwenImageFlowV4_3TrainingModule,
    parse_layer_list,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# 固定样本加载（不走 ImageDataset，避免 repeat/shuffle；图像加载逻辑对齐 crop_and_resize）
# ---------------------------------------------------------------------------

def crop_and_resize(image: Image.Image, target_h: int, target_w: int) -> Image.Image:
    """与 ImageDataset.crop_and_resize 一致（确定性，无随机增强）。"""
    import torchvision
    w, h = image.size
    scale = max(target_w / w, target_h / h)
    image = torchvision.transforms.functional.resize(
        image, (round(h * scale), round(w * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    image = torchvision.transforms.functional.center_crop(image, (target_h, target_w))
    return image


def get_hw(image: Image.Image, max_pixels: int, factor: int = 16):
    w, h = image.size
    if w * h > max_pixels:
        scale = (w * h / max_pixels) ** 0.5
        h, w = int(h / scale), int(w / scale)
    h = h // factor * factor
    w = w // factor * factor
    return h, w


def load_fixed_samples(csv_path: str, base_path: str, max_pixels: int):
    """加载固定分层样本，返回 list of data dict（image/edit_image=PIL, flow_gt=tensor）。"""
    rows = list(csv.DictReader(open(csv_path)))
    samples = []
    for r in rows:
        # image = corrected(GT 矫正图), edit_image = warped
        img_path  = os.path.join(base_path, r["image"])
        edit_path = os.path.join(base_path, r["edit_image"])
        flow_path = r["flow_gt_path"]
        if not (os.path.exists(img_path) and os.path.exists(edit_path) and os.path.exists(flow_path)):
            print(f"跳过缺失样本: {r['image']}")
            continue
        corrected = Image.open(img_path).convert("RGB")
        h, w = get_hw(corrected, max_pixels)
        corrected = crop_and_resize(corrected, h, w)
        warped    = crop_and_resize(Image.open(edit_path).convert("RGB"), h, w)
        flow_gt   = torch.from_numpy(np.load(flow_path).astype(np.float32))  # (2,H,W)
        samples.append({
            "image": corrected,
            "edit_image": warped,
            "prompt": r.get("prompt", "Apply geometric correction to the input image."),
            "flow_gt": flow_gt,
            "_layer": r.get("_layer", "?"),
            "_zero_epe": float(r.get("_zero_epe", 0.0)),
        })
    return samples


# ---------------------------------------------------------------------------
# 评估：在固定样本上算分层 EPE / <5px（不反向）
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, samples, device, iters):
    model.eval()
    by_layer = {}
    for s in samples:
        inputs = model.forward_preprocess(s)
        q_feats, k_feats = model._extract_qk_features(inputs)
        corrected_t = model.pipe.preprocess_image(s["image"]).to(device=device, dtype=torch.float32)
        warped_t    = model.pipe.preprocess_image(s["edit_image"]).to(device=device, dtype=torch.float32)
        preds = model.flow_head(corrected_t, warped_t, q_feats, k_feats, iters=iters)
        pred = preds[-1] if isinstance(preds, list) else preds
        fh, fw = pred.shape[-2], pred.shape[-1]
        flow_gt = s["flow_gt"].unsqueeze(0).to(device)
        scale = float(fh) / flow_gt.shape[-2]
        flow_gt_r = F.interpolate(flow_gt, size=(fh, fw), mode="bilinear", align_corners=True) * scale
        epe_map = torch.norm(pred - flow_gt_r, dim=1)
        layer = s["_layer"]
        by_layer.setdefault(layer, []).append({
            "epe": epe_map.mean().item(),
            "p5": (epe_map < 5).float().mean().item(),
            "p3": (epe_map < 3).float().mean().item(),
            "zero": torch.norm(flow_gt_r, dim=1).mean().item(),
        })
    model.train()
    return by_layer


def fmt_eval(by_layer):
    lines = []
    for layer in ["low", "mid", "high"]:
        if layer not in by_layer:
            continue
        rs = by_layer[layer]
        epe = np.mean([r["epe"] for r in rs])
        p5  = np.mean([r["p5"] for r in rs]) * 100
        p3  = np.mean([r["p3"] for r in rs]) * 100
        z   = np.mean([r["zero"] for r in rs])
        lines.append(f"{layer}: EPE={epe:6.2f}(zero={z:6.2f}) <3px={p3:5.1f}% <5px={p5:5.1f}%")
    return "  |  ".join(lines)


# ---------------------------------------------------------------------------
# 主诊断循环
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_csv", type=str,
                        default=os.path.join(_HERE, "overfit_samples.csv"))
    parser.add_argument("--dataset_base_path", type=str,
                        default="/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511")
    parser.add_argument("--model_paths", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, required=True)
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--dit_target_layers", type=str, default="35")
    parser.add_argument("--k_source", type=str, default="warped", choices=["warped", "corrected"])
    parser.add_argument("--diff_out_ch", type=int, default=96)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_warp", type=float, default=0.5)
    parser.add_argument("--gradloss_ratio", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="过拟合诊断用稍大 lr（小集，要快速收敛）")
    parser.add_argument("--num_steps", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--train_iters", type=int, default=12)
    parser.add_argument("--eval_iters", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_layers = parse_layer_list(args.dit_target_layers)
    print(f"=== V4.2 过拟合诊断 ===")
    print(f"目标层: {target_layers}, k_source: {args.k_source}, lr: {args.learning_rate}")
    print(f"训练 iters: {args.train_iters}, 评估 iters: {args.eval_iters}")

    # 加载固定样本
    samples = load_fixed_samples(args.samples_csv, args.dataset_base_path, args.max_pixels)
    print(f"加载 {len(samples)} 个固定分层样本")
    from collections import Counter
    print(f"分层分布: {dict(Counter(s['_layer'] for s in samples))}")

    # 构造 V4.2 训练模块（与正式训练完全一致）
    model = QwenImageFlowV4_3TrainingModule(
        model_paths=args.model_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=True,
        extra_inputs="edit_image",
        lambda_flow=args.lambda_flow,
        lambda_warp=args.lambda_warp,
        gradloss_ratio=args.gradloss_ratio,
        dit_target_layers=target_layers,
        diff_out_ch=args.diff_out_ch,
        freeze_dit=True,
        loss_print_interval=10 ** 9,   # 关掉它自己的打印，用我们的评估
        k_source=args.k_source,
    )
    # 把整个 module（含 pipe 的 DiT/VAE/TextEncoder + flow_head）搬到 GPU。
    # V4.2 module 加载时 pipe device="cpu"，正式训练靠 accelerate.prepare 搬运；
    # 诊断单卡用 model.to(device) 递归搬运，并显式修正 pipe.device，
    # 否则 forward_preprocess 里 rand_device=self.pipe.device 会用错设备。
    model.to(device)
    model.pipe.device = device
    if hasattr(model.pipe, 'scheduler'):
        # scheduler.timesteps 等张量也需在正确设备（add_noise 用到）
        pass
    model.flow_head.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.flow_head.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=0.0)   # 过拟合诊断：关 weight decay

    print("\n=== 开始过拟合（在 18 个固定样本上反复训练）===")
    print("判据：若各层 <5px 都能逼近 100% → 容量够，瓶颈在数据/训练；"
          "若卡在 40-60% → 容量是真瓶颈\n")

    # 初始评估
    init = evaluate(model, samples, device, args.eval_iters)
    print(f"[step      0] {fmt_eval(init)}")

    step = 0
    while step < args.num_steps:
        for s in samples:
            optimizer.zero_grad()
            loss = model(s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.flow_head.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.eval_every == 0:
                ev = evaluate(model, samples, device, args.eval_iters)
                print(f"[step {step:>6d}] loss={loss.item():.4f}  {fmt_eval(ev)}", flush=True)
            if step >= args.num_steps:
                break

    print("\n=== 诊断结论参考 ===")
    final = evaluate(model, samples, device, args.eval_iters)
    print(f"最终: {fmt_eval(final)}")
    overall_p5 = np.mean([r["p5"] for rs in final.values() for r in rs]) * 100
    print(f"\n整体 <5px = {overall_p5:.1f}%")
    if overall_p5 > 90:
        print("→ 能过拟合到 ~100%：容量够，瓶颈在数据/训练动力学 → 走【工程档】")
    elif overall_p5 < 65:
        print("→ 卡在 <65%：容量/特征是真瓶颈 → 走【架构档】")
    else:
        print("→ 中间地带：部分层受限，检查是否高位移层拖累（看 high 层）")


if __name__ == "__main__":
    main()
