#!/usr/bin/env python3
"""
FlowHead V2 独立训练脚本
========================
训练一个轻量级双图光流网络，替代推理时的 RAFT。

数据：复用现有 metadata_with_flow.csv
  - image:          corrected_gt（GT 矫正图）
  - edit_image:     warped（弯曲输入图）
  - flow_gt_path:   flow_gt.npy，shape (2, 1024, 1024)

训练不依赖 DiT，纯粹是图像对 → 光流的监督学习。

用法：
  accelerate launch --config_file scripts/Acceconfig_8A800.yaml train_flow_head_v2.py \\
    --dataset_metadata_path /juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white/metadata_with_flow.csv \\
    --output_path /juicefs-algorithm/data/IPT/zhuochu_yang/flow_head_v2_ckpts/ \\
    --train_size 512 \\
    --batch_size 4 \\
    --num_steps 50000 \\
    --learning_rate 2e-4
"""

import os
import sys
import csv
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

from utils.flow_head_v2 import FlowHeadV2, sequence_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FlowDatasetV2(Dataset):
    """
    读取 metadata_with_flow.csv，返回：
      corrected: PIL Image  GT 矫正图
      warped:    PIL Image  弯曲输入图
      flow_gt:   (2, train_size, train_size) float32 tensor
    """

    def __init__(self, csv_path: str, train_size: int = 512):
        self.train_size = train_size
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if os.path.exists(row.get("flow_gt_path", "")):
                    self.rows.append(row)
        print(f"FlowDatasetV2: {len(self.rows)} 条有效样本")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row  = self.rows[idx]
        sz   = self.train_size
        corrected = Image.open(row["image"]).convert("RGB").resize((sz, sz), Image.LANCZOS)
        warped    = Image.open(row["edit_image"]).convert("RGB").resize((sz, sz), Image.LANCZOS)

        # 加载 flow_gt 并缩放到 train_size
        flow_np = np.load(row["flow_gt_path"]).astype(np.float32)  # (2, 1024, 1024)
        flow_gt = torch.from_numpy(flow_np)                          # (2, 1024, 1024)
        # 缩放到 train_size，同时按比例缩放偏移量数值
        scale = sz / 1024.0
        flow_gt = F.interpolate(flow_gt.unsqueeze(0), size=(sz, sz),
                                mode="bilinear", align_corners=True).squeeze(0) * scale

        return {"corrected": corrected, "warped": warped, "flow_gt": flow_gt}


def collate_fn(batch):
    return {
        "corrected": [b["corrected"] for b in batch],
        "warped":    [b["warped"]    for b in batch],
        "flow_gt":   torch.stack([b["flow_gt"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def pil_list_to_tensor(imgs: list, device, dtype=torch.float32) -> torch.Tensor:
    """PIL 列表 → (B, 3, H, W)，值域 [-1, 1]"""
    arrs = [np.array(img, dtype=np.float32) * (2.0 / 255.0) - 1.0 for img in imgs]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_metadata_path", type=str, required=True)
    parser.add_argument("--output_path",    type=str, required=True)
    parser.add_argument("--flow_head_init", type=str, default=None,
                        help="FlowHead V2 初始权重（.pth），用于断点续训")
    parser.add_argument("--train_size",     type=int, default=512,
                        help="训练分辨率，推理时可用 1024")
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--num_steps",      type=int, default=50000)
    parser.add_argument("--learning_rate",  type=float, default=2e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-5)
    parser.add_argument("--gamma",          type=float, default=0.8,
                        help="序列 loss 的衰减系数")
    parser.add_argument("--lambda_warp",    type=float, default=0.0,
                        help="warp loss 权重，0 表示不用（训练初期建议为 0）")
    parser.add_argument("--iters_train",    type=int, default=4,
                        help="训练时迭代次数")
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--save_steps",     type=int, default=5000)
    parser.add_argument("--log_steps",      type=int, default=100)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs(args.output_path, exist_ok=True)

    # ---- 模型 ----
    model = FlowHeadV2(iters=args.iters_train)
    if args.flow_head_init and os.path.exists(args.flow_head_init):
        model.load_state_dict(torch.load(args.flow_head_init, map_location="cpu"))
        if accelerator.is_main_process:
            print(f"断点续训: {args.flow_head_init}")
    if accelerator.is_main_process:
        total = sum(p.numel() for p in model.parameters())
        print(f"FlowHead V2 参数量: {total:,}")

    # ---- 数据 ----
    dataset = FlowDatasetV2(args.dataset_metadata_path, train_size=args.train_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ---- 优化器：OneCycleLR，收敛更快 ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 估算 epoch 数（步数固定）
    steps_per_epoch = len(dataset) // (args.batch_size * accelerator.num_processes)
    num_epochs = max(1, args.num_steps // steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=args.num_steps,
        pct_start=0.05,
        anneal_strategy="cos",
    )

    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler)

    # ---- 训练循环 ----
    step = 0
    epoch = 0
    running_loss = 0.0

    while step < args.num_steps:
        epoch += 1
        for batch in loader:
            if step >= args.num_steps:
                break

            corrected_imgs = batch["corrected"]   # list[PIL]
            warped_imgs    = batch["warped"]       # list[PIL]
            flow_gt        = batch["flow_gt"].to(device)  # (B, 2, H, W)

            # PIL → tensor [-1, 1]
            corrected_t = pil_list_to_tensor(corrected_imgs, device)
            warped_t    = pil_list_to_tensor(warped_imgs,    device)

            # 前向（训练模式返回 predictions list）
            predictions = model(corrected_t, warped_t, iters=args.iters_train)

            # 序列 L1 loss
            loss = sequence_loss(predictions, flow_gt, gamma=args.gamma)

            # 可选 warp loss
            if args.lambda_warp > 0:
                flow_final = predictions[-1]
                B, _, H, W = flow_final.shape
                gy, gx = torch.meshgrid(
                    torch.linspace(-1, 1, H, device=device),
                    torch.linspace(-1, 1, W, device=device),
                    indexing="ij",
                )
                base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
                grid = base + torch.stack([
                    flow_final[:, 0] * 2.0 / (W - 1),
                    flow_final[:, 1] * 2.0 / (H - 1),
                ], dim=-1)
                warp_result = F.grid_sample(warped_t, grid, mode="bilinear",
                                            padding_mode="border", align_corners=True)
                loss = loss + args.lambda_warp * F.l1_loss(warp_result, corrected_t)

            optimizer.zero_grad()
            accelerator.backward(loss)
            # 梯度裁剪，防止训练初期爆炸
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            running_loss += loss.item()

            if accelerator.is_main_process and step % args.log_steps == 0:
                avg = running_loss / args.log_steps
                lr  = scheduler.get_last_lr()[0]
                print(f"Step {step}/{args.num_steps} | loss={avg:.4f} | lr={lr:.2e}")
                running_loss = 0.0

            if accelerator.is_main_process and step % args.save_steps == 0:
                ckpt = os.path.join(args.output_path, f"step-{step}.pth")
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt)
                print(f"  → checkpoint: {ckpt}")

    # 保存最终权重
    if accelerator.is_main_process:
        final = os.path.join(args.output_path, "flow_head_v2_final.pth")
        torch.save(accelerator.unwrap_model(model).state_dict(), final)
        print(f"\n训练完成 → {final}")


if __name__ == "__main__":
    main()
