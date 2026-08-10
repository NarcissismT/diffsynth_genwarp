#!/usr/bin/env python3
"""
FlowHead V4.1 — V4 + gradient loss + BN→InstanceNorm
=====================================================
相对 V4 (utils/flow_head_v4_layer_probe.py) 的两点改进：

1. **ContextEncoder 的 BatchNorm2d → InstanceNorm2d**
   原因：V4 用 BatchNorm + DiffusionTrainingModule 的 export_trainable_state_dict
   只保 requires_grad=True 的参数，BN 的 running_mean/var/num_batches_tracked
   是 buffer 不会被保存 → 推理时 BN 用初始统计量 → 训练-推理不一致。
   InstanceNorm 不依赖 batch 统计，无 buffer 问题。

2. **新增 gradient_loss 和 sequence_loss_with_grad**
   参考 dewarp_dino/train_warp_DDP.py 的 gradient_loss（已验证可解决 flow 高频振荡）。
   每次 RAFT 迭代的 flow 都加 L2 梯度惩罚，强制 flow 空间平滑，
   消除水波纹（diagnose.md Step 3 + 5.1 Item 3）。

模型结构与 V4 完全一致，参数量也相同，可以直接续训 V4 ckpt。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础模块
# ---------------------------------------------------------------------------

def _in_with_affine(num_channels: int) -> nn.Module:
    """
    InstanceNorm2d with affine=True，保留通道级可学习 scale/shift。
    注意：默认 affine=False 时 IN 没有可训练参数，相当于"硬 normalize"，
    会损失模型容量。这里强制 affine=True 让它和 BatchNorm2d 行为对齐。
    """
    return nn.InstanceNorm2d(num_channels, affine=True)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=_in_with_affine):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = norm(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            norm(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        return F.relu(
            self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))) +
            self.downsample(x)
        )


class ContextEncoder(nn.Module):
    """
    V4.1 改动：BatchNorm2d → InstanceNorm2d (affine=True)。
    - 避免 BN running 统计量在 ckpt 里缺失导致的训练-推理不一致
    - affine=True 保留可学习 weight/bias（与 BN 相同的容量）
    """
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.norm1  = _in_with_affine(64)
        self.layer1 = ResBlock(64, 128, stride=2, norm=_in_with_affine)
        self.layer2 = ResBlock(128, 128, stride=2, norm=_in_with_affine)
        self.out_conv = nn.Conv2d(128, 96 + 32, 1)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out_conv(x)
        hidden, context = x[:, :96], x[:, 96:]
        return torch.tanh(hidden), torch.relu(context)


class CorrBlock:
    """
    RAFT 风格 all-pairs cosine 相关查表。

    【V4.4 致命 bug 修复 (2026-06-10)】原实现把"位移量 flow"直接归一化当成
    "绝对采样坐标"，丢失了每个像素自身的 identity 网格坐标 coords0：

        # 错误：sample_coords = flow/(W-1)*2-1 + offset    → flow=0 时所有像素都采左上角
        # 正确：sample_coords = (coords0 + flow + offset) 归一化

    标准 RAFT 的查表中心是 coords0(像素自身坐标) + flow(位移)。漏掉 coords0 后，
    correlation volume 与几何位置完全脱钩 → 整条 warped→flow 相关通路失效。
    数值验证：修复前逐像素命中正确位置 0.0%、center-tap 相关 0.015(≈随机)；
    修复后命中 93.8%、center-tap 1.000。该 bug 从 V4 延续到 V4.4 全部版本。
    """

    def __init__(self, fmap_c: torch.Tensor, fmap_w: torch.Tensor, radius: int = 4):
        self.radius = radius
        B, C, H, W = fmap_c.shape
        self.fmap_c = F.normalize(fmap_c, dim=1)
        self.fmap_w = F.normalize(fmap_w, dim=1)

        # identity base grid（特征分辨率的绝对像素坐标，单位与 flow 一致 = H/8 像素）
        # coords0[...,0]=列(x, 对应 flow[:,0]=dx)，coords0[...,1]=行(y, 对应 flow[:,1]=dy)
        gy, gx = torch.meshgrid(
            torch.arange(H, device=fmap_c.device).float(),
            torch.arange(W, device=fmap_c.device).float(),
            indexing="ij",
        )
        self.coords0 = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = self.fmap_c.shape
        r = self.radius
        dy, dx = torch.meshgrid(
            torch.arange(-r, r + 1, device=flow.device),
            torch.arange(-r, r + 1, device=flow.device),
            indexing="ij",
        )
        delta = torch.stack([dx, dy], dim=-1).float()        # (2r+1, 2r+1, 2)
        scale = torch.tensor([W - 1, H - 1], device=flow.device).float()

        # 查表中心 = 像素自身坐标 + 位移（这一项是修复的核心，原代码缺 coords0）
        coords = self.coords0.to(flow.device) + flow.permute(0, 2, 3, 1)  # (B, H, W, 2) 绝对像素坐标

        corr_list = []
        for i in range(2 * r + 1):
            for j in range(2 * r + 1):
                sample_pix = coords + delta[i, j]            # 加 ±r 搜索窗（像素单位）
                sample_norm = sample_pix / scale * 2 - 1     # 统一归一化到 [-1, 1]
                fmap_sample = F.grid_sample(
                    self.fmap_w, sample_norm,
                    mode="bilinear", padding_mode="border", align_corners=True,
                )
                corr = (self.fmap_c * fmap_sample).sum(dim=1, keepdim=True)
                corr_list.append(corr)
        return torch.cat(corr_list, dim=1)


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 96, input_dim: int = 115):
        super().__init__()
        self.conv_z = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.conv_r = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.conv_q = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.conv_z(hx))
        r = torch.sigmoid(self.conv_r(hx))
        q = torch.tanh(self.conv_q(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class FlowPredHead(nn.Module):
    def __init__(self, hidden_dim: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.conv1(x)))


# ---------------------------------------------------------------------------
# DPT 上采样头
# ---------------------------------------------------------------------------

class DPTHead(nn.Module):
    def __init__(self, in_channels: int = 3072, out_channels: int = 96, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.projs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for _ in range(num_layers)
        ])
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, features: list, spatial_shape: tuple) -> torch.Tensor:
        H, W = spatial_shape
        h, w = H // 16, W // 16
        weights = F.softmax(self.layer_weights, dim=0)
        fused = None
        for i, (feat, proj) in enumerate(zip(features, self.projs)):
            B, T, C = feat.shape
            feat_2d = feat.permute(0, 2, 1).reshape(B, C, h, w).float()
            feat_proj = proj(feat_2d)
            fused = feat_proj * weights[i] if fused is None else fused + feat_proj * weights[i]
        return F.interpolate(fused, size=(H // 8, W // 8), mode="bilinear", align_corners=True)


class DPTHeads(nn.Module):
    def __init__(self, in_channels: int = 3072, out_channels: int = 96, num_layers: int = 4):
        super().__init__()
        self.dpt_q   = DPTHead(in_channels, out_channels, num_layers)
        self.dpt_k   = DPTHead(in_channels, out_channels, num_layers)
        self.dpt_ctx = DPTHead(in_channels, out_channels // 3, num_layers)

    def forward(self, q_features: list, k_features: list, spatial_shape: tuple):
        F_Q   = self.dpt_q(q_features, spatial_shape)
        F_K   = self.dpt_k(k_features, spatial_shape)
        F_ctx = self.dpt_ctx(q_features, spatial_shape)
        return F_Q, F_K, F_ctx


# ---------------------------------------------------------------------------
# 混合特征编码器
# ---------------------------------------------------------------------------

class HybridEncoder(nn.Module):
    def __init__(self, cnn_out_ch: int = 96, diff_out_ch: int = 96):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)
        self.norm1  = _in_with_affine(32)
        self.layer1 = ResBlock(32, 64, stride=2, norm=_in_with_affine)
        self.layer2 = ResBlock(64, cnn_out_ch, stride=2, norm=_in_with_affine)

    def forward(self, img: torch.Tensor, diff_feat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(img)))
        x = self.layer1(x)
        x = self.layer2(x)
        return torch.cat([x, diff_feat], dim=1)


# ---------------------------------------------------------------------------
# FlowHead V4.1 主网络（结构与 V4 完全相同）
# ---------------------------------------------------------------------------

class FlowHeadV4_1(nn.Module):
    """
    V4.1 主网络。结构与 V4 完全相同，仅 ContextEncoder 内 BN→IN。
    可直接加载 V4 ckpt 续训（context_encoder 的 BN 参数会迁移到 IN，
    虽然 IN 不需要 running 统计量，但 weight/bias 是兼容的）。
    """
    def __init__(self, iters: int = 4, radius: int = 4,
                 dit_channels: int = 3072, diff_out_ch: int = 96,
                 cnn_out_ch: int = 96, num_dit_layers: int = 4):
        super().__init__()
        self.iters  = iters
        self.radius = radius

        self.dpt_heads = DPTHeads(
            in_channels=dit_channels, out_channels=diff_out_ch, num_layers=num_dit_layers)
        self.feat_encoder = HybridEncoder(cnn_out_ch=cnn_out_ch, diff_out_ch=diff_out_ch)
        self.context_encoder = ContextEncoder()

        corr_ch = (2 * radius + 1) ** 2
        self.update_block = ConvGRU(hidden_dim=96, input_dim=corr_ch + 2 + 32)
        self.flow_head = FlowPredHead(hidden_dim=96)

    def upsample_flow(self, flow: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        _, _, h, w = flow.shape
        flow_up = F.interpolate(flow, size=(target_h, target_w),
                                mode="bilinear", align_corners=True)
        flow_up[:, 0] *= float(target_w) / w
        flow_up[:, 1] *= float(target_h) / h
        return flow_up

    def forward(self, corrected: torch.Tensor, warped: torch.Tensor,
                q_features: list, k_features: list, iters: int = None):
        iters = iters or self.iters
        B, _, H, W = corrected.shape

        # 【V4.4 修复 (2026-06-10)】接回 F_ctx：原 forward 把 dpt_ctx 的输出丢弃
        # (F_Q, F_K, _ = ...)，导致 DiT 上下文特征成为死参数(梯度 None)，且 GRU 的
        # context 只含纯 CNN 外观。F_ctx 与 CNN context 同为 32ch×H/8，相加融合，
        # 给 DiT 几何/语义多一条直达 GRU 的通路（GRU input_dim 不变，ckpt 兼容）。
        F_Q, F_K, F_ctx = self.dpt_heads(q_features, k_features, (H, W))
        fmap_c = self.feat_encoder(corrected, F_Q)
        fmap_w = self.feat_encoder(warped,    F_K)
        hidden, context_cnn = self.context_encoder(corrected)
        context = context_cnn + F_ctx
        corr_block = CorrBlock(fmap_c, fmap_w, radius=self.radius)

        _, _, fh, fw = fmap_c.shape
        flow = torch.zeros(B, 2, fh, fw, device=corrected.device)

        predictions = []
        for _ in range(iters):
            flow = flow.detach()
            corr_feat = corr_block(flow)
            x = torch.cat([corr_feat, flow, context], dim=1)
            hidden = self.update_block(hidden, x)
            delta_flow = self.flow_head(hidden)
            flow = flow + delta_flow
            predictions.append(self.upsample_flow(flow, H, W))

        if self.training:
            return predictions
        else:
            return predictions[-1]


# ---------------------------------------------------------------------------
# Loss 函数（V4.1 新增）
# ---------------------------------------------------------------------------

def gradient_loss(flow: torch.Tensor) -> torch.Tensor:
    """
    L2 梯度损失：惩罚 flow 在空间上的高频振荡，强制平滑。

    参考 dewarp_dino/train_warp_DDP.py:
        dy = abs(s[:, :, 1:, :] - s[:, :, :-1, :]) ** 2
        dx = abs(s[:, :, :, 1:] - s[:, :, :, :-1]) ** 2
        return (mean(dx) + mean(dy)) / 2.0

    flow: (B, 2, H, W)
    """
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]) ** 2
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]) ** 2
    return (dx.mean() + dy.mean()) / 2.0


def sequence_loss(predictions: list, flow_gt: torch.Tensor,
                  gamma: float = 0.8) -> torch.Tensor:
    """RAFT 序列 L1 loss（与 V4 相同，保留供兼容）"""
    N = len(predictions)
    loss = 0.0
    for i, pred in enumerate(predictions):
        weight = gamma ** (N - 1 - i)
        loss += weight * F.l1_loss(pred, flow_gt)
    return loss


def sequence_loss_with_grad(predictions: list, flow_gt: torch.Tensor,
                             gamma: float = 0.8,
                             gradloss_ratio: float = 1.0) -> tuple:
    """
    V4.1 核心：序列 L1 + 每次迭代的 L2 gradient loss

    返回:
        L_flow:  序列 L1 loss
        L_grad:  序列 L2 梯度 loss（已乘 gradloss_ratio）

    用法:
        L_flow, L_grad = sequence_loss_with_grad(preds, gt, 0.8, 1.0)
        total = L_flow + L_grad + lambda_warp * L_warp
    """
    N = len(predictions)
    L_flow = 0.0
    L_grad = 0.0
    for i, pred in enumerate(predictions):
        weight = gamma ** (N - 1 - i)
        L_flow += weight * F.l1_loss(pred, flow_gt)
        L_grad += weight * gradloss_ratio * gradient_loss(pred)
    return L_flow, L_grad


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    print("=== FlowHead V4.1 自检 ===")
    for L in [1, 3, 4]:
        m = FlowHeadV4_1(num_dit_layers=L)
        n = sum(p.numel() for p in m.parameters())
        print(f"  num_dit_layers={L}: {n:,} 参数")

    # 检查没有 BN buffer
    m = FlowHeadV4_1(num_dit_layers=1)
    bn_buffers = [k for k in m.state_dict().keys()
                  if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
    assert len(bn_buffers) == 0, f"还有 BN buffer: {bn_buffers}"
    print("✓ 无 BN buffer，IN 替换成功")

    # 前向 + grad loss 验证
    m.train()
    B, H, W = 1, 512, 512
    T = (H // 16) * (W // 16)
    c = torch.randn(B, 3, H, W); w_ = torch.randn(B, 3, H, W)
    q = [torch.randn(B, T, 3072)]; k = [torch.randn(B, T, 3072)]
    preds = m(c, w_, q, k, iters=4)
    flow_gt = torch.randn(B, 2, H, W)
    L_flow, L_grad = sequence_loss_with_grad(preds, flow_gt, 0.8, 1.0)
    total = L_flow + L_grad
    total.backward()
    print(f"✓ Loss: L_flow={L_flow.item():.4f}, L_grad={L_grad.item():.6f}, total={total.item():.4f}")
    print(f"✓ 反向传播 OK")
