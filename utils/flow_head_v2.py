#!/usr/bin/env python3
"""
FlowHead V2 - 轻量级双图光流网络
==================================
输入：corrected_low + warped 两张像素图
输出：光流位移场 (dx, dy)，单位像素

架构参考 RAFT，但大幅精简（~600k 参数 vs RAFT ~5M）：
  Feature Encoder   → 提取多尺度图像特征（corrected 和 warped 共用权重）
  Context Encoder   → 提取 corrected 的上下文，初始化 GRU hidden state
  Correlation Volume → 在 H/8 分辨率上计算 all-pairs 相似度
  ConvGRU           → 迭代精化光流（训练 4 次，推理 12 次）
  Flow Head         → 从 hidden state 预测 Δflow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础模块
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.InstanceNorm2d):
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
        return F.relu(self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))) +
                      self.downsample(x))


# ---------------------------------------------------------------------------
# Feature Encoder（corrected 和 warped 共用权重）
# ---------------------------------------------------------------------------

class FeatureEncoder(nn.Module):
    """
    输入: (B, 3, H, W)，值域 [-1, 1]
    输出: (B, 96, H/8, W/8)
    """
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)
        self.norm1  = nn.InstanceNorm2d(32)
        self.layer1 = ResBlock(32, 64, stride=2, norm=nn.InstanceNorm2d)
        self.layer2 = ResBlock(64, 96, stride=2, norm=nn.InstanceNorm2d)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# ---------------------------------------------------------------------------
# Context Encoder（只处理 corrected，初始化 hidden state）
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """
    输入: (B, 3, H, W)，值域 [-1, 1]
    输出: hidden (B, 96, H/8, W/8)，context (B, 32, H/8, W/8)
    """
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.norm1  = nn.BatchNorm2d(64)
        self.layer1 = ResBlock(64, 128, stride=2, norm=nn.BatchNorm2d)
        self.layer2 = ResBlock(128, 128, stride=2, norm=nn.BatchNorm2d)
        # 分裂为 hidden(96) + context(32)
        self.out_conv = nn.Conv2d(128, 96 + 32, 1)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out_conv(x)
        hidden, context = x[:, :96], x[:, 96:]
        return torch.tanh(hidden), torch.relu(context)


# ---------------------------------------------------------------------------
# Correlation Volume
# ---------------------------------------------------------------------------

class CorrBlock:
    """
    在 H/8 分辨率上计算 all-pairs dot-product 相似度。
    搜索半径 r=4，即 (2r+1)² = 81 个候选位置。
    """
    def __init__(self, fmap_c: torch.Tensor, fmap_w: torch.Tensor, radius: int = 4):
        """
        fmap_c: corrected 特征图 (B, C, H, W)
        fmap_w: warped    特征图 (B, C, H, W)
        """
        self.radius = radius
        self.fmap_w = fmap_w  # 在 warped 特征图上查找
        B, C, H, W = fmap_c.shape
        # 归一化特征，使 dot-product 等价于余弦相似度
        self.fmap_c = F.normalize(fmap_c, dim=1)
        self.fmap_w = F.normalize(fmap_w, dim=1)

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        """
        在当前 flow 处采样 correlation features。
        flow: (B, 2, H, W) 当前光流估计
        返回: (B, (2r+1)², H, W)
        """
        B, C, H, W = self.fmap_c.shape
        r = self.radius
        fmap_w = self.fmap_w

        # 构建基础 grid
        dy, dx = torch.meshgrid(
            torch.arange(-r, r + 1, device=flow.device),
            torch.arange(-r, r + 1, device=flow.device),
            indexing="ij",
        )
        delta = torch.stack([dx, dy], dim=-1).float()  # (2r+1, 2r+1, 2)

        # 以 flow 为中心，在 warped 特征图上采样
        centroid = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        # 归一化坐标
        centroid_norm = centroid / torch.tensor(
            [W - 1, H - 1], device=flow.device) * 2 - 1  # (B, H, W, 2)

        corr_list = []
        for i in range(2 * r + 1):
            for j in range(2 * r + 1):
                offset = delta[i, j]  # (2,) [dx, dy]
                offset_norm = offset / torch.tensor(
                    [W - 1, H - 1], device=flow.device) * 2
                sample_coords = centroid_norm + offset_norm  # (B, H, W, 2)
                # grid_sample 在 warped 特征图上采样 (B, C, H, W)
                fmap_sample = F.grid_sample(
                    fmap_w, sample_coords,
                    mode="bilinear", padding_mode="border", align_corners=True,
                )  # (B, C, H, W)
                # dot product with corrected features
                corr = (self.fmap_c * fmap_sample).sum(dim=1, keepdim=True)  # (B, 1, H, W)
                corr_list.append(corr)

        return torch.cat(corr_list, dim=1)  # (B, (2r+1)², H, W)


# ---------------------------------------------------------------------------
# ConvGRU 更新模块
# ---------------------------------------------------------------------------

class ConvGRU(nn.Module):
    """
    输入:
      h:    hidden state (B, hidden_dim, H, W)
      x:    corr_feat(81) + flow(2) + context(32) = 115 通道
    输出:
      h_new: 更新后的 hidden state (B, hidden_dim, H, W)
    """
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


# ---------------------------------------------------------------------------
# Flow Head（从 hidden state 预测 Δflow）
# ---------------------------------------------------------------------------

class FlowHead(nn.Module):
    """
    输入: hidden (B, hidden_dim, H, W)
    输出: Δflow  (B, 2, H, W)，像素偏移增量
    """
    def __init__(self, hidden_dim: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.conv1(x)))


# ---------------------------------------------------------------------------
# FlowHead V2 主网络
# ---------------------------------------------------------------------------

class FlowHeadV2(nn.Module):
    """
    轻量级双图光流网络（~600k 参数）。

    Args:
        iters:   推理/训练时的迭代次数（训练 4，推理 12）
        radius:  Correlation 搜索半径（默认 4，即 81 候选位置）

    Forward:
        corrected: (B, 3, H, W)  矫正图，值域 [-1, 1]
        warped:    (B, 3, H, W)  弯曲图，值域 [-1, 1]
        iters:     迭代次数（可在推理时覆盖）

    Returns:
        训练模式：list of (B, 2, H, W)，共 iters 个，用于序列 loss
        推理模式：(B, 2, H, W)，最终光流，单位像素
    """

    def __init__(self, iters: int = 4, radius: int = 4):
        super().__init__()
        self.iters  = iters
        self.radius = radius

        self.feature_encoder  = FeatureEncoder()
        self.context_encoder  = ContextEncoder()
        self.update_block     = ConvGRU(hidden_dim=96, input_dim=(2 * radius + 1) ** 2 + 2 + 32)
        self.flow_head        = FlowHead(hidden_dim=96)

    def upsample_flow(self, flow: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """将光流从 H/8 上采样到目标分辨率，并缩放偏移量数值。"""
        _, _, h, w = flow.shape
        flow_up = F.interpolate(flow, size=(target_h, target_w),
                                mode="bilinear", align_corners=True)
        flow_up[:, 0] *= float(target_w) / w
        flow_up[:, 1] *= float(target_h) / h
        return flow_up

    def forward(self, corrected: torch.Tensor, warped: torch.Tensor,
                iters: int = None):
        iters = iters or self.iters
        B, _, H, W = corrected.shape

        # --- 特征提取 ---
        fmap_c = self.feature_encoder(corrected)   # (B, 96, H/8, W/8)
        fmap_w = self.feature_encoder(warped)       # (B, 96, H/8, W/8)
        hidden, context = self.context_encoder(corrected)  # (B,96,H/8,W/8), (B,32,H/8,W/8)

        # --- 构建相关体积 ---
        corr_block = CorrBlock(fmap_c, fmap_w, radius=self.radius)

        # --- 初始化光流 ---
        _, _, fh, fw = fmap_c.shape
        flow = torch.zeros(B, 2, fh, fw, device=corrected.device)

        # --- 迭代更新 ---
        predictions = []
        for _ in range(iters):
            flow = flow.detach()  # 截断梯度，每次迭代独立
            corr_feat = corr_block(flow)                         # (B, 81, fh, fw)
            x = torch.cat([corr_feat, flow, context], dim=1)    # (B, 115, fh, fw)
            hidden = self.update_block(hidden, x)                # (B, 96, fh, fw)
            delta_flow = self.flow_head(hidden)                  # (B, 2, fh, fw)
            flow = flow + delta_flow
            # 上采样到输入分辨率，加入 predictions
            predictions.append(self.upsample_flow(flow, H, W))

        if self.training:
            return predictions   # list of (B, 2, H, W)，用于序列 loss
        else:
            return predictions[-1]  # (B, 2, H, W)，最终光流


# ---------------------------------------------------------------------------
# 序列 Loss（训练用）
# ---------------------------------------------------------------------------

def sequence_loss(predictions: list, flow_gt: torch.Tensor,
                  gamma: float = 0.8) -> torch.Tensor:
    """
    RAFT 风格的序列 L1 loss。
    越晚的迭代权重越大（gamma^(N-i)），鼓励逐步精化。

    predictions: list of (B, 2, H, W)，共 N 个
    flow_gt:     (B, 2, H, W)  GT 光流
    gamma:       衰减系数（默认 0.8）
    """
    N = len(predictions)
    loss = 0.0
    for i, pred in enumerate(predictions):
        weight = gamma ** (N - 1 - i)
        loss += weight * F.l1_loss(pred, flow_gt)
    return loss


# ---------------------------------------------------------------------------
# 参数量统计（调试用）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = FlowHeadV2(iters=4)
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total:,}")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n:,}")

    # 前向验证
    model.train()
    c = torch.randn(2, 3, 512, 512)
    w = torch.randn(2, 3, 512, 512)
    preds = model(c, w, iters=4)
    print(f"\n训练模式输出: {len(preds)} 个预测，shape={preds[0].shape}")

    model.eval()
    with torch.no_grad():
        flow = model(c, w, iters=12)
    print(f"推理模式输出: shape={flow.shape}")
