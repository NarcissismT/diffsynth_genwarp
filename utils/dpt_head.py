#!/usr/bin/env python3
"""
DPT 上采样头（参考 DA-Flow Section 4.4.1）
==========================================
将 DiT 中间层 Q/K 特征从 latent 分辨率（H/16 × W/16）
上采样到 RAFT feature encoder 分辨率（H/8 × W/8）。

DA-Flow 使用三个独立的 DPT head：
  DPT_Q   → query 特征，用于构建 correlation volume（corrected 侧）
  DPT_K   → key   特征，用于构建 correlation volume（warped   侧）
  DPT_ctx → context 特征，用于 RAFT 的 update operator

每个 DPT head 聚合 L=4 层 DiT 特征，通过可学习的加权融合 + 2× 上采样实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPTHead(nn.Module):
    """
    把 L 层 DiT Q/K 特征聚合并上采样到 H/8 × W/8。

    输入：list of (B, T, C_in) — T = (H/16)*(W/16) 个 token，C_in = 3072
    输出：(B, C_out, H/8, W/8)

    处理流程：
      1. 每层特征 (B, T, C_in) reshape → (B, C_in, h, w)，h = H/16，w = W/16
      2. 1×1 conv 降维至 C_out
      3. 可学习权重加权融合 L 层
      4. 2× 双线性上采样到 H/8 × W/8
    """

    def __init__(self, in_channels: int = 3072, out_channels: int = 96,
                 num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        # 每层独立的 1×1 投影
        self.projs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for _ in range(num_layers)
        ])
        # 可学习层融合权重（softmax 归一化）
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, features: list, spatial_shape: tuple) -> torch.Tensor:
        """
        features:      list of (B, T, C_in)，长度 = num_layers
        spatial_shape: (H, W) — 原始输入图像分辨率，用于计算 latent 空间大小

        Returns: (B, C_out, H/8, W/8)
        """
        H, W = spatial_shape
        h, w = H // 16, W // 16  # latent 分辨率

        weights = F.softmax(self.layer_weights, dim=0)  # (L,)

        fused = None
        for i, (feat, proj) in enumerate(zip(features, self.projs)):
            # (B, T, C_in) → (B, C_in, h, w)
            B, T, C = feat.shape
            feat_2d = feat.permute(0, 2, 1).reshape(B, C, h, w).float()
            feat_proj = proj(feat_2d)  # (B, C_out, h, w)
            fused = feat_proj * weights[i] if fused is None else fused + feat_proj * weights[i]

        # 2× 上采样：H/16 → H/8
        out = F.interpolate(fused, size=(H // 8, W // 8),
                            mode="bilinear", align_corners=True)
        return out  # (B, C_out, H/8, W/8)


class DAFlowDPTHeads(nn.Module):
    """
    DA-Flow 的三个 DPT head，对应 Q、K、ctx 三种用途。

    输入：
      q_features:   list[L] of (B, T, C_in)  — corrected 分支的 Q 特征
      k_features:   list[L] of (B, T, C_in)  — warped condition 分支的 K 特征
      spatial_shape: (H, W)

    输出：
      F_Q:   (B, C_out, H/8, W/8)  — 用于 correlation（corrected 侧）
      F_K:   (B, C_out, H/8, W/8)  — 用于 correlation（warped 侧）
      F_ctx: (B, C_out, H/8, W/8)  — 用于 update operator context
    """

    def __init__(self, in_channels: int = 3072, out_channels: int = 96,
                 num_layers: int = 4):
        super().__init__()
        self.dpt_q   = DPTHead(in_channels, out_channels, num_layers)
        self.dpt_k   = DPTHead(in_channels, out_channels, num_layers)
        self.dpt_ctx = DPTHead(in_channels, out_channels // 3, num_layers)  # ctx 通道数较小

    def forward(self, q_features: list, k_features: list,
                spatial_shape: tuple):
        F_Q   = self.dpt_q(q_features,   spatial_shape)
        F_K   = self.dpt_k(k_features,   spatial_shape)
        F_ctx = self.dpt_ctx(q_features, spatial_shape)  # ctx 用 Q 侧特征
        return F_Q, F_K, F_ctx


if __name__ == "__main__":
    # 验证
    H, W = 1024, 1024
    B, L = 1, 4
    T = (H // 16) * (W // 16)  # 64*64 = 4096
    C_in = 3072

    heads = DAFlowDPTHeads(in_channels=C_in, out_channels=96, num_layers=L)
    total = sum(p.numel() for p in heads.parameters())
    print(f"DAFlowDPTHeads 参数量: {total:,}")

    q_feats = [torch.randn(B, T, C_in) for _ in range(L)]
    k_feats = [torch.randn(B, T, C_in) for _ in range(L)]
    F_Q, F_K, F_ctx = heads(q_feats, k_feats, (H, W))
    print(f"F_Q:   {tuple(F_Q.shape)}")    # (1, 96, 128, 128)
    print(f"F_K:   {tuple(F_K.shape)}")    # (1, 96, 128, 128)
    print(f"F_ctx: {tuple(F_ctx.shape)}")  # (1, 32, 128, 128)
