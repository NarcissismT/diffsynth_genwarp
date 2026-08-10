#!/usr/bin/env python3
"""
FlowHead V4 - 任意层选择的 DiT 特征光流网络
=============================================
在 V3 (DA-Flow 风格) 的基础上，核心改进是支持用户指定任意 DiT 层组合，
方便做 layer selection probing 实验（对比哪些层的几何先验最强）。

与 V3 的区别：
  V3：只能取"最后 num_dit_layers 层"（硬编码 top-L）
  V4：接受任意层索引列表，例如 [11], [23,35,47], [11,23,35,47]

架构与 V3 完全相同，不引入新结构：
  CNN local encoder   → 提取 warped 图局部纹理/边缘
  DPT Head (Q/K/ctx)  → 将 DiT 指定层 Q/K 特征上采样到 H/8
  HybridFeatureEncoder→ CNN + DPT 拼接
  ContextEncoder      → RAFT-style hidden state 初始化
  CorrBlock           → all-pairs 相似度
  ConvGRU + FlowHead  → 迭代精化，输出 backward displacement field

所有模块均从本文件内部定义，不依赖 v2/v3，便于独立回滚。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础模块（与 V2/V3 相同，独立定义）
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
        return F.relu(
            self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))) +
            self.downsample(x)
        )


class ContextEncoder(nn.Module):
    """输出 hidden (B,96,H/8,W/8) 和 context (B,32,H/8,W/8)，用于 RAFT GRU 初始化。"""
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.norm1  = nn.BatchNorm2d(64)
        self.layer1 = ResBlock(64, 128, stride=2, norm=nn.BatchNorm2d)
        self.layer2 = ResBlock(128, 128, stride=2, norm=nn.BatchNorm2d)
        self.out_conv = nn.Conv2d(128, 96 + 32, 1)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out_conv(x)
        hidden, context = x[:, :96], x[:, 96:]
        return torch.tanh(hidden), torch.relu(context)


class CorrBlock:
    """在 H/8 分辨率上计算 all-pairs cosine 相似度，搜索半径 r。"""
    def __init__(self, fmap_c: torch.Tensor, fmap_w: torch.Tensor, radius: int = 4):
        self.radius = radius
        B, C, H, W = fmap_c.shape
        self.fmap_c = F.normalize(fmap_c, dim=1)
        self.fmap_w = F.normalize(fmap_w, dim=1)

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = self.fmap_c.shape
        r = self.radius
        dy, dx = torch.meshgrid(
            torch.arange(-r, r + 1, device=flow.device),
            torch.arange(-r, r + 1, device=flow.device),
            indexing="ij",
        )
        delta = torch.stack([dx, dy], dim=-1).float()
        centroid = flow.permute(0, 2, 3, 1)
        centroid_norm = centroid / torch.tensor(
            [W - 1, H - 1], device=flow.device) * 2 - 1

        corr_list = []
        for i in range(2 * r + 1):
            for j in range(2 * r + 1):
                offset = delta[i, j]
                offset_norm = offset / torch.tensor([W - 1, H - 1], device=flow.device) * 2
                sample_coords = centroid_norm + offset_norm
                fmap_sample = F.grid_sample(
                    self.fmap_w, sample_coords,
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
# DPT 上采样头（与 dpt_head.py 相同，独立定义）
# ---------------------------------------------------------------------------

class DPTHead(nn.Module):
    """
    聚合 L 层 DiT Q/K token 特征并上采样到 H/8。
    输入：list of (B, T, C_in)，T = (H/16)*(W/16)
    输出：(B, C_out, H/8, W/8)
    """
    def __init__(self, in_channels: int = 3072, out_channels: int = 96,
                 num_layers: int = 4):
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
    """三个独立 DPT head，分别处理 Q、K、ctx。"""
    def __init__(self, in_channels: int = 3072, out_channels: int = 96,
                 num_layers: int = 4):
        super().__init__()
        self.dpt_q   = DPTHead(in_channels, out_channels, num_layers)
        self.dpt_k   = DPTHead(in_channels, out_channels, num_layers)
        self.dpt_ctx = DPTHead(in_channels, out_channels // 3, num_layers)

    def forward(self, q_features: list, k_features: list,
                spatial_shape: tuple):
        F_Q   = self.dpt_q(q_features, spatial_shape)
        F_K   = self.dpt_k(k_features, spatial_shape)
        F_ctx = self.dpt_ctx(q_features, spatial_shape)
        return F_Q, F_K, F_ctx


# ---------------------------------------------------------------------------
# 混合特征编码器
# ---------------------------------------------------------------------------

class HybridEncoder(nn.Module):
    """CNN 局部特征 + DPT 扩散特征拼接。"""
    def __init__(self, cnn_out_ch: int = 96, diff_out_ch: int = 96):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)
        self.norm1  = nn.InstanceNorm2d(32)
        self.layer1 = ResBlock(32, 64, stride=2, norm=nn.InstanceNorm2d)
        self.layer2 = ResBlock(64, cnn_out_ch, stride=2, norm=nn.InstanceNorm2d)

    def forward(self, img: torch.Tensor, diff_feat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(img)))
        x = self.layer1(x)
        x = self.layer2(x)
        return torch.cat([x, diff_feat], dim=1)


# ---------------------------------------------------------------------------
# FlowHeadV4 主网络
# ---------------------------------------------------------------------------

class FlowHeadV4LayerProbe(nn.Module):
    """
    Layer probing 版光流网络。

    相比 V3 的唯一变化是：
      - num_dit_layers 由外部根据 len(target_layers) 推断传入，不再硬编码取"最后 N 层"
      - 内部结构与 V3 完全相同

    Args:
        iters:         迭代次数（训练 4，推理 12）
        radius:        Correlation 搜索半径（默认 4）
        dit_channels:  DiT 隐层维度（默认 3072）
        diff_out_ch:   DPT Head 输出通道数（默认 96）
        cnn_out_ch:    CNN 输出通道数（默认 96）
        num_dit_layers: 等于 len(target_layers)，由训练脚本传入
    """

    def __init__(self, iters: int = 4, radius: int = 4,
                 dit_channels: int = 3072, diff_out_ch: int = 96,
                 cnn_out_ch: int = 96, num_dit_layers: int = 4):
        super().__init__()
        self.iters  = iters
        self.radius = radius

        self.dpt_heads = DPTHeads(
            in_channels=dit_channels,
            out_channels=diff_out_ch,
            num_layers=num_dit_layers,
        )
        self.feat_encoder = HybridEncoder(
            cnn_out_ch=cnn_out_ch, diff_out_ch=diff_out_ch)

        self.context_encoder = ContextEncoder()

        corr_ch = (2 * radius + 1) ** 2
        self.update_block = ConvGRU(hidden_dim=96, input_dim=corr_ch + 2 + 32)
        self.flow_head = FlowPredHead(hidden_dim=96)

    def upsample_flow(self, flow: torch.Tensor,
                      target_h: int, target_w: int) -> torch.Tensor:
        _, _, h, w = flow.shape
        flow_up = F.interpolate(flow, size=(target_h, target_w),
                                mode="bilinear", align_corners=True)
        flow_up[:, 0] *= float(target_w) / w
        flow_up[:, 1] *= float(target_h) / h
        return flow_up

    def forward(self,
                corrected: torch.Tensor,
                warped: torch.Tensor,
                q_features: list,
                k_features: list,
                iters: int = None):
        """
        corrected:  (B, 3, H, W)，矫正图，值域 [-1,1]
        warped:     (B, 3, H, W)，弯曲输入图，值域 [-1,1]
        q_features: list[L] of (B, T, C)，选定层的 img Q 特征
        k_features: list[L] of (B, T, C)，选定层的 img K 特征
        iters:      推理时可覆盖

        Returns:
            训练模式：list of (B, 2, H, W)，共 iters 个（供 sequence loss）
            推理模式：(B, 2, H, W)，最终 backward displacement field，单位像素
        """
        iters = iters or self.iters
        B, _, H, W = corrected.shape

        F_Q, F_K, _ = self.dpt_heads(q_features, k_features, (H, W))

        fmap_c = self.feat_encoder(corrected, F_Q)
        fmap_w = self.feat_encoder(warped,    F_K)

        hidden, context = self.context_encoder(corrected)

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
# 序列 Loss
# ---------------------------------------------------------------------------

def sequence_loss(predictions: list, flow_gt: torch.Tensor,
                  gamma: float = 0.8) -> torch.Tensor:
    N = len(predictions)
    loss = 0.0
    for i, pred in enumerate(predictions):
        weight = gamma ** (N - 1 - i)
        loss += weight * F.l1_loss(pred, flow_gt)
    return loss


# ---------------------------------------------------------------------------
# 参数量统计
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for num_layers, label in [(1, "单层"), (3, "三层"), (4, "四层")]:
        model = FlowHeadV4LayerProbe(num_dit_layers=num_layers)
        total = sum(p.numel() for p in model.parameters())
        print(f"FlowHeadV4 ({label}, num_dit_layers={num_layers}): {total:,} 参数")

    # 前向验证（四层）
    model = FlowHeadV4LayerProbe(num_dit_layers=4)
    model.train()
    B, H, W, L = 1, 512, 512, 4
    T = (H // 16) * (W // 16)
    corrected = torch.randn(B, 3, H, W)
    warped    = torch.randn(B, 3, H, W)
    q_feats   = [torch.randn(B, T, 3072) for _ in range(L)]
    k_feats   = [torch.randn(B, T, 3072) for _ in range(L)]
    preds = model(corrected, warped, q_feats, k_feats, iters=4)
    print(f"\n训练模式: {len(preds)} 个预测，shape={tuple(preds[0].shape)}")
    loss = sequence_loss(preds, torch.randn(B, 2, H, W))
    loss.backward()
    print(f"Loss={loss.item():.4f}，反向传播 OK")
