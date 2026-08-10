#!/usr/bin/env python3
"""
FlowHead V3 (DA-Flow 风格) - 混合光流网络
==========================================
在 FlowHead V2 基础上，注入 DiT 中间层 Q/K 特征（DA-Flow Section 4.4）。

核心改进：
  V2：feature_encoder(corrected) + feature_encoder(warped) → Correlation → flow
  V3：[feature_encoder(corrected) + DPT_Q] + [feature_encoder(warped) + DPT_K] → Correlation → flow

  扩散特征提供语义/几何感知（对退化/折痕不敏感）
  CNN 特征提供局部纹理细节（边界/表格线精确定位）
  两者互补，和 DA-Flow 完全一致

参数量：V2 ~137万 + DPT heads ~275万 = ~412万
        vs RAFT large ~500万，接近但更轻量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.flow_head_v2 import (
    ResBlock, ContextEncoder, CorrBlock, ConvGRU, FlowHead, sequence_loss
)
from utils.dpt_head import DAFlowDPTHeads


# ---------------------------------------------------------------------------
# 混合特征编码器（CNN + 扩散特征拼接）
# ---------------------------------------------------------------------------

class HybridFeatureEncoder(nn.Module):
    """
    DA-Flow Eq.(15)-(16) 的实现：
      F_hybrid = concat(F_CNN, F_diff)

    CNN 特征：从像素图提取局部纹理（与 V2 的 FeatureEncoder 相同）
    扩散特征：由外部 DPT Head 提供，传入 forward
    """

    def __init__(self, cnn_out_ch: int = 96, diff_out_ch: int = 96):
        super().__init__()
        # CNN feature encoder（与 V2 相同结构）
        self.conv1  = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)
        self.norm1  = nn.InstanceNorm2d(32)
        self.layer1 = ResBlock(32, 64, stride=2, norm=nn.InstanceNorm2d)
        self.layer2 = ResBlock(64, cnn_out_ch, stride=2, norm=nn.InstanceNorm2d)
        self.cnn_out_ch = cnn_out_ch
        self.diff_out_ch = diff_out_ch
        self.out_ch = cnn_out_ch + diff_out_ch  # 拼接后的总通道数

    def forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """纯 CNN 前向，返回 (B, cnn_out_ch, H/8, W/8)"""
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward(self, img: torch.Tensor,
                diff_feat: torch.Tensor) -> torch.Tensor:
        """
        img:       (B, 3, H, W)，像素图，值域 [-1, 1]
        diff_feat: (B, diff_out_ch, H/8, W/8)，DPT Head 输出的扩散特征

        Returns: (B, cnn_out_ch + diff_out_ch, H/8, W/8)
        """
        cnn_feat = self.forward_cnn(img)
        return torch.cat([cnn_feat, diff_feat], dim=1)


# ---------------------------------------------------------------------------
# FlowHead V3 主网络（DA-Flow 风格）
# ---------------------------------------------------------------------------

class FlowHeadV3DAFlow(nn.Module):
    """
    DA-Flow 风格的混合光流网络（~412万参数）。

    和 V2 的区别：
      - 新增 DAFlowDPTHeads：将 DiT Q/K 特征上采样到 H/8
      - feature encoder 改为 HybridFeatureEncoder：CNN + DPT 特征拼接
      - Correlation 和 Update 与 V2 完全相同

    推理时需要额外传入 q_features / k_features（DiT 中间层特征）
    训练时同样需要（从 GT 矫正图的 DiT 推理中提取）

    Args:
        iters:          迭代次数（训练 4，推理 12）
        radius:         Correlation 搜索半径（默认 4）
        dit_channels:   DiT 隐层维度（默认 3072）
        diff_out_ch:    DPT Head 输出通道数（默认 96）
        cnn_out_ch:     CNN 输出通道数（默认 96）
        num_dit_layers: 提取的 DiT 层数（默认 4，与 DA-Flow 一致）
    """

    def __init__(self, iters: int = 4, radius: int = 4,
                 dit_channels: int = 3072, diff_out_ch: int = 96,
                 cnn_out_ch: int = 96, num_dit_layers: int = 4):
        super().__init__()
        self.iters  = iters
        self.radius = radius

        # DPT 上采样头（DA-Flow Section 4.4.1）
        ctx_ch = diff_out_ch // 3  # context 通道数较小
        self.dpt_heads = DAFlowDPTHeads(
            in_channels=dit_channels,
            out_channels=diff_out_ch,
            num_layers=num_dit_layers,
        )

        # 混合特征编码器（DA-Flow Section 4.4.2 Hybrid feature encoding）
        self.feat_encoder = HybridFeatureEncoder(
            cnn_out_ch=cnn_out_ch, diff_out_ch=diff_out_ch)
        hybrid_ch = cnn_out_ch + diff_out_ch  # 192

        # Context encoder（只处理 corrected，不用扩散特征）
        self.context_encoder = ContextEncoder()

        # ConvGRU：输入 = corr(81) + flow(2) + context(32)
        corr_ch = (2 * radius + 1) ** 2  # 81
        self.update_block = ConvGRU(
            hidden_dim=96,
            input_dim=corr_ch + 2 + 32,
        )
        self.flow_head = FlowHead(hidden_dim=96)

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
        corrected:  (B, 3, H, W)，矫正图（GT 或扩散输出），值域 [-1,1]
        warped:     (B, 3, H, W)，弯曲输入图，值域 [-1,1]
        q_features: list[L] of (B, T, C)，DiT top-L 层 Q 特征（corrected 分支）
        k_features: list[L] of (B, T, C)，DiT top-L 层 K 特征（warped condition 分支）
        iters:      迭代次数（推理时可覆盖）

        Returns:
            训练模式：list of (B, 2, H, W)，共 iters 个
            推理模式：(B, 2, H, W)，最终光流
        """
        iters = iters or self.iters
        B, _, H, W = corrected.shape

        # 1. DPT 上采样：DiT Q/K → (B, diff_out_ch, H/8, W/8)
        F_Q, F_K, F_ctx_diff = self.dpt_heads(q_features, k_features, (H, W))

        # 2. 混合特征编码（CNN + DPT 拼接）
        fmap_c = self.feat_encoder(corrected, F_Q)   # (B, 192, H/8, W/8)
        fmap_w = self.feat_encoder(warped,    F_K)   # (B, 192, H/8, W/8)

        # 3. Context encoder（只用 corrected）
        hidden, context = self.context_encoder(corrected)  # (B,96,H/8,W/8)

        # 4. Correlation volume
        corr_block = CorrBlock(fmap_c, fmap_w, radius=self.radius)

        # 5. 迭代更新
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
# 参数量统计
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    model = FlowHeadV3DAFlow(iters=4)
    total = sum(p.numel() for p in model.parameters())
    print(f"FlowHead V3 (DA-Flow) 总参数量: {total:,}")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n:,}")

    # 前向验证
    B, H, W = 1, 512, 512
    L = 4
    T = (H // 16) * (W // 16)
    C = 3072

    corrected = torch.randn(B, 3, H, W)
    warped    = torch.randn(B, 3, H, W)
    q_feats   = [torch.randn(B, T, C) for _ in range(L)]
    k_feats   = [torch.randn(B, T, C) for _ in range(L)]

    # 训练模式
    model.train()
    preds = model(corrected, warped, q_feats, k_feats, iters=4)
    print(f"\n训练模式: {len(preds)} 个预测，shape={tuple(preds[0].shape)}")

    flow_gt = torch.randn(B, 2, H, W)
    loss = sequence_loss(preds, flow_gt)
    loss.backward()
    print(f"Loss: {loss.item():.4f}，反向传播 OK")

    # 推理模式
    model.eval()
    with torch.no_grad():
        flow = model(corrected, warped, q_feats, k_feats, iters=12)
    print(f"\n推理模式: flow shape={tuple(flow.shape)}")
