#!/usr/bin/env python3
"""
CorrBlock 修复自检 (2026-06-10)
================================
开训前的最后一道闸。验证四件事：
  1. flow=0 时 center-tap 相关 ≈ 1.0（两图相同 → 应处处自匹配）
  2. 已知位移时，相关峰落在正确的搜索窗位置（方向/通道约定正确）
  3. forward 跑通，且 F_ctx(dpt_ctx) 不再是死参数（梯度非 None）
  4. 合成 overfit：单样本能在几百步内把 EPE 压到亚像素（相关通路真的通了）

跑法：
  cd versions/20260603-v4_4-hardweight
  /juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python verify_corrblock_fix.py
"""
import torch
import torch.nn.functional as F

from utils.flow_head_v4_4 import (
    CorrBlock, FlowHeadV4_1, sequence_loss_with_grad,
)


def test_1_self_match():
    """两张相同特征图、flow=0：center-tap 相关应处处≈1.0。"""
    print("\n=== Test 1: flow=0 自匹配 (center-tap 应≈1.0) ===")
    B, C, H, W = 1, 16, 64, 64
    fmap = torch.randn(B, C, H, W)
    cb = CorrBlock(fmap.clone(), fmap.clone(), radius=4)
    flow = torch.zeros(B, 2, H, W)
    corr = cb(flow)  # (B, (2r+1)^2, H, W)
    r = 4
    center_idx = (2 * r + 1) * r + r  # (dy=0, dx=0) 在 i*..+j 展开里的位置
    center = corr[:, center_idx]      # (B, H, W)
    # 排除 border（grid_sample border padding 会让边缘自匹配略偏）
    inner = center[:, r:-r, r:-r]
    print(f"  center-tap 相关: mean={inner.mean():.4f} (期望≈1.0), min={inner.min():.4f}")
    ok = inner.mean() > 0.95
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_2_known_shift():
    """纯水平位移 dx=+2：相关峰应落在 (dy=0, dx=+2)。验证方向/通道约定。"""
    print("\n=== Test 2: 已知位移峰值定位 (方向约定) ===")
    B, C, H, W = 1, 16, 64, 64
    torch.manual_seed(0)
    base = torch.randn(B, C, H, W)
    # warped = base 向左移 2px（即 warped[x] = base[x+2]）；
    # 那么 corrected 像素 p 的对应在 warped 的 p-? —— 直接用 grid 构造保证一致性：
    # 让 fmap_w 是 fmap_c 沿 W 平移 +2，则 flow=(dx=2,dy=0) 时应自匹配。
    fmap_c = base
    # fmap_w[x] = base[x+2]，即把 base 沿 W 左移 2px。
    # corrected 像素 p 的对应特征在 fmap_w 的 (p-2) 处 → 应在搜索窗 dx=-2 处出现峰值。
    fmap_w = torch.roll(base, shifts=-2, dims=3)
    cb = CorrBlock(fmap_c, fmap_w, radius=4)
    flow = torch.zeros(B, 2, H, W)
    corr = cb(flow)  # 在 flow=0 处搜索窗内找峰
    r = 4
    # 取内部区域平均，找哪个 (dy,dx) 窗口相关最大
    inner = corr[:, :, r:-r, r:-r].mean(dim=(0, 2, 3))  # ((2r+1)^2,)
    peak = inner.argmax().item()
    pdy, pdx = peak // (2 * r + 1) - r, peak % (2 * r + 1) - r
    print(f"  峰值落在 (dy={pdy}, dx={pdx}) (期望 dx=-2, dy=0)")
    ok = (pdx == -2 and pdy == 0)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_3_forward_and_ctx_grad():
    """forward 跑通 + F_ctx(dpt_ctx) 梯度非 None（接回成功）。"""
    print("\n=== Test 3: forward + dpt_ctx 梯度回灌 ===")
    B, H, W = 1, 512, 512
    T = (H // 16) * (W // 16)
    m = FlowHeadV4_1(num_dit_layers=1)
    m.train()
    c = torch.randn(B, 3, H, W)
    w = torch.randn(B, 3, H, W)
    q = [torch.randn(B, T, 3072)]
    k = [torch.randn(B, T, 3072)]
    preds = m(c, w, q, k, iters=4)
    gt = torch.randn(B, 2, H, W)
    L_flow, L_grad = sequence_loss_with_grad(preds, gt, 0.8, 1.0)
    (L_flow + L_grad).backward()
    ctx_grad = m.dpt_heads.dpt_ctx.projs[0].weight.grad
    has_grad = ctx_grad is not None and ctx_grad.abs().sum() > 0
    print(f"  forward OK, preds={len(preds)}, L_flow={L_flow.item():.4f}")
    print(f"  dpt_ctx 梯度: {'非零 ✓ (接回成功)' if has_grad else 'None/零 ✗ (仍是死参数)'}")
    return has_grad


def test_4_overfit():
    """合成单样本 overfit：相关通路通了的话，EPE 应能压到亚像素。"""
    print("\n=== Test 4: 合成单样本 overfit (相关通路连通性) ===")
    torch.manual_seed(0)
    B, H, W = 1, 512, 512
    T = (H // 16) * (W // 16)
    m = FlowHeadV4_1(num_dit_layers=1)
    m.train()
    c = torch.randn(B, 3, H, W)
    w = torch.randn(B, 3, H, W)
    q = [torch.randn(B, T, 3072)]
    k = [torch.randn(B, T, 3072)]
    # 造一个平滑的 GT flow（低频，模拟真实扭曲场），H/8 尺度
    fh, fw = H // 8, W // 8
    gt_low = F.interpolate(torch.randn(B, 2, 8, 8) * 3, size=(fh, fw),
                           mode="bilinear", align_corners=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    epe0 = None
    for step in range(400):
        opt.zero_grad()
        preds = m(c, w, q, k, iters=4)
        pred_low = F.interpolate(preds[-1], size=(fh, fw),
                                 mode="bilinear", align_corners=True) * (fh / H)
        loss = F.l1_loss(pred_low, gt_low)
        loss.backward()
        opt.step()
        if step == 0:
            epe0 = torch.norm(pred_low.detach() - gt_low, dim=1).mean().item()
    epe = torch.norm(pred_low.detach() - gt_low, dim=1).mean().item()
    print(f"  EPE: {epe0:.3f}px (step0) → {epe:.3f}px (step400)")
    ok = epe < 1.0
    print(f"  {'✓ PASS (相关通路连通，能拟合几何 flow)' if ok else '✗ FAIL (仍学不出)'}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("CorrBlock 修复自检")
    print("=" * 60)
    results = {
        "Test1 自匹配":      test_1_self_match(),
        "Test2 峰值定位":    test_2_known_shift(),
        "Test3 dpt_ctx梯度": test_3_forward_and_ctx_grad(),
        "Test4 overfit":     test_4_overfit(),
    }
    print("\n" + "=" * 60)
    print("汇总:")
    for k, v in results.items():
        print(f"  {'✓' if v else '✗'} {k}")
    all_ok = all(results.values())
    print("=" * 60)
    print("✅ 全部通过，可以开训" if all_ok else "❌ 有失败项，先排查再开训")
