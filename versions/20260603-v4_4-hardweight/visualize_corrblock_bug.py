#!/usr/bin/env python3
"""
Visualize the CorrBlock bug: coords0+flow mistakenly used as just `flow`
========================================================================
复现根因 bug 并和修复版并排对比，全部用【真实三元组】数据。

bug 本质：RAFT 相关查表中心应是 coords0(像素自身坐标)+flow(位移)；旧代码只用了
flow，丢掉 coords0 → flow=0 时每个像素都去 warped 特征图左上角采样 → correlation
volume 与几何脱钩 → 估不出位移。

【真实数据】（这是与上一版的关键区别——不再合成扭曲场）：
  - warped 图 = img/Doc3d_crop/Doc3d_crop_0000000.png（真实扭曲文档）
  - gt 图     = gt/Doc3d_crop/Doc3d_crop_0000000.png（真实矫正结果）
  - flow_gt   = flow/.../*.npy（真实 backward flow，已验证 warp(warped,flow_gt)≈gt，L1=0.007）
    方向约定：gt[p] = warped[p + flow_gt[p]]

【特征的诚实说明】：CorrBlock 实际作用在 DiT 特征上，而 DiT 特征无法离线复现
（需跑完整扩散）。因此相关计算用【判别性代理特征】，但其几何对应关系 100% 由真实
flow_gt 定义（fmap_w = warp(fmap_c, -flow_gt)）。即：几何真实、特征是代理。
这恰好隔离出 CorrBlock 的坐标 bug——给定良好特征 + 真实几何，buggy 仍失败、fixed 成功。

公平性自检（末尾断言）：fixed 恢复真实 flow_gt 的 EPE 必须 << buggy，且明显低于
zero-flow 基线，否则说明 demo 构造偏袒，对比不可信。

跑法：
  cd versions/20260603-v4_4-hardweight
  /juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python visualize_corrblock_bug.py
输出：corrblock_bug_vis/ 下 5 张 png
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corrblock_bug_vis")
os.makedirs(OUT, exist_ok=True)
_BASE = "/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511"
WARP = f"{_BASE}/img/Doc3d_crop/Doc3d_crop_0000000.png"   # 真实扭曲文档
GT   = f"{_BASE}/gt/Doc3d_crop/Doc3d_crop_0000000.png"    # 真实矫正 gt
FLOW = f"{_BASE}/flow/Doc3d_crop/Doc3d_crop_0000000.npy"  # 真实 backward flow

IMG = 512           # 像素 warp 用的分辨率
H = W = 64          # 特征分辨率（= 512/8）
R = 4               # 相关半径
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 两版 CorrBlock（唯一区别：fixed 版加了 coords0 base grid）
# ---------------------------------------------------------------------------
def corr_buggy(fmap_c, fmap_w, flow, r=R):
    """旧代码（逐字复现 V4.2 flow_head_v4_1_grad.py）：centroid = flow，无 coords0。"""
    B, C, Hh, Ww = fmap_c.shape
    dy, dx = torch.meshgrid(torch.arange(-r, r + 1.), torch.arange(-r, r + 1.), indexing="ij")
    delta = torch.stack([dx, dy], dim=-1)
    centroid = flow.permute(0, 2, 3, 1)
    centroid_norm = centroid / torch.tensor([Ww - 1., Hh - 1.]) * 2 - 1   # ← 把位移当绝对坐标
    out = []
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            off = delta[i, j] / torch.tensor([Ww - 1., Hh - 1.]) * 2
            samp = centroid_norm + off
            fs = F.grid_sample(fmap_w, samp, mode="bilinear",
                               padding_mode="border", align_corners=True)
            out.append((fmap_c * fs).sum(1, keepdim=True))
    return torch.cat(out, 1)


def corr_fixed(fmap_c, fmap_w, flow, r=R):
    """修复版：coords = coords0(identity grid) + flow，再加搜索窗。"""
    B, C, Hh, Ww = fmap_c.shape
    gy, gx = torch.meshgrid(torch.arange(float(Hh)), torch.arange(float(Ww)), indexing="ij")
    coords0 = torch.stack([gx, gy], dim=-1).unsqueeze(0)                # (1,H,W,2)
    coords = coords0 + flow.permute(0, 2, 3, 1)                         # ← 关键：+coords0
    dy, dx = torch.meshgrid(torch.arange(-r, r + 1.), torch.arange(-r, r + 1.), indexing="ij")
    delta = torch.stack([dx, dy], dim=-1)
    scale = torch.tensor([Ww - 1., Hh - 1.])
    out = []
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            samp = (coords + delta[i, j]) / scale * 2 - 1
            fs = F.grid_sample(fmap_w, samp, mode="bilinear",
                               padding_mode="border", align_corners=True)
            out.append((fmap_c * fs).sum(1, keepdim=True))
    return torch.cat(out, 1)


def softargmax_offset(corr, r=R, temp=8.0):
    """correlation volume → 每像素相关峰的亚像素偏移（soft-argmax，近似真实 GRU 行为）。"""
    B, L, Hh, Ww = corr.shape
    dy, dx = torch.meshgrid(torch.arange(-r, r + 1.), torch.arange(-r, r + 1.), indexing="ij")
    off = torch.stack([dx.reshape(-1), dy.reshape(-1)], 0)  # (2, L)
    wgt = torch.softmax(corr * temp, dim=1)
    return torch.einsum("blhw,kl->bkhw", wgt, off)          # (B,2,H,W)


def iterate(corr_fn, fmap_c, fmap_w, iters=20, damp=0.6):
    """RAFT 风格纯几何迭代（无学习权重）：每步用相关峰亚像素残差更新 flow，阻尼防过冲。"""
    flow = torch.zeros(1, 2, H, W)
    traj = []
    for _ in range(iters):
        delta = softargmax_offset(corr_fn(fmap_c, fmap_w, flow))
        flow = flow + damp * delta
        traj.append(flow.clone())
    return flow, traj


def epe(pred, gt):
    return torch.norm(pred - gt, dim=1).mean().item()


# ---------------------------------------------------------------------------
# 真实数据加载
# ---------------------------------------------------------------------------
def load_real_flow_feat():
    """真实 flow_gt（2,1024,1024）下采到特征分辨率 (1,2,64,64)，数值按比例缩放。"""
    f = np.load(FLOW).astype(np.float32)               # (2, 1024, 1024)
    t = torch.from_numpy(f).unsqueeze(0)               # (1,2,1024,1024)
    src = t.shape[-1]
    fl = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=True)
    fl = fl * (float(W) / src)                         # 位移数值同比缩放到特征尺度
    return fl


def make_discriminative_feat(C=32):
    """
    判别性代理特征（每像素近似唯一的高维向量 + 轻度空间平滑）。
    用 gt 图低频亮度调制一路，让特征与真实图像有结构关联（仅观感）。
    DiT 真特征无法离线复现，这里用代理；几何对应由真实 flow_gt 定义。
    """
    rand = torch.randn(1, C, H, W)
    k = torch.tensor([1., 4., 6., 4., 1.]); k = k / k.sum()
    ker = (k[:, None] * k[None, :])[None, None]
    rand = F.conv2d(rand.reshape(C, 1, H, W), ker, padding=2).reshape(1, C, H, W)
    a = torch.from_numpy(
        np.array(Image.open(GT).convert("L").resize((W, H), Image.BILINEAR),
                 dtype=np.float32) / 255.)
    rand[:, 0] = rand[:, 0] + 2.0 * a
    return rand


def warp_feat(fmap, flow):
    """backward warp: out[p] = fmap[p + flow[p]]（特征像素单位）。"""
    B, C, h, w = fmap.shape
    gy, gx = torch.meshgrid(torch.arange(float(h)), torch.arange(float(w)), indexing="ij")
    c0 = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    grid = (c0 + flow.permute(0, 2, 3, 1)) / torch.tensor([w - 1., h - 1.]) * 2 - 1
    return F.grid_sample(fmap, grid, mode="bilinear", padding_mode="border", align_corners=True)


def warp_img(img, flow):
    """对像素图按 flow backward warp（flow 为像素单位，与 img 同分辨率）。"""
    B, C, h, w = img.shape
    gy, gx = torch.meshgrid(torch.arange(float(h)), torch.arange(float(w)), indexing="ij")
    c0 = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    grid = (c0 + flow.permute(0, 2, 3, 1)) / torch.tensor([w - 1., h - 1.]) * 2 - 1
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def upscale_flow(flow, s=IMG):
    f = F.interpolate(flow, size=(s, s), mode="bilinear", align_corners=True)
    f[:, 0] *= s / W; f[:, 1] *= s / H
    return f


def flow_to_rgb(flow):
    dx, dy = flow[0, 0].numpy(), flow[0, 1].numpy()
    mag = np.sqrt(dx ** 2 + dy ** 2); ang = np.arctan2(dy, dx)
    hsv = np.zeros((*dx.shape, 3))
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / (mag.max() + 1e-6), 0, 1)
    return mcolors.hsv_to_rgb(hsv)


def to_img(t):
    return ((t[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
def main():
    # ---- 真实数据 ----
    warped_pil = Image.open(WARP).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
    gt_pil     = Image.open(GT).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
    warped_t = torch.from_numpy(np.array(warped_pil, np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0)
    gt_t     = torch.from_numpy(np.array(gt_pil, np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0)

    gt_flow = load_real_flow_feat()                    # 真实 flow_gt（特征尺度）
    zero_epe = torch.norm(gt_flow, dim=1).mean().item()

    # ---- 特征：判别性代理，几何对应由真实 flow_gt 定义 ----
    fmap_c = make_discriminative_feat()                # 代表 gt/corrected 特征
    fmap_w = warp_feat(fmap_c, -gt_flow)               # warped 特征：对应关系 = 真实 flow

    # ---- 两版 CorrBlock 迭代估计 ----
    flow_bug, traj_bug = iterate(corr_buggy, fmap_c, fmap_w)
    flow_fix, traj_fix = iterate(corr_fixed, fmap_c, fmap_w)
    epe_bug, epe_fix = epe(flow_bug, gt_flow), epe(flow_fix, gt_flow)

    print(f"真实 flow_gt 平均幅度(zero-EPE 基线，特征尺度) = {zero_epe:.3f} px")
    print(f"buggy 版估计 EPE = {epe_bug:.3f} px")
    print(f"fixed 版估计 EPE = {epe_fix:.3f} px")
    print(f"fixed 相对 buggy 误差降低 {(1 - epe_fix / max(epe_bug,1e-6)) * 100:.1f}%")

    # ---- 图0：真实三元组（warped / gt / 真实 flow_gt）----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(warped_pil); axes[0].set_title("REAL warped (input)", fontsize=12)
    axes[1].imshow(gt_pil);     axes[1].set_title("REAL gt (target)", fontsize=12)
    axes[2].imshow(flow_to_rgb(gt_flow))
    axes[2].set_title(f"REAL flow_gt (mean={zero_epe:.2f}px @feat)", fontsize=12)
    for ax in axes: ax.axis("off")
    plt.suptitle("Real data triplet (this document's actual distortion)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT}/0_real_data.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- 图1：flow 场对比（真实 GT / buggy / fixed）----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, fl, ttl in zip(
            axes, [gt_flow, flow_bug, flow_fix],
            [f"REAL flow_gt (zero-EPE={zero_epe:.2f}px)",
             f"buggy: lost coords0 (EPE={epe_bug:.2f}px)",
             f"fixed: coords0+flow (EPE={epe_fix:.2f}px)"]):
        ax.imshow(flow_to_rgb(fl)); ax.set_title(ttl, fontsize=11); ax.axis("off")
    plt.suptitle("Flow field (HSV: hue=direction, brightness=magnitude)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT}/1_flow_compare.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- 图2：flow=0 时相关查表中心（核心 bug 图解）----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ys, xs = np.meshgrid(np.arange(4, H, 10), np.arange(4, W, 10), indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    axes[0].scatter(pts[:, 0], pts[:, 1], c="tab:blue", s=18, label="pixel's own position")
    axes[0].scatter([0]*len(pts), [0]*len(pts), c="red", s=40, marker="x",
                    label="actual sample center @flow=0")
    for (px, py) in pts:
        axes[0].annotate("", xy=(0, 0), xytext=(px, py),
                         arrowprops=dict(arrowstyle="->", color="red", alpha=0.25, lw=0.6))
    axes[0].set_title("buggy: every pixel samples top-left (0,0)\n→ geometry destroyed", fontsize=11)
    axes[0].set_xlim(-2, W); axes[0].set_ylim(H, -2); axes[0].legend(fontsize=8)
    axes[1].scatter(pts[:, 0], pts[:, 1], c="tab:blue", s=18, label="pixel's own position")
    axes[1].scatter(pts[:, 0], pts[:, 1], c="green", s=40, marker="x",
                    label="actual sample center @flow=0")
    axes[1].set_title("fixed: each pixel samples its own\nposition → correct geometry", fontsize=11)
    axes[1].set_xlim(-2, W); axes[1].set_ylim(H, -2); axes[1].legend(fontsize=8)
    plt.suptitle("Root cause: correlation lookup center @flow=0 (±4 window around it)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT}/2_sampling_center.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- 图3：EPE 随迭代 ----
    eb = [epe(f, gt_flow) for f in traj_bug]
    ef = [epe(f, gt_flow) for f in traj_fix]
    plt.figure(figsize=(7, 4.5))
    plt.axhline(zero_epe, color="gray", ls="--", label=f"zero-flow baseline ({zero_epe:.2f}px)")
    plt.plot(range(1, len(eb)+1), eb, "o-", color="red", label="buggy (lost coords0)")
    plt.plot(range(1, len(ef)+1), ef, "o-", color="green", label="fixed (coords0+flow)")
    plt.xlabel("RAFT iteration"); plt.ylabel("EPE (px)")
    plt.title("EPE over iterations: fixed converges, buggy diverges")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/3_epe_curve.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- 图4：特征尺度(64×64) warp 还原 ----
    # 关键：所有量都在 CorrBlock 实际工作的 64 尺度，【不做 8× 上采】，
    # 避免朴素双线性上采把误差放大成假扭曲（真实模型有学习的上采头+512训练，
    # 此 demo 没有，故不在 512 尺度比，以免误导）。
    # 真实关系：gt = warp(warped, flow_gt)。把真实 warped/gt 下采到 64，用 64 尺度 flow 直接 warp。
    warped64 = F.interpolate(warped_t, size=(H, W), mode="bilinear", align_corners=True)
    gt64     = F.interpolate(gt_t,     size=(H, W), mode="bilinear", align_corners=True)
    rec_bug = warp_img(warped64, flow_bug)
    rec_fix = warp_img(warped64, flow_fix)
    rec_gt  = warp_img(warped64, gt_flow)                 # 真实 flow 还原（同尺度上界）
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    panels = [(to_img(warped64), "REAL warped @64 (input)"),
              (to_img(rec_bug), f"buggy recon (EPE={epe_bug:.2f}px @feat)"),
              (to_img(rec_fix), f"fixed recon (EPE={epe_fix:.2f}px @feat)"),
              (to_img(rec_gt),  "flow_gt recon (upper bound)"),
              (to_img(gt64), "REAL gt @64 (target)")]
    for ax, (im, ttl) in zip(axes, panels):
        ax.imshow(im); ax.set_title(ttl, fontsize=10); ax.axis("off")
    plt.suptitle("Warp at feature scale (64×64, no upsampling): "
                 "buggy garbled, fixed ≈ gt", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT}/4_warp_featscale.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- 公平性自检 ----
    print("\n=== 公平性自检 ===")
    ok1 = epe_fix < zero_epe * 0.7
    ok2 = epe_fix < epe_bug * 0.6
    print(f"  [{'✓' if ok1 else '✗'}] fixed EPE({epe_fix:.2f}) 明显低于 zero-flow 基线"
          f"({zero_epe:.2f}, 阈值 {zero_epe*0.7:.2f}) —— fixed 确实学到了位移")
    print(f"  [{'✓' if ok2 else '✗'}] fixed EPE({epe_fix:.2f}) < buggy EPE({epe_bug:.2f}) 的 60%"
          f"({epe_bug*0.6:.2f}) —— 对比公平、差距显著")
    print("  ✅ 自检通过：可视化对比可信" if (ok1 and ok2)
          else "  ⚠️ 自检未全过：demo 构造可能偏袒或搜索范围不足，结论需谨慎")

    print(f"\n✅ 5 张图已保存到 {OUT}/")
    for f in ["0_real_data.png", "1_flow_compare.png", "2_sampling_center.png",
              "3_epe_curve.png", "4_warp_featscale.png"]:
        print(f"   - {f}")
    # 清理旧的 512 尺度图，避免混淆
    old = os.path.join(OUT, "4_pixel_warp.png")
    if os.path.exists(old):
        os.remove(old)


if __name__ == "__main__":
    main()
