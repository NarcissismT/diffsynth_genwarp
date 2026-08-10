#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段光流 Warp 工具函数
========================
提供三个核心功能：
  1. load_raft_model   - 加载 torchvision 内置 RAFT 光流模型
  2. estimate_flow     - 用 RAFT 估计两张图之间的密集光流场
  3. upscale_flow      - 将低分辨率光流上采样到目标分辨率
  4. warp_image_with_flow - 用光流对高清图做像素重采样

光流方向约定
------------
estimate_flow 返回的是"反向光流"（backward flow）：
  flow[y, x] = (dx, dy) 表示输出图坐标 (x, y) 对应输入图的 (x+dx, y+dy)。
这与 grid_sample 的语义一致，可直接用于 warp_image_with_flow。

具体地：
  img1 = warped（变形图），img2 = corrected（矫正图）
  我们要生成高清矫正图，即对于矫正图每个坐标，去原始高清变形图中取像素。
  所以需要的是 corrected→warped 方向的映射，即反向光流。
  RAFT 默认返回前向光流（img1→img2），因此调用时交换参数顺序：
    estimate_flow(raft_model, corrected_1024, warped_1024) → 得到反向光流。
"""

import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# 预存权重路径（juicefs 上，容器内只读也能读）
_RAFT_WEIGHTS_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "torch_cache", "hub", "checkpoints"
)
# torch.hub 要求 TORCH_HOME 指向可写路径，/tmp 在 SLURM 容器内始终可写
_TORCH_HOME_WRITABLE = "/tmp/diffsynth_torch_home"


def _prepare_torch_home() -> str:
    """把 RAFT 权重从只读的 juicefs 复制到可写的 /tmp，返回 TORCH_HOME 路径。"""
    dst = os.path.join(_TORCH_HOME_WRITABLE, "hub", "checkpoints")
    os.makedirs(dst, exist_ok=True)
    if os.path.isdir(_RAFT_WEIGHTS_SRC):
        for fname in os.listdir(_RAFT_WEIGHTS_SRC):
            dst_file = os.path.join(dst, fname)
            if not os.path.exists(dst_file):
                shutil.copy2(os.path.join(_RAFT_WEIGHTS_SRC, fname), dst_file)
    return _TORCH_HOME_WRITABLE


def load_raft_model(device: torch.device, model_size: str = "large") -> torch.nn.Module:
    """
    加载 torchvision 内置的 RAFT 光流模型。
    权重预存在项目 models/torch_cache，运行时复制到 /tmp 以绕过容器只读限制。
    """
    os.environ["TORCH_HOME"] = _prepare_torch_home()

    if model_size == "large":
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    else:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)

    model = model.to(device).eval()
    return model


def _pil_to_raft_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """
    将 PIL 图像转为 RAFT 所需格式：float32 (1, 3, H, W)，值域 [0, 1]。
    宽高若不是 8 的倍数则 pad 到最近的 8 的倍数（RAFT 内部 8× 下采样）。
    torchvision RAFT 要求 float32 且值域 [0, 1]。
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)  # (H, W, 3)
    h, w = arr.shape[:2]

    # 向上对齐到 8 的倍数（pad 右侧和底部）
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    tensor = torch.from_numpy(arr).float().div(255.0).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H', W')
    return tensor.to(device), h, w  # 返回原始 h/w 用于裁回


@torch.no_grad()
def estimate_flow_batch(
    raft_model: torch.nn.Module,
    imgs_src: list,
    imgs_dst: list,
    device: torch.device,
    num_flow_updates: int = 20,
) -> list:
    """
    批量版本的 estimate_flow，支持一次处理多张图片。

    Args:
        raft_model: load_raft_model 返回的模型
        imgs_src:   PIL 图像列表（源图）
        imgs_dst:   PIL 图像列表（目标图），与 imgs_src 等长
        device:     运行设备
        num_flow_updates: RAFT 迭代精化次数

    Returns:
        flows: list of (1, 2, H, W) float32 tensor，与输入顺序一致
    """
    assert len(imgs_src) == len(imgs_dst), "src 和 dst 图像数量必须相等"

    tensors_src, tensors_dst, orig_sizes = [], [], []
    for src, dst in zip(imgs_src, imgs_dst):
        t_src, h, w = _pil_to_raft_tensor(src, device)
        t_dst, _, _ = _pil_to_raft_tensor(dst, device)
        tensors_src.append(t_src)
        tensors_dst.append(t_dst)
        orig_sizes.append((h, w))

    # 拼成 batch（所有图必须是同一尺寸，由调用方保证 resize 一致）
    batch_src = torch.cat(tensors_src, dim=0)  # (B, 3, H, W)
    batch_dst = torch.cat(tensors_dst, dim=0)

    flow_predictions = raft_model(batch_src, batch_dst, num_flow_updates=num_flow_updates)
    flow_batch = flow_predictions[-1]  # (B, 2, H', W')

    # 按原始尺寸裁回并拆分
    flows = []
    for i, (h, w) in enumerate(orig_sizes):
        flows.append(flow_batch[i:i+1, :, :h, :w].float())
    return flows


@torch.no_grad()
def estimate_flow(
    raft_model: torch.nn.Module,
    img_src: Image.Image,
    img_dst: Image.Image,
    device: torch.device,
    num_flow_updates: int = 20,
) -> torch.Tensor:
    """
    用 RAFT 估计从 img_src 到 img_dst 的前向光流。

    调用惯例（与 warp_image_with_flow 配套）：
        若要生成矫正图（corrected），应传入：
          img_src = corrected_1024,  img_dst = warped_1024
        即估计反向光流，直接用于从 warped 高清图中采样。

    Args:
        raft_model: load_raft_model 返回的模型
        img_src:    源图（PIL），光流从此坐标系出发
        img_dst:    目标图（PIL），光流指向此坐标系
        device:     运行设备
        num_flow_updates: RAFT 迭代精化次数（越大越精确，默认 20）

    Returns:
        flow: (1, 2, H, W) float32 tensor，单位为像素偏移
              flow[:, 0, ...] = dx（水平方向偏移），flow[:, 1, ...] = dy（垂直方向偏移）
              裁剪回 img_src 原始分辨率
    """
    t_src, h, w = _pil_to_raft_tensor(img_src, device)
    t_dst, _, _ = _pil_to_raft_tensor(img_dst, device)

    # RAFT 返回多个迭代结果的列表，取最后一个最精确的
    flow_predictions = raft_model(t_src, t_dst, num_flow_updates=num_flow_updates)
    flow = flow_predictions[-1]  # (1, 2, H', W')

    # 裁回原始尺寸（去掉 pad）
    flow = flow[:, :, :h, :w]
    return flow.float()


def upscale_flow(
    flow: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    将低分辨率光流场上采样到目标分辨率，同时按比例缩放偏移量数值。

    Args:
        flow:     (1, 2, src_h, src_w) 的光流张量
        target_h: 目标高度（原始高清图像的高度）
        target_w: 目标宽度（原始高清图像的宽度）

    Returns:
        flow_hires: (1, 2, target_h, target_w) 上采样后的光流
    """
    _, _, src_h, src_w = flow.shape

    # 双线性上采样空间维度
    flow_hires = F.interpolate(
        flow,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=True,
    )

    # 偏移量按分辨率比例缩放
    flow_hires[:, 0, :, :] *= target_w / src_w  # dx 分量
    flow_hires[:, 1, :, :] *= target_h / src_h  # dy 分量

    return flow_hires


def warp_image_with_flow(
    hires_img: Image.Image,
    flow: torch.Tensor,
    padding_mode: str = "zeros",
) -> Image.Image:
    """
    用光流对高清图做反向映射像素重采样（backward warping）。

    对于输出图中每个坐标 (x, y)，去原图 (x + dx, y + dy) 处采样。
    flow 应是"反向光流"（corrected→warped 方向），与 estimate_flow 约定一致。

    Args:
        hires_img:    原始高清变形图（PIL），尺寸应与 flow 空间维度一致
        flow:         (1, 2, H, W) 反向光流，单位像素偏移
        padding_mode: 越界像素处理方式，'zeros'=黑色填充，'border'=边界像素填充

    Returns:
        warped PIL 图像（与 hires_img 同尺寸）
    """
    device = flow.device
    _, _, H, W = flow.shape

    # 将 PIL 图转为 float32 tensor (1, 3, H, W)，值域 [-1, 1]（grid_sample 要求）
    arr = np.array(hires_img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # 构建 identity grid（归一化坐标，align_corners=True）
    # grid[..., 0] = x_normalized ∈ [-1, 1]，grid[..., 1] = y_normalized ∈ [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # 将像素偏移转换为归一化偏移
    # 归一化单位：1 像素 = 2 / (W-1) in x，2 / (H-1) in y
    dx_norm = flow[:, 0, :, :] * 2.0 / (W - 1)  # (1, H, W)
    dy_norm = flow[:, 1, :, :] * 2.0 / (H - 1)  # (1, H, W)
    delta_grid = torch.stack([dx_norm, dy_norm], dim=-1)  # (1, H, W, 2)

    sample_grid = base_grid + delta_grid  # 采样坐标 = 当前位置 + 偏移

    # 双线性采样
    warped = F.grid_sample(
        img_tensor,
        sample_grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )

    # 转回 PIL
    warped_arr = ((warped.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5)
    warped_arr = warped_arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(warped_arr)
