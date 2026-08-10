"""Latent-space backward-flow guidance for image editing diffusion.

The flow convention is the same as ``grid_sample`` backward warping:
``output(y, x) = source(y + dy, x + dx)``.  Flow values are expressed in
pixels on ``source_size`` (normally the input RGB canvas), then converted to
the latent canvas before sampling.  Keeping this operation outside the model
architecture makes it usable with existing Qwen checkpoints.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


SizeHW = Tuple[int, int]


def resize_backward_flow_to_latents(
    flow: torch.Tensor,
    latent_size: SizeHW,
    *,
    source_size: SizeHW | None = None,
) -> torch.Tensor:
    """Resize a pixel-unit backward flow to a latent grid.

    Args:
        flow: ``[B, 2, H, W]`` or ``[2, H, W]`` backward displacement.
        latent_size: target ``(height, width)`` of the latent tensor.
        source_size: ``(height, width)`` of the coordinate canvas represented
            by flow values.  If omitted, the flow grid size is used.
    """
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must have shape [B,2,H,W], got {tuple(flow.shape)}")
    target_h, target_w = (int(latent_size[0]), int(latent_size[1]))
    if target_h < 1 or target_w < 1:
        raise ValueError(f"latent_size must be positive, got {latent_size}")
    flow_h, flow_w = (int(flow.shape[-2]), int(flow.shape[-1]))
    canvas_h, canvas_w = source_size or (flow_h, flow_w)
    if canvas_h < 1 or canvas_w < 1:
        raise ValueError(f"source_size must be positive, got {source_size}")

    resized = F.interpolate(
        flow.float(), size=(target_h, target_w), mode="bilinear", align_corners=True
    )
    # A displacement is a coordinate quantity, not an image intensity.  Scale
    # using the endpoint convention used by align_corners=True.
    resized[:, 0] *= (target_w - 1) / max(canvas_w - 1, 1)
    resized[:, 1] *= (target_h - 1) / max(canvas_h - 1, 1)
    return resized


def apply_backward_flow_guidance(
    latents: torch.Tensor,
    flow: torch.Tensor,
    anchor_latents: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    sigma: torch.Tensor | float | None = None,
    scale: float = 1.0,
    source_size: SizeHW | None = None,
    noisy_anchor: bool = True,
    detach_anchor: bool = True,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Blend a flow-warped clean anchor into a denoising-step latent.

    ``anchor_latents`` must contain the original input pixels encoded by the
    VAE.  If ``noisy_anchor`` is enabled, the anchor is moved to the current
    flow-matching noise level with the same noise tensor as the pipeline.
    """
    if anchor_latents.ndim != 4 or anchor_latents.shape[0] not in (1, latents.shape[0]):
        raise ValueError(
            "anchor_latents must be [B,C,H,W] with a compatible batch, got "
            f"{tuple(anchor_latents.shape)} for {tuple(latents.shape)}"
        )
    anchor = backward_warp_latents(
        anchor_latents,
        flow,
        source_size=source_size,
        padding_mode=padding_mode,
    )
    if anchor.shape[0] == 1 and latents.shape[0] > 1:
        anchor = anchor.expand(latents.shape[0], -1, -1, -1)
    if noisy_anchor:
        if noise is None or sigma is None:
            raise ValueError("noisy flow guidance requires both noise and sigma")
        sigma = torch.as_tensor(sigma, device=anchor.device, dtype=anchor.dtype)
        noise = noise.to(device=anchor.device, dtype=anchor.dtype)
        anchor = (1.0 - sigma) * anchor + sigma * noise
    if detach_anchor:
        anchor = anchor.detach()
    strength = max(0.0, min(1.0, float(scale)))
    return (1.0 - strength) * latents + strength * anchor


def backward_warp_latents(
    latents: torch.Tensor,
    flow: torch.Tensor,
    *,
    source_size: SizeHW | None = None,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Sample ``latents`` at target-plus-flow coordinates.

    ``latents`` is both the source and target canvas after downsampling; the
    flow itself may have been predicted on the original RGB canvas.
    """
    if latents.ndim != 4:
        raise ValueError(f"latents must have shape [B,C,H,W], got {tuple(latents.shape)}")
    flow_latent = resize_backward_flow_to_latents(
        flow, latents.shape[-2:], source_size=source_size
    ).to(device=latents.device, dtype=torch.float32)
    if flow_latent.shape[0] not in (1, latents.shape[0]):
        raise ValueError(
            f"flow batch {flow_latent.shape[0]} does not match latent batch {latents.shape[0]}"
        )
    if flow_latent.shape[0] == 1 and latents.shape[0] != 1:
        flow_latent = flow_latent.expand(latents.shape[0], -1, -1, -1)

    height, width = latents.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=latents.device, dtype=torch.float32),
        torch.arange(width, device=latents.device, dtype=torch.float32),
        indexing="ij",
    )
    base = torch.stack((x, y), dim=-1).unsqueeze(0)
    base = base.expand(latents.shape[0], -1, -1, -1)
    coords = base + flow_latent.permute(0, 2, 3, 1)
    denom = latents.new_tensor((max(width - 1, 1), max(height - 1, 1)))
    grid = coords * (2.0 / denom) - 1.0
    warped = F.grid_sample(
        latents.float(), grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    )
    return warped.to(dtype=latents.dtype)
