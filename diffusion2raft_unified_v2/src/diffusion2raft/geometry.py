"""Geometry primitives for document rectification.

Coordinate contract
-------------------
Every flow tensor is a *backward displacement* in pixel units with shape
``[B, 2, H_target, W_target]``. For an output pixel ``p=(x, y)``, the source
sample is ``p + flow(p)``. Channel 0 is x and channel 1 is y.

Keeping this contract explicit is essential: torchvision RAFT predicts flow
from its first image to its second image, so the rectified guide must be passed
as image 1 and the warped/pre-rectified source as image 2.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def make_pixel_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return an ``[B, 2, H, W]`` grid ordered as x, y."""

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def backward_flow_to_map(flow: Tensor) -> Tensor:
    """Convert backward displacement to absolute source pixel coordinates."""

    _check_flow(flow)
    b, _, h, w = flow.shape
    return flow + make_pixel_grid(b, h, w, device=flow.device, dtype=flow.dtype)


def map_to_backward_flow(pixel_map: Tensor) -> Tensor:
    """Convert absolute source pixel coordinates to backward displacement."""

    _check_flow(pixel_map)
    b, _, h, w = pixel_map.shape
    return pixel_map - make_pixel_grid(b, h, w, device=pixel_map.device, dtype=pixel_map.dtype)


def _pixel_map_to_normalized_grid(
    pixel_map: Tensor,
    source_size: Sequence[int],
    *,
    align_corners: bool,
) -> Tensor:
    """Convert ``[B,2,H,W]`` absolute pixel coordinates for grid_sample."""

    source_h, source_w = int(source_size[0]), int(source_size[1])
    x, y = pixel_map[:, 0], pixel_map[:, 1]
    if align_corners:
        x_norm = 2.0 * x / max(source_w - 1, 1) - 1.0
        y_norm = 2.0 * y / max(source_h - 1, 1) - 1.0
    else:
        x_norm = (2.0 * x + 1.0) / max(source_w, 1) - 1.0
        y_norm = (2.0 * y + 1.0) / max(source_h, 1) - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def backward_warp(
    source: Tensor,
    backward_flow: Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    return_valid: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sample ``source`` into the target canvas using a backward flow.

    ``source`` may have a different spatial size from ``backward_flow``. The
    flow itself must already express coordinates in the source pixel system.
    """

    if source.ndim != 4:
        raise ValueError(f"source must be [B,C,H,W], got {tuple(source.shape)}")
    _check_flow(backward_flow)
    if source.shape[0] != backward_flow.shape[0]:
        raise ValueError("source and flow batch sizes differ")

    pixel_map = backward_flow_to_map(backward_flow)
    grid = _pixel_map_to_normalized_grid(
        pixel_map,
        source.shape[-2:],
        align_corners=align_corners,
    )
    warped = F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    if not return_valid:
        return warped
    valid = pixel_map_valid_mask(pixel_map, source.shape[-2:])
    return warped, valid


def pixel_map_valid_mask(pixel_map: Tensor, source_size: Sequence[int]) -> Tensor:
    """Return ``[B,1,H,W]`` mask for mappings inside the source canvas."""

    _check_flow(pixel_map)
    source_h, source_w = int(source_size[0]), int(source_size[1])
    x, y = pixel_map[:, 0:1], pixel_map[:, 1:2]
    return (x >= 0) & (x <= source_w - 1) & (y >= 0) & (y <= source_h - 1)


def flow_valid_mask(backward_flow: Tensor, source_size: Sequence[int]) -> Tensor:
    """Return the in-bounds mask for a backward displacement field."""

    return pixel_map_valid_mask(backward_flow_to_map(backward_flow), source_size)


def resize_backward_flow(
    flow: Tensor,
    target_size: Sequence[int],
    *,
    source_size_from: Sequence[int] | None = None,
    source_size_to: Sequence[int] | None = None,
) -> Tensor:
    """Resize a backward flow by transforming its *absolute coordinate map*.

    Multiplying interpolated displacements by width/height ratios is only
    correct when the source and target canvases undergo identical scaling.
    This function also handles anisotropic and different source/target sizes.

    Args:
        flow: ``[B,2,Ht0,Wt0]`` backward displacement.
        target_size: new target canvas ``(Ht1, Wt1)``.
        source_size_from: old source canvas. Defaults to ``(Ht0, Wt0)``.
        source_size_to: new source canvas. Defaults to ``target_size``.
    """

    _check_flow(flow)
    b, _, target_h0, target_w0 = flow.shape
    target_h1, target_w1 = int(target_size[0]), int(target_size[1])
    source_h0, source_w0 = (
        (target_h0, target_w0)
        if source_size_from is None
        else (int(source_size_from[0]), int(source_size_from[1]))
    )
    source_h1, source_w1 = (
        (target_h1, target_w1)
        if source_size_to is None
        else (int(source_size_to[0]), int(source_size_to[1]))
    )

    pixel_map = backward_flow_to_map(flow)
    pixel_map = F.interpolate(
        pixel_map,
        size=(target_h1, target_w1),
        mode="bilinear",
        align_corners=True,
    )
    scale_x = (source_w1 - 1) / max(source_w0 - 1, 1)
    scale_y = (source_h1 - 1) / max(source_h0 - 1, 1)
    pixel_map = pixel_map.clone()
    pixel_map[:, 0] *= scale_x
    pixel_map[:, 1] *= scale_y

    target_grid = make_pixel_grid(
        b,
        target_h1,
        target_w1,
        device=flow.device,
        dtype=flow.dtype,
    )
    return pixel_map - target_grid


def compose_backward_flows(base_flow: Tensor, residual_flow: Tensor) -> Tensor:
    """Compose target->intermediate residual with intermediate->source base.

    If ``P(x)=W(x+B(x))`` and RAFT predicts ``R`` from the final target to
    ``P``, then the correct final flow is ``R(x) + B(x+R(x))``. Directly adding
    ``B + R`` is wrong whenever ``B`` varies spatially.
    """

    _check_flow(base_flow)
    _check_flow(residual_flow)
    if base_flow.shape != residual_flow.shape:
        raise ValueError(
            "base and residual flows must share shape for composition, got "
            f"{tuple(base_flow.shape)} and {tuple(residual_flow.shape)}"
        )
    sampled_base = backward_warp(
        base_flow,
        residual_flow,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return residual_flow + sampled_base


def endpoint_error(prediction: Tensor, target: Tensor, valid: Tensor | None = None) -> Tensor:
    """Mean optical-flow endpoint error."""

    error = torch.linalg.vector_norm(prediction - target, dim=1, keepdim=True)
    if valid is None:
        return error.mean()
    valid = valid.bool()
    return error[valid].mean() if valid.any() else error.new_zeros(())


def _check_flow(flow: Tensor) -> None:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be [B,2,H,W], got {tuple(flow.shape)}")

