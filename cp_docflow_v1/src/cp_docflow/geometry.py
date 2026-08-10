"""Geometry primitives for the CP-DocFlow coordinate contract.

The public representation is always an absolute backward map in source-image
pixel coordinates with shape ``[B, 2, H_target, W_target]``. Channel 0 is x,
channel 1 is y, and ``align_corners=False`` is deliberately not configurable.
Keeping that choice global prevents train/eval/inference drift.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

ALIGN_CORNERS = False


def _size(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must be (height, width), got {value!r}")
    height, width = int(value[0]), int(value[1])
    if height < 1 or width < 1:
        raise ValueError(f"{name} must be positive, got {(height, width)}")
    return height, width


def _check_map(backward_map: Tensor) -> None:
    if backward_map.ndim != 4 or backward_map.shape[1] != 2:
        raise ValueError(
            "backward_map must be [B,2,H_target,W_target], got "
            f"{tuple(backward_map.shape)}"
        )


def make_pixel_grid(
    batch: int,
    size: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return pixel-centre coordinates ``[B,2,H,W]`` ordered x, y."""

    height, width = _size(size, "size")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(int(batch), -1, -1, -1)


def canonical_backward_map(
    batch: int,
    target_size: Sequence[int],
    source_size: Sequence[int] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Map target pixel centres to the same normalized source positions.

    For equal source and target canvases this is exactly the identity map. For
    different sizes it follows PyTorch's ``align_corners=False`` half-pixel
    convention.
    """

    target_h, target_w = _size(target_size, "target_size")
    source_h, source_w = _size(
        source_size if source_size is not None else target_size,
        "source_size",
    )
    grid = make_pixel_grid(
        batch,
        (target_h, target_w),
        device=device,
        dtype=dtype,
    ).clone()
    grid[:, 0] = (grid[:, 0] + 0.5) * (source_w / target_w) - 0.5
    grid[:, 1] = (grid[:, 1] + 0.5) * (source_h / target_h) - 0.5
    return grid


def canonical_backward_map_window(
    batch: int,
    target_size: Sequence[int],
    source_size: Sequence[int],
    target_canvas_size: Sequence[int],
    target_window: Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Canonical map for a crop that still references the complete canvas.

    ``target_window`` is ``[B,4]`` in ``(x0,y0,width,height)`` target-canvas
    pixel-boundary coordinates.  The returned map remains in complete-source
    pixel coordinates; no source-coordinate rebasing is performed.
    """

    target_h, target_w = _size(target_size, "target_size")
    source_h, source_w = _size(source_size, "source_size")
    canvas_h, canvas_w = _size(target_canvas_size, "target_canvas_size")
    if target_window.shape != (int(batch), 4):
        raise ValueError(
            f"target_window must be [B,4], got {tuple(target_window.shape)}"
        )
    window = target_window.to(device=device, dtype=dtype)
    x0, y0, window_w, window_h = window.unbind(dim=1)
    if bool((window_w <= 0).any()) or bool((window_h <= 0).any()):
        raise ValueError("target window width/height must be positive")
    if bool((x0 < 0).any()) or bool((y0 < 0).any()):
        raise ValueError("target window origin must be non-negative")
    tolerance = 1.0e-4
    if bool((x0 + window_w > canvas_w + tolerance).any()) or bool(
        (y0 + window_h > canvas_h + tolerance).any()
    ):
        raise ValueError("target window exceeds target_canvas_size")
    grid = make_pixel_grid(
        int(batch), (target_h, target_w), device=device, dtype=dtype
    ).clone()
    canvas_x = x0[:, None, None] + (
        grid[:, 0] + 0.5
    ) * (window_w[:, None, None] / target_w)
    canvas_y = y0[:, None, None] + (
        grid[:, 1] + 0.5
    ) * (window_h[:, None, None] / target_h)
    grid[:, 0] = canvas_x * (source_w / canvas_w) - 0.5
    grid[:, 1] = canvas_y * (source_h / canvas_h) - 0.5
    return grid


def rescale_source_pixel_map(
    backward_map: Tensor,
    source_size_from: Sequence[int],
    source_size_to: Sequence[int],
) -> Tensor:
    """Change only the source-pixel units of an absolute map."""

    _check_map(backward_map)
    source_h0, source_w0 = _size(source_size_from, "source_size_from")
    source_h1, source_w1 = _size(source_size_to, "source_size_to")
    result = backward_map.float().clone()
    result[:, 0] = (result[:, 0] + 0.5) * (source_w1 / source_w0) - 0.5
    result[:, 1] = (result[:, 1] + 0.5) * (source_h1 / source_h0) - 0.5
    return result


def pixel_map_to_normalized_grid(
    backward_map: Tensor,
    source_size: Sequence[int],
) -> Tensor:
    """Convert an absolute pixel map to a ``grid_sample`` grid."""

    _check_map(backward_map)
    source_h, source_w = _size(source_size, "source_size")
    x = (2.0 * backward_map[:, 0] + 1.0) / source_w - 1.0
    y = (2.0 * backward_map[:, 1] + 1.0) / source_h - 1.0
    return torch.stack((x, y), dim=-1)


def normalized_grid_to_pixel_map(
    normalized_grid: Tensor,
    source_size: Sequence[int],
) -> Tensor:
    """Inverse of :func:`pixel_map_to_normalized_grid`."""

    if normalized_grid.ndim != 4 or normalized_grid.shape[-1] != 2:
        raise ValueError(
            "normalized_grid must be [B,H,W,2], got "
            f"{tuple(normalized_grid.shape)}"
        )
    source_h, source_w = _size(source_size, "source_size")
    x = ((normalized_grid[..., 0] + 1.0) * source_w - 1.0) / 2.0
    y = ((normalized_grid[..., 1] + 1.0) * source_h - 1.0) / 2.0
    return torch.stack((x, y), dim=1)


def backward_map_valid_mask(
    backward_map: Tensor,
    source_size: Sequence[int],
) -> Tensor:
    """Return positions inside the ``align_corners=False`` sampling canvas."""

    _check_map(backward_map)
    source_h, source_w = _size(source_size, "source_size")
    finite = torch.isfinite(backward_map).all(dim=1, keepdim=True)
    x, y = backward_map[:, 0:1], backward_map[:, 1:2]
    return (
        finite
        & (x >= -0.5)
        & (x <= source_w - 0.5)
        & (y >= -0.5)
        & (y <= source_h - 0.5)
    )


def warp_with_backward_map(
    source: Tensor,
    backward_map: Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
    return_valid: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sample ``source`` exactly once using an absolute backward map."""

    if source.ndim != 4:
        raise ValueError(f"source must be [B,C,H,W], got {tuple(source.shape)}")
    _check_map(backward_map)
    if source.shape[0] != backward_map.shape[0]:
        raise ValueError("source and backward_map batch sizes differ")
    grid = pixel_map_to_normalized_grid(backward_map, source.shape[-2:])
    result = F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=ALIGN_CORNERS,
    )
    if not return_valid:
        return result
    return result, backward_map_valid_mask(backward_map, source.shape[-2:])


def resize_backward_map(
    backward_map: Tensor,
    target_size: Sequence[int],
    *,
    source_size_from: Sequence[int],
    source_size_to: Sequence[int],
) -> Tensor:
    """Resize both domains of an absolute backward map without coordinate drift.

    Interpolating absolute maps directly corrupts the half-pixel boundary. We
    instead interpolate displacement from the canonical source/target map,
    scale that displacement in source-pixel units, and add the new canonical
    map. Canonical/identity maps therefore remain exact at every size.
    """

    _check_map(backward_map)
    target_h0, target_w0 = backward_map.shape[-2:]
    target_h1, target_w1 = _size(target_size, "target_size")
    source_h0, source_w0 = _size(source_size_from, "source_size_from")
    source_h1, source_w1 = _size(source_size_to, "source_size_to")
    canonical0 = canonical_backward_map(
        backward_map.shape[0],
        (target_h0, target_w0),
        (source_h0, source_w0),
        device=backward_map.device,
        dtype=backward_map.dtype,
    )
    residual = backward_map - canonical0
    residual = F.interpolate(
        residual,
        size=(target_h1, target_w1),
        mode="bilinear",
        align_corners=ALIGN_CORNERS,
    )
    residual = residual.clone()
    residual[:, 0] *= source_w1 / source_w0
    residual[:, 1] *= source_h1 / source_h0
    canonical1 = canonical_backward_map(
        backward_map.shape[0],
        (target_h1, target_w1),
        (source_h1, source_w1),
        device=backward_map.device,
        dtype=backward_map.dtype,
    )
    return canonical1 + residual


def resize_backward_map_window(
    backward_map: Tensor,
    target_size: Sequence[int],
    *,
    source_size_from: Sequence[int],
    source_size_to: Sequence[int],
    target_canvas_size: Sequence[int],
    target_window: Tensor,
) -> Tensor:
    """Resize a target-window map while preserving its complete-canvas frame."""

    _check_map(backward_map)
    target_h1, target_w1 = _size(target_size, "target_size")
    source_h0, source_w0 = _size(source_size_from, "source_size_from")
    source_h1, source_w1 = _size(source_size_to, "source_size_to")
    canonical0 = canonical_backward_map_window(
        backward_map.shape[0],
        backward_map.shape[-2:],
        (source_h0, source_w0),
        target_canvas_size,
        target_window,
        device=backward_map.device,
        dtype=backward_map.dtype,
    )
    residual = F.interpolate(
        backward_map - canonical0,
        size=(target_h1, target_w1),
        mode="bilinear",
        align_corners=ALIGN_CORNERS,
    )
    residual = residual.clone()
    residual[:, 0] *= source_w1 / source_w0
    residual[:, 1] *= source_h1 / source_h0
    canonical1 = canonical_backward_map_window(
        backward_map.shape[0],
        (target_h1, target_w1),
        (source_h1, source_w1),
        target_canvas_size,
        target_window,
        device=backward_map.device,
        dtype=backward_map.dtype,
    )
    return canonical1 + residual


def resize_backward_map_window_with_mask(
    backward_map: Tensor,
    valid_mask: Tensor,
    target_size: Sequence[int],
    *,
    source_size_from: Sequence[int],
    source_size_to: Sequence[int],
    target_canvas_size: Sequence[int],
    target_window: Tensor,
    full_support_tolerance: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    """Mask-aware resize for an unre-based target-window map."""

    _check_map(backward_map)
    if valid_mask.shape != (backward_map.shape[0], 1, *backward_map.shape[-2:]):
        raise ValueError("valid_mask must be [B,1,H_target,W_target]")
    if not 0.0 <= float(full_support_tolerance) < 1.0:
        raise ValueError("full_support_tolerance must be in [0,1)")
    target_h1, target_w1 = _size(target_size, "target_size")
    source_h0, source_w0 = _size(source_size_from, "source_size_from")
    source_h1, source_w1 = _size(source_size_to, "source_size_to")
    canonical0 = canonical_backward_map_window(
        backward_map.shape[0],
        backward_map.shape[-2:],
        (source_h0, source_w0),
        target_canvas_size,
        target_window,
        device=backward_map.device,
        dtype=backward_map.dtype,
    )
    valid = valid_mask.bool() & torch.isfinite(backward_map).all(dim=1, keepdim=True)
    residual = torch.where(valid.expand_as(backward_map), backward_map - canonical0, 0.0)
    weights = F.interpolate(
        valid.to(backward_map.dtype),
        size=(target_h1, target_w1),
        mode="bilinear",
        align_corners=ALIGN_CORNERS,
    )
    weighted = F.interpolate(
        residual,
        size=(target_h1, target_w1),
        mode="bilinear",
        align_corners=ALIGN_CORNERS,
    )
    resized_valid = weights >= (1.0 - float(full_support_tolerance))
    normalized = weighted / weights.clamp_min(1.0e-8)
    normalized = normalized.clone()
    normalized[:, 0] *= source_w1 / source_w0
    normalized[:, 1] *= source_h1 / source_h0
    canonical1 = canonical_backward_map_window(
        backward_map.shape[0],
        (target_h1, target_w1),
        (source_h1, source_w1),
        target_canvas_size,
        target_window,
        device=backward_map.device,
        dtype=backward_map.dtype,
    )
    result = canonical1 + torch.where(
        resized_valid.expand_as(normalized), normalized, 0.0
    )
    return result, resized_valid


def resize_backward_map_with_mask(
    backward_map: Tensor,
    valid_mask: Tensor,
    target_size: Sequence[int],
    *,
    source_size_from: Sequence[int],
    source_size_to: Sequence[int],
    full_support_tolerance: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    """Mask-aware counterpart of :func:`resize_backward_map`.

    Invalid map values must never bleed into a pixel that remains supervised.
    Residuals are therefore interpolated with normalized valid weights, while
    the returned mask conservatively requires the entire bilinear footprint to
    be valid. Pixels touching an invalid neighbor receive a finite canonical
    map but remain invalid for losses and metrics.
    """

    _check_map(backward_map)
    if valid_mask.shape != (backward_map.shape[0], 1, *backward_map.shape[-2:]):
        raise ValueError("valid_mask must be [B,1,H_target,W_target]")
    if not 0.0 <= float(full_support_tolerance) < 1.0:
        raise ValueError("full_support_tolerance must be in [0,1)")
    target_h0, target_w0 = backward_map.shape[-2:]
    target_h1, target_w1 = _size(target_size, "target_size")
    source_h0, source_w0 = _size(source_size_from, "source_size_from")
    source_h1, source_w1 = _size(source_size_to, "source_size_to")
    canonical0 = canonical_backward_map(
        backward_map.shape[0],
        (target_h0, target_w0),
        (source_h0, source_w0),
        device=backward_map.device,
        dtype=backward_map.dtype,
    )
    valid = valid_mask.bool() & torch.isfinite(backward_map).all(dim=1, keepdim=True)
    residual = torch.where(valid.expand_as(backward_map), backward_map - canonical0, 0.0)
    weights = F.interpolate(
        valid.to(backward_map.dtype),
        size=(target_h1, target_w1),
        mode="bilinear",
        align_corners=ALIGN_CORNERS,
    )
    weighted_residual = F.interpolate(
        residual,
        size=(target_h1, target_w1),
        mode="bilinear",
        align_corners=ALIGN_CORNERS,
    )
    normalized_residual = weighted_residual / weights.clamp_min(1.0e-8)
    resized_valid = weights >= (1.0 - float(full_support_tolerance))
    normalized_residual = torch.where(
        resized_valid.expand_as(normalized_residual),
        normalized_residual,
        torch.zeros_like(normalized_residual),
    )
    normalized_residual = normalized_residual.clone()
    normalized_residual[:, 0] *= source_w1 / source_w0
    normalized_residual[:, 1] *= source_h1 / source_h0
    canonical1 = canonical_backward_map(
        backward_map.shape[0],
        (target_h1, target_w1),
        (source_h1, source_w1),
        device=backward_map.device,
        dtype=backward_map.dtype,
    )
    return canonical1 + normalized_residual, resized_valid


def crop_backward_map(
    backward_map: Tensor,
    *,
    target_box: Sequence[int],
    source_offset: Sequence[int] = (0, 0),
) -> Tensor:
    """Crop the target domain and optionally rebase a cropped source domain.

    ``target_box`` is ``(left, top, right, bottom)``. ``source_offset`` is the
    ``(left, top)`` removed from the source image.
    """

    _check_map(backward_map)
    if len(target_box) != 4 or len(source_offset) != 2:
        raise ValueError("target_box must have 4 and source_offset 2 entries")
    left, top, right, bottom = (int(value) for value in target_box)
    height, width = backward_map.shape[-2:]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"invalid target_box={tuple(target_box)} for {(height, width)}")
    source_left, source_top = float(source_offset[0]), float(source_offset[1])
    result = backward_map[..., top:bottom, left:right].clone()
    result[:, 0] -= source_left
    result[:, 1] -= source_top
    return result


def pad_backward_map(
    backward_map: Tensor,
    valid_mask: Tensor,
    *,
    target_padding: Sequence[int],
    source_padding: Sequence[int] = (0, 0, 0, 0),
) -> tuple[Tensor, Tensor]:
    """Pad target/source domains while making target padding explicitly invalid.

    Padding order is ``(left, right, top, bottom)`` as in ``torch.nn.functional``.
    Existing source coordinates are shifted by source left/top padding.
    """

    _check_map(backward_map)
    if valid_mask.shape != (backward_map.shape[0], 1, *backward_map.shape[-2:]):
        raise ValueError("valid_mask must be [B,1,H_target,W_target]")
    if len(target_padding) != 4 or len(source_padding) != 4:
        raise ValueError("padding must be (left,right,top,bottom)")
    target_left, target_right, target_top, target_bottom = (
        int(value) for value in target_padding
    )
    source_left, source_right, source_top, source_bottom = (
        int(value) for value in source_padding
    )
    if min(*target_padding, *source_padding) < 0:
        raise ValueError("padding values must be non-negative")
    del source_right, source_bottom
    batch, _, old_h, old_w = backward_map.shape
    new_h = old_h + target_top + target_bottom
    new_w = old_w + target_left + target_right
    # Values outside the pasted region are irrelevant because validity is
    # false, but zeros keep serialized maps finite and easy to inspect.
    result = backward_map.new_zeros((batch, 2, new_h, new_w))
    shifted = backward_map.clone()
    shifted[:, 0] += source_left
    shifted[:, 1] += source_top
    result[
        ...,
        target_top : target_top + old_h,
        target_left : target_left + old_w,
    ] = shifted
    padded_valid = F.pad(
        valid_mask.bool(),
        (target_left, target_right, target_top, target_bottom),
        value=False,
    )
    return result, padded_valid


def flip_backward_map(
    backward_map: Tensor,
    source_size: Sequence[int],
    *,
    horizontal: bool = False,
    vertical: bool = False,
) -> Tensor:
    """Apply synchronized source-and-target flips to a backward map."""

    _check_map(backward_map)
    source_h, source_w = _size(source_size, "source_size")
    result = backward_map
    if horizontal:
        result = torch.flip(result, dims=(-1,)).clone()
        result[:, 0] = source_w - 1.0 - result[:, 0]
    if vertical:
        result = torch.flip(result, dims=(-2,)).clone()
        result[:, 1] = source_h - 1.0 - result[:, 1]
    return result
