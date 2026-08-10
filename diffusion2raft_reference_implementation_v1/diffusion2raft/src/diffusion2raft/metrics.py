"""Evaluation metrics beyond PSNR/SSIM.

README_0714.md is explicit that the primary metrics must include, at minimum:
flow EPE, Jacobian fold rate, OCR CER, text-edge reprojection error, straight
line bending, and the invalid-sampling ratio. PSNR/SSIM alone hide the ripple
and folding failure modes this project is built to avoid.

Everything here is pure ``torch`` so it runs in the same process as training and
inference. ``ocr_character_error_rate`` is the only optional piece: it needs an
external OCR engine and ground-truth text, so it is guarded and returns ``None``
when its dependencies or inputs are unavailable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .geometry import backward_flow_to_map, endpoint_error, pixel_map_valid_mask
from .losses import jacobian_determinant


def _cell_valid(valid: Tensor) -> Tensor:
    """Reduce a per-pixel ``[B,1,H,W]`` mask to per-cell ``[B,H-1,W-1]``."""

    valid_bool = valid[:, 0].bool()
    return (
        valid_bool[:, :-1, :-1]
        & valid_bool[:, 1:, :-1]
        & valid_bool[:, :-1, 1:]
        & valid_bool[:, 1:, 1:]
    )


def fold_statistics(
    backward_flow: Tensor,
    valid: Tensor,
    *,
    min_jacobian: float = 0.05,
) -> dict[str, float]:
    """Jacobian fold rate plus the near-fold tail of the determinant."""

    determinant = jacobian_determinant(backward_flow)
    cell_valid = _cell_valid(valid)
    if not cell_valid.any():
        return {"fold_rate": float("nan"), "jacobian_p01": float("nan"), "near_fold_rate": float("nan")}
    valid_det = determinant[cell_valid].float()
    return {
        "fold_rate": float((valid_det <= 0).float().mean()),
        "near_fold_rate": float((valid_det < float(min_jacobian)).float().mean()),
        "jacobian_p01": float(torch.quantile(valid_det, 0.01)),
    }


def invalid_sampling_ratio(backward_flow: Tensor, source_size: tuple[int, int]) -> float:
    """Fraction of output pixels whose source coordinate leaves the source."""

    pixel_map = backward_flow_to_map(backward_flow)
    valid = pixel_map_valid_mask(pixel_map, source_size)
    return float(1.0 - valid.float().mean())


def line_bending(backward_flow: Tensor, valid: Tensor) -> float:
    """Mean second-order curvature of the coordinate map.

    A straight source line stays straight only if the target->source map has no
    spurious local curvature. This is the metric analogue of the second-order
    bending loss and is the quantitative stand-in for "直线弯曲度".
    """

    pixel_map = backward_flow_to_map(backward_flow)
    valid_bool = valid[:, 0:1].bool()
    terms: list[Tensor] = []
    if pixel_map.shape[-1] >= 3:
        dxx = (pixel_map[..., 2:] - 2.0 * pixel_map[..., 1:-1] + pixel_map[..., :-2]).abs()
        mask = valid_bool[..., 1:-1].expand_as(dxx[:, :1]).expand(-1, 2, -1, -1)
        terms.append(dxx[mask].mean() if mask.any() else dxx.new_zeros(()))
    if pixel_map.shape[-2] >= 3:
        dyy = (pixel_map[..., 2:, :] - 2.0 * pixel_map[..., 1:-1, :] + pixel_map[..., :-2, :]).abs()
        mask = valid_bool[..., 1:-1, :].expand_as(dyy[:, :1]).expand(-1, 2, -1, -1)
        terms.append(dyy[mask].mean() if mask.any() else dyy.new_zeros(()))
    if not terms:
        return float("nan")
    return float(torch.stack(terms).mean())


def _image_edges(image: Tensor) -> Tensor:
    """Sobel-free edge magnitude, ``[B,1,H,W]`` in [0, ~1]."""

    gray = image.mean(dim=1, keepdim=True)
    dx = torch.zeros_like(gray)
    dy = torch.zeros_like(gray)
    dx[..., :, 1:] = (gray[..., :, 1:] - gray[..., :, :-1]).abs()
    dy[..., 1:, :] = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
    return (dx + dy).clamp(0.0, 1.0)


def text_edge_reprojection_error(
    prediction: Tensor,
    target_flow: Tensor,
    target_image: Tensor,
    valid: Tensor,
    *,
    edge_quantile: float = 0.90,
) -> float:
    """Endpoint error restricted to high-contrast (text/line) target pixels.

    Overall EPE is dominated by flat page interior. Text fidelity depends on the
    flow being correct exactly where ink is, so this weights EPE toward strong
    edges of the *target* (ground-truth rectified) image.
    """

    edges = _image_edges(target_image)
    error = torch.linalg.vector_norm(prediction - target_flow, dim=1, keepdim=True)
    valid_bool = valid.bool()
    if not valid_bool.any():
        return float("nan")
    edge_valid = edges[valid_bool]
    if edge_valid.numel() == 0:
        return float("nan")
    threshold = torch.quantile(edge_valid.float(), float(edge_quantile))
    edge_mask = valid_bool & (edges >= threshold)
    if not edge_mask.any():
        edge_mask = valid_bool
    return float(error[edge_mask].mean())


def ocr_character_error_rate(
    rectified: Tensor,
    ground_truth_text: list[str] | None,
    *,
    languages: tuple[str, ...] = ("en",),
) -> float | None:
    """Optional OCR CER. Returns ``None`` if OCR deps or GT text are missing.

    Kept out of the default loop so training/eval never take a hard dependency
    on a heavy OCR stack. Enable it only when the manifest carries GT text.
    """

    if ground_truth_text is None:
        return None
    try:  # pragma: no cover - optional heavy dependency
        import numpy as np
        from PIL import Image  # noqa: F401
    except ImportError:
        return None
    try:  # pragma: no cover - optional heavy dependency
        import easyocr  # type: ignore
    except ImportError:
        return None

    reader = easyocr.Reader(list(languages), gpu=rectified.is_cuda)
    total_distance = 0
    total_length = 0
    images = (
        rectified.detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().permute(0, 2, 3, 1).numpy()
    )
    for image, reference in zip(images, ground_truth_text, strict=False):
        prediction = " ".join(reader.readtext(image, detail=0))
        total_distance += _levenshtein(prediction, reference)
        total_length += max(len(reference), 1)
    return total_distance / max(total_length, 1)


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


@torch.no_grad()
def compute_geometry_metrics(
    prediction: Tensor,
    target_flow: Tensor,
    valid: Tensor,
    *,
    source_size: tuple[int, int],
    target_image: Tensor | None = None,
    min_jacobian: float = 0.05,
) -> dict[str, float]:
    """Bundle the pure-geometry metrics from README's evaluation list."""

    metrics: dict[str, float] = {
        "epe": float(endpoint_error(prediction, target_flow, valid)),
        "invalid_ratio": invalid_sampling_ratio(prediction, source_size),
        "line_bending": line_bending(prediction, valid),
    }
    metrics.update(fold_statistics(prediction, valid, min_jacobian=min_jacobian))
    if target_image is not None:
        metrics["text_edge_epe"] = text_edge_reprojection_error(
            prediction, target_flow, target_image, valid
        )
    return metrics


def aggregate_metrics(per_sample: list[dict[str, float]]) -> dict[str, float]:
    """Mean over samples, ignoring NaNs per key."""

    if not per_sample:
        return {}
    keys: set[str] = set()
    for entry in per_sample:
        keys.update(entry)
    result: dict[str, float] = {}
    for key in sorted(keys):
        values = [
            entry[key]
            for entry in per_sample
            if key in entry and entry[key] == entry[key]  # drop NaN
        ]
        result[key] = float(sum(values) / len(values)) if values else float("nan")
    return result
