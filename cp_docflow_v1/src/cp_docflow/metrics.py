"""Geometry-first metrics used by every CP-DocFlow stage."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _check_maps(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError(
            "prediction and target must share [B,2,H,W], got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )


def endpoint_error_map(prediction: Tensor, target: Tensor) -> Tensor:
    _check_maps(prediction, target)
    return torch.linalg.vector_norm(prediction - target, dim=1, keepdim=True)


def _values(value: Tensor, valid_mask: Tensor | None) -> Tensor:
    finite = torch.isfinite(value)
    if valid_mask is not None:
        if valid_mask.shape != value.shape:
            raise ValueError(
                f"valid_mask must match {tuple(value.shape)}, got {tuple(valid_mask.shape)}"
            )
        finite &= valid_mask.bool()
    return value[finite]


def endpoint_error(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    values = _values(endpoint_error_map(prediction, target), valid_mask)
    return values.mean() if values.numel() else prediction.new_tensor(float("nan"))


def endpoint_error_p95(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    values = _values(endpoint_error_map(prediction, target), valid_mask)
    return (
        torch.quantile(values.float(), 0.95)
        if values.numel()
        else prediction.new_tensor(float("nan"))
    )


def jacobian_determinant(backward_map: Tensor) -> Tensor:
    """Forward-difference Jacobian determinant on map cells."""

    if backward_map.ndim != 4 or backward_map.shape[1] != 2:
        raise ValueError("backward_map must be [B,2,H,W]")
    if min(backward_map.shape[-2:]) < 2:
        raise ValueError("backward_map needs at least a 2x2 target grid")
    derivative_x = backward_map[..., :, 1:] - backward_map[..., :, :-1]
    derivative_y = backward_map[..., 1:, :] - backward_map[..., :-1, :]
    dx = derivative_x[..., :-1, :]
    dy = derivative_y[..., :, :-1]
    return (
        dx[:, 0:1] * dy[:, 1:2]
        - dx[:, 1:2] * dy[:, 0:1]
    )


def cell_valid_mask(valid_mask: Tensor) -> Tensor:
    if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
        raise ValueError("valid_mask must be [B,1,H,W]")
    valid = valid_mask.bool()
    return (
        valid[..., :-1, :-1]
        & valid[..., :-1, 1:]
        & valid[..., 1:, :-1]
        & valid[..., 1:, 1:]
    )


def fold_rate(backward_map: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    determinant = jacobian_determinant(backward_map)
    cell_mask = None if valid_mask is None else cell_valid_mask(valid_mask)
    values = _values((determinant <= 0.0).float(), cell_mask)
    return values.mean() if values.numel() else backward_map.new_tensor(float("nan"))


def final_win_rate(
    coarse_map: Tensor,
    final_map: Tensor,
    target_map: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    coarse_error = endpoint_error_map(coarse_map, target_map)
    final_error = endpoint_error_map(final_map, target_map)
    values = _values((final_error < coarse_error).float(), valid_mask)
    return values.mean() if values.numel() else final_map.new_tensor(float("nan"))


def high_confidence_damage_rate(
    coarse_map: Tensor,
    final_map: Tensor,
    target_map: Tensor,
    valid_mask: Tensor | None = None,
    *,
    coarse_error_threshold: float = 1.0,
    damage_margin: float = 1.0,
) -> Tensor:
    """Rate from the project goal: good coarse pixels damaged by >1 px."""

    coarse_error = endpoint_error_map(coarse_map, target_map)
    final_error = endpoint_error_map(final_map, target_map)
    eligible = coarse_error < float(coarse_error_threshold)
    if valid_mask is not None:
        eligible &= valid_mask.bool()
    eligible &= torch.isfinite(coarse_error) & torch.isfinite(final_error)
    if not bool(eligible.any()):
        return final_map.new_tensor(float("nan"))
    damaged = (final_error - coarse_error) > float(damage_margin)
    return damaged[eligible].float().mean()


def confidence_brier_score(
    confidence: Tensor,
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    accurate_threshold: float = 1.0,
) -> Tensor:
    """Brier score for confidence as P(EPE < ``accurate_threshold``)."""

    error = endpoint_error_map(prediction, target)
    if confidence.shape != error.shape:
        raise ValueError("confidence must be [B,1,H,W] matching the map grid")
    accuracy = (error < float(accurate_threshold)).to(confidence.dtype)
    values = _values((confidence - accuracy).square(), valid_mask)
    return values.mean() if values.numel() else confidence.new_tensor(float("nan"))


def confidence_ece(
    confidence: Tensor,
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    accurate_threshold: float = 1.0,
    bins: int = 10,
) -> Tensor:
    """Expected calibration error for the same 1-pixel confidence event."""

    if bins < 2:
        raise ValueError("bins must be at least 2")
    error = endpoint_error_map(prediction, target)
    if confidence.shape != error.shape:
        raise ValueError("confidence must be [B,1,H,W] matching the map grid")
    mask = valid_mask.bool() & torch.isfinite(confidence) & torch.isfinite(error)
    if not bool(mask.any()):
        return confidence.new_tensor(float("nan"))
    probabilities = confidence[mask].float().clamp(0.0, 1.0)
    outcomes = (error[mask] < float(accurate_threshold)).float()
    total = probabilities.numel()
    result = probabilities.new_zeros(())
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    for index in range(bins):
        selected = (probabilities >= boundaries[index]) & (
            probabilities < boundaries[index + 1]
            if index + 1 < bins
            else probabilities <= boundaries[index + 1]
        )
        if bool(selected.any()):
            gap = torch.abs(probabilities[selected].mean() - outcomes[selected].mean())
            result = result + gap * (selected.sum() / total)
    return result


def coarse_metric_dict(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> dict[str, Tensor]:
    return {
        "epe": endpoint_error(prediction, target, valid_mask),
        "epe_p95": endpoint_error_p95(prediction, target, valid_mask),
        "fold_rate": fold_rate(prediction, valid_mask),
    }


def _masked_scalar_mean(value: Tensor, mask: Tensor) -> Tensor:
    selected = mask.bool() & torch.isfinite(value)
    return value[selected].mean() if bool(selected.any()) else value.new_tensor(float("nan"))


def geometry_quality_metrics(
    backward_map: Tensor,
    canonical_map: Tensor,
    valid_mask: Tensor,
    *,
    minimum_relative_scale: float = 0.25,
    maximum_relative_scale: float = 4.0,
) -> dict[str, Tensor]:
    """Geometry diagnostics required by the DocGrid-Flow v2 evaluator."""

    _check_maps(backward_map, canonical_map)
    if valid_mask.shape != backward_map[:, :1].shape:
        raise ValueError("valid_mask must be [B,1,H,W]")
    if min(backward_map.shape[-2:]) < 3:
        raise ValueError("geometry quality metrics need at least a 3x3 grid")
    cells = cell_valid_mask(valid_mask)
    dx = (backward_map[..., :, 1:] - backward_map[..., :, :-1])[..., :-1, :]
    dy = (backward_map[..., 1:, :] - backward_map[..., :-1, :])[..., :, :-1]
    canonical_dx = (
        canonical_map[..., :, 1:] - canonical_map[..., :, :-1]
    )[..., :-1, :]
    canonical_dy = (
        canonical_map[..., 1:, :] - canonical_map[..., :-1, :]
    )[..., :, :-1]
    scale_x = canonical_dx[:, 0:1].abs().clamp_min(1.0e-6)
    scale_y = canonical_dy[:, 1:2].abs().clamp_min(1.0e-6)
    relative_matrix = torch.stack(
        (
            torch.stack((dx[:, 0] / scale_x[:, 0], dy[:, 0] / scale_y[:, 0]), dim=-1),
            torch.stack((dx[:, 1] / scale_x[:, 0], dy[:, 1] / scale_y[:, 0]), dim=-1),
        ),
        dim=-2,
    )
    singular = torch.linalg.svdvals(relative_matrix.float())
    scale_anomaly = (
        (singular[..., 1] < float(minimum_relative_scale))
        | (singular[..., 0] > float(maximum_relative_scale))
    )[:, None]
    dx_norm = torch.linalg.vector_norm(dx, dim=1, keepdim=True).clamp_min(1.0e-6)
    dy_norm = torch.linalg.vector_norm(dy, dim=1, keepdim=True).clamp_min(1.0e-6)
    orthogonality = torch.abs((dx * dy).sum(dim=1, keepdim=True)) / (dx_norm * dy_norm)
    determinant = jacobian_determinant(backward_map)
    determinant_values = determinant[cells]

    displacement = backward_map - canonical_map
    dxx = displacement[..., 2:] - 2.0 * displacement[..., 1:-1] + displacement[..., :-2]
    dyy = displacement[..., 2:, :] - 2.0 * displacement[..., 1:-1, :] + displacement[..., :-2, :]
    valid_x = valid_mask[..., 2:] & valid_mask[..., 1:-1] & valid_mask[..., :-2]
    valid_y = valid_mask[..., 2:, :] & valid_mask[..., 1:-1, :] & valid_mask[..., :-2, :]
    bend_x = _masked_scalar_mean(torch.linalg.vector_norm(dxx, dim=1, keepdim=True), valid_x)
    bend_y = _masked_scalar_mean(torch.linalg.vector_norm(dyy, dim=1, keepdim=True), valid_y)
    bending = torch.nanmean(torch.stack((bend_x, bend_y)))

    border_errors: list[Tensor] = []
    for batch_index in range(backward_map.shape[0]):
        edge_pairs = (
            (backward_map[batch_index, :, 0, :].T, valid_mask[batch_index, 0, 0, :]),
            (backward_map[batch_index, :, -1, :].T, valid_mask[batch_index, 0, -1, :]),
            (backward_map[batch_index, :, :, 0].T, valid_mask[batch_index, 0, :, 0]),
            (backward_map[batch_index, :, :, -1].T, valid_mask[batch_index, 0, :, -1]),
        )
        for points, mask in edge_pairs:
            points = points[mask.bool() & torch.isfinite(points).all(dim=1)].float()
            if points.shape[0] < 2:
                continue
            centered = points - points.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
            normal = torch.stack((-direction[1], direction[0]))
            border_errors.append(torch.abs(centered @ normal).mean())
    page_border = (
        torch.stack(border_errors).mean()
        if border_errors
        else backward_map.new_tensor(float("nan"))
    )
    return {
        "jacobian_determinant_mean": (
            determinant_values.float().mean()
            if determinant_values.numel()
            else backward_map.new_tensor(float("nan"))
        ),
        "jacobian_determinant_p05": (
            torch.quantile(determinant_values.float(), 0.05)
            if determinant_values.numel()
            else backward_map.new_tensor(float("nan"))
        ),
        "jacobian_determinant_p95": (
            torch.quantile(determinant_values.float(), 0.95)
            if determinant_values.numel()
            else backward_map.new_tensor(float("nan"))
        ),
        "local_scale_anomaly_rate": _masked_scalar_mean(
            scale_anomaly.float(), cells
        ),
        "orthogonality_error": _masked_scalar_mean(orthogonality, cells),
        "bending_energy": bending,
        "page_border_line_error_px": page_border,
    }


def _image_gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    gray = image.float().mean(dim=1, keepdim=True)
    gradient_x = F.pad(torch.abs(gray[..., 1:] - gray[..., :-1]), (0, 1, 0, 0))
    gradient_y = F.pad(torch.abs(gray[..., 1:, :] - gray[..., :-1, :]), (0, 0, 0, 1))
    return gradient_x, gradient_y


def _finite_masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    selected = mask.bool().expand_as(value) & torch.isfinite(value)
    return value[selected].mean() if bool(selected.any()) else value.new_tensor(float("nan"))


def _connectivity_recall(
    target_active: Tensor,
    predicted_active: Tensor,
    *,
    dimension: int,
) -> Tensor:
    if dimension == -1:
        target_pairs = target_active[..., 1:] & target_active[..., :-1]
        predicted_pairs = predicted_active[..., 1:] & predicted_active[..., :-1]
    elif dimension == -2:
        target_pairs = target_active[..., 1:, :] & target_active[..., :-1, :]
        predicted_pairs = predicted_active[..., 1:, :] & predicted_active[..., :-1, :]
    else:
        raise ValueError("connectivity dimension must be -1 or -2")
    return (
        predicted_pairs[target_pairs].float().mean()
        if bool(target_pairs.any())
        else target_active.new_tensor(float("nan"), dtype=torch.float32)
    )


def image_quality_metrics(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    structure: Tensor | None = None,
) -> dict[str, Tensor]:
    """Auxiliary RGB/text/table preservation metrics at the evaluation canvas.

    Geometry remains the primary signal.  These metrics quantify how closely a
    single sampling of the warped source approaches the photographed/renderer
    target and whether target edge runs remain connected.
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target RGB must share [B,C,H,W]")
    if prediction.shape[1] not in {1, 3}:
        raise ValueError("image metrics support one or three channels")
    if valid_mask.shape != prediction[:, :1].shape:
        raise ValueError("valid_mask must be [B,1,H,W]")
    valid = valid_mask.bool()
    difference = prediction.float() - target.float()
    l1 = _finite_masked_mean(difference.abs(), valid)
    mse = _finite_masked_mean(difference.square(), valid)
    psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-12))

    # Local SSIM with a dependency-free 7x7 box window. Invalid pixels are
    # excluded from the final reduction rather than filled into the images.
    kernel = min(7, prediction.shape[-2], prediction.shape[-1])
    if kernel % 2 == 0:
        kernel -= 1
    kernel = max(kernel, 1)
    padding = kernel // 2
    pred = prediction.float()
    truth = target.float()
    mu_pred = F.avg_pool2d(pred, kernel, stride=1, padding=padding)
    mu_truth = F.avg_pool2d(truth, kernel, stride=1, padding=padding)
    variance_pred = (
        F.avg_pool2d(pred.square(), kernel, stride=1, padding=padding)
        - mu_pred.square()
    ).clamp_min(0.0)
    variance_truth = (
        F.avg_pool2d(truth.square(), kernel, stride=1, padding=padding)
        - mu_truth.square()
    ).clamp_min(0.0)
    covariance = (
        F.avg_pool2d(pred * truth, kernel, stride=1, padding=padding)
        - mu_pred * mu_truth
    )
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = (
        (2.0 * mu_pred * mu_truth + c1) * (2.0 * covariance + c2)
        / (
            (mu_pred.square() + mu_truth.square() + c1)
            * (variance_pred + variance_truth + c2)
        ).clamp_min(1.0e-12)
    )
    ssim = _finite_masked_mean(ssim_map, valid)

    predicted_x, predicted_y = _image_gradients(prediction)
    target_x, target_y = _image_gradients(target)
    predicted_edge = predicted_x + predicted_y
    target_edge = target_x + target_y
    edge_threshold = (
        target_edge.flatten(1).mean(dim=1)[:, None, None, None] * 2.0
    ).clamp_min(0.02)
    target_edges = valid & (target_edge >= edge_threshold)
    relative_edge_error = (predicted_edge - target_edge).abs() / target_edge.clamp_min(0.02)
    edge_preservation = (
        1.0 - _finite_masked_mean(relative_edge_error, target_edges)
    ).clamp(0.0, 1.0)

    if structure is None:
        horizontal = vertical = valid.float()
    else:
        if structure.ndim != 4 or structure.shape[0] != prediction.shape[0] or structure.shape[1] < 2:
            raise ValueError("structure must be [B,>=2,H,W]")
        if structure.shape[-2:] != prediction.shape[-2:]:
            structure = F.interpolate(
                structure.float(), prediction.shape[-2:], mode="bilinear", align_corners=False
            )
        horizontal, vertical = structure[:, 0:1], structure[:, 1:2]
    threshold_y = (
        target_y.flatten(1).mean(dim=1)[:, None, None, None] * 1.5
    ).clamp_min(0.01)
    threshold_x = (
        target_x.flatten(1).mean(dim=1)[:, None, None, None] * 1.5
    ).clamp_min(0.01)
    target_horizontal = valid & (horizontal >= 0.25) & (target_y >= threshold_y)
    target_vertical = valid & (vertical >= 0.25) & (target_x >= threshold_x)
    predicted_horizontal = valid & (horizontal >= 0.25) & (
        predicted_y >= 0.5 * target_y.clamp_min(threshold_y)
    )
    predicted_vertical = valid & (vertical >= 0.25) & (
        predicted_x >= 0.5 * target_x.clamp_min(threshold_x)
    )
    connectivity_values = torch.stack(
        (
            _connectivity_recall(
                target_horizontal, predicted_horizontal, dimension=-1
            ),
            _connectivity_recall(target_vertical, predicted_vertical, dimension=-2),
        )
    )
    finite_connectivity = connectivity_values[torch.isfinite(connectivity_values)]
    connectivity = (
        finite_connectivity.mean()
        if finite_connectivity.numel()
        else prediction.new_tensor(float("nan"))
    )
    return {
        "rgb_l1": l1,
        "rgb_psnr": psnr,
        "rgb_ssim": ssim,
        "character_edge_preservation": edge_preservation,
        "table_line_connectivity": connectivity,
    }
