"""Losses that supervise geometry and suppress ripple/folding artifacts."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .geometry import (
    backward_warp,
    endpoint_error,
    flow_valid_mask,
    residual_from_composed_flow,
    resize_backward_flow,
)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.bool()
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    return value[mask].mean() if mask.any() else value.new_zeros(())


def _masked_weighted_mean(value: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
    """Mean over valid pixels with a soft spatial weight."""

    mask_float = mask.to(dtype=value.dtype)
    weight_float = weight.to(dtype=value.dtype)
    if mask_float.shape != value.shape:
        mask_float = mask_float.expand_as(value)
    if weight_float.shape != value.shape:
        weight_float = weight_float.expand_as(value)
    combined = mask_float * weight_float.clamp_min(0.0)
    numerator = (value * combined).sum()
    denominator = combined.sum()
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-6),
        numerator * 0.0,
    )


def _masked_quantile(value: Tensor, mask: Tensor, quantile: float) -> Tensor:
    mask = mask.bool()
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    selected = value[mask]
    return (
        torch.quantile(selected.float(), float(quantile))
        if selected.numel()
        else value.new_zeros(())
    )


def robust_l1(value: Tensor, epsilon: float = 1e-3) -> Tensor:
    return torch.sqrt(value.square() + epsilon * epsilon)


def sequence_flow_loss(
    predictions: list[Tensor],
    target: Tensor,
    valid: Tensor,
    *,
    gamma: float = 0.8,
) -> Tensor:
    if not predictions:
        return target.new_zeros(())
    total = target.new_zeros(())
    count = len(predictions)
    for index, prediction in enumerate(predictions):
        weight = gamma ** (count - index - 1)
        total = total + weight * _masked_mean(robust_l1(prediction - target), valid)
    return total


def weighted_sequence_flow_loss(
    predictions: list[Tensor],
    target: Tensor,
    valid: Tensor,
    spatial_weight: Tensor,
    *,
    gamma: float = 0.8,
) -> Tensor:
    """Sequence EPE concentrated on target text edges and long rules."""

    if not predictions:
        return target.new_zeros(())
    total = target.new_zeros(())
    count = len(predictions)
    for index, prediction in enumerate(predictions):
        iteration_weight = gamma ** (count - index - 1)
        error = torch.linalg.vector_norm(prediction - target, dim=1, keepdim=True)
        total = total + iteration_weight * _masked_weighted_mean(
            robust_l1(error), valid, spatial_weight
        )
    return total


def second_order_bending_loss(flow: Tensor, valid: Tensor) -> Tensor:
    """Penalize local curvature in a flow or prediction-error field."""

    terms: list[Tensor] = []
    if flow.shape[-1] >= 3:
        dxx = flow[..., 2:] - 2.0 * flow[..., 1:-1] + flow[..., :-2]
        terms.append(_masked_mean(robust_l1(dxx), valid[..., 1:-1]))
    if flow.shape[-2] >= 3:
        dyy = flow[..., 2:, :] - 2.0 * flow[..., 1:-1, :] + flow[..., :-2, :]
        terms.append(_masked_mean(robust_l1(dyy), valid[..., 1:-1, :]))
    return torch.stack(terms).mean() if terms else flow.new_zeros(())


def jacobian_determinant(backward_flow: Tensor) -> Tensor:
    """Jacobian determinant of the absolute target->source coordinate map."""

    u, v = backward_flow[:, 0], backward_flow[:, 1]
    du_dx = u[:, :-1, 1:] - u[:, :-1, :-1]
    du_dy = u[:, 1:, :-1] - u[:, :-1, :-1]
    dv_dx = v[:, :-1, 1:] - v[:, :-1, :-1]
    dv_dy = v[:, 1:, :-1] - v[:, :-1, :-1]
    return (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx


def anti_fold_loss(flow: Tensor, valid: Tensor, min_jacobian: float = 0.05) -> tuple[Tensor, Tensor]:
    determinant = jacobian_determinant(flow)
    valid_bool = valid[:, 0].bool()
    cell_valid = (
        valid_bool[:, :-1, :-1]
        & valid_bool[:, 1:, :-1]
        & valid_bool[:, :-1, 1:]
        & valid_bool[:, 1:, 1:]
    )
    penalty = torch.relu(float(min_jacobian) - determinant)
    loss = _masked_mean(penalty, cell_valid)
    fold_rate = _masked_mean((determinant <= 0).to(flow.dtype), cell_valid)
    return loss, fold_rate


def _image_gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    return image[..., 1:] - image[..., :-1], image[..., 1:, :] - image[..., :-1, :]


def reconstruction_losses(
    warped: Tensor,
    target: Tensor,
    flow: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    with torch.autocast(device_type=warped.device.type, enabled=False):
        rectified = backward_warp(warped.float(), flow.float(), padding_mode="border")
        target_float = target.float()
        photo = _masked_mean(robust_l1(rectified - target_float), valid)
        rect_dx, rect_dy = _image_gradients(rectified)
        target_dx, target_dy = _image_gradients(target_float)
        grad_x = _masked_mean(robust_l1(rect_dx - target_dx), valid[..., 1:])
        grad_y = _masked_mean(robust_l1(rect_dy - target_dy), valid[..., 1:, :])
    return photo, 0.5 * (grad_x + grad_y), rectified


def _odd_kernel(requested: int, extent: int) -> int:
    kernel = max(1, min(int(requested), int(extent)))
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    return kernel


def target_structure_maps(
    target: Tensor,
    *,
    edge_threshold: float = 0.08,
    edge_temperature: float = 0.04,
    line_kernel: int = 15,
    line_threshold: float = 0.20,
    line_temperature: float = 0.05,
) -> dict[str, Tensor]:
    """Detect soft text/edge and long horizontal/vertical rule masks."""

    if target.ndim != 4 or target.shape[1] not in {1, 3}:
        raise ValueError(f"target must be [B,1|3,H,W], got {tuple(target.shape)}")
    image = target.float()
    if image.shape[1] == 3:
        luminance = (
            0.2989 * image[:, 0:1]
            + 0.5870 * image[:, 1:2]
            + 0.1140 * image[:, 2:3]
        )
    else:
        luminance = image
    sobel_x = luminance.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    padded = F.pad(luminance, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded, sobel_x).abs()
    grad_y = F.conv2d(padded, sobel_y).abs()
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    edge = torch.sigmoid(
        (magnitude - float(edge_threshold)) / max(float(edge_temperature), 1e-4)
    )
    denominator = grad_x + grad_y + 1e-6
    horizontal_seed = edge * (grad_y / denominator)
    vertical_seed = edge * (grad_x / denominator)
    kernel_x = _odd_kernel(line_kernel, target.shape[-1])
    kernel_y = _odd_kernel(line_kernel, target.shape[-2])
    horizontal_response = F.avg_pool2d(
        horizontal_seed, (1, kernel_x), stride=1, padding=(0, kernel_x // 2)
    )
    vertical_response = F.avg_pool2d(
        vertical_seed, (kernel_y, 1), stride=1, padding=(kernel_y // 2, 0)
    )
    temperature = max(float(line_temperature), 1e-4)
    horizontal = horizontal_seed * torch.sigmoid(
        (horizontal_response - float(line_threshold)) / temperature
    )
    vertical = vertical_seed * torch.sigmoid(
        (vertical_response - float(line_threshold)) / temperature
    )
    return {
        "edge": edge,
        "horizontal": horizontal,
        "vertical": vertical,
        "line": torch.maximum(horizontal, vertical),
    }


def first_order_flow_matching_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    weight: Tensor,
) -> Tensor:
    """Match GT first derivatives instead of suppressing legitimate warping."""

    error = prediction - target
    terms: list[Tensor] = []
    if error.shape[-1] >= 2:
        difference = error[..., 1:] - error[..., :-1]
        pair_valid = valid[..., 1:] & valid[..., :-1]
        pair_weight = 0.5 * (weight[..., 1:] + weight[..., :-1])
        value = torch.linalg.vector_norm(difference, dim=1, keepdim=True)
        terms.append(_masked_weighted_mean(robust_l1(value), pair_valid, pair_weight))
    if error.shape[-2] >= 2:
        difference = error[..., 1:, :] - error[..., :-1, :]
        pair_valid = valid[..., 1:, :] & valid[..., :-1, :]
        pair_weight = 0.5 * (weight[..., 1:, :] + weight[..., :-1, :])
        value = torch.linalg.vector_norm(difference, dim=1, keepdim=True)
        terms.append(_masked_weighted_mean(robust_l1(value), pair_valid, pair_weight))
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def line_straightness_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    horizontal_weight: Tensor,
    vertical_weight: Tensor,
    *,
    robust: bool = True,
) -> Tensor:
    """Penalize changes of normal flow error along each line's tangent."""

    error = prediction - target
    terms: list[Tensor] = []
    if error.shape[-1] >= 2:
        normal_change = error[:, 1:2, :, 1:] - error[:, 1:2, :, :-1]
        pair_valid = valid[..., 1:] & valid[..., :-1]
        pair_weight = 0.5 * (
            horizontal_weight[..., 1:] + horizontal_weight[..., :-1]
        )
        value = robust_l1(normal_change) if robust else normal_change.abs()
        terms.append(_masked_weighted_mean(value, pair_valid, pair_weight))
    if error.shape[-2] >= 2:
        normal_change = error[:, 0:1, 1:, :] - error[:, 0:1, :-1, :]
        pair_valid = valid[..., 1:, :] & valid[..., :-1, :]
        pair_weight = 0.5 * (
            vertical_weight[..., 1:, :] + vertical_weight[..., :-1, :]
        )
        value = robust_l1(normal_change) if robust else normal_change.abs()
        terms.append(_masked_weighted_mean(value, pair_valid, pair_weight))
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def local_match_supervision(
    logits: Tensor,
    residual_target: Tensor,
    valid: Tensor,
    confidence: Tensor,
    *,
    radius: int,
    gate_temperature_px: float = 1.0,
) -> dict[str, Tensor]:
    """Supervise Qwen matching and calibrate its reliability gate.

    The cost volume lives at the feature resolution.  A bilinear four-neighbour
    target avoids rounding a sub-pixel residual to a single offset.  The gate's
    oracle target is whether Qwen's argmax is more accurate than the safe
    zero-residual CNN fallback at that pixel.
    """

    if logits.ndim != 4:
        raise ValueError(f"match logits must be [B,K,H,W], got {tuple(logits.shape)}")
    batch, classes, height, width = logits.shape
    kernel = 2 * int(radius) + 1
    if classes != kernel * kernel:
        raise ValueError(
            f"match logits have {classes} classes, expected {(kernel * kernel)} "
            f"for radius={radius}"
        )
    full_size = tuple(int(value) for value in residual_target.shape[-2:])
    residual_low = resize_backward_flow(
        residual_target,
        (height, width),
        source_size_from=full_size,
        source_size_to=(height, width),
    )
    valid_low = F.interpolate(valid.float(), size=(height, width), mode="nearest") > 0.5

    x = residual_low[:, 0]
    y = residual_low[:, 1]
    inside = (
        (x >= -radius)
        & (x <= radius)
        & (y >= -radius)
        & (y <= radius)
    ).unsqueeze(1)
    match_valid = valid_low & inside

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    wx = x - x0
    wy = y - y0
    log_prob = F.log_softmax(logits.float(), dim=1)

    def gather(dx: Tensor, dy: Tensor) -> Tensor:
        index_x = (dx + radius).long().clamp(0, kernel - 1)
        index_y = (dy + radius).long().clamp(0, kernel - 1)
        index = (index_y * kernel + index_x).unsqueeze(1)
        return torch.gather(log_prob, 1, index).squeeze(1)

    soft_ce = -(
        (1.0 - wx) * (1.0 - wy) * gather(x0, y0)
        + wx * (1.0 - wy) * gather(x1, y0)
        + (1.0 - wx) * wy * gather(x0, y1)
        + wx * wy * gather(x1, y1)
    ).unsqueeze(1)
    match_loss = _masked_mean(soft_ce, match_valid)

    prediction = logits.detach().argmax(dim=1)
    pred_x = (prediction.remainder(kernel) - radius).to(x.dtype)
    pred_y = (prediction.div(kernel, rounding_mode="floor") - radius).to(y.dtype)
    scale_x = (full_size[1] - 1) / max(width - 1, 1)
    scale_y = (full_size[0] - 1) / max(height - 1, 1)
    match_error = torch.sqrt(
        ((pred_x - x) * scale_x).square()
        + ((pred_y - y) * scale_y).square()
    ).unsqueeze(1)
    fallback_error = torch.sqrt(
        (x * scale_x).square() + (y * scale_y).square()
    ).unsqueeze(1)

    # Confidence answers a concrete question: should this pixel use the Qwen
    # match instead of the zero-residual source-faithful fallback?
    oracle_confidence = torch.sigmoid(
        (fallback_error - match_error) / max(float(gate_temperature_px), 1e-4)
    )
    oracle_confidence = torch.where(inside, oracle_confidence, torch.zeros_like(oracle_confidence))
    if confidence.shape[-2:] != (height, width):
        confidence = F.interpolate(
            confidence, size=(height, width), mode="bilinear", align_corners=True
        )
    confidence_float = confidence.float().clamp(1e-5, 1.0 - 1e-5)
    # binary_cross_entropy is unsafe under (bf16) autocast; run it in FP32 with
    # autocast disabled. Inputs are already clamped, so FP32 BCE is stable.
    with torch.autocast(device_type=confidence_float.device.type, enabled=False):
        confidence_map = F.binary_cross_entropy(
            confidence_float,
            oracle_confidence.float(),
            reduction="none",
        )
    confidence_loss = _masked_mean(confidence_map, valid_low)

    probability = log_prob.exp()
    entropy = -(
        probability * log_prob
    ).sum(dim=1, keepdim=True) / max(math.log(max(classes, 2)), 1e-6)
    low_error = torch.sqrt((pred_x - x).square() + (pred_y - y).square()).unsqueeze(1)
    return {
        "loss": match_loss,
        "confidence_loss": confidence_loss,
        "epe": _masked_mean(match_error, match_valid),
        "acc1": _masked_mean((low_error <= 1.0).to(logits.dtype), match_valid),
        "entropy": _masked_mean(entropy, match_valid),
        "coverage": _masked_mean(inside.to(logits.dtype), valid_low),
        "advantage": _masked_mean(fallback_error - match_error, match_valid),
        "win_rate": _masked_mean((match_error < fallback_error).to(logits.dtype), match_valid),
        "gate_target": _masked_mean(oracle_confidence, valid_low),
        "gate_mae": _masked_mean(
            torch.abs(confidence_float - oracle_confidence), valid_low
        ),
    }


class RectificationLoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        target_flow = batch["flow"]
        valid = batch["valid"].bool()
        finite = torch.isfinite(target_flow).all(dim=1, keepdim=True)
        magnitude = torch.linalg.vector_norm(target_flow, dim=1, keepdim=True)
        valid = valid & finite & (magnitude < float(self.config.get("max_valid_flow", 1000.0)))

        with torch.no_grad(), torch.autocast(
            device_type=target_flow.device.type, enabled=False
        ):
            structures = target_structure_maps(
                batch["target"].float(),
                edge_threshold=float(self.config.get("structure_edge_threshold", 0.08)),
                edge_temperature=float(self.config.get("structure_edge_temperature", 0.04)),
                line_kernel=int(self.config.get("structure_line_kernel", 15)),
                line_threshold=float(self.config.get("structure_line_threshold", 0.20)),
                line_temperature=float(self.config.get("structure_line_temperature", 0.05)),
            )
        structure_weight = (structures["edge"] + structures["line"]).clamp_max(2.0)

        final_flow = outputs["final_flow"]
        # During teacher warmup the refiner keeps predicting the full residual,
        # while only a scheduled fraction is allowed to affect final geometry.
        # Falling back to the historical key preserves legacy callers/tests.
        raw_residuals = outputs.get("raw_residuals", outputs["residuals"])
        applied_residuals = outputs["residuals"]
        flow_loss = sequence_flow_loss(
            outputs["flows"],
            target_flow,
            valid,
            gamma=float(self.config.get("sequence_gamma", 0.8)),
        )
        structure_flow_loss = weighted_sequence_flow_loss(
            outputs["flows"],
            target_flow,
            valid,
            structure_weight,
            gamma=float(self.config.get("sequence_gamma", 0.8)),
        )
        sequence_gamma = float(self.config.get("sequence_gamma", 0.8))
        sequence_weight_sum = sum(
            sequence_gamma ** (len(outputs["flows"]) - index - 1)
            for index in range(len(outputs["flows"]))
        )
        flow_per_iteration = flow_loss.detach() / max(sequence_weight_sum, 1e-6)
        final_component_l1 = _masked_mean(
            robust_l1(final_flow.detach() - target_flow), valid
        ).detach()
        prior_loss = _masked_mean(
            robust_l1(outputs["prior_flow"] - target_flow),
            valid,
        )
        photo, gradient, rectified = reconstruction_losses(
            batch["warped"], batch["target"], final_flow, valid
        )
        line_photo_map = robust_l1(
            rectified - batch["target"].float()
        ).mean(dim=1, keepdim=True)
        line_reconstruction = _masked_weighted_mean(
            line_photo_map, valid, structures["line"]
        )
        flow_gradient = first_order_flow_matching_loss(
            final_flow, target_flow, valid, structure_weight
        )
        curvature = second_order_bending_loss(final_flow - target_flow, valid)
        line_straightness = line_straightness_loss(
            final_flow,
            target_flow,
            valid,
            structures["horizontal"],
            structures["vertical"],
        )
        bending_field = final_flow
        if outputs.get("stage") == "unified" and raw_residuals:
            bending_field = raw_residuals[-1]
        bending = second_order_bending_loss(bending_field, valid)
        anti_fold, fold_rate = anti_fold_loss(
            final_flow,
            valid,
            min_jacobian=float(self.config.get("min_jacobian", 0.05)),
        )
        with torch.no_grad():
            determinant = jacobian_determinant(final_flow.detach())
            valid_bool = valid[:, 0].bool()
            valid_cells = (
                valid_bool[:, :-1, :-1]
                & valid_bool[:, 1:, :-1]
                & valid_bool[:, :-1, 1:]
                & valid_bool[:, 1:, 1:]
            )
            jacobian_p01 = _masked_quantile(
                determinant, valid_cells, 0.01
            ).detach()
        applied_residual_p95 = final_flow.new_zeros(())
        if raw_residuals:
            residual_norm = torch.linalg.vector_norm(
                raw_residuals[-1], dim=1, keepdim=True
            )
            residual = _masked_mean(residual_norm, valid)
            residual_p95 = _masked_quantile(residual_norm.detach(), valid, 0.95)
            applied_residual_norm = torch.linalg.vector_norm(
                applied_residuals[-1], dim=1, keepdim=True
            )
            applied_residual_p95 = _masked_quantile(
                applied_residual_norm.detach(), valid, 0.95
            )
        else:
            residual = final_flow.new_zeros(())
            residual_p95 = final_flow.new_zeros(())

        # Supervise the geometrically correct residual R satisfying
        # F_gt(x) = R(x) + B(x + R(x)).  Directly using F_gt - B is biased for
        # curved/projective priors and makes the refiner learn through a harder
        # indirect composition gradient only.
        residual_flow_loss = final_flow.new_zeros(())
        residual_epe = final_flow.new_zeros(())
        residual_target_valid_rate = final_flow.new_zeros(())
        residual_target_consistency = final_flow.new_zeros(())
        residual_target = torch.zeros_like(final_flow)
        residual_valid = torch.zeros_like(valid)
        if raw_residuals:
            with torch.no_grad(), torch.autocast(
                device_type=final_flow.device.type, enabled=False
            ):
                safe_target = torch.where(
                    finite.expand_as(target_flow),
                    target_flow.float(),
                    outputs["prior_flow"].detach().float(),
                )
                residual_target, residual_consistency = residual_from_composed_flow(
                    outputs["prior_flow"].detach().float(),
                    safe_target,
                    iterations=int(self.config.get("residual_target_iterations", 6)),
                )
                max_residual = float(self.config.get("max_residual_target", 24.0))
                residual_in_range = (
                    residual_target.abs().amax(dim=1, keepdim=True) <= max_residual
                )
                residual_valid = (
                    valid
                    & flow_valid_mask(residual_target, residual_target.shape[-2:])
                    & residual_in_range
                    & (
                        residual_consistency
                        <= float(self.config.get("max_residual_consistency", 1.0))
                    )
                )
                residual_target_consistency = _masked_mean(
                    residual_consistency, residual_valid
                )
                residual_target_valid_rate = _masked_mean(
                    residual_valid.to(final_flow.dtype), valid
                )
            residual_flow_loss = sequence_flow_loss(
                raw_residuals,
                residual_target,
                residual_valid,
                gamma=float(self.config.get("sequence_gamma", 0.8)),
            )
            residual_epe = endpoint_error(
                raw_residuals[-1].detach(),
                residual_target,
                residual_valid,
            )

        match_metrics = {
            key: final_flow.new_zeros(())
            for key in (
                "loss",
                "confidence_loss",
                "epe",
                "acc1",
                "entropy",
                "coverage",
                "advantage",
                "win_rate",
                "gate_target",
                "gate_mae",
            )
        }
        if "qwen_match_logits" in outputs and raw_residuals:
            match_metrics = local_match_supervision(
                outputs["qwen_match_logits"],
                residual_target,
                residual_valid,
                outputs["feature_confidence"],
                radius=int(outputs["qwen_match_radius"]),
                gate_temperature_px=float(
                    self.config.get("gate_temperature_px", 1.0)
                ),
            )

        prior_weight_key = (
            "prior_flow_unified" if outputs.get("stage") == "unified" else "prior_flow"
        )
        weighted = {
            "flow": float(self.config.get("flow", 1.0)) * flow_loss,
            "structure_flow": float(self.config.get("structure_flow", 0.0))
            * structure_flow_loss,
            "prior_flow": float(
                self.config.get(prior_weight_key, self.config.get("prior_flow", 0.5))
            )
            * prior_loss,
            "reconstruction": float(self.config.get("reconstruction", 0.15)) * photo,
            "line_reconstruction": float(self.config.get("line_reconstruction", 0.0))
            * line_reconstruction,
            "gradient": float(self.config.get("gradient", 0.10)) * gradient,
            "flow_gradient": float(self.config.get("flow_gradient", 0.0))
            * flow_gradient,
            "curvature": float(self.config.get("curvature", 0.0)) * curvature,
            "line_straightness": float(self.config.get("line_straightness", 0.0))
            * line_straightness,
            "bending": float(self.config.get("bending", 0.02)) * bending,
            "anti_fold": float(self.config.get("anti_fold", 0.20)) * anti_fold,
            "residual": float(self.config.get("residual", 0.002)) * residual,
            "residual_flow": float(self.config.get("residual_flow", 0.0))
            * residual_flow_loss,
            "qwen_match": float(self.config.get("qwen_match", 0.0))
            * match_metrics["loss"],
            "confidence": float(self.config.get("confidence", 0.0))
            * match_metrics["confidence_loss"],
        }
        total = torch.stack(tuple(weighted.values())).sum()
        final_epe = endpoint_error(final_flow.detach(), target_flow, valid).detach()
        prior_epe = endpoint_error(
            outputs["prior_flow"].detach(), target_flow, valid
        ).detach()
        final_error_map = torch.linalg.vector_norm(
            final_flow.detach() - target_flow, dim=1, keepdim=True
        )
        prior_error_map = torch.linalg.vector_norm(
            outputs["prior_flow"].detach() - target_flow, dim=1, keepdim=True
        )
        epe_p95 = _masked_quantile(final_error_map, valid, 0.95).detach()
        edge_epe = _masked_weighted_mean(final_error_map, valid, structures["edge"]).detach()
        line_epe = _masked_weighted_mean(final_error_map, valid, structures["line"]).detach()
        prior_edge_epe = _masked_weighted_mean(
            prior_error_map, valid, structures["edge"]
        ).detach()
        prior_line_epe = _masked_weighted_mean(
            prior_error_map, valid, structures["line"]
        ).detach()
        edge_mass = (valid.to(final_flow.dtype) * structures["edge"]).sum()
        line_mass = (valid.to(final_flow.dtype) * structures["line"]).sum()
        edge_epe = torch.where(edge_mass > 1e-6, edge_epe, final_epe)
        line_epe = torch.where(line_mass > 1e-6, line_epe, edge_epe)
        prior_edge_epe = torch.where(
            edge_mass > 1e-6, prior_edge_epe, prior_epe
        )
        prior_line_epe = torch.where(
            line_mass > 1e-6, prior_line_epe, prior_edge_epe
        )
        horizontal = structures["horizontal"]
        vertical = structures["vertical"]
        normal_weight = horizontal + vertical
        normal_error = (
            (final_flow[:, 1:2].detach() - target_flow[:, 1:2]).abs() * horizontal
            + (final_flow[:, 0:1].detach() - target_flow[:, 0:1]).abs() * vertical
        ) / normal_weight.clamp_min(1e-6)
        line_normal_mae = _masked_weighted_mean(
            normal_error, valid, normal_weight
        ).detach()
        line_straightness_error = line_straightness_loss(
            final_flow.detach(), target_flow, valid, horizontal, vertical, robust=False
        ).detach()
        prior_line_straightness_error = line_straightness_loss(
            outputs["prior_flow"].detach(),
            target_flow,
            valid,
            horizontal,
            vertical,
            robust=False,
        ).detach()
        structure_coverage = _masked_mean(structures["line"], valid).detach()
        result = {
            "total": total,
            **weighted,
            "raw_flow": flow_loss.detach(),
            "raw_structure_flow": structure_flow_loss.detach(),
            "flow_per_iteration": flow_per_iteration,
            "final_component_l1": final_component_l1,
            "sequence_weight_sum": final_flow.new_tensor(sequence_weight_sum),
            "raw_prior_flow": prior_loss.detach(),
            "raw_reconstruction": photo.detach(),
            "raw_line_reconstruction": line_reconstruction.detach(),
            "raw_gradient": gradient.detach(),
            "raw_flow_gradient": flow_gradient.detach(),
            "raw_curvature": curvature.detach(),
            "raw_line_straightness": line_straightness.detach(),
            "raw_bending": bending.detach(),
            "raw_anti_fold": anti_fold.detach(),
            "raw_residual": residual.detach(),
            "raw_residual_flow": residual_flow_loss.detach(),
            "raw_qwen_match": match_metrics["loss"].detach(),
            "raw_confidence": match_metrics["confidence_loss"].detach(),
            "epe": final_epe,
            "epe_p95": epe_p95,
            "edge_epe": edge_epe,
            "line_epe": line_epe,
            "prior_line_epe": prior_line_epe,
            "line_epe_gain": prior_line_epe - line_epe,
            "line_normal_mae": line_normal_mae,
            "line_straightness_error": line_straightness_error,
            "prior_line_straightness_error": prior_line_straightness_error,
            "line_straightness_gain": (
                prior_line_straightness_error - line_straightness_error
            ),
            "structure_coverage": structure_coverage,
            "prior_epe": prior_epe,
            "epe_gain": prior_epe - final_epe,
            "relative_epe_gain": (prior_epe - final_epe)
            / prior_epe.clamp_min(1e-6),
            "final_win_rate": _masked_mean(
                (final_error_map < prior_error_map).to(final_flow.dtype), valid
            ).detach(),
            "fold_rate": fold_rate.detach(),
            "jacobian_p01": jacobian_p01,
            "residual_p95": residual_p95.detach(),
            "applied_residual_p95": applied_residual_p95.detach(),
            "residual_epe": residual_epe.detach(),
            "residual_target_valid_rate": residual_target_valid_rate.detach(),
            "residual_target_consistency": residual_target_consistency.detach(),
            "qwen_match_epe": match_metrics["epe"].detach(),
            "qwen_match_acc1": match_metrics["acc1"].detach(),
            "qwen_match_entropy": match_metrics["entropy"].detach(),
            "qwen_match_coverage": match_metrics["coverage"].detach(),
            "qwen_advantage": match_metrics["advantage"].detach(),
            "qwen_win_rate": match_metrics["win_rate"].detach(),
            "gate_target": match_metrics["gate_target"].detach(),
            "gate_mae": match_metrics["gate_mae"].detach(),
            "residual_application_scale": final_flow.new_tensor(
                float(outputs.get("residual_application_scale", 1.0))
            ),
            "rectified": rectified.detach(),
        }
        if "feature_confidence" in outputs:
            result["feature_confidence"] = outputs["feature_confidence"].detach().mean()
        if "matching_feature_confidence" in outputs:
            result["matching_feature_confidence"] = (
                outputs["matching_feature_confidence"].detach().mean()
            )
        if "context_feature_confidence" in outputs:
            result["context_feature_confidence"] = (
                outputs["context_feature_confidence"].detach().mean()
            )
        return result
