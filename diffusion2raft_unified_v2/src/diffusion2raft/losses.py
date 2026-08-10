"""Losses that supervise geometry and suppress ripple/folding artifacts."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .geometry import backward_warp, endpoint_error


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.bool()
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    return value[mask].mean() if mask.any() else value.new_zeros(())


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


def second_order_bending_loss(flow: Tensor, valid: Tensor) -> Tensor:
    """Penalize local flow curvature while preserving affine/projective trends."""

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

        final_flow = outputs["final_flow"]
        flow_loss = sequence_flow_loss(
            outputs["flows"],
            target_flow,
            valid,
            gamma=float(self.config.get("sequence_gamma", 0.8)),
        )
        prior_loss = _masked_mean(
            robust_l1(outputs["prior_flow"] - target_flow),
            valid,
        )
        photo, gradient, rectified = reconstruction_losses(
            batch["warped"], batch["target"], final_flow, valid
        )
        bending = second_order_bending_loss(final_flow, valid)
        anti_fold, fold_rate = anti_fold_loss(
            final_flow,
            valid,
            min_jacobian=float(self.config.get("min_jacobian", 0.05)),
        )
        if outputs["residuals"]:
            residual_norm = torch.linalg.vector_norm(
                outputs["residuals"][-1], dim=1, keepdim=True
            )
            residual = _masked_mean(residual_norm, valid)
            residual_p95 = _masked_quantile(residual_norm.detach(), valid, 0.95)
        else:
            residual = final_flow.new_zeros(())
            residual_p95 = final_flow.new_zeros(())

        prior_weight_key = (
            "prior_flow_unified" if outputs.get("stage") == "unified" else "prior_flow"
        )
        weighted = {
            "flow": float(self.config.get("flow", 1.0)) * flow_loss,
            "prior_flow": float(
                self.config.get(prior_weight_key, self.config.get("prior_flow", 0.5))
            )
            * prior_loss,
            "reconstruction": float(self.config.get("reconstruction", 0.15)) * photo,
            "gradient": float(self.config.get("gradient", 0.10)) * gradient,
            "bending": float(self.config.get("bending", 0.02)) * bending,
            "anti_fold": float(self.config.get("anti_fold", 0.20)) * anti_fold,
            "residual": float(self.config.get("residual", 0.002)) * residual,
        }
        total = torch.stack(tuple(weighted.values())).sum()
        result = {
            "total": total,
            **weighted,
            "raw_flow": flow_loss.detach(),
            "raw_prior_flow": prior_loss.detach(),
            "raw_reconstruction": photo.detach(),
            "raw_gradient": gradient.detach(),
            "raw_bending": bending.detach(),
            "raw_anti_fold": anti_fold.detach(),
            "raw_residual": residual.detach(),
            "epe": endpoint_error(final_flow.detach(), target_flow, valid).detach(),
            "prior_epe": endpoint_error(
                outputs["prior_flow"].detach(), target_flow, valid
            ).detach(),
            "fold_rate": fold_rate.detach(),
            "residual_p95": residual_p95.detach(),
            "rectified": rectified.detach(),
        }
        if "feature_confidence" in outputs:
            result["feature_confidence"] = outputs["feature_confidence"].detach().mean()
        return result
