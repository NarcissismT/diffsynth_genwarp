"""Stage-aware map losses; RGB remains a low-weight auxiliary signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .checkpoint import COORDINATE_CONTRACT
from .geometry import (
    canonical_backward_map,
    resize_backward_map_with_mask,
    resize_backward_map_window_with_mask,
    warp_with_backward_map,
)
from .metrics import cell_valid_mask, endpoint_error_map, jacobian_determinant


def masked_mean(value: Tensor, valid_mask: Tensor) -> Tensor:
    if value.shape != valid_mask.shape:
        try:
            valid_mask = valid_mask.expand_as(value)
        except RuntimeError as exc:
            raise ValueError(
                f"cannot broadcast mask {tuple(valid_mask.shape)} to {tuple(value.shape)}"
            ) from exc
    mask = valid_mask.bool() & torch.isfinite(value)
    # Keep the zero connected to the graph so an unexpected empty batch does
    # not turn ``total.backward()`` into a detached-scalar RuntimeError. The
    # dataset also rejects samples with no valid map pixels fail-closed.
    return value[mask].mean() if bool(mask.any()) else value.sum() * 0.0


def masked_local_ssim_loss(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    window_size: int = 7,
) -> Tensor:
    """Differentiable local ``1-SSIM`` over valid RGB/image pixels.

    Local moments are mask-normalized, so invalid map borders do not leak
    white/border padding into the geometry auxiliary loss.  Map EPE remains
    the primary objective; this term only penalizes structure lost by the
    single source-image sampling operation.
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM prediction and target must share [B,C,H,W]")
    if valid_mask.shape != prediction[:, :1].shape:
        raise ValueError("SSIM valid_mask must be [B,1,H,W]")
    kernel = min(int(window_size), prediction.shape[-2], prediction.shape[-1])
    if kernel % 2 == 0:
        kernel -= 1
    if kernel < 1:
        raise ValueError("SSIM window_size must be positive")
    padding = kernel // 2
    prediction = prediction.float()
    target = target.float()
    mask = valid_mask.float().expand_as(prediction)

    def local_mean(value: Tensor) -> Tensor:
        numerator = F.avg_pool2d(
            value * mask, kernel, stride=1, padding=padding
        )
        denominator = F.avg_pool2d(
            mask, kernel, stride=1, padding=padding
        ).clamp_min(1.0e-6)
        return numerator / denominator

    mean_prediction = local_mean(prediction)
    mean_target = local_mean(target)
    variance_prediction = (
        local_mean(prediction.square()) - mean_prediction.square()
    ).clamp_min(0.0)
    variance_target = (
        local_mean(target.square()) - mean_target.square()
    ).clamp_min(0.0)
    covariance = (
        local_mean(prediction * target) - mean_prediction * mean_target
    )
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (
        (2.0 * mean_prediction * mean_target + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (variance_prediction + variance_target + c2)
        ).clamp_min(1.0e-12)
    )
    return masked_mean((1.0 - ssim).clamp(0.0, 2.0), mask.bool())


def second_order_bending_loss(
    backward_map: Tensor,
    valid_mask: Tensor,
    source_size: tuple[int, int],
    canonical_map: Tensor | None = None,
) -> Tensor:
    canonical = (
        canonical_backward_map(
            backward_map.shape[0],
            backward_map.shape[-2:],
            source_size,
            device=backward_map.device,
            dtype=backward_map.dtype,
        )
        if canonical_map is None
        else canonical_map.to(backward_map)
    )
    if canonical.shape != backward_map.shape:
        raise ValueError("canonical_map and backward_map shapes differ")
    displacement = backward_map - canonical
    terms: list[Tensor] = []
    if displacement.shape[-1] >= 3:
        dxx = displacement[..., 2:] - 2.0 * displacement[..., 1:-1] + displacement[..., :-2]
        mask_x = valid_mask[..., 2:] & valid_mask[..., 1:-1] & valid_mask[..., :-2]
        terms.append(masked_mean(torch.linalg.vector_norm(dxx, dim=1, keepdim=True), mask_x))
    if displacement.shape[-2] >= 3:
        dyy = displacement[..., 2:, :] - 2.0 * displacement[..., 1:-1, :] + displacement[..., :-2, :]
        mask_y = valid_mask[..., 2:, :] & valid_mask[..., 1:-1, :] & valid_mask[..., :-2, :]
        terms.append(masked_mean(torch.linalg.vector_norm(dyy, dim=1, keepdim=True), mask_y))
    if min(displacement.shape[-2:]) >= 2:
        dxy = (
            displacement[..., 1:, 1:]
            - displacement[..., 1:, :-1]
            - displacement[..., :-1, 1:]
            + displacement[..., :-1, :-1]
        )
        mask_xy = (
            valid_mask[..., 1:, 1:]
            & valid_mask[..., 1:, :-1]
            & valid_mask[..., :-1, 1:]
            & valid_mask[..., :-1, :-1]
        )
        terms.append(
            2.0
            * masked_mean(
                torch.linalg.vector_norm(dxy, dim=1, keepdim=True), mask_xy
            )
        )
    return torch.stack(terms).mean() if terms else backward_map.new_zeros(())


def anti_fold_loss(
    backward_map: Tensor,
    valid_mask: Tensor,
    *,
    minimum_determinant: float = 0.01,
) -> Tensor:
    determinant = jacobian_determinant(backward_map)
    penalty = F.relu(float(minimum_determinant) - determinant).square()
    return masked_mean(penalty, cell_valid_mask(valid_mask))


def jacobian_scale_loss(
    backward_map: Tensor,
    valid_mask: Tensor,
    source_size: tuple[int, int],
    *,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
    canonical_map: Tensor | None = None,
) -> Tensor:
    """Penalize singular values far from the canonical source/target scale."""

    target_h, target_w = backward_map.shape[-2:]
    source_h, source_w = source_size
    derivative_x = backward_map[..., :, 1:] - backward_map[..., :, :-1]
    derivative_y = backward_map[..., 1:, :] - backward_map[..., :-1, :]
    dx = derivative_x[..., :-1, :]
    dy = derivative_y[..., :, :-1]
    if canonical_map is None:
        expected_x: Tensor | float = source_w / target_w
        expected_y: Tensor | float = source_h / target_h
    else:
        if canonical_map.shape != backward_map.shape:
            raise ValueError("canonical_map and backward_map shapes differ")
        canonical_dx = canonical_map[..., :, 1:] - canonical_map[..., :, :-1]
        canonical_dy = canonical_map[..., 1:, :] - canonical_map[..., :-1, :]
        expected_x = canonical_dx[:, 0, :-1, :].abs().clamp_min(1.0e-6)
        expected_y = canonical_dy[:, 1, :, :-1].abs().clamp_min(1.0e-6)
    matrix = torch.stack(
        (
            torch.stack((dx[:, 0] / expected_x, dy[:, 0] / expected_y), dim=-1),
            torch.stack((dx[:, 1] / expected_x, dy[:, 1] / expected_y), dim=-1),
        ),
        dim=-2,
    )
    singular = torch.linalg.svdvals(matrix.float())
    penalty = F.relu(float(minimum_scale) - singular[..., 1]).square()
    penalty = penalty + F.relu(singular[..., 0] - float(maximum_scale)).square()
    return masked_mean(penalty[:, None], cell_valid_mask(valid_mask))


def _normalized_structure_response(value: Tensor) -> Tensor:
    scale = value.flatten(1).mean(dim=1).clamp_min(1.0e-4)[:, None, None, None]
    return (value / (4.0 * scale)).clamp(0.0, 1.0)


def structure_targets(
    batch: dict[str, Tensor], output_size: tuple[int, int]
) -> Tensor:
    """Load optional H/V/B labels or derive deterministic image-based pseudo labels."""

    explicit_keys = ("horizontal_structure", "vertical_structure", "boundary_structure")
    if all(key in batch for key in explicit_keys):
        targets = torch.cat([batch[key].float() for key in explicit_keys], dim=1)
        return F.interpolate(targets, output_size, mode="bilinear", align_corners=False)
    image = F.interpolate(
        batch["warped_image"].float(), output_size, mode="bilinear", align_corners=False
    )
    gray = image.mean(dim=1, keepdim=True)
    # Horizontal ink/line structures generate strong vertical intensity change;
    # vertical structures generate strong horizontal change.
    response_h = F.pad(torch.abs(gray[..., 1:, :] - gray[..., :-1, :]), (0, 0, 0, 1))
    response_v = F.pad(torch.abs(gray[..., 1:] - gray[..., :-1]), (0, 1, 0, 0))
    response_h = F.avg_pool2d(response_h, (3, 9), stride=1, padding=(1, 4))
    response_v = F.avg_pool2d(response_v, (9, 3), stride=1, padding=(4, 1))
    valid = F.interpolate(
        batch["valid_mask"].float(), output_size, mode="nearest"
    )
    eroded = -F.max_pool2d(-valid, 3, stride=1, padding=1)
    boundary = (valid - eroded).clamp(0.0, 1.0)
    return torch.cat(
        (
            _normalized_structure_response(response_h),
            _normalized_structure_response(response_v),
            boundary,
        ),
        dim=1,
    ).detach()


def structure_straightness_loss(
    backward_map: Tensor,
    canonical_map: Tensor,
    valid_mask: Tensor,
    structure: Tensor,
) -> Tensor:
    """Suppress tangential coordinate curvature along H/V structures."""

    displacement = backward_map - canonical_map
    horizontal = structure[:, 0:1]
    vertical = structure[:, 1:2]
    terms: list[Tensor] = []
    if displacement.shape[-1] >= 3:
        dxx = displacement[..., 2:] - 2 * displacement[..., 1:-1] + displacement[..., :-2]
        valid_x = valid_mask[..., 2:] & valid_mask[..., 1:-1] & valid_mask[..., :-2]
        weight_x = horizontal[..., 1:-1]
        terms.append(
            masked_mean(
                torch.linalg.vector_norm(dxx, dim=1, keepdim=True) * weight_x,
                valid_x,
            )
        )
    if displacement.shape[-2] >= 3:
        dyy = displacement[..., 2:, :] - 2 * displacement[..., 1:-1, :] + displacement[..., :-2, :]
        valid_y = valid_mask[..., 2:, :] & valid_mask[..., 1:-1, :] & valid_mask[..., :-2, :]
        weight_y = vertical[..., 1:-1, :]
        terms.append(
            masked_mean(
                torch.linalg.vector_norm(dyy, dim=1, keepdim=True) * weight_y,
                valid_y,
            )
        )
    return torch.stack(terms).mean() if terms else backward_map.sum() * 0.0


@dataclass(frozen=True)
class CoarseLossWeights:
    map: float = 1.0
    confidence: float = 0.05
    warp: float = 0.10
    ssim: float = 0.0
    gradient: float = 0.05
    bending: float = 0.01
    anti_fold: float = 0.10

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CoarseLossWeights":
        if value is None:
            return cls()
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown coarse loss weights: {sorted(unknown)}")
        return cls(**{key: float(item) for key, item in value.items()})


class CoarseRectificationLoss(nn.Module):
    """Primary map supervision plus uncertainty and gentle structure terms."""

    def __init__(
        self,
        weights: CoarseLossWeights | None = None,
        *,
        charbonnier_epsilon: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.weights = weights or CoarseLossWeights()
        self.charbonnier_epsilon = float(charbonnier_epsilon)

    @staticmethod
    def _image_gradient(image: Tensor) -> tuple[Tensor, Tensor]:
        return image[..., :, 1:] - image[..., :, :-1], image[..., 1:, :] - image[..., :-1, :]

    def forward(
        self,
        prediction: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        predicted_map = prediction["backward_map"]
        target_map = batch["backward_map"].float()
        valid = batch["valid_mask"].bool()
        error = endpoint_error_map(predicted_map, target_map)
        map_loss = masked_mean(
            torch.sqrt(error.square() + self.charbonnier_epsilon**2),
            valid,
        )

        log_variance = prediction["log_variance"]
        confidence_nll = 0.5 * torch.exp(-log_variance) * error.square() + 0.5 * log_variance
        confidence_loss = masked_mean(confidence_nll, valid)

        rectified = prediction["rectified_image"]
        # Resizing source, photographed target, and map independently does not
        # commute with warping. It can therefore produce a non-zero RGB loss at
        # the exact GT map. Re-render the work-size source with the work-size GT
        # map so auxiliary photometric gradients are geometry-consistent and
        # cannot chase lighting/VAE/resize differences in rectified_image.
        with torch.no_grad():
            target_image = warp_with_backward_map(
                batch["warped_image"].float(),
                target_map,
                padding_mode="border",
            )
        image_valid = valid.expand(-1, rectified.shape[1], -1, -1)
        warp_loss = masked_mean(torch.abs(rectified - target_image), image_valid)
        ssim_loss = masked_local_ssim_loss(
            rectified, target_image, valid
        )
        pred_dx, pred_dy = self._image_gradient(rectified)
        target_dx, target_dy = self._image_gradient(target_image)
        gradient_x = masked_mean(
            torch.abs(pred_dx - target_dx),
            image_valid[..., :, 1:] & image_valid[..., :, :-1],
        )
        gradient_y = masked_mean(
            torch.abs(pred_dy - target_dy),
            image_valid[..., 1:, :] & image_valid[..., :-1, :],
        )
        gradient_loss = 0.5 * (gradient_x + gradient_y)
        source_size = tuple(int(value) for value in batch["warped_image"].shape[-2:])
        bending_loss = second_order_bending_loss(predicted_map, valid, source_size)
        fold_loss = anti_fold_loss(predicted_map, valid)
        parts = {
            "map": map_loss,
            "confidence": confidence_loss,
            "warp": warp_loss,
            "ssim": ssim_loss,
            "gradient": gradient_loss,
            "bending": bending_loss,
            "anti_fold": fold_loss,
        }
        total = sum(
            getattr(self.weights, name) * value
            for name, value in parts.items()
        )
        return {"total": total, **parts}


@dataclass(frozen=True)
class FullLossWeights:
    velocity: float = 1.0
    residual_endpoint: float = 0.5
    map: float = 1.0
    coarse_map: float = 0.35
    sequence: float = 0.5
    confidence: float = 0.05
    warp: float = 0.10
    ssim: float = 0.05
    gradient: float = 0.05
    hv: float = 0.20
    straightness: float = 0.10
    bending: float = 0.01
    anti_fold: float = 0.10
    jacobian_scale: float = 0.02
    preserve: float = 0.10

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "FullLossWeights":
        if value is None:
            return cls()
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown full loss weights: {sorted(unknown)}")
        return cls(**{key: float(item) for key, item in value.items()})


class CPDocFlowLoss(nn.Module):
    """Loss for coordinate velocity, all recurrent maps, and final geometry."""

    def __init__(
        self,
        weights: FullLossWeights | None = None,
        *,
        sequence_gamma: float = 0.8,
        charbonnier_epsilon: float = 1.0e-3,
        preserve_confidence_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        if not 0.0 < sequence_gamma <= 1.0:
            raise ValueError("sequence_gamma must be in (0,1]")
        if not 0.0 <= preserve_confidence_threshold <= 1.0:
            raise ValueError("preserve confidence threshold must be in [0,1]")
        self.weights = weights or FullLossWeights()
        self.sequence_gamma = float(sequence_gamma)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.preserve_confidence_threshold = float(preserve_confidence_threshold)

    def _charbonnier_epe(
        self, prediction: Tensor, target: Tensor, valid: Tensor
    ) -> Tensor:
        error = endpoint_error_map(prediction, target)
        return masked_mean(
            torch.sqrt(error.square() + self.charbonnier_epsilon**2), valid
        )

    def _validate_absolute_map_sequence(
        self,
        sequence: Any,
        target: Tensor,
        *,
        name: str,
        coordinate_contract: Any,
    ) -> list[Tensor]:
        if coordinate_contract != COORDINATE_CONTRACT:
            raise ValueError(
                f"{name} requires coordinate contract {COORDINATE_CONTRACT!r}; "
                f"got {coordinate_contract!r}"
            )
        if not isinstance(sequence, (list, tuple)):
            raise ValueError(f"{name} must be a list or tuple of absolute maps")
        checked: list[Tensor] = []
        for index, current in enumerate(sequence):
            if not isinstance(current, Tensor):
                raise ValueError(f"{name}[{index}] must be a tensor")
            if current.ndim != 4 or current.shape[1] != 2:
                raise ValueError(
                    f"{name}[{index}] must have shape [B,2,H,W], got "
                    f"{tuple(current.shape)}"
                )
            if current.shape[0] != target.shape[0] or min(current.shape[-2:]) < 1:
                raise ValueError(
                    f"{name}[{index}] batch/grid is incompatible with target "
                    f"{tuple(target.shape)}"
                )
            if not current.is_floating_point():
                raise ValueError(f"{name}[{index}] must be floating point")
            if not bool(torch.isfinite(current).all()):
                raise ValueError(f"{name}[{index}] contains non-finite coordinates")
            checked.append(current)
        return checked

    def _absolute_map_sequence_loss(
        self,
        sequence: list[Tensor],
        target: Tensor,
        valid: Tensor,
        source_size: tuple[int, int],
        target_canvas_size: tuple[int, int] | None = None,
        target_window: Tensor | None = None,
    ) -> Tensor:
        if not sequence:
            return target.sum() * 0.0
        terms: list[Tensor] = []
        count = len(sequence)
        for index, current in enumerate(sequence):
            if target_canvas_size is not None and target_window is not None:
                resized_target, resized_valid = resize_backward_map_window_with_mask(
                    target,
                    valid,
                    current.shape[-2:],
                    source_size_from=source_size,
                    source_size_to=source_size,
                    target_canvas_size=target_canvas_size,
                    target_window=target_window,
                )
            else:
                resized_target, resized_valid = resize_backward_map_with_mask(
                    target,
                    valid,
                    current.shape[-2:],
                    source_size_from=source_size,
                    source_size_to=source_size,
                )
            weight = self.sequence_gamma ** (count - index - 1)
            terms.append(
                current.new_tensor(weight)
                * self._charbonnier_epe(current, resized_target, resized_valid)
            )
        return torch.stack(terms).sum() / sum(
            self.sequence_gamma ** (count - index - 1) for index in range(count)
        )

    @staticmethod
    def _image_gradient(image: Tensor) -> tuple[Tensor, Tensor]:
        return image[..., :, 1:] - image[..., :, :-1], image[..., 1:, :] - image[..., :-1, :]

    def forward(
        self,
        prediction: dict[str, Any],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        target = batch["backward_map"].float()
        valid = batch["valid_mask"].bool()
        source = batch["warped_image"].float()
        source_size = tuple(int(value) for value in source.shape[-2:])
        final_map = prediction["backward_map"]
        coarse_map = prediction["coarse_backward_map"]
        map_loss = self._charbonnier_epe(final_map, target, valid)
        coarse_loss = self._charbonnier_epe(coarse_map, target, valid)

        if "velocity_prediction" in prediction:
            velocity_error = torch.linalg.vector_norm(
                prediction["velocity_prediction"] - prediction["velocity_target"],
                dim=1,
                keepdim=True,
            )
            confidence_low = F.interpolate(
                prediction["confidence"],
                velocity_error.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).detach()
            velocity_weight = 0.25 + (1.0 - confidence_low)
            velocity_loss = masked_mean(
                torch.sqrt(velocity_error.square() + self.charbonnier_epsilon**2)
                * velocity_weight,
                prediction["flow_matching_valid_mask"],
            )
            residual_endpoint_loss = self._charbonnier_epe(
                prediction["residual_proposal"],
                prediction["residual_proposal_target"],
                prediction["flow_matching_valid_mask"],
            )
        else:
            velocity_loss = final_map.sum() * 0.0
            residual_endpoint_loss = final_map.sum() * 0.0

        map_sequence_contract = prediction.get("map_sequence_coordinate_contract")
        if "flow_matching_map_sequence" in prediction:
            raw_fm_sequence = prediction["flow_matching_map_sequence"]
            compatibility_sequence = prediction.get("flow_matching_sequence")
            if (
                compatibility_sequence is not None
                and compatibility_sequence is not raw_fm_sequence
            ):
                raise ValueError(
                    "flow_matching_sequence must alias flow_matching_map_sequence"
                )
        else:
            raw_fm_sequence = prediction.get("flow_matching_sequence", [])
        raw_refiner_sequence = prediction.get("refiner_sequence", [])
        fm_sequence = self._validate_absolute_map_sequence(
            raw_fm_sequence,
            target,
            name="flow_matching_map_sequence",
            coordinate_contract=map_sequence_contract,
        )
        refiner_sequence = self._validate_absolute_map_sequence(
            raw_refiner_sequence,
            target,
            name="refiner_sequence",
            coordinate_contract=map_sequence_contract,
        )
        sequence_terms = [
            self._absolute_map_sequence_loss(
                sequence,
                target,
                valid,
                source_size,
                prediction.get("target_canvas_size"),
                prediction.get("target_window"),
            )
            for sequence in (fm_sequence, refiner_sequence)
            if sequence
        ]
        sequence_loss = (
            torch.stack(sequence_terms).mean()
            if sequence_terms
            else final_map.sum() * 0.0
        )

        coarse_error = endpoint_error_map(coarse_map, target)
        log_variance = prediction["coarse_log_variance"]
        confidence_nll = (
            0.5 * torch.exp(-log_variance) * coarse_error.square()
            + 0.5 * log_variance
        )
        confidence_event = (coarse_error.detach() < 1.0).to(
            prediction["confidence"].dtype
        )
        confidence_bce = F.binary_cross_entropy(
            prediction["confidence"].clamp(1.0e-6, 1.0 - 1.0e-6),
            confidence_event,
            reduction="none",
        )
        confidence_loss = masked_mean(confidence_nll + confidence_bce, valid)

        rectified = prediction["rectified_image"]
        with torch.no_grad():
            target_image = warp_with_backward_map(source, target, padding_mode="border")
        image_valid = valid.expand(-1, rectified.shape[1], -1, -1)
        warp_loss = masked_mean(torch.abs(rectified - target_image), image_valid)
        ssim_loss = masked_local_ssim_loss(rectified, target_image, valid)
        pred_dx, pred_dy = self._image_gradient(rectified)
        target_dx, target_dy = self._image_gradient(target_image)
        gradient_loss = 0.5 * (
            masked_mean(
                torch.abs(pred_dx - target_dx),
                image_valid[..., :, 1:] & image_valid[..., :, :-1],
            )
            + masked_mean(
                torch.abs(pred_dy - target_dy),
                image_valid[..., 1:, :] & image_valid[..., :-1, :],
            )
        )
        canonical_final = prediction.get("canonical_map")
        bending_loss = second_order_bending_loss(
            final_map, valid, source_size, canonical_final
        )
        fold_loss = anti_fold_loss(final_map, valid)
        scale_loss = jacobian_scale_loss(
            final_map,
            valid,
            source_size,
            canonical_map=canonical_final,
        )
        structure = structure_targets(batch, final_map.shape[-2:])
        hv_logits = prediction["hv_logits"]
        hv_loss = masked_mean(
            F.binary_cross_entropy_with_logits(
                hv_logits, structure.to(hv_logits.dtype), reduction="none"
            ),
            valid.expand_as(hv_logits),
        )
        if canonical_final is None:
            canonical_final = canonical_backward_map(
                final_map.shape[0],
                final_map.shape[-2:],
                source_size,
                device=final_map.device,
                dtype=final_map.dtype,
            )
        straightness_loss = structure_straightness_loss(
            final_map, canonical_final, valid, structure
        )
        reliable = (
            valid
            & (prediction["confidence"].detach() >= self.preserve_confidence_threshold)
        )
        preserve_error = torch.linalg.vector_norm(final_map - coarse_map, dim=1, keepdim=True)
        preserve_loss = masked_mean(preserve_error, reliable)
        parts = {
            "velocity": velocity_loss,
            "residual_endpoint": residual_endpoint_loss,
            "map": map_loss,
            "coarse_map": coarse_loss,
            "sequence": sequence_loss,
            "confidence": confidence_loss,
            "warp": warp_loss,
            "ssim": ssim_loss,
            "gradient": gradient_loss,
            "hv": hv_loss,
            "straightness": straightness_loss,
            "bending": bending_loss,
            "anti_fold": fold_loss,
            "jacobian_scale": scale_loss,
            "preserve": preserve_loss,
        }
        total = sum(getattr(self.weights, name) * value for name, value in parts.items())
        return {"total": total, **parts}
