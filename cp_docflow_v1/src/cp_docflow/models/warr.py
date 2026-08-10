"""Canonical WARR and structure-aware convex backward-map upsampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry import canonical_backward_map, resize_backward_map, warp_with_backward_map
from .coarse import ConvNormAct
from .coordinate_flow_transformer import normalize_pixel_delta, normalize_pixel_map


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        combined = input_channels + hidden_channels
        self.update = nn.Conv2d(combined, hidden_channels, 3, padding=1)
        self.reset = nn.Conv2d(combined, hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(combined, hidden_channels, 3, padding=1)

    def forward(self, hidden: Tensor, value: Tensor) -> Tensor:
        combined = torch.cat((hidden, value), dim=1)
        update = torch.sigmoid(self.update(combined))
        reset = torch.sigmoid(self.reset(combined))
        candidate = torch.tanh(
            self.candidate(torch.cat((reset * hidden, value), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


def _forward_x(value: Tensor) -> Tensor:
    return F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0), mode="replicate")


def _forward_y(value: Tensor) -> Tensor:
    return F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1), mode="replicate")


def geometry_jacobian_cues(
    backward_map: Tensor,
    canonical_map: Tensor,
    source_size: tuple[int, int],
) -> Tensor:
    """Displacement, first derivatives, determinant, and second derivatives."""

    normalized_map = normalize_pixel_map(backward_map, source_size)
    displacement = normalize_pixel_delta(backward_map - canonical_map, source_size)
    derivative_x = _forward_x(normalized_map)
    derivative_y = _forward_y(normalized_map)
    determinant = (
        derivative_x[:, 0:1] * derivative_y[:, 1:2]
        - derivative_x[:, 1:2] * derivative_y[:, 0:1]
    )
    second_x = _forward_x(derivative_x)
    second_y = _forward_y(derivative_y)
    mixed = 0.5 * (_forward_y(derivative_x) + _forward_x(derivative_y))
    return torch.cat(
        (displacement, derivative_x, derivative_y, determinant, second_x, second_y, mixed),
        dim=1,
    )


# Compatibility alias for older tests/importers.
def jacobian_cues(backward_map: Tensor, source_size: tuple[int, int]) -> Tensor:
    canonical = canonical_backward_map(
        backward_map.shape[0],
        backward_map.shape[-2:],
        source_size,
        device=backward_map.device,
        dtype=backward_map.dtype,
    )
    return geometry_jacobian_cues(backward_map, canonical, source_size)


class HighResolutionMapRefiner(nn.Module):
    """Single-image Warp-Aware Recurrent Refinement at the 1/4 target grid.

    Confidence is intentionally absent.  It affects only the composed initial
    map; WARR judges local geometry from warped source/HV features and map
    derivatives, as required by the newest architecture.
    """

    def __init__(
        self,
        source_feature_channels: int,
        hv_feature_channels: int,
        hidden_channels: int = 128,
        iterations: int = 4,
        max_step_px: float = 2.0,
        confidence_preserve_strength: float | None = None,
    ) -> None:
        super().__init__()
        del confidence_preserve_strength  # Accepted only for old config compatibility.
        if iterations < 1:
            raise ValueError("refiner iterations must be positive")
        if max_step_px <= 0.0:
            raise ValueError("max_step_px must be positive")
        self.iterations = int(iterations)
        self.max_step_px = float(max_step_px)
        self.hidden_channels = int(hidden_channels)
        # warped source, warped H/V, and 13 geometry/Jacobian channels.
        motion_channels = source_feature_channels + hv_feature_channels + 13
        self.geometry_encoder = nn.Sequential(
            ConvNormAct(motion_channels, hidden_channels),
            ConvNormAct(hidden_channels, hidden_channels),
        )
        self.initial_hidden = ConvNormAct(
            source_feature_channels + hv_feature_channels, hidden_channels
        )
        self.gru = ConvGRUCell(hidden_channels, hidden_channels)
        self.delta_head = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        self.update_gate_head = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, 1, 3, padding=1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.update_gate_head[-1].weight)
        nn.init.constant_(self.update_gate_head[-1].bias, -1.0)

    @property
    def delta(self) -> nn.Sequential:
        """Compatibility alias for the previous refiner output head."""

        return self.delta_head

    @staticmethod
    def _warp_source_feature(
        feature: Tensor,
        current_map: Tensor,
        source_size: tuple[int, int],
    ) -> Tensor:
        feature_map = resize_backward_map(
            current_map,
            current_map.shape[-2:],
            source_size_from=source_size,
            source_size_to=feature.shape[-2:],
        )
        return warp_with_backward_map(feature, feature_map, padding_mode="border")

    def forward(
        self,
        initial_map: Tensor,
        *,
        source_feature: Tensor,
        hv_feature: Tensor,
        source_size: tuple[int, int],
        confidence: Tensor | None = None,
        coarse_map: Tensor | None = None,
        canonical_map: Tensor | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        # Old keyword inputs are rejected semantically but accepted so existing
        # checkpoints/config call sites can migrate without a Python signature break.
        del confidence, coarse_map
        if source_feature.shape[-2:] != hv_feature.shape[-2:]:
            raise ValueError("source and H/V feature grids differ")
        current = initial_map.float()
        canonical = (
            canonical_backward_map(
                current.shape[0],
                current.shape[-2:],
                source_size,
                device=current.device,
                dtype=torch.float32,
            )
            if canonical_map is None
            else canonical_map.float()
        )
        if canonical.shape != current.shape:
            raise ValueError("canonical_map and initial_map grids differ")
        initial_source = self._warp_source_feature(source_feature, current, source_size)
        initial_hv = self._warp_source_feature(hv_feature, current, source_size)
        hidden = torch.tanh(
            self.initial_hidden(torch.cat((initial_source, initial_hv), dim=1))
        )
        sequence: list[Tensor] = []
        deltas: list[Tensor] = []
        update_gates: list[Tensor] = []
        for _ in range(self.iterations):
            warped_source = self._warp_source_feature(source_feature, current, source_size)
            warped_hv = self._warp_source_feature(hv_feature, current, source_size)
            geometry = geometry_jacobian_cues(current, canonical, source_size)
            motion = self.geometry_encoder(
                torch.cat((warped_source, warped_hv, geometry), dim=1)
            )
            hidden = self.gru(hidden, motion)
            update_gate = torch.sigmoid(self.update_gate_head(hidden).float())
            bounded_delta = torch.tanh(self.delta_head(hidden).float()) * self.max_step_px
            delta = update_gate * bounded_delta
            current = current + delta
            deltas.append(delta)
            update_gates.append(update_gate)
            sequence.append(current)
        return {
            "backward_map": current,
            "sequence": sequence,
            "deltas": deltas,
            "update_gates": update_gates,
            "hidden": hidden,
        }


class ConvexMapUpsampler(nn.Module):
    """RAFT-style learned convex upsampling of displacement, not absolute map."""

    def __init__(
        self,
        context_channels: int,
        hidden_channels: int = 128,
        scale: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if scale < 1 or kernel_size % 2 != 1:
            raise ValueError("convex upsampling needs positive scale and odd kernel")
        self.scale = int(scale)
        self.kernel_size = int(kernel_size)
        weights = scale * scale * kernel_size * kernel_size
        self.mask = nn.Sequential(
            ConvNormAct(context_channels, hidden_channels),
            nn.Conv2d(hidden_channels, weights, 1),
        )
        # Uniform convex weights at initialization are stable; canonical
        # identity remains exact because only displacement is upsampled.
        nn.init.zeros_(self.mask[-1].weight)
        nn.init.zeros_(self.mask[-1].bias)

    def forward(
        self,
        low_map: Tensor,
        context: Tensor,
        *,
        output_size: tuple[int, int],
        source_size: tuple[int, int],
        canonical_low: Tensor | None = None,
        canonical_high: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if context.shape[0] != low_map.shape[0] or context.shape[-2:] != low_map.shape[-2:]:
            raise ValueError("convex context and low map grids differ")
        batch, _, low_h, low_w = low_map.shape
        if canonical_low is None:
            canonical_low = canonical_backward_map(
                batch,
                (low_h, low_w),
                source_size,
                device=low_map.device,
                dtype=low_map.dtype,
            )
        if canonical_low.shape != low_map.shape:
            raise ValueError("canonical_low and low_map grids differ")
        displacement = low_map - canonical_low
        mask_logits = self.mask(context)
        neighborhood = self.kernel_size * self.kernel_size
        mask = mask_logits.view(
            batch, 1, neighborhood, self.scale, self.scale, low_h, low_w
        ).softmax(dim=2)
        unfolded = F.unfold(
            displacement,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        ).view(batch, 2, neighborhood, 1, 1, low_h, low_w)
        high_displacement = (mask * unfolded).sum(dim=2)
        high_displacement = (
            high_displacement.permute(0, 1, 4, 2, 5, 3)
            .reshape(batch, 2, low_h * self.scale, low_w * self.scale)
        )
        output_h, output_w = (int(value) for value in output_size)
        if output_h > high_displacement.shape[-2] or output_w > high_displacement.shape[-1]:
            raise ValueError("convex low grid is too small for requested output size")
        high_displacement = high_displacement[..., :output_h, :output_w]
        if canonical_high is None:
            canonical_high = canonical_backward_map(
                batch,
                (output_h, output_w),
                source_size,
                device=low_map.device,
                dtype=low_map.dtype,
            )
        if canonical_high.shape != (batch, 2, output_h, output_w):
            raise ValueError("canonical_high grid differs from output_size")
        return {
            "backward_map": canonical_high + high_displacement,
            "convex_mask": mask,
            "displacement": high_displacement,
            "canonical_high": canonical_high,
        }
