"""Stage-1 deterministic coarse backward-map and confidence predictor."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry import canonical_backward_map, warp_with_backward_map


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        # At the smallest supported 16x16 input, Stage-16 is spatially 1x1.
        # Each normalization group must still contain at least two channels so
        # batch=1 has more than one value per group.
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels),
            ConvNormAct(channels, channels, activation=False),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.body(value))


class DeterministicCoarseRectifier(nn.Module):
    """CNN/FPN baseline producing an absolute map and calibrated confidence.

    The displacement head operates at 1/8 resolution but expresses residuals
    in full source-image pixels. It is initialized to zero, making a fresh
    model an exact canonical/identity warp rather than an arbitrary geometric
    corruption. No Qwen or diffusion component is used at this stage.
    """

    def __init__(
        self,
        *,
        base_channels: int = 32,
        feature_channels: int = 128,
        max_displacement_ratio: float = 0.45,
        min_log_variance: float = -6.0,
        max_log_variance: float = 6.0,
    ) -> None:
        super().__init__()
        base_channels = int(base_channels)
        feature_channels = int(feature_channels)
        if base_channels < 8 or feature_channels < 16:
            raise ValueError("base_channels>=8 and feature_channels>=16 are required")
        if not 0.0 < float(max_displacement_ratio) <= 1.0:
            raise ValueError("max_displacement_ratio must be in (0,1]")
        if float(min_log_variance) >= float(max_log_variance):
            raise ValueError("min_log_variance must be below max_log_variance")
        self.max_displacement_ratio = float(max_displacement_ratio)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)

        c = base_channels
        self.stem = nn.Sequential(
            ConvNormAct(3, c, kernel_size=5, stride=2),
            ResidualBlock(c),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(c, 2 * c, stride=2),
            ResidualBlock(2 * c),
        )
        self.stage8 = nn.Sequential(
            ConvNormAct(2 * c, 4 * c, stride=2),
            ResidualBlock(4 * c),
            ResidualBlock(4 * c),
        )
        self.stage16 = nn.Sequential(
            ConvNormAct(4 * c, 8 * c, stride=2),
            ResidualBlock(8 * c),
            ResidualBlock(8 * c),
        )
        self.lateral8 = ConvNormAct(4 * c, feature_channels, kernel_size=1)
        self.lateral16 = ConvNormAct(8 * c, feature_channels, kernel_size=1)
        self.fpn8 = nn.Sequential(
            ConvNormAct(feature_channels, feature_channels),
            ResidualBlock(feature_channels),
        )
        self.local4 = ConvNormAct(2 * c, feature_channels // 2, kernel_size=1)
        self.map_head = nn.Sequential(
            ConvNormAct(feature_channels, feature_channels),
            nn.Conv2d(feature_channels, 2, 3, padding=1),
        )
        self.log_variance_head = nn.Sequential(
            ConvNormAct(feature_channels, feature_channels // 2),
            nn.Conv2d(feature_channels // 2, 1, 3, padding=1),
        )
        nn.init.zeros_(self.map_head[-1].weight)
        nn.init.zeros_(self.map_head[-1].bias)
        nn.init.zeros_(self.log_variance_head[-1].weight)
        nn.init.zeros_(self.log_variance_head[-1].bias)

    def encode(self, warped_image: Tensor) -> dict[str, Tensor]:
        feature2 = self.stem(warped_image)
        feature4 = self.stage4(feature2)
        feature8 = self.stage8(feature4)
        feature16 = self.stage16(feature8)
        top = self.lateral16(feature16)
        top = F.interpolate(
            top,
            size=feature8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fpn8 = self.fpn8(self.lateral8(feature8) + top)
        return {
            "fpn8": fpn8,
            # Kept for the Stage-2 1/4 recurrent updater. Exposing it now makes
            # the checkpoint boundary explicit without activating that stage.
            "local4": self.local4(feature4),
        }

    def forward(
        self,
        warped_image: Tensor,
        *,
        output_size: Sequence[int] | None = None,
        render: bool = True,
    ) -> dict[str, Tensor]:
        if warped_image.ndim != 4 or warped_image.shape[1] != 3:
            raise ValueError(
                f"warped_image must be [B,3,H,W], got {tuple(warped_image.shape)}"
            )
        input_h, input_w = warped_image.shape[-2:]
        if min(input_h, input_w) < 16:
            raise ValueError("warped_image spatial dimensions must be at least 16")
        if output_size is None:
            output_h, output_w = input_h, input_w
        else:
            if len(output_size) != 2:
                raise ValueError("output_size must be (height,width)")
            output_h, output_w = int(output_size[0]), int(output_size[1])
            if min(output_h, output_w) < 1:
                raise ValueError("output_size must be positive")

        features = self.encode(warped_image)
        fpn8 = features["fpn8"]
        # Coordinate tensors remain FP32 even under mixed-precision feature
        # extraction; BF16 cannot represent every coordinate on large pages.
        raw_residual = self.map_head(fpn8).float()
        scale = raw_residual.new_tensor(
            (input_w, input_h),
        ).view(1, 2, 1, 1) * self.max_displacement_ratio
        residual_low = torch.tanh(raw_residual) * scale
        residual = F.interpolate(
            residual_low,
            size=(output_h, output_w),
            mode="bilinear",
            align_corners=False,
        )
        canonical = canonical_backward_map(
            warped_image.shape[0],
            (output_h, output_w),
            (input_h, input_w),
            device=warped_image.device,
            dtype=torch.float32,
        )
        backward_map = canonical + residual

        log_variance_low = self.log_variance_head(fpn8).float().clamp(
            self.min_log_variance,
            self.max_log_variance,
        )
        log_variance = F.interpolate(
            log_variance_low,
            size=(output_h, output_w),
            mode="bilinear",
            align_corners=False,
        )
        # Under the isotropic 2-D Gaussian used by the NLL, radial squared
        # error follows chi-square(2). This is P(EPE < 1 px), matching the
        # evaluator's Brier/ECE event instead of using an uncalibrated sigmoid.
        confidence = -torch.expm1(-0.5 * torch.exp(-log_variance))
        low_canonical = canonical_backward_map(
            warped_image.shape[0],
            fpn8.shape[-2:],
            (input_h, input_w),
            device=warped_image.device,
            dtype=torch.float32,
        )
        output: dict[str, Tensor] = {
            "backward_map": backward_map,
            "coarse_backward_map": backward_map,
            "canonical_map": canonical,
            "residual_map": residual,
            "low_resolution_backward_map": low_canonical + residual_low,
            "log_variance": log_variance,
            "confidence": confidence,
            "fpn8": fpn8,
            "local4": features["local4"],
        }
        if render:
            rectified, valid = warp_with_backward_map(
                warped_image.float(),
                backward_map,
                padding_mode="border",
                return_valid=True,
            )
            output["rectified_image"] = rectified
            output["valid_mask"] = valid
        return output
