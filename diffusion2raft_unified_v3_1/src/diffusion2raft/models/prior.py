"""Single-image smooth geometry prior for document dewarping.

The local U-Net predicts page curl well, but a bounded displacement head alone
cannot represent arbitrary page orientation.  The optional global pose branch
therefore predicts a projective transform and composes it *after* the local
target-to-intermediate map.  It is identity initialized and disabled by
default, preserving both the state-dict keys and numerical behaviour of v3.1.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry import backward_flow_to_map, make_pixel_grid


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
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
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class DecodeBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse(torch.cat((x, skip), dim=1))


class GlobalProjectiveHead(nn.Module):
    """Predict and apply one image-level projective transform.

    The transform acts in align-corners normalized coordinates around the
    canvas centre.  Its eight bounded residual coefficients directly form a
    homography.  Unlike an angle initialized at zero, this parameterization
    has a useful first-order gradient for an exact half-turn target.

    The input ``local_flow`` maps the final target to an intermediate canvas.
    Applying the homography to that absolute map produces the composed
    target-to-source mapping.  This ordering is important for augmented
    training samples, whose source-only transform post-multiplies the original
    backward coordinate map.
    """

    PARAMETER_COUNT = 8

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int = 256,
        pool_size: int = 4,
        max_linear_delta: float = 2.50,
        max_translation_ratio: float = 0.50,
        max_perspective: float = 0.15,
    ) -> None:
        super().__init__()
        self.pool_size = int(pool_size)
        if self.pool_size < 1:
            raise ValueError("global pose pool_size must be >= 1")
        self.max_linear_delta = float(max_linear_delta)
        self.max_translation_ratio = float(max_translation_ratio)
        self.max_perspective = float(max_perspective)
        if not math.isfinite(self.max_linear_delta) or self.max_linear_delta <= 2.0:
            raise ValueError(
                "max_linear_delta must be > 2 so a half-turn is reachable "
                "without tanh saturation"
            )
        if (
            not math.isfinite(self.max_translation_ratio)
            or self.max_translation_ratio < 0.0
        ):
            raise ValueError("max_translation_ratio must be finite and non-negative")
        if (
            not math.isfinite(self.max_perspective)
            or not 0.0 <= self.max_perspective < 0.5
        ):
            raise ValueError("max_perspective must be finite and in [0, 0.5)")

        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * self.pool_size * self.pool_size, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, self.PARAMETER_COUNT),
        )
        # The old local prior is the complete prediction at initialization.
        # Zeroing only the last layer still lets it learn immediately through
        # that layer, while leaving the first projection normally initialized.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _homography(self, raw: Tensor) -> Tensor:
        if raw.ndim != 2 or raw.shape[1] != self.PARAMETER_COUNT:
            raise ValueError(
                f"global pose parameters must be [B,{self.PARAMETER_COUNT}], "
                f"got {tuple(raw.shape)}"
            )
        delta = torch.tanh(raw)
        a00 = 1.0 + self.max_linear_delta * delta[:, 0]
        a01 = self.max_linear_delta * delta[:, 1]
        translate_x = 2.0 * self.max_translation_ratio * delta[:, 2]
        a10 = self.max_linear_delta * delta[:, 3]
        a11 = 1.0 + self.max_linear_delta * delta[:, 4]
        translate_y = 2.0 * self.max_translation_ratio * delta[:, 5]
        perspective_x = self.max_perspective * delta[:, 6]
        perspective_y = self.max_perspective * delta[:, 7]

        one = torch.ones_like(a00)
        return torch.stack(
            (
                a00,
                a01,
                translate_x,
                a10,
                a11,
                translate_y,
                perspective_x,
                perspective_y,
                one,
            ),
            dim=1,
        ).reshape(-1, 3, 3)

    @staticmethod
    def _safe_denominator(value: Tensor, epsilon: float = 1e-4) -> Tensor:
        sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
        return torch.where(value.abs() < epsilon, sign * epsilon, value)

    def forward(self, features: Tensor, local_flow: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must be [B,C,H,W], got {tuple(features.shape)}")
        if local_flow.ndim != 4 or local_flow.shape[1] != 2:
            raise ValueError(
                f"local_flow must be [B,2,H,W], got {tuple(local_flow.shape)}"
            )
        if features.shape[0] != local_flow.shape[0]:
            raise ValueError("features and local_flow batch sizes differ")

        batch, _, height, width = local_flow.shape
        # Pose coordinates are geometry, not appearance features.  Keep their
        # predictor and arithmetic in FP32 under the outer BF16 autocast;
        # otherwise near-half-turn parameters and 512px coordinates can jump.
        with torch.autocast(device_type=local_flow.device.type, enabled=False):
            raw = self.mlp(self.pool(features.float()))
            local_flow_float = local_flow.float()
            homography = self._homography(raw)
            local_map = backward_flow_to_map(local_flow_float)
            x = 2.0 * local_map[:, 0] / max(width - 1, 1) - 1.0
            y = 2.0 * local_map[:, 1] / max(height - 1, 1) - 1.0
            ones = torch.ones_like(x)
            homogeneous = torch.stack((x, y, ones), dim=1).flatten(2)
            transformed = torch.bmm(homography, homogeneous).reshape(
                batch, 3, height, width
            )
            denominator = self._safe_denominator(transformed[:, 2])
            source_x = (transformed[:, 0] / denominator + 1.0) * 0.5 * max(
                width - 1, 1
            )
            source_y = (transformed[:, 1] / denominator + 1.0) * 0.5 * max(
                height - 1, 1
            )
            source_map = torch.stack((source_x, source_y), dim=1)
            target_grid = make_pixel_grid(
                batch,
                height,
                width,
                device=local_flow.device,
                dtype=local_flow_float.dtype,
            )
            return source_map - target_grid


class DocumentGeometryPrior(nn.Module):
    """Coordinate-aware U-Net that predicts a coarse backward flow.

    This branch sees only the warped source. It prevents the system from being
    fully dependent on text/content synthesized by the diffusion guide.
    """

    def __init__(
        self,
        base_channels: int = 32,
        max_displacement_ratio: float = 0.35,
        control_stride: int = 8,
        global_pose_enabled: bool = False,
        global_pose_hidden_channels: int = 256,
        global_pose_pool_size: int = 4,
        global_pose_max_linear_delta: float = 2.50,
        global_pose_max_translation_ratio: float = 0.50,
        global_pose_max_perspective: float = 0.15,
    ) -> None:
        super().__init__()
        c = base_channels
        self.max_displacement_ratio = float(max_displacement_ratio)
        self.control_stride = int(control_stride)
        if self.control_stride < 1:
            raise ValueError("control_stride must be >= 1")
        self.stem = nn.Sequential(ConvNormAct(5, c, kernel_size=5), ResidualBlock(c))
        self.down1 = nn.Sequential(ConvNormAct(c, 2 * c, stride=2), ResidualBlock(2 * c))
        self.down2 = nn.Sequential(ConvNormAct(2 * c, 4 * c, stride=2), ResidualBlock(4 * c))
        self.down3 = nn.Sequential(ConvNormAct(4 * c, 8 * c, stride=2), ResidualBlock(8 * c))
        self.bottleneck = nn.Sequential(ResidualBlock(8 * c), ResidualBlock(8 * c))
        self.up2 = DecodeBlock(8 * c, 4 * c, 4 * c)
        self.up1 = DecodeBlock(4 * c, 2 * c, 2 * c)
        self.up0 = DecodeBlock(2 * c, c, c)
        self.head = nn.Sequential(
            ConvNormAct(c, c),
            nn.Conv2d(c, 2, kernel_size=3, padding=1),
        )
        self.global_pose_head: GlobalProjectiveHead | None
        if global_pose_enabled:
            self.global_pose_head = GlobalProjectiveHead(
                8 * c,
                hidden_channels=int(global_pose_hidden_channels),
                pool_size=int(global_pose_pool_size),
                max_linear_delta=float(global_pose_max_linear_delta),
                max_translation_ratio=float(global_pose_max_translation_ratio),
                max_perspective=float(global_pose_max_perspective),
            )
        else:
            # None registers no state-dict entries, which keeps old v3.1
            # checkpoints strictly compatible when this feature is disabled.
            self.global_pose_head = None

        # Identity mapping is the safest initial prediction.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _coordinate_channels(x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        y, x_coord = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack((x_coord, y), dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return coords

    def forward(self, warped: Tensor) -> Tensor:
        if warped.ndim != 4 or warped.shape[1] != 3:
            raise ValueError(f"warped must be [B,3,H,W], got {tuple(warped.shape)}")
        _, _, h, w = warped.shape
        x = torch.cat((warped, self._coordinate_channels(warped)), dim=1)
        e0 = self.stem(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.bottleneck(self.down3(e2))
        d2 = self.up2(e3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)

        raw = self.head(d0)
        if self.control_stride > 1:
            coarse_size = (
                max(2, (h + self.control_stride - 1) // self.control_stride),
                max(2, (w + self.control_stride - 1) // self.control_stride),
            )
            raw = F.adaptive_avg_pool2d(raw, coarse_size)
            raw = F.interpolate(raw, size=(h, w), mode="bicubic", align_corners=True)
        raw = torch.tanh(raw)
        scale = raw.new_tensor((w, h)).view(1, 2, 1, 1)
        local_flow = raw * scale * self.max_displacement_ratio
        if self.global_pose_head is None:
            return local_flow
        return self.global_pose_head(e3, local_flow)
