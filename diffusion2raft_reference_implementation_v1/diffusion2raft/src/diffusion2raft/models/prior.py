"""Single-image smooth geometry prior for document dewarping."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
        return raw * scale * self.max_displacement_ratio
