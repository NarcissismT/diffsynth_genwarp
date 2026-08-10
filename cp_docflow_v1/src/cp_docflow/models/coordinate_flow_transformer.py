"""Canonical confidence-protected residual Coordinate Flow Transformer.

The transported state is a residual proposal, never an absolute backward map:

``Mc -> R_gt/G(C) -> residual ODE -> R_hat -> Mc + G(C)*R_hat``.

All map values remain source-pixel coordinates.  Only inputs to neural layers
are normalized for numerical conditioning.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sinusoidal_time_embedding(time: Tensor, channels: int) -> Tensor:
    if time.ndim != 1:
        raise ValueError("time must be [B]")
    half = channels // 2
    if half < 1:
        raise ValueError("time embedding needs at least two channels")
    exponent = -math.log(10_000.0) * torch.arange(
        half, device=time.device, dtype=time.dtype
    ) / max(half - 1, 1)
    phase = time[:, None] * torch.exp(exponent)[None]
    embedding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
    return F.pad(embedding, (0, channels - embedding.shape[1]))


def normalize_pixel_map(backward_map: Tensor, source_size: tuple[int, int]) -> Tensor:
    source_h, source_w = (int(value) for value in source_size)
    result = backward_map.float().clone()
    result[:, 0] = (2.0 * result[:, 0] + 1.0) / source_w - 1.0
    result[:, 1] = (2.0 * result[:, 1] + 1.0) / source_h - 1.0
    return result


def normalize_pixel_delta(delta: Tensor, source_size: tuple[int, int]) -> Tensor:
    source_h, source_w = (int(value) for value in source_size)
    scale = delta.new_tensor((2.0 / source_w, 2.0 / source_h)).view(1, 2, 1, 1)
    return delta.float() * scale


def confidence_gate(confidence: Tensor, minimum: float) -> Tensor:
    if not 0.0 < float(minimum) <= 1.0:
        raise ValueError("minimum confidence gate must be in (0,1]")
    return float(minimum) + (1.0 - float(minimum)) * (1.0 - confidence.float())


def build_residual_proposal_target(
    target_map: Tensor,
    coarse_map: Tensor,
    confidence: Tensor,
    *,
    minimum_gate: float,
    residual_clip_px: float,
    detach_confidence: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return raw residual, confidence-consistent proposal target, and gate."""

    if target_map.shape != coarse_map.shape:
        raise ValueError("target_map and coarse_map shapes differ")
    confidence_for_flow = confidence.detach() if detach_confidence else confidence
    gate = confidence_gate(confidence_for_flow, minimum_gate)
    residual = target_map.float() - coarse_map.float()
    proposal = residual / gate.clamp_min(1.0e-6)
    if residual_clip_px <= 0.0:
        raise ValueError("residual_clip_px must be positive")
    proposal = proposal.clamp(-float(residual_clip_px), float(residual_clip_px))
    return residual, proposal, gate


def residual_noise_start(
    confidence: Tensor,
    *,
    sigma_min: float,
    sigma_max: float,
    noise: Tensor,
    detach_confidence: bool = True,
) -> Tensor:
    if noise.ndim != 4 or noise.shape[1] != 2:
        raise ValueError("residual noise must be [B,2,H,W]")
    if confidence.shape != noise[:, :1].shape:
        raise ValueError("confidence and residual noise grids differ")
    if sigma_min < 0.0 or sigma_max < sigma_min:
        raise ValueError("require 0 <= sigma_min <= sigma_max")
    value = confidence.detach() if detach_confidence else confidence
    amplitude = float(sigma_min) + (float(sigma_max) - float(sigma_min)) * (
        1.0 - value.float()
    )
    return amplitude * noise.float()


def confidence_protected_start(
    coarse_map: Tensor,
    confidence: Tensor,
    *,
    sigma_max: float,
    noise: Tensor,
) -> Tensor:
    """Backward-compatible absolute-map helper used by older call sites/tests."""

    return coarse_map.float() + residual_noise_start(
        confidence,
        sigma_min=0.0,
        sigma_max=sigma_max,
        noise=noise,
    )


def flow_matching_state(
    start_state: Tensor,
    target_state: Tensor,
    time: Tensor,
) -> tuple[Tensor, Tensor]:
    if start_state.shape != target_state.shape:
        raise ValueError("flow-matching endpoint shapes differ")
    if time.shape != (start_state.shape[0],):
        raise ValueError("time must be [B]")
    weight = time[:, None, None, None].float()
    target_velocity = target_state.float() - start_state.float()
    state = (1.0 - weight) * start_state.float() + weight * target_state.float()
    return state, target_velocity


@torch.no_grad()
def deterministic_noise_like(value: Tensor, seed: int) -> Tensor:
    generator = torch.Generator(device=value.device).manual_seed(int(seed))
    return torch.randn(
        value.shape, device=value.device, dtype=torch.float32, generator=generator
    )


def _window_partition(value: Tensor, window_size: int) -> tuple[Tensor, tuple[int, ...]]:
    batch, channels, height, width = value.shape
    window = min(int(window_size), height, width)
    pad_h = (window - height % window) % window
    pad_w = (window - width % window) % window
    padded = F.pad(value, (0, pad_w, 0, pad_h))
    padded_h, padded_w = padded.shape[-2:]
    tokens = (
        padded.view(batch, channels, padded_h // window, window, padded_w // window, window)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(-1, window * window, channels)
    )
    return tokens, (batch, channels, height, width, padded_h, padded_w, window)


def _window_unpartition(tokens: Tensor, metadata: tuple[int, ...]) -> Tensor:
    batch, channels, height, width, padded_h, padded_w, window = metadata
    value = (
        tokens.view(batch, padded_h // window, padded_w // window, window, window, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, padded_h, padded_w)
    )
    return value[..., :height, :width]


class CoordinateFlowBlock(nn.Module):
    """AdaLN -> local/global self-attn -> visual cross-attn -> H/V FFN."""

    def __init__(
        self,
        channels: int,
        heads: int,
        time_channels: int,
        hv_channels: int,
        *,
        window_size: int = 8,
        global_pool_size: int = 8,
        minimum_gate: float = 0.25,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("transformer channels must be divisible by heads")
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.global_pool_size = int(global_pool_size)
        self.minimum_gate = float(minimum_gate)
        self.norm_self = nn.LayerNorm(channels, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(channels, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(channels, elementwise_affine=False)
        self.local_attention = nn.MultiheadAttention(
            channels, heads, batch_first=True
        )
        self.global_attention = nn.MultiheadAttention(
            channels, heads, batch_first=True
        )
        self.visual_attention = nn.MultiheadAttention(
            channels, heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        self.time_modulation = nn.Linear(time_channels, 6 * channels)
        self.hv_gate = nn.Sequential(
            nn.Conv2d(hv_channels, channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        # AdaLN-Zero style initialization keeps a newly inserted CFT close to
        # a no-op until velocity supervision begins to shape it.
        nn.init.zeros_(self.time_modulation.weight)
        nn.init.zeros_(self.time_modulation.bias)

    @staticmethod
    def _modulate(tokens: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return tokens * (1.0 + scale[:, None]) + shift[:, None]

    def forward(
        self,
        feature: Tensor,
        *,
        visual: Tensor,
        hv_feature: Tensor,
        confidence: Tensor,
        time_embedding: Tensor,
    ) -> Tensor:
        batch, channels, height, width = feature.shape
        modulation = self.time_modulation(time_embedding).chunk(6, dim=1)
        shift_self, scale_self, shift_cross, scale_cross, shift_ffn, scale_ffn = modulation
        confidence_residual = confidence_gate(
            confidence.detach(), self.minimum_gate
        )

        local_tokens, metadata = _window_partition(feature, self.window_size)
        repeated_shift = shift_self.repeat_interleave(
            local_tokens.shape[0] // batch, dim=0
        )
        repeated_scale = scale_self.repeat_interleave(
            local_tokens.shape[0] // batch, dim=0
        )
        normalized = self._modulate(
            self.norm_self(local_tokens), repeated_shift, repeated_scale
        )
        local_update = self.local_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        local_update = _window_unpartition(local_update, metadata)

        query = feature.flatten(2).transpose(1, 2)
        query_normalized = self._modulate(
            self.norm_self(query), shift_self, scale_self
        )
        pooled_feature = F.adaptive_avg_pool2d(
            feature,
            (min(self.global_pool_size, height), min(self.global_pool_size, width)),
        ).flatten(2).transpose(1, 2)
        pooled_feature = self.norm_self(pooled_feature)
        global_update = self.global_attention(
            query_normalized, pooled_feature, pooled_feature, need_weights=False
        )[0].transpose(1, 2).reshape(batch, channels, height, width)
        feature = feature + confidence_residual * (local_update + global_update)

        query = feature.flatten(2).transpose(1, 2)
        query = self._modulate(self.norm_cross(query), shift_cross, scale_cross)
        visual_tokens = F.adaptive_avg_pool2d(
            visual,
            (min(self.global_pool_size, height), min(self.global_pool_size, width)),
        ).flatten(2).transpose(1, 2)
        cross_update = self.visual_attention(
            query, visual_tokens, visual_tokens, need_weights=False
        )[0].transpose(1, 2).reshape(batch, channels, height, width)
        feature = feature + confidence_residual * cross_update

        tokens = feature.flatten(2).transpose(1, 2)
        tokens = self._modulate(self.norm_ffn(tokens), shift_ffn, scale_ffn)
        ffn_update = self.ffn(tokens).transpose(1, 2).reshape(
            batch, channels, height, width
        )
        structure_gate = torch.sigmoid(self.hv_gate(hv_feature))
        return feature + structure_gate * ffn_update


class ResidualCoordinateFlowTransformer(nn.Module):
    """Tokenize residual coordinates and predict an instantaneous 2-D velocity."""

    def __init__(
        self,
        visual_channels: int,
        hv_channels: int,
        hidden_channels: int = 256,
        time_channels: int = 256,
        blocks: int = 8,
        heads: int = 8,
        window_size: int = 8,
        global_pool_size: int = 8,
        max_velocity_px: float = 64.0,
        minimum_gate: float = 0.25,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("coordinate transformer blocks must be positive")
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        if max_velocity_px <= 0.0:
            raise ValueError("max_velocity_px must be positive")
        self.time_channels = int(time_channels)
        self.max_velocity_px = float(max_velocity_px)
        # R_t(2), Mc-P(2), canonical P(2), confidence(1).
        self.tokenizer = nn.Sequential(
            nn.Conv2d(7, hidden_channels, 3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        )
        self.position = nn.Conv2d(2, hidden_channels, 1)
        self.visual_projection = nn.Conv2d(visual_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                CoordinateFlowBlock(
                    hidden_channels,
                    heads,
                    time_channels,
                    hv_channels,
                    window_size=window_size,
                    global_pool_size=global_pool_size,
                    minimum_gate=minimum_gate,
                )
                for _ in range(blocks)
            ]
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels),
            nn.SiLU(inplace=True),
            nn.Linear(time_channels, time_channels),
        )
        self.output_norm = nn.GroupNorm(1, hidden_channels)
        self.velocity_head = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    @property
    def head(self) -> nn.Conv2d:
        """Compatibility alias for the previous convolutional decoder tests."""

        return self.velocity_head

    def forward(
        self,
        residual_state: Tensor,
        time: Tensor,
        *,
        coarse_map: Tensor,
        confidence: Tensor,
        canonical_map: Tensor,
        visual_condition: Tensor,
        hv_condition: Tensor,
        source_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        if residual_state.shape != coarse_map.shape or coarse_map.shape != canonical_map.shape:
            raise ValueError("residual/coarse/canonical coordinate grids differ")
        if confidence.shape != residual_state[:, :1].shape:
            raise ValueError("confidence grid differs from residual state")
        if time.shape != (residual_state.shape[0],):
            raise ValueError("time must be [B]")
        spatial = residual_state.shape[-2:]
        if visual_condition.shape[-2:] != spatial or hv_condition.shape[-2:] != spatial:
            raise ValueError("CFT visual/HV conditions must share the residual grid")
        coordinate_input = torch.cat(
            (
                normalize_pixel_delta(residual_state, source_size),
                normalize_pixel_delta(coarse_map - canonical_map, source_size),
                normalize_pixel_map(canonical_map, source_size),
                confidence.float(),
            ),
            dim=1,
        )
        feature = self.tokenizer(coordinate_input)
        feature = feature + self.position(
            normalize_pixel_map(canonical_map, source_size)
        )
        visual = self.visual_projection(visual_condition)
        time_embedding = self.time_mlp(
            sinusoidal_time_embedding(time.float(), self.time_channels)
        )
        for block in self.blocks:
            feature = block(
                feature,
                visual=visual,
                hv_feature=hv_condition,
                confidence=confidence,
                time_embedding=time_embedding,
            )
        raw_velocity = self.velocity_head(F.silu(self.output_norm(feature))).float()
        velocity = torch.tanh(raw_velocity) * self.max_velocity_px
        return {"velocity": velocity, "coordinate_tokens": feature}


# Compatibility name retained for external imports; the implementation is now
# the Transformer mandated by the newest DocGrid-Flow plan.
CoordinateVelocityDecoder = ResidualCoordinateFlowTransformer
