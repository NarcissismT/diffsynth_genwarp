"""Canonical unified DocGrid-Flow model from the newest specification."""

from __future__ import annotations

import time as wall_time
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..checkpoint import COORDINATE_CONTRACT
from ..geometry import (
    canonical_backward_map,
    canonical_backward_map_window,
    rescale_source_pixel_map,
    resize_backward_map,
    resize_backward_map_with_mask,
    resize_backward_map_window,
    resize_backward_map_window_with_mask,
    warp_with_backward_map,
)
from .coarse import ConvNormAct, DeterministicCoarseRectifier, ResidualBlock
from .coordinate_flow_transformer import (
    ResidualCoordinateFlowTransformer,
    build_residual_proposal_target,
    confidence_gate,
    deterministic_noise_like,
    flow_matching_state,
    residual_noise_start,
)
from .qwen_feature_probe import QwenDPTFPNAdapter, build_qwen_feature_source
from .warr import ConvexMapUpsampler, HighResolutionMapRefiner


def _target_scale_size(size: Sequence[int], stride: int) -> tuple[int, int]:
    height, width = int(size[0]), int(size[1])
    return (
        max(1, (height + stride - 1) // stride),
        max(1, (width + stride - 1) // stride),
    )


def _map_to_feature_grid(
    backward_map: Tensor,
    feature: Tensor,
    source_size: tuple[int, int],
) -> Tensor:
    return resize_backward_map(
        backward_map,
        backward_map.shape[-2:],
        source_size_from=source_size,
        source_size_to=feature.shape[-2:],
    )


def _resolve_target_geometry(
    batch: int,
    final_size: tuple[int, int],
    *,
    target_canvas_size: Sequence[int] | Tensor | None,
    target_window: Tensor | None,
    device: torch.device,
) -> tuple[tuple[int, int], Tensor]:
    if target_canvas_size is None:
        canvas_size = final_size
    elif isinstance(target_canvas_size, Tensor):
        value = target_canvas_size.detach().to("cpu")
        if value.ndim == 1:
            value = value[None].expand(batch, -1)
        if value.shape != (batch, 2) or not bool((value == value[:1]).all()):
            raise ValueError("target_canvas_size must be one shared [H,W] per batch")
        canvas_size = (int(value[0, 0]), int(value[0, 1]))
    else:
        canvas_size = (int(target_canvas_size[0]), int(target_canvas_size[1]))
    if min(canvas_size) < 1:
        raise ValueError("target_canvas_size must be positive")
    if target_window is None:
        window = torch.tensor(
            (0.0, 0.0, float(canvas_size[1]), float(canvas_size[0])),
            device=device,
        )[None].expand(batch, -1)
    else:
        window = target_window.to(device=device, dtype=torch.float32)
        if window.ndim == 1:
            window = window[None].expand(batch, -1)
        if window.shape != (batch, 4):
            raise ValueError("target_window must be [B,4] in x0,y0,width,height order")
    return canvas_size, window


def _sample_full_feature_for_window(
    feature: Tensor,
    output_size: tuple[int, int],
    *,
    source_size: tuple[int, int],
    target_canvas_size: tuple[int, int],
    target_window: Tensor,
) -> Tensor:
    canonical_source = canonical_backward_map_window(
        feature.shape[0],
        output_size,
        source_size,
        target_canvas_size,
        target_window,
        device=feature.device,
        dtype=torch.float32,
    )
    feature_map = rescale_source_pixel_map(
        canonical_source, source_size, feature.shape[-2:]
    )
    return warp_with_backward_map(feature, feature_map, padding_mode="border")


class HVStructureEncoder(nn.Module):
    """Predict horizontal, vertical, and page-boundary structure features."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        if channels < 4:
            raise ValueError("hv channels must be at least four")
        self.channels = int(channels)
        self.stem = nn.Sequential(
            ConvNormAct(5, channels, kernel_size=5, stride=2),
            ResidualBlock(channels),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(channels, channels, stride=2), ResidualBlock(channels)
        )
        self.stage8 = nn.Sequential(
            ConvNormAct(channels, channels, stride=2), ResidualBlock(channels)
        )
        self.structure_head = nn.Sequential(
            ConvNormAct(channels, channels), nn.Conv2d(channels, 3, 3, padding=1)
        )

    def forward(
        self, image: Tensor, *, output_size: tuple[int, int]
    ) -> dict[str, Tensor]:
        gray = image.float().mean(dim=1, keepdim=True)
        horizontal_response = F.pad(
            gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1)
        )
        vertical_response = F.pad(
            gray[..., 1:] - gray[..., :-1], (0, 1, 0, 0)
        )
        feature2 = self.stem(
            torch.cat((image, horizontal_response, vertical_response), dim=1)
        )
        feature4 = self.stage4(feature2)
        feature8 = self.stage8(feature4)
        logits4 = self.structure_head(feature4).float()
        logits = F.interpolate(
            logits4, output_size, mode="bilinear", align_corners=False
        )
        return {
            "hv4": feature4,
            "hv8": feature8,
            "structure_logits4": logits4,
            "structure_logits": logits,
            "horizontal_probability": torch.sigmoid(logits[:, 0:1]),
            "vertical_probability": torch.sigmoid(logits[:, 1:2]),
            "boundary_probability": torch.sigmoid(logits[:, 2:3]),
        }


class GatedMultiScaleFusion(nn.Module):
    """Three-way Qwen/CNN/HV projection with spatial softmax weights."""

    def __init__(
        self,
        cnn_channels: int,
        qwen_channels: int,
        hv_channels: int,
        output_channels: int,
        mode: str = "gated",
    ) -> None:
        super().__init__()
        normalized = str(mode).lower()
        if normalized not in {"cnn_only", "qwen_only", "concat", "gated"}:
            raise ValueError("fusion mode must be cnn_only, qwen_only, concat, or gated")
        self.mode = normalized
        self.cnn = ConvNormAct(cnn_channels, output_channels, kernel_size=1)
        self.qwen = ConvNormAct(qwen_channels, output_channels, kernel_size=1)
        self.hv = ConvNormAct(hv_channels, output_channels, kernel_size=1)
        self.concat = nn.Sequential(
            ConvNormAct(3 * output_channels, output_channels, kernel_size=1),
            ResidualBlock(output_channels),
        )
        self.weight_logits = nn.Sequential(
            ConvNormAct(3 * output_channels, output_channels),
            nn.Conv2d(output_channels, 3, 3, padding=1),
        )

    def forward(
        self,
        cnn_feature: Tensor,
        qwen_feature: Tensor | None,
        hv_feature: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cnn = self.cnn(cnn_feature)
        hv = self.hv(hv_feature)
        qwen_available = qwen_feature is not None
        qwen = self.qwen(qwen_feature) if qwen_available else torch.zeros_like(cnn)
        batch, _, height, width = cnn.shape
        if self.mode == "cnn_only":
            weights = cnn.new_zeros(batch, 3, height, width)
            weights[:, 1] = 1.0
            return cnn, weights
        if self.mode == "qwen_only":
            if not qwen_available:
                raise ValueError("qwen_only fusion requires a Qwen feature source")
            weights = cnn.new_zeros(batch, 3, height, width)
            weights[:, 0] = 1.0
            return qwen, weights
        stacked = torch.cat((qwen, cnn, hv), dim=1)
        if self.mode == "concat":
            weights = cnn.new_full((batch, 3, height, width), 1.0 / 3.0)
            if not qwen_available:
                weights[:, 0] = 0.0
                weights[:, 1:] = 0.5
            return self.concat(stacked), weights
        logits = self.weight_logits(stacked)
        if not qwen_available:
            logits = logits.clone()
            logits[:, 0] = -10_000.0
        weights = torch.softmax(logits.float(), dim=1)
        fused = (
            weights[:, 0:1] * qwen
            + weights[:, 1:2] * cnn
            + weights[:, 2:3] * hv
        )
        return fused, weights


# Compatibility name retained for existing imports.
ConfidenceGatedCondition = GatedMultiScaleFusion


class CPDocFlow(nn.Module):
    """Predict a final absolute backward map from one warped document image.

    The only image condition is ``warped_image``.  ``target_map`` and
    ``valid_mask`` are used solely to construct residual Flow-Matching targets.
    """

    def __init__(
        self,
        *,
        coarse: dict[str, Any] | None = None,
        qwen_backend: str = "none",
        qwen: dict[str, Any] | None = None,
        qwen_feature_channels: int = 96,
        instantiate_qwen_adapter: bool = True,
        fusion_channels: int | None = None,
        fusion_mode: str = "gated",
        hv_channels: int = 32,
        velocity_hidden_channels: int = 256,
        velocity_time_channels: int = 256,
        flow_blocks: int = 8,
        flow_heads: int = 8,
        flow_window_size: int = 8,
        flow_global_pool_size: int = 8,
        max_velocity_px: float = 64.0,
        minimum_residual_gate: float = 0.25,
        residual_clip_px: float = 64.0,
        sigma_min: float = 0.0,
        sigma_max: float = 4.0,
        composition_uses_confidence: bool = True,
        detach_confidence_for_flow: bool = True,
        fm_steps: int = 4,
        enable_flow_matching: bool = True,
        enable_refiner: bool = True,
        enable_hv_condition: bool = True,
        use_qwen_condition: bool = True,
        refiner_hidden_channels: int = 128,
        refiner_iterations: int = 4,
        refiner_max_step_px: float = 2.0,
        convex_hidden_channels: int = 128,
        upsampling_mode: str = "convex",
        inference_seed: int = 0,
        anchor_strength: float | None = None,
        confidence_preserve_strength: float | None = None,
    ) -> None:
        super().__init__()
        del anchor_strength, confidence_preserve_strength
        if fm_steps < 1:
            raise ValueError("fm_steps must be positive")
        if sigma_min < 0.0 or sigma_max < sigma_min:
            raise ValueError("require 0 <= sigma_min <= sigma_max")
        coarse_config = dict(coarse or {})
        self.coarse = DeterministicCoarseRectifier(**coarse_config)
        coarse_channels = int(coarse_config.get("feature_channels", 128))
        if coarse_channels < 2:
            raise ValueError("coarse feature_channels must be at least two")
        fused_channels = coarse_channels if fusion_channels is None else int(fusion_channels)
        if fused_channels != coarse_channels:
            raise ValueError(
                "fusion_channels must equal coarse.feature_channels so the unified "
                "coarse/confidence heads share deterministic initialization"
            )
        self.qwen_backend = str(qwen_backend).lower()
        self.qwen_config = dict(qwen or {})
        self.qwen_source = build_qwen_feature_source(self.qwen_backend, self.qwen_config)
        self.qwen_adapter: QwenDPTFPNAdapter | None
        if self.qwen_source is None or not bool(instantiate_qwen_adapter):
            self.qwen_adapter = None
        else:
            self.qwen_adapter = QwenDPTFPNAdapter(
                input_channels=int(self.qwen_source.hidden_channels),
                feature_channels=int(qwen_feature_channels),
                layer_count=int(self.qwen_source.layer_count),
            )
        self.hv = HVStructureEncoder(hv_channels)
        self.fusion = GatedMultiScaleFusion(
            coarse_channels,
            int(qwen_feature_channels),
            int(hv_channels),
            fused_channels,
            mode=fusion_mode,
        )
        self.velocity = ResidualCoordinateFlowTransformer(
            fused_channels,
            int(hv_channels),
            hidden_channels=int(velocity_hidden_channels),
            time_channels=int(velocity_time_channels),
            blocks=int(flow_blocks),
            heads=int(flow_heads),
            window_size=int(flow_window_size),
            global_pool_size=int(flow_global_pool_size),
            max_velocity_px=float(max_velocity_px),
            minimum_gate=float(minimum_residual_gate),
        )
        self.refiner = HighResolutionMapRefiner(
            source_feature_channels=coarse_channels // 2,
            hv_feature_channels=int(hv_channels),
            hidden_channels=int(refiner_hidden_channels),
            iterations=int(refiner_iterations),
            max_step_px=float(refiner_max_step_px),
        )
        self.convex_context = nn.Sequential(
            ConvNormAct(coarse_channels // 2 + int(hv_channels), refiner_hidden_channels),
            ResidualBlock(refiner_hidden_channels),
        )
        self.convex_upsampler = ConvexMapUpsampler(
            int(refiner_hidden_channels),
            hidden_channels=int(convex_hidden_channels),
            scale=4,
        )
        self.upsampling_mode = str(upsampling_mode).lower()
        if self.upsampling_mode not in {"convex", "bilinear"}:
            raise ValueError("upsampling_mode must be convex or bilinear")
        self.minimum_residual_gate = float(minimum_residual_gate)
        self.residual_clip_px = float(residual_clip_px)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.composition_uses_confidence = bool(composition_uses_confidence)
        self.detach_confidence_for_flow = bool(detach_confidence_for_flow)
        self.fm_steps = int(fm_steps)
        self.enable_flow_matching = bool(enable_flow_matching)
        self.enable_refiner = bool(enable_refiner)
        self.enable_hv_condition = bool(enable_hv_condition)
        self.use_qwen_condition = bool(use_qwen_condition)
        self.instantiate_qwen_adapter = bool(instantiate_qwen_adapter)
        self.inference_seed = int(inference_seed)

    def set_execution_stage(self, stage: str) -> None:
        """Apply runtime topology for staged training/checkpoint inference."""

        normalized = str(stage).lower()
        aliases = {"refiner": "warr", "joint": "full_page"}
        normalized = aliases.get(normalized, normalized)
        if normalized == "coarse":
            self.enable_flow_matching = False
            self.enable_refiner = False
            self.enable_hv_condition = False
            self.use_qwen_condition = False
        elif normalized == "warr":
            self.enable_flow_matching = False
            self.enable_refiner = True
            self.enable_hv_condition = True
            self.use_qwen_condition = False
        elif normalized == "coord_fm":
            self.enable_flow_matching = True
            self.enable_refiner = True
            self.enable_hv_condition = True
            self.use_qwen_condition = False
        elif normalized in {"qwen", "full_page"}:
            self.enable_flow_matching = True
            self.enable_refiner = True
            self.enable_hv_condition = True
            self.use_qwen_condition = (
                self.qwen_source is not None and self.qwen_adapter is not None
            )
        else:
            raise ValueError(
                "execution stage must be coarse, warr, coord_fm, qwen, or full_page"
            )

    def _qwen_condition(
        self,
        warped_image: Tensor,
        output_size: tuple[int, int],
        *,
        target_canvas_size: tuple[int, int],
        target_window: Tensor,
    ) -> Tensor | None:
        if (
            not self.use_qwen_condition
            or self.qwen_source is None
            or self.qwen_adapter is None
        ):
            return None
        full_size = _target_scale_size(target_canvas_size, 8)
        full_feature = self.qwen_adapter(self.qwen_source(warped_image), full_size)
        return _sample_full_feature_for_window(
            full_feature,
            output_size,
            source_size=target_canvas_size,
            target_canvas_size=target_canvas_size,
            target_window=target_window,
        )

    def _predict_coarse_low(
        self,
        fused: Tensor,
        *,
        low_size: tuple[int, int],
        source_size: tuple[int, int],
        canonical_map: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if fused.shape[-2:] != low_size:
            fused = F.interpolate(fused, low_size, mode="bilinear", align_corners=False)
        raw_residual = self.coarse.map_head(fused).float()
        source_h, source_w = source_size
        scale = raw_residual.new_tensor((source_w, source_h)).view(1, 2, 1, 1)
        scale = scale * self.coarse.max_displacement_ratio
        residual = torch.tanh(raw_residual) * scale
        canonical = (
            canonical_backward_map(
                fused.shape[0],
                low_size,
                source_size,
                device=fused.device,
                dtype=torch.float32,
            )
            if canonical_map is None
            else canonical_map.float()
        )
        if canonical.shape != raw_residual.shape:
            raise ValueError("coarse canonical_map and residual grids differ")
        log_variance = self.coarse.log_variance_head(fused).float().clamp(
            self.coarse.min_log_variance, self.coarse.max_log_variance
        )
        confidence = -torch.expm1(-0.5 * torch.exp(-log_variance))
        return {
            "map": canonical + residual,
            "canonical": canonical,
            "residual": residual,
            "log_variance": log_variance,
            "confidence": confidence,
        }

    def _velocity_call(
        self,
        residual_state: Tensor,
        time: Tensor,
        *,
        coarse_map: Tensor,
        confidence: Tensor,
        canonical_map: Tensor,
        visual: Tensor,
        hv: Tensor,
        source_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        return self.velocity(
            residual_state,
            time,
            coarse_map=coarse_map,
            confidence=confidence,
            canonical_map=canonical_map,
            visual_condition=visual,
            hv_condition=hv,
            source_size=source_size,
        )

    def forward(
        self,
        warped_image: Tensor,
        *,
        output_size: Sequence[int] | None = None,
        target_map: Tensor | None = None,
        valid_mask: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        render: bool = True,
        target_canvas_size: Sequence[int] | Tensor | None = None,
        target_window: Tensor | None = None,
        profile: bool = False,
    ) -> dict[str, Any]:
        if warped_image.ndim != 4 or warped_image.shape[1] != 3:
            raise ValueError("warped_image must be [B,3,H,W]")
        source_size = tuple(int(value) for value in warped_image.shape[-2:])
        final_size = source_size if output_size is None else (
            int(output_size[0]), int(output_size[1])
        )
        if min(final_size) < 4:
            raise ValueError("output_size dimensions must be at least four")
        canvas_size, window = _resolve_target_geometry(
            warped_image.shape[0],
            final_size,
            target_canvas_size=target_canvas_size,
            target_window=target_window,
            device=warped_image.device,
        )
        runtime_breakdown: dict[str, float] = {}

        def timestamp() -> float:
            if profile and warped_image.device.type == "cuda":
                torch.cuda.synchronize(warped_image.device)
            return wall_time.perf_counter()

        native_started = timestamp()
        cnn = self.coarse.encode(warped_image)
        hv = self.hv(warped_image, output_size=source_size)
        hv_logits = _sample_full_feature_for_window(
            hv["structure_logits"],
            final_size,
            source_size=source_size,
            target_canvas_size=canvas_size,
            target_window=window,
        ).float()
        if profile:
            runtime_breakdown["native_geometry_seconds"] = timestamp() - native_started
        low_size = _target_scale_size(final_size, 8)
        cnn8 = _sample_full_feature_for_window(
            cnn["fpn8"],
            low_size,
            source_size=source_size,
            target_canvas_size=canvas_size,
            target_window=window,
        )
        hv8 = _sample_full_feature_for_window(
            hv["hv8"],
            low_size,
            source_size=source_size,
            target_canvas_size=canvas_size,
            target_window=window,
        )
        hv8_condition = (
            hv8 if self.enable_hv_condition else torch.zeros_like(hv8)
        )
        hv4_condition = (
            hv["hv4"]
            if self.enable_hv_condition
            else torch.zeros_like(hv["hv4"])
        )
        qwen_started = timestamp()
        qwen_feature = self._qwen_condition(
            warped_image,
            low_size,
            target_canvas_size=canvas_size,
            target_window=window,
        )
        if profile:
            runtime_breakdown["qwen_feature_probe_seconds"] = timestamp() - qwen_started
        coarse_started = timestamp()
        fused, fusion_weights = self.fusion(cnn8, qwen_feature, hv8_condition)
        canonical_low = canonical_backward_map_window(
            warped_image.shape[0],
            low_size,
            source_size,
            canvas_size,
            window,
            device=warped_image.device,
            dtype=torch.float32,
        )
        coarse_low_output = self._predict_coarse_low(
            fused,
            low_size=low_size,
            source_size=source_size,
            canonical_map=canonical_low,
        )
        coarse_low = coarse_low_output["map"]
        confidence_low = coarse_low_output["confidence"]
        canonical_low = coarse_low_output["canonical"]
        coarse_full = resize_backward_map_window(
            coarse_low,
            final_size,
            source_size_from=source_size,
            source_size_to=source_size,
            target_canvas_size=canvas_size,
            target_window=window,
        )
        confidence_full = F.interpolate(
            confidence_low, final_size, mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        log_variance_full = F.interpolate(
            coarse_low_output["log_variance"],
            final_size,
            mode="bilinear",
            align_corners=False,
        )
        if profile:
            runtime_breakdown["fusion_and_coarse_seconds"] = timestamp() - coarse_started

        coordinate_started = timestamp()
        training_fm: dict[str, Tensor] = {}
        if self.enable_flow_matching:
            if noise is None:
                residual_noise = (
                    torch.randn_like(coarse_low, dtype=torch.float32)
                    if target_map is not None
                    else deterministic_noise_like(coarse_low, self.inference_seed)
                )
            else:
                residual_noise = F.interpolate(
                    noise.float(), low_size, mode="bilinear", align_corners=False
                )
            residual_start = residual_noise_start(
                confidence_low,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                noise=residual_noise,
                detach_confidence=self.detach_confidence_for_flow,
            )
            if target_map is not None:
                if valid_mask is None:
                    raise ValueError("valid_mask is required with target_map")
                target_low, valid_low = resize_backward_map_window_with_mask(
                    target_map.float(),
                    valid_mask.bool(),
                    low_size,
                    source_size_from=source_size,
                    source_size_to=source_size,
                    target_canvas_size=canvas_size,
                    target_window=window,
                )
                raw_residual, proposal_target, proposal_gate = (
                    build_residual_proposal_target(
                        target_low,
                        coarse_low,
                        confidence_low,
                        minimum_gate=self.minimum_residual_gate,
                        residual_clip_px=self.residual_clip_px,
                        detach_confidence=self.detach_confidence_for_flow,
                    )
                )
                sampled_time = (
                    torch.rand(
                        warped_image.shape[0],
                        device=warped_image.device,
                        dtype=torch.float32,
                    )
                    if time is None
                    else time.float()
                )
                residual_state, velocity_target = flow_matching_state(
                    residual_start, proposal_target, sampled_time
                )
                velocity_prediction = self._velocity_call(
                    residual_state,
                    sampled_time,
                    coarse_map=coarse_low,
                    confidence=confidence_low,
                    canonical_map=canonical_low,
                    visual=fused,
                    hv=hv8_condition,
                    source_size=source_size,
                )
                training_fm = {
                    "flow_matching_state": residual_state,
                    "flow_matching_time": sampled_time,
                    "velocity_prediction": velocity_prediction["velocity"],
                    "velocity_target": velocity_target,
                    "residual_target": raw_residual,
                    "residual_proposal_target": proposal_target,
                    "residual_gate": proposal_gate,
                    "flow_matching_target_map": target_low,
                    "flow_matching_valid_mask": valid_low,
                }
            composition_gate = (
                confidence_gate(
                    confidence_low.detach()
                    if self.detach_confidence_for_flow
                    else confidence_low,
                    self.minimum_residual_gate,
                )
                if self.composition_uses_confidence
                else torch.ones_like(confidence_low)
            )
            residual_current = residual_start
            flow_matching_residual_sequence: list[Tensor] = []
            flow_matching_map_sequence: list[Tensor] = []
            for step in range(self.fm_steps):
                step_time = torch.full(
                    (warped_image.shape[0],),
                    step / self.fm_steps,
                    device=warped_image.device,
                    dtype=torch.float32,
                )
                velocity = self._velocity_call(
                    residual_current,
                    step_time,
                    coarse_map=coarse_low,
                    confidence=confidence_low,
                    canonical_map=canonical_low,
                    visual=fused,
                    hv=hv8_condition,
                    source_size=source_size,
                )["velocity"]
                residual_current = residual_current + velocity / self.fm_steps
                flow_matching_residual_sequence.append(residual_current)
                # Every public FM map is an absolute backward map on the low
                # target grid, with values in source-image pixel coordinates.
                flow_matching_map_sequence.append(
                    coarse_low + composition_gate * residual_current
                )
            residual_proposal = residual_current
            composed_low = flow_matching_map_sequence[-1]
        else:
            residual_start = torch.zeros_like(coarse_low)
            residual_proposal = torch.zeros_like(coarse_low)
            flow_matching_residual_sequence = []
            flow_matching_map_sequence = []
            composition_gate = torch.ones_like(confidence_low)
            composed_low = coarse_low
        if profile:
            runtime_breakdown["coordinate_ode_seconds"] = timestamp() - coordinate_started

        warr_started = timestamp()
        quarter_size = _target_scale_size(final_size, 4)
        canonical_quarter = canonical_backward_map_window(
            warped_image.shape[0],
            quarter_size,
            source_size,
            canvas_size,
            window,
            device=warped_image.device,
            dtype=torch.float32,
        )
        composed_quarter = resize_backward_map_window(
            composed_low,
            quarter_size,
            source_size_from=source_size,
            source_size_to=source_size,
            target_canvas_size=canvas_size,
            target_window=window,
        )
        if self.enable_refiner:
            refined = self.refiner(
                composed_quarter,
                source_feature=cnn["local4"],
                hv_feature=hv4_condition,
                source_size=source_size,
                canonical_map=canonical_quarter,
            )
            refined_map = refined["backward_map"]
            refiner_sequence = refined["sequence"]
            refiner_deltas = refined["deltas"]
            refiner_gates = refined["update_gates"]
            convex_context = refined["hidden"]
        else:
            refined_map = composed_quarter
            refiner_sequence = []
            refiner_deltas = []
            refiner_gates = []
            local_map = _map_to_feature_grid(
                refined_map, cnn["local4"], source_size
            )
            hv4_map = _map_to_feature_grid(refined_map, hv4_condition, source_size)
            warped_local = warp_with_backward_map(
                cnn["local4"], local_map, padding_mode="border"
            )
            warped_hv = warp_with_backward_map(
                hv4_condition, hv4_map, padding_mode="border"
            )
            convex_context = self.convex_context(
                torch.cat((warped_local, warped_hv), dim=1)
            )
        if profile:
            runtime_breakdown["warr_seconds"] = timestamp() - warr_started
        convex_started = timestamp()
        canonical_final = canonical_backward_map_window(
            warped_image.shape[0],
            final_size,
            source_size,
            canvas_size,
            window,
            device=warped_image.device,
            dtype=torch.float32,
        )
        if self.upsampling_mode == "convex":
            convex = self.convex_upsampler(
                refined_map,
                convex_context,
                output_size=final_size,
                source_size=source_size,
                canonical_low=canonical_quarter,
                canonical_high=canonical_final,
            )
        else:
            bilinear_map = resize_backward_map_window(
                refined_map,
                final_size,
                source_size_from=source_size,
                source_size_to=source_size,
                target_canvas_size=canvas_size,
                target_window=window,
            )
            convex = {
                "backward_map": bilinear_map,
                "convex_mask": bilinear_map.new_empty(0),
                "displacement": bilinear_map - canonical_final,
                "canonical_high": canonical_final,
            }
        final_map = convex["backward_map"]
        if profile:
            runtime_breakdown["convex_upsampling_seconds"] = timestamp() - convex_started
        output: dict[str, Any] = {
            "backward_map": final_map,
            "final_backward_map": final_map,
            "coarse_backward_map": coarse_full,
            "coarse_log_variance": log_variance_full,
            "confidence": confidence_full,
            "canonical_map": canonical_final,
            "target_canvas_size": canvas_size,
            "target_window": window,
            "coarse_low": coarse_low,
            "flow_start_map": residual_start,
            "residual_start": residual_start,
            "residual_proposal": residual_proposal,
            "flow_matching_residual_sequence": flow_matching_residual_sequence,
            # Compatibility alias: residual states, not absolute maps.
            "residual_sequence": flow_matching_residual_sequence,
            "flow_matching_map": composed_low,
            "composed_map": composed_low,
            "composition_gate": composition_gate,
            "flow_matching_map_sequence": flow_matching_map_sequence,
            "map_sequence_coordinate_contract": COORDINATE_CONTRACT,
            # Compatibility alias: absolute source-pixel backward maps.
            "flow_matching_sequence": flow_matching_map_sequence,
            "refiner_sequence": refiner_sequence,
            "refiner_deltas": refiner_deltas,
            "refiner_update_gates": refiner_gates,
            "convex_mask": convex["convex_mask"],
            "upsampling_mode": self.upsampling_mode,
            "fusion_weights": fusion_weights,
            "qwen_gate": fusion_weights[:, 0:1],
            "hv_logits": hv_logits,
            "horizontal_probability": torch.sigmoid(hv_logits[:, 0:1]),
            "vertical_probability": torch.sigmoid(hv_logits[:, 1:2]),
            "boundary_probability": torch.sigmoid(hv_logits[:, 2:3]),
            "qwen_backend": self.qwen_backend if self.use_qwen_condition else "none",
            **training_fm,
        }
        if profile:
            output["runtime_breakdown"] = runtime_breakdown
        if render:
            rectified, valid = warp_with_backward_map(
                warped_image.float(),
                final_map,
                padding_mode="border",
                return_valid=True,
            )
            output["rectified_image"] = rectified
            output["valid_mask"] = valid
        return output
