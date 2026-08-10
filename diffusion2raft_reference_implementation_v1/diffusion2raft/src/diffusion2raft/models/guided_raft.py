"""Diffusion-guide + bounded RAFT residual rectifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..geometry import backward_warp, compose_backward_flows, flow_valid_mask
from .prior import DocumentGeometryPrior


def _load_torchvision_raft(
    size: str,
    *,
    pretrained: bool,
    checkpoint: str | None,
) -> nn.Module:
    try:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except ImportError as exc:
        raise RuntimeError(
            "torchvision optical-flow models are required. Install torchvision>=0.16."
        ) from exc

    size = size.lower()
    if size == "large":
        weights = Raft_Large_Weights.DEFAULT if pretrained and checkpoint is None else None
        raft = raft_large(weights=weights, progress=True)
    elif size == "small":
        weights = Raft_Small_Weights.DEFAULT if pretrained and checkpoint is None else None
        raft = raft_small(weights=weights, progress=True)
    else:
        raise ValueError(f"raft_size must be 'small' or 'large', got {size!r}")

    if checkpoint:
        payload = torch.load(Path(checkpoint), map_location="cpu")
        state = payload.get("model", payload.get("state_dict", payload))
        # Accept both standalone RAFT and a full model with a `raft.` prefix.
        if any(key.startswith("raft.") for key in state):
            state = {key.removeprefix("raft."): value for key, value in state.items() if key.startswith("raft.")}
        missing, unexpected = raft.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "RAFT checkpoint is incompatible: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
    return raft


class GuidedRectificationRAFT(nn.Module):
    """Predict a safe target->source sampling flow.

    1. A warped-only U-Net predicts the global/coarse backward map.
    2. The source is pre-rectified with that map.
    3. RAFT predicts a bounded guide->pre-rectified residual.
    4. The two maps are composed geometrically (not simply added).

    The diffusion-generated image is never returned as the final image and is
    never used as a pixel-level supervision target.
    """

    def __init__(
        self,
        prior: DocumentGeometryPrior,
        raft: nn.Module | None,
        *,
        max_residual_px: float = 32.0,
        guide_dropout_prob: float = 0.10,
        residual_mode: str = "compose",
    ) -> None:
        super().__init__()
        self.prior = prior
        self.raft = raft
        self.max_residual_px = float(max_residual_px)
        self.guide_dropout_prob = float(guide_dropout_prob)
        if residual_mode not in {"compose", "direct"}:
            raise ValueError(
                f"residual_mode must be 'compose' or 'direct', got {residual_mode!r}"
            )
        # 'compose'  : prior pre-rectifies, RAFT predicts a bounded residual,
        #              flows are composed geometrically. This is the method.
        # 'direct'   : README ablation #2 -- RAFT(guide, warped) predicts the
        #              full flow with no prior pre-rectification and no bound.
        #              Kept so the baseline that chases hallucinated guide text
        #              can be measured against the safe composed model.
        self.residual_mode = residual_mode

    def _bound_residual(self, residual: Tensor) -> Tensor:
        if self.max_residual_px <= 0:
            return residual
        limit = self.max_residual_px
        return limit * torch.tanh(residual / limit)

    def forward(
        self,
        warped: Tensor,
        guide: Tensor | None = None,
        *,
        stage: str = "joint",
    ) -> dict[str, Any]:
        prior_flow = self.prior(warped)
        prior_rectified, prior_valid = backward_warp(
            warped,
            prior_flow,
            padding_mode="border",
            return_valid=True,
        )
        output: dict[str, Any] = {
            "prior_flow": prior_flow,
            "prior_rectified": prior_rectified,
            "prior_valid": prior_valid,
        }
        if stage == "prior":
            output.update(
                flows=[prior_flow],
                residuals=[],
                final_flow=prior_flow,
                final_valid=prior_valid,
            )
            return output
        if stage != "joint":
            raise ValueError(f"stage must be 'prior' or 'joint', got {stage!r}")
        if guide is None:
            raise ValueError("guide is required in joint stage")
        if self.raft is None:
            raise RuntimeError("joint stage requested but the RAFT module was not constructed")
        if guide.shape != warped.shape:
            raise ValueError(
                "guide and warped tensors must share shape after preprocessing, got "
                f"{tuple(guide.shape)} and {tuple(warped.shape)}"
            )

        # torchvision RAFT expects tensors normalized to [-1, 1]. Passing the
        # guide first gives the required rectified->source direction.
        if self.residual_mode == "direct":
            # Ablation #2: RAFT(guide, warped) predicts the full flow directly.
            # No pre-rectification, no residual bound, no composition -- this is
            # the baseline the method is meant to beat, so it deliberately keeps
            # every guard off.
            flows = self.raft(2.0 * guide - 1.0, 2.0 * warped - 1.0)
            final_flow = flows[-1]
            output.update(
                flows=flows,
                residuals=[],
                final_flow=final_flow,
                final_valid=flow_valid_mask(final_flow, warped.shape[-2:]),
            )
            return output

        if self.training and self.guide_dropout_prob > 0:
            drop = torch.rand(
                (guide.shape[0], 1, 1, 1), device=guide.device
            ) < self.guide_dropout_prob
            # A dropped guide becomes the current pre-rectified estimate. RAFT
            # then learns a near-zero residual and the system falls back to the
            # warped-only prior instead of chasing hallucinated content.
            guide = torch.where(drop, prior_rectified.detach(), guide)

        residual_predictions = self.raft(2.0 * guide - 1.0, 2.0 * prior_rectified - 1.0)
        residuals = [self._bound_residual(residual) for residual in residual_predictions]
        flows = [compose_backward_flows(prior_flow, residual) for residual in residuals]
        final_flow = flows[-1]
        output.update(
            flows=flows,
            residuals=residuals,
            final_flow=final_flow,
            final_valid=flow_valid_mask(final_flow, warped.shape[-2:]),
        )
        return output


def build_guided_rectifier(model_config: dict[str, Any], *, stage: str) -> GuidedRectificationRAFT:
    prior = DocumentGeometryPrior(
        base_channels=int(model_config.get("prior_base_channels", 32)),
        max_displacement_ratio=float(model_config.get("prior_max_displacement_ratio", 0.35)),
        control_stride=int(model_config.get("prior_control_stride", 8)),
    )
    raft = None
    if stage == "joint":
        raft = _load_torchvision_raft(
            str(model_config.get("raft_size", "large")),
            pretrained=bool(model_config.get("raft_pretrained", True)),
            checkpoint=model_config.get("raft_checkpoint"),
        )
    return GuidedRectificationRAFT(
        prior,
        raft,
        max_residual_px=float(model_config.get("max_residual_px", 32.0)),
        guide_dropout_prob=float(model_config.get("guide_dropout_prob", 0.10)),
        residual_mode=str(model_config.get("residual_mode", "compose")),
    )
