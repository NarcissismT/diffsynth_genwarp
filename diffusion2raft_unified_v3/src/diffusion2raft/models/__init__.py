from __future__ import annotations

from typing import Any

import torch

from .guided_raft import GuidedRectificationRAFT, build_guided_rectifier
from .unified import UnifiedDocumentRectifier, build_unified_rectifier


def build_rectifier(
    model_config: dict[str, Any],
    qwen_config: dict[str, Any],
    *,
    stage: str,
    device: torch.device | str,
) -> GuidedRectificationRAFT | UnifiedDocumentRectifier:
    """Construct a stage-specific model without loading Qwen for Stage A."""

    if stage in {"prior", "joint"}:
        return build_guided_rectifier(model_config, stage=stage)
    if stage == "unified":
        return build_unified_rectifier(model_config, qwen_config, device=device)
    raise ValueError(f"stage must be 'prior', 'joint', or 'unified', got {stage!r}")


__all__ = [
    "GuidedRectificationRAFT",
    "UnifiedDocumentRectifier",
    "build_guided_rectifier",
    "build_rectifier",
    "build_unified_rectifier",
]
