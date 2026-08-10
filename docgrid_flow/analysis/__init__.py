"""Analysis-only code; nothing in this package contains trainable modules."""

from .mmdit_correspondence import (
    EvaluationContext,
    ManifestSample,
    SourceResizeTransform,
    build_evaluation_context,
    evaluate_baselines,
    evaluate_similarity,
    load_config,
    read_manifest,
)

__all__ = [
    "EvaluationContext",
    "ManifestSample",
    "SourceResizeTransform",
    "build_evaluation_context",
    "evaluate_baselines",
    "evaluate_similarity",
    "load_config",
    "read_manifest",
]

