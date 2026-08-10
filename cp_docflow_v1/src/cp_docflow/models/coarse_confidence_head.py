"""Canonical import for deterministic coarse-map/confidence prediction."""

from .coarse import DeterministicCoarseRectifier

CoarseConfidenceHead = DeterministicCoarseRectifier

__all__ = ["CoarseConfidenceHead", "DeterministicCoarseRectifier"]

