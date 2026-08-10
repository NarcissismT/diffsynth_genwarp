"""Canonical entry points for CNN, H/V, and three-way feature fusion."""

from .coarse import DeterministicCoarseRectifier
from .docgrid_flow import GatedMultiScaleFusion, HVStructureEncoder

__all__ = [
    "DeterministicCoarseRectifier",
    "GatedMultiScaleFusion",
    "HVStructureEncoder",
]
