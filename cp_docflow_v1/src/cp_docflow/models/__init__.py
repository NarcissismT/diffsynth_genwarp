"""Model components for CP-DocFlow."""

from .coarse import DeterministicCoarseRectifier
from .docgrid_flow import CPDocFlow

DocGridFlow = CPDocFlow

__all__ = ["CPDocFlow", "DocGridFlow", "DeterministicCoarseRectifier"]
