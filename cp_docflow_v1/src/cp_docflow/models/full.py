"""Compatibility import for the pre-DocGrid-Flow module name.

New code must import :mod:`cp_docflow.models.docgrid_flow`.  This shim keeps
old checkpoints/tests importable without duplicating model implementation.
"""

from .docgrid_flow import (
    CPDocFlow,
    ConfidenceGatedCondition,
    GatedMultiScaleFusion,
    HVStructureEncoder,
)

DocGridFlow = CPDocFlow

__all__ = [
    "CPDocFlow",
    "DocGridFlow",
    "ConfidenceGatedCondition",
    "GatedMultiScaleFusion",
    "HVStructureEncoder",
]
