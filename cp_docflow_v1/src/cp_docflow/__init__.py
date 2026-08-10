"""CP-DocFlow: source-faithful document rectification via backward maps.

Model imports are deliberately lazy.  Data/rendering utilities must remain
usable in a lightweight CPU environment without importing the full model tree.
"""

from typing import Any

from .geometry import (
    ALIGN_CORNERS,
    canonical_backward_map,
    resize_backward_map,
    warp_with_backward_map,
)

__all__ = [
    "ALIGN_CORNERS",
    "DeterministicCoarseRectifier",
    "canonical_backward_map",
    "resize_backward_map",
    "warp_with_backward_map",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "DeterministicCoarseRectifier":
        from .models.coarse import DeterministicCoarseRectifier

        return DeterministicCoarseRectifier
    raise AttributeError(name)
