#!/usr/bin/env python3
"""CLI for the isolated oracle-C4 full-geometry solver/capacity grid."""

from __future__ import annotations

import sys

from diffusion2raft.teacher_quarter_turn_diagnostic import main


if __name__ == "__main__":
    main(
        [
            "--canonical-frame-v2",
            "--full-geometry-per-sample",
            "1",
            "--residual-target-iterations-override",
            "12",
            "--full-geometry-solver-iteration-sweep",
            "12",
            "24",
            "--full-geometry-residual-cap-sweep",
            "24",
            "32",
            "40",
            *sys.argv[1:],
        ]
    )
