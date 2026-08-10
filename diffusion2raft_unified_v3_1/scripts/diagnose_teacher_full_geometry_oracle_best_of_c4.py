#!/usr/bin/env python3
"""CLI for the isolated full-geometry all-four-C4 boundary audit."""

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
            "--full-geometry-best-of-c4",
            "--c4-candidate-batch-size",
            "1",
            *sys.argv[1:],
        ]
    )
