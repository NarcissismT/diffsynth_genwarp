#!/usr/bin/env python3
"""CLI for the isolated oracle-C4 canonical-frame residual diagnostic."""

from __future__ import annotations

import sys

from diffusion2raft.teacher_quarter_turn_diagnostic import main


if __name__ == "__main__":
    main(["--canonical-frame-v2", *sys.argv[1:]])
