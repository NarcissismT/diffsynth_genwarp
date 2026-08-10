#!/usr/bin/env python3
"""Run the production teacher-capacity evidence generator/verifier."""

from __future__ import annotations

import sys
from pathlib import Path


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from diffusion2raft.teacher_capacity_production import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
