#!/usr/bin/env python3
"""Print an actionable decision for a raw teacher-capacity audit report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from diffusion2raft.teacher_capacity_policy import (  # noqa: E402
    evaluate_teacher_capacity_policy,
)


_AGGREGATE_NAMES = ("original", "rotation_augmented", "full_geometry_augmented")
_METRIC_NAMES = (
    "oracle_solver_coverage",
    "oracle_residual_overflow_given_solvable_any_axis_pixel_rate",
    "trainable_coverage",
    "stride_trainable_oracle_reconstruction_epe_px",
)


def _read_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read capacity report {path}: {exc}") from exc
    if type(value) is not dict:
        raise ValueError("capacity report root must be an exact JSON object")
    return value


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "missing"
    numeric = float(value)
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.6f}"


def _metrics_line(label: str, metrics: Any) -> str:
    if not isinstance(metrics, Mapping):
        return f"{label} metrics=missing"
    return (
        f"{label} samples={metrics.get('sample_count', 'missing')} "
        f"solver={_number(metrics.get(_METRIC_NAMES[0]))} "
        f"overflow={_number(metrics.get(_METRIC_NAMES[1]))} "
        f"trainable={_number(metrics.get(_METRIC_NAMES[2]))} "
        f"stride_epe={_number(metrics.get(_METRIC_NAMES[3]))}"
    )


def render_summary(report: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    """Return a compact human-readable policy summary."""

    passed = decision.get("passed") is True
    results = report.get("results")
    lines = [
        "TEACHER_CAPACITY_DECISION=" + ("PASS" if passed else "REJECT"),
        f"policy={decision.get('policy_id', 'unknown')}",
    ]
    if isinstance(results, Mapping):
        for name in _AGGREGATE_NAMES:
            lines.append(_metrics_line(f"aggregate.{name}", results.get(name)))
        rotation_bins = results.get("rotation_bins")
        if isinstance(rotation_bins, Sequence) and not isinstance(
            rotation_bins, (str, bytes)
        ):
            for item in rotation_bins:
                if not isinstance(item, Mapping):
                    continue
                bounds = item.get("absolute_rotation_deg")
                if isinstance(bounds, Mapping):
                    interval = (
                        f"[{bounds.get('lower_inclusive', '?')},"
                        f"{bounds.get('upper', '?')}]"
                    )
                else:
                    interval = "[?,?]"
                lines.append(
                    _metrics_line(
                        f"rotation_bin.{item.get('index', '?')} {interval}deg",
                        item.get("metrics"),
                    )
                )
    failures = decision.get("failures")
    if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)):
        lines.append(f"failed_check_count={len(failures)}")
        for failure in failures:
            if not isinstance(failure, Mapping):
                continue
            lines.append(
                "FAIL "
                f"{failure.get('code', 'unknown')} "
                f"actual={failure.get('actual')!r} "
                f"expected={failure.get('operator', '?')} "
                f"{failure.get('threshold')!r}"
            )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="raw capacity audit JSON")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return exit code 2 when the frozen production policy rejects the report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = _read_report(args.report)
        decision = evaluate_teacher_capacity_policy(report)
        print(render_summary(report, decision), flush=True)
        if args.require_pass and decision.get("passed") is not True:
            return 2
        return 0
    except Exception as exc:
        print(f"teacher-capacity report evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
