"""Pure production acceptance policy for teacher-capacity reports.

This module deliberately has no dependency on PyTorch and performs no file
system I/O.  It consumes the raw mapping produced by the capacity preflight
and returns a JSON-compatible, itemized decision.  Missing, empty, non-numeric,
or non-finite metrics fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


POLICY_SCHEMA_VERSION = 1
POLICY_ID = "teacher_capacity_production_v1"
DECISION_SCHEMA_VERSION = 1
DECISION_KIND = "teacher_capacity_production_policy_decision"

_REPORT_KIND = "frozen_teacher_residual_capacity_preflight"
_AGGREGATE_GROUPS = (
    "original",
    "rotation_augmented",
    "full_geometry_augmented",
)
_ROTATION_BIN_EDGES = (0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)

_THRESHOLD_METRICS = {
    "oracle_solver_coverage": (">=", 0.99),
    "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": ("<=", 0.005),
    "trainable_coverage": (">=", 0.985),
    "stride_trainable_oracle_reconstruction_epe_px": ("<=", 1.0),
}
_ROTATION_BIN_THRESHOLD_METRICS = {
    "oracle_solver_coverage": (">=", 0.98),
    "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": ("<=", 0.01),
    "trainable_coverage": (">=", 0.97),
    "stride_trainable_oracle_reconstruction_epe_px": ("<=", 1.5),
}

# Every aggregate emitted by teacher_capacity_preflight has this metric
# contract.  Sample-level overflow rates are required to be finite and in
# [0,1], but intentionally have no production threshold in policy v1.
_REQUIRED_METRIC_PATHS = (
    ("sample_count",),
    ("eval_sample_count",),
    ("eval_pixels",),
    ("teacher_epe_px",),
    ("oracle_solver_coverage",),
    ("oracle_solver_any_sample_rate",),
    ("oracle_solver_full_sample_rate",),
    ("oracle_residual_overflow_given_solvable_x_pixel_rate",),
    ("oracle_residual_overflow_given_solvable_y_pixel_rate",),
    ("oracle_residual_overflow_given_solvable_any_axis_pixel_rate",),
    ("oracle_residual_overflow_given_solvable_x_sample_rate",),
    ("oracle_residual_overflow_given_solvable_y_sample_rate",),
    ("oracle_residual_overflow_given_solvable_any_axis_sample_rate",),
    ("trainable_coverage",),
    ("residual_target_valid_rate",),
    ("oracle_residual_axis_absmax_px", "x"),
    ("oracle_residual_axis_absmax_px", "y"),
    ("stride_oracle_residual_reconstruction_epe_px",),
    ("stride_trainable_oracle_reconstruction_epe_px",),
)
_RATE_METRIC_NAMES = frozenset(
    {
        "oracle_solver_coverage",
        "oracle_solver_any_sample_rate",
        "oracle_solver_full_sample_rate",
        "oracle_residual_overflow_given_solvable_x_pixel_rate",
        "oracle_residual_overflow_given_solvable_y_pixel_rate",
        "oracle_residual_overflow_given_solvable_any_axis_pixel_rate",
        "oracle_residual_overflow_given_solvable_x_sample_rate",
        "oracle_residual_overflow_given_solvable_y_sample_rate",
        "oracle_residual_overflow_given_solvable_any_axis_sample_rate",
        "trainable_coverage",
        "residual_target_valid_rate",
    }
)


def _threshold_policy(values: Mapping[str, tuple[str, float]]) -> dict[str, Any]:
    return {
        name: {"operator": operator, "threshold": threshold}
        for name, (operator, threshold) in values.items()
    }


_POLICY_V1: dict[str, Any] = {
    "schema_version": POLICY_SCHEMA_VERSION,
    "policy_id": POLICY_ID,
    "report_contract": {
        "report_version": 1,
        "kind": _REPORT_KIND,
    },
    "protocol": {
        "manifest_record_count": 300,
        "manifest_split": "val",
        "selected_sample_count": 300,
        "selected_indices": "exactly_0_through_299",
        "work_size": [512, 512],
        "feature_stride": 8,
        "max_residual_px": 24.0,
        "max_residual_target": 24.0,
        "rotation_bin_edges_deg": list(_ROTATION_BIN_EDGES),
    },
    "aggregate_groups": {
        name: {
            "sample_count": 300,
            "thresholds": _threshold_policy(_THRESHOLD_METRICS),
        }
        for name in _AGGREGATE_GROUPS
    },
    "rotation_bins": {
        "minimum_sample_count": 20,
        "thresholds": _threshold_policy(_ROTATION_BIN_THRESHOLD_METRICS),
    },
    "diagnostics": {
        "sample_level_overflow_rates": "finite_and_in_range_but_not_gated",
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


_COMPUTED_POLICY_SHA256 = hashlib.sha256(_canonical_json(_POLICY_V1)).hexdigest()
# This literal freezes both the production thresholds and their canonical
# serialization.  The accompanying test also asserts the same digest.
CANONICAL_POLICY_SHA256 = (
    "f762d9e96a53b3404815c0437c5a5535810323c7d50dd4baa774393c27c688fa"
)
if _COMPUTED_POLICY_SHA256 != CANONICAL_POLICY_SHA256:  # pragma: no cover
    raise RuntimeError(
        "teacher capacity production policy drifted; "
        f"expected SHA-256 {CANONICAL_POLICY_SHA256}, "
        f"computed {_COMPUTED_POLICY_SHA256}"
    )


_MISSING = object()


def production_policy_v1() -> dict[str, Any]:
    """Return a caller-owned copy of the frozen policy mapping."""

    return copy.deepcopy(_POLICY_V1)


def _get(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_exact_int(value: Any) -> bool:
    return type(value) is int


def _safe(value: Any) -> Any:
    """Make malformed actual values JSON-compatible for failure reports."""

    if value is _MISSING:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _typed_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_typed_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(_typed_equal(left, right) for left, right in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


class _Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(
        self,
        *,
        code: str,
        scope: str,
        passed: bool,
        actual: Any,
        operator: str,
        threshold: Any,
    ) -> None:
        safe_actual = _safe(actual)
        safe_threshold = _safe(threshold)
        self.items.append(
            {
                "code": code,
                "scope": scope,
                "passed": bool(passed),
                "actual": safe_actual,
                "operator": operator,
                "threshold": safe_threshold,
                "message": (
                    f"{code}: expected actual {operator} {safe_threshold!r}; "
                    f"actual={safe_actual!r}"
                ),
            }
        )

    def exact(
        self, *, code: str, scope: str, actual: Any, expected: Any
    ) -> None:
        self.add(
            code=code,
            scope=scope,
            passed=_typed_equal(actual, expected),
            actual=actual,
            operator="== (typed)",
            threshold=expected,
        )

    def finite(self, *, code: str, scope: str, actual: Any) -> None:
        self.add(
            code=code,
            scope=scope,
            passed=_is_finite_number(actual),
            actual=actual,
            operator="is finite numeric",
            threshold=True,
        )

    def numeric_gate(
        self,
        *,
        code: str,
        scope: str,
        actual: Any,
        operator: str,
        threshold: float,
    ) -> None:
        finite = _is_finite_number(actual)
        if operator == ">=":
            passed = finite and float(actual) >= threshold
        elif operator == "<=":
            passed = finite and float(actual) <= threshold
        else:  # Internal frozen policy invariant.
            raise AssertionError(f"unsupported operator {operator!r}")
        self.add(
            code=code,
            scope=scope,
            passed=passed,
            actual=actual,
            operator=operator,
            threshold=threshold,
        )


def _metric(metrics: Any, path: tuple[str, ...]) -> Any:
    return _get(metrics, *path)


def _metric_leaves(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        if not value:
            return [(prefix, _MISSING)]
        result: list[tuple[tuple[str, ...], Any]] = []
        for key in sorted(value, key=lambda item: str(item)):
            result.extend(_metric_leaves(value[key], (*prefix, str(key))))
        return result
    if isinstance(value, (list, tuple)):
        if not value:
            return [(prefix, _MISSING)]
        result = []
        for index, item in enumerate(value):
            result.extend(_metric_leaves(item, (*prefix, str(index))))
        return result
    return [(prefix, value)]


def _add_metric_contract(
    checks: _Checks,
    metrics: Any,
    *,
    code_prefix: str,
    scope: str,
) -> None:
    checks.add(
        code=f"{code_prefix}.metrics_nonempty",
        scope=scope,
        passed=isinstance(metrics, Mapping) and bool(metrics),
        actual=(len(metrics) if isinstance(metrics, Mapping) else metrics),
        operator=">",
        threshold=0,
    )
    observed_paths: set[tuple[str, ...]] = set()
    if isinstance(metrics, Mapping):
        for path, value in _metric_leaves(metrics):
            observed_paths.add(path)
            checks.finite(
                code=f"{code_prefix}.metric.{'.'.join(path) or '<root>'}.finite",
                scope=scope,
                actual=value,
            )
    for path in _REQUIRED_METRIC_PATHS:
        if path not in observed_paths:
            checks.finite(
                code=f"{code_prefix}.metric.{'.'.join(path)}.finite",
                scope=scope,
                actual=_MISSING,
            )

    for name in _RATE_METRIC_NAMES:
        value = _metric(metrics, (name,))
        checks.add(
            code=f"{code_prefix}.metric.{name}.unit_interval",
            scope=scope,
            passed=_is_finite_number(value) and 0.0 <= float(value) <= 1.0,
            actual=value,
            operator="in closed interval",
            threshold=[0.0, 1.0],
        )
    trainable = _metric(metrics, ("trainable_coverage",))
    target_valid = _metric(metrics, ("residual_target_valid_rate",))
    checks.add(
        code=f"{code_prefix}.trainable_target_valid_identity",
        scope=scope,
        passed=(
            _is_finite_number(trainable)
            and _is_finite_number(target_valid)
            and float(trainable) == float(target_valid)
        ),
        actual={"trainable_coverage": trainable, "residual_target_valid_rate": target_valid},
        operator="values are identical",
        threshold=True,
    )


def _add_thresholds(
    checks: _Checks,
    metrics: Any,
    *,
    code_prefix: str,
    scope: str,
    thresholds: Mapping[str, tuple[str, float]],
) -> None:
    for name, (operator, threshold) in thresholds.items():
        checks.numeric_gate(
            code=f"{code_prefix}.{name}",
            scope=scope,
            actual=_metric(metrics, (name,)),
            operator=operator,
            threshold=threshold,
        )


def _valid_metric_count(value: Any, expected: int) -> bool:
    return _is_exact_int(value) and value == expected


def evaluate_teacher_capacity_policy(report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one raw capacity report against frozen production policy v1.

    The function never mutates ``report`` and returns a JSON-compatible
    decision even when nested report fields are missing or non-finite.
    """

    if not isinstance(report, Mapping):
        raise TypeError("teacher capacity report must be a mapping")

    checks = _Checks()
    checks.exact(
        code="report.report_version",
        scope="report",
        actual=_get(report, "report_version"),
        expected=1,
    )
    checks.exact(
        code="report.kind",
        scope="report",
        actual=_get(report, "kind"),
        expected=_REPORT_KIND,
    )

    checks.exact(
        code="protocol.manifest.record_count",
        scope="protocol",
        actual=_get(report, "identities", "manifest", "record_count"),
        expected=300,
    )
    checks.exact(
        code="protocol.manifest.split",
        scope="protocol",
        actual=_get(report, "identities", "manifest", "split"),
        expected="val",
    )
    checks.exact(
        code="protocol.selected_sample_count",
        scope="protocol",
        actual=_get(report, "protocol", "selected_sample_count"),
        expected=300,
    )
    expected_indices = list(range(300))
    checks.exact(
        code="protocol.selected_indices",
        scope="protocol",
        actual=_get(report, "protocol", "selected_indices"),
        expected=expected_indices,
    )
    checks.exact(
        code="protocol.work_size",
        scope="protocol",
        actual=_get(report, "protocol", "work_size"),
        expected=[512, 512],
    )
    checks.exact(
        code="protocol.feature_stride",
        scope="protocol",
        actual=_get(report, "protocol", "feature_stride"),
        expected=8,
    )
    checks.exact(
        code="protocol.max_residual_px",
        scope="protocol",
        actual=_get(report, "protocol", "max_residual_px"),
        expected=24.0,
    )
    checks.exact(
        code="protocol.max_residual_target",
        scope="protocol",
        actual=_get(report, "protocol", "max_residual_target"),
        expected=24.0,
    )
    checks.exact(
        code="protocol.rotation_bin_edges_deg",
        scope="protocol",
        actual=_get(report, "protocol", "source_rotation", "bin_edges_deg"),
        expected=list(_ROTATION_BIN_EDGES),
    )

    results = _get(report, "results")
    aggregate_metrics: dict[str, Any] = {}
    for group_name in _AGGREGATE_GROUPS:
        code_prefix = f"aggregate.{group_name}"
        value = _get(results, group_name)
        checks.add(
            code=f"{code_prefix}.present",
            scope="aggregate",
            passed=isinstance(value, Mapping),
            actual=value if not isinstance(value, Mapping) else True,
            operator="is mapping",
            threshold=True,
        )
        aggregate_metrics[group_name] = value
        sample_count = _get(value, "sample_count")
        checks.add(
            code=f"{code_prefix}.sample_count",
            scope="aggregate",
            passed=_valid_metric_count(sample_count, 300),
            actual=sample_count,
            operator="== (integer)",
            threshold=300,
        )
        eval_sample_count = _get(value, "eval_sample_count")
        checks.add(
            code=f"{code_prefix}.eval_sample_count",
            scope="aggregate",
            passed=_valid_metric_count(eval_sample_count, 300),
            actual=eval_sample_count,
            operator="== (integer)",
            threshold=300,
        )
        eval_pixels = _get(value, "eval_pixels")
        checks.add(
            code=f"{code_prefix}.eval_pixels_positive",
            scope="aggregate",
            passed=_is_exact_int(eval_pixels) and eval_pixels > 0,
            actual=eval_pixels,
            operator="> (integer)",
            threshold=0,
        )
        _add_metric_contract(
            checks,
            value,
            code_prefix=code_prefix,
            scope="aggregate",
        )
        _add_thresholds(
            checks,
            value,
            code_prefix=code_prefix,
            scope="aggregate",
            thresholds=_THRESHOLD_METRICS,
        )

    raw_bins = _get(results, "rotation_bins")
    bins = raw_bins if isinstance(raw_bins, list) else []
    checks.add(
        code="rotation_bins.count",
        scope="rotation_bins",
        passed=len(bins) == len(_ROTATION_BIN_EDGES) - 1,
        actual=len(bins) if isinstance(raw_bins, list) else raw_bins,
        operator="==",
        threshold=len(_ROTATION_BIN_EDGES) - 1,
    )
    bin_sample_total = 0
    for index in range(len(_ROTATION_BIN_EDGES) - 1):
        item = bins[index] if index < len(bins) else _MISSING
        prefix = f"rotation_bin.{index}"
        checks.exact(
            code=f"{prefix}.index",
            scope="rotation_bin",
            actual=_get(item, "index"),
            expected=index,
        )
        expected_interval = {
            "lower_inclusive": _ROTATION_BIN_EDGES[index],
            "upper": _ROTATION_BIN_EDGES[index + 1],
            "upper_inclusive": index == len(_ROTATION_BIN_EDGES) - 2,
        }
        checks.exact(
            code=f"{prefix}.absolute_rotation_deg",
            scope="rotation_bin",
            actual=_get(item, "absolute_rotation_deg"),
            expected=expected_interval,
        )
        metrics = _get(item, "metrics")
        _add_metric_contract(
            checks,
            metrics,
            code_prefix=prefix,
            scope="rotation_bin",
        )
        sample_count = _get(metrics, "sample_count")
        if _is_exact_int(sample_count):
            bin_sample_total += sample_count
        checks.add(
            code=f"{prefix}.sample_count",
            scope="rotation_bin",
            passed=_is_exact_int(sample_count) and sample_count >= 20,
            actual=sample_count,
            operator=">= (integer)",
            threshold=20,
        )
        eval_sample_count = _get(metrics, "eval_sample_count")
        checks.add(
            code=f"{prefix}.eval_sample_count_identity",
            scope="rotation_bin",
            passed=(
                _is_exact_int(sample_count)
                and _is_exact_int(eval_sample_count)
                and eval_sample_count == sample_count
            ),
            actual={"sample_count": sample_count, "eval_sample_count": eval_sample_count},
            operator="values are identical integers",
            threshold=True,
        )
        eval_pixels = _get(metrics, "eval_pixels")
        checks.add(
            code=f"{prefix}.eval_pixels_positive",
            scope="rotation_bin",
            passed=_is_exact_int(eval_pixels) and eval_pixels > 0,
            actual=eval_pixels,
            operator="> (integer)",
            threshold=0,
        )
        _add_thresholds(
            checks,
            metrics,
            code_prefix=prefix,
            scope="rotation_bin",
            thresholds=_ROTATION_BIN_THRESHOLD_METRICS,
        )
    checks.add(
        code="rotation_bins.sample_count_sum",
        scope="rotation_bins",
        passed=(
            len(bins) == len(_ROTATION_BIN_EDGES) - 1
            and bin_sample_total == 300
        ),
        actual=bin_sample_total,
        operator="==",
        threshold=300,
    )

    # The raw report's per-sample section is diagnostic.  Production decisions
    # intentionally use aggregates/bins only; sample-level overflow *aggregate
    # rates* above are still required to be finite and semantically valid.
    failures = [dict(item) for item in checks.items if not item["passed"]]
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "kind": DECISION_KIND,
        "policy_id": POLICY_ID,
        "policy_sha256": CANONICAL_POLICY_SHA256,
        "policy": production_policy_v1(),
        "passed": not failures,
        "checks": checks.items,
        "failures": failures,
        "summary": {
            "check_count": len(checks.items),
            "passed_check_count": len(checks.items) - len(failures),
            "failed_check_count": len(failures),
            "aggregate_groups": list(_AGGREGATE_GROUPS),
            "rotation_bin_count": len(bins),
        },
    }


__all__ = [
    "CANONICAL_POLICY_SHA256",
    "DECISION_KIND",
    "DECISION_SCHEMA_VERSION",
    "POLICY_ID",
    "POLICY_SCHEMA_VERSION",
    "evaluate_teacher_capacity_policy",
    "production_policy_v1",
]
