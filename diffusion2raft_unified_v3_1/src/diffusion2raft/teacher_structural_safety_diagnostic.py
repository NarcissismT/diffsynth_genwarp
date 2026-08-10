"""Structural safety audit for the GT-selected best-of-C4 teacher upper bound.

This v7 diagnostic consumes and cryptographically binds a v6 best-of-C4
report.  It replays exactly the same full-geometry samples, uses the frozen
best-teacher-EPE C4 label, and evaluates the 24-iteration stride-8 oracle
residual on several per-axis capacity supports.  The audit measures capacity,
in-bounds support, folds/Jacobians, line geometry, curvature, and texture loss
relative to an exact-GT-flow resampling reference.

The selector still reads GT flow and is therefore not deployable.  A passing
v7 result is only permission to investigate a real image-only router; it can
never approve production capacity or unlock training.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import time
from collections import defaultdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .config import load_config
from .data import DocumentFlowDataset, SourceGeometryAugment
from .geometry import (
    backward_warp,
    compose_backward_flows,
    flow_valid_mask,
    residual_from_composed_flow,
    resize_backward_flow,
)
from .losses import (
    jacobian_determinant,
    line_straightness_loss,
    second_order_bending_loss,
    target_structure_maps,
)
from .models.teacher_prior import TorchScriptGeometryPrior
from .teacher_capacity_preflight import (
    _Accumulator,
    _assert_path_matches_identity,
    _atomic_write_json,
    _canonical_json,
    _file_identity,
    _resolve_configured_path,
    _runtime_identity,
    _sample_indices,
    build_full_geometry_plan,
    capacity_sample_statistics,
)
from .teacher_quarter_turn_diagnostic import (
    BEST_OF_C4_REPORT_KIND,
    BEST_OF_C4_REPORT_VERSION,
    _QUARTER_TURNS,
    _full_geometry_oracle_variants,
    _validate_expected_sha256,
    _validate_output_path,
    rotate_source_by_quarter_turn,
    transform_backward_flow_source_map_by_quarter_turn,
)


REPORT_VERSION = 7
REPORT_KIND = (
    "teacher_quarter_turn_oracle_canonical_full_geometry_"
    "structural_safety_diagnostic"
)
POLICY_ID = "best_of_c4_stride8_structural_safety_v1"
DEFAULT_CAP_SWEEP = (24.0, 32.0, 40.0)

CAPACITY_THRESHOLDS = {
    "min_oracle_solver_coverage": 0.99,
    "max_overflow_any_axis_pixel_rate": 0.005,
    "min_trainable_coverage": 0.985,
    "max_stride_trainable_reconstruction_epe_px": 1.0,
}
STRUCTURAL_THRESHOLDS = {
    # Reuse the frozen typical-v3.3 validation/inference topology limits.
    "min_candidate_in_bounds_given_trainable": 0.999,
    "max_aggregate_fold_rate": 0.0004,
    "max_per_sample_fold_rate": 0.001,
    "min_per_sample_jacobian_p01": 0.01,
    "max_stride_flow_epe_px": 1.0,
    "max_line_stride_flow_epe_px": 1.0,
    # Resampling error must recover at least 90% of the teacher discrepancy,
    # except for a one-8-bit-level numerical/interpolation floor.
    "min_texture_recovery_fraction": 0.90,
    "texture_absolute_floor": 1.0 / 255.0,
}


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _regular_input_identity(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size <= 0:
        raise ValueError(f"{label} must be a regular non-empty non-symlink file: {candidate}")
    resolved = candidate.resolve(strict=True)
    return resolved, _file_identity(resolved, hash_contents=True)


def _same_sha256(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    left = actual.get("sha256")
    right = expected.get("sha256")
    if not isinstance(left, str) or not isinstance(right, str) or not hmac.compare_digest(left, right):
        raise ValueError(f"{label} SHA-256 does not match the bound v6 report")


def _grid_key(iterations: int, cap: float) -> str:
    return f"iterations={int(iterations)},max_residual_px={float(cap):g}"


def _find_grid_entry(
    entries: Any, *, iterations: int, cap: float, label: str
) -> dict[str, Any]:
    if not isinstance(entries, list):
        raise ValueError(f"{label} must be a list")
    matches = [
        value
        for value in entries
        if isinstance(value, dict)
        and value.get("residual_target_iterations") == int(iterations)
        and value.get("max_residual_px") == float(cap)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{label} must contain exactly one {iterations}x{float(cap):g} cell"
        )
    return matches[0]


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _capacity_failures(entry: dict[str, Any]) -> list[dict[str, Any]]:
    contexts: list[tuple[str, dict[str, Any]]] = []
    aggregate = _require_mapping(
        entry.get("canonical_full_geometry_augmented"),
        "capacity aggregate",
    )
    contexts.append(("aggregate", aggregate))
    for family in ("canonical_rotation_bins", "nearest_angle_residual_rotation_bins"):
        values = entry.get(family)
        if not isinstance(values, list):
            raise ValueError(f"capacity cell {family} must be a list")
        for record in values:
            record = _require_mapping(record, f"capacity cell {family} record")
            metrics = _require_mapping(record.get("metrics"), f"capacity cell {family} metrics")
            # Empty bins do not weaken a tiny CPU fixture and cannot occur in
            # the frozen 300-sample production-scale audit.
            if int(metrics.get("sample_count", 0)) > 0:
                contexts.append((f"{family}[{record.get('index')}]", metrics))

    checks = (
        ("oracle_solver_coverage", ">=", CAPACITY_THRESHOLDS["min_oracle_solver_coverage"]),
        (
            "oracle_residual_overflow_given_solvable_any_axis_pixel_rate",
            "<=",
            CAPACITY_THRESHOLDS["max_overflow_any_axis_pixel_rate"],
        ),
        ("trainable_coverage", ">=", CAPACITY_THRESHOLDS["min_trainable_coverage"]),
        (
            "stride_trainable_oracle_reconstruction_epe_px",
            "<=",
            CAPACITY_THRESHOLDS["max_stride_trainable_reconstruction_epe_px"],
        ),
    )
    failures: list[dict[str, Any]] = []
    for context, metrics in contexts:
        for metric, operator, threshold in checks:
            raw = metrics.get(metric)
            actual = None
            passed = False
            if isinstance(raw, Real) and not isinstance(raw, bool):
                actual = float(raw)
                if math.isfinite(actual):
                    passed = actual >= threshold if operator == ">=" else actual <= threshold
            if not passed:
                failures.append(
                    {
                        "context": context,
                        "metric": metric,
                        "actual": actual,
                        "operator": operator,
                        "threshold": threshold,
                    }
                )
    return failures


def _candidate_record(sample: dict[str, Any], quarter_turn: int) -> dict[str, Any]:
    c4 = _require_mapping(sample.get("c4_best_of_four"), "v6 sample c4_best_of_four")
    candidates = c4.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("v6 sample candidates must be a list")
    matches = [
        value
        for value in candidates
        if isinstance(value, dict) and value.get("quarter_turn_deg") == quarter_turn
    ]
    if len(matches) != 1:
        raise ValueError(f"v6 sample must contain one q={quarter_turn} candidate")
    return matches[0]


def _assert_capacity_metric_match(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    exact_keys = ("eval_pixels",)
    numeric_keys = (
        "teacher_epe_px",
        "oracle_solver_coverage",
        "oracle_residual_overflow_given_solvable_any_axis_pixel_rate",
        "trainable_coverage",
        "stride_trainable_oracle_reconstruction_epe_px",
    )
    for key in exact_keys:
        if actual.get(key) != expected.get(key):
            raise RuntimeError(f"v7 capacity replay diverged from v6 at {key}")
    for key in numeric_keys:
        left = actual.get(key)
        right = expected.get(key)
        if left is None or right is None:
            if left is not right:
                raise RuntimeError(f"v7 capacity replay diverged from v6 at {key}")
            continue
        left_value = _finite_number(left, f"actual {key}")
        right_value = _finite_number(right, f"expected {key}")
        if not math.isclose(left_value, right_value, rel_tol=2.0e-5, abs_tol=2.0e-5):
            raise RuntimeError(
                f"v7 capacity replay diverged from v6 at {key}: "
                f"actual={left_value} expected={right_value}"
            )


def _masked_stat(value: Tensor, mask: Tensor) -> tuple[float, float, float | None]:
    mask = mask.bool()
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    denominator = float(mask.sum().item())
    if denominator == 0.0:
        return 0.0, 0.0, None
    numerator = float(value[mask].sum().item())
    return numerator, denominator, numerator / denominator


def _weighted_stat(
    value: Tensor, mask: Tensor, weight: Tensor
) -> tuple[float, float, float | None]:
    mask_float = mask.to(dtype=value.dtype)
    weight_float = weight.to(dtype=value.dtype).clamp_min(0.0)
    if mask_float.shape != value.shape:
        mask_float = mask_float.expand_as(value)
    if weight_float.shape != value.shape:
        weight_float = weight_float.expand_as(value)
    combined = mask_float * weight_float
    denominator = float(combined.sum().item())
    if denominator <= 1.0e-12:
        return 0.0, 0.0, None
    numerator = float((value * combined).sum().item())
    return numerator, denominator, numerator / denominator


def _gradient_stat(
    prediction: Tensor, reference: Tensor, mask: Tensor
) -> tuple[float, float, float | None]:
    dx = (prediction[..., 1:] - prediction[..., :-1]) - (
        reference[..., 1:] - reference[..., :-1]
    )
    dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (
        reference[..., 1:, :] - reference[..., :-1, :]
    )
    dx_value = dx.abs().mean(dim=1, keepdim=True)
    dy_value = dy.abs().mean(dim=1, keepdim=True)
    dx_mask = mask[..., 1:] & mask[..., :-1]
    dy_mask = mask[..., 1:, :] & mask[..., :-1, :]
    x_num, x_den, _ = _masked_stat(dx_value, dx_mask)
    y_num, y_den, _ = _masked_stat(dy_value, dy_mask)
    denominator = x_den + y_den
    numerator = x_num + y_num
    return numerator, denominator, None if denominator == 0.0 else numerator / denominator


def _valid_cells(mask: Tensor) -> Tensor:
    value = mask[:, 0].bool()
    return (
        value[:, :-1, :-1]
        & value[:, 1:, :-1]
        & value[:, :-1, 1:]
        & value[:, 1:, 1:]
    )


def _sample_structural_metrics(
    *,
    canonical_source: Tensor,
    target_image: Tensor,
    teacher_flow: Tensor,
    target_flow: Tensor,
    stride_flow: Tensor,
    valid_mask: Tensor,
    trainable_mask: Tensor,
    structures: dict[str, Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_in_bounds = flow_valid_mask(stride_flow, canonical_source.shape[-2:])
    teacher_in_bounds = flow_valid_mask(teacher_flow, canonical_source.shape[-2:])
    target_in_bounds = flow_valid_mask(target_flow, canonical_source.shape[-2:])
    structural_valid = trainable_mask & candidate_in_bounds & target_in_bounds

    flow_error = torch.linalg.vector_norm(stride_flow - target_flow, dim=1, keepdim=True)
    teacher_error = torch.linalg.vector_norm(teacher_flow - target_flow, dim=1, keepdim=True)
    flow_stat = _masked_stat(flow_error, structural_valid)
    teacher_flow_stat = _masked_stat(teacher_error, structural_valid)
    line_flow_stat = _weighted_stat(flow_error, structural_valid, structures["line"])
    teacher_line_flow_stat = _weighted_stat(
        teacher_error, structural_valid, structures["line"]
    )

    line_straightness = float(
        line_straightness_loss(
            stride_flow,
            target_flow,
            structural_valid,
            structures["horizontal"],
            structures["vertical"],
            robust=False,
        ).item()
    )
    teacher_line_straightness = float(
        line_straightness_loss(
            teacher_flow,
            target_flow,
            structural_valid,
            structures["horizontal"],
            structures["vertical"],
            robust=False,
        ).item()
    )
    curvature = float(
        second_order_bending_loss(stride_flow - target_flow, structural_valid).item()
    )
    teacher_curvature = float(
        second_order_bending_loss(teacher_flow - target_flow, structural_valid).item()
    )

    reference = backward_warp(
        canonical_source, target_flow, padding_mode="border"
    )
    candidate_rectified = backward_warp(
        canonical_source, stride_flow, padding_mode="border"
    )
    teacher_rectified = backward_warp(
        canonical_source, teacher_flow, padding_mode="border"
    )
    candidate_rgb_error = (candidate_rectified - reference).abs().mean(dim=1, keepdim=True)
    teacher_rgb_error = (teacher_rectified - reference).abs().mean(dim=1, keepdim=True)
    rgb_stat = _masked_stat(candidate_rgb_error, structural_valid)
    teacher_rgb_stat = _masked_stat(teacher_rgb_error, structural_valid)
    gradient_stat = _gradient_stat(candidate_rectified, reference, structural_valid)
    teacher_gradient_stat = _gradient_stat(teacher_rectified, reference, structural_valid)
    line_rgb_stat = _weighted_stat(
        candidate_rgb_error, structural_valid, structures["line"]
    )
    teacher_line_rgb_stat = _weighted_stat(
        teacher_rgb_error, structural_valid, structures["line"]
    )

    cells = _valid_cells(structural_valid)
    candidate_det = jacobian_determinant(stride_flow)
    target_det = jacobian_determinant(target_flow)
    cell_count = int(cells.sum().item())
    if cell_count:
        candidate_selected = candidate_det[cells]
        target_selected = target_det[cells]
        candidate_fold_count = int((candidate_selected <= 0.0).sum().item())
        target_fold_count = int((target_selected <= 0.0).sum().item())
        introduced_fold_count = int(
            ((candidate_selected <= 0.0) & (target_selected > 0.0)).sum().item()
        )
        candidate_jacobian_p01: float | None = float(
            torch.quantile(candidate_selected.float(), 0.01).item()
        )
        target_jacobian_p01: float | None = float(
            torch.quantile(target_selected.float(), 0.01).item()
        )
    else:
        candidate_fold_count = 0
        target_fold_count = 0
        introduced_fold_count = 0
        candidate_jacobian_p01 = None
        target_jacobian_p01 = None

    eval_pixels = int(valid_mask.sum().item())
    trainable_pixels = int(trainable_mask.sum().item())
    structural_pixels = int(structural_valid.sum().item())
    candidate_in_bounds_pixels = int((trainable_mask & candidate_in_bounds).sum().item())
    teacher_in_bounds_pixels = int((trainable_mask & teacher_in_bounds).sum().item())
    target_in_bounds_pixels = int((trainable_mask & target_in_bounds).sum().item())

    def rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else float(numerator / denominator)

    report = {
        "eval_pixels": eval_pixels,
        "trainable_pixels": trainable_pixels,
        "trainable_coverage": rate(trainable_pixels, eval_pixels),
        "structural_eval_pixels": structural_pixels,
        "candidate_in_bounds_given_trainable": rate(
            candidate_in_bounds_pixels, trainable_pixels
        ),
        "teacher_in_bounds_given_trainable": rate(
            teacher_in_bounds_pixels, trainable_pixels
        ),
        "target_in_bounds_given_trainable": rate(
            target_in_bounds_pixels, trainable_pixels
        ),
        "stride_flow_epe_px": flow_stat[2],
        "teacher_flow_epe_px": teacher_flow_stat[2],
        "line_stride_flow_epe_px": line_flow_stat[2],
        "teacher_line_flow_epe_px": teacher_line_flow_stat[2],
        "line_straightness_error_px": line_straightness,
        "teacher_line_straightness_error_px": teacher_line_straightness,
        "curvature_error_px": curvature,
        "teacher_curvature_error_px": teacher_curvature,
        "reference_rgb_l1": rgb_stat[2],
        "teacher_reference_rgb_l1": teacher_rgb_stat[2],
        "reference_gradient_l1": gradient_stat[2],
        "teacher_reference_gradient_l1": teacher_gradient_stat[2],
        "line_reference_rgb_l1": line_rgb_stat[2],
        "teacher_line_reference_rgb_l1": teacher_line_rgb_stat[2],
        "topology_cell_count": cell_count,
        "candidate_fold_rate": rate(candidate_fold_count, cell_count),
        "target_fold_rate": rate(target_fold_count, cell_count),
        "introduced_fold_rate": rate(introduced_fold_count, cell_count),
        "candidate_jacobian_p01": candidate_jacobian_p01,
        "target_jacobian_p01": target_jacobian_p01,
    }
    raw = {
        "eval_pixels": eval_pixels,
        "trainable_pixels": trainable_pixels,
        "structural_pixels": structural_pixels,
        "candidate_in_bounds_pixels": candidate_in_bounds_pixels,
        "teacher_in_bounds_pixels": teacher_in_bounds_pixels,
        "target_in_bounds_pixels": target_in_bounds_pixels,
        "topology_cells": cell_count,
        "candidate_folds": candidate_fold_count,
        "target_folds": target_fold_count,
        "introduced_folds": introduced_fold_count,
        "weighted": {
            "stride_flow_epe_px": flow_stat[:2],
            "teacher_flow_epe_px": teacher_flow_stat[:2],
            "line_stride_flow_epe_px": line_flow_stat[:2],
            "teacher_line_flow_epe_px": teacher_line_flow_stat[:2],
            "reference_rgb_l1": rgb_stat[:2],
            "teacher_reference_rgb_l1": teacher_rgb_stat[:2],
            "reference_gradient_l1": gradient_stat[:2],
            "teacher_reference_gradient_l1": teacher_gradient_stat[:2],
            "line_reference_rgb_l1": line_rgb_stat[:2],
            "teacher_line_reference_rgb_l1": teacher_line_rgb_stat[:2],
        },
    }
    return report, raw


def _new_aggregate_state() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "eval_pixels": 0,
        "trainable_pixels": 0,
        "structural_pixels": 0,
        "candidate_in_bounds_pixels": 0,
        "teacher_in_bounds_pixels": 0,
        "target_in_bounds_pixels": 0,
        "topology_cells": 0,
        "candidate_folds": 0,
        "target_folds": 0,
        "introduced_folds": 0,
        "weighted": defaultdict(lambda: [0.0, 0.0]),
        "sample_values": defaultdict(list),
    }


def _add_aggregate(
    state: dict[str, Any], sample_report: dict[str, Any], raw: dict[str, Any]
) -> None:
    state["sample_count"] += 1
    for key in (
        "eval_pixels",
        "trainable_pixels",
        "structural_pixels",
        "candidate_in_bounds_pixels",
        "teacher_in_bounds_pixels",
        "target_in_bounds_pixels",
        "topology_cells",
        "candidate_folds",
        "target_folds",
        "introduced_folds",
    ):
        state[key] += raw[key]
    for key, (numerator, denominator) in raw["weighted"].items():
        state["weighted"][key][0] += numerator
        state["weighted"][key][1] += denominator
    for key in (
        "candidate_fold_rate",
        "target_fold_rate",
        "introduced_fold_rate",
        "candidate_jacobian_p01",
        "target_jacobian_p01",
        "line_straightness_error_px",
        "teacher_line_straightness_error_px",
        "curvature_error_px",
        "teacher_curvature_error_px",
    ):
        value = sample_report.get(key)
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
            state["sample_values"][key].append(float(value))


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _sample_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": None if not values else float(sum(values) / len(values)),
        "min": None if not values else float(min(values)),
        "max": None if not values else float(max(values)),
    }


def _recovery_fraction(candidate: float | None, teacher: float | None) -> float | None:
    if candidate is None or teacher is None:
        return None
    if teacher <= 1.0e-12:
        return 1.0 if candidate <= 1.0e-12 else None
    return float(1.0 - candidate / teacher)


def _finalize_aggregate(state: dict[str, Any]) -> dict[str, Any]:
    weighted = {
        key: _ratio(values[0], values[1])
        for key, values in state["weighted"].items()
    }
    sample_values = state["sample_values"]
    result: dict[str, Any] = {
        "sample_count": state["sample_count"],
        "eval_pixels": state["eval_pixels"],
        "trainable_pixels": state["trainable_pixels"],
        "trainable_coverage": _ratio(state["trainable_pixels"], state["eval_pixels"]),
        "structural_eval_pixels": state["structural_pixels"],
        "candidate_in_bounds_given_trainable": _ratio(
            state["candidate_in_bounds_pixels"], state["trainable_pixels"]
        ),
        "teacher_in_bounds_given_trainable": _ratio(
            state["teacher_in_bounds_pixels"], state["trainable_pixels"]
        ),
        "target_in_bounds_given_trainable": _ratio(
            state["target_in_bounds_pixels"], state["trainable_pixels"]
        ),
        **weighted,
        "topology_cell_count": state["topology_cells"],
        "candidate_fold_rate": _ratio(
            state["candidate_folds"], state["topology_cells"]
        ),
        "target_fold_rate": _ratio(state["target_folds"], state["topology_cells"]),
        "introduced_fold_rate": _ratio(
            state["introduced_folds"], state["topology_cells"]
        ),
        "candidate_fold_rate_per_sample": _sample_summary(
            sample_values["candidate_fold_rate"]
        ),
        "target_fold_rate_per_sample": _sample_summary(
            sample_values["target_fold_rate"]
        ),
        "introduced_fold_rate_per_sample": _sample_summary(
            sample_values["introduced_fold_rate"]
        ),
        "candidate_jacobian_p01_per_sample": _sample_summary(
            sample_values["candidate_jacobian_p01"]
        ),
        "target_jacobian_p01_per_sample": _sample_summary(
            sample_values["target_jacobian_p01"]
        ),
        "line_straightness_error_px_per_sample": _sample_summary(
            sample_values["line_straightness_error_px"]
        ),
        "teacher_line_straightness_error_px_per_sample": _sample_summary(
            sample_values["teacher_line_straightness_error_px"]
        ),
        "curvature_error_px_per_sample": _sample_summary(
            sample_values["curvature_error_px"]
        ),
        "teacher_curvature_error_px_per_sample": _sample_summary(
            sample_values["teacher_curvature_error_px"]
        ),
    }
    for metric in (
        "reference_rgb_l1",
        "reference_gradient_l1",
        "line_reference_rgb_l1",
    ):
        result[f"{metric}_recovery_fraction_vs_teacher"] = _recovery_fraction(
            result.get(metric), result.get(f"teacher_{metric}")
        )
    return result


def _new_support_metrics(
    *,
    stride_flow: Tensor,
    target_flow: Tensor,
    candidate_rectified: Tensor,
    reference: Tensor,
    new_mask: Tensor,
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    flow_error = torch.linalg.vector_norm(stride_flow - target_flow, dim=1, keepdim=True)
    rgb_error = (candidate_rectified - reference).abs().mean(dim=1, keepdim=True)
    flow_stat = _masked_stat(flow_error, new_mask)
    rgb_stat = _masked_stat(rgb_error, new_mask)
    return (
        {
            "pixels": int(new_mask.sum().item()),
            "stride_flow_epe_px": flow_stat[2],
            "reference_rgb_l1": rgb_stat[2],
        },
        {
            "stride_flow_epe_px": flow_stat[:2],
            "reference_rgb_l1": rgb_stat[:2],
        },
    )


def _evaluate_structural_policy(
    *, capacity_failures: list[dict[str, Any]], metrics: dict[str, Any], new_support: dict[str, Any]
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(code: str, actual: Any, operator: str, threshold: Any, passed: bool) -> None:
        checks.append(
            {
                "code": code,
                "actual": actual,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )

    def numeric(code: str, value: Any, operator: str, threshold: Any) -> None:
        actual = None
        passed = False
        finite_threshold = (
            isinstance(threshold, Real)
            and not isinstance(threshold, bool)
            and math.isfinite(float(threshold))
        )
        if finite_threshold and isinstance(value, Real) and not isinstance(value, bool):
            actual = float(value)
            if math.isfinite(actual):
                passed = actual <= threshold if operator == "<=" else actual >= threshold
        add(code, actual, operator, threshold, passed)

    add(
        "capacity.selected_cell_strict_all_bins",
        len(capacity_failures),
        "==",
        0.0,
        not capacity_failures,
    )
    numeric(
        "structure.candidate_in_bounds_given_trainable",
        metrics.get("candidate_in_bounds_given_trainable"),
        ">=",
        STRUCTURAL_THRESHOLDS["min_candidate_in_bounds_given_trainable"],
    )
    numeric(
        "structure.aggregate_candidate_fold_rate",
        metrics.get("candidate_fold_rate"),
        "<=",
        STRUCTURAL_THRESHOLDS["max_aggregate_fold_rate"],
    )
    numeric(
        "structure.aggregate_introduced_fold_rate",
        metrics.get("introduced_fold_rate"),
        "<=",
        STRUCTURAL_THRESHOLDS["max_aggregate_fold_rate"],
    )
    numeric(
        "structure.max_per_sample_candidate_fold_rate",
        _require_mapping(
            metrics.get("candidate_fold_rate_per_sample"),
            "candidate fold sample summary",
        ).get("max"),
        "<=",
        STRUCTURAL_THRESHOLDS["max_per_sample_fold_rate"],
    )
    numeric(
        "structure.max_per_sample_introduced_fold_rate",
        _require_mapping(
            metrics.get("introduced_fold_rate_per_sample"),
            "introduced fold sample summary",
        ).get("max"),
        "<=",
        STRUCTURAL_THRESHOLDS["max_per_sample_fold_rate"],
    )
    numeric(
        "structure.min_per_sample_candidate_jacobian_p01",
        _require_mapping(
            metrics.get("candidate_jacobian_p01_per_sample"),
            "candidate Jacobian sample summary",
        ).get("min"),
        ">=",
        STRUCTURAL_THRESHOLDS["min_per_sample_jacobian_p01"],
    )
    numeric(
        "geometry.stride_flow_epe_px",
        metrics.get("stride_flow_epe_px"),
        "<=",
        STRUCTURAL_THRESHOLDS["max_stride_flow_epe_px"],
    )
    numeric(
        "geometry.line_stride_flow_epe_px",
        metrics.get("line_stride_flow_epe_px"),
        "<=",
        STRUCTURAL_THRESHOLDS["max_line_stride_flow_epe_px"],
    )

    candidate_line = _require_mapping(
        metrics.get("line_straightness_error_px_per_sample"),
        "line straightness sample summary",
    ).get("mean")
    teacher_line = _require_mapping(
        metrics.get("teacher_line_straightness_error_px_per_sample"),
        "teacher line straightness sample summary",
    ).get("mean")
    line_passed = (
        isinstance(candidate_line, Real)
        and isinstance(teacher_line, Real)
        and float(candidate_line) <= float(teacher_line) + 1.0e-6
    )
    add(
        "geometry.line_straightness_not_worse_than_teacher",
        candidate_line,
        "<=",
        float(teacher_line) + 1.0e-6 if isinstance(teacher_line, Real) else None,
        line_passed,
    )
    candidate_curvature = _require_mapping(
        metrics.get("curvature_error_px_per_sample"),
        "curvature sample summary",
    ).get("mean")
    teacher_curvature = _require_mapping(
        metrics.get("teacher_curvature_error_px_per_sample"),
        "teacher curvature sample summary",
    ).get("mean")
    curvature_passed = (
        isinstance(candidate_curvature, Real)
        and isinstance(teacher_curvature, Real)
        and float(candidate_curvature) <= float(teacher_curvature) + 1.0e-6
    )
    add(
        "geometry.curvature_not_worse_than_teacher",
        candidate_curvature,
        "<=",
        float(teacher_curvature) + 1.0e-6
        if isinstance(teacher_curvature, Real)
        else None,
        curvature_passed,
    )

    for metric in ("reference_rgb_l1", "reference_gradient_l1", "line_reference_rgb_l1"):
        teacher_metric = metrics.get(f"teacher_{metric}")
        threshold = None
        if isinstance(teacher_metric, Real) and not isinstance(teacher_metric, bool):
            threshold = max(
                STRUCTURAL_THRESHOLDS["texture_absolute_floor"],
                (1.0 - STRUCTURAL_THRESHOLDS["min_texture_recovery_fraction"])
                * float(teacher_metric),
            )
        numeric(
            f"texture.{metric}_90pct_recovery_or_8bit_floor",
            metrics.get(metric),
            "<=",
            threshold,
        )

    if int(new_support.get("pixels", 0)) > 0:
        numeric(
            "capacity.new_support_vs_baseline.stride_flow_epe_px",
            new_support.get("stride_flow_epe_px"),
            "<=",
            STRUCTURAL_THRESHOLDS["max_stride_flow_epe_px"],
        )

    failures = [value for value in checks if not value["passed"]]
    return {
        "policy_id": POLICY_ID,
        "passed": not failures,
        "structural_upper_bound_passed": not failures,
        "can_approve_production": False,
        "next_action": (
            "implement_and_evaluate_image_only_c4_router"
            if not failures
            else "reject_24x40_upper_bound_and_investigate_failed_structure_checks"
        ),
        "checks": checks,
        "failures": failures,
        "bound_v6_capacity_failures": capacity_failures,
    }


def run_structural_safety_diagnostic(
    *,
    config_path: Path,
    checkpoint_path: Path,
    teacher_path: Path,
    expected_teacher_sha256: str,
    best_of_c4_report_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    split: str = "val",
    sample_count: int | None = 300,
    seed: int = 42,
    batch_size: int = 1,
    device: torch.device | str = "cuda:0",
    residual_target_iterations: int = 24,
    residual_cap_sweep: Sequence[float] = DEFAULT_CAP_SWEEP,
    selected_max_residual_px: float = 40.0,
    baseline_max_residual_px: float = 24.0,
    teacher_factory: type[TorchScriptGeometryPrior] = TorchScriptGeometryPrior,
) -> dict[str, Any]:
    """Replay the v6 best-EPE labels and audit stride-8 structural safety."""

    started_at = time.time()
    started_monotonic = time.monotonic()
    output = _validate_output_path(output_path)
    if int(batch_size) != 1 or isinstance(batch_size, bool):
        raise ValueError("v7 structural audit freezes batch_size=1")
    if isinstance(residual_target_iterations, bool) or not isinstance(
        residual_target_iterations, Integral
    ) or int(residual_target_iterations) < 1:
        raise ValueError("residual_target_iterations must be a positive integer")
    residual_target_iterations = int(residual_target_iterations)
    raw_caps = tuple(residual_cap_sweep)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_caps):
        raise ValueError("residual_cap_sweep must contain only numbers")
    caps = tuple(float(value) for value in raw_caps)
    if (
        not caps
        or any(not math.isfinite(value) or value <= 0.0 for value in caps)
        or tuple(sorted(set(caps))) != caps
    ):
        raise ValueError("residual_cap_sweep must be unique, positive, and increasing")
    selected_cap = float(selected_max_residual_px)
    baseline_cap = float(baseline_max_residual_px)
    if selected_cap not in caps or baseline_cap not in caps or baseline_cap > selected_cap:
        raise ValueError("baseline and selected residual caps must be ordered members of the sweep")

    expected_teacher_sha256 = _validate_expected_sha256(expected_teacher_sha256)
    best_path, best_identity = _regular_input_identity(
        best_of_c4_report_path, label="best_of_c4_report"
    )
    try:
        best_report = json.loads(best_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid best-of-C4 JSON report: {best_path}") from error
    best_report = _require_mapping(best_report, "best_of_c4_report")
    if (
        best_report.get("kind") != BEST_OF_C4_REPORT_KIND
        or best_report.get("report_version") != BEST_OF_C4_REPORT_VERSION
        or best_report.get("diagnostic_only") is not True
        or best_report.get("can_approve_production") is not False
        or best_report.get("uses_ground_truth_flow_for_candidate_selection") is not True
    ):
        raise ValueError("best_of_c4_report is not a canonical diagnostic-only v6 report")

    best_protocol = _require_mapping(best_report.get("protocol"), "v6 protocol")
    best_results = _require_mapping(best_report.get("results"), "v6 results")
    best_identities = _require_mapping(best_report.get("identities"), "v6 identities")
    best_c4_protocol = _require_mapping(best_protocol.get("best_of_c4"), "v6 best_of_c4 protocol")
    if best_c4_protocol.get("candidate_order") != list(_QUARTER_TURNS):
        raise ValueError("v6 candidate order is not the frozen C4 order")
    selected_v6_grid = _find_grid_entry(
        best_results.get("best_teacher_epe_capacity_grid"),
        iterations=residual_target_iterations,
        cap=selected_cap,
        label="v6 best_teacher_epe_capacity_grid",
    )
    capacity_failures = _capacity_failures(selected_v6_grid)
    if capacity_failures:
        raise ValueError("selected v6 capacity cell does not pass the frozen strict all-bin gate")

    config_path = config_path.expanduser().resolve(strict=True)
    config = load_config(config_path)
    project_root = config_path.parent.parent
    implementation_root = Path(__file__).resolve().parent
    implementation_identities = {
        name: _file_identity(
            implementation_root / name,
            hash_contents=True,
        )
        for name in (
            "geometry.py",
            "losses.py",
            "teacher_capacity_preflight.py",
            "teacher_quarter_turn_diagnostic.py",
            "teacher_structural_safety_diagnostic.py",
        )
    }
    data_config = _require_mapping(config.get("data"), "config.data")
    model_config = _require_mapping(config.get("model"), "config.model")
    loss_config = _require_mapping(config.get("loss"), "config.loss")
    if str(model_config.get("prior_backend", "learned")).lower() != "torchscript":
        raise ValueError("structural diagnostic requires prior_backend=torchscript")
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    configured_manifest = data_config.get(f"{split}_manifest")
    if manifest_path is None:
        if not configured_manifest:
            raise ValueError(f"config.data.{split}_manifest is required")
        manifest = _resolve_configured_path(
            str(configured_manifest), project_root=project_root, explicit=False
        )
    else:
        manifest = _resolve_configured_path(
            manifest_path, project_root=project_root, explicit=True
        )
    manifest = manifest.resolve(strict=True)
    checkpoint = _resolve_configured_path(
        checkpoint_path, project_root=project_root, explicit=True
    ).resolve(strict=True)
    teacher_path = _resolve_configured_path(
        teacher_path, project_root=project_root, explicit=True
    ).resolve(strict=True)

    config_identity = _file_identity(config_path, hash_contents=True)
    manifest_identity = _file_identity(manifest, hash_contents=True)
    checkpoint_identity = _file_identity(checkpoint, hash_contents=True)
    teacher_identity = _file_identity(teacher_path, hash_contents=True)
    _same_sha256(config_identity, _require_mapping(best_identities.get("config"), "v6 config identity"), "config")
    _same_sha256(manifest_identity, _require_mapping(best_identities.get("manifest"), "v6 manifest identity"), "manifest")
    _same_sha256(checkpoint_identity, _require_mapping(best_identities.get("checkpoint"), "v6 checkpoint identity"), "checkpoint")
    best_teacher_identity = _require_mapping(best_identities.get("teacher"), "v6 teacher identity")
    _same_sha256(
        teacher_identity,
        _require_mapping(best_teacher_identity.get("checkpoint"), "v6 teacher checkpoint identity"),
        "teacher",
    )
    actual_teacher_sha256 = teacher_identity.get("sha256")
    if not isinstance(actual_teacher_sha256, str) or not hmac.compare_digest(
        actual_teacher_sha256, expected_teacher_sha256
    ):
        raise ValueError("teacher SHA-256 does not match the explicit expected digest")

    work_size_value = data_config.get("work_size")
    if not isinstance(work_size_value, (list, tuple)) or len(work_size_value) != 2:
        raise ValueError("config.data.work_size must be [height,width]")
    work_size = (int(work_size_value[0]), int(work_size_value[1]))
    if work_size[0] != work_size[1] or list(work_size) != best_protocol.get("work_size"):
        raise ValueError("work_size is non-square or diverges from v6")
    feature_stride = int(model_config.get("feature_stride", 8))
    if feature_stride != int(best_protocol.get("feature_stride", -1)):
        raise ValueError("feature_stride diverges from v6")
    max_residual_consistency = float(loss_config.get("max_residual_consistency", 1.0))
    max_valid_flow = float(loss_config.get("max_valid_flow", 1000.0))
    if max_residual_consistency != float(best_protocol.get("max_residual_consistency")):
        raise ValueError("max_residual_consistency diverges from v6")
    if max_valid_flow != float(best_protocol.get("max_valid_flow")):
        raise ValueError("max_valid_flow diverges from v6")

    dataset = DocumentFlowDataset(manifest, work_size, augment_guide=False)
    indices = _sample_indices(len(dataset), sample_count)
    if indices != best_protocol.get("selected_indices"):
        raise ValueError("selected dataset indices diverge from v6")
    selected_digest = hashlib.sha256(_canonical_json(indices)).hexdigest()
    if selected_digest != best_protocol.get("selected_indices_sha256"):
        raise ValueError("selected dataset index digest diverges from v6")
    raw_augment = _require_mapping(
        data_config.get("source_geometry_augment"),
        "config.data.source_geometry_augment",
    )
    augment = SourceGeometryAugment.from_config(raw_augment)
    source_protocol = _require_mapping(
        best_protocol.get("source_full_geometry"), "v6 source_full_geometry"
    )
    if int(seed) != int(source_protocol.get("seed", -1)):
        raise ValueError("requested seed diverges from v6")
    transformations_per_sample = int(source_protocol.get("transformations_per_sample", -1))
    if transformations_per_sample != 1:
        raise ValueError("v7 requires the frozen one-transform-per-sample v6 protocol")
    full_plan, regenerated_protocol = build_full_geometry_plan(
        indices,
        transformations_per_sample=transformations_per_sample,
        seed=int(seed),
    )
    if regenerated_protocol["seed_plan_sha256"] != best_protocol.get("full_geometry_seed_plan_sha256"):
        raise ValueError("full-geometry seed plan diverges from v6")

    best_samples = best_results.get("samples")
    if not isinstance(best_samples, list) or len(best_samples) != len(indices):
        raise ValueError("v6 sample records do not match the selected sample count")
    rotation_edges = source_protocol.get("rotation_bin_edges_deg")
    if not isinstance(rotation_edges, list):
        raise ValueError("v6 rotation bin edges are missing")

    requested_device = torch.device(device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run this diagnostic on a GPU node")
        device_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        torch.cuda.set_device(device_index)
        requested_device = torch.device("cuda", device_index)
    teacher = teacher_factory(
        teacher_path,
        device=requested_device,
        input_size=int(model_config.get("prior_torchscript_size", 512)),
        flow_size=int(
            model_config.get(
                "prior_torchscript_flow_size",
                model_config.get("prior_torchscript_size", 512),
            )
        ),
        blur_kernel=int(model_config.get("prior_torchscript_blur_kernel", 39)),
        autocast_dtype=str(model_config.get("prior_torchscript_autocast_dtype", "float16")),
        requires_logical_cuda0=bool(
            model_config.get("prior_torchscript_requires_logical_cuda0", False)
        ),
        expected_sha256=expected_teacher_sha256,
    ).eval()
    if not isinstance(teacher, nn.Module):
        raise TypeError("teacher_factory must return an nn.Module")

    capacity_aggregates = {cap: _Accumulator() for cap in caps}
    structural_states = {cap: _new_aggregate_state() for cap in caps}
    sample_reports: list[dict[str, Any]] = []
    new_support_weighted: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    new_support_pixels = 0
    grid_key_by_cap = {cap: _grid_key(residual_target_iterations, cap) for cap in caps}

    variants = _full_geometry_oracle_variants(
        dataset,
        indices,
        full_plan,
        augment,
        rotation_edges,
    )
    for ordinal, variant in enumerate(variants):
        if ordinal >= len(best_samples):
            raise RuntimeError("v7 replay produced more variants than v6")
        v6_sample = _require_mapping(best_samples[ordinal], f"v6 sample[{ordinal}]")
        if (
            v6_sample.get("dataset_index") != variant.dataset_index
            or v6_sample.get("id") != variant.sample_id
            or v6_sample.get("full_geometry_seed") != variant.full_geometry_seed
            or v6_sample.get("oracle_quarter_turn_deg") != variant.oracle_quarter_turn_deg
            or not math.isclose(
                float(v6_sample.get("injected_rotation_deg")),
                variant.injected_rotation_deg,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
        ):
            raise RuntimeError(f"v7 replay variant {ordinal} diverges from v6 metadata")
        expected_homography = torch.tensor(v6_sample.get("source_homography"), dtype=torch.float64)
        actual_homography = variant.source_homography
        if actual_homography is None or not torch.allclose(
            actual_homography.double(), expected_homography, rtol=0.0, atol=1.0e-7
        ):
            raise RuntimeError(f"v7 replay variant {ordinal} homography diverges from v6")
        c4 = _require_mapping(v6_sample.get("c4_best_of_four"), f"v6 sample[{ordinal}] C4")
        if c4.get("selection_uses_ground_truth_flow") is not True:
            raise ValueError("v6 sample selection is not explicitly GT-flow based")
        selected_turn = int(c4.get("best_teacher_epe_quarter_turn_deg"))
        if selected_turn not in _QUARTER_TURNS:
            raise ValueError("v6 selected an invalid quarter turn")
        if c4.get("best_capacity_quarter_turn_by_grid", {}).get(
            grid_key_by_cap[selected_cap]
        ) != selected_turn:
            raise ValueError("v6 best-EPE and selected-cell capacity labels disagree")

        nearest_turn = variant.oracle_quarter_turn_deg
        inverse_nearest = 180 if abs(nearest_turn) == 180 else -nearest_turn
        augmented_source = rotate_source_by_quarter_turn(
            variant.canonicalized_warped, inverse_nearest
        )
        canonical_source_cpu = (
            variant.canonicalized_warped
            if selected_turn == nearest_turn
            else rotate_source_by_quarter_turn(augmented_source, selected_turn)
        )
        canonical_source = canonical_source_cpu.unsqueeze(0).to(requested_device)
        target_image = variant.target.unsqueeze(0).to(requested_device)
        target_flow = variant.target_flow.unsqueeze(0).to(requested_device)
        valid = variant.valid.unsqueeze(0).to(requested_device)
        canonical_target = transform_backward_flow_source_map_by_quarter_turn(
            target_flow, [selected_turn]
        )
        with torch.inference_mode():
            canonical_teacher = teacher(canonical_source).float()
        if not bool(torch.isfinite(canonical_teacher).all()):
            raise ValueError("teacher produced NaN or infinite flow")

        capacity_reports: dict[str, Any] = {}
        for cap in caps:
            stats = capacity_sample_statistics(
                canonical_teacher,
                canonical_target,
                valid,
                max_residual_px=cap,
                residual_target_iterations=residual_target_iterations,
                max_residual_consistency=max_residual_consistency,
                max_valid_flow=max_valid_flow,
                feature_stride=feature_stride,
            )[0]
            capacity_aggregates[cap].add(stats)
            capacity_report = stats.as_report()
            capacity_reports[grid_key_by_cap[cap]] = capacity_report
            candidate = _candidate_record(v6_sample, selected_turn)
            expected_by_grid = _require_mapping(
                candidate.get("canonical_metrics_by_capacity_grid"),
                "v6 candidate capacity metrics",
            )
            expected_metric = _require_mapping(
                expected_by_grid.get(grid_key_by_cap[cap]),
                f"v6 candidate metric {grid_key_by_cap[cap]}",
            )
            _assert_capacity_metric_match(capacity_report, expected_metric)

        with torch.no_grad(), torch.autocast(
            device_type=requested_device.type, enabled=False
        ):
            teacher_float = canonical_teacher.detach().float()
            target_float = canonical_target.detach().float()
            finite = torch.isfinite(target_float).all(dim=1, keepdim=True)
            magnitude = torch.linalg.vector_norm(target_float, dim=1, keepdim=True)
            valid_mask = valid.bool() & finite & (magnitude < max_valid_flow)
            safe_target = torch.where(finite.expand_as(target_float), target_float, teacher_float)
            residual, consistency = residual_from_composed_flow(
                teacher_float,
                safe_target,
                iterations=residual_target_iterations,
            )
            residual_finite = torch.isfinite(residual).all(dim=1, keepdim=True)
            safe_residual = torch.where(torch.isfinite(residual), residual, torch.zeros_like(residual))
            solver_valid = (
                valid_mask
                & residual_finite
                & torch.isfinite(consistency)
                & flow_valid_mask(safe_residual, residual.shape[-2:])
                & (consistency <= max_residual_consistency)
            )
            low_size = (
                target_float.shape[-2] // feature_stride,
                target_float.shape[-1] // feature_stride,
            )
            low_residual = resize_backward_flow(
                safe_residual,
                low_size,
                source_size_from=target_float.shape[-2:],
                source_size_to=low_size,
            )
            restored_residual = resize_backward_flow(
                low_residual,
                target_float.shape[-2:],
                source_size_from=low_size,
                source_size_to=target_float.shape[-2:],
            )
            stride_flow = compose_backward_flows(teacher_float, restored_residual)
            trainable_masks = {
                cap: solver_valid
                & (safe_residual[:, 0:1].abs() <= cap)
                & (safe_residual[:, 1:2].abs() <= cap)
                for cap in caps
            }
            structures = target_structure_maps(
                target_image.float(),
                edge_threshold=float(loss_config.get("structure_edge_threshold", 0.08)),
                edge_temperature=float(loss_config.get("structure_edge_temperature", 0.04)),
                line_kernel=int(loss_config.get("structure_line_kernel", 15)),
                line_threshold=float(loss_config.get("structure_line_threshold", 0.20)),
                line_temperature=float(loss_config.get("structure_line_temperature", 0.05)),
            )

        structural_by_cap: dict[str, Any] = {}
        for cap in caps:
            metric, raw = _sample_structural_metrics(
                canonical_source=canonical_source.float(),
                target_image=target_image.float(),
                teacher_flow=teacher_float,
                target_flow=target_float,
                stride_flow=stride_flow,
                valid_mask=valid_mask,
                trainable_mask=trainable_masks[cap],
                structures=structures,
            )
            structural_by_cap[f"max_residual_px={cap:g}"] = metric
            _add_aggregate(structural_states[cap], metric, raw)

        with torch.no_grad():
            reference = backward_warp(canonical_source.float(), target_float, padding_mode="border")
            candidate_rectified = backward_warp(
                canonical_source.float(), stride_flow, padding_mode="border"
            )
            new_mask = trainable_masks[selected_cap] & ~trainable_masks[baseline_cap]
            new_report, new_weighted = _new_support_metrics(
                stride_flow=stride_flow,
                target_flow=target_float,
                candidate_rectified=candidate_rectified,
                reference=reference,
                new_mask=new_mask,
            )
        new_support_pixels += int(new_report["pixels"])
        for key, (numerator, denominator) in new_weighted.items():
            new_support_weighted[key][0] += numerator
            new_support_weighted[key][1] += denominator

        sample_reports.append(
            {
                "dataset_index": variant.dataset_index,
                "id": variant.sample_id,
                "full_geometry_seed": variant.full_geometry_seed,
                "injected_rotation_deg": variant.injected_rotation_deg,
                "nearest_angle_quarter_turn_deg": nearest_turn,
                "selected_best_teacher_epe_quarter_turn_deg": selected_turn,
                "best_teacher_epe_top1_top2_margin_px": c4.get(
                    "best_teacher_epe_top1_top2_margin_px"
                ),
                "capacity_replay": capacity_reports,
                "structural_metrics_by_cap": structural_by_cap,
                "selected_new_support_vs_baseline": new_report,
            }
        )

    if len(sample_reports) != len(best_samples):
        raise RuntimeError("v7 replay produced fewer variants than v6")

    _assert_path_matches_identity(config_identity)
    _assert_path_matches_identity(manifest_identity)
    _assert_path_matches_identity(checkpoint_identity)
    _assert_path_matches_identity(teacher_identity)
    _assert_path_matches_identity(best_identity)
    for identity in implementation_identities.values():
        _assert_path_matches_identity(identity)

    structural_grid = [
        {
            "residual_target_iterations": residual_target_iterations,
            "max_residual_px": cap,
            "capacity_replay": capacity_aggregates[cap].report(),
            "structural_metrics": _finalize_aggregate(structural_states[cap]),
        }
        for cap in caps
    ]
    selected_entry = next(
        value for value in structural_grid if value["max_residual_px"] == selected_cap
    )
    new_support_aggregate = {
        "baseline_max_residual_px": baseline_cap,
        "selected_max_residual_px": selected_cap,
        "pixels": new_support_pixels,
        "fraction_of_selected_eval_pixels": _ratio(
            new_support_pixels,
            selected_entry["structural_metrics"]["eval_pixels"],
        ),
        **{
            key: _ratio(values[0], values[1])
            for key, values in new_support_weighted.items()
        },
    }
    decision = _evaluate_structural_policy(
        capacity_failures=capacity_failures,
        metrics=selected_entry["structural_metrics"],
        new_support=new_support_aggregate,
    )

    completed_at = time.time()
    report = {
        "report_version": REPORT_VERSION,
        "kind": REPORT_KIND,
        "diagnostic_only": True,
        "can_approve_production": False,
        "uses_ground_truth_flow_for_candidate_selection": True,
        "uses_ground_truth_flow_for_oracle_residual": True,
        "identities": {
            "config": config_identity,
            "checkpoint": {
                **checkpoint_identity,
                "role": "migration provenance only; payload was not loaded",
            },
            "teacher": {
                "checkpoint": teacher_identity,
                "expected_sha256": expected_teacher_sha256,
            },
            "manifest": {
                **manifest_identity,
                "split": split,
                "record_count": len(dataset),
            },
            "bound_best_of_c4_v6_report": best_identity,
            "implementation": implementation_identities,
        },
        "protocol": {
            "scope": "v6_best_teacher_epe_selected_stride8_structural_safety_v7",
            "work_size": list(work_size),
            "selected_indices": indices,
            "selected_indices_sha256": selected_digest,
            "sample_count": len(sample_reports),
            "seed": int(seed),
            "full_geometry_seed_plan_sha256": regenerated_protocol[
                "seed_plan_sha256"
            ],
            "selection": {
                "source": "bound v6 per-sample best_teacher_epe_quarter_turn_deg",
                "candidate_order": list(_QUARTER_TURNS),
                "deployment_available": False,
                "requires_best_epe_equal_capacity_label_at_selected_cell": True,
            },
            "residual_target_iterations": residual_target_iterations,
            "residual_cap_sweep_px": list(caps),
            "selected_max_residual_px": selected_cap,
            "baseline_max_residual_px": baseline_cap,
            "feature_stride": feature_stride,
            "max_residual_consistency": max_residual_consistency,
            "max_valid_flow": max_valid_flow,
            "stride_oracle": (
                "training-identical residual target; absolute-map downsample/upscale; "
                "metrics restricted to solver-valid per-axis-cap support"
            ),
            "texture_reference": (
                "same canonical source sampled with canonical GT flow; isolates "
                "stride residual geometry/resampling from source-target appearance mismatch"
            ),
            "policy": {
                "id": POLICY_ID,
                "capacity_thresholds": CAPACITY_THRESHOLDS,
                "structural_thresholds": STRUCTURAL_THRESHOLDS,
                "topology_threshold_source": (
                    "configs/typical_v33_quality_v2.yaml validation/inference gates"
                ),
            },
            "batch_size": 1,
        },
        "runtime": {
            **_runtime_identity(requested_device),
            "started_unix_seconds": started_at,
            "completed_unix_seconds": completed_at,
            "elapsed_seconds": float(time.monotonic() - started_monotonic),
        },
        "results": {
            "structural_grid": structural_grid,
            "selected_cell": {
                "residual_target_iterations": residual_target_iterations,
                "max_residual_px": selected_cap,
                "capacity_replay": selected_entry["capacity_replay"],
                "structural_metrics": selected_entry["structural_metrics"],
            },
            "selected_new_support_vs_baseline": new_support_aggregate,
            "decision": decision,
            "samples": sample_reports,
        },
    }
    _atomic_write_json(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/unified_v3_3_teacher_anchor.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--best-of-c4-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--residual-target-iterations", type=int, default=24)
    parser.add_argument(
        "--residual-cap-sweep",
        type=float,
        nargs="+",
        default=list(DEFAULT_CAP_SWEEP),
    )
    parser.add_argument("--selected-max-residual-px", type=float, default=40.0)
    parser.add_argument("--baseline-max-residual-px", type=float, default=24.0)
    args = parser.parse_args(argv)
    if args.threads < 1:
        raise ValueError("threads must be at least one")
    torch.set_num_threads(args.threads)
    report = run_structural_safety_diagnostic(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        teacher_path=args.teacher,
        expected_teacher_sha256=args.expected_teacher_sha256,
        best_of_c4_report_path=args.best_of_c4_report,
        output_path=args.output,
        manifest_path=args.manifest,
        split=args.split,
        sample_count=args.sample_count,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        residual_target_iterations=args.residual_target_iterations,
        residual_cap_sweep=args.residual_cap_sweep,
        selected_max_residual_px=args.selected_max_residual_px,
        baseline_max_residual_px=args.baseline_max_residual_px,
    )
    selected = report["results"]["selected_cell"]
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().absolute()),
                "kind": report["kind"],
                "diagnostic_only": report["diagnostic_only"],
                "can_approve_production": report["can_approve_production"],
                "selected_cell": {
                    "residual_target_iterations": selected[
                        "residual_target_iterations"
                    ],
                    "max_residual_px": selected["max_residual_px"],
                    "capacity_replay": selected["capacity_replay"],
                    "structural_metrics": selected["structural_metrics"],
                },
                "selected_new_support_vs_baseline": report["results"][
                    "selected_new_support_vs_baseline"
                ],
                "decision": report["results"]["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
