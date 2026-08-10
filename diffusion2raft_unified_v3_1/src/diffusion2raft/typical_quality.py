"""Pure, fail-closed quality policy for the v3.3 typical all-40 run.

The evaluator deliberately performs no filesystem I/O.  Callers parse policy
text with :func:`parse_quality_policy_yaml` and supply already-loaded
validation, LSD, and per-image inference records.  A malformed contract raises
``QualityGateInputError``; a well-formed run always returns a structured pass
or failure report.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import PurePath
from typing import Any, Mapping, Sequence

import yaml


POLICY_SCHEMA_VERSION = 1
POLICY_ID = "typical_v33_quality_v1"
SUBJECTS = {
    "candidate": "v33_best",
    "reference": "target_first",
    "anchor": "v33_anchor",
}
LINE_CANDIDATES = {
    "candidate": "v33_best_full",
    "reference": "target_first",
    "anchor": "v33_anchor_full",
}
LINE_METRIC = "orientation_error_deg_length_weighted"
FROZEN_VALIDATION_THRESHOLDS = {
    "max_epe_exclusive": 5.7501,
    "min_epe_gain_exclusive": 0.0,
    "min_final_win_rate_exclusive": 0.5,
    "max_fold_rate_exclusive": 0.0004,
    "min_jacobian_p01_inclusive": 0.01,
}
FROZEN_TYPICAL_THRESHOLDS = {
    "expected_images": 40,
    "lsd_metric": LINE_METRIC,
    "max_mean_delta_reference_deg_inclusive": 0.0,
    "min_reference_win_rate_inclusive": 0.75,
    "max_reference_delta_p95_deg_inclusive": 2.0,
    "max_reference_worst_delta_deg_inclusive": 5.0,
    "min_reference_total_line_length_ratio_inclusive": 0.70,
    "min_reference_line_length_ratio_p05_inclusive": 0.50,
}
FROZEN_INFERENCE_THRESHOLDS = {
    "max_mean_fold_rate_inclusive": 0.0004,
    "max_per_image_fold_rate_inclusive": 0.001,
    "min_per_image_jacobian_p01_inclusive": 0.01,
    "max_mean_evaluation_valid_drop_inclusive": 0.01,
    "max_per_image_evaluation_valid_drop_inclusive": 0.05,
}
FROZEN_LINE_REPORT_CONFIG = {
    "max_dimension": 1600,
    "min_length_fraction": 0.03,
    "axis_threshold_deg": 5.0,
    "mask_min_coverage": 0.90,
    "candidate_suffixes": ["_rectified", "-rectified", ".rectified"],
    "mask_suffixes": [
        "_evaluation_valid",
        "_eval_valid",
        "_valid",
        "-valid",
        ".valid",
    ],
}


class QualityGateInputError(ValueError):
    """The policy or one of the supplied metric contracts is malformed."""


@dataclass(frozen=True)
class ValidationPolicy:
    required_stage: str
    required_feature_backend: str
    max_epe_exclusive: float
    min_epe_gain_exclusive: float
    min_final_win_rate_exclusive: float
    max_fold_rate_exclusive: float
    min_jacobian_p01_inclusive: float
    require_line_epe_below_anchor: bool
    require_line_straightness_below_anchor: bool


@dataclass(frozen=True)
class TypicalPolicy:
    expected_images: int
    lsd_metric: str
    max_mean_delta_reference_deg_inclusive: float
    min_reference_win_rate_inclusive: float
    max_reference_delta_p95_deg_inclusive: float
    max_reference_worst_delta_deg_inclusive: float
    min_reference_total_line_length_ratio_inclusive: float
    min_reference_line_length_ratio_p05_inclusive: float


@dataclass(frozen=True)
class InferencePolicy:
    max_mean_fold_rate_inclusive: float
    max_per_image_fold_rate_inclusive: float
    min_per_image_jacobian_p01_inclusive: float
    max_mean_evaluation_valid_drop_inclusive: float
    max_per_image_evaluation_valid_drop_inclusive: float


@dataclass(frozen=True)
class QualityPolicy:
    """Validated v1 policy.  ``as_dict`` is JSON/YAML-serializable."""

    schema_version: int
    policy_id: str
    validation: ValidationPolicy
    typical: TypicalPolicy
    inference: InferencePolicy

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "subjects": dict(SUBJECTS),
            "validation": {
                "required_stage": self.validation.required_stage,
                "required_feature_backend": (
                    self.validation.required_feature_backend
                ),
                "max_epe_exclusive": self.validation.max_epe_exclusive,
                "min_epe_gain_exclusive": (
                    self.validation.min_epe_gain_exclusive
                ),
                "min_final_win_rate_exclusive": (
                    self.validation.min_final_win_rate_exclusive
                ),
                "max_fold_rate_exclusive": (
                    self.validation.max_fold_rate_exclusive
                ),
                "min_jacobian_p01_inclusive": (
                    self.validation.min_jacobian_p01_inclusive
                ),
                "require_line_epe_below_anchor": (
                    self.validation.require_line_epe_below_anchor
                ),
                "require_line_straightness_below_anchor": (
                    self.validation.require_line_straightness_below_anchor
                ),
            },
            "typical": {
                "expected_images": self.typical.expected_images,
                "line_candidates": dict(LINE_CANDIDATES),
                "lsd_metric": self.typical.lsd_metric,
                "max_mean_delta_reference_deg_inclusive": (
                    self.typical.max_mean_delta_reference_deg_inclusive
                ),
                "min_reference_win_rate_inclusive": (
                    self.typical.min_reference_win_rate_inclusive
                ),
                "max_reference_delta_p95_deg_inclusive": (
                    self.typical.max_reference_delta_p95_deg_inclusive
                ),
                "max_reference_worst_delta_deg_inclusive": (
                    self.typical.max_reference_worst_delta_deg_inclusive
                ),
                "min_reference_total_line_length_ratio_inclusive": (
                    self.typical.min_reference_total_line_length_ratio_inclusive
                ),
                "min_reference_line_length_ratio_p05_inclusive": (
                    self.typical.min_reference_line_length_ratio_p05_inclusive
                ),
            },
            "inference": {
                "max_mean_fold_rate_inclusive": (
                    self.inference.max_mean_fold_rate_inclusive
                ),
                "max_per_image_fold_rate_inclusive": (
                    self.inference.max_per_image_fold_rate_inclusive
                ),
                "min_per_image_jacobian_p01_inclusive": (
                    self.inference.min_per_image_jacobian_p01_inclusive
                ),
                "max_mean_evaluation_valid_drop_inclusive": (
                    self.inference.max_mean_evaluation_valid_drop_inclusive
                ),
                "max_per_image_evaluation_valid_drop_inclusive": (
                    self.inference.max_per_image_evaluation_valid_drop_inclusive
                ),
            },
        }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityGateInputError(f"{label} must be a mapping")
    return value


def _strict_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise QualityGateInputError(
            f"{label} keys must be exactly {sorted(expected)!r}; "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise QualityGateInputError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise QualityGateInputError(f"{label} must be a finite number")
    return result


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise QualityGateInputError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise QualityGateInputError(f"{label} must be >= {minimum}")
    return result


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise QualityGateInputError(f"{label} must be a boolean")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualityGateInputError(f"{label} must be a non-empty string")
    return value


def _assert_all_numbers_finite(value: Any, *, label: str) -> None:
    """Reject a NaN/Inf anywhere in a supplied metric subtree."""

    if isinstance(value, bool):
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise QualityGateInputError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_all_numbers_finite(item, label=f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _assert_all_numbers_finite(item, label=f"{label}[{index}]")


def _require_range(
    value: float,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> None:
    if minimum is not None:
        valid = value > minimum if minimum_exclusive else value >= minimum
        if not valid:
            operator = ">" if minimum_exclusive else ">="
            raise QualityGateInputError(f"{label} must be {operator} {minimum}")
    if maximum is not None and value > maximum:
        raise QualityGateInputError(f"{label} must be <= {maximum}")


def parse_quality_policy(payload: Mapping[str, Any]) -> QualityPolicy:
    """Validate a mapping against the exact v1 quality-policy schema."""

    root = _mapping(payload, label="policy")
    _strict_keys(
        root,
        {
            "schema_version",
            "policy_id",
            "subjects",
            "validation",
            "typical",
            "inference",
        },
        label="policy",
    )
    version = _integer(root["schema_version"], label="policy.schema_version")
    if version != POLICY_SCHEMA_VERSION:
        raise QualityGateInputError(
            f"unsupported policy.schema_version={version}; "
            f"expected {POLICY_SCHEMA_VERSION}"
        )
    policy_id = _string(root["policy_id"], label="policy.policy_id")
    if policy_id != POLICY_ID:
        raise QualityGateInputError(
            f"policy.policy_id must be {POLICY_ID!r}"
        )

    subjects = _mapping(root["subjects"], label="policy.subjects")
    _strict_keys(subjects, set(SUBJECTS), label="policy.subjects")
    if dict(subjects) != SUBJECTS:
        raise QualityGateInputError(
            f"policy.subjects is fixed to {SUBJECTS!r}"
        )

    validation_raw = _mapping(root["validation"], label="policy.validation")
    validation_keys = {
        "required_stage",
        "required_feature_backend",
        "max_epe_exclusive",
        "min_epe_gain_exclusive",
        "min_final_win_rate_exclusive",
        "max_fold_rate_exclusive",
        "min_jacobian_p01_inclusive",
        "require_line_epe_below_anchor",
        "require_line_straightness_below_anchor",
    }
    _strict_keys(validation_raw, validation_keys, label="policy.validation")
    required_stage = _string(
        validation_raw["required_stage"], label="policy.validation.required_stage"
    )
    required_backend = _string(
        validation_raw["required_feature_backend"],
        label="policy.validation.required_feature_backend",
    )
    if required_stage != "unified" or required_backend != "qwen":
        raise QualityGateInputError(
            "v1 requires validation stage='unified' and feature backend='qwen'"
        )
    max_epe = _finite(
        validation_raw["max_epe_exclusive"],
        label="policy.validation.max_epe_exclusive",
    )
    min_gain = _finite(
        validation_raw["min_epe_gain_exclusive"],
        label="policy.validation.min_epe_gain_exclusive",
    )
    min_win = _finite(
        validation_raw["min_final_win_rate_exclusive"],
        label="policy.validation.min_final_win_rate_exclusive",
    )
    max_fold = _finite(
        validation_raw["max_fold_rate_exclusive"],
        label="policy.validation.max_fold_rate_exclusive",
    )
    min_jacobian = _finite(
        validation_raw["min_jacobian_p01_inclusive"],
        label="policy.validation.min_jacobian_p01_inclusive",
    )
    line_epe_required = _boolean(
        validation_raw["require_line_epe_below_anchor"],
        label="policy.validation.require_line_epe_below_anchor",
    )
    line_straightness_required = _boolean(
        validation_raw["require_line_straightness_below_anchor"],
        label="policy.validation.require_line_straightness_below_anchor",
    )
    if not line_epe_required or not line_straightness_required:
        raise QualityGateInputError(
            "v1 requires both anchor-relative line improvements"
        )
    _require_range(max_epe, label="max_epe_exclusive", minimum=0.0, minimum_exclusive=True)
    _require_range(min_win, label="min_final_win_rate_exclusive", minimum=0.0, maximum=1.0)
    _require_range(max_fold, label="max_fold_rate_exclusive", minimum=0.0, maximum=1.0)

    typical_raw = _mapping(root["typical"], label="policy.typical")
    typical_keys = {
        "expected_images",
        "line_candidates",
        "lsd_metric",
        "max_mean_delta_reference_deg_inclusive",
        "min_reference_win_rate_inclusive",
        "max_reference_delta_p95_deg_inclusive",
        "max_reference_worst_delta_deg_inclusive",
        "min_reference_total_line_length_ratio_inclusive",
        "min_reference_line_length_ratio_p05_inclusive",
    }
    _strict_keys(typical_raw, typical_keys, label="policy.typical")
    expected_images = _integer(
        typical_raw["expected_images"],
        label="policy.typical.expected_images",
        minimum=1,
    )
    if expected_images != 40:
        raise QualityGateInputError("v1 is fixed to the typical all-40 set")
    line_candidates = _mapping(
        typical_raw["line_candidates"], label="policy.typical.line_candidates"
    )
    _strict_keys(
        line_candidates, set(LINE_CANDIDATES), label="policy.typical.line_candidates"
    )
    if dict(line_candidates) != LINE_CANDIDATES:
        raise QualityGateInputError(
            f"policy.typical.line_candidates is fixed to {LINE_CANDIDATES!r}"
        )
    lsd_metric = _string(
        typical_raw["lsd_metric"], label="policy.typical.lsd_metric"
    )
    if lsd_metric != LINE_METRIC:
        raise QualityGateInputError(
            f"policy.typical.lsd_metric must be {LINE_METRIC!r}"
        )
    typical_numbers = {
        key: _finite(typical_raw[key], label=f"policy.typical.{key}")
        for key in typical_keys
        if key
        not in {"expected_images", "line_candidates", "lsd_metric"}
    }
    _require_range(
        typical_numbers["min_reference_win_rate_inclusive"],
        label="min_reference_win_rate_inclusive",
        minimum=0.0,
        maximum=1.0,
    )
    for key in (
        "max_reference_delta_p95_deg_inclusive",
        "max_reference_worst_delta_deg_inclusive",
        "min_reference_total_line_length_ratio_inclusive",
        "min_reference_line_length_ratio_p05_inclusive",
    ):
        _require_range(typical_numbers[key], label=key, minimum=0.0)

    inference_raw = _mapping(root["inference"], label="policy.inference")
    inference_keys = {
        "max_mean_fold_rate_inclusive",
        "max_per_image_fold_rate_inclusive",
        "min_per_image_jacobian_p01_inclusive",
        "max_mean_evaluation_valid_drop_inclusive",
        "max_per_image_evaluation_valid_drop_inclusive",
    }
    _strict_keys(inference_raw, inference_keys, label="policy.inference")
    inference_numbers = {
        key: _finite(inference_raw[key], label=f"policy.inference.{key}")
        for key in inference_keys
    }
    for key in (
        "max_mean_fold_rate_inclusive",
        "max_per_image_fold_rate_inclusive",
        "max_mean_evaluation_valid_drop_inclusive",
        "max_per_image_evaluation_valid_drop_inclusive",
    ):
        _require_range(inference_numbers[key], label=key, minimum=0.0, maximum=1.0)
    if (
        inference_numbers["max_mean_fold_rate_inclusive"]
        > inference_numbers["max_per_image_fold_rate_inclusive"]
    ):
        raise QualityGateInputError(
            "max_mean_fold_rate_inclusive cannot exceed the per-image limit"
        )
    if (
        inference_numbers["max_mean_evaluation_valid_drop_inclusive"]
        > inference_numbers["max_per_image_evaluation_valid_drop_inclusive"]
    ):
        raise QualityGateInputError(
            "mean evaluation-valid drop cannot exceed the per-image limit"
        )

    policy = QualityPolicy(
        schema_version=version,
        policy_id=policy_id,
        validation=ValidationPolicy(
            required_stage=required_stage,
            required_feature_backend=required_backend,
            max_epe_exclusive=max_epe,
            min_epe_gain_exclusive=min_gain,
            min_final_win_rate_exclusive=min_win,
            max_fold_rate_exclusive=max_fold,
            min_jacobian_p01_inclusive=min_jacobian,
            require_line_epe_below_anchor=line_epe_required,
            require_line_straightness_below_anchor=line_straightness_required,
        ),
        typical=TypicalPolicy(
            expected_images=expected_images,
            lsd_metric=lsd_metric,
            **typical_numbers,
        ),
        inference=InferencePolicy(**inference_numbers),
    )
    observed_validation = {
        key: getattr(policy.validation, key)
        for key in FROZEN_VALIDATION_THRESHOLDS
    }
    observed_typical = {
        key: getattr(policy.typical, key) for key in FROZEN_TYPICAL_THRESHOLDS
    }
    observed_inference = {
        key: getattr(policy.inference, key)
        for key in FROZEN_INFERENCE_THRESHOLDS
    }
    if observed_validation != FROZEN_VALIDATION_THRESHOLDS:
        raise QualityGateInputError(
            "typical_v33_quality_v1 validation thresholds are frozen; "
            f"expected={FROZEN_VALIDATION_THRESHOLDS!r}"
        )
    if observed_typical != FROZEN_TYPICAL_THRESHOLDS:
        raise QualityGateInputError(
            "typical_v33_quality_v1 typical thresholds are frozen; "
            f"expected={FROZEN_TYPICAL_THRESHOLDS!r}"
        )
    if observed_inference != FROZEN_INFERENCE_THRESHOLDS:
        raise QualityGateInputError(
            "typical_v33_quality_v1 inference thresholds are frozen; "
            f"expected={FROZEN_INFERENCE_THRESHOLDS!r}"
        )
    return policy


def parse_quality_policy_yaml(text: str) -> QualityPolicy:
    """Parse policy YAML text without reading or writing the filesystem."""

    if not isinstance(text, str) or not text.strip():
        raise QualityGateInputError("policy YAML must be a non-empty string")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise QualityGateInputError(f"invalid policy YAML: {error}") from error
    return parse_quality_policy(_mapping(payload, label="policy YAML root"))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise QualityGateInputError("cannot compute a mean over an empty sequence")
    return math.fsum(values) / len(values)


def _quantile(values: Sequence[float], quantile: float) -> float:
    """NumPy-compatible linear quantile over finite, non-empty values."""

    if not values:
        raise QualityGateInputError("cannot compute a quantile over an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _valid_basename(value: Any, *, label: str) -> str:
    basename = _string(value, label=label)
    if PurePath(basename).name != basename or basename in {".", ".."}:
        raise QualityGateInputError(f"{label} must be a basename, got {basename!r}")
    return basename


@dataclass(frozen=True)
class _LineRow:
    orientation_error_deg: float
    total_line_length_px: float


def _line_index(
    report: Mapping[str, Any], candidate_name: str, *, expected_count: int
) -> dict[str, _LineRow]:
    candidates = _mapping(report.get("candidates"), label="line_report.candidates")
    if candidate_name not in candidates:
        raise QualityGateInputError(
            f"line_report.candidates is missing fixed candidate {candidate_name!r}"
        )
    candidate = _mapping(
        candidates[candidate_name],
        label=f"line_report.candidates.{candidate_name}",
    )
    if candidate.get("valid_mask_directory") is not None:
        raise QualityGateInputError(
            f"{candidate_name} must be an unmasked full-frame line candidate"
        )
    rows = candidate.get("per_image")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise QualityGateInputError(
            f"{candidate_name}.per_image must contain exactly {expected_count} rows"
        )
    result: dict[str, _LineRow] = {}
    for index, raw_row in enumerate(rows):
        label = f"{candidate_name}.per_image[{index}]"
        row = _mapping(raw_row, label=label)
        _assert_all_numbers_finite(row, label=label)
        basename = _valid_basename(row.get("basename"), label=f"{label}.basename")
        if basename in result:
            raise QualityGateInputError(
                f"{candidate_name}.per_image has duplicate basename {basename!r}"
            )
        if row.get("status") != "ok":
            raise QualityGateInputError(
                f"{candidate_name}/{basename} status must be 'ok'"
            )
        if row.get("evaluation_valid_mask") is not None:
            raise QualityGateInputError(
                f"{candidate_name}/{basename} must not use an evaluation mask"
            )
        metrics = _mapping(row.get("metrics"), label=f"{label}.metrics")
        line_count = _integer(
            metrics.get("line_count"), label=f"{label}.metrics.line_count", minimum=1
        )
        if line_count <= 0:  # defensive; minimum=1 already enforces this
            raise QualityGateInputError(f"{candidate_name}/{basename} has no lines")
        orientation = _finite(
            metrics.get(LINE_METRIC), label=f"{label}.metrics.{LINE_METRIC}"
        )
        length = _finite(
            metrics.get("total_line_length_px"),
            label=f"{label}.metrics.total_line_length_px",
        )
        _require_range(
            orientation,
            label=f"{candidate_name}/{basename} LSD orientation error",
            minimum=0.0,
            maximum=45.0,
        )
        _require_range(
            length,
            label=f"{candidate_name}/{basename} line length",
            minimum=0.0,
            minimum_exclusive=True,
        )
        result[basename] = _LineRow(orientation, length)

    summary = _mapping(candidate.get("summary"), label=f"{candidate_name}.summary")
    _assert_all_numbers_finite(summary, label=f"{candidate_name}.summary")
    expected_counts = {
        "indexed_images": expected_count,
        "paired_images": expected_count,
        "evaluated_images": expected_count,
        "missing_images": 0,
        "missing_masks": 0,
        "no_line_images": 0,
    }
    for field, expected in expected_counts.items():
        actual = _integer(summary.get(field), label=f"{candidate_name}.summary.{field}")
        if actual != expected:
            raise QualityGateInputError(
                f"{candidate_name}.summary.{field}={actual}, expected {expected}"
            )
    reported_mean = _finite(
        summary.get(f"image_mean_{LINE_METRIC}"),
        label=f"{candidate_name}.summary.image_mean_{LINE_METRIC}",
    )
    reported_length = _finite(
        summary.get("total_line_length_px"),
        label=f"{candidate_name}.summary.total_line_length_px",
    )
    computed_mean = _mean([row.orientation_error_deg for row in result.values()])
    computed_length = math.fsum(row.total_line_length_px for row in result.values())
    if not math.isclose(reported_mean, computed_mean, rel_tol=1e-10, abs_tol=1e-10):
        raise QualityGateInputError(
            f"{candidate_name} summary LSD mean disagrees with per_image rows"
        )
    if not math.isclose(reported_length, computed_length, rel_tol=1e-10, abs_tol=1e-8):
        raise QualityGateInputError(
            f"{candidate_name} summary line length disagrees with per_image rows"
        )
    return result


def _validation_record(
    validation: Mapping[str, Any], name: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], int, float, str, str]:
    raw = _mapping(validation.get(name), label=f"validation.{name}")
    _assert_all_numbers_finite(raw, label=f"validation.{name}")
    stage = _string(raw.get("stage"), label=f"validation.{name}.stage")
    backend = _string(
        raw.get("feature_backend"), label=f"validation.{name}.feature_backend"
    )
    epoch = _integer(
        raw.get("epoch_index"), label=f"validation.{name}.epoch_index"
    )
    residual_scale = _finite(
        raw.get("residual_scale"), label=f"validation.{name}.residual_scale"
    )
    metrics = _mapping(raw.get("metrics"), label=f"validation.{name}.metrics")
    return raw, metrics, epoch, residual_scale, stage, backend


def _inference_index(
    inference_metrics: Mapping[str, Any], name: str
) -> Mapping[str, Mapping[str, Any]]:
    raw = _mapping(
        inference_metrics.get(name), label=f"inference_metrics.{name}"
    )
    result: dict[str, Mapping[str, Any]] = {}
    for key, metadata in raw.items():
        basename = _valid_basename(key, label=f"inference_metrics.{name} key")
        item = _mapping(
            metadata, label=f"inference_metrics.{name}.{basename}"
        )
        _assert_all_numbers_finite(
            item, label=f"inference_metrics.{name}.{basename}"
        )
        result[basename] = item
    return result


def _gate_record(
    *,
    code: str,
    scope: str,
    passed: bool,
    actual: Any,
    operator: str,
    threshold: Any,
    basename: str | None = None,
) -> dict[str, Any]:
    subject = f" for {basename!r}" if basename is not None else ""
    record: dict[str, Any] = {
        "code": code,
        "scope": scope,
        "passed": bool(passed),
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "message": (
            f"{code}{subject}: expected actual {operator} {threshold!r}; "
            f"actual={actual!r}"
        ),
    }
    if basename is not None:
        record["basename"] = basename
    return record


def evaluate_typical_quality(
    policy: QualityPolicy | Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
    line_report: Mapping[str, Any],
    inference_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate v3.3 best against the fixed anchor and target-first reference.

    ``validation`` has exactly ``v33_anchor`` and ``v33_best`` records.  Each
    record supplies ``stage``, ``feature_backend``, ``epoch_index``,
    ``residual_scale``, and a ``metrics`` mapping.

    ``inference_metrics`` has the same two top-level keys and maps every
    canonical basename to its inference metadata.  Best metadata must include
    ``fold_rate``, ``jacobian_p01``, and ``evaluation_valid_fraction``; anchor
    metadata must include ``evaluation_valid_fraction``.
    """

    # Public dataclass constructors must not be an alternate, unvalidated
    # policy channel.  Re-parse even a QualityPolicy instance so the frozen v1
    # thresholds and every range/schema invariant are always enforced.
    policy = parse_quality_policy(
        policy.as_dict()
        if isinstance(policy, QualityPolicy)
        else _mapping(policy, label="policy")
    )
    validation = _mapping(validation, label="validation")
    _strict_keys(validation, {"v33_anchor", "v33_best"}, label="validation")
    line_report = _mapping(line_report, label="line_report")
    if _integer(line_report.get("schema_version"), label="line_report.schema_version") != 1:
        raise QualityGateInputError("line_report.schema_version must be 1")
    if line_report.get("metric_family") != "opencv_lsd_axis_alignment":
        raise QualityGateInputError(
            "line_report.metric_family must be 'opencv_lsd_axis_alignment'"
        )
    line_config = _mapping(
        line_report.get("config"), label="line_report.config"
    )
    _strict_keys(
        line_config, set(FROZEN_LINE_REPORT_CONFIG), label="line_report.config"
    )
    if dict(line_config) != FROZEN_LINE_REPORT_CONFIG:
        raise QualityGateInputError(
            "line_report.config does not match the frozen v1 OpenCV LSD protocol"
        )
    inference_metrics = _mapping(inference_metrics, label="inference_metrics")
    _strict_keys(
        inference_metrics,
        {"v33_anchor", "v33_best"},
        label="inference_metrics",
    )

    failures: list[dict[str, Any]] = []
    aggregate_gates: list[dict[str, Any]] = []

    def add_gate(**kwargs: Any) -> dict[str, Any]:
        record = _gate_record(**kwargs)
        aggregate_gates.append(record)
        if not record["passed"]:
            failures.append(dict(record))
        return record

    _, anchor_metrics, anchor_epoch, anchor_scale, anchor_stage, anchor_backend = (
        _validation_record(validation, "v33_anchor")
    )
    _, best_metrics, best_epoch, best_scale, best_stage, best_backend = (
        _validation_record(validation, "v33_best")
    )
    for name, stage, backend in (
        ("v33_anchor", anchor_stage, anchor_backend),
        ("v33_best", best_stage, best_backend),
    ):
        add_gate(
            code=f"validation.{name}.stage",
            scope="validation",
            passed=stage == policy.validation.required_stage,
            actual=stage,
            operator="==",
            threshold=policy.validation.required_stage,
        )
        add_gate(
            code=f"validation.{name}.feature_backend",
            scope="validation",
            passed=backend == policy.validation.required_feature_backend,
            actual=backend,
            operator="==",
            threshold=policy.validation.required_feature_backend,
        )

    required_best_metrics = (
        "epe",
        "prior_epe",
        "epe_gain",
        "final_win_rate",
        "fold_rate",
        "jacobian_p01",
        "line_epe",
        "line_straightness_error",
    )
    best_values = {
        name: _finite(
            best_metrics.get(name), label=f"validation.v33_best.metrics.{name}"
        )
        for name in required_best_metrics
    }
    anchor_values = {
        name: _finite(
            anchor_metrics.get(name),
            label=f"validation.v33_anchor.metrics.{name}",
        )
        for name in ("line_epe", "line_straightness_error")
    }
    for key in ("epe", "prior_epe", "line_epe", "line_straightness_error"):
        _require_range(best_values[key], label=f"best {key}", minimum=0.0)
    for key in ("line_epe", "line_straightness_error"):
        _require_range(anchor_values[key], label=f"anchor {key}", minimum=0.0)
    _require_range(
        best_values["final_win_rate"],
        label="best final_win_rate",
        minimum=0.0,
        maximum=1.0,
    )
    _require_range(
        best_values["fold_rate"],
        label="best fold_rate",
        minimum=0.0,
        maximum=1.0,
    )

    add_gate(
        code="validation.best_epoch_after_anchor",
        scope="validation",
        passed=best_epoch > anchor_epoch,
        actual=best_epoch,
        operator=">",
        threshold=anchor_epoch,
    )
    add_gate(
        code="validation.best_residual_scale_positive",
        scope="validation",
        passed=best_scale > 0.0,
        actual=best_scale,
        operator=">",
        threshold=0.0,
    )
    add_gate(
        code="validation.best_epe",
        scope="validation",
        passed=best_values["epe"] < policy.validation.max_epe_exclusive,
        actual=best_values["epe"],
        operator="<",
        threshold=policy.validation.max_epe_exclusive,
    )
    add_gate(
        code="validation.best_epe_gain",
        scope="validation",
        passed=(
            best_values["epe_gain"] > policy.validation.min_epe_gain_exclusive
        ),
        actual=best_values["epe_gain"],
        operator=">",
        threshold=policy.validation.min_epe_gain_exclusive,
    )
    expected_gain = best_values["prior_epe"] - best_values["epe"]
    add_gate(
        code="validation.best_computed_epe_gain",
        scope="validation",
        passed=expected_gain > policy.validation.min_epe_gain_exclusive,
        actual=expected_gain,
        operator=">",
        threshold=policy.validation.min_epe_gain_exclusive,
    )
    add_gate(
        code="validation.best_epe_gain_identity",
        scope="validation",
        passed=math.isclose(
            best_values["epe_gain"],
            expected_gain,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        actual=best_values["epe_gain"],
        operator="approximately == prior_epe - epe",
        threshold=expected_gain,
    )
    add_gate(
        code="validation.best_final_win_rate",
        scope="validation",
        passed=(
            best_values["final_win_rate"]
            > policy.validation.min_final_win_rate_exclusive
        ),
        actual=best_values["final_win_rate"],
        operator=">",
        threshold=policy.validation.min_final_win_rate_exclusive,
    )
    add_gate(
        code="validation.best_fold_rate",
        scope="validation",
        passed=(
            best_values["fold_rate"] < policy.validation.max_fold_rate_exclusive
        ),
        actual=best_values["fold_rate"],
        operator="<",
        threshold=policy.validation.max_fold_rate_exclusive,
    )
    add_gate(
        code="validation.best_jacobian_p01",
        scope="validation",
        passed=(
            best_values["jacobian_p01"]
            >= policy.validation.min_jacobian_p01_inclusive
        ),
        actual=best_values["jacobian_p01"],
        operator=">=",
        threshold=policy.validation.min_jacobian_p01_inclusive,
    )
    add_gate(
        code="validation.best_line_epe_below_anchor",
        scope="validation",
        passed=best_values["line_epe"] < anchor_values["line_epe"],
        actual=best_values["line_epe"],
        operator="<",
        threshold=anchor_values["line_epe"],
    )
    add_gate(
        code="validation.best_line_straightness_below_anchor",
        scope="validation",
        passed=(
            best_values["line_straightness_error"]
            < anchor_values["line_straightness_error"]
        ),
        actual=best_values["line_straightness_error"],
        operator="<",
        threshold=anchor_values["line_straightness_error"],
    )

    line_indexes = {
        role: _line_index(
            line_report, candidate_name, expected_count=policy.typical.expected_images
        )
        for role, candidate_name in LINE_CANDIDATES.items()
    }
    canonical_basenames = set(line_indexes["reference"])
    for role in ("candidate", "anchor"):
        actual = set(line_indexes[role])
        if actual != canonical_basenames:
            raise QualityGateInputError(
                f"line_report basename set mismatch for {role}; "
                f"missing={sorted(canonical_basenames - actual)!r}, "
                f"extra={sorted(actual - canonical_basenames)!r}"
            )
    if len(canonical_basenames) != policy.typical.expected_images:
        raise QualityGateInputError(
            f"line_report must cover exactly {policy.typical.expected_images} basenames"
        )

    anchor_inference = _inference_index(inference_metrics, "v33_anchor")
    best_inference = _inference_index(inference_metrics, "v33_best")
    for name, index in (
        ("v33_anchor", anchor_inference),
        ("v33_best", best_inference),
    ):
        actual = set(index)
        if actual != canonical_basenames:
            raise QualityGateInputError(
                f"inference_metrics.{name} basename set mismatch; "
                f"missing={sorted(canonical_basenames - actual)!r}, "
                f"extra={sorted(actual - canonical_basenames)!r}"
            )

    per_image: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    reference_errors: list[float] = []
    anchor_errors: list[float] = []
    reference_deltas: list[float] = []
    line_length_ratios: list[float] = []
    candidate_lengths: list[float] = []
    reference_lengths: list[float] = []
    best_fold_rates: list[float] = []
    best_jacobians: list[float] = []
    best_valid_fractions: list[float] = []
    anchor_valid_fractions: list[float] = []
    valid_drops: list[float] = []
    reference_wins = 0

    for basename in sorted(canonical_basenames, key=lambda value: (value.casefold(), value)):
        candidate_line = line_indexes["candidate"][basename]
        reference_line = line_indexes["reference"][basename]
        anchor_line = line_indexes["anchor"][basename]
        delta_reference = (
            candidate_line.orientation_error_deg
            - reference_line.orientation_error_deg
        )
        delta_anchor = (
            candidate_line.orientation_error_deg - anchor_line.orientation_error_deg
        )
        length_ratio = (
            candidate_line.total_line_length_px
            / reference_line.total_line_length_px
        )
        reference_win = delta_reference <= 0.0
        reference_wins += int(reference_win)

        best_metadata = best_inference[basename]
        anchor_metadata = anchor_inference[basename]
        best_fold = _finite(
            best_metadata.get("fold_rate"),
            label=f"inference_metrics.v33_best.{basename}.fold_rate",
        )
        best_jacobian = _finite(
            best_metadata.get("jacobian_p01"),
            label=f"inference_metrics.v33_best.{basename}.jacobian_p01",
        )
        best_valid = _finite(
            best_metadata.get("evaluation_valid_fraction"),
            label=(
                f"inference_metrics.v33_best.{basename}.evaluation_valid_fraction"
            ),
        )
        anchor_valid = _finite(
            anchor_metadata.get("evaluation_valid_fraction"),
            label=(
                f"inference_metrics.v33_anchor.{basename}.evaluation_valid_fraction"
            ),
        )
        _require_range(
            best_fold,
            label=f"{basename} fold_rate",
            minimum=0.0,
            maximum=1.0,
        )
        _require_range(
            best_valid,
            label=f"{basename} best evaluation_valid_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        _require_range(
            anchor_valid,
            label=f"{basename} anchor evaluation_valid_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        valid_drop = anchor_valid - best_valid

        image_gates = [
            _gate_record(
                code="typical.per_image.reference_worst_delta_deg",
                scope="per_image",
                passed=(
                    delta_reference
                    <= policy.typical.max_reference_worst_delta_deg_inclusive
                ),
                actual=delta_reference,
                operator="<=",
                threshold=policy.typical.max_reference_worst_delta_deg_inclusive,
                basename=basename,
            ),
            _gate_record(
                code="inference.per_image.fold_rate",
                scope="per_image",
                passed=(
                    best_fold
                    <= policy.inference.max_per_image_fold_rate_inclusive
                ),
                actual=best_fold,
                operator="<=",
                threshold=policy.inference.max_per_image_fold_rate_inclusive,
                basename=basename,
            ),
            _gate_record(
                code="inference.per_image.jacobian_p01",
                scope="per_image",
                passed=(
                    best_jacobian
                    >= policy.inference.min_per_image_jacobian_p01_inclusive
                ),
                actual=best_jacobian,
                operator=">=",
                threshold=policy.inference.min_per_image_jacobian_p01_inclusive,
                basename=basename,
            ),
            _gate_record(
                code="inference.per_image.evaluation_valid_drop",
                scope="per_image",
                passed=(
                    valid_drop
                    <= policy.inference.max_per_image_evaluation_valid_drop_inclusive
                ),
                actual=valid_drop,
                operator="<=",
                threshold=(
                    policy.inference.max_per_image_evaluation_valid_drop_inclusive
                ),
                basename=basename,
            ),
        ]
        image_failures = [dict(gate) for gate in image_gates if not gate["passed"]]
        failures.extend(image_failures)
        per_image.append(
            {
                "basename": basename,
                "passed": not image_failures,
                "failures": image_failures,
                "lsd": {
                    "candidate_error_deg": candidate_line.orientation_error_deg,
                    "reference_error_deg": reference_line.orientation_error_deg,
                    "anchor_error_deg": anchor_line.orientation_error_deg,
                    "delta_reference_deg": delta_reference,
                    "delta_anchor_deg": delta_anchor,
                    "reference_win": reference_win,
                },
                "line_length": {
                    "candidate_px": candidate_line.total_line_length_px,
                    "reference_px": reference_line.total_line_length_px,
                    "ratio_to_reference": length_ratio,
                },
                "inference": {
                    "best_fold_rate": best_fold,
                    "best_jacobian_p01": best_jacobian,
                    "best_evaluation_valid_fraction": best_valid,
                    "anchor_evaluation_valid_fraction": anchor_valid,
                    "evaluation_valid_drop": valid_drop,
                },
            }
        )

        candidate_errors.append(candidate_line.orientation_error_deg)
        reference_errors.append(reference_line.orientation_error_deg)
        anchor_errors.append(anchor_line.orientation_error_deg)
        reference_deltas.append(delta_reference)
        candidate_lengths.append(candidate_line.total_line_length_px)
        reference_lengths.append(reference_line.total_line_length_px)
        line_length_ratios.append(length_ratio)
        best_fold_rates.append(best_fold)
        best_jacobians.append(best_jacobian)
        best_valid_fractions.append(best_valid)
        anchor_valid_fractions.append(anchor_valid)
        valid_drops.append(valid_drop)

    candidate_mean = _mean(candidate_errors)
    reference_mean = _mean(reference_errors)
    anchor_mean = _mean(anchor_errors)
    reference_delta_p95 = _quantile(reference_deltas, 0.95)
    reference_worst_delta = max(reference_deltas)
    total_length_ratio = math.fsum(candidate_lengths) / math.fsum(reference_lengths)
    length_ratio_p05 = _quantile(line_length_ratios, 0.05)
    reference_win_rate = reference_wins / policy.typical.expected_images
    best_mean_fold = _mean(best_fold_rates)
    best_max_fold = max(best_fold_rates)
    best_min_jacobian = min(best_jacobians)
    best_mean_valid = _mean(best_valid_fractions)
    anchor_mean_valid = _mean(anchor_valid_fractions)
    mean_valid_drop = anchor_mean_valid - best_mean_valid
    worst_valid_drop = max(valid_drops)

    add_gate(
        code="typical.full_frame_lsd_mean_at_most_reference",
        scope="typical_summary",
        passed=(
            candidate_mean - reference_mean
            <= policy.typical.max_mean_delta_reference_deg_inclusive
        ),
        actual=candidate_mean - reference_mean,
        operator="<=",
        threshold=policy.typical.max_mean_delta_reference_deg_inclusive,
    )
    add_gate(
        code="typical.reference_win_rate",
        scope="typical_summary",
        passed=reference_win_rate >= policy.typical.min_reference_win_rate_inclusive,
        actual=reference_win_rate,
        operator=">=",
        threshold=policy.typical.min_reference_win_rate_inclusive,
    )
    add_gate(
        code="typical.reference_delta_p95_deg",
        scope="typical_summary",
        passed=(
            reference_delta_p95
            <= policy.typical.max_reference_delta_p95_deg_inclusive
        ),
        actual=reference_delta_p95,
        operator="<=",
        threshold=policy.typical.max_reference_delta_p95_deg_inclusive,
    )
    add_gate(
        code="typical.reference_worst_delta_deg",
        scope="typical_summary",
        passed=(
            reference_worst_delta
            <= policy.typical.max_reference_worst_delta_deg_inclusive
        ),
        actual=reference_worst_delta,
        operator="<=",
        threshold=policy.typical.max_reference_worst_delta_deg_inclusive,
    )
    add_gate(
        code="typical.reference_total_line_length_ratio",
        scope="typical_summary",
        passed=(
            total_length_ratio
            >= policy.typical.min_reference_total_line_length_ratio_inclusive
        ),
        actual=total_length_ratio,
        operator=">=",
        threshold=policy.typical.min_reference_total_line_length_ratio_inclusive,
    )
    add_gate(
        code="typical.reference_line_length_ratio_p05",
        scope="typical_summary",
        passed=(
            length_ratio_p05
            >= policy.typical.min_reference_line_length_ratio_p05_inclusive
        ),
        actual=length_ratio_p05,
        operator=">=",
        threshold=policy.typical.min_reference_line_length_ratio_p05_inclusive,
    )
    add_gate(
        code="inference.best_mean_fold_rate",
        scope="inference_summary",
        passed=(
            best_mean_fold <= policy.inference.max_mean_fold_rate_inclusive
        ),
        actual=best_mean_fold,
        operator="<=",
        threshold=policy.inference.max_mean_fold_rate_inclusive,
    )
    add_gate(
        code="inference.best_max_fold_rate",
        scope="inference_summary",
        passed=(
            best_max_fold <= policy.inference.max_per_image_fold_rate_inclusive
        ),
        actual=best_max_fold,
        operator="<=",
        threshold=policy.inference.max_per_image_fold_rate_inclusive,
    )
    add_gate(
        code="inference.best_min_jacobian_p01",
        scope="inference_summary",
        passed=(
            best_min_jacobian
            >= policy.inference.min_per_image_jacobian_p01_inclusive
        ),
        actual=best_min_jacobian,
        operator=">=",
        threshold=policy.inference.min_per_image_jacobian_p01_inclusive,
    )
    add_gate(
        code="inference.mean_evaluation_valid_drop",
        scope="inference_summary",
        passed=(
            mean_valid_drop
            <= policy.inference.max_mean_evaluation_valid_drop_inclusive
        ),
        actual=mean_valid_drop,
        operator="<=",
        threshold=policy.inference.max_mean_evaluation_valid_drop_inclusive,
    )
    add_gate(
        code="inference.worst_evaluation_valid_drop",
        scope="inference_summary",
        passed=(
            worst_valid_drop
            <= policy.inference.max_per_image_evaluation_valid_drop_inclusive
        ),
        actual=worst_valid_drop,
        operator="<=",
        threshold=policy.inference.max_per_image_evaluation_valid_drop_inclusive,
    )

    return {
        "schema_version": 1,
        "policy_id": policy.policy_id,
        "passed": not failures,
        "failures": failures,
        "per_image": per_image,
        "summary": {
            "expected_images": policy.typical.expected_images,
            "evaluated_images": len(per_image),
            "validation": {
                "anchor_epoch_index": anchor_epoch,
                "best_epoch_index": best_epoch,
                "anchor_residual_scale": anchor_scale,
                "best_residual_scale": best_scale,
                "anchor_metrics": anchor_values,
                "best_metrics": best_values,
            },
            "typical": {
                "candidate_mean_lsd_error_deg": candidate_mean,
                "reference_mean_lsd_error_deg": reference_mean,
                "anchor_mean_lsd_error_deg": anchor_mean,
                "mean_delta_reference_deg": candidate_mean - reference_mean,
                "reference_win_rate": reference_win_rate,
                "reference_delta_p95_deg": reference_delta_p95,
                "reference_worst_delta_deg": reference_worst_delta,
                "reference_total_line_length_ratio": total_length_ratio,
                "reference_line_length_ratio_p05": length_ratio_p05,
            },
            "inference": {
                "best_mean_fold_rate": best_mean_fold,
                "best_max_fold_rate": best_max_fold,
                "best_min_jacobian_p01": best_min_jacobian,
                "best_mean_evaluation_valid_fraction": best_mean_valid,
                "anchor_mean_evaluation_valid_fraction": anchor_mean_valid,
                "mean_evaluation_valid_drop": mean_valid_drop,
                "worst_evaluation_valid_drop": worst_valid_drop,
            },
            "gates": aggregate_gates,
        },
    }


__all__ = [
    "InferencePolicy",
    "QualityGateInputError",
    "QualityPolicy",
    "TypicalPolicy",
    "ValidationPolicy",
    "evaluate_typical_quality",
    "parse_quality_policy",
    "parse_quality_policy_yaml",
]
