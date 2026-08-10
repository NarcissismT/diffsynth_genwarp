"""Versioned dual-baseline release gate for the v3.3 typical all-40 run.

Version 1 remains frozen in :mod:`diffusion2raft.typical_quality` and proves
quality only against ``target_first``.  Version 2 reuses the exact same
validation, inference, and line thresholds, but independently applies every
typical line gate to both historical baselines.  A v1 report therefore cannot
silently acquire v2 release status.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import yaml

from . import typical_quality as v1


POLICY_SCHEMA_VERSION = 2
POLICY_ID = "typical_v33_quality_v2"
REFERENCE_NAMES = ("target_first", "target_second")
SUBJECTS = {
    "candidate": "v33_best",
    "references": list(REFERENCE_NAMES),
    "anchor": "v33_anchor",
}
LINE_CANDIDATES = {
    "candidate": "v33_best_full",
    "references": {
        "target_first": "target_first",
        "target_second": "target_second",
    },
    "anchor": "v33_anchor_full",
}

QualityGateInputError = v1.QualityGateInputError
ValidationPolicy = v1.ValidationPolicy
TypicalPolicy = v1.TypicalPolicy
InferencePolicy = v1.InferencePolicy


@dataclass(frozen=True)
class QualityPolicy:
    """Validated v2 policy with the frozen v1 numeric thresholds."""

    schema_version: int
    policy_id: str
    validation: ValidationPolicy
    typical: TypicalPolicy
    inference: InferencePolicy

    def as_dict(self) -> dict[str, Any]:
        legacy = _legacy_policy(self).as_dict()
        legacy.update(
            {
                "schema_version": self.schema_version,
                "policy_id": self.policy_id,
                "subjects": copy.deepcopy(SUBJECTS),
            }
        )
        legacy["typical"]["line_candidates"] = copy.deepcopy(
            LINE_CANDIDATES
        )
        return legacy


def _legacy_policy(policy: QualityPolicy) -> v1.QualityPolicy:
    return v1.QualityPolicy(
        schema_version=v1.POLICY_SCHEMA_VERSION,
        policy_id=v1.POLICY_ID,
        validation=policy.validation,
        typical=policy.typical,
        inference=policy.inference,
    )


def parse_quality_policy(payload: Mapping[str, Any]) -> QualityPolicy:
    """Validate the exact v2 schema without weakening the frozen v1 limits."""

    root = v1._mapping(payload, label="policy")
    v1._strict_keys(
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
    version = v1._integer(root["schema_version"], label="policy.schema_version")
    if version != POLICY_SCHEMA_VERSION:
        raise QualityGateInputError(
            f"unsupported policy.schema_version={version}; "
            f"expected {POLICY_SCHEMA_VERSION}"
        )
    policy_id = v1._string(root["policy_id"], label="policy.policy_id")
    if policy_id != POLICY_ID:
        raise QualityGateInputError(f"policy.policy_id must be {POLICY_ID!r}")

    subjects = v1._mapping(root["subjects"], label="policy.subjects")
    v1._strict_keys(subjects, set(SUBJECTS), label="policy.subjects")
    if dict(subjects) != SUBJECTS:
        raise QualityGateInputError(
            f"policy.subjects is fixed to {SUBJECTS!r}"
        )

    typical = v1._mapping(root["typical"], label="policy.typical")
    line_candidates = v1._mapping(
        typical.get("line_candidates"),
        label="policy.typical.line_candidates",
    )
    v1._strict_keys(
        line_candidates,
        set(LINE_CANDIDATES),
        label="policy.typical.line_candidates",
    )
    references = v1._mapping(
        line_candidates.get("references"),
        label="policy.typical.line_candidates.references",
    )
    v1._strict_keys(
        references,
        set(REFERENCE_NAMES),
        label="policy.typical.line_candidates.references",
    )
    if dict(line_candidates) != LINE_CANDIDATES:
        raise QualityGateInputError(
            "policy.typical.line_candidates is fixed to both target baselines"
        )

    # Re-express only the versioned identity fields as v1, then let the frozen
    # v1 parser validate every numeric threshold and range invariant.
    legacy_typical = dict(typical)
    legacy_typical["line_candidates"] = dict(v1.LINE_CANDIDATES)
    legacy = v1.parse_quality_policy(
        {
            "schema_version": v1.POLICY_SCHEMA_VERSION,
            "policy_id": v1.POLICY_ID,
            "subjects": dict(v1.SUBJECTS),
            "validation": dict(
                v1._mapping(root["validation"], label="policy.validation")
            ),
            "typical": legacy_typical,
            "inference": dict(
                v1._mapping(root["inference"], label="policy.inference")
            ),
        }
    )
    return QualityPolicy(
        schema_version=version,
        policy_id=policy_id,
        validation=legacy.validation,
        typical=legacy.typical,
        inference=legacy.inference,
    )


def parse_quality_policy_yaml(text: str) -> QualityPolicy:
    if not isinstance(text, str) or not text.strip():
        raise QualityGateInputError("policy YAML must be a non-empty string")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise QualityGateInputError(f"invalid policy YAML: {error}") from error
    return parse_quality_policy(v1._mapping(payload, label="policy YAML root"))


def _line_report_for_reference(
    line_report: Mapping[str, Any], reference_name: str
) -> dict[str, Any]:
    report = v1._mapping(line_report, label="line_report")
    candidates = v1._mapping(
        report.get("candidates"), label="line_report.candidates"
    )
    if reference_name not in candidates:
        raise QualityGateInputError(
            f"line_report.candidates is missing required {reference_name!r}"
        )
    transformed = dict(report)
    transformed["candidates"] = {
        **dict(candidates),
        "target_first": candidates[reference_name],
    }
    return transformed


def _secondary_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    old_code = str(result["code"])
    if old_code.startswith("typical.per_image."):
        new_code = old_code.replace(
            "typical.per_image.",
            "typical.per_image.target_second.",
            1,
        )
    elif old_code.startswith("typical."):
        new_code = old_code.replace(
            "typical.", "typical.target_second.", 1
        )
    else:  # pragma: no cover - caller filters records before renaming
        raise QualityGateInputError(
            f"cannot qualify non-typical gate code {old_code!r}"
        )
    result["code"] = new_code
    result["message"] = str(result["message"]).replace(
        old_code, new_code, 1
    )
    return result


def _reference_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report["summary"]["typical"]
    return {
        "reference_mean_lsd_error_deg": summary[
            "reference_mean_lsd_error_deg"
        ],
        "mean_delta_reference_deg": summary["mean_delta_reference_deg"],
        "reference_win_rate": summary["reference_win_rate"],
        "reference_delta_p95_deg": summary["reference_delta_p95_deg"],
        "reference_worst_delta_deg": summary[
            "reference_worst_delta_deg"
        ],
        "reference_total_line_length_ratio": summary[
            "reference_total_line_length_ratio"
        ],
        "reference_line_length_ratio_p05": summary[
            "reference_line_length_ratio_p05"
        ],
    }


def evaluate_typical_quality(
    policy: QualityPolicy | Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
    line_report: Mapping[str, Any],
    inference_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the full frozen line policy independently to both references."""

    policy = parse_quality_policy(
        policy.as_dict()
        if isinstance(policy, QualityPolicy)
        else v1._mapping(policy, label="policy")
    )
    legacy_policy = _legacy_policy(policy)
    first = v1.evaluate_typical_quality(
        legacy_policy,
        validation=validation,
        line_report=_line_report_for_reference(line_report, "target_first"),
        inference_metrics=inference_metrics,
    )
    second = v1.evaluate_typical_quality(
        legacy_policy,
        validation=validation,
        line_report=_line_report_for_reference(line_report, "target_second"),
        inference_metrics=inference_metrics,
    )

    result = copy.deepcopy(first)
    secondary_failures = [
        _secondary_record(item)
        for item in second["failures"]
        if str(item.get("code", "")).startswith("typical.")
    ]
    result["failures"].extend(secondary_failures)
    secondary_by_basename: dict[str, list[dict[str, Any]]] = {}
    for failure in secondary_failures:
        basename = failure.get("basename")
        if isinstance(basename, str):
            secondary_by_basename.setdefault(basename, []).append(failure)

    second_rows = {row["basename"]: row for row in second["per_image"]}
    for row in result["per_image"]:
        basename = row["basename"]
        second_row = second_rows[basename]
        row["lsd"]["references"] = {
            "target_first": {
                "error_deg": row["lsd"]["reference_error_deg"],
                "delta_deg": row["lsd"]["delta_reference_deg"],
                "win": row["lsd"]["reference_win"],
            },
            "target_second": {
                "error_deg": second_row["lsd"]["reference_error_deg"],
                "delta_deg": second_row["lsd"]["delta_reference_deg"],
                "win": second_row["lsd"]["reference_win"],
            },
        }
        row["line_length"]["references"] = {
            "target_first": {
                "reference_px": row["line_length"]["reference_px"],
                "ratio": row["line_length"]["ratio_to_reference"],
            },
            "target_second": {
                "reference_px": second_row["line_length"]["reference_px"],
                "ratio": second_row["line_length"]["ratio_to_reference"],
            },
        }
        row["failures"].extend(secondary_by_basename.get(basename, []))
        row["passed"] = not row["failures"]

    result["summary"]["typical"]["references"] = {
        "target_first": _reference_summary(first),
        "target_second": _reference_summary(second),
    }
    secondary_gates = [
        _secondary_record(item)
        for item in second["summary"]["gates"]
        if item.get("scope") == "typical_summary"
        and str(item.get("code", "")).startswith("typical.")
    ]
    result["summary"]["gates"].extend(secondary_gates)
    result["schema_version"] = POLICY_SCHEMA_VERSION
    result["policy_id"] = policy.policy_id
    result["passed"] = not result["failures"]
    return result


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
