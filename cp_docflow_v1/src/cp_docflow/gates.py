"""Evaluate plan thresholds and write an immutable DocGrid-Flow gate receipt."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoint import file_sha256

_GATE_STAGE = {
    "gate1": "coarse",
    "gate2": "warr",
    "gate3": "coord_fm",
    "gate4": "qwen",
    "gate5": "full_page",
}
_VERIFIED_PROVENANCE = {"analytic_gt", "renderer_gt"}


def _load_json(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return resolved, value


def _number(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _check(
    checks: list[dict[str, Any]],
    name: str,
    actual: Any,
    passed: bool,
    requirement: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "actual": actual,
            "requirement": requirement,
        }
    )


def _matching_hard_subset(
    current: dict[str, Any], baseline: dict[str, Any]
) -> tuple[str | None, float | None, float | None]:
    current_groups = current.get("groups", {})
    baseline_groups = baseline.get("groups", {})
    if not isinstance(current_groups, dict) or not isinstance(baseline_groups, dict):
        return None, None, None
    candidates = (
        "severity:hard",
        "severity:heavy",
        "severity:severe",
        "severity:extreme",
        "subset:difficulty:hard",
        "subset:warp_difficulty:hard",
        "subset:is_hard:true",
    )
    for name in candidates:
        if name in current_groups and name in baseline_groups:
            return (
                name,
                _number(current_groups[name], "epe"),
                _number(baseline_groups[name], "epe"),
            )
    return None, None, None


def _validate_gate5_ocr_evidence(
    evaluation: dict[str, Any],
    evidence: dict[str, Any] | None,
    *,
    evidence_file: Path | None,
) -> dict[str, Any]:
    """Require scored OCR evidence from the exact geometry evaluation run."""

    if not isinstance(evidence, dict):
        raise ValueError("Gate-5 requires scored OCR evidence")
    report_value = evidence.get("ocr_evaluation")
    declared_sha = evidence.get("ocr_evaluation_sha256")
    if not isinstance(report_value, str) or not report_value.strip():
        raise ValueError("Gate-5 evidence lacks ocr_evaluation")
    report_path = Path(report_value)
    if not report_path.is_absolute():
        if evidence_file is None:
            raise ValueError("relative ocr_evaluation requires an evidence file")
        report_path = evidence_file.parent / report_path
    report_path = report_path.resolve()
    report_path, report = _load_json(report_path, "OCR evaluation")
    if report.get("schema") != "docgrid_flow.ocr_evaluation.v2":
        raise ValueError("Gate-5 OCR evidence has an unsupported schema")
    actual_sha = file_sha256(report_path)
    if declared_sha != actual_sha:
        raise ValueError("Gate-5 OCR report SHA does not match the evidence")
    for key in ("ocr_cer", "oracle_ocr_cer", "ocr_wer", "oracle_ocr_wer"):
        evidence_value = _number(evidence, key)
        report_value_number = _number(report, key)
        if (
            evidence_value is None
            or report_value_number is None
            or abs(evidence_value - report_value_number) > 1.0e-12
        ):
            raise ValueError(f"Gate-5 OCR value {key!r} differs from its report")
    geometry = report.get("geometry_identity")
    if not isinstance(geometry, dict):
        raise ValueError("Gate-5 OCR report is not bound to geometry sample IDs")
    for key in ("checkpoint_sha256", "manifest_sha256"):
        if geometry.get(key) != evaluation.get(key):
            raise ValueError(f"Gate-5 OCR {key} differs from geometry evaluation")
    if geometry.get("dataset_payload_sha256") != evaluation.get(
        "evaluation_dataset_payload_sha256"
    ):
        raise ValueError("Gate-5 OCR validation payload differs from geometry evaluation")
    ocr_export = evaluation.get("ocr_image_export")
    if not isinstance(ocr_export, dict) or geometry.get(
        "ocr_image_manifest_sha256"
    ) != ocr_export.get("manifest_sha256"):
        raise ValueError("Gate-5 OCR is not bound to exported geometry images")
    evaluation_samples = _number(evaluation.get("aggregate", {}), "samples")
    report_samples = _number(report, "samples")
    if (
        evaluation_samples is None
        or report_samples is None
        or int(evaluation_samples) != int(report_samples)
    ):
        raise ValueError("Gate-5 OCR and geometry evaluation sample counts differ")
    return {
        "path": str(report_path),
        "sha256": actual_sha,
        "samples": int(report_samples),
        "ocr_engine": report.get("ocr_engine"),
        "ocr_engine_version": report.get("ocr_engine_version"),
    }


def evaluate_gate_criteria(
    gate: str,
    evaluation: dict[str, Any],
    baseline: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Translate the newest plan's Gate 1-5 requirements into checks."""

    current = evaluation.get("aggregate")
    if not isinstance(current, dict):
        raise ValueError("evaluation lacks aggregate metrics")
    base = None if baseline is None else baseline.get("aggregate")
    if baseline is not None and not isinstance(base, dict):
        raise ValueError("baseline evaluation lacks aggregate metrics")
    facts = dict(evidence or {})
    checks: list[dict[str, Any]] = []
    epe = _number(current, "epe")
    p95 = _number(current, "epe_p95")
    fold = _number(current, "fold_rate")
    line = _number(current, "line_epe")
    edge = _number(current, "edge_epe")
    straight = _number(current, "straightness_error")
    win = _number(current, "final_win_rate")
    damage = _number(current, "high_confidence_damage_rate")

    if gate == "gate1":
        base_fold = None if base is None else _number(base, "fold_rate")
        monotonic = _number(current, "confidence_monotonic_rate")
        _check(checks, "epe", epe, epe is not None and epe <= 5.75, "EPE <= 5.75 px")
        _check(
            checks,
            "fold_not_worse_than_prior",
            {"current": fold, "prior": base_fold},
            fold is not None and base_fold is not None and fold <= base_fold + 1.0e-12,
            "fold rate <= frozen prior",
        )
        _check(
            checks,
            "confidence_monotonic",
            monotonic,
            monotonic is not None and monotonic >= 0.75,
            "at least 75% of adjacent confidence bins have non-increasing EPE",
        )
        _check(
            checks,
            "full_page_scale_stable",
            facts.get("full_page_scale_stable"),
            facts.get("full_page_scale_stable") is True,
            "reviewed full-page inference has no systematic scale drift",
        )
    elif gate == "gate2":
        base_epe = None if base is None else _number(base, "epe")
        base_p95 = None if base is None else _number(base, "epe_p95")
        base_line = None if base is None else _number(base, "line_epe")
        base_straight = None if base is None else _number(base, "straightness_error")
        base_fold = None if base is None else _number(base, "fold_rate")
        _check(
            checks, "epe_improvement", None if epe is None or base_epe is None else base_epe - epe,
            epe is not None and base_epe is not None and base_epe - epe >= 0.3,
            "EPE improves by >= 0.3 px over coarse",
        )
        for name, value, reference in (
            ("p95_improves", p95, base_p95),
            ("line_epe_improves", line, base_line),
            ("straightness_improves", straight, base_straight),
        ):
            _check(
                checks, name, {"current": value, "baseline": reference},
                value is not None and reference is not None and value < reference,
                "strictly improves over coarse",
            )
        _check(checks, "final_win_rate", win, win is not None and win >= 0.60, "win rate >= 0.60")
        _check(
            checks, "fold_not_worse", {"current": fold, "baseline": base_fold},
            fold is not None and base_fold is not None and fold <= base_fold + 1.0e-12,
            "fold rate does not increase",
        )
        _check(
            checks, "warr_monotonic", current.get("warr_monotonic"),
            current.get("warr_monotonic") is True,
            "aggregate WARR iteration EPE is monotonic non-increasing",
        )
    elif gate == "gate3":
        base_epe = None if base is None else _number(base, "epe")
        absolute = None if epe is None or base_epe is None else base_epe - epe
        relative = None if absolute is None or not base_epe else absolute / base_epe
        _check(
            checks, "epe_improvement", {"absolute_px": absolute, "relative": relative},
            absolute is not None and (absolute >= 0.2 or (relative is not None and relative >= 0.05)),
            "EPE improves by >= 0.2 px or >= 5% over WARR",
        )
        _check(checks, "final_win_rate", win, win is not None and win >= 0.60, "win rate >= 0.60")
        _check(
            checks, "high_confidence_damage", damage,
            damage is not None and damage < 0.05,
            "high-confidence damage rate < 5%",
        )
        for key, requirement in (
            ("multi_seed_stable", "three-seed result is stable"),
            ("fixed_seed_repeatable", "fixed inference seed is exactly repeatable"),
            ("ode_steps_validated", "4-6 ODE step cost/benefit was reviewed"),
        ):
            _check(checks, key, facts.get(key), facts.get(key) is True, requirement)
    elif gate == "gate4":
        base_epe = None if base is None else _number(base, "epe")
        base_line = None if base is None else _number(base, "line_epe")
        base_edge = None if base is None else _number(base, "edge_epe")
        subset, hard_epe, hard_base = _matching_hard_subset(evaluation, baseline or {})
        hard_gain = None if hard_epe is None or not hard_base else (hard_base - hard_epe) / hard_base
        _check(
            checks, "hard_subset_improvement",
            {"subset": subset, "current": hard_epe, "baseline": hard_base, "relative": hard_gain},
            hard_gain is not None and hard_gain >= 0.05,
            "matching hard/heavy/severe subset EPE improves by >= 5%",
        )
        for name, value, reference, requirement in (
            ("global_epe_not_worse", epe, base_epe, "global EPE does not increase"),
            ("line_epe_not_worse", line, base_line, "Line EPE does not increase"),
            ("edge_epe_not_worse", edge, base_edge, "Edge EPE does not increase"),
        ):
            _check(
                checks, name, {"current": value, "baseline": reference},
                value is not None and reference is not None and value <= reference + 1.0e-12,
                requirement,
            )
        _check(
            checks, "efficiency_acceptable", facts.get("efficiency_acceptable"),
            facts.get("efficiency_acceptable") is True,
            "Qwen latency and memory overhead reviewed as acceptable",
        )
    elif gate == "gate5":
        base_p95 = None if base is None else _number(base, "epe_p95")
        base_line = None if base is None else _number(base, "line_epe")
        base_fold = None if base is None else _number(base, "fold_rate")
        _check(checks, "epe", epe, epe is not None and epe <= 5.18, "EPE <= 5.18 px")
        _check(checks, "final_win_rate", win, win is not None and win >= 0.65, "win rate >= 0.65")
        _check(
            checks, "straightness", straight,
            straight is not None and straight <= 0.10,
            "straightness error <= 0.10",
        )
        _check(
            checks, "p95_vs_deterministic", {"current": p95, "deterministic": base_p95},
            p95 is not None and base_p95 is not None and p95 <= 0.90 * base_p95,
            "P95 improves >= 10% over deterministic model",
        )
        _check(
            checks, "line_vs_deterministic", {"current": line, "deterministic": base_line},
            line is not None and base_line is not None and line <= 0.92 * base_line,
            "Line EPE improves >= 8% over deterministic model",
        )
        _check(
            checks, "edge_ratio", {"edge_epe": edge, "epe": epe},
            edge is not None and epe is not None and edge <= 1.15 * epe,
            "Edge EPE <= 1.15 * global EPE",
        )
        _check(
            checks, "fold", {"current": fold, "deterministic": base_fold},
            fold is not None and base_fold is not None
            and fold <= base_fold + 1.0e-12 and fold <= 0.0045,
            "fold <= deterministic and <= 0.0045",
        )
        _check(
            checks, "high_confidence_damage", damage,
            damage is not None and damage < 0.05,
            "high-confidence damage rate < 5%",
        )
        ocr = _number(facts, "ocr_cer")
        oracle_ocr = _number(facts, "oracle_ocr_cer")
        _check(
            checks, "ocr_cer", {"model": ocr, "oracle": oracle_ocr},
            ocr is not None and oracle_ocr is not None and ocr <= oracle_ocr + 0.01,
            "OCR CER <= oracle CER + 0.01",
        )
        for key, requirement in (
            ("multi_seed_stable", "three-seed final result is stable"),
            ("fixed_seed_repeatable", "complete inference is repeatable"),
            ("visual_no_water_ripple", "full-page review finds no visible water ripple/fold"),
            ("visual_text_table_preserved", "text and table lines are preserved"),
        ):
            _check(checks, key, facts.get(key), facts.get(key) is True, requirement)
    return checks


def write_gate_receipt(
    evaluation_path: str | Path,
    output_path: str | Path,
    *,
    gate: str,
    passed: bool,
    reviewer: str,
    review_note: str,
    baseline_evaluation: str | Path | None = None,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    normalized_gate = str(gate).lower()
    if normalized_gate not in _GATE_STAGE:
        raise ValueError(f"gate must be one of {sorted(_GATE_STAGE)}")
    evaluation_file, evaluation = _load_json(evaluation_path, "evaluation report")
    if evaluation.get("schema") != "docgrid_flow.full_gate_evaluation.v3":
        raise ValueError(
            "receipt requires a v3 gate evaluation with bound work-size identity"
        )
    if evaluation.get("evaluation_role") != "gate" or evaluation.get("gate_eligible") is not True:
        raise ValueError("receipt requires evaluate_full --gate output, not exploratory metrics")
    expected_stage = _GATE_STAGE[normalized_gate]
    actual_stage = {"refiner": "warr", "joint": "full_page"}.get(
        str(evaluation.get("training_stage", "")).lower(),
        str(evaluation.get("training_stage", "")).lower(),
    )
    if actual_stage != expected_stage:
        raise ValueError(
            f"{normalized_gate} requires stage {expected_stage!r}, got {actual_stage!r}"
        )
    contract = evaluation.get("training_data_contract")
    if not isinstance(contract, dict):
        raise ValueError("evaluation lacks the frozen training data contract")
    provenance = {
        str(value).lower() for value in contract.get("allowed_label_provenance", [])
    }
    if not provenance or not provenance <= _VERIFIED_PROVENANCE:
        raise ValueError("gate receipt only accepts verified analytic_gt/renderer_gt provenance")
    baseline_file: Path | None = None
    baseline: dict[str, Any] | None = None
    if baseline_evaluation is not None:
        baseline_file, baseline = _load_json(baseline_evaluation, "baseline evaluation")
        if baseline.get("schema") not in {
            "docgrid_flow.full_gate_evaluation.v3",
            "docgrid_flow.full_exploratory_evaluation.v3",
        }:
            raise ValueError(
                "baseline requires a v3 evaluation with bound work-size identity"
            )
        current_manifest = evaluation.get("manifest_sha256")
        baseline_manifest = baseline.get("manifest_sha256")
        if current_manifest and baseline_manifest and current_manifest != baseline_manifest:
            raise ValueError("current and baseline evaluations use different manifests")
        current_payload = evaluation.get("evaluation_dataset_payload_sha256")
        baseline_payload = baseline.get("evaluation_dataset_payload_sha256")
        if (
            not isinstance(current_payload, str)
            or len(current_payload) != 64
            or not isinstance(baseline_payload, str)
            or len(baseline_payload) != 64
        ):
            raise ValueError("current and baseline evaluations must bind payload SHA-256")
        if current_payload != baseline_payload:
            raise ValueError("current and baseline evaluations use different payload assets")
        for field in (
            "evaluation_input_work_size",
            "evaluation_output_work_size",
        ):
            current_size = evaluation.get(field)
            baseline_size = baseline.get(field)
            valid_current = (
                isinstance(current_size, list)
                and len(current_size) == 2
                and all(isinstance(value, int) and value > 0 for value in current_size)
            )
            valid_baseline = (
                isinstance(baseline_size, list)
                and len(baseline_size) == 2
                and all(isinstance(value, int) and value > 0 for value in baseline_size)
            )
            if not valid_current or not valid_baseline:
                raise ValueError(f"current and baseline evaluations must bind {field}")
            if current_size != baseline_size:
                raise ValueError(
                    f"current and baseline evaluations use different {field}: "
                    f"{current_size} != {baseline_size}"
                )
    evidence_file: Path | None = None
    evidence: dict[str, Any] | None = None
    if evidence_path is not None:
        evidence_file, evidence = _load_json(evidence_path, "external evidence")
    ocr_evidence_identity: dict[str, Any] | None = None
    if normalized_gate == "gate5" and (
        passed
        or (
            isinstance(evidence, dict)
            and any(
                key in evidence
                for key in ("ocr_cer", "oracle_ocr_cer", "ocr_evaluation")
            )
        )
    ):
        ocr_evidence_identity = _validate_gate5_ocr_evidence(
            evaluation, evidence, evidence_file=evidence_file
        )
    checks = evaluate_gate_criteria(normalized_gate, evaluation, baseline, evidence)
    quantitative_passed = bool(checks) and all(item["passed"] for item in checks)
    if passed and not quantitative_passed:
        failed = [item["name"] for item in checks if not item["passed"]]
        raise ValueError(
            "cannot write a passing receipt; failed or missing criteria: "
            + ", ".join(failed)
        )
    reviewer_value = str(reviewer).strip()
    note_value = str(review_note).strip()
    if not reviewer_value or not note_value:
        raise ValueError("reviewer and review_note must be non-empty")
    receipt: dict[str, Any] = {
        "schema": f"docgrid_flow.{normalized_gate}.v2",
        "passed": bool(passed),
        "criteria_passed": quantitative_passed,
        "criteria": checks,
        "verified_gt_only": True,
        "reviewer": reviewer_value,
        "review_note": note_value,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": str(evaluation_file),
        "evaluation_sha256": file_sha256(evaluation_file),
        "baseline_evaluation": None if baseline_file is None else str(baseline_file),
        "baseline_evaluation_sha256": (
            None if baseline_file is None else file_sha256(baseline_file)
        ),
        "external_evidence": None if evidence_file is None else str(evidence_file),
        "external_evidence_sha256": (
            None if evidence_file is None else file_sha256(evidence_file)
        ),
        "ocr_evidence": ocr_evidence_identity,
        "checkpoint": evaluation.get("checkpoint"),
        "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
        "manifest": evaluation.get("manifest"),
        "manifest_sha256": evaluation.get("manifest_sha256"),
        "training_stage": actual_stage,
        "aggregate": evaluation.get("aggregate"),
        "criteria_source": "Diffusion2RAFT_Plan_and_Goals_newest.md",
    }
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing gate receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=sorted(_GATE_STAGE), required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--baseline-evaluation")
    parser.add_argument("--evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-note", required=True)
    decision = parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--passed", action="store_true")
    decision.add_argument("--failed", action="store_true")
    args = parser.parse_args()
    receipt = write_gate_receipt(
        args.evaluation,
        args.output,
        gate=args.gate,
        passed=args.passed and not args.failed,
        reviewer=args.reviewer,
        review_note=args.review_note,
        baseline_evaluation=args.baseline_evaluation,
        evidence_path=args.evidence,
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
