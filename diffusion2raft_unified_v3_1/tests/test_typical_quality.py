from __future__ import annotations

import copy
from dataclasses import replace
import math
from pathlib import Path
import unittest

import yaml

from diffusion2raft.typical_quality import (
    QualityGateInputError,
    evaluate_typical_quality,
    parse_quality_policy,
    parse_quality_policy_yaml,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "typical_v33_quality_v1.yaml"
COUNT = 40


def _policy():
    return parse_quality_policy_yaml(POLICY_PATH.read_text(encoding="utf-8"))


def _validation() -> dict:
    return {
        "v33_anchor": {
            "stage": "unified",
            "feature_backend": "qwen",
            "epoch_index": 20,
            "residual_scale": 0.0,
            "metrics": {
                "line_epe": 5.0,
                "line_straightness_error": 0.10,
            },
        },
        "v33_best": {
            "stage": "unified",
            "feature_backend": "qwen",
            "epoch_index": 24,
            "residual_scale": 0.5,
            "metrics": {
                "epe": 5.0,
                "prior_epe": 5.25,
                "epe_gain": 0.25,
                "final_win_rate": 0.60,
                "fold_rate": 0.0001,
                "jacobian_p01": 0.02,
                "line_epe": 4.5,
                "line_straightness_error": 0.08,
            },
        },
    }


def _line_candidate(errors: list[float], lengths: list[float]) -> dict:
    rows = []
    for index, (error, length) in enumerate(zip(errors, lengths)):
        rows.append(
            {
                "basename": f"page_{index:02d}",
                "status": "ok",
                "metrics": {
                    "line_count": 10,
                    "total_line_length_px": length,
                    "orientation_error_deg_length_weighted": error,
                },
            }
        )
    return {
        "valid_mask_directory": None,
        "summary": {
            "indexed_images": COUNT,
            "paired_images": COUNT,
            "evaluated_images": COUNT,
            "missing_images": 0,
            "missing_masks": 0,
            "no_line_images": 0,
            "total_line_length_px": math.fsum(lengths),
            "image_mean_orientation_error_deg_length_weighted": (
                math.fsum(errors) / COUNT
            ),
        },
        "per_image": rows,
    }


def _line_report(
    *,
    candidate_errors: list[float] | None = None,
    reference_errors: list[float] | None = None,
    anchor_errors: list[float] | None = None,
    candidate_lengths: list[float] | None = None,
    reference_lengths: list[float] | None = None,
) -> dict:
    candidate_errors = candidate_errors or [1.0] * COUNT
    reference_errors = reference_errors or [2.0] * COUNT
    anchor_errors = anchor_errors or [2.5] * COUNT
    candidate_lengths = candidate_lengths or [80.0] * COUNT
    reference_lengths = reference_lengths or [100.0] * COUNT
    report = {
        "schema_version": 1,
        "metric_family": "opencv_lsd_axis_alignment",
        "config": {
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
        },
        "candidates": {
            "target_first": _line_candidate(reference_errors, reference_lengths),
            "v33_anchor_full": _line_candidate(anchor_errors, [90.0] * COUNT),
            "v33_best_full": _line_candidate(candidate_errors, candidate_lengths),
            # The production report has other candidates.  They must not alter
            # this fixed target-first/anchor/best decision.
            "target_second": {"ignored": True},
        },
    }
    return report


def _inference() -> dict:
    return {
        "v33_anchor": {
            f"page_{index:02d}": {"evaluation_valid_fraction": 0.90}
            for index in range(COUNT)
        },
        "v33_best": {
            f"page_{index:02d}": {
                "fold_rate": 0.0001,
                "jacobian_p01": 0.02,
                "evaluation_valid_fraction": 0.90,
            }
            for index in range(COUNT)
        },
    }


def _evaluate(
    *,
    validation: dict | None = None,
    line_report: dict | None = None,
    inference: dict | None = None,
) -> dict:
    return evaluate_typical_quality(
        _policy(),
        validation=validation or _validation(),
        line_report=line_report or _line_report(),
        inference_metrics=inference or _inference(),
    )


def _failure_codes(result: dict) -> set[str]:
    return {failure["code"] for failure in result["failures"]}


class TypicalQualityPolicyTest(unittest.TestCase):
    def test_checked_in_policy_is_strict_versioned_and_round_trips(self) -> None:
        text = POLICY_PATH.read_text(encoding="utf-8")
        policy = parse_quality_policy_yaml(text)
        self.assertEqual(policy.schema_version, 1)
        self.assertEqual(policy.policy_id, "typical_v33_quality_v1")
        self.assertEqual(policy.typical.expected_images, 40)
        self.assertEqual(parse_quality_policy(policy.as_dict()), policy)

        payload = yaml.safe_load(text)
        for mutation, message in (
            (lambda item: item.update(extra=True), "keys must be exactly"),
            (lambda item: item.update(schema_version=2), "unsupported"),
            (
                lambda item: item["subjects"].update(reference="target_second"),
                "fixed",
            ),
            (
                lambda item: item["validation"].update(
                    max_epe_exclusive=float("nan")
                ),
                "finite",
            ),
        ):
            with self.subTest(message=message):
                broken = copy.deepcopy(payload)
                mutation(broken)
                with self.assertRaisesRegex(QualityGateInputError, message):
                    parse_quality_policy(broken)

        relaxed = yaml.safe_load(text)
        relaxed["validation"]["min_final_win_rate_exclusive"] = 0.0
        with self.assertRaisesRegex(QualityGateInputError, "thresholds are frozen"):
            parse_quality_policy(relaxed)

        policy = _policy()
        bypass = replace(
            policy,
            validation=replace(
                policy.validation, min_final_win_rate_exclusive=0.0
            ),
        )
        with self.assertRaisesRegex(QualityGateInputError, "thresholds are frozen"):
            evaluate_typical_quality(
                bypass,
                validation=_validation(),
                line_report=_line_report(),
                inference_metrics=_inference(),
            )

    def test_passing_report_is_complete_stable_and_does_not_mutate_inputs(self) -> None:
        validation = _validation()
        lines = _line_report()
        inference = _inference()
        before = copy.deepcopy((validation, lines, inference))
        result = _evaluate(
            validation=validation, line_report=lines, inference=inference
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(len(result["per_image"]), COUNT)
        self.assertTrue(all(row["passed"] for row in result["per_image"]))
        self.assertEqual(
            set(result),
            {"schema_version", "policy_id", "passed", "failures", "per_image", "summary"},
        )
        self.assertAlmostEqual(
            result["summary"]["typical"]["reference_win_rate"], 1.0
        )
        self.assertAlmostEqual(
            result["summary"]["typical"][
                "reference_total_line_length_ratio"
            ],
            0.8,
        )
        self.assertEqual((validation, lines, inference), before)

    def test_every_validation_gate_is_enforced_with_strict_boundaries(self) -> None:
        cases = (
            (
                "validation.v33_best.stage",
                lambda value: value["v33_best"].update(stage="joint"),
            ),
            (
                "validation.v33_anchor.feature_backend",
                lambda value: value["v33_anchor"].update(feature_backend="lite"),
            ),
            (
                "validation.best_epoch_after_anchor",
                lambda value: value["v33_best"].update(epoch_index=20),
            ),
            (
                "validation.best_residual_scale_positive",
                lambda value: value["v33_best"].update(residual_scale=0.0),
            ),
            (
                "validation.best_epe",
                lambda value: value["v33_best"]["metrics"].update(epe=5.7501),
            ),
            (
                "validation.best_epe_gain",
                lambda value: value["v33_best"]["metrics"].update(epe_gain=0.0),
            ),
            (
                "validation.best_final_win_rate",
                lambda value: value["v33_best"]["metrics"].update(
                    final_win_rate=0.5
                ),
            ),
            (
                "validation.best_fold_rate",
                lambda value: value["v33_best"]["metrics"].update(
                    fold_rate=0.0004
                ),
            ),
            (
                "validation.best_jacobian_p01",
                lambda value: value["v33_best"]["metrics"].update(
                    jacobian_p01=0.009
                ),
            ),
            (
                "validation.best_line_epe_below_anchor",
                lambda value: value["v33_best"]["metrics"].update(line_epe=5.0),
            ),
            (
                "validation.best_line_straightness_below_anchor",
                lambda value: value["v33_best"]["metrics"].update(
                    line_straightness_error=0.10
                ),
            ),
        )
        for code, mutation in cases:
            with self.subTest(code=code):
                validation = _validation()
                mutation(validation)
                result = _evaluate(validation=validation)
                self.assertFalse(result["passed"])
                self.assertIn(code, _failure_codes(result))
                failure = next(
                    item for item in result["failures"] if item["code"] == code
                )
                self.assertIn(code, failure["message"])

        validation = _validation()
        validation["v33_best"]["metrics"]["prior_epe"] = 9.0
        result = _evaluate(validation=validation)
        self.assertIn(
            "validation.best_epe_gain_identity", _failure_codes(result)
        )

        # The identity tolerance may absorb harmless aggregation roundoff, but
        # it must never turn an actually negative prior-final gain into a pass.
        validation = _validation()
        validation["v33_best"]["metrics"].update(
            prior_epe=4.9999995,
            epe=5.0,
            epe_gain=0.0000005,
        )
        result = _evaluate(validation=validation)
        self.assertIn(
            "validation.best_computed_epe_gain", _failure_codes(result)
        )

        # Inclusive jacobian boundary is intentionally different from the
        # strict EPE/gain/win/fold gates above.
        validation = _validation()
        validation["v33_best"]["metrics"]["jacobian_p01"] = 0.01
        self.assertTrue(_evaluate(validation=validation)["passed"])

    def test_typical_inclusive_boundaries_pass_exactly(self) -> None:
        deltas = [-1.0] * 30 + [1.0] * 7 + [2.0] * 2 + [5.0]
        reference = [10.0] * COUNT
        candidate = [base + delta for base, delta in zip(reference, deltas)]
        high_ratio = (28.0 - 1.5) / 37.0
        ratios = [0.5] * 3 + [high_ratio] * 37
        lines = _line_report(
            candidate_errors=candidate,
            reference_errors=reference,
            candidate_lengths=[100.0 * ratio for ratio in ratios],
            reference_lengths=[100.0] * COUNT,
        )
        result = _evaluate(line_report=lines)
        self.assertTrue(result["passed"], result["failures"])
        summary = result["summary"]["typical"]
        self.assertAlmostEqual(summary["reference_win_rate"], 0.75)
        self.assertAlmostEqual(summary["reference_delta_p95_deg"], 2.0)
        self.assertAlmostEqual(summary["reference_worst_delta_deg"], 5.0)
        self.assertAlmostEqual(summary["reference_total_line_length_ratio"], 0.70)
        self.assertAlmostEqual(summary["reference_line_length_ratio_p05"], 0.50)

    def test_typical_aggregate_failures_have_stable_codes(self) -> None:
        cases = []

        errors = [2.1] * COUNT
        cases.append(
            (
                "typical.full_frame_lsd_mean_at_most_reference",
                _line_report(candidate_errors=errors),
            )
        )
        errors = [1.0] * 29 + [2.1] * 11
        cases.append(("typical.reference_win_rate", _line_report(candidate_errors=errors)))
        errors = [1.0] * 37 + [5.0] * 3
        cases.append(
            ("typical.reference_delta_p95_deg", _line_report(candidate_errors=errors))
        )
        errors = [1.0] * 39 + [8.0]
        cases.append(
            ("typical.reference_worst_delta_deg", _line_report(candidate_errors=errors))
        )
        cases.append(
            (
                "typical.reference_total_line_length_ratio",
                _line_report(candidate_lengths=[69.0] * COUNT),
            )
        )
        lengths = [40.0] * 3 + [80.0] * 37
        cases.append(
            (
                "typical.reference_line_length_ratio_p05",
                _line_report(candidate_lengths=lengths),
            )
        )
        for code, lines in cases:
            with self.subTest(code=code):
                result = _evaluate(line_report=lines)
                self.assertFalse(result["passed"])
                self.assertIn(code, _failure_codes(result))

    def test_per_image_worst_delta_is_attributed_to_the_basename(self) -> None:
        errors = [1.0] * COUNT
        errors[7] = 8.0
        result = _evaluate(line_report=_line_report(candidate_errors=errors))
        failure = next(
            item
            for item in result["failures"]
            if item["code"] == "typical.per_image.reference_worst_delta_deg"
        )
        self.assertEqual(failure["basename"], "page_07")
        row = next(item for item in result["per_image"] if item["basename"] == "page_07")
        self.assertFalse(row["passed"])

    def test_inference_fold_and_jacobian_boundaries_and_violations(self) -> None:
        inference = _inference()
        for metadata in inference["v33_best"].values():
            metadata["fold_rate"] = 0.0004
            metadata["jacobian_p01"] = 0.01
        self.assertTrue(_evaluate(inference=inference)["passed"])

        inference = _inference()
        inference["v33_best"]["page_03"]["fold_rate"] = 0.0011
        inference["v33_best"]["page_09"]["jacobian_p01"] = 0.009
        result = _evaluate(inference=inference)
        codes = _failure_codes(result)
        self.assertIn("inference.per_image.fold_rate", codes)
        self.assertIn("inference.best_max_fold_rate", codes)
        self.assertIn("inference.per_image.jacobian_p01", codes)
        self.assertIn("inference.best_min_jacobian_p01", codes)

    def test_mean_fold_gate_is_independent_of_per_image_limit(self) -> None:
        inference = _inference()
        for metadata in inference["v33_best"].values():
            metadata["fold_rate"] = 0.0005
        result = _evaluate(inference=inference)
        self.assertIn("inference.best_mean_fold_rate", _failure_codes(result))
        self.assertNotIn("inference.per_image.fold_rate", _failure_codes(result))

    def test_evaluation_valid_support_collapse_is_caught_at_both_scales(self) -> None:
        inference = _inference()
        for metadata in inference["v33_best"].values():
            metadata["evaluation_valid_fraction"] = 0.88
        result = _evaluate(inference=inference)
        self.assertIn("inference.mean_evaluation_valid_drop", _failure_codes(result))
        self.assertNotIn(
            "inference.per_image.evaluation_valid_drop", _failure_codes(result)
        )

        inference = _inference()
        inference["v33_best"]["page_17"]["evaluation_valid_fraction"] = 0.80
        result = _evaluate(inference=inference)
        codes = _failure_codes(result)
        self.assertIn("inference.per_image.evaluation_valid_drop", codes)
        self.assertIn("inference.worst_evaluation_valid_drop", codes)
        row = next(item for item in result["per_image"] if item["basename"] == "page_17")
        self.assertAlmostEqual(row["inference"]["evaluation_valid_drop"], 0.10)

    def test_none_nan_and_infinity_in_required_metrics_fail_closed(self) -> None:
        cases = []
        inference = _inference()
        inference["v33_best"]["page_00"]["fold_rate"] = None
        cases.append((None, None, inference, "fold_rate"))
        inference = _inference()
        inference["v33_best"]["page_00"]["jacobian_p01"] = float("nan")
        cases.append((None, None, inference, "non-finite"))
        validation = _validation()
        validation["v33_best"]["metrics"]["epe"] = float("inf")
        cases.append((validation, None, None, "non-finite"))
        lines = _line_report()
        lines["candidates"]["v33_best_full"]["per_image"][0]["metrics"][
            "orientation_error_deg_length_weighted"
        ] = float("nan")
        cases.append((None, lines, None, "non-finite"))

        for validation, lines, inference, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(QualityGateInputError, message):
                    _evaluate(
                        validation=validation,
                        line_report=lines,
                        inference=inference,
                    )

    def test_basename_sets_and_line_summaries_must_match_exactly(self) -> None:
        inference = _inference()
        inference["v33_best"]["unexpected"] = inference["v33_best"].pop(
            "page_39"
        )
        with self.assertRaisesRegex(QualityGateInputError, "basename set mismatch"):
            _evaluate(inference=inference)

        lines = _line_report()
        lines["candidates"]["v33_best_full"]["per_image"][0]["basename"] = "other"
        with self.assertRaisesRegex(QualityGateInputError, "basename set mismatch"):
            _evaluate(line_report=lines)

        lines = _line_report()
        lines["candidates"]["target_first"]["summary"][
            "image_mean_orientation_error_deg_length_weighted"
        ] += 0.01
        with self.assertRaisesRegex(QualityGateInputError, "summary LSD mean disagrees"):
            _evaluate(line_report=lines)

        lines = _line_report()
        lines["config"]["max_dimension"] = 800
        with self.assertRaisesRegex(QualityGateInputError, "frozen v1 OpenCV LSD"):
            _evaluate(line_report=lines)

        lines = _line_report()
        lines["candidates"]["v33_best_full"]["valid_mask_directory"] = "/mask"
        with self.assertRaisesRegex(QualityGateInputError, "unmasked full-frame"):
            _evaluate(line_report=lines)


if __name__ == "__main__":
    unittest.main()
