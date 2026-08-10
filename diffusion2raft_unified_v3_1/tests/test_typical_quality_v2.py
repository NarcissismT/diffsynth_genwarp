from __future__ import annotations

import copy
import math
from pathlib import Path
import unittest

import yaml

from diffusion2raft.typical_quality_v2 import (
    QualityGateInputError,
    evaluate_typical_quality,
    parse_quality_policy,
    parse_quality_policy_yaml,
)
from test_typical_quality import (
    COUNT,
    _inference,
    _line_candidate,
    _line_report as _v1_line_report,
    _validation,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "typical_v33_quality_v2.yaml"


def _policy():
    return parse_quality_policy_yaml(POLICY_PATH.read_text(encoding="utf-8"))


def _line_report(
    *,
    candidate_errors: list[float] | None = None,
    first_errors: list[float] | None = None,
    second_errors: list[float] | None = None,
    candidate_lengths: list[float] | None = None,
    first_lengths: list[float] | None = None,
    second_lengths: list[float] | None = None,
) -> dict:
    candidate_errors = candidate_errors or [1.0] * COUNT
    first_errors = first_errors or [2.0] * COUNT
    second_errors = second_errors or [1.5] * COUNT
    candidate_lengths = candidate_lengths or [80.0] * COUNT
    first_lengths = first_lengths or [100.0] * COUNT
    second_lengths = second_lengths or [90.0] * COUNT
    report = _v1_line_report(
        candidate_errors=candidate_errors,
        reference_errors=first_errors,
        candidate_lengths=candidate_lengths,
        reference_lengths=first_lengths,
    )
    report["candidates"]["target_second"] = _line_candidate(
        second_errors, second_lengths
    )
    return report


def _evaluate(*, line_report: dict | None = None) -> dict:
    return evaluate_typical_quality(
        _policy(),
        validation=_validation(),
        line_report=line_report or _line_report(),
        inference_metrics=_inference(),
    )


def _failure_codes(result: dict) -> set[str]:
    return {failure["code"] for failure in result["failures"]}


class TypicalQualityV2Test(unittest.TestCase):
    def test_v2_is_strict_versioned_round_trips_and_v1_remains_separate(self) -> None:
        text = POLICY_PATH.read_text(encoding="utf-8")
        policy = parse_quality_policy_yaml(text)
        self.assertEqual(policy.schema_version, 2)
        self.assertEqual(policy.policy_id, "typical_v33_quality_v2")
        self.assertEqual(parse_quality_policy(policy.as_dict()), policy)

        payload = yaml.safe_load(text)
        payload["subjects"]["references"] = ["target_first"]
        with self.assertRaisesRegex(QualityGateInputError, "fixed"):
            parse_quality_policy(payload)

        v1_path = ROOT / "configs" / "typical_v33_quality_v1.yaml"
        with self.assertRaisesRegex(QualityGateInputError, "schema_version"):
            parse_quality_policy_yaml(v1_path.read_text(encoding="utf-8"))

    def test_passing_report_has_both_references_and_target_first_aliases(self) -> None:
        result = _evaluate()
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["schema_version"], 2)
        references = result["summary"]["typical"]["references"]
        self.assertEqual(set(references), {"target_first", "target_second"})
        self.assertEqual(
            result["summary"]["typical"]["reference_mean_lsd_error_deg"],
            references["target_first"]["reference_mean_lsd_error_deg"],
        )
        row = result["per_image"][0]
        self.assertEqual(
            set(row["lsd"]["references"]),
            {"target_first", "target_second"},
        )

    def test_candidate_beating_first_but_losing_second_is_rejected(self) -> None:
        result = _evaluate(
            line_report=_line_report(
                candidate_errors=[1.0] * COUNT,
                first_errors=[2.0] * COUNT,
                second_errors=[0.5] * COUNT,
            )
        )
        self.assertFalse(result["passed"])
        codes = _failure_codes(result)
        self.assertIn(
            "typical.target_second.full_frame_lsd_mean_at_most_reference",
            codes,
        )
        self.assertNotIn(
            "typical.full_frame_lsd_mean_at_most_reference", codes
        )

    def test_candidate_beating_second_but_losing_first_is_rejected(self) -> None:
        result = _evaluate(
            line_report=_line_report(
                candidate_errors=[2.5] * COUNT,
                first_errors=[2.0] * COUNT,
                second_errors=[3.0] * COUNT,
            )
        )
        self.assertFalse(result["passed"])
        codes = _failure_codes(result)
        self.assertIn(
            "typical.full_frame_lsd_mean_at_most_reference", codes
        )
        self.assertNotIn(
            "typical.target_second.full_frame_lsd_mean_at_most_reference",
            codes,
        )

    def test_exact_equality_to_both_references_passes(self) -> None:
        errors = [2.0] * COUNT
        lengths = [100.0] * COUNT
        result = _evaluate(
            line_report=_line_report(
                candidate_errors=errors,
                first_errors=errors,
                second_errors=errors,
                candidate_lengths=lengths,
                first_lengths=lengths,
                second_lengths=lengths,
            )
        )
        self.assertTrue(result["passed"], result["failures"])
        for reference in result["summary"]["typical"]["references"].values():
            self.assertEqual(reference["mean_delta_reference_deg"], 0.0)
            self.assertEqual(reference["reference_win_rate"], 1.0)

    def test_secondary_reference_is_fully_validated(self) -> None:
        report = _line_report()
        report["candidates"]["target_second"]["per_image"][0]["metrics"][
            "orientation_error_deg_length_weighted"
        ] = float("nan")
        with self.assertRaisesRegex(QualityGateInputError, "non-finite"):
            _evaluate(line_report=report)

        report = _line_report()
        report["candidates"]["target_second"]["per_image"][0][
            "basename"
        ] = "other"
        with self.assertRaisesRegex(QualityGateInputError, "basename set mismatch"):
            _evaluate(line_report=report)

    def test_crossed_per_image_strength_cannot_be_hidden_by_aggregate_mean(self) -> None:
        candidate = [2.0] * COUNT
        first = [1.0] * 20 + [3.0] * 20
        second = [3.0] * 20 + [1.0] * 20
        result = _evaluate(
            line_report=_line_report(
                candidate_errors=candidate,
                first_errors=first,
                second_errors=second,
            )
        )
        codes = _failure_codes(result)
        self.assertIn("typical.reference_win_rate", codes)
        self.assertIn("typical.target_second.reference_win_rate", codes)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
