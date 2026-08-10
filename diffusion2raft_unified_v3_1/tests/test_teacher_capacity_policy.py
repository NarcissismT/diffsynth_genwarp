from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion2raft.teacher_capacity_policy import (  # noqa: E402
    CANONICAL_POLICY_SHA256,
    DECISION_KIND,
    POLICY_ID,
    evaluate_teacher_capacity_policy,
    production_policy_v1,
)


BIN_EDGES = [0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
BIN_COUNTS = [40, 40, 40, 40, 40, 50, 50]


def _metrics(
    sample_count: int,
    *,
    solver: float,
    overflow: float,
    trainable: float,
    stride_trainable: float,
) -> dict:
    return {
        "sample_count": sample_count,
        "eval_sample_count": sample_count,
        "eval_pixels": sample_count * 100,
        "teacher_epe_px": 5.0,
        "oracle_solver_coverage": solver,
        "oracle_solver_any_sample_rate": 1.0,
        "oracle_solver_full_sample_rate": 0.99,
        "oracle_residual_overflow_given_solvable_x_pixel_rate": overflow / 2.0,
        "oracle_residual_overflow_given_solvable_y_pixel_rate": overflow / 2.0,
        "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": overflow,
        # Deliberately much larger than the pixel gate.  Production policy v1
        # retains these sample-level rates as finite diagnostics only.
        "oracle_residual_overflow_given_solvable_x_sample_rate": 0.90,
        "oracle_residual_overflow_given_solvable_y_sample_rate": 0.95,
        "oracle_residual_overflow_given_solvable_any_axis_sample_rate": 1.0,
        "trainable_coverage": trainable,
        "residual_target_valid_rate": trainable,
        "oracle_residual_axis_absmax_px": {"x": 30.0, "y": 31.0},
        "stride_oracle_residual_reconstruction_epe_px": 0.8,
        "stride_trainable_oracle_reconstruction_epe_px": stride_trainable,
    }


def _aggregate_metrics() -> dict:
    return _metrics(
        300,
        solver=0.995,
        overflow=0.004,
        trainable=0.99,
        stride_trainable=0.9,
    )


def _bin_metrics(sample_count: int) -> dict:
    return _metrics(
        sample_count,
        solver=0.985,
        overflow=0.009,
        trainable=0.975,
        stride_trainable=1.4,
    )


def _valid_report() -> dict:
    bins = []
    for index, count in enumerate(BIN_COUNTS):
        bins.append(
            {
                "index": index,
                "absolute_rotation_deg": {
                    "lower_inclusive": BIN_EDGES[index],
                    "upper": BIN_EDGES[index + 1],
                    "upper_inclusive": index == len(BIN_COUNTS) - 1,
                },
                "metrics": _bin_metrics(count),
            }
        )
    return {
        "report_version": 1,
        "kind": "frozen_teacher_residual_capacity_preflight",
        "identities": {
            "manifest": {
                "split": "val",
                "record_count": 300,
            }
        },
        "protocol": {
            "work_size": [512, 512],
            "selected_sample_count": 300,
            "selected_indices": list(range(300)),
            "source_rotation": {"bin_edges_deg": list(BIN_EDGES)},
            "feature_stride": 8,
            "max_residual_px": 24.0,
            "max_residual_target": 24.0,
        },
        "results": {
            "original": _aggregate_metrics(),
            "rotation_augmented": _aggregate_metrics(),
            "full_geometry_augmented": _aggregate_metrics(),
            "rotation_bins": bins,
            "samples": [],
        },
    }


def _failure_codes(decision: dict) -> set[str]:
    return {item["code"] for item in decision["failures"]}


class TeacherCapacityPolicyTest(unittest.TestCase):
    def test_frozen_policy_sha_passing_decision_and_input_immutability(self) -> None:
        self.assertEqual(
            CANONICAL_POLICY_SHA256,
            "f762d9e96a53b3404815c0437c5a5535810323c7d50dd4baa774393c27c688fa",
        )
        report = _valid_report()
        before = copy.deepcopy(report)
        decision = evaluate_teacher_capacity_policy(report)
        self.assertTrue(decision["passed"], decision["failures"])
        self.assertEqual(decision["failures"], [])
        self.assertEqual(decision["kind"], DECISION_KIND)
        self.assertEqual(decision["policy_id"], POLICY_ID)
        self.assertEqual(decision["policy_sha256"], CANONICAL_POLICY_SHA256)
        self.assertEqual(report, before)
        self.assertEqual(
            decision["summary"]["check_count"], len(decision["checks"])
        )
        self.assertEqual(
            len({item["code"] for item in decision["checks"]}),
            len(decision["checks"]),
        )
        json.dumps(decision, allow_nan=False)

        caller_copy = production_policy_v1()
        caller_copy["protocol"]["feature_stride"] = 4
        self.assertEqual(production_policy_v1()["protocol"]["feature_stride"], 8)

    def test_all_three_aggregate_thresholds_are_inclusive_and_enforced(self) -> None:
        boundary = _valid_report()
        for group in ("original", "rotation_augmented", "full_geometry_augmented"):
            metrics = boundary["results"][group]
            metrics["oracle_solver_coverage"] = 0.99
            metrics[
                "oracle_residual_overflow_given_solvable_any_axis_pixel_rate"
            ] = 0.005
            metrics["trainable_coverage"] = 0.985
            metrics["residual_target_valid_rate"] = 0.985
            metrics["stride_trainable_oracle_reconstruction_epe_px"] = 1.0
        self.assertTrue(evaluate_teacher_capacity_policy(boundary)["passed"])

        cases = (
            ("oracle_solver_coverage", 0.989999),
            (
                "oracle_residual_overflow_given_solvable_any_axis_pixel_rate",
                0.005001,
            ),
            ("trainable_coverage", 0.984999),
            ("stride_trainable_oracle_reconstruction_epe_px", 1.000001),
        )
        for group in ("original", "rotation_augmented", "full_geometry_augmented"):
            for metric, value in cases:
                with self.subTest(group=group, metric=metric):
                    report = _valid_report()
                    report["results"][group][metric] = value
                    if metric == "trainable_coverage":
                        report["results"][group]["residual_target_valid_rate"] = value
                    decision = evaluate_teacher_capacity_policy(report)
                    self.assertFalse(decision["passed"])
                    self.assertIn(
                        f"aggregate.{group}.{metric}", _failure_codes(decision)
                    )

    def test_rotation_bins_require_twenty_samples_and_the_looser_profile(self) -> None:
        boundary = _valid_report()
        boundary_counts = [20, 40, 40, 40, 40, 60, 60]
        for item, count in zip(
            boundary["results"]["rotation_bins"], boundary_counts, strict=True
        ):
            metrics = item["metrics"]
            metrics["sample_count"] = count
            metrics["eval_sample_count"] = count
            metrics["eval_pixels"] = count * 100
            metrics["oracle_solver_coverage"] = 0.98
            metrics[
                "oracle_residual_overflow_given_solvable_any_axis_pixel_rate"
            ] = 0.01
            metrics["trainable_coverage"] = 0.97
            metrics["residual_target_valid_rate"] = 0.97
            metrics["stride_trainable_oracle_reconstruction_epe_px"] = 1.5
        self.assertTrue(evaluate_teacher_capacity_policy(boundary)["passed"])

        too_small = copy.deepcopy(boundary)
        too_small["results"]["rotation_bins"][0]["metrics"].update(
            sample_count=19,
            eval_sample_count=19,
            eval_pixels=1900,
        )
        too_small["results"]["rotation_bins"][-1]["metrics"].update(
            sample_count=61,
            eval_sample_count=61,
            eval_pixels=6100,
        )
        decision = evaluate_teacher_capacity_policy(too_small)
        self.assertIn("rotation_bin.0.sample_count", _failure_codes(decision))
        self.assertNotIn(
            "rotation_bins.sample_count_sum", _failure_codes(decision)
        )

        cases = (
            ("oracle_solver_coverage", 0.979999),
            (
                "oracle_residual_overflow_given_solvable_any_axis_pixel_rate",
                0.010001,
            ),
            ("trainable_coverage", 0.969999),
            ("stride_trainable_oracle_reconstruction_epe_px", 1.500001),
        )
        for metric, value in cases:
            with self.subTest(metric=metric):
                report = _valid_report()
                report["results"]["rotation_bins"][3]["metrics"][metric] = value
                if metric == "trainable_coverage":
                    report["results"]["rotation_bins"][3]["metrics"][
                        "residual_target_valid_rate"
                    ] = value
                decision = evaluate_teacher_capacity_policy(report)
                self.assertIn(
                    f"rotation_bin.3.{metric}", _failure_codes(decision)
                )

        # Sample-level overflow remains diagnostic even at its semantic maximum.
        diagnostic = _valid_report()
        self.assertTrue(evaluate_teacher_capacity_policy(diagnostic)["passed"])

    def test_protocol_contract_is_itemized_and_fails_closed(self) -> None:
        cases = (
            (
                "report version",
                lambda item: item.update(report_version=2),
                "report.report_version",
            ),
            (
                "kind",
                lambda item: item.update(kind="other"),
                "report.kind",
            ),
            (
                "manifest count",
                lambda item: item["identities"]["manifest"].update(record_count=299),
                "protocol.manifest.record_count",
            ),
            (
                "manifest split",
                lambda item: item["identities"]["manifest"].update(split="train"),
                "protocol.manifest.split",
            ),
            (
                "selected count",
                lambda item: item["protocol"].update(selected_sample_count=299),
                "protocol.selected_sample_count",
            ),
            (
                "selected indices",
                lambda item: item["protocol"].update(
                    selected_indices=list(reversed(range(300)))
                ),
                "protocol.selected_indices",
            ),
            (
                "work size",
                lambda item: item["protocol"].update(work_size=[512, 511]),
                "protocol.work_size",
            ),
            (
                "stride",
                lambda item: item["protocol"].update(feature_stride=4),
                "protocol.feature_stride",
            ),
            (
                "residual cap",
                lambda item: item["protocol"].update(max_residual_px=25.0),
                "protocol.max_residual_px",
            ),
            (
                "residual target",
                lambda item: item["protocol"].update(max_residual_target=23.0),
                "protocol.max_residual_target",
            ),
            (
                "bin edges",
                lambda item: item["protocol"]["source_rotation"].update(
                    bin_edges_deg=[0.0, 180.0]
                ),
                "protocol.rotation_bin_edges_deg",
            ),
            (
                "original sample count",
                lambda item: item["results"]["original"].update(sample_count=299),
                "aggregate.original.sample_count",
            ),
            (
                "rotation sample count",
                lambda item: item["results"]["rotation_augmented"].update(
                    sample_count=299
                ),
                "aggregate.rotation_augmented.sample_count",
            ),
            (
                "full geometry missing",
                lambda item: item["results"].pop("full_geometry_augmented"),
                "aggregate.full_geometry_augmented.present",
            ),
        )
        for label, mutation, expected_code in cases:
            with self.subTest(label=label):
                report = _valid_report()
                mutation(report)
                decision = evaluate_teacher_capacity_policy(report)
                self.assertFalse(decision["passed"])
                self.assertIn(expected_code, _failure_codes(decision))

    def test_every_metric_must_be_present_numeric_finite_and_nonempty(self) -> None:
        cases = []
        report = _valid_report()
        report["results"]["original"]["teacher_epe_px"] = None
        cases.append(
            (
                "missing value",
                report,
                "aggregate.original.metric.teacher_epe_px.finite",
            )
        )
        report = _valid_report()
        report["results"]["rotation_bins"][2]["metrics"][
            "oracle_residual_overflow_given_solvable_x_sample_rate"
        ] = float("nan")
        cases.append(
            (
                "diagnostic nan",
                report,
                "rotation_bin.2.metric.oracle_residual_overflow_given_solvable_x_sample_rate.finite",
            )
        )
        report = _valid_report()
        report["results"]["full_geometry_augmented"] = {}
        cases.append(
            (
                "empty aggregate",
                report,
                "aggregate.full_geometry_augmented.metrics_nonempty",
            )
        )
        report = _valid_report()
        del report["results"]["rotation_augmented"][
            "oracle_residual_axis_absmax_px"
        ]["y"]
        cases.append(
            (
                "missing nested metric",
                report,
                "aggregate.rotation_augmented.metric.oracle_residual_axis_absmax_px.y.finite",
            )
        )
        report = _valid_report()
        report["results"]["original"]["future_metric"] = float("inf")
        cases.append(
            (
                "nonfinite extra metric",
                report,
                "aggregate.original.metric.future_metric.finite",
            )
        )

        for label, report, expected_code in cases:
            with self.subTest(label=label):
                before = copy.deepcopy(report)
                decision = evaluate_teacher_capacity_policy(report)
                self.assertFalse(decision["passed"])
                self.assertIn(expected_code, _failure_codes(decision))
                self.assertEqual(
                    _normalize_nonfinite(report), _normalize_nonfinite(before)
                )
                json.dumps(decision, allow_nan=False)

    def test_non_mapping_report_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            evaluate_teacher_capacity_policy([])  # type: ignore[arg-type]


def _normalize_nonfinite(value):
    if isinstance(value, dict):
        return {key: _normalize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_nonfinite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


if __name__ == "__main__":
    unittest.main()
