from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - source-only environments
    torch = None


if torch is not None:

    class _ZeroFlowTeacher(torch.nn.Module):
        def forward(self, image1, image2):
            del image2
            return [image1[:, :2] * 0.0]


    class _CountingFlowPrior(torch.nn.Module):
        forward_calls = 0

        def __init__(self, checkpoint_path, **kwargs):
            super().__init__()
            del checkpoint_path, kwargs

        def forward(self, warped):
            type(self).forward_calls += 1
            return warped[:, :2] * 0.0


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TeacherQuarterTurnDiagnosticTest(unittest.TestCase):
    def test_known_angle_router_boundaries_and_ties(self) -> None:
        from diffusion2raft.teacher_quarter_turn_diagnostic import (
            oracle_quarter_turn_degrees,
            wrap_rotation_degrees,
        )

        cases = {
            0.0: 0,
            44.9: 0,
            45.0: -90,
            45.1: -90,
            89.0: -90,
            91.0: -90,
            134.9: -90,
            135.0: 180,
            135.1: 180,
            180.0: 180,
            -44.9: 0,
            -45.0: 0,
            -45.1: 90,
            -89.0: 90,
            -91.0: 90,
            -134.9: 90,
            -135.0: 90,
            -135.1: 180,
            -180.0: 180,
        }
        for angle, expected in cases.items():
            with self.subTest(angle=angle):
                actual = oracle_quarter_turn_degrees(angle)
                self.assertEqual(actual, expected)
                residual = wrap_rotation_degrees(angle + actual)
                self.assertGreaterEqual(residual, -45.0 - 1.0e-9)
                self.assertLess(residual, 45.0)
        for invalid in (float("nan"), float("inf"), 180.1):
            with self.assertRaises(ValueError):
                oracle_quarter_turn_degrees(invalid)

    def test_absolute_map_is_restored_by_inverse_quarter_turn(self) -> None:
        from diffusion2raft.data import apply_source_homography, source_affine_homography
        from diffusion2raft.teacher_quarter_turn_diagnostic import (
            restore_teacher_flow_from_quarter_turn,
            transform_backward_flow_source_map_by_quarter_turn,
        )

        size = 5
        source = torch.arange(3 * size * size).reshape(3, size, size).float()
        zero = torch.zeros(2, size, size)
        valid = torch.ones(1, size, size, dtype=torch.bool)
        for quarter_turn in (0, -90, 90, 180):
            with self.subTest(quarter_turn=quarter_turn):
                restored = restore_teacher_flow_from_quarter_turn(
                    zero.unsqueeze(0), [quarter_turn]
                ).squeeze(0)
                # The restored map lives before Q, so it equals applying Q^-1
                # to an identity absolute map.
                inverse_angle = -quarter_turn if quarter_turn != 180 else 180
                expected_h = source_affine_homography(
                    (size, size), angle_deg=inverse_angle
                )
                _, expected, _ = apply_source_homography(
                    source, zero, valid, expected_h
                )
                torch.testing.assert_close(restored, expected, rtol=0.0, atol=1.0e-6)

                transformed = transform_backward_flow_source_map_by_quarter_turn(
                    zero.unsqueeze(0), [quarter_turn]
                ).squeeze(0)
                forward_h = source_affine_homography(
                    (size, size), angle_deg=quarter_turn
                )
                _, expected_forward, _ = apply_source_homography(
                    source, zero, valid, forward_h
                )
                torch.testing.assert_close(
                    transformed, expected_forward, rtol=0.0, atol=1.0e-6
                )

    def test_mixed_batch_absolute_map_round_trip(self) -> None:
        from diffusion2raft.teacher_quarter_turn_diagnostic import (
            restore_teacher_flow_from_quarter_turn,
            transform_backward_flow_source_map_by_quarter_turn,
        )

        torch.manual_seed(7)
        flow = torch.randn(4, 2, 9, 9)
        turns = [0, -90, 90, 180]
        canonical = transform_backward_flow_source_map_by_quarter_turn(flow, turns)
        restored = restore_teacher_flow_from_quarter_turn(canonical, turns)
        torch.testing.assert_close(restored, flow, rtol=0.0, atol=4.0e-6)
        self.assertTrue(torch.equal(canonical[0], flow[0]))
        self.assertTrue(torch.equal(restored[0], flow[0]))

        inverse_turns = [0, 90, -90, 180]
        augmented = transform_backward_flow_source_map_by_quarter_turn(
            flow, inverse_turns
        )
        recanonicalized = transform_backward_flow_source_map_by_quarter_turn(
            augmented, turns
        )
        torch.testing.assert_close(
            recanonicalized, flow, rtol=0.0, atol=4.0e-6
        )

    def test_image_rotation_and_zero_turn_control(self) -> None:
        from diffusion2raft.geometry import backward_warp
        from diffusion2raft.teacher_quarter_turn_diagnostic import (
            restore_teacher_flow_from_quarter_turn,
            rotate_source_by_quarter_turn,
        )

        size = 5
        source = torch.arange(3 * size * size).reshape(3, size, size).float()
        zero = torch.zeros(1, 2, size, size)
        for quarter_turn in (0, -90, 90, 180):
            with self.subTest(quarter_turn=quarter_turn):
                canonical_source = rotate_source_by_quarter_turn(
                    source, quarter_turn
                )
                expected = torch.rot90(
                    source,
                    k=-(quarter_turn // 90),
                    dims=(-2, -1),
                )
                self.assertTrue(torch.equal(canonical_source, expected))
                restored = restore_teacher_flow_from_quarter_turn(
                    zero, [quarter_turn]
                )
                canonical_sample = backward_warp(
                    canonical_source.unsqueeze(0), zero
                )
                restored_sample = backward_warp(source.unsqueeze(0), restored)
                torch.testing.assert_close(
                    restored_sample, canonical_sample, rtol=0.0, atol=1.0e-6
                )

        arbitrary = torch.randn(1, 2, size, size)
        q0 = restore_teacher_flow_from_quarter_turn(arbitrary, [0])
        self.assertTrue(torch.equal(q0, arbitrary))

        with self.assertRaisesRegex(ValueError, "square source canvas"):
            rotate_source_by_quarter_turn(torch.zeros(3, 4, 5), -90)

    def test_fixed_point_solver_must_run_before_final_map_back(self) -> None:
        from diffusion2raft.geometry import (
            compose_backward_flows,
            residual_from_composed_flow,
        )
        from diffusion2raft.teacher_quarter_turn_diagnostic import (
            restore_teacher_flow_from_quarter_turn,
        )

        size = 64
        canonical_base = torch.zeros(1, 2, size, size)
        true_residual = torch.zeros_like(canonical_base)
        true_residual[:, 0] = 1.0
        true_residual[:, 1] = 0.5
        canonical_final = compose_backward_flows(
            canonical_base, true_residual
        )
        recovered, consistency = residual_from_composed_flow(
            canonical_base, canonical_final, iterations=6
        )
        torch.testing.assert_close(recovered, true_residual, rtol=0.0, atol=1.0e-6)
        self.assertEqual(float(consistency.max()), 0.0)

        interior = (slice(None), slice(None), slice(8, -8), slice(8, -8))
        for quarter_turn in (0, -90, 90, 180):
            with self.subTest(quarter_turn=quarter_turn):
                mapped_base = restore_teacher_flow_from_quarter_turn(
                    canonical_base, [quarter_turn]
                )
                mapped_final = restore_teacher_flow_from_quarter_turn(
                    canonical_final, [quarter_turn]
                )
                canonical_epe = torch.linalg.vector_norm(
                    canonical_base - canonical_final, dim=1
                )
                mapped_epe = torch.linalg.vector_norm(
                    mapped_base - mapped_final, dim=1
                )
                torch.testing.assert_close(
                    mapped_epe, canonical_epe, rtol=0.0, atol=4.0e-6
                )
                _, mapped_consistency = residual_from_composed_flow(
                    mapped_base, mapped_final, iterations=6
                )
                mean_consistency = float(
                    mapped_consistency[interior].mean().item()
                )
                if quarter_turn == 0:
                    self.assertEqual(mean_consistency, 0.0)
                else:
                    self.assertGreater(mean_consistency, 5.0)

    def test_cpu_end_to_end_is_diagnostic_only_and_policy_rejects_it(self) -> None:
        from diffusion2raft.teacher_capacity_policy import (
            evaluate_teacher_capacity_policy,
        )
        from diffusion2raft.teacher_quarter_turn_diagnostic import (
            BEST_OF_C4_REPORT_KIND,
            BEST_OF_C4_REPORT_VERSION,
            CANONICAL_FRAME_REPORT_KIND,
            CANONICAL_FRAME_REPORT_VERSION,
            FULL_GEOMETRY_REPORT_KIND,
            FULL_GEOMETRY_REPORT_VERSION,
            FULL_GEOMETRY_GRID_REPORT_KIND,
            FULL_GEOMETRY_GRID_REPORT_VERSION,
            REPORT_KIND,
            SOLVER_SWEEP_REPORT_KIND,
            SOLVER_SWEEP_REPORT_VERSION,
            run_quarter_turn_oracle_diagnostic,
        )
        from diffusion2raft.teacher_structural_safety_diagnostic import (
            REPORT_KIND as STRUCTURAL_SAFETY_REPORT_KIND,
            REPORT_VERSION as STRUCTURAL_SAFETY_REPORT_VERSION,
            run_structural_safety_diagnostic,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "data").mkdir()
            (root / "runs").mkdir()
            size = 8
            teacher_path = root / "teacher.pt"
            example = torch.rand(1, 3, size, size)
            traced = torch.jit.trace(
                _ZeroFlowTeacher().eval(), (example, example), strict=False
            )
            torch.jit.save(traced, str(teacher_path))
            teacher_sha256 = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
            checkpoint_path = root / "runs" / "epoch_0020.pt"
            checkpoint_path.write_bytes(b"diagnostic-provenance-only")

            image = np.arange(size * size * 3, dtype=np.uint8).reshape(
                size, size, 3
            )
            Image.fromarray(image).save(root / "data" / "warped.png")
            Image.fromarray(image).save(root / "data" / "target.png")
            np.save(
                root / "data" / "flow.npy",
                np.zeros((size, size, 2), dtype=np.float32),
            )
            manifest = root / "data" / "val.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "warped": "warped.png",
                        "target": "target.png",
                        "flow": "flow.npy",
                        "flow_format": "displacement",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "data": {
                    "train_manifest": "data/val.jsonl",
                    "val_manifest": "data/val.jsonl",
                    "work_size": [size, size],
                    "source_geometry_augment": {
                        "probability": 0.7,
                        "max_rotation_deg": 180.0,
                        "scale": [0.85, 1.05],
                        "translation": [0.04, 0.04],
                        "perspective": 0.025,
                    },
                },
                "model": {
                    "prior_backend": "torchscript",
                    "prior_torchscript_path": str(teacher_path),
                    "prior_torchscript_sha256": teacher_sha256,
                    "prior_torchscript_size": size,
                    "prior_torchscript_flow_size": size,
                    "prior_torchscript_blur_kernel": 1,
                    "prior_torchscript_autocast_dtype": "float32",
                    "prior_torchscript_requires_logical_cuda0": False,
                    "feature_stride": 2,
                    "max_residual_px": 24.0,
                },
                "train": {"resume": str(checkpoint_path)},
                "loss": {
                    "max_valid_flow": 1000.0,
                    "max_residual_target": 24.0,
                    "max_residual_consistency": 1.0,
                    "residual_target_iterations": 3,
                },
            }
            config_path = root / "configs" / "audit.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = root / "diagnostics" / "quarter-turn.json"
            report = run_quarter_turn_oracle_diagnostic(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                teacher_path=teacher_path,
                expected_teacher_sha256=teacher_sha256,
                output_path=output,
                split="val",
                sample_count=1,
                explicit_rotation_angles=[90.0, -90.0, 180.0],
                batch_size=2,
                device="cpu",
            )

            self.assertEqual(report["kind"], REPORT_KIND)
            self.assertTrue(report["diagnostic_only"])
            self.assertFalse(report["can_approve_production"])
            self.assertTrue(report["uses_ground_truth_rotation"])
            aggregate = report["results"]["oracle_rotation_augmented"]
            self.assertEqual(aggregate["sample_count"], 3)
            self.assertAlmostEqual(aggregate["teacher_epe_px"], 0.0, places=5)
            self.assertAlmostEqual(aggregate["trainable_coverage"], 1.0, places=5)
            self.assertEqual(
                report["results"]["quarter_turn_histogram"],
                {"0": 0, "-90": 1, "90": 1, "180": 1},
            )
            self.assertNotIn("solver_iteration_sweep", report["results"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

            canonical_output = (
                root / "diagnostics" / "quarter-turn-canonical-v2.json"
            )
            canonical_report = run_quarter_turn_oracle_diagnostic(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                teacher_path=teacher_path,
                expected_teacher_sha256=teacher_sha256,
                output_path=canonical_output,
                split="val",
                sample_count=1,
                explicit_rotation_angles=[90.0, -90.0, 180.0],
                batch_size=2,
                device="cpu",
                canonical_frame_v2=True,
            )
            self.assertEqual(
                canonical_report["kind"], CANONICAL_FRAME_REPORT_KIND
            )
            self.assertEqual(
                canonical_report["report_version"],
                CANONICAL_FRAME_REPORT_VERSION,
            )
            canonical_aggregate = canonical_report["results"][
                "canonical_rotation_augmented"
            ]
            mapped_aggregate = canonical_report["results"][
                "mapped_back_control_rotation_augmented"
            ]
            self.assertEqual(canonical_aggregate["sample_count"], 3)
            self.assertAlmostEqual(
                canonical_aggregate["teacher_epe_px"], 0.0, places=5
            )
            self.assertAlmostEqual(
                canonical_aggregate["trainable_coverage"], 1.0, places=5
            )
            self.assertAlmostEqual(
                mapped_aggregate["teacher_epe_px"], 0.0, places=5
            )
            self.assertEqual(
                canonical_report["results"][
                    "canonical_teacher_epe_over_50px_sample_count"
                ],
                0,
            )
            self.assertLessEqual(
                canonical_report["results"][
                    "teacher_epe_isometry_absmax_px"
                ],
                1.0e-5,
            )
            self.assertLessEqual(
                canonical_report["results"][
                    "gt_absolute_map_roundtrip_absmax_px"
                ],
                1.0e-5,
            )
            quarter_counts = {
                value["oracle_quarter_turn_deg"]: value["canonical_metrics"][
                    "sample_count"
                ]
                for value in canonical_report["results"]["quarter_turns"]
            }
            self.assertEqual(quarter_counts, {0: 0, -90: 1, 90: 1, 180: 1})
            self.assertEqual(
                json.loads(canonical_output.read_text(encoding="utf-8")),
                canonical_report,
            )
            self.assertNotIn(
                "solver_iteration_sweep", canonical_report["results"]
            )

            sweep_output = (
                root / "diagnostics" / "quarter-turn-canonical-solver-sweep-v3.json"
            )
            _CountingFlowPrior.forward_calls = 0
            sweep_report = run_quarter_turn_oracle_diagnostic(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                teacher_path=teacher_path,
                expected_teacher_sha256=teacher_sha256,
                output_path=sweep_output,
                split="val",
                sample_count=1,
                explicit_rotation_angles=[90.0, -90.0, 180.0],
                batch_size=2,
                device="cpu",
                canonical_frame_v2=True,
                residual_target_iteration_sweep=[3, 6],
                teacher_factory=_CountingFlowPrior,
            )
            # Three variants at batch size two require exactly two teacher
            # forwards; the two solver settings must reuse those same flows.
            self.assertEqual(_CountingFlowPrior.forward_calls, 2)
            self.assertEqual(sweep_report["kind"], SOLVER_SWEEP_REPORT_KIND)
            self.assertEqual(
                sweep_report["report_version"], SOLVER_SWEEP_REPORT_VERSION
            )
            self.assertTrue(sweep_report["diagnostic_only"])
            self.assertFalse(sweep_report["can_approve_production"])
            self.assertEqual(
                sweep_report["protocol"]["residual_target_iteration_sweep"],
                [3, 6],
            )
            sweep_results = sweep_report["results"]
            self.assertEqual(
                sweep_results["baseline_residual_target_iterations"], 3
            )
            sweep_entries = sweep_results["solver_iteration_sweep"]
            self.assertEqual(
                [entry["residual_target_iterations"] for entry in sweep_entries],
                [3, 6],
            )
            for entry in sweep_entries:
                with self.subTest(
                    residual_target_iterations=entry[
                        "residual_target_iterations"
                    ]
                ):
                    self.assertEqual(
                        entry["canonical_rotation_augmented"]["sample_count"],
                        3,
                    )
                    self.assertEqual(
                        sum(
                            item["metrics"]["sample_count"]
                            for item in entry["canonical_rotation_bins"]
                        ),
                        3,
                    )
                    self.assertEqual(
                        sum(
                            item["metrics"]["sample_count"]
                            for item in entry[
                                "canonical_residual_rotation_bins"
                            ]
                        ),
                        3,
                    )
                    quarter_counts = {
                        value["oracle_quarter_turn_deg"]: value[
                            "canonical_metrics"
                        ]["sample_count"]
                        for value in entry["quarter_turns"]
                    }
                    self.assertEqual(
                        quarter_counts, {0: 0, -90: 1, 90: 1, 180: 1}
                    )
            self.assertEqual(
                sweep_results[
                    "baseline_mapped_back_control_rotation_augmented"
                ]["sample_count"],
                3,
            )
            self.assertIn(
                "baseline_mapped_back_control_rotation_bins", sweep_results
            )
            self.assertIn(
                "baseline_mapped_back_control_quarter_turns", sweep_results
            )
            self.assertEqual(
                sweep_results["quarter_turn_histogram"],
                {"0": 0, "-90": 1, "90": 1, "180": 1},
            )
            self.assertEqual(len(sweep_results["samples"]), 3)
            for sample in sweep_results["samples"]:
                self.assertEqual(
                    set(
                        sample[
                            "canonical_metrics_by_residual_target_iterations"
                        ]
                    ),
                    {"3", "6"},
                )
            self.assertEqual(
                json.loads(sweep_output.read_text(encoding="utf-8")),
                sweep_report,
            )

            full_geometry_output = (
                root / "diagnostics" / "quarter-turn-canonical-full-geometry-v4.json"
            )
            _CountingFlowPrior.forward_calls = 0
            full_geometry_report = run_quarter_turn_oracle_diagnostic(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                teacher_path=teacher_path,
                expected_teacher_sha256=teacher_sha256,
                output_path=full_geometry_output,
                split="val",
                sample_count=1,
                batch_size=1,
                device="cpu",
                canonical_frame_v2=True,
                full_geometry_per_sample=1,
                residual_target_iterations_override=6,
                teacher_factory=_CountingFlowPrior,
            )
            self.assertEqual(_CountingFlowPrior.forward_calls, 1)
            self.assertEqual(
                full_geometry_report["kind"], FULL_GEOMETRY_REPORT_KIND
            )
            self.assertEqual(
                full_geometry_report["report_version"],
                FULL_GEOMETRY_REPORT_VERSION,
            )
            self.assertTrue(full_geometry_report["diagnostic_only"])
            self.assertFalse(full_geometry_report["can_approve_production"])
            full_protocol = full_geometry_report["protocol"]
            self.assertEqual(full_protocol["configured_residual_target_iterations"], 3)
            self.assertEqual(full_protocol["residual_target_iterations"], 6)
            self.assertEqual(
                full_protocol["source_full_geometry"]["transformations_per_sample"],
                1,
            )
            self.assertTrue(
                full_protocol["source_full_geometry"][
                    "reuses_source_geometry_augment_sample_homography"
                ]
            )
            full_results = full_geometry_report["results"]
            self.assertEqual(
                full_results["canonical_full_geometry_augmented"]["sample_count"],
                1,
            )
            self.assertEqual(
                sum(
                    item["metrics"]["sample_count"]
                    for item in full_results["canonical_rotation_bins"]
                ),
                1,
            )
            self.assertEqual(
                sum(
                    item["metrics"]["sample_count"]
                    for item in full_results["canonical_residual_rotation_bins"]
                ),
                1,
            )
            full_sample = full_results["samples"][0]
            self.assertIsInstance(full_sample["full_geometry_seed"], int)
            self.assertEqual(len(full_sample["source_homography"]), 3)
            self.assertEqual(
                json.loads(full_geometry_output.read_text(encoding="utf-8")),
                full_geometry_report,
            )
            self.assertNotIn("capacity_grid", full_results)

            capacity_grid_output = (
                root / "diagnostics" / "quarter-turn-full-geometry-grid-v5.json"
            )
            _CountingFlowPrior.forward_calls = 0
            capacity_grid_report = run_quarter_turn_oracle_diagnostic(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                teacher_path=teacher_path,
                expected_teacher_sha256=teacher_sha256,
                output_path=capacity_grid_output,
                split="val",
                sample_count=1,
                batch_size=1,
                device="cpu",
                canonical_frame_v2=True,
                full_geometry_per_sample=1,
                residual_target_iterations_override=6,
                full_geometry_solver_iteration_sweep=[6, 9],
                full_geometry_residual_cap_sweep=[24, 32],
                teacher_factory=_CountingFlowPrior,
            )
            self.assertEqual(_CountingFlowPrior.forward_calls, 1)
            self.assertEqual(
                capacity_grid_report["kind"], FULL_GEOMETRY_GRID_REPORT_KIND
            )
            self.assertEqual(
                capacity_grid_report["report_version"],
                FULL_GEOMETRY_GRID_REPORT_VERSION,
            )
            grid_protocol = capacity_grid_report["protocol"]
            self.assertEqual(
                grid_protocol["full_geometry_solver_iteration_sweep"], [6, 9]
            )
            self.assertEqual(
                grid_protocol["full_geometry_residual_cap_sweep_px"],
                [24.0, 32.0],
            )
            grid_results = capacity_grid_report["results"]
            grid_entries = grid_results["capacity_grid"]
            self.assertEqual(
                [
                    (
                        value["residual_target_iterations"],
                        value["max_residual_px"],
                    )
                    for value in grid_entries
                ],
                [(6, 24.0), (6, 32.0), (9, 24.0), (9, 32.0)],
            )
            for entry in grid_entries:
                self.assertEqual(
                    entry["canonical_full_geometry_augmented"]["sample_count"],
                    1,
                )
            self.assertEqual(
                grid_entries[0]["canonical_full_geometry_augmented"],
                full_results["canonical_full_geometry_augmented"],
            )
            self.assertEqual(
                len(
                    grid_results["samples"][0][
                        "canonical_metrics_by_capacity_grid"
                    ]
                ),
                4,
            )
            self.assertEqual(
                json.loads(capacity_grid_output.read_text(encoding="utf-8")),
                capacity_grid_report,
            )

            best_of_c4_output = (
                root / "diagnostics" / "quarter-turn-best-of-c4-v6.json"
            )
            _CountingFlowPrior.forward_calls = 0
            best_of_c4_report = run_quarter_turn_oracle_diagnostic(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                teacher_path=teacher_path,
                expected_teacher_sha256=teacher_sha256,
                output_path=best_of_c4_output,
                split="val",
                sample_count=1,
                batch_size=1,
                device="cpu",
                canonical_frame_v2=True,
                full_geometry_per_sample=1,
                residual_target_iterations_override=6,
                full_geometry_solver_iteration_sweep=[6, 9],
                full_geometry_residual_cap_sweep=[24, 32],
                full_geometry_best_of_c4=True,
                c4_candidate_batch_size=2,
                teacher_factory=_CountingFlowPrior,
            )
            self.assertEqual(_CountingFlowPrior.forward_calls, 2)
            self.assertEqual(best_of_c4_report["kind"], BEST_OF_C4_REPORT_KIND)
            self.assertEqual(
                best_of_c4_report["report_version"],
                BEST_OF_C4_REPORT_VERSION,
            )
            self.assertTrue(
                best_of_c4_report[
                    "uses_ground_truth_flow_for_candidate_selection"
                ]
            )
            best_protocol = best_of_c4_report["protocol"]["best_of_c4"]
            self.assertEqual(best_protocol["candidate_batch_size"], 2)
            self.assertEqual(best_protocol["candidate_order"], [0, -90, 90, 180])
            best_results = best_of_c4_report["results"]
            self.assertEqual(best_results["capacity_grid"], grid_entries)
            self.assertEqual(
                best_results["nearest_angle_capacity_grid"], grid_entries
            )
            self.assertEqual(
                len(best_results["best_teacher_epe_capacity_grid"]), 4
            )
            self.assertEqual(
                len(best_results["best_capacity_aware_capacity_grid"]), 4
            )
            self.assertEqual(len(best_results["all_candidate_capacity_grid"]), 4)
            self.assertEqual(
                best_results["routing_comparison"]["sample_count"], 1
            )
            best_sample = best_results["samples"][0]["c4_best_of_four"]
            self.assertEqual(len(best_sample["candidates"]), 4)
            self.assertEqual(
                {value["quarter_turn_deg"] for value in best_sample["candidates"]},
                {0, -90, 90, 180},
            )
            self.assertEqual(
                json.loads(best_of_c4_output.read_text(encoding="utf-8")),
                best_of_c4_report,
            )

            structural_output = (
                root / "diagnostics" / "quarter-turn-structural-safety-v7.json"
            )
            _CountingFlowPrior.forward_calls = 0
            structural_report = run_structural_safety_diagnostic(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                teacher_path=teacher_path,
                expected_teacher_sha256=teacher_sha256,
                best_of_c4_report_path=best_of_c4_output,
                output_path=structural_output,
                split="val",
                sample_count=1,
                seed=42,
                batch_size=1,
                device="cpu",
                residual_target_iterations=6,
                residual_cap_sweep=[24, 32],
                baseline_max_residual_px=24,
                selected_max_residual_px=32,
                teacher_factory=_CountingFlowPrior,
            )
            # v7 consumes the already-frozen best C4 label and performs one
            # selected teacher forward, rather than rerunning all four labels.
            self.assertEqual(_CountingFlowPrior.forward_calls, 1)
            self.assertEqual(
                structural_report["kind"], STRUCTURAL_SAFETY_REPORT_KIND
            )
            self.assertEqual(
                structural_report["report_version"],
                STRUCTURAL_SAFETY_REPORT_VERSION,
            )
            self.assertTrue(structural_report["diagnostic_only"])
            self.assertFalse(structural_report["can_approve_production"])
            self.assertEqual(
                structural_report["protocol"]["selection"]["candidate_order"],
                [0, -90, 90, 180],
            )
            self.assertEqual(
                structural_report["results"]["selected_cell"][
                    "residual_target_iterations"
                ],
                6,
            )
            self.assertEqual(
                structural_report["results"]["selected_cell"][
                    "max_residual_px"
                ],
                32.0,
            )
            self.assertEqual(
                len(structural_report["results"]["structural_grid"]), 2
            )
            self.assertEqual(len(structural_report["results"]["samples"]), 1)
            self.assertFalse(
                structural_report["results"]["decision"][
                    "can_approve_production"
                ]
            )
            self.assertEqual(
                json.loads(structural_output.read_text(encoding="utf-8")),
                structural_report,
            )

            decision = evaluate_teacher_capacity_policy(report)
            self.assertFalse(decision["passed"])
            self.assertIn("report.kind", {item["code"] for item in decision["failures"]})
            canonical_decision = evaluate_teacher_capacity_policy(canonical_report)
            self.assertFalse(canonical_decision["passed"])
            self.assertIn(
                "report.kind",
                {item["code"] for item in canonical_decision["failures"]},
            )
            sweep_decision = evaluate_teacher_capacity_policy(sweep_report)
            self.assertFalse(sweep_decision["passed"])
            self.assertIn(
                "report.kind",
                {item["code"] for item in sweep_decision["failures"]},
            )
            full_geometry_decision = evaluate_teacher_capacity_policy(
                full_geometry_report
            )
            self.assertFalse(full_geometry_decision["passed"])
            self.assertIn(
                "report.kind",
                {item["code"] for item in full_geometry_decision["failures"]},
            )
            capacity_grid_decision = evaluate_teacher_capacity_policy(
                capacity_grid_report
            )
            self.assertFalse(capacity_grid_decision["passed"])
            self.assertIn(
                "report.kind",
                {item["code"] for item in capacity_grid_decision["failures"]},
            )
            best_of_c4_decision = evaluate_teacher_capacity_policy(
                best_of_c4_report
            )
            self.assertFalse(best_of_c4_decision["passed"])
            self.assertIn(
                "report.kind",
                {item["code"] for item in best_of_c4_decision["failures"]},
            )
            structural_decision = evaluate_teacher_capacity_policy(
                structural_report
            )
            self.assertFalse(structural_decision["passed"])
            self.assertIn(
                "report.kind",
                {item["code"] for item in structural_decision["failures"]},
            )
            self.assertFalse((root / "approved.json").exists())
            self.assertFalse(
                (root / "runs" / "preflight_v33_teacher_capacity" / "approved.json").exists()
            )

            common_kwargs = {
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "teacher_path": teacher_path,
                "expected_teacher_sha256": teacher_sha256,
                "output_path": root / "diagnostics" / "invalid-sweep.json",
                "split": "val",
                "sample_count": 1,
                "explicit_rotation_angles": [90.0],
                "batch_size": 1,
                "device": "cpu",
            }
            with self.assertRaisesRegex(
                ValueError, "best-of-C4 requires the full geometry capacity grid"
            ):
                run_quarter_turn_oracle_diagnostic(
                    **common_kwargs,
                    canonical_frame_v2=True,
                    full_geometry_best_of_c4=True,
                )
            with self.assertRaisesRegex(
                ValueError, "c4_candidate_batch_size must be a positive integer"
            ):
                run_quarter_turn_oracle_diagnostic(
                    **common_kwargs,
                    c4_candidate_batch_size=0,
                )
            with self.assertRaisesRegex(ValueError, "requires canonical_frame_v2"):
                run_quarter_turn_oracle_diagnostic(
                    **common_kwargs,
                    residual_target_iteration_sweep=[3, 6],
                )
            for invalid_sweep in ([], [0, 3], [3, 3], [6, 3]):
                with self.subTest(invalid_sweep=invalid_sweep):
                    with self.assertRaisesRegex(
                        ValueError,
                        "non-empty, unique, strictly increasing sequence",
                    ):
                        run_quarter_turn_oracle_diagnostic(
                            **common_kwargs,
                            canonical_frame_v2=True,
                            residual_target_iteration_sweep=invalid_sweep,
                        )
            with self.assertRaisesRegex(ValueError, "include the configured baseline"):
                full_grid_kwargs = {
                    **common_kwargs,
                    "explicit_rotation_angles": None,
                }
                run_quarter_turn_oracle_diagnostic(
                    **full_grid_kwargs,
                    canonical_frame_v2=True,
                    residual_target_iteration_sweep=[6, 12],
                )
            with self.assertRaisesRegex(ValueError, "sequence of integers"):
                run_quarter_turn_oracle_diagnostic(
                    **common_kwargs,
                    canonical_frame_v2=True,
                    residual_target_iteration_sweep=[3, 6.5],
                )
            with self.assertRaisesRegex(ValueError, "requires canonical_frame_v2"):
                run_quarter_turn_oracle_diagnostic(
                    **common_kwargs,
                    full_geometry_per_sample=1,
                    residual_target_iterations_override=6,
                )
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                run_quarter_turn_oracle_diagnostic(
                    **common_kwargs,
                    canonical_frame_v2=True,
                    full_geometry_per_sample=1,
                    residual_target_iterations_override=6,
                    residual_target_iteration_sweep=[3, 6],
                )
            with self.assertRaisesRegex(ValueError, "restricted to full geometry"):
                run_quarter_turn_oracle_diagnostic(
                    **common_kwargs,
                    canonical_frame_v2=True,
                    residual_target_iterations_override=6,
                )
            with self.assertRaisesRegex(ValueError, "requires both"):
                run_quarter_turn_oracle_diagnostic(
                    **common_kwargs,
                    canonical_frame_v2=True,
                    full_geometry_per_sample=1,
                    residual_target_iterations_override=6,
                    full_geometry_solver_iteration_sweep=[6, 9],
                )
            with self.assertRaisesRegex(ValueError, "include the configured baseline"):
                run_quarter_turn_oracle_diagnostic(
                    **full_grid_kwargs,
                    canonical_frame_v2=True,
                    full_geometry_per_sample=1,
                    residual_target_iterations_override=6,
                    full_geometry_solver_iteration_sweep=[6, 9],
                    full_geometry_residual_cap_sweep=[32, 40],
                )

    def test_output_path_cannot_impersonate_production_approval(self) -> None:
        from diffusion2raft import teacher_quarter_turn_diagnostic as diagnostic

        with self.assertRaisesRegex(ValueError, "approved.json"):
            diagnostic._validate_output_path(Path("approved.json"))
        with self.assertRaisesRegex(ValueError, "production capacity directory"):
            diagnostic._validate_output_path(
                Path("runs/preflight_v33_teacher_capacity/oracle.json")
            )


if __name__ == "__main__":
    unittest.main()
