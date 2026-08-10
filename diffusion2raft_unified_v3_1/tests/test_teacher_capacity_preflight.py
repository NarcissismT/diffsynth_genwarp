from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TeacherCapacityPreflightTest(unittest.TestCase):
    def test_identity_check_rejects_replaced_checkpoint(self) -> None:
        from diffusion2raft import teacher_capacity_preflight as capacity

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            path.write_bytes(b"epoch-one")
            identity = capacity._file_identity(path)
            replacement = path.with_name("next.pt")
            replacement.write_bytes(b"epoch-two")
            replacement.replace(path)
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                capacity._assert_path_matches_identity(identity)

    def test_rotation_plan_is_reproducible_and_deduplicates_zero_and_180(self) -> None:
        from diffusion2raft.teacher_capacity_preflight import build_rotation_plan

        explicit, protocol = build_rotation_plan(
            [2, 7],
            explicit_angles=[0.0, 90.0, 90.0, -90.0, 180.0, -180.0],
            max_rotation_deg=180.0,
            rotations_per_sample=99,
            seed=1,
        )
        self.assertEqual(explicit[2], [90.0, -90.0, 180.0])
        self.assertEqual(explicit[7], [90.0, -90.0, 180.0])
        self.assertEqual(protocol["mode"], "explicit_cross_product")

        first, _ = build_rotation_plan(
            [0, 4, 9],
            explicit_angles=None,
            max_rotation_deg=120.0,
            rotations_per_sample=2,
            seed=17,
        )
        second, _ = build_rotation_plan(
            [0, 4, 9],
            explicit_angles=None,
            max_rotation_deg=120.0,
            rotations_per_sample=2,
            seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual(sum(map(len, first.values())), 6)
        self.assertTrue(
            all(abs(angle) <= 120.0 for values in first.values() for angle in values)
        )

    def test_full_geometry_seed_plan_is_reproducible_and_index_scoped(self) -> None:
        from diffusion2raft.teacher_capacity_preflight import (
            build_full_geometry_plan,
        )

        first, protocol = build_full_geometry_plan(
            [2, 7], transformations_per_sample=3, seed=17
        )
        second, second_protocol = build_full_geometry_plan(
            [2, 7], transformations_per_sample=3, seed=17
        )
        reordered, _ = build_full_geometry_plan(
            [7, 2], transformations_per_sample=3, seed=17
        )
        different_seed, _ = build_full_geometry_plan(
            [2, 7], transformations_per_sample=3, seed=18
        )

        self.assertEqual(first, second)
        self.assertEqual(protocol, second_protocol)
        self.assertEqual(first[2], reordered[2])
        self.assertEqual(first[7], reordered[7])
        self.assertNotEqual(first[2], first[7])
        self.assertNotEqual(first, different_seed)
        self.assertEqual(len(set(first[2] + first[7])), 6)
        self.assertTrue(protocol["enabled"])
        self.assertEqual(protocol["transformations_per_sample"], 3)
        self.assertEqual(len(protocol["seed_plan_sha256"]), 64)

        disabled, disabled_protocol = build_full_geometry_plan(
            [2, 7], transformations_per_sample=0, seed=17
        )
        self.assertEqual(disabled, {2: [], 7: []})
        self.assertFalse(disabled_protocol["enabled"])
        self.assertEqual(disabled_protocol["mode"], "disabled")

    def test_full_geometry_sampler_is_reproducible_and_restores_cpu_rng(self) -> None:
        from diffusion2raft.data import SourceGeometryAugment
        from diffusion2raft.teacher_capacity_preflight import (
            _sample_deterministic_full_homography,
        )

        augment = SourceGeometryAugment(
            # The audit samples the configured distribution conditional on the
            # branch being triggered, so probability must not suppress it.
            probability=0.0,
            max_rotation_deg=180.0,
            scale=(0.85, 1.05),
            translation=(0.04, 0.04),
            perspective=0.025,
        )
        source = torch.zeros(3, 16, 16)
        torch.manual_seed(123456)
        state_before = torch.random.get_rng_state().clone()
        with patch.object(
            augment,
            "sample_homography",
            wraps=augment.sample_homography,
        ) as sampler:
            first = _sample_deterministic_full_homography(
                augment, source, seed=2026
            )
        self.assertEqual(sampler.call_count, 1)
        self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))

        second = _sample_deterministic_full_homography(
            augment, source, seed=2026
        )
        different = _sample_deterministic_full_homography(
            augment, source, seed=2027
        )
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))

    def test_component_linf_not_vector_l2_defines_overflow(self) -> None:
        from diffusion2raft.teacher_capacity_preflight import (
            capacity_sample_statistics,
        )

        size = 8
        teacher = torch.zeros(1, 2, size, size)
        # L2 is >2, but each component is <2.  This must remain within the
        # refiner's per-axis tanh bound.
        target = torch.full_like(teacher, 1.5)
        valid = torch.ones(1, 1, size, size, dtype=torch.bool)
        stats = capacity_sample_statistics(
            teacher,
            target,
            valid,
            max_residual_px=2.0,
            residual_target_iterations=2,
            max_residual_consistency=1.0e-5,
            max_valid_flow=1000.0,
            feature_stride=2,
        )[0]
        self.assertGreater(stats.solver_pixels, 0)
        self.assertEqual(stats.overflow_x_solvable_pixels, 0)
        self.assertEqual(stats.overflow_y_solvable_pixels, 0)
        self.assertEqual(stats.overflow_any_solvable_pixels, 0)
        self.assertEqual(stats.trainable_pixels, stats.solver_pixels)
        self.assertAlmostEqual(
            stats.as_report()["teacher_epe_px"], 2**0.5 * 1.5, places=6
        )

    def test_solver_failure_is_not_counted_as_overflow(self) -> None:
        from diffusion2raft import teacher_capacity_preflight as capacity

        size = 8
        teacher = torch.zeros(1, 2, size, size)
        target = torch.zeros_like(teacher)
        valid = torch.ones(1, 1, size, size, dtype=torch.bool)
        failed_residual = torch.full_like(teacher, 99.0)
        failed_consistency = torch.full((1, 1, size, size), 3.0)
        with patch.object(
            capacity,
            "residual_from_composed_flow",
            return_value=(failed_residual, failed_consistency),
        ):
            stats = capacity.capacity_sample_statistics(
                teacher,
                target,
                valid,
                max_residual_px=24.0,
                residual_target_iterations=6,
                max_residual_consistency=1.0,
                max_valid_flow=1000.0,
                feature_stride=2,
            )[0]
        self.assertEqual(stats.solver_pixels, 0)
        self.assertEqual(stats.overflow_any_solvable_pixels, 0)
        self.assertFalse(stats.overflow_any_given_solvable_sample)
        self.assertEqual(stats.trainable_pixels, 0)
        self.assertIsNone(
            stats.as_report()[
                "oracle_residual_overflow_given_solvable_any_axis_pixel_rate"
            ]
        )

    def test_stride_down_up_reports_spatial_bandwidth_error(self) -> None:
        from diffusion2raft.teacher_capacity_preflight import (
            capacity_sample_statistics,
        )

        size = 8
        teacher = torch.zeros(1, 2, size, size)
        x = torch.arange(size).view(1, 1, 1, size)
        y = torch.arange(size).view(1, 1, size, 1)
        checkerboard = ((x + y) % 2).float().mul(0.5).sub(0.25)
        target = torch.cat((checkerboard.expand(1, 1, size, size), torch.zeros_like(checkerboard).expand(1, 1, size, size)), dim=1)
        valid = torch.ones(1, 1, size, size, dtype=torch.bool)
        stats = capacity_sample_statistics(
            teacher,
            target,
            valid,
            max_residual_px=24.0,
            residual_target_iterations=2,
            max_residual_consistency=1.0e-5,
            max_valid_flow=1000.0,
            feature_stride=2,
        )[0]
        report = stats.as_report()
        self.assertGreater(stats.solver_pixels, 0)
        self.assertGreater(
            report["stride_oracle_residual_reconstruction_epe_px"], 0.0
        )

    def test_cpu_end_to_end_writes_atomic_provenance_report(self) -> None:
        from diffusion2raft.teacher_capacity_preflight import run_capacity_preflight

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "data").mkdir()
            (root / "runs").mkdir()
            size = 8
            teacher_path = root / "teacher.pt"
            teacher_override_path = root / "teacher_override.pt"
            example = torch.rand(1, 3, size, size)
            traced = torch.jit.trace(
                _ZeroFlowTeacher().eval(), (example, example), strict=False
            )
            torch.jit.save(traced, str(teacher_path))
            torch.jit.save(traced, str(teacher_override_path))
            checkpoint_path = root / "runs" / "latest.pt"
            checkpoint_path.write_bytes(b"identity-only-checkpoint")

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
                    "prior_torchscript_path": "teacher.pt",
                    "prior_torchscript_sha256": hashlib.sha256(
                        teacher_path.read_bytes()
                    ).hexdigest(),
                    "prior_torchscript_size": size,
                    "prior_torchscript_flow_size": size,
                    "prior_torchscript_blur_kernel": 1,
                    "prior_torchscript_autocast_dtype": "float32",
                    "prior_torchscript_requires_logical_cuda0": False,
                    "feature_stride": 2,
                    "max_residual_px": 24.0,
                },
                "train": {"resume": "runs/latest.pt"},
                "loss": {
                    "max_valid_flow": 1000.0,
                    "max_residual_target": 24.0,
                    "max_residual_consistency": 1.0,
                    "residual_target_iterations": 3,
                },
            }
            config_path = root / "configs" / "audit.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = root / "report" / "capacity.json"
            report = run_capacity_preflight(
                config_path=config_path,
                output_path=output,
                teacher_path=teacher_override_path,
                split="val",
                sample_count=1,
                explicit_rotation_angles=[0.0, 90.0, -90.0, 180.0, -180.0],
                full_geometry_per_sample=2,
                batch_size=2,
                device="cpu",
                hash_external_files=True,
            )

            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(report["results"]["original"]["sample_count"], 1)
            self.assertEqual(
                report["results"]["rotation_augmented"]["sample_count"], 3
            )
            self.assertEqual(
                report["results"]["full_geometry_augmented"]["sample_count"],
                2,
            )
            self.assertTrue(
                report["protocol"]["source_rotation"]["rotation_isolated"]
            )
            self.assertFalse(
                report["protocol"]["source_rotation"][
                    "equivalent_to_full_source_geometry_augment"
                ]
            )
            full_protocol = report["protocol"]["source_full_geometry"]
            self.assertTrue(full_protocol["enabled"])
            self.assertEqual(full_protocol["transformations_per_sample"], 2)
            self.assertTrue(full_protocol["conditional_on_augmentation_trigger"])
            self.assertTrue(
                full_protocol[
                    "reuses_source_geometry_augment_sample_homography"
                ]
            )
            self.assertEqual(len(full_protocol["seed_plan_sha256"]), 64)
            full_samples = [
                sample
                for sample in report["results"]["samples"]
                if sample["mode"] == "full_geometry_augmented"
            ]
            self.assertEqual(len(full_samples), 2)
            self.assertEqual(
                len({sample["full_geometry_seed"] for sample in full_samples}), 2
            )
            for sample in full_samples:
                self.assertIsNone(sample["rotation_deg"])
                self.assertIsNone(sample["absolute_rotation_deg"])
                self.assertIsNone(sample["rotation_bin_index"])
                self.assertEqual(
                    np.asarray(sample["source_homography"]).shape, (3, 3)
                )
            self.assertEqual(report["identities"]["manifest"]["record_count"], 1)
            self.assertIsNotNone(
                report["identities"]["teacher"]["checkpoint"]["sha256"]
            )
            teacher_selection = report["identities"]["teacher"]["selection"]
            self.assertEqual(teacher_selection["source"], "explicit_override")
            self.assertEqual(
                teacher_selection["config_resolved_path"], str(teacher_path)
            )
            self.assertEqual(
                teacher_selection["effective_path"], str(teacher_override_path)
            )
            self.assertEqual(
                report["identities"]["teacher"]["checkpoint"]["resolved_path"],
                str(teacher_override_path),
            )
            self.assertIn(
                "not loaded", report["identities"]["checkpoint"]["role"]
            )
            self.assertFalse(any(output.parent.glob(f".{output.name}.tmp.*")))


if __name__ == "__main__":
    unittest.main()
