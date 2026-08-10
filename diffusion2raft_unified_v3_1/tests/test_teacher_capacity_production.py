from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion2raft import teacher_capacity_production as production  # noqa: E402
from diffusion2raft.teacher_capacity_receipt import (  # noqa: E402
    build_teacher_capacity_receipt,
    decode_teacher_capacity_receipt_base64,
)


BIN_EDGES = [0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
BIN_COUNTS = [40, 40, 40, 40, 40, 50, 50]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(count: int, *, aggregate: bool) -> dict:
    return {
        "sample_count": count,
        "eval_sample_count": count,
        "eval_pixels": count * 10,
        "teacher_epe_px": 5.0,
        "oracle_solver_coverage": 0.995 if aggregate else 0.985,
        "oracle_solver_any_sample_rate": 1.0,
        "oracle_solver_full_sample_rate": 0.99,
        "oracle_residual_overflow_given_solvable_x_pixel_rate": 0.002,
        "oracle_residual_overflow_given_solvable_y_pixel_rate": 0.002,
        "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": (
            0.004 if aggregate else 0.009
        ),
        "oracle_residual_overflow_given_solvable_x_sample_rate": 0.9,
        "oracle_residual_overflow_given_solvable_y_sample_rate": 0.9,
        "oracle_residual_overflow_given_solvable_any_axis_sample_rate": 1.0,
        "trainable_coverage": 0.99 if aggregate else 0.975,
        "residual_target_valid_rate": 0.99 if aggregate else 0.975,
        "oracle_residual_axis_absmax_px": {"x": 30.0, "y": 31.0},
        "stride_oracle_residual_reconstruction_epe_px": 0.8,
        "stride_trainable_oracle_reconstruction_epe_px": (
            0.9 if aggregate else 1.4
        ),
    }


class TeacherCapacityProductionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        (self.root / "data").mkdir()
        (self.root / "runs").mkdir()
        self.teacher = self.root / "teacher.pt"
        self.teacher.write_bytes(b"teacher-bytes")
        self.manifest = self.root / "data" / "val.jsonl"
        for name, value in (
            ("warped.bin", b"warped"),
            ("target.bin", b"target"),
            ("flow.bin", b"flow"),
        ):
            (self.root / "data" / name).write_bytes(value)
        records = [
            {
                "id": f"sample-{index}",
                "warped": "warped.bin",
                "target": "target.bin",
                "flow": "flow.bin",
            }
            for index in range(300)
        ]
        self.manifest.write_text(
            "".join(json.dumps(value) + "\n" for value in records),
            encoding="utf-8",
        )
        self.learned = self.root / "runs" / "learned.pt"
        torch.save(
            {
                "model": {"prior.weight": torch.ones(1)},
                "optimizer": {"param_groups": [{}]},
                "stage": "unified",
                "epoch": 19,
                "prior_backend": "learned",
                "config": {"model": {"prior_backend": "learned"}},
            },
            self.learned,
        )
        config = {
            "data": {
                "val_manifest": str(self.manifest),
                "work_size": [512, 512],
            },
            "model": {
                "prior_backend": "torchscript",
                "prior_torchscript_path": str(self.teacher),
                "prior_torchscript_sha256": _sha(self.teacher),
                "prior_torchscript_size": 512,
                "prior_torchscript_flow_size": 512,
                "prior_torchscript_blur_kernel": 39,
                "prior_torchscript_autocast_dtype": "float16",
                "prior_torchscript_requires_logical_cuda0": False,
                "feature_stride": 8,
                "max_residual_px": 24.0,
            },
            "train": {"resume": str(self.learned)},
            "loss": {"max_residual_target": 24.0},
        }
        self.config = self.root / "configs" / "production.yaml"
        self.config.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.output = self.root / "approved"
        self.pointer = self.output / "approved.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _report(self) -> dict:
        bins = [
            {
                "index": index,
                "absolute_rotation_deg": {
                    "lower_inclusive": BIN_EDGES[index],
                    "upper": BIN_EDGES[index + 1],
                    "upper_inclusive": index == len(BIN_COUNTS) - 1,
                },
                "metrics": _metrics(count, aggregate=False),
            }
            for index, count in enumerate(BIN_COUNTS)
        ]
        return {
            "report_version": 1,
            "kind": "frozen_teacher_residual_capacity_preflight",
            "identities": {
                "config": {"sha256": _sha(self.config)},
                "checkpoint": {"sha256": _sha(self.learned)},
                "teacher": {
                    "backend": "torchscript",
                    "selection": {
                        "source": "config",
                        "override_value": None,
                        "config_resolved_path": str(self.teacher),
                        "effective_path": str(self.teacher),
                    },
                    "checkpoint": {
                        "sha256": _sha(self.teacher),
                        "size_bytes": self.teacher.stat().st_size,
                    },
                    "input_size": 512,
                    "flow_size": 512,
                    "blur_kernel": 39,
                    "autocast_dtype": "float16",
                    "requires_logical_cuda0": False,
                },
                "manifest": {
                    "configured_path": str(self.manifest),
                    "sha256": _sha(self.manifest),
                    "split": "val",
                    "record_count": 300,
                },
            },
            "protocol": {
                "work_size": [512, 512],
                "requested_sample_count": 0,
                "selected_sample_count": 300,
                "selected_indices": list(range(300)),
                "selected_indices_sha256": "3" * 64,
                "rotation_plan_sha256": "4" * 64,
                "source_rotation": {
                    "mode": "config_stratified_uniform",
                    "rotations_per_sample": 1,
                    "bin_edges_deg": BIN_EDGES,
                },
                "source_full_geometry": {
                    "enabled": True,
                    "mode": "deterministic_config_distribution_conditional_on_trigger",
                    "transformations_per_sample": 1,
                    "seed_plan_sha256": "5" * 64,
                    "reuses_source_geometry_augment_sample_homography": True,
                    "conditional_on_augmentation_trigger": True,
                    "equivalent_to_conditional_full_source_geometry_augment": True,
                },
                "feature_stride": 8,
                "max_residual_px": 24.0,
                "max_residual_target": 24.0,
                "external_file_sha256_enabled": True,
            },
            "results": {
                "original": _metrics(300, aggregate=True),
                "rotation_augmented": _metrics(300, aggregate=True),
                "full_geometry_augmented": _metrics(300, aggregate=True),
                "rotation_bins": bins,
                "samples": [],
            },
        }

    def _generate(self, report: dict | None = None) -> dict:
        with mock.patch.object(
            production,
            "run_capacity_preflight",
            return_value=self._report() if report is None else report,
        ) as audit:
            result = production.generate_teacher_capacity_production(
                config_path=self.config,
                pointer_path=self.pointer,
                output_directory=self.output,
                threads=1,
                repository_root=ROOT,
            )
        kwargs = audit.call_args.kwargs
        self.assertEqual(kwargs["sample_count"], 0)
        self.assertEqual(kwargs["rotations_per_sample"], 1)
        self.assertEqual(kwargs["full_geometry_per_sample"], 1)
        self.assertTrue(kwargs["hash_external_files"])
        self.assertIsNone(kwargs["teacher_path"])
        self.assertIsNone(kwargs["manifest_path"])
        return result

    def test_generate_pass_and_learned_verify(self) -> None:
        generated = self._generate()
        self.assertTrue(Path(generated["evidence_path"]).is_file())
        self.assertEqual(Path(generated["pointer_path"]), self.pointer)
        pointer, evidence = production.read_teacher_capacity_evidence_pointer(
            self.pointer
        )
        self.assertEqual(pointer["evidence_report_sha256"], evidence.stem)
        encoded = production.verify_teacher_capacity_production(
            config_path=self.config,
            pointer_path=self.pointer,
            resume_path=self.learned,
            repository_root=ROOT,
        )
        self.assertEqual(encoded, generated["receipt_b64"])

    def test_audit_or_policy_failure_does_not_update_pointer(self) -> None:
        self.output.mkdir()
        self.pointer.write_bytes(b"old-pointer")
        with mock.patch.object(
            production, "run_capacity_preflight", side_effect=RuntimeError("GPU failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                production.generate_teacher_capacity_production(
                    config_path=self.config,
                    pointer_path=self.pointer,
                    output_directory=self.output,
                    threads=1,
                    repository_root=ROOT,
                )
        self.assertEqual(self.pointer.read_bytes(), b"old-pointer")

        failed = self._report()
        failed["results"]["original"]["trainable_coverage"] = 0.1
        failed["results"]["original"]["residual_target_valid_rate"] = 0.1
        with mock.patch.object(
            production, "run_capacity_preflight", return_value=failed
        ):
            with self.assertRaisesRegex(ValueError, "fails production"):
                production.generate_teacher_capacity_production(
                    config_path=self.config,
                    pointer_path=self.pointer,
                    output_directory=self.output,
                    threads=1,
                    repository_root=ROOT,
                )
        self.assertEqual(self.pointer.read_bytes(), b"old-pointer")

    def _teacher_resume(self, receipt: dict) -> Path:
        teacher_stat = self.teacher.stat()
        identity = {
            "version": 2,
            "resolved_path": str(self.teacher),
            "file_size": teacher_stat.st_size,
            "mtime_ns": teacher_stat.st_mtime_ns,
            "sha256": _sha(self.teacher),
            "input_size": 512,
            "flow_size": 512,
            "blur_kernel": 39,
            "autocast_dtype": "float16",
            "requires_logical_cuda0": False,
        }
        path = self.root / "runs" / "teacher.pt"
        torch.save(
            {
                "model": {"prior._teacher_backend_marker": torch.ones(1)},
                "optimizer": {"param_groups": [{}]},
                "stage": "unified",
                "epoch": 20,
                "prior_backend": "torchscript",
                "capacity_evidence_receipt": receipt,
                "teacher_prior_identity": identity,
                "config": {
                    "model": {
                        "prior_backend": "torchscript",
                        "prior_torchscript_sha256": identity["sha256"],
                    }
                },
            },
            path,
        )
        return path

    def test_teacher_resume_and_receipt_mismatch(self) -> None:
        generated = self._generate()
        receipt = decode_teacher_capacity_receipt_base64(generated["receipt_b64"])
        teacher_resume = self._teacher_resume(receipt)
        encoded = production.verify_teacher_capacity_production(
            config_path=self.config,
            pointer_path=self.pointer,
            resume_path=teacher_resume,
            repository_root=ROOT,
        )
        self.assertEqual(encoded, generated["receipt_b64"])

        mismatched = build_teacher_capacity_receipt(
            evidence_report_sha256=receipt["evidence_report_sha256"],
            capacity_contract_sha256="f" * 64,
            capacity_config_projection_sha256=receipt[
                "capacity_config_projection_sha256"
            ],
            protocol_sha256=receipt["protocol_sha256"],
            implementation_sha256=receipt["implementation_sha256"],
            policy=receipt["policy"],
            migration_seed=receipt["migration_seed"],
            teacher=receipt["teacher"],
            manifest=receipt["manifest"],
        )
        bad_resume = self._teacher_resume(mismatched)
        with self.assertRaisesRegex(
            production.TeacherCapacityProductionError, "stored receipt differs"
        ):
            production.verify_teacher_capacity_production(
                config_path=self.config,
                pointer_path=self.pointer,
                resume_path=bad_resume,
                repository_root=ROOT,
            )

    def test_pointer_tamper_and_symlinks_are_rejected(self) -> None:
        generated = self._generate()
        pointer = json.loads(self.pointer.read_text(encoding="utf-8"))
        pointer["evidence_report_sha256"] = "f" * 64
        self.pointer.write_bytes(production._canonical_json(pointer))
        with self.assertRaises(production.TeacherCapacityProductionError):
            production.verify_teacher_capacity_production(
                config_path=self.config,
                pointer_path=self.pointer,
                resume_path=self.learned,
                repository_root=ROOT,
            )

        self.pointer.unlink()
        real_pointer = self.output / "real-pointer.json"
        pointer["evidence_report_sha256"] = Path(generated["evidence_path"]).stem
        body = {key: value for key, value in pointer.items() if key != "pointer_sha256"}
        pointer["pointer_sha256"] = hashlib.sha256(
            production._canonical_json(body)
        ).hexdigest()
        real_pointer.write_bytes(production._canonical_json(pointer))
        self.pointer.symlink_to(real_pointer)
        with self.assertRaisesRegex(
            production.TeacherCapacityProductionError, "symlink"
        ):
            production.verify_teacher_capacity_production(
                config_path=self.config,
                pointer_path=self.pointer,
                resume_path=self.learned,
                repository_root=ROOT,
            )

        self.pointer.unlink()
        self.pointer.write_bytes(real_pointer.read_bytes())
        evidence = Path(generated["evidence_path"])
        target = evidence.with_name("evidence-target.json")
        evidence.rename(target)
        evidence.symlink_to(target)
        with self.assertRaisesRegex(
            production.TeacherCapacityProductionError, "symlink"
        ):
            production.verify_teacher_capacity_production(
                config_path=self.config,
                pointer_path=self.pointer,
                resume_path=self.learned,
                repository_root=ROOT,
            )

    def test_verify_main_stdout_is_only_base64(self) -> None:
        generated = self._generate()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = production.main(
                [
                    "verify",
                    "--config",
                    str(self.config),
                    "--pointer",
                    str(self.pointer),
                    "--resume",
                    str(self.learned),
                ]
            )
        self.assertEqual(status, 0, stderr.getvalue())
        self.assertEqual(stdout.getvalue(), generated["receipt_b64"] + "\n")
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
