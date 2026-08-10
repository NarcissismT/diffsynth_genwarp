from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion2raft.teacher_capacity_evidence import (  # noqa: E402
    EVIDENCE_KIND,
    TeacherCapacityEvidenceError,
    build_implementation_identity,
    build_production_evidence,
    canonical_sha256,
    read_verify_production_evidence,
    validate_production_raw_report,
    write_production_evidence,
)


BIN_EDGES = [0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
BIN_COUNTS = [40, 40, 40, 40, 40, 50, 50]
IMPLEMENTATION_PATHS = (
    "src/diffusion2raft/teacher_capacity_evidence.py",
    "src/diffusion2raft/teacher_capacity_preflight.py",
    "src/diffusion2raft/teacher_capacity_policy.py",
    "src/diffusion2raft/teacher_capacity_assets.py",
    "src/diffusion2raft/teacher_capacity_receipt.py",
    "src/diffusion2raft/teacher_capacity_production.py",
    "src/diffusion2raft/external_file.py",
    "src/diffusion2raft/data.py",
    "src/diffusion2raft/geometry.py",
    "src/diffusion2raft/models/teacher_prior.py",
    "src/diffusion2raft/models/unified.py",
    "src/diffusion2raft/losses.py",
    "src/diffusion2raft/config.py",
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _metrics(sample_count: int, *, aggregate: bool) -> dict[str, Any]:
    return {
        "sample_count": sample_count,
        "eval_sample_count": sample_count,
        "eval_pixels": sample_count * 100,
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


def _valid_report(
    *, teacher_path: Path, manifest_path: Path, teacher_sha256: str
) -> dict[str, Any]:
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
    manifest_sha = _sha(manifest_path.read_bytes())
    teacher_size = teacher_path.stat().st_size
    return {
        "report_version": 1,
        "kind": "frozen_teacher_residual_capacity_preflight",
        "identities": {
            "config": {"sha256": "1" * 64},
            "checkpoint": {"sha256": "2" * 64},
            "teacher": {
                "backend": "torchscript",
                "selection": {
                    "source": "config",
                    "override_value": None,
                    "config_resolved_path": str(teacher_path),
                    "effective_path": str(teacher_path),
                },
                "checkpoint": {
                    "sha256": teacher_sha256,
                    "size_bytes": teacher_size,
                },
                "input_size": 512,
                "flow_size": 512,
                "blur_kernel": 39,
                "autocast_dtype": "float16",
                "requires_logical_cuda0": True,
            },
            "manifest": {
                "configured_path": str(manifest_path),
                "sha256": manifest_sha,
                "split": "val",
                "record_count": 300,
            },
        },
        "protocol": {
            "work_size": [512, 512],
            "selected_sample_count": 300,
            "selected_indices": list(range(300)),
            "selected_indices_sha256": "3" * 64,
            "rotation_plan_sha256": "4" * 64,
            "source_rotation": {
                "mode": "config_stratified_uniform",
                "rotations_per_sample": 1,
                "bin_edges_deg": list(BIN_EDGES),
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


class TeacherCapacityEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.teacher = self.root / "teacher.pt"
        self.teacher.write_bytes(b"frozen-teacher")
        self.teacher_sha = _sha(self.teacher.read_bytes())
        self.manifest = self.root / "val.jsonl"
        self.manifest.write_bytes(b'{"id":"sample"}\n')
        self.implementation_root = self.root / "repository"
        for index, relative in enumerate(IMPLEMENTATION_PATHS):
            path = self.implementation_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"implementation file {index}\n", encoding="utf-8")
        self.report = _valid_report(
            teacher_path=self.teacher,
            manifest_path=self.manifest,
            teacher_sha256=self.teacher_sha,
        )
        self.asset_identity = {
            "kind": "authenticated_assets",
            "aggregate_sha256": "a" * 64,
            "items": [{"name": "asset", "sha256": "b" * 64}],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self) -> dict[str, Any]:
        return build_production_evidence(
            self.report,
            configured_teacher_sha256=self.teacher_sha,
            repository_root=self.implementation_root,
            asset_identity=self.asset_identity,
        )

    def test_build_write_read_verify_pass_and_receipt_bindings(self) -> None:
        implementation = build_implementation_identity(
            repository_root=self.implementation_root
        )
        self.assertEqual(len(implementation["files"]), len(IMPLEMENTATION_PATHS))
        evidence = self._build()
        self.assertEqual(evidence["kind"], EVIDENCE_KIND)
        self.assertTrue(evidence["policy_decision"]["passed"])
        self.assertEqual(
            evidence["capacity_contract"]["asset_identity_binding"][
                "aggregate_sha256"
            ],
            self.asset_identity["aggregate_sha256"],
        )
        path = write_production_evidence(self.root / "evidence", evidence)
        self.assertEqual(path.name, f"{canonical_sha256(evidence)}.json")
        bindings = read_verify_production_evidence(
            path,
            configured_teacher_sha256=self.teacher_sha,
            repository_root=self.implementation_root,
            current_teacher_path=self.teacher,
            current_manifest_path=self.manifest,
            asset_identity=self.asset_identity,
        )
        self.assertEqual(bindings["evidence_report_sha256"], path.stem)
        self.assertEqual(bindings["teacher"]["sha256"], self.teacher_sha)
        self.assertEqual(bindings["manifest"]["record_count"], 300)
        self.assertEqual(bindings["migration_seed"], {"sha256": "2" * 64})
        self.assertEqual(
            bindings["implementation_sha256"],
            implementation["implementation_sha256"],
        )

    def test_report_tamper_is_rejected_even_with_matching_filename(self) -> None:
        evidence = self._build()
        evidence["raw_audit"]["runtime"] = {"tampered": True}
        payload = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        path = self.root / f"{_sha(payload)}.json"
        path.write_bytes(payload)
        with self.assertRaisesRegex(
            TeacherCapacityEvidenceError, "capacity contract"
        ):
            read_verify_production_evidence(
                path,
                configured_teacher_sha256=self.teacher_sha,
                repository_root=self.implementation_root,
                current_teacher_path=self.teacher,
                current_manifest_path=self.manifest,
                asset_identity=self.asset_identity,
            )

    def test_policy_failure_cannot_build_evidence(self) -> None:
        self.report["results"]["original"]["trainable_coverage"] = 0.1
        self.report["results"]["original"]["residual_target_valid_rate"] = 0.1
        with self.assertRaisesRegex(TeacherCapacityEvidenceError, "fails production"):
            self._build()

    def test_override_fast_rotation_or_missing_full_geometry_fail_closed(self) -> None:
        cases = (
            (
                ("identities", "teacher", "selection", "source"),
                "explicit_override",
            ),
            (
                ("identities", "teacher", "selection", "override_value"),
                "/tmp/override.pt",
            ),
            (
                ("identities", "teacher", "selection", "effective_path"),
                "/tmp/not-configured.pt",
            ),
            (("protocol", "external_file_sha256_enabled"), False),
            (("protocol", "source_rotation", "mode"), "explicit_cross_product"),
            (("protocol", "source_rotation", "rotations_per_sample"), 2),
            (("protocol", "source_full_geometry", "enabled"), False),
            (("protocol", "source_full_geometry", "transformations_per_sample"), 0),
            (
                (
                    "protocol",
                    "source_full_geometry",
                    "reuses_source_geometry_augment_sample_homography",
                ),
                False,
            ),
        )
        for path, replacement in cases:
            with self.subTest(path=path):
                report = copy.deepcopy(self.report)
                current = report
                for key in path[:-1]:
                    current = current[key]
                current[path[-1]] = replacement
                with self.assertRaises(TeacherCapacityEvidenceError):
                    validate_production_raw_report(
                        report, configured_teacher_sha256=self.teacher_sha
                    )

        missing = copy.deepcopy(self.report)
        del missing["protocol"]["source_full_geometry"]
        with self.assertRaises(TeacherCapacityEvidenceError):
            validate_production_raw_report(
                missing, configured_teacher_sha256=self.teacher_sha
            )

    def test_teacher_pin_incomplete_hashes_and_asset_change_fail(self) -> None:
        with self.assertRaisesRegex(TeacherCapacityEvidenceError, "teacher SHA"):
            validate_production_raw_report(
                self.report, configured_teacher_sha256="f" * 64
            )
        incomplete = copy.deepcopy(self.report)
        incomplete["identities"]["checkpoint"]["sha256"] = None
        with self.assertRaises(TeacherCapacityEvidenceError):
            validate_production_raw_report(
                incomplete, configured_teacher_sha256=self.teacher_sha
            )

        evidence = self._build()
        path = write_production_evidence(self.root / "identity", evidence)
        changed = copy.deepcopy(self.asset_identity)
        changed["kind"] = "changed"
        with self.assertRaisesRegex(TeacherCapacityEvidenceError, "asset identity"):
            read_verify_production_evidence(
                path,
                configured_teacher_sha256=self.teacher_sha,
                repository_root=self.implementation_root,
                current_teacher_path=self.teacher,
                current_manifest_path=self.manifest,
                asset_identity=changed,
            )

    def test_implementation_drift_and_external_file_drift_are_rejected(self) -> None:
        evidence = self._build()
        path = write_production_evidence(self.root / "drift", evidence)
        drifted = self.implementation_root / IMPLEMENTATION_PATHS[-1]
        drifted.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(TeacherCapacityEvidenceError, "contract"):
            read_verify_production_evidence(
                path,
                configured_teacher_sha256=self.teacher_sha,
                repository_root=self.implementation_root,
                current_teacher_path=self.teacher,
                current_manifest_path=self.manifest,
                asset_identity=self.asset_identity,
            )

        # Restore the implementation so external file verification is reached.
        drifted.write_text(
            f"implementation file {len(IMPLEMENTATION_PATHS) - 1}\n",
            encoding="utf-8",
        )
        self.manifest.write_bytes(b"changed manifest\n")
        with self.assertRaisesRegex(TeacherCapacityEvidenceError, "manifest SHA"):
            read_verify_production_evidence(
                path,
                configured_teacher_sha256=self.teacher_sha,
                repository_root=self.implementation_root,
                current_teacher_path=self.teacher,
                current_manifest_path=self.manifest,
                asset_identity=self.asset_identity,
            )

    def test_content_addressed_writer_refuses_overwrite(self) -> None:
        evidence = self._build()
        output = self.root / "exclusive"
        first = write_production_evidence(output, evidence)
        with self.assertRaises(FileExistsError):
            write_production_evidence(output, evidence)
        self.assertEqual(first.read_bytes(), json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
