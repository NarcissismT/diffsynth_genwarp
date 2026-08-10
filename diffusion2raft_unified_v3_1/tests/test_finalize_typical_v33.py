from __future__ import annotations

import hashlib
import json
import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from finalize_typical_v33 import (
    COMPARISON_CANDIDATES,
    LINE_CANDIDATES,
    FinalizationError,
    REQUIRED_OUTPUTS,
    _artifact_record,
    _exclusive_lock,
    _expected_artifact_paths,
    _quality_validation_inputs,
    _require_fresh_release_inference,
    _run_inference,
    _verify_checkpoint_artifacts,
    checkpoint_summary,
    expected_v33_extra_keys,
    validate_checkpoint_set,
    validate_comparison_report,
    validate_image_sets,
    validate_inference_output,
    validate_line_report,
)
from diffusion2raft.teacher_capacity_policy import (
    CANONICAL_POLICY_SHA256,
    POLICY_ID,
    POLICY_SCHEMA_VERSION,
)
from diffusion2raft.teacher_capacity_receipt import build_teacher_capacity_receipt


INFERENCE_CONFIG = {
    "data": {"work_size": [512, 512]},
    "model": {
        "prior_backend": "torchscript",
        "prior_torchscript_sha256": "0" * 64,
        "feature_backend": "qwen",
        "match_confidence_cap": 0.05,
    },
    "qwen": {"prompt": "rectify", "feature_layers": [-24, -12, -1]},
    "inference": {"resize_policy": "stretch", "image_decoder": "opencv"},
}


def _capacity_receipt(identity: dict[str, object]) -> dict[str, object]:
    return build_teacher_capacity_receipt(
        evidence_report_sha256="1" * 64,
        capacity_contract_sha256="2" * 64,
        capacity_config_projection_sha256="3" * 64,
        protocol_sha256="4" * 64,
        implementation_sha256="5" * 64,
        policy={
            "id": POLICY_ID,
            "version": POLICY_SCHEMA_VERSION,
            "sha256": CANONICAL_POLICY_SHA256,
        },
        migration_seed={
            "sha256": "6" * 64,
            "stage": "unified",
            "epoch_index": 19,
            "completed_epochs": 20,
        },
        teacher={
            key: identity[key]
            for key in (
                "sha256",
                "file_size",
                "input_size",
                "flow_size",
                "blur_kernel",
                "autocast_dtype",
                "requires_logical_cuda0",
            )
        },
        manifest={
            "sha256": "8" * 64,
            "split": "val",
            "record_count": 300,
        },
    )


def _summary(epoch: int, completed: int, scale: float) -> dict[str, object]:
    teacher = {
        "version": 2,
        "resolved_path": "/teacher.pt",
        "file_size": 100,
        "mtime_ns": 200,
        "sha256": "0" * 64,
        "input_size": 512,
        "flow_size": 512,
        "blur_kernel": 39,
        "autocast_dtype": "float16",
        "requires_logical_cuda0": True,
    }
    residual = {
        "version": 1,
        "origin_epoch": 20,
        "warmup_epochs": 1,
        "ramp_epochs": 6,
        "max_scale": 1.0,
        "scale": scale,
    }
    return {
        "path": f"/{epoch}.pt",
        "stage": "unified",
        "epoch_index": epoch,
        "completed_epochs": completed,
        "configured_epochs": 32,
        "inference_critical_config": copy.deepcopy(INFERENCE_CONFIG),
        "training_revision": "v3_3_teacher_anchor_residual_warmup",
        "prior_backend": "torchscript",
        "teacher_prior_identity": teacher,
        "residual_application": residual,
        "deployment_contract": {
            "version": 2,
            "teacher_prior_identity": teacher,
            "inpaint": {"enabled": False},
        },
        "best_metric": {
            "name": "line_epe",
            "mode": "min",
            "value": 4.0,
        },
        "metrics": {"line_epe": 4.0},
    }


def _write_valid_checkpoint(
    path: Path,
    *,
    teacher_path: Path,
    metric: float = 4.0,
) -> None:
    import torch

    teacher_stat = teacher_path.stat()
    teacher_sha256 = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    teacher = {
        "version": 2,
        "resolved_path": str(teacher_path.resolve()),
        "file_size": int(teacher_stat.st_size),
        "mtime_ns": int(teacher_stat.st_mtime_ns),
        "sha256": teacher_sha256,
        "input_size": 512,
        "flow_size": 512,
        "blur_kernel": 39,
        "autocast_dtype": "float16",
        "requires_logical_cuda0": True,
    }
    payload = {
        "model": {"prior._teacher_backend_marker": 1},
        "optimizer": {"state": {"present": True}},
        "stage": "unified",
        "epoch": 20,
        "config": {
            **copy.deepcopy(INFERENCE_CONFIG),
            "model": {
                **copy.deepcopy(INFERENCE_CONFIG["model"]),
                "prior_torchscript_sha256": teacher_sha256,
            },
            "train": {"epochs": 32},
        },
        "training_revision": "v3_3_teacher_anchor_residual_warmup",
        "prior_backend": "torchscript",
        "teacher_prior_identity": teacher,
        "capacity_evidence_receipt": _capacity_receipt(teacher),
        "residual_application": {
            "version": 1,
            "origin_epoch": 20,
            "warmup_epochs": 1,
            "ramp_epochs": 6,
            "max_scale": 1.0,
            "scale": 0.0,
        },
        "deployment_contract": {
            "version": 2,
            "teacher_prior_identity": teacher,
            "inpaint": {"enabled": False},
        },
        "best_metric": {"name": "line_epe", "mode": "min", "value": metric},
        "metrics": {"line_epe": metric},
    }
    torch.save(payload, path)


class FinalizeTypicalV33Test(unittest.TestCase):
    def test_production_release_rejects_skip_inference(self) -> None:
        _require_fresh_release_inference(False)
        with self.assertRaisesRegex(FinalizationError, "禁止 --skip-inference"):
            _require_fresh_release_inference(True)

    def test_run_inference_passes_complete_checkpoint_artifact(self) -> None:
        artifact = {
            "path": "/canonical/best.pt",
            "size_bytes": 123,
            "mtime_ns": 456,
            "sha256": "a" * 64,
        }
        with mock.patch("subprocess.run") as run_process:
            _run_inference(
                checkpoint=Path("/canonical/best.pt"),
                checkpoint_artifact=artifact,
                output_dir=Path("/output"),
                input_dir=Path("/input"),
                config=Path("/config.yaml"),
            )
        environment = run_process.call_args.kwargs["env"]
        self.assertEqual(environment["EXPECTED_CHECKPOINT_PATH"], artifact["path"])
        self.assertEqual(environment["EXPECTED_CHECKPOINT_SIZE_BYTES"], "123")
        self.assertEqual(environment["EXPECTED_CHECKPOINT_MTIME_NS"], "456")
        self.assertEqual(environment["EXPECTED_CHECKPOINT_SHA256"], "a" * 64)

    def test_exclusive_lock_never_follows_or_truncates_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protected = root / "protected.txt"
            protected.write_bytes(b"must-stay-intact")
            malicious_lock = root / "lock"
            malicious_lock.symlink_to(protected)

            with self.assertRaisesRegex(FinalizationError, "安全打开"):
                with _exclusive_lock(malicious_lock):
                    self.fail("symlink lock must never be acquired")
            self.assertEqual(protected.read_bytes(), b"must-stay-intact")

            malicious_lock.unlink()
            with _exclusive_lock(malicious_lock):
                self.assertTrue(malicious_lock.is_file())

    def test_checkpoint_artifacts_are_bound_by_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            summaries = {}
            for name in ("anchor", "best", "latest"):
                path = root / f"{name}.pt"
                path.write_bytes(name.encode("ascii"))
                paths[name] = path
                summaries[name] = _artifact_record(path)

            _verify_checkpoint_artifacts(paths, summaries)

            target = paths["best"]
            original_stat = target.stat()
            target.write_bytes(b"BEST")
            os.utime(
                target,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            with self.assertRaisesRegex(FinalizationError, "best checkpoint"):
                _verify_checkpoint_artifacts(paths, summaries)

    def test_checkpoint_summary_rejects_symlink_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher.pt"
            teacher.write_bytes(b"teacher")
            checkpoint = root / "checkpoint.pt"
            _write_valid_checkpoint(checkpoint, teacher_path=teacher)
            symlink = root / "checkpoint-link.pt"
            symlink.symlink_to(checkpoint)

            with self.assertRaisesRegex(FinalizationError, "安全打开"):
                checkpoint_summary(symlink)

    def test_checkpoint_summary_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fifo = Path(temporary) / "checkpoint.pt"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(FinalizationError, "不是普通文件"):
                checkpoint_summary(fifo)

    def test_checkpoint_summary_rewinds_same_fd_for_legacy_torch(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher.pt"
            teacher.write_bytes(b"teacher")
            checkpoint = root / "checkpoint.pt"
            _write_valid_checkpoint(checkpoint, teacher_path=teacher, metric=3.5)
            expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            real_torch_load = torch.load
            positions: list[int] = []

            def emulate_legacy_torch(
                source: object, *args: object, **kwargs: object
            ):
                if "weights_only" in kwargs:
                    source.read(13)  # type: ignore[attr-defined]
                    raise TypeError("legacy torch has no weights_only")
                positions.append(source.tell())  # type: ignore[attr-defined]
                return real_torch_load(source, *args, **kwargs)

            with mock.patch("torch.load", side_effect=emulate_legacy_torch):
                summary = checkpoint_summary(checkpoint)
            self.assertEqual(positions, [0])
            self.assertEqual(summary["metrics"], {"line_epe": 3.5})
            self.assertEqual(summary["sha256"], expected_sha256)

    def test_checkpoint_summary_rejects_path_replacement_during_load(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher.pt"
            teacher.write_bytes(b"teacher")
            checkpoint = root / "checkpoint.pt"
            replacement = root / "replacement.pt"
            _write_valid_checkpoint(checkpoint, teacher_path=teacher, metric=4.0)
            _write_valid_checkpoint(replacement, teacher_path=teacher, metric=9.0)

            real_torch_load = torch.load
            loaded_from_descriptor: list[bool] = []

            def replace_after_load(source: object, *args: object, **kwargs: object):
                payload = real_torch_load(source, *args, **kwargs)
                loaded_from_descriptor.append(
                    hasattr(source, "fileno") and not isinstance(source, (str, Path))
                )
                os.replace(replacement, checkpoint)
                return payload

            with mock.patch("torch.load", side_effect=replace_after_load):
                with self.assertRaisesRegex(FinalizationError, "路径被替换"):
                    checkpoint_summary(checkpoint)
            self.assertEqual(loaded_from_descriptor, [True])

    def test_quality_gate_adapter_uses_fixed_best_and_anchor_contract(self) -> None:
        anchor = _summary(20, 21, 0.0)
        best = _summary(24, 25, 0.5)
        best["metrics"].update(
            {
                "epe": 4.5,
                "epe_gain": 1.0,
                "final_win_rate": 0.7,
                "fold_rate": 0.0001,
                "jacobian_p01": 0.2,
                "line_straightness_error": 0.05,
            }
        )
        adapted = _quality_validation_inputs({"anchor": anchor, "best": best})
        self.assertEqual(set(adapted), {"v33_anchor", "v33_best"})
        self.assertEqual(adapted["v33_anchor"]["residual_scale"], 0.0)
        self.assertEqual(adapted["v33_best"]["residual_scale"], 0.5)
        self.assertEqual(adapted["v33_best"]["feature_backend"], "qwen")
        self.assertEqual(adapted["v33_best"]["metrics"]["epe"], 4.5)

    def test_checkpoint_relationships_require_exact_anchor_and_completed_latest(self) -> None:
        anchor = _summary(20, 21, 0.0)
        best = _summary(27, 28, 1.0)
        latest = _summary(31, 32, 1.0)
        validate_checkpoint_set(
            anchor,
            best,
            latest,
            expected_total_epochs=32,
            current_inference_config=INFERENCE_CONFIG,
        )

        bad_anchor = _summary(20, 21, 0.1)
        with self.assertRaisesRegex(FinalizationError, "anchor residual scale"):
            validate_checkpoint_set(
                bad_anchor,
                best,
                latest,
                expected_total_epochs=32,
                current_inference_config=INFERENCE_CONFIG,
            )

        incomplete = _summary(30, 31, 1.0)
        with self.assertRaisesRegex(FinalizationError, "严格等于"):
            validate_checkpoint_set(
                anchor,
                best,
                incomplete,
                expected_total_epochs=32,
                current_inference_config=INFERENCE_CONFIG,
            )

        overcomplete = _summary(32, 33, 1.0)
        with self.assertRaisesRegex(FinalizationError, "严格等于"):
            validate_checkpoint_set(
                anchor,
                best,
                overcomplete,
                expected_total_epochs=32,
                current_inference_config=INFERENCE_CONFIG,
            )

        mismatched_best = _summary(27, 28, 1.0)
        mismatched_best["best_metric"] = {
            "name": "line_epe",
            "mode": "min",
            "value": 3.5,
        }
        mismatched_best["metrics"] = {"line_epe": 3.5}
        with self.assertRaisesRegex(FinalizationError, "最终 best_metric"):
            validate_checkpoint_set(
                anchor,
                mismatched_best,
                latest,
                expected_total_epochs=32,
                current_inference_config=INFERENCE_CONFIG,
            )

        changed_config = copy.deepcopy(INFERENCE_CONFIG)
        changed_config["model"]["match_confidence_cap"] = 0.5
        with self.assertRaisesRegex(FinalizationError, "inference-critical"):
            validate_checkpoint_set(
                anchor,
                best,
                latest,
                expected_total_epochs=32,
                current_inference_config=changed_config,
            )

    def test_image_sets_require_exact_count_lowercase_jpg_and_stems(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / name for name in ("source", "first", "second")]
            for directory in directories:
                directory.mkdir()
                for name in ("MixedCase_A.jpg", "page_b.jpg"):
                    cv2.imwrite(str(directory / name), np.zeros((8, 9, 3), np.uint8))
            indexes = validate_image_sets(*directories, expected_count=2)
            self.assertEqual(set(indexes["warped"]), {"mixedcase_a", "page_b"})

            cv2.imwrite(
                str(directories[1] / "page_b.jpg"),
                np.zeros((7, 9, 3), np.uint8),
            )
            with self.assertRaisesRegex(FinalizationError, "尺寸与 source 不一致"):
                validate_image_sets(*directories, expected_count=2)
            cv2.imwrite(
                str(directories[1] / "page_b.jpg"),
                np.zeros((8, 9, 3), np.uint8),
            )

            (directories[2] / "page_b.jpg").rename(directories[2] / "other.jpg")
            with self.assertRaisesRegex(FinalizationError, "basename"):
                validate_image_sets(*directories, expected_count=2)

    def test_line_and_comparison_reports_fail_closed(self) -> None:
        candidates: dict[str, object] = {}
        for name in LINE_CANDIDATES:
            candidates[name] = {
                "summary": {
                    "indexed_images": 1,
                    "paired_images": 1,
                    "evaluated_images": 1,
                    "missing_images": 0,
                    "missing_masks": 0,
                    "no_line_images": 0,
                    "image_mean_orientation_error_deg_length_weighted": 2.0,
                },
                "per_image": [{"basename": "Mixed", "status": "ok"}],
            }
        weighted = validate_line_report(
            {"candidates": candidates}, expected_count=1
        )
        self.assertEqual(weighted["v33_best_valid"], 2.0)

        source_keys = ["Mixed"]
        extras = sorted(expected_v33_extra_keys(source_keys))
        comparison_candidates = {
            "target_first": {
                "matched_count": 1,
                "missing_count": 0,
                "extra_keys": [],
            },
            "target_second": {
                "matched_count": 1,
                "missing_count": 0,
                "extra_keys": [],
            },
            "v33_anchor": {
                "matched_count": 1,
                "missing_count": 0,
                "extra_keys": extras,
            },
            "v33_best": {
                "matched_count": 1,
                "missing_count": 0,
                "extra_keys": extras,
            },
        }
        comparison = {
            "candidate_order": list(COMPARISON_CANDIDATES),
            "pairing": {"source_count": 1, "complete_row_count": 1},
            "candidates": comparison_candidates,
        }
        validate_comparison_report(
            comparison, source_keys=source_keys, expected_count=1
        )
        comparison_candidates["v33_best"]["missing_count"] = 1
        with self.assertRaisesRegex(FinalizationError, "配对不完整"):
            validate_comparison_report(
                comparison, source_keys=source_keys, expected_count=1
            )

    def test_inference_report_validates_artifacts_masks_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            output_dir.mkdir()
            source = source_dir / "Mixed_A.jpg"
            cv2.imwrite(str(source), np.full((8, 10, 3), 127, np.uint8))
            checkpoint = root / "anchor.pt"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_stat = checkpoint.stat()
            checkpoint_artifact = {
                "path": str(checkpoint.resolve()),
                "size_bytes": int(checkpoint_stat.st_size),
                "mtime_ns": int(checkpoint_stat.st_mtime_ns),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            teacher = {"version": 1, "path": "/teacher.pt"}
            residual = {"version": 1, "scale": 0.0}
            deployment = {"version": 1, "teacher_prior_identity": teacher}

            paths: dict[str, str] = {}
            expected_paths = _expected_artifact_paths(output_dir, source.stem)
            for name in REQUIRED_OUTPUTS:
                path = expected_paths[name]
                if name in {"flow", "prior_flow", "residual_flow"}:
                    np.save(path, np.zeros((8, 10, 2), np.float32))
                elif name == "metadata":
                    path.write_text(
                        json.dumps(
                            {
                                "checkpoint": str(checkpoint.resolve()),
                                "checkpoint_artifact": checkpoint_artifact,
                                "warped": str(source.resolve()),
                                "final_image_inpainted": True,
                                "teacher_prior_identity": teacher,
                                "residual_application": residual,
                                "deployment_contract": deployment,
                                "valid_fraction": 1.0,
                                "inpaint_fraction": 1.0 / 80.0,
                                "evaluation_valid_fraction": 79.0 / 80.0,
                                "fold_rate": 0.0,
                                "jacobian_p01": 1.0,
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    image = np.full((8, 10, 3), 255, np.uint8)
                    if name == "inpaint_mask":
                        image.fill(0)
                        image[0, 0, :] = 255
                    elif name == "evaluation_valid":
                        image[0, 0, :] = 0
                    cv2.imwrite(str(path), image)
                paths[name] = str(path.resolve())

            report = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_artifact": checkpoint_artifact,
                "source_dir": str(source_dir.resolve()),
                "resize_policy": "stretch",
                "padding_mode": "replicate",
                "image_decoder": "opencv",
                "resize_interpolation": "opencv_baseline",
                "input_count": 1,
                "success_count": 1,
                "error_count": 0,
                "successes": [
                    {"input": str(source.resolve()), "outputs": paths}
                ],
                "errors": [],
            }
            (output_dir / "inference_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            metadata = {
                **checkpoint_artifact,
                "teacher_prior_identity": teacher,
                "residual_application": residual,
                "deployment_contract": deployment,
            }
            validated = validate_inference_output(
                output_dir,
                checkpoint=checkpoint,
                checkpoint_metadata=metadata,
                source_index={"mixed_a": source},
                expected_count=1,
            )
            self.assertEqual(validated["success_count"], 1)
            self.assertAlmostEqual(
                validated["_finalizer_validation"][
                    "mean_evaluation_valid_fraction"
                ],
                79.0 / 80.0,
            )
            self.assertEqual(
                validated["_finalizer_validation"]["per_image"]["Mixed_A"],
                {
                    "valid_fraction": 1.0,
                    "inpaint_fraction": 1.0 / 80.0,
                    "evaluation_valid_fraction": 79.0 / 80.0,
                    "fold_rate": 0.0,
                    "jacobian_p01": 1.0,
                },
            )

            report_path = output_dir / "inference_report.json"
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            report_payload["checkpoint_artifact"]["sha256"] = "0" * 64
            report_path.write_text(json.dumps(report_payload), encoding="utf-8")
            with self.assertRaisesRegex(
                FinalizationError, "inference report checkpoint_artifact"
            ):
                validate_inference_output(
                    output_dir,
                    checkpoint=checkpoint,
                    checkpoint_metadata=metadata,
                    source_index={"mixed_a": source},
                    expected_count=1,
                )
            report_path.write_text(json.dumps(report), encoding="utf-8")

            metadata_path = Path(paths["metadata"])
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_payload["checkpoint_artifact"]["mtime_ns"] += 1
            metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")
            with self.assertRaisesRegex(
                FinalizationError, "metadata checkpoint_artifact"
            ):
                validate_inference_output(
                    output_dir,
                    checkpoint=checkpoint,
                    checkpoint_metadata=metadata,
                    source_index={"mixed_a": source},
                    expected_count=1,
                )
            metadata_payload["checkpoint_artifact"] = checkpoint_artifact
            metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_payload["fold_rate"] = 0.5
            metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationError, "持久化 flow 不一致"):
                validate_inference_output(
                    output_dir,
                    checkpoint=checkpoint,
                    checkpoint_metadata=metadata,
                    source_index={"mixed_a": source},
                    expected_count=1,
                )
            metadata_payload["fold_rate"] = 0.0
            metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

            corrupt_valid = np.full((8, 10, 3), 255, np.uint8)
            corrupt_valid[1, 1, 1] = 0
            cv2.imwrite(paths["valid"], corrupt_valid)
            with self.assertRaisesRegex(FinalizationError, "三通道必须完全相同"):
                validate_inference_output(
                    output_dir,
                    checkpoint=checkpoint,
                    checkpoint_metadata=metadata,
                    source_index={"mixed_a": source},
                    expected_count=1,
                )
            cv2.imwrite(
                paths["valid"], np.full((8, 10, 3), 255, np.uint8)
            )

            nonbinary_valid = np.full((8, 10, 3), 255, np.uint8)
            nonbinary_valid[1, 1, :] = 128
            cv2.imwrite(paths["valid"], nonbinary_valid)
            with self.assertRaisesRegex(FinalizationError, "0/255 二值 mask"):
                validate_inference_output(
                    output_dir,
                    checkpoint=checkpoint,
                    checkpoint_metadata=metadata,
                    source_index={"mixed_a": source},
                    expected_count=1,
                )
            cv2.imwrite(
                paths["valid"], np.full((8, 10, 3), 255, np.uint8)
            )

            cv2.imwrite(
                paths["evaluation_valid"],
                np.full((8, 10, 3), 255, np.uint8),
            )
            with self.assertRaisesRegex(FinalizationError, "evaluation_valid"):
                validate_inference_output(
                    output_dir,
                    checkpoint=checkpoint,
                    checkpoint_metadata=metadata,
                    source_index={"mixed_a": source},
                    expected_count=1,
                )

    def test_shell_wrapper_selects_a_python_with_runtime_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / name for name in ("source", "first", "second")]
            for directory in directories:
                directory.mkdir()
                cv2.imwrite(
                    str(directory / "page.jpg"),
                    np.zeros((4, 5, 3), np.uint8),
                )
            environment = os.environ.copy()
            environment.pop("PYTHON", None)
            environment.update(
                {
                    "PREFLIGHT_ONLY": "1",
                    "EXPECTED_COUNT": "1",
                    "INPUT_DIR": str(directories[0]),
                    "TARGET_FIRST_DIR": str(directories[1]),
                    "TARGET_SECOND_DIR": str(directories[2]),
                    "RUN_ROOT": str(root / "missing_run"),
                }
            )
            completed = subprocess.run(
                ["bash", "scripts/finalize_typical_v33_all40.sh"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 65, completed.stdout)
            self.assertIn("checkpoint 校验失败", completed.stdout)
            self.assertNotIn("ModuleNotFoundError", completed.stdout)


if __name__ == "__main__":
    unittest.main()
