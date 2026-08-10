from __future__ import annotations

import hashlib
import tempfile
import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:

    class _ZeroPrior(torch.nn.Module):
        backend_name = "learned"

        def forward(self, image):
            return image.new_zeros((image.shape[0], 2, *image.shape[-2:]))


    class _ZeroFlowTeacher(torch.nn.Module):
        def forward(self, image1, image2):
            del image2
            return [image1[:, :2] * 0.0]


    class _FixedElevenFlowTeacher(torch.nn.Module):
        def forward(self, image1, image2):
            del image2
            return [
                torch.nn.functional.interpolate(
                    image1[:, :2],
                    size=(11, 11),
                    mode="bilinear",
                    align_corners=False,
                )
            ]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TeacherResidualWarmupTest(unittest.TestCase):
    @staticmethod
    def _lite_model():
        from diffusion2raft.models.unified import (
            LiteEditFeatureEncoder,
            UnifiedDocumentRectifier,
        )

        return UnifiedDocumentRectifier(
            _ZeroPrior(),
            LiteEditFeatureEncoder(8),
            feature_channels=8,
            cnn_channels=8,
            hidden_channels=8,
            correlation_radius=1,
            iterations=1,
            max_residual_px=8.0,
            feature_dropout_prob=0.0,
        )

    @staticmethod
    def _save_teacher(path: Path, size: int) -> None:
        example = torch.rand(1, 3, size, size)
        traced = torch.jit.trace(
            _ZeroFlowTeacher().eval(), (example, example), strict=False
        )
        torch.jit.save(traced, str(path))

    @staticmethod
    def _model_config(teacher_path: Path | None = None, size: int = 16):
        config = {
            "feature_backend": "lite",
            "feature_channels": 8,
            "feature_stride": 8,
            "cnn_feature_channels": 8,
            "refiner_hidden_channels": 8,
            "refiner_iterations": 1,
            "correlation_radius": 1,
            "feature_dropout_prob": 0.0,
            "prior_base_channels": 4,
            "prior_control_stride": 8,
        }
        if teacher_path is not None:
            config.update(
                prior_backend="torchscript",
                prior_torchscript_path=str(teacher_path),
                prior_torchscript_sha256=hashlib.sha256(
                    teacher_path.read_bytes()
                ).hexdigest(),
                prior_torchscript_size=size,
                prior_torchscript_blur_kernel=1,
                prior_torchscript_autocast_dtype="float16",
            )
        return config

    @staticmethod
    def _capacity_receipt(
        teacher_path: Path,
        *,
        migration_sha256: str = "6" * 64,
        evidence_sha256: str = "1" * 64,
    ):
        from diffusion2raft.teacher_capacity_policy import (
            CANONICAL_POLICY_SHA256,
            POLICY_ID,
            POLICY_SCHEMA_VERSION,
        )
        from diffusion2raft.teacher_capacity_receipt import (
            build_teacher_capacity_receipt,
        )

        return build_teacher_capacity_receipt(
            evidence_report_sha256=evidence_sha256,
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
                "sha256": migration_sha256,
                "stage": "unified",
                "epoch_index": 19,
                "completed_epochs": 20,
            },
            teacher={
                "sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                "file_size": teacher_path.stat().st_size,
                "input_size": 16,
                "flow_size": 16,
                "blur_kernel": 1,
                "autocast_dtype": "float16",
                "requires_logical_cuda0": False,
            },
            manifest={
                "sha256": "8" * 64,
                "split": "val",
                "record_count": 300,
            },
        )

    def test_alpha_zero_is_exact_prior_but_raw_auxiliary_has_gradient(self) -> None:
        from diffusion2raft.geometry import flow_valid_mask
        from diffusion2raft.losses import RectificationLoss

        torch.manual_seed(7)
        model = self._lite_model().train()
        state_keys = tuple(model.state_dict())
        with torch.no_grad():
            model.refiner.flow_head[-1].bias.copy_(torch.tensor([0.30, -0.20]))
        model.set_residual_application_scale(0.0)
        self.assertEqual(tuple(model.state_dict()), state_keys)

        warped = torch.rand(1, 3, 32, 32)
        outputs = model(warped, stage="unified")
        self.assertGreater(
            float(outputs["raw_residuals"][-1].detach().abs().max()), 0.0
        )
        self.assertEqual(
            float(outputs["residuals"][-1].detach().abs().max()), 0.0
        )
        self.assertTrue(torch.equal(outputs["final_flow"], outputs["prior_flow"]))
        self.assertTrue(torch.equal(outputs["final_valid"], outputs["prior_valid"]))

        target_flow = torch.zeros_like(outputs["prior_flow"])
        target_flow[:, 0] = 1.0
        valid = flow_valid_mask(target_flow, target_flow.shape[-2:])
        criterion = RectificationLoss(
            {
                "flow": 0.0,
                "prior_flow": 0.0,
                "prior_flow_unified": 0.0,
                "reconstruction": 0.0,
                "gradient": 0.0,
                "structure_flow": 0.0,
                "line_reconstruction": 0.0,
                "flow_gradient": 0.0,
                "curvature": 0.0,
                "line_straightness": 0.0,
                "bending": 0.0,
                "anti_fold": 0.0,
                "residual": 0.0,
                "residual_flow": 1.0,
                "qwen_match": 0.0,
                "confidence": 0.0,
                "max_residual_target": 8.0,
                "max_residual_consistency": 1.0,
            }
        )
        losses = criterion(
            outputs,
            {
                "warped": warped,
                "target": warped.clone(),
                "flow": target_flow,
                "valid": valid,
            },
        )
        self.assertGreater(float(losses["raw_residual_flow"]), 0.0)
        losses["total"].backward()
        gradient = model.refiner.flow_head[-1].bias.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_relative_schedule_boundaries(self) -> None:
        from diffusion2raft.train import (
            _scheduled_correlation_temperature,
            _scheduled_residual_application_scale,
        )

        values = [
            _scheduled_residual_application_scale(
                epoch,
                origin_epoch=10,
                warmup_epochs=1,
                ramp_epochs=6,
                max_scale=1.0,
            )
            for epoch in range(10, 18)
        ]
        expected = [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0, 1.0]
        for actual, target in zip(values, expected):
            self.assertAlmostEqual(actual, target)
        for value in (float("nan"), float("inf"), -float("inf"), 0.0):
            with self.subTest(correlation_temperature=value):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    _scheduled_correlation_temperature(
                        {"model": {"correlation_temperature": value}}, 0
                    )

    def test_capacity_receipt_environment_is_teacher_only(self) -> None:
        from diffusion2raft.train import (
            TEACHER_CAPACITY_RECEIPT_ENV,
            _teacher_capacity_receipt_from_environment,
        )

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(
                _teacher_capacity_receipt_from_environment(
                    {"model": {}}, stage="unified"
                )
            )
            with self.assertRaisesRegex(RuntimeError, "requires"):
                _teacher_capacity_receipt_from_environment(
                    {"model": {"prior_backend": "torchscript"}},
                    stage="unified",
                )
            with self.assertRaisesRegex(RuntimeError, "only accepted"):
                _teacher_capacity_receipt_from_environment(
                    {"model": {}, "capacity_evidence_receipt": None},
                    stage="unified",
                )
        with patch.dict(
            os.environ, {TEACHER_CAPACITY_RECEIPT_ENV: "stray"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "only accepted"):
                _teacher_capacity_receipt_from_environment(
                    {"model": {}}, stage="unified"
                )

    def test_best_guard_is_opt_in_and_anchor_is_always_admitted(self) -> None:
        from diffusion2raft.train import _best_candidate_guard

        unsafe = {"epe_gain": -0.1, "fold_rate": 0.01, "jacobian_p01": 0.001}
        self.assertEqual(
            _best_candidate_guard({}, unsafe, is_anchor=False), (True, None)
        )
        guarded = {
            "best_guard": {
                "min_epe_gain": 0.0,
                "max_fold_rate": 0.001,
                "min_jacobian_p01": 0.01,
            }
        }
        self.assertEqual(
            _best_candidate_guard(guarded, unsafe, is_anchor=True), (True, None)
        )
        passed, reason = _best_candidate_guard(
            guarded, unsafe, is_anchor=False
        )
        self.assertFalse(passed)
        self.assertIn("epe_gain", reason)
        self.assertIn("fold_rate", reason)
        self.assertIn("jacobian_p01", reason)
        safe = {"epe_gain": 0.0, "fold_rate": 0.001, "jacobian_p01": 0.01}
        self.assertEqual(
            _best_candidate_guard(guarded, safe, is_anchor=False), (True, None)
        )

        strict_guarded = {
            "best_guard": {
                "max_epe_exclusive": 5.7501,
                "min_epe_gain_exclusive": 0.0,
                "min_final_win_rate_exclusive": 0.5,
                "max_fold_rate_exclusive": 0.0004,
                "min_line_epe_gain_exclusive": 0.0,
                "min_line_straightness_gain_exclusive": 0.0,
                "min_jacobian_p01": 0.01,
            }
        }
        boundary = {
            "epe": 5.7501,
            "epe_gain": 0.0,
            "final_win_rate": 0.5,
            "fold_rate": 0.0004,
            "line_epe_gain": 0.0,
            "line_straightness_gain": 0.0,
            "jacobian_p01": 0.01,
        }
        passed, reason = _best_candidate_guard(
            strict_guarded, boundary, is_anchor=False
        )
        self.assertFalse(passed)
        for metric in (
            "epe",
            "epe_gain",
            "final_win_rate",
            "fold_rate",
            "line_epe_gain",
            "line_straightness_gain",
        ):
            self.assertIn(metric, reason)
        better = {
            **boundary,
            "epe": 5.7500,
            "epe_gain": 1e-6,
            "final_win_rate": 0.500001,
            "fold_rate": 0.000399,
            "line_epe_gain": 1e-6,
            "line_straightness_gain": 1e-6,
        }
        self.assertEqual(
            _best_candidate_guard(strict_guarded, better, is_anchor=False),
            (True, None),
        )
        self.assertEqual(
            _best_candidate_guard(strict_guarded, boundary, is_anchor=True),
            (True, None),
        )
        with self.assertRaisesRegex(ValueError, "unknown train.best_guard"):
            _best_candidate_guard(
                {"best_guard": {"unknown": 1.0}},
                {},
                is_anchor=True,
            )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _best_candidate_guard(
                {"best_guard": {"max_epe_exclusive": float("nan")}},
                {},
                is_anchor=True,
            )

    def test_teacher_input_and_flow_canvas_sizes_are_independent(self) -> None:
        from diffusion2raft.models.teacher_prior import TorchScriptGeometryPrior

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed-flow.pt"
            example = torch.rand(1, 3, 17, 17)
            traced = torch.jit.trace(
                _FixedElevenFlowTeacher().eval(),
                (example, example),
                strict=False,
            )
            torch.jit.save(traced, str(path))
            prior = TorchScriptGeometryPrior(
                path,
                device="cpu",
                input_size=17,
                flow_size=11,
                blur_kernel=1,
            )
            output = prior(torch.rand(1, 3, 7, 9))
            self.assertEqual(tuple(output.shape), (1, 2, 7, 9))
            self.assertEqual(prior.input_size, 17)
            self.assertEqual(prior.flow_size, 11)

    def test_migration_resets_best_and_teacher_resume_restores_contract(self) -> None:
        from diffusion2raft.infer import RectificationSession
        from diffusion2raft.deployment import build_teacher_deployment_contract
        from diffusion2raft.models.unified import build_unified_rectifier
        from diffusion2raft.train import (
            TEACHER_WARMUP_REVISION,
            _load_checkpoint,
            _set_residual_application_schedule,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_path = root / "teacher.pt"
            checkpoint_path = root / "checkpoint.pt"
            self._save_teacher(teacher_path, 16)
            learned_config = self._model_config()
            teacher_config = self._model_config(teacher_path)
            learned = build_unified_rectifier(learned_config, {}, device="cpu")
            anchored = build_unified_rectifier(teacher_config, {}, device="cpu")
            legacy = {
                "model": learned.state_dict(),
                "stage": "unified",
                "epoch": 4,
                "config": {"model": learned_config},
                "prior_backend": "learned",
                "prior_state_keys": sorted(
                    key for key in learned.state_dict() if key.startswith("prior.")
                ),
                "best_metric": {"name": "line_epe", "mode": "min", "value": 0.1},
            }
            torch.save(legacy, checkpoint_path)
            start_epoch, best_name, best_value = _load_checkpoint(
                anchored, checkpoint_path, target_stage="unified"
            )
            self.assertEqual(start_epoch, 5)
            self.assertIsNone(best_name)
            self.assertIsNone(best_value)
            self.assertEqual(anchored.residual_application_scale, 0.0)
            self.assertEqual(anchored._residual_schedule_origin_epoch, 5)

            schedule = {
                "version": 1,
                "origin_epoch": 5,
                "warmup_epochs": 1,
                "ramp_epochs": 8,
                "max_scale": 1.0,
            }
            _set_residual_application_schedule(anchored, schedule, scale=0.375)
            session_config = {
                "device": "cpu",
                "data": {"work_size": [16, 16]},
                "model": teacher_config,
                "qwen": {},
            }
            payload = {
                "model": anchored.state_dict(),
                "stage": "unified",
                # age=3 with a one-epoch hold and eight-step ramp -> 3/8.
                "epoch": 8,
                "prior_backend": "torchscript",
                "training_revision": TEACHER_WARMUP_REVISION,
                "teacher_prior_identity": anchored.teacher_prior_identity,
                "residual_application": {**schedule, "scale": 0.375},
                "best_metric": {"name": "line_epe", "mode": "min", "value": 0.05},
                "config": session_config,
                "deployment_contract": build_teacher_deployment_contract(
                    session_config,
                    teacher_identity=anchored.teacher_prior_identity,
                ),
            }
            torch.save(payload, checkpoint_path)

            resumed = build_unified_rectifier(teacher_config, {}, device="cpu")
            resumed_start, resumed_best, resumed_value = _load_checkpoint(
                resumed, checkpoint_path, target_stage="unified"
            )
            self.assertEqual(resumed_start, 9)
            self.assertEqual(resumed_best, "line_epe")
            self.assertEqual(resumed_value, 0.05)
            self.assertEqual(resumed.residual_application_scale, 0.375)
            self.assertEqual(resumed._residual_application_schedule, schedule)

            session = RectificationSession(
                session_config,
                checkpoint_path,
                stage="unified",
            )
            self.assertEqual(session.model.residual_application_scale, 0.375)

            mismatched_deployment_config = {
                **session_config,
                "inference": {"image_decoder": "opencv"},
            }
            with self.assertRaisesRegex(
                RuntimeError, "deployment contract mismatch"
            ):
                RectificationSession(
                    mismatched_deployment_config,
                    checkpoint_path,
                    stage="unified",
                )

            missing_deployment = dict(payload)
            missing_deployment.pop("deployment_contract")
            torch.save(missing_deployment, checkpoint_path)
            with self.assertRaisesRegex(
                RuntimeError, "missing deployment_contract"
            ):
                RectificationSession(
                    session_config,
                    checkpoint_path,
                    stage="unified",
                )

            missing_correlation_config = dict(payload)
            missing_correlation_config.pop("config")
            torch.save(missing_correlation_config, checkpoint_path)
            with self.assertRaisesRegex(
                RuntimeError, "requires config and an integer epoch"
            ):
                RectificationSession(
                    session_config,
                    checkpoint_path,
                    stage="unified",
                )

            invalid_correlation_config = {
                **payload,
                "config": {
                    "model": {
                        **teacher_config,
                        "correlation_temperature": 0.0,
                    }
                },
            }
            torch.save(invalid_correlation_config, checkpoint_path)
            with self.assertRaisesRegex(
                RuntimeError, "invalid correlation-temperature schedule"
            ):
                RectificationSession(
                    session_config,
                    checkpoint_path,
                    stage="unified",
                )

            bad_scale_payload = {
                **payload,
                "residual_application": {
                    **payload["residual_application"],
                    "scale": 0.5,
                },
            }
            torch.save(bad_scale_payload, checkpoint_path)
            bad_scale_model = build_unified_rectifier(
                teacher_config, {}, device="cpu"
            )
            with self.assertRaisesRegex(
                RuntimeError, "scale does not match its epoch/schedule"
            ):
                _load_checkpoint(
                    bad_scale_model, checkpoint_path, target_stage="unified"
                )

            bad_epoch_payload = {**payload, "epoch": 8.0}
            torch.save(bad_epoch_payload, checkpoint_path)
            bad_epoch_model = build_unified_rectifier(
                teacher_config, {}, device="cpu"
            )
            with self.assertRaisesRegex(
                RuntimeError, "checkpoint epoch must be an integer"
            ):
                _load_checkpoint(
                    bad_epoch_model, checkpoint_path, target_stage="unified"
                )

            for field, invalid in (
                ("version", 1.9),
                ("origin_epoch", 5.9),
                ("warmup_epochs", True),
                ("ramp_epochs", 8.9),
                ("max_scale", "1.0"),
                ("scale", False),
            ):
                with self.subTest(strict_residual_field=field):
                    bad_metadata_payload = {
                        **payload,
                        "residual_application": {
                            **payload["residual_application"],
                            field: invalid,
                        },
                    }
                    torch.save(bad_metadata_payload, checkpoint_path)
                    bad_metadata_model = build_unified_rectifier(
                        teacher_config, {}, device="cpu"
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "residual_application (integer|numeric) field",
                    ):
                        _load_checkpoint(
                            bad_metadata_model,
                            checkpoint_path,
                            target_stage="unified",
                        )

            # Inference must never infer a safe teacher deployment merely from
            # the marker.  Even an older/missing training_revision still needs
            # the external identity and residual application contract.
            metadata_free = dict(payload)
            metadata_free.pop("training_revision")
            metadata_free.pop("teacher_prior_identity")
            metadata_free.pop("residual_application")
            torch.save(metadata_free, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "missing teacher_prior_identity"):
                RectificationSession(
                    session_config,
                    checkpoint_path,
                    stage="unified",
                )

            bad_payload = dict(payload)
            bad_identity = dict(payload["teacher_prior_identity"])
            bad_identity["mtime_ns"] += 1
            bad_payload["teacher_prior_identity"] = bad_identity
            torch.save(bad_payload, checkpoint_path)
            mismatch = build_unified_rectifier(teacher_config, {}, device="cpu")
            with self.assertRaisesRegex(RuntimeError, "teacher identity mismatch"):
                _load_checkpoint(mismatch, checkpoint_path, target_stage="unified")

    def test_capacity_receipt_enforces_migration_and_teacher_resume(self) -> None:
        from diffusion2raft.external_file import stable_external_file_identity
        from diffusion2raft.models.unified import build_unified_rectifier
        from diffusion2raft.train import (
            _load_checkpoint,
            _set_residual_application_schedule,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_path = root / "teacher.pt"
            learned_path = root / "learned.pt"
            teacher_checkpoint_path = root / "teacher-checkpoint.pt"
            self._save_teacher(teacher_path, 16)
            learned_config = self._model_config()
            teacher_config = self._model_config(teacher_path)
            learned = build_unified_rectifier(learned_config, {}, device="cpu")
            learned_payload = {
                "model": learned.state_dict(),
                "stage": "unified",
                "epoch": 19,
                "config": {"model": learned_config},
                "prior_backend": "learned",
                "prior_state_keys": sorted(
                    key
                    for key in learned.state_dict()
                    if key.startswith("prior.")
                ),
            }
            torch.save(learned_payload, learned_path)
            learned_identity = stable_external_file_identity(learned_path)
            receipt = self._capacity_receipt(
                teacher_path,
                migration_sha256=learned_identity["sha256"],
            )

            missing_receipt_target = build_unified_rectifier(
                teacher_config, {}, device="cpu"
            )
            with self.assertRaisesRegex(RuntimeError, "requires a capacity"):
                _load_checkpoint(
                    missing_receipt_target,
                    learned_path,
                    target_stage="unified",
                    require_capacity_evidence_receipt=True,
                    resume_file_identity=learned_identity,
                )

            wrong_seed_receipt = self._capacity_receipt(
                teacher_path, migration_sha256="a" * 64
            )
            wrong_seed_target = build_unified_rectifier(
                teacher_config, {}, device="cpu"
            )
            with self.assertRaisesRegex(RuntimeError, "migration_seed.*sha256"):
                _load_checkpoint(
                    wrong_seed_target,
                    learned_path,
                    target_stage="unified",
                    capacity_evidence_receipt=wrong_seed_receipt,
                    resume_file_identity=learned_identity,
                    require_capacity_evidence_receipt=True,
                )

            anchored = build_unified_rectifier(
                teacher_config, {}, device="cpu"
            )
            start_epoch, best_name, best_value = _load_checkpoint(
                anchored,
                learned_path,
                target_stage="unified",
                capacity_evidence_receipt=receipt,
                resume_file_identity=learned_identity,
                require_capacity_evidence_receipt=True,
            )
            self.assertEqual(start_epoch, 20)
            self.assertIsNone(best_name)
            self.assertIsNone(best_value)

            schedule = {
                "version": 1,
                "origin_epoch": 20,
                "warmup_epochs": 1,
                "ramp_epochs": 6,
                "max_scale": 1.0,
            }
            _set_residual_application_schedule(anchored, schedule, scale=0.0)
            teacher_payload = {
                "model": anchored.state_dict(),
                "stage": "unified",
                "epoch": 20,
                "config": {"model": teacher_config},
                "prior_backend": "torchscript",
                "teacher_prior_identity": anchored.teacher_prior_identity,
                "residual_application": {**schedule, "scale": 0.0},
                "best_metric": {
                    "name": "line_epe",
                    "mode": "min",
                    "value": 0.25,
                },
                "capacity_evidence_receipt": receipt,
            }
            torch.save(teacher_payload, teacher_checkpoint_path)

            resumed = build_unified_rectifier(teacher_config, {}, device="cpu")
            resumed_start, _, _ = _load_checkpoint(
                resumed,
                teacher_checkpoint_path,
                target_stage="unified",
                capacity_evidence_receipt=receipt,
                require_capacity_evidence_receipt=True,
            )
            self.assertEqual(resumed_start, 21)
            self.assertEqual(
                teacher_payload["capacity_evidence_receipt"]["migration_seed"],
                receipt["migration_seed"],
            )

            missing = dict(teacher_payload)
            missing.pop("capacity_evidence_receipt")
            torch.save(missing, teacher_checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "no strict-valid"):
                _load_checkpoint(
                    build_unified_rectifier(teacher_config, {}, device="cpu"),
                    teacher_checkpoint_path,
                    target_stage="unified",
                    capacity_evidence_receipt=receipt,
                    require_capacity_evidence_receipt=True,
                )

            tampered = dict(teacher_payload)
            tampered["capacity_evidence_receipt"] = dict(receipt)
            tampered["capacity_evidence_receipt"]["evidence_report_sha256"] = (
                "f" * 64
            )
            torch.save(tampered, teacher_checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "no strict-valid"):
                _load_checkpoint(
                    build_unified_rectifier(teacher_config, {}, device="cpu"),
                    teacher_checkpoint_path,
                    target_stage="unified",
                    capacity_evidence_receipt=receipt,
                    require_capacity_evidence_receipt=True,
                )

            different = self._capacity_receipt(
                teacher_path,
                migration_sha256=learned_identity["sha256"],
                evidence_sha256="9" * 64,
            )
            teacher_payload["capacity_evidence_receipt"] = different
            torch.save(teacher_payload, teacher_checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "differs from"):
                _load_checkpoint(
                    build_unified_rectifier(teacher_config, {}, device="cpu"),
                    teacher_checkpoint_path,
                    target_stage="unified",
                    capacity_evidence_receipt=receipt,
                    require_capacity_evidence_receipt=True,
                )

    def test_jacobian_p01_reports_near_singular_positive_map(self) -> None:
        from diffusion2raft.losses import RectificationLoss

        height = width = 16
        x = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
        flow = torch.zeros(1, 2, height, width)
        flow[:, 0:1] = -0.8 * x
        valid = torch.ones(1, 1, height, width, dtype=torch.bool)
        image = torch.rand(1, 3, height, width)
        outputs = {
            "stage": "prior",
            "flows": [flow],
            "residuals": [],
            "raw_residuals": [],
            "prior_flow": flow,
            "final_flow": flow,
        }
        losses = RectificationLoss({})(
            outputs,
            {"warped": image, "target": image, "flow": flow, "valid": valid},
        )
        self.assertEqual(float(losses["fold_rate"]), 0.0)
        self.assertAlmostEqual(float(losses["jacobian_p01"]), 0.2, places=5)

    def test_first_teacher_epoch_writes_atomic_anchor_metadata(self) -> None:
        from diffusion2raft.train import (
            TEACHER_WARMUP_REVISION,
            _validate_existing_teacher_anchor,
            train,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_path = root / "teacher.pt"
            self._save_teacher(teacher_path, 16)
            image_path = root / "image.png"
            flow_path = root / "flow.npy"
            manifest_path = root / "manifest.jsonl"
            image = np.full((16, 16, 3), 127, dtype=np.uint8)
            Image.fromarray(image).save(image_path)
            np.save(flow_path, np.zeros((16, 16, 2), dtype=np.float32))
            manifest_path.write_text(
                json.dumps(
                    {
                        "warped": str(image_path),
                        "target": str(image_path),
                        "flow": str(flow_path),
                        "flow_format": "backward_displacement",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_root = root / "run"
            config = {
                "seed": 3,
                "device": "cpu",
                "data": {
                    "train_manifest": str(manifest_path),
                    "val_manifest": str(manifest_path),
                    "work_size": [16, 16],
                    "batch_size": 1,
                    "num_workers": 0,
                },
                "model": self._model_config(teacher_path),
                "train": {
                    "stage": "unified",
                    "epochs": 1,
                    "lr_unified": 1.0e-4,
                    "weight_decay": 0.0,
                    "amp": False,
                    "amp_dtype": "bfloat16",
                    "max_train_steps": 1,
                    "max_val_batches": 1,
                    "log_every": 10,
                    "save_every": 1,
                    "preview_every": 0,
                    "output_dir": str(output_root),
                    "best_metric": "line_epe",
                    "best_metric_mode": "min",
                    "residual_warmup_epochs": 1,
                    "residual_ramp_epochs": 6,
                    "residual_max_scale": 1.0,
                },
                "loss": {
                    "residual_flow": 1.0,
                    "max_residual_target": 8.0,
                },
                "qwen": {},
            }
            receipt = self._capacity_receipt(teacher_path)
            from diffusion2raft.teacher_capacity_receipt import (
                encode_teacher_capacity_receipt_base64,
            )

            with patch.dict(
                os.environ,
                {
                    "D2R_TEACHER_CAPACITY_RECEIPT_B64": (
                        encode_teacher_capacity_receipt_base64(receipt)
                    )
                },
            ):
                train(config)
            checkpoint_dir = output_root / "unified"
            for name in ("anchor.pt", "latest.pt", "best.pt", "epoch_0001.pt"):
                self.assertTrue((checkpoint_dir / name).is_file(), name)
                saved = torch.load(
                    checkpoint_dir / name,
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(saved["capacity_evidence_receipt"], receipt)
            payload = torch.load(
                checkpoint_dir / "anchor.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["training_revision"], TEACHER_WARMUP_REVISION)
            self.assertEqual(payload["residual_application"]["origin_epoch"], 0)
            self.assertEqual(payload["residual_application"]["scale"], 0.0)
            self.assertIn("flow_size", payload["teacher_prior_identity"])
            self.assertEqual(payload["metrics"]["epe"], payload["metrics"]["prior_epe"])
            _validate_existing_teacher_anchor(
                checkpoint_dir / "anchor.pt",
                teacher_identity=payload["teacher_prior_identity"],
                residual_metadata=payload["residual_application"],
                deployment_contract=payload["deployment_contract"],
                capacity_evidence_receipt=receipt,
            )

            mutations = {
                "identity": lambda candidate: candidate["teacher_prior_identity"].update(
                    mtime_ns=candidate["teacher_prior_identity"]["mtime_ns"] + 1
                ),
                "identity_type": lambda candidate: candidate[
                    "teacher_prior_identity"
                ].update(version=1.0),
                "scale": lambda candidate: candidate["residual_application"].update(
                    scale=0.5
                ),
                "scale_type": lambda candidate: candidate[
                    "residual_application"
                ].update(scale=False),
                "origin": lambda candidate: candidate["residual_application"].update(
                    origin_epoch=1
                ),
                "warmup": lambda candidate: candidate["residual_application"].update(
                    warmup_epochs=2
                ),
                "ramp": lambda candidate: candidate["residual_application"].update(
                    ramp_epochs=12
                ),
                "max_scale": lambda candidate: candidate["residual_application"].update(
                    max_scale=0.75
                ),
                "schedule_version": lambda candidate: candidate[
                    "residual_application"
                ].update(version=999),
                "checkpoint_epoch": lambda candidate: candidate.update(epoch=1),
                "stage": lambda candidate: candidate.update(stage="prior"),
                "backend": lambda candidate: candidate.update(prior_backend="learned"),
                "revision": lambda candidate: candidate.update(
                    training_revision="stale-teacher-contract"
                ),
                "deployment": lambda candidate: candidate[
                    "deployment_contract"
                ]["source_preprocess"].update(image_decoder="opencv"),
                "receipt": lambda candidate: candidate.update(
                    capacity_evidence_receipt=self._capacity_receipt(
                        teacher_path, evidence_sha256="9" * 64
                    )
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    candidate = {
                        **payload,
                        "teacher_prior_identity": dict(
                            payload["teacher_prior_identity"]
                        ),
                        "residual_application": dict(
                            payload["residual_application"]
                        ),
                        "deployment_contract": {
                            **payload["deployment_contract"],
                            "source_preprocess": dict(
                                payload["deployment_contract"]["source_preprocess"]
                            ),
                        },
                    }
                    mutate(candidate)
                    bad_path = checkpoint_dir / f"bad_anchor_{name}.pt"
                    torch.save(candidate, bad_path)
                    with self.assertRaisesRegex(
                        RuntimeError, "existing teacher anchor checkpoint is incompatible"
                    ):
                        _validate_existing_teacher_anchor(
                            bad_path,
                            teacher_identity=payload["teacher_prior_identity"],
                            residual_metadata=payload["residual_application"],
                            deployment_contract=payload["deployment_contract"],
                            capacity_evidence_receipt=receipt,
                        )


if __name__ == "__main__":
    unittest.main()
