from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - lightweight source-only environments
    torch = None


if torch is not None:

    class _InputProbeTeacher(torch.nn.Module):
        def forward(self, image1, image2):
            # x encodes the first input channel (which must be original blue);
            # y proves both scripted arguments receive the identical tensor.
            return [
                torch.cat(
                    (
                        image1[:, 0:1],
                        (image1 - image2).abs().sum(dim=1, keepdim=True),
                    ),
                    dim=1,
                )
            ]


    class _WrongShapeTeacher(torch.nn.Module):
        def forward(self, image1, image2):
            del image2
            return [image1[:, :2, :-1]]


    class _NonFiniteTeacher(torch.nn.Module):
        def forward(self, image1, image2):
            del image2
            return [image1[:, :2] / 0.0]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchScriptGeometryPriorTest(unittest.TestCase):
    @staticmethod
    def _save_trace(module, path: Path, size: int) -> None:
        example = torch.rand(1, 3, size, size)
        traced = torch.jit.trace(module.eval(), (example, example), strict=False)
        torch.jit.save(traced, str(path))

    def test_bgr_two_inputs_blur_and_absolute_coordinate_resize(self) -> None:
        from torchvision.transforms.functional import gaussian_blur

        from diffusion2raft.geometry import resize_backward_flow
        from diffusion2raft.models.teacher_prior import TorchScriptGeometryPrior

        teacher_size = 17
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            self._save_trace(_InputProbeTeacher(), path, teacher_size)
            prior = TorchScriptGeometryPrior(
                path,
                device="cpu",
                input_size=teacher_size,
                blur_kernel=5,
                autocast_dtype="float16",
            )

            warped = torch.zeros(1, 3, 7, 11)
            warped[:, 0] = 0.10
            warped[:, 1] = 0.20
            warped[:, 2, 3, 5] = 0.90
            flow = prior(warped)

            resized = torch.nn.functional.interpolate(
                warped,
                size=(teacher_size, teacher_size),
                mode="bilinear",
                align_corners=False,
            )
            # Fake teacher x = B and y = |arg1-arg2| = 0.
            teacher_flow = torch.cat(
                (resized[:, 2:3], torch.zeros_like(resized[:, 0:1])), dim=1
            )
            teacher_flow = gaussian_blur(teacher_flow, [5, 5])
            expected = resize_backward_flow(
                teacher_flow,
                (7, 11),
                source_size_from=(teacher_size, teacher_size),
                source_size_to=(7, 11),
            )
            self.assertEqual(tuple(flow.shape), (1, 2, 7, 11))
            self.assertTrue(torch.allclose(flow, expected, atol=1e-6, rtol=0.0))

            # The frozen scripted module remains external and in eval mode even
            # when the lightweight wrapper follows the rectifier into train().
            prior.train()
            self.assertTrue(prior.training)
            self.assertFalse(prior.teacher.training)
            self.assertFalse(any(prior.teacher.parameters()))
            self.assertEqual(
                list(prior.state_dict()), ["_teacher_backend_marker"]
            )
            self.assertFalse(
                any(name.startswith("_teacher") for name, _ in prior.named_modules())
            )
            frozen_identity = (
                prior.resolved_checkpoint_path,
                prior.checkpoint_size_bytes,
                prior.checkpoint_mtime_ns,
            )
            with path.open("ab") as stream:
                stream.write(b"changed-after-load")
            self.assertEqual(
                (
                    prior.resolved_checkpoint_path,
                    prior.checkpoint_size_bytes,
                    prior.checkpoint_mtime_ns,
                ),
                frozen_identity,
            )

    def test_jit_load_reads_authenticated_procfd_not_replaced_path(self) -> None:
        from diffusion2raft.models.teacher_prior import TorchScriptGeometryPrior

        size = 9
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "teacher.pt"
            replacement = root / "replacement.pt"
            backup = root / "original.pt"
            self._save_trace(_InputProbeTeacher(), path, size)
            self._save_trace(_WrongShapeTeacher(), replacement, size)
            example = torch.rand(1, 3, size, size)
            real_load = torch.jit.load
            captured = {}

            def replace_then_load(load_path, *args, **kwargs):
                captured["load_path"] = str(load_path)
                path.rename(backup)
                replacement.rename(path)
                module = real_load(load_path, *args, **kwargs)
                captured["shape"] = tuple(module(example, example)[-1].shape)
                return module

            with mock.patch("torch.jit.load", side_effect=replace_then_load):
                with self.assertRaisesRegex(RuntimeError, "changed while in use"):
                    TorchScriptGeometryPrior(
                        path, device="cpu", input_size=size, blur_kernel=1
                    )
            self.assertTrue(captured["load_path"].startswith("/proc/self/fd/"))
            self.assertEqual(captured["shape"], (1, 2, size, size))

    def test_teacher_output_shape_and_finiteness_are_strict(self) -> None:
        from diffusion2raft.models.teacher_prior import TorchScriptGeometryPrior

        size = 9
        image = torch.rand(1, 3, size, size)
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            wrong_path = directory_path / "wrong.pt"
            nan_path = directory_path / "nan.pt"
            self._save_trace(_WrongShapeTeacher(), wrong_path, size)
            self._save_trace(_NonFiniteTeacher(), nan_path, size)
            wrong = TorchScriptGeometryPrior(
                wrong_path, device="cpu", input_size=size, blur_kernel=1
            )
            nonfinite = TorchScriptGeometryPrior(
                nan_path, device="cpu", input_size=size, blur_kernel=1
            )
            with self.assertRaisesRegex(ValueError, "wrong flow shape"):
                wrong(image)
            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                nonfinite(image)

    def test_expected_sha256_is_checked_before_jit_load(self) -> None:
        from diffusion2raft.models.teacher_prior import TorchScriptGeometryPrior

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            self._save_trace(_InputProbeTeacher(), path, 9)
            with mock.patch("torch.jit.load") as jit_load:
                with self.assertRaisesRegex(
                    RuntimeError, "sha256 differs from configured expected digest"
                ):
                    TorchScriptGeometryPrior(
                        path,
                        device="cpu",
                        input_size=9,
                        blur_kernel=1,
                        expected_sha256="0" * 64,
                    )
            jit_load.assert_not_called()

    def test_logical_cuda0_contract_fails_before_loading_on_cpu(self) -> None:
        from diffusion2raft.models.teacher_prior import TorchScriptGeometryPrior

        with self.assertRaisesRegex(RuntimeError, "logical cuda:0"):
            TorchScriptGeometryPrior(
                "/path/that/must/not/be-loaded.pt",
                device="cpu",
                requires_logical_cuda0=True,
            )

    def test_learned_migration_is_scoped_and_teacher_resume_is_strict(self) -> None:
        from diffusion2raft.models.unified import build_unified_rectifier
        from diffusion2raft.train import (
            _build_optimizer,
            _load_checkpoint,
            _set_residual_application_schedule,
        )

        size = 17
        base = {
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
        train_config = {
            "train": {
                "lr_prior_joint": 5e-6,
                "lr_unified": 1e-4,
                "weight_decay": 1e-5,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            teacher_path = directory_path / "teacher.pt"
            checkpoint_path = directory_path / "checkpoint.pt"
            self._save_trace(_InputProbeTeacher(), teacher_path, size)
            teacher_config = {
                **base,
                "prior_backend": "torchscript",
                "prior_torchscript_path": str(teacher_path),
                "prior_torchscript_sha256": hashlib.sha256(
                    teacher_path.read_bytes()
                ).hexdigest(),
                "prior_torchscript_size": size,
                "prior_torchscript_blur_kernel": 1,
            }
            learned = build_unified_rectifier(base, {}, device="cpu")
            anchored = build_unified_rectifier(teacher_config, {}, device="cpu")
            old_optimizer = _build_optimizer(learned, train_config, "unified")
            anchored_optimizer = _build_optimizer(
                anchored, train_config, "unified"
            )
            self.assertEqual(
                [group["name"] for group in old_optimizer.param_groups],
                ["prior", "unified_heads"],
            )
            self.assertEqual(
                [group["name"] for group in anchored_optimizer.param_groups],
                ["unified_heads"],
            )
            prior_keys = [
                key for key in anchored.state_dict() if key.startswith("prior.")
            ]
            self.assertEqual(prior_keys, ["prior._teacher_backend_marker"])

            legacy_payload = {
                "model": learned.state_dict(),
                "optimizer": old_optimizer.state_dict(),
                "stage": "unified",
                "epoch": 19,
                "config": {"model": base},
            }
            falsely_declared_teacher = {
                **legacy_payload,
                "prior_backend": "torchscript",
            }
            torch.save(falsely_declared_teacher, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "has no teacher marker"):
                _load_checkpoint(
                    learned, checkpoint_path, target_stage="unified"
                )

            invalid_legacy_epoch = {**legacy_payload, "epoch": 19.9}
            torch.save(invalid_legacy_epoch, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "epoch must be an integer"):
                _load_checkpoint(
                    anchored, checkpoint_path, target_stage="unified"
                )

            torch.save(legacy_payload, checkpoint_path)
            start_epoch, _, _ = _load_checkpoint(
                anchored,
                checkpoint_path,
                target_stage="unified",
                optimizer=anchored_optimizer,
            )
            self.assertEqual(start_epoch, 20)
            self.assertFalse(anchored_optimizer.state)

            # Any mismatch outside the replaced prior branch is still fatal.
            broken_legacy = dict(legacy_payload)
            broken_state = dict(legacy_payload["model"])
            broken_state.pop("cnn_encoder.net.0.0.weight")
            broken_legacy["model"] = broken_state
            torch.save(broken_legacy, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
                _load_checkpoint(
                    anchored, checkpoint_path, target_stage="unified"
                )

            # Source-only keys do not appear as target ``missing`` entries, so
            # migration explicitly reconstructs and validates the full learned
            # prior schema rather than accepting a truncated branch.
            truncated_legacy = dict(legacy_payload)
            truncated_state = {
                key: value
                for key, value in legacy_payload["model"].items()
                if not key.startswith("prior.")
            }
            one_prior_key = next(
                key for key in legacy_payload["model"] if key.startswith("prior.")
            )
            truncated_state[one_prior_key] = legacy_payload["model"][one_prior_key]
            truncated_legacy["model"] = truncated_state
            torch.save(truncated_legacy, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "complete source prior"):
                _load_checkpoint(
                    anchored, checkpoint_path, target_stage="unified"
                )

            # Once a teacher checkpoint exists, marker and head restoration are
            # fully strict and the same-shape optimizer state is resumed.
            anchored_optimizer.param_groups[0]["lr"] = 0.0123
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
                "optimizer": anchored_optimizer.state_dict(),
                "stage": "unified",
                "epoch": 20,
                "prior_backend": "torchscript",
                "teacher_prior_identity": anchored.teacher_prior_identity,
                "residual_application": {**schedule, "scale": 0.0},
                "best_metric": {
                    "name": "line_epe",
                    "mode": "min",
                    "value": 0.25,
                },
                "config": {"model": teacher_config},
            }
            torch.save(teacher_payload, checkpoint_path)
            resumed = build_unified_rectifier(teacher_config, {}, device="cpu")
            resumed_optimizer = _build_optimizer(resumed, train_config, "unified")
            start_epoch, _, _ = _load_checkpoint(
                resumed,
                checkpoint_path,
                target_stage="unified",
                optimizer=resumed_optimizer,
            )
            self.assertEqual(start_epoch, 21)
            self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], 0.0123)

            wrong_best_mode = {
                **teacher_payload,
                "best_metric": {
                    **teacher_payload["best_metric"],
                    "mode": "max",
                },
            }
            torch.save(wrong_best_mode, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "mode disagrees"):
                _load_checkpoint(
                    resumed,
                    checkpoint_path,
                    target_stage="unified",
                    optimizer=resumed_optimizer,
                    expected_best_metric_name="line_epe",
                    expected_best_metric_mode="min",
                )

            missing_optimizer = dict(teacher_payload)
            missing_optimizer.pop("optimizer")
            torch.save(missing_optimizer, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "missing optimizer state"):
                _load_checkpoint(
                    resumed,
                    checkpoint_path,
                    target_stage="unified",
                    optimizer=resumed_optimizer,
                )

            renamed_optimizer = {
                **teacher_payload,
                "optimizer": {
                    **teacher_payload["optimizer"],
                    "param_groups": [
                        {
                            **group,
                            "name": "wrong_group",
                        }
                        for group in teacher_payload["optimizer"]["param_groups"]
                    ],
                },
            }
            torch.save(renamed_optimizer, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "group names differ"):
                _load_checkpoint(
                    resumed,
                    checkpoint_path,
                    target_stage="unified",
                    optimizer=resumed_optimizer,
                )

            missing_identity = dict(teacher_payload)
            missing_identity.pop("teacher_prior_identity")
            torch.save(missing_identity, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "missing teacher_prior_identity"):
                _load_checkpoint(resumed, checkpoint_path, target_stage="unified")

            missing_residual = dict(teacher_payload)
            missing_residual.pop("residual_application")
            torch.save(missing_residual, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "missing residual_application"):
                _load_checkpoint(resumed, checkpoint_path, target_stage="unified")

            broken_teacher = dict(teacher_payload)
            broken_teacher_state = dict(teacher_payload["model"])
            broken_teacher_state.pop("prior._teacher_backend_marker")
            broken_teacher["model"] = broken_teacher_state
            torch.save(broken_teacher, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "has no teacher marker"):
                _load_checkpoint(resumed, checkpoint_path, target_stage="unified")

            invalid_marker = dict(teacher_payload)
            invalid_marker_state = dict(teacher_payload["model"])
            invalid_marker_state["prior._teacher_backend_marker"] = torch.tensor(
                [0, 0, 0], dtype=torch.uint8
            )
            invalid_marker["model"] = invalid_marker_state
            torch.save(invalid_marker, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "invalid TorchScript prior"):
                _load_checkpoint(resumed, checkpoint_path, target_stage="unified")

    def test_v33_config_selects_exact_teacher_and_stretch_contract(self) -> None:
        from diffusion2raft.config import load_config

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "unified_v3_3_teacher_anchor.yaml"
        )
        config = load_config(config_path)
        model = config["model"]
        self.assertEqual(model["prior_backend"], "torchscript")
        self.assertEqual(int(model["prior_torchscript_size"]), 512)
        self.assertEqual(int(model["prior_torchscript_flow_size"]), 512)
        self.assertEqual(int(model["prior_torchscript_blur_kernel"]), 39)
        self.assertEqual(model["prior_torchscript_autocast_dtype"], "float16")
        self.assertTrue(model["prior_torchscript_requires_logical_cuda0"])
        self.assertTrue(str(model["prior_torchscript_path"]).endswith(
            "/259999_raft_unwarp.pt"
        ))
        self.assertEqual(
            model["prior_torchscript_sha256"],
            "3d079e19445168169144f2af741362f673289b6510df4a4c1af348449ae045b9",
        )
        self.assertEqual(config["inference"]["resize_policy"], "stretch")
        self.assertEqual(config["inference"]["image_decoder"], "opencv")
        self.assertEqual(
            config["inference"]["resize_interpolation"], "opencv_baseline"
        )
        inpaint = config["inference"]["inpaint"]
        self.assertTrue(inpaint["enabled"])
        self.assertEqual(int(inpaint["size"]), 512)
        self.assertEqual(int(inpaint["dilation"]), 11)
        self.assertTrue(str(inpaint["path"]).endswith("/common_erase.pt"))
        self.assertEqual(
            inpaint["sha256"],
            "f93d5573e0433d4d24b80f2c0cb0f3c445b891b17baf3aa939fb9af0421ebb53",
        )
        self.assertEqual(int(config["train"]["epochs"]), 32)
        self.assertEqual(
            config["train"]["output_dir"], "runs/d2r_v3_3_teacher_anchor"
        )
        self.assertEqual(
            config["train"]["best_guard"],
            {
                "max_epe_exclusive": 5.7501,
                "min_epe_gain_exclusive": 0.0,
                "min_final_win_rate_exclusive": 0.5,
                "max_fold_rate_exclusive": 0.0004,
                "min_line_epe_gain_exclusive": 0.0,
                "min_line_straightness_gain_exclusive": 0.0,
                "min_jacobian_p01": 0.01,
            },
        )


if __name__ == "__main__":
    unittest.main()
