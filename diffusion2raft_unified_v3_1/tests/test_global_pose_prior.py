from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - lightweight source-only environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GlobalPosePriorTest(unittest.TestCase):
    def test_v32_production_config_couples_pose_bigrot_and_safe_fusion(self) -> None:
        from diffusion2raft.config import load_config

        config_path = Path(__file__).resolve().parents[1] / "configs" / "unified_v3_2.yaml"
        config = load_config(config_path)
        model = config["model"]
        augment = config["data"]["source_geometry_augment"]
        self.assertTrue(model["prior_global_pose_enabled"])
        self.assertGreater(float(model["prior_global_pose_max_linear_delta"]), 2.0)
        self.assertEqual(float(augment["max_rotation_deg"]), 180.0)
        self.assertGreater(float(augment["probability"]), 0.0)
        self.assertEqual(float(model["match_confidence_cap"]), 0.05)
        self.assertTrue(model["detach_confidence_from_refiner"])
        self.assertGreater(float(config["train"]["lr_global_pose"]), 0.0)

    def test_identity_pose_preserves_legacy_prior_and_state_keys(self) -> None:
        from diffusion2raft.models.prior import DocumentGeometryPrior

        torch.manual_seed(4)
        legacy = DocumentGeometryPrior(base_channels=4)
        upgraded = DocumentGeometryPrior(
            base_channels=4,
            global_pose_enabled=True,
            global_pose_hidden_channels=16,
            global_pose_pool_size=2,
        )
        missing, unexpected = upgraded.load_state_dict(legacy.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(all(key.startswith("global_pose_head.") for key in missing))
        self.assertFalse(
            any(key.startswith("global_pose_head.") for key in legacy.state_dict())
        )

        image = torch.rand(2, 3, 48, 64)
        legacy.eval()
        upgraded.eval()
        with torch.no_grad():
            legacy_flow = legacy(image)
            upgraded_flow = upgraded(image)
        self.assertTrue(torch.allclose(legacy_flow, upgraded_flow, atol=1e-5, rtol=0.0))

    def test_pose_head_represents_exact_half_turn_beyond_local_cap(self) -> None:
        from diffusion2raft.models.prior import GlobalProjectiveHead

        head = GlobalProjectiveHead(
            4,
            hidden_channels=8,
            pool_size=1,
            max_linear_delta=2.5,
        )
        # 1 + 2.5*tanh(raw_diagonal) = -1.  The finite logit leaves
        # meaningful gradients on both diagonal coefficients.
        with torch.no_grad():
            half_turn_logit = torch.atanh(torch.tensor(-2.0 / 2.5))
            head.mlp[-1].bias[0] = half_turn_logit
            head.mlp[-1].bias[4] = half_turn_logit
        height, width = 7, 9
        flow = head(
            torch.randn(1, 4, 3, 3),
            torch.zeros(1, 2, height, width),
        )
        y, x = torch.meshgrid(
            torch.arange(height, dtype=flow.dtype),
            torch.arange(width, dtype=flow.dtype),
            indexing="ij",
        )
        expected = torch.stack((width - 1 - 2 * x, height - 1 - 2 * y), dim=0)
        self.assertTrue(torch.allclose(flow[0], expected, atol=2e-5, rtol=0.0))
        self.assertGreater(float(flow.detach().abs().max()), 0.35 * max(height, width))

    def test_pose_head_fp32_autocast_and_half_turn_gradient_are_stable(self) -> None:
        from diffusion2raft.models.prior import GlobalProjectiveHead

        torch.manual_seed(8)
        head = GlobalProjectiveHead(4, hidden_channels=8, pool_size=1)
        features = torch.randn(1, 4, 3, 3)
        height, width = 9, 11
        local_flow = torch.zeros(1, 2, height, width)
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        target = torch.stack((width - 1 - 2 * x, height - 1 - 2 * y), dim=0)[None]

        reference = head(features, local_flow)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            autocast_flow = head(features, local_flow)
        self.assertEqual(autocast_flow.dtype, torch.float32)
        self.assertTrue(torch.equal(reference, autocast_flow))

        loss = torch.nn.functional.mse_loss(reference, target)
        loss.backward()
        gradient = head.mlp[-1].bias.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[[0, 4]].abs().min()), 1.0)

    def test_pose_head_non_square_quarter_turn_matches_pixel_homography(self) -> None:
        from diffusion2raft.data import apply_source_homography, source_affine_homography
        from diffusion2raft.models.prior import GlobalProjectiveHead

        height, width = 49, 81
        head = GlobalProjectiveHead(4, hidden_channels=8, pool_size=1)
        # Convert a +90 degree pixel-space rotation about the canvas centre to
        # align-corners normalized coordinates. A non-square canvas exercises
        # the otherwise easy-to-miss x/y aspect-ratio factors.
        normalized = torch.tensor(
            (
                0.0,
                -(height - 1) / (width - 1),
                0.0,
                (width - 1) / (height - 1),
                0.0,
                0.0,
                0.0,
                0.0,
            )
        )
        identity_coefficients = torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        with torch.no_grad():
            head.mlp[-1].bias.copy_(
                torch.atanh((normalized - identity_coefficients) / 2.5)
            )

        local_flow = torch.zeros(1, 2, height, width)
        predicted = head(torch.zeros(1, 4, 2, 3), local_flow)
        _, expected, _ = apply_source_homography(
            torch.zeros(3, height, width),
            local_flow[0],
            torch.ones(1, height, width, dtype=torch.bool),
            source_affine_homography((height, width), angle_deg=90.0),
        )
        torch.testing.assert_close(predicted[0], expected, rtol=0.0, atol=1.0e-5)

    def test_pose_head_rejects_nonfinite_limits(self) -> None:
        from diffusion2raft.models.prior import GlobalProjectiveHead

        for kwargs in (
            {"max_linear_delta": float("nan")},
            {"max_linear_delta": float("inf")},
            {"max_translation_ratio": float("nan")},
            {"max_translation_ratio": float("inf")},
            {"max_perspective": float("nan")},
            {"max_perspective": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GlobalProjectiveHead(4, hidden_channels=8, pool_size=1, **kwargs)

    def test_v31_checkpoint_migrates_strictly_and_resets_optimizer(self) -> None:
        from diffusion2raft.models.unified import build_unified_rectifier
        from diffusion2raft.train import _build_optimizer, _load_checkpoint

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
        training = {
            "train": {
                "lr_prior_joint": 5e-6,
                "lr_unified": 1e-4,
                "lr_global_pose": 2e-4,
                "weight_decay": 1e-5,
            }
        }
        legacy = build_unified_rectifier(base, {}, device="cpu")
        upgraded = build_unified_rectifier(
            {
                **base,
                "prior_global_pose_enabled": True,
                "prior_global_pose_hidden_channels": 16,
                "prior_global_pose_pool_size": 2,
            },
            {},
            device="cpu",
        )
        old_optimizer = _build_optimizer(legacy, training, "unified")
        new_optimizer = _build_optimizer(upgraded, training, "unified")
        self.assertEqual(
            [group["name"] for group in old_optimizer.param_groups],
            ["prior", "unified_heads"],
        )
        self.assertEqual(
            [group["name"] for group in new_optimizer.param_groups],
            ["prior", "global_pose", "unified_heads"],
        )

        checkpoint_config = {
            "model": {**base, "correlation_temperature": 0.1},
            "train": {"stage": "unified"},
        }
        payload = {
            "model": legacy.state_dict(),
            "optimizer": old_optimizer.state_dict(),
            "stage": "unified",
            "epoch": 9,
            # Real v3.1 checkpoints always carry their authoritative config;
            # the loader now uses it to reconstruct the runtime correlation
            # temperature before accepting a same-stage migration.
            "config": checkpoint_config,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(payload, path)
            start_epoch, _, _ = _load_checkpoint(
                upgraded,
                path,
                target_stage="unified",
                optimizer=new_optimizer,
            )
            self.assertEqual(start_epoch, 10)
            self.assertFalse(new_optimizer.state)

            # Compatibility is prefix-scoped: an unrelated missing tensor is
            # still a hard error rather than silently weakening the model.
            broken = dict(payload)
            broken_state = dict(payload["model"])
            broken_state.pop("prior.stem.0.0.weight")
            broken["model"] = broken_state
            torch.save(broken, path)
            with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
                _load_checkpoint(upgraded, path, target_stage="unified")

            # A current checkpoint with only part of the pose branch missing
            # is corruption, not a valid v3.1 -> v3.2 migration.
            partial_pose = {
                "model": dict(upgraded.state_dict()),
                "stage": "unified",
                "epoch": 9,
                "config": checkpoint_config,
            }
            partial_pose["model"].pop("prior.global_pose_head.mlp.3.bias")
            torch.save(partial_pose, path)
            with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
                _load_checkpoint(upgraded, path, target_stage="unified")


if __name__ == "__main__":
    unittest.main()
