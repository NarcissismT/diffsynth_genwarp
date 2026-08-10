from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - lightweight source-only environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class UnifiedModelTest(unittest.TestCase):
    def test_lite_backend_is_one_input_and_differentiable(self) -> None:
        from diffusion2raft.models.unified import build_unified_rectifier

        config = {
            "feature_backend": "lite",
            "feature_channels": 16,
            "feature_stride": 8,
            "cnn_feature_channels": 16,
            "refiner_hidden_channels": 16,
            "refiner_iterations": 2,
            "correlation_radius": 1,
            "correlation_temperature": 0.10,
            "shared_qwen_projection": True,
            "feature_dropout_prob": 0.0,
            "prior_base_channels": 4,
            "prior_control_stride": 8,
            "prior_max_displacement_ratio": 0.25,
            "max_residual_px": 8.0,
        }
        model = build_unified_rectifier(config, {}, device="cpu")
        warped = torch.rand(1, 3, 64, 64)
        outputs = model(warped, stage="unified")
        self.assertEqual(tuple(outputs["final_flow"].shape), (1, 2, 64, 64))
        self.assertEqual(len(outputs["flows"]), 2)
        self.assertEqual(len(outputs["residuals"]), 2)
        self.assertEqual(tuple(outputs["qwen_match_logits"].shape), (1, 9, 8, 8))
        outputs["final_flow"].mean().backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(trainable_gradients)

    def test_residual_target_inverts_geometric_composition(self) -> None:
        from diffusion2raft.geometry import (
            compose_backward_flows,
            make_pixel_grid,
            residual_from_composed_flow,
        )

        grid = make_pixel_grid(1, 32, 40)
        base = torch.empty_like(grid)
        base[:, 0] = 0.04 * grid[:, 0] + 0.01 * grid[:, 1]
        base[:, 1] = -0.02 * grid[:, 0] + 0.03 * grid[:, 1]
        residual = torch.zeros_like(base)
        residual[:, 0] = 1.25
        residual[:, 1] = -0.75
        composed = compose_backward_flows(base, residual)
        recovered, consistency = residual_from_composed_flow(
            base, composed, iterations=8
        )
        self.assertLess(float((recovered - residual).abs().max()), 1e-3)
        self.assertLess(float(consistency.max()), 1e-3)

    def test_full_v3_loss_is_differentiable(self) -> None:
        from diffusion2raft.losses import RectificationLoss
        from diffusion2raft.models.unified import build_unified_rectifier

        model_config = {
            "feature_backend": "lite",
            "feature_channels": 16,
            "feature_stride": 8,
            "cnn_feature_channels": 16,
            "refiner_hidden_channels": 16,
            "refiner_iterations": 2,
            "correlation_radius": 1,
            "correlation_temperature": 0.10,
            "shared_qwen_projection": True,
            "feature_dropout_prob": 0.0,
            "prior_base_channels": 4,
            "prior_control_stride": 8,
            "prior_max_displacement_ratio": 0.25,
            "max_residual_px": 8.0,
        }
        model = build_unified_rectifier(model_config, {}, device="cpu")
        warped = torch.rand(1, 3, 64, 64)
        target_flow = torch.zeros(1, 2, 64, 64)
        target_flow[:, 0] = 1.0
        outputs = model(warped, stage="unified")
        criterion = RectificationLoss(
            {
                "flow": 1.0,
                "prior_flow_unified": 0.25,
                "reconstruction": 0.1,
                "gradient": 0.1,
                "bending": 0.01,
                "anti_fold": 0.1,
                "residual": 0.002,
                "residual_flow": 0.25,
                "qwen_match": 0.05,
                "confidence": 0.02,
                "max_residual_target": 8.0,
            }
        )
        losses = criterion(
            outputs,
            {
                "warped": warped,
                "target": warped.clone(),
                "flow": target_flow,
                "valid": torch.ones(1, 1, 64, 64, dtype=torch.bool),
            },
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertIn("qwen_match_epe", losses)
        self.assertIn("gate_target", losses)
        losses["total"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )


if __name__ == "__main__":
    unittest.main()
