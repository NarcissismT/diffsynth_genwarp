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
        outputs["final_flow"].mean().backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(trainable_gradients)


if __name__ == "__main__":
    unittest.main()
