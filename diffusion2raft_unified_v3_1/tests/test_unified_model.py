from __future__ import annotations

import math
import unittest
from pathlib import Path

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
        self.assertEqual(
            tuple(outputs["matching_feature_confidence"].shape), (1, 1, 8, 8)
        )
        self.assertEqual(
            tuple(outputs["context_feature_confidence"].shape), (1, 1, 8, 8)
        )
        outputs["final_flow"].mean().backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(trainable_gradients)
        for invalid_temperature in (0.0, float("nan"), float("inf")):
            with self.subTest(invalid_temperature=invalid_temperature):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    model.set_correlation_temperature(invalid_temperature)

    @staticmethod
    def _set_constant_confidence(module, probability: float) -> None:
        final = module.confidence[-1]
        logit = math.log(probability / (1.0 - probability))
        with torch.no_grad():
            final.weight.zero_()
            final.bias.fill_(logit)

    def test_safe_fusion_separates_match_and_context_gates(self) -> None:
        from diffusion2raft.models.unified import FeatureReliabilityFusion

        fusion = FeatureReliabilityFusion(
            8,
            8,
            8,
            match_confidence_power=2.0,
            match_confidence_cap=0.05,
            context_confidence_floor=0.0,
            detach_confidence_from_refiner=True,
        )
        self._set_constant_confidence(fusion, 0.203)
        tensors = (
            torch.randn(1, 8, 6, 6),
            torch.randn(1, 8, 6, 6),
            torch.randn(1, 8, 6, 6),
            torch.randn(1, 2, 6, 6),
        )
        outputs = fusion(*tensors)
        raw, effective, matching, context = outputs[3:7]
        self.assertTrue(torch.allclose(raw, torch.full_like(raw, 0.203), atol=1e-6))
        self.assertTrue(torch.allclose(effective, raw, atol=1e-7))
        self.assertTrue(
            torch.allclose(matching, torch.full_like(matching, 0.203**2), atol=1e-6)
        )
        self.assertTrue(torch.allclose(context, raw, atol=1e-7))

        # A spuriously high gate is still bounded on the matching path, while
        # the context path retains its generation feature amplitude.
        self._set_constant_confidence(fusion, 0.9)
        outputs = fusion(*tensors)
        self.assertTrue(
            torch.allclose(outputs[5], torch.full_like(outputs[5], 0.05), atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(outputs[6], torch.full_like(outputs[6], 0.9), atol=1e-6)
        )

    def test_bad_matching_can_be_disabled_without_deleting_qwen_context(self) -> None:
        from diffusion2raft.models.unified import FeatureReliabilityFusion

        torch.manual_seed(7)
        fusion = FeatureReliabilityFusion(
            8,
            8,
            8,
            match_confidence_power=2.0,
            match_confidence_cap=0.0,
            context_confidence_floor=0.0,
            detach_confidence_from_refiner=True,
        ).eval()
        self._set_constant_confidence(fusion, 0.2)
        qwen_target = torch.randn(1, 8, 6, 6)
        qwen_source = torch.randn(1, 8, 6, 6)
        cnn = torch.randn(1, 8, 6, 6)
        prior = torch.randn(1, 2, 6, 6)
        first = fusion(qwen_target, qwen_source, cnn, prior)
        second = fusion(-qwen_target, -qwen_source, cnn, prior)

        # The cost-volume inputs are pure CNN fallback and cannot be polluted
        # by either set of Qwen features.
        self.assertTrue(torch.allclose(first[0], second[0], atol=1e-6))
        self.assertTrue(torch.allclose(first[1], second[1], atol=1e-6))
        # Qwen generation features still reach the separately gated context.
        self.assertFalse(torch.allclose(first[2], second[2], atol=1e-5))

        # Explicit Qwen-off/dropout remains a complete fallback for both paths.
        force_fallback = torch.ones(1, 1, 1, 1)
        first_off = fusion(qwen_target, qwen_source, cnn, prior, force_fallback)
        second_off = fusion(-qwen_target, -qwen_source, cnn, prior, force_fallback)
        self.assertTrue(torch.allclose(first_off[2], second_off[2], atol=1e-6))
        self.assertTrue(torch.count_nonzero(first_off[5]) == 0)
        self.assertTrue(torch.count_nonzero(first_off[6]) == 0)

        # The runtime ablation controls can isolate the two Qwen jobs without
        # mutating weights or checkpoint schemas.
        matching_off_first = fusion(
            qwen_target,
            qwen_source,
            cnn,
            prior,
            force_matching_fallback=force_fallback,
        )
        matching_off_second = fusion(
            -qwen_target,
            -qwen_source,
            cnn,
            prior,
            force_matching_fallback=force_fallback,
        )
        self.assertTrue(
            torch.allclose(matching_off_first[0], matching_off_second[0], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(matching_off_first[1], matching_off_second[1], atol=1e-6)
        )
        self.assertFalse(
            torch.allclose(matching_off_first[2], matching_off_second[2], atol=1e-5)
        )
        self.assertTrue(torch.count_nonzero(matching_off_first[5]) == 0)
        self.assertGreater(int(torch.count_nonzero(matching_off_first[6])), 0)

        split_fusion = FeatureReliabilityFusion(
            8,
            8,
            8,
            match_confidence_power=1.0,
            match_confidence_cap=1.0,
            context_confidence_floor=0.0,
            detach_confidence_from_refiner=True,
        ).eval()
        self._set_constant_confidence(split_fusion, 0.2)
        context_off_first = split_fusion(
            qwen_target,
            qwen_source,
            cnn,
            prior,
            force_context_fallback=force_fallback,
        )
        context_off_second = split_fusion(
            -qwen_target,
            -qwen_source,
            cnn,
            prior,
            force_context_fallback=force_fallback,
        )
        self.assertTrue(
            torch.allclose(context_off_first[2], context_off_second[2], atol=1e-6)
        )
        self.assertTrue(torch.count_nonzero(context_off_first[6]) == 0)
        self.assertGreater(int(torch.count_nonzero(context_off_first[5])), 0)

    def test_refiner_loss_cannot_inflate_detached_confidence_head(self) -> None:
        from diffusion2raft.models.unified import FeatureReliabilityFusion

        fusion = FeatureReliabilityFusion(
            8,
            8,
            8,
            match_confidence_power=2.0,
            match_confidence_cap=0.05,
            context_confidence_floor=0.0,
            detach_confidence_from_refiner=True,
        )
        tensors = (
            torch.randn(1, 8, 6, 6),
            torch.randn(1, 8, 6, 6),
            torch.randn(1, 8, 6, 6),
            torch.randn(1, 2, 6, 6),
        )
        outputs = fusion(*tensors)
        refiner_loss = outputs[0].square().mean()
        refiner_loss = refiner_loss + outputs[1].square().mean()
        refiner_loss = refiner_loss + outputs[2].square().mean()
        refiner_loss.backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in fusion.confidence.parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in fusion.context.parameters())
        )

        # The explicit confidence objective still trains the calibration head.
        fusion.zero_grad(set_to_none=True)
        fusion(*tensors)[3].mean().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in fusion.confidence.parameters())
        )

    def test_safe_fusion_strictly_loads_legacy_model_state_dict(self) -> None:
        from diffusion2raft.models.unified import build_unified_rectifier

        base = {
            "feature_backend": "lite",
            "feature_channels": 16,
            "feature_stride": 8,
            "cnn_feature_channels": 16,
            "refiner_hidden_channels": 16,
            "refiner_iterations": 1,
            "correlation_radius": 1,
            "feature_dropout_prob": 0.0,
            "prior_base_channels": 4,
            "prior_control_stride": 8,
        }
        legacy = build_unified_rectifier(
            {
                **base,
                "match_confidence_power": 1.0,
                "match_confidence_cap": 1.0,
                "context_confidence_floor": 0.0,
                "detach_confidence_from_refiner": False,
            },
            {},
            device="cpu",
        )
        safe = build_unified_rectifier(
            {
                **base,
                "match_confidence_power": 2.0,
                "match_confidence_cap": 0.05,
                "context_confidence_floor": 0.0,
                "detach_confidence_from_refiner": True,
            },
            {},
            device="cpu",
        )
        incompatible = safe.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(set(legacy.state_dict()), set(safe.state_dict()))

        # A config written before these knobs existed retains the exact legacy
        # gate transform; v3.2 must opt into the conservative behavior.
        default_legacy = build_unified_rectifier(base, {}, device="cpu")
        self.assertEqual(default_legacy.fusion.match_confidence_power, 1.0)
        self.assertEqual(default_legacy.fusion.match_confidence_cap, 1.0)
        self.assertEqual(default_legacy.fusion.context_confidence_floor, 0.0)
        self.assertFalse(default_legacy.fusion.detach_confidence_from_refiner)

    def test_v32_config_explicitly_enables_safe_fusion(self) -> None:
        from diffusion2raft.config import load_config

        path = Path(__file__).resolve().parents[1] / "configs" / "unified_v3_2_safe_fusion.yaml"
        model = load_config(path)["model"]
        self.assertEqual(float(model["match_confidence_power"]), 2.0)
        self.assertEqual(float(model["match_confidence_cap"]), 0.05)
        self.assertEqual(float(model["context_confidence_floor"]), 0.0)
        self.assertTrue(model["detach_confidence_from_refiner"])

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
                "structure_flow": 0.15,
                "line_reconstruction": 0.05,
                "flow_gradient": 0.05,
                "curvature": 0.02,
                "line_straightness": 0.10,
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
        self.assertIn("matching_feature_confidence", losses)
        self.assertIn("context_feature_confidence", losses)
        self.assertIn("epe_p95", losses)
        self.assertIn("line_epe", losses)
        self.assertIn("prior_line_epe", losses)
        self.assertIn("line_epe_gain", losses)
        self.assertIn("line_straightness_error", losses)
        self.assertIn("prior_line_straightness_error", losses)
        self.assertIn("line_straightness_gain", losses)
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
