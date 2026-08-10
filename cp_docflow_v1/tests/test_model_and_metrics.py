from __future__ import annotations

import unittest

import torch

from cp_docflow.geometry import canonical_backward_map, warp_with_backward_map
from cp_docflow.losses import CoarseRectificationLoss, masked_local_ssim_loss
from cp_docflow.metrics import (
    confidence_brier_score,
    confidence_ece,
    fold_rate,
    geometry_quality_metrics,
    high_confidence_damage_rate,
    image_quality_metrics,
)
from cp_docflow.models.coarse import DeterministicCoarseRectifier


class ModelAndMetricsTest(unittest.TestCase):
    def _model(self) -> DeterministicCoarseRectifier:
        return DeterministicCoarseRectifier(base_channels=8, feature_channels=16)

    def test_zero_initialized_model_is_source_faithful(self) -> None:
        torch.manual_seed(1)
        image = torch.rand(1, 3, 32, 40)
        output = self._model().eval()(image)
        expected = canonical_backward_map(1, (32, 40))
        torch.testing.assert_close(output["backward_map"], expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(output["rectified_image"], image, atol=2.0e-6, rtol=0.0)
        torch.testing.assert_close(
            output["confidence"],
            torch.full_like(
                output["confidence"],
                1.0 - torch.exp(torch.tensor(-0.5)).item(),
            ),
            atol=1.0e-7,
            rtol=0.0,
        )

    def test_identity_geometry_quality_has_no_distortion(self) -> None:
        identity = canonical_backward_map(1, (24, 28))
        valid = torch.ones(1, 1, 24, 28, dtype=torch.bool)
        metrics = geometry_quality_metrics(identity, identity, valid)
        self.assertEqual(float(metrics["local_scale_anomaly_rate"]), 0.0)
        self.assertAlmostEqual(float(metrics["orthogonality_error"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["bending_energy"]), 0.0, places=7)
        self.assertLess(float(metrics["page_border_line_error_px"]), 1.0e-5)

    def test_identical_images_have_perfect_preservation_metrics(self) -> None:
        image = torch.rand(1, 3, 24, 28)
        valid = torch.ones(1, 1, 24, 28, dtype=torch.bool)
        structure = torch.ones(1, 3, 24, 28)
        metrics = image_quality_metrics(image, image, valid, structure)
        self.assertEqual(float(metrics["rgb_l1"]), 0.0)
        self.assertGreaterEqual(float(metrics["rgb_psnr"]), 119.0)
        self.assertAlmostEqual(float(metrics["rgb_ssim"]), 1.0, places=5)
        self.assertAlmostEqual(
            float(metrics["character_edge_preservation"]), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(metrics["table_line_connectivity"]), 1.0, places=6
        )

    def test_masked_ssim_is_differentiable_and_ignores_invalid_pixels(self) -> None:
        target = torch.rand(1, 3, 16, 18)
        prediction = target.clone().requires_grad_(True)
        valid = torch.ones(1, 1, 16, 18, dtype=torch.bool)
        valid[..., :3, :3] = False
        with torch.no_grad():
            prediction[..., :3, :3] = 0.0
        identical_loss = masked_local_ssim_loss(prediction, target, valid)
        self.assertLess(float(identical_loss), 1.0e-5)
        modified = prediction.clone()
        modified[..., 8:11, 8:11] = 0.0
        loss = masked_local_ssim_loss(modified, target, valid)
        self.assertGreater(float(loss), float(identical_loss))
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    def test_minimum_16px_canvas_supports_batch_one(self) -> None:
        output = self._model().eval()(torch.rand(1, 3, 16, 16), render=False)
        self.assertEqual(tuple(output["backward_map"].shape), (1, 2, 16, 16))

    def test_model_supports_different_target_canvas(self) -> None:
        image = torch.rand(1, 3, 32, 40)
        output = self._model().eval()(image, output_size=(24, 28), render=False)
        expected = canonical_backward_map(1, (24, 28), (32, 40))
        torch.testing.assert_close(output["backward_map"], expected, atol=0.0, rtol=0.0)
        self.assertNotIn("rectified_image", output)

    def test_loss_backpropagates_to_map_and_confidence_heads(self) -> None:
        torch.manual_seed(2)
        model = self._model().train()
        image = torch.rand(1, 3, 32, 32)
        target_map = canonical_backward_map(1, (32, 32)).clone()
        target_map[:, 0] += 1.0
        batch = {
            "warped_image": image,
            "rectified_image": image,
            "backward_map": target_map,
            "valid_mask": torch.ones(1, 1, 32, 32, dtype=torch.bool),
        }
        losses = CoarseRectificationLoss()(model(image), batch)
        losses["total"].backward()
        self.assertIsNotNone(model.map_head[-1].weight.grad)
        self.assertIsNotNone(model.log_variance_head[-1].weight.grad)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_rgb_auxiliary_is_zero_at_gt_map_despite_appearance_target(self) -> None:
        image = torch.rand(1, 3, 24, 28)
        target_map = canonical_backward_map(1, (24, 28)).clone()
        target_map[:, 0] += 0.4
        rendered = warp_with_backward_map(image, target_map)
        prediction = {
            "backward_map": target_map,
            "log_variance": torch.zeros(1, 1, 24, 28),
            "rectified_image": rendered,
        }
        batch = {
            "warped_image": image,
            # Deliberately unrelated appearance: it must not pull geometry.
            "rectified_image": torch.zeros_like(image),
            "backward_map": target_map,
            "valid_mask": torch.ones(1, 1, 24, 28, dtype=torch.bool),
        }
        losses = CoarseRectificationLoss()(prediction, batch)
        self.assertLess(float(losses["warp"]), 1.0e-7)
        self.assertLess(float(losses["gradient"]), 1.0e-7)

    def test_identity_has_no_folds(self) -> None:
        identity = canonical_backward_map(1, (16, 20))
        valid = torch.ones(1, 1, 16, 20, dtype=torch.bool)
        self.assertEqual(float(fold_rate(identity, valid)), 0.0)

    def test_high_confidence_damage_metric_matches_definition(self) -> None:
        target = canonical_backward_map(1, (2, 2))
        coarse = target.clone()
        final = target.clone()
        final[:, 0, 0, 0] += 2.0
        valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        self.assertAlmostEqual(
            float(high_confidence_damage_rate(coarse, final, target, valid)),
            0.25,
        )

    def test_confidence_calibration_metrics(self) -> None:
        target = canonical_backward_map(1, (2, 2))
        prediction = target.clone()
        confidence = torch.ones(1, 1, 2, 2)
        valid = torch.ones_like(confidence, dtype=torch.bool)
        self.assertEqual(
            float(confidence_brier_score(confidence, prediction, target, valid)),
            0.0,
        )
        self.assertEqual(
            float(confidence_ece(confidence, prediction, target, valid)),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
