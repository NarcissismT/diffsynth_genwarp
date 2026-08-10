from __future__ import annotations

import inspect
import unittest

import torch

from cp_docflow.checkpoint import COORDINATE_CONTRACT
from cp_docflow.geometry import canonical_backward_map
from cp_docflow.models.coordinate_flow_transformer import (
    ResidualCoordinateFlowTransformer,
    build_residual_proposal_target,
)
from cp_docflow.models.docgrid_flow import CPDocFlow
from cp_docflow.models.warr import ConvexMapUpsampler, HighResolutionMapRefiner


class DocGridV2ArchitectureTest(unittest.TestCase):
    def test_residual_target_and_composition_are_mathematically_consistent(self) -> None:
        coarse = canonical_backward_map(1, (4, 5))
        target = coarse + torch.tensor([3.0, -2.0]).view(1, 2, 1, 1)
        confidence = torch.full((1, 1, 4, 5), 0.8)
        residual, proposal, gate = build_residual_proposal_target(
            target,
            coarse,
            confidence,
            minimum_gate=0.25,
            residual_clip_px=100.0,
        )
        torch.testing.assert_close(residual, target - coarse)
        torch.testing.assert_close(coarse + gate * proposal, target)
        self.assertFalse(gate.requires_grad)

    def test_model_exposes_absolute_and_residual_sequences_separately(self) -> None:
        model = CPDocFlow(
            coarse={"base_channels": 8, "feature_channels": 16},
            qwen_backend="none",
            qwen_feature_channels=12,
            fusion_channels=16,
            hv_channels=8,
            velocity_hidden_channels=16,
            velocity_time_channels=16,
            flow_blocks=2,
            flow_heads=4,
            fm_steps=2,
            refiner_hidden_channels=16,
            refiner_iterations=2,
        ).eval()
        output = model(torch.rand(1, 3, 32, 32), render=False)
        self.assertIs(output["flow_matching_sequence"], output["flow_matching_map_sequence"])
        self.assertEqual(output["map_sequence_coordinate_contract"], COORDINATE_CONTRACT)
        self.assertEqual(len(output["flow_matching_residual_sequence"]), 2)
        self.assertEqual(len(output["flow_matching_map_sequence"]), 2)
        torch.testing.assert_close(
            output["fusion_weights"].sum(dim=1),
            torch.ones_like(output["fusion_weights"][:, 0]),
        )
        torch.testing.assert_close(
            output["fusion_weights"][:, 0],
            torch.zeros_like(output["fusion_weights"][:, 0]),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_coordinate_decoder_is_a_transformer_not_a_conv_unet(self) -> None:
        decoder = ResidualCoordinateFlowTransformer(
            16, 8, hidden_channels=16, time_channels=16, blocks=8, heads=4
        )
        self.assertEqual(len(decoder.blocks), 8)
        for block in decoder.blocks:
            self.assertIsInstance(block.local_attention, torch.nn.MultiheadAttention)
            self.assertIsInstance(block.visual_attention, torch.nn.MultiheadAttention)

    def test_warr_has_no_operational_confidence_input(self) -> None:
        source = inspect.getsource(HighResolutionMapRefiner.forward)
        self.assertIn("del confidence, coarse_map", source)
        self.assertNotIn("confidence_gate", source)

    def test_convex_upsampling_weights_sum_to_one_and_preserve_identity(self) -> None:
        upsampler = ConvexMapUpsampler(8, hidden_channels=8, scale=4).eval()
        low = canonical_backward_map(1, (3, 4), (12, 16))
        output = upsampler(
            low,
            torch.randn(1, 8, 3, 4),
            output_size=(12, 16),
            source_size=(12, 16),
        )
        expected = canonical_backward_map(1, (12, 16), (12, 16))
        torch.testing.assert_close(output["backward_map"], expected, atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(
            output["convex_mask"].sum(dim=2),
            torch.ones_like(output["convex_mask"][:, :, 0]),
        )


if __name__ == "__main__":
    unittest.main()

