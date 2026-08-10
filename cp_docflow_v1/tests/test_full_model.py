from __future__ import annotations

import unittest
import sys
import types
from unittest import mock

import torch

from cp_docflow.geometry import canonical_backward_map
from cp_docflow.losses import CPDocFlowLoss
from cp_docflow.models.coordinate_flow_transformer import confidence_protected_start
from cp_docflow.models.docgrid_flow import CPDocFlow
from cp_docflow.models.qwen_feature_probe import FrozenQwenImageEditFeatureSource


class FullModelTest(unittest.TestCase):
    @staticmethod
    def _model(**overrides: object) -> CPDocFlow:
        config: dict[str, object] = {
            "coarse": {"base_channels": 8, "feature_channels": 16},
            "qwen_backend": "none",
            "qwen_feature_channels": 12,
            "fusion_channels": 16,
            "hv_channels": 8,
            "velocity_hidden_channels": 16,
            "velocity_time_channels": 16,
            "sigma_max": 0.0,
            "fm_steps": 2,
            "refiner_hidden_channels": 16,
            "refiner_iterations": 2,
            "refiner_max_step_px": 1.0,
        }
        config.update(overrides)
        return CPDocFlow(**config)

    def test_zero_initialized_full_graph_preserves_source_pixels(self) -> None:
        image = torch.rand(1, 3, 32, 40)
        output = self._model().eval()(image)
        identity = canonical_backward_map(1, (32, 40))
        torch.testing.assert_close(output["backward_map"], identity, atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(output["rectified_image"], image, atol=2.0e-6, rtol=0.0)
        self.assertEqual(len(output["flow_matching_sequence"]), 2)
        self.assertEqual(len(output["refiner_sequence"]), 2)

    def test_training_target_is_supervision_not_encoder_input(self) -> None:
        torch.manual_seed(4)
        image = torch.rand(1, 3, 32, 32)
        target = canonical_backward_map(1, (32, 32)).clone()
        target[:, 0] += 1.0
        valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
        model = self._model().train()
        prediction = model(
            image,
            target_map=target,
            valid_mask=valid,
            time=torch.tensor([0.25]),
        )
        losses = CPDocFlowLoss()(
            prediction,
            {
                "warped_image": image,
                "backward_map": target,
                "valid_mask": valid,
            },
        )
        losses["total"].backward()
        self.assertIsNotNone(model.velocity.head.weight.grad)
        self.assertIsNotNone(model.refiner.delta[-1].weight.grad)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(tuple(prediction["velocity_target"].shape), (1, 2, 4, 4))

    def test_confidence_protected_noise_only_changes_uncertain_pixels(self) -> None:
        coarse = torch.zeros(1, 2, 2, 2)
        confidence = torch.tensor([[[[1.0, 0.0], [0.5, 1.0]]]])
        noise = torch.ones_like(coarse)
        start = confidence_protected_start(
            coarse, confidence, sigma_max=4.0, noise=noise
        )
        torch.testing.assert_close(start[..., 0, 0], torch.zeros(1, 2))
        torch.testing.assert_close(start[..., 0, 1], torch.full((1, 2), 4.0))
        torch.testing.assert_close(start[..., 1, 0], torch.full((1, 2), 2.0))

    def test_lite_qwen_uses_same_adapter_and_gate_contract(self) -> None:
        model = self._model(
            qwen_backend="lite",
            qwen={"hidden_channels": 16, "feature_layers": [-3, -2, -1]},
        ).eval()
        output = model(torch.rand(1, 3, 32, 32), render=False)
        self.assertEqual(output["qwen_backend"], "lite")
        self.assertEqual(tuple(output["qwen_gate"].shape), (1, 1, 4, 4))
        self.assertEqual(tuple(output["backward_map"].shape), (1, 2, 32, 32))

    def test_refiner_can_target_a_different_canvas(self) -> None:
        output = self._model().eval()(
            torch.rand(1, 3, 32, 40), output_size=(24, 28), render=False
        )
        self.assertEqual(tuple(output["backward_map"].shape), (1, 2, 24, 28))

    def test_zero_initialized_model_renders_a_target_window_without_rebasing(self) -> None:
        image = torch.rand(1, 3, 32, 40)
        window = torch.tensor([[5.0, 7.0, 20.0, 16.0]])
        output = self._model().eval()(
            image,
            output_size=(16, 20),
            target_canvas_size=(32, 40),
            target_window=window,
        )
        expected_map = canonical_backward_map(1, (16, 20)).clone()
        expected_map[:, 0] += 5.0
        expected_map[:, 1] += 7.0
        torch.testing.assert_close(
            output["backward_map"], expected_map, atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            output["rectified_image"],
            image[..., 7:23, 5:25],
            atol=2.0e-6,
            rtol=0.0,
        )

    def test_bilinear_upsampling_ablation_preserves_the_coordinate_contract(self) -> None:
        image = torch.rand(1, 3, 32, 40)
        output = self._model(upsampling_mode="bilinear").eval()(
            image, render=True, profile=True
        )
        identity = canonical_backward_map(1, (32, 40))
        torch.testing.assert_close(
            output["backward_map"], identity, atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            output["rectified_image"], image, atol=2.0e-6, rtol=0.0
        )
        self.assertEqual(output["upsampling_mode"], "bilinear")
        self.assertIn("convex_upsampling_seconds", output["runtime_breakdown"])

    def test_hv_off_removes_condition_from_fusion_cft_and_warr(self) -> None:
        model = self._model(enable_hv_condition=False).eval()
        fusion_inputs: list[torch.Tensor] = []
        cft_inputs: list[torch.Tensor] = []
        warr_inputs: list[torch.Tensor] = []

        fusion_original = model.fusion.forward
        cft_original = model.velocity.forward
        warr_original = model.refiner.forward

        def fusion_spy(cnn: torch.Tensor, qwen: torch.Tensor, hv: torch.Tensor):
            fusion_inputs.append(hv)
            return fusion_original(cnn, qwen, hv)

        def cft_spy(*args: object, **kwargs: object):
            cft_inputs.append(kwargs["hv_condition"])
            return cft_original(*args, **kwargs)

        def warr_spy(*args: object, **kwargs: object):
            warr_inputs.append(kwargs["hv_feature"])
            return warr_original(*args, **kwargs)

        with mock.patch.object(model.fusion, "forward", side_effect=fusion_spy), mock.patch.object(
            model.velocity, "forward", side_effect=cft_spy
        ), mock.patch.object(model.refiner, "forward", side_effect=warr_spy):
            model(torch.rand(1, 3, 32, 32), render=False)
        self.assertTrue(fusion_inputs and cft_inputs and warr_inputs)
        for value in (*fusion_inputs, *cft_inputs, *warr_inputs):
            self.assertEqual(int(torch.count_nonzero(value)), 0)

    def test_frozen_qwen_path_requests_latent_and_never_decodes_vae(self) -> None:
        class DummyBlock(torch.nn.Module):
            def forward(self, text: torch.Tensor, image: torch.Tensor):
                return text, image

        class DummyTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner_dim = 32
                self.transformer_blocks = torch.nn.ModuleList(
                    [DummyBlock() for _ in range(4)]
                )

        class DummyVAE(torch.nn.Module):
            def decode(self, *_args: object, **_kwargs: object) -> None:
                raise AssertionError("the Qwen VAE decoder must never run")

        class QwenImageEditPipeline:
            def __init__(self) -> None:
                self.transformer = DummyTransformer()
                self.vae = DummyVAE()
                self.text_encoder = torch.nn.Identity()
                self.vae_scale_factor = 8
                self.last_kwargs: dict[str, object] = {}

            @classmethod
            def from_pretrained(cls, *_args: object, **_kwargs: object):
                return cls()

            def to(self, _device: torch.device):
                return self

            def set_progress_bar_config(self, **_kwargs: object) -> None:
                return None

            def __call__(self, **kwargs: object) -> object:
                self.last_kwargs = kwargs
                text = torch.zeros(1, 1, 32)
                # At 32x32, target_count=(32/8/2)^2=4; add four source tokens.
                image = torch.zeros(1, 8, 32)
                for block in self.transformer.transformer_blocks:
                    text, image = block(text, image)
                return object()

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.QwenImageEditPipeline = QwenImageEditPipeline
        source = FrozenQwenImageEditFeatureSource(
            {
                "model_id": "/fake/Qwen-Image-Edit",
                "hidden_channels": 32,
                "feature_layers": [-3, -2, -1],
                "feature_num_inference_steps": 1,
            }
        )
        with mock.patch.dict(sys.modules, {"diffusers": fake_diffusers}):
            features = source(torch.rand(1, 3, 32, 32))
        self.assertEqual(len(features), 3)
        self.assertEqual(tuple(features[0][0].shape), (1, 32, 2, 2))
        self.assertEqual(tuple(features[0][1].shape), (1, 32, 2, 2))
        self.assertEqual(source.pipeline.last_kwargs["output_type"], "latent")


if __name__ == "__main__":
    unittest.main()
