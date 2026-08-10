import unittest

try:
    import torch
except ImportError:  # pragma: no cover - source-only environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class InferenceCanvasTest(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio(self) -> None:
        from diffusion2raft.infer import compute_canvas_transform

        transform = compute_canvas_transform((300, 100), (512, 512), "letterbox")
        self.assertEqual(transform.content_size[0], 512)
        self.assertLessEqual(
            abs(transform.content_size[0] / transform.content_size[1] - 3.0),
            0.02,
        )
        self.assertGreater(transform.content_left, 0)

    def test_letterboxed_identity_flow_restores_native_identity(self) -> None:
        from diffusion2raft.infer import compute_canvas_transform, flow_from_model_canvas

        native_size = (300, 100)
        transform = compute_canvas_transform(native_size, (512, 512), "letterbox")
        identity = torch.zeros((1, 2, 512, 512), dtype=torch.float32)
        restored = flow_from_model_canvas(identity, transform, native_size)
        self.assertLess(float(restored.abs().max()), 2e-4)

    def test_stretch_identity_is_backward_compatible(self) -> None:
        from diffusion2raft.infer import compute_canvas_transform, flow_from_model_canvas

        native_size = (123, 321)
        transform = compute_canvas_transform(native_size, (64, 96), "stretch")
        identity = torch.zeros((1, 2, 64, 96), dtype=torch.float32)
        restored = flow_from_model_canvas(identity, transform, native_size)
        self.assertLess(float(restored.abs().max()), 2e-4)

    def test_letterbox_translation_uses_native_pixel_units(self) -> None:
        from diffusion2raft.infer import compute_canvas_transform, flow_from_model_canvas

        native_size = (300, 100)
        transform = compute_canvas_transform(native_size, (512, 512), "letterbox")
        flow = torch.zeros((1, 2, 512, 512), dtype=torch.float32)
        expected_x = 10.0
        flow[:, 0] = (
            expected_x
            * (transform.content_size[1] - 1)
            / (native_size[1] - 1)
        )
        restored = flow_from_model_canvas(flow, transform, native_size)
        self.assertLess(float((restored[:, 0] - expected_x).abs().max()), 2e-4)
        self.assertLess(float(restored[:, 1].abs().max()), 2e-4)

    def test_residual_crop_removes_letterbox_padding(self) -> None:
        from diffusion2raft.infer import (
            compute_canvas_transform,
            residual_from_model_canvas,
        )

        native_size = (300, 100)
        transform = compute_canvas_transform(native_size, (512, 512), "letterbox")
        residual = torch.zeros((1, 2, 512, 512), dtype=torch.float32)
        expected_y = 12.0
        residual[:, 1] = (
            expected_y
            * (transform.content_size[0] - 1)
            / (native_size[0] - 1)
        )
        restored = residual_from_model_canvas(residual, transform, native_size)
        self.assertLess(float(restored[:, 0].abs().max()), 2e-4)
        self.assertLess(float((restored[:, 1] - expected_y).abs().max()), 2e-4)

    def test_image_padding_is_not_anisotropic(self) -> None:
        from diffusion2raft.infer import compute_canvas_transform, image_to_model_canvas

        native = torch.rand((1, 3, 300, 100))
        transform = compute_canvas_transform((300, 100), (512, 512), "letterbox")
        canvas = image_to_model_canvas(native, transform, padding_mode="replicate")
        self.assertEqual(tuple(canvas.shape), (1, 3, 512, 512))


if __name__ == "__main__":
    unittest.main()
