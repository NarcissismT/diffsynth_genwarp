from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional legacy deployment dependency
    cv2 = None

try:
    import torch
except ImportError:  # pragma: no cover - lightweight source-only environments
    torch = None

from PIL import Image


if torch is not None:

    class _ConstantBgrLama(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.color = torch.nn.Parameter(
                # Values land above the half-byte boundary, so the session
                # test distinguishes historical truncation from rounding.
                torch.tensor((0.203, 0.403, 0.803)).view(1, 3, 1, 1)
            )

        def forward(self, image, mask):
            del mask
            return torch.ones_like(image) * self.color


    class _AllInvalidRectifier(torch.nn.Module):
        prior_backend = "learned"

        def set_correlation_temperature(self, value):
            self.correlation_temperature = float(value)

        def forward(self, warped, guide=None, *, stage="unified"):
            del guide
            batch, _, height, width = warped.shape
            prior = torch.zeros(
                batch, 2, height, width, device=warped.device, dtype=torch.float32
            )
            final = prior.clone()
            final[:, 0] = 100.0
            return {
                "stage": stage,
                "prior_flow": prior,
                "final_flow": final,
                "residuals": [],
            }


    class _LastColumnInvalidRectifier(torch.nn.Module):
        prior_backend = "learned"

        def set_correlation_temperature(self, value):
            self.correlation_temperature = float(value)

        def forward(self, warped, guide=None, *, stage="unified"):
            del guide
            batch, _, height, width = warped.shape
            flow = torch.zeros(
                batch, 2, height, width, device=warped.device, dtype=torch.float32
            )
            flow[:, 0, :, -1] = float(width)
            return {
                "stage": stage,
                "prior_flow": torch.zeros_like(flow),
                "final_flow": flow,
                "residuals": [],
            }


@unittest.skipIf(torch is None or cv2 is None, "PyTorch and OpenCV are required")
class Corrected512PreprocessingTest(unittest.TestCase):
    @staticmethod
    def _pattern(height: int, width: int) -> np.ndarray:
        y = np.arange(height, dtype=np.uint16)[:, None]
        x = np.arange(width, dtype=np.uint16)[None, :]
        return np.stack(
            (
                np.broadcast_to((3 * x + 5 * y) % 256, (height, width)),
                np.broadcast_to((7 * x + 2 * y + 11) % 256, (height, width)),
                np.broadcast_to((x + 9 * y + 23) % 256, (height, width)),
            ),
            axis=-1,
        ).astype(np.uint8)

    @staticmethod
    def _reference_resize(rgb: np.ndarray, target: int) -> np.ndarray:
        height, width = rgb.shape[:2]
        if min(height, width) > 2048:
            scale = 1024.0 / min(height, width)
            middle_width = int(width * scale)
            middle_height = int(height * scale)
            rgb = cv2.resize(
                rgb,
                (middle_width, middle_height),
                interpolation=cv2.INTER_AREA,
            )
            return cv2.resize(
                rgb, (target, target), interpolation=cv2.INTER_AREA
            )
        interpolation = (
            cv2.INTER_AREA if min(height, width) > target else cv2.INTER_LINEAR
        )
        return cv2.resize(rgb, (target, target), interpolation=interpolation)

    def test_opencv_decoder_preserves_cv2_rgb_bytes(self) -> None:
        from diffusion2raft.infer import _load_image

        rgb = self._pattern(37, 53)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            self.assertTrue(cv2.imwrite(str(path), rgb[:, :, ::-1]))
            tensor, _ = _load_image(path, decoder="opencv")
        restored = (
            tensor.squeeze(0)
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .byte()
            .numpy()
        )
        self.assertTrue(np.array_equal(restored, rgb))

    def test_all_three_opencv_resize_branches_match_reference(self) -> None:
        from diffusion2raft.infer import _opencv_baseline_resize

        for height, width in ((127, 193), (600, 901), (2049, 2053)):
            with self.subTest(size=(height, width)):
                rgb = self._pattern(height, width)
                tensor = (
                    torch.from_numpy(rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .div(255.0)
                )
                actual = _opencv_baseline_resize(tensor, (512, 512))
                actual_u8 = (
                    actual.squeeze(0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .round()
                    .byte()
                    .numpy()
                )
                expected = self._reference_resize(rgb, 512)
                self.assertTrue(np.array_equal(actual_u8, expected))

    def test_default_resize_is_unchanged_and_opencv_mode_is_restricted(self) -> None:
        from diffusion2raft.infer import (
            compute_canvas_transform,
            image_to_model_canvas,
        )

        image = torch.rand(1, 3, 19, 31)
        transform = compute_canvas_transform((19, 31), (32, 48), "stretch")
        actual = image_to_model_canvas(image, transform)
        expected = torch.nn.functional.interpolate(
            image, size=(32, 48), mode="bilinear", align_corners=True
        )
        self.assertTrue(torch.equal(actual, expected))

        with self.assertRaisesRegex(ValueError, "square model canvas"):
            image_to_model_canvas(
                image,
                transform,
                resize_interpolation="opencv_baseline",
            )
        letterbox = compute_canvas_transform((19, 31), (32, 32), "letterbox")
        with self.assertRaisesRegex(ValueError, "only supported"):
            image_to_model_canvas(
                image,
                letterbox,
                resize_interpolation="opencv_baseline",
            )


@unittest.skipIf(torch is None or cv2 is None, "PyTorch and OpenCV are required")
class Corrected512LamaSessionTest(unittest.TestCase):
    @staticmethod
    def _save_lama(path: Path, size: int) -> None:
        image = torch.rand(1, 3, size, size)
        mask = torch.zeros(1, 1, size, size)
        traced = torch.jit.trace(
            _ConstantBgrLama().eval(), (image, mask), strict=False
        )
        torch.jit.save(traced, str(path))

    def test_session_saves_raw_then_inpainted_final_and_explicit_prior(self) -> None:
        from diffusion2raft.infer import RectificationSession

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.png"
            checkpoint_path = root / "rectifier.pt"
            lama_path = root / "lama.pt"
            output_dir = root / "outputs"

            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            rgb[..., 0] = 25
            rgb[..., 1] = 75
            rgb[..., 2] = 125
            self.assertTrue(cv2.imwrite(str(image_path), rgb[:, :, ::-1]))
            torch.save(
                {
                    "model": {},
                    "stage": "unified",
                    "epoch": 0,
                    "config": {"model": {"correlation_temperature": 1.0}},
                    "correlation_temperature": 1.0,
                },
                checkpoint_path,
            )
            self._save_lama(lama_path, 16)

            config = {
                "device": "cpu",
                "data": {"work_size": [8, 8]},
                "model": {"feature_backend": "lite"},
                "inference": {
                    "resize_policy": "stretch",
                    "padding_mode": "replicate",
                    "image_decoder": "opencv",
                    "resize_interpolation": "opencv_baseline",
                    "inpaint": {
                        "enabled": True,
                        "path": str(lama_path),
                        "sha256": hashlib.sha256(
                            lama_path.read_bytes()
                        ).hexdigest(),
                        "size": 16,
                        "dilation": 11,
                    },
                },
            }
            fake_rectifier = _AllInvalidRectifier()
            with mock.patch(
                "diffusion2raft.infer.build_rectifier", return_value=fake_rectifier
            ):
                session = RectificationSession(config, checkpoint_path)
                paths = session.rectify(image_path, None, output_dir)

            raw = np.asarray(Image.open(paths["raw_image"]).convert("RGB"))
            final = np.asarray(Image.open(paths["image"]).convert("RGB"))
            prior = np.asarray(Image.open(paths["prior_image"]).convert("RGB"))
            valid = np.asarray(Image.open(paths["valid"]).convert("RGB"))
            inpaint_mask = np.asarray(
                Image.open(paths["inpaint_mask"]).convert("RGB")
            )
            evaluation_valid = np.asarray(
                Image.open(paths["evaluation_valid"]).convert("RGB")
            )
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

            self.assertTrue(np.array_equal(raw, np.zeros_like(raw)))
            expected_final = np.empty_like(final)
            expected_final[..., 0] = 204
            expected_final[..., 1] = 102
            expected_final[..., 2] = 51
            self.assertTrue(np.array_equal(final, expected_final))
            self.assertTrue(np.array_equal(prior, rgb))
            self.assertFalse(valid.any())
            self.assertTrue(inpaint_mask.all())
            self.assertFalse(evaluation_valid.any())
            self.assertEqual(metadata["image_decoder"], "opencv")
            self.assertEqual(metadata["resize_interpolation"], "opencv_baseline")
            self.assertEqual(metadata["final_raw_padding_mode"], "zeros")
            self.assertTrue(metadata["final_image_inpainted"])
            self.assertEqual(
                metadata["prior_diagnostic"],
                {"padding_mode": "border", "inpainted": False},
            )
            identity = metadata["inpaint_identity"]
            self.assertTrue(identity["enabled"])
            self.assertEqual(identity["input_size"], 16)
            self.assertEqual(identity["dilation_kernel"], 11)
            self.assertEqual(identity["size_bytes"], lama_path.stat().st_size)
            self.assertEqual(
                metadata["mask_semantics"]["valid"],
                "flow_valid_before_inpainting",
            )
            self.assertEqual(metadata["inpaint_fraction"], 1.0)
            self.assertEqual(metadata["evaluation_valid_fraction"], 0.0)
            self.assertIsNone(metadata["fold_rate"])
            self.assertIsNone(metadata["jacobian_p01"])
            self.assertNotIn("NaN", paths["metadata"].read_text(encoding="utf-8"))

    def test_partial_invalid_region_exposes_composite_and_evaluation_masks(self) -> None:
        from diffusion2raft.infer import RectificationSession

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.png"
            checkpoint_path = root / "rectifier.pt"
            lama_path = root / "lama.pt"
            output_dir = root / "outputs"
            rgb = np.full((32, 32, 3), 127, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), rgb[:, :, ::-1]))
            torch.save(
                {
                    "model": {},
                    "stage": "unified",
                    "epoch": 0,
                    "config": {"model": {"correlation_temperature": 1.0}},
                    "correlation_temperature": 1.0,
                },
                checkpoint_path,
            )
            self._save_lama(lama_path, 16)
            config = {
                "device": "cpu",
                "data": {"work_size": [32, 32]},
                "model": {"feature_backend": "lite"},
                "inference": {
                    "resize_policy": "stretch",
                    "image_decoder": "opencv",
                    "resize_interpolation": "opencv_baseline",
                    "inpaint": {
                        "enabled": True,
                        "path": str(lama_path),
                        "sha256": hashlib.sha256(
                            lama_path.read_bytes()
                        ).hexdigest(),
                        "size": 16,
                        "dilation": 11,
                    },
                },
            }
            with mock.patch(
                "diffusion2raft.infer.build_rectifier",
                return_value=_LastColumnInvalidRectifier(),
            ):
                paths = RectificationSession(
                    config, checkpoint_path
                ).rectify(image_path, None, output_dir)

            flow_valid = np.asarray(Image.open(paths["valid"]).convert("L")) > 0
            inpaint_mask = (
                np.asarray(Image.open(paths["inpaint_mask"]).convert("L")) > 0
            )
            evaluation_valid = (
                np.asarray(Image.open(paths["evaluation_valid"]).convert("L")) > 0
            )
            self.assertTrue(flow_valid[:, :-1].all())
            self.assertFalse(flow_valid[:, -1].any())
            self.assertFalse(inpaint_mask[:, :-6].any())
            self.assertTrue(inpaint_mask[:, -6:].all())
            self.assertTrue(evaluation_valid[:, :-6].all())
            self.assertFalse(evaluation_valid[:, -6:].any())


if __name__ == "__main__":
    unittest.main()
