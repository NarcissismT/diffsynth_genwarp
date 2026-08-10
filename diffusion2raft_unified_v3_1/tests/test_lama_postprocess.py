from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - lightweight source-only environments
    torch = None


if torch is not None:

    class _ChannelSwapLama(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # A real parameter verifies that the external scripted model is
            # frozen and absent from the wrapper's parameters/state_dict.
            self.gain = torch.nn.Parameter(torch.ones(()))

        def forward(self, image, mask):
            del mask
            return torch.cat(
                (image[:, 2:3], image[:, 1:2], image[:, 0:1]), dim=1
            ) * self.gain


    class _WrongShapeLama(torch.nn.Module):
        def forward(self, image, mask):
            del mask
            return image[:, :, :-1, :]


    class _NonFiniteLama(torch.nn.Module):
        def forward(self, image, mask):
            del mask
            return image / 0.0


    class _PreprocessProbeLama(torch.nn.Module):
        def forward(self, image, mask):
            return image * 0.5 + mask.expand_as(image) * 0.25


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchScriptLamaInpainterTest(unittest.TestCase):
    @staticmethod
    def _save_trace(
        module: "torch.nn.Module", path: Path, size: int = 512
    ) -> None:
        image = torch.rand(1, 3, size, size)
        mask = torch.zeros(1, 1, size, size)
        traced = torch.jit.trace(module.eval(), (image, mask), strict=False)
        torch.jit.save(traced, str(path))

    def test_only_dilated_mask_is_replaced_and_bgr_is_restored(self) -> None:
        from diffusion2raft.postprocess import TorchScriptLamaInpainter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "channel_swap.pt"
            self._save_trace(_ChannelSwapLama(), path)
            inpainter = TorchScriptLamaInpainter(path, device="cpu")

            image = torch.empty(1, 3, 21, 23)
            image[:, 0] = 0.10
            image[:, 1] = 0.40
            image[:, 2] = 0.80
            invalid = torch.zeros(1, 1, 21, 23, dtype=torch.bool)
            invalid[:, :, 10, 11] = True

            result, actual_mask = inpainter.forward_with_mask(image, invalid)
            expected_mask = torch.zeros_like(invalid)
            expected_mask[:, :, 5:16, 6:17] = True
            expected_fill = (
                torch.tensor((204, 102, 25), dtype=torch.float32)
                .div(255.0)
                .view(1, 3, 1, 1)
            )
            quantized = image.mul(255.0).to(torch.uint8).float().div(255.0)

            self.assertEqual(int(expected_mask.sum()), 11 * 11)
            self.assertTrue(torch.equal(actual_mask, expected_mask))
            self.assertTrue(
                torch.equal(
                    result.masked_select(~expected_mask),
                    quantized.masked_select(~expected_mask),
                )
            )
            expanded_mask = expected_mask.expand_as(result)
            expected_inside = expected_fill.expand_as(result).masked_select(expanded_mask)
            self.assertTrue(
                torch.allclose(
                    result.masked_select(expanded_mask),
                    expected_inside,
                    atol=1e-6,
                    rtol=0.0,
                )
            )

    def test_uint8_opencv_image_and_mask_preprocessing_matches_reference(self) -> None:
        try:
            import cv2
        except ImportError:  # pragma: no cover
            self.skipTest("OpenCV is required")
        from diffusion2raft.postprocess import TorchScriptLamaInpainter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            size = 16
            self._save_trace(_PreprocessProbeLama(), path, size=size)
            inpainter = TorchScriptLamaInpainter(
                path,
                device="cpu",
                inpaint_size=size,
                dilation_kernel=3,
            )
            height, width = 11, 13
            values = torch.arange(3 * height * width, dtype=torch.float32)
            image = values.view(1, 3, height, width).remainder(251).div(250.0)
            invalid = torch.zeros(1, 1, height, width, dtype=torch.bool)
            invalid[:, :, 4:7, 5:8] = True

            actual, actual_mask = inpainter.forward_with_mask(image, invalid)

            rgb_u8 = image.mul(255.0).to(torch.uint8)[0].permute(1, 2, 0).numpy()
            bgr_u8 = np.ascontiguousarray(rgb_u8[:, :, ::-1])
            resized_image = cv2.resize(
                bgr_u8, (size, size), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32) / 255.0
            invalid_u8 = invalid[0, 0].numpy().astype(np.uint8) * 255
            dilated_u8 = cv2.dilate(invalid_u8, np.ones((3, 3), np.uint8))
            resized_mask = cv2.resize(
                dilated_u8, (size, size), interpolation=cv2.INTER_LINEAR
            )
            binary_mask = (resized_mask > 100).astype(np.float32)
            probe_bgr = resized_image * 0.5 + binary_mask[..., None] * 0.25
            probe = torch.from_numpy(probe_bgr).permute(2, 0, 1).unsqueeze(0)
            probe = torch.nn.functional.interpolate(
                probe,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[:, (2, 1, 0)]
            expected_mask = torch.from_numpy(dilated_u8 > 0).view(
                1, 1, height, width
            )
            quantized = image.mul(255.0).to(torch.uint8).float().div(255.0)
            expected = torch.where(expected_mask, probe, quantized).clamp(0.0, 1.0)

            self.assertTrue(torch.equal(actual_mask, expected_mask))
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))

    def test_external_jit_is_frozen_and_absent_from_wrapper_state(self) -> None:
        from diffusion2raft.postprocess import TorchScriptLamaInpainter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameterized.pt"
            self._save_trace(_ChannelSwapLama(), path)
            inpainter = TorchScriptLamaInpainter(path, device="cpu")

            self.assertTrue(any(True for _ in inpainter.lama.parameters()))
            self.assertTrue(
                all(not parameter.requires_grad for parameter in inpainter.lama.parameters())
            )
            self.assertEqual(list(inpainter.parameters()), [])
            self.assertEqual(list(inpainter.state_dict()), [])
            self.assertEqual([name for name, _ in inpainter.named_modules()], [""])

            frozen_identity = dict(inpainter.identity)
            with path.open("ab") as stream:
                stream.write(b"changed-after-load")
            self.assertEqual(inpainter.identity, frozen_identity)

            inpainter.train()
            self.assertTrue(inpainter.training)
            self.assertFalse(inpainter.lama.training)

    def test_jit_load_reads_authenticated_procfd_not_replaced_path(self) -> None:
        from diffusion2raft.postprocess import TorchScriptLamaInpainter

        size = 16
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "lama.pt"
            replacement = root / "replacement.pt"
            backup = root / "original.pt"
            self._save_trace(_ChannelSwapLama(), path, size=size)
            self._save_trace(_WrongShapeLama(), replacement, size=size)
            image = torch.rand(1, 3, size, size)
            mask = torch.zeros(1, 1, size, size)
            real_load = torch.jit.load
            captured = {}

            def replace_then_load(load_path, *args, **kwargs):
                captured["load_path"] = str(load_path)
                path.rename(backup)
                replacement.rename(path)
                module = real_load(load_path, *args, **kwargs)
                captured["shape"] = tuple(module(image, mask).shape)
                return module

            with mock.patch("torch.jit.load", side_effect=replace_then_load):
                with self.assertRaisesRegex(RuntimeError, "changed while in use"):
                    TorchScriptLamaInpainter(
                        path, device="cpu", inpaint_size=size
                    )
            self.assertTrue(captured["load_path"].startswith("/proc/self/fd/"))
            self.assertEqual(captured["shape"], (1, 3, size, size))

    def test_wrong_shape_and_nonfinite_outputs_are_rejected(self) -> None:
        from diffusion2raft.postprocess import TorchScriptLamaInpainter

        image = torch.rand(1, 3, 9, 13)
        invalid = torch.zeros(1, 1, 9, 13, dtype=torch.bool)
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            wrong_path = directory_path / "wrong.pt"
            nonfinite_path = directory_path / "nonfinite.pt"
            self._save_trace(_WrongShapeLama(), wrong_path)
            self._save_trace(_NonFiniteLama(), nonfinite_path)

            wrong = TorchScriptLamaInpainter(wrong_path, device="cpu")
            nonfinite = TorchScriptLamaInpainter(nonfinite_path, device="cpu")
            with self.assertRaisesRegex(ValueError, "wrong output shape"):
                wrong(image, invalid)
            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                nonfinite(image, invalid)

    def test_expected_sha256_is_checked_before_jit_load(self) -> None:
        from diffusion2raft.postprocess import TorchScriptLamaInpainter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lama.pt"
            self._save_trace(_ChannelSwapLama(), path, size=16)
            with mock.patch("torch.jit.load") as jit_load:
                with self.assertRaisesRegex(
                    RuntimeError, "sha256 differs from configured expected digest"
                ):
                    TorchScriptLamaInpainter(
                        path,
                        device="cpu",
                        inpaint_size=16,
                        expected_sha256="0" * 64,
                    )
            jit_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
