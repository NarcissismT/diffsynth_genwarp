from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cp_docflow.checkpoint import checkpoint_payload
from cp_docflow.geometry import canonical_backward_map
from cp_docflow.infer import infer
from cp_docflow.models.coarse import DeterministicCoarseRectifier


class InferenceContractTest(unittest.TestCase):
    def test_native_image_is_sampled_once_without_qwen_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3)
            image_path = root / "page.png"
            Image.fromarray(array).save(image_path)
            config = {"base_channels": 8, "feature_channels": 16}
            model = DeterministicCoarseRectifier(**config)
            payload = checkpoint_payload(
                model,
                model_config=config,
                input_work_size=(32, 40),
                output_work_size=(32, 40),
                epoch=0,
            )
            checkpoint = root / "model.pt"
            torch.save(payload, checkpoint)
            metadata = infer(checkpoint, image_path, root / "output", device_name="cpu")
            result = np.asarray(Image.open(metadata["rectified_image"]))
            np.testing.assert_array_equal(result, array)
            backward_map = np.load(metadata["backward_map"])
            expected = (
                canonical_backward_map(1, (32, 40))
                .squeeze(0)
                .permute(1, 2, 0)
                .numpy()
            )
            np.testing.assert_allclose(backward_map, expected, atol=1.0e-6, rtol=0.0)
            self.assertFalse(metadata["qwen_vae_decoder_used"])
            self.assertEqual(
                metadata["final_rgb_source"],
                "single_grid_sample_from_native_warped_image",
            )


if __name__ == "__main__":
    unittest.main()

