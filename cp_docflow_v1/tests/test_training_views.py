from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cp_docflow.data import DocumentMapDataset
from cp_docflow.make_smoke_data import _write_split
from cp_docflow.training_views import FullPageMixedViewDataset


class TrainingViewsTest(unittest.TestCase):
    @staticmethod
    def _config(view: str) -> dict[str, object]:
        return {
            "low_input_size": [32, 32],
            "low_output_size": [32, 32],
            "structure_patch_size": [32, 32],
            "probabilities": {
                "low_page": 1.0 if view == "low_page" else 0.0,
                "structure_patch": 1.0 if view == "structure_patch" else 0.0,
                "full_page": 1.0 if view == "full_page" else 0.0,
            },
            "samples_per_epoch_multiplier": 1.0,
        }

    def test_three_stage5_views_have_explicit_coordinate_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_split(Path(directory), "train", 1, (64, 64), 0)
            base = DocumentMapDataset(
                manifest, input_work_size=(64, 64), output_work_size=(64, 64)
            )
            for view in FullPageMixedViewDataset.VIEW_NAMES:
                mixed = FullPageMixedViewDataset(base, self._config(view), seed=5)
                sample = mixed[0]
                self.assertEqual(sample["training_view"], view)
                self.assertIn("target_canvas_size", sample)
                self.assertIn("target_window", sample)
                self.assertTrue(bool(sample["valid_mask"].any()))
                for key in (
                    "horizontal_structure",
                    "vertical_structure",
                    "boundary_structure",
                ):
                    self.assertEqual(
                        tuple(sample[key].shape[-2:]),
                        tuple(sample["backward_map"].shape[-2:]),
                    )

    def test_structure_patch_does_not_rebase_source_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_split(Path(directory), "train", 1, (64, 64), 0)
            base = DocumentMapDataset(manifest)
            mixed = FullPageMixedViewDataset(
                base, self._config("structure_patch"), seed=9
            )
            patch = mixed[0]
            x0, y0, width, height = [int(value) for value in patch["target_window"]]
            expected = base[0]["backward_map"][..., y0 : y0 + height, x0 : x0 + width]
            torch.testing.assert_close(patch["backward_map"], expected)
            self.assertEqual(tuple(patch["warped_image"].shape[-2:]), (64, 64))
            self.assertEqual(tuple(patch["target_canvas_size"]), (64, 64))


if __name__ == "__main__":
    unittest.main()

