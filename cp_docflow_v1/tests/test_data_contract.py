from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cp_docflow.data import (
    DocumentMapDataset,
    assert_document_disjoint,
    read_manifest,
)
from cp_docflow.geometry import canonical_backward_map


class DataContractTest(unittest.TestCase):
    def _fixture(self, root: Path, *, document_id: str = "doc-1") -> Path:
        height, width = 8, 10
        image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
        Image.fromarray(image).save(root / "warped.png")
        Image.fromarray(image).save(root / "rectified.png")
        backward_map = canonical_backward_map(1, (height, width))
        np.save(
            root / "map.npy",
            backward_map.squeeze(0).permute(1, 2, 0).numpy(),
        )
        np.save(root / "valid.npy", np.ones((height, width), dtype=np.uint8))
        record = {
            "sample_id": "sample-1",
            "document_id": document_id,
            "warp_severity": "light",
            "label_provenance": "synthetic_analytic",
            "label_source": "unit_test_fixture.v1",
            "map_direction": "output_to_warped_source",
            "coordinate_convention": "absolute_source_pixel_xy",
            "warped_image": "warped.png",
            "rectified_image": "rectified.png",
            "backward_map": "map.npy",
            "valid_mask": "valid.npy",
            "input_size": [height, width],
            "output_size": [height, width],
        }
        manifest = root / "manifest.jsonl"
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return manifest

    def test_load_and_resize_keeps_absolute_map_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._fixture(Path(directory))
            dataset = DocumentMapDataset(
                manifest,
                input_work_size=(16, 20),
                output_work_size=(16, 20),
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["warped_image"].shape), (3, 16, 20))
            self.assertEqual(sample["document_id"], "doc-1")
            self.assertEqual(sample["label_provenance"], "synthetic_analytic")
            expected = canonical_backward_map(1, (16, 20)).squeeze(0)
            torch.testing.assert_close(
                sample["backward_map"], expected, atol=1.0e-6, rtol=0.0
            )
            self.assertTrue(bool(sample["valid_mask"].all()))

    def test_required_provenance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._fixture(Path(directory))
            value = json.loads(manifest.read_text(encoding="utf-8"))
            del value["label_provenance"]
            manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "label_provenance"):
                read_manifest(manifest)

    def test_wrong_map_direction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._fixture(Path(directory))
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["map_direction"] = "warped_to_output_forward"
            manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "map_direction"):
                read_manifest(manifest)

    def test_raft_pseudo_requires_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._fixture(Path(directory))
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["label_provenance"] = "raft_pseudo"
            manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "label_checkpoint_sha256"):
                read_manifest(manifest)

    def test_document_split_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            train = read_manifest(self._fixture(Path(first), document_id="same-doc"))
            val = read_manifest(self._fixture(Path(second), document_id="same-doc"))
            with self.assertRaisesRegex(ValueError, "leaks across splits"):
                assert_document_disjoint(train, val)

    def test_sample_with_no_valid_map_pixels_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            np.save(root / "valid.npy", np.zeros((8, 10), dtype=np.uint8))
            dataset = DocumentMapDataset(manifest)
            with self.assertRaisesRegex(ValueError, "no valid map pixels"):
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
