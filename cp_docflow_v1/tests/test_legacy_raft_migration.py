from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cp_docflow.checkpoint import file_sha256
from cp_docflow.data import DocumentMapDataset
from cp_docflow.geometry import canonical_backward_map
from cp_docflow.migrate_legacy_raft import migrate_legacy_raft_csv


class LegacyRaftMigrationTest(unittest.TestCase):
    def test_pseudo_displacement_is_materialized_as_native_absolute_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
            Image.fromarray(image).save(root / "rectified.png")
            Image.fromarray(image).save(root / "warped.png")
            np.save(root / "flow.npy", np.zeros((2, 16, 20), dtype=np.float32))
            checkpoint = root / "raft.pth"
            checkpoint.write_bytes(b"frozen raft weights")
            generator = root / "generate_flow.py"
            generator.write_text("# frozen generator\n", encoding="utf-8")
            source_csv = root / "legacy.csv"
            with source_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("image", "edit_image", "category", "flow_gt_path"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image": "rectified.png",
                        "edit_image": "warped.png",
                        "category": "doc",
                        "flow_gt_path": "flow.npy",
                    }
                )
            manifest = root / "train.jsonl"
            report = migrate_legacy_raft_csv(
                source_csv,
                manifest,
                root / "maps",
                label_checkpoint=checkpoint,
                generator_script=generator,
                flow_source_size=(16, 20),
            )
            self.assertFalse(report["gate_eligible"])
            self.assertEqual(report["label_provenance"], "raft_pseudo")
            record = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(record["label_checkpoint_sha256"], file_sha256(checkpoint))
            self.assertEqual(record["legacy_flow_format"], "backward_displacement_xy")
            dataset = DocumentMapDataset(manifest)
            sample = dataset[0]
            expected = canonical_backward_map(1, (8, 10)).squeeze(0)
            torch.testing.assert_close(
                sample["backward_map"], expected, atol=1.0e-6, rtol=0.0
            )
            self.assertTrue(bool(sample["valid_mask"].all()))

    def test_existing_manifest_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "existing.jsonl"
            manifest.write_text("owned by user\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                migrate_legacy_raft_csv(
                    root / "missing.csv",
                    manifest,
                    root / "maps",
                    label_checkpoint=root / "missing.pth",
                    generator_script=root / "missing.py",
                )


if __name__ == "__main__":
    unittest.main()
