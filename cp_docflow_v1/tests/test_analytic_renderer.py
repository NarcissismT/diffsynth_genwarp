from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cp_docflow.audit_data import audit_manifests
from cp_docflow.data import DocumentMapDataset, assert_document_disjoint
from cp_docflow.merge_analytic_shards import merge_analytic_shards
from cp_docflow.render_analytic_gt import render_analytic_dataset


def _write_flat_documents(root: Path, count: int) -> Path:
    source = root / "flat"
    source.mkdir()
    csv_path = root / "documents.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "category"))
        writer.writeheader()
        for index in range(count):
            yy, xx = np.mgrid[:24, :32]
            base = ((xx * 13 + yy * 7 + index * 11) % 256).astype(np.uint8)
            array = np.stack(
                (base, np.roll(base, 2, 0), np.roll(base, 3, 1)), axis=-1
            )
            image_path = source / f"document-{index:03d}.png"
            Image.fromarray(array).save(image_path)
            writer.writerow({"image": str(image_path), "category": "fixture"})
    return csv_path


class AnalyticRendererTest(unittest.TestCase):
    def test_renderer_creates_verified_document_disjoint_absolute_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = _write_flat_documents(root, 40)
            report = render_analytic_dataset(
                csv_path,
                root / "rendered",
                variants_per_document=1,
                seed=9,
                split_ratios=(0.6, 0.2, 0.2),
                output_size=(24, 32),
                device_name="cpu",
            )
            self.assertEqual(sum(report["documents"].values()), 40)
            self.assertLess(report["oracle_rgb_l1_mean"], 0.20)
            datasets = {
                name: DocumentMapDataset(path)
                for name, path in report["manifests"].items()
            }
            assert_document_disjoint(*(dataset.records for dataset in datasets.values()))
            for dataset in datasets.values():
                sample = dataset[0]
                self.assertEqual(sample["label_provenance"], "analytic_gt")
                self.assertEqual(tuple(sample["backward_map"].shape), (2, 24, 32))
                self.assertTrue(bool(sample["valid_mask"].any()))
                self.assertIn("horizontal_structure", sample)
            audit = audit_manifests(
                report["manifests"],
                root / "audit",
                allowed_label_provenance={"analytic_gt"},
                seeds=[1337, 2027, 3407],
            )
            self.assertTrue(audit["verified_gt_only"])
            self.assertTrue(audit["document_disjoint_verified"])

    def test_renderer_shards_merge_into_one_auditable_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = _write_flat_documents(root, 12)
            shard_root = root / "shards"
            for shard_index in range(2):
                report = render_analytic_dataset(
                    csv_path,
                    shard_root / f"shard-{shard_index:05d}-of-00002",
                    variants_per_document=1,
                    seed=9,
                    split_ratios=(0.6, 0.2, 0.2),
                    output_size=(24, 32),
                    device_name="cpu",
                    shard_index=shard_index,
                    num_shards=2,
                )
                self.assertEqual(report["selected_document_count_before_sharding"], 12)
                self.assertEqual(sum(report["documents"].values()), 6)

            merged = merge_analytic_shards(shard_root, root / "merged")
            self.assertEqual(sum(merged["documents"].values()), 12)
            self.assertEqual(sum(merged["samples"].values()), 12)
            datasets = {
                split: DocumentMapDataset(manifest)
                for split, manifest in merged["manifests"].items()
            }
            assert_document_disjoint(*(dataset.records for dataset in datasets.values()))
            audit = audit_manifests(
                merged["manifests"],
                root / "sharded_audit",
                allowed_label_provenance={"analytic_gt"},
                seeds=[1337, 2027, 3407],
            )
            self.assertTrue(audit["verified_gt_only"])
            self.assertTrue(audit["document_disjoint_verified"])


if __name__ == "__main__":
    unittest.main()
