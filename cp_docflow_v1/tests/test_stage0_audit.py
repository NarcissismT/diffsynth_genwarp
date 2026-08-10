from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cp_docflow.audit_data import audit_manifests
from cp_docflow.checkpoint import COORDINATE_CONTRACT, file_sha256
from cp_docflow.data import DocumentMapDataset
from cp_docflow.make_smoke_data import _write_split
from cp_docflow.train_full import _validate_frozen_data_contract


class Stage0AuditTest(unittest.TestCase):
    def test_audit_freezes_manifest_payload_and_document_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_split(root, "train", 2, (32, 32), 0)
            val = _write_split(root, "val", 1, (32, 32), 2)
            report = audit_manifests(
                {"train": train, "val": val},
                root / "audit",
                allowed_label_provenance={"synthetic_analytic"},
                seeds=[1, 2, 3],
            )
            self.assertTrue(report["document_disjoint_verified"])
            self.assertFalse(report["verified_gt_only"])
            config = {
                "_project_root": str(root),
                "data": {"frozen_contract": "audit/frozen_contract.json"},
            }
            datasets = {
                "train": DocumentMapDataset(train),
                "val": DocumentMapDataset(val),
            }
            identity = _validate_frozen_data_contract(
                config, datasets, enforce=False
            )
            self.assertIsNotNone(identity)
            image_path = root / "images" / "train-0000-warped.png"
            image = Image.open(image_path).convert("RGB")
            image.putpixel((0, 0), (1, 2, 3))
            image.save(image_path)
            with self.assertRaisesRegex(ValueError, "payload changed"):
                _validate_frozen_data_contract(config, datasets, enforce=False)

    def test_baseline_must_bind_the_same_validation_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_split(root, "train", 2, (32, 32), 0)
            val = _write_split(root, "val", 1, (32, 32), 2)
            checkpoint = root / "teacher.pt"
            checkpoint.write_bytes(b"teacher")
            config = root / "baseline.yaml"
            config.write_text("schema: fixture\n", encoding="utf-8")
            metrics = {
                "schema": "docgrid_flow.full_exploratory_evaluation.v3",
                "evaluation_role": "exploratory",
                "gate_eligible": False,
                "training_stage": "frozen_prior",
                "coordinate_contract": COORDINATE_CONTRACT,
                "manifest_sha256": "0" * 64,
                "checkpoint_sha256": file_sha256(checkpoint),
                "evaluation_input_work_size": [32, 32],
                "evaluation_output_work_size": [32, 32],
                "baseline_identity": {
                    "config_sha256": file_sha256(config),
                    "checkpoint_sha256": file_sha256(checkpoint),
                },
            }
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen val manifest"):
                audit_manifests(
                    {"train": train, "val": val},
                    root / "audit",
                    allowed_label_provenance={"synthetic_analytic"},
                    seeds=[1, 2, 3],
                    baseline_checkpoint=checkpoint,
                    baseline_config=config,
                    baseline_metrics=metrics_path,
                )


if __name__ == "__main__":
    unittest.main()
