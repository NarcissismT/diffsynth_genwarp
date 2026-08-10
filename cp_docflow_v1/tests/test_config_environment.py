from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cp_docflow.config import load_config


class ConfigEnvironmentTest(unittest.TestCase):
    def test_docgrid_stage_configs_have_stable_defaults_and_env_overrides(self) -> None:
        root = Path(__file__).resolve().parents[1]
        clean_environment = dict(os.environ)
        for key in (
            "DOCGRID_TRAIN_MANIFEST",
            "DOCGRID_VAL_MANIFEST",
            "DOCGRID_FROZEN_CONTRACT",
            "DOCGRID_QWEN_MODEL",
            "DOCGRID_STAGE2_PARENT_CHECKPOINT",
            "DOCGRID_GATE1_RECEIPT",
            "DOCGRID_STAGE5_TRAIN_MANIFEST",
            "DOCGRID_STAGE5_VAL_MANIFEST",
            "DOCGRID_STAGE5_FROZEN_CONTRACT",
            "DOCGRID_STAGE5_PARENT_CHECKPOINT",
        ):
            clean_environment.pop(key, None)
        with patch.dict(os.environ, clean_environment, clear=True):
            stage2 = load_config(root / "configs/docgrid_v2/stage2_warr.yaml")
            self.assertEqual(stage2["data"]["train_manifest"], "data/train_docgrid_v2.jsonl")
            self.assertFalse(stage2["model"]["instantiate_qwen_adapter"])
            self.assertEqual(
                stage2["train"]["parent_checkpoint"],
                "runs/docgrid_v2/stage1_coarse/seed-1337/best.pt",
            )

        overrides = {
            "DOCGRID_TRAIN_MANIFEST": "/datasets/merged/train.jsonl",
            "DOCGRID_VAL_MANIFEST": "/datasets/merged/val.jsonl",
            "DOCGRID_FROZEN_CONTRACT": "/runs/audit/frozen_contract.json",
            "DOCGRID_QWEN_MODEL": "/models/Qwen-Image-Edit",
            "DOCGRID_STAGE2_PARENT_CHECKPOINT": "/runs/stage1/best.pt",
            "DOCGRID_GATE1_RECEIPT": "/runs/gates/gate1.json",
            "DOCGRID_STAGE5_TRAIN_MANIFEST": "/datasets/full/train.jsonl",
            "DOCGRID_STAGE5_VAL_MANIFEST": "/datasets/full/val.jsonl",
            "DOCGRID_STAGE5_FROZEN_CONTRACT": "/runs/full_audit/frozen_contract.json",
            "DOCGRID_STAGE5_PARENT_CHECKPOINT": "/runs/stage4/best.pt",
        }
        with patch.dict(os.environ, overrides, clear=False):
            stage2 = load_config(root / "configs/docgrid_v2/stage2_warr.yaml")
            self.assertEqual(stage2["data"]["train_manifest"], overrides["DOCGRID_TRAIN_MANIFEST"])
            self.assertEqual(stage2["data"]["val_manifest"], overrides["DOCGRID_VAL_MANIFEST"])
            self.assertEqual(stage2["data"]["frozen_contract"], overrides["DOCGRID_FROZEN_CONTRACT"])
            self.assertEqual(stage2["model"]["qwen"]["model_id"], overrides["DOCGRID_QWEN_MODEL"])
            self.assertEqual(
                stage2["train"]["parent_checkpoint"],
                overrides["DOCGRID_STAGE2_PARENT_CHECKPOINT"],
            )
            self.assertEqual(stage2["train"]["gate_receipts"]["gate1"], overrides["DOCGRID_GATE1_RECEIPT"])
            stage5 = load_config(root / "configs/docgrid_v2/stage5_full_page.yaml")
            stage4 = load_config(root / "configs/docgrid_v2/stage4_qwen.yaml")
            self.assertTrue(stage4["model"]["instantiate_qwen_adapter"])
            self.assertTrue(stage5["model"]["instantiate_qwen_adapter"])
            self.assertEqual(stage5["data"]["train_manifest"], overrides["DOCGRID_STAGE5_TRAIN_MANIFEST"])
            self.assertEqual(stage5["data"]["val_manifest"], overrides["DOCGRID_STAGE5_VAL_MANIFEST"])
            self.assertEqual(stage5["data"]["frozen_contract"], overrides["DOCGRID_STAGE5_FROZEN_CONTRACT"])
            self.assertEqual(
                stage5["train"]["parent_checkpoint"],
                overrides["DOCGRID_STAGE5_PARENT_CHECKPOINT"],
            )

    def test_required_environment_reference_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("value: ${DOCGRID_REQUIRED_FOR_TEST}\n", encoding="utf-8")
            environment = dict(os.environ)
            environment.pop("DOCGRID_REQUIRED_FOR_TEST", None)
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(ValueError, "DOCGRID_REQUIRED_FOR_TEST"):
                    load_config(config)


if __name__ == "__main__":
    unittest.main()
