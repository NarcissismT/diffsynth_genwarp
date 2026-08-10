from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cp_docflow.checkpoint import file_sha256, full_checkpoint_payload, load_full_checkpoint
from cp_docflow.config import build_full_model
from cp_docflow.evaluate_full import evaluate_full
from cp_docflow.geometry import canonical_backward_map
from cp_docflow.infer_full import infer_full
from cp_docflow.make_smoke_data import _write_split
from cp_docflow.train_full import _load_parent, _read_gate_receipts


class FullInferenceCheckpointTest(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, object]:
        return {
            "coarse": {"base_channels": 8, "feature_channels": 16},
            "qwen_backend": "none",
            "qwen_feature_channels": 12,
            "fusion_channels": 16,
            "hv_channels": 8,
            "velocity_hidden_channels": 16,
            "velocity_time_channels": 16,
            "sigma_max": 0.0,
            "fm_steps": 2,
            "refiner_hidden_channels": 16,
            "refiner_iterations": 2,
        }

    def test_native_full_inference_uses_one_source_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3)
            image_path = root / "page.png"
            Image.fromarray(array).save(image_path)
            config = self._config()
            model = build_full_model(config)
            payload = full_checkpoint_payload(
                model,
                model_config=config,
                input_work_size=(32, 40),
                output_work_size=(32, 40),
                epoch=0,
                training_stage="joint",
            )
            checkpoint = root / "model.pt"
            torch.save(payload, checkpoint)
            metadata = infer_full(
                checkpoint, image_path, root / "result", device_name="cpu"
            )
            np.testing.assert_array_equal(
                np.asarray(Image.open(metadata["rectified_image"])), array
            )
            expected = (
                canonical_backward_map(1, (32, 40))
                .squeeze(0)
                .permute(1, 2, 0)
                .numpy()
            )
            np.testing.assert_allclose(
                np.load(metadata["backward_map"]), expected, atol=1.0e-6, rtol=0.0
            )
            self.assertFalse(metadata["qwen_vae_decoder_used"])
            self.assertEqual(
                metadata["final_rgb_source"],
                "single_grid_sample_from_native_warped_image",
            )

    def test_qwen_checkpoint_has_adapter_but_no_pretrained_backbone(self) -> None:
        config = {
            **self._config(),
            "qwen_backend": "qwen",
            "qwen": {
                "model_id": "/model/not-loaded-during-construction",
                "hidden_channels": 32,
                "feature_layers": [-3, -2, -1],
            },
        }
        model = build_full_model(config)
        keys = set(model.state_dict())
        self.assertTrue(any(name.startswith("qwen_adapter.") for name in keys))
        self.assertFalse(any(name.startswith("qwen_source.") for name in keys))

    def test_gate5_evaluator_exports_sha_bound_ocr_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_split(root, "val", 1, (32, 32), 0)
            config = self._config()
            model = build_full_model(config)
            checkpoint = root / "model.pt"
            torch.save(
                full_checkpoint_payload(
                    model,
                    model_config=config,
                    input_work_size=(32, 32),
                    output_work_size=(32, 32),
                    epoch=0,
                    training_stage="full_page",
                ),
                checkpoint,
            )
            report = evaluate_full(
                checkpoint,
                manifest,
                root / "evaluation",
                device_name="cpu",
                allowed_label_provenance={"synthetic_analytic"},
                max_visualizations=0,
                export_ocr_images=True,
            )
            export = report["ocr_image_export"]
            self.assertEqual(export["samples"], 1)
            image_manifest = Path(export["manifest"])
            self.assertEqual(export["manifest_sha256"], file_sha256(image_manifest))
            row = json.loads(image_manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                row["model_image_sha256"], file_sha256(row["model_image"])
            )
            self.assertEqual(
                row["oracle_image_sha256"], file_sha256(row["oracle_image"])
            )

    def test_stage4_initializes_qwen_adapter_after_adapter_free_stage3_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage3_config = {
                **self._config(),
                "qwen_backend": "qwen",
                "qwen": {
                    "model_id": "/model/not-loaded-during-construction",
                    "hidden_channels": 32,
                    "feature_layers": [-3, -2, -1],
                },
                "instantiate_qwen_adapter": False,
            }
            stage4_config = {**stage3_config, "instantiate_qwen_adapter": True}
            stage3_model = build_full_model(stage3_config)
            self.assertIsNone(stage3_model.qwen_adapter)
            checkpoint = Path(directory) / "stage3.pt"
            torch.save(
                full_checkpoint_payload(
                    stage3_model,
                    model_config=stage3_config,
                    input_work_size=(32, 32),
                    output_work_size=(32, 32),
                    epoch=0,
                    training_stage="coord_fm",
                ),
                checkpoint,
            )
            stage4_model = build_full_model(stage4_config)
            self.assertIsNotNone(stage4_model.qwen_adapter)
            _load_parent(stage4_model, checkpoint)
            self.assertTrue(
                any(name.startswith("qwen_adapter.") for name in stage4_model.state_dict())
            )

    def test_adapter_free_stage_accepts_only_legacy_extra_adapter_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy_config = {
                **self._config(),
                "qwen_backend": "qwen",
                "qwen": {
                    "model_id": "/model/not-loaded-during-construction",
                    "hidden_channels": 32,
                    "feature_layers": [-3, -2, -1],
                },
                "instantiate_qwen_adapter": True,
            }
            current_config = {**legacy_config, "instantiate_qwen_adapter": False}
            checkpoint = Path(directory) / "legacy_stage3.pt"
            torch.save(
                full_checkpoint_payload(
                    build_full_model(legacy_config),
                    model_config=legacy_config,
                    input_work_size=(32, 32),
                    output_work_size=(32, 32),
                    epoch=0,
                    training_stage="coord_fm",
                ),
                checkpoint,
            )
            current = build_full_model(current_config)
            self.assertIsNone(current.qwen_adapter)
            _load_parent(current, checkpoint)

    def test_checkpoint_loader_rejects_vae_decoder_contract_violation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            config = self._config()
            payload = full_checkpoint_payload(
                build_full_model(config),
                model_config=config,
                input_work_size=(32, 32),
                output_work_size=(32, 32),
                epoch=0,
                training_stage="joint",
            )
            payload["qwen_vae_decoder_used"] = True
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "no-Qwen-VAE-decoder"):
                load_full_checkpoint(path)

    def test_qwen_stage_fails_closed_without_gate_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "_project_root": directory,
                "train": {"gate_receipts": {}},
            }
            with self.assertRaisesRegex(ValueError, "gate1"):
                _read_gate_receipts(config, "qwen", enforce=True)


if __name__ == "__main__":
    unittest.main()
