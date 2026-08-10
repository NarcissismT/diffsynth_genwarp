from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cp_docflow.checkpoint import file_sha256
from cp_docflow.validate_qwen_runtime import check_qwen_runtime_report


class QwenRuntimeValidationTest(unittest.TestCase):
    def test_report_is_bound_to_local_model_contract_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "qwen"
            (model / "transformer").mkdir(parents=True)
            model_index = model / "model_index.json"
            transformer = model / "transformer" / "config.json"
            model_index.write_text('{"_class_name":"QwenImageEditPipeline"}', encoding="utf-8")
            transformer.write_text('{"num_layers":60}', encoding="utf-8")
            report = {
                "schema": "docgrid_flow.qwen_runtime_validation.v2",
                "validated": True,
                "qwen_vae_decoder_called": False,
                "model_id": str(model.resolve()),
                "model_index_sha256": file_sha256(model_index),
                "transformer_config_sha256": file_sha256(transformer),
                "input_size": [512, 512],
                "feature_type": "hidden",
                "feature_layers": [-1],
                "probe_contract": {
                    "input_size": [512, 512],
                    "feature_type": "hidden",
                    "feature_layers": [-1],
                    "feature_dtype": "bfloat16",
                    "feature_quantization": "none",
                    "feature_num_inference_steps": 1,
                    "guidance_scale": 1.0,
                    "feature_seed": 0,
                    "hidden_channels": 3072,
                    "cpu_offload": True,
                    "local_files_only": True,
                    "output_type": "latent",
                    "vae_decoder_forbidden": True,
                },
                "features": [
                    {
                        "layer": -1,
                        "target": [1, 3072, 32, 32],
                        "source": [1, 3072, 64, 64],
                        "finite": True,
                    }
                ],
            }
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            checked = check_qwen_runtime_report(path, model)
            self.assertTrue(checked["validated"])
            checked = check_qwen_runtime_report(
                path,
                model,
                feature_type="hidden",
                feature_layers=(-1,),
                input_size=(512, 512),
                feature_dtype="bfloat16",
                cpu_offload=True,
                feature_num_inference_steps=1,
                guidance_scale=1.0,
                feature_seed=0,
                hidden_channels=3072,
            )
            self.assertTrue(checked["validated"])
            with self.assertRaisesRegex(ValueError, "input_size"):
                check_qwen_runtime_report(path, model, input_size=(1024, 768))
            with self.assertRaisesRegex(ValueError, "cpu_offload"):
                check_qwen_runtime_report(path, model, cpu_offload=False)
            transformer.write_text('{"num_layers":61}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                check_qwen_runtime_report(path, model)


if __name__ == "__main__":
    unittest.main()
