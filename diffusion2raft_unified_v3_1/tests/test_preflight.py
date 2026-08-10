from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight_v31.py"
SPEC = importlib.util.spec_from_file_location("preflight_v31", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class PreflightTest(unittest.TestCase):
    def test_sample_indices_cover_manifest_ends(self) -> None:
        self.assertEqual(PREFLIGHT._sample_indices(100, 3), [0, 50, 99])
        self.assertEqual(PREFLIGHT._sample_indices(2, 9), [0, 1])

    def test_independent_flow_grid_requires_source_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            Image.fromarray(image).save(root / "warped.png")
            Image.fromarray(image).save(root / "target.png")
            np.save(root / "flow.npy", np.zeros((64, 64, 2), dtype=np.float32))
            manifest = root / "data.jsonl"
            record = {
                "id": "canvas-check",
                "warped": "warped.png",
                "target": "target.png",
                "flow": "flow.npy",
            }
            with self.assertRaisesRegex(ValueError, "flow_source_size is required"):
                PREFLIGHT._validate_sample(manifest, record, 0)

            record["flow_source_size"] = [32, 32]
            record["flow_target_size"] = [64, 64]
            report = PREFLIGHT._validate_sample(manifest, record, 0)
            self.assertEqual(report["flow_grid_size"], (64, 64))
            self.assertEqual(report["flow_source_size"], (32, 32))

    def test_inpaint_preflight_records_external_file_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lama = root / "fake-lama.pt"
            lama.write_bytes(b"fake-jit-for-byte-identity")
            config = {
                "inference": {
                    "resize_policy": "stretch",
                    "image_decoder": "pil",
                    "resize_interpolation": "bilinear",
                    "inpaint": {
                        "enabled": True,
                        "path": lama.name,
                        "sha256": hashlib.sha256(lama.read_bytes()).hexdigest(),
                        "size": 512,
                        "dilation": 11,
                    },
                }
            }
            summary = PREFLIGHT._inference_dependency_summary(
                config, root, (512, 512)
            )
            identity = summary["inpaint_identity"]
            stat = lama.stat()
            self.assertTrue(identity["enabled"])
            self.assertEqual(identity["path"], str(lama.resolve()))
            self.assertEqual(identity["size_bytes"], stat.st_size)
            self.assertEqual(identity["mtime_ns"], stat.st_mtime_ns)
            self.assertEqual(
                identity["sha256"], hashlib.sha256(lama.read_bytes()).hexdigest()
            )
            self.assertEqual(identity["input_size"], 512)
            self.assertEqual(identity["dilation_kernel"], 11)

            lama.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                PREFLIGHT._inference_dependency_summary(config, root, (512, 512))


if __name__ == "__main__":
    unittest.main()
