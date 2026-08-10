from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cp_docflow.checkpoint import file_sha256
from cp_docflow.evaluation.ocr_metrics import edit_distance, score_ocr_transcripts


class OCRMetricsTest(unittest.TestCase):
    def test_edit_distance(self) -> None:
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance([], []), 0)

    def test_rejects_unknown_engine_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcripts = root / "transcripts.jsonl"
            transcripts.write_text(
                json.dumps(
                    {
                        "sample_id": "a",
                        "reference_text": "text",
                        "model_text": "text",
                        "oracle_text": "text",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exact fixed engine"):
                score_ocr_transcripts(
                    transcripts,
                    root / "output",
                    ocr_engine="fixture",
                    ocr_engine_version="unknown",
                )

    def test_scores_corpus_and_binds_geometry_sample_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_rows = []
            for sample_id in ("a", "b"):
                model_image = root / f"{sample_id}-model.png"
                oracle_image = root / f"{sample_id}-oracle.png"
                model_image.write_bytes(f"model-{sample_id}".encode())
                oracle_image.write_bytes(f"oracle-{sample_id}".encode())
                image_rows.append(
                    {
                        "sample_id": sample_id,
                        "model_image": str(model_image),
                        "model_image_sha256": file_sha256(model_image),
                        "oracle_image": str(oracle_image),
                        "oracle_image_sha256": file_sha256(oracle_image),
                    }
                )
            image_manifest = root / "ocr_images.jsonl"
            image_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in image_rows),
                encoding="utf-8",
            )
            image_identity = {row["sample_id"]: row for row in image_rows}
            transcripts = root / "transcripts.jsonl"
            with transcripts.open("w", encoding="utf-8") as handle:
                for value in (
                    {
                        "sample_id": "a",
                        "reference_text": "hello world",
                        "model_text": "hallo world",
                        "oracle_text": "hello world",
                        "model_image_sha256": image_identity["a"][
                            "model_image_sha256"
                        ],
                        "oracle_image_sha256": image_identity["a"][
                            "oracle_image_sha256"
                        ],
                    },
                    {
                        "sample_id": "b",
                        "reference_text": "table line",
                        "model_text": "table lime",
                        "oracle_text": "table line",
                        "model_image_sha256": image_identity["b"][
                            "model_image_sha256"
                        ],
                        "oracle_image_sha256": image_identity["b"][
                            "oracle_image_sha256"
                        ],
                    },
                ):
                    handle.write(json.dumps(value) + "\n")
            geometry_rows = root / "geometry.csv"
            with geometry_rows.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("sample_id", "epe"))
                writer.writeheader()
                writer.writerows(
                    ({"sample_id": "a", "epe": 1.0}, {"sample_id": "b", "epe": 2.0})
                )
            geometry_report = root / "geometry.json"
            geometry_report.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": "c" * 64,
                        "manifest_sha256": "m" * 64,
                        "evaluation_dataset_payload_sha256": "d" * 64,
                        "ocr_image_export": {
                            "manifest_sha256": file_sha256(image_manifest)
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = score_ocr_transcripts(
                transcripts,
                root / "output",
                ocr_engine="fixture",
                ocr_engine_version="1",
                geometry_evaluation=geometry_report,
                geometry_per_sample=geometry_rows,
                ocr_image_manifest=image_manifest,
            )
            self.assertEqual(report["schema"], "docgrid_flow.ocr_evaluation.v2")
            self.assertAlmostEqual(report["ocr_cer"], 2.0 / 21.0)
            self.assertEqual(report["oracle_ocr_cer"], 0.0)
            self.assertAlmostEqual(report["ocr_wer"], 2.0 / 4.0)
            fragment = json.loads(
                (root / "output" / "gate_evidence.fragment.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(fragment["ocr_cer"], report["ocr_cer"])

    def test_rejects_geometry_sample_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_image = root / "model.png"
            oracle_image = root / "oracle.png"
            model_image.write_bytes(b"model")
            oracle_image.write_bytes(b"oracle")
            image_manifest = root / "ocr_images.jsonl"
            image_manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "geometry-only",
                        "model_image": str(model_image),
                        "model_image_sha256": file_sha256(model_image),
                        "oracle_image": str(oracle_image),
                        "oracle_image_sha256": file_sha256(oracle_image),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            transcript = root / "transcripts.csv"
            with transcript.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "sample_id", "reference_text", "model_text", "oracle_text"
                ))
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "ocr-only",
                        "reference_text": "x",
                        "model_text": "x",
                        "oracle_text": "x",
                    }
                )
            geometry_rows = root / "geometry.csv"
            geometry_rows.write_text("sample_id\ngeometry-only\n", encoding="utf-8")
            geometry_report = root / "geometry.json"
            geometry_report.write_text(
                json.dumps(
                    {
                        "ocr_image_export": {
                            "manifest_sha256": file_sha256(image_manifest)
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sample sets differ"):
                score_ocr_transcripts(
                    transcript,
                    root / "output",
                    ocr_engine="fixture",
                    ocr_engine_version="1",
                    geometry_evaluation=geometry_report,
                    geometry_per_sample=geometry_rows,
                    ocr_image_manifest=image_manifest,
                )


if __name__ == "__main__":
    unittest.main()
