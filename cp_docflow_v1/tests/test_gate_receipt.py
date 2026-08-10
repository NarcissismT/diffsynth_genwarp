from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cp_docflow.checkpoint import file_sha256
from cp_docflow.gates import write_gate_receipt


class GateReceiptTest(unittest.TestCase):
    @staticmethod
    def _report(role: str = "gate") -> dict[str, object]:
        return {
            "schema": "docgrid_flow.full_gate_evaluation.v3",
            "evaluation_role": role,
            "gate_eligible": role == "gate",
            "training_stage": "coarse",
            "checkpoint": "/frozen/model.pt",
            "checkpoint_sha256": "a" * 64,
            "manifest": "/frozen/val.jsonl",
            "manifest_sha256": "b" * 64,
            "evaluation_dataset_payload_sha256": "c" * 64,
            "evaluation_input_work_size": [512, 512],
            "evaluation_output_work_size": [512, 512],
            "training_data_contract": {
                "allowed_label_provenance": ["renderer_gt"],
            },
            "aggregate": {
                "epe": 5.1,
                "fold_rate": 0.001,
                "confidence_monotonic_rate": 1.0,
            },
        }

    def test_verified_gate_report_writes_immutable_v2_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "metrics.json"
            report_path.write_text(json.dumps(self._report()), encoding="utf-8")
            baseline_path = root / "baseline.json"
            baseline = self._report()
            baseline["aggregate"] = {"fold_rate": 0.002}
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            evidence_path = root / "evidence.json"
            evidence_path.write_text(
                json.dumps({"full_page_scale_stable": True}), encoding="utf-8"
            )
            output = root / "gate1.json"
            receipt = write_gate_receipt(
                report_path,
                output,
                gate="gate1",
                passed=True,
                reviewer="unit-test",
                review_note="quantitative and visual criteria reviewed",
                baseline_evaluation=baseline_path,
                evidence_path=evidence_path,
            )
            self.assertEqual(receipt["schema"], "docgrid_flow.gate1.v2")
            self.assertTrue(receipt["verified_gt_only"])
            with self.assertRaises(FileExistsError):
                write_gate_receipt(
                    report_path,
                    output,
                    gate="gate1",
                    passed=True,
                    reviewer="unit-test",
                    review_note="must not overwrite",
                    baseline_evaluation=baseline_path,
                    evidence_path=evidence_path,
                )

    def test_passing_receipt_rejects_missing_plan_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "metrics.json"
            report_path.write_text(json.dumps(self._report()), encoding="utf-8")
            baseline_path = root / "baseline.json"
            baseline = self._report()
            baseline["aggregate"] = {"fold_rate": 0.002}
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full_page_scale_stable"):
                write_gate_receipt(
                    report_path,
                    root / "gate1.json",
                    gate="gate1",
                    passed=True,
                    reviewer="unit-test",
                    review_note="missing visual evidence",
                    baseline_evaluation=baseline_path,
                )

    def test_exploratory_report_cannot_be_promoted_to_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "metrics.json"
            report_path.write_text(
                json.dumps(self._report("exploratory")), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "--gate"):
                write_gate_receipt(
                    report_path,
                    root / "gate1.json",
                    gate="gate1",
                    passed=True,
                    reviewer="unit-test",
                    review_note="invalid source",
                )

    def test_baseline_must_use_the_same_work_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "metrics.json"
            report_path.write_text(json.dumps(self._report()), encoding="utf-8")
            baseline = self._report()
            baseline["schema"] = "docgrid_flow.full_exploratory_evaluation.v3"
            baseline["evaluation_input_work_size"] = [1024, 768]
            baseline["evaluation_output_work_size"] = [1024, 768]
            baseline["aggregate"] = {"fold_rate": 0.002}
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different evaluation_input_work_size"):
                write_gate_receipt(
                    report_path,
                    root / "gate1.json",
                    gate="gate1",
                    passed=False,
                    reviewer="unit-test",
                    review_note="mismatched work size must fail",
                    baseline_evaluation=baseline_path,
                )

    def test_gate5_ocr_must_bind_same_checkpoint_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._report()
            report["training_stage"] = "full_page"
            report["aggregate"] = {
                "epe": 5.0,
                "epe_p95": 8.0,
                "line_epe": 8.0,
                "edge_epe": 5.0,
                "straightness_error": 0.08,
                "final_win_rate": 0.70,
                "fold_rate": 0.001,
                "high_confidence_damage_rate": 0.01,
                "samples": 2.0,
            }
            report_path = root / "metrics.json"
            report["ocr_image_export"] = {"manifest_sha256": "i" * 64}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            baseline = self._report()
            baseline["manifest_sha256"] = report["manifest_sha256"]
            baseline["aggregate"] = {
                "epe_p95": 10.0,
                "line_epe": 10.0,
                "fold_rate": 0.002,
            }
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            ocr_report = {
                "schema": "docgrid_flow.ocr_evaluation.v2",
                "samples": 2,
                "ocr_engine": "fixture",
                "ocr_engine_version": "1",
                "ocr_cer": 0.02,
                "oracle_ocr_cer": 0.015,
                "ocr_wer": 0.03,
                "oracle_ocr_wer": 0.02,
                "geometry_identity": {
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "manifest_sha256": report["manifest_sha256"],
                    "dataset_payload_sha256": report[
                        "evaluation_dataset_payload_sha256"
                    ],
                    "ocr_image_manifest_sha256": "i" * 64,
                },
            }
            ocr_path = root / "ocr_metrics.json"
            ocr_path.write_text(json.dumps(ocr_report), encoding="utf-8")
            evidence = {
                "ocr_cer": 0.02,
                "oracle_ocr_cer": 0.015,
                "ocr_wer": 0.03,
                "oracle_ocr_wer": 0.02,
                "ocr_evaluation": str(ocr_path),
                "ocr_evaluation_sha256": file_sha256(ocr_path),
                "multi_seed_stable": True,
                "fixed_seed_repeatable": True,
                "visual_no_water_ripple": True,
                "visual_text_table_preserved": True,
            }
            evidence_path = root / "evidence.json"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            receipt = write_gate_receipt(
                report_path,
                root / "gate5.json",
                gate="gate5",
                passed=True,
                reviewer="unit-test",
                review_note="all final criteria reviewed",
                baseline_evaluation=baseline_path,
                evidence_path=evidence_path,
            )
            self.assertEqual(receipt["ocr_evidence"]["sha256"], file_sha256(ocr_path))

            evidence["ocr_cer"] = 0.01
            tampered_path = root / "tampered_evidence.json"
            tampered_path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from its report"):
                write_gate_receipt(
                    report_path,
                    root / "tampered_gate5.json",
                    gate="gate5",
                    passed=True,
                    reviewer="unit-test",
                    review_note="tampered OCR must fail",
                    baseline_evaluation=baseline_path,
                    evidence_path=tampered_path,
                )


if __name__ == "__main__":
    unittest.main()
