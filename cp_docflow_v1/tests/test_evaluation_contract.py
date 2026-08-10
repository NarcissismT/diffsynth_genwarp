from __future__ import annotations

import unittest

from cp_docflow.evaluate import _require_gate_contract


class EvaluationContractTest(unittest.TestCase):
    @staticmethod
    def _payload(provenance: str) -> dict[str, object]:
        return {
            "data_contract": {
                "val_manifest_sha256": "a" * 64,
                "allowed_label_provenance": [provenance],
                "train_document_ids_sha256": "b" * 64,
                "val_document_ids_sha256": "c" * 64,
                "train_document_count": 2,
                "val_document_count": 1,
                "document_disjoint_verified": True,
                "frozen_contract": {
                    "splits": {
                        "val": {"dataset_payload_sha256": "d" * 64},
                    }
                },
            }
        }

    def test_verified_renderer_gt_can_form_a_gate_contract(self) -> None:
        _require_gate_contract(
            self._payload("renderer_gt"),
            manifest_sha256="a" * 64,
            allowed_label_provenance={"renderer_gt"},
            dataset_payload_sha256_value="d" * 64,
        )

    def test_synthetic_smoke_cannot_claim_gate_eligibility(self) -> None:
        with self.assertRaisesRegex(ValueError, "only accepts verified"):
            _require_gate_contract(
                self._payload("synthetic_analytic"),
                manifest_sha256="a" * 64,
                allowed_label_provenance={"synthetic_analytic"},
                dataset_payload_sha256_value="d" * 64,
            )

    def test_changed_validation_payload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload changed"):
            _require_gate_contract(
                self._payload("renderer_gt"),
                manifest_sha256="a" * 64,
                allowed_label_provenance={"renderer_gt"},
                dataset_payload_sha256_value="e" * 64,
            )


if __name__ == "__main__":
    unittest.main()
