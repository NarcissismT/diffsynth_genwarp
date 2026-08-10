from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion2raft.teacher_capacity_policy import (
    CANONICAL_POLICY_SHA256,
    POLICY_ID,
    POLICY_SCHEMA_VERSION,
)
from diffusion2raft.teacher_capacity_receipt import (
    RECEIPT_KIND,
    TeacherCapacityReceiptError,
    build_teacher_capacity_receipt,
    decode_teacher_capacity_receipt_base64,
    encode_teacher_capacity_receipt_base64,
    strict_validate_teacher_capacity_receipt,
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reseal(receipt: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(receipt)
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return value


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def _valid_receipt() -> dict[str, Any]:
    return build_teacher_capacity_receipt(
        evidence_report_sha256="1" * 64,
        capacity_contract_sha256="2" * 64,
        capacity_config_projection_sha256="3" * 64,
        protocol_sha256="4" * 64,
        implementation_sha256="5" * 64,
        policy={
            "id": POLICY_ID,
            "version": POLICY_SCHEMA_VERSION,
            "sha256": CANONICAL_POLICY_SHA256,
        },
        migration_seed={
            "sha256": "6" * 64,
            "stage": "unified",
            "epoch_index": 19,
            "completed_epochs": 20,
        },
        teacher={
            "sha256": "7" * 64,
            "file_size": 3_470_866_785,
            "input_size": 512,
            "flow_size": 512,
            "blur_kernel": 39,
            "autocast_dtype": "float16",
            "requires_logical_cuda0": True,
        },
        manifest={
            "sha256": "8" * 64,
            "split": "val",
            "record_count": 300,
        },
    )


class TeacherCapacityReceiptTest(unittest.TestCase):
    def test_build_validate_and_canonical_base64_round_trip(self) -> None:
        receipt = _valid_receipt()
        self.assertEqual(receipt["kind"], RECEIPT_KIND)
        self.assertEqual(receipt["decision"], "pass")
        self.assertIs(receipt["passed"], True)

        body = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        self.assertEqual(
            receipt["receipt_sha256"],
            hashlib.sha256(_canonical_json(body)).hexdigest(),
        )
        self.assertEqual(strict_validate_teacher_capacity_receipt(receipt), receipt)

        encoded = encode_teacher_capacity_receipt_base64(receipt)
        self.assertNotIn("=", encoded)
        self.assertEqual(decode_teacher_capacity_receipt_base64(encoded), receipt)
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        self.assertEqual(raw, _canonical_json(receipt))

    def test_builder_owns_nested_values(self) -> None:
        policy = {
            "id": POLICY_ID,
            "version": POLICY_SCHEMA_VERSION,
            "sha256": CANONICAL_POLICY_SHA256,
        }
        seed = {
            "sha256": "6" * 64,
            "stage": "unified",
            "epoch_index": 19,
            "completed_epochs": 20,
        }
        teacher = {
            "sha256": "7" * 64,
            "file_size": 1,
            "input_size": 512,
            "flow_size": 512,
            "blur_kernel": 39,
            "autocast_dtype": "float16",
            "requires_logical_cuda0": True,
        }
        manifest = {"sha256": "8" * 64, "split": "val", "record_count": 300}
        receipt = build_teacher_capacity_receipt(
            evidence_report_sha256="1" * 64,
            capacity_contract_sha256="2" * 64,
            capacity_config_projection_sha256="3" * 64,
            protocol_sha256="4" * 64,
            implementation_sha256="5" * 64,
            policy=policy,
            migration_seed=seed,
            teacher=teacher,
            manifest=manifest,
        )
        policy["id"] = "mutated"
        seed["completed_epochs"] = 21
        teacher["input_size"] = 1
        manifest["split"] = "train"
        self.assertEqual(receipt["policy"]["id"], POLICY_ID)
        self.assertEqual(receipt["migration_seed"]["completed_epochs"], 20)
        self.assertEqual(receipt["teacher"]["input_size"], 512)
        self.assertEqual(receipt["manifest"]["split"], "val")
        self.assertEqual(strict_validate_teacher_capacity_receipt(receipt), receipt)

    def test_every_digest_rejects_noncanonical_or_tampered_value(self) -> None:
        digest_paths = (
            ("evidence_report_sha256",),
            ("capacity_contract_sha256",),
            ("capacity_config_projection_sha256",),
            ("protocol_sha256",),
            ("implementation_sha256",),
            ("policy", "sha256"),
            ("migration_seed", "sha256"),
            ("teacher", "sha256"),
            ("manifest", "sha256"),
            ("receipt_sha256",),
        )
        for path in digest_paths:
            with self.subTest(path=path):
                receipt = _valid_receipt()
                _set_path(receipt, path, "A" * 64)
                with self.assertRaises(TeacherCapacityReceiptError):
                    strict_validate_teacher_capacity_receipt(receipt)

        receipt = _valid_receipt()
        receipt["evidence_report_sha256"] = "9" * 64
        with self.assertRaisesRegex(TeacherCapacityReceiptError, "receipt_sha256"):
            strict_validate_teacher_capacity_receipt(receipt)

    def test_exact_keys_are_required_at_every_level(self) -> None:
        mapping_paths = (
            (),
            ("policy",),
            ("migration_seed",),
            ("teacher",),
            ("manifest",),
        )
        for path in mapping_paths:
            with self.subTest(path=path, mutation="extra"):
                receipt = _valid_receipt()
                target = receipt
                for key in path:
                    target = target[key]
                target["unexpected"] = "value"
                with self.assertRaisesRegex(TeacherCapacityReceiptError, "keys differ"):
                    strict_validate_teacher_capacity_receipt(_reseal(receipt))

            with self.subTest(path=path, mutation="missing"):
                receipt = _valid_receipt()
                target = receipt
                for key in path:
                    target = target[key]
                removable = next(key for key in target if key != "receipt_sha256")
                del target[removable]
                with self.assertRaisesRegex(TeacherCapacityReceiptError, "keys differ"):
                    strict_validate_teacher_capacity_receipt(_reseal(receipt))

    def test_type_tampering_is_rejected_even_after_reseal(self) -> None:
        mutations = (
            (("version",), True),
            (("kind",), 1),
            (("decision",), True),
            (("passed",), 1),
            (("policy",), []),
            (("policy", "version"), True),
            (("migration_seed", "epoch_index"), True),
            (("migration_seed", "completed_epochs"), 20.0),
            (("teacher", "file_size"), True),
            (("teacher", "input_size"), 512.0),
            (("teacher", "autocast_dtype"), 16),
            (("teacher", "requires_logical_cuda0"), 1),
            (("manifest", "record_count"), 300.0),
        )
        for path, replacement in mutations:
            with self.subTest(path=path):
                receipt = _valid_receipt()
                _set_path(receipt, path, replacement)
                with self.assertRaises(TeacherCapacityReceiptError):
                    strict_validate_teacher_capacity_receipt(_reseal(receipt))

    def test_semantic_tampering_is_rejected_even_after_reseal(self) -> None:
        mutations = (
            (("kind",), "other"),
            (("decision",), "fail"),
            (("passed",), False),
            (("policy", "id"), "other_policy"),
            (("policy", "version"), 2),
            (("policy", "sha256"), "9" * 64),
            (("migration_seed", "stage"), "joint"),
            (("migration_seed", "completed_epochs"), 21),
            (("migration_seed", "epoch_index"), 18),
            (("teacher", "file_size"), 0),
            (("teacher", "input_size"), -1),
            (("teacher", "flow_size"), 0),
            (("teacher", "blur_kernel"), 0),
            (("teacher", "autocast_dtype"), ""),
            (("manifest", "split"), "train"),
            (("manifest", "record_count"), 299),
        )
        for path, replacement in mutations:
            with self.subTest(path=path):
                receipt = _valid_receipt()
                _set_path(receipt, path, replacement)
                with self.assertRaises(TeacherCapacityReceiptError):
                    strict_validate_teacher_capacity_receipt(_reseal(receipt))

    def test_noncanonical_base64_and_json_are_rejected(self) -> None:
        receipt = _valid_receipt()
        encoded = encode_teacher_capacity_receipt_base64(receipt)
        with self.assertRaisesRegex(TeacherCapacityReceiptError, "canonical"):
            decode_teacher_capacity_receipt_base64(encoded + "=")
        with self.assertRaises(TeacherCapacityReceiptError):
            decode_teacher_capacity_receipt_base64(encoded + "!")
        with self.assertRaises(TeacherCapacityReceiptError):
            decode_teacher_capacity_receipt_base64(b"not-a-string")

        noncanonical_json = json.dumps(receipt, sort_keys=False, indent=2).encode(
            "utf-8"
        )
        noncanonical_encoded = base64.urlsafe_b64encode(noncanonical_json).decode(
            "ascii"
        ).rstrip("=")
        with self.assertRaisesRegex(
            TeacherCapacityReceiptError, "JSON is not canonical"
        ):
            decode_teacher_capacity_receipt_base64(noncanonical_encoded)


if __name__ == "__main__":
    unittest.main()
