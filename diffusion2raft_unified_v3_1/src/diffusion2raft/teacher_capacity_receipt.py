"""Compact, strict receipt for an approved teacher-capacity audit.

The raw capacity report contains per-sample diagnostics and is intentionally
not embedded in every training checkpoint.  This module defines the compact
JSON contract that is safe to persist instead.  It is deliberately pure: it
does no file-system I/O and has no PyTorch dependency.

Receipts are integrity envelopes, not signatures.  Callers must still compare
the bound digests with the files and runtime contract they authenticated.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import hmac
import json
import re
from typing import Any

from .teacher_capacity_policy import (
    CANONICAL_POLICY_SHA256,
    POLICY_ID,
    POLICY_SCHEMA_VERSION,
)


RECEIPT_VERSION = 1
RECEIPT_KIND = "teacher_capacity_evidence_receipt"
RECEIPT_DECISION = "pass"

_TOP_LEVEL_KEYS = frozenset(
    {
        "version",
        "kind",
        "decision",
        "passed",
        "evidence_report_sha256",
        "capacity_contract_sha256",
        "capacity_config_projection_sha256",
        "protocol_sha256",
        "implementation_sha256",
        "policy",
        "migration_seed",
        "teacher",
        "manifest",
        "receipt_sha256",
    }
)
_POLICY_KEYS = frozenset({"id", "version", "sha256"})
_MIGRATION_SEED_KEYS = frozenset(
    {"sha256", "stage", "epoch_index", "completed_epochs"}
)
_TEACHER_KEYS = frozenset(
    {
        "sha256",
        "file_size",
        "input_size",
        "flow_size",
        "blur_kernel",
        "autocast_dtype",
        "requires_logical_cuda0",
    }
)
_MANIFEST_KEYS = frozenset({"sha256", "split", "record_count"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BASE64_RE = re.compile(r"[A-Za-z0-9_-]+\Z")


class TeacherCapacityReceiptError(ValueError):
    """Raised when a capacity receipt violates the frozen v1 contract."""


def _fail(message: str) -> None:
    raise TeacherCapacityReceiptError(message)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherCapacityReceiptError(
            f"teacher capacity receipt is not canonical-JSON compatible: {exc}"
        ) from exc


def _require_exact_dict(
    value: Any, *, path: str, keys: frozenset[str]
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{path} must be an exact dict")
    actual_keys = set(value)
    if actual_keys != keys:
        missing = sorted(keys - actual_keys)
        extra = sorted(actual_keys - keys, key=str)
        _fail(f"{path} keys differ; missing={missing}, extra={extra}")
    return value


def _require_exact_int(value: Any, *, path: str, positive: bool = False) -> int:
    if type(value) is not int:
        _fail(f"{path} must be an exact int")
    if positive and value <= 0:
        _fail(f"{path} must be positive")
    return value


def _require_exact_bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        _fail(f"{path} must be an exact bool")
    return value


def _require_exact_string(
    value: Any, *, path: str, nonempty: bool = False
) -> str:
    if type(value) is not str:
        _fail(f"{path} must be an exact str")
    if nonempty and not value:
        _fail(f"{path} must be non-empty")
    return value


def _require_sha256(value: Any, *, path: str) -> str:
    digest = _require_exact_string(value, path=path)
    if _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{path} must be exactly 64 lowercase hexadecimal characters")
    return digest


def _body_sha256(receipt: dict[str, Any]) -> str:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def strict_validate_teacher_capacity_receipt(receipt: Any) -> dict[str, Any]:
    """Validate and return a caller-owned copy of one strict schema-v1 receipt.

    Validation is intentionally type-sensitive: for example, ``True`` is not
    accepted where an integer is required, despite ``bool`` subclassing
    ``int`` in Python.
    """

    value = _require_exact_dict(receipt, path="receipt", keys=_TOP_LEVEL_KEYS)

    if _require_exact_int(value["version"], path="receipt.version") != RECEIPT_VERSION:
        _fail(f"receipt.version must be {RECEIPT_VERSION}")
    if _require_exact_string(value["kind"], path="receipt.kind") != RECEIPT_KIND:
        _fail(f"receipt.kind must be {RECEIPT_KIND!r}")
    if (
        _require_exact_string(value["decision"], path="receipt.decision")
        != RECEIPT_DECISION
    ):
        _fail(f"receipt.decision must be {RECEIPT_DECISION!r}")
    if not _require_exact_bool(value["passed"], path="receipt.passed"):
        _fail("receipt.passed must be true")

    for key in (
        "evidence_report_sha256",
        "capacity_contract_sha256",
        "capacity_config_projection_sha256",
        "protocol_sha256",
        "implementation_sha256",
    ):
        _require_sha256(value[key], path=f"receipt.{key}")

    policy = _require_exact_dict(
        value["policy"], path="receipt.policy", keys=_POLICY_KEYS
    )
    policy_id = _require_exact_string(
        policy["id"], path="receipt.policy.id", nonempty=True
    )
    if policy_id != POLICY_ID:
        _fail(f"receipt.policy.id must be {POLICY_ID!r}")
    policy_version = _require_exact_int(
        policy["version"], path="receipt.policy.version", positive=True
    )
    if policy_version != POLICY_SCHEMA_VERSION:
        _fail(f"receipt.policy.version must be {POLICY_SCHEMA_VERSION}")
    policy_sha256 = _require_sha256(
        policy["sha256"], path="receipt.policy.sha256"
    )
    if policy_sha256 != CANONICAL_POLICY_SHA256:
        _fail(
            "receipt.policy.sha256 does not identify the frozen production "
            "capacity policy"
        )

    migration_seed = _require_exact_dict(
        value["migration_seed"],
        path="receipt.migration_seed",
        keys=_MIGRATION_SEED_KEYS,
    )
    _require_sha256(
        migration_seed["sha256"], path="receipt.migration_seed.sha256"
    )
    seed_stage = _require_exact_string(
        migration_seed["stage"],
        path="receipt.migration_seed.stage",
        nonempty=True,
    )
    if seed_stage != "unified":
        _fail("receipt.migration_seed.stage must be 'unified'")
    epoch_index = _require_exact_int(
        migration_seed["epoch_index"], path="receipt.migration_seed.epoch_index"
    )
    completed_epochs = _require_exact_int(
        migration_seed["completed_epochs"],
        path="receipt.migration_seed.completed_epochs",
    )
    if completed_epochs != epoch_index + 1:
        _fail(
            "receipt.migration_seed.completed_epochs must equal "
            "epoch_index + 1"
        )
    if completed_epochs < 20:
        _fail("receipt.migration_seed.completed_epochs must be at least 20")

    teacher = _require_exact_dict(
        value["teacher"], path="receipt.teacher", keys=_TEACHER_KEYS
    )
    _require_sha256(teacher["sha256"], path="receipt.teacher.sha256")
    for key in ("file_size", "input_size", "flow_size", "blur_kernel"):
        _require_exact_int(
            teacher[key], path=f"receipt.teacher.{key}", positive=True
        )
    _require_exact_string(
        teacher["autocast_dtype"],
        path="receipt.teacher.autocast_dtype",
        nonempty=True,
    )
    _require_exact_bool(
        teacher["requires_logical_cuda0"],
        path="receipt.teacher.requires_logical_cuda0",
    )

    manifest = _require_exact_dict(
        value["manifest"], path="receipt.manifest", keys=_MANIFEST_KEYS
    )
    _require_sha256(manifest["sha256"], path="receipt.manifest.sha256")
    if _require_exact_string(manifest["split"], path="receipt.manifest.split") != "val":
        _fail("receipt.manifest.split must be 'val'")
    if (
        _require_exact_int(
            manifest["record_count"], path="receipt.manifest.record_count"
        )
        != 300
    ):
        _fail("receipt.manifest.record_count must be 300")

    supplied_receipt_sha256 = _require_sha256(
        value["receipt_sha256"], path="receipt.receipt_sha256"
    )
    expected_receipt_sha256 = _body_sha256(value)
    if not hmac.compare_digest(supplied_receipt_sha256, expected_receipt_sha256):
        _fail(
            "receipt.receipt_sha256 does not match canonical JSON excluding "
            "receipt_sha256"
        )

    return copy.deepcopy(value)


def build_teacher_capacity_receipt(
    *,
    evidence_report_sha256: str,
    capacity_contract_sha256: str,
    capacity_config_projection_sha256: str,
    protocol_sha256: str,
    implementation_sha256: str,
    policy: dict[str, Any],
    migration_seed: dict[str, Any],
    teacher: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build a validated pass receipt from already authenticated bindings."""

    receipt: dict[str, Any] = {
        "version": RECEIPT_VERSION,
        "kind": RECEIPT_KIND,
        "decision": RECEIPT_DECISION,
        "passed": True,
        "evidence_report_sha256": evidence_report_sha256,
        "capacity_contract_sha256": capacity_contract_sha256,
        "capacity_config_projection_sha256": capacity_config_projection_sha256,
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "policy": copy.deepcopy(policy),
        "migration_seed": copy.deepcopy(migration_seed),
        "teacher": copy.deepcopy(teacher),
        "manifest": copy.deepcopy(manifest),
    }
    receipt["receipt_sha256"] = _body_sha256(receipt)
    return strict_validate_teacher_capacity_receipt(receipt)


def encode_teacher_capacity_receipt_base64(receipt: Any) -> str:
    """Encode a receipt as canonical, unpadded URL-safe base64."""

    validated = strict_validate_teacher_capacity_receipt(receipt)
    return (
        base64.urlsafe_b64encode(_canonical_json(validated))
        .decode("ascii")
        .rstrip("=")
    )


def decode_teacher_capacity_receipt_base64(encoded: Any) -> dict[str, Any]:
    """Decode only the canonical base64/JSON representation of a v1 receipt."""

    if type(encoded) is not str or not encoded:
        _fail("encoded teacher capacity receipt must be a non-empty exact str")
    if _BASE64_RE.fullmatch(encoded) is None:
        _fail("encoded teacher capacity receipt is not canonical URL-safe base64")
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        raw = base64.b64decode(
            padded.encode("ascii"), altchars=b"-_", validate=True
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise TeacherCapacityReceiptError(
            "encoded teacher capacity receipt is invalid base64"
        ) from exc
    canonical_encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    if not hmac.compare_digest(encoded, canonical_encoded):
        _fail("encoded teacher capacity receipt is not canonical base64")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeacherCapacityReceiptError(
            "encoded teacher capacity receipt is not valid UTF-8 JSON"
        ) from exc
    validated = strict_validate_teacher_capacity_receipt(decoded)
    if raw != _canonical_json(validated):
        _fail("encoded teacher capacity receipt JSON is not canonical")
    return validated


__all__ = [
    "RECEIPT_DECISION",
    "RECEIPT_KIND",
    "RECEIPT_VERSION",
    "TeacherCapacityReceiptError",
    "build_teacher_capacity_receipt",
    "decode_teacher_capacity_receipt_base64",
    "encode_teacher_capacity_receipt_base64",
    "strict_validate_teacher_capacity_receipt",
]
