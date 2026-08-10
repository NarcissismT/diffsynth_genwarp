"""Integrity envelope for production teacher-capacity evidence.

This module turns a raw GPU capacity audit into a content-addressed evidence
document.  It deliberately does not run the audit, authenticate deployment
assets, or construct the final training receipt.  Verification fails closed
and re-evaluates every locally reproducible binding.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .teacher_capacity_policy import (
    CANONICAL_POLICY_SHA256,
    POLICY_ID,
    POLICY_SCHEMA_VERSION,
    evaluate_teacher_capacity_policy,
)


EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_KIND = "teacher_capacity_production_evidence"
EVIDENCE_PROFILE = "production"
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_KIND = "teacher_capacity_production_contract"
IMPLEMENTATION_IDENTITY_VERSION = 1
IMPLEMENTATION_IDENTITY_KIND = "teacher_capacity_implementation_identity"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IMPLEMENTATION_FILES = (
    "src/diffusion2raft/teacher_capacity_evidence.py",
    "src/diffusion2raft/teacher_capacity_preflight.py",
    "src/diffusion2raft/teacher_capacity_policy.py",
    "src/diffusion2raft/teacher_capacity_assets.py",
    "src/diffusion2raft/teacher_capacity_receipt.py",
    "src/diffusion2raft/teacher_capacity_production.py",
    "src/diffusion2raft/external_file.py",
    "src/diffusion2raft/data.py",
    "src/diffusion2raft/geometry.py",
    "src/diffusion2raft/models/teacher_prior.py",
    "src/diffusion2raft/models/unified.py",
    "src/diffusion2raft/losses.py",
    "src/diffusion2raft/config.py",
)
_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile",
        "raw_audit",
        "policy_decision",
        "capacity_contract",
        "asset_identity",
    }
)


class TeacherCapacityEvidenceError(ValueError):
    """Raised when production capacity evidence is incomplete or inconsistent."""


def _fail(message: str) -> None:
    raise TeacherCapacityEvidenceError(message)


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
        raise TeacherCapacityEvidenceError(
            f"value is not canonical-JSON compatible: {exc}"
        ) from exc


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of a value's canonical JSON encoding."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise RuntimeError(f"file changed while hashing it: {path}")
    return digest.hexdigest(), int(after.st_size)


def _get(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            _fail(f"missing required field: {'.'.join(path)}")
        current = current[key]
    return current


def _require_sha256(value: Any, *, path: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{path} must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_exact(value: Any, expected: Any, *, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(f"{path} must be {expected!r}; got {value!r}")


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def validate_production_raw_report(
    report: Mapping[str, Any], *, configured_teacher_sha256: str
) -> dict[str, Any]:
    """Validate provenance/protocol fields required beyond numeric policy v1.

    The numeric acceptance decision is intentionally evaluated separately by
    :func:`evaluate_teacher_capacity_policy`.  This check ensures the report
    was made with the configured teacher and the complete production sampling
    modes, with no path override or unhashed external input.
    """

    if not isinstance(report, Mapping):
        raise TypeError("teacher capacity report must be a mapping")
    configured_pin = _require_sha256(
        configured_teacher_sha256, path="configured_teacher_sha256"
    )

    exact_fields = (
        (("identities", "teacher", "selection", "source"), "config"),
        (("identities", "teacher", "selection", "override_value"), None),
        (("protocol", "external_file_sha256_enabled"), True),
        (("protocol", "source_full_geometry", "enabled"), True),
        (
            ("protocol", "source_full_geometry", "mode"),
            "deterministic_config_distribution_conditional_on_trigger",
        ),
        (("protocol", "source_full_geometry", "transformations_per_sample"), 1),
        (
            (
                "protocol",
                "source_full_geometry",
                "reuses_source_geometry_augment_sample_homography",
            ),
            True,
        ),
        (
            (
                "protocol",
                "source_full_geometry",
                "conditional_on_augmentation_trigger",
            ),
            True,
        ),
        (
            (
                "protocol",
                "source_full_geometry",
                "equivalent_to_conditional_full_source_geometry_augment",
            ),
            True,
        ),
        (
            ("protocol", "source_rotation", "mode"),
            "config_stratified_uniform",
        ),
        (("protocol", "source_rotation", "rotations_per_sample"), 1),
    )
    for path, expected in exact_fields:
        _require_exact(_get(report, *path), expected, path=".".join(path))

    selection = _get(report, "identities", "teacher", "selection")
    config_resolved_path = _get(selection, "config_resolved_path")
    effective_path = _get(selection, "effective_path")
    if type(config_resolved_path) is not str or not config_resolved_path:
        _fail("identities.teacher.selection.config_resolved_path must be non-empty")
    if type(effective_path) is not str or not effective_path:
        _fail("identities.teacher.selection.effective_path must be non-empty")
    if effective_path != config_resolved_path:
        _fail(
            "configured teacher selection must use config_resolved_path as its "
            "effective_path"
        )

    required_identity_sha_paths = (
        ("identities", "config", "sha256"),
        ("identities", "checkpoint", "sha256"),
        ("identities", "teacher", "checkpoint", "sha256"),
        ("identities", "manifest", "sha256"),
        ("protocol", "selected_indices_sha256"),
        ("protocol", "rotation_plan_sha256"),
        ("protocol", "source_full_geometry", "seed_plan_sha256"),
    )
    for path in required_identity_sha_paths:
        _require_sha256(_get(report, *path), path=".".join(path))

    actual_teacher_sha256 = _get(
        report, "identities", "teacher", "checkpoint", "sha256"
    )
    if not hmac.compare_digest(actual_teacher_sha256, configured_pin):
        _fail(
            "raw report teacher SHA-256 does not equal the configured teacher pin"
        )
    return copy.deepcopy(dict(report))


def build_implementation_identity(
    *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Hash every source file that can affect capacity evidence semantics."""

    root = (
        Path(repository_root).expanduser().absolute()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    files: list[dict[str, Any]] = []
    for relative in _IMPLEMENTATION_FILES:
        path = root / relative
        try:
            digest, size = _file_sha256(path.resolve(strict=True))
        except FileNotFoundError as exc:
            raise TeacherCapacityEvidenceError(
                f"required implementation file is missing: {relative}"
            ) from exc
        files.append(
            {"path": relative, "sha256": digest, "size_bytes": size}
        )
    body: dict[str, Any] = {
        "schema_version": IMPLEMENTATION_IDENTITY_VERSION,
        "kind": IMPLEMENTATION_IDENTITY_KIND,
        "files": files,
    }
    return {**body, "implementation_sha256": canonical_sha256(body)}


def _capacity_config_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    teacher = _get(report, "identities", "teacher")
    return {
        "config_sha256": _get(report, "identities", "config", "sha256"),
        "teacher_sha256": _get(teacher, "checkpoint", "sha256"),
        "teacher": {
            key: _get(teacher, key)
            for key in (
                "backend",
                "input_size",
                "flow_size",
                "blur_kernel",
                "autocast_dtype",
                "requires_logical_cuda0",
            )
        },
        "work_size": copy.deepcopy(_get(report, "protocol", "work_size")),
        "feature_stride": _get(report, "protocol", "feature_stride"),
        "max_residual_px": _get(report, "protocol", "max_residual_px"),
        "max_residual_target": _get(report, "protocol", "max_residual_target"),
    }


def _asset_binding(asset_identity: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if asset_identity is None:
        return None
    if not isinstance(asset_identity, Mapping):
        raise TypeError("asset_identity must be a mapping or None")
    aggregate = _require_sha256(
        _get(asset_identity, "aggregate_sha256"),
        path="asset_identity.aggregate_sha256",
    )
    # Canonical serialization also establishes that no opaque/non-JSON value
    # can enter an evidence document through an external identity.
    _canonical_json(asset_identity)
    return {
        "aggregate_sha256": aggregate,
        "identity_sha256": canonical_sha256(asset_identity),
    }


def build_capacity_contract(
    raw_report: Mapping[str, Any],
    policy_decision: Mapping[str, Any],
    implementation_identity: Mapping[str, Any],
    *,
    asset_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and seal the capacity-relevant production contract."""

    if not isinstance(policy_decision, Mapping):
        raise TypeError("policy_decision must be a mapping")
    if not isinstance(implementation_identity, Mapping):
        raise TypeError("implementation_identity must be a mapping")
    if policy_decision.get("passed") is not True:
        _fail("cannot build a production contract from a failing policy decision")
    implementation_sha256 = _require_sha256(
        _get(implementation_identity, "implementation_sha256"),
        path="implementation_identity.implementation_sha256",
    )
    implementation_body = {
        key: value
        for key, value in implementation_identity.items()
        if key != "implementation_sha256"
    }
    if canonical_sha256(implementation_body) != implementation_sha256:
        _fail("implementation identity self-hash is invalid")

    protocol = copy.deepcopy(_get(raw_report, "protocol"))
    projection = _capacity_config_projection(raw_report)
    body: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "raw_report_sha256": canonical_sha256(raw_report),
        "policy": {
            "id": _get(policy_decision, "policy_id"),
            "version": POLICY_SCHEMA_VERSION,
            "sha256": _get(policy_decision, "policy_sha256"),
            "decision_sha256": canonical_sha256(policy_decision),
        },
        "capacity_config_projection": projection,
        "capacity_config_projection_sha256": canonical_sha256(projection),
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "implementation_identity": copy.deepcopy(dict(implementation_identity)),
        "implementation_sha256": implementation_sha256,
        "asset_identity_binding": _asset_binding(asset_identity),
        "teacher_sha256": _get(
            raw_report, "identities", "teacher", "checkpoint", "sha256"
        ),
        "manifest_sha256": _get(raw_report, "identities", "manifest", "sha256"),
        "migration_seed_sha256": _get(
            raw_report, "identities", "checkpoint", "sha256"
        ),
    }
    return {**body, "capacity_contract_sha256": canonical_sha256(body)}


def build_production_evidence(
    raw_report: Mapping[str, Any],
    *,
    configured_teacher_sha256: str,
    repository_root: str | Path | None = None,
    asset_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a passing raw audit and build its production evidence mapping."""

    validated_report = validate_production_raw_report(
        raw_report, configured_teacher_sha256=configured_teacher_sha256
    )
    decision = evaluate_teacher_capacity_policy(validated_report)
    if decision["passed"] is not True:
        failures = [item.get("code") for item in decision.get("failures", [])]
        _fail(f"raw report fails production capacity policy: {failures}")
    implementation = build_implementation_identity(repository_root=repository_root)
    contract = build_capacity_contract(
        validated_report,
        decision,
        implementation,
        asset_identity=asset_identity,
    )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "profile": EVIDENCE_PROFILE,
        "raw_audit": validated_report,
        "policy_decision": decision,
        "capacity_contract": contract,
        "asset_identity": (
            None if asset_identity is None else copy.deepcopy(dict(asset_identity))
        ),
    }


def write_production_evidence(
    output_directory: str | Path, evidence: Mapping[str, Any]
) -> Path:
    """Exclusively write ``<canonical evidence SHA-256>.json`` without overwrite."""

    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    payload = _canonical_json(evidence)
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(output_directory).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{digest}.json"
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A failed exclusive creation must not leave a seemingly valid partial
        # evidence file.  This path is recoverable and targets only our file.
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        raise
    directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
                raise
    finally:
        os.close(directory_fd)
    return output


def _read_stable_json(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise TeacherCapacityEvidenceError("evidence file changed while reading")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeacherCapacityEvidenceError(f"invalid evidence JSON: {exc}") from exc
    if type(value) is not dict:
        _fail("evidence JSON root must be an exact dict")
    if payload != _canonical_json(value):
        _fail("evidence JSON is not in canonical form")
    return value, payload


def _current_external_sha(path: str | Path, *, label: str) -> tuple[str, int]:
    try:
        return _file_sha256(Path(path).expanduser().resolve(strict=True))
    except FileNotFoundError as exc:
        raise TeacherCapacityEvidenceError(f"current {label} file is missing") from exc


def read_verify_production_evidence(
    evidence_path: str | Path,
    *,
    configured_teacher_sha256: str,
    repository_root: str | Path | None = None,
    current_teacher_path: str | Path | None = None,
    current_manifest_path: str | Path | None = None,
    asset_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stably read evidence, recompute all bindings, and return receipt inputs."""

    path = Path(evidence_path).expanduser().absolute()
    evidence, payload = _read_stable_json(path)
    evidence_sha256 = hashlib.sha256(payload).hexdigest()
    if path.name != f"{evidence_sha256}.json":
        _fail("evidence filename does not match its canonical content SHA-256")

    if set(evidence) != _EVIDENCE_KEYS:
        missing = sorted(_EVIDENCE_KEYS - set(evidence))
        extra = sorted(set(evidence) - _EVIDENCE_KEYS, key=str)
        _fail(f"evidence schema fields differ; missing={missing}, extra={extra}")

    _require_exact(evidence.get("schema_version"), 1, path="schema_version")
    _require_exact(evidence.get("kind"), EVIDENCE_KIND, path="kind")
    _require_exact(evidence.get("profile"), EVIDENCE_PROFILE, path="profile")
    raw_report = _get(evidence, "raw_audit")
    validate_production_raw_report(
        raw_report, configured_teacher_sha256=configured_teacher_sha256
    )
    decision = evaluate_teacher_capacity_policy(raw_report)
    if decision["passed"] is not True:
        _fail("stored raw report no longer passes production capacity policy")
    if not _typed_equal(decision, _get(evidence, "policy_decision")):
        _fail("stored policy decision differs from re-evaluation")

    stored_assets = evidence.get("asset_identity")
    expected_assets = None if asset_identity is None else dict(asset_identity)
    if not _typed_equal(stored_assets, expected_assets):
        _fail("supplied asset identity differs from stored evidence identity")

    current_implementation = build_implementation_identity(
        repository_root=repository_root
    )
    expected_contract = build_capacity_contract(
        raw_report,
        decision,
        current_implementation,
        asset_identity=asset_identity,
    )
    stored_contract = _get(evidence, "capacity_contract")
    if not _typed_equal(expected_contract, stored_contract):
        _fail("capacity contract differs from current recomputation")

    teacher_identity = _get(raw_report, "identities", "teacher")
    teacher_checkpoint = _get(teacher_identity, "checkpoint")
    teacher_path = (
        current_teacher_path
        if current_teacher_path is not None
        else _get(teacher_identity, "selection", "effective_path")
    )
    teacher_sha256, teacher_size = _current_external_sha(
        teacher_path, label="teacher"
    )
    configured_pin = _require_sha256(
        configured_teacher_sha256, path="configured_teacher_sha256"
    )
    if not hmac.compare_digest(teacher_sha256, configured_pin):
        _fail("current teacher file SHA-256 differs from configured pin")
    if teacher_size != _get(teacher_checkpoint, "size_bytes"):
        _fail("current teacher file size differs from raw audit identity")

    manifest_identity = _get(raw_report, "identities", "manifest")
    manifest_path = (
        current_manifest_path
        if current_manifest_path is not None
        else _get(manifest_identity, "configured_path")
    )
    manifest_sha256, _ = _current_external_sha(manifest_path, label="manifest")
    if not hmac.compare_digest(
        manifest_sha256,
        _require_sha256(manifest_identity.get("sha256"), path="manifest.sha256"),
    ):
        _fail("current manifest SHA-256 differs from raw audit identity")

    contract_sha256 = _require_sha256(
        stored_contract.get("capacity_contract_sha256"),
        path="capacity_contract.capacity_contract_sha256",
    )
    return {
        "evidence_report_sha256": evidence_sha256,
        "capacity_contract_sha256": contract_sha256,
        "capacity_config_projection_sha256": stored_contract[
            "capacity_config_projection_sha256"
        ],
        "protocol_sha256": stored_contract["protocol_sha256"],
        "implementation_sha256": stored_contract["implementation_sha256"],
        "policy": {
            "id": POLICY_ID,
            "version": POLICY_SCHEMA_VERSION,
            "sha256": CANONICAL_POLICY_SHA256,
        },
        "migration_seed": {
            "sha256": _get(raw_report, "identities", "checkpoint", "sha256")
        },
        "teacher": {
            "sha256": teacher_sha256,
            "file_size": teacher_size,
            "input_size": _get(teacher_identity, "input_size"),
            "flow_size": _get(teacher_identity, "flow_size"),
            "blur_kernel": _get(teacher_identity, "blur_kernel"),
            "autocast_dtype": _get(teacher_identity, "autocast_dtype"),
            "requires_logical_cuda0": _get(
                teacher_identity, "requires_logical_cuda0"
            ),
        },
        "manifest": {
            "sha256": manifest_sha256,
            "split": _get(manifest_identity, "split"),
            "record_count": _get(manifest_identity, "record_count"),
        },
    }


# Descriptive aliases retained for callers that prefer explicit I/O verbs.
write_evidence_json_exclusive = write_production_evidence
read_and_verify_production_evidence = read_verify_production_evidence


__all__ = [
    "CONTRACT_KIND",
    "CONTRACT_SCHEMA_VERSION",
    "EVIDENCE_KIND",
    "EVIDENCE_PROFILE",
    "EVIDENCE_SCHEMA_VERSION",
    "IMPLEMENTATION_IDENTITY_KIND",
    "IMPLEMENTATION_IDENTITY_VERSION",
    "TeacherCapacityEvidenceError",
    "build_capacity_contract",
    "build_implementation_identity",
    "build_production_evidence",
    "canonical_sha256",
    "read_and_verify_production_evidence",
    "read_verify_production_evidence",
    "validate_production_raw_report",
    "write_evidence_json_exclusive",
    "write_production_evidence",
]
