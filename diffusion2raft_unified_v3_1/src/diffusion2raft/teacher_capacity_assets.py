"""Stable identities for the production teacher-capacity validation assets.

The capacity audit evaluates the ground-truth validation tuple consumed by
``DocumentFlowDataset``: ``warped``, ``target``, ``flow``, and optional
``valid``.  ``guide`` is deliberately excluded because it is a conditioning
input to the student, not an input to the teacher-capacity measurement.  Each
record reference records that exclusion explicitly so it cannot be mistaken
for an accidentally omitted dependency.

Generation hashes stable, already-open regular files.  Fast verification
rehashes the JSONL manifest itself, but checks data assets by resolved path and
stat identity only; it is intended for cheap repeated preflight after a
content-hashed identity has been approved and pinned by its aggregate digest.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import stat as stat_module
from collections.abc import Mapping
from pathlib import Path
from typing import Any


TEACHER_CAPACITY_ASSET_SCHEMA_VERSION = 1
TEACHER_CAPACITY_ASSET_KIND = "teacher_capacity_validation_assets"
TEACHER_CAPACITY_RECORD_COUNT = 300
GUIDE_EXCLUDED_REASON = "not_used_by_teacher_capacity_audit"

_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOP_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "manifest",
        "record_count",
        "unique_assets",
        "record_refs",
        "aggregate_sha256",
    }
)
_PAYLOAD_KEYS = _TOP_KEYS - {"aggregate_sha256"}
_FILE_KEYS = frozenset(
    {"path", "size", "mtime_ns", "device", "inode", "sha256"}
)
_ASSET_KEYS = _FILE_KEYS | {"asset_index"}
_RECORD_KEYS = frozenset(
    {
        "dataset_index",
        "id",
        "warped",
        "target",
        "flow",
        "valid",
        "guide_excluded_reason",
    }
)
_REQUIRED_ROLES = ("warped", "target", "flow")
_ALL_ROLES = (*_REQUIRED_ROLES, "valid")


class TeacherCapacityAssetsError(RuntimeError):
    """The validation manifest or one of its assets failed authentication."""


def _fail(message: str) -> None:
    raise TeacherCapacityAssetsError(message)


def _exact_dict(value: Any, *, label: str, keys: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact dict")
    actual = set(value)
    if actual != keys:
        _fail(
            f"{label} fields differ from schema: "
            f"missing={sorted(keys - actual)}, extra={sorted(actual - keys, key=str)}"
        )
    return value


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be exactly 64 lowercase hexadecimal characters")
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherCapacityAssetsError(
            f"teacher-capacity asset identity is not canonical-JSON compatible: {exc}"
        ) from exc


def _canonical_absolute_path(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        _fail(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or os.path.abspath(value) != value:
        _fail(f"{label} must be a normalized absolute path")
    return value


def _normalize_file_identity(
    value: Any, *, label: str, include_index: bool
) -> dict[str, Any]:
    keys = _ASSET_KEYS if include_index else _FILE_KEYS
    entry = _exact_dict(value, label=label, keys=keys)
    normalized: dict[str, Any] = {}
    if include_index:
        normalized["asset_index"] = _exact_int(
            entry["asset_index"], label=f"{label}.asset_index"
        )
    normalized.update(
        {
            "path": _canonical_absolute_path(entry["path"], label=f"{label}.path"),
            "size": _exact_int(entry["size"], label=f"{label}.size"),
            "mtime_ns": _exact_int(
                entry["mtime_ns"], label=f"{label}.mtime_ns", minimum=-(2**63)
            ),
            "device": _exact_int(entry["device"], label=f"{label}.device"),
            "inode": _exact_int(entry["inode"], label=f"{label}.inode"),
            "sha256": _sha256(entry["sha256"], label=f"{label}.sha256"),
        }
    )
    return normalized


def _normalize_payload(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        _fail("teacher-capacity asset identity must be a mapping")
    actual = frozenset(identity.keys())
    if actual not in {_PAYLOAD_KEYS, _TOP_KEYS}:
        # Produce the useful strict-schema diagnostic against a complete value.
        _exact_dict(identity, label="identity", keys=_TOP_KEYS)

    version = identity.get("schema_version")
    if type(version) is not int or version != TEACHER_CAPACITY_ASSET_SCHEMA_VERSION:
        _fail("identity.schema_version must be integer 1")
    if identity.get("kind") != TEACHER_CAPACITY_ASSET_KIND:
        _fail(f"identity.kind must be {TEACHER_CAPACITY_ASSET_KIND!r}")
    record_count = identity.get("record_count")
    if type(record_count) is not int or record_count != TEACHER_CAPACITY_RECORD_COUNT:
        _fail(f"identity.record_count must be {TEACHER_CAPACITY_RECORD_COUNT}")

    manifest = _normalize_file_identity(
        identity.get("manifest"), label="identity.manifest", include_index=False
    )
    raw_assets = identity.get("unique_assets")
    if type(raw_assets) is not list:
        _fail("identity.unique_assets must be a list")
    assets: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    observed_indexes: set[int] = set()
    for position, raw_asset in enumerate(raw_assets):
        asset = _normalize_file_identity(
            raw_asset,
            label=f"identity.unique_assets[{position}]",
            include_index=True,
        )
        index = asset["asset_index"]
        if index in observed_indexes:
            _fail("identity.unique_assets contains a duplicate asset_index")
        if index != position:
            _fail("identity.unique_assets asset_index values must be contiguous and ordered")
        if asset["path"] in observed_paths:
            _fail("identity.unique_assets contains a duplicate resolved path")
        observed_indexes.add(index)
        observed_paths.add(asset["path"])
        assets.append(asset)

    raw_refs = identity.get("record_refs")
    if type(raw_refs) is not list:
        _fail("identity.record_refs must be a list")
    if len(raw_refs) != TEACHER_CAPACITY_RECORD_COUNT:
        _fail(
            f"identity.record_refs must contain exactly {TEACHER_CAPACITY_RECORD_COUNT} records"
        )
    refs: list[dict[str, Any]] = []
    observed_dataset_indexes: set[int] = set()
    referenced_asset_indexes: set[int] = set()
    for position, raw_ref in enumerate(raw_refs):
        ref = _exact_dict(
            raw_ref, label=f"identity.record_refs[{position}]", keys=_RECORD_KEYS
        )
        dataset_index = _exact_int(
            ref["dataset_index"],
            label=f"identity.record_refs[{position}].dataset_index",
        )
        if dataset_index in observed_dataset_indexes:
            _fail("identity.record_refs contains a duplicate dataset_index")
        if dataset_index != position:
            _fail("identity.record_refs dataset_index values must be contiguous and ordered")
        observed_dataset_indexes.add(dataset_index)
        record_id = ref["id"]
        if type(record_id) is not str:
            _fail(f"identity.record_refs[{position}].id must be an exact string")

        normalized_ref: dict[str, Any] = {
            "dataset_index": dataset_index,
            "id": record_id,
        }
        for role in _ALL_ROLES:
            asset_index = ref[role]
            if role == "valid" and asset_index is None:
                normalized_ref[role] = None
                continue
            asset_index = _exact_int(
                asset_index, label=f"identity.record_refs[{position}].{role}"
            )
            if asset_index >= len(assets):
                _fail(
                    f"identity.record_refs[{position}].{role} references a missing asset"
                )
            normalized_ref[role] = asset_index
            referenced_asset_indexes.add(asset_index)
        if ref["guide_excluded_reason"] != GUIDE_EXCLUDED_REASON:
            _fail(
                f"identity.record_refs[{position}].guide_excluded_reason must be "
                f"{GUIDE_EXCLUDED_REASON!r}"
            )
        normalized_ref["guide_excluded_reason"] = GUIDE_EXCLUDED_REASON
        refs.append(normalized_ref)

    if referenced_asset_indexes != set(range(len(assets))):
        _fail("identity.unique_assets contains an unreferenced asset")

    return {
        "schema_version": TEACHER_CAPACITY_ASSET_SCHEMA_VERSION,
        "kind": TEACHER_CAPACITY_ASSET_KIND,
        "manifest": manifest,
        "record_count": TEACHER_CAPACITY_RECORD_COUNT,
        "unique_assets": assets,
        "record_refs": refs,
    }


def canonical_teacher_capacity_asset_digest(identity: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 of the strict payload (excluding itself)."""

    return hashlib.sha256(_canonical_json(_normalize_payload(identity))).hexdigest()


def validate_teacher_capacity_asset_identity(
    identity: Mapping[str, Any], *, expected_aggregate_sha256: str | None = None
) -> dict[str, Any]:
    """Validate the exact v1 schema and self digest without filesystem I/O."""

    if type(identity) is not dict:
        _fail("teacher-capacity asset identity must be an exact dict")
    _exact_dict(identity, label="identity", keys=_TOP_KEYS)
    payload = _normalize_payload(identity)
    recorded = _sha256(
        identity["aggregate_sha256"], label="identity.aggregate_sha256"
    )
    calculated = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if not hmac.compare_digest(recorded, calculated):
        _fail("identity.aggregate_sha256 does not match its canonical contents")
    if expected_aggregate_sha256 is not None:
        expected = _sha256(
            expected_aggregate_sha256, label="expected aggregate_sha256"
        )
        if not hmac.compare_digest(calculated, expected):
            _fail("identity aggregate digest differs from the configured expected digest")
    return copy.deepcopy({**payload, "aggregate_sha256": calculated})


def _stat_tuple(value: os.stat_result) -> tuple[int, ...]:
    # ctime and mode are used to detect instability while reading.  They are
    # intentionally absent from the public fast-verification contract.
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_mode),
    )


def _require_regular(value: os.stat_result, *, label: str) -> None:
    if not stat_module.S_ISREG(value.st_mode):
        _fail(f"{label} is a symlink or is not a regular file")


def _resolve_regular(path_value: str | os.PathLike[str], *, label: str) -> Path:
    try:
        raw = os.fspath(path_value)
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise ValueError("invalid path")
        lexical = Path(os.path.abspath(raw))
        resolved = Path(raw).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TeacherCapacityAssetsError(f"{label} is missing or cannot be resolved") from exc
    # Dataset path lookup follows the original name.  Reject every symlinked
    # component, rather than silently recording only its current target.
    if lexical != resolved:
        _fail(f"{label} contains a symlink")
    try:
        path_stat = os.lstat(lexical)
    except OSError as exc:
        raise TeacherCapacityAssetsError(f"{label} is missing") from exc
    _require_regular(path_stat, label=label)
    return lexical


def _file_flags() -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _read_and_hash_fd(fd: int) -> tuple[bytes, str]:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        try:
            block = os.read(fd, _HASH_CHUNK_BYTES)
        except InterruptedError:
            continue
        except BlockingIOError as exc:
            raise TeacherCapacityAssetsError(
                "regular file unexpectedly blocked while hashing"
            ) from exc
        if not block:
            return b"".join(chunks), digest.hexdigest()
        chunks.append(block)
        digest.update(block)


def _hash_fd(fd: int) -> str:
    """Hash an asset with bounded memory and without a stat-keyed cache."""

    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        try:
            block = os.read(fd, _HASH_CHUNK_BYTES)
        except InterruptedError:
            continue
        except BlockingIOError as exc:
            raise TeacherCapacityAssetsError(
                "regular file unexpectedly blocked while hashing"
            ) from exc
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _stable_file_identity(
    path_value: str | os.PathLike[str], *, label: str, retain_bytes: bool
) -> tuple[dict[str, Any], bytes | None]:
    resolved = _resolve_regular(path_value, label=label)
    try:
        before_path = os.lstat(resolved)
        fd = os.open(resolved, _file_flags())
    except OSError as exc:
        raise TeacherCapacityAssetsError(f"{label} cannot be opened safely") from exc
    try:
        before_fd = os.fstat(fd)
        _require_regular(before_fd, label=label)
        before = _stat_tuple(before_fd)
        if _stat_tuple(before_path) != before:
            _fail(f"{label} changed between stat and open")
        if retain_bytes:
            content, digest = _read_and_hash_fd(fd)
        else:
            content, digest = None, _hash_fd(fd)
        after_fd = os.fstat(fd)
        try:
            after_path = os.lstat(resolved)
        except OSError as exc:
            raise TeacherCapacityAssetsError(f"{label} disappeared while hashing") from exc
        if _stat_tuple(after_fd) != before or _stat_tuple(after_path) != before:
            _fail(f"{label} changed while hashing")
        if _resolve_regular(resolved, label=label) != resolved:
            _fail(f"{label} path was retargeted while hashing")
        identity = {
            "path": str(resolved),
            "size": int(before_fd.st_size),
            "mtime_ns": int(before_fd.st_mtime_ns),
            "device": int(before_fd.st_dev),
            "inode": int(before_fd.st_ino),
            "sha256": digest,
        }
        return identity, content
    finally:
        os.close(fd)


def _parse_manifest(content: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TeacherCapacityAssetsError(f"{label} is not valid UTF-8") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise TeacherCapacityAssetsError(
                f"{label} line {line_number} is not valid JSON"
            ) from exc
        if type(record) is not dict:
            _fail(f"{label} line {line_number} must contain a JSON object")
        records.append(record)
    if len(records) != TEACHER_CAPACITY_RECORD_COUNT:
        _fail(
            f"{label} must contain exactly {TEACHER_CAPACITY_RECORD_COUNT} records; "
            f"found {len(records)}"
        )
    return records


def _record_path(root: Path, record: dict[str, Any], role: str, index: int) -> Path | None:
    value = record.get(role)
    if role == "valid" and value is None:
        return None
    if type(value) is not str or not value or "\x00" in value:
        _fail(f"manifest record {index} {role!r} must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _record_id(record: dict[str, Any], index: int) -> str:
    # This deliberately matches DocumentFlowDataset.__getitem__ exactly.
    return str(record.get("id", index))


def build_teacher_capacity_asset_manifest(
    manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Hash and bind one exact 300-record validation JSONL manifest."""

    manifest, content = _stable_file_identity(
        manifest_path, label="validation manifest", retain_bytes=True
    )
    assert content is not None
    records = _parse_manifest(content, label="validation manifest")
    root = Path(manifest["path"]).parent

    assets: list[dict[str, Any]] = []
    asset_by_path: dict[str, int] = {}
    refs: list[dict[str, Any]] = []
    for dataset_index, record in enumerate(records):
        ref: dict[str, Any] = {
            "dataset_index": dataset_index,
            "id": _record_id(record, dataset_index),
        }
        for role in _ALL_ROLES:
            configured = _record_path(root, record, role, dataset_index)
            if configured is None:
                ref[role] = None
                continue
            resolved = _resolve_regular(
                configured, label=f"manifest record {dataset_index} {role}"
            )
            path_key = str(resolved)
            asset_index = asset_by_path.get(path_key)
            if asset_index is None:
                file_identity, _ = _stable_file_identity(
                    resolved,
                    label=f"manifest record {dataset_index} {role}",
                    retain_bytes=False,
                )
                asset_index = len(assets)
                assets.append({"asset_index": asset_index, **file_identity})
                asset_by_path[path_key] = asset_index
            else:
                _compare_disk_stat(
                    assets[asset_index],
                    label=f"manifest record {dataset_index} {role}",
                )
            ref[role] = asset_index
        ref["guide_excluded_reason"] = GUIDE_EXCLUDED_REASON
        refs.append(ref)

    # Prove that early assets and the manifest name were not replaced while
    # the rest of the (potentially large) asset set was being hashed.
    _compare_disk_stat(manifest, label="validation manifest")
    for asset in assets:
        _compare_disk_stat(asset, label=f"asset {asset['asset_index']}")

    payload: dict[str, Any] = {
        "schema_version": TEACHER_CAPACITY_ASSET_SCHEMA_VERSION,
        "kind": TEACHER_CAPACITY_ASSET_KIND,
        "manifest": manifest,
        "record_count": TEACHER_CAPACITY_RECORD_COUNT,
        "unique_assets": assets,
        "record_refs": refs,
    }
    payload["aggregate_sha256"] = canonical_teacher_capacity_asset_digest(payload)
    return validate_teacher_capacity_asset_identity(payload)


def _compare_disk_stat(recorded: Mapping[str, Any], *, label: str) -> Path:
    resolved = _resolve_regular(recorded["path"], label=label)
    if str(resolved) != recorded["path"]:
        _fail(f"{label} resolved path differs from recorded identity")
    try:
        current = os.lstat(resolved)
    except OSError as exc:
        raise TeacherCapacityAssetsError(f"{label} is missing") from exc
    actual = {
        "path": str(resolved),
        "size": int(current.st_size),
        "mtime_ns": int(current.st_mtime_ns),
        "device": int(current.st_dev),
        "inode": int(current.st_ino),
    }
    for key, actual_value in actual.items():
        if recorded[key] != actual_value:
            _fail(f"{label} {key} differs from recorded identity")
    return resolved


def fast_verify_teacher_capacity_asset_manifest(
    manifest_path: str | os.PathLike[str],
    identity: Mapping[str, Any],
    *,
    expected_aggregate_sha256: str | None = None,
) -> dict[str, Any]:
    """Fast-check schema, manifest bytes, path bindings, and asset stat identity.

    Asset SHA-256 values remain covered by ``aggregate_sha256`` but asset bytes
    are intentionally not read here.  The JSONL manifest is stably read and
    rehashed on every call.
    """

    validated = validate_teacher_capacity_asset_identity(
        identity, expected_aggregate_sha256=expected_aggregate_sha256
    )
    current_manifest, content = _stable_file_identity(
        manifest_path, label="validation manifest", retain_bytes=True
    )
    assert content is not None
    recorded_manifest = validated["manifest"]
    for key in _FILE_KEYS:
        if current_manifest[key] != recorded_manifest[key]:
            _fail(f"validation manifest {key} differs from recorded identity")
    records = _parse_manifest(content, label="validation manifest")
    root = Path(current_manifest["path"]).parent
    assets = validated["unique_assets"]

    # Check every unique on-disk inode once, then prove every current manifest
    # role still resolves to the referenced unique asset.
    for asset in assets:
        _compare_disk_stat(asset, label=f"asset {asset['asset_index']}")
    for dataset_index, (record, ref) in enumerate(
        zip(records, validated["record_refs"])
    ):
        if ref["dataset_index"] != dataset_index:
            _fail(f"record {dataset_index} dataset_index differs")
        if ref["id"] != _record_id(record, dataset_index):
            _fail(f"record {dataset_index} id differs from the manifest")
        for role in _ALL_ROLES:
            configured = _record_path(root, record, role, dataset_index)
            expected_index = ref[role]
            if configured is None:
                if expected_index is not None:
                    _fail(f"record {dataset_index} valid reference differs")
                continue
            if expected_index is None:
                _fail(f"record {dataset_index} {role} reference is missing")
            resolved = _resolve_regular(
                configured, label=f"manifest record {dataset_index} {role}"
            )
            if str(resolved) != assets[expected_index]["path"]:
                _fail(f"manifest record {dataset_index} {role} path was retargeted")
            _compare_disk_stat(
                assets[expected_index],
                label=f"manifest record {dataset_index} {role}",
            )
    return copy.deepcopy(validated)


# Clear identity-oriented aliases for callers that do not treat the returned
# mapping as a file to be written.
build_teacher_capacity_assets_identity = build_teacher_capacity_asset_manifest
fast_verify_teacher_capacity_assets = fast_verify_teacher_capacity_asset_manifest
canonical_teacher_capacity_assets_digest = canonical_teacher_capacity_asset_digest
canonical_aggregate_sha256 = canonical_teacher_capacity_asset_digest


__all__ = [
    "GUIDE_EXCLUDED_REASON",
    "TEACHER_CAPACITY_ASSET_KIND",
    "TEACHER_CAPACITY_ASSET_SCHEMA_VERSION",
    "TEACHER_CAPACITY_RECORD_COUNT",
    "TeacherCapacityAssetsError",
    "build_teacher_capacity_asset_manifest",
    "build_teacher_capacity_assets_identity",
    "canonical_aggregate_sha256",
    "canonical_teacher_capacity_asset_digest",
    "canonical_teacher_capacity_assets_digest",
    "fast_verify_teacher_capacity_asset_manifest",
    "fast_verify_teacher_capacity_assets",
    "validate_teacher_capacity_asset_identity",
]
