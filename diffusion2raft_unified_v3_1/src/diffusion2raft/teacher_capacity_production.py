"""Production orchestration for the frozen teacher-capacity audit.

The lower-level capacity modules intentionally have separate responsibilities:
the preflight produces a raw diagnostic, the asset module authenticates the
complete validation set, the evidence module seals a passing report, and the
receipt module produces the small value consumed by training.  This module is
the fail-closed production entry point that joins those contracts together.

Approved evidence is reached only through a canonical ``approved.json``
pointer in the same directory.  Raw audit reports are retained for diagnosis,
but their names cannot be confused with content-addressed evidence files.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import hmac
import json
import os
import re
import stat as stat_module
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import load_config
from .external_file import canonical_sha256, open_stable_external_file
from .teacher_capacity_assets import (
    build_teacher_capacity_assets_identity,
    fast_verify_teacher_capacity_assets,
)
from .teacher_capacity_evidence import (
    build_production_evidence,
    canonical_sha256 as canonical_json_sha256,
    read_verify_production_evidence,
    write_production_evidence,
)
from .teacher_capacity_receipt import (
    build_teacher_capacity_receipt,
    encode_teacher_capacity_receipt_base64,
    strict_validate_teacher_capacity_receipt,
)


DEFAULT_CONFIG_PATH = Path("configs/unified_v3_3_teacher_anchor.yaml")
DEFAULT_POINTER_PATH = Path("runs/preflight_v33_teacher_capacity/approved.json")
DEFAULT_ROTATION_BIN_EDGES = (
    0.0,
    15.0,
    30.0,
    60.0,
    90.0,
    120.0,
    150.0,
    180.0,
)

POINTER_VERSION = 1
POINTER_KIND = "teacher_capacity_evidence_pointer"
CHECKPOINT_RECEIPT_KEY = "capacity_evidence_receipt"
LEGACY_AMBIGUOUS_RECEIPT_KEY = "teacher_capacity_evidence_receipt"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EVIDENCE_FILENAME_RE = re.compile(r"([0-9a-f]{64})\.json\Z")
_POINTER_KEYS = frozenset(
    {
        "version",
        "kind",
        "evidence_filename",
        "evidence_report_sha256",
        "pointer_sha256",
    }
)
_TEACHER_RECEIPT_BINDING_KEYS = (
    "sha256",
    "file_size",
    "input_size",
    "flow_size",
    "blur_kernel",
    "autocast_dtype",
    "requires_logical_cuda0",
)


class TeacherCapacityProductionError(RuntimeError):
    """A production audit, pointer, or resume checkpoint failed validation."""


@dataclass(frozen=True)
class _ProductionInputs:
    config_path: Path
    config: dict[str, Any]
    config_sha256: str
    project_root: Path
    teacher_path: Path
    teacher_sha256: str
    manifest_path: Path


@dataclass(frozen=True)
class _CheckpointInfo:
    path: Path
    identity: dict[str, Any]
    payload: Mapping[str, Any]
    backend: str
    stage: str
    epoch_index: int
    completed_epochs: int


def _fail(message: str) -> None:
    raise TeacherCapacityProductionError(message)


def run_capacity_preflight(**kwargs: Any) -> dict[str, Any]:
    """Lazily import the GPU audit so ``verify --help`` stays lightweight."""

    from .teacher_capacity_preflight import run_capacity_preflight as implementation

    return implementation(**kwargs)


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
        raise TeacherCapacityProductionError(
            f"value is not canonical-JSON compatible: {exc}"
        ) from exc


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
    return bool(left == right)


def _absolute_path(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TeacherCapacityProductionError("path is not filesystem-compatible") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        _fail("path must be a non-empty string without NUL bytes")
    return Path(os.path.abspath(os.path.expanduser(raw)))


def _stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_mode),
    )


def _regular_without_symlinks(
    value: str | os.PathLike[str], *, label: str
) -> Path:
    path = _absolute_path(value)
    try:
        resolved = path.resolve(strict=True)
        path_stat = os.lstat(path)
    except (OSError, RuntimeError) as exc:
        raise TeacherCapacityProductionError(f"{label} is missing") from exc
    if resolved != path:
        _fail(f"{label} contains a symlink")
    if not stat_module.S_ISREG(path_stat.st_mode):
        _fail(f"{label} is a symlink or is not a regular file")
    return path


def _read_canonical_regular_json(
    value: str | os.PathLike[str], *, label: str
) -> tuple[dict[str, Any], bytes, Path]:
    """Read canonical JSON from one unchanged, non-symlink regular file."""

    path = _regular_without_symlinks(value, label=label)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        before_path = os.lstat(path)
        fd = os.open(path, flags)
    except OSError as exc:
        raise TeacherCapacityProductionError(f"{label} cannot be opened safely") from exc
    try:
        before_fd = os.fstat(fd)
        if (
            not stat_module.S_ISREG(before_fd.st_mode)
            or _stable_stat(before_fd) != _stable_stat(before_path)
        ):
            _fail(f"{label} changed between lstat and open")
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(fd, 1024 * 1024)
            except InterruptedError:
                continue
            except BlockingIOError as exc:
                raise TeacherCapacityProductionError(
                    f"{label} unexpectedly blocked while reading"
                ) from exc
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after_fd = os.fstat(fd)
        try:
            after_path = os.lstat(path)
        except OSError as exc:
            raise TeacherCapacityProductionError(
                f"{label} disappeared while reading"
            ) from exc
        if (
            _stable_stat(after_fd) != _stable_stat(before_fd)
            or _stable_stat(after_path) != _stable_stat(before_fd)
            or path.resolve(strict=True) != path
        ):
            _fail(f"{label} changed while reading")
    finally:
        os.close(fd)

    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TeacherCapacityProductionError(f"{label} is not valid JSON") from exc
    if type(decoded) is not dict:
        _fail(f"{label} root must be an exact dict")
    if payload != _canonical_json(decoded):
        _fail(f"{label} is not canonical JSON")
    return decoded, payload, path


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
                raise
    finally:
        os.close(directory_fd)


def _ensure_safe_directory(value: str | os.PathLike[str]) -> Path:
    directory = _absolute_path(value)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        resolved = directory.resolve(strict=True)
        status = os.lstat(directory)
    except (OSError, RuntimeError) as exc:
        raise TeacherCapacityProductionError(
            f"output directory cannot be resolved safely: {directory}"
        ) from exc
    if resolved != directory:
        _fail("output directory contains a symlink")
    if not stat_module.S_ISDIR(status.st_mode):
        _fail("output directory is not a directory")
    return directory


def build_teacher_capacity_evidence_pointer(
    evidence_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the strict pointer body for one content-addressed evidence file."""

    path = _regular_without_symlinks(evidence_path, label="production evidence")
    match = _EVIDENCE_FILENAME_RE.fullmatch(path.name)
    if match is None:
        _fail("production evidence filename must be '<sha256>.json'")
    _, payload, _ = _read_canonical_regular_json(path, label="production evidence")
    digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(match.group(1), digest):
        _fail("production evidence filename differs from its canonical contents")
    body: dict[str, Any] = {
        "version": POINTER_VERSION,
        "kind": POINTER_KIND,
        "evidence_filename": path.name,
        "evidence_report_sha256": digest,
    }
    return {**body, "pointer_sha256": hashlib.sha256(_canonical_json(body)).hexdigest()}


def strict_validate_teacher_capacity_evidence_pointer(value: Any) -> dict[str, Any]:
    """Validate the exact v1 pointer schema without filesystem access."""

    if type(value) is not dict:
        _fail("teacher-capacity evidence pointer must be an exact dict")
    if set(value) != _POINTER_KEYS:
        _fail(
            "teacher-capacity evidence pointer fields differ from the exact v1 schema"
        )
    if type(value["version"]) is not int or value["version"] != POINTER_VERSION:
        _fail("pointer.version must be integer 1")
    if type(value["kind"]) is not str or value["kind"] != POINTER_KIND:
        _fail(f"pointer.kind must be {POINTER_KIND!r}")
    filename = value["evidence_filename"]
    if type(filename) is not str or not filename or "\x00" in filename:
        _fail("pointer.evidence_filename must be a non-empty exact string")
    filename_path = Path(filename)
    if (
        filename_path.is_absolute()
        or filename_path.name != filename
        or filename in {".", ".."}
        or len(filename_path.parts) != 1
    ):
        _fail("pointer.evidence_filename must be a basename without '..'")
    match = _EVIDENCE_FILENAME_RE.fullmatch(filename)
    if match is None:
        _fail("pointer.evidence_filename must be '<sha256>.json'")
    report_sha = value["evidence_report_sha256"]
    if type(report_sha) is not str or _SHA256_RE.fullmatch(report_sha) is None:
        _fail("pointer.evidence_report_sha256 must be canonical lowercase SHA-256")
    if not hmac.compare_digest(match.group(1), report_sha):
        _fail("pointer evidence filename and report SHA-256 differ")
    supplied_sha = value["pointer_sha256"]
    if type(supplied_sha) is not str or _SHA256_RE.fullmatch(supplied_sha) is None:
        _fail("pointer.pointer_sha256 must be canonical lowercase SHA-256")
    body = {key: value[key] for key in value if key != "pointer_sha256"}
    expected_sha = hashlib.sha256(_canonical_json(body)).hexdigest()
    if not hmac.compare_digest(supplied_sha, expected_sha):
        _fail("pointer.pointer_sha256 does not match its canonical body")
    return json.loads(_canonical_json(value))


def read_teacher_capacity_evidence_pointer(
    pointer_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], Path]:
    """Read a stable canonical pointer and resolve its sibling evidence file."""

    pointer, _, path = _read_canonical_regular_json(
        pointer_path, label="teacher-capacity evidence pointer"
    )
    validated = strict_validate_teacher_capacity_evidence_pointer(pointer)
    evidence_path = path.parent / validated["evidence_filename"]
    _regular_without_symlinks(evidence_path, label="production evidence")
    return validated, evidence_path


def _atomic_publish_pointer(pointer_path: Path, pointer: Mapping[str, Any]) -> Path:
    validated = strict_validate_teacher_capacity_evidence_pointer(dict(pointer))
    payload = _canonical_json(validated)
    path = _absolute_path(pointer_path)
    directory = _ensure_safe_directory(path.parent)
    if path.parent != directory:
        _fail("pointer must be written in the authenticated output directory")
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise TeacherCapacityProductionError("existing pointer cannot be inspected") from exc
    if existing is not None and not stat_module.S_ISREG(existing.st_mode):
        _fail("existing pointer is a symlink or is not a regular file")

    temporary = directory / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            current = os.lstat(path)
        except FileNotFoundError:
            current = None
        if current is not None and not stat_module.S_ISREG(current.st_mode):
            _fail("existing pointer became a symlink or non-regular file")
        os.replace(temporary, path)
        _fsync_directory(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _resolve_configured_path(value: Any, *, project_root: Path, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        _fail(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TeacherCapacityProductionError(f"{label} is missing") from exc


def _load_production_inputs(config_path: str | os.PathLike[str]) -> _ProductionInputs:
    try:
        resolved_config = _absolute_path(config_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TeacherCapacityProductionError("production config is missing") from exc
    with open_stable_external_file(resolved_config, label="production config") as opened:
        config = load_config(opened.load_path)
        config_sha = canonical_sha256(
            opened.identity["sha256"], label="production config sha256"
        )
    project_root = resolved_config.parent.parent
    model = config.get("model")
    data = config.get("data")
    if type(model) is not dict:
        _fail("config.model must be an exact mapping")
    if type(data) is not dict:
        _fail("config.data must be an exact mapping")
    backend = model.get("prior_backend")
    if type(backend) is not str or backend.lower() != "torchscript":
        _fail("production config requires model.prior_backend='torchscript'")
    teacher_pin = canonical_sha256(
        model.get("prior_torchscript_sha256"),
        label="config.model.prior_torchscript_sha256",
    )
    teacher_path = _resolve_configured_path(
        model.get("prior_torchscript_path"),
        project_root=project_root,
        label="config.model.prior_torchscript_path",
    )
    manifest_path = _resolve_configured_path(
        data.get("val_manifest"),
        project_root=project_root,
        label="config.data.val_manifest",
    )
    return _ProductionInputs(
        config_path=resolved_config,
        config=config,
        config_sha256=config_sha,
        project_root=project_root,
        teacher_path=teacher_path,
        teacher_sha256=teacher_pin,
        manifest_path=manifest_path,
    )


def _torch_load(source: str) -> Mapping[str, Any]:
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        _fail("resume checkpoint payload must be a mapping")
    return payload


def _checkpoint_backend(payload: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    declared_values: list[str] = []
    declared = payload.get("prior_backend")
    if declared is not None:
        declared_values.append(str(declared).lower())
    saved_config = payload.get("config")
    saved_model = saved_config.get("model") if isinstance(saved_config, Mapping) else None
    if isinstance(saved_model, Mapping) and saved_model.get("prior_backend") is not None:
        declared_values.append(str(saved_model.get("prior_backend")).lower())
    if any(value not in {"learned", "torchscript"} for value in declared_values):
        _fail(f"resume checkpoint has unknown prior_backend metadata: {declared_values}")
    if len(set(declared_values)) > 1:
        _fail("resume checkpoint has conflicting prior_backend metadata")
    marker_present = "prior._teacher_backend_marker" in model
    backend = declared_values[0] if declared_values else (
        "torchscript" if marker_present else "learned"
    )
    if backend == "learned" and marker_present:
        _fail("learned resume checkpoint contains a teacher backend marker")
    if backend == "torchscript" and not marker_present:
        _fail("torchscript resume checkpoint has no teacher backend marker")
    return backend


def _load_checkpoint_info(
    path_value: str | os.PathLike[str], *, expected_backend: str | None = None
) -> _CheckpointInfo:
    try:
        path = _absolute_path(path_value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TeacherCapacityProductionError("resume checkpoint is missing") from exc
    with open_stable_external_file(path, label="resume checkpoint") as opened:
        payload = _torch_load(opened.load_path)
        identity = dict(opened.identity)
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    if not isinstance(model, Mapping) or not model:
        _fail("resume checkpoint has no non-empty model state")
    if not isinstance(optimizer, Mapping) or not optimizer:
        _fail("resume checkpoint has no non-empty optimizer state")
    stage = payload.get("stage")
    if type(stage) is not str or stage != "unified":
        _fail("resume checkpoint stage must be exactly 'unified'")
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch < 19:
        _fail("resume checkpoint epoch_index must be an integer >= 19")
    backend = _checkpoint_backend(payload, model)
    if expected_backend is not None and backend != expected_backend:
        _fail(
            f"resume checkpoint prior backend must be {expected_backend!r}; got {backend!r}"
        )
    if backend == "learned":
        if (
            CHECKPOINT_RECEIPT_KEY in payload
            or LEGACY_AMBIGUOUS_RECEIPT_KEY in payload
        ):
            _fail("learned resume checkpoint contains a teacher-capacity receipt")
    return _CheckpointInfo(
        path=path,
        identity=identity,
        payload=payload,
        backend=backend,
        stage=stage,
        epoch_index=epoch,
        completed_epochs=epoch + 1,
    )


def _mapping_path(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            _fail(f"missing required field: {'.'.join(keys)}")
        current = current[key]
    return current


def _validate_generated_report(
    report: Mapping[str, Any],
    *,
    inputs: _ProductionInputs,
    checkpoint: _CheckpointInfo,
) -> None:
    """Enforce orchestration details not independently frozen by evidence v1."""

    if not isinstance(report, Mapping):
        _fail("capacity preflight did not return a report mapping")
    exact = (
        (("protocol", "requested_sample_count"), 0),
        (("protocol", "selected_sample_count"), 300),
        (("protocol", "source_rotation", "rotations_per_sample"), 1),
        (
            ("protocol", "source_rotation", "bin_edges_deg"),
            list(DEFAULT_ROTATION_BIN_EDGES),
        ),
        (("protocol", "source_full_geometry", "enabled"), True),
        (("protocol", "source_full_geometry", "transformations_per_sample"), 1),
        (("protocol", "external_file_sha256_enabled"), True),
        (("identities", "teacher", "backend"), "torchscript"),
        (("identities", "teacher", "selection", "source"), "config"),
        (("identities", "teacher", "selection", "override_value"), None),
        (("identities", "manifest", "split"), "val"),
        (("identities", "manifest", "record_count"), 300),
    )
    for path, expected in exact:
        actual = _mapping_path(report, *path)
        if not _typed_equal(actual, expected):
            _fail(
                f"raw audit violates production protocol at {'.'.join(path)}: "
                f"expected {expected!r}, got {actual!r}"
            )
    digests = (
        (
            _mapping_path(report, "identities", "config", "sha256"),
            inputs.config_sha256,
            "config",
        ),
        (
            _mapping_path(report, "identities", "checkpoint", "sha256"),
            checkpoint.identity["sha256"],
            "migration checkpoint",
        ),
        (
            _mapping_path(
                report, "identities", "teacher", "checkpoint", "sha256"
            ),
            inputs.teacher_sha256,
            "teacher",
        ),
    )
    for actual, expected, label in digests:
        if type(actual) is not str or not hmac.compare_digest(actual, expected):
            _fail(f"raw audit {label} SHA-256 differs from the authenticated input")

    configured_teacher = _mapping_path(
        report, "identities", "teacher", "selection", "config_resolved_path"
    )
    effective_teacher = _mapping_path(
        report, "identities", "teacher", "selection", "effective_path"
    )
    for label, value in (
        ("configured teacher", configured_teacher),
        ("effective teacher", effective_teacher),
    ):
        if type(value) is not str:
            _fail(f"raw audit {label} path must be an exact string")
        try:
            resolved = Path(value).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TeacherCapacityProductionError(
                f"raw audit {label} path cannot be resolved"
            ) from exc
        if resolved != inputs.teacher_path:
            _fail(f"raw audit {label} path is not the config teacher")
    manifest_value = _mapping_path(
        report, "identities", "manifest", "configured_path"
    )
    if type(manifest_value) is not str:
        _fail("raw audit manifest path must be an exact string")
    try:
        report_manifest = Path(manifest_value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TeacherCapacityProductionError(
            "raw audit manifest path cannot be resolved"
        ) from exc
    if report_manifest != inputs.manifest_path:
        _fail("raw audit manifest is not config.data.val_manifest")


def _receipt_from_bindings(
    bindings: Mapping[str, Any], *, migration_seed: Mapping[str, Any]
) -> dict[str, Any]:
    required = (
        "evidence_report_sha256",
        "capacity_contract_sha256",
        "capacity_config_projection_sha256",
        "protocol_sha256",
        "implementation_sha256",
        "policy",
        "teacher",
        "manifest",
    )
    missing = [key for key in required if key not in bindings]
    if missing:
        _fail(f"verified evidence bindings are incomplete: {missing}")
    return build_teacher_capacity_receipt(
        evidence_report_sha256=bindings["evidence_report_sha256"],
        capacity_contract_sha256=bindings["capacity_contract_sha256"],
        capacity_config_projection_sha256=bindings[
            "capacity_config_projection_sha256"
        ],
        protocol_sha256=bindings["protocol_sha256"],
        implementation_sha256=bindings["implementation_sha256"],
        policy=dict(bindings["policy"]),
        migration_seed=dict(migration_seed),
        teacher=dict(bindings["teacher"]),
        manifest=dict(bindings["manifest"]),
    )


def _read_and_verify_evidence_from_pointer(
    *,
    pointer_path: str | os.PathLike[str],
    inputs: _ProductionInputs,
    repository_root: str | os.PathLike[str] | None,
) -> tuple[dict[str, Any], dict[str, Any], Path, bytes, bytes]:
    pointer, pointer_payload, canonical_pointer_path = _read_canonical_regular_json(
        pointer_path, label="teacher-capacity evidence pointer"
    )
    validated_pointer = strict_validate_teacher_capacity_evidence_pointer(pointer)
    evidence_path = canonical_pointer_path.parent / validated_pointer[
        "evidence_filename"
    ]
    evidence, evidence_payload, evidence_path = _read_canonical_regular_json(
        evidence_path, label="production evidence"
    )
    actual_sha = hashlib.sha256(evidence_payload).hexdigest()
    if not hmac.compare_digest(
        actual_sha, validated_pointer["evidence_report_sha256"]
    ):
        _fail("pointer does not bind the current production evidence bytes")

    stored_config_sha = _mapping_path(
        evidence, "raw_audit", "identities", "config", "sha256"
    )
    if type(stored_config_sha) is not str or not hmac.compare_digest(
        stored_config_sha, inputs.config_sha256
    ):
        _fail("current production config SHA-256 differs from approved evidence")

    stored_assets = evidence.get("asset_identity")
    if type(stored_assets) is not dict:
        _fail("production evidence has no exact validation asset identity")
    expected_asset_digest = _mapping_path(
        evidence,
        "capacity_contract",
        "asset_identity_binding",
        "aggregate_sha256",
    )
    verified_assets = fast_verify_teacher_capacity_assets(
        inputs.manifest_path,
        stored_assets,
        expected_aggregate_sha256=expected_asset_digest,
    )
    bindings = read_verify_production_evidence(
        evidence_path,
        configured_teacher_sha256=inputs.teacher_sha256,
        repository_root=repository_root,
        current_teacher_path=inputs.teacher_path,
        current_manifest_path=inputs.manifest_path,
        asset_identity=verified_assets,
    )

    # Detect ordinary replacement/retargeting across the longer verification.
    pointer_after, pointer_payload_after, _ = _read_canonical_regular_json(
        canonical_pointer_path, label="teacher-capacity evidence pointer"
    )
    evidence_after, evidence_payload_after, _ = _read_canonical_regular_json(
        evidence_path, label="production evidence"
    )
    if (
        pointer_payload_after != pointer_payload
        or evidence_payload_after != evidence_payload
        or not _typed_equal(pointer_after, pointer)
        or not _typed_equal(evidence_after, evidence)
    ):
        _fail("pointer or production evidence changed during verification")
    return bindings, evidence, evidence_path, pointer_payload, evidence_payload


def _configured_resume(inputs: _ProductionInputs) -> Path:
    train = inputs.config.get("train")
    if type(train) is not dict:
        _fail("config.train must be an exact mapping")
    return _resolve_configured_path(
        train.get("resume"),
        project_root=inputs.project_root,
        label="config.train.resume",
    )


def generate_teacher_capacity_production(
    *,
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
    pointer_path: str | os.PathLike[str] = DEFAULT_POINTER_PATH,
    resume_path: str | os.PathLike[str] | None = None,
    output_directory: str | os.PathLike[str] | None = None,
    seed: int = 42,
    batch_size: int = 1,
    device: str = "cuda:0",
    threads: int = 16,
    repository_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run, seal, and atomically approve the exact production audit."""

    inputs = _load_production_inputs(config_path)
    pointer = _absolute_path(pointer_path)
    directory = _ensure_safe_directory(
        pointer.parent if output_directory is None else output_directory
    )
    if pointer.parent != directory:
        _fail("approved pointer and production evidence must share one directory")
    resume = _configured_resume(inputs) if resume_path is None else _absolute_path(
        resume_path
    ).resolve(strict=True)

    # Authenticate and semantically inspect the migration payload before any
    # GPU work or validation-asset approval begins.
    checkpoint = _load_checkpoint_info(resume, expected_backend="learned")
    assets = build_teacher_capacity_assets_identity(inputs.manifest_path)

    torch.set_num_threads(max(1, int(threads)))
    raw_output = directory / f"raw-audit-{uuid.uuid4().hex}.json"
    report = run_capacity_preflight(
        config_path=inputs.config_path,
        output_path=raw_output,
        checkpoint_path=checkpoint.path,
        teacher_path=None,
        manifest_path=None,
        split="val",
        sample_count=0,
        explicit_rotation_angles=None,
        rotations_per_sample=1,
        full_geometry_per_sample=1,
        rotation_bin_edges=DEFAULT_ROTATION_BIN_EDGES,
        seed=int(seed),
        batch_size=int(batch_size),
        device=device,
        hash_external_files=True,
    )
    _validate_generated_report(report, inputs=inputs, checkpoint=checkpoint)
    verified_assets = fast_verify_teacher_capacity_assets(
        inputs.manifest_path,
        assets,
        expected_aggregate_sha256=assets["aggregate_sha256"],
    )
    evidence = build_production_evidence(
        report,
        configured_teacher_sha256=inputs.teacher_sha256,
        repository_root=repository_root,
        asset_identity=verified_assets,
    )
    evidence_path = write_production_evidence(directory, evidence)
    _regular_without_symlinks(evidence_path, label="production evidence")

    bindings = read_verify_production_evidence(
        evidence_path,
        configured_teacher_sha256=inputs.teacher_sha256,
        repository_root=repository_root,
        current_teacher_path=inputs.teacher_path,
        current_manifest_path=inputs.manifest_path,
        asset_identity=verified_assets,
    )
    if not hmac.compare_digest(
        bindings["migration_seed"]["sha256"], checkpoint.identity["sha256"]
    ):
        _fail("sealed evidence migration seed differs from the loaded checkpoint")
    migration_seed = {
        "sha256": checkpoint.identity["sha256"],
        "stage": checkpoint.stage,
        "epoch_index": checkpoint.epoch_index,
        "completed_epochs": checkpoint.completed_epochs,
    }
    receipt = _receipt_from_bindings(bindings, migration_seed=migration_seed)
    strict_validate_teacher_capacity_receipt(receipt)
    encoded = encode_teacher_capacity_receipt_base64(receipt)

    pointer_value = build_teacher_capacity_evidence_pointer(evidence_path)
    published_pointer = _atomic_publish_pointer(pointer, pointer_value)
    return {
        "evidence_path": str(evidence_path),
        "pointer_path": str(published_pointer),
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_b64": encoded,
    }


def _validate_teacher_checkpoint_identity(
    checkpoint: _CheckpointInfo,
    *,
    receipt: Mapping[str, Any],
    configured_teacher_path: Path,
) -> None:
    identity = checkpoint.payload.get("teacher_prior_identity")
    required = {
        "version",
        "resolved_path",
        "file_size",
        "mtime_ns",
        "sha256",
        "input_size",
        "flow_size",
        "blur_kernel",
        "autocast_dtype",
    }
    optional = {"requires_logical_cuda0"}
    if not isinstance(identity, Mapping) or not (
        required <= set(identity) and set(identity) <= required | optional
    ):
        _fail("teacher resume has invalid teacher_prior_identity schema")
    if type(identity["version"]) is not int or identity["version"] != 2:
        _fail("teacher resume teacher_prior_identity must be strict version 2")
    for key in ("file_size", "mtime_ns", "input_size", "flow_size", "blur_kernel"):
        if type(identity[key]) is not int:
            _fail(f"teacher_prior_identity.{key} must be an exact integer")
    for key in ("resolved_path", "autocast_dtype"):
        if type(identity[key]) is not str or not identity[key]:
            _fail(f"teacher_prior_identity.{key} must be a non-empty exact string")
    identity_sha = canonical_sha256(
        identity["sha256"], label="teacher_prior_identity.sha256"
    )
    requires_cuda0 = identity.get("requires_logical_cuda0", False)
    if type(requires_cuda0) is not bool:
        _fail("teacher_prior_identity.requires_logical_cuda0 must be boolean")
    normalized_identity = dict(identity)
    normalized_identity["requires_logical_cuda0"] = requires_cuda0
    receipt_teacher = receipt["teacher"]
    differing = [
        key
        for key in _TEACHER_RECEIPT_BINDING_KEYS
        if key not in normalized_identity
        or not _typed_equal(receipt_teacher[key], normalized_identity[key])
    ]
    if differing:
        _fail(
            "teacher resume receipt differs from teacher_prior_identity; "
            f"differing_fields={differing}"
        )
    try:
        identity_path = Path(identity["resolved_path"]).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TeacherCapacityProductionError(
            "teacher_prior_identity.resolved_path is missing"
        ) from exc
    if identity_path != configured_teacher_path:
        _fail("teacher resume identity path differs from the config teacher")
    with open_stable_external_file(
        identity_path,
        expected_sha256=identity_sha,
        label="teacher resume external teacher",
    ) as opened:
        if opened.identity["file_size"] != identity["file_size"]:
            _fail("teacher resume external teacher size differs from its identity")
        if opened.identity["mtime_ns"] != identity["mtime_ns"]:
            _fail("teacher resume external teacher mtime differs from its identity")
    saved_config = checkpoint.payload.get("config")
    saved_model = saved_config.get("model") if isinstance(saved_config, Mapping) else None
    if not isinstance(saved_model, Mapping):
        _fail("teacher resume has no config.model metadata")
    if str(saved_model.get("prior_backend", "")).lower() != "torchscript":
        _fail("teacher resume config.model.prior_backend is not torchscript")
    if saved_model.get("prior_torchscript_sha256") != identity_sha:
        _fail("teacher resume config teacher pin differs from teacher identity")


def verify_teacher_capacity_production(
    *,
    resume_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
    pointer_path: str | os.PathLike[str] = DEFAULT_POINTER_PATH,
    repository_root: str | os.PathLike[str] | None = None,
) -> str:
    """Verify an approved environment and return its canonical receipt base64."""

    inputs = _load_production_inputs(config_path)
    bindings, _, _, _, _ = _read_and_verify_evidence_from_pointer(
        pointer_path=pointer_path,
        inputs=inputs,
        repository_root=repository_root,
    )
    checkpoint = _load_checkpoint_info(resume_path)
    evidence_seed_sha = bindings["migration_seed"]["sha256"]

    if checkpoint.backend == "learned":
        if not hmac.compare_digest(checkpoint.identity["sha256"], evidence_seed_sha):
            _fail("learned resume SHA-256 differs from evidence migration seed")
        migration_seed = {
            "sha256": checkpoint.identity["sha256"],
            "stage": checkpoint.stage,
            "epoch_index": checkpoint.epoch_index,
            "completed_epochs": checkpoint.completed_epochs,
        }
        receipt = _receipt_from_bindings(bindings, migration_seed=migration_seed)
    else:
        if LEGACY_AMBIGUOUS_RECEIPT_KEY in checkpoint.payload:
            _fail("teacher resume contains an ambiguous receipt field")
        if CHECKPOINT_RECEIPT_KEY not in checkpoint.payload:
            _fail("teacher resume has no stored teacher-capacity receipt")
        try:
            stored_receipt = strict_validate_teacher_capacity_receipt(
                checkpoint.payload[CHECKPOINT_RECEIPT_KEY]
            )
        except ValueError as exc:
            raise TeacherCapacityProductionError(
                f"teacher resume stored receipt is invalid: {exc}"
            ) from exc
        stored_seed = stored_receipt["migration_seed"]
        if not hmac.compare_digest(stored_seed["sha256"], evidence_seed_sha):
            _fail("teacher resume receipt migration seed differs from evidence")
        receipt = _receipt_from_bindings(bindings, migration_seed=stored_seed)
        if not _typed_equal(receipt, stored_receipt):
            _fail("teacher resume stored receipt differs from approved evidence")
        _validate_teacher_checkpoint_identity(
            checkpoint,
            receipt=stored_receipt,
            configured_teacher_path=inputs.teacher_path,
        )
    strict_validate_teacher_capacity_receipt(receipt)
    return encode_teacher_capacity_receipt_base64(receipt)


# Short aliases for callers that treat the subcommands as an API.
generate = generate_teacher_capacity_production
verify = verify_teacher_capacity_production


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="run and atomically approve the fixed production audit"
    )
    generate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    generate_parser.add_argument("--pointer", type=Path)
    generate_parser.add_argument("--resume", type=Path)
    generate_parser.add_argument("--output-dir", type=Path)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--batch-size", type=int, default=1)
    generate_parser.add_argument("--device", default="cuda:0")
    generate_parser.add_argument("--threads", type=int, default=16)

    verify_parser = subparsers.add_parser(
        "verify", help="verify the pointer and current training resume"
    )
    verify_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    verify_parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER_PATH)
    verify_parser.add_argument("--resume", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            output_directory = args.output_dir
            if args.pointer is None:
                pointer = (
                    DEFAULT_POINTER_PATH
                    if output_directory is None
                    else Path(output_directory) / "approved.json"
                )
            else:
                pointer = args.pointer
            result = generate_teacher_capacity_production(
                config_path=args.config,
                pointer_path=pointer,
                resume_path=args.resume,
                output_directory=output_directory,
                seed=args.seed,
                batch_size=args.batch_size,
                device=args.device,
                threads=args.threads,
            )
            print(_canonical_json(result).decode("utf-8"), flush=True)
            return 0
        encoded = verify_teacher_capacity_production(
            config_path=args.config,
            pointer_path=args.pointer,
            resume_path=args.resume,
        )
        print(encoded, flush=True)
        return 0
    except Exception as exc:
        print(f"teacher-capacity production {args.command} failed: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "CHECKPOINT_RECEIPT_KEY",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_POINTER_PATH",
    "POINTER_KIND",
    "POINTER_VERSION",
    "TeacherCapacityProductionError",
    "build_teacher_capacity_evidence_pointer",
    "generate",
    "generate_teacher_capacity_production",
    "main",
    "read_teacher_capacity_evidence_pointer",
    "strict_validate_teacher_capacity_evidence_pointer",
    "verify",
    "verify_teacher_capacity_production",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
