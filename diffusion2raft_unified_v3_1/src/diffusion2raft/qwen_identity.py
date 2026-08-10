"""Strict content identity and authenticated loading tree for local Qwen.

The local Qwen model is intentionally treated as a closed set of deployment
dependencies.  A manifest binds every required relative path to its byte size
and SHA-256 digest; unrelated files in the model directory are ignored.  Model
files are opened relative to an authenticated root directory descriptor and
are kept open while a temporary ``from_pretrained``-compatible tree exposes
their already-authenticated inodes through ``/proc/self/fd``.

The v1 manifest digest is SHA-256 over canonical UTF-8 JSON containing exactly
``schema_version`` and ``files`` (sorted object keys, no insignificant
whitespace).  The ``manifest_sha256`` field is therefore not self-referential.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat as stat_module
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


QWEN_MANIFEST_SCHEMA_VERSION = 1

# This is the complete set read by the Qwen-Image-Edit Diffusers pipeline used
# by this project.  Keep it explicit: globbing would make a download marker,
# README, cache file, or attacker-created file part of the release identity.
QWEN_MODEL_RELATIVE_PATHS: tuple[str, ...] = (
    "model_index.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)

_CANONICAL_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_MANIFEST_KEYS = frozenset({"schema_version", "files", "manifest_sha256"})
_PAYLOAD_KEYS = frozenset({"schema_version", "files"})
_FILE_KEYS = frozenset({"path", "size", "sha256"})


class QwenIdentityError(RuntimeError):
    """The Qwen manifest or on-disk model tree failed authentication."""


@dataclass(frozen=True)
class VerifiedQwenTree:
    """A temporary model tree whose leaves resolve to authenticated file fds."""

    load_path: str
    manifest_sha256: str
    manifest: dict[str, Any]


@dataclass
class _HeldDirectory:
    relative_path: str
    fd: int
    parent_fd: int | None
    name: str | None
    identity: tuple[int, int, int]


@dataclass
class _HeldFile:
    relative_path: str
    fd: int
    parent_fd: int
    name: str
    identity: tuple[int, int, int, int, int, int]


@dataclass
class _HeldTree:
    root_path: Path
    directories: dict[str, _HeldDirectory]
    files: dict[str, _HeldFile]


def _canonical_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _CANONICAL_SHA256.fullmatch(value) is None:
        raise QwenIdentityError(
            f"{label} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = frozenset(value.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(repr(key) for key in actual - expected)
        raise QwenIdentityError(
            f"{label} fields differ from the v1 schema: missing={missing}, extra={extra}"
        )


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise QwenIdentityError(f"{label} must be a non-empty relative POSIX path")
    if "\\" in value:
        raise QwenIdentityError(f"{label} must use POSIX '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise QwenIdentityError(f"{label} must not be absolute or contain '.'/'..'")
    canonical = "/".join(path.parts)
    if value != canonical:
        raise QwenIdentityError(f"{label} is not a canonical relative POSIX path")
    return value


def _manifest_payload(files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": QWEN_MANIFEST_SCHEMA_VERSION,
        "files": files,
    }


def _digest_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_qwen_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Validate v1 payload fields and return its canonical SHA-256 digest.

    ``manifest`` may be either the two-field digest payload or a complete v1
    manifest.  A supplied ``manifest_sha256`` is deliberately ignored by this
    calculation and is checked by :func:`validate_qwen_manifest`.
    """

    if not isinstance(manifest, Mapping):
        raise QwenIdentityError("Qwen manifest must be a mapping")
    actual_keys = frozenset(manifest.keys())
    if actual_keys not in {_PAYLOAD_KEYS, _MANIFEST_KEYS}:
        _require_exact_keys(manifest, _MANIFEST_KEYS, label="Qwen manifest")
    normalized = _validate_payload(manifest)
    return _digest_payload(normalized)


def _validate_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    version = manifest.get("schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != QWEN_MANIFEST_SCHEMA_VERSION
    ):
        raise QwenIdentityError("Qwen manifest schema_version must be integer 1")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise QwenIdentityError("Qwen manifest files must be a list")

    normalized_files: list[dict[str, Any]] = []
    observed_paths: list[str] = []
    for index, raw_entry in enumerate(raw_files):
        if not isinstance(raw_entry, Mapping):
            raise QwenIdentityError(f"Qwen manifest files[{index}] must be a mapping")
        _require_exact_keys(raw_entry, _FILE_KEYS, label=f"Qwen files[{index}]")
        path = _canonical_relative_path(
            raw_entry["path"], label=f"Qwen files[{index}].path"
        )
        size = raw_entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise QwenIdentityError(
                f"Qwen files[{index}].size must be a non-negative integer"
            )
        sha256 = _canonical_sha256(
            raw_entry["sha256"], label=f"Qwen files[{index}].sha256"
        )
        observed_paths.append(path)
        normalized_files.append({"path": path, "size": size, "sha256": sha256})

    if len(set(observed_paths)) != len(observed_paths):
        raise QwenIdentityError("Qwen manifest contains duplicate relative paths")
    if tuple(observed_paths) != QWEN_MODEL_RELATIVE_PATHS:
        missing = sorted(set(QWEN_MODEL_RELATIVE_PATHS) - set(observed_paths))
        extra = sorted(set(observed_paths) - set(QWEN_MODEL_RELATIVE_PATHS))
        raise QwenIdentityError(
            "Qwen manifest paths/order differ from the fixed v1 dependency list: "
            f"missing={missing}, extra={extra}"
        )
    return _manifest_payload(normalized_files)


def validate_qwen_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a normalized copy of an exact, self-consistent v1 manifest."""

    if not isinstance(manifest, Mapping):
        raise QwenIdentityError("Qwen manifest must be a mapping")
    _require_exact_keys(manifest, _MANIFEST_KEYS, label="Qwen manifest")
    payload = _validate_payload(manifest)
    recorded_digest = _canonical_sha256(
        manifest["manifest_sha256"], label="Qwen manifest_sha256"
    )
    calculated_digest = _digest_payload(payload)
    if recorded_digest != calculated_digest:
        raise QwenIdentityError("Qwen manifest_sha256 does not match its contents")
    if expected_manifest_sha256 is not None:
        expected_digest = _canonical_sha256(
            expected_manifest_sha256, label="expected Qwen manifest_sha256"
        )
        if calculated_digest != expected_digest:
            raise QwenIdentityError(
                "Qwen manifest digest differs from the configured expected digest"
            )
    return {**payload, "manifest_sha256": calculated_digest}


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_mode))


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_mode),
    )


def _require_directory(value: os.stat_result, *, label: str) -> None:
    if not stat_module.S_ISDIR(value.st_mode):
        raise QwenIdentityError(f"{label} is a symlink or is not a directory")


def _require_regular(value: os.stat_result, *, label: str) -> None:
    if not stat_module.S_ISREG(value.st_mode):
        raise QwenIdentityError(f"{label} is a symlink or is not a regular file")


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _file_flags() -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _hash_fd(fd: int) -> str:
    """Hash current bytes from an fd; never consult or populate a stat cache."""

    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        try:
            block = os.read(fd, _HASH_CHUNK_BYTES)
        except InterruptedError:
            continue
        except BlockingIOError as exc:
            raise QwenIdentityError(
                "Qwen regular file unexpectedly blocked while hashing"
            ) from exc
        if not block:
            os.lseek(fd, 0, os.SEEK_SET)
            return digest.hexdigest()
        digest.update(block)


def _all_relative_directories() -> tuple[str, ...]:
    directories: set[str] = set()
    for relative_path in QWEN_MODEL_RELATIVE_PATHS:
        parts = PurePosixPath(relative_path).parts[:-1]
        for length in range(1, len(parts) + 1):
            directories.add("/".join(parts[:length]))
    return tuple(sorted(directories, key=lambda value: (value.count("/"), value)))


def _open_child_directory(parent_fd: int, name: str, *, label: str) -> tuple[int, tuple[int, int, int]]:
    try:
        path_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _require_directory(path_before, label=label)
        fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except (OSError, ValueError) as exc:
        raise QwenIdentityError(f"cannot open {label} safely") from exc
    try:
        fd_stat = os.fstat(fd)
        _require_directory(fd_stat, label=label)
        identity = _directory_identity(fd_stat)
        if _directory_identity(path_before) != identity:
            raise QwenIdentityError(f"{label} changed between lstat and open")
        return fd, identity
    except BaseException:
        os.close(fd)
        raise


def _open_child_file(parent_fd: int, name: str, *, label: str) -> tuple[int, tuple[int, int, int, int, int, int]]:
    try:
        path_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _require_regular(path_before, label=label)
        fd = os.open(name, _file_flags(), dir_fd=parent_fd)
    except (OSError, ValueError) as exc:
        raise QwenIdentityError(f"cannot open {label} safely") from exc
    try:
        fd_stat = os.fstat(fd)
        _require_regular(fd_stat, label=label)
        identity = _file_identity(fd_stat)
        if _file_identity(path_before) != identity:
            raise QwenIdentityError(f"{label} changed between lstat and open")
        return fd, identity
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def _hold_qwen_tree(root_value: str | Path) -> Iterator[_HeldTree]:
    root_text = os.path.abspath(os.path.expanduser(os.fspath(root_value)))
    root_path = Path(root_text)
    root_fd: int | None = None
    directory_fds: list[int] = []
    file_fds: list[int] = []
    try:
        try:
            root_before = os.lstat(root_path)
            _require_directory(root_before, label=f"Qwen root {root_path}")
            root_fd = os.open(root_path, _directory_flags())
        except (OSError, ValueError) as exc:
            raise QwenIdentityError(f"cannot open Qwen root safely: {root_path}") from exc
        root_after = os.fstat(root_fd)
        _require_directory(root_after, label=f"Qwen root {root_path}")
        root_identity = _directory_identity(root_after)
        if _directory_identity(root_before) != root_identity:
            raise QwenIdentityError("Qwen root changed between lstat and open")

        directories: dict[str, _HeldDirectory] = {
            "": _HeldDirectory("", root_fd, None, None, root_identity)
        }
        for relative_directory in _all_relative_directories():
            path = PurePosixPath(relative_directory)
            parent_relative = "/".join(path.parts[:-1])
            parent_fd = directories[parent_relative].fd
            name = path.name
            fd, identity = _open_child_directory(
                parent_fd, name, label=f"Qwen directory {relative_directory}"
            )
            directory_fds.append(fd)
            directories[relative_directory] = _HeldDirectory(
                relative_directory, fd, parent_fd, name, identity
            )

        files: dict[str, _HeldFile] = {}
        for relative_path in QWEN_MODEL_RELATIVE_PATHS:
            path = PurePosixPath(relative_path)
            parent_relative = "/".join(path.parts[:-1])
            parent_fd = directories[parent_relative].fd
            fd, identity = _open_child_file(
                parent_fd, path.name, label=f"Qwen file {relative_path}"
            )
            file_fds.append(fd)
            files[relative_path] = _HeldFile(
                relative_path, fd, parent_fd, path.name, identity
            )
        yield _HeldTree(root_path, directories, files)
    finally:
        for fd in reversed(file_fds):
            os.close(fd)
        for fd in reversed(directory_fds):
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)


def _assert_tree_paths_unchanged(tree: _HeldTree, *, phase: str) -> None:
    root = tree.directories[""]
    try:
        root_fd_stat = os.fstat(root.fd)
        root_path_stat = os.lstat(tree.root_path)
    except OSError as exc:
        raise QwenIdentityError(f"Qwen root disappeared {phase}") from exc
    _require_directory(root_fd_stat, label="open Qwen root")
    _require_directory(root_path_stat, label="Qwen root path")
    if (
        _directory_identity(root_fd_stat) != root.identity
        or _directory_identity(root_path_stat) != root.identity
    ):
        raise QwenIdentityError(f"Qwen root changed {phase}")

    for relative_path, held in tree.directories.items():
        if not relative_path:
            continue
        try:
            fd_stat = os.fstat(held.fd)
            path_stat = os.stat(
                held.name, dir_fd=held.parent_fd, follow_symlinks=False
            )
        except OSError as exc:
            raise QwenIdentityError(
                f"Qwen directory {relative_path} disappeared {phase}"
            ) from exc
        _require_directory(fd_stat, label=f"open Qwen directory {relative_path}")
        _require_directory(path_stat, label=f"Qwen directory {relative_path}")
        if (
            _directory_identity(fd_stat) != held.identity
            or _directory_identity(path_stat) != held.identity
        ):
            raise QwenIdentityError(f"Qwen directory {relative_path} changed {phase}")

    for relative_path, held in tree.files.items():
        try:
            fd_stat = os.fstat(held.fd)
            path_stat = os.stat(
                held.name, dir_fd=held.parent_fd, follow_symlinks=False
            )
        except OSError as exc:
            raise QwenIdentityError(
                f"Qwen file {relative_path} disappeared {phase}"
            ) from exc
        _require_regular(fd_stat, label=f"open Qwen file {relative_path}")
        _require_regular(path_stat, label=f"Qwen file {relative_path}")
        if _file_identity(fd_stat) != held.identity or _file_identity(path_stat) != held.identity:
            raise QwenIdentityError(f"Qwen file {relative_path} changed {phase}")


def _hash_held_files(tree: _HeldTree) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative_path in QWEN_MODEL_RELATIVE_PATHS:
        held = tree.files[relative_path]
        before = _file_identity(os.fstat(held.fd))
        digest = _hash_fd(held.fd)
        after = _file_identity(os.fstat(held.fd))
        if before != held.identity or after != held.identity:
            raise QwenIdentityError(f"Qwen file {relative_path} changed while hashing")
        digests[relative_path] = digest
    _assert_tree_paths_unchanged(tree, phase="while hashing")
    return digests


def build_qwen_manifest(root_value: str | Path) -> dict[str, Any]:
    """Hash the fixed Qwen dependency set and return a canonical v1 manifest."""

    with _hold_qwen_tree(root_value) as tree:
        digests = _hash_held_files(tree)
        files = [
            {
                "path": relative_path,
                "size": int(os.fstat(tree.files[relative_path].fd).st_size),
                "sha256": digests[relative_path],
            }
            for relative_path in QWEN_MODEL_RELATIVE_PATHS
        ]
        payload = _manifest_payload(files)
        manifest = {**payload, "manifest_sha256": _digest_payload(payload)}
        # This also protects the builder from accidentally emitting a looser
        # representation than the parser accepts.
        return validate_qwen_manifest(manifest)


def _create_procfd_tree(tree: _HeldTree) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="diffusion2raft-qwen-"))
    try:
        temp_root.chmod(0o700)
        for relative_path in QWEN_MODEL_RELATIVE_PATHS:
            destination = temp_root.joinpath(*PurePosixPath(relative_path).parts)
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.symlink(f"/proc/self/fd/{tree.files[relative_path].fd}", destination)
        return temp_root
    except BaseException:
        shutil.rmtree(temp_root)
        raise


@contextmanager
def open_verified_qwen_tree(
    root_value: str | Path,
    manifest: Mapping[str, Any],
    *,
    expected_manifest_sha256: str | None = None,
) -> Iterator[VerifiedQwenTree]:
    """Yield a ``from_pretrained`` path backed only by authenticated open fds.

    All 37 model descriptors remain open for the full context.  On context
    exit, every path/inode/stat identity and every content digest is checked
    again before descriptors are closed and the temporary symlink tree is
    removed.  The final re-hash detects same-inode writes even on filesystems
    whose timestamp granularity is too coarse to expose a rapid rewrite.
    """

    normalized = validate_qwen_manifest(
        manifest, expected_manifest_sha256=expected_manifest_sha256
    )
    expected_by_path = {
        entry["path"]: entry for entry in normalized["files"]
    }
    temp_root: Path | None = None
    with _hold_qwen_tree(root_value) as tree:
        initial_digests = _hash_held_files(tree)
        for relative_path in QWEN_MODEL_RELATIVE_PATHS:
            entry = expected_by_path[relative_path]
            held_size = int(os.fstat(tree.files[relative_path].fd).st_size)
            if held_size != entry["size"]:
                raise QwenIdentityError(
                    f"Qwen file {relative_path} size differs from its manifest"
                )
            if initial_digests[relative_path] != entry["sha256"]:
                raise QwenIdentityError(
                    f"Qwen file {relative_path} sha256 differs from its manifest"
                )

        if not Path("/proc/self/fd").is_dir():
            raise QwenIdentityError(
                "/proc/self/fd is required for authenticated Qwen model loading"
            )
        temp_root = _create_procfd_tree(tree)
        try:
            yield VerifiedQwenTree(
                load_path=str(temp_root),
                manifest_sha256=normalized["manifest_sha256"],
                manifest={
                    "schema_version": normalized["schema_version"],
                    "files": [dict(entry) for entry in normalized["files"]],
                    "manifest_sha256": normalized["manifest_sha256"],
                },
            )
        finally:
            try:
                _assert_tree_paths_unchanged(tree, phase="while in use")
                final_digests = _hash_held_files(tree)
                for relative_path in QWEN_MODEL_RELATIVE_PATHS:
                    if final_digests[relative_path] != expected_by_path[relative_path]["sha256"]:
                        raise QwenIdentityError(
                            f"Qwen file {relative_path} content changed while in use"
                        )
            finally:
                shutil.rmtree(temp_root)


__all__ = [
    "QWEN_MANIFEST_SCHEMA_VERSION",
    "QWEN_MODEL_RELATIVE_PATHS",
    "QwenIdentityError",
    "VerifiedQwenTree",
    "build_qwen_manifest",
    "canonical_qwen_manifest_digest",
    "open_verified_qwen_tree",
    "validate_qwen_manifest",
]
