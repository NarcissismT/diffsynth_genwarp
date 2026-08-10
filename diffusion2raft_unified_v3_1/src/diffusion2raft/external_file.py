"""Stable byte identities for external deployment artifacts.

Large TorchScript assets stay outside unified checkpoints, so path, size, and
mtime are not a sufficient binding: a same-sized replacement can restore its
mtime.  This module hashes an already-open regular-file descriptor and proves
that the descriptor and canonical path name the same unchanged object before
and after hashing.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat as stat_module
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


_CANONICAL_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class ExternalFileIdentityError(RuntimeError):
    """An external artifact could not be given a stable byte identity."""


@dataclass(frozen=True)
class StableExternalFile:
    """An authenticated open file and the procfd path bound to that inode."""

    identity: dict[str, Any]
    load_path: str


def canonical_sha256(value: Any, *, label: str = "sha256") -> str:
    """Return *value* only when it is canonical lowercase SHA-256 hex."""

    if not isinstance(value, str) or _CANONICAL_SHA256.fullmatch(value) is None:
        raise ExternalFileIdentityError(
            f"{label} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
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
        raise ExternalFileIdentityError(f"{label} is not a regular file")


def _hash_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        try:
            block = os.read(fd, _HASH_CHUNK_BYTES)
        except InterruptedError:
            continue
        except BlockingIOError as exc:  # A verified regular file cannot block.
            raise ExternalFileIdentityError(
                "external regular file unexpectedly blocked while hashing"
            ) from exc
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _assert_unchanged(
    fd: int,
    resolved: Path,
    expected: tuple[int, ...],
    *,
    phase: str,
) -> None:
    fd_stat = os.fstat(fd)
    try:
        path_stat = os.lstat(resolved)
    except OSError as exc:
        raise ExternalFileIdentityError(
            f"external file disappeared {phase}: {resolved}"
        ) from exc
    _require_regular(fd_stat, label=f"external file {resolved}")
    _require_regular(path_stat, label=f"external file {resolved}")
    if (
        _stable_stat_identity(fd_stat) != expected
        or _stable_stat_identity(path_stat) != expected
    ):
        raise ExternalFileIdentityError(
            f"external file changed {phase}: {resolved}"
        )


@contextmanager
def open_stable_external_file(
    path_value: str | Path,
    *,
    expected_sha256: str | None = None,
    label: str = "external file",
) -> Iterator[StableExternalFile]:
    """Authenticate and hold one fd for a consumer such as ``torch.jit.load``.

    ``load_path`` is ``/proc/self/fd/<fd>`` rather than a Python file object:
    current PyTorch reads an entire file object into one ``bytes`` allocation,
    which is unsafe for the 3.5GB teacher.  Opening the procfd still resolves to
    this already-authenticated inode and never re-resolves the configured path.
    """

    expected_sha = (
        None
        if expected_sha256 is None
        else canonical_sha256(
            expected_sha256, label=f"{label} expected sha256"
        )
    )

    try:
        resolved = Path(path_value).expanduser().resolve(strict=True)
        path_before = os.lstat(resolved)
    except (OSError, RuntimeError) as exc:
        raise ExternalFileIdentityError(
            f"external file cannot be resolved safely: {path_value}"
        ) from exc
    _require_regular(path_before, label=f"external file {resolved}")

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(resolved, flags)
    except OSError as exc:
        raise ExternalFileIdentityError(
            f"external file cannot be opened safely: {resolved}"
        ) from exc

    try:
        fd_before = os.fstat(fd)
        _require_regular(fd_before, label=f"external file {resolved}")
        before_identity = _stable_stat_identity(fd_before)
        if _stable_stat_identity(path_before) != before_identity:
            raise ExternalFileIdentityError(
                f"external file changed between lstat and open: {resolved}"
            )
        # Never memoize a content digest from stat fields.  JuiceFS can expose
        # the same ctime_ns for two rapid same-inode rewrites, so even a key
        # containing inode/size/mtime/ctime can otherwise return stale bytes.
        # Callers avoid redundant multi-GB hashes by propagating the identity
        # produced by the authenticated loader, not by trusting a stat cache.
        sha256 = _hash_fd(fd)
        _assert_unchanged(fd, resolved, before_identity, phase="while hashing")

        identity = {
            "resolved_path": str(resolved),
            "file_size": int(fd_before.st_size),
            "mtime_ns": int(fd_before.st_mtime_ns),
            "sha256": canonical_sha256(sha256),
        }
        if expected_sha is not None:
            if identity["sha256"] != expected_sha:
                raise ExternalFileIdentityError(
                    f"{label} sha256 differs from configured expected digest"
                )

        procfd = Path("/proc/self/fd") / str(fd)
        if not procfd.exists():
            raise ExternalFileIdentityError(
                "/proc/self/fd is required for bounded-memory stable model loading"
            )
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            yield StableExternalFile(dict(identity), str(procfd))
        finally:
            _assert_unchanged(fd, resolved, before_identity, phase="while in use")
    finally:
        os.close(fd)


def stable_external_file_identity(
    path_value: str | Path,
    *,
    expected_sha256: str | None = None,
    label: str = "external file",
) -> dict[str, Any]:
    """Return a stable public identity without retaining the authenticated fd."""

    with open_stable_external_file(
        path_value,
        expected_sha256=expected_sha256,
        label=label,
    ) as opened:
        return dict(opened.identity)


def validate_external_file_identity(
    identity: Mapping[str, Any],
    *,
    path_key: str,
    size_key: str,
    label: str,
) -> dict[str, Any]:
    """Re-hash an artifact and require exact persisted size/mtime/SHA-256."""

    configured = identity.get(path_key)
    if not isinstance(configured, str) or not configured:
        raise ExternalFileIdentityError(f"{label} identity has no path")
    expected_size = identity.get(size_key)
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise ExternalFileIdentityError(f"{label} identity has invalid size")
    expected_mtime = identity.get("mtime_ns")
    if isinstance(expected_mtime, bool) or not isinstance(expected_mtime, int):
        raise ExternalFileIdentityError(f"{label} identity has invalid mtime_ns")
    expected_sha256 = canonical_sha256(
        identity.get("sha256"), label=f"{label} identity sha256"
    )

    current = stable_external_file_identity(configured)
    if expected_size != current["file_size"]:
        raise ExternalFileIdentityError(
            f"{label} external file size differs from recorded identity"
        )
    if expected_mtime != current["mtime_ns"]:
        raise ExternalFileIdentityError(
            f"{label} external file mtime differs from recorded identity"
        )
    if expected_sha256 != current["sha256"]:
        raise ExternalFileIdentityError(
            f"{label} external file sha256 differs from recorded identity"
        )
    return current


__all__ = [
    "ExternalFileIdentityError",
    "StableExternalFile",
    "canonical_sha256",
    "open_stable_external_file",
    "stable_external_file_identity",
    "validate_external_file_identity",
]
