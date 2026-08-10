#!/usr/bin/env python3
"""Re-verify the persisted formal v3.3 smoke report before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class SmokeReportVerificationError(ValueError):
    pass


def _fail(message: str) -> None:
    raise SmokeReportVerificationError(message)


def _stable_read(path: Path, *, label: str) -> tuple[bytes, Path]:
    absolute = path.expanduser().absolute()
    try:
        path_stat = os.lstat(absolute)
    except OSError as exc:
        raise SmokeReportVerificationError(f"{label} is missing: {absolute}") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        _fail(f"{label} is a symlink or is not a regular file: {absolute}")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        final_path_stat = os.lstat(absolute)
    finally:
        os.close(descriptor)
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after) or identity(before) != identity(final_path_stat):
        _fail(f"{label} changed while it was being read: {absolute}")
    if before.st_size <= 0:
        _fail(f"{label} is empty: {absolute}")
    return b"".join(chunks), absolute.resolve(strict=True)


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes, Path]:
    payload_bytes, resolved = _stable_read(path, label=label)
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeReportVerificationError(f"{label} is not valid JSON") from exc
    if type(payload) is not dict:
        _fail(f"{label} must be an exact JSON object")
    return payload, payload_bytes, resolved


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{label} must be a canonical lowercase SHA-256")
    return value


def verify_formal_smoke_report(
    *,
    overall_report: Path,
    expected_seed: Path,
    expected_seed_sha256: str,
    expected_config: Path,
    expected_teacher_sha256: str,
) -> dict[str, Any]:
    expected_seed_sha256 = _require_sha256(
        expected_seed_sha256, label="expected seed SHA-256"
    )
    expected_teacher_sha256 = _require_sha256(
        expected_teacher_sha256, label="expected teacher SHA-256"
    )
    seed_bytes, seed_path = _stable_read(expected_seed, label="migration seed")
    if _sha256(seed_bytes) != expected_seed_sha256:
        _fail("current migration seed differs from its pinned SHA-256")
    config_bytes, config_path = _stable_read(expected_config, label="formal config")
    config_sha256 = _sha256(config_bytes)

    overall, overall_bytes, overall_path = _load_json(
        overall_report, label="overall smoke report"
    )
    exact_overall = {
        "schema_version": 1,
        "kind": "v33_teacher_qwen_ddp_smoke_overall",
        "decision": "pass",
        "passed": True,
        "formal": True,
        "functional_world_size": 8,
        "seed_completed_epochs": 20,
        "teacher_sha256": expected_teacher_sha256,
        "failure_world_sizes": [2, 8],
    }
    differing = [key for key, value in exact_overall.items() if overall.get(key) != value]
    if differing:
        _fail(f"overall smoke report is not the exact formal pass; differing={differing}")
    failure_evidence = overall.get("failure_evidence")
    if type(failure_evidence) is not list or len(failure_evidence) != 2:
        _fail("overall smoke report must contain exactly two failure evidence rows")
    for item, world_size in zip(failure_evidence, (2, 8)):
        expected_isolation = [[rank, rank, 1, 0] for rank in range(world_size)]
        if type(item) is not dict or item.get("world_size") != world_size:
            _fail("failure evidence rows must be ordered world sizes 2 and 8")
        status_value = item.get("exit_status")
        if type(status_value) is not int or status_value == 0 or status_value in {124, 137, 143}:
            _fail(f"failure evidence world_size={world_size} has an invalid exit status")
        if item.get("rank_isolation") != expected_isolation:
            _fail(f"failure evidence world_size={world_size} lacks exact rank isolation")
        _require_sha256(
            item.get("log_sha256"),
            label=f"failure evidence world_size={world_size} log SHA-256",
        )

    reference = overall.get("functional_report")
    if type(reference) is not dict:
        _fail("overall smoke report has no functional_report identity")
    functional_path_value = reference.get("path")
    if type(functional_path_value) is not str or not functional_path_value:
        _fail("functional_report.path must be a non-empty string")
    functional, functional_bytes, functional_path = _load_json(
        Path(functional_path_value), label="functional smoke report"
    )
    expected_functional_path = overall_path.parent / "functional_report.json"
    if functional_path != expected_functional_path.resolve(strict=True):
        _fail("overall report references a functional report outside its report directory")
    if reference.get("size_bytes") != len(functional_bytes):
        _fail("functional report size differs from the overall report binding")
    if _require_sha256(reference.get("sha256"), label="functional report SHA-256") \
        != _sha256(functional_bytes):
        _fail("functional report content differs from the overall report binding")

    exact_functional = {
        "schema_version": 1,
        "kind": "v33_real_teacher_qwen_ddp_smoke",
        "scope": "functional_substage_only",
        "passed": True,
        "invoked_world_size": 8,
        "seed_completed_epochs": 20,
        "output_completed_epochs": 21,
        "teacher_sha256": expected_teacher_sha256,
    }
    differing = [key for key, value in exact_functional.items() if functional.get(key) != value]
    if differing:
        _fail(f"functional smoke report differs from the formal contract; differing={differing}")
    verified_checkpoints = functional.get("verified_checkpoints")
    if type(verified_checkpoints) is not dict or set(verified_checkpoints) != {
        "anchor",
        "best",
        "latest",
    }:
        _fail("functional smoke did not verify exact anchor/best/latest checkpoints")
    for name, item in verified_checkpoints.items():
        if (
            type(item) is not dict
            or item.get("output_completed_epochs") != 21
            or type(item.get("optimizer_state_count")) is not int
            or item["optimizer_state_count"] < 1
            or item.get("optimizer_step_min") != 1.0
            or item.get("optimizer_step_max") != 1.0
            or item.get("teacher_sha256") != expected_teacher_sha256
        ):
            _fail(f"functional smoke checkpoint {name!r} lacks exact one-step evidence")
    artifacts = functional.get("artifacts")
    if type(artifacts) is not dict:
        _fail("functional smoke report has no artifacts")
    seed_artifact = artifacts.get("seed")
    config_artifact = artifacts.get("config")
    if type(seed_artifact) is not dict or type(config_artifact) is not dict:
        _fail("functional smoke report lacks seed/config artifacts")
    if seed_artifact.get("source_path") != str(seed_path):
        _fail("functional smoke used a different seed path")
    if seed_artifact.get("sha256") != expected_seed_sha256:
        _fail("functional smoke used a different seed SHA-256")
    if config_artifact.get("path") != str(config_path):
        _fail("functional smoke used a different config path")
    if config_artifact.get("sha256") != config_sha256:
        _fail("functional smoke used different config bytes")

    return {
        "overall_report": str(overall_path),
        "overall_report_sha256": _sha256(overall_bytes),
        "functional_report": str(functional_path),
        "functional_report_sha256": _sha256(functional_bytes),
        "seed_sha256": expected_seed_sha256,
        "config_sha256": config_sha256,
        "teacher_sha256": expected_teacher_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overall-report", type=Path, required=True)
    parser.add_argument("--expected-seed", type=Path, required=True)
    parser.add_argument("--expected-seed-sha256", required=True)
    parser.add_argument("--expected-config", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    args = parser.parse_args()
    result = verify_formal_smoke_report(
        overall_report=args.overall_report,
        expected_seed=args.expected_seed,
        expected_seed_sha256=args.expected_seed_sha256,
        expected_config=args.expected_config,
        expected_teacher_sha256=args.expected_teacher_sha256,
    )
    print("D2R_V33_FORMAL_SMOKE_REPORT_VERIFIED " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
