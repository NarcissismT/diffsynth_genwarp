#!/usr/bin/env python3
"""Create the persisted overall decision after every v3.3 smoke stage passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_ISOLATION_PREFIX = "D2R_DDP_ISOLATION_OK "
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORMAL_TEACHER_SHA256 = (
    "3d079e19445168169144f2af741362f673289b6510df4a4c1af348449ae045b9"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_failure_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    world_size = int(item["world_size"])
    status = int(item["exit_status"])
    token = str(item["failure_token"])
    text = str(item["log_text"])
    if world_size < 2:
        raise ValueError("failure evidence must use at least two ranks")
    if status == 0 or status in {124, 137, 143}:
        raise ValueError(
            "failure evidence has a success/timeout/forced-termination status; "
            f"world_size={world_size}, status={status}"
        )
    expected_line = f"D2R_EXPECTED_RANK_FAILURE {token}"
    if text.splitlines().count(expected_line) != 1:
        raise ValueError(
            "failure evidence does not contain exactly one matching failure token; "
            f"world_size={world_size}"
        )
    isolation_lines = [
        line[len(_ISOLATION_PREFIX) :]
        for line in text.splitlines()
        if line.startswith(_ISOLATION_PREFIX)
    ]
    if len(isolation_lines) != 1:
        raise ValueError(
            "failure evidence does not contain exactly one isolation receipt; "
            f"world_size={world_size}"
        )
    try:
        isolation = json.loads(isolation_lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("failure isolation receipt is invalid JSON") from error
    expected_ranks = [[rank, rank, 1, 0] for rank in range(world_size)]
    if not isinstance(isolation, Mapping) or isolation.get("world_size") != world_size:
        raise ValueError("failure isolation receipt has the wrong world size")
    if isolation.get("ranks") != expected_ranks:
        raise ValueError(
            "failure isolation receipt has incomplete or non-isolated ranks; "
            f"world_size={world_size}"
        )
    return {
        "world_size": world_size,
        "exit_status": status,
        "failure_token": token,
        "rank_isolation": expected_ranks,
        "log_sha256": str(item["log_sha256"]),
        "log_path": str(item["log_path"]),
    }


def _require_sha256(value: Any, *, field: str) -> str:
    result = str(value)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _validate_functional_report(
    report: Mapping[str, Any],
    *,
    formal: bool,
    formal_config_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if report.get("schema_version") != 1:
        raise ValueError("functional smoke report has the wrong schema version")
    if report.get("kind") != "v33_real_teacher_qwen_ddp_smoke":
        raise ValueError("functional smoke report has the wrong kind")
    if report.get("scope") != "functional_substage_only":
        raise ValueError("functional smoke report has the wrong scope")
    if report.get("passed") is not True:
        raise ValueError("functional smoke substage did not pass")
    world_size = report.get("invoked_world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("functional smoke report has no valid world size")
    expected_rows = [[rank, rank] for rank in range(world_size)]
    if report.get("verified_rank_isolation") != expected_rows:
        raise ValueError("functional smoke report has incomplete rank isolation evidence")
    seed_completed = report.get("seed_completed_epochs")
    output_completed = report.get("output_completed_epochs")
    if (
        isinstance(seed_completed, bool)
        or not isinstance(seed_completed, int)
        or seed_completed < 1
        or output_completed != seed_completed + 1
    ):
        raise ValueError("functional smoke report has an invalid epoch transition")
    state_count = report.get("optimizer_state_count")
    if isinstance(state_count, bool) or not isinstance(state_count, int) or state_count < 1:
        raise ValueError("functional smoke report has no optimizer state")
    if report.get("optimizer_step_min") != 1.0 or report.get("optimizer_step_max") != 1.0:
        raise ValueError("functional smoke report does not prove exactly one Adam step")
    teacher_sha = _require_sha256(
        report.get("teacher_sha256"), field="functional teacher_sha256"
    )
    metrics = report.get("validation_metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("functional smoke report has no validation metrics")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("functional smoke report has no artifact identities")
    for name in ("config", "seed", "functional_log", "anchor", "best", "latest"):
        identity = artifacts.get(name)
        if not isinstance(identity, Mapping):
            raise ValueError(f"functional smoke report lacks artifact {name!r}")
        _require_sha256(identity.get("sha256"), field=f"artifact {name}.sha256")
        size = identity.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"functional smoke artifact {name!r} has invalid size")
    for name in ("anchor", "best", "latest"):
        if artifacts[name].get("ephemeral") is not True:
            raise ValueError(f"functional smoke artifact {name!r} must be marked ephemeral")
    verified_checkpoints = report.get("verified_checkpoints")
    if not isinstance(verified_checkpoints, Mapping) or set(verified_checkpoints) != {
        "anchor",
        "best",
        "latest",
    }:
        raise ValueError("functional smoke did not deeply verify anchor/best/latest")
    for name, item in verified_checkpoints.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"verified checkpoint {name!r} has invalid semantics")
        if item.get("output_completed_epochs") != output_completed:
            raise ValueError(f"verified checkpoint {name!r} has the wrong epoch")
        count = item.get("optimizer_state_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"verified checkpoint {name!r} has no optimizer state")
        if item.get("optimizer_step_min") != 1.0 or item.get("optimizer_step_max") != 1.0:
            raise ValueError(f"verified checkpoint {name!r} lacks one-step evidence")
        if item.get("teacher_sha256") != teacher_sha:
            raise ValueError(f"verified checkpoint {name!r} has the wrong teacher")
    report_sha = _require_sha256(
        report.get("_artifact_sha256"), field="functional report artifact SHA-256"
    )
    report_size = report.get("_artifact_size_bytes")
    if isinstance(report_size, bool) or not isinstance(report_size, int) or report_size <= 0:
        raise ValueError("functional report artifact size is invalid")
    report_path = str(report.get("_artifact_path", ""))
    if not report_path:
        raise ValueError("functional report artifact path is missing")
    if formal:
        if world_size != 8:
            raise ValueError("formal v3.3 smoke requires an eight-rank functional run")
        if seed_completed < 20:
            raise ValueError("formal v3.3 smoke requires a seed with at least 20 epochs")
        if teacher_sha != _FORMAL_TEACHER_SHA256:
            raise ValueError("formal v3.3 smoke used the wrong teacher SHA-256")
        if not isinstance(formal_config_identity, Mapping):
            raise ValueError("formal config identity is required")
        config_artifact = artifacts["config"]
        if str(config_artifact.get("path")) != str(formal_config_identity.get("path")):
            raise ValueError("formal v3.3 smoke used the wrong config path")
        if config_artifact.get("sha256") != formal_config_identity.get("sha256"):
            raise ValueError("formal v3.3 smoke used the wrong config content")
    return {
        "world_size": world_size,
        "seed_completed_epochs": seed_completed,
        "output_completed_epochs": output_completed,
        "teacher_sha256": teacher_sha,
        "report_path": report_path,
        "report_size_bytes": report_size,
        "report_sha256": report_sha,
        "config": dict(artifacts["config"]),
    }


def build_overall_report(
    functional_report: Mapping[str, Any],
    failure_evidence: Sequence[Mapping[str, Any]],
    *,
    formal: bool,
    formal_config_identity: Mapping[str, Any] | None = None,
    preflight_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all persisted evidence and return the final decision payload."""

    functional = _validate_functional_report(
        functional_report,
        formal=formal,
        formal_config_identity=formal_config_identity,
    )
    preflight_sha = _require_sha256(
        preflight_evidence.get("sha256"), field="preflight report SHA-256"
    )
    preflight_size = preflight_evidence.get("size_bytes")
    if (
        isinstance(preflight_size, bool)
        or not isinstance(preflight_size, int)
        or preflight_size <= 0
    ):
        raise ValueError("preflight report artifact size is invalid")
    if preflight_evidence.get("errors") != []:
        raise ValueError("preflight report contains errors")
    parsed = [_parse_failure_evidence(item) for item in failure_evidence]
    sizes = [item["world_size"] for item in parsed]
    if len(sizes) != len(set(sizes)):
        raise ValueError("failure evidence contains duplicate world sizes")
    if formal:
        if not {2, 8}.issubset(sizes):
            raise ValueError("formal v3.3 smoke requires both two- and eight-rank failures")
    return {
        "schema_version": 1,
        "kind": "v33_teacher_qwen_ddp_smoke_overall",
        "decision": "pass" if formal else "partial_only",
        "passed": bool(formal),
        "formal": bool(formal),
        "functional_world_size": functional["world_size"],
        "seed_completed_epochs": functional["seed_completed_epochs"],
        "teacher_sha256": functional["teacher_sha256"],
        "failure_world_sizes": sorted(sizes),
        "functional_report": {
            "path": functional["report_path"],
            "size_bytes": functional["report_size_bytes"],
            "sha256": functional["report_sha256"],
        },
        "preflight_report": {
            "path": str(preflight_evidence.get("path", "")),
            "size_bytes": preflight_size,
            "sha256": preflight_sha,
        },
        "failure_evidence": sorted(parsed, key=lambda item: item["world_size"]),
        "limitations": [
            "This is a bounded one-train-step and one-validation-batch-per-rank smoke, not a quality result.",
            "LAMA inference is outside the training smoke graph.",
        ],
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite final smoke report: {path}") from error
        temporary.unlink()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "partial"), required=True)
    parser.add_argument("--functional-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument(
        "--failure-evidence",
        action="append",
        nargs=4,
        metavar=("WORLD_SIZE", "EXIT_STATUS", "TOKEN", "LOG"),
        default=[],
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.functional_report.open("r", encoding="utf-8") as handle:
        functional = json.load(handle)
    if not isinstance(functional, dict):
        raise TypeError("functional report must be a JSON object")
    functional["_artifact_path"] = str(args.functional_report.resolve())
    functional["_artifact_sha256"] = _sha256(args.functional_report)
    functional["_artifact_size_bytes"] = args.functional_report.stat().st_size
    with args.preflight_report.open("r", encoding="utf-8") as handle:
        preflight_payload = json.load(handle)
    if not isinstance(preflight_payload, Mapping):
        raise TypeError("preflight report must be a JSON object")
    preflight_evidence = {
        "path": str(args.preflight_report.resolve()),
        "size_bytes": args.preflight_report.stat().st_size,
        "sha256": _sha256(args.preflight_report),
        "errors": preflight_payload.get("errors"),
    }
    evidence: list[dict[str, Any]] = []
    for world_size, status, token, log_value in args.failure_evidence:
        log = Path(log_value)
        if not log.is_file():
            raise FileNotFoundError(f"failure log is missing: {log}")
        evidence.append(
            {
                "world_size": int(world_size),
                "exit_status": int(status),
                "failure_token": token,
                "log_text": log.read_text(encoding="utf-8", errors="replace"),
                "log_sha256": _sha256(log),
                "log_path": str(log.resolve()),
            }
        )
    formal = args.mode == "formal"
    formal_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "unified_v3_3_teacher_anchor.yaml"
    )
    formal_config_identity = {
        "path": str(formal_config.resolve(strict=True)),
        "size_bytes": formal_config.stat().st_size,
        "sha256": _sha256(formal_config),
    }
    report = build_overall_report(
        functional,
        evidence,
        formal=formal,
        formal_config_identity=formal_config_identity if formal else None,
        preflight_evidence=preflight_evidence,
    )
    _atomic_write(args.report, report)
    token = "D2R_V33_SMOKE_PASS" if formal else "D2R_V33_SMOKE_PARTIAL_ONLY"
    print(f"{token} report={args.report}", flush=True)


if __name__ == "__main__":
    main()
