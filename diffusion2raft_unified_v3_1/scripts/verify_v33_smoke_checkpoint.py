#!/usr/bin/env python3
"""Verify that a v3.3 smoke run performed one real optimizer step."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import torch

from checkpoint_status import inspect_checkpoint_payload
from diffusion2raft.config import load_config


REQUIRED_METRICS = (
    "epe",
    "prior_epe",
    "epe_gain",
    "final_win_rate",
    "fold_rate",
    "jacobian_p01",
    "line_epe",
    "prior_line_epe",
    "line_epe_gain",
    "line_straightness_error",
    "prior_line_straightness_error",
    "line_straightness_gain",
    "residual_application_scale",
)

_ISOLATION_LINE = re.compile(
    r"^\[info\] global_rank=(\d+) physical_local_rank=(\d+) "
    r"device='[^']+' -> logical cuda:0$",
    re.MULTILINE,
)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"smoke validation metric {name!r} is missing or non-numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"smoke validation metric {name!r} is non-finite")
    return result


def parse_functional_rank_isolation(
    text: str, *, expected_world_size: int
) -> list[tuple[int, int]]:
    """Extract and strictly validate the per-worker isolation evidence."""

    rows = [(int(rank), int(local_rank)) for rank, local_rank in _ISOLATION_LINE.findall(text)]
    expected = [(rank, rank) for rank in range(expected_world_size)]
    if sorted(rows) != expected or len(rows) != expected_world_size:
        raise ValueError(
            "functional log does not prove exactly one isolated worker per rank; "
            f"expected={expected}, observed={rows}"
        )
    return sorted(rows)


def verify_smoke_payloads(
    seed: Mapping[str, Any],
    output: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    expected_seed_completed_epochs: int,
    invoked_world_size: int,
    observed_rank_isolation: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Pure contract checks used by the CLI and CPU unit tests."""

    if invoked_world_size < 1:
        raise ValueError("invoked_world_size must be positive")
    isolation_rows = [(int(rank), int(local_rank)) for rank, local_rank in observed_rank_isolation]
    expected_rows = [(rank, rank) for rank in range(invoked_world_size)]
    if sorted(isolation_rows) != expected_rows or len(isolation_rows) != invoked_world_size:
        raise ValueError(
            "rank-isolation evidence disagrees with invoked_world_size; "
            f"expected={expected_rows}, observed={isolation_rows}"
        )
    seed_epoch = seed.get("epoch")
    if isinstance(seed_epoch, bool) or not isinstance(seed_epoch, int):
        raise ValueError("seed checkpoint epoch must be an integer")
    seed_completed = seed_epoch + 1
    if seed_completed != expected_seed_completed_epochs:
        raise ValueError(
            "seed completed epoch mismatch; "
            f"expected={expected_seed_completed_epochs}, got={seed_completed}"
        )
    seed_model = seed.get("model")
    if not isinstance(seed_model, Mapping) or any(
        key == "prior._teacher_backend_marker" for key in seed_model
    ):
        raise ValueError("smoke seed must be a learned-prior unified checkpoint")
    if str(seed.get("stage")) != "unified":
        raise ValueError("smoke seed stage must be unified")
    if str(seed.get("prior_backend", "learned")) != "learned":
        raise ValueError("smoke seed prior_backend must be learned")

    output_epoch = output.get("epoch")
    if output_epoch != seed_epoch + 1:
        raise ValueError(
            "smoke output must advance exactly one epoch index; "
            f"seed={seed_epoch}, output={output_epoch}"
        )
    if str(output.get("stage")) != "unified":
        raise ValueError("smoke output stage must be unified")
    if str(output.get("prior_backend")) != "torchscript":
        raise ValueError("smoke output prior_backend must be torchscript")
    model = output.get("model")
    if not isinstance(model, Mapping) or "prior._teacher_backend_marker" not in model:
        raise ValueError("smoke output is missing the TorchScript teacher marker")

    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("config.model must be a mapping")
    if str(model_config.get("feature_backend")) != "qwen":
        raise ValueError("smoke config must use the real Qwen feature backend")
    if str(model_config.get("prior_backend")) != "torchscript":
        raise ValueError("smoke config must use the TorchScript prior backend")
    expected_teacher_sha = str(model_config.get("prior_torchscript_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", expected_teacher_sha) is None:
        raise ValueError("smoke config teacher SHA-256 is missing or malformed")
    qwen_config = config.get("qwen")
    if not isinstance(qwen_config, Mapping) or not str(qwen_config.get("model_id", "")):
        raise ValueError("smoke config must identify the real Qwen model")
    saved_config = output.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("smoke output does not preserve its effective config")
    saved_model_config = saved_config.get("model")
    saved_qwen_config = saved_config.get("qwen")
    if not isinstance(saved_model_config, Mapping) or dict(saved_model_config) != dict(
        model_config
    ):
        raise ValueError("smoke output model config differs from the invoked config")
    if not isinstance(saved_qwen_config, Mapping) or dict(saved_qwen_config) != dict(
        qwen_config
    ):
        raise ValueError("smoke output Qwen config differs from the invoked config")
    saved_train_config = saved_config.get("train")
    if not isinstance(saved_train_config, Mapping):
        raise ValueError("smoke output has no effective train config")
    expected_train_values = {
        "stage": "unified",
        "epochs": seed_completed + 1,
        "max_train_steps": 1,
        "max_val_batches": 1,
        "preview_every": 0,
    }
    for name, expected in expected_train_values.items():
        if saved_train_config.get(name) != expected:
            raise ValueError(
                "smoke output train config does not prove the bounded one-step run; "
                f"{name} expected={expected!r}, got={saved_train_config.get(name)!r}"
            )
    identity = output.get("teacher_prior_identity")
    if not isinstance(identity, Mapping) or identity.get("sha256") != expected_teacher_sha:
        raise ValueError("smoke teacher identity does not match the configured SHA-256")
    deployment = output.get("deployment_contract")
    if not isinstance(deployment, Mapping) or deployment.get(
        "teacher_prior_identity"
    ) != identity:
        raise ValueError("smoke deployment contract does not bind the teacher identity")

    residual = output.get("residual_application")
    if not isinstance(residual, Mapping):
        raise ValueError("smoke output has no residual application metadata")
    if int(residual.get("origin_epoch", -1)) != seed_completed:
        raise ValueError("smoke residual origin must equal the seed completed epoch")
    if float(residual.get("scale", float("nan"))) != 0.0:
        raise ValueError("the first teacher smoke epoch must keep residual scale at zero")

    metrics = output.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("smoke output has no validation metrics")
    selected_metrics = {name: _finite_metric(metrics, name) for name in REQUIRED_METRICS}
    if selected_metrics["residual_application_scale"] != 0.0:
        raise ValueError("smoke validation did not evaluate the alpha-zero anchor")

    optimizer = output.get("optimizer")
    state = optimizer.get("state") if isinstance(optimizer, Mapping) else None
    if not isinstance(state, Mapping) or not state:
        raise ValueError("smoke optimizer.state is empty; no real optimizer step is proven")
    steps: list[float] = []
    for item in state.values():
        if not isinstance(item, Mapping) or "step" not in item:
            raise ValueError("smoke Adam state is missing its step counter")
        raw_step = item["step"]
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1:
                raise ValueError("smoke Adam step tensor must be scalar")
            step = float(raw_step.item())
        elif isinstance(raw_step, (int, float)) and not isinstance(raw_step, bool):
            step = float(raw_step)
        else:
            raise ValueError("smoke Adam step must be numeric")
        if step != 1.0:
            raise ValueError(f"smoke Adam step must be exactly 1, got {step}")
        steps.append(step)

    return {
        "schema_version": 1,
        "kind": "v33_real_teacher_qwen_ddp_smoke",
        "scope": "functional_substage_only",
        "passed": True,
        "invoked_world_size": invoked_world_size,
        "verified_rank_isolation": [list(row) for row in sorted(isolation_rows)],
        "seed_completed_epochs": seed_completed,
        "output_completed_epochs": int(output_epoch) + 1,
        "optimizer_state_count": len(state),
        "optimizer_step_min": min(steps),
        "optimizer_step_max": max(steps),
        "teacher_sha256": expected_teacher_sha,
        "validation_metrics": selected_metrics,
        "limitations": [
            "LAMA inference is not executed by the training smoke",
            "max_val_batches=1 is applied independently on every DDP rank",
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
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-checkpoint", type=Path, required=True)
    parser.add_argument("--smoke-output-root", type=Path, required=True)
    parser.add_argument("--expected-seed-completed-epochs", type=int, required=True)
    parser.add_argument("--invoked-world-size", type=int, required=True)
    parser.add_argument("--functional-log", type=Path, required=True)
    parser.add_argument("--seed-source", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        name: args.smoke_output_root / "unified" / f"{name}.pt"
        for name in ("anchor", "best", "latest")
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"smoke {name} checkpoint is missing: {path}")
    seed = _load(args.seed_checkpoint)
    outputs = {name: _load(path) for name, path in paths.items()}
    if not args.functional_log.is_file():
        raise FileNotFoundError(f"functional log is missing: {args.functional_log}")
    functional_log_text = args.functional_log.read_text(encoding="utf-8", errors="replace")
    isolation_rows = parse_functional_rank_isolation(
        functional_log_text, expected_world_size=args.invoked_world_size
    )
    inspect_checkpoint_payload(seed, expect_stage="unified", require_optimizer=True)
    for name, path in paths.items():
        inspect_checkpoint_payload(
            outputs[name], expect_stage="unified", require_optimizer=True
        )
    effective_config = load_config(args.config)
    semantic_reports = {
        name: verify_smoke_payloads(
            seed,
            output,
            effective_config,
            expected_seed_completed_epochs=args.expected_seed_completed_epochs,
            invoked_world_size=args.invoked_world_size,
            observed_rank_isolation=isolation_rows,
        )
        for name, output in outputs.items()
    }
    report = semantic_reports["latest"]
    report["verified_checkpoints"] = {
        name: {
            "output_completed_epochs": item["output_completed_epochs"],
            "optimizer_state_count": item["optimizer_state_count"],
            "optimizer_step_min": item["optimizer_step_min"],
            "optimizer_step_max": item["optimizer_step_max"],
            "teacher_sha256": item["teacher_sha256"],
        }
        for name, item in semantic_reports.items()
    }
    report["artifacts"] = {
        "config": {
            "path": str(args.config.resolve()),
            "sha256": _sha256(args.config),
            "size_bytes": args.config.stat().st_size,
        },
        "seed": {
            "source_path": str(
                (args.seed_source or args.seed_checkpoint).resolve()
            ),
            "frozen_path": str(args.seed_checkpoint.resolve()),
            "sha256": _sha256(args.seed_checkpoint),
            "size_bytes": args.seed_checkpoint.stat().st_size,
            "frozen_copy_is_ephemeral": True,
        },
        "functional_log": {
            "path": str(args.functional_log.resolve()),
            "sha256": _sha256(args.functional_log),
            "size_bytes": args.functional_log.stat().st_size,
        },
        **{
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "ephemeral": True,
            }
            for name, path in paths.items()
        },
    }
    _atomic_write(args.report, report)
    print(
        "D2R_V33_FUNCTIONAL_SMOKE_OK "
        f"world_size={args.invoked_world_size} optimizer_step=1 "
        f"report={args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
