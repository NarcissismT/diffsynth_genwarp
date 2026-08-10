"""Small, explicit Stage-1 checkpoint contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT = "cp_docflow.det_coarse.v1"
FULL_CHECKPOINT_FORMAT = "docgrid_flow.full.v2"
COORDINATE_CONTRACT = "absolute_backward_map_xy_pixel_align_corners_false_v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256() -> str:
    """Hash every package source file together with its relative path."""

    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source_path in sorted(package_root.rglob("*.py")):
        relative = source_path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_payload(
    model: nn.Module,
    *,
    model_config: dict[str, Any],
    input_work_size: tuple[int, int],
    output_work_size: tuple[int, int],
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict[str, float] | None = None,
    data_contract: dict[str, Any] | None = None,
    config_sha256: str | None = None,
    best_epe: float | None = None,
    parent_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "coordinate_contract": COORDINATE_CONTRACT,
        "implementation_sha256": implementation_sha256(),
        "model_state": model.state_dict(),
        "model_config": dict(model_config),
        "input_work_size": tuple(int(v) for v in input_work_size),
        "output_work_size": tuple(int(v) for v in output_work_size),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
    }
    if data_contract is not None:
        payload["data_contract"] = dict(data_contract)
    if config_sha256 is not None:
        payload["config_sha256"] = str(config_sha256)
    if best_epe is not None:
        payload["best_epe"] = float(best_epe)
    if parent_checkpoint is not None:
        payload["parent_checkpoint"] = dict(parent_checkpoint)
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    return payload


def full_checkpoint_payload(
    model: nn.Module,
    *,
    model_config: dict[str, Any],
    input_work_size: tuple[int, int],
    output_work_size: tuple[int, int],
    epoch: int,
    training_stage: str,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict[str, float] | None = None,
    data_contract: dict[str, Any] | None = None,
    config_sha256: str | None = None,
    best_epe: float | None = None,
    parent_checkpoint: dict[str, Any] | None = None,
    gate_receipts: dict[str, Any] | None = None,
    training_seed: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": FULL_CHECKPOINT_FORMAT,
        "architecture_name": "DocGrid-Flow",
        "architecture_version": "2.0",
        "plan_contract": "Diffusion2RAFT_Plan_and_Goals_newest",
        "flow_state_contract": "confidence_gated_residual_backward_map",
        "coordinate_contract": COORDINATE_CONTRACT,
        "implementation_sha256": implementation_sha256(),
        "model_state": model.state_dict(),
        "model_config": dict(model_config),
        "input_work_size": tuple(int(v) for v in input_work_size),
        "output_work_size": tuple(int(v) for v in output_work_size),
        "epoch": int(epoch),
        "training_stage": str(training_stage),
        "metrics": dict(metrics or {}),
        "qwen_weights_embedded": False,
        "qwen_vae_decoder_used": False,
        "final_decoder": "WARR_then_convex_map_upsampler",
        "final_rgb_renderer": "single_grid_sample_from_original_warped_image",
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if data_contract is not None:
        payload["data_contract"] = dict(data_contract)
    if config_sha256 is not None:
        payload["config_sha256"] = str(config_sha256)
    if best_epe is not None:
        payload["best_epe"] = float(best_epe)
    if parent_checkpoint is not None:
        payload["parent_checkpoint"] = dict(parent_checkpoint)
    if gate_receipts is not None:
        payload["gate_receipts"] = dict(gate_receipts)
    if training_seed is not None:
        payload["training_seed"] = int(training_seed)
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a mapping: {checkpoint_path}")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {payload.get('format')!r}; "
            f"expected {CHECKPOINT_FORMAT!r}"
        )
    if payload.get("coordinate_contract") != COORDINATE_CONTRACT:
        raise ValueError("checkpoint coordinate contract does not match DocGrid-Flow v2")
    return payload


def load_full_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a mapping: {checkpoint_path}")
    if payload.get("format") != FULL_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {payload.get('format')!r}; "
            f"expected {FULL_CHECKPOINT_FORMAT!r}"
        )
    if payload.get("coordinate_contract") != COORDINATE_CONTRACT:
        raise ValueError("checkpoint coordinate contract does not match DocGrid-Flow v2")
    if payload.get("qwen_weights_embedded") is not False:
        raise ValueError("full checkpoint must not embed the frozen Qwen backbone")
    if payload.get("qwen_vae_decoder_used") is not False:
        raise ValueError("full checkpoint violates the no-Qwen-VAE-decoder contract")
    return payload
