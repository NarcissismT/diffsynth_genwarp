"""Train staged DocGrid-Flow coordinate prediction from warped images."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from .checkpoint import (
    CHECKPOINT_FORMAT,
    FULL_CHECKPOINT_FORMAT,
    file_sha256,
    full_checkpoint_payload,
    load_checkpoint,
    load_full_checkpoint,
)
from .config import build_full_model, load_config, project_path
from .data import (
    DocumentMapDataset,
    assert_document_disjoint,
    dataset_payload_sha256,
)
from .losses import CPDocFlowLoss, FullLossWeights
from .metrics import cell_valid_mask, endpoint_error_map, jacobian_determinant
from .train_coarse import (
    _assert_provenance,
    _device,
    _identity_set_sha256,
    _parse_size,
    _seed_everything,
    _tensor_batch,
)
from .training_views import FullPageMixedViewDataset


_STAGES = {"coarse", "warr", "coord_fm", "qwen", "full_page"}
_STAGE_ALIASES = {"refiner": "warr", "joint": "full_page"}
_DEFAULT_PROFILE = {
    "coarse": "coarse",
    "warr": "warr",
    "coord_fm": "coord_fm",
    "qwen": "qwen",
    "full_page": "full_page",
}
_STAGE_PROFILES = {
    "coarse": {"coarse"},
    "warr": {"warr", "warr_joint"},
    "coord_fm": {"coord_fm", "coord_fm_joint"},
    "qwen": {"qwen"},
    "full_page": {"full_page"},
}


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    profile: str
    duration_fraction: float
    learning_rate: float


def canonical_stage_name(stage: str) -> str:
    return _STAGE_ALIASES.get(str(stage).lower(), str(stage).lower())


def _read_gate_receipts(
    config: dict[str, Any],
    stage: str,
    *,
    enforce: bool,
    qwen_enabled: bool = False,
) -> dict[str, Any]:
    stage = canonical_stage_name(stage)
    required = {
        "coarse": (),
        "warr": ("gate1",),
        "coord_fm": ("gate1", "gate2"),
        "qwen": ("gate1", "gate2", "gate3"),
        "full_page": ("gate1", "gate2", "gate3", "gate4")
        if qwen_enabled
        else ("gate1", "gate2", "gate3"),
    }[stage]
    configured = dict(config.get("train", {}).get("gate_receipts", {}))
    receipts: dict[str, Any] = {}
    for gate in required:
        raw_path = configured.get(gate)
        if raw_path is None:
            if enforce:
                raise ValueError(
                    f"training stage {stage!r} requires train.gate_receipts.{gate}"
                )
            receipts[gate] = {"bypassed_for_exploratory_run": True}
            continue
        path = project_path(config, raw_path)
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        expected_schema = f"docgrid_flow.{gate}.v2"
        if not isinstance(value, dict) or value.get("schema") != expected_schema:
            raise ValueError(f"invalid {gate} receipt schema in {path}")
        if value.get("passed") is not True:
            raise ValueError(f"{gate} receipt did not pass: {path}")
        if value.get("verified_gt_only") is not True:
            raise ValueError(f"{gate} receipt is not based on verified GT: {path}")
        receipts[gate] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "schema": expected_schema,
        }
    return receipts


def configure_training_profile(
    model: torch.nn.Module,
    stage: str,
    profile: str | None = None,
) -> None:
    """Set the runtime topology and exactly one trainable-module profile."""

    stage = canonical_stage_name(stage)
    if stage not in _STAGES:
        raise ValueError(f"train.stage must be one of {sorted(_STAGES)}")
    selected = _DEFAULT_PROFILE[stage] if profile is None else str(profile).lower()
    if selected not in _STAGE_PROFILES[stage]:
        raise ValueError(
            f"training profile {selected!r} is invalid for stage {stage!r}; "
            f"choose one of {sorted(_STAGE_PROFILES[stage])}"
        )
    model.set_execution_stage(stage)
    model.requires_grad_(False)
    # Qwen is a frozen feature source in every stage.  Only its adapter is
    # trainable; this also freezes the lite stand-in during smoke tests.
    if model.qwen_source is not None:
        model.qwen_source.requires_grad_(False)
    if selected == "coarse":
        model.coarse.requires_grad_(True)
        model.fusion.requires_grad_(True)
        model.convex_context.requires_grad_(True)
        model.convex_upsampler.requires_grad_(True)
    elif selected in {"warr", "warr_joint"}:
        model.hv.requires_grad_(True)
        model.refiner.requires_grad_(True)
        model.convex_context.requires_grad_(True)
        model.convex_upsampler.requires_grad_(True)
        if selected == "warr_joint":
            model.coarse.requires_grad_(True)
            model.fusion.requires_grad_(True)
    elif selected in {"coord_fm", "coord_fm_joint"}:
        model.velocity.requires_grad_(True)
        if selected == "coord_fm_joint":
            model.coarse.requires_grad_(True)
            model.fusion.requires_grad_(True)
            model.hv.requires_grad_(True)
            model.refiner.requires_grad_(True)
            model.convex_context.requires_grad_(True)
            model.convex_upsampler.requires_grad_(True)
    elif selected == "qwen":
        if model.qwen_source is None or model.qwen_adapter is None:
            raise ValueError("qwen stage requires model.qwen_backend=lite or qwen")
        model.qwen_adapter.requires_grad_(True)
        model.fusion.requires_grad_(True)
    else:
        model.requires_grad_(True)
        if model.qwen_source is not None:
            model.qwen_source.requires_grad_(False)


def configure_training_stage(model: torch.nn.Module, stage: str) -> None:
    """Compatibility entry point selecting the stage's default profile."""

    configure_training_profile(model, stage)


def resolve_training_phases(
    train_config: dict[str, Any], stage: str
) -> list[TrainingPhase]:
    """Validate the freeze/joint-unfreeze schedule for one training stage."""

    stage = canonical_stage_name(stage)
    base_lr = float(train_config.get("learning_rate", 1.0e-4))
    raw = train_config.get("phase_schedule")
    if raw is None:
        return [TrainingPhase(stage, _DEFAULT_PROFILE[stage], 1.0, base_lr)]
    if not isinstance(raw, list) or not raw:
        raise ValueError("train.phase_schedule must be a non-empty list")
    phases: list[TrainingPhase] = []
    names: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise ValueError(f"phase_schedule[{index}] must be a mapping")
        unknown = set(value) - {
            "name", "profile", "duration_fraction", "learning_rate"
        }
        if unknown:
            raise ValueError(
                f"unknown phase_schedule[{index}] keys: {sorted(unknown)}"
            )
        name = str(value.get("name", f"phase-{index}")).strip()
        profile = str(value.get("profile", _DEFAULT_PROFILE[stage])).lower()
        duration = float(value.get("duration_fraction", 0.0))
        learning_rate = float(value.get("learning_rate", base_lr))
        if not name or name in names:
            raise ValueError("training phase names must be non-empty and unique")
        if profile not in _STAGE_PROFILES[stage]:
            raise ValueError(
                f"profile {profile!r} is invalid for stage {stage!r}"
            )
        if not 0.0 < duration <= 1.0:
            raise ValueError("phase duration_fraction must be in (0,1]")
        if not 0.0 < learning_rate:
            raise ValueError("phase learning_rate must be positive")
        names.add(name)
        phases.append(TrainingPhase(name, profile, duration, learning_rate))
    if abs(sum(phase.duration_fraction for phase in phases) - 1.0) > 1.0e-6:
        raise ValueError("phase duration_fraction values must sum to 1.0")
    return phases


def training_phase_for_step(
    phases: list[TrainingPhase], step: int, total_steps: int
) -> TrainingPhase:
    if total_steps < 1 or step < 0:
        raise ValueError("total_steps must be positive and step non-negative")
    progress = min(float(step) / total_steps, 1.0 - 1.0e-12)
    boundary = 0.0
    for phase in phases:
        boundary += phase.duration_fraction
        if progress < boundary:
            return phase
    return phases[-1]


def _validate_frozen_data_contract(
    config: dict[str, Any],
    datasets: dict[str, DocumentMapDataset],
    *,
    enforce: bool,
) -> dict[str, Any] | None:
    raw_path = config.get("data", {}).get("frozen_contract")
    if raw_path is None:
        if enforce:
            raise ValueError(
                "formal training requires data.frozen_contract from Stage-0 audit"
            )
        return None
    path = project_path(config, str(raw_path))
    with path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if not isinstance(frozen, dict) or frozen.get("schema") != "docgrid_flow.frozen_data_contract.v2":
        raise ValueError(f"invalid frozen data contract: {path}")
    if frozen.get("coordinate_contract") != "absolute_backward_map_xy_pixel_align_corners_false_v1":
        raise ValueError("frozen data coordinate contract differs")
    if enforce and frozen.get("verified_gt_only") is not True:
        raise ValueError("formal training frozen contract is not verified-GT-only")
    splits = frozen.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("frozen data contract lacks split identities")
    identities: dict[str, Any] = {}
    for name, dataset in datasets.items():
        expected = splits.get(name)
        if not isinstance(expected, dict):
            raise ValueError(f"frozen data contract lacks split {name!r}")
        manifest_hash = file_sha256(dataset.manifest)
        payload_hash = dataset_payload_sha256(dataset.records)
        if expected.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"{name} manifest changed after Stage-0 freeze")
        if expected.get("dataset_payload_sha256") != payload_hash:
            raise ValueError(f"{name} payload changed after Stage-0 freeze")
        identities[name] = {
            "manifest_sha256": manifest_hash,
            "dataset_payload_sha256": payload_hash,
        }
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "verified_gt_only": frozen.get("verified_gt_only") is True,
        "splits": identities,
    }


def _load_parent(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    checkpoint_format = raw.get("format") if isinstance(raw, dict) else None
    if checkpoint_format == CHECKPOINT_FORMAT:
        payload = load_checkpoint(path)
        model.coarse.load_state_dict(payload["model_state"], strict=True)
    elif checkpoint_format == FULL_CHECKPOINT_FORMAT:
        payload = load_full_checkpoint(path)
        missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
        allowed_missing = all(
            name.startswith(("qwen_adapter.", "qwen_source.")) for name in missing
        )
        # Stage 1-3 checkpoints written before adapter isolation may contain an
        # untrained Qwen adapter.  It is safe to discard only that historical
        # state when the current deterministic stage deliberately omitted it.
        allowed_unexpected = all(
            name.startswith("qwen_adapter.") for name in unexpected
        )
        if not allowed_unexpected or not allowed_missing:
            raise ValueError(
                "parent full checkpoint is structurally incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )
    else:
        raise ValueError(f"unsupported parent checkpoint format in {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "format": checkpoint_format,
        "epoch": int(payload["epoch"]),
    }


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: CPDocFlowLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    final_errors: list[Tensor] = []
    coarse_errors: list[Tensor] = []
    wins = 0
    valid_pixels = 0
    eligible = 0
    damaged = 0
    folds = 0
    cells = 0
    loss_sum = 0.0
    samples = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        output = model(
            batch["warped_image"],
            output_size=batch["backward_map"].shape[-2:],
            target_canvas_size=batch.get("target_canvas_size"),
            target_window=batch.get("target_window"),
        )
        losses = criterion(output, batch)
        valid = batch["valid_mask"].bool()
        final_error = endpoint_error_map(output["backward_map"], batch["backward_map"].float())
        coarse_error = endpoint_error_map(
            output["coarse_backward_map"], batch["backward_map"].float()
        )
        final_errors.append(final_error[valid].detach().cpu())
        coarse_errors.append(coarse_error[valid].detach().cpu())
        wins += int(((final_error < coarse_error) & valid).sum().item())
        valid_pixels += int(valid.sum().item())
        reliable = valid & (coarse_error < 1.0)
        eligible += int(reliable.sum().item())
        damaged += int((((final_error - coarse_error) > 1.0) & reliable).sum().item())
        determinant = jacobian_determinant(output["backward_map"])
        cell_mask = cell_valid_mask(valid)
        folds += int(((determinant <= 0.0) & cell_mask).sum().item())
        cells += int(cell_mask.sum().item())
        count = int(batch["warped_image"].shape[0])
        loss_sum += float(losses["total"].item()) * count
        samples += count
    values = torch.cat(final_errors).float() if final_errors else torch.empty(0)
    coarse_values = torch.cat(coarse_errors).float() if coarse_errors else torch.empty(0)
    return {
        "loss": loss_sum / max(samples, 1),
        "epe": float(values.mean()) if values.numel() else float("nan"),
        "epe_p95": float(torch.quantile(values, 0.95)) if values.numel() else float("nan"),
        "coarse_epe": float(coarse_values.mean()) if coarse_values.numel() else float("nan"),
        "final_win_rate": wins / max(valid_pixels, 1),
        "high_confidence_damage_rate": damaged / max(eligible, 1),
        "fold_rate": folds / max(cells, 1),
        "samples": float(samples),
    }


def train(
    config: dict[str, Any],
    *,
    resume: str | None = None,
    output_dir_override: str | None = None,
    seed_override: int | None = None,
    parent_checkpoint_override: str | None = None,
) -> Path:
    config = copy.deepcopy(config)
    seed = int(config.get("seed", 1337) if seed_override is None else seed_override)
    config["seed"] = seed
    if parent_checkpoint_override is not None:
        config.setdefault("train", {})["parent_checkpoint"] = str(
            parent_checkpoint_override
        )
    _seed_everything(seed)
    data_config = dict(config.get("data", {}))
    train_config = dict(config.get("train", {}))
    model_config = dict(config.get("model", {}))
    stage = canonical_stage_name(str(train_config.get("stage", "full_page")))
    if stage not in _STAGES:
        raise ValueError(f"train.stage must be one of {sorted(_STAGES)}")
    gate_receipts = _read_gate_receipts(
        config,
        stage,
        enforce=bool(train_config.get("enforce_stage_gates", True)),
        qwen_enabled=str(model_config.get("qwen_backend", "none")).lower() == "qwen",
    )
    output_value = output_dir_override or train_config.get("output_dir", f"runs/{stage}")
    if seed_override is not None and output_dir_override is None:
        output_path_value = Path(str(output_value))
        seed_name = f"seed-{seed}"
        output_value = (
            output_path_value.with_name(seed_name)
            if output_path_value.name.startswith("seed-")
            else output_path_value / seed_name
        )
        config.setdefault("train", {})["output_dir"] = str(output_value)
        train_config["output_dir"] = str(output_value)
    output_dir = project_path(config, output_value)
    resume_path = None if resume is None else Path(resume).resolve()
    if resume_path is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to use non-empty output_dir={output_dir}")
    if resume_path is not None and resume_path.parent != output_dir.resolve():
        raise ValueError("resume checkpoint must be inside the configured output_dir")

    input_size = _parse_size(data_config["input_work_size"], "input_work_size")
    output_size = _parse_size(data_config["output_work_size"], "output_work_size")
    train_manifest = project_path(config, data_config["train_manifest"])
    val_manifest = project_path(config, data_config["val_manifest"])
    base_train_dataset = DocumentMapDataset(
        train_manifest, input_work_size=input_size, output_work_size=output_size
    )
    val_dataset = DocumentMapDataset(
        val_manifest, input_work_size=input_size, output_work_size=output_size
    )
    stage5_mix = dict(data_config.get("stage5_mix", {}))
    mix_enabled = stage == "full_page" and bool(stage5_mix.get("enabled", False))
    train_dataset: Any = (
        FullPageMixedViewDataset(base_train_dataset, stage5_mix, seed=seed)
        if mix_enabled
        else base_train_dataset
    )
    assert_document_disjoint(base_train_dataset.records, val_dataset.records)
    frozen_data_contract = _validate_frozen_data_contract(
        config,
        {"train": base_train_dataset, "val": val_dataset},
        enforce=bool(train_config.get("enforce_stage_gates", True)),
    )
    allowed_provenance = {
        str(value).strip().lower()
        for value in data_config.get("allowed_label_provenance", [])
    }
    _assert_provenance(base_train_dataset, allowed_provenance, "train")
    _assert_provenance(val_dataset, allowed_provenance, "val")
    data_contract = {
        "train_manifest": str(train_manifest.resolve()),
        "train_manifest_sha256": file_sha256(train_manifest),
        "val_manifest": str(val_manifest.resolve()),
        "val_manifest_sha256": file_sha256(val_manifest),
        "allowed_label_provenance": sorted(allowed_provenance),
        "train_document_count": len({item.document_id for item in base_train_dataset.records}),
        "train_document_ids_sha256": _identity_set_sha256(
            {item.document_id for item in base_train_dataset.records}
        ),
        "val_document_count": len({item.document_id for item in val_dataset.records}),
        "val_document_ids_sha256": _identity_set_sha256(
            {item.document_id for item in val_dataset.records}
        ),
        "document_disjoint_verified": True,
        "stage5_mix": stage5_mix if mix_enabled else None,
        "frozen_contract": frozen_data_contract,
    }
    device = _device(str(train_config.get("device", "auto")))
    model = build_full_model(model_config).to(device)
    parent_identity: dict[str, Any] | None = None
    parent_value = train_config.get("parent_checkpoint")
    if resume_path is None and parent_value:
        parent_path = project_path(config, str(parent_value))
        parent_identity = _load_parent(model, parent_path)
    phases = resolve_training_phases(train_config, stage)
    optimizer_parameter_ids: set[int] = set()
    trainable_parameters_by_phase: dict[str, int] = {}
    for phase in phases:
        configure_training_profile(model, stage, phase.profile)
        current = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not current:
            raise ValueError(f"training phase {phase.name!r} has no trainable parameters")
        optimizer_parameter_ids.update(id(parameter) for parameter in current)
        trainable_parameters_by_phase[phase.name] = sum(
            parameter.numel() for parameter in current
        )
    optimizer_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) in optimizer_parameter_ids
    ]
    configure_training_profile(model, stage, phases[0].profile)
    if not optimizer_parameters:
        raise ValueError(f"stage {stage!r} has no trainable parameters")
    criterion = CPDocFlowLoss(
        FullLossWeights.from_mapping(config.get("loss")),
        sequence_gamma=float(train_config.get("sequence_gamma", 0.8)),
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=phases[0].learning_rate,
        weight_decay=float(train_config.get("weight_decay", 1.0e-4)),
    )
    start_epoch, best_epe = 0, float("inf")
    if resume_path is not None:
        payload = load_full_checkpoint(resume_path, map_location=device)
        if payload["model_config"] != model_config or payload["training_stage"] != stage:
            raise ValueError("resume model config or training stage differs")
        if payload.get("data_contract") != data_contract:
            raise ValueError("resume data contract differs from current manifests")
        if payload.get("gate_receipts", {}) != gate_receipts:
            raise ValueError("resume gate receipts differ")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_epe = float(payload.get("best_epe", float("inf")))

    batch_size = int(train_config.get("batch_size", 1))
    if mix_enabled and batch_size != 1:
        raise ValueError(
            "Stage-5 mixed views have different tensor shapes and require batch_size=1"
        )
    workers = int(train_config.get("num_workers", 4))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_config.get("eval_batch_size", batch_size)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        key: value for key, value in config.items() if not str(key).startswith("_")
    }
    resolved_config_path = output_dir / "config.yaml"
    if resume_path is None:
        shutil.copy2(config["_config_path"], output_dir / "config_source.yaml")
        with resolved_config_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(resolved_config, handle, sort_keys=False, allow_unicode=True)
    else:
        if not resolved_config_path.is_file():
            raise ValueError("resume run lacks the resolved config.yaml")
        with resolved_config_path.open("r", encoding="utf-8") as handle:
            frozen_resolved_config = yaml.safe_load(handle)
        if frozen_resolved_config != resolved_config:
            raise ValueError("resume resolved config differs from frozen config.yaml")
    run_manifest = {
        "schema": "docgrid_flow.training_run.v2",
        "training_stage": stage,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "optimizer_parameters": sum(
            parameter.numel() for parameter in optimizer_parameters
        ),
        "trainable_parameters_by_phase": trainable_parameters_by_phase,
        "phase_schedule": [
            {
                "name": phase.name,
                "profile": phase.profile,
                "duration_fraction": phase.duration_fraction,
                "learning_rate": phase.learning_rate,
            }
            for phase in phases
        ],
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "resolved_config_sha256": file_sha256(resolved_config_path),
        "source_config": str(Path(config["_config_path"]).resolve()),
        "source_config_sha256": file_sha256(config["_config_path"]),
        "parent_checkpoint": parent_identity,
        "gate_receipts": gate_receipts,
        "data_contract": data_contract,
    }
    run_manifest_path = output_dir / "run_manifest.json"
    if resume_path is None:
        with run_manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(run_manifest, handle, indent=2, ensure_ascii=False)
    elif not run_manifest_path.is_file():
        raise ValueError("resume run lacks run_manifest.json")
    history: list[dict[str, Any]] = []
    metrics_path = output_dir / "metrics.json"
    if resume_path is not None and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
    epochs = int(train_config.get("epochs", 20))
    total_steps = max(epochs * len(train_loader), 1)
    global_step = start_epoch * len(train_loader)
    use_amp = bool(train_config.get("mixed_precision", True)) and device.type == "cuda"
    gradient_clip = float(train_config.get("gradient_clip", 1.0))
    for epoch in range(start_epoch, epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        running_loss, seen = 0.0, 0
        running_parts: dict[str, float] = {}
        view_counts: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        active_phase: TrainingPhase | None = None
        active_trainable: list[torch.nn.Parameter] = []
        for raw_batch in train_loader:
            requested_phase = training_phase_for_step(phases, global_step, total_steps)
            if active_phase != requested_phase:
                active_phase = requested_phase
                configure_training_profile(model, stage, active_phase.profile)
                active_trainable = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                for group in optimizer.param_groups:
                    group["lr"] = active_phase.learning_rate
            phase_counts[active_phase.name] = phase_counts.get(active_phase.name, 0) + 1
            batch = _tensor_batch(raw_batch, device)
            for view in raw_batch.get("training_view", ["unspecified"]):
                name = str(view)
                view_counts[name] = view_counts.get(name, 0) + 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                output = model(
                    batch["warped_image"],
                    output_size=batch["backward_map"].shape[-2:],
                    target_map=batch["backward_map"],
                    valid_mask=batch["valid_mask"],
                    target_canvas_size=batch.get("target_canvas_size"),
                    target_window=batch.get("target_window"),
                )
                losses = criterion(output, batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(active_trainable, gradient_clip)
            optimizer.step()
            global_step += 1
            count = int(batch["warped_image"].shape[0])
            running_loss += float(losses["total"].detach().item()) * count
            for name, value in losses.items():
                running_parts[name] = running_parts.get(name, 0.0) + float(
                    value.detach().item()
                ) * count
            seen += count
        metrics = evaluate_loader(model, val_loader, criterion, device)
        metrics.update(
            train_loss=running_loss / max(seen, 1),
            epoch=float(epoch),
            training_stage=stage,
            training_view_counts=view_counts,
            training_phase_counts=phase_counts,
            training_phase=(active_phase.name if active_phase is not None else "none"),
            training_profile=(active_phase.profile if active_phase is not None else "none"),
            global_step=float(global_step),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            **{
                f"train_{name}": value / max(seen, 1)
                for name, value in sorted(running_parts.items())
                if name != "total"
            },
        )
        history.append(metrics)
        is_best = metrics["epe"] < best_epe
        if is_best:
            best_epe = metrics["epe"]
        payload = full_checkpoint_payload(
            model,
            model_config=model_config,
            input_work_size=input_size,
            output_work_size=output_size,
            epoch=epoch,
            training_stage=stage,
            optimizer=optimizer,
            metrics=metrics,
            data_contract=data_contract,
            config_sha256=file_sha256(resolved_config_path),
            best_epe=best_epe,
            parent_checkpoint=parent_identity,
            gate_receipts=gate_receipts,
            training_seed=seed,
        )
        torch.save(payload, output_dir / f"epoch_{epoch:04d}.pt")
        torch.save(payload, output_dir / "latest.pt")
        if is_best:
            torch.save(payload, output_dir / "best.pt")
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, sort_keys=True), flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parent-checkpoint")
    args = parser.parse_args()
    train(
        load_config(args.config),
        resume=args.resume,
        output_dir_override=args.output_dir,
        seed_override=args.seed,
        parent_checkpoint_override=args.parent_checkpoint,
    )


if __name__ == "__main__":
    main()
