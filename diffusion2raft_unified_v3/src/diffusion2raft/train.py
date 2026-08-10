"""Train Stage A, the legacy RGB-guide baseline, or the unified model."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import load_config
from .data import DocumentFlowDataset
from .losses import RectificationLoss
from .models import build_rectifier


TRAIN_LOG_KEYS = (
    "total",
    "flow_per_iteration",
    "epe",
    "prior_epe",
    "epe_gain",
    "final_win_rate",
    "fold_rate",
    "residual_p95",
    "residual_epe",
    "residual_target_valid_rate",
    "feature_confidence",
    "gate_target",
    "qwen_match_epe",
    "qwen_match_acc1",
    "qwen_advantage",
    "qwen_win_rate",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distributed_device(config: dict[str, Any]) -> tuple[torch.device, int, int, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    distributed = local_rank >= 0
    requested = str(config.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if distributed:
        backend = "nccl" if requested.startswith("cuda") else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if requested.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(requested)
        return device, rank, world_size, True
    return torch.device(requested), 0, 1, False


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _make_loader(
    config: dict[str, Any],
    manifest: str,
    *,
    stage: str,
    training: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    data_config = config["data"]
    dataset = DocumentFlowDataset(
        manifest,
        tuple(data_config["work_size"]),
        augment_guide=training and stage == "joint",
        guide_artifact_prob=float(data_config.get("guide_artifact_prob", 0.65)),
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            drop_last=training,
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 1)),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=True,
        drop_last=training,
        persistent_workers=int(data_config.get("num_workers", 4)) > 0,
    )
    return loader, sampler


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _global_running_means(
    running: dict[str, float],
    steps: int,
    *,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    """Aggregate training logs across ranks instead of reporting rank 0 only."""

    values = torch.tensor(
        [*[running.get(key, 0.0) for key in TRAIN_LOG_KEYS], float(steps)],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    denominator = max(float(values[-1].item()), 1.0)
    return {
        key: float(values[index].item()) / denominator
        for index, key in enumerate(TRAIN_LOG_KEYS)
    }


def _scheduled_correlation_temperature(
    config: dict[str, Any], epoch: int
) -> float:
    """Log-linearly ramp correlation contrast using one-based epoch numbers."""

    model_config = config["model"]
    target = float(model_config.get("correlation_temperature", 0.10))
    start = float(model_config.get("correlation_temperature_start", target))
    if target <= 0 or start <= 0:
        raise ValueError("correlation temperatures must be positive")
    start_epoch = int(model_config.get("correlation_ramp_start_epoch", 1))
    ramp_epochs = int(model_config.get("correlation_ramp_epochs", 1))
    display_epoch = int(epoch) + 1
    if display_epoch <= start_epoch:
        return start
    if ramp_epochs <= 1 or display_epoch >= start_epoch + ramp_epochs - 1:
        return target
    progress = (display_epoch - start_epoch) / max(ramp_epochs - 1, 1)
    return math.exp((1.0 - progress) * math.log(start) + progress * math.log(target))


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any], stage: str) -> AdamW:
    train_config = config["train"]
    raw = _raw_model(model)
    prior_parameters = [parameter for parameter in raw.prior.parameters() if parameter.requires_grad]
    groups: list[dict[str, Any]] = [
        {
            "params": prior_parameters,
            "lr": float(
                train_config.get("lr_prior_joint", 2e-5)
                if stage == "unified"
                else train_config.get("lr_prior", 2e-4)
            ),
            "name": "prior",
        }
    ]
    prior_ids = {id(parameter) for parameter in prior_parameters}
    if stage == "joint" and getattr(raw, "raft", None) is not None:
        groups.append(
            {
                "params": [
                    parameter
                    for parameter in raw.raft.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(train_config.get("lr_raft", 2e-5)),
                "name": "raft",
            }
        )
    elif stage == "unified":
        joint_parameters = [
            parameter
            for parameter in raw.parameters()
            if parameter.requires_grad and id(parameter) not in prior_ids
        ]
        groups.append(
            {
                "params": joint_parameters,
                "lr": float(train_config.get("lr_unified", 1e-4)),
                "name": "unified_heads",
            }
        )
    groups = [group for group in groups if group["params"]]
    if not groups:
        raise RuntimeError("optimizer has no trainable parameters")
    return AdamW(groups, weight_decay=float(train_config.get("weight_decay", 1e-5)))


def _normalize_checkpoint_keys(state: dict[str, Tensor]) -> dict[str, Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    target_stage: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    payload = torch.load(path, map_location="cpu")
    state = _normalize_checkpoint_keys(payload.get("model", payload))
    source_stage = str(payload.get("stage", target_stage))
    missing, unexpected = model.load_state_dict(state, strict=False)
    same_stage = source_stage == target_stage
    if same_stage:
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch; missing={missing[:12]}, unexpected={unexpected[:12]}"
            )
    elif source_stage == "prior" and target_stage in {"joint", "unified"}:
        # Stage-A checkpoints contain only the shared `prior.*` branch.  Every
        # missing key must belong to the newly constructed joint backend.
        bad_missing = [key for key in missing if key.startswith("prior.")]
        if unexpected or bad_missing:
            raise RuntimeError(
                "Stage-A migration failed; "
                f"missing_prior={bad_missing[:12]}, unexpected={unexpected[:12]}"
            )
    else:
        raise RuntimeError(
            f"unsupported checkpoint migration {source_stage!r} -> {target_stage!r}"
        )
    if same_stage and optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("epoch", -1)) + 1 if same_stage else 0


def _forward(
    model: torch.nn.Module,
    batch: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    guide = batch["guide"] if stage == "joint" else None
    return model(batch["warped"], guide, stage=stage)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: RectificationLoss,
    device: torch.device,
    stage: str,
    *,
    distributed: bool,
) -> dict[str, float]:
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        if stage == "joint" and not bool(batch["guide_available"].all()):
            raise RuntimeError("legacy joint validation sample is missing a Qwen RGB guide")
        losses = criterion(_forward(model, batch, stage), batch)
        for key, value in losses.items():
            if value.ndim == 0:
                totals[key] += float(value)
        count += 1
    keys = sorted(totals)
    values = torch.tensor(
        [*[totals[key] for key in keys], float(count)],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    global_count = max(float(values[-1].item()), 1.0)
    return {key: float(values[index].item()) / global_count for index, key in enumerate(keys)}


def train(config: dict[str, Any]) -> Path:
    train_config = config["train"]
    stage = str(train_config.get("stage", "prior"))
    if stage not in {"prior", "joint", "unified"}:
        raise ValueError("train.stage must be 'prior', 'joint', or 'unified'")
    device, rank, world_size, distributed = _distributed_device(config)
    _seed_everything(int(config.get("seed", 42)) + rank)
    is_main = rank == 0

    model_config = dict(config["model"])
    model_config.setdefault(
        "guide_dropout_prob", float(config["data"].get("guide_dropout_prob", 0.10))
    )
    if stage == "unified" and str(model_config.get("feature_backend", "qwen")) == "qwen":
        batch_size = int(config["data"].get("batch_size", 1))
        if batch_size != 1:
            raise ValueError(
                "QwenImageEditPlus currently requires data.batch_size=1 per rank for unified training"
            )
    model = build_rectifier(
        model_config,
        dict(config.get("qwen", {})),
        stage=stage,
        device=device,
    ).to(device)
    criterion = RectificationLoss(config["loss"]).to(device)
    optimizer = _build_optimizer(model, config, stage)

    start_epoch = 0
    resume = train_config.get("resume")
    if resume:
        start_epoch = _load_checkpoint(
            model,
            resume,
            target_stage=stage,
            optimizer=optimizer,
        )
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    train_loader, train_sampler = _make_loader(
        config,
        config["data"]["train_manifest"],
        stage=stage,
        training=True,
        rank=rank,
        world_size=world_size,
    )
    val_manifest = config["data"].get("val_manifest")
    if val_manifest and Path(val_manifest).exists():
        val_loader, _ = _make_loader(
            config,
            val_manifest,
            stage=stage,
            training=False,
            rank=rank,
            world_size=world_size,
        )
    else:
        val_loader = None

    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    amp_dtype_name = str(train_config.get("amp_dtype", "bfloat16")).lower()
    if amp_dtype_name in {"bfloat16", "bf16"}:
        amp_dtype = torch.bfloat16
    elif amp_dtype_name in {"float16", "fp16"}:
        amp_dtype = torch.float16
    else:
        raise ValueError("train.amp_dtype must be bfloat16 or float16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    output_dir = Path(train_config.get("output_dir", "runs/d2r")) / stage
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    epochs = int(train_config.get("epochs", 20))
    log_every = int(train_config.get("log_every", 20))
    grad_clip = float(train_config.get("grad_clip", 1.0))
    freeze_prior_epochs = int(train_config.get("freeze_prior_epochs", 0)) if stage == "unified" else 0

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        raw = _raw_model(model)
        correlation_temperature = float(
            getattr(raw, "correlation_temperature", float("nan"))
        )
        if stage == "unified":
            # Do not toggle requires_grad after DDP construction.  A zero LR
            # provides the same stabilization while keeping reducer topology
            # fixed across the later unfreeze boundary.
            configured_prior_lr = float(train_config.get("lr_prior_joint", 2e-5))
            correlation_temperature = _scheduled_correlation_temperature(config, epoch)
            raw.set_correlation_temperature(correlation_temperature)
            for group in optimizer.param_groups:
                if group.get("name") == "prior":
                    group["lr"] = 0.0 if epoch < freeze_prior_epochs else configured_prior_lr
                elif group.get("name") == "unified_heads":
                    # A resumed optimizer stores the old LR.  Re-apply the
                    # current experiment config so v2 -> v3 continuation uses
                    # the intended schedule without discarding Adam moments.
                    group["lr"] = float(train_config.get("lr_unified", 1e-4))
        running: defaultdict[str, float] = defaultdict(float)
        running_steps = 0
        for step, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            if stage == "joint" and not bool(batch["guide_available"].all()):
                raise RuntimeError(
                    "legacy joint training requires a guide path; unified training does not"
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = _forward(model, batch, stage)
                losses = criterion(outputs, batch)
                loss = losses["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for key, value in losses.items():
                if value.ndim == 0:
                    running[key] += float(value.detach())
            running_steps += 1
            if (step + 1) % log_every == 0:
                means = _global_running_means(
                    running,
                    running_steps,
                    device=device,
                    distributed=distributed,
                )
                if is_main:
                    summary = (
                        f"total={means['total']:.4f} "
                        f"seq={means['flow_per_iteration']:.3f} "
                        f"epe={means['epe']:.4f} "
                        f"prior={means['prior_epe']:.4f} "
                        f"gain={means['epe_gain']:.4f} "
                        f"win={means['final_win_rate']:.3f} "
                        f"fold={means['fold_rate']:.6f} "
                        f"r95={means['residual_p95']:.2f} "
                        f"r_epe={means['residual_epe']:.2f} "
                        f"r_valid={means['residual_target_valid_rate']:.3f} "
                        f"gate={means['feature_confidence']:.3f}/"
                        f"{means['gate_target']:.3f} "
                        f"q_epe={means['qwen_match_epe']:.2f} "
                        f"q_acc1={means['qwen_match_acc1']:.3f} "
                        f"q_adv={means['qwen_advantage']:.2f} "
                        f"q_win={means['qwen_win_rate']:.3f} "
                        f"corr_t={correlation_temperature:.3f}"
                    )
                    print(
                        f"epoch={epoch + 1}/{epochs} step={step + 1} {summary}",
                        flush=True,
                    )
                running.clear()
                running_steps = 0

        metrics = (
            _evaluate(
                model,
                val_loader,
                criterion,
                device,
                stage,
                distributed=distributed,
            )
            if val_loader
            else {}
        )
        if is_main and metrics:
            print(
                f"validation epoch={epoch + 1} "
                f"epe={metrics.get('epe', float('nan')):.4f} "
                f"seq={metrics.get('flow_per_iteration', float('nan')):.3f} "
                f"prior_epe={metrics.get('prior_epe', float('nan')):.4f} "
                f"gain={metrics.get('epe_gain', float('nan')):.4f} "
                f"win={metrics.get('final_win_rate', float('nan')):.3f} "
                f"fold_rate={metrics.get('fold_rate', float('nan')):.6f} "
                f"residual_p95={metrics.get('residual_p95', float('nan')):.2f} "
                f"residual_epe={metrics.get('residual_epe', float('nan')):.2f} "
                f"r_valid={metrics.get('residual_target_valid_rate', float('nan')):.3f} "
                f"gate={metrics.get('feature_confidence', float('nan')):.3f}/"
                f"{metrics.get('gate_target', float('nan')):.3f} "
                f"qwen_epe={metrics.get('qwen_match_epe', float('nan')):.2f} "
                f"qwen_acc1={metrics.get('qwen_match_acc1', float('nan')):.3f} "
                f"qwen_adv={metrics.get('qwen_advantage', float('nan')):.2f} "
                f"qwen_win={metrics.get('qwen_win_rate', float('nan')):.3f} "
                f"corr_t={getattr(raw, 'correlation_temperature', float('nan')):.3f}",
                flush=True,
            )

        if is_main:
            checkpoint = {
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "stage": stage,
                "architecture": type(raw).__name__,
                "training_revision": "v3_supervised_matching",
                "config": config,
                "metrics": metrics,
            }
            latest_path = output_dir / "latest.pt"
            torch.save(checkpoint, latest_path)
            if (epoch + 1) % int(train_config.get("save_every", 1)) == 0:
                torch.save(checkpoint, output_dir / f"epoch_{epoch + 1:04d}.pt")
        if distributed:
            dist.barrier()

    result = output_dir / "latest.pt"
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--stage", choices=("prior", "joint", "unified"))
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage:
        config["train"]["stage"] = args.stage
    if args.resume:
        config["train"]["resume"] = args.resume
    if args.device:
        config["device"] = args.device
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    result = train(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"saved checkpoint: {result}")


if __name__ == "__main__":
    main()
