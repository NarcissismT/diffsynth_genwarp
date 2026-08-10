"""Train Stage A, the legacy RGB-guide baseline, or the unified model."""

from __future__ import annotations

import argparse
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
    # Our own checkpoints contain a config dict, so weights_only=False is
    # required; these are trusted local files written by this trainer.
    payload = torch.load(path, map_location="cpu", weights_only=False)
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
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
        if stage == "unified":
            # Do not toggle requires_grad after DDP construction.  A zero LR
            # provides the same stabilization while keeping reducer topology
            # fixed across the later unfreeze boundary.
            configured_prior_lr = float(train_config.get("lr_prior_joint", 2e-5))
            for group in optimizer.param_groups:
                if group.get("name") == "prior":
                    group["lr"] = 0.0 if epoch < freeze_prior_epochs else configured_prior_lr
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
                if is_main:
                    summary = " ".join(
                        f"{key}={value / max(running_steps, 1):.4f}"
                        for key, value in running.items()
                        if key
                        in {
                            "total",
                            "epe",
                            "prior_epe",
                            "fold_rate",
                            "residual_p95",
                            "feature_confidence",
                        }
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
                f"prior_epe={metrics.get('prior_epe', float('nan')):.4f} "
                f"fold_rate={metrics.get('fold_rate', float('nan')):.6f} "
                f"residual_p95={metrics.get('residual_p95', float('nan')):.2f} "
                f"confidence={metrics.get('feature_confidence', float('nan')):.3f}",
                flush=True,
            )

        if is_main:
            checkpoint = {
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "stage": stage,
                "architecture": type(raw).__name__,
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
