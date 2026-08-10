"""Two-stage trainer for the diffusion-guided document rectifier."""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from .config import load_config
from .data import DocumentFlowDataset
from .losses import RectificationLoss
from .metrics import invalid_sampling_ratio, line_bending
from .models import build_guided_rectifier


@dataclass
class DistContext:
    """Holds the distributed topology for the current process."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _init_distributed() -> DistContext:
    """Initialize DDP from ``torchrun`` env vars, or return a single-process ctx.

    Launch multi-GPU with e.g. ``torchrun --nproc_per_node=8 -m diffusion2raft.train``.
    Without torchrun the WORLD_SIZE var is absent and this stays single-process,
    so the exact same code path serves both cases.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistContext(enabled=False, rank=0, local_rank=0, world_size=1)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable but WORLD_SIZE > 1")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def _cleanup_distributed(ctx: DistContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _make_loader(
    config: dict[str, Any],
    manifest: str,
    *,
    training: bool,
    ctx: DistContext | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    data_config = config["data"]
    dataset = DocumentFlowDataset(
        manifest,
        tuple(data_config["work_size"]),
        augment_guide=training,
        guide_artifact_prob=float(data_config.get("guide_artifact_prob", 0.65)),
    )
    sampler: DistributedSampler | None = None
    # Only the training loader is sharded across ranks. Validation runs on rank 0
    # over the whole set so the reported metrics match the single-GPU numbers.
    if training and ctx is not None and ctx.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            drop_last=True,
        )
    num_workers = int(data_config.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 2)),
        shuffle=(training and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any], stage: str) -> AdamW:
    train_config = config["train"]
    groups: list[dict[str, Any]] = [
        {
            "params": [parameter for parameter in model.prior.parameters() if parameter.requires_grad],
            "lr": float(train_config.get("lr_prior", 2e-4)),
        }
    ]
    if stage == "joint" and model.raft is not None:
        groups.append(
            {
                "params": [parameter for parameter in model.raft.parameters() if parameter.requires_grad],
                "lr": float(train_config.get("lr_raft", 2e-5)),
            }
        )
    return AdamW(groups, weight_decay=float(train_config.get("weight_decay", 1e-5)))


def _load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    same_stage: bool = False,
) -> int:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = all(key.startswith("raft.") for key in missing)
    if unexpected or (missing and not allowed_missing):
        raise RuntimeError(
            f"checkpoint mismatch; missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
    if same_stage and optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("epoch", -1)) + 1 if same_stage else 0


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: RectificationLoss,
    device: torch.device,
    stage: str,
) -> dict[str, float]:
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        if stage == "joint" and not bool(batch["guide_available"].all()):
            raise RuntimeError("joint-stage validation sample is missing a precomputed Qwen guide")
        outputs = model(batch["warped"], batch["guide"], stage=stage)
        losses = criterion(outputs, batch)
        for key, value in losses.items():
            if value.ndim == 0:
                totals[key] += float(value)
        final_flow = outputs["final_flow"].float()
        valid = batch["valid"].bool()
        totals["invalid_ratio"] += invalid_sampling_ratio(
            final_flow, tuple(final_flow.shape[-2:])
        )
        totals["line_bending"] += line_bending(final_flow, valid)
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(config: dict[str, Any], ctx: DistContext | None = None) -> Path:
    ctx = ctx or DistContext(enabled=False, rank=0, local_rank=0, world_size=1)
    # Offset the seed per rank so augmentation/dropout differ across replicas.
    seed = int(config.get("seed", 42)) + ctx.rank
    _seed_everything(seed)
    train_config = config["train"]
    stage = str(train_config.get("stage", "prior"))
    if stage not in {"prior", "joint"}:
        raise ValueError("train.stage must be 'prior' or 'joint'")

    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested_device.startswith("cuda"):
        device = torch.device(f"cuda:{ctx.local_rank}")
    else:
        device = torch.device(requested_device)

    # Keep guide-dropout with model settings so it is checkpointed as an
    # architectural/training contract rather than a dataset-side coordinate edit.
    model_config = dict(config["model"])
    model_config.setdefault(
        "guide_dropout_prob", float(config["data"].get("guide_dropout_prob", 0.10))
    )
    model = build_guided_rectifier(model_config, stage=stage).to(device)
    criterion = RectificationLoss(config["loss"]).to(device)
    # Build the optimizer on the raw parameters before DDP wrapping; DDP shares
    # the same Parameter objects, so gradients still reach these groups.
    optimizer = _build_optimizer(model, config, stage)

    start_epoch = 0
    resume = train_config.get("resume")
    if resume:
        payload = torch.load(resume, map_location="cpu")
        same_stage = str(payload.get("stage", stage)) == stage
        start_epoch = _load_checkpoint(
            model,
            resume,
            optimizer=optimizer,
            same_stage=same_stage,
        )

    if ctx.enabled:
        # Both stages use every parameter each step, so the default is off (the
        # extra autograd traversal is pure overhead). Enable via config only if a
        # future variant leaves parameters unused on some steps.
        model = DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if device.type == "cuda" else None,
            output_device=ctx.local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(
                train_config.get("ddp_find_unused_parameters", False)
            ),
        )

    train_loader, train_sampler = _make_loader(
        config, config["data"]["train_manifest"], training=True, ctx=ctx
    )
    val_manifest = config["data"].get("val_manifest")
    # Validation is rank-0 only over the full set -> metrics match single-GPU.
    val_loader = None
    if ctx.is_main and val_manifest and Path(val_manifest).exists():
        val_loader, _ = _make_loader(config, val_manifest, training=False, ctx=None)

    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(train_config.get("output_dir", "runs/d2r")) / stage
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(train_config.get("epochs", 20))
    log_every = int(train_config.get("log_every", 20))
    grad_clip = float(train_config.get("grad_clip", 1.0))

    for epoch in range(start_epoch, epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running: defaultdict[str, float] = defaultdict(float)
        for step, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            if stage == "joint" and not bool(batch["guide_available"].all()):
                raise RuntimeError(
                    "joint-stage training requires a guide path for every manifest record; "
                    "run d2r-generate-guides first"
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(batch["warped"], batch["guide"], stage=stage)
                losses = criterion(outputs, batch)
                loss = losses["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for key, value in losses.items():
                if value.ndim == 0:
                    running[key] += float(value.detach())
            if ctx.is_main and (step + 1) % log_every == 0:
                summary = " ".join(
                    f"{key}={value / log_every:.4f}"
                    for key, value in running.items()
                    if key in {"total", "epe", "fold_rate", "raw_reconstruction"}
                )
                print(f"epoch={epoch + 1}/{epochs} step={step + 1} {summary}", flush=True)
                running.clear()

        metrics: dict[str, float] = {}
        if val_loader is not None:
            metrics = _evaluate(_unwrap(model), val_loader, criterion, device, stage)
            print(
                f"validation epoch={epoch + 1} "
                f"epe={metrics.get('epe', float('nan')):.4f} "
                f"fold_rate={metrics.get('fold_rate', float('nan')):.6f} "
                f"invalid_ratio={metrics.get('invalid_ratio', float('nan')):.4f} "
                f"line_bending={metrics.get('line_bending', float('nan')):.4f}",
                flush=True,
            )

        if ctx.is_main:
            checkpoint = {
                "model": _unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "stage": stage,
                "config": config,
                "metrics": metrics,
            }
            latest_path = output_dir / "latest.pt"
            torch.save(checkpoint, latest_path)
            if (epoch + 1) % int(train_config.get("save_every", 1)) == 0:
                torch.save(checkpoint, output_dir / f"epoch_{epoch + 1:04d}.pt")
        if ctx.enabled:
            dist.barrier()
    return output_dir / "latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--stage", choices=("prior", "joint"))
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
    ctx = _init_distributed()
    try:
        result = train(config, ctx)
    finally:
        _cleanup_distributed(ctx)
    if ctx.is_main:
        print(f"saved checkpoint: {result}")


if __name__ == "__main__":
    main()

