"""Train Stage A, the legacy RGB-guide baseline, or the unified model."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from .config import load_config
from .data import DocumentFlowDataset
from .deployment import (
    build_teacher_deployment_contract,
    validate_teacher_deployment_contract,
)
from .external_file import stable_external_file_identity
from .losses import RectificationLoss
from .models import build_learned_geometry_prior, build_rectifier
from .teacher_capacity_receipt import (
    decode_teacher_capacity_receipt_base64,
    strict_validate_teacher_capacity_receipt,
)


TRAIN_LOG_KEYS = (
    "total",
    "flow_per_iteration",
    "epe",
    "epe_p95",
    "edge_epe",
    "line_epe",
    "prior_line_epe",
    "line_epe_gain",
    "line_normal_mae",
    "line_straightness_error",
    "prior_line_straightness_error",
    "line_straightness_gain",
    "structure_coverage",
    "prior_epe",
    "epe_gain",
    "final_win_rate",
    "fold_rate",
    "jacobian_p01",
    "residual_p95",
    "applied_residual_p95",
    "residual_epe",
    "residual_target_valid_rate",
    "residual_application_scale",
    "feature_confidence",
    "gate_target",
    "qwen_match_epe",
    "qwen_match_acc1",
    "qwen_advantage",
    "qwen_win_rate",
)


TEACHER_WARMUP_REVISION = "v3_3_teacher_anchor_residual_warmup"
RESIDUAL_APPLICATION_VERSION = 1
TEACHER_CAPACITY_RECEIPT_ENV = "D2R_TEACHER_CAPACITY_RECEIPT_B64"


def _teacher_capacity_receipt_from_environment(
    config: dict[str, Any], *, stage: str
) -> dict[str, Any] | None:
    """Read the teacher audit receipt before any distributed/model setup."""

    encoded = os.environ.get(TEACHER_CAPACITY_RECEIPT_ENV)
    embedded_receipt_present = "capacity_evidence_receipt" in config
    model_config = config.get("model")
    teacher_training = (
        stage == "unified"
        and isinstance(model_config, dict)
        and str(model_config.get("prior_backend", "learned")).lower()
        == "torchscript"
    )
    if not teacher_training:
        if encoded is not None or embedded_receipt_present:
            raise RuntimeError(
                "capacity evidence receipts are only accepted from "
                f"{TEACHER_CAPACITY_RECEIPT_ENV} for unified "
                "TorchScript-teacher training"
            )
        return None
    if embedded_receipt_present:
        raise RuntimeError(
            "capacity_evidence_receipt must not be embedded in the training "
            f"config; provide only {TEACHER_CAPACITY_RECEIPT_ENV}"
        )
    if encoded is None:
        raise RuntimeError(
            f"unified TorchScript-teacher training requires "
            f"{TEACHER_CAPACITY_RECEIPT_ENV}"
        )
    return decode_teacher_capacity_receipt_base64(encoded)


def _typed_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like receipt values without bool/int coercion."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _validate_capacity_receipt_teacher(
    receipt: dict[str, Any], teacher_identity: dict[str, Any]
) -> None:
    """Bind the approved capacity audit to the teacher loaded by this rank."""

    approved = receipt["teacher"]
    current = {
        "sha256": teacher_identity.get("sha256"),
        "file_size": teacher_identity.get("file_size"),
        "input_size": teacher_identity.get("input_size"),
        "flow_size": teacher_identity.get("flow_size"),
        "blur_kernel": teacher_identity.get("blur_kernel"),
        "autocast_dtype": teacher_identity.get("autocast_dtype"),
        "requires_logical_cuda0": teacher_identity.get(
            "requires_logical_cuda0", False
        ),
    }
    if not _typed_equal(approved, current):
        differing = sorted(
            key
            for key in approved
            if not _typed_equal(approved.get(key), current.get(key))
        )
        raise RuntimeError(
            "teacher capacity receipt does not match the loaded TorchScript "
            f"teacher; differing_fields={differing}"
        )


def _tensor_to_pil(image: Tensor) -> Image.Image:
    array = (
        image.detach()
        .float()
        .clamp(0.0, 1.0)
        .cpu()
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _save_validation_preview(
    batch: dict[str, Any],
    outputs: dict[str, Any],
    rectified: Tensor,
    path: Path,
) -> None:
    """Save a fixed visual guardrail without an additional model forward."""

    panels = [
        ("warped", batch["warped"][0]),
        ("prior", outputs["prior_rectified"][0]),
        ("unified", rectified[0]),
        ("GT target", batch["target"][0]),
    ]
    images = [(label, _tensor_to_pil(tensor)) for label, tensor in panels]
    width, height = images[0][1].size
    label_height = 24
    canvas = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        x = index * width
        canvas.paste(image, (x, label_height))
        draw.text((x + 5, 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


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
) -> tuple[DataLoader, Sampler[int] | None]:
    data_config = config["data"]
    dataset = DocumentFlowDataset(
        manifest,
        tuple(data_config["work_size"]),
        augment_guide=training and stage == "joint",
        guide_artifact_prob=float(data_config.get("guide_artifact_prob", 0.65)),
        source_geometry_augment=(
            data_config.get("source_geometry_augment") if training else None
        ),
    )
    if world_size <= 1:
        sampler: Sampler[int] | None = None
    elif training:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    else:
        sampler = ExactDistributedEvalSampler(
            len(dataset), num_replicas=world_size, rank=rank
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


class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices without padding or duplication.

    ``DistributedSampler(drop_last=False)`` pads a 300-item validation set to
    304 items for eight ranks, silently counting the first four samples twice.
    Evaluation performs its collective only after all local batches finish,
    so ranks may safely own counts that differ by one.
    """

    def __init__(self, size: int, *, num_replicas: int, rank: int) -> None:
        self.size = int(size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.size < 0:
            raise ValueError("evaluation dataset size must be non-negative")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must be in [0,{self.num_replicas}), got {self.rank}"
            )

    def __iter__(self):
        return iter(range(self.rank, self.size, self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return (self.size - 1 - self.rank) // self.num_replicas + 1


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
    if not math.isfinite(target) or not math.isfinite(start) or target <= 0 or start <= 0:
        raise ValueError("correlation temperatures must be finite and positive")
    start_epoch = int(model_config.get("correlation_ramp_start_epoch", 1))
    ramp_epochs = int(model_config.get("correlation_ramp_epochs", 1))
    display_epoch = int(epoch) + 1
    if display_epoch <= start_epoch:
        return start
    if ramp_epochs <= 1 or display_epoch >= start_epoch + ramp_epochs - 1:
        return target
    progress = (display_epoch - start_epoch) / max(ramp_epochs - 1, 1)
    return math.exp((1.0 - progress) * math.log(start) + progress * math.log(target))


def _checkpoint_correlation_temperature(
    payload: dict[str, Any], *, required: bool = True
) -> float | None:
    """Return and cross-check the exact runtime temperature of a checkpoint.

    New checkpoints store the actual value directly.  Legacy checkpoints can
    be reconstructed exactly from their saved config and zero-based epoch.
    When both forms exist they must agree, preventing a renamed/copied config
    or a stale runtime attribute from silently changing refiner behavior.
    """

    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload must be a mapping")
    saved_present = "correlation_temperature" in payload
    saved: float | None = None
    if saved_present:
        value = payload.get("correlation_temperature")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError(
                "checkpoint correlation_temperature must be a real number"
            )
        saved = float(value)
        if not math.isfinite(saved) or saved <= 0.0:
            raise RuntimeError(
                "checkpoint correlation_temperature must be finite and positive"
            )

    config = payload.get("config")
    epoch = payload.get("epoch")
    reconstructable = (
        isinstance(config, dict)
        and not isinstance(epoch, bool)
        and isinstance(epoch, Integral)
        and int(epoch) >= 0
    )
    reconstructed: float | None = None
    if reconstructable:
        try:
            reconstructed = float(
                _scheduled_correlation_temperature(config, int(epoch))
            )
        except Exception as exc:
            raise RuntimeError(
                f"checkpoint has an invalid correlation-temperature schedule: {exc}"
            ) from exc
    elif required:
        raise RuntimeError(
            "unified checkpoint requires config and an integer epoch "
            "(non-negative) to bind correlation_temperature"
        )

    if saved is not None and reconstructed is not None and not math.isclose(
        saved, reconstructed, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "checkpoint runtime correlation_temperature disagrees with its "
            f"saved schedule: runtime={saved}, scheduled={reconstructed}"
        )
    if saved is not None:
        return saved
    if reconstructed is not None:
        return reconstructed
    if required:
        raise RuntimeError("checkpoint correlation_temperature is unavailable")
    return None


def _scheduled_residual_application_scale(
    epoch: int,
    *,
    origin_epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    max_scale: float,
) -> float:
    """Ramp applied residuals relative to the first teacher-backed epoch."""

    epoch = int(epoch)
    origin_epoch = int(origin_epoch)
    warmup_epochs = int(warmup_epochs)
    ramp_epochs = int(ramp_epochs)
    max_scale = float(max_scale)
    if origin_epoch < 0:
        raise ValueError("residual schedule origin_epoch must be non-negative")
    if warmup_epochs < 1:
        raise ValueError(
            "teacher residual warmup must hold alpha=0 for at least one epoch"
        )
    if ramp_epochs < 0:
        raise ValueError("residual ramp_epochs must be non-negative")
    if not math.isfinite(max_scale) or not 0.0 <= max_scale <= 1.0:
        raise ValueError("residual max_scale must be finite and in [0,1]")
    age = epoch - origin_epoch
    if age < warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return max_scale
    progress = min(max((age - warmup_epochs + 1) / ramp_epochs, 0.0), 1.0)
    return max_scale * progress


def _teacher_schedule_from_config(
    config: dict[str, Any], origin_epoch: int
) -> dict[str, Any]:
    train_config = config["train"]
    schedule = {
        "version": RESIDUAL_APPLICATION_VERSION,
        "origin_epoch": int(origin_epoch),
        "warmup_epochs": int(train_config.get("residual_warmup_epochs", 1)),
        "ramp_epochs": int(train_config.get("residual_ramp_epochs", 6)),
        "max_scale": float(train_config.get("residual_max_scale", 1.0)),
    }
    # Validate all fields through the public scheduler at the first epoch.
    _scheduled_residual_application_scale(
        int(origin_epoch),
        origin_epoch=schedule["origin_epoch"],
        warmup_epochs=schedule["warmup_epochs"],
        ramp_epochs=schedule["ramp_epochs"],
        max_scale=schedule["max_scale"],
    )
    return schedule


def _set_residual_application_schedule(
    model: torch.nn.Module,
    schedule: dict[str, Any],
    *,
    scale: float,
) -> None:
    raw = _raw_model(model)
    setter = getattr(raw, "set_residual_application_scale", None)
    if setter is None:
        raise RuntimeError("unified model does not support residual application warmup")
    normalized = {
        "version": int(schedule["version"]),
        "origin_epoch": int(schedule["origin_epoch"]),
        "warmup_epochs": int(schedule["warmup_epochs"]),
        "ramp_epochs": int(schedule["ramp_epochs"]),
        "max_scale": float(schedule["max_scale"]),
    }
    if normalized["version"] != RESIDUAL_APPLICATION_VERSION:
        raise RuntimeError(
            "unsupported residual application metadata version "
            f"{normalized['version']}"
        )
    _scheduled_residual_application_scale(
        normalized["origin_epoch"],
        origin_epoch=normalized["origin_epoch"],
        warmup_epochs=normalized["warmup_epochs"],
        ramp_epochs=normalized["ramp_epochs"],
        max_scale=normalized["max_scale"],
    )
    setter(float(scale))
    raw._residual_application_schedule = normalized


def _residual_application_metadata(model: torch.nn.Module) -> dict[str, Any] | None:
    raw = _raw_model(model)
    schedule = getattr(raw, "_residual_application_schedule", None)
    if schedule is None:
        return None
    return {
        **dict(schedule),
        "scale": float(getattr(raw, "residual_application_scale")),
    }


def _validate_residual_application_metadata_schema(
    metadata: Any,
) -> dict[str, Any]:
    """Validate checkpoint metadata without any lossy type coercion."""

    if not isinstance(metadata, dict):
        raise RuntimeError(
            "checkpoint residual_application metadata must be a mapping"
        )
    required_keys = {
        "version",
        "origin_epoch",
        "warmup_epochs",
        "ramp_epochs",
        "max_scale",
        "scale",
    }
    if set(metadata) != required_keys:
        raise RuntimeError(
            "checkpoint residual_application metadata has the wrong schema; "
            f"expected={sorted(required_keys)}, got={sorted(metadata)}"
        )
    integer_keys = ("version", "origin_epoch", "warmup_epochs", "ramp_epochs")
    for key in integer_keys:
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise RuntimeError(
                "checkpoint residual_application integer field "
                f"{key!r} must be an integer, got {value!r}"
            )
    for key in ("max_scale", "scale"):
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError(
                "checkpoint residual_application numeric field "
                f"{key!r} must be a real number, got {value!r}"
            )
    return metadata


def _restore_residual_application_from_checkpoint(
    model: torch.nn.Module,
    payload: dict[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
    metadata = payload.get("residual_application")
    if metadata is None:
        if required:
            raise RuntimeError(
                "teacher checkpoint is missing residual_application metadata"
            )
        return None
    metadata = _validate_residual_application_metadata_schema(metadata)
    required_keys = set(metadata)
    schedule = {key: metadata[key] for key in required_keys if key != "scale"}
    checkpoint_epoch = payload.get("epoch")
    if isinstance(checkpoint_epoch, bool) or not isinstance(
        checkpoint_epoch, Integral
    ):
        raise RuntimeError(
            "teacher checkpoint epoch must be an integer for residual schedule "
            f"validation, got {checkpoint_epoch!r}"
        )
    try:
        _set_residual_application_schedule(
            model, schedule, scale=float(metadata["scale"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"invalid checkpoint residual_application metadata: {exc}"
        ) from exc
    restored = _residual_application_metadata(model)
    if restored is None:  # Defensive: _set above must install it.
        raise RuntimeError("failed to restore residual application metadata")
    if int(checkpoint_epoch) < int(restored["origin_epoch"]):
        raise RuntimeError(
            "teacher checkpoint epoch predates its residual schedule origin; "
            f"epoch={int(checkpoint_epoch)}, origin={int(restored['origin_epoch'])}"
        )
    expected_scale = _scheduled_residual_application_scale(
        int(checkpoint_epoch),
        origin_epoch=int(restored["origin_epoch"]),
        warmup_epochs=int(restored["warmup_epochs"]),
        ramp_epochs=int(restored["ramp_epochs"]),
        max_scale=float(restored["max_scale"]),
    )
    saved_scale = float(restored["scale"])
    if not math.isclose(saved_scale, expected_scale, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(
            "checkpoint residual_application scale does not match its epoch/schedule; "
            f"epoch={int(checkpoint_epoch)}, saved={saved_scale:.12g}, "
            f"expected={expected_scale:.12g}"
        )
    return restored


def _validate_teacher_prior_identity(
    model: torch.nn.Module,
    payload: dict[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
    raw = _raw_model(model)
    current = getattr(raw, "teacher_prior_identity", None)
    if current is None:
        raise RuntimeError("target model does not expose a TorchScript teacher identity")
    saved = payload.get("teacher_prior_identity")
    if saved is None:
        if required:
            raise RuntimeError("teacher checkpoint is missing teacher_prior_identity")
        return None
    if not isinstance(saved, dict):
        raise RuntimeError("checkpoint teacher_prior_identity must be a mapping")
    differing = sorted(
        key
        for key in set(saved) | set(current)
        if key not in saved
        or key not in current
        or type(saved[key]) is not type(current[key])
        or saved[key] != current[key]
    )
    if differing:
        raise RuntimeError(
            "TorchScript teacher identity mismatch; "
            f"differing_fields={differing}, saved={saved}, current={current}"
        )
    return dict(current)


def _best_candidate_guard(
    train_config: dict[str, Any],
    metrics: dict[str, float],
    *,
    is_anchor: bool,
) -> tuple[bool, str | None]:
    """Apply optional safety thresholds before replacing ``best.pt``.

    The guard is opt-in so existing v3.1/v3.2 configurations keep their exact
    historical checkpoint-selection behavior.  The alpha=0 teacher anchor is
    always admitted after validating the guard schema.
    """

    guard = train_config.get("best_guard")
    if guard is None:
        return True, None
    if not isinstance(guard, dict):
        raise ValueError("train.best_guard must be a mapping when enabled")
    checks = {
        "min_epe_gain": ("epe_gain", "min"),
        "max_fold_rate": ("fold_rate", "max"),
        "min_jacobian_p01": ("jacobian_p01", "min"),
        "max_epe_exclusive": ("epe", "max_exclusive"),
        "min_epe_gain_exclusive": ("epe_gain", "min_exclusive"),
        "min_final_win_rate_exclusive": (
            "final_win_rate",
            "min_exclusive",
        ),
        "max_fold_rate_exclusive": ("fold_rate", "max_exclusive"),
        "min_line_epe_gain_exclusive": (
            "line_epe_gain",
            "min_exclusive",
        ),
        "min_line_straightness_gain_exclusive": (
            "line_straightness_gain",
            "min_exclusive",
        ),
    }
    unknown = set(guard) - set(checks)
    if unknown:
        raise ValueError(f"unknown train.best_guard keys: {sorted(unknown)}")
    thresholds: dict[str, float] = {}
    for key, value in guard.items():
        threshold = float(value)
        if not math.isfinite(threshold):
            raise ValueError(f"train.best_guard.{key} must be finite")
        thresholds[key] = threshold
    if is_anchor:
        return True, None

    failures: list[str] = []
    for key, threshold in thresholds.items():
        metric_name, mode = checks[key]
        if metric_name not in metrics:
            raise KeyError(
                f"train.best_guard.{key} requires validation metric {metric_name!r}"
            )
        value = float(metrics[metric_name])
        if not math.isfinite(value):
            failures.append(f"{metric_name}=non-finite")
        elif mode == "min" and value < threshold:
            failures.append(f"{metric_name}={value:.6g} < {threshold:.6g}")
        elif mode == "max" and value > threshold:
            failures.append(f"{metric_name}={value:.6g} > {threshold:.6g}")
        elif mode == "min_exclusive" and value <= threshold:
            failures.append(f"{metric_name}={value:.6g} <= {threshold:.6g}")
        elif mode == "max_exclusive" and value >= threshold:
            failures.append(f"{metric_name}={value:.6g} >= {threshold:.6g}")
    return (not failures), "; ".join(failures) if failures else None


def _validate_existing_teacher_anchor(
    path: Path,
    *,
    teacher_identity: dict[str, Any],
    residual_metadata: dict[str, Any],
    deployment_contract: dict[str, Any],
    capacity_evidence_receipt: dict[str, Any] | None = None,
) -> None:
    """Refuse to reuse anything but the exact immutable teacher anchor.

    An anchor is not merely an alpha-zero checkpoint.  Its complete residual
    schedule is part of the continuation contract: silently keeping an older
    anchor with a different warmup or ramp would make later deployment and
    recovery metadata disagree with the run that owns it.
    """

    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"existing teacher anchor checkpoint is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"existing teacher anchor is not a checkpoint mapping: {path}")
    saved_identity = payload.get("teacher_prior_identity")
    saved_residual = payload.get("residual_application")
    failures: list[str] = []
    if capacity_evidence_receipt is not None:
        try:
            expected_receipt = strict_validate_teacher_capacity_receipt(
                capacity_evidence_receipt
            )
            saved_receipt = strict_validate_teacher_capacity_receipt(
                payload.get("capacity_evidence_receipt")
            )
            if not _typed_equal(saved_receipt, expected_receipt):
                failures.append("capacity_evidence_receipt differs")
        except ValueError as exc:
            failures.append(f"capacity_evidence_receipt is invalid: {exc}")
    if payload.get("stage") != "unified":
        failures.append(f"stage is {payload.get('stage')!r}, not 'unified'")
    if payload.get("prior_backend") != "torchscript":
        failures.append(
            f"prior_backend is {payload.get('prior_backend')!r}, not 'torchscript'"
        )
    if payload.get("training_revision") != TEACHER_WARMUP_REVISION:
        failures.append(
            "training_revision differs "
            f"({payload.get('training_revision')!r} != {TEACHER_WARMUP_REVISION!r})"
        )
    try:
        validate_teacher_deployment_contract(
            payload.get("deployment_contract"),
            deployment_contract,
            external_files_authenticated=True,
        )
    except RuntimeError as exc:
        failures.append(str(exc))
    if not isinstance(saved_identity, dict) or any(
        key not in saved_identity
        or type(saved_identity[key]) is not type(value)
        or saved_identity[key] != value
        for key, value in teacher_identity.items()
    ) or set(saved_identity) != set(teacher_identity):
        failures.append("teacher_prior_identity differs")
    if not isinstance(saved_residual, dict):
        failures.append("residual_application is missing")
    else:
        try:
            _validate_residual_application_metadata_schema(saved_residual)
        except RuntimeError as exc:
            failures.append(f"residual_application is invalid: {exc}")
        try:
            saved_scale = float(saved_residual.get("scale", float("nan")))
        except (TypeError, ValueError):
            saved_scale = float("nan")
        if saved_scale != 0.0:
            failures.append("saved residual scale is not zero")
        if saved_residual != residual_metadata:
            failures.append("residual_application differs")
    expected_origin = residual_metadata.get("origin_epoch")
    saved_epoch = payload.get("epoch")
    if (
        isinstance(saved_epoch, bool)
        or not isinstance(saved_epoch, Integral)
        or saved_epoch != expected_origin
    ):
        failures.append(
            f"anchor epoch differs ({saved_epoch!r} != {expected_origin!r})"
        )
    if failures:
        raise RuntimeError(
            f"existing teacher anchor checkpoint is incompatible: {path}: "
            + "; ".join(failures)
        )


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any], stage: str) -> AdamW:
    train_config = config["train"]
    raw = _raw_model(model)
    pose_head = getattr(raw.prior, "global_pose_head", None)
    pose_parameters = (
        [parameter for parameter in pose_head.parameters() if parameter.requires_grad]
        if pose_head is not None
        else []
    )
    pose_ids = {id(parameter) for parameter in pose_parameters}
    prior_parameters = [
        parameter
        for parameter in raw.prior.parameters()
        if parameter.requires_grad and id(parameter) not in pose_ids
    ]
    groups: list[dict[str, Any]] = []
    if prior_parameters:
        groups.append(
            {
                "params": prior_parameters,
                "lr": float(
                    train_config.get("lr_prior_joint", 2e-5)
                    if stage == "unified"
                    else train_config.get("lr_prior", 2e-4)
                ),
                "name": "prior",
            }
        )
    if pose_parameters:
        groups.append(
            {
                "params": pose_parameters,
                "lr": float(
                    train_config.get(
                        "lr_global_pose",
                        train_config.get("lr_unified", 1e-4)
                        if stage == "unified"
                        else train_config.get("lr_prior", 2e-4),
                    )
                ),
                "name": "global_pose",
            }
        )
    prior_ids = {id(parameter) for parameter in raw.prior.parameters() if parameter.requires_grad}
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


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Avoid truncated checkpoints when a job is pre-empted."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_prior_backend(payload: dict[str, Any], state: dict[str, Tensor]) -> str:
    """Read prior-backend metadata, with marker fallback for raw state dicts."""

    declared = payload.get("prior_backend")
    if declared is None:
        saved_config = payload.get("config")
        if isinstance(saved_config, dict):
            saved_model_config = saved_config.get("model")
            if isinstance(saved_model_config, dict):
                declared = saved_model_config.get("prior_backend")
    marker_present = "prior._teacher_backend_marker" in state
    if declared is None:
        return "torchscript" if marker_present else "learned"
    normalized = str(declared).lower()
    if normalized not in {"learned", "torchscript"}:
        raise RuntimeError(f"checkpoint has unknown prior_backend={declared!r}")
    if normalized == "learned" and marker_present:
        raise RuntimeError(
            "checkpoint prior metadata says learned but contains a teacher marker"
        )
    if normalized == "torchscript" and not marker_present:
        raise RuntimeError(
            "checkpoint prior metadata says torchscript but has no teacher marker"
        )
    return normalized


def _expected_learned_prior_keys(payload: dict[str, Any]) -> set[str]:
    """Recover the complete source-prior schema for a strict backend swap."""

    saved_keys = payload.get("prior_state_keys")
    if isinstance(saved_keys, (list, tuple)) and saved_keys:
        expected = {str(key) for key in saved_keys if str(key).startswith("prior.")}
        if expected:
            return expected
    saved_config = payload.get("config")
    model_config = (
        saved_config.get("model") if isinstance(saved_config, dict) else None
    )
    if not isinstance(model_config, dict):
        raise RuntimeError(
            "strict learned -> TorchScript prior migration requires the source "
            "checkpoint config or prior_state_keys metadata"
        )
    # Construct only the small CPU learned prior, not Qwen or the unified
    # model. Preserve RNG state so schema validation cannot perturb training.
    rng_state = torch.random.get_rng_state()
    try:
        source_prior = build_learned_geometry_prior(dict(model_config))
        return {f"prior.{key}" for key in source_prior.state_dict()}
    finally:
        torch.random.set_rng_state(rng_state)


def _load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    target_stage: str,
    optimizer: torch.optim.Optimizer | None = None,
    expected_best_metric_name: str | None = None,
    expected_best_metric_mode: str | None = None,
    current_deployment_contract: dict[str, Any] | None = None,
    capacity_evidence_receipt: dict[str, Any] | None = None,
    resume_file_identity: dict[str, Any] | None = None,
    require_capacity_evidence_receipt: bool = False,
) -> tuple[int, str | None, float | None]:
    # Unified checkpoints contain optimizer/config metadata in addition to
    # tensors.  Be explicit for PyTorch >= 2.6 while retaining old-version
    # compatibility; callers only pass trusted local training checkpoints.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = _normalize_checkpoint_keys(payload.get("model", payload))
    source_stage = str(payload.get("stage", target_stage))
    same_stage = source_stage == target_stage
    checkpoint_epoch = payload.get("epoch", -1)
    if same_stage and (
        isinstance(checkpoint_epoch, bool)
        or not isinstance(checkpoint_epoch, Integral)
        or int(checkpoint_epoch) < -1
    ):
        raise RuntimeError(
            "checkpoint epoch must be an integer >= -1 for same-stage resume, "
            f"got {checkpoint_epoch!r}"
        )
    if same_stage and target_stage == "unified":
        _checkpoint_correlation_temperature(payload, required=True)
    start_epoch = int(checkpoint_epoch) + 1 if same_stage else 0
    target_prior_backend = str(getattr(model, "prior_backend", "learned")).lower()
    source_prior_backend = _checkpoint_prior_backend(payload, state)
    teacher_backend_migration = (
        same_stage
        and source_prior_backend == "learned"
        and target_prior_backend == "torchscript"
    )
    strict_teacher_resume = (
        same_stage
        and source_prior_backend == "torchscript"
        and target_prior_backend == "torchscript"
    )
    validated_capacity_receipt: dict[str, Any] | None = None
    if require_capacity_evidence_receipt and (
        teacher_backend_migration or strict_teacher_resume
    ):
        if capacity_evidence_receipt is None:
            raise RuntimeError(
                "teacher checkpoint loading requires a capacity evidence receipt"
            )
        try:
            validated_capacity_receipt = strict_validate_teacher_capacity_receipt(
                capacity_evidence_receipt
            )
        except ValueError as exc:
            raise RuntimeError(
                f"teacher capacity evidence receipt is invalid: {exc}"
            ) from exc

    # Authenticate the migration and continuation envelopes before allowing
    # even a partial state-dict mutation of the target model.
    if teacher_backend_migration and validated_capacity_receipt is not None:
        if resume_file_identity is None:
            resume_file_identity = stable_external_file_identity(
                path, label="learned migration resume checkpoint"
            )
        actual_sha256 = resume_file_identity.get("sha256")
        seed = validated_capacity_receipt["migration_seed"]
        seed_failures: list[str] = []
        if type(actual_sha256) is not str or actual_sha256 != seed["sha256"]:
            seed_failures.append("sha256")
        actual_stage = payload.get("stage")
        if type(actual_stage) is not str or actual_stage != seed["stage"]:
            seed_failures.append("stage")
        if (
            isinstance(checkpoint_epoch, bool)
            or not isinstance(checkpoint_epoch, Integral)
            or int(checkpoint_epoch) != seed["epoch_index"]
        ):
            seed_failures.append("epoch_index")
        actual_completed = (
            int(checkpoint_epoch) + 1
            if not isinstance(checkpoint_epoch, bool)
            and isinstance(checkpoint_epoch, Integral)
            else None
        )
        if actual_completed != seed["completed_epochs"]:
            seed_failures.append("completed_epochs")
        if seed_failures:
            raise RuntimeError(
                "teacher capacity receipt migration_seed does not match the "
                f"learned resume checkpoint; differing_fields={seed_failures}"
            )

    if strict_teacher_resume and validated_capacity_receipt is not None:
        try:
            checkpoint_receipt = strict_validate_teacher_capacity_receipt(
                payload.get("capacity_evidence_receipt")
            )
        except ValueError as exc:
            raise RuntimeError(
                "teacher resume checkpoint has no strict-valid "
                f"capacity_evidence_receipt: {exc}"
            ) from exc
        if not _typed_equal(checkpoint_receipt, validated_capacity_receipt):
            raise RuntimeError(
                "teacher resume checkpoint capacity_evidence_receipt differs "
                "from the approved environment receipt"
            )

    missing, unexpected = model.load_state_dict(state, strict=False)
    compatible_prefixes = tuple(
        getattr(model, "checkpoint_compatible_missing_prefixes", ())
    )
    # A migration may omit a newly introduced branch only as one complete
    # unit.  Accepting an arbitrary tensor under a compatible prefix would
    # silently treat a truncated/corrupt current checkpoint as an old one.
    model_keys = set(model.state_dict())
    missing_set = set(missing)
    compatible_missing: list[str] = []
    for prefix in compatible_prefixes:
        expected_group = {key for key in model_keys if key.startswith(prefix)}
        missing_group = missing_set & expected_group
        if expected_group and missing_group == expected_group:
            compatible_missing.extend(sorted(missing_group))
    teacher_compatible_missing: list[str] = []
    teacher_compatible_unexpected: list[str] = []
    if teacher_backend_migration:
        teacher_marker = getattr(model, "teacher_checkpoint_marker_key", None)
        if teacher_marker is None or teacher_marker not in missing_set:
            raise RuntimeError(
                "learned -> TorchScript prior migration is missing its complete "
                "teacher backend marker transition"
            )
        teacher_compatible_missing.append(teacher_marker)
        teacher_compatible_unexpected = [
            key for key in unexpected if key.startswith("prior.")
        ]
        expected_source_prior = _expected_learned_prior_keys(payload)
        actual_source_prior = {
            key for key in state if key.startswith("prior.")
        }
        if actual_source_prior != expected_source_prior:
            missing_source = sorted(expected_source_prior - actual_source_prior)
            extra_source = sorted(actual_source_prior - expected_source_prior)
            raise RuntimeError(
                "learned -> TorchScript prior migration requires a complete source "
                f"prior; missing_source={missing_source[:12]}, "
                f"extra_source={extra_source[:12]}"
            )
    bad_missing = [
        key
        for key in missing
        if key not in compatible_missing and key not in teacher_compatible_missing
    ]
    bad_unexpected = [
        key for key in unexpected if key not in teacher_compatible_unexpected
    ]
    if same_stage:
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "checkpoint mismatch; "
                f"missing={bad_missing[:12]}, unexpected={bad_unexpected[:12]}"
            )
        if compatible_missing or teacher_backend_migration:
            migrated = compatible_missing + teacher_compatible_missing
            print(
                "checkpoint migration: initialized new parameters "
                f"{migrated[:12]} and reset optimizer state",
                flush=True,
            )
    elif source_stage == "prior" and target_stage in {"joint", "unified"}:
        # Stage-A checkpoints contain only the shared `prior.*` branch.  Every
        # missing key must belong to the newly constructed joint backend.
        missing_prior = [
            key
            for key in missing
            if key.startswith("prior.") and key not in compatible_missing
        ]
        if unexpected or missing_prior:
            raise RuntimeError(
                "Stage-A migration failed; "
                f"missing_prior={missing_prior[:12]}, unexpected={unexpected[:12]}"
            )
    else:
        raise RuntimeError(
            f"unsupported checkpoint migration {source_stage!r} -> {target_stage!r}"
        )
    if same_stage and optimizer is not None and not compatible_missing:
        saved_optimizer = payload.get("optimizer")
        if strict_teacher_resume and (
            not isinstance(saved_optimizer, dict) or not saved_optimizer
        ):
            raise RuntimeError(
                "teacher checkpoint is missing optimizer state; exact training "
                "continuation is unsafe"
            )
        if saved_optimizer is not None and not teacher_backend_migration:
            if strict_teacher_resume:
                saved_groups = saved_optimizer.get("param_groups")
                current_groups = optimizer.state_dict().get("param_groups")
                if not isinstance(saved_groups, list) or not isinstance(
                    current_groups, list
                ):
                    raise RuntimeError(
                        "teacher checkpoint optimizer param_groups are invalid"
                    )
                saved_names = [group.get("name") for group in saved_groups]
                current_names = [group.get("name") for group in current_groups]
                if saved_names != current_names:
                    raise RuntimeError(
                        "teacher checkpoint optimizer group names differ; "
                        f"saved={saved_names}, current={current_names}"
                    )
            optimizer.load_state_dict(saved_optimizer)
    if target_prior_backend == "torchscript":
        if teacher_backend_migration:
            # The old refiner was trained around a much weaker/different prior.
            # Hide it completely until raw-residual supervision adapts it.
            setter = getattr(model, "set_residual_application_scale", None)
            if setter is None:
                raise RuntimeError(
                    "TorchScript target does not support residual application warmup"
                )
            setter(0.0)
            model._residual_schedule_origin_epoch = int(start_epoch)
        elif source_prior_backend == "torchscript":
            # There are no production checkpoints predating this contract.
            # Never infer/relax it from a free-form revision string: every
            # teacher->teacher resume must bind the same external model and
            # restore the exact residual deployment homotopy.
            _validate_teacher_prior_identity(model, payload, required=True)
            _restore_residual_application_from_checkpoint(
                model, payload, required=True
            )
            if current_deployment_contract is not None:
                validate_teacher_deployment_contract(
                    payload.get("deployment_contract"),
                    current_deployment_contract,
                    external_files_authenticated=True,
                )
    # A best score measured with the learned prior is not comparable to the
    # teacher-anchored graph.  Reset it so the exact alpha=0 anchor is always a
    # recoverable first candidate.
    best_payload = (
        payload.get("best_metric", {})
        if same_stage and not teacher_backend_migration
        else {}
    )
    if strict_teacher_resume:
        if not isinstance(best_payload, dict) or set(best_payload) != {
            "name",
            "mode",
            "value",
        }:
            raise RuntimeError(
                "teacher checkpoint best_metric must contain exactly "
                "name/mode/value"
            )
        if not isinstance(best_payload["name"], str) or not best_payload["name"]:
            raise RuntimeError("teacher checkpoint best_metric.name must be non-empty")
        if best_payload["mode"] not in {"min", "max"}:
            raise RuntimeError(
                "teacher checkpoint best_metric.mode must be 'min' or 'max'"
            )
        best_number = best_payload["value"]
        if isinstance(best_number, bool) or not isinstance(best_number, Real):
            raise RuntimeError(
                "teacher checkpoint best_metric.value must be a real number"
            )
        if math.isnan(float(best_number)):
            raise RuntimeError("teacher checkpoint best_metric.value must not be NaN")
    best_name = (
        str(best_payload.get("name"))
        if isinstance(best_payload, dict) and best_payload.get("name") is not None
        else None
    )
    best_value = (
        float(best_payload.get("value"))
        if isinstance(best_payload, dict) and best_payload.get("value") is not None
        else None
    )
    if (
        strict_teacher_resume
        and
        expected_best_metric_name is not None
        and best_name is not None
        and best_name != expected_best_metric_name
    ):
        raise RuntimeError(
            "checkpoint best metric name disagrees with current config; "
            f"checkpoint={best_name!r}, config={expected_best_metric_name!r}"
        )
    stored_best_mode = (
        str(best_payload.get("mode")).lower()
        if isinstance(best_payload, dict) and best_payload.get("mode") is not None
        else None
    )
    if (
        strict_teacher_resume
        and
        expected_best_metric_mode is not None
        and stored_best_mode is not None
        and stored_best_mode != expected_best_metric_mode
    ):
        raise RuntimeError(
            "checkpoint best metric mode disagrees with current config; "
            f"checkpoint={stored_best_mode!r}, config={expected_best_metric_mode!r}"
        )
    return start_epoch, best_name, best_value


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
    max_batches: int | None = None,
    preview_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _to_device(batch, device)
        if stage == "joint" and not bool(batch["guide_available"].all()):
            raise RuntimeError("legacy joint validation sample is missing a Qwen RGB guide")
        outputs = _forward(model, batch, stage)
        losses = criterion(outputs, batch)
        if preview_path is not None and batch_index == 0:
            _save_validation_preview(batch, outputs, losses["rectified"], preview_path)
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
    capacity_evidence_receipt = _teacher_capacity_receipt_from_environment(
        config, stage=stage
    )
    device, rank, world_size, distributed = _distributed_device(config)
    _seed_everything(int(config.get("seed", 42)) + rank)
    is_main = rank == 0
    resume = train_config.get("resume")
    resume_file_identity: dict[str, Any] | None = None
    if capacity_evidence_receipt is not None and resume:
        # Every rank authenticates the exact resume artifact it will load.  The
        # digest is reused below rather than hashing the checkpoint twice.
        resume_file_identity = stable_external_file_identity(
            resume, label="training resume checkpoint"
        )

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
    current_deployment_contract: dict[str, Any] | None = None
    if (
        stage == "unified"
        and str(getattr(model, "prior_backend", "learned")) == "torchscript"
    ):
        teacher_identity = getattr(model, "teacher_prior_identity", None)
        if teacher_identity is None:
            raise RuntimeError("TorchScript model does not expose teacher identity")
        if capacity_evidence_receipt is None:  # Defensive: checked before setup.
            raise RuntimeError("TorchScript teacher training has no capacity receipt")
        _validate_capacity_receipt_teacher(
            capacity_evidence_receipt, teacher_identity
        )
        current_deployment_contract = build_teacher_deployment_contract(
            config,
            teacher_identity=teacher_identity,
        )
    criterion = RectificationLoss(config["loss"]).to(device)
    optimizer = _build_optimizer(model, config, stage)

    best_metric_name = str(train_config.get("best_metric", "epe"))
    best_metric_mode = str(train_config.get("best_metric_mode", "min")).lower()
    if best_metric_mode not in {"min", "max"}:
        raise ValueError("train.best_metric_mode must be 'min' or 'max'")
    best_metric_value = float("inf") if best_metric_mode == "min" else -float("inf")
    start_epoch = 0
    if resume:
        start_epoch, stored_best_name, stored_best_value = _load_checkpoint(
            model,
            resume,
            target_stage=stage,
            optimizer=optimizer,
            expected_best_metric_name=best_metric_name,
            expected_best_metric_mode=best_metric_mode,
            current_deployment_contract=current_deployment_contract,
            capacity_evidence_receipt=capacity_evidence_receipt,
            resume_file_identity=resume_file_identity,
            require_capacity_evidence_receipt=(
                capacity_evidence_receipt is not None
            ),
        )
        if stored_best_name == best_metric_name and stored_best_value is not None:
            best_metric_value = stored_best_value
    residual_schedule: dict[str, Any] | None = None
    if (
        stage == "unified"
        and str(getattr(model, "prior_backend", "learned")) == "torchscript"
    ):
        restored_schedule = getattr(model, "_residual_application_schedule", None)
        origin_epoch = int(
            restored_schedule["origin_epoch"]
            if restored_schedule is not None
            else getattr(model, "_residual_schedule_origin_epoch", start_epoch)
        )
        requested_schedule = _teacher_schedule_from_config(config, origin_epoch)
        if restored_schedule is not None:
            for key in ("version", "warmup_epochs", "ramp_epochs", "max_scale"):
                if restored_schedule[key] != requested_schedule[key]:
                    raise RuntimeError(
                        "resumed teacher residual schedule disagrees with the current "
                        f"config at {key}: checkpoint={restored_schedule[key]!r}, "
                        f"config={requested_schedule[key]!r}"
                    )
            residual_schedule = dict(restored_schedule)
        else:
            residual_schedule = requested_schedule
            initial_scale = _scheduled_residual_application_scale(
                start_epoch,
                origin_epoch=residual_schedule["origin_epoch"],
                warmup_epochs=residual_schedule["warmup_epochs"],
                ramp_epochs=residual_schedule["ramp_epochs"],
                max_scale=residual_schedule["max_scale"],
            )
            _set_residual_application_schedule(
                model, residual_schedule, scale=initial_scale
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
    max_train_steps_value = train_config.get("max_train_steps")
    max_val_batches_value = train_config.get("max_val_batches")
    max_train_steps = (
        int(max_train_steps_value)
        if max_train_steps_value is not None and int(max_train_steps_value) > 0
        else None
    )
    max_val_batches = (
        int(max_val_batches_value)
        if max_val_batches_value is not None and int(max_val_batches_value) > 0
        else None
    )
    preview_every = int(train_config.get("preview_every", 1))
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
            if residual_schedule is not None:
                residual_scale = _scheduled_residual_application_scale(
                    epoch,
                    origin_epoch=residual_schedule["origin_epoch"],
                    warmup_epochs=residual_schedule["warmup_epochs"],
                    ramp_epochs=residual_schedule["ramp_epochs"],
                    max_scale=residual_schedule["max_scale"],
                )
                _set_residual_application_schedule(
                    raw, residual_schedule, scale=residual_scale
                )
            for group in optimizer.param_groups:
                if group.get("name") == "prior":
                    group["lr"] = 0.0 if epoch < freeze_prior_epochs else configured_prior_lr
                elif group.get("name") == "global_pose":
                    # Unlike the mature local curl prior, the new identity-
                    # initialized pose branch must learn immediately from the
                    # source-only big-rotation augmentation.
                    group["lr"] = float(
                        train_config.get("lr_global_pose", train_config.get("lr_unified", 1e-4))
                    )
                elif group.get("name") == "unified_heads":
                    # A resumed optimizer stores the old LR.  Re-apply the
                    # current experiment config so v2 -> v3 continuation uses
                    # the intended schedule without discarding Adam moments.
                    group["lr"] = float(train_config.get("lr_unified", 1e-4))
        running: defaultdict[str, float] = defaultdict(float)
        running_steps = 0
        for step, batch in enumerate(train_loader):
            if max_train_steps is not None and step >= max_train_steps:
                break
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
                        f"epe95={means['epe_p95']:.3f} "
                        f"edge={means['edge_epe']:.3f} "
                        f"line={means['line_epe']:.3f} "
                        f"line_gain={means['line_epe_gain']:.3f} "
                        f"line_n={means['line_normal_mae']:.3f} "
                        f"line_bend={means['line_straightness_error']:.4f} "
                        f"bend_gain={means['line_straightness_gain']:.4f} "
                        f"line_cov={means['structure_coverage']:.4f} "
                        f"prior={means['prior_epe']:.4f} "
                        f"gain={means['epe_gain']:.4f} "
                        f"win={means['final_win_rate']:.3f} "
                        f"fold={means['fold_rate']:.6f} "
                        f"jac_p01={means['jacobian_p01']:.4f} "
                        f"r95={means['residual_p95']:.2f}/"
                        f"{means['applied_residual_p95']:.2f} "
                        f"r_epe={means['residual_epe']:.2f} "
                        f"r_valid={means['residual_target_valid_rate']:.3f} "
                        f"r_alpha={means['residual_application_scale']:.3f} "
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

        preview_path = (
            output_dir / "previews" / f"epoch_{epoch + 1:04d}.jpg"
            if is_main and preview_every > 0 and (epoch + 1) % preview_every == 0
            else None
        )
        metrics = (
            _evaluate(
                model,
                val_loader,
                criterion,
                device,
                stage,
                distributed=distributed,
                max_batches=max_val_batches,
                preview_path=preview_path,
            )
            if val_loader
            else {}
        )
        if is_main and metrics:
            print(
                f"validation epoch={epoch + 1} "
                f"epe={metrics.get('epe', float('nan')):.4f} "
                f"epe95={metrics.get('epe_p95', float('nan')):.3f} "
                f"edge_epe={metrics.get('edge_epe', float('nan')):.3f} "
                f"line_epe={metrics.get('line_epe', float('nan')):.3f} "
                f"prior_line={metrics.get('prior_line_epe', float('nan')):.3f} "
                f"line_gain={metrics.get('line_epe_gain', float('nan')):.3f} "
                f"line_normal={metrics.get('line_normal_mae', float('nan')):.3f} "
                f"line_bend={metrics.get('line_straightness_error', float('nan')):.4f} "
                f"prior_bend={metrics.get('prior_line_straightness_error', float('nan')):.4f} "
                f"bend_gain={metrics.get('line_straightness_gain', float('nan')):.4f} "
                f"line_cov={metrics.get('structure_coverage', float('nan')):.4f} "
                f"seq={metrics.get('flow_per_iteration', float('nan')):.3f} "
                f"prior_epe={metrics.get('prior_epe', float('nan')):.4f} "
                f"gain={metrics.get('epe_gain', float('nan')):.4f} "
                f"win={metrics.get('final_win_rate', float('nan')):.3f} "
                f"fold_rate={metrics.get('fold_rate', float('nan')):.6f} "
                f"jac_p01={metrics.get('jacobian_p01', float('nan')):.4f} "
                f"residual_p95={metrics.get('residual_p95', float('nan')):.2f}/"
                f"{metrics.get('applied_residual_p95', float('nan')):.2f} "
                f"residual_epe={metrics.get('residual_epe', float('nan')):.2f} "
                f"r_valid={metrics.get('residual_target_valid_rate', float('nan')):.3f} "
                f"r_alpha={metrics.get('residual_application_scale', float('nan')):.3f} "
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
            teacher_backend = (
                str(getattr(raw, "prior_backend", "learned")) == "torchscript"
            )
            teacher_identity: dict[str, Any] | None = None
            residual_metadata: dict[str, Any] | None = None
            is_anchor = False
            if teacher_backend:
                teacher_identity = getattr(raw, "teacher_prior_identity", None)
                residual_metadata = _residual_application_metadata(raw)
                if teacher_identity is None or residual_metadata is None:
                    raise RuntimeError(
                        "teacher checkpoint cannot be saved without identity and "
                        "residual application metadata"
                    )
                if current_deployment_contract is None:
                    raise RuntimeError(
                        "teacher checkpoint cannot be saved without a deployment contract"
                    )
                is_anchor = (
                    epoch == int(residual_metadata["origin_epoch"])
                    and float(residual_metadata["scale"]) == 0.0
                )

            is_best = False
            if metrics:
                if best_metric_name not in metrics:
                    raise KeyError(
                        f"train.best_metric={best_metric_name!r} is unavailable; "
                        f"available metrics: {sorted(metrics)}"
                    )
                candidate = float(metrics[best_metric_name])
                if math.isfinite(candidate):
                    metric_improved = (
                        candidate < best_metric_value
                        if best_metric_mode == "min"
                        else candidate > best_metric_value
                    )
                    guard_passed, guard_reason = _best_candidate_guard(
                        train_config, metrics, is_anchor=is_anchor
                    )
                    is_best = metric_improved and guard_passed
                    if metric_improved and not guard_passed:
                        print(
                            "best checkpoint candidate rejected by safety guard: "
                            f"{guard_reason}",
                            flush=True,
                        )
                    if is_best:
                        best_metric_value = candidate
            checkpoint = {
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "stage": stage,
                "architecture": type(raw).__name__,
                "prior_backend": str(getattr(raw, "prior_backend", "learned")),
                "prior_state_keys": sorted(
                    key for key in raw.state_dict() if key.startswith("prior.")
                ),
                "training_revision": (
                    TEACHER_WARMUP_REVISION
                    if teacher_backend
                    else (
                        "v3_2_global_pose_bigrot_safe_fusion"
                        if getattr(raw.prior, "global_pose_head", None) is not None
                        else "v3_1_line_aware"
                    )
                ),
                "config": config,
                "metrics": metrics,
                "best_metric": {
                    "name": best_metric_name,
                    "mode": best_metric_mode,
                    "value": best_metric_value,
                },
            }
            if stage == "unified":
                runtime_temperature = float(
                    getattr(raw, "correlation_temperature", float("nan"))
                )
                scheduled_temperature = _scheduled_correlation_temperature(
                    config, epoch
                )
                if not math.isclose(
                    runtime_temperature,
                    scheduled_temperature,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        "runtime correlation_temperature drifted from the training "
                        f"schedule: runtime={runtime_temperature}, "
                        f"scheduled={scheduled_temperature}"
                    )
                checkpoint["correlation_temperature"] = runtime_temperature
                # Exercise the same strict reader used by inference before any
                # checkpoint reaches disk.
                _checkpoint_correlation_temperature(checkpoint, required=True)
            if teacher_backend:
                if capacity_evidence_receipt is None:
                    raise RuntimeError(
                        "teacher checkpoint cannot be saved without a capacity "
                        "evidence receipt"
                    )
                checkpoint["teacher_prior_identity"] = teacher_identity
                checkpoint["residual_application"] = residual_metadata
                checkpoint["deployment_contract"] = current_deployment_contract
                checkpoint["capacity_evidence_receipt"] = (
                    capacity_evidence_receipt
                )

                # The first teacher epoch is alpha=0, so it is an exact strong-
                # prior recovery point even though its raw residual heads have
                # already received one epoch of auxiliary supervision.
                anchor_path = output_dir / "anchor.pt"
                if is_anchor:
                    if anchor_path.exists():
                        _validate_existing_teacher_anchor(
                            anchor_path,
                            teacher_identity=teacher_identity,
                            residual_metadata=residual_metadata,
                            deployment_contract=current_deployment_contract,
                            capacity_evidence_receipt=capacity_evidence_receipt,
                        )
                        print(
                            "kept compatible immutable teacher anchor checkpoint: "
                            f"{anchor_path}",
                            flush=True,
                        )
                    else:
                        _atomic_torch_save(checkpoint, anchor_path)
                        print(
                            "saved immutable teacher anchor checkpoint: "
                            f"{anchor_path}",
                            flush=True,
                        )
            latest_path = output_dir / "latest.pt"
            _atomic_torch_save(checkpoint, latest_path)
            if (epoch + 1) % int(train_config.get("save_every", 1)) == 0:
                _atomic_torch_save(
                    checkpoint, output_dir / f"epoch_{epoch + 1:04d}.pt"
                )
            if is_best:
                _atomic_torch_save(checkpoint, output_dir / "best.pt")
                print(
                    f"new best checkpoint: {best_metric_name}={best_metric_value:.6f} "
                    f"({best_metric_mode})",
                    flush=True,
                )
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
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--preview-every", type=int)
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
    if args.max_train_steps is not None:
        config["train"]["max_train_steps"] = args.max_train_steps
    if args.max_val_batches is not None:
        config["train"]["max_val_batches"] = args.max_val_batches
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.preview_every is not None:
        config["train"]["preview_every"] = args.preview_every
    result = train(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"saved checkpoint: {result}")


if __name__ == "__main__":
    main()
