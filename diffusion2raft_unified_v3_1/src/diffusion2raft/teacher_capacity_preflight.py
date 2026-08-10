"""Read-only capacity audit for the frozen v3.3 geometry teacher.

The audit measures whether the bounded recurrent residual head can, even with
an oracle prediction, correct the frozen teacher.  It evaluates the manifest's
original source image, deterministic source-only rotations, and optionally the
complete configured source-geometry distribution.  It then recovers the exact
residual ``R`` satisfying ``F_gt = compose(B_teacher, R)`` with the same
fixed-point helper and validity rules used by :class:`RectificationLoss`.

Only the JSON report is written.  No trainable model or optimizer is built and
the supplied migration checkpoint is identified for provenance but not loaded.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch import Tensor, nn

from .config import load_config
from .data import (
    DocumentFlowDataset,
    SourceGeometryAugment,
    apply_source_homography,
    source_affine_homography,
)
from .geometry import (
    compose_backward_flows,
    flow_valid_mask,
    residual_from_composed_flow,
    resize_backward_flow,
)
from .models.teacher_prior import TorchScriptGeometryPrior


REPORT_VERSION = 1
DEFAULT_ROTATION_BIN_EDGES = (0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)
FULL_GEOMETRY_SEED_NAMESPACE = "teacher_capacity_full_geometry_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_identity(path: Path, *, hash_contents: bool = True) -> dict[str, Any]:
    """Describe one stable open inode, including a streaming digest by default."""

    configured = path.expanduser().absolute()
    resolved = configured.resolve(strict=True)
    digest = hashlib.sha256() if hash_contents else None
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if digest is not None:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(handle.fileno())
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_key = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_key != after_key:
        raise RuntimeError(f"file changed while identifying it: {resolved}")
    return {
        "configured_path": str(configured),
        "resolved_path": str(resolved),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "device_id": int(after.st_dev),
        "inode": int(after.st_ino),
        "sha256": None if digest is None else digest.hexdigest(),
        "identity_mode": "stat" if digest is None else "stat+sha256",
    }


def _assert_path_matches_identity(identity: dict[str, Any]) -> None:
    configured = Path(str(identity["configured_path"]))
    path = configured.resolve(strict=True)
    if str(path) != str(identity["resolved_path"]):
        raise RuntimeError(
            f"input path was retargeted during capacity audit: {configured}"
        )
    stat = path.stat()
    observed = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    expected = (
        identity["device_id"],
        identity["inode"],
        identity["size_bytes"],
        identity["mtime_ns"],
    )
    if observed != expected:
        raise RuntimeError(f"input file changed during capacity audit: {path}")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            try:
                os.fsync(directory_fd)
            except OSError as error:
                if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
                    raise
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sample_indices(length: int, count: int | None) -> list[int]:
    if length <= 0:
        raise ValueError("dataset must contain at least one sample")
    if count is None or int(count) == 0 or int(count) >= length:
        return list(range(length))
    if int(count) < 0:
        raise ValueError(f"sample_count must be non-negative, got {count}")
    count = int(count)
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _validate_rotation_angles(angles: Iterable[float]) -> list[float]:
    raw = [float(value) for value in angles]
    if not raw:
        raise ValueError("at least one explicit rotation angle is required")
    result: list[float] = []
    seen: set[float] = set()
    for value in raw:
        if not math.isfinite(value) or abs(value) > 180.0:
            raise ValueError(
                f"rotation angles must be finite and in [-180,180], got {value}"
            )
        # The canonical sample already supplies 0 degrees.  Positive and
        # negative 180 degrees are the same square-canvas transformation.
        if abs(value) <= 1.0e-12:
            continue
        canonical = 180.0 if abs(abs(value) - 180.0) <= 1.0e-12 else value
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    if not result:
        raise ValueError(
            "explicit rotations contain only 0 degrees; original samples are "
            "already evaluated, so provide at least one non-zero angle"
        )
    return result


def build_rotation_plan(
    sample_indices: Sequence[int],
    *,
    explicit_angles: Sequence[float] | None,
    max_rotation_deg: float,
    rotations_per_sample: int,
    seed: int,
) -> tuple[dict[int, list[float]], dict[str, Any]]:
    """Build deterministic rotations for every selected sample.

    With no explicit angle list, stratified jitter covers the configured
    uniform interval instead of relying on a small, potentially clustered
    pseudo-random draw.  This is conditional on rotation being applied: the
    training augmentation probability is reported separately and original
    samples are always measured in their own group.
    """

    if rotations_per_sample < 1:
        raise ValueError("rotations_per_sample must be at least one")
    if explicit_angles is not None:
        angles = _validate_rotation_angles(explicit_angles)
        return (
            {int(index): list(angles) for index in sample_indices},
            {
                "mode": "explicit_cross_product",
                "explicit_angles_deg": angles,
                "rotations_per_sample": len(angles),
            },
        )

    maximum = float(max_rotation_deg)
    if not math.isfinite(maximum) or not 0.0 < maximum <= 180.0:
        raise ValueError(
            "config source_geometry_augment.max_rotation_deg must be in "
            f"(0,180], got {maximum}"
        )
    assignment_count = len(sample_indices) * int(rotations_per_sample)
    rng = random.Random(int(seed))
    angles = [
        -maximum
        + 2.0 * maximum * ((position + rng.random()) / assignment_count)
        for position in range(assignment_count)
    ]
    rng.shuffle(angles)
    plan: dict[int, list[float]] = {}
    cursor = 0
    for index in sample_indices:
        plan[int(index)] = angles[cursor : cursor + rotations_per_sample]
        cursor += rotations_per_sample
    return plan, {
        "mode": "config_stratified_uniform",
        "configured_max_rotation_deg": maximum,
        "rotations_per_sample": int(rotations_per_sample),
        "seed": int(seed),
    }


def _derive_full_geometry_seed(
    base_seed: int,
    dataset_index: int,
    repetition_index: int,
) -> int:
    """Derive an order- and batch-independent CPU RNG seed for one variant."""

    dataset_index = int(dataset_index)
    repetition_index = int(repetition_index)
    if dataset_index < 0:
        raise ValueError(f"dataset_index must be non-negative, got {dataset_index}")
    if repetition_index < 0:
        raise ValueError(
            f"repetition_index must be non-negative, got {repetition_index}"
        )
    payload = {
        "namespace": FULL_GEOMETRY_SEED_NAMESPACE,
        "base_seed": int(base_seed),
        "dataset_index": dataset_index,
        "repetition_index": repetition_index,
    }
    digest = hashlib.sha256(_canonical_json(payload)).digest()
    # torch.Generator.manual_seed accepts a signed 64-bit seed.  Masking the
    # high bit keeps the representation portable while retaining 63 bits.
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def build_full_geometry_plan(
    sample_indices: Sequence[int],
    *,
    transformations_per_sample: int,
    seed: int,
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    """Build deterministic seeds for conditional full-geometry transforms.

    Zero transformations is the backwards-compatible diagnostic default: no
    additional teacher forward is scheduled.  Non-zero plans exercise the
    exact ``SourceGeometryAugment.sample_homography`` distribution, conditional
    on the augmentation branch having been selected.
    """

    transformations_per_sample = int(transformations_per_sample)
    if transformations_per_sample < 0:
        raise ValueError("transformations_per_sample must be non-negative")
    plan = {
        int(index): [
            _derive_full_geometry_seed(seed, int(index), repetition)
            for repetition in range(transformations_per_sample)
        ]
        for index in sample_indices
    }
    seed_records = [
        {"dataset_index": int(index), "seeds": plan[int(index)]}
        for index in sample_indices
    ]
    return plan, {
        "enabled": transformations_per_sample > 0,
        "mode": (
            "deterministic_config_distribution_conditional_on_trigger"
            if transformations_per_sample > 0
            else "disabled"
        ),
        "transformations_per_sample": transformations_per_sample,
        "seed": int(seed),
        "seed_derivation": f"sha256:{FULL_GEOMETRY_SEED_NAMESPACE}",
        "seed_plan_sha256": _sha256_bytes(_canonical_json(seed_records)),
    }


def _sample_deterministic_full_homography(
    augment: SourceGeometryAugment,
    source: Tensor,
    *,
    seed: int,
) -> Tensor:
    """Call the training sampler deterministically without consuming its RNG."""

    # DocumentFlowDataset returns CPU tensors and variants are formed before
    # batching/to(device).  Keeping this explicit makes the RNG protocol
    # independent of the audit GPU and avoids silently seeding a CUDA stream.
    if source.device.type != "cpu":
        raise ValueError(
            "deterministic full-geometry variants must be sampled on CPU, got "
            f"{source.device}"
        )
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.default_generator.manual_seed(int(seed))
        return augment.sample_homography(source)


def _validate_bin_edges(
    edges: Sequence[float], *, maximum_angle: float
) -> tuple[float, ...]:
    result = tuple(float(value) for value in edges)
    if len(result) < 2:
        raise ValueError("rotation bin edges require at least two values")
    if result[0] != 0.0:
        raise ValueError("rotation bin edges must start at 0")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("rotation bin edges must be finite")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("rotation bin edges must be strictly increasing")
    if result[-1] + 1.0e-9 < abs(float(maximum_angle)):
        raise ValueError(
            f"last rotation bin edge {result[-1]} does not cover {maximum_angle} degrees"
        )
    return result


def _rotation_bin(angle_deg: float, edges: Sequence[float]) -> int:
    magnitude = abs(float(angle_deg))
    for index, upper in enumerate(edges[1:]):
        if magnitude < upper or (
            index == len(edges) - 2 and magnitude <= upper + 1.0e-9
        ):
            return index
    raise ValueError(f"absolute rotation {magnitude} is outside bin edges {edges}")


@dataclass(frozen=True)
class CapacitySampleStats:
    eval_pixels: int
    teacher_epe_sum: float
    solver_pixels: int
    overflow_x_solvable_pixels: int
    overflow_y_solvable_pixels: int
    overflow_any_solvable_pixels: int
    trainable_pixels: int
    overflow_x_given_solvable_sample: bool
    overflow_y_given_solvable_sample: bool
    overflow_any_given_solvable_sample: bool
    solver_any_sample: bool
    solver_full_sample: bool
    oracle_residual_absmax_x_px: float | None
    oracle_residual_absmax_y_px: float | None
    stride_oracle_reconstruction_epe_sum: float
    stride_trainable_reconstruction_epe_sum: float

    def as_report(self) -> dict[str, Any]:
        eval_denominator = self.eval_pixels
        solver_denominator = self.solver_pixels
        trainable_denominator = self.trainable_pixels

        def rate(numerator: float | int, denominator: int) -> float | None:
            return None if denominator == 0 else float(numerator / denominator)

        return {
            "eval_pixels": eval_denominator,
            "teacher_epe_px": (
                None
                if eval_denominator == 0
                else float(self.teacher_epe_sum / eval_denominator)
            ),
            "oracle_solver_coverage": rate(solver_denominator, eval_denominator),
            "oracle_solver_any_sample": self.solver_any_sample,
            "oracle_solver_full_sample": self.solver_full_sample,
            "oracle_residual_overflow_given_solvable_x_pixel_rate": rate(
                self.overflow_x_solvable_pixels, solver_denominator
            ),
            "oracle_residual_overflow_given_solvable_y_pixel_rate": rate(
                self.overflow_y_solvable_pixels, solver_denominator
            ),
            "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": rate(
                self.overflow_any_solvable_pixels, solver_denominator
            ),
            "oracle_residual_overflow_given_solvable_x_sample": (
                self.overflow_x_given_solvable_sample
            ),
            "oracle_residual_overflow_given_solvable_y_sample": (
                self.overflow_y_given_solvable_sample
            ),
            "oracle_residual_overflow_given_solvable_any_axis_sample": (
                self.overflow_any_given_solvable_sample
            ),
            "trainable_coverage": rate(trainable_denominator, eval_denominator),
            # Alias the training metric name so the report can be compared
            # directly with RectificationLoss logs.
            "residual_target_valid_rate": rate(
                trainable_denominator, eval_denominator
            ),
            "oracle_residual_axis_absmax_px": {
                "x": self.oracle_residual_absmax_x_px,
                "y": self.oracle_residual_absmax_y_px,
            },
            "stride_oracle_residual_reconstruction_epe_px": rate(
                self.stride_oracle_reconstruction_epe_sum, solver_denominator
            ),
            "stride_trainable_oracle_reconstruction_epe_px": rate(
                self.stride_trainable_reconstruction_epe_sum,
                trainable_denominator,
            ),
        }


def capacity_sample_statistics(
    teacher_flow: Tensor,
    target_flow: Tensor,
    valid: Tensor,
    *,
    max_residual_px: float,
    residual_target_iterations: int,
    max_residual_consistency: float,
    max_valid_flow: float,
    feature_stride: int = 8,
) -> list[CapacitySampleStats]:
    """Compute per-sample capacity statistics with training-identical masks."""

    if teacher_flow.ndim != 4 or teacher_flow.shape[1] != 2:
        raise ValueError(
            f"teacher_flow must be [B,2,H,W], got {tuple(teacher_flow.shape)}"
        )
    if target_flow.shape != teacher_flow.shape:
        raise ValueError(
            "target_flow must match teacher_flow, got "
            f"{tuple(target_flow.shape)} and {tuple(teacher_flow.shape)}"
        )
    if valid.shape != (teacher_flow.shape[0], 1, *teacher_flow.shape[-2:]):
        raise ValueError(
            f"valid must be [B,1,H,W], got {tuple(valid.shape)}"
        )
    max_residual_px = float(max_residual_px)
    max_residual_consistency = float(max_residual_consistency)
    max_valid_flow = float(max_valid_flow)
    if not math.isfinite(max_residual_px) or max_residual_px <= 0.0:
        raise ValueError("max_residual_px must be finite and positive")
    if not math.isfinite(max_residual_consistency) or max_residual_consistency < 0.0:
        raise ValueError("max_residual_consistency must be finite and non-negative")
    if not math.isfinite(max_valid_flow) or max_valid_flow <= 0.0:
        raise ValueError("max_valid_flow must be finite and positive")
    feature_stride = int(feature_stride)
    if feature_stride < 1:
        raise ValueError("feature_stride must be at least one")
    height, width = (int(value) for value in teacher_flow.shape[-2:])
    if height % feature_stride or width % feature_stride:
        raise ValueError(
            "flow size must be divisible by feature_stride, got "
            f"{(height, width)} and stride={feature_stride}"
        )

    with torch.no_grad(), torch.autocast(
        device_type=teacher_flow.device.type, enabled=False
    ):
        teacher = teacher_flow.detach().float()
        target = target_flow.detach().float()
        finite = torch.isfinite(target).all(dim=1, keepdim=True)
        magnitude = torch.linalg.vector_norm(target, dim=1, keepdim=True)
        valid_mask = valid.bool() & finite & (magnitude < max_valid_flow)
        if not bool(torch.isfinite(teacher).all()):
            raise ValueError("teacher flow contains NaN or infinite values")
        safe_target = torch.where(finite.expand_as(target), target, teacher)
        residual, consistency = residual_from_composed_flow(
            teacher,
            safe_target,
            iterations=int(residual_target_iterations),
        )
        residual_finite = torch.isfinite(residual).all(dim=1, keepdim=True)
        consistency_finite = torch.isfinite(consistency)
        safe_residual = torch.where(
            torch.isfinite(residual), residual, torch.zeros_like(residual)
        )
        residual_map_valid = flow_valid_mask(safe_residual, residual.shape[-2:])
        solver_valid = (
            valid_mask
            & residual_finite
            & consistency_finite
            & residual_map_valid
            & (consistency <= max_residual_consistency)
        )
        over_x_raw = safe_residual[:, 0:1].abs() > max_residual_px
        over_y_raw = safe_residual[:, 1:2].abs() > max_residual_px
        over_any_raw = over_x_raw | over_y_raw
        over_x = solver_valid & over_x_raw
        over_y = solver_valid & over_y_raw
        over_any = solver_valid & over_any_raw
        trainable_valid = solver_valid & ~over_any_raw
        teacher_epe = torch.linalg.vector_norm(teacher - target, dim=1, keepdim=True)

        # The refiner predicts on its feature grid, and resize_backward_flow is
        # exactly how its low-resolution residual is restored.  This oracle
        # down/up pass isolates spatial-bandwidth loss from the 24 px L-inf cap.
        low_size = (height // feature_stride, width // feature_stride)
        low_residual = resize_backward_flow(
            safe_residual,
            low_size,
            source_size_from=(height, width),
            source_size_to=low_size,
        )
        restored_residual = resize_backward_flow(
            low_residual,
            (height, width),
            source_size_from=low_size,
            source_size_to=(height, width),
        )
        stride_recomposed = compose_backward_flows(teacher, restored_residual)
        stride_reconstruction_epe = torch.linalg.vector_norm(
            stride_recomposed - target, dim=1, keepdim=True
        )

    results: list[CapacitySampleStats] = []
    for batch_index in range(int(teacher.shape[0])):
        sample_eval = valid_mask[batch_index, 0]
        eval_pixels = int(sample_eval.sum().item())
        sample_solver = solver_valid[batch_index, 0]
        solver_pixels = int(sample_solver.sum().item())
        sample_trainable = trainable_valid[batch_index, 0]
        trainable_pixels = int(sample_trainable.sum().item())

        def count(mask: Tensor, support: Tensor) -> int:
            return int((mask[batch_index, 0] & support).sum().item())

        x_count = count(over_x, sample_solver)
        y_count = count(over_y, sample_solver)
        any_count = count(over_any, sample_solver)
        if solver_pixels:
            residual_x = safe_residual[batch_index, 0].abs()[sample_solver]
            residual_y = safe_residual[batch_index, 1].abs()[sample_solver]
            axis_max_x: float | None = float(residual_x.max().item())
            axis_max_y: float | None = float(residual_y.max().item())
        else:
            axis_max_x = None
            axis_max_y = None
        if eval_pixels:
            epe_sum = float(teacher_epe[batch_index, 0][sample_eval].sum().item())
        else:
            epe_sum = 0.0
        results.append(
            CapacitySampleStats(
                eval_pixels=eval_pixels,
                teacher_epe_sum=epe_sum,
                solver_pixels=solver_pixels,
                overflow_x_solvable_pixels=x_count,
                overflow_y_solvable_pixels=y_count,
                overflow_any_solvable_pixels=any_count,
                trainable_pixels=trainable_pixels,
                overflow_x_given_solvable_sample=x_count > 0,
                overflow_y_given_solvable_sample=y_count > 0,
                overflow_any_given_solvable_sample=any_count > 0,
                solver_any_sample=solver_pixels > 0,
                solver_full_sample=eval_pixels > 0 and solver_pixels == eval_pixels,
                oracle_residual_absmax_x_px=axis_max_x,
                oracle_residual_absmax_y_px=axis_max_y,
                stride_oracle_reconstruction_epe_sum=float(
                    stride_reconstruction_epe[batch_index, 0][sample_solver]
                    .sum()
                    .item()
                ),
                stride_trainable_reconstruction_epe_sum=float(
                    stride_reconstruction_epe[batch_index, 0][sample_trainable]
                    .sum()
                    .item()
                ),
            )
        )
    return results


class _Accumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.eval_samples = 0
        self.eval_pixels = 0
        self.teacher_epe_sum = 0.0
        self.solver_pixels = 0
        self.trainable_pixels = 0
        self.solver_any_samples = 0
        self.solver_full_samples = 0
        self.overflow_x_solvable_pixels = 0
        self.overflow_y_solvable_pixels = 0
        self.overflow_any_solvable_pixels = 0
        self.overflow_x_solvable_samples = 0
        self.overflow_y_solvable_samples = 0
        self.overflow_any_solvable_samples = 0
        self.absmax_x: float | None = None
        self.absmax_y: float | None = None
        self.stride_reconstruction_epe_sum = 0.0
        self.stride_trainable_reconstruction_epe_sum = 0.0

    def add(self, value: CapacitySampleStats) -> None:
        self.samples += 1
        if value.eval_pixels > 0:
            self.eval_samples += 1
        self.eval_pixels += value.eval_pixels
        self.teacher_epe_sum += value.teacher_epe_sum
        self.solver_pixels += value.solver_pixels
        self.trainable_pixels += value.trainable_pixels
        self.solver_any_samples += int(value.solver_any_sample)
        self.solver_full_samples += int(value.solver_full_sample)
        self.overflow_x_solvable_pixels += value.overflow_x_solvable_pixels
        self.overflow_y_solvable_pixels += value.overflow_y_solvable_pixels
        self.overflow_any_solvable_pixels += value.overflow_any_solvable_pixels
        self.overflow_x_solvable_samples += int(
            value.overflow_x_given_solvable_sample
        )
        self.overflow_y_solvable_samples += int(
            value.overflow_y_given_solvable_sample
        )
        self.overflow_any_solvable_samples += int(
            value.overflow_any_given_solvable_sample
        )
        self.stride_reconstruction_epe_sum += (
            value.stride_oracle_reconstruction_epe_sum
        )
        self.stride_trainable_reconstruction_epe_sum += (
            value.stride_trainable_reconstruction_epe_sum
        )
        if value.oracle_residual_absmax_x_px is not None:
            self.absmax_x = (
                value.oracle_residual_absmax_x_px
                if self.absmax_x is None
                else max(self.absmax_x, value.oracle_residual_absmax_x_px)
            )
        if value.oracle_residual_absmax_y_px is not None:
            self.absmax_y = (
                value.oracle_residual_absmax_y_px
                if self.absmax_y is None
                else max(self.absmax_y, value.oracle_residual_absmax_y_px)
            )

    def report(self) -> dict[str, Any]:
        def rate(numerator: float | int, denominator: int) -> float | None:
            return None if denominator == 0 else float(numerator / denominator)

        # Overflow sample rates are conditional on at least one solver-covered
        # pixel.  Fixed-point failures therefore cannot masquerade as cap
        # overflow at either pixel or sample level.
        solvable_samples = self.solver_any_samples

        return {
            "sample_count": self.samples,
            "eval_sample_count": self.eval_samples,
            "eval_pixels": self.eval_pixels,
            "teacher_epe_px": (
                None
                if self.eval_pixels == 0
                else float(self.teacher_epe_sum / self.eval_pixels)
            ),
            "oracle_solver_coverage": rate(self.solver_pixels, self.eval_pixels),
            "oracle_solver_any_sample_rate": rate(
                self.solver_any_samples, self.eval_samples
            ),
            "oracle_solver_full_sample_rate": rate(
                self.solver_full_samples, self.eval_samples
            ),
            "oracle_residual_overflow_given_solvable_x_pixel_rate": rate(
                self.overflow_x_solvable_pixels, self.solver_pixels
            ),
            "oracle_residual_overflow_given_solvable_y_pixel_rate": rate(
                self.overflow_y_solvable_pixels, self.solver_pixels
            ),
            "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": rate(
                self.overflow_any_solvable_pixels, self.solver_pixels
            ),
            "oracle_residual_overflow_given_solvable_x_sample_rate": rate(
                self.overflow_x_solvable_samples, solvable_samples
            ),
            "oracle_residual_overflow_given_solvable_y_sample_rate": rate(
                self.overflow_y_solvable_samples, solvable_samples
            ),
            "oracle_residual_overflow_given_solvable_any_axis_sample_rate": rate(
                self.overflow_any_solvable_samples, solvable_samples
            ),
            "trainable_coverage": rate(self.trainable_pixels, self.eval_pixels),
            "residual_target_valid_rate": rate(
                self.trainable_pixels, self.eval_pixels
            ),
            "oracle_residual_axis_absmax_px": {
                "x": self.absmax_x,
                "y": self.absmax_y,
            },
            "stride_oracle_residual_reconstruction_epe_px": rate(
                self.stride_reconstruction_epe_sum, self.solver_pixels
            ),
            "stride_trainable_oracle_reconstruction_epe_px": rate(
                self.stride_trainable_reconstruction_epe_sum,
                self.trainable_pixels,
            ),
        }


@dataclass(frozen=True)
class _Variant:
    dataset_index: int
    sample_id: str
    mode: str
    rotation_deg: float | None
    rotation_bin_index: int | None
    full_geometry_seed: int | None
    source_homography: Tensor | None
    warped: Tensor
    target_flow: Tensor
    valid: Tensor


def _variants(
    dataset: DocumentFlowDataset,
    indices: Sequence[int],
    rotation_plan: dict[int, list[float]],
    bin_edges: Sequence[float],
    *,
    source_geometry_augment: SourceGeometryAugment,
    full_geometry_plan: dict[int, list[int]],
) -> Iterator[_Variant]:
    for index in indices:
        sample = dataset[int(index)]
        sample_id = str(sample["id"])
        yield _Variant(
            dataset_index=int(index),
            sample_id=sample_id,
            mode="original",
            rotation_deg=0.0,
            rotation_bin_index=None,
            full_geometry_seed=None,
            source_homography=None,
            warped=sample["warped"],
            target_flow=sample["flow"],
            valid=sample["valid"],
        )
        for angle in rotation_plan[int(index)]:
            homography = source_affine_homography(
                sample["warped"].shape[-2:],
                angle_deg=float(angle),
                dtype=sample["warped"].dtype,
                device=sample["warped"].device,
            )
            warped, target_flow, valid = apply_source_homography(
                sample["warped"], sample["flow"], sample["valid"], homography
            )
            yield _Variant(
                dataset_index=int(index),
                sample_id=sample_id,
                mode="rotation_augmented",
                rotation_deg=float(angle),
                rotation_bin_index=_rotation_bin(float(angle), bin_edges),
                full_geometry_seed=None,
                source_homography=None,
                warped=warped,
                target_flow=target_flow,
                valid=valid,
            )
        for full_geometry_seed in full_geometry_plan[int(index)]:
            homography = _sample_deterministic_full_homography(
                source_geometry_augment,
                sample["warped"],
                seed=full_geometry_seed,
            )
            warped, target_flow, valid = apply_source_homography(
                sample["warped"], sample["flow"], sample["valid"], homography
            )
            yield _Variant(
                dataset_index=int(index),
                sample_id=sample_id,
                mode="full_geometry_augmented",
                rotation_deg=None,
                rotation_bin_index=None,
                full_geometry_seed=full_geometry_seed,
                source_homography=homography.detach().cpu(),
                warped=warped,
                target_flow=target_flow,
                valid=valid,
            )


def _batched(values: Iterable[_Variant], batch_size: int) -> Iterator[list[_Variant]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    batch: list[_Variant] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        result.update(
            cuda_device_name=torch.cuda.get_device_name(index),
            cuda_capability=list(torch.cuda.get_device_capability(index)),
            cudnn_version=torch.backends.cudnn.version(),
        )
    return result


def _resolve_configured_path(
    value: str | Path, *, project_root: Path, explicit: bool
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() if explicit else project_root) / path


def run_capacity_preflight(
    *,
    config_path: Path,
    output_path: Path,
    checkpoint_path: Path | None = None,
    teacher_path: Path | None = None,
    manifest_path: Path | None = None,
    split: str = "val",
    sample_count: int | None = 300,
    explicit_rotation_angles: Sequence[float] | None = None,
    rotations_per_sample: int = 1,
    full_geometry_per_sample: int = 0,
    rotation_bin_edges: Sequence[float] = DEFAULT_ROTATION_BIN_EDGES,
    seed: int = 42,
    batch_size: int = 1,
    device: torch.device | str = "cuda:0",
    hash_external_files: bool = True,
    teacher_factory: type[TorchScriptGeometryPrior] = TorchScriptGeometryPrior,
) -> dict[str, Any]:
    """Run the read-only audit and atomically publish its JSON report."""

    started_at = _utc_now()
    started = time.monotonic()
    config_path = config_path.expanduser().resolve(strict=True)
    config = load_config(config_path)
    project_root = config_path.parent.parent
    data_config = config.get("data")
    model_config = config.get("model")
    loss_config = config.get("loss")
    train_config = config.get("train")
    for name, value in (
        ("data", data_config),
        ("model", model_config),
        ("loss", loss_config),
        ("train", train_config),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"config.{name} must be a mapping")
    assert isinstance(data_config, dict)
    assert isinstance(model_config, dict)
    assert isinstance(loss_config, dict)
    assert isinstance(train_config, dict)
    if str(model_config.get("prior_backend", "learned")).lower() != "torchscript":
        raise ValueError("capacity preflight requires model.prior_backend=torchscript")

    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    if manifest_path is None:
        configured_manifest = data_config.get(f"{split}_manifest")
        if not configured_manifest:
            raise ValueError(f"config.data.{split}_manifest is required")
        manifest = _resolve_configured_path(
            str(configured_manifest), project_root=project_root, explicit=False
        )
    else:
        manifest = _resolve_configured_path(
            manifest_path, project_root=project_root, explicit=True
        )
    manifest = manifest.resolve(strict=True)

    if checkpoint_path is None:
        configured_checkpoint = train_config.get("resume")
        if not configured_checkpoint:
            raise ValueError("--checkpoint or config.train.resume is required")
        checkpoint = _resolve_configured_path(
            str(configured_checkpoint), project_root=project_root, explicit=False
        )
    else:
        checkpoint = _resolve_configured_path(
            checkpoint_path, project_root=project_root, explicit=True
        )

    configured_teacher_value = model_config.get("prior_torchscript_path")
    if not configured_teacher_value:
        raise ValueError("model.prior_torchscript_path is required")
    configured_teacher_path = _resolve_configured_path(
        str(configured_teacher_value), project_root=project_root, explicit=False
    )
    if teacher_path is None:
        effective_teacher_path = configured_teacher_path
        teacher_selection_source = "config"
        teacher_override_path: str | None = None
    else:
        effective_teacher_path = _resolve_configured_path(
            teacher_path, project_root=project_root, explicit=True
        )
        teacher_selection_source = "explicit_override"
        teacher_override_path = str(teacher_path.expanduser())

    config_identity = _file_identity(config_path, hash_contents=True)
    manifest_identity = _file_identity(manifest, hash_contents=True)
    checkpoint_identity = _file_identity(
        checkpoint, hash_contents=hash_external_files
    )
    teacher_file_identity = _file_identity(
        effective_teacher_path, hash_contents=hash_external_files
    )

    work_size_value = data_config.get("work_size")
    if not isinstance(work_size_value, (list, tuple)) or len(work_size_value) != 2:
        raise ValueError("config.data.work_size must be [height,width]")
    work_size = (int(work_size_value[0]), int(work_size_value[1]))
    dataset = DocumentFlowDataset(manifest, work_size, augment_guide=False)
    indices = _sample_indices(len(dataset), sample_count)

    raw_augment_config = data_config.get("source_geometry_augment")
    if not isinstance(raw_augment_config, dict):
        raise ValueError("config.data.source_geometry_augment must be a mapping")
    parsed_augment = SourceGeometryAugment.from_config(raw_augment_config)
    rotation_plan, rotation_protocol = build_rotation_plan(
        indices,
        explicit_angles=explicit_rotation_angles,
        max_rotation_deg=parsed_augment.max_rotation_deg,
        rotations_per_sample=int(rotations_per_sample),
        seed=int(seed),
    )
    all_angles = [angle for values in rotation_plan.values() for angle in values]
    maximum_angle = max((abs(value) for value in all_angles), default=0.0)
    bin_edges = _validate_bin_edges(
        rotation_bin_edges, maximum_angle=maximum_angle
    )
    rotation_protocol.update(
        {
            "angle_convention": (
                "positive rotates visually clockwise in image coordinates"
            ),
            "binning": "absolute rotation magnitude; [lower,upper), final bin inclusive",
            "bin_edges_deg": list(bin_edges),
            "configured_augmentation_probability": parsed_augment.probability,
            "rotation_isolated": True,
            "equivalent_to_full_source_geometry_augment": False,
            "scope": (
                "pure source-only rotation; configured scale/translation/perspective "
                "are intentionally excluded to isolate residual rotation capacity"
            ),
        }
    )
    full_geometry_plan, full_geometry_protocol = build_full_geometry_plan(
        indices,
        transformations_per_sample=int(full_geometry_per_sample),
        seed=int(seed),
    )
    full_geometry_enabled = bool(full_geometry_protocol["enabled"])
    full_geometry_protocol.update(
        {
            "conditional_on_augmentation_trigger": full_geometry_enabled,
            "configured_augmentation_probability": parsed_augment.probability,
            "configured_parameters": {
                "max_rotation_deg": parsed_augment.max_rotation_deg,
                "scale": list(parsed_augment.scale),
                "translation": list(parsed_augment.translation),
                "perspective": parsed_augment.perspective,
            },
            "reuses_source_geometry_augment_sample_homography": True,
            "equivalent_to_conditional_full_source_geometry_augment": (
                full_geometry_enabled
            ),
            "equivalent_to_unconditional_training_augmentation": False,
            "scope": (
                "full configured source-only rotation/scale/translation/perspective; "
                "sampled conditional on the augmentation branch being triggered"
            ),
        }
    )

    selected_indices_digest = _sha256_bytes(_canonical_json(indices))
    angle_records = [
        {"dataset_index": index, "angles_deg": rotation_plan[index]}
        for index in indices
    ]
    angle_plan_digest = _sha256_bytes(_canonical_json(angle_records))
    max_residual_px = float(model_config.get("max_residual_px", 24.0))
    configured_target_limit = float(
        loss_config.get("max_residual_target", max_residual_px)
    )
    if configured_target_limit != max_residual_px:
        raise ValueError(
            "model.max_residual_px and loss.max_residual_target must match for a "
            "capacity audit; got "
            f"{max_residual_px} and {configured_target_limit}"
        )
    residual_target_iterations = int(
        loss_config.get("residual_target_iterations", 6)
    )
    max_residual_consistency = float(
        loss_config.get("max_residual_consistency", 1.0)
    )
    max_valid_flow = float(loss_config.get("max_valid_flow", 1000.0))
    feature_stride = int(model_config.get("feature_stride", 8))

    requested_device = torch.device(device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run the teacher audit on a GPU node")
        device_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        torch.cuda.set_device(device_index)
        requested_device = torch.device("cuda", device_index)
    teacher = teacher_factory(
        effective_teacher_path,
        device=requested_device,
        input_size=int(model_config.get("prior_torchscript_size", 512)),
        flow_size=int(
            model_config.get(
                "prior_torchscript_flow_size",
                model_config.get("prior_torchscript_size", 512),
            )
        ),
        blur_kernel=int(model_config.get("prior_torchscript_blur_kernel", 39)),
        autocast_dtype=str(
            model_config.get("prior_torchscript_autocast_dtype", "float16")
        ),
        requires_logical_cuda0=bool(
            model_config.get("prior_torchscript_requires_logical_cuda0", False)
        ),
    ).eval()
    if not isinstance(teacher, nn.Module):
        raise TypeError("teacher_factory must return an nn.Module")

    original = _Accumulator()
    augmented = _Accumulator()
    full_geometry_augmented = _Accumulator()
    bins = [_Accumulator() for _ in range(len(bin_edges) - 1)]
    sample_reports: list[dict[str, Any]] = []
    variant_iterator = _variants(
        dataset,
        indices,
        rotation_plan,
        bin_edges,
        source_geometry_augment=parsed_augment,
        full_geometry_plan=full_geometry_plan,
    )
    for variant_batch in _batched(variant_iterator, int(batch_size)):
        warped = torch.stack([value.warped for value in variant_batch]).to(
            requested_device, non_blocking=False
        )
        target_flow = torch.stack(
            [value.target_flow for value in variant_batch]
        ).to(requested_device, non_blocking=False)
        valid = torch.stack([value.valid for value in variant_batch]).to(
            requested_device, non_blocking=False
        )
        with torch.inference_mode():
            teacher_flow = teacher(warped)
        batch_statistics = capacity_sample_statistics(
            teacher_flow,
            target_flow,
            valid,
            max_residual_px=max_residual_px,
            residual_target_iterations=residual_target_iterations,
            max_residual_consistency=max_residual_consistency,
            max_valid_flow=max_valid_flow,
            feature_stride=feature_stride,
        )
        for variant, statistics in zip(variant_batch, batch_statistics):
            if variant.mode == "original":
                original.add(statistics)
            elif variant.mode == "rotation_augmented":
                augmented.add(statistics)
                assert variant.rotation_bin_index is not None
                bins[variant.rotation_bin_index].add(statistics)
            elif variant.mode == "full_geometry_augmented":
                full_geometry_augmented.add(statistics)
            else:  # pragma: no cover - _variants owns this closed schema
                raise RuntimeError(f"unknown capacity variant mode: {variant.mode}")
            sample_reports.append(
                {
                    "dataset_index": variant.dataset_index,
                    "id": variant.sample_id,
                    "mode": variant.mode,
                    "rotation_deg": variant.rotation_deg,
                    "absolute_rotation_deg": (
                        None
                        if variant.rotation_deg is None
                        else abs(variant.rotation_deg)
                    ),
                    "rotation_bin_index": variant.rotation_bin_index,
                    "full_geometry_seed": variant.full_geometry_seed,
                    "source_homography": (
                        None
                        if variant.source_homography is None
                        else variant.source_homography.tolist()
                    ),
                    "metrics": statistics.as_report(),
                }
            )

    # The teacher, manifest, and config govern every measured number.  Abort
    # instead of publishing a report if any path changed while the audit ran.
    _assert_path_matches_identity(config_identity)
    _assert_path_matches_identity(manifest_identity)
    _assert_path_matches_identity(checkpoint_identity)
    _assert_path_matches_identity(teacher_file_identity)

    completed_at = _utc_now()
    report = {
        "report_version": REPORT_VERSION,
        "kind": "frozen_teacher_residual_capacity_preflight",
        "identities": {
            "config": config_identity,
            "checkpoint": {
                **checkpoint_identity,
                "role": "migration provenance only; checkpoint payload was not loaded",
            },
            "teacher": {
                "backend": "torchscript",
                "selection": {
                    "source": teacher_selection_source,
                    "config_value": str(configured_teacher_value),
                    "config_resolved_path": str(
                        configured_teacher_path.expanduser().absolute()
                    ),
                    "override_value": teacher_override_path,
                    "effective_path": str(
                        effective_teacher_path.expanduser().absolute()
                    ),
                },
                "checkpoint": teacher_file_identity,
                "input_size": int(model_config.get("prior_torchscript_size", 512)),
                "flow_size": int(
                    model_config.get(
                        "prior_torchscript_flow_size",
                        model_config.get("prior_torchscript_size", 512),
                    )
                ),
                "blur_kernel": int(
                    model_config.get("prior_torchscript_blur_kernel", 39)
                ),
                "autocast_dtype": str(
                    model_config.get("prior_torchscript_autocast_dtype", "float16")
                ),
                "requires_logical_cuda0": bool(
                    model_config.get(
                        "prior_torchscript_requires_logical_cuda0", False
                    )
                ),
            },
            "manifest": {
                **manifest_identity,
                "split": split,
                "record_count": len(dataset),
            },
        },
        "protocol": {
            "work_size": list(work_size),
            "requested_sample_count": sample_count,
            "selected_sample_count": len(indices),
            "selected_indices": indices,
            "selected_indices_sha256": selected_indices_digest,
            "rotation_plan_sha256": angle_plan_digest,
            "source_rotation": rotation_protocol,
            "source_full_geometry": full_geometry_protocol,
            "batch_size": int(batch_size),
            "feature_stride": feature_stride,
            "max_residual_px": max_residual_px,
            "max_residual_target": configured_target_limit,
            "oracle_solver_coverage_definition": (
                "m_eval and finite oracle residual and residual flow in bounds and "
                "finite composition consistency <= max_residual_consistency"
            ),
            "overflow_given_solvable_definition": (
                "within oracle solver coverage, "
                "abs(oracle_residual_axis)>max_residual_px; any-axis is channel L-inf"
            ),
            "residual_target_iterations": residual_target_iterations,
            "max_residual_consistency": max_residual_consistency,
            "max_valid_flow": max_valid_flow,
            "residual_target_valid_definition": (
                "training-identical: GT valid+finite+magnitude, residual map in canvas, "
                "within max_residual_px on both axes, consistency within threshold"
            ),
            "external_file_sha256_enabled": bool(hash_external_files),
        },
        "runtime": {
            **_runtime_identity(requested_device),
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_seconds": float(time.monotonic() - started),
        },
        "results": {
            "original": original.report(),
            "rotation_augmented": augmented.report(),
            "full_geometry_augmented": full_geometry_augmented.report(),
            "rotation_bins": [
                {
                    "index": index,
                    "absolute_rotation_deg": {
                        "lower_inclusive": bin_edges[index],
                        "upper": bin_edges[index + 1],
                        "upper_inclusive": index == len(bin_edges) - 2,
                    },
                    "metrics": accumulator.report(),
                }
                for index, accumulator in enumerate(bins)
            ],
            "samples": sample_reports,
        },
    }
    _atomic_write_json(output_path.expanduser().absolute(), report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unified_v3_3_teacher_anchor.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--teacher",
        type=Path,
        help=(
            "TorchScript teacher override; relative paths resolve from the current "
            "directory, while the configured teacher remains recorded in provenance"
        ),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=300,
        help="evenly spaced manifest samples; 0 means the complete manifest",
    )
    parser.add_argument(
        "--rotation-angles",
        type=float,
        nargs="+",
        help="explicit angles applied to every sample; default samples config range",
    )
    parser.add_argument("--rotations-per-sample", type=int, default=1)
    parser.add_argument(
        "--full-geometry-per-sample",
        type=int,
        default=0,
        help=(
            "deterministic full source-geometry transforms per selected sample; "
            "0 preserves the rotation-only diagnostic"
        ),
    )
    parser.add_argument(
        "--rotation-bin-edges",
        type=float,
        nargs="+",
        default=list(DEFAULT_ROTATION_BIN_EDGES),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--fast-external-identity",
        action="store_true",
        help="record stat identity but skip SHA-256 for the large teacher/checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/preflight_v33/teacher_capacity.json"),
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, int(args.threads)))
    report = run_capacity_preflight(
        config_path=args.config,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        teacher_path=args.teacher,
        manifest_path=args.manifest,
        split=args.split,
        sample_count=args.sample_count,
        explicit_rotation_angles=args.rotation_angles,
        rotations_per_sample=args.rotations_per_sample,
        full_geometry_per_sample=args.full_geometry_per_sample,
        rotation_bin_edges=args.rotation_bin_edges,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        hash_external_files=not args.fast_external_identity,
    )
    summary = {
        "output": str(args.output.expanduser().absolute()),
        "original": report["results"]["original"],
        "rotation_augmented": report["results"]["rotation_augmented"],
        "full_geometry_augmented": report["results"][
            "full_geometry_augmented"
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
