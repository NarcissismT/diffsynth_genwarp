"""Diagnostic upper bound for an oracle quarter-turn teacher router.

This module is deliberately isolated from the production teacher-capacity
approval path.  It evaluates only deterministic, rotation-only source
variants, uses the known injected angle to undo the nearest quarter turn, and
then measures the remaining frozen-teacher residual with the same capacity
statistics as training.

The resulting report is diagnostic evidence, not production evidence.  Its
kind is intentionally incompatible with :mod:`teacher_capacity_policy`, and
the writer rejects both the production capacity directory and
``approved.json`` as output targets.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor, nn

from .config import load_config
from .data import (
    DocumentFlowDataset,
    SourceGeometryAugment,
    apply_source_homography,
    source_affine_homography,
)
from .geometry import backward_flow_to_map, map_to_backward_flow
from .models.teacher_prior import TorchScriptGeometryPrior
from .teacher_capacity_preflight import (
    DEFAULT_ROTATION_BIN_EDGES,
    _Accumulator,
    _assert_path_matches_identity,
    _atomic_write_json,
    _canonical_json,
    _file_identity,
    _resolve_configured_path,
    _rotation_bin,
    _runtime_identity,
    _sample_deterministic_full_homography,
    _sample_indices,
    build_full_geometry_plan,
    build_rotation_plan,
    capacity_sample_statistics,
)


REPORT_VERSION = 1
REPORT_KIND = "teacher_quarter_turn_oracle_diagnostic"
CANONICAL_FRAME_REPORT_VERSION = 2
CANONICAL_FRAME_REPORT_KIND = (
    "teacher_quarter_turn_oracle_canonical_frame_diagnostic"
)
SOLVER_SWEEP_REPORT_VERSION = 3
SOLVER_SWEEP_REPORT_KIND = (
    "teacher_quarter_turn_oracle_canonical_solver_sweep_diagnostic"
)
FULL_GEOMETRY_REPORT_VERSION = 4
FULL_GEOMETRY_REPORT_KIND = (
    "teacher_quarter_turn_oracle_canonical_full_geometry_diagnostic"
)
FULL_GEOMETRY_GRID_REPORT_VERSION = 5
FULL_GEOMETRY_GRID_REPORT_KIND = (
    "teacher_quarter_turn_oracle_canonical_full_geometry_capacity_grid_diagnostic"
)
BEST_OF_C4_REPORT_VERSION = 6
BEST_OF_C4_REPORT_KIND = (
    "teacher_quarter_turn_oracle_canonical_full_geometry_best_of_c4_diagnostic"
)
DEFAULT_SOLVER_ITERATION_SWEEP = (6, 12, 24)
RESIDUAL_ROTATION_BIN_EDGES = (0.0, 15.0, 30.0, 40.0, 45.0)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_QUARTER_TURNS = (0, -90, 90, 180)


def _validate_angle(angle_deg: float) -> float:
    angle = float(angle_deg)
    if not math.isfinite(angle) or abs(angle) > 180.0:
        raise ValueError(
            f"rotation angle must be finite and in [-180,180], got {angle_deg}"
        )
    return angle


def wrap_rotation_degrees(angle_deg: float) -> float:
    """Wrap one angle to ``[-180, 180)`` with an exact zero when possible."""

    angle = float(angle_deg)
    if not math.isfinite(angle):
        raise ValueError(f"rotation angle must be finite, got {angle_deg}")
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 0.0 if abs(wrapped) <= 1.0e-12 else wrapped


def oracle_quarter_turn_degrees(angle_deg: float) -> int:
    """Return the nearest inverse quarter turn for a known injected angle.

    Angles are normalized to ``(-180, 180]`` and exact half-way cases follow a
    frozen half-open residual convention: ``[-45, 45)``.  Positive angles
    follow the project's image-coordinate convention (visual clockwise
    rotation), so the returned canonicalization has the opposite sign.  Both
    signs of 180 degrees use the single canonical value ``180``.
    """

    angle = wrap_rotation_degrees(_validate_angle(angle_deg))
    if abs(angle + 180.0) <= 1.0e-12:
        angle = 180.0
    turns = int(math.floor((angle + 45.0) / 90.0))
    result = -90 * turns
    if abs(result) == 180:
        result = 180
    if result not in _QUARTER_TURNS:
        raise RuntimeError(f"invalid oracle quarter turn {result} for angle {angle}")
    residual = wrap_rotation_degrees(angle + result)
    if residual < -45.0 - 1.0e-9 or residual >= 45.0:
        raise RuntimeError(
            f"quarter-turn routing left an invalid residual angle {residual}"
        )
    return result


def rotate_source_by_quarter_turn(source: Tensor, quarter_turn_deg: int) -> Tensor:
    """Apply an exact square-canvas quarter turn without interpolation."""

    quarter_turn = int(quarter_turn_deg)
    if quarter_turn not in _QUARTER_TURNS:
        raise ValueError(
            f"quarter_turn_deg must be one of {_QUARTER_TURNS}, got {quarter_turn_deg}"
        )
    if source.ndim != 3:
        raise ValueError(f"source must be [C,H,W], got {tuple(source.shape)}")
    if source.shape[-2] != source.shape[-1]:
        raise ValueError(
            "oracle quarter-turn diagnostic requires a square source canvas, got "
            f"{tuple(source.shape[-2:])}"
        )
    return torch.rot90(source, k=-(quarter_turn // 90), dims=(-2, -1))


def restore_teacher_flow_from_quarter_turn(
    canonical_flow: Tensor,
    quarter_turn_degrees: Sequence[int],
) -> Tensor:
    """Map teacher source coordinates back before the oracle quarter turn.

    ``canonical_flow`` maps target pixels into the quarter-turned source.  A
    final rectifier must sample the original rotated source, so its absolute
    source map is transformed by the inverse quarter turn.  Rotating the flow
    tensor spatially, or rotating only its vector channels, would violate the
    backward-flow coordinate contract.
    """

    inverse_turns = [
        180 if abs(int(value)) == 180 else -int(value)
        for value in quarter_turn_degrees
    ]
    return transform_backward_flow_source_map_by_quarter_turn(
        canonical_flow,
        inverse_turns,
    )


def transform_backward_flow_source_map_by_quarter_turn(
    flow: Tensor,
    quarter_turn_degrees: Sequence[int],
) -> Tensor:
    """Apply a quarter turn to each absolute source-map value.

    The target grid is unchanged.  For ``M(x)=x+F(x)``, this returns the flow
    whose absolute source map is ``Q M(x)``.  It deliberately does not rotate
    the flow tensor spatially and does not rotate displacement channels in
    isolation.  Positive angles follow :func:`source_affine_homography`.
    """

    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(
            "flow must be [B,2,H,W], got "
            f"{tuple(flow.shape)}"
        )
    if flow.shape[-2] != flow.shape[-1]:
        raise ValueError(
            "oracle quarter-turn diagnostic requires square flow canvases, got "
            f"{tuple(flow.shape[-2:])}"
        )
    turns = [int(value) for value in quarter_turn_degrees]
    if len(turns) != flow.shape[0]:
        raise ValueError(
            "quarter-turn count must equal flow batch size, got "
            f"{len(turns)} and {flow.shape[0]}"
        )
    if any(value not in _QUARTER_TURNS for value in turns):
        raise ValueError(
            f"quarter turns must be drawn from {_QUARTER_TURNS}, got {turns}"
        )

    source_map = backward_flow_to_map(flow)
    transformed_map = torch.empty_like(source_map)
    edge = float(flow.shape[-1] - 1)
    for index, quarter_turn in enumerate(turns):
        x = source_map[index, 0]
        y = source_map[index, 1]
        if quarter_turn == 0:
            source_x, source_y = x, y
        elif quarter_turn == 90:
            source_x, source_y = edge - y, x
        elif quarter_turn == -90:
            source_x, source_y = y, edge - x
        else:  # 180 degrees is its own inverse.
            source_x, source_y = edge - x, edge - y
        transformed_map[index, 0] = source_x
        transformed_map[index, 1] = source_y
    transformed_flow = map_to_backward_flow(transformed_map)
    # Preserve the q=0 control bit-for-bit instead of introducing an avoidable
    # add/subtract round trip through absolute coordinates.
    for index, quarter_turn in enumerate(turns):
        if quarter_turn == 0:
            transformed_flow[index].copy_(flow[index])
    return transformed_flow


@dataclass(frozen=True)
class _OracleVariant:
    dataset_index: int
    sample_id: str
    injected_rotation_deg: float
    rotation_bin_index: int
    oracle_quarter_turn_deg: int
    residual_rotation_deg: float
    canonicalized_warped: Tensor
    target: Tensor
    target_flow: Tensor
    valid: Tensor
    full_geometry_seed: int | None = None
    source_homography: Tensor | None = None


def _oracle_variants(
    dataset: DocumentFlowDataset,
    indices: Sequence[int],
    rotation_plan: dict[int, list[float]],
    bin_edges: Sequence[float],
) -> Iterator[_OracleVariant]:
    for index in indices:
        sample = dataset[int(index)]
        sample_id = str(sample["id"])
        for angle_value in rotation_plan[int(index)]:
            angle = float(angle_value)
            homography = source_affine_homography(
                sample["warped"].shape[-2:],
                angle_deg=angle,
                dtype=sample["warped"].dtype,
                device=sample["warped"].device,
            )
            warped, target_flow, valid = apply_source_homography(
                sample["warped"], sample["flow"], sample["valid"], homography
            )
            quarter_turn = oracle_quarter_turn_degrees(angle)
            residual_angle = wrap_rotation_degrees(angle + quarter_turn)
            yield _OracleVariant(
                dataset_index=int(index),
                sample_id=sample_id,
                injected_rotation_deg=angle,
                rotation_bin_index=_rotation_bin(angle, bin_edges),
                oracle_quarter_turn_deg=quarter_turn,
                residual_rotation_deg=residual_angle,
                canonicalized_warped=rotate_source_by_quarter_turn(
                    warped, quarter_turn
                ),
                target=sample["target"],
                target_flow=target_flow,
                valid=valid,
            )


def _full_geometry_rotation_from_seed(
    augment: SourceGeometryAugment,
    *,
    seed: int,
) -> float:
    """Recover the exact affine rotation drawn first by the training sampler."""

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.default_generator.manual_seed(int(seed))
        return float(augment._sample_symmetric(augment.max_rotation_deg))


def _full_geometry_oracle_variants(
    dataset: DocumentFlowDataset,
    indices: Sequence[int],
    full_geometry_plan: dict[int, list[int]],
    augment: SourceGeometryAugment,
    bin_edges: Sequence[float],
) -> Iterator[_OracleVariant]:
    """Yield formal full-geometry samples with known-angle C4 routing."""

    for index in indices:
        sample = dataset[int(index)]
        sample_id = str(sample["id"])
        for full_geometry_seed in full_geometry_plan[int(index)]:
            homography = _sample_deterministic_full_homography(
                augment,
                sample["warped"],
                seed=full_geometry_seed,
            )
            angle = _full_geometry_rotation_from_seed(
                augment,
                seed=full_geometry_seed,
            )
            warped, target_flow, valid = apply_source_homography(
                sample["warped"],
                sample["flow"],
                sample["valid"],
                homography,
            )
            quarter_turn = oracle_quarter_turn_degrees(angle)
            residual_angle = wrap_rotation_degrees(angle + quarter_turn)
            yield _OracleVariant(
                dataset_index=int(index),
                sample_id=sample_id,
                injected_rotation_deg=angle,
                rotation_bin_index=_rotation_bin(angle, bin_edges),
                oracle_quarter_turn_deg=quarter_turn,
                residual_rotation_deg=residual_angle,
                canonicalized_warped=rotate_source_by_quarter_turn(
                    warped,
                    quarter_turn,
                ),
                target=sample["target"],
                target_flow=target_flow,
                valid=valid,
                full_geometry_seed=int(full_geometry_seed),
                source_homography=homography.detach().cpu(),
            )


def _batched(
    values: Iterator[_OracleVariant], batch_size: int
) -> Iterator[list[_OracleVariant]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    batch: list[_OracleVariant] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _validate_output_path(path: Path) -> Path:
    output = path.expanduser().absolute()
    if output.name == "approved.json":
        raise ValueError("a diagnostic may not write production approved.json")
    if "preflight_v33_teacher_capacity" in output.parts:
        raise ValueError(
            "quarter-turn diagnostic output must not use the production capacity directory"
        )
    return output


def _validate_expected_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            "expected_teacher_sha256 must be a canonical lowercase SHA-256"
        )
    return value


def _validate_solver_iteration_sweep(
    *,
    configured_iterations: int,
    requested_iterations: Sequence[int] | None,
    canonical_frame_v2: bool,
) -> tuple[int, ...]:
    """Validate an opt-in solver sweep without changing legacy protocols."""

    baseline = int(configured_iterations)
    if baseline < 1:
        raise ValueError("loss.residual_target_iterations must be at least one")
    if requested_iterations is None:
        return (baseline,)
    if not canonical_frame_v2:
        raise ValueError(
            "residual target iteration sweep requires canonical_frame_v2"
        )
    raw = tuple(requested_iterations)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in raw
    ):
        raise ValueError(
            "residual target iteration sweep must be a non-empty sequence of "
            f"integers, got {raw}"
        )
    result = tuple(int(value) for value in raw)
    if (
        not result
        or any(value < 1 for value in result)
        or len(set(result)) != len(result)
        or tuple(sorted(result)) != result
    ):
        raise ValueError(
            "residual target iteration sweep must be a non-empty, unique, "
            f"strictly increasing sequence of positive integers, got {result}"
        )
    if baseline not in result:
        raise ValueError(
            "residual target iteration sweep must include the configured "
            f"baseline {baseline}, got {result}"
        )
    return result


def run_quarter_turn_oracle_diagnostic(
    *,
    config_path: Path,
    checkpoint_path: Path,
    teacher_path: Path,
    expected_teacher_sha256: str,
    output_path: Path,
    manifest_path: Path | None = None,
    split: str = "val",
    sample_count: int | None = 300,
    explicit_rotation_angles: Sequence[float] | None = None,
    rotations_per_sample: int = 1,
    rotation_bin_edges: Sequence[float] = DEFAULT_ROTATION_BIN_EDGES,
    seed: int = 42,
    batch_size: int = 1,
    device: torch.device | str = "cuda:0",
    teacher_factory: type[TorchScriptGeometryPrior] = TorchScriptGeometryPrior,
    canonical_frame_v2: bool = False,
    residual_target_iteration_sweep: Sequence[int] | None = None,
    full_geometry_per_sample: int = 0,
    residual_target_iterations_override: int | None = None,
    full_geometry_solver_iteration_sweep: Sequence[int] | None = None,
    full_geometry_residual_cap_sweep: Sequence[float] | None = None,
    full_geometry_best_of_c4: bool = False,
    c4_candidate_batch_size: int = 1,
) -> dict[str, Any]:
    """Run the known-angle, rotation-only oracle diagnostic.

    ``canonical_frame_v2=False`` preserves the job-78932 mapped-back v1
    report.  The opt-in v2 protocol additionally evaluates residual capacity
    before the final inverse quarter turn, where the fixed-point target solver
    remains a contraction for residual rotations below 45 degrees.  Supplying
    ``residual_target_iteration_sweep`` creates an isolated v3 report that
    reuses each teacher forward and varies only the offline solver iterations.
    ``full_geometry_per_sample>0`` instead creates a v4 upper-bound report over
    the formal rotation/scale/translation/perspective sampler.  Supplying both
    full-geometry sweep arguments creates a v5 solver/capacity grid while still
    reusing each canonical teacher forward.  ``full_geometry_best_of_c4=True``
    creates an isolated v6 boundary audit: every full-geometry source is sent
    through all four exact C4 candidates, and GT flow is used only after the
    forwards to compare the nearest-angle label with teacher-EPE and
    capacity-aware best-of-four upper bounds.
    """

    started_at = time.time()
    started_monotonic = time.monotonic()
    output = _validate_output_path(output_path)
    expected_teacher_sha256 = _validate_expected_sha256(expected_teacher_sha256)
    config_path = config_path.expanduser().resolve(strict=True)
    config = load_config(config_path)
    project_root = config_path.parent.parent
    data_config = config.get("data")
    model_config = config.get("model")
    loss_config = config.get("loss")
    if not isinstance(data_config, dict):
        raise ValueError("config.data must be a mapping")
    if not isinstance(model_config, dict):
        raise ValueError("config.model must be a mapping")
    if not isinstance(loss_config, dict):
        raise ValueError("config.loss must be a mapping")
    if str(model_config.get("prior_backend", "learned")).lower() != "torchscript":
        raise ValueError("quarter-turn diagnostic requires prior_backend=torchscript")
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    configured_manifest = data_config.get(f"{split}_manifest")
    if manifest_path is None:
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
    checkpoint = _resolve_configured_path(
        checkpoint_path, project_root=project_root, explicit=True
    ).resolve(strict=True)
    teacher_path = _resolve_configured_path(
        teacher_path, project_root=project_root, explicit=True
    ).resolve(strict=True)

    config_identity = _file_identity(config_path, hash_contents=True)
    manifest_identity = _file_identity(manifest, hash_contents=True)
    checkpoint_identity = _file_identity(checkpoint, hash_contents=True)
    teacher_identity = _file_identity(teacher_path, hash_contents=True)
    actual_teacher_sha256 = teacher_identity.get("sha256")
    if not isinstance(actual_teacher_sha256, str) or not hmac.compare_digest(
        actual_teacher_sha256, expected_teacher_sha256
    ):
        raise ValueError(
            "teacher SHA-256 mismatch: "
            f"expected={expected_teacher_sha256} actual={actual_teacher_sha256}"
        )

    work_size_value = data_config.get("work_size")
    if not isinstance(work_size_value, (list, tuple)) or len(work_size_value) != 2:
        raise ValueError("config.data.work_size must be [height,width]")
    work_size = (int(work_size_value[0]), int(work_size_value[1]))
    if work_size[0] != work_size[1]:
        raise ValueError(
            f"quarter-turn diagnostic requires square work_size, got {work_size}"
        )
    dataset = DocumentFlowDataset(manifest, work_size, augment_guide=False)
    indices = _sample_indices(len(dataset), sample_count)

    raw_augment_config = data_config.get("source_geometry_augment")
    if not isinstance(raw_augment_config, dict):
        raise ValueError("config.data.source_geometry_augment must be a mapping")
    parsed_augment = SourceGeometryAugment.from_config(raw_augment_config)
    if isinstance(full_geometry_per_sample, bool) or not isinstance(
        full_geometry_per_sample, Integral
    ):
        raise ValueError("full_geometry_per_sample must be a non-negative integer")
    full_geometry_per_sample = int(full_geometry_per_sample)
    if full_geometry_per_sample < 0:
        raise ValueError("full_geometry_per_sample must be a non-negative integer")
    full_geometry_grid_enabled = (
        full_geometry_solver_iteration_sweep is not None
        or full_geometry_residual_cap_sweep is not None
    )
    if not isinstance(full_geometry_best_of_c4, bool):
        raise ValueError("full_geometry_best_of_c4 must be boolean")
    if isinstance(c4_candidate_batch_size, bool) or not isinstance(
        c4_candidate_batch_size, Integral
    ):
        raise ValueError("c4_candidate_batch_size must be a positive integer")
    c4_candidate_batch_size = int(c4_candidate_batch_size)
    if c4_candidate_batch_size < 1:
        raise ValueError("c4_candidate_batch_size must be a positive integer")
    if (full_geometry_solver_iteration_sweep is None) != (
        full_geometry_residual_cap_sweep is None
    ):
        raise ValueError(
            "full geometry capacity grid requires both solver iteration and "
            "residual cap sweeps"
        )
    full_geometry_enabled = full_geometry_per_sample > 0
    if full_geometry_best_of_c4 and not full_geometry_grid_enabled:
        raise ValueError(
            "full geometry best-of-C4 requires the full geometry capacity grid"
        )
    if full_geometry_enabled:
        if not canonical_frame_v2:
            raise ValueError("full geometry oracle requires canonical_frame_v2")
        if residual_target_iteration_sweep is not None:
            raise ValueError("full geometry oracle cannot be combined with solver sweep")
        if explicit_rotation_angles is not None:
            raise ValueError(
                "full geometry oracle samples rotation from the formal augmentation; "
                "explicit_rotation_angles is not allowed"
            )
        rotation_plan: dict[int, list[float]] = {}
        rotation_protocol: dict[str, Any] | None = None
        full_geometry_plan, full_geometry_protocol = build_full_geometry_plan(
            indices,
            transformations_per_sample=full_geometry_per_sample,
            seed=int(seed),
        )
        full_geometry_protocol.update(
            {
                "conditional_on_augmentation_trigger": True,
                "configured_augmentation_probability": parsed_augment.probability,
                "configured_parameters": {
                    "max_rotation_deg": parsed_augment.max_rotation_deg,
                    "scale": list(parsed_augment.scale),
                    "translation": list(parsed_augment.translation),
                    "perspective": parsed_augment.perspective,
                },
                "reuses_source_geometry_augment_sample_homography": True,
                "equivalent_to_conditional_full_source_geometry_augment": True,
                "equivalent_to_unconditional_training_augmentation": False,
                "oracle_rotation_metadata": (
                    "exact first RNG draw used by sample_homography; unavailable "
                    "to a deployable router"
                ),
                "scope": (
                    "full configured source-only rotation/scale/translation/"
                    "perspective; sampled conditional on augmentation trigger"
                ),
            }
        )
        maximum_angle = parsed_augment.max_rotation_deg
    else:
        if full_geometry_grid_enabled:
            raise ValueError("full geometry capacity grid requires full geometry")
        if residual_target_iterations_override is not None:
            raise ValueError(
                "residual_target_iterations_override is restricted to full geometry"
            )
        rotation_plan, rotation_protocol = build_rotation_plan(
            indices,
            explicit_angles=explicit_rotation_angles,
            max_rotation_deg=parsed_augment.max_rotation_deg,
            rotations_per_sample=int(rotations_per_sample),
            seed=int(seed),
        )
        full_geometry_plan = {int(index): [] for index in indices}
        full_geometry_protocol = None
        all_angles = [angle for values in rotation_plan.values() for angle in values]
        maximum_angle = max((abs(value) for value in all_angles), default=0.0)
    bin_edges = tuple(float(value) for value in rotation_bin_edges)
    if (
        len(bin_edges) < 2
        or bin_edges[0] != 0.0
        or any(not math.isfinite(value) for value in bin_edges)
        or any(right <= left for left, right in zip(bin_edges, bin_edges[1:]))
        or bin_edges[-1] + 1.0e-9 < maximum_angle
    ):
        raise ValueError(
            f"rotation bin edges {bin_edges} do not cover maximum angle {maximum_angle}"
        )

    max_residual_px = float(model_config.get("max_residual_px", 24.0))
    max_residual_target = float(
        loss_config.get("max_residual_target", max_residual_px)
    )
    if max_residual_target != max_residual_px:
        raise ValueError(
            "model.max_residual_px and loss.max_residual_target must match"
        )
    configured_residual_target_iterations = int(
        loss_config.get("residual_target_iterations", 6)
    )
    if residual_target_iterations_override is None:
        residual_target_iterations = configured_residual_target_iterations
    else:
        if isinstance(residual_target_iterations_override, bool) or not isinstance(
            residual_target_iterations_override, Integral
        ):
            raise ValueError(
                "residual_target_iterations_override must be a positive integer"
            )
        residual_target_iterations = int(residual_target_iterations_override)
        if residual_target_iterations < 1:
            raise ValueError(
                "residual_target_iterations_override must be a positive integer"
            )
    solver_iterations = _validate_solver_iteration_sweep(
        configured_iterations=residual_target_iterations,
        requested_iterations=residual_target_iteration_sweep,
        canonical_frame_v2=canonical_frame_v2,
    )
    if full_geometry_grid_enabled:
        assert full_geometry_solver_iteration_sweep is not None
        assert full_geometry_residual_cap_sweep is not None
        full_geometry_grid_iterations = _validate_solver_iteration_sweep(
            configured_iterations=residual_target_iterations,
            requested_iterations=full_geometry_solver_iteration_sweep,
            canonical_frame_v2=canonical_frame_v2,
        )
        raw_caps = tuple(full_geometry_residual_cap_sweep)
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in raw_caps
        ):
            raise ValueError(
                "full geometry residual cap sweep must be a sequence of numbers"
            )
        full_geometry_grid_caps = tuple(float(value) for value in raw_caps)
        if (
            not full_geometry_grid_caps
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in full_geometry_grid_caps
            )
            or len(set(full_geometry_grid_caps)) != len(full_geometry_grid_caps)
            or tuple(sorted(full_geometry_grid_caps)) != full_geometry_grid_caps
        ):
            raise ValueError(
                "full geometry residual cap sweep must be a non-empty, unique, "
                "strictly increasing sequence of positive finite numbers"
            )
        if max_residual_px not in full_geometry_grid_caps:
            raise ValueError(
                "full geometry residual cap sweep must include the configured "
                f"baseline {max_residual_px}"
            )
    else:
        full_geometry_grid_iterations = ()
        full_geometry_grid_caps = ()
    full_geometry_grid_cells = tuple(
        (iterations, cap)
        for iterations in full_geometry_grid_iterations
        for cap in full_geometry_grid_caps
    )
    max_residual_consistency = float(
        loss_config.get("max_residual_consistency", 1.0)
    )
    max_valid_flow = float(loss_config.get("max_valid_flow", 1000.0))
    feature_stride = int(model_config.get("feature_stride", 8))

    requested_device = torch.device(device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run this diagnostic on a GPU node")
        device_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        torch.cuda.set_device(device_index)
        requested_device = torch.device("cuda", device_index)

    teacher = teacher_factory(
        teacher_path,
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
        expected_sha256=expected_teacher_sha256,
    ).eval()
    if not isinstance(teacher, nn.Module):
        raise TypeError("teacher_factory must return an nn.Module")

    canonical_aggregate = _Accumulator()
    mapped_back_aggregate = _Accumulator()
    canonical_bins = [_Accumulator() for _ in range(len(bin_edges) - 1)]
    mapped_back_bins = [_Accumulator() for _ in range(len(bin_edges) - 1)]
    canonical_quarters = {value: _Accumulator() for value in _QUARTER_TURNS}
    mapped_back_quarters = {value: _Accumulator() for value in _QUARTER_TURNS}
    canonical_iteration_aggregates = {
        iterations: (
            canonical_aggregate
            if iterations == residual_target_iterations
            else _Accumulator()
        )
        for iterations in solver_iterations
    }
    canonical_iteration_bins = {
        iterations: (
            canonical_bins
            if iterations == residual_target_iterations
            else [_Accumulator() for _ in range(len(bin_edges) - 1)]
        )
        for iterations in solver_iterations
    }
    canonical_iteration_residual_bins = {
        iterations: [
            _Accumulator() for _ in range(len(RESIDUAL_ROTATION_BIN_EDGES) - 1)
        ]
        for iterations in solver_iterations
    }
    canonical_iteration_quarters = {
        iterations: (
            canonical_quarters
            if iterations == residual_target_iterations
            else {value: _Accumulator() for value in _QUARTER_TURNS}
        )
        for iterations in solver_iterations
    }
    full_geometry_grid_aggregates = {
        cell: _Accumulator() for cell in full_geometry_grid_cells
    }
    full_geometry_grid_bins = {
        cell: [_Accumulator() for _ in range(len(bin_edges) - 1)]
        for cell in full_geometry_grid_cells
    }
    full_geometry_grid_residual_bins = {
        cell: [
            _Accumulator() for _ in range(len(RESIDUAL_ROTATION_BIN_EDGES) - 1)
        ]
        for cell in full_geometry_grid_cells
    }
    full_geometry_grid_quarters = {
        cell: {value: _Accumulator() for value in _QUARTER_TURNS}
        for cell in full_geometry_grid_cells
    }
    # v6 keeps the v5 nearest-angle grid as a bit-for-bit baseline, then adds
    # two GT-only upper bounds over the same four teacher forwards.  The first
    # picks the smallest teacher EPE; the second maximizes trainable pixels for
    # each grid cell and uses solver pixels/EPE only as deterministic
    # tie-breakers.  Neither selector is deployable.
    best_epe_grid_aggregates = {
        cell: _Accumulator() for cell in full_geometry_grid_cells
    }
    best_epe_grid_bins = {
        cell: [_Accumulator() for _ in range(len(bin_edges) - 1)]
        for cell in full_geometry_grid_cells
    }
    best_epe_grid_residual_bins = {
        cell: [
            _Accumulator() for _ in range(len(RESIDUAL_ROTATION_BIN_EDGES) - 1)
        ]
        for cell in full_geometry_grid_cells
    }
    best_capacity_grid_aggregates = {
        cell: _Accumulator() for cell in full_geometry_grid_cells
    }
    best_capacity_grid_bins = {
        cell: [_Accumulator() for _ in range(len(bin_edges) - 1)]
        for cell in full_geometry_grid_cells
    }
    best_capacity_grid_residual_bins = {
        cell: [
            _Accumulator() for _ in range(len(RESIDUAL_ROTATION_BIN_EDGES) - 1)
        ]
        for cell in full_geometry_grid_cells
    }
    candidate_grid_aggregates = {
        cell: {quarter_turn: _Accumulator() for quarter_turn in _QUARTER_TURNS}
        for cell in full_geometry_grid_cells
    }
    best_epe_histogram = {str(value): 0 for value in _QUARTER_TURNS}
    best_capacity_histograms = {
        cell: {str(value): 0 for value in _QUARTER_TURNS}
        for cell in full_geometry_grid_cells
    }
    nearest_to_best_epe_confusion = {
        str(nearest): {str(best): 0 for best in _QUARTER_TURNS}
        for nearest in _QUARTER_TURNS
    }
    nearest_best_epe_match_count = 0
    best_epe_margins_px: list[float] = []
    boundary_best_epe_match_count = 0
    boundary_sample_count = 0
    histogram = {str(value): 0 for value in _QUARTER_TURNS}
    sample_reports: list[dict[str, Any]] = []
    residual_absmax = 0.0
    epe_isometry_absmax = 0.0
    gt_roundtrip_absmax = 0.0
    catastrophic_epe_sample_count = 0
    if full_geometry_enabled:
        variants = _full_geometry_oracle_variants(
            dataset,
            indices,
            full_geometry_plan,
            parsed_augment,
            bin_edges,
        )
    else:
        variants = _oracle_variants(dataset, indices, rotation_plan, bin_edges)
    for variant_batch in _batched(variants, int(batch_size)):
        warped = torch.stack(
            [value.canonicalized_warped for value in variant_batch]
        ).to(requested_device, non_blocking=False)
        target_flow = torch.stack([value.target_flow for value in variant_batch]).to(
            requested_device, non_blocking=False
        )
        valid = torch.stack([value.valid for value in variant_batch]).to(
            requested_device, non_blocking=False
        )
        turns = [value.oracle_quarter_turn_deg for value in variant_batch]
        candidate_teacher_flows: dict[int, Tensor] = {}
        candidate_target_flows: dict[int, Tensor] = {}
        with torch.inference_mode():
            if full_geometry_best_of_c4:
                flat_images: list[Tensor] = []
                flat_pairs: list[tuple[int, int]] = []
                for sample_index, variant in enumerate(variant_batch):
                    nearest_turn = variant.oracle_quarter_turn_deg
                    inverse_nearest = (
                        180 if abs(nearest_turn) == 180 else -nearest_turn
                    )
                    augmented_source = rotate_source_by_quarter_turn(
                        variant.canonicalized_warped,
                        inverse_nearest,
                    )
                    for candidate_turn in _QUARTER_TURNS:
                        candidate_source = (
                            variant.canonicalized_warped
                            if candidate_turn == nearest_turn
                            else rotate_source_by_quarter_turn(
                                augmented_source,
                                candidate_turn,
                            )
                        )
                        flat_images.append(candidate_source)
                        flat_pairs.append((sample_index, candidate_turn))
                flat_teacher_chunks: list[Tensor] = []
                for start in range(0, len(flat_images), c4_candidate_batch_size):
                    candidate_input = torch.stack(
                        flat_images[start : start + c4_candidate_batch_size]
                    ).to(requested_device, non_blocking=False)
                    flat_teacher_chunks.append(teacher(candidate_input).float())
                flat_teacher = torch.cat(flat_teacher_chunks, dim=0)
                flat_target = torch.stack(
                    [target_flow[sample_index] for sample_index, _ in flat_pairs]
                )
                flat_turns = [candidate_turn for _, candidate_turn in flat_pairs]
                flat_canonical_target = (
                    transform_backward_flow_source_map_by_quarter_turn(
                        flat_target,
                        flat_turns,
                    )
                )
                for candidate_turn in _QUARTER_TURNS:
                    flat_indices = [
                        index
                        for index, (_, turn) in enumerate(flat_pairs)
                        if turn == candidate_turn
                    ]
                    candidate_teacher_flows[candidate_turn] = flat_teacher[
                        flat_indices
                    ]
                    candidate_target_flows[candidate_turn] = (
                        flat_canonical_target[flat_indices]
                    )
                canonical_teacher_flow = torch.stack(
                    [
                        candidate_teacher_flows[turn][sample_index]
                        for sample_index, turn in enumerate(turns)
                    ]
                )
                canonical_target_flow = torch.stack(
                    [
                        candidate_target_flows[turn][sample_index]
                        for sample_index, turn in enumerate(turns)
                    ]
                )
            else:
                canonical_teacher_flow = teacher(warped).float()
                canonical_target_flow = (
                    transform_backward_flow_source_map_by_quarter_turn(
                        target_flow,
                        turns,
                    )
                )
            restored_teacher_flow = restore_teacher_flow_from_quarter_turn(
                canonical_teacher_flow, turns
            )
            roundtrip_target_flow = restore_teacher_flow_from_quarter_turn(
                canonical_target_flow,
                turns,
            )
        roundtrip_errors = (
            roundtrip_target_flow - target_flow
        ).abs().flatten(start_dim=1).amax(dim=1)
        gt_roundtrip_absmax = max(
            gt_roundtrip_absmax,
            max((float(value.item()) for value in roundtrip_errors), default=0.0),
        )
        canonical_statistics_by_iteration = {
            iterations: capacity_sample_statistics(
                canonical_teacher_flow,
                canonical_target_flow,
                valid,
                max_residual_px=max_residual_px,
                residual_target_iterations=iterations,
                max_residual_consistency=max_residual_consistency,
                max_valid_flow=max_valid_flow,
                feature_stride=feature_stride,
            )
            for iterations in solver_iterations
        }
        canonical_statistics = canonical_statistics_by_iteration[
            residual_target_iterations
        ]
        full_geometry_grid_statistics = {}
        for cell in full_geometry_grid_cells:
            grid_iterations, grid_cap = cell
            if (
                grid_iterations == residual_target_iterations
                and grid_cap == max_residual_px
            ):
                grid_statistics = canonical_statistics
            else:
                grid_statistics = capacity_sample_statistics(
                    canonical_teacher_flow,
                    canonical_target_flow,
                    valid,
                    max_residual_px=grid_cap,
                    residual_target_iterations=grid_iterations,
                    max_residual_consistency=max_residual_consistency,
                    max_valid_flow=max_valid_flow,
                    feature_stride=feature_stride,
                )
            full_geometry_grid_statistics[cell] = grid_statistics
        candidate_full_geometry_grid_statistics: dict[
            int, dict[tuple[int, float], list[Any]]
        ] = {}
        if full_geometry_best_of_c4:
            for candidate_turn in _QUARTER_TURNS:
                candidate_full_geometry_grid_statistics[candidate_turn] = {}
                for cell in full_geometry_grid_cells:
                    grid_iterations, grid_cap = cell
                    # Reuse the already-computed nearest-angle statistics only
                    # at per-sample selection time: a batch can contain
                    # different nearest turns, so there is no single candidate
                    # tensor that aliases the baseline aggregate here.
                    candidate_statistics = capacity_sample_statistics(
                        candidate_teacher_flows[candidate_turn],
                        candidate_target_flows[candidate_turn],
                        valid,
                        max_residual_px=grid_cap,
                        residual_target_iterations=grid_iterations,
                        max_residual_consistency=max_residual_consistency,
                        max_valid_flow=max_valid_flow,
                        feature_stride=feature_stride,
                    )
                    # The matching candidate tensors already supplied the v5
                    # nearest baseline above. Reuse those exact stats objects
                    # for matching samples so the v6 report cannot acquire a
                    # meaningless second-reduction rounding difference.
                    for sample_index, nearest_turn in enumerate(turns):
                        if candidate_turn == nearest_turn:
                            candidate_statistics[sample_index] = (
                                full_geometry_grid_statistics[cell][sample_index]
                            )
                    candidate_full_geometry_grid_statistics[candidate_turn][
                        cell
                    ] = candidate_statistics
        mapped_target_flow = (
            roundtrip_target_flow if canonical_frame_v2 else target_flow
        )
        mapped_back_statistics = capacity_sample_statistics(
            restored_teacher_flow,
            mapped_target_flow,
            valid,
            max_residual_px=max_residual_px,
            residual_target_iterations=residual_target_iterations,
            max_residual_consistency=max_residual_consistency,
            max_valid_flow=max_valid_flow,
            feature_stride=feature_stride,
        )
        for sample_index, (
            variant,
            canonical_stats,
            mapped_back_stats,
            roundtrip_error,
        ) in enumerate(
            zip(
                variant_batch,
                canonical_statistics,
                mapped_back_statistics,
                roundtrip_errors,
                strict=True,
            )
        ):
            mapped_back_aggregate.add(mapped_back_stats)
            mapped_back_bins[variant.rotation_bin_index].add(mapped_back_stats)
            mapped_back_quarters[variant.oracle_quarter_turn_deg].add(
                mapped_back_stats
            )
            residual_bin_index = _rotation_bin(
                variant.residual_rotation_deg,
                RESIDUAL_ROTATION_BIN_EDGES,
            )
            c4_candidate_reports: list[dict[str, Any]] = []
            best_teacher_epe_turn: int | None = None
            best_teacher_epe_margin_px: float | None = None
            best_capacity_turns_by_grid: dict[str, int] = {}
            if full_geometry_best_of_c4:
                representative_cell = full_geometry_grid_cells[0]

                def candidate_epe(candidate_turn: int) -> float:
                    value = candidate_full_geometry_grid_statistics[
                        candidate_turn
                    ][representative_cell][sample_index]
                    return (
                        math.inf
                        if value.eval_pixels == 0
                        else float(value.teacher_epe_sum / value.eval_pixels)
                    )

                ranked_by_epe = sorted(
                    _QUARTER_TURNS,
                    key=lambda candidate_turn: (
                        candidate_epe(candidate_turn),
                        _QUARTER_TURNS.index(candidate_turn),
                    ),
                )
                best_teacher_epe_turn = ranked_by_epe[0]
                best_teacher_epe_margin_px = float(
                    candidate_epe(ranked_by_epe[1])
                    - candidate_epe(best_teacher_epe_turn)
                )
                if not math.isfinite(best_teacher_epe_margin_px):
                    raise RuntimeError(
                        "best-of-C4 teacher EPE margin is non-finite"
                    )
                best_epe_margins_px.append(best_teacher_epe_margin_px)
                best_epe_histogram[str(best_teacher_epe_turn)] += 1
                nearest_to_best_epe_confusion[
                    str(variant.oracle_quarter_turn_deg)
                ][str(best_teacher_epe_turn)] += 1
                nearest_matches_best = (
                    best_teacher_epe_turn == variant.oracle_quarter_turn_deg
                )
                nearest_best_epe_match_count += int(nearest_matches_best)
                if abs(variant.residual_rotation_deg) >= 40.0:
                    boundary_sample_count += 1
                    boundary_best_epe_match_count += int(nearest_matches_best)

                candidate_reports_by_turn: dict[int, dict[str, Any]] = {
                    candidate_turn: {
                        "quarter_turn_deg": candidate_turn,
                        "residual_rotation_deg": wrap_rotation_degrees(
                            variant.injected_rotation_deg + candidate_turn
                        ),
                        "canonical_metrics_by_capacity_grid": {},
                    }
                    for candidate_turn in _QUARTER_TURNS
                }
                for cell in full_geometry_grid_cells:
                    grid_iterations, grid_cap = cell
                    grid_key = (
                        f"iterations={grid_iterations},"
                        f"max_residual_px={grid_cap:g}"
                    )
                    nearest_candidate_stats = (
                        candidate_full_geometry_grid_statistics[
                            variant.oracle_quarter_turn_deg
                        ][cell][sample_index]
                    )
                    if nearest_candidate_stats != full_geometry_grid_statistics[
                        cell
                    ][sample_index]:
                        raise RuntimeError(
                            "best-of-C4 nearest candidate diverged from the "
                            "frozen v5 baseline"
                        )
                    for candidate_turn in _QUARTER_TURNS:
                        candidate_stats = (
                            candidate_full_geometry_grid_statistics[
                                candidate_turn
                            ][cell][sample_index]
                        )
                        candidate_grid_aggregates[cell][candidate_turn].add(
                            candidate_stats
                        )
                        candidate_reports_by_turn[candidate_turn][
                            "canonical_metrics_by_capacity_grid"
                        ][grid_key] = candidate_stats.as_report()

                    assert best_teacher_epe_turn is not None
                    best_epe_stats = candidate_full_geometry_grid_statistics[
                        best_teacher_epe_turn
                    ][cell][sample_index]
                    best_epe_grid_aggregates[cell].add(best_epe_stats)
                    best_epe_grid_bins[cell][variant.rotation_bin_index].add(
                        best_epe_stats
                    )
                    best_epe_grid_residual_bins[cell][residual_bin_index].add(
                        best_epe_stats
                    )

                    def capacity_key(candidate_turn: int) -> tuple[Any, ...]:
                        value = candidate_full_geometry_grid_statistics[
                            candidate_turn
                        ][cell][sample_index]
                        return (
                            -value.trainable_pixels,
                            -value.solver_pixels,
                            candidate_epe(candidate_turn),
                            _QUARTER_TURNS.index(candidate_turn),
                        )

                    best_capacity_turn = min(
                        _QUARTER_TURNS,
                        key=capacity_key,
                    )
                    best_capacity_turns_by_grid[grid_key] = best_capacity_turn
                    best_capacity_histograms[cell][str(best_capacity_turn)] += 1
                    best_capacity_stats = (
                        candidate_full_geometry_grid_statistics[
                            best_capacity_turn
                        ][cell][sample_index]
                    )
                    best_capacity_grid_aggregates[cell].add(best_capacity_stats)
                    best_capacity_grid_bins[cell][variant.rotation_bin_index].add(
                        best_capacity_stats
                    )
                    best_capacity_grid_residual_bins[cell][residual_bin_index].add(
                        best_capacity_stats
                    )
                c4_candidate_reports = [
                    candidate_reports_by_turn[candidate_turn]
                    for candidate_turn in _QUARTER_TURNS
                ]
            canonical_reports_by_iteration: dict[str, Any] = {}
            full_geometry_grid_reports: dict[str, Any] = {}
            for iterations in solver_iterations:
                iteration_stats = canonical_statistics_by_iteration[iterations][
                    sample_index
                ]
                canonical_iteration_aggregates[iterations].add(iteration_stats)
                canonical_iteration_bins[iterations][
                    variant.rotation_bin_index
                ].add(iteration_stats)
                canonical_iteration_residual_bins[iterations][
                    residual_bin_index
                ].add(iteration_stats)
                canonical_iteration_quarters[iterations][
                    variant.oracle_quarter_turn_deg
                ].add(iteration_stats)
                if residual_target_iteration_sweep is not None:
                    canonical_reports_by_iteration[str(iterations)] = (
                        iteration_stats.as_report()
                    )
            for cell in full_geometry_grid_cells:
                grid_iterations, grid_cap = cell
                grid_stats = full_geometry_grid_statistics[cell][sample_index]
                full_geometry_grid_aggregates[cell].add(grid_stats)
                full_geometry_grid_bins[cell][variant.rotation_bin_index].add(
                    grid_stats
                )
                full_geometry_grid_residual_bins[cell][residual_bin_index].add(
                    grid_stats
                )
                full_geometry_grid_quarters[cell][
                    variant.oracle_quarter_turn_deg
                ].add(grid_stats)
                full_geometry_grid_reports[
                    f"iterations={grid_iterations},max_residual_px={grid_cap:g}"
                ] = grid_stats.as_report()
            histogram[str(variant.oracle_quarter_turn_deg)] += 1
            residual_absmax = max(
                residual_absmax, abs(variant.residual_rotation_deg)
            )
            canonical_report = canonical_stats.as_report()
            mapped_back_report = mapped_back_stats.as_report()
            canonical_epe = canonical_report["teacher_epe_px"]
            mapped_back_epe = mapped_back_report["teacher_epe_px"]
            if canonical_epe is not None and mapped_back_epe is not None:
                epe_isometry_absmax = max(
                    epe_isometry_absmax,
                    abs(float(canonical_epe) - float(mapped_back_epe)),
                )
                catastrophic_epe_sample_count += int(float(canonical_epe) > 50.0)
            sample_report = {
                "dataset_index": variant.dataset_index,
                "id": variant.sample_id,
                "injected_rotation_deg": variant.injected_rotation_deg,
                "absolute_rotation_deg": abs(variant.injected_rotation_deg),
                "rotation_bin_index": variant.rotation_bin_index,
                "oracle_quarter_turn_deg": variant.oracle_quarter_turn_deg,
                "residual_rotation_deg": variant.residual_rotation_deg,
                "canonical_metrics": canonical_report,
                "mapped_back_control_metrics": mapped_back_report,
                "teacher_epe_isometry_abs_error_px": (
                    None
                    if canonical_epe is None or mapped_back_epe is None
                    else abs(float(canonical_epe) - float(mapped_back_epe))
                ),
                "gt_absolute_map_roundtrip_absmax_px": float(
                    roundtrip_error.item()
                ),
            }
            if residual_target_iteration_sweep is not None:
                sample_report.update(
                    {
                        "residual_rotation_bin_index": residual_bin_index,
                        "canonical_metrics_by_residual_target_iterations": (
                            canonical_reports_by_iteration
                        ),
                    }
                )
            if full_geometry_enabled:
                sample_report.update(
                    {
                        "residual_rotation_bin_index": residual_bin_index,
                        "full_geometry_seed": variant.full_geometry_seed,
                        "source_homography": (
                            None
                            if variant.source_homography is None
                            else variant.source_homography.tolist()
                        ),
                    }
                )
            if full_geometry_grid_enabled:
                sample_report["canonical_metrics_by_capacity_grid"] = (
                    full_geometry_grid_reports
                )
            if full_geometry_best_of_c4:
                assert best_teacher_epe_turn is not None
                assert best_teacher_epe_margin_px is not None
                sample_report["c4_best_of_four"] = {
                    "candidate_order": list(_QUARTER_TURNS),
                    "selection_uses_ground_truth_flow": True,
                    "best_teacher_epe_quarter_turn_deg": (
                        best_teacher_epe_turn
                    ),
                    "nearest_angle_matches_best_teacher_epe": (
                        variant.oracle_quarter_turn_deg
                        == best_teacher_epe_turn
                    ),
                    "best_teacher_epe_top1_top2_margin_px": (
                        best_teacher_epe_margin_px
                    ),
                    "best_capacity_quarter_turn_by_grid": (
                        best_capacity_turns_by_grid
                    ),
                    "candidates": c4_candidate_reports,
                }
            sample_reports.append(sample_report)

    _assert_path_matches_identity(config_identity)
    _assert_path_matches_identity(manifest_identity)
    _assert_path_matches_identity(checkpoint_identity)
    _assert_path_matches_identity(teacher_identity)

    selected_indices_digest = hashlib.sha256(_canonical_json(indices)).hexdigest()
    if full_geometry_enabled:
        assert full_geometry_protocol is not None
        variant_plan_digest = str(full_geometry_protocol["seed_plan_sha256"])
    else:
        angle_records = [
            {"dataset_index": index, "angles_deg": rotation_plan[index]}
            for index in indices
        ]
        variant_plan_digest = hashlib.sha256(
            _canonical_json(angle_records)
        ).hexdigest()
    completed_at = time.time()
    rotation_bin_ranges = [
        {
            "index": index,
            "absolute_rotation_deg": {
                "lower_inclusive": bin_edges[index],
                "upper": bin_edges[index + 1],
                "upper_inclusive": index == len(bin_edges) - 2,
            },
        }
        for index in range(len(bin_edges) - 1)
    ]
    residual_rotation_bin_ranges = [
        {
            "index": index,
            "absolute_residual_rotation_deg": {
                "lower_inclusive": RESIDUAL_ROTATION_BIN_EDGES[index],
                "upper": RESIDUAL_ROTATION_BIN_EDGES[index + 1],
                "upper_inclusive": (
                    index == len(RESIDUAL_ROTATION_BIN_EDGES) - 2
                ),
            },
        }
        for index in range(len(RESIDUAL_ROTATION_BIN_EDGES) - 1)
    ]
    if full_geometry_grid_enabled:
        report_version = (
            BEST_OF_C4_REPORT_VERSION
            if full_geometry_best_of_c4
            else FULL_GEOMETRY_GRID_REPORT_VERSION
        )
        report_kind = (
            BEST_OF_C4_REPORT_KIND
            if full_geometry_best_of_c4
            else FULL_GEOMETRY_GRID_REPORT_KIND
        )
        scope = (
            "full_geometry_all_four_c4_gt_ranked_capacity_grid_v6"
            if full_geometry_best_of_c4
            else "full_geometry_known_angle_oracle_canonical_capacity_grid_v5"
        )
        flow_transform = (
            (
                "all four exact quarter turns are evaluated after the full "
                "source homography; GT ranks candidates only after all teacher "
                "forwards; nearest-angle remains the frozen v5 baseline"
            )
            if full_geometry_best_of_c4
            else (
                "known affine rotation selects an exact quarter turn after the "
                "full source homography; one teacher flow feeds every "
                "solver/cap cell"
            )
        )
        residual_frame = {
            "capacity_statistics": "canonical_source_frame",
            "canonical_gt_map": "C_q(H_full(x + F_original(x)))",
            "canonical_composition": "R(x) + B_canonical(x + R(x))",
            "capacity_grid": (
                "same teacher and GT flows; only fixed-point iterations and "
                "per-axis residual cap vary"
            ),
            "mapped_back_control": (
                "evaluated once at the effective iteration/configured-cap baseline"
            ),
        }
        results = {
            "baseline_residual_target_iterations": residual_target_iterations,
            "baseline_max_residual_px": max_residual_px,
            "capacity_grid": [
                {
                    "residual_target_iterations": cell[0],
                    "max_residual_px": cell[1],
                    "canonical_full_geometry_augmented": (
                        full_geometry_grid_aggregates[cell].report()
                    ),
                    "canonical_rotation_bins": [
                        {**record, "metrics": accumulator.report()}
                        for record, accumulator in zip(
                            rotation_bin_ranges,
                            full_geometry_grid_bins[cell],
                            strict=True,
                        )
                    ],
                    "canonical_residual_rotation_bins": [
                        {**record, "metrics": accumulator.report()}
                        for record, accumulator in zip(
                            residual_rotation_bin_ranges,
                            full_geometry_grid_residual_bins[cell],
                            strict=True,
                        )
                    ],
                    "quarter_turns": [
                        {
                            "oracle_quarter_turn_deg": quarter_turn,
                            "canonical_metrics": full_geometry_grid_quarters[cell][
                                quarter_turn
                            ].report(),
                        }
                        for quarter_turn in _QUARTER_TURNS
                    ],
                }
                for cell in full_geometry_grid_cells
            ],
            "baseline_mapped_back_control_full_geometry_augmented": (
                mapped_back_aggregate.report()
            ),
            "quarter_turn_histogram": histogram,
            "residual_rotation_absmax_deg": residual_absmax,
            "teacher_epe_isometry_absmax_px": epe_isometry_absmax,
            "gt_absolute_map_roundtrip_absmax_px": gt_roundtrip_absmax,
            "canonical_teacher_epe_over_50px_sample_count": (
                catastrophic_epe_sample_count
            ),
            "samples": sample_reports,
        }
        if full_geometry_best_of_c4:
            sorted_margins = sorted(best_epe_margins_px)

            def margin_quantile(quantile: float) -> float | None:
                if not sorted_margins:
                    return None
                position = float(quantile) * (len(sorted_margins) - 1)
                lower = int(math.floor(position))
                upper = int(math.ceil(position))
                fraction = position - lower
                return float(
                    sorted_margins[lower] * (1.0 - fraction)
                    + sorted_margins[upper] * fraction
                )

            def selected_grid_report(
                aggregates: dict[tuple[int, float], _Accumulator],
                rotation_bins: dict[tuple[int, float], list[_Accumulator]],
                residual_bins: dict[tuple[int, float], list[_Accumulator]],
                selected_histograms: dict[
                    tuple[int, float], dict[str, int]
                ] | None,
            ) -> list[dict[str, Any]]:
                return [
                    {
                        "residual_target_iterations": cell[0],
                        "max_residual_px": cell[1],
                        "canonical_full_geometry_augmented": (
                            aggregates[cell].report()
                        ),
                        "canonical_rotation_bins": [
                            {**record, "metrics": accumulator.report()}
                            for record, accumulator in zip(
                                rotation_bin_ranges,
                                rotation_bins[cell],
                                strict=True,
                            )
                        ],
                        "nearest_angle_residual_rotation_bins": [
                            {**record, "metrics": accumulator.report()}
                            for record, accumulator in zip(
                                residual_rotation_bin_ranges,
                                residual_bins[cell],
                                strict=True,
                            )
                        ],
                        **(
                            {}
                            if selected_histograms is None
                            else {
                                "selected_quarter_turn_histogram": (
                                    selected_histograms[cell]
                                )
                            }
                        ),
                    }
                    for cell in full_geometry_grid_cells
                ]

            results.update(
                {
                    "nearest_angle_capacity_grid": results["capacity_grid"],
                    "best_teacher_epe_capacity_grid": selected_grid_report(
                        best_epe_grid_aggregates,
                        best_epe_grid_bins,
                        best_epe_grid_residual_bins,
                        None,
                    ),
                    "best_capacity_aware_capacity_grid": selected_grid_report(
                        best_capacity_grid_aggregates,
                        best_capacity_grid_bins,
                        best_capacity_grid_residual_bins,
                        best_capacity_histograms,
                    ),
                    "all_candidate_capacity_grid": [
                        {
                            "residual_target_iterations": cell[0],
                            "max_residual_px": cell[1],
                            "quarter_turns": [
                                {
                                    "quarter_turn_deg": candidate_turn,
                                    "canonical_full_geometry_augmented": (
                                        candidate_grid_aggregates[cell][
                                            candidate_turn
                                        ].report()
                                    ),
                                }
                                for candidate_turn in _QUARTER_TURNS
                            ],
                        }
                        for cell in full_geometry_grid_cells
                    ],
                    "routing_comparison": {
                        "sample_count": len(sample_reports),
                        "nearest_angle_matches_best_teacher_epe_count": (
                            nearest_best_epe_match_count
                        ),
                        "nearest_angle_matches_best_teacher_epe_rate": (
                            None
                            if not sample_reports
                            else nearest_best_epe_match_count
                            / len(sample_reports)
                        ),
                        "nearest_angle_to_best_teacher_epe_confusion": (
                            nearest_to_best_epe_confusion
                        ),
                        "best_teacher_epe_quarter_turn_histogram": (
                            best_epe_histogram
                        ),
                        "best_teacher_epe_top1_top2_margin_px": {
                            "min": margin_quantile(0.0),
                            "p10": margin_quantile(0.10),
                            "p50": margin_quantile(0.50),
                            "mean": (
                                None
                                if not best_epe_margins_px
                                else float(
                                    sum(best_epe_margins_px)
                                    / len(best_epe_margins_px)
                                )
                            ),
                            "p90": margin_quantile(0.90),
                            "max": margin_quantile(1.0),
                        },
                        "nearest_residual_abs_ge_40deg": {
                            "sample_count": boundary_sample_count,
                            "nearest_angle_matches_best_teacher_epe_count": (
                                boundary_best_epe_match_count
                            ),
                            "nearest_angle_matches_best_teacher_epe_rate": (
                                None
                                if boundary_sample_count == 0
                                else boundary_best_epe_match_count
                                / boundary_sample_count
                            ),
                        },
                    },
                }
            )
    elif full_geometry_enabled:
        report_version = FULL_GEOMETRY_REPORT_VERSION
        report_kind = FULL_GEOMETRY_REPORT_KIND
        scope = "full_geometry_known_angle_oracle_canonical_frame_v4"
        flow_transform = (
            "known affine rotation selects an exact quarter turn after the full "
            "source homography; GT capacity stays in the canonical source frame"
        )
        residual_frame = {
            "capacity_statistics": "canonical_source_frame",
            "canonical_gt_map": "C_q(H_full(x + F_original(x)))",
            "canonical_composition": "R(x) + B_canonical(x + R(x))",
            "mapped_back_control": (
                "C_q^-1(x + B_canonical(x)) - x; EPE/isometry control only"
            ),
            "deployment_order": (
                "full-geometry source -> oracle C4 canonicalization -> teacher/"
                "refiner composition -> inverse quarter turn on final map"
            ),
        }
        results = {
            "canonical_full_geometry_augmented": canonical_aggregate.report(),
            "mapped_back_control_full_geometry_augmented": (
                mapped_back_aggregate.report()
            ),
            "canonical_rotation_bins": [
                {**record, "metrics": accumulator.report()}
                for record, accumulator in zip(
                    rotation_bin_ranges,
                    canonical_bins,
                    strict=True,
                )
            ],
            "canonical_residual_rotation_bins": [
                {**record, "metrics": accumulator.report()}
                for record, accumulator in zip(
                    residual_rotation_bin_ranges,
                    canonical_iteration_residual_bins[
                        residual_target_iterations
                    ],
                    strict=True,
                )
            ],
            "mapped_back_control_rotation_bins": [
                {**record, "metrics": accumulator.report()}
                for record, accumulator in zip(
                    rotation_bin_ranges,
                    mapped_back_bins,
                    strict=True,
                )
            ],
            "quarter_turns": [
                {
                    "oracle_quarter_turn_deg": quarter_turn,
                    "canonical_metrics": canonical_quarters[quarter_turn].report(),
                    "mapped_back_control_metrics": (
                        mapped_back_quarters[quarter_turn].report()
                    ),
                }
                for quarter_turn in _QUARTER_TURNS
            ],
            "quarter_turn_histogram": histogram,
            "residual_rotation_absmax_deg": residual_absmax,
            "teacher_epe_isometry_absmax_px": epe_isometry_absmax,
            "gt_absolute_map_roundtrip_absmax_px": gt_roundtrip_absmax,
            "canonical_teacher_epe_over_50px_sample_count": (
                catastrophic_epe_sample_count
            ),
            "samples": sample_reports,
        }
    elif residual_target_iteration_sweep is not None:
        report_version = SOLVER_SWEEP_REPORT_VERSION
        report_kind = SOLVER_SWEEP_REPORT_KIND
        scope = "rotation_only_known_angle_oracle_canonical_solver_sweep_v3"
        flow_transform = (
            "quarter turn applied to GT absolute source map for canonical "
            "capacity statistics; inverse applied only for baseline mapped-back "
            "control"
        )
        residual_frame = {
            "capacity_statistics": "canonical_source_frame",
            "canonical_gt_map": "C_q(x + F_augmented(x))",
            "canonical_composition": "R(x) + B_canonical(x + R(x))",
            "solver_sweep": (
                "repeat only the fixed-point residual-target solver on the same "
                "teacher and GT flows"
            ),
            "mapped_back_control": (
                "C_q^-1(x + B_canonical(x)) - x; evaluated once with the "
                f"configured {residual_target_iterations}-iteration baseline"
            ),
            "deployment_order": (
                "canonicalize source -> teacher/refiner composition -> inverse "
                "quarter turn on final absolute source map"
            ),
        }
        results = {
            "baseline_residual_target_iterations": residual_target_iterations,
            "solver_iteration_sweep": [
                {
                    "residual_target_iterations": iterations,
                    "canonical_rotation_augmented": (
                        canonical_iteration_aggregates[iterations].report()
                    ),
                    "canonical_rotation_bins": [
                        {**record, "metrics": accumulator.report()}
                        for record, accumulator in zip(
                            rotation_bin_ranges,
                            canonical_iteration_bins[iterations],
                            strict=True,
                        )
                    ],
                    "canonical_residual_rotation_bins": [
                        {**record, "metrics": accumulator.report()}
                        for record, accumulator in zip(
                            residual_rotation_bin_ranges,
                            canonical_iteration_residual_bins[iterations],
                            strict=True,
                        )
                    ],
                    "quarter_turns": [
                        {
                            "oracle_quarter_turn_deg": quarter_turn,
                            "canonical_metrics": (
                                canonical_iteration_quarters[iterations][
                                    quarter_turn
                                ].report()
                            ),
                        }
                        for quarter_turn in _QUARTER_TURNS
                    ],
                }
                for iterations in solver_iterations
            ],
            "baseline_mapped_back_control_rotation_augmented": (
                mapped_back_aggregate.report()
            ),
            "baseline_mapped_back_control_rotation_bins": [
                {**record, "metrics": accumulator.report()}
                for record, accumulator in zip(
                    rotation_bin_ranges,
                    mapped_back_bins,
                    strict=True,
                )
            ],
            "baseline_mapped_back_control_quarter_turns": [
                {
                    "oracle_quarter_turn_deg": quarter_turn,
                    "mapped_back_control_metrics": (
                        mapped_back_quarters[quarter_turn].report()
                    ),
                }
                for quarter_turn in _QUARTER_TURNS
            ],
            "quarter_turn_histogram": histogram,
            "residual_rotation_absmax_deg": residual_absmax,
            "teacher_epe_isometry_absmax_px": epe_isometry_absmax,
            "gt_absolute_map_roundtrip_absmax_px": gt_roundtrip_absmax,
            "canonical_teacher_epe_over_50px_sample_count": (
                catastrophic_epe_sample_count
            ),
            "samples": sample_reports,
        }
    elif canonical_frame_v2:
        report_version = CANONICAL_FRAME_REPORT_VERSION
        report_kind = CANONICAL_FRAME_REPORT_KIND
        scope = "rotation_only_known_angle_oracle_canonical_frame_v2"
        flow_transform = (
            "quarter turn applied to GT absolute source map for canonical "
            "capacity statistics; inverse applied only for mapped-back control"
        )
        residual_frame: dict[str, Any] | None = {
            "capacity_statistics": "canonical_source_frame",
            "canonical_gt_map": "C_q(x + F_augmented(x))",
            "canonical_composition": "R(x) + B_canonical(x + R(x))",
            "mapped_back_control": (
                "C_q^-1(x + B_canonical(x)) - x; EPE control only because "
                "the six-step fixed-point iteration is not C4-equivariant"
            ),
            "deployment_order": (
                "canonicalize source -> teacher/refiner composition -> inverse "
                "quarter turn on final absolute source map"
            ),
        }
        results: dict[str, Any] = {
            # Deliberately not named "rotation_augmented": production policy
            # requires a different report kind and three unmodified aggregates.
            "canonical_rotation_augmented": canonical_aggregate.report(),
            "mapped_back_control_rotation_augmented": (
                mapped_back_aggregate.report()
            ),
            "canonical_rotation_bins": [
                {**record, "metrics": accumulator.report()}
                for record, accumulator in zip(
                    rotation_bin_ranges,
                    canonical_bins,
                    strict=True,
                )
            ],
            "mapped_back_control_rotation_bins": [
                {**record, "metrics": accumulator.report()}
                for record, accumulator in zip(
                    rotation_bin_ranges,
                    mapped_back_bins,
                    strict=True,
                )
            ],
            "quarter_turns": [
                {
                    "oracle_quarter_turn_deg": quarter_turn,
                    "canonical_metrics": canonical_quarters[quarter_turn].report(),
                    "mapped_back_control_metrics": (
                        mapped_back_quarters[quarter_turn].report()
                    ),
                }
                for quarter_turn in _QUARTER_TURNS
            ],
            "quarter_turn_histogram": histogram,
            "residual_rotation_absmax_deg": residual_absmax,
            "teacher_epe_isometry_absmax_px": epe_isometry_absmax,
            "gt_absolute_map_roundtrip_absmax_px": gt_roundtrip_absmax,
            "canonical_teacher_epe_over_50px_sample_count": (
                catastrophic_epe_sample_count
            ),
            "samples": sample_reports,
        }
    else:
        report_version = REPORT_VERSION
        report_kind = REPORT_KIND
        scope = "rotation_only_known_angle_oracle"
        flow_transform = "inverse quarter turn applied to teacher absolute source map"
        residual_frame = None
        legacy_samples = [
            {
                "dataset_index": value["dataset_index"],
                "id": value["id"],
                "injected_rotation_deg": value["injected_rotation_deg"],
                "absolute_rotation_deg": value["absolute_rotation_deg"],
                "rotation_bin_index": value["rotation_bin_index"],
                "oracle_quarter_turn_deg": value["oracle_quarter_turn_deg"],
                "residual_rotation_deg": value["residual_rotation_deg"],
                "metrics": value["mapped_back_control_metrics"],
            }
            for value in sample_reports
        ]
        results = {
            "oracle_rotation_augmented": mapped_back_aggregate.report(),
            "rotation_bins": [
                {**record, "metrics": accumulator.report()}
                for record, accumulator in zip(
                    rotation_bin_ranges,
                    mapped_back_bins,
                    strict=True,
                )
            ],
            "quarter_turn_histogram": histogram,
            "residual_rotation_absmax_deg": residual_absmax,
            "samples": legacy_samples,
        }
    report = {
        "report_version": report_version,
        "kind": report_kind,
        "diagnostic_only": True,
        "can_approve_production": False,
        "uses_ground_truth_rotation": True,
        "uses_ground_truth_flow_for_candidate_selection": (
            full_geometry_best_of_c4
        ),
        "identities": {
            "config": config_identity,
            "checkpoint": {
                **checkpoint_identity,
                "role": "migration provenance only; payload was not loaded",
            },
            "teacher": {
                "selection": "explicit_cli_parameter",
                "checkpoint": teacher_identity,
                "expected_sha256": expected_teacher_sha256,
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
            },
            "manifest": {
                **manifest_identity,
                "split": split,
                "record_count": len(dataset),
            },
        },
        "protocol": {
            "scope": scope,
            "work_size": list(work_size),
            "requested_sample_count": sample_count,
            "selected_sample_count": len(indices),
            "selected_indices": indices,
            "selected_indices_sha256": selected_indices_digest,
            **(
                {
                    "full_geometry_seed_plan_sha256": variant_plan_digest,
                    "source_full_geometry": {
                        **full_geometry_protocol,
                        "rotation_bin_edges_deg": list(bin_edges),
                    },
                }
                if full_geometry_enabled
                else {
                    "rotation_plan_sha256": variant_plan_digest,
                    "source_rotation": {
                        **rotation_protocol,
                        "bin_edges_deg": list(bin_edges),
                        "configured_max_rotation_deg": (
                            parsed_augment.max_rotation_deg
                        ),
                    },
                }
            ),
            "oracle_quarter_turn": {
                "candidate_degrees": list(_QUARTER_TURNS),
                "selection": (
                    (
                        "v5 baseline uses nearest inverse quarter turn from the "
                        "injected angle; v6 additionally runs every candidate "
                        "before GT-flow teacher-EPE and capacity-aware ranking"
                    )
                    if full_geometry_best_of_c4
                    else (
                        "nearest inverse quarter turn from injected ground-truth "
                        "angle; angles normalized to (-180,180] with residual "
                        "in [-45,45)"
                    )
                ),
                "image_transform": "exact torch.rot90 on a square canvas",
                "flow_transform": flow_transform,
                "per_pixel_selection": False,
            },
            **(
                {
                    "best_of_c4": {
                        "candidate_batch_size": c4_candidate_batch_size,
                        "teacher_epe_selection": (
                            "minimum mean GT teacher EPE; ties use frozen "
                            "candidate order"
                        ),
                        "capacity_aware_selection": (
                            "per grid cell maximize GT trainable pixels, then "
                            "solver pixels, then minimize teacher EPE; ties use "
                            "frozen candidate order"
                        ),
                        "candidate_order": list(_QUARTER_TURNS),
                        "deployment_available": False,
                    }
                }
                if full_geometry_best_of_c4
                else {}
            ),
            **({} if residual_frame is None else {"residual_frame": residual_frame}),
            "batch_size": int(batch_size),
            "feature_stride": feature_stride,
            "max_residual_px": max_residual_px,
            "max_residual_target": max_residual_target,
            "residual_target_iterations": residual_target_iterations,
            **(
                {
                    "configured_residual_target_iterations": (
                        configured_residual_target_iterations
                    ),
                    "residual_target_iterations_override": (
                        residual_target_iterations
                    ),
                }
                if full_geometry_enabled
                else {}
            ),
            **(
                {
                    "full_geometry_solver_iteration_sweep": list(
                        full_geometry_grid_iterations
                    ),
                    "full_geometry_residual_cap_sweep_px": list(
                        full_geometry_grid_caps
                    ),
                    "full_geometry_capacity_grid_order": (
                        "solver_iterations_outer_residual_cap_inner"
                    ),
                }
                if full_geometry_grid_enabled
                else {}
            ),
            **(
                {}
                if residual_target_iteration_sweep is None
                else {
                    "residual_target_iteration_sweep": list(solver_iterations)
                }
            ),
            "max_residual_consistency": max_residual_consistency,
            "max_valid_flow": max_valid_flow,
        },
        "runtime": {
            **_runtime_identity(requested_device),
            "started_unix_seconds": started_at,
            "completed_unix_seconds": completed_at,
            "elapsed_seconds": float(time.monotonic() - started_monotonic),
        },
        "results": results,
    }
    _atomic_write_json(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/unified_v3_3_teacher_anchor.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--canonical-frame-v2",
        action="store_true",
        help=(
            "evaluate residual capacity in the canonical source frame and keep "
            "the mapped-back calculation only as an EPE/control diagnostic"
        ),
    )
    parser.add_argument(
        "--residual-target-iteration-sweep",
        type=int,
        nargs="+",
        default=None,
        help=(
            "opt-in canonical-frame fixed-point solver iteration counts; the "
            "strictly increasing list must include the configured baseline"
        ),
    )
    parser.add_argument(
        "--full-geometry-per-sample",
        type=int,
        default=0,
        help=(
            "evaluate the formal conditional rotation/scale/translation/"
            "perspective sampler this many times per selected sample"
        ),
    )
    parser.add_argument(
        "--residual-target-iterations-override",
        type=int,
        default=None,
        help="diagnostic-only solver iteration override for full geometry",
    )
    parser.add_argument(
        "--full-geometry-solver-iteration-sweep",
        type=int,
        nargs="+",
        default=None,
        help="solver iterations for the full-geometry capacity grid",
    )
    parser.add_argument(
        "--full-geometry-residual-cap-sweep",
        type=float,
        nargs="+",
        default=None,
        help="per-axis residual caps in pixels for the full-geometry capacity grid",
    )
    parser.add_argument(
        "--full-geometry-best-of-c4",
        action="store_true",
        help=(
            "run all four C4 teacher candidates and use GT flow only for the "
            "offline v6 ranking audit"
        ),
    )
    parser.add_argument(
        "--c4-candidate-batch-size",
        type=int,
        default=1,
        help="teacher batch size for the four v6 C4 candidate forwards",
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, int(args.threads)))
    report = run_quarter_turn_oracle_diagnostic(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        teacher_path=args.teacher,
        expected_teacher_sha256=args.expected_teacher_sha256,
        output_path=args.output,
        sample_count=args.sample_count,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        canonical_frame_v2=args.canonical_frame_v2,
        residual_target_iteration_sweep=args.residual_target_iteration_sweep,
        full_geometry_per_sample=args.full_geometry_per_sample,
        residual_target_iterations_override=(
            args.residual_target_iterations_override
        ),
        full_geometry_solver_iteration_sweep=(
            args.full_geometry_solver_iteration_sweep
        ),
        full_geometry_residual_cap_sweep=(
            args.full_geometry_residual_cap_sweep
        ),
        full_geometry_best_of_c4=args.full_geometry_best_of_c4,
        c4_candidate_batch_size=args.c4_candidate_batch_size,
    )
    summary = {
        "output": str(args.output.expanduser().absolute()),
        "kind": report["kind"],
        "diagnostic_only": report["diagnostic_only"],
        "can_approve_production": report["can_approve_production"],
        "quarter_turn_histogram": report["results"]["quarter_turn_histogram"],
    }
    if args.full_geometry_solver_iteration_sweep is not None:
        summary.update(
            {
                "baseline_residual_target_iterations": report["results"][
                    "baseline_residual_target_iterations"
                ],
                "baseline_max_residual_px": report["results"][
                    "baseline_max_residual_px"
                ],
                "capacity_grid": [
                    {
                        "residual_target_iterations": value[
                            "residual_target_iterations"
                        ],
                        "max_residual_px": value["max_residual_px"],
                        "canonical_full_geometry_augmented": value[
                            "canonical_full_geometry_augmented"
                        ],
                        "canonical_residual_rotation_bins": value[
                            "canonical_residual_rotation_bins"
                        ],
                    }
                    for value in report["results"]["capacity_grid"]
                ],
                "baseline_mapped_back_control_full_geometry_augmented": report[
                    "results"
                ]["baseline_mapped_back_control_full_geometry_augmented"],
            }
        )
        if args.full_geometry_best_of_c4:
            summary.update(
                {
                    "routing_comparison": report["results"][
                        "routing_comparison"
                    ],
                    "best_teacher_epe_capacity_grid": [
                        {
                            "residual_target_iterations": value[
                                "residual_target_iterations"
                            ],
                            "max_residual_px": value["max_residual_px"],
                            "canonical_full_geometry_augmented": value[
                                "canonical_full_geometry_augmented"
                            ],
                            "canonical_rotation_bins": value[
                                "canonical_rotation_bins"
                            ],
                        }
                        for value in report["results"][
                            "best_teacher_epe_capacity_grid"
                        ]
                    ],
                    "best_capacity_aware_capacity_grid": [
                        {
                            "residual_target_iterations": value[
                                "residual_target_iterations"
                            ],
                            "max_residual_px": value["max_residual_px"],
                            "canonical_full_geometry_augmented": value[
                                "canonical_full_geometry_augmented"
                            ],
                            "canonical_rotation_bins": value[
                                "canonical_rotation_bins"
                            ],
                            "selected_quarter_turn_histogram": value[
                                "selected_quarter_turn_histogram"
                            ],
                        }
                        for value in report["results"][
                            "best_capacity_aware_capacity_grid"
                        ]
                    ],
                }
            )
    elif args.full_geometry_per_sample > 0:
        summary.update(
            {
                "residual_target_iterations": report["protocol"][
                    "residual_target_iterations"
                ],
                "canonical_full_geometry_augmented": report["results"][
                    "canonical_full_geometry_augmented"
                ],
                "mapped_back_control_full_geometry_augmented": report["results"][
                    "mapped_back_control_full_geometry_augmented"
                ],
                "canonical_rotation_bins": report["results"][
                    "canonical_rotation_bins"
                ],
                "canonical_residual_rotation_bins": report["results"][
                    "canonical_residual_rotation_bins"
                ],
                "quarter_turns": report["results"]["quarter_turns"],
                "teacher_epe_isometry_absmax_px": report["results"][
                    "teacher_epe_isometry_absmax_px"
                ],
                "gt_absolute_map_roundtrip_absmax_px": report["results"][
                    "gt_absolute_map_roundtrip_absmax_px"
                ],
            }
        )
    elif args.residual_target_iteration_sweep is not None:
        summary.update(
            {
                "baseline_residual_target_iterations": report["results"][
                    "baseline_residual_target_iterations"
                ],
                "solver_iteration_sweep": [
                    {
                        "residual_target_iterations": value[
                            "residual_target_iterations"
                        ],
                        "canonical_rotation_augmented": value[
                            "canonical_rotation_augmented"
                        ],
                        "canonical_residual_rotation_bins": value[
                            "canonical_residual_rotation_bins"
                        ],
                    }
                    for value in report["results"]["solver_iteration_sweep"]
                ],
                "baseline_mapped_back_control_rotation_augmented": report[
                    "results"
                ]["baseline_mapped_back_control_rotation_augmented"],
                "teacher_epe_isometry_absmax_px": report["results"][
                    "teacher_epe_isometry_absmax_px"
                ],
                "gt_absolute_map_roundtrip_absmax_px": report["results"][
                    "gt_absolute_map_roundtrip_absmax_px"
                ],
            }
        )
    elif args.canonical_frame_v2:
        summary.update(
            {
                "canonical_rotation_augmented": report["results"][
                    "canonical_rotation_augmented"
                ],
                "mapped_back_control_rotation_augmented": report["results"][
                    "mapped_back_control_rotation_augmented"
                ],
                "quarter_turns": report["results"]["quarter_turns"],
                "teacher_epe_isometry_absmax_px": report["results"][
                    "teacher_epe_isometry_absmax_px"
                ],
                "gt_absolute_map_roundtrip_absmax_px": report["results"][
                    "gt_absolute_map_roundtrip_absmax_px"
                ],
                "canonical_teacher_epe_over_50px_sample_count": report[
                    "results"
                ]["canonical_teacher_epe_over_50px_sample_count"],
            }
        )
    else:
        summary["oracle_rotation_augmented"] = report["results"][
            "oracle_rotation_augmented"
        ]
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


__all__ = [
    "BEST_OF_C4_REPORT_KIND",
    "BEST_OF_C4_REPORT_VERSION",
    "CANONICAL_FRAME_REPORT_KIND",
    "CANONICAL_FRAME_REPORT_VERSION",
    "DEFAULT_SOLVER_ITERATION_SWEEP",
    "FULL_GEOMETRY_REPORT_KIND",
    "FULL_GEOMETRY_REPORT_VERSION",
    "FULL_GEOMETRY_GRID_REPORT_KIND",
    "FULL_GEOMETRY_GRID_REPORT_VERSION",
    "REPORT_KIND",
    "REPORT_VERSION",
    "SOLVER_SWEEP_REPORT_KIND",
    "SOLVER_SWEEP_REPORT_VERSION",
    "oracle_quarter_turn_degrees",
    "restore_teacher_flow_from_quarter_turn",
    "rotate_source_by_quarter_turn",
    "run_quarter_turn_oracle_diagnostic",
    "transform_backward_flow_source_map_by_quarter_turn",
    "wrap_rotation_degrees",
]
