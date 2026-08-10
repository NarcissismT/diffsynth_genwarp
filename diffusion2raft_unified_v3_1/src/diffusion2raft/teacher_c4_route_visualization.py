"""Render one real C4 routing failure with the frozen geometry teacher.

This is an explanatory, diagnostic-only replay.  It binds the v6 best-of-C4
report, reconstructs the exact source homography for one recorded sample, and
runs the same frozen teacher separately on the recorded nearest-angle and
teacher-optimal quarter-turn candidates.  Ground-truth flow is used to form an
offline residual oracle and therefore none of the outputs are deployment or
production-approval evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor

from .config import load_config
from .data import DocumentFlowDataset, SourceGeometryAugment
from .geometry import (
    backward_warp,
    compose_backward_flows,
    flow_valid_mask,
    residual_from_composed_flow,
    resize_backward_flow,
)
from .models.teacher_prior import TorchScriptGeometryPrior
from .teacher_capacity_preflight import (
    _assert_path_matches_identity,
    _atomic_write_json,
    _file_identity,
    _resolve_configured_path,
    _runtime_identity,
    capacity_sample_statistics,
)
from .teacher_quarter_turn_diagnostic import (
    BEST_OF_C4_REPORT_KIND,
    BEST_OF_C4_REPORT_VERSION,
    _QUARTER_TURNS,
    _full_geometry_oracle_variants,
    rotate_source_by_quarter_turn,
    transform_backward_flow_source_map_by_quarter_turn,
    wrap_rotation_degrees,
)


REPORT_VERSION = 1
REPORT_KIND = "teacher_c4_route_failure_visualization"
DEFAULT_SAMPLE_ID = "Pers_NoAug_0010947"
DEFAULT_V6_SHA256 = (
    "abf55cc22ea65665d175563c73d18ce993d7401923c4afe4e824073900655176"
)
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class RouteRender:
    role: str
    quarter_turn_deg: int
    residual_rotation_deg: float
    candidate: Image.Image
    teacher: Image.Image
    capped_oracle: Image.Image
    reference: Image.Image
    metrics: dict[str, Any]


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return _require_mapping(json.load(handle), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error


def _validate_sha256(value: str, label: str) -> str:
    normalized = str(value)
    if (
        len(normalized) != _SHA256_LENGTH
        or normalized.lower() != normalized
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{label} must be a canonical lowercase SHA-256")
    return normalized


def _assert_bound_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for key in ("sha256", "size_bytes"):
        if actual.get(key) != expected.get(key):
            raise RuntimeError(
                f"{label} diverged from the v6-bound identity at {key}: "
                f"actual={actual.get(key)!r} expected={expected.get(key)!r}"
            )


def _artifact_identity(path: Path, relative_name: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {
        "relative_path": relative_name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _tensor_to_image(value: Tensor) -> Image.Image:
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"image batch must contain one sample, got {value.shape[0]}")
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"image tensor must be [3,H,W], got {tuple(value.shape)}")
    array = (
        value.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .add(0.5)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def project_residual_to_bounded_feature_grid(
    residual: Tensor,
    *,
    feature_stride: int,
    max_residual_px: float,
) -> tuple[Tensor, Tensor, tuple[float, float]]:
    """Project an oracle residual through the refiner's stride and L-inf cap.

    The refiner predicts on a low-resolution feature grid.  Its configured
    full-resolution displacement cap is converted to that grid using the same
    endpoint-coordinate scaling as ``ResidualFlowRefiner``.  Clamping an
    oracle low-grid field is an illustrative upper-bound projection; it is not
    a prediction made by a learned model and is labelled accordingly.
    """

    if residual.ndim != 4 or residual.shape[1] != 2:
        raise ValueError(f"residual must be [B,2,H,W], got {tuple(residual.shape)}")
    feature_stride = int(feature_stride)
    max_residual_px = float(max_residual_px)
    if feature_stride < 1:
        raise ValueError("feature_stride must be at least one")
    if not math.isfinite(max_residual_px) or max_residual_px <= 0.0:
        raise ValueError("max_residual_px must be finite and positive")
    full_h, full_w = (int(value) for value in residual.shape[-2:])
    if full_h % feature_stride or full_w % feature_stride:
        raise ValueError(
            f"residual size {(full_h, full_w)} is not divisible by stride "
            f"{feature_stride}"
        )
    low_h, low_w = full_h // feature_stride, full_w // feature_stride
    low = resize_backward_flow(
        residual,
        (low_h, low_w),
        source_size_from=(full_h, full_w),
        source_size_to=(low_h, low_w),
    )
    limit_x = max_residual_px * (low_w - 1) / max(full_w - 1, 1)
    limit_y = max_residual_px * (low_h - 1) / max(full_h - 1, 1)
    bounded = low.clone()
    bounded[:, 0].clamp_(min=-limit_x, max=limit_x)
    bounded[:, 1].clamp_(min=-limit_y, max=limit_y)
    restored = resize_backward_flow(
        bounded,
        (full_h, full_w),
        source_size_from=(low_h, low_w),
        source_size_to=(full_h, full_w),
    )
    return bounded, restored, (float(limit_x), float(limit_y))


def _mean_on_mask(value: Tensor, mask: Tensor) -> float | None:
    selected = value[mask]
    if selected.numel() == 0:
        return None
    return float(selected.double().mean().item())


def _evaluate_route(
    *,
    role: str,
    quarter_turn_deg: int,
    injected_rotation_deg: float,
    candidate_source_cpu: Tensor,
    target_flow_cpu: Tensor,
    valid_cpu: Tensor,
    teacher: TorchScriptGeometryPrior,
    device: torch.device,
    residual_target_iterations: int,
    max_residual_px: float,
    max_residual_consistency: float,
    max_valid_flow: float,
    feature_stride: int,
) -> tuple[RouteRender, dict[str, Any]]:
    candidate_source = candidate_source_cpu.unsqueeze(0).to(device)
    target_flow = target_flow_cpu.unsqueeze(0).to(device)
    valid = valid_cpu.unsqueeze(0).to(device)
    canonical_target = transform_backward_flow_source_map_by_quarter_turn(
        target_flow,
        [quarter_turn_deg],
    )
    with torch.inference_mode():
        teacher_flow = teacher(candidate_source).float()
        teacher_rectified = backward_warp(
            candidate_source.float(),
            teacher_flow,
            padding_mode="border",
        )
        reference = backward_warp(
            candidate_source.float(),
            canonical_target.float(),
            padding_mode="border",
        )

    statistics = capacity_sample_statistics(
        teacher_flow,
        canonical_target,
        valid,
        max_residual_px=max_residual_px,
        residual_target_iterations=residual_target_iterations,
        max_residual_consistency=max_residual_consistency,
        max_valid_flow=max_valid_flow,
        feature_stride=feature_stride,
    )[0].as_report()

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        teacher_float = teacher_flow.detach().float()
        target_float = canonical_target.detach().float()
        finite = torch.isfinite(target_float).all(dim=1, keepdim=True)
        target_magnitude = torch.linalg.vector_norm(
            target_float, dim=1, keepdim=True
        )
        valid_mask = valid.bool() & finite & (target_magnitude < max_valid_flow)
        safe_target = torch.where(
            finite.expand_as(target_float), target_float, teacher_float
        )
        residual, consistency = residual_from_composed_flow(
            teacher_float,
            safe_target,
            iterations=residual_target_iterations,
        )
        residual_finite = torch.isfinite(residual).all(dim=1, keepdim=True)
        safe_residual = torch.where(
            torch.isfinite(residual), residual, torch.zeros_like(residual)
        )
        solver_valid = (
            valid_mask
            & residual_finite
            & torch.isfinite(consistency)
            & flow_valid_mask(safe_residual, safe_residual.shape[-2:])
            & (consistency <= max_residual_consistency)
        )
        low_residual, restored_bounded, low_limits = (
            project_residual_to_bounded_feature_grid(
                safe_residual,
                feature_stride=feature_stride,
                max_residual_px=max_residual_px,
            )
        )
        capped_flow = compose_backward_flows(teacher_float, restored_bounded)
        capped_rectified = backward_warp(
            candidate_source.float(),
            capped_flow,
            padding_mode="border",
        )
        capped_epe = torch.linalg.vector_norm(
            capped_flow - target_float, dim=1, keepdim=True
        )
        teacher_epe = torch.linalg.vector_norm(
            teacher_float - target_float, dim=1, keepdim=True
        )
        restored_axis_max = restored_bounded.abs().amax(dim=(0, 2, 3))
        low_axis_max = low_residual.abs().amax(dim=(0, 2, 3))

    metrics = {
        **statistics,
        "residual_rotation_deg": wrap_rotation_degrees(
            injected_rotation_deg + quarter_turn_deg
        ),
        "capped_projection_epe_on_eval_px": _mean_on_mask(
            capped_epe[:, 0], valid_mask[:, 0]
        ),
        "capped_projection_epe_on_solver_px": _mean_on_mask(
            capped_epe[:, 0], solver_valid[:, 0]
        ),
        "teacher_epe_direct_reduction_px": _mean_on_mask(
            teacher_epe[:, 0], valid_mask[:, 0]
        ),
        "bounded_low_grid_limit": {"x": low_limits[0], "y": low_limits[1]},
        "unbounded_low_grid_axis_absmax": {
            "x": float(low_axis_max[0].item()),
            "y": float(low_axis_max[1].item()),
        },
        "restored_bounded_axis_absmax_px": {
            "x": float(restored_axis_max[0].item()),
            "y": float(restored_axis_max[1].item()),
        },
        "capped_projection_is_oracle_not_model_prediction": True,
    }
    route = RouteRender(
        role=role,
        quarter_turn_deg=quarter_turn_deg,
        residual_rotation_deg=float(metrics["residual_rotation_deg"]),
        candidate=_tensor_to_image(candidate_source),
        teacher=_tensor_to_image(teacher_rectified),
        capped_oracle=_tensor_to_image(capped_rectified),
        reference=_tensor_to_image(reference),
        metrics=metrics,
    )
    tensors = {
        "teacher_flow": teacher_flow.detach().cpu(),
        "target_flow": canonical_target.detach().cpu(),
        "oracle_residual": safe_residual.detach().cpu(),
        "capped_projection_flow": capped_flow.detach().cpu(),
    }
    return route, tensors


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        ]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 5,
    anchor: str = "la",
) -> int:
    x, y = xy
    lines = _wrap_text(draw, text, font, max_width)
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox[3] - bbox[1] + line_gap
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, anchor=anchor)
        y += line_height
    return y


def _format_turn(value: int) -> str:
    return f"{value:+d}°" if value else "0°"


def _metric_float(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"route metric {key} must be finite")
    return float(value)


def _required_axis_max(metrics: Mapping[str, Any]) -> float:
    axes = _require_mapping(
        metrics.get("oracle_residual_axis_absmax_px"),
        "oracle residual axis maximum",
    )
    return max(float(axes["x"]), float(axes["y"]))


def render_comparison_figure(
    wrong: RouteRender,
    correct: RouteRender,
    *,
    sample_id: str,
    injected_rotation_deg: float,
    max_residual_px: float,
) -> Image.Image:
    """Create a self-contained two-row explanation from real route images."""

    width, height = 2140, 1480
    background = "#F7F9FC"
    ink = "#172033"
    muted = "#596579"
    red = "#C83E4D"
    red_light = "#FCE8EB"
    green = "#16835B"
    green_light = "#E4F5ED"
    blue = "#2D6CDF"
    grid = "#D8DEE8"
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)

    title_font = _font(42, bold=True)
    subtitle_font = _font(21)
    header_font = _font(21, bold=True)
    row_font = _font(28, bold=True)
    metric_label_font = _font(18)
    metric_value_font = _font(25, bold=True)
    small_font = _font(17)
    footnote_font = _font(16)

    draw.text(
        (44, 28),
        "Why the C4 direction must be chosen before rectification",
        font=title_font,
        fill=ink,
    )
    subtitle = (
        f"Real boundary sample {sample_id}  •  injected rotation "
        f"{injected_rotation_deg:+.3f}°  •  actual frozen-teacher outputs"
    )
    draw.text((46, 82), subtitle, font=subtitle_font, fill=muted)

    label_width = 300
    panel_size = 420
    gap = 24
    panel_x0 = 44 + label_width
    columns = [
        "C4 candidate sent\nto the teacher",
        "Actual teacher\nrectification",
        f"{max_residual_px:g} px-capped\noracle projection",
        "Flat GT geometry\nreference",
    ]
    for index, label in enumerate(columns):
        x = panel_x0 + index * (panel_size + gap) + panel_size // 2
        lines = label.splitlines()
        for line_index, line in enumerate(lines):
            draw.text(
                (x, 119 + line_index * 25),
                line,
                font=header_font,
                fill=ink,
                anchor="ma",
            )

    rows = [(wrong, 180, red, red_light), (correct, 680, green, green_light)]
    for route, y, color, light in rows:
        draw.rounded_rectangle(
            (36, y - 10, width - 36, y + panel_size + 18),
            radius=18,
            fill=light,
            outline=color,
            width=3,
        )
        role_title = "WRONG ROUTE" if route.role == "wrong" else "CORRECT ROUTE"
        role_subtitle = (
            "nearest-angle choice" if route.role == "wrong" else "teacher-optimal choice"
        )
        draw.text((60, y + 17), role_title, font=row_font, fill=color)
        draw.text(
            (60, y + 57),
            f"q = {_format_turn(route.quarter_turn_deg)}",
            font=metric_value_font,
            fill=ink,
        )
        draw.text((60, y + 91), role_subtitle, font=small_font, fill=muted)
        draw.text((60, y + 129), "Teacher error", font=metric_label_font, fill=muted)
        draw.text(
            (60, y + 154),
            f"{_metric_float(route.metrics, 'teacher_epe_px'):.1f} px",
            font=metric_value_font,
            fill=ink,
        )
        axes = _require_mapping(
            route.metrics.get("oracle_residual_axis_absmax_px"),
            "oracle residual axis maximum",
        )
        draw.text((60, y + 202), "Correction required", font=metric_label_font, fill=muted)
        draw.text(
            (60, y + 227),
            f"x {float(axes['x']):.1f} / y {float(axes['y']):.1f} px",
            font=metric_value_font,
            fill=ink,
        )
        draw.text((60, y + 275), f"Within ±{max_residual_px:g} px", font=metric_label_font, fill=muted)
        draw.text(
            (60, y + 300),
            f"{100.0 * _metric_float(route.metrics, 'trainable_coverage'):.1f}%",
            font=metric_value_font,
            fill=color,
        )
        overflow = _metric_float(
            route.metrics,
            "oracle_residual_overflow_given_solvable_any_axis_pixel_rate",
        )
        _draw_wrapped(
            draw,
            (60, y + 348),
            f"{100.0 * overflow:.1f}% of solvable pixels need more than the cap.",
            font=small_font,
            fill=muted,
            max_width=245,
        )

        images = [route.candidate, route.teacher, route.capped_oracle, route.reference]
        for index, panel in enumerate(images):
            x = panel_x0 + index * (panel_size + gap)
            resized = panel.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
            canvas.paste(resized, (x, y))
            draw.rectangle(
                (x, y, x + panel_size - 1, y + panel_size - 1),
                outline=color,
                width=3,
            )

    chart_top = 1145
    draw.text(
        (44, chart_top),
        "Largest residual displacement the downstream model would have to supply",
        font=_font(25, bold=True),
        fill=ink,
    )
    draw.text(
        (44, chart_top + 37),
        "If the route is wrong, the required correction is nearly ten times the configured ±40 px range.",
        font=small_font,
        fill=muted,
    )
    chart_x0, chart_x1 = 360, width - 70
    wrong_required = _required_axis_max(wrong.metrics)
    correct_required = _required_axis_max(correct.metrics)
    chart_max = max(400.0, math.ceil(max(wrong_required, max_residual_px) / 50.0) * 50.0)
    bar_width = chart_x1 - chart_x0
    cap_x = chart_x0 + int(bar_width * max_residual_px / chart_max)
    draw.line((cap_x, chart_top + 76, cap_x, chart_top + 212), fill=blue, width=4)
    draw.text(
        (cap_x, chart_top + 65),
        f"±{max_residual_px:g} px cap",
        font=_font(16, bold=True),
        fill=blue,
        anchor="ms",
    )
    for name, required, y, color in (
        ("Wrong route", wrong_required, chart_top + 100, red),
        ("Correct route", correct_required, chart_top + 166, green),
    ):
        draw.text((44, y + 14), name, font=_font(18, bold=True), fill=color)
        draw.rounded_rectangle(
            (chart_x0, y, chart_x1, y + 28),
            radius=14,
            fill="#E5E9F0",
        )
        end = chart_x0 + max(4, int(bar_width * required / chart_max))
        draw.rounded_rectangle((chart_x0, y, end, y + 28), radius=14, fill=color)
        draw.text(
            (min(end + 12, chart_x1 - 5), y + 14),
            f"{required:.1f} px",
            font=_font(17, bold=True),
            fill=ink,
            anchor="lm" if end + 90 < chart_x1 else "rm",
        )
    for tick in (0.0, 100.0, 200.0, 300.0, 400.0):
        if tick > chart_max:
            continue
        x = chart_x0 + int(bar_width * tick / chart_max)
        draw.line((x, chart_top + 205, x, chart_top + 213), fill=grid, width=2)
        draw.text((x, chart_top + 218), f"{tick:g}", font=_font(14), fill=muted, anchor="ma")

    footnote = (
        "The teacher panels are real model outputs. The capped panels use GT flow to construct an "
        "illustrative oracle projection; they are not predictions from a trained downstream model."
    )
    draw.text((44, height - 31), footnote, font=footnote_font, fill=muted, anchor="ls")
    return canvas


def _save_image(path: Path, image: Image.Image) -> None:
    image.save(path, format="PNG", optimize=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"failed to save non-empty image: {path}")


def _route_file_stem(route: RouteRender) -> str:
    turn = (
        "q0"
        if route.quarter_turn_deg == 0
        else f"qplus{route.quarter_turn_deg}"
        if route.quarter_turn_deg > 0
        else f"qminus{abs(route.quarter_turn_deg)}"
    )
    return f"{route.role}_route_{turn}"


def _candidate_record(sample: Mapping[str, Any], quarter_turn: int) -> dict[str, Any]:
    c4 = _require_mapping(sample.get("c4_best_of_four"), "v6 C4 record")
    candidates = c4.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("v6 C4 candidates must be a list")
    matches = [
        value
        for value in candidates
        if isinstance(value, dict) and value.get("quarter_turn_deg") == quarter_turn
    ]
    if len(matches) != 1:
        raise ValueError(
            f"v6 sample must contain exactly one q={quarter_turn} candidate"
        )
    return matches[0]


def _capacity_drift(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    tolerances = {
        "teacher_epe_px": (0.25, 0.005),
        "oracle_solver_coverage": (0.005, 0.0),
        "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": (0.005, 0.0),
        "trainable_coverage": (0.005, 0.0),
        "stride_trainable_oracle_reconstruction_epe_px": (0.10, 0.01),
    }
    result: dict[str, Any] = {}
    failures: list[str] = []
    for key, (absolute_tolerance, relative_tolerance) in tolerances.items():
        actual_value = float(actual[key])
        expected_value = float(expected[key])
        difference = actual_value - expected_value
        allowed = absolute_tolerance + relative_tolerance * abs(expected_value)
        result[key] = {
            "actual": actual_value,
            "v6_expected": expected_value,
            "difference": difference,
            "allowed_abs_difference": allowed,
            "within_tolerance": abs(difference) <= allowed,
        }
        if abs(difference) > allowed:
            failures.append(f"{key}: difference={difference} allowed={allowed}")
    actual_axes = _require_mapping(
        actual.get("oracle_residual_axis_absmax_px"), "actual residual axes"
    )
    expected_axes = _require_mapping(
        expected.get("oracle_residual_axis_absmax_px"), "expected residual axes"
    )
    axis_result: dict[str, Any] = {}
    for axis in ("x", "y"):
        actual_value = float(actual_axes[axis])
        expected_value = float(expected_axes[axis])
        difference = actual_value - expected_value
        axis_result[axis] = {
            "actual": actual_value,
            "v6_expected": expected_value,
            "difference": difference,
            "allowed_abs_difference": 1.0,
            "within_tolerance": abs(difference) <= 1.0,
        }
        if abs(difference) > 1.0:
            failures.append(f"residual {axis}: difference={difference} allowed=1.0")
    result["oracle_residual_axis_absmax_px"] = axis_result
    if failures:
        raise RuntimeError(
            "visualization replay materially diverged from v6: " + "; ".join(failures)
        )
    return result


def run_c4_route_failure_visualization(
    *,
    config_path: Path,
    checkpoint_path: Path,
    teacher_path: Path,
    expected_teacher_sha256: str,
    v6_report_path: Path,
    expected_v6_report_sha256: str,
    output_dir: Path,
    sample_id: str = DEFAULT_SAMPLE_ID,
    manifest_path: Path | None = None,
    device: torch.device | str = "cuda:0",
    residual_target_iterations: int = 24,
    max_residual_px: float = 40.0,
    feature_stride: int = 8,
) -> dict[str, Any]:
    """Replay and render one v6 route disagreement using real teacher forwards."""

    started_unix = time.time()
    started_monotonic = time.monotonic()
    expected_teacher_sha256 = _validate_sha256(
        expected_teacher_sha256, "expected_teacher_sha256"
    )
    expected_v6_report_sha256 = _validate_sha256(
        expected_v6_report_sha256, "expected_v6_report_sha256"
    )
    output = output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    if output.name == "approved.json" or "preflight_v33_teacher_capacity" in output.parts:
        raise ValueError("visualization output may not use a production evidence path")

    config_path = config_path.expanduser().resolve(strict=True)
    config = load_config(config_path)
    project_root = config_path.parent.parent
    data_config = _require_mapping(config.get("data"), "config.data")
    model_config = _require_mapping(config.get("model"), "config.model")
    loss_config = _require_mapping(config.get("loss"), "config.loss")
    if str(model_config.get("prior_backend", "learned")).lower() != "torchscript":
        raise ValueError("C4 visualization requires prior_backend=torchscript")

    checkpoint = _resolve_configured_path(
        checkpoint_path, project_root=project_root, explicit=True
    ).resolve(strict=True)
    teacher_path = _resolve_configured_path(
        teacher_path, project_root=project_root, explicit=True
    ).resolve(strict=True)
    v6_report_path = _resolve_configured_path(
        v6_report_path, project_root=project_root, explicit=True
    ).resolve(strict=True)
    if manifest_path is None:
        configured_manifest = data_config.get("val_manifest")
        if not configured_manifest:
            raise ValueError("config.data.val_manifest is required")
        manifest = _resolve_configured_path(
            str(configured_manifest), project_root=project_root, explicit=False
        ).resolve(strict=True)
    else:
        manifest = _resolve_configured_path(
            manifest_path, project_root=project_root, explicit=True
        ).resolve(strict=True)

    identities = {
        "config": _file_identity(config_path, hash_contents=True),
        "checkpoint": _file_identity(checkpoint, hash_contents=True),
        "teacher": _file_identity(teacher_path, hash_contents=True),
        "manifest": _file_identity(manifest, hash_contents=True),
        "v6_report": _file_identity(v6_report_path, hash_contents=True),
    }
    if not hmac.compare_digest(
        str(identities["teacher"]["sha256"]), expected_teacher_sha256
    ):
        raise ValueError(
            "teacher SHA-256 mismatch: "
            f"expected={expected_teacher_sha256} "
            f"actual={identities['teacher']['sha256']}"
        )
    if not hmac.compare_digest(
        str(identities["v6_report"]["sha256"]), expected_v6_report_sha256
    ):
        raise ValueError(
            "v6 report SHA-256 mismatch: "
            f"expected={expected_v6_report_sha256} "
            f"actual={identities['v6_report']['sha256']}"
        )

    v6 = _read_json(v6_report_path, "v6 report")
    if (
        v6.get("report_version") != BEST_OF_C4_REPORT_VERSION
        or v6.get("kind") != BEST_OF_C4_REPORT_KIND
        or v6.get("diagnostic_only") is not True
        or v6.get("can_approve_production") is not False
        or v6.get("uses_ground_truth_flow_for_candidate_selection") is not True
    ):
        raise ValueError("input report is not the diagnostic-only v6 best-of-C4 report")
    v6_identities = _require_mapping(v6.get("identities"), "v6 identities")
    _assert_bound_identity(
        identities["config"],
        _require_mapping(v6_identities.get("config"), "v6 config identity"),
        "config",
    )
    _assert_bound_identity(
        identities["checkpoint"],
        _require_mapping(v6_identities.get("checkpoint"), "v6 checkpoint identity"),
        "checkpoint",
    )
    v6_teacher = _require_mapping(v6_identities.get("teacher"), "v6 teacher")
    _assert_bound_identity(
        identities["teacher"],
        _require_mapping(v6_teacher.get("checkpoint"), "v6 teacher identity"),
        "teacher",
    )
    _assert_bound_identity(
        identities["manifest"],
        _require_mapping(v6_identities.get("manifest"), "v6 manifest identity"),
        "manifest",
    )

    work_size_value = data_config.get("work_size")
    if not isinstance(work_size_value, (list, tuple)) or len(work_size_value) != 2:
        raise ValueError("config.data.work_size must be [height,width]")
    work_size = (int(work_size_value[0]), int(work_size_value[1]))
    if work_size[0] != work_size[1]:
        raise ValueError("C4 visualization requires a square work canvas")
    dataset = DocumentFlowDataset(manifest, work_size, augment_guide=False)
    results = _require_mapping(v6.get("results"), "v6 results")
    samples = results.get("samples")
    if not isinstance(samples, list):
        raise ValueError("v6 results.samples must be a list")
    matches = [
        value for value in samples if isinstance(value, dict) and value.get("id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"v6 report must contain exactly one sample id {sample_id!r}, got {len(matches)}"
        )
    v6_sample = matches[0]
    dataset_index = int(v6_sample["dataset_index"])
    full_geometry_seed = int(v6_sample["full_geometry_seed"])
    raw_augment = _require_mapping(
        data_config.get("source_geometry_augment"),
        "config.data.source_geometry_augment",
    )
    augment = SourceGeometryAugment.from_config(raw_augment)
    protocol = _require_mapping(v6.get("protocol"), "v6 protocol")
    source_geometry = _require_mapping(
        protocol.get("source_full_geometry"), "v6 source_full_geometry"
    )
    bin_edges = source_geometry.get("rotation_bin_edges_deg")
    if not isinstance(bin_edges, list):
        raise ValueError("v6 rotation_bin_edges_deg must be a list")
    variants = list(
        _full_geometry_oracle_variants(
            dataset,
            [dataset_index],
            {dataset_index: [full_geometry_seed]},
            augment,
            [float(value) for value in bin_edges],
        )
    )
    if len(variants) != 1:
        raise RuntimeError("full-geometry replay did not produce exactly one variant")
    variant = variants[0]
    if (
        variant.dataset_index != dataset_index
        or variant.sample_id != sample_id
        or variant.full_geometry_seed != full_geometry_seed
        or variant.oracle_quarter_turn_deg != int(v6_sample["oracle_quarter_turn_deg"])
        or not math.isclose(
            variant.injected_rotation_deg,
            float(v6_sample["injected_rotation_deg"]),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        )
    ):
        raise RuntimeError("replayed sample metadata diverged from v6")
    expected_homography = torch.tensor(v6_sample["source_homography"], dtype=torch.float64)
    if variant.source_homography is None or not torch.allclose(
        variant.source_homography.double(),
        expected_homography,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("replayed source homography diverged from v6")

    c4 = _require_mapping(v6_sample.get("c4_best_of_four"), "v6 sample C4")
    nearest_turn = int(variant.oracle_quarter_turn_deg)
    best_turn = int(c4.get("best_teacher_epe_quarter_turn_deg"))
    if nearest_turn not in _QUARTER_TURNS or best_turn not in _QUARTER_TURNS:
        raise ValueError("v6 sample contains an invalid C4 turn")
    if nearest_turn == best_turn:
        raise ValueError(
            "selected sample has no nearest-vs-best route disagreement to visualize"
        )
    inverse_nearest = 180 if abs(nearest_turn) == 180 else -nearest_turn
    augmented_source = rotate_source_by_quarter_turn(
        variant.canonicalized_warped, inverse_nearest
    )

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    residual_target_iterations = int(residual_target_iterations)
    if residual_target_iterations < 1:
        raise ValueError("residual_target_iterations must be at least one")
    feature_stride = int(feature_stride)
    max_residual_px = float(max_residual_px)
    max_residual_consistency = float(loss_config.get("max_residual_consistency", 1.0))
    max_valid_flow = float(loss_config.get("max_valid_flow", 1000.0))
    teacher = TorchScriptGeometryPrior(
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

    route_specs = (("wrong", nearest_turn), ("correct", best_turn))
    route_renders: dict[str, RouteRender] = {}
    route_tensors: dict[str, dict[str, Any]] = {}
    route_reports: dict[str, Any] = {}
    grid_key = (
        f"iterations={residual_target_iterations},max_residual_px={max_residual_px:g}"
    )
    for role, turn in route_specs:
        candidate_source = (
            variant.canonicalized_warped
            if turn == nearest_turn
            else rotate_source_by_quarter_turn(augmented_source, turn)
        )
        route, tensors = _evaluate_route(
            role=role,
            quarter_turn_deg=turn,
            injected_rotation_deg=variant.injected_rotation_deg,
            candidate_source_cpu=candidate_source,
            target_flow_cpu=variant.target_flow,
            valid_cpu=variant.valid,
            teacher=teacher,
            device=requested_device,
            residual_target_iterations=residual_target_iterations,
            max_residual_px=max_residual_px,
            max_residual_consistency=max_residual_consistency,
            max_valid_flow=max_valid_flow,
            feature_stride=feature_stride,
        )
        expected_candidate = _candidate_record(v6_sample, turn)
        expected_grid = _require_mapping(
            expected_candidate.get("canonical_metrics_by_capacity_grid"),
            "v6 candidate capacity grid",
        )
        expected_metrics = _require_mapping(
            expected_grid.get(grid_key), f"v6 candidate metrics {grid_key}"
        )
        drift = _capacity_drift(route.metrics, expected_metrics)
        route_renders[role] = route
        route_tensors[role] = tensors
        route_reports[role] = {
            "role": role,
            "quarter_turn_deg": turn,
            "residual_rotation_deg": route.residual_rotation_deg,
            "actual_metrics": route.metrics,
            "v6_expected_metrics": expected_metrics,
            "v6_replay_drift": drift,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        artifact_paths: list[Path] = []
        for route in route_renders.values():
            stem = _route_file_stem(route)
            for suffix, image in (
                ("candidate", route.candidate),
                ("teacher", route.teacher),
                (f"cap{max_residual_px:g}_oracle", route.capped_oracle),
                ("reference", route.reference),
            ):
                path = stage / f"{stem}_{suffix}.png"
                _save_image(path, image)
                artifact_paths.append(path)
        comparison = render_comparison_figure(
            route_renders["wrong"],
            route_renders["correct"],
            sample_id=sample_id,
            injected_rotation_deg=variant.injected_rotation_deg,
            max_residual_px=max_residual_px,
        )
        comparison_path = stage / "c4_route_comparison.png"
        _save_image(comparison_path, comparison)
        artifact_paths.insert(0, comparison_path)

        _assert_path_matches_identity(identities["config"])
        _assert_path_matches_identity(identities["checkpoint"])
        _assert_path_matches_identity(identities["teacher"])
        _assert_path_matches_identity(identities["manifest"])
        _assert_path_matches_identity(identities["v6_report"])
        completed_unix = time.time()
        report = {
            "report_version": REPORT_VERSION,
            "kind": REPORT_KIND,
            "diagnostic_only": True,
            "can_approve_production": False,
            "uses_ground_truth_flow_for_oracle_projection": True,
            "actual_teacher_outputs": True,
            "purpose": (
                "explain why a wrong C4 teacher route demands a residual far beyond "
                "the downstream model cap, while the teacher-optimal route remains "
                "correctable"
            ),
            "identities": {
                "config": identities["config"],
                "checkpoint": {
                    **identities["checkpoint"],
                    "role": "v6 migration provenance only; payload was not loaded",
                },
                "teacher": identities["teacher"],
                "manifest": identities["manifest"],
                "bound_v6_report": identities["v6_report"],
            },
            "protocol": {
                "work_size": list(work_size),
                "feature_stride": feature_stride,
                "residual_target_iterations": residual_target_iterations,
                "max_residual_px": max_residual_px,
                "max_residual_consistency": max_residual_consistency,
                "max_valid_flow": max_valid_flow,
                "teacher_forward_batch_size": 1,
                "route_order": [nearest_turn, best_turn],
                "wrong_route_definition": "v6 nearest-angle C4 choice",
                "correct_route_definition": "v6 GT-flow-ranked minimum-teacher-EPE C4 choice",
                "capped_projection": (
                    "GT-derived fixed-point residual -> stride feature grid -> "
                    "per-axis model-equivalent cap -> full-resolution flow composition"
                ),
                "capped_projection_is_learned_model_output": False,
            },
            "sample": {
                "id": sample_id,
                "dataset_index": dataset_index,
                "full_geometry_seed": full_geometry_seed,
                "injected_rotation_deg": variant.injected_rotation_deg,
                "source_homography": variant.source_homography.tolist(),
                "nearest_angle_quarter_turn_deg": nearest_turn,
                "best_teacher_epe_quarter_turn_deg": best_turn,
                "best_teacher_epe_top1_top2_margin_px": c4.get(
                    "best_teacher_epe_top1_top2_margin_px"
                ),
            },
            "routes": route_reports,
            "artifacts": [
                _artifact_identity(path, path.name) for path in artifact_paths
            ],
            "runtime": {
                **_runtime_identity(requested_device),
                "started_unix_seconds": started_unix,
                "completed_unix_seconds": completed_unix,
                "elapsed_seconds": time.monotonic() - started_monotonic,
            },
        }
        _atomic_write_json(stage / "report.json", report)
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        # Keep the large scripted teacher out of subsequent interactive work.
        del teacher
        if requested_device.type == "cuda":
            torch.cuda.empty_cache()
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
    parser.add_argument("--v6-report", type=Path, required=True)
    parser.add_argument(
        "--expected-v6-report-sha256",
        default=DEFAULT_V6_SHA256,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--residual-target-iterations", type=int, default=24)
    parser.add_argument("--max-residual-px", type=float, default=40.0)
    parser.add_argument("--feature-stride", type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least one")
    torch.set_num_threads(args.threads)
    report = run_c4_route_failure_visualization(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        teacher_path=args.teacher,
        expected_teacher_sha256=args.expected_teacher_sha256,
        v6_report_path=args.v6_report,
        expected_v6_report_sha256=args.expected_v6_report_sha256,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        manifest_path=args.manifest,
        device=args.device,
        residual_target_iterations=args.residual_target_iterations,
        max_residual_px=args.max_residual_px,
        feature_stride=args.feature_stride,
    )
    wrong = report["routes"]["wrong"]["actual_metrics"]
    correct = report["routes"]["correct"]["actual_metrics"]
    print(
        json.dumps(
            {
                "kind": report["kind"],
                "output_dir": str(args.output_dir.expanduser().absolute()),
                "sample_id": report["sample"]["id"],
                "wrong_route": {
                    "quarter_turn_deg": report["routes"]["wrong"][
                        "quarter_turn_deg"
                    ],
                    "teacher_epe_px": wrong["teacher_epe_px"],
                    "trainable_coverage": wrong["trainable_coverage"],
                },
                "correct_route": {
                    "quarter_turn_deg": report["routes"]["correct"][
                        "quarter_turn_deg"
                    ],
                    "teacher_epe_px": correct["teacher_epe_px"],
                    "trainable_coverage": correct["trainable_coverage"],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


__all__ = [
    "DEFAULT_SAMPLE_ID",
    "DEFAULT_V6_SHA256",
    "REPORT_KIND",
    "REPORT_VERSION",
    "RouteRender",
    "main",
    "project_residual_to_bounded_feature_grid",
    "render_comparison_figure",
    "run_c4_route_failure_visualization",
]
