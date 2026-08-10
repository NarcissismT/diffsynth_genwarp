"""Evaluate a full CP-DocFlow checkpoint with exact pixel aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from PIL import Image

from .checkpoint import COORDINATE_CONTRACT, file_sha256, load_full_checkpoint
from .config import build_full_model
from .data import DocumentMapDataset, dataset_payload_sha256
from .evaluate import _calibration, _device, _require_gate_contract
from .geometry import (
    canonical_backward_map,
    resize_backward_map,
    resize_backward_map_with_mask,
    warp_with_backward_map,
)
from .losses import structure_straightness_loss, structure_targets
from .metrics import (
    cell_valid_mask,
    endpoint_error_map,
    geometry_quality_metrics,
    image_quality_metrics,
    jacobian_determinant,
)


def _apply_runtime_overrides(model: torch.nn.Module, overrides: dict[str, Any]) -> None:
    allowed = {
        "execution_stage",
        "fm_steps",
        "refiner_iterations",
        "use_qwen_condition",
        "enable_hv_condition",
        "fusion_mode",
        "upsampling_mode",
        "composition_uses_confidence",
        "sigma_min",
        "sigma_max",
        "inference_seed",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"unknown evaluation runtime overrides: {sorted(unknown)}")
    if "execution_stage" in overrides:
        model.set_execution_stage(str(overrides["execution_stage"]))
    if "fm_steps" in overrides:
        value = int(overrides["fm_steps"])
        if value < 1:
            raise ValueError("fm_steps override must be positive")
        model.fm_steps = value
    if "refiner_iterations" in overrides:
        value = int(overrides["refiner_iterations"])
        if value < 1:
            raise ValueError("refiner_iterations override must be positive")
        model.refiner.iterations = value
    for name in (
        "use_qwen_condition",
        "enable_hv_condition",
        "composition_uses_confidence",
    ):
        if name in overrides:
            setattr(model, name, bool(overrides[name]))
    if "fusion_mode" in overrides:
        value = str(overrides["fusion_mode"]).lower()
        if value not in {"cnn_only", "qwen_only", "concat", "gated"}:
            raise ValueError("invalid fusion_mode override")
        model.fusion.mode = value
    if "upsampling_mode" in overrides:
        value = str(overrides["upsampling_mode"]).lower()
        if value not in {"convex", "bilinear"}:
            raise ValueError("invalid upsampling_mode override")
        model.upsampling_mode = value
    sigma_min = float(overrides.get("sigma_min", model.sigma_min))
    sigma_max = float(overrides.get("sigma_max", model.sigma_max))
    if sigma_min < 0.0 or sigma_max < sigma_min:
        raise ValueError("runtime sigma override requires 0 <= min <= max")
    model.sigma_min, model.sigma_max = sigma_min, sigma_max
    if "inference_seed" in overrides:
        model.inference_seed = int(overrides["inference_seed"])


def _confidence_reliability(
    error: Tensor, confidence: Tensor, bins: int = 10
) -> tuple[list[dict[str, float]], float]:
    if not error.numel() or error.shape != confidence.shape:
        return [], float("nan")
    rows: list[dict[str, float]] = []
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (confidence >= boundaries[index]) & (
            confidence < boundaries[index + 1]
            if index + 1 < bins
            else confidence <= boundaries[index + 1]
        )
        if bool(selected.any()):
            rows.append(
                {
                    "confidence_min": float(boundaries[index]),
                    "confidence_max": float(boundaries[index + 1]),
                    "count": float(selected.sum()),
                    "confidence_mean": float(confidence[selected].mean()),
                    "coarse_epe": float(error[selected].mean()),
                    "accurate_1px_rate": float((error[selected] < 1.0).float().mean()),
                }
            )
    if len(rows) < 2:
        return rows, float("nan")
    comparisons = [
        rows[index]["coarse_epe"] <= rows[index - 1]["coarse_epe"] + 1.0e-6
        for index in range(1, len(rows))
    ]
    return rows, sum(comparisons) / len(comparisons)


def _rgb_array(value: Tensor, size: tuple[int, int] | None = None) -> np.ndarray:
    image = value.detach().float()
    if size is not None and tuple(image.shape[-2:]) != size:
        image = torch.nn.functional.interpolate(
            image, size, mode="bilinear", align_corners=False
        )
    return (
        image[0].permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
    )


def _save_visual_artifacts(
    root: Path,
    sample_id: str,
    *,
    warped: Tensor,
    coarse_rectified: Tensor,
    final_rectified: Tensor,
    target_rectified: Tensor,
    oracle_rectified: Tensor,
    final_error: Tensor,
    fold_mask: Tensor,
    line_mask: Tensor,
) -> None:
    output_size = tuple(int(value) for value in final_rectified.shape[-2:])
    panels = [
        _rgb_array(value, output_size)
        for value in (
            warped,
            coarse_rectified,
            final_rectified,
            target_rectified,
            oracle_rectified,
        )
    ]
    Image.fromarray(np.concatenate(panels, axis=1)).save(
        root / "warped_coarse_final_gt_oracle_panel" / f"{sample_id}.png"
    )
    normalized = final_error.detach().float().cpu().squeeze().clamp(0.0, 10.0) / 10.0
    heat = torch.stack(
        (normalized, 1.0 - normalized, torch.zeros_like(normalized)), dim=-1
    ).mul(255.0).round().byte().numpy()
    Image.fromarray(heat).save(root / "map_error_heatmaps" / f"{sample_id}.png")
    fold = fold_mask.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    Image.fromarray(fold).save(root / "fold_masks" / f"{sample_id}.png")
    line = line_mask.detach().cpu().squeeze().numpy().astype(bool)
    overlay = _rgb_array(final_rectified)
    overlay[line] = np.array((255, 0, 0), dtype=np.uint8)
    Image.fromarray(overlay).save(root / "line_visualizations" / f"{sample_id}.png")


def _summary(
    final_errors: list[Tensor],
    coarse_errors: list[Tensor],
    residual_errors: list[Tensor],
    confidences: list[Tensor],
    oracle_rgb_errors: list[Tensor],
    image_values: dict[str, list[Tensor]],
    oracle_image_values: dict[str, list[Tensor]],
    qwen_gates: list[Tensor],
    line_errors: list[Tensor],
    edge_errors: list[Tensor],
    straightness_values: list[Tensor],
    iteration_errors: dict[int, list[Tensor]],
    iteration_straightness: dict[int, list[Tensor]],
    iteration_folds: dict[int, int],
    iteration_cells: dict[int, int],
    geometry_values: dict[str, list[Tensor]],
    folds: int,
    cells: int,
    samples: int,
) -> dict[str, Any]:
    final = torch.cat(final_errors).float() if final_errors else torch.empty(0)
    coarse = torch.cat(coarse_errors).float() if coarse_errors else torch.empty(0)
    residual = torch.cat(residual_errors).float() if residual_errors else torch.empty(0)
    confidence = torch.cat(confidences).float() if confidences else torch.empty(0)
    oracle_rgb = (
        torch.cat(oracle_rgb_errors).float() if oracle_rgb_errors else torch.empty(0)
    )
    gate = torch.cat(qwen_gates).float() if qwen_gates else torch.empty(0)
    line = torch.cat(line_errors).float() if line_errors else torch.empty(0)
    edge = torch.cat(edge_errors).float() if edge_errors else torch.empty(0)
    straightness = (
        torch.stack(straightness_values).float()
        if straightness_values
        else torch.empty(0)
    )
    iteration_means = {
        index: float(torch.cat(values).float().mean())
        for index, values in sorted(iteration_errors.items())
        if values
    }
    iteration_straightness_means = {
        index: float(torch.nanmean(torch.stack(values).float()))
        for index, values in sorted(iteration_straightness.items())
        if values
    }
    iteration_fold_rates = {
        index: iteration_folds.get(index, 0) / max(iteration_cells.get(index, 0), 1)
        for index in sorted(set(iteration_folds) | set(iteration_cells))
    }
    image_summary = {
        f"rectified_{name}": float(torch.nanmean(torch.stack(values).float()))
        for name, values in sorted(image_values.items())
        if values
    }
    oracle_image_summary = {
        f"oracle_{name}": float(torch.nanmean(torch.stack(values).float()))
        for name, values in sorted(oracle_image_values.items())
        if values
    }
    geometry_summary = {
        name: float(torch.stack(values).float().mean())
        for name, values in sorted(geometry_values.items())
        if values
    }
    brier, ece = _calibration(coarse, confidence)
    reliability, confidence_monotonic_rate = _confidence_reliability(
        coarse, confidence
    )
    reliable = coarse < 1.0
    damage = (final - coarse) > 1.0
    return {
        "epe": float(final.mean()) if final.numel() else float("nan"),
        "epe_p95": float(torch.quantile(final, 0.95)) if final.numel() else float("nan"),
        "coarse_epe": float(coarse.mean()) if coarse.numel() else float("nan"),
        "coarse_epe_p95": (
            float(torch.quantile(coarse, 0.95)) if coarse.numel() else float("nan")
        ),
        "residual_epe": (
            float(residual.mean()) if residual.numel() else float("nan")
        ),
        "residual_epe_p95": (
            float(torch.quantile(residual, 0.95))
            if residual.numel()
            else float("nan")
        ),
        "final_win_rate": (
            float((final < coarse).float().mean()) if final.numel() else float("nan")
        ),
        "high_confidence_damage_rate": (
            float(damage[reliable].float().mean())
            if bool(reliable.any())
            else float("nan")
        ),
        "fold_rate": folds / max(cells, 1),
        "confidence_brier_1px": brier,
        "confidence_ece_1px": ece,
        "confidence_monotonic_rate": confidence_monotonic_rate,
        "confidence_reliability": reliability,
        "qwen_gate_mean": float(gate.mean()) if gate.numel() else float("nan"),
        "line_epe": float(line.mean()) if line.numel() else float("nan"),
        "edge_epe": float(edge.mean()) if edge.numel() else float("nan"),
        "straightness_error": (
            float(straightness.mean()) if straightness.numel() else float("nan")
        ),
        "warr_iteration_epe": {
            str(index + 1): value for index, value in iteration_means.items()
        },
        "warr_iteration_fold_rate": {
            str(index + 1): value for index, value in iteration_fold_rates.items()
        },
        "warr_iteration_straightness": {
            str(index + 1): value
            for index, value in iteration_straightness_means.items()
        },
        "warr_monotonic": all(
            iteration_means[index] <= iteration_means[index - 1] + 1.0e-6
            for index in sorted(iteration_means)
            if index - 1 in iteration_means
        ),
        **geometry_summary,
        **image_summary,
        **oracle_image_summary,
        "worksize_oracle_rgb_l1": (
            oracle_image_summary.get(
                "oracle_rgb_l1",
                float(oracle_rgb.mean()) if oracle_rgb.numel() else float("nan"),
            )
        ),
        "valid_pixels": float(final.numel()),
        "samples": float(samples),
    }


@torch.no_grad()
def evaluate_full(
    checkpoint_path: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    allowed_label_provenance: set[str],
    gate: bool = False,
    max_visualizations: int = 32,
    runtime_overrides: dict[str, Any] | None = None,
    input_work_size_override: tuple[int, int] | None = None,
    output_work_size_override: tuple[int, int] | None = None,
    prepared_model: torch.nn.Module | None = None,
    prepared_payload: dict[str, Any] | None = None,
    checkpoint_sha256: str | None = None,
    report_metadata: dict[str, Any] | None = None,
    export_ocr_images: bool = False,
) -> dict[str, Any]:
    if max_visualizations < 0:
        raise ValueError("max_visualizations must be non-negative")
    overrides = dict(runtime_overrides or {})
    if gate and overrides:
        raise ValueError("gate evaluation cannot use runtime ablation overrides")
    work_size_overridden = (
        input_work_size_override is not None or output_work_size_override is not None
    )
    if (input_work_size_override is None) != (output_work_size_override is None):
        raise ValueError(
            "input and output work-size overrides must be supplied together"
        )
    if gate and work_size_overridden:
        raise ValueError("gate evaluation cannot override checkpoint work sizes")
    if (prepared_model is None) != (prepared_payload is None):
        raise ValueError("prepared_model and prepared_payload must be supplied together")
    if prepared_model is not None and gate:
        raise ValueError("prepared external models are baseline-only, not gate eligible")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty evaluation output: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "warped_coarse_final_gt_oracle_panel",
        "map_error_heatmaps",
        "fold_masks",
        "line_visualizations",
    ):
        (destination / name).mkdir(exist_ok=True)
    if export_ocr_images:
        for name in (
            "ocr_model_rectified",
            "ocr_oracle_rectified",
            "ocr_target_rectified",
        ):
            (destination / name).mkdir(exist_ok=True)
    device = _device(device_name)
    if prepared_model is None:
        payload = load_full_checkpoint(checkpoint_path, map_location=device)
        model = build_full_model(payload["model_config"]).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.set_execution_stage(payload["training_stage"])
        _apply_runtime_overrides(model, overrides)
    else:
        payload = dict(prepared_payload or {})
        required_payload = {
            "model_config",
            "training_stage",
            "input_work_size",
            "output_work_size",
        }
        missing_payload = sorted(required_payload - set(payload))
        if missing_payload:
            raise ValueError(
                "prepared baseline payload is missing: " + ", ".join(missing_payload)
            )
        if overrides:
            raise ValueError("prepared baseline models do not accept runtime overrides")
        model = prepared_model.to(device)
    model.eval()
    input_size = tuple(
        int(value)
        for value in (
            input_work_size_override
            if input_work_size_override is not None
            else payload["input_work_size"]
        )
    )
    output_size = tuple(
        int(value)
        for value in (
            output_work_size_override
            if output_work_size_override is not None
            else payload["output_work_size"]
        )
    )
    if len(input_size) != 2 or min(input_size) < 1:
        raise ValueError("input work size must contain two positive integers")
    if len(output_size) != 2 or min(output_size) < 1:
        raise ValueError("output work size must contain two positive integers")
    dataset = DocumentMapDataset(
        manifest, input_work_size=input_size, output_work_size=output_size
    )
    evaluation_dataset_payload_sha256 = dataset_payload_sha256(dataset.records)
    requested_provenance = {value.lower() for value in allowed_label_provenance}
    if not requested_provenance:
        raise ValueError("allowed_label_provenance must be an explicit non-empty set")
    rejected = sorted(
        {record.label_provenance for record in dataset.records} - requested_provenance
    )
    if rejected:
        raise ValueError(f"evaluation contains unapproved label provenance: {rejected}")
    manifest_sha256 = file_sha256(manifest)
    if gate:
        _require_gate_contract(
            payload,
            manifest_sha256=manifest_sha256,
            allowed_label_provenance=requested_provenance,
            dataset_payload_sha256_value=evaluation_dataset_payload_sha256,
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    evaluation_started = time.perf_counter()
    model_seconds = 0.0
    component_runtime: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    ocr_image_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {
        "final": [],
        "coarse": [],
        "residual": [],
        "confidence": [],
        "oracle_rgb": [],
        "image": defaultdict(list),
        "oracle_image": defaultdict(list),
        "qwen_gate": [],
        "line": [],
        "edge": [],
        "straightness": [],
        "iterations": defaultdict(list),
        "iteration_straightness": defaultdict(list),
        "iteration_folds": defaultdict(int),
        "iteration_cells": defaultdict(int),
        "geometry": defaultdict(list),
        "folds": 0,
        "cells": 0,
        "samples": 0,
    }
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "final": [],
            "coarse": [],
            "residual": [],
            "confidence": [],
            "oracle_rgb": [],
            "image": defaultdict(list),
            "oracle_image": defaultdict(list),
            "qwen_gate": [],
            "line": [],
            "edge": [],
            "straightness": [],
            "iterations": defaultdict(list),
            "iteration_straightness": defaultdict(list),
            "iteration_folds": defaultdict(int),
            "iteration_cells": defaultdict(int),
            "geometry": defaultdict(list),
            "folds": 0,
            "cells": 0,
            "samples": 0,
        }
    )
    for batch in loader:
        warped = batch["warped_image"].to(device)
        target = batch["backward_map"].to(device).float()
        valid = batch["valid_mask"].to(device).bool()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_started = time.perf_counter()
        output = model(warped, output_size=output_size, render=False, profile=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_seconds += time.perf_counter() - model_started
        for name, value in output.get("runtime_breakdown", {}).items():
            component_runtime[name].append(float(value))
        final_error = endpoint_error_map(output["backward_map"], target)
        coarse_error = endpoint_error_map(output["coarse_backward_map"], target)
        final_values = final_error[valid].detach().cpu()
        coarse_values = coarse_error[valid].detach().cpu()
        target_low, valid_low = resize_backward_map_with_mask(
            target,
            valid,
            output["coarse_low"].shape[-2:],
            source_size_from=tuple(int(value) for value in warped.shape[-2:]),
            source_size_to=tuple(int(value) for value in warped.shape[-2:]),
        )
        residual_target = target_low - output["coarse_low"]
        residual_prediction = output["composition_gate"] * output["residual_proposal"]
        residual_values = endpoint_error_map(
            residual_prediction, residual_target
        )[valid_low].detach().cpu()
        confidence_values = output["confidence"][valid].detach().cpu()
        gate_values = output["qwen_gate"].detach().cpu().flatten()
        structure_batch = {"warped_image": warped, "valid_mask": valid}
        for structure_key in (
            "horizontal_structure",
            "vertical_structure",
            "boundary_structure",
        ):
            if structure_key in batch:
                structure_batch[structure_key] = batch[structure_key].to(device)
        structure = structure_targets(structure_batch, output_size)
        line_mask = valid & (structure[:, 0:2].amax(dim=1, keepdim=True) >= 0.25)
        target_rgb = batch["rectified_image"].to(device).float()
        gray = target_rgb.mean(dim=1, keepdim=True)
        edge_x = torch.nn.functional.pad(
            torch.abs(gray[..., 1:] - gray[..., :-1]), (0, 1, 0, 0)
        )
        edge_y = torch.nn.functional.pad(
            torch.abs(gray[..., 1:, :] - gray[..., :-1, :]), (0, 0, 0, 1)
        )
        edge_response = edge_x + edge_y
        edge_threshold = edge_response.flatten(1).mean(dim=1)[:, None, None, None] * 2.0
        edge_mask = valid & (
            (edge_response >= edge_threshold.clamp_min(0.05))
            | (structure[:, 2:3] >= 0.5)
        )
        line_values = final_error[line_mask].detach().cpu()
        edge_values = final_error[edge_mask].detach().cpu()
        canonical = canonical_backward_map(
            target.shape[0],
            output_size,
            tuple(int(value) for value in warped.shape[-2:]),
            device=device,
            dtype=target.dtype,
        )
        straightness_value = structure_straightness_loss(
            output["backward_map"], canonical, valid, structure
        ).detach().cpu()
        iteration_values: dict[int, Tensor] = {}
        iteration_straightness_values: dict[int, Tensor] = {}
        iteration_fold_counts: dict[int, int] = {}
        iteration_cell_counts: dict[int, int] = {}
        for iteration, current in enumerate(output["refiner_sequence"]):
            current_full = resize_backward_map(
                current,
                output_size,
                source_size_from=tuple(int(value) for value in warped.shape[-2:]),
                source_size_to=tuple(int(value) for value in warped.shape[-2:]),
            )
            iteration_values[iteration] = endpoint_error_map(
                current_full, target
            )[valid].detach().cpu()
            iteration_straightness_values[iteration] = structure_straightness_loss(
                current_full, canonical, valid, structure
            ).detach().cpu()
            current_determinant = jacobian_determinant(current_full)
            current_cells = cell_valid_mask(valid)
            iteration_fold_counts[iteration] = int(
                ((current_determinant <= 0.0) & current_cells).sum().item()
            )
            iteration_cell_counts[iteration] = int(current_cells.sum().item())
        final_rectified = warp_with_backward_map(
            warped.float(), output["backward_map"], padding_mode="border"
        )
        oracle_rectified = warp_with_backward_map(warped.float(), target, padding_mode="border")
        rgb_valid = valid.expand(-1, oracle_rectified.shape[1], -1, -1)
        oracle_rgb_values = torch.abs(
            oracle_rectified - batch["rectified_image"].to(device).float()
        )[rgb_valid].detach().cpu()
        final_image_metrics = {
            name: value.detach().cpu()
            for name, value in image_quality_metrics(
                final_rectified, target_rgb, valid, structure
            ).items()
        }
        oracle_image_metrics = {
            name: value.detach().cpu()
            for name, value in image_quality_metrics(
                oracle_rectified, target_rgb, valid, structure
            ).items()
        }
        determinant = jacobian_determinant(output["backward_map"])
        geometry_metrics = {
            name: value.detach().cpu()
            for name, value in geometry_quality_metrics(
                output["backward_map"], output["canonical_map"], valid
            ).items()
        }
        cell_mask = cell_valid_mask(valid)
        fold_count = int(((determinant <= 0.0) & cell_mask).sum().item())
        cell_count = int(cell_mask.sum().item())
        if export_ocr_images:
            image_name = f"{len(ocr_image_rows):08d}.png"
            model_image_path = destination / "ocr_model_rectified" / image_name
            oracle_image_path = destination / "ocr_oracle_rectified" / image_name
            target_image_path = destination / "ocr_target_rectified" / image_name
            Image.fromarray(_rgb_array(final_rectified)).save(model_image_path)
            Image.fromarray(_rgb_array(oracle_rectified)).save(oracle_image_path)
            Image.fromarray(_rgb_array(target_rgb)).save(target_image_path)
            ocr_image_rows.append(
                {
                    "sample_id": str(batch["sample_id"][0]),
                    "document_id": str(batch["document_id"][0]),
                    "model_image": str(model_image_path),
                    "model_image_sha256": file_sha256(model_image_path),
                    "oracle_image": str(oracle_image_path),
                    "oracle_image_sha256": file_sha256(oracle_image_path),
                    "target_image": str(target_image_path),
                    "target_image_sha256": file_sha256(target_image_path),
                }
            )
        if len(rows) < max_visualizations:
            coarse_rectified = warp_with_backward_map(
                warped.float(), output["coarse_backward_map"], padding_mode="border"
            )
            _save_visual_artifacts(
                destination,
                str(batch["sample_id"][0]),
                warped=warped,
                coarse_rectified=coarse_rectified,
                final_rectified=final_rectified,
                target_rectified=batch["rectified_image"].to(device).float(),
                oracle_rectified=oracle_rectified,
                final_error=final_error,
                fold_mask=(determinant <= 0.0) & cell_mask,
                line_mask=line_mask,
            )
        values = {
            "final": final_values,
            "coarse": coarse_values,
            "residual": residual_values,
            "confidence": confidence_values,
            "oracle_rgb": oracle_rgb_values,
            "qwen_gate": gate_values,
            "line": line_values,
            "edge": edge_values,
        }
        for key, value in values.items():
            aggregate[key].append(value)
        aggregate["folds"] += fold_count
        aggregate["cells"] += cell_count
        aggregate["samples"] += 1
        aggregate["straightness"].append(straightness_value)
        for name, value in final_image_metrics.items():
            aggregate["image"][name].append(value)
        for name, value in oracle_image_metrics.items():
            aggregate["oracle_image"][name].append(value)
        for iteration, value in iteration_values.items():
            aggregate["iterations"][iteration].append(value)
            aggregate["iteration_straightness"][iteration].append(
                iteration_straightness_values[iteration]
            )
            aggregate["iteration_folds"][iteration] += iteration_fold_counts[iteration]
            aggregate["iteration_cells"][iteration] += iteration_cell_counts[iteration]
        for name, value in geometry_metrics.items():
            aggregate["geometry"][name].append(value)
        severity = str(batch["warp_severity"][0])
        provenance = str(batch["label_provenance"][0])
        label_source = str(batch["label_source"][0])
        subset_tags = json.loads(str(batch["subset_tags_json"][0]))
        group_names = [
            f"severity:{severity}",
            f"provenance:{provenance}",
            f"label_source:{label_source}",
        ]
        group_names.extend(
            f"subset:{key}:{value}" for key, value in sorted(subset_tags.items())
        )
        for name in group_names:
            group = grouped[name]
            for key, value in values.items():
                group[key].append(value)
            group["folds"] += fold_count
            group["cells"] += cell_count
            group["samples"] += 1
            group["straightness"].append(straightness_value)
            for metric_name, value in final_image_metrics.items():
                group["image"][metric_name].append(value)
            for metric_name, value in oracle_image_metrics.items():
                group["oracle_image"][metric_name].append(value)
            for iteration, value in iteration_values.items():
                group["iterations"][iteration].append(value)
                group["iteration_straightness"][iteration].append(
                    iteration_straightness_values[iteration]
                )
                group["iteration_folds"][iteration] += iteration_fold_counts[iteration]
                group["iteration_cells"][iteration] += iteration_cell_counts[iteration]
            for metric_name, value in geometry_metrics.items():
                group["geometry"][metric_name].append(value)
        reliable = coarse_values < 1.0
        rows.append(
            {
                "sample_id": batch["sample_id"][0],
                "document_id": batch["document_id"][0],
                "warp_severity": severity,
                "label_provenance": provenance,
                "label_source": label_source,
                "subset_tags": json.dumps(subset_tags, sort_keys=True, ensure_ascii=False),
                "epe": float(final_values.mean()),
                "epe_p95": float(torch.quantile(final_values.float(), 0.95)),
                "coarse_epe": float(coarse_values.mean()),
                "residual_epe": float(residual_values.mean()),
                "final_win_rate": float((final_values < coarse_values).float().mean()),
                "high_confidence_damage_rate": (
                    float(((final_values - coarse_values)[reliable] > 1.0).float().mean())
                    if bool(reliable.any())
                    else float("nan")
                ),
                "fold_rate": fold_count / max(cell_count, 1),
                "qwen_gate_mean": float(gate_values.mean()),
                "line_epe": (
                    float(line_values.float().mean()) if line_values.numel() else float("nan")
                ),
                "edge_epe": (
                    float(edge_values.float().mean()) if edge_values.numel() else float("nan")
                ),
                "straightness_error": float(straightness_value),
                "warr_iteration_epe": json.dumps(
                    {
                        str(index + 1): float(value.float().mean())
                        for index, value in iteration_values.items()
                    },
                    sort_keys=True,
                ),
                "warr_iteration_fold_rate": json.dumps(
                    {
                        str(index + 1): iteration_fold_counts[index]
                        / max(iteration_cell_counts[index], 1)
                        for index in iteration_values
                    },
                    sort_keys=True,
                ),
                "warr_iteration_straightness": json.dumps(
                    {
                        str(index + 1): float(iteration_straightness_values[index])
                        for index in iteration_values
                    },
                    sort_keys=True,
                ),
                "warr_monotonic": all(
                    float(iteration_values[index].float().mean())
                    <= float(iteration_values[index - 1].float().mean()) + 1.0e-6
                    for index in range(1, len(iteration_values))
                ),
                **{name: float(value) for name, value in geometry_metrics.items()},
                **{
                    f"rectified_{name}": float(value)
                    for name, value in final_image_metrics.items()
                },
                **{
                    f"oracle_{name}": float(value)
                    for name, value in oracle_image_metrics.items()
                },
                "valid_pixels": int(final_values.numel()),
            }
        )

    def summarize(value: dict[str, Any]) -> dict[str, Any]:
        return _summary(
            value["final"],
            value["coarse"],
            value["residual"],
            value["confidence"],
            value["oracle_rgb"],
            value["image"],
            value["oracle_image"],
            value["qwen_gate"],
            value["line"],
            value["edge"],
            value["straightness"],
            value["iterations"],
            value["iteration_straightness"],
            value["iteration_folds"],
            value["iteration_cells"],
            value["geometry"],
            value["folds"],
            value["cells"],
            value["samples"],
        )

    ocr_image_export: dict[str, Any] | None = None
    if export_ocr_images:
        ocr_manifest_path = destination / "ocr_images.jsonl"
        with ocr_manifest_path.open("x", encoding="utf-8") as handle:
            for row in ocr_image_rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        ocr_image_export = {
            "schema": "docgrid_flow.ocr_image_export.v1",
            "manifest": str(ocr_manifest_path),
            "manifest_sha256": file_sha256(ocr_manifest_path),
            "samples": len(ocr_image_rows),
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(sorted(row["sample_id"] for row in ocr_image_rows)).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "model_image_directory": str(destination / "ocr_model_rectified"),
            "oracle_image_directory": str(destination / "ocr_oracle_rectified"),
            "target_image_directory": str(destination / "ocr_target_rectified"),
            "image_format": "PNG_RGB8",
        }

    report: dict[str, Any] = {
        "schema": (
            "docgrid_flow.full_gate_evaluation.v3"
            if gate
            else "docgrid_flow.full_exploratory_evaluation.v3"
        ),
        "evaluation_role": "gate" if gate else "exploratory",
        "gate_eligible": bool(gate),
        "runtime_overrides": overrides,
        "coordinate_contract": COORDINATE_CONTRACT,
        "training_stage": payload["training_stage"],
        "training_seed": payload.get("training_seed"),
        "qwen_backend": payload["model_config"].get("qwen_backend", "none"),
        "qwen_vae_decoder_used": False,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256 or file_sha256(checkpoint_path),
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_sha256,
        "evaluation_dataset_payload_sha256": evaluation_dataset_payload_sha256,
        "evaluation_input_work_size": list(input_size),
        "evaluation_output_work_size": list(output_size),
        "checkpoint_input_work_size": [
            int(value) for value in payload["input_work_size"]
        ],
        "checkpoint_output_work_size": [
            int(value) for value in payload["output_work_size"]
        ],
        "work_size_overridden": work_size_overridden,
        "ocr_image_export": ocr_image_export,
        "training_data_contract": payload.get("data_contract"),
        "gate_receipts": payload.get("gate_receipts"),
        "aggregate": summarize(aggregate),
        "groups": {name: summarize(value) for name, value in sorted(grouped.items())},
    }
    protected_report_keys = set(report)
    for name, value in dict(report_metadata or {}).items():
        if name in protected_report_keys:
            raise ValueError(f"report metadata cannot replace protected field {name!r}")
        report[name] = value
    elapsed = time.perf_counter() - evaluation_started
    runtime = {
        "schema": "docgrid_flow.runtime.v2",
        "device": str(device),
        "samples": len(dataset),
        "total_seconds": elapsed,
        "model_seconds": model_seconds,
        "model_seconds_per_page": model_seconds / max(len(dataset), 1),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "docgrid_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "external_qwen_parameter_count": 0,
        "external_teacher_parameter_count": int(
            getattr(model, "external_teacher_parameter_count", 0)
        ),
        "component_seconds_per_page": {
            name: sum(values) / max(len(values), 1)
            for name, values in sorted(component_runtime.items())
        },
    }
    pipeline = getattr(getattr(model, "qwen_source", None), "pipeline", None)
    if pipeline is not None:
        external_ids: set[int] = set()
        external_count = 0
        for component_name in ("transformer", "vae", "text_encoder"):
            component = getattr(pipeline, component_name, None)
            if component is None:
                continue
            for parameter in component.parameters():
                if id(parameter) not in external_ids:
                    external_ids.add(id(parameter))
                    external_count += parameter.numel()
        runtime["external_qwen_parameter_count"] = external_count
    runtime["deployment_parameter_count"] = (
        runtime["docgrid_parameter_count"]
        + runtime["external_qwen_parameter_count"]
        + runtime["external_teacher_parameter_count"]
    )
    report["runtime"] = runtime
    with (destination / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (destination / "confidence_calibration.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema": "docgrid_flow.confidence_calibration.v2",
                "brier_1px": report["aggregate"]["confidence_brier_1px"],
                "ece_1px": report["aggregate"]["confidence_ece_1px"],
                "monotonic_rate": report["aggregate"]["confidence_monotonic_rate"],
                "bins": report["aggregate"]["confidence_reliability"],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with (destination / "runtime.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2, ensure_ascii=False)
    with (destination / "per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--allowed-label-provenance", nargs="+", required=True)
    parser.add_argument("--max-visualizations", type=int, default=32)
    parser.add_argument("--runtime-overrides-json")
    parser.add_argument("--input-work-size", nargs=2, type=int)
    parser.add_argument("--output-work-size", nargs=2, type=int)
    parser.add_argument(
        "--export-ocr-images",
        action="store_true",
        help="Export every model/oracle/target rectified image with SHA-bound JSONL.",
    )
    args = parser.parse_args()
    overrides = None
    if args.runtime_overrides_json:
        with Path(args.runtime_overrides_json).open("r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        if not isinstance(overrides, dict):
            parser.error("--runtime-overrides-json must contain a JSON object")
    report = evaluate_full(
        args.checkpoint,
        args.manifest,
        args.output_dir,
        device_name=args.device,
        allowed_label_provenance=set(args.allowed_label_provenance),
        gate=args.gate,
        max_visualizations=args.max_visualizations,
        runtime_overrides=overrides,
        input_work_size_override=(
            None if args.input_work_size is None else tuple(args.input_work_size)
        ),
        output_work_size_override=(
            None if args.output_work_size is None else tuple(args.output_work_size)
        ),
        export_ocr_images=args.export_ocr_images,
    )
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
