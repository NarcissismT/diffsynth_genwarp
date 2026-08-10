"""Evaluate a deterministic coarse checkpoint with exact pixel aggregation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .checkpoint import COORDINATE_CONTRACT, file_sha256, load_checkpoint
from .config import build_coarse_model
from .data import DocumentMapDataset, dataset_payload_sha256
from .geometry import warp_with_backward_map
from .metrics import cell_valid_mask, endpoint_error_map, jacobian_determinant

GATE_LABEL_PROVENANCE = frozenset({"analytic_gt", "renderer_gt"})


def _calibration(errors: Tensor, confidence: Tensor, bins: int = 10) -> tuple[float, float]:
    if errors.numel() == 0:
        return float("nan"), float("nan")
    probabilities = confidence.float().clamp(0.0, 1.0)
    outcomes = (errors.float() < 1.0).float()
    brier = float((probabilities - outcomes).square().mean())
    ece = probabilities.new_zeros(())
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (probabilities >= boundaries[index]) & (
            probabilities < boundaries[index + 1]
            if index + 1 < bins
            else probabilities <= boundaries[index + 1]
        )
        if bool(selected.any()):
            ece += (
                torch.abs(probabilities[selected].mean() - outcomes[selected].mean())
                * selected.sum()
                / probabilities.numel()
            )
    return brier, float(ece)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _require_gate_contract(
    payload: dict[str, Any],
    *,
    manifest_sha256: str,
    allowed_label_provenance: set[str],
    dataset_payload_sha256_value: str | None = None,
) -> None:
    contract = payload.get("data_contract")
    if not isinstance(contract, dict):
        raise ValueError("gate evaluation requires a checkpoint data_contract")
    if contract.get("val_manifest_sha256") != manifest_sha256:
        raise ValueError(
            "gate manifest SHA-256 differs from the checkpoint's frozen val manifest"
        )
    frozen_allowed = {
        str(value).lower() for value in contract.get("allowed_label_provenance", [])
    }
    requested_allowed = {value.lower() for value in allowed_label_provenance}
    if not requested_allowed or not requested_allowed <= GATE_LABEL_PROVENANCE:
        raise ValueError(
            "gate evaluation only accepts verified analytic_gt/renderer_gt; "
            f"got {sorted(requested_allowed)}"
        )
    if not frozen_allowed or requested_allowed != frozen_allowed:
        raise ValueError(
            "gate provenance allowlist must exactly match the checkpoint data contract"
        )
    required_split_evidence = (
        "train_document_ids_sha256",
        "val_document_ids_sha256",
        "train_document_count",
        "val_document_count",
    )
    if not bool(contract.get("document_disjoint_verified")) or any(
        key not in contract for key in required_split_evidence
    ):
        raise ValueError("gate checkpoint lacks frozen document-disjoint split evidence")
    frozen_contract = contract.get("frozen_contract")
    if not isinstance(frozen_contract, dict):
        raise ValueError("gate checkpoint lacks its Stage-0 frozen contract identity")
    frozen_splits = frozen_contract.get("splits")
    frozen_val = frozen_splits.get("val") if isinstance(frozen_splits, dict) else None
    expected_payload = (
        frozen_val.get("dataset_payload_sha256")
        if isinstance(frozen_val, dict)
        else None
    )
    if not isinstance(expected_payload, str) or len(expected_payload) != 64:
        raise ValueError("gate checkpoint lacks frozen validation payload identity")
    if dataset_payload_sha256_value is None:
        raise ValueError("gate evaluation did not hash validation payload assets")
    if expected_payload != dataset_payload_sha256_value:
        raise ValueError("gate validation payload changed after Stage-0 freeze")


def _summary(
    errors: list[Tensor],
    confidences: list[Tensor],
    oracle_rgb_errors: list[Tensor],
    folds: int,
    cells: int,
    samples: int,
) -> dict[str, float]:
    values = torch.cat(errors).float() if errors else torch.empty(0)
    confidence = torch.cat(confidences).float() if confidences else torch.empty(0)
    rgb_values = (
        torch.cat(oracle_rgb_errors).float() if oracle_rgb_errors else torch.empty(0)
    )
    brier, ece = _calibration(values, confidence)
    return {
        "epe": float(values.mean()) if values.numel() else float("nan"),
        "epe_p95": (
            float(torch.quantile(values, 0.95)) if values.numel() else float("nan")
        ),
        "fold_rate": folds / max(cells, 1),
        "confidence_brier_1px": brier,
        "confidence_ece_1px": ece,
        "worksize_oracle_rgb_l1": (
            float(rgb_values.mean()) if rgb_values.numel() else float("nan")
        ),
        "valid_pixels": float(values.numel()),
        "samples": float(samples),
    }


@torch.no_grad()
def evaluate(
    checkpoint_path: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    allowed_label_provenance: set[str],
    gate: bool = False,
) -> dict[str, Any]:
    device = _device(device_name)
    payload = load_checkpoint(checkpoint_path, map_location=device)
    model = build_coarse_model(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    input_size = tuple(int(v) for v in payload["input_work_size"])
    output_size = tuple(int(v) for v in payload["output_work_size"])
    dataset = DocumentMapDataset(
        manifest,
        input_work_size=input_size,
        output_work_size=output_size,
    )
    if not allowed_label_provenance:
        raise ValueError("allowed_label_provenance must be an explicit non-empty set")
    rejected = sorted(
        {record.label_provenance for record in dataset.records}
        - {value.lower() for value in allowed_label_provenance}
    )
    if rejected:
        raise ValueError(f"evaluation contains unapproved label provenance: {rejected}")
    manifest_sha256 = file_sha256(manifest)
    evaluation_dataset_payload_sha256 = dataset_payload_sha256(dataset.records)
    if gate:
        _require_gate_contract(
            payload,
            manifest_sha256=manifest_sha256,
            allowed_label_provenance=allowed_label_provenance,
            dataset_payload_sha256_value=evaluation_dataset_payload_sha256,
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows: list[dict[str, Any]] = []
    aggregate_errors: list[Tensor] = []
    aggregate_confidences: list[Tensor] = []
    aggregate_oracle_rgb_errors: list[Tensor] = []
    aggregate_folds = 0
    aggregate_cells = 0
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "errors": [],
            "confidences": [],
            "oracle_rgb_errors": [],
            "folds": 0,
            "cells": 0,
            "samples": 0,
        }
    )
    for batch in loader:
        warped = batch["warped_image"].to(device)
        target_map = batch["backward_map"].to(device).float()
        valid = batch["valid_mask"].to(device).bool()
        output = model(warped, output_size=output_size, render=False)
        error = endpoint_error_map(output["backward_map"], target_map)
        values = error[valid].detach().cpu()
        confidence_values = output["confidence"][valid].detach().cpu()
        oracle_rectified = warp_with_backward_map(
            warped.float(),
            target_map,
            padding_mode="border",
        )
        rgb_valid = valid.expand(-1, oracle_rectified.shape[1], -1, -1)
        oracle_rgb_values = torch.abs(
            oracle_rectified - batch["rectified_image"].to(device).float()
        )[rgb_valid].detach().cpu()
        determinant = jacobian_determinant(output["backward_map"])
        cell_mask = cell_valid_mask(valid)
        fold_count = int(((determinant <= 0.0) & cell_mask).sum().item())
        cell_count = int(cell_mask.sum().item())
        severity = str(batch["warp_severity"][0])
        provenance = str(batch["label_provenance"][0])
        label_source = str(batch["label_source"][0])
        for group_name in (
            f"severity:{severity}",
            f"provenance:{provenance}",
            f"label_source:{label_source}",
        ):
            group = grouped[group_name]
            group["errors"].append(values)
            group["confidences"].append(confidence_values)
            group["oracle_rgb_errors"].append(oracle_rgb_values)
            group["folds"] += fold_count
            group["cells"] += cell_count
            group["samples"] += 1
        aggregate_errors.append(values)
        aggregate_confidences.append(confidence_values)
        aggregate_oracle_rgb_errors.append(oracle_rgb_values)
        aggregate_folds += fold_count
        aggregate_cells += cell_count
        sample_brier, sample_ece = _calibration(values, confidence_values)
        rows.append(
            {
                "sample_id": batch["sample_id"][0],
                "document_id": batch["document_id"][0],
                "warp_severity": severity,
                "label_provenance": provenance,
                "label_source": label_source,
                "epe": float(values.mean()) if values.numel() else float("nan"),
                "epe_p95": (
                    float(torch.quantile(values.float(), 0.95))
                    if values.numel()
                    else float("nan")
                ),
                "fold_rate": fold_count / max(cell_count, 1),
                "confidence_brier_1px": sample_brier,
                "confidence_ece_1px": sample_ece,
                "worksize_oracle_rgb_l1": (
                    float(oracle_rgb_values.mean())
                    if oracle_rgb_values.numel()
                    else float("nan")
                ),
                "valid_pixels": int(values.numel()),
            }
        )
    report: dict[str, Any] = {
        "schema": (
            "cp_docflow.coarse_gate_evaluation.v1"
            if gate
            else "cp_docflow.coarse_exploratory_evaluation.v1"
        ),
        "evaluation_role": "gate" if gate else "exploratory",
        "gate_eligible": bool(gate),
        "coordinate_contract": COORDINATE_CONTRACT,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_sha256,
        "evaluation_dataset_payload_sha256": evaluation_dataset_payload_sha256,
        "training_data_contract": payload.get("data_contract"),
        "aggregate": _summary(
            aggregate_errors,
            aggregate_confidences,
            aggregate_oracle_rgb_errors,
            aggregate_folds,
            aggregate_cells,
            len(rows),
        ),
        "groups": {
            name: _summary(
                value["errors"],
                value["confidences"],
                value["oracle_rgb_errors"],
                value["folds"],
                value["cells"],
                value["samples"],
            )
            for name, value in sorted(grouped.items())
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
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
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Require the checkpoint's frozen val-manifest/provenance/split contract.",
    )
    parser.add_argument(
        "--allowed-label-provenance",
        nargs="+",
        required=True,
        help="Fail closed if the manifest contains any other label source.",
    )
    args = parser.parse_args()
    report = evaluate(
        args.checkpoint,
        args.manifest,
        args.output_dir,
        device_name=args.device,
        allowed_label_provenance=set(args.allowed_label_provenance),
        gate=args.gate,
    )
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
