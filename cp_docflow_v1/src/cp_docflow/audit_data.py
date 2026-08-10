"""Freeze and audit verified DocGrid-Flow manifests before Stage-1 training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .checkpoint import COORDINATE_CONTRACT, file_sha256
from .data import (
    DocumentMapDataset,
    assert_document_disjoint,
    dataset_payload_sha256,
)
from .geometry import canonical_backward_map, warp_with_backward_map
from .metrics import cell_valid_mask, endpoint_error_map, jacobian_determinant

_VERIFIED_PROVENANCE = {"analytic_gt", "renderer_gt"}


def _identity_digest(values: set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if not values.numel():
        return {name: float("nan") for name in ("mean", "p50", "p95", "max")}
    value = values.float()
    return {
        "mean": float(value.mean()),
        "p50": float(torch.quantile(value, 0.50)),
        "p95": float(torch.quantile(value, 0.95)),
        "max": float(value.max()),
    }


@torch.no_grad()
def _audit_split(
    name: str,
    dataset: DocumentMapDataset,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    residual_values: list[torch.Tensor] = []
    oracle_values: list[torch.Tensor] = []
    total_valid = 0
    total_pixels = 0
    total_fold = 0
    total_cells = 0
    for index, record in enumerate(dataset.records):
        sample = dataset[index]
        source = sample["warped_image"][None].float()
        target_rgb = sample["rectified_image"][None].float()
        backward_map = sample["backward_map"][None].float()
        valid = sample["valid_mask"][None].bool()
        oracle = warp_with_backward_map(source, backward_map, padding_mode="border")
        rgb_valid = valid.expand(-1, 3, -1, -1)
        rgb_error = torch.abs(oracle - target_rgb)[rgb_valid]
        canonical = canonical_backward_map(
            1,
            backward_map.shape[-2:],
            source.shape[-2:],
            dtype=backward_map.dtype,
        )
        residual = endpoint_error_map(backward_map, canonical)[valid]
        determinant = jacobian_determinant(backward_map)
        cells = cell_valid_mask(valid)
        fold_count = int(((determinant <= 0.0) & cells).sum())
        cell_count = int(cells.sum())
        valid_count = int(valid.sum())
        pixel_count = int(valid.numel())
        residual_values.append(residual.cpu())
        oracle_values.append(rgb_error.cpu())
        total_valid += valid_count
        total_pixels += pixel_count
        total_fold += fold_count
        total_cells += cell_count
        mse = float(rgb_error.square().mean()) if rgb_error.numel() else float("nan")
        psnr = -10.0 * math.log10(max(mse, 1.0e-12))
        rows.append(
            {
                "split": name,
                "sample_id": record.sample_id,
                "document_id": record.document_id,
                "warp_severity": record.warp_severity,
                "label_provenance": record.label_provenance,
                "label_source": record.label_source,
                "subset_tags": json.dumps(
                    dict(record.subset_tags), sort_keys=True, ensure_ascii=False
                ),
                "input_height": record.input_size[0],
                "input_width": record.input_size[1],
                "output_height": record.output_size[0],
                "output_width": record.output_size[1],
                "valid_ratio": valid_count / max(pixel_count, 1),
                "fold_rate": fold_count / max(cell_count, 1),
                "residual_mean_px": float(residual.mean()),
                "residual_p95_px": float(torch.quantile(residual.float(), 0.95)),
                "oracle_rgb_l1": float(rgb_error.mean()),
                "oracle_rgb_psnr": psnr,
            }
        )
    residual_all = torch.cat(residual_values) if residual_values else torch.empty(0)
    oracle_all = torch.cat(oracle_values) if oracle_values else torch.empty(0)
    documents = {record.document_id for record in dataset.records}
    provenance = sorted({record.label_provenance for record in dataset.records})
    subset_counts: dict[str, int] = {}
    for record in dataset.records:
        for key, value in record.subset_tags:
            name = f"{key}:{value}"
            subset_counts[name] = subset_counts.get(name, 0) + 1
    summary = {
        "manifest": str(dataset.manifest.resolve()),
        "manifest_sha256": file_sha256(dataset.manifest),
        "dataset_payload_sha256": dataset_payload_sha256(dataset.records),
        "samples": len(dataset),
        "documents": len(documents),
        "document_ids_sha256": _identity_digest(documents),
        "label_provenance": provenance,
        "subset_counts": dict(sorted(subset_counts.items())),
        "verified_gt_only": bool(provenance)
        and set(provenance) <= _VERIFIED_PROVENANCE,
        "valid_ratio": total_valid / max(total_pixels, 1),
        "fold_rate": total_fold / max(total_cells, 1),
        "residual_px": _quantiles(residual_all),
        "oracle_rgb_l1": _quantiles(oracle_all),
    }
    return summary, rows


def audit_manifests(
    manifests: dict[str, str | Path],
    output_dir: str | Path,
    *,
    allowed_label_provenance: set[str],
    seeds: list[int],
    baseline_checkpoint: str | Path | None = None,
    baseline_config: str | Path | None = None,
    baseline_metrics: str | Path | None = None,
) -> dict[str, Any]:
    requested = {str(value).lower() for value in allowed_label_provenance}
    if not requested:
        raise ValueError("allowed_label_provenance must be non-empty")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique list")
    datasets = {
        name: DocumentMapDataset(Path(path).resolve())
        for name, path in manifests.items()
    }
    assert_document_disjoint(*(dataset.records for dataset in datasets.values()))
    report_splits: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, dataset in datasets.items():
        actual = {record.label_provenance for record in dataset.records}
        rejected = actual - requested
        if rejected:
            raise ValueError(f"{name} contains unapproved provenance: {sorted(rejected)}")
        summary, split_rows = _audit_split(name, dataset)
        report_splits[name] = summary
        rows.extend(split_rows)
    baseline_values = (baseline_checkpoint, baseline_config, baseline_metrics)
    if any(value is not None for value in baseline_values) and not all(
        value is not None for value in baseline_values
    ):
        raise ValueError(
            "baseline_checkpoint, baseline_config, and baseline_metrics must be supplied together"
        )
    baseline: dict[str, Any] | None = None
    if all(value is not None for value in baseline_values):
        baseline = {}
        for name, raw_path in zip(
            ("checkpoint", "config", "metrics"), baseline_values
        ):
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"baseline {name} does not exist: {path}")
            baseline[name] = str(path)
            baseline[f"{name}_sha256"] = file_sha256(path)
        with Path(baseline["metrics"]).open("r", encoding="utf-8") as handle:
            baseline_report = json.load(handle)
        if not isinstance(baseline_report, dict):
            raise ValueError("baseline metrics must contain a JSON object")
        if baseline_report.get("schema") != "docgrid_flow.full_exploratory_evaluation.v3":
            raise ValueError("baseline metrics must be a v3 exploratory evaluation")
        if baseline_report.get("evaluation_role") != "exploratory" or baseline_report.get(
            "gate_eligible"
        ) is not False:
            raise ValueError("frozen prior baseline must not claim gate eligibility")
        if baseline_report.get("training_stage") != "frozen_prior":
            raise ValueError("Stage-0 baseline metrics must come from the frozen prior")
        if baseline_report.get("coordinate_contract") != COORDINATE_CONTRACT:
            raise ValueError("baseline metrics use a different coordinate contract")
        if baseline_report.get("manifest_sha256") != report_splits["val"][
            "manifest_sha256"
        ]:
            raise ValueError("baseline metrics were not evaluated on the frozen val manifest")
        if baseline_report.get("evaluation_dataset_payload_sha256") != report_splits[
            "val"
        ]["dataset_payload_sha256"]:
            raise ValueError("baseline metrics used different validation payload assets")
        if baseline_report.get("checkpoint_sha256") != baseline["checkpoint_sha256"]:
            raise ValueError("baseline metrics checkpoint identity does not match")
        baseline_identity = baseline_report.get("baseline_identity")
        if not isinstance(baseline_identity, dict):
            raise ValueError("baseline metrics lack frozen baseline identity")
        if baseline_identity.get("config_sha256") != baseline["config_sha256"]:
            raise ValueError("baseline metrics config identity does not match")
        if baseline_identity.get("checkpoint_sha256") != baseline[
            "checkpoint_sha256"
        ]:
            raise ValueError("baseline identity checkpoint SHA-256 does not match")
        val_input_sizes = {tuple(record.input_size) for record in datasets["val"].records}
        val_output_sizes = {tuple(record.output_size) for record in datasets["val"].records}
        if len(val_input_sizes) != 1 or len(val_output_sizes) != 1:
            raise ValueError("frozen val manifest must have one input/output canvas size")
        expected_input_size = list(next(iter(val_input_sizes)))
        expected_output_size = list(next(iter(val_output_sizes)))
        if baseline_report.get("evaluation_input_work_size") != expected_input_size:
            raise ValueError("baseline input work size differs from the frozen val manifest")
        if baseline_report.get("evaluation_output_work_size") != expected_output_size:
            raise ValueError("baseline output work size differs from the frozen val manifest")
        baseline["manifest_sha256"] = baseline_report["manifest_sha256"]
        baseline["dataset_payload_sha256"] = baseline_report[
            "evaluation_dataset_payload_sha256"
        ]
        baseline["evaluation_input_work_size"] = expected_input_size
        baseline["evaluation_output_work_size"] = expected_output_size
        baseline["evaluation_schema"] = baseline_report["schema"]
    report: dict[str, Any] = {
        "schema": "docgrid_flow.stage0_data_audit.v2",
        "coordinate_contract": COORDINATE_CONTRACT,
        "document_disjoint_verified": True,
        "allowed_label_provenance": sorted(requested),
        "verified_gt_only": all(
            value["verified_gt_only"] for value in report_splits.values()
        ),
        "seeds": [int(value) for value in seeds],
        "splits": report_splits,
        "baseline": baseline,
    }
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit dir: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "dataset_audit.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (destination / "per_sample.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    frozen = {
        "schema": "docgrid_flow.frozen_data_contract.v2",
        "audit": str(report_path),
        "audit_sha256": file_sha256(report_path),
        "coordinate_contract": COORDINATE_CONTRACT,
        "document_disjoint_verified": True,
        "verified_gt_only": report["verified_gt_only"],
        "seeds": report["seeds"],
        "splits": {
            name: {
                key: value[key]
                for key in (
                    "manifest",
                    "manifest_sha256",
                    "dataset_payload_sha256",
                    "document_ids_sha256",
                    "samples",
                    "documents",
                )
            }
            for name, value in report_splits.items()
        },
        "baseline": baseline,
    }
    with (destination / "frozen_contract.json").open("x", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-label-provenance", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 2027, 3407])
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--baseline-config")
    parser.add_argument("--baseline-metrics")
    args = parser.parse_args()
    manifests = {"train": args.train_manifest, "val": args.val_manifest}
    if args.test_manifest:
        manifests["test"] = args.test_manifest
    result = audit_manifests(
        manifests,
        args.output_dir,
        allowed_label_provenance=set(args.allowed_label_provenance),
        seeds=args.seeds,
        baseline_checkpoint=args.baseline_checkpoint,
        baseline_config=args.baseline_config,
        baseline_metrics=args.baseline_metrics,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
