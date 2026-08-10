"""Run a named runtime ablation matrix against one frozen checkpoint/manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from .checkpoint import file_sha256
from .evaluate_full import evaluate_full

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def run_ablation_suite(
    checkpoint: str | Path,
    manifest: str | Path,
    variants_path: str | Path,
    output_dir: str | Path,
    *,
    allowed_label_provenance: set[str],
    device_name: str = "auto",
    max_visualizations: int = 4,
) -> dict[str, Any]:
    variant_file = Path(variants_path).resolve()
    with variant_file.open("r", encoding="utf-8") as handle:
        variants = json.load(handle)
    if not isinstance(variants, dict) or not variants:
        raise ValueError("ablation variants must be a non-empty name-to-object mapping")
    for name, overrides in variants.items():
        if not _SAFE_NAME.fullmatch(str(name)) or not isinstance(overrides, dict):
            raise ValueError(f"invalid ablation variant {name!r}")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite ablation suite: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for name, overrides in variants.items():
        report = evaluate_full(
            checkpoint,
            manifest,
            destination / name,
            device_name=device_name,
            allowed_label_provenance=allowed_label_provenance,
            gate=False,
            max_visualizations=max_visualizations,
            runtime_overrides=overrides,
        )
        aggregate = report["aggregate"]
        row = {
            "variant": name,
            "runtime_overrides": json.dumps(overrides, sort_keys=True),
            **{
                metric: aggregate.get(metric)
                for metric in (
                    "epe",
                    "epe_p95",
                    "line_epe",
                    "edge_epe",
                    "straightness_error",
                    "fold_rate",
                    "final_win_rate",
                    "high_confidence_damage_rate",
                    "local_scale_anomaly_rate",
                )
            },
            "model_seconds_per_page": report["runtime"]["model_seconds_per_page"],
            "peak_cuda_memory_bytes": report["runtime"]["peak_cuda_memory_bytes"],
        }
        rows.append(row)
        reports[name] = {
            "metrics": str((destination / name / "metrics.json").resolve()),
            "aggregate": aggregate,
            "runtime": report["runtime"],
        }
    result = {
        "schema": "docgrid_flow.ablation_suite.v2",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": file_sha256(manifest),
        "variants_file": str(variant_file),
        "variants_file_sha256": file_sha256(variant_file),
        "variants": reports,
    }
    with (destination / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    with (destination / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-label-provenance", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-visualizations", type=int, default=4)
    args = parser.parse_args()
    result = run_ablation_suite(
        args.checkpoint,
        args.manifest,
        args.variants,
        args.output_dir,
        allowed_label_provenance=set(args.allowed_label_provenance),
        device_name=args.device,
        max_visualizations=args.max_visualizations,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

