"""Aggregate at least three immutable evaluation reports for stability evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .checkpoint import file_sha256

_METRICS = (
    "epe",
    "epe_p95",
    "line_epe",
    "edge_epe",
    "straightness_error",
    "fold_rate",
    "final_win_rate",
    "high_confidence_damage_rate",
)


def aggregate_seed_reports(
    evaluation_paths: list[str | Path],
    output_path: str | Path,
    *,
    max_epe_std: float = 0.15,
    max_fold_std: float = 0.001,
    max_win_rate_std: float = 0.03,
) -> dict[str, Any]:
    if len(evaluation_paths) < 3:
        raise ValueError("multi-seed evidence requires at least three evaluations")
    reports: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in evaluation_paths:
        path = Path(raw_path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if not isinstance(report, dict) or not isinstance(report.get("aggregate"), dict):
            raise ValueError(f"invalid evaluation report: {path}")
        reports.append((path, report))
    manifests = {report.get("manifest_sha256") for _, report in reports}
    stages = {report.get("training_stage") for _, report in reports}
    seeds = [report.get("training_seed") for _, report in reports]
    if len(manifests) != 1 or None in manifests:
        raise ValueError("seed evaluations must use one frozen manifest")
    if len(stages) != 1 or None in stages:
        raise ValueError("seed evaluations must use one training stage")
    if any(seed is None for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation reports need distinct checkpoint training_seed values")
    summary: dict[str, dict[str, float]] = {}
    for metric in _METRICS:
        values = [report["aggregate"].get(metric) for _, report in reports]
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        numeric = [float(value) for value in values]
        summary[metric] = {
            "mean": statistics.fmean(numeric),
            "std": statistics.pstdev(numeric),
            "min": min(numeric),
            "max": max(numeric),
        }
    stability_checks = {
        "epe_std": {
            "actual": summary.get("epe", {}).get("std"),
            "maximum": float(max_epe_std),
        },
        "fold_rate_std": {
            "actual": summary.get("fold_rate", {}).get("std"),
            "maximum": float(max_fold_std),
        },
        "final_win_rate_std": {
            "actual": summary.get("final_win_rate", {}).get("std"),
            "maximum": float(max_win_rate_std),
        },
    }
    for value in stability_checks.values():
        value["passed"] = (
            value["actual"] is not None and value["actual"] <= value["maximum"]
        )
    result: dict[str, Any] = {
        "schema": "docgrid_flow.multi_seed_evidence.v2",
        "training_stage": next(iter(stages)),
        "manifest_sha256": next(iter(manifests)),
        "seeds": [int(seed) for seed in seeds],
        "multi_seed_stable": all(
            bool(value["passed"]) for value in stability_checks.values()
        ),
        "stability_checks": stability_checks,
        "metrics": summary,
        "evaluations": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "checkpoint": report.get("checkpoint"),
                "checkpoint_sha256": report.get("checkpoint_sha256"),
                "training_seed": report.get("training_seed"),
            }
            for path, report in reports
        ],
    }
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite seed evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-epe-std", type=float, default=0.15)
    parser.add_argument("--max-fold-std", type=float, default=0.001)
    parser.add_argument("--max-win-rate-std", type=float, default=0.03)
    args = parser.parse_args()
    result = aggregate_seed_reports(
        args.evaluations,
        args.output,
        max_epe_std=args.max_epe_std,
        max_fold_std=args.max_fold_std,
        max_win_rate_std=args.max_win_rate_std,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

