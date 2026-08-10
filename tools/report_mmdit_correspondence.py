#!/usr/bin/env python3
"""Merge probe shards, select configurations, plot results, and write the report."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from docgrid_flow.analysis.mmdit_correspondence import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    load_config,
    read_manifest,
    stable_sha256,
)


CONFIG_FIELDS = ("layer", "step_index", "rope_state", "temperature")


def iter_jsonl(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield value


_CRITICAL_METRICS = {
    "hard_epe_mean_px",
    "hard_epe_median_px",
    "hard_epe_p95_px",
    "soft_epe_mean_px",
    "soft_epe_median_px",
    "soft_epe_p95_px",
    "normalized_entropy",
    "recall_at_10_r1",
}


@dataclass
class WeightedMetrics:
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    weights: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    sample_sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    sample_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sketches: dict[str, list[tuple[float, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    rows: int = 0
    tokens: int = 0

    def _add_sketch(self, name: str, value: Any) -> None:
        if not isinstance(value, Mapping):
            raise ValueError(f"metric {name} must be a quantile-sketch object")
        means = value.get("means")
        weights = value.get("weights")
        if not isinstance(means, list) or not isinstance(weights, list) or len(means) != len(weights):
            raise ValueError(f"metric {name} has an invalid quantile sketch")
        for mean, weight in zip(means, weights):
            number = float(mean)
            count = int(weight)
            if not math.isfinite(number) or count <= 0:
                raise ValueError(f"metric {name} has invalid centroid {mean!r}/{weight!r}")
            self.sketches[name].append((number, count))

    @staticmethod
    def _sketch_quantile(centroids: Sequence[tuple[float, int]], q: float) -> float:
        ordered = sorted(centroids)
        total = sum(weight for _mean, weight in ordered)
        target = q * max(total - 1, 0)
        cumulative = 0
        for mean, weight in ordered:
            if cumulative + weight > target:
                return mean
            cumulative += weight
        return ordered[-1][0]

    def add(self, metrics: Mapping[str, Any]) -> None:
        weight = int(metrics.get("valid_tokens") or 0)
        if weight <= 0:
            return
        missing = sorted(name for name in _CRITICAL_METRICS if name in metrics and metrics[name] is None)
        if missing:
            raise ValueError(f"non-empty metric row contains null critical metrics: {missing}")
        self.rows += 1
        self.tokens += weight
        for name, value in metrics.items():
            if name.endswith("_epe_px_sketch"):
                self._add_sketch(name, value)
                continue
            if name in {"valid_tokens", "false_identity_tokens"} or value is None:
                continue
            if isinstance(value, bool):
                number = float(value)
            elif isinstance(value, (int, float)):
                number = float(value)
            else:
                continue
            if not math.isfinite(number):
                raise ValueError(f"metric {name} is non-finite: {number}")
            if name.endswith(("_epe_median_px", "_epe_p95_px")):
                self.sample_sums[name] += number
                self.sample_counts[name] += 1
            metric_weight = (
                float(metrics.get("false_identity_tokens") or 0)
                if name == "false_identity_rate"
                else float(weight)
            )
            if metric_weight <= 0:
                continue
            self.sums[name] += number * metric_weight
            self.weights[name] += metric_weight

    def result(self) -> dict[str, Any]:
        result = {
            "rows": self.rows,
            "valid_tokens": self.tokens,
            **{
                name: self.sums[name] / self.weights[name]
                for name in sorted(self.sums)
                if self.weights[name] > 0
            },
        }
        for name, count in self.sample_counts.items():
            if count:
                result[f"{name}_sample_macro_mean"] = self.sample_sums[name] / count
        for sketch_name, centroids in self.sketches.items():
            if not centroids:
                continue
            prefix = sketch_name.removesuffix("_epe_px_sketch")
            result[f"{prefix}_epe_median_px"] = self._sketch_quantile(centroids, 0.50)
            result[f"{prefix}_epe_p95_px"] = self._sketch_quantile(centroids, 0.95)
            result[f"{prefix}_epe_quantile_centroids"] = len(centroids)
        return result


def config_key(row: Mapping[str, Any]) -> tuple[int, int, str, float]:
    return (
        int(row["layer"]),
        int(row["step_index"]),
        str(row["rope_state"]),
        float(row["temperature"]),
    )


def config_mapping(key: tuple[int, int, str, float]) -> dict[str, Any]:
    return dict(zip(CONFIG_FIELDS, key))


def config_label(key: tuple[int, int, str, float]) -> str:
    return f"L{key[0]:02d}/S{key[1]:02d}/{key[2]}/tau={key[3]:.3g}"


def flatten_for_parquet(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    value: dict[str, Any] = {
        "record_type": str(row.get("record_type") or ""),
        "stage": str(row.get("stage") or ""),
        "sample_id": str(row.get("sample_id") or ""),
        "document_id": str(row.get("document_id") or ""),
        "key_sample_id": str(row.get("key_sample_id") or ""),
        "warp_severity": str(row.get("warp_severity") or ""),
        "seed": int(row.get("seed") or 0),
        "rank": int(row.get("rank") or 0),
        "layer": int(row.get("layer") or 0),
        "step_index": int(row.get("step_index") or 0),
        "scheduler_timestep": float(row.get("scheduler_timestep") or 0.0),
        "sigma": float(row.get("sigma") or 0.0),
        "rope_state": str(row.get("rope_state") or ""),
        "temperature": float(row.get("temperature") or 0.0),
        "target_grid": json.dumps(row.get("target_grid")),
        "source_grid": json.dumps(row.get("source_grid")),
        "structure_labels": row.get("structure_labels"),
        "subgroups_json": json.dumps(row.get("subgroups", {}), sort_keys=True),
    }
    for name, metric in metrics.items():
        if name.endswith("_epe_px_sketch"):
            continue
        value[f"metric_{name}"] = float("nan") if metric is None else metric
    return value


class ParquetSink:
    def __init__(self, path: Path, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.rows: list[dict[str, Any]] = []
        self._writer: Any = None
        self._schema: Any = None
        self._batch_size = 4096

    def add(self, row: Mapping[str, Any]) -> None:
        # Baselines have a deliberately smaller schema and live in the
        # aggregate JSON.  The Parquet product is the per-sample Q/K table.
        if self.enabled and row.get("record_type") != "baseline":
            self.rows.append(flatten_for_parquet(row))
            if len(self.rows) >= self._batch_size:
                self._flush()

    def _flush(self) -> None:
        if not self.rows:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("write_parquet=true requires pyarrow") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._writer is None:
            table = pa.Table.from_pylist(self.rows)
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self.path, self._schema, compression="zstd"
            )
        else:
            assert self._schema is not None
            normalized = [
                {name: row.get(name) for name in self._schema.names}
                for row in self.rows
            ]
            table = pa.Table.from_pylist(normalized, schema=self._schema)
        self._writer.write_table(table)
        self.rows.clear()

    def close(self) -> str | None:
        if not self.enabled:
            return None
        self._flush()
        if self._writer is None:
            raise RuntimeError("cannot write an empty per-sample Parquet table")
        self._writer.close()
        return str(self.path)


def validate_shards(
    stage_dir: Path, expected_ranks: int, *, require_all: bool
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths: list[Path] = []
    metadata: list[dict[str, Any]] = []
    missing: list[int] = []
    for rank in range(expected_ranks):
        done = stage_dir / f"done_rank{rank:03d}.json"
        metrics = stage_dir / f"metrics_rank{rank:03d}.jsonl"
        meta = stage_dir / f"metadata_rank{rank:03d}.json"
        if not (done.is_file() and metrics.is_file() and meta.is_file()):
            missing.append(rank)
            continue
        paths.append(metrics)
        metadata.append(json.loads(meta.read_text(encoding="utf-8")))
    if missing and require_all:
        raise RuntimeError(f"stage {stage_dir.name} is missing completed ranks {missing}")
    if not paths:
        raise RuntimeError(f"no completed metric shards in {stage_dir}")
    fingerprints = {
        (
            item.get("model_id"),
            item.get("revision"),
            item.get("pipeline_class"),
            item.get("lora_checkpoint_sha256"),
            item.get("lora_rank"),
            item.get("lora_alpha"),
            item.get("lora_tensor_count"),
            item.get("prompt_sha256"),
            item.get("config_sha256"),
            tuple(item.get("resolved_model_commit_candidates") or []),
            item.get("runtime_core_sha256"),
            item.get("scheduler_config_sha256"),
            item.get("transformer_config_sha256"),
            item.get("block_count"),
        )
        for item in metadata
    }
    if len(fingerprints) != 1:
        raise RuntimeError(f"rank runtime fingerprints differ: {fingerprints}")
    schedules = {item.get("scheduler_trace_sha256") for item in metadata}
    if len(schedules) != 1:
        raise RuntimeError(f"rank scheduler traces differ: {schedules}")
    return paths, metadata


def aggregate_stage(
    stage: str,
    stage_dir: Path,
    paths: Sequence[Path],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    configurations: dict[str, dict[tuple[int, int, str, float], WeightedMetrics]] = {
        "normal": defaultdict(WeightedMetrics),
        "batch_shuffle": defaultdict(WeightedMetrics),
        "determinism_repeat": defaultdict(WeightedMetrics),
    }
    subgroups: dict[
        str, dict[tuple[int, int, str, float], dict[str, WeightedMetrics]]
    ] = {
        name: defaultdict(lambda: defaultdict(WeightedMetrics))
        for name in configurations
    }
    severities: dict[
        str, dict[tuple[int, int, str, float], dict[str, WeightedMetrics]]
    ] = {
        name: defaultdict(lambda: defaultdict(WeightedMetrics))
        for name in configurations
    }
    baselines: dict[str, WeightedMetrics] = defaultdict(WeightedMetrics)
    baseline_severity: dict[str, dict[str, WeightedMetrics]] = defaultdict(
        lambda: defaultdict(WeightedMetrics)
    )
    repeat_rows: dict[
        tuple[str, int, int, str, float], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    feature_row_counts: dict[
        tuple[str, str, int, int, str, float], int
    ] = defaultdict(int)
    parquet = ParquetSink(
        stage_dir / "per_sample_metrics.parquet",
        enabled=bool(config["output"].get("write_parquet", True)),
    )
    row_count = 0
    sample_ids: set[str] = set()
    for row in iter_jsonl(paths):
        row_count += 1
        sample_ids.add(str(row.get("sample_id")))
        parquet.add(row)
        record_type = str(row.get("record_type"))
        metrics = row.get("metrics", {})
        if record_type == "baseline":
            baseline = str(row["baseline"])
            baselines[baseline].add(metrics)
            baseline_severity[baseline][str(row.get("warp_severity", "unknown"))].add(metrics)
            continue
        if record_type not in configurations:
            raise ValueError(f"unknown record_type={record_type!r}")
        if int(metrics.get("valid_tokens") or 0) > 0:
            missing_critical = sorted(
                name
                for name in _CRITICAL_METRICS
                if name not in metrics or metrics[name] is None
            )
            if missing_critical:
                raise ValueError(
                    f"{record_type} row for sample={row.get('sample_id')} has "
                    f"missing/null critical metrics: {missing_critical}"
                )
        key = config_key(row)
        feature_identity = (
            record_type,
            str(row["sample_id"]),
            int(row["layer"]),
            int(row["step_index"]),
            str(row["rope_state"]),
            float(row["temperature"]),
        )
        feature_row_counts[feature_identity] += 1
        configurations[record_type][key].add(metrics)
        severity = str(row.get("warp_severity", "unknown"))
        severities[record_type][key][severity].add(metrics)
        for subgroup_name, subgroup_metrics in row.get("subgroups", {}).items():
            subgroups[record_type][key][str(subgroup_name)].add(subgroup_metrics)
        if stage == "sanity" and record_type in {"normal", "determinism_repeat"}:
            repeat_key = (
                str(row["sample_id"]),
                int(row["layer"]),
                int(row["step_index"]),
                str(row["rope_state"]),
                float(row["temperature"]),
            )
            repeat_rows[repeat_key][record_type] = dict(metrics)
    parquet_path = parquet.close()

    def encode_config_metrics(
        record_type: str,
        values: Mapping[tuple[int, int, str, float], WeightedMetrics],
    ) -> list[dict[str, Any]]:
        encoded: list[dict[str, Any]] = []
        for key in sorted(values):
            metrics = values[key].result()
            hard_groups = [
                severities[record_type][key][name].result()
                for name in ("hard", "extreme")
                if name in severities[record_type][key]
            ]
            hard_tokens = sum(int(group.get("valid_tokens") or 0) for group in hard_groups)
            if hard_tokens:
                metric_name = "recall_at_10_r1"
                metrics["hard_extreme_recall_at_10_r1"] = sum(
                    float(group[metric_name]) * int(group["valid_tokens"])
                    for group in hard_groups
                    if group.get(metric_name) is not None
                ) / hard_tokens
            encoded.append(
                {**config_mapping(key), "label": config_label(key), "metrics": metrics}
            )
        return encoded

    aggregate = {
        "format_version": 1,
        "stage": stage,
        "row_count": row_count,
        "sample_count": len(sample_ids),
        "aggregation_note": (
            "Rates/means are valid-token weighted. EPE median/P95 are token-micro "
            "pooled estimates from mergeable equal-mass centroids; explicit "
            "*_sample_macro_mean fields summarize the sample-level quantiles."
        ),
        "normal": encode_config_metrics("normal", configurations["normal"]),
        "batch_shuffle": encode_config_metrics(
            "batch_shuffle", configurations["batch_shuffle"]
        ),
        "determinism_repeat": encode_config_metrics(
            "determinism_repeat", configurations["determinism_repeat"]
        ),
        "baselines": {name: value.result() for name, value in sorted(baselines.items())},
        "parquet": parquet_path,
    }
    subgroup_payload = {
        record_type: [
            {
                **config_mapping(key),
                "label": config_label(key),
                "subgroups": {
                    name: accumulator.result()
                    for name, accumulator in sorted(groups.items())
                },
                "severities": {
                    name: accumulator.result()
                    for name, accumulator in sorted(severities[record_type][key].items())
                },
            }
            for key, groups in sorted(subgroups[record_type].items())
        ]
        for record_type in configurations
    }
    subgroup_payload["baseline_severities"] = {
        baseline: {
            severity: accumulator.result()
            for severity, accumulator in sorted(values.items())
        }
        for baseline, values in sorted(baseline_severity.items())
    }
    atomic_write_json(stage_dir / "subgroup_metrics.json", subgroup_payload)
    if stage == "sanity":
        mismatches: list[dict[str, Any]] = []
        missing_repeat: list[str] = []
        tolerance = 1.0e-7
        for key, variants in repeat_rows.items():
            if set(variants) != {"normal", "determinism_repeat"}:
                missing_repeat.append(str(key))
                continue
            left, right = variants["normal"], variants["determinism_repeat"]
            names = sorted(set(left) | set(right))
            for name in names:
                a, b = left.get(name), right.get(name)
                if a is None or b is None:
                    if a != b:
                        mismatches.append({"key": key, "metric": name, "a": a, "b": b})
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if abs(float(a) - float(b)) > tolerance:
                        mismatches.append({"key": key, "metric": name, "a": a, "b": b})
        shuffle_present = bool(aggregate["batch_shuffle"])
        gate_config = config.get("sanity_gate", {})
        shuffle_config = gate_config.get("shuffle", {})
        primary = str(shuffle_config.get("primary_metric", "recall_at_10_r1"))
        finite_normal = [
            row
            for row in aggregate["normal"]
            if row.get("metrics", {}).get(primary) is not None
        ]
        best_normal = (
            max(finite_normal, key=lambda row: float(row["metrics"][primary]))
            if finite_normal
            else None
        )
        matched_shuffle = (
            None
            if best_normal is None
            else _lookup_config(aggregate["batch_shuffle"], best_normal)
        )
        normal_value = (
            None if best_normal is None else float(best_normal["metrics"][primary])
        )
        shuffle_value = (
            None
            if matched_shuffle is None
            else matched_shuffle.get("metrics", {}).get(primary)
        )
        shuffle_value = None if shuffle_value is None else float(shuffle_value)
        absolute_drop = (
            None
            if normal_value is None or shuffle_value is None
            else normal_value - shuffle_value
        )
        ratio = (
            None
            if normal_value in (None, 0.0) or shuffle_value is None
            else shuffle_value / normal_value
        )
        max_ratio = float(shuffle_config.get("max_ratio_to_normal", 0.5))
        min_drop = float(shuffle_config.get("min_absolute_drop", 0.05))
        degradation_ok = bool(
            absolute_drop is not None
            and ratio is not None
            and absolute_drop >= min_drop
            and ratio <= max_ratio
        )
        degradation_required = bool(
            gate_config.get("require_shuffle_degradation", True)
        )
        missing_feature_rows: list[str] = []
        duplicate_feature_rows: list[str] = []
        sanity_split = config.get("data", {}).get("splits", {}).get("sanity")
        coverage_required = bool(gate_config.get("require_complete_lattice", True))
        coverage_checked = sanity_split is not None
        expected_feature_rows = 0
        if coverage_checked:
            expected_samples = {
                sample.sample_id for sample in read_manifest(Path(str(sanity_split)))
            }
            sanity_probe = config["probe"]["sanity"]
            expected_record_types = {"normal"}
            if bool(sanity_probe.get("determinism_repeat", False)):
                expected_record_types.add("determinism_repeat")
            if bool(sanity_probe.get("batch_shuffle", False)):
                expected_record_types.add("batch_shuffle")
            expected_keys = {
                (
                    record_type,
                    sample_id,
                    int(layer),
                    int(step),
                    str(rope),
                    float(temperature),
                )
                for record_type in expected_record_types
                for sample_id in expected_samples
                for layer in sanity_probe["layers"]
                for step in sanity_probe["steps"]
                for rope in config["probe"]["rope_states"]
                for temperature in config["probe"]["temperatures"]
            }
            expected_feature_rows = len(expected_keys)
            missing_feature_rows = [
                str(key) for key in sorted(expected_keys - set(feature_row_counts))
            ]
            unexpected = set(feature_row_counts) - expected_keys
            missing_feature_rows.extend(
                f"unexpected:{key}" for key in sorted(unexpected)
            )
            duplicate_feature_rows = [
                f"{key}:count={count}"
                for key, count in sorted(feature_row_counts.items())
                if count != 1
            ]
        coverage_ok = bool(
            (coverage_checked and not missing_feature_rows and not duplicate_feature_rows)
            or not coverage_required
        )
        gate = {
            "pass": bool(
                not mismatches
                and not missing_repeat
                and shuffle_present
                and coverage_ok
                and (degradation_ok or not degradation_required)
            ),
            "determinism_tolerance": tolerance,
            "determinism_mismatches": mismatches[:100],
            "missing_repeat_pairs": missing_repeat[:100],
            "batch_shuffle_present": shuffle_present,
            "lattice_coverage_required": coverage_required,
            "lattice_coverage_checked": coverage_checked,
            "lattice_coverage_pass": coverage_ok,
            "expected_feature_rows": expected_feature_rows,
            "observed_feature_rows": len(feature_row_counts),
            "missing_or_unexpected_feature_rows": missing_feature_rows[:100],
            "duplicate_feature_rows": duplicate_feature_rows[:100],
            "shuffle_degradation_required": degradation_required,
            "shuffle_degradation_pass": degradation_ok,
            "shuffle_primary_metric": primary,
            "shuffle_normal_value": normal_value,
            "shuffle_value": shuffle_value,
            "shuffle_ratio_to_normal": ratio,
            "shuffle_absolute_drop": absolute_drop,
            "shuffle_max_ratio_to_normal": max_ratio,
            "shuffle_min_absolute_drop": min_drop,
            "finite_json_metrics": True,
        }
        atomic_write_json(stage_dir / "sanity_gate.json", gate)
        aggregate["sanity_gate"] = gate
    atomic_write_json(stage_dir / "aggregate_metrics.json", aggregate)
    return {"aggregate": aggregate, "subgroups": subgroup_payload}


def _metric(entry: Mapping[str, Any], name: str, default: float) -> float:
    value = entry.get("metrics", {}).get(name)
    return default if value is None else float(value)


def _selection_sort_key(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[float, ...]:
    """Apply the same preregistered ranking to Discovery and Confirmation."""

    primary = str(config["selection"]["primary_metric"])
    fields = [primary, *config["selection"].get("tie_breakers", [])]
    maximize = {primary, "hard_extreme_recall_at_10_r1"}
    result: list[float] = []
    for name in fields:
        value = row.get("metrics", {}).get(str(name))
        if value is None:
            raise ValueError(
                f"selection metric {name!r} is missing for {config_label(config_key(row))}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"selection metric {name!r} is non-finite: {number}")
        result.append(-number if name in maximize else number)
    return tuple(result)


def select_discovery_configs(
    aggregate: Mapping[str, Any], config: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    rows = list(aggregate["normal"])
    if not rows:
        raise RuntimeError("discovery has no normal feature rows")
    primary = str(config["selection"]["primary_metric"])
    by_temperature: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_temperature[float(row["temperature"])].append(row)
    temperature_scores: list[
        tuple[tuple[float, ...], float, Mapping[str, Any]]
    ] = []
    for temperature, values in by_temperature.items():
        best_at_temperature = min(
            values, key=lambda item: _selection_sort_key(item, config)
        )
        temperature_scores.append(
            (
                _selection_sort_key(best_at_temperature, config),
                temperature,
                best_at_temperature,
            )
        )
    _score, selected_temperature, _best_temperature_row = min(
        temperature_scores, key=lambda value: (value[0], value[1])
    )
    candidates = [
        row for row in rows if math.isclose(float(row["temperature"]), selected_temperature)
    ]
    candidates.sort(key=lambda item: _selection_sort_key(item, config))
    max_configs = int(config["selection"].get("max_configs", 6))
    max_layer = max(int(row["layer"]) for row in candidates)

    def depth_region(row: Mapping[str, Any]) -> int:
        return min(2, (3 * int(row["layer"])) // max(max_layer + 1, 1))

    selected: list[Mapping[str, Any]] = []
    if candidates:
        selected.append(candidates[0])
    required_regions = int(config["selection"].get("min_depth_regions", 2))
    for row in candidates[1:]:
        if len({depth_region(value) for value in selected}) >= required_regions:
            break
        if depth_region(row) not in {depth_region(value) for value in selected}:
            selected.append(row)
    for row in candidates:
        identity = (row["layer"], row["step_index"], row["rope_state"])
        if any(
            identity == (value["layer"], value["step_index"], value["rope_state"])
            for value in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= max_configs:
            break
    anchor_layers = {int(value) for value in config["selection"].get("anchor_layers", [])}
    anchor = next((row for row in candidates if int(row["layer"]) in anchor_layers), None)
    if anchor is not None:
        anchor_identity = (anchor["layer"], anchor["step_index"], anchor["rope_state"])
        selected_identities = {
            (row["layer"], row["step_index"], row["rope_state"]) for row in selected
        }
        if anchor_identity not in selected_identities:
            if len(selected) < max_configs:
                selected.append(anchor)
            else:
                replacement: list[Mapping[str, Any]] | None = None
                for index in range(len(selected) - 1, -1, -1):
                    trial = [*selected]
                    trial[index] = anchor
                    if len({depth_region(row) for row in trial}) >= required_regions:
                        replacement = trial
                        break
                if replacement is None:
                    raise RuntimeError(
                        "cannot retain an anchor configuration without violating "
                        f"min_depth_regions={required_regions}"
                    )
                selected = replacement
    selected = selected[:max_configs]
    selected_regions = {depth_region(row) for row in selected}
    if len(selected_regions) < required_regions:
        raise RuntimeError(
            f"selected configs cover {len(selected_regions)} depth regions, "
            f"but {required_regions} are required"
        )

    def minimal(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "layer": int(row["layer"]),
            "step_index": int(row["step_index"]),
            "rope_state": str(row["rope_state"]),
            "temperature": float(row["temperature"]),
            "discovery_metrics": row["metrics"],
        }

    payload = {
        "format_version": 1,
        "selection_source": "discovery_v1_only",
        "primary_metric": primary,
        "global_temperature": selected_temperature,
        "temperature_selection": [
            {
                "temperature": temperature,
                "best_config": minimal(row),
                "selection_sort_key": list(score),
            }
            for score, temperature, row in sorted(
                temperature_scores, key=lambda value: value[1]
            )
        ],
        "configs": [minimal(row) for row in selected],
        "seed_configs": [
            minimal(row)
            for row in selected[: int(config["seed_stability"].get("top_configs", 3))]
        ],
        "anchor_config": None if anchor is None else minimal(anchor),
        "depth_regions": sorted({depth_region(row) for row in selected}),
        "discovery_aggregate_sha256": stable_sha256(aggregate),
    }
    atomic_write_json(run_dir / "selected_configs.json", payload)
    return payload


def update_seed_configs_from_confirmation(
    aggregate: Mapping[str, Any], config: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    path = run_dir / "selected_configs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary = str(config["selection"]["primary_metric"])
    rows = sorted(
        aggregate["normal"],
        key=lambda row: _selection_sort_key(row, config),
    )
    count = int(config["seed_stability"].get("top_configs", 3))
    payload["seed_configs"] = [
        {
            "layer": int(row["layer"]),
            "step_index": int(row["step_index"]),
            "rope_state": str(row["rope_state"]),
            "temperature": float(row["temperature"]),
            "confirmation_metrics": row["metrics"],
        }
        for row in rows[:count]
    ]
    payload["seed_selection_source"] = "confirmation_v1_top3_without_reselection"
    payload["confirmation_aggregate_sha256"] = stable_sha256(aggregate)
    atomic_write_json(path, payload)
    return payload


def plot_discovery_heatmaps(
    aggregate: Mapping[str, Any], stage_dir: Path, selected_temperature: float
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in aggregate["normal"]
        if math.isclose(float(row["temperature"]), float(selected_temperature))
    ]
    if not rows:
        return []
    layers = sorted({int(row["layer"]) for row in rows})
    steps = sorted({int(row["step_index"]) for row in rows})
    layer_to_index = {value: index for index, value in enumerate(layers)}
    step_to_index = {value: index for index, value in enumerate(steps)}
    metrics = (
        ("recall_at_10_r1", "R@10, r=1", "viridis"),
        ("soft_epe_median_px", "Median soft EPE (px)", "magma_r"),
        ("normalized_entropy", "Normalized entropy", "viridis_r"),
        ("false_identity_rate", "False identity rate", "magma_r"),
    )
    output = stage_dir / "heatmaps"
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for metric_name, title, color_map in metrics:
        all_values = [
            float(row["metrics"][metric_name])
            for row in rows
            if row["metrics"].get(metric_name) is not None
        ]
        shared_min = min(all_values) if all_values else None
        shared_max = max(all_values) if all_values else None
        if metric_name in {"recall_at_10_r1", "normalized_entropy", "false_identity_rate"}:
            shared_min, shared_max = 0.0, 1.0
        for rope in ("pre", "post"):
            rope_rows = [row for row in rows if row["rope_state"] == rope]
            matrix = np.full((len(layers), len(steps)), np.nan, dtype=np.float32)
            for row in rope_rows:
                value = row["metrics"].get(metric_name)
                if value is not None:
                    matrix[layer_to_index[int(row["layer"])], step_to_index[int(row["step_index"])]] = value
            figure, axis = plt.subplots(figsize=(12, max(5, len(layers) * 0.18)))
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                vmin=shared_min,
                vmax=shared_max,
            )
            axis.set_xticks(range(len(steps)), labels=steps)
            tick_step = max(1, len(layers) // 20)
            tick_positions = list(range(0, len(layers), tick_step))
            axis.set_yticks(tick_positions, labels=[layers[index] for index in tick_positions])
            axis.set_xlabel("Denoising step index")
            axis.set_ylabel("MMDiT block")
            axis.set_title(f"{title} ({rope}-RoPE, tau={selected_temperature:g})")
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            path = output / f"{metric_name}_{rope}.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            written.append(str(path))
    return written


def plot_confirmation_subgroups(
    aggregate: Mapping[str, Any],
    subgroup_payload: Mapping[str, Any],
    stage_dir: Path,
    config: Mapping[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary_metric = str(config["selection"]["primary_metric"])
    rows = sorted(
        aggregate["normal"],
        key=lambda row: _selection_sort_key(row, config),
    )
    if not rows:
        return []
    best = rows[0]
    subgroup = _lookup_config(subgroup_payload["normal"], best)
    if subgroup is None:
        return []
    plots = stage_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for group_name, groups in (
        ("severity", subgroup["severities"]),
        ("structure_and_displacement", subgroup["subgroups"]),
    ):
        labels = [name.replace("structure/", "").replace("displacement/", "") for name in groups]
        recalls = [groups[name].get(primary_metric) for name in groups]
        epes = [groups[name].get("soft_epe_median_px") for name in groups]
        positions = np.arange(len(labels))
        figure, recall_axis = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
        recall_values = [float("nan") if value is None else value for value in recalls]
        epe_values = [float("nan") if value is None else value for value in epes]
        recall_axis.bar(positions - 0.2, recall_values, 0.4, label="R@10,r=1", color="#2a9d8f")
        recall_axis.set_ylabel("Recall")
        recall_axis.set_ylim(0, 1)
        recall_axis.set_xticks(positions, labels=labels, rotation=35, ha="right")
        epe_axis = recall_axis.twinx()
        epe_axis.bar(positions + 0.2, epe_values, 0.4, label="soft median EPE", color="#e76f51")
        epe_axis.set_ylabel("EPE (px)")
        handles_a, labels_a = recall_axis.get_legend_handles_labels()
        handles_b, labels_b = epe_axis.get_legend_handles_labels()
        recall_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")
        recall_axis.set_title(
            f"Best confirmation config: L{best['layer']} S{best['step_index']} {best['rope_state']}"
        )
        figure.tight_layout()
        path = plots / f"best_{group_name}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(str(path))
    return written


def _lookup_config(
    rows: Sequence[Mapping[str, Any]], config_value: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    for row in rows:
        if all(
            (
                math.isclose(float(row[field]), float(config_value[field]))
                if field == "temperature"
                else row[field] == config_value[field]
            )
            for field in CONFIG_FIELDS
        ):
            return row
    return None


def _combine_metric_groups(
    groups: Mapping[str, Mapping[str, Any]], names: Sequence[str], metric: str
) -> float | None:
    numerator = 0.0
    denominator = 0
    for name in names:
        group = groups.get(name)
        if not group or group.get(metric) is None:
            continue
        weight = int(group.get("valid_tokens") or 0)
        numerator += float(group[metric]) * weight
        denominator += weight
    return None if denominator == 0 else numerator / denominator


def seed_statistics(
    aggregate: Mapping[str, Any], stage_dir: Path
) -> dict[str, Any]:
    # Re-read the compact rows to preserve per-seed separation.
    paths = sorted(stage_dir.glob("metrics_rank*.jsonl"))
    per_seed: dict[tuple[int, int, str, float], dict[int, WeightedMetrics]] = defaultdict(
        lambda: defaultdict(WeightedMetrics)
    )
    for row in iter_jsonl(paths):
        if row.get("record_type") != "normal":
            continue
        per_seed[config_key(row)][int(row["seed"])].add(row["metrics"])
    configs: list[dict[str, Any]] = []
    rankings_by_seed: dict[int, list[tuple[int, int, str, float]]] = defaultdict(list)
    for key, seeds in sorted(per_seed.items()):
        values = {seed: accumulator.result() for seed, accumulator in sorted(seeds.items())}
        recalls = [value.get("recall_at_10_r1") for value in values.values()]
        epes = [value.get("soft_epe_median_px") for value in values.values()]
        recalls = [float(value) for value in recalls if value is not None]
        epes = [float(value) for value in epes if value is not None]
        configs.append(
            {
                **config_mapping(key),
                "per_seed": values,
                "recall_at_10_r1_mean": statistics.fmean(recalls) if recalls else None,
                "recall_at_10_r1_std": statistics.pstdev(recalls) if len(recalls) > 1 else 0.0,
                "soft_epe_median_px_mean": statistics.fmean(epes) if epes else None,
                "soft_epe_median_px_std": statistics.pstdev(epes) if len(epes) > 1 else 0.0,
            }
        )
    for seed in sorted({seed for values in per_seed.values() for seed in values}):
        rankings_by_seed[seed] = sorted(
            [key for key, values in per_seed.items() if seed in values],
            key=lambda key: -float(per_seed[key][seed].result().get("recall_at_10_r1") or -1),
        )
    rank_consistency: list[dict[str, Any]] = []
    seeds = sorted(rankings_by_seed)
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            left = rankings_by_seed[left_seed]
            right = rankings_by_seed[right_seed]
            common = [key for key in left if key in right]
            if len(common) < 2:
                correlation = None
            else:
                left_rank = {key: index for index, key in enumerate(left)}
                right_rank = {key: index for index, key in enumerate(right)}
                square = sum((left_rank[key] - right_rank[key]) ** 2 for key in common)
                correlation = 1.0 - 6.0 * square / (len(common) * (len(common) ** 2 - 1))
            rank_consistency.append(
                {"seed_a": left_seed, "seed_b": right_seed, "spearman": correlation}
            )
    detail_groups: dict[tuple[str, int, int, str], dict[int, np.ndarray]] = defaultdict(dict)
    for path in stage_dir.glob("token_details/rank_*/*.npz"):
        with np.load(path) as payload:
            key = (
                str(payload["sample_id"].item()),
                int(payload["layer"].item()),
                int(payload["step_index"].item()),
                str(payload["rope_state"].item()),
            )
            detail_groups[key][int(payload["seed"].item())] = payload["top1_xy"].copy()
    variances: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for (_sample, layer, step, rope), values in detail_groups.items():
        if len(values) < 2:
            continue
        arrays = list(values.values())
        if len({array.shape for array in arrays}) != 1:
            continue
        coordinate_variance = np.stack(arrays, axis=0).var(axis=0).sum(axis=-1).mean()
        variances[(layer, step, rope)].append(float(coordinate_variance))
    for value in configs:
        key = (int(value["layer"]), int(value["step_index"]), str(value["rope_state"]))
        samples = variances.get(key, [])
        value["top1_coordinate_variance_token2"] = (
            statistics.fmean(samples) if samples else None
        )
    return {"configs": configs, "rank_consistency": rank_consistency}


def generate_qualitative_visuals(
    confirmation_dir: Path,
    output_dir: Path,
    best: Mapping[str, Any],
    limit: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    candidates: list[Path] = []
    for path in confirmation_dir.glob("artifacts/rank_*/*.npz"):
        with np.load(path) as payload:
            if (
                int(payload["layer"].item()) == int(best["layer"])
                and int(payload["step_index"].item()) == int(best["step_index"])
                and str(payload["rope_state"].item()) == str(best["rope_state"])
            ):
                candidates.append(path)
    candidates.sort()
    seen: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for path in candidates:
        with np.load(path) as payload:
            sample_id = str(payload["sample_id"].item())
            if sample_id in seen:
                continue
            seen.add(sample_id)
            source_grid = tuple(int(value) for value in payload["source_grid"])
            target_grid = tuple(int(value) for value in payload["target_grid"])
            artifact_positions = payload["artifact_query_positions"]
            target_indices = payload["target_indices"]
            gt = payload["gt_source_token_xy"]
            identity = payload["identity_source_token_xy"]
            top_indices = payload["top_indices"]
            cost = payload["cost"]
            local_position = int(artifact_positions[0])
            cost_map = cost[0].reshape(source_grid)
            ranked = top_indices[local_position]
            top1 = int(ranked[0])
            top1_xy = np.asarray([top1 % source_grid[1], top1 // source_grid[1]])
            figure, axis = plt.subplots(figsize=(6, 5))
            image = axis.imshow(cost_map, cmap="viridis")
            axis.scatter(gt[local_position, 0], gt[local_position, 1], c="lime", marker="x", s=80, label="GT")
            axis.scatter(top1_xy[0], top1_xy[1], c="red", marker="+", s=80, label="top-1")
            top5 = ranked[1:5]
            top10 = ranked[5:10]
            if len(top5):
                axis.scatter(
                    top5 % source_grid[1],
                    top5 // source_grid[1],
                    facecolors="none",
                    edgecolors="orange",
                    s=45,
                    label="top-2..5",
                )
            if len(top10):
                axis.scatter(
                    top10 % source_grid[1],
                    top10 // source_grid[1],
                    facecolors="none",
                    edgecolors="yellow",
                    s=30,
                    label="top-6..10",
                )
            axis.scatter(identity[local_position, 0], identity[local_position, 1], c="white", marker="o", s=35, label="identity")
            from matplotlib.patches import Rectangle

            gt_round = np.round(gt[local_position])
            axis.add_patch(
                Rectangle(
                    (gt_round[0] - 1.5, gt_round[1] - 1.5),
                    3,
                    3,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.2,
                    linestyle="--",
                    label="GT r=1",
                )
            )
            axis.set_title(f"{sample_id}: cost map")
            axis.legend(loc="upper right")
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            cost_path = output_dir / f"{_safe_file(sample_id)}_cost.png"
            figure.savefig(cost_path, dpi=170)
            plt.close(figure)

            rectified_path = str(payload["rectified_image"].item())
            warped_path = str(payload["warped_image"].item())
            if not (
                rectified_path
                and Path(rectified_path).is_file()
                and Path(warped_path).is_file()
            ):
                raise FileNotFoundError(
                    f"qualitative sample {sample_id} is missing warped/rectified imagery"
                )
            with Image.open(rectified_path) as opened:
                rectified = opened.convert("RGB").resize((512, 512))
            with Image.open(warped_path) as opened:
                warped = opened.convert("RGB").resize((512, 512))
            canvas = Image.new("RGB", (1024, 512), "white")
            canvas.paste(rectified, (0, 0))
            canvas.paste(warped, (512, 0))
            figure, axis = plt.subplots(figsize=(14, 7))
            axis.imshow(canvas)
            chosen = artifact_positions[: min(24, len(artifact_positions))]
            colors = plt.cm.turbo(np.linspace(0, 1, len(chosen)))
            for color, position in zip(colors, chosen):
                position = int(position)
                target_flat = int(target_indices[position])
                tx = ((target_flat % target_grid[1]) + 0.5) * 512 / target_grid[1]
                ty = ((target_flat // target_grid[1]) + 0.5) * 512 / target_grid[0]
                prediction = int(top_indices[position, 0])
                px = ((prediction % source_grid[1]) + 0.5) * 512 / source_grid[1] + 512
                py = ((prediction // source_grid[1]) + 0.5) * 512 / source_grid[0]
                gx = (gt[position, 0] + 0.5) * 512 / source_grid[1] + 512
                gy = (gt[position, 1] + 0.5) * 512 / source_grid[0]
                axis.plot([tx, px], [ty, py], color=color, linewidth=0.8, alpha=0.75)
                axis.scatter([gx], [gy], color=[color], marker="x", s=18)
            axis.axvline(512, color="black", linewidth=1)
            axis.set_title(f"{sample_id}: target → predicted source lines; x = GT")
            axis.axis("off")
            figure.tight_layout()
            line_path = output_dir / f"{_safe_file(sample_id)}_matches.png"
            figure.savefig(line_path, dpi=170)
            plt.close(figure)
            written.append(str(line_path))
            written.append(str(cost_path))
            if len(seen) >= limit:
                break
    if len(seen) < limit:
        raise RuntimeError(
            f"expected {limit} qualitative samples for the best configuration, "
            f"but only found {len(seen)} complete artifacts"
        )
    return written


def _safe_file(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def final_report(
    config: Mapping[str, Any], run_dir: Path, confirmation: Mapping[str, Any], confirmation_subgroups: Mapping[str, Any], seed: Mapping[str, Any]
) -> dict[str, Any]:
    normal = list(confirmation["normal"])
    if not normal:
        raise RuntimeError("confirmation has no normal results")
    primary = str(config["selection"]["primary_metric"])
    normal.sort(key=lambda row: _selection_sort_key(row, config))
    best = normal[0]
    shuffle = _lookup_config(confirmation["batch_shuffle"], best)
    random_recall = confirmation["baselines"].get("random_candidate", {}).get(primary)
    identity_epe = confirmation["baselines"].get("identity", {}).get("hard_epe_median_px")
    best_recall = _metric(best, primary, 0.0)
    shuffle_recall = 0.0 if shuffle is None else _metric(shuffle, primary, 0.0)
    control_level = max(float(random_recall or 0.0), shuffle_recall, 1.0e-12)
    control_multiple = best_recall / control_level
    soft_epe = best["metrics"].get("soft_epe_median_px")
    epe_reduction = (
        None
        if identity_epe in (None, 0) or soft_epe is None
        else (float(identity_epe) - float(soft_epe)) / float(identity_epe)
    )
    subgroup_entry = _lookup_config(confirmation_subgroups["normal"], best)
    groups = {} if subgroup_entry is None else subgroup_entry["subgroups"]
    severities = {} if subgroup_entry is None else subgroup_entry["severities"]
    hard_extreme = _combine_metric_groups(severities, ("hard", "extreme"), primary)
    text_edge = _combine_metric_groups(
        groups,
        (
            "structure/text_or_horizontal",
            "structure/vertical",
            "structure/boundary",
            "structure/pseudo_edge",
            "structure/pseudo_boundary",
        ),
        primary,
    )
    seed_entry = next(
        (
            value
            for value in seed.get("configs", [])
            if int(value["layer"]) == int(best["layer"])
            and int(value["step_index"]) == int(best["step_index"])
            and str(value["rope_state"]) == str(best["rope_state"])
        ),
        None,
    )
    seed_std = None if seed_entry is None else seed_entry.get("recall_at_10_r1_std")
    fir = best["metrics"].get("false_identity_rate")
    strong = config["decision_thresholds"]["strong"]
    strong_checks = {
        "overall_recall": best_recall >= float(strong["overall_recall_at_10_r1_min"]),
        "hard_extreme_recall": hard_extreme is not None
        and hard_extreme >= float(strong["hard_extreme_recall_at_10_r1_min"]),
        "text_edge_recall": text_edge is not None
        and text_edge >= float(strong["text_edge_recall_at_10_r1_min"]),
        "control_multiple": control_multiple >= float(strong["control_multiple_min"]),
        "epe_reduction": epe_reduction is not None
        and epe_reduction >= float(strong["soft_median_epe_reduction_vs_identity_min"]),
        "false_identity": fir is not None and float(fir) < float(strong["false_identity_rate_max"]),
        "seed_stability": seed_std is not None and float(seed_std) <= float(strong["seed_recall_std_max"]),
    }
    large_displacement = _combine_metric_groups(
        groups, ("displacement/>96px",), primary
    )
    weak = config["decision_thresholds"]["weak"]
    hard_near_chance = hard_extreme is not None and hard_extreme <= float(
        weak["hard_extreme_recall_at_10_r1_max"]
    )
    large_motion_near_chance = (
        large_displacement is not None
        and large_displacement
        <= float(weak["large_displacement_recall_at_10_r1_max"])
    )
    weak_checks = {
        "overall_recall": best_recall < float(weak["best_recall_at_10_r1_max"]),
        "control_multiple": control_multiple <= float(weak["control_multiple_max"]),
        "epe_not_better_than_identity": epe_reduction is not None
        and epe_reduction <= float(weak.get("epe_reduction_vs_identity_max", 0.0)),
        "hard_or_large_motion_near_chance": bool(
            hard_near_chance or large_motion_near_chance
        ),
        "post_rope_identity_bias": str(best["rope_state"]) == "post"
        and fir is not None
        and float(fir) >= float(weak["post_rope_false_identity_rate_min"]),
        "seed_instability": seed_std is not None
        and float(seed_std) >= float(weak["seed_recall_std_min"]),
    }
    weak_pattern = all(weak_checks.values())
    if all(strong_checks.values()):
        verdict = "强"
    elif weak_pattern:
        verdict = "弱/不存在"
    else:
        verdict = "条件性存在"
    failure_cases = [
        {
            "category": "large_displacement_gt_96px",
            "recall_at_10_r1": _combine_metric_groups(
                groups, ("displacement/>96px",), primary
            ),
        },
        {
            "category": "blank_or_low_texture",
            "recall_at_10_r1": _combine_metric_groups(
                groups,
                ("structure/blank", "structure/pseudo_non_edge"),
                primary,
            ),
        },
        {
            "category": "identity_bias_on_large_motion",
            "false_identity_rate": fir,
        },
        {
            "category": "seed_sensitivity",
            "recall_standard_deviation": seed_std,
        },
    ]
    profile = str(config.get("split", {}).get("profile", "full"))
    pilot = profile == "pilot"
    visuals = generate_qualitative_visuals(
        run_dir / "confirmation",
        run_dir / "visualizations",
        best,
        int(config["visualization"].get("qualitative_samples", 24)),
    )
    ranking_lines = os.linesep.join(
        "| {rank} | L{layer:02d}/S{step:02d}/{rope}/tau={temperature:g} | {recall} | {epe} | {fir_value} |".format(
            rank=index,
            layer=int(row["layer"]),
            step=int(row["step_index"]),
            rope=str(row["rope_state"]),
            temperature=float(row["temperature"]),
            recall=format_optional(row["metrics"].get(primary)),
            epe=format_optional(row["metrics"].get("soft_epe_median_px")),
            fir_value=format_optional(row["metrics"].get("false_identity_rate")),
        )
        for index, row in enumerate(normal, start=1)
    )
    rope_lines: list[str] = []
    for rope_state in ("pre", "post"):
        candidates = [row for row in normal if row["rope_state"] == rope_state]
        if candidates:
            rope_best = candidates[0]
            rope_lines.append(
                f"| {rope_state} | L{int(rope_best['layer']):02d}/S{int(rope_best['step_index']):02d} "
                f"| {format_optional(rope_best['metrics'].get(primary))} "
                f"| {format_optional(rope_best['metrics'].get('soft_epe_median_px'))} |"
            )
        else:
            rope_lines.append(f"| {rope_state} | 未入选 Discovery 候选 | N/A | N/A |")
    seed_lines = os.linesep.join(
        f"| L{int(value['layer']):02d}/S{int(value['step_index']):02d}/{value['rope_state']} "
        f"| {format_optional(value.get('recall_at_10_r1_mean'))} "
        f"| {format_optional(value.get('recall_at_10_r1_std'))} "
        f"| {format_optional(value.get('soft_epe_median_px_mean'))} "
        f"| {format_optional(value.get('top1_coordinate_variance_token2'))} |"
        for value in seed.get("configs", [])
    ) or "| N/A | N/A | N/A | N/A | N/A |"
    runtime_path = run_dir / "runtime_fingerprint.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {}
    )
    runtime_core = runtime.get("runtime_core", {})
    result = {
        "verdict": verdict,
        "model_variant": (
            "lora_finetuned" if config.get("model", {}).get("lora_checkpoint") else "base"
        ),
        "profile": profile,
        "pilot_not_formal": pilot,
        "best_config": best,
        "best_recall_at_10_r1": best_recall,
        "hard_extreme_recall_at_10_r1": hard_extreme,
        "large_displacement_recall_at_10_r1": large_displacement,
        "text_edge_recall_at_10_r1": text_edge,
        "shuffle_recall_at_10_r1": shuffle_recall,
        "random_recall_at_10_r1": random_recall,
        "control_multiple": control_multiple,
        "identity_median_epe_px": identity_epe,
        "soft_median_epe_px": soft_epe,
        "epe_reduction_vs_identity": epe_reduction,
        "false_identity_rate": fir,
        "seed_recall_std": seed_std,
        "strong_checks": strong_checks,
        "weak_checks": weak_checks,
        "failure_case_categories": failure_cases,
        "qualitative_visuals": visuals,
        "qualitative_sample_count": len(visuals) // 2,
    }
    atomic_write_json(run_dir / "final_decision.json", result)
    failure_lines = os.linesep.join(
        f"| {item['category']} | "
        + ", ".join(
            f"{key}={format_optional(value)}"
            for key, value in item.items()
            if key != "category"
        )
        + " |"
        for item in failure_cases
    )
    pilot_notice = (
        "**本次使用 pilot（8+32+128）规模，只能作为预实验，不能作为正式最终结论。**"
        if pilot
        else "本次使用 full（8+64+256）预注册规模。"
    )
    evaluation_label = (
        "LoRA 微调模型冻结对应评估"
        if config.get("model", {}).get("lora_checkpoint")
        else "零样本文档对应评估"
    )
    markdown = f"""# Qwen-Image-Edit MMDiT {evaluation_label}实验一报告

## 单一结论

**{verdict}**。

{pilot_notice}

最终排名只使用独立的 Confirmation 子集。最佳配置为 block `{best['layer']}`、
denoising step `{best['step_index']}`、`{best['rope_state']}-RoPE`、
temperature `{best['temperature']}`。

## 核心指标

| 指标 | 数值 |
|---|---:|
| Confirmation R@10, r=1 | {best_recall:.4f} |
| hard/extreme R@10, r=1 | {format_optional(hard_extreme)} |
| >96px 位移 R@10, r=1 | {format_optional(large_displacement)} |
| 文字/边缘 R@10, r=1 | {format_optional(text_edge)} |
| batch-shuffle R@10, r=1 | {shuffle_recall:.4f} |
| random-candidate R@10, r=1 | {format_optional(random_recall)} |
| 相对最强对照倍数 | {control_multiple:.3f} |
| soft median EPE | {format_optional(soft_epe)} px |
| identity median EPE | {format_optional(identity_epe)} px |
| 相对 identity EPE 降幅 | {format_optional(epe_reduction)} |
| false identity rate | {format_optional(fir)} |
| 三 seed R@10 标准差 | {format_optional(seed_std)} |

## Confirmation 候选排名

| 排名 | 配置 | R@10,r=1 | soft median EPE (px) | false identity rate |
|---:|---|---:|---:|---:|
{ranking_lines}

## pre-RoPE / post-RoPE

这里只比较由 Discovery 冻结后进入 Confirmation 的候选；完整全层对比见 Discovery 热力图。

| RoPE 状态 | 最佳入选配置 | R@10,r=1 | soft median EPE (px) |
|---|---|---:|---:|
{os.linesep.join(rope_lines)}

## Seed 稳定性

| 配置 | R@10 均值 | R@10 标准差 | soft median EPE 均值 | top-1 坐标方差 (token²) |
|---|---:|---:|---:|---:|
{seed_lines}

## 预注册强能力条件

{os.linesep.join(f'- {name}: {"PASS" if passed else "FAIL"}' for name, passed in strong_checks.items())}

## 预注册弱/不存在条件

{os.linesep.join(f'- {name}: {"PASS" if passed else "FAIL"}' for name, passed in weak_checks.items())}

## 失败类型分解

| 类型 | 观测量 |
|---|---|
{failure_lines}

## 产物

- Discovery 热力图：`discovery/heatmaps/`
- Confirmation 总体指标：`confirmation/aggregate_metrics.json`
- Confirmation 分组指标：`confirmation/subgroup_metrics.json`
- Seed 稳定性：`seed_stability/seed_stability.json`
- 定性可视化：`visualizations/`
- 冻结配置与环境：`frozen_config.yaml`、`environment.json`

## 运行指纹

- model: `{runtime.get('model_id', config.get('model', {}).get('model_id'))}`
- revision: `{runtime.get('revision')}`
- pipeline: `{runtime.get('pipeline_class', config.get('model', {}).get('pipeline_class'))}`
- Diffusers: `{runtime_core.get('diffusers', 'unknown')}`
- prompt SHA-256: `{runtime.get('prompt_sha256', 'unknown')}`
- scheduler config SHA-256: `{runtime.get('scheduler_config_sha256', 'unknown')}`
- qualitative samples: `{len(visuals) // 2}`

注：聚合 JSON 中的 rate/mean 按有效 token 加权；EPE median/P95 使用可跨样本、
跨 rank 合并的等质量 centroid 估计 token-micro pooled quantile，并另存样本级 macro
均值。模型始终以 `output_type=latent` 运行，报告没有使用生成 RGB 做选择。
"""
    atomic_write_text(
        run_dir / "MMDiT_correspondence_experiment_1_report.md", markdown
    )
    return result


def format_optional(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("sanity", "discovery", "confirmation", "seed_stability", "final"),
    )
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(config["output"]["run_dir"]).resolve()
    expected_ranks = args.expected_ranks or int(config["resources"].get("expected_gpus", 8))
    if args.stage == "final":
        confirmation = json.loads(
            (run_dir / "confirmation" / "aggregate_metrics.json").read_text(encoding="utf-8")
        )
        confirmation_subgroups = json.loads(
            (run_dir / "confirmation" / "subgroup_metrics.json").read_text(encoding="utf-8")
        )
        seed_path = run_dir / "seed_stability" / "seed_stability.json"
        seed = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.is_file() else {}
        decision = final_report(config, run_dir, confirmation, confirmation_subgroups, seed)
        print(json.dumps({"verdict": decision["verdict"], "report": str(run_dir)}, ensure_ascii=False))
        return
    stage_dir = run_dir / args.stage
    paths, metadata = validate_shards(
        stage_dir,
        expected_ranks,
        require_all=bool(config["sanity_gate"].get("require_all_ranks", True)),
    )
    runtime_fingerprint_path = run_dir / "runtime_fingerprint.json"
    runtime_fingerprint = {
        key: value
        for key, value in metadata[0].items()
        if key
        not in {
            "device",
            "rank",
            "local_rank",
            "elapsed_seconds",
            "samples_completed",
            "local_work_items",
        }
    }
    if runtime_fingerprint_path.is_file():
        existing = json.loads(runtime_fingerprint_path.read_text(encoding="utf-8"))
        stable_fields = (
            "model_id",
            "revision",
            "pipeline_class",
            "lora_checkpoint_sha256",
            "lora_rank",
            "lora_alpha",
            "lora_tensor_count",
            "prompt_sha256",
            "config_sha256",
            "resolved_model_commit_candidates",
            "runtime_core_sha256",
            "scheduler_config_sha256",
            "transformer_config_sha256",
            "block_count",
        )
        if any(existing.get(key) != runtime_fingerprint.get(key) for key in stable_fields):
            raise RuntimeError("runtime fingerprint changed between experiment stages")
    else:
        atomic_write_json(runtime_fingerprint_path, runtime_fingerprint)
        environment_path = run_dir / "environment.json"
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment["qwen_runtime_fingerprint"] = str(runtime_fingerprint_path)
        environment["qwen_runtime_fingerprint_sha256"] = stable_sha256(runtime_fingerprint)
        atomic_write_json(environment_path, environment)
    result = aggregate_stage(args.stage, stage_dir, paths, config)
    if args.stage == "sanity":
        gate = result["aggregate"]["sanity_gate"]
        expected_steps = int(config["sanity_gate"].get("require_exact_step_count", 50))
        trace_lengths_ok = all(
            len(item.get("scheduler_trace", [])) == expected_steps for item in metadata
        )
        trace_indices_ok = all(
            [entry.get("step_index") for entry in item.get("scheduler_trace", [])]
            == list(range(expected_steps))
            for item in metadata
        )
        single_forward_required = bool(
            config["sanity_gate"].get("require_single_conditional_forward", True)
        )
        conditional_forward_ok = all(
            int(entry.get("transformer_forwards", 0)) == 1
            and entry.get("branches") == ["conditional"]
            for item in metadata
            for entry in item.get("scheduler_trace", [])
        )
        gate.update(
            {
                "exact_step_count": trace_lengths_ok,
                "contiguous_step_indices": trace_indices_ok,
                "single_conditional_forward": conditional_forward_ok,
            }
        )
        gate["pass"] = bool(gate["pass"] and trace_lengths_ok and trace_indices_ok)
        if single_forward_required:
            gate["pass"] = bool(gate["pass"] and conditional_forward_ok)
        atomic_write_json(stage_dir / "sanity_gate.json", gate)
        atomic_write_json(stage_dir / "aggregate_metrics.json", result["aggregate"])
        print(json.dumps(gate, ensure_ascii=False))
        if args.require_pass and not gate["pass"]:
            raise SystemExit(2)
    elif args.stage == "discovery":
        selected = select_discovery_configs(result["aggregate"], config, run_dir)
        heatmaps = plot_discovery_heatmaps(
            result["aggregate"], stage_dir, selected["global_temperature"]
        )
        print(
            json.dumps(
                {"selected_configs": len(selected["configs"]), "heatmaps": len(heatmaps)},
                ensure_ascii=False,
            )
        )
    elif args.stage == "seed_stability":
        statistics_value = seed_statistics(result["aggregate"], stage_dir)
        atomic_write_json(stage_dir / "seed_stability.json", statistics_value)
        print(json.dumps({"configs": len(statistics_value["configs"])}, ensure_ascii=False))
    else:
        if args.stage == "confirmation":
            update_seed_configs_from_confirmation(result["aggregate"], config, run_dir)
            plot_confirmation_subgroups(
                result["aggregate"],
                result["subgroups"],
                stage_dir,
                config,
            )
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "samples": result["aggregate"]["sample_count"],
                    "configs": len(result["aggregate"]["normal"]),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
