from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from report_mmdit_correspondence import (  # noqa: E402
    WeightedMetrics,
    aggregate_stage,
    select_discovery_configs,
)


def _feature_row(record_type: str, recall: float) -> dict:
    metrics = {
        "valid_tokens": 10,
        "false_identity_tokens": 5,
        "recall_at_10_r1": recall,
        "hard_epe_mean_px": 5.0,
        "hard_epe_median_px": 4.5,
        "hard_epe_p95_px": 9.0,
        "soft_epe_mean_px": 4.5,
        "soft_epe_median_px": 4.0,
        "soft_epe_p95_px": 8.0,
        "hard_epe_px_sketch": {"means": [4.5], "weights": [10]},
        "soft_epe_px_sketch": {"means": [4.0], "weights": [10]},
        "false_identity_rate": 0.2,
        "normalized_entropy": 0.5,
    }
    return {
        "record_type": record_type,
        "stage": "sanity",
        "sample_id": "sample",
        "document_id": "document",
        "warp_severity": "hard",
        "seed": 0,
        "rank": 0,
        "layer": 1,
        "step_index": 2,
        "scheduler_timestep": 500.0,
        "sigma": 0.5,
        "rope_state": "pre",
        "temperature": 0.03,
        "target_grid": [2, 2],
        "source_grid": [3, 3],
        "structure_labels": "explicit",
        "metrics": metrics,
        "subgroups": {"displacement/48-96px": metrics},
    }


def test_sanity_aggregation_checks_repeat_and_shuffle(tmp_path: Path) -> None:
    rows = [
        _feature_row("normal", 0.8),
        _feature_row("determinism_repeat", 0.8),
        _feature_row("batch_shuffle", 0.1),
        {
            "record_type": "baseline",
            "baseline": "identity",
            "stage": "sanity",
            "sample_id": "sample",
            "document_id": "document",
            "warp_severity": "hard",
            "seed": 0,
            "rank": 0,
            "target_grid": [2, 2],
            "source_grid": [3, 3],
            "metrics": {"valid_tokens": 10, "hard_epe_median_px": 8.0},
        },
    ]
    shard = tmp_path / "metrics_rank000.jsonl"
    shard.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = {
        "output": {"write_parquet": False},
        "sanity_gate": {"require_complete_lattice": False},
    }
    result = aggregate_stage("sanity", tmp_path, [shard], config)
    assert result["aggregate"]["sanity_gate"]["pass"]
    assert result["aggregate"]["normal"][0]["metrics"]["recall_at_10_r1"] == 0.8


def test_discovery_selection_freezes_one_temperature_and_depths(tmp_path: Path) -> None:
    rows = []
    for layer, recall, epe in ((1, 0.9, 5.0), (30, 0.8, 3.0), (59, 0.7, 2.0)):
        for temperature in (0.03, 0.10):
            rows.append(
                {
                    "layer": layer,
                    "step_index": 9,
                    "rope_state": "post",
                    "temperature": temperature,
                    "metrics": {
                        "recall_at_10_r1": recall,
                        "hard_epe_median_px": epe + 1,
                        "soft_epe_median_px": epe + (10 if temperature == 0.10 else 0),
                        "hard_extreme_recall_at_10_r1": recall - 0.1,
                        "false_identity_rate": 0.1,
                    },
                }
            )
    aggregate = {"normal": rows}
    config = {
        "selection": {
            "primary_metric": "recall_at_10_r1",
            "tie_breakers": [
                "hard_epe_median_px",
                "soft_epe_median_px",
                "hard_extreme_recall_at_10_r1",
                "false_identity_rate",
            ],
            "max_configs": 3,
            "min_depth_regions": 2,
            "anchor_layers": [17, 18, 20, 21],
        },
        "seed_stability": {"top_configs": 2},
    }
    selected = select_discovery_configs(aggregate, config, tmp_path)
    assert selected["global_temperature"] == 0.03
    assert len(selected["configs"]) == 3
    assert len(selected["depth_regions"]) >= 2
    assert (tmp_path / "selected_configs.json").is_file()


def test_pooled_epe_quantile_uses_mergeable_sketch() -> None:
    accumulator = WeightedMetrics()
    accumulator.add(
        {
            "valid_tokens": 100,
            "soft_epe_median_px": 0.0,
            "soft_epe_p95_px": 0.0,
            "soft_epe_px_sketch": {"means": [0.0], "weights": [100]},
        }
    )
    accumulator.add(
        {
            "valid_tokens": 1,
            "soft_epe_median_px": 100.0,
            "soft_epe_p95_px": 100.0,
            "soft_epe_px_sketch": {"means": [100.0], "weights": [1]},
        }
    )
    result = accumulator.result()
    assert result["soft_epe_median_px"] == 0.0
    assert result["soft_epe_p95_px"] == 0.0
    assert result["soft_epe_median_px_sample_macro_mean"] == 50.0


def test_sanity_gate_rejects_shuffle_without_degradation(tmp_path: Path) -> None:
    rows = [
        _feature_row("normal", 0.8),
        _feature_row("determinism_repeat", 0.8),
        _feature_row("batch_shuffle", 0.8),
    ]
    shard = tmp_path / "metrics_rank000.jsonl"
    shard.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = {
        "output": {"write_parquet": False},
        "sanity_gate": {
            "require_shuffle_degradation": True,
            "shuffle": {
                "primary_metric": "recall_at_10_r1",
                "max_ratio_to_normal": 0.5,
                "min_absolute_drop": 0.05,
            },
        },
    }
    result = aggregate_stage("sanity", tmp_path, [shard], config)
    gate = result["aggregate"]["sanity_gate"]
    assert not gate["pass"]
    assert not gate["shuffle_degradation_pass"]
    assert gate["shuffle_ratio_to_normal"] == 1.0


def test_nonempty_feature_row_rejects_missing_critical_metrics(tmp_path: Path) -> None:
    row = _feature_row("normal", 0.8)
    del row["metrics"]["normalized_entropy"]
    shard = tmp_path / "metrics_rank000.jsonl"
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = {
        "output": {"write_parquet": False},
        "sanity_gate": {"require_complete_lattice": False},
    }
    try:
        aggregate_stage("sanity", tmp_path, [shard], config)
    except ValueError as exc:
        assert "missing/null critical metrics" in str(exc)
    else:
        raise AssertionError("missing critical metric must fail aggregation")

