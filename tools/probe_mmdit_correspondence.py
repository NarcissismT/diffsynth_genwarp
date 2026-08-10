#!/usr/bin/env python3
"""Run one distributed shard of the frozen MMDiT correspondence probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from docgrid_flow.analysis.mmdit_correspondence import (  # noqa: E402
    FORMAT_VERSION,
    EvaluationContext,
    JsonlWriter,
    ManifestSample,
    SourceResizeTransform,
    atomic_write_json,
    build_evaluation_context,
    evaluate_baselines,
    evaluate_similarity,
    load_config,
    read_manifest,
    runtime_environment,
    stable_sha256,
)
from docgrid_flow.providers.qwen_diffusers import (  # noqa: E402
    DiffusersQwenCorrespondenceProbe,
    FeatureMetadata,
    ImageTokenLayout,
    build_selections,
)


@dataclass(frozen=True)
class WorkItem:
    sample: ManifestSample
    seed: int
    global_index: int


@dataclass
class QueryBank:
    context: EvaluationContext
    sample_id: str
    document_id: str
    seed: int
    queries: dict[tuple[int, int, str], torch.Tensor]


def _rank_info(args: argparse.Namespace) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", args.rank if args.rank is not None else 0))
    local_rank = int(
        os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank is not None else rank)
    )
    world_size = int(
        os.environ.get("WORLD_SIZE", args.world_size if args.world_size is not None else 1)
    )
    if not (0 <= rank < world_size):
        raise ValueError(f"invalid rank/world_size={rank}/{world_size}")
    return rank, local_rank, world_size


def _stage_directory(run_dir: Path, stage: str) -> Path:
    return run_dir / stage


def _sample_count(value: Any) -> int | None:
    if value in (None, "all", "full"):
        return None
    result = int(value)
    if result < 1:
        raise ValueError("target_tokens_per_sample must be positive or 'all'")
    return result


def _read_selected(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"selected configuration file is missing: {path}; run discovery reporting first"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    configs = value.get(key)
    if not isinstance(configs, list) or not configs:
        raise ValueError(f"{path} contains no non-empty {key!r} list")
    required = {"layer", "step_index", "rope_state", "temperature"}
    for index, config in enumerate(configs):
        missing = required - set(config)
        if missing:
            raise ValueError(f"{path}:{key}[{index}] missing {sorted(missing)}")
    return configs


def selections_for_stage(
    stage: str,
    config: Mapping[str, Any],
    block_count: int,
    selected_path: Path,
) -> dict[int, dict[int, tuple[str, ...]]]:
    probe = config["probe"]
    if stage in {"sanity", "discovery"}:
        stage_config = probe[stage]
        layers = stage_config["layers"]
        return build_selections(
            block_count=block_count,
            steps=stage_config["steps"],
            layers=layers,
            rope_states=probe["rope_states"],
        )
    key = "configs" if stage == "confirmation" else "seed_configs"
    configs = _read_selected(selected_path, key)
    selections: dict[int, dict[int, set[str]]] = {}
    for value in configs:
        step = int(value["step_index"])
        layer = int(value["layer"])
        selections.setdefault(step, {}).setdefault(layer, set()).add(
            str(value["rope_state"])
        )
    return {
        step: {layer: tuple(sorted(states)) for layer, states in layers.items()}
        for step, layers in selections.items()
    }


def configs_for_stage(
    stage: str, config: Mapping[str, Any], selected_path: Path
) -> set[tuple[int, int, str, float]] | None:
    if stage in {"sanity", "discovery"}:
        return None
    key = "configs" if stage == "confirmation" else "seed_configs"
    return {
        (
            int(value["layer"]),
            int(value["step_index"]),
            str(value["rope_state"]),
            float(value["temperature"]),
        )
        for value in _read_selected(selected_path, key)
    }


def _tag_true(sample: ManifestSample, key: str) -> bool:
    return str(sample.subset_tags.get(key, "false")).lower() in {"1", "true", "yes"}


def make_work_items(
    samples: list[ManifestSample], stage: str, config: Mapping[str, Any]
) -> list[WorkItem]:
    base_seed = int(config["inference"].get("seed", 0))
    if stage != "seed_stability":
        return [
            WorkItem(sample=sample, seed=base_seed, global_index=index)
            for index, sample in enumerate(samples)
        ]
    selected = [sample for sample in samples if _tag_true(sample, "mmdit_seed_repeat")]
    expected = min(int(config["seed_stability"]["sample_count"]), len(samples))
    if len(selected) != expected:
        raise ValueError(
            f"seed_stability expected {expected} tagged samples, found {len(selected)}"
        )
    items: list[WorkItem] = []
    index = 0
    for sample in selected:
        for seed in config["seed_stability"]["seeds"]:
            items.append(WorkItem(sample=sample, seed=int(seed), global_index=index))
            index += 1
    return items


def _processed_image(sample: ManifestSample, config: Mapping[str, Any]) -> Image.Image:
    work_size = tuple(int(value) for value in config["data"]["work_size"])
    source_size = sample.input_size
    if source_size is None:
        with Image.open(sample.warped_image) as image:
            source_size = int(image.height), int(image.width)
    transform = SourceResizeTransform.create(
        source_size, work_size, str(config["data"].get("resize_mode", "stretch"))
    )
    with Image.open(sample.warped_image) as image:
        return transform.apply_image(image)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


class SampleConsumer:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        stage: str,
        item: WorkItem,
        writer: JsonlWriter,
        output_dir: Path,
        rank: int,
        sample_count: int | None,
        allowed_configs: set[tuple[int, int, str, float]] | None,
        collect_query_bank: bool,
        shuffled_bank: QueryBank | None,
        normal_metrics: bool = True,
        normal_record_type: str = "normal",
        write_baselines: bool = True,
    ) -> None:
        self.config = config
        self.stage = stage
        self.item = item
        self.writer = writer
        self.output_dir = output_dir
        self.rank = rank
        self.sample_count = sample_count
        self.allowed_configs = allowed_configs
        self.collect_query_bank = collect_query_bank
        self.shuffled_bank = shuffled_bank
        self.normal_metrics = normal_metrics
        self.normal_record_type = normal_record_type
        self.write_baselines = write_baselines
        self.capture_current_query = normal_metrics or collect_query_bank
        self.context: EvaluationContext | None = None
        self.layout: ImageTokenLayout | None = None
        self.current_queries: dict[tuple[int, int, str], torch.Tensor] = {}
        self.step_trace: list[dict[str, Any]] = []
        self._baselines_written = False
        self._artifact_written: set[tuple[int, int, str]] = set()
        self._details_written: set[tuple[int, int, str]] = set()

    @property
    def temperatures(self) -> tuple[float, ...]:
        if self.allowed_configs is None:
            return tuple(float(value) for value in self.config["probe"]["temperatures"])
        return tuple(
            sorted(
                {
                    temperature
                    for layer, step, rope, temperature in self.allowed_configs
                }
            )
        )

    def _ensure_context(self, layout: ImageTokenLayout) -> EvaluationContext:
        if self.context is not None:
            if self.context.target_grid != layout.target_grid or self.context.source_grid != layout.source_grid:
                raise RuntimeError(
                    f"token layout changed within sample {self.item.sample.sample_id}: "
                    f"{self.context.target_grid}/{self.context.source_grid} -> "
                    f"{layout.target_grid}/{layout.source_grid}"
                )
            return self.context
        work_size = tuple(int(value) for value in self.config["data"]["work_size"])
        self.context = build_evaluation_context(
            self.item.sample,
            target_grid=layout.target_grid,
            source_grid=layout.source_grid,
            work_size=work_size,
            resize_mode=str(self.config["data"].get("resize_mode", "stretch")),
            sample_count=self.sample_count,
            sample_seed=int(self.config["split"].get("seed", 20260803)),
            keep_pil=_tag_true(self.item.sample, "mmdit_qualitative"),
        )
        return self.context

    def on_layout(self, layout: ImageTokenLayout) -> None:
        self.layout = layout
        if self.normal_metrics:
            context = self._ensure_context(layout)
            if self.write_baselines and not self._baselines_written:
                random_seed = int(
                    hashlib.sha256(
                        f"{self.item.sample.sample_id}:{self.item.seed}".encode("utf-8")
                    ).hexdigest()[:8],
                    16,
                )
                for row in evaluate_baselines(
                    context,
                    topks=self.config["probe"]["topk"],
                    radii=self.config["probe"]["radii"],
                    random_seed=random_seed,
                ):
                    self.writer.write(
                        {
                            **row,
                            "stage": self.stage,
                            "seed": self.item.seed,
                            "rank": self.rank,
                        }
                    )
                self._baselines_written = True

    def _base_row(
        self,
        metadata: FeatureMetadata,
        *,
        record_type: str,
        context: EvaluationContext,
    ) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "record_type": record_type,
            "stage": self.stage,
            "sample_id": context.sample.sample_id,
            "document_id": context.sample.document_id,
            "warp_severity": context.sample.warp_severity,
            "seed": (
                self.item.seed
                if record_type in {"normal", "determinism_repeat"}
                else self.shuffled_bank.seed
            ),
            "rank": self.rank,
            "layer": metadata.layer,
            "step_index": metadata.step_index,
            "scheduler_timestep": metadata.scheduler_timestep,
            "sigma": metadata.sigma,
            "branch": metadata.branch,
            "rope_state": metadata.rope_state,
            "target_grid": list(context.target_grid),
            "source_grid": list(context.source_grid),
            "structure_labels": "pseudo" if context.structure_is_pseudo else "explicit",
            "qk_direction": "target_query_to_warped_source_key",
        }

    def _allowed_temperatures(self, metadata: FeatureMetadata) -> tuple[float, ...]:
        if self.allowed_configs is None:
            return self.temperatures
        return tuple(
            sorted(
                temperature
                for layer, step, rope, temperature in self.allowed_configs
                if layer == metadata.layer
                and step == metadata.step_index
                and rope == metadata.rope_state
            )
        )

    def _artifact_positions(self, context: EvaluationContext) -> torch.Tensor | None:
        if not self.normal_metrics or not _tag_true(context.sample, "mmdit_qualitative"):
            return None
        count = min(
            int(self.config["visualization"].get("target_points_per_sample", 32)),
            context.count,
        )
        if count < 1:
            return None
        return torch.linspace(0, context.count - 1, count).round().long().unique()

    def _save_artifact(
        self,
        metadata: FeatureMetadata,
        context: EvaluationContext,
        artifact: dict[str, np.ndarray],
    ) -> None:
        key = (metadata.layer, metadata.step_index, metadata.rope_state)
        if key in self._artifact_written:
            return
        directory = self.output_dir / "artifacts" / f"rank_{self.rank:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{_safe_name(context.sample.sample_id)}_seed{self.item.seed}_"
            f"l{metadata.layer:02d}_s{metadata.step_index:02d}_{metadata.rope_state}.npz"
        )
        np.savez_compressed(
            directory / filename,
            **artifact,
            sample_id=np.asarray(context.sample.sample_id),
            document_id=np.asarray(context.sample.document_id),
            warped_image=np.asarray(str(context.sample.warped_image)),
            rectified_image=np.asarray(
                "" if context.sample.rectified_image is None else str(context.sample.rectified_image)
            ),
            target_grid=np.asarray(context.target_grid, dtype=np.int32),
            source_grid=np.asarray(context.source_grid, dtype=np.int32),
            layer=np.asarray(metadata.layer, dtype=np.int32),
            step_index=np.asarray(metadata.step_index, dtype=np.int32),
            rope_state=np.asarray(metadata.rope_state),
        )
        self._artifact_written.add(key)

    def _save_seed_details(
        self,
        metadata: FeatureMetadata,
        context: EvaluationContext,
        row: Mapping[str, Any],
    ) -> None:
        if self.stage != "seed_stability":
            return
        key = (metadata.layer, metadata.step_index, metadata.rope_state)
        if key in self._details_written:
            return
        directory = self.output_dir / "token_details" / f"rank_{self.rank:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{_safe_name(context.sample.sample_id)}_seed{self.item.seed}_"
            f"l{metadata.layer:02d}_s{metadata.step_index:02d}_{metadata.rope_state}.npz"
        )
        np.savez_compressed(
            directory / filename,
            sample_id=np.asarray(context.sample.sample_id),
            seed=np.asarray(self.item.seed, dtype=np.int32),
            layer=np.asarray(metadata.layer, dtype=np.int32),
            step_index=np.asarray(metadata.step_index, dtype=np.int32),
            rope_state=np.asarray(metadata.rope_state),
            top1_xy=np.asarray(row["top1_xy"], dtype=np.float32),
            target_indices=np.asarray(row["target_indices"], dtype=np.int32),
        )
        self._details_written.add(key)

    def _evaluate_and_write(
        self,
        metadata: FeatureMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        context: EvaluationContext,
        *,
        record_type: str,
        artifact_positions: torch.Tensor | None,
    ) -> None:
        temperatures = self._allowed_temperatures(metadata)
        if not temperatures:
            return
        rows, artifact = evaluate_similarity(
            query,
            key,
            context,
            temperatures=temperatures,
            topks=self.config["probe"]["topk"],
            radii=self.config["probe"]["radii"],
            source_chunk_size=int(self.config["probe"].get("source_chunk_size", 2048)),
            artifact_query_positions=artifact_positions,
            return_token_details=(
                record_type == "normal" and self.stage == "seed_stability"
            ),
        )
        base = self._base_row(metadata, record_type=record_type, context=context)
        if record_type == "batch_shuffle":
            base["key_sample_id"] = self.item.sample.sample_id
            base["key_document_id"] = self.item.sample.document_id
        for result in rows:
            self.writer.write(
                {
                    **base,
                    "temperature": result["temperature"],
                    "metrics": result["metrics"],
                    "subgroups": result["subgroups"],
                }
            )
        if record_type == "normal" and rows:
            self._save_seed_details(metadata, context, rows[0])
        if record_type == "normal" and artifact is not None:
            self._save_artifact(metadata, context, artifact)

    def on_feature(
        self,
        metadata: FeatureMetadata,
        query_target: torch.Tensor | None,
        key_source: torch.Tensor,
    ) -> None:
        feature_key = (metadata.layer, metadata.step_index, metadata.rope_state)
        if self.normal_metrics:
            context = self._ensure_context(metadata.layout)
            if query_target is None:
                raise RuntimeError("normal probe feature arrived without target query")
            selected_query = query_target[context.target_indices.to(query_target.device)]
            self._evaluate_and_write(
                metadata,
                selected_query,
                key_source,
                context,
                record_type=self.normal_record_type,
                artifact_positions=(
                    self._artifact_positions(context)
                    if self.normal_record_type == "normal"
                    else None
                ),
            )
            if self.collect_query_bank:
                normalized = F.normalize(selected_query.float(), dim=-1, eps=1.0e-8)
                self.current_queries[feature_key] = normalized.to(
                    device="cpu", dtype=torch.bfloat16
                )
        if self.shuffled_bank is not None:
            shuffled_query = self.shuffled_bank.queries.get(feature_key)
            if shuffled_query is None:
                raise RuntimeError(
                    f"shuffled query bank lacks layer/step/RoPE={feature_key}"
                )
            shuffled_context = self.shuffled_bank.context.with_source_grid(
                metadata.layout.source_grid
            )
            self._evaluate_and_write(
                metadata,
                shuffled_query.to(device=key_source.device, dtype=key_source.dtype),
                key_source,
                shuffled_context,
                record_type="batch_shuffle",
                artifact_positions=None,
            )

    def on_step_end(self, trace: Mapping[str, Any]) -> None:
        self.step_trace.append(dict(trace))

    def query_bank(self) -> QueryBank:
        if self.context is None:
            raise RuntimeError("cannot create query bank before a selected feature was evaluated")
        if not self.current_queries:
            raise RuntimeError("batch-shuffle requested but no query features were collected")
        return QueryBank(
            context=self.context,
            sample_id=self.item.sample.sample_id,
            document_id=self.item.sample.document_id,
            seed=self.item.seed,
            queries=self.current_queries,
        )


def _config_qwen(config: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(config["model"]), **dict(config["inference"])}


def _split_for_stage(config: Mapping[str, Any], stage: str) -> Path:
    key = "confirmation" if stage == "seed_stability" else stage
    return Path(config["data"]["splits"][key])


def _stage_probe_config(config: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    key = "seed_stability" if stage == "seed_stability" else stage
    return config["probe"][key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen config produced by split builder")
    parser.add_argument(
        "--stage", required=True, choices=("sanity", "discovery", "confirmation", "seed_stability")
    )
    parser.add_argument("--selected-configs")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rank, local_rank, world_size = _rank_info(args)
    if not torch.cuda.is_available():
        raise RuntimeError("the real Qwen probe requires CUDA")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    stagger_seconds = float(os.environ.get("MMDIT_LOAD_STAGGER_SECONDS", "3"))
    if stagger_seconds < 0:
        raise ValueError("MMDIT_LOAD_STAGGER_SECONDS must be non-negative")
    if local_rank and stagger_seconds:
        time.sleep(local_rank * stagger_seconds)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if not bool(config.get("experiment", {}).get("frozen", False)):
        raise ValueError("refusing to run an unfrozen config; use build_mmdit_probe_split.py")
    run_dir = Path(config["output"]["run_dir"]).resolve()
    output_dir = _stage_directory(run_dir, args.stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    done_path = output_dir / f"done_rank{rank:03d}.json"
    final_metrics = output_dir / f"metrics_rank{rank:03d}.jsonl"
    final_meta = output_dir / f"metadata_rank{rank:03d}.json"
    if done_path.exists() and not args.overwrite:
        print(f"rank {rank}: already complete ({done_path})")
        return
    if (final_metrics.exists() or final_meta.exists()) and not args.overwrite:
        raise FileExistsError(
            f"rank {rank} has outputs without a done marker; inspect them, then pass --overwrite"
        )
    if args.overwrite:
        for path in (done_path, final_metrics, final_meta):
            if path.is_file():
                path.unlink()
    selected_path = Path(
        args.selected_configs or (run_dir / "selected_configs.json")
    ).resolve()
    split_path = _split_for_stage(config, args.stage)
    samples = read_manifest(split_path)
    all_items = make_work_items(samples, args.stage, config)
    local_items = [item for item in all_items if item.global_index % world_size == rank]
    if not local_items:
        raise RuntimeError(
            f"rank {rank}/{world_size} has no work items for stage={args.stage}; reduce world size"
        )
    stage_config = _stage_probe_config(config, args.stage)
    target_sample_count = _sample_count(stage_config["target_tokens_per_sample"])
    batch_shuffle = bool(stage_config.get("batch_shuffle", False))
    allowed_configs = configs_for_stage(args.stage, config, selected_path)
    temporary_metrics = final_metrics.with_suffix(
        final_metrics.suffix + f".partial.{os.getpid()}"
    )
    start_time = time.time()
    schedule_hash: str | None = None
    samples_completed = 0
    feature_rows_before = 0

    def selection_builder(block_count: int) -> dict[int, dict[int, tuple[str, ...]]]:
        return selections_for_stage(args.stage, config, block_count, selected_path)

    writer = JsonlWriter(temporary_metrics)
    try:
        with DiffusersQwenCorrespondenceProbe(
            _config_qwen(config),
            device=device,
            selection_builder=selection_builder,
        ) as probe:
            runtime = runtime_environment()
            runtime_core = {
                key: runtime.get(key)
                for key in (
                    "python",
                    "torch",
                    "cuda_runtime",
                    "diffusers",
                    "diffusers_direct_url",
                    "transformers",
                    "accelerate",
                    "numpy",
                )
            }
            fingerprint = probe.fingerprint()
            fingerprint.update(
                {
                    "prompt": config["inference"]["prompt"],
                    "prompt_sha256": hashlib.sha256(
                        str(config["inference"]["prompt"]).encode("utf-8")
                    ).hexdigest(),
                    "config_sha256": stable_sha256(config),
                    "runtime_core": runtime_core,
                    "runtime_core_sha256": stable_sha256(runtime_core),
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "stage": args.stage,
                }
            )
            pending_bank: QueryBank | None = None
            first_item: WorkItem | None = None
            for item_index, item in enumerate(local_items):
                if first_item is None:
                    first_item = item
                consumer = SampleConsumer(
                    config=config,
                    stage=args.stage,
                    item=item,
                    writer=writer,
                    output_dir=output_dir,
                    rank=rank,
                    sample_count=target_sample_count,
                    allowed_configs=allowed_configs,
                    collect_query_bank=batch_shuffle,
                    shuffled_bank=pending_bank if batch_shuffle and len(local_items) > 1 else None,
                    normal_metrics=True,
                )
                probe.run_sample(
                    _processed_image(item.sample, config),
                    seed=item.seed,
                    consumer=consumer,
                )
                current_schedule_hash = stable_sha256(consumer.step_trace)
                if schedule_hash is None:
                    schedule_hash = current_schedule_hash
                elif schedule_hash != current_schedule_hash:
                    raise RuntimeError(
                        "scheduler timestep/sigma trace changed between samples; experiment is not frozen"
                    )
                if batch_shuffle:
                    pending_bank = consumer.query_bank()
                if args.stage == "sanity" and bool(stage_config.get("determinism_repeat", False)):
                    repeat_consumer = SampleConsumer(
                        config=config,
                        stage=args.stage,
                        item=item,
                        writer=writer,
                        output_dir=output_dir,
                        rank=rank,
                        sample_count=target_sample_count,
                        allowed_configs=allowed_configs,
                        collect_query_bank=False,
                        shuffled_bank=None,
                        normal_metrics=True,
                        normal_record_type="determinism_repeat",
                        write_baselines=False,
                    )
                    probe.run_sample(
                        _processed_image(item.sample, config),
                        seed=item.seed,
                        consumer=repeat_consumer,
                    )
                    repeat_schedule_hash = stable_sha256(repeat_consumer.step_trace)
                    if schedule_hash != repeat_schedule_hash:
                        raise RuntimeError("determinism repeat changed scheduler trace")
                samples_completed += 1
                print(
                    f"rank {rank}: {args.stage} {item_index + 1}/{len(local_items)} "
                    f"sample={item.sample.sample_id} seed={item.seed}",
                    flush=True,
                )

            # Complete the cyclic shuffled baseline with one extra key-only
            # trajectory per rank.  With a one-sample sanity shard, use the next
            # global document so Q and K never come from the same document.
            if batch_shuffle:
                assert pending_bank is not None and first_item is not None
                if len(local_items) > 1:
                    key_item = first_item
                else:
                    partner_index = (first_item.global_index + 1) % len(all_items)
                    key_item = all_items[partner_index]
                    if key_item.sample.document_id == first_item.sample.document_id:
                        raise RuntimeError("could not construct a document-disjoint shuffle partner")
                key_only = SampleConsumer(
                    config=config,
                    stage=args.stage,
                    item=key_item,
                    writer=writer,
                    output_dir=output_dir,
                    rank=rank,
                    sample_count=target_sample_count,
                    allowed_configs=allowed_configs,
                    collect_query_bank=False,
                    shuffled_bank=pending_bank,
                    normal_metrics=False,
                )
                probe.run_sample(
                    _processed_image(key_item.sample, config),
                    seed=key_item.seed,
                    consumer=key_only,
                )
                key_schedule_hash = stable_sha256(key_only.step_trace)
                if schedule_hash != key_schedule_hash:
                    raise RuntimeError("batch-shuffle key trajectory changed scheduler trace")
            metadata = {
                **fingerprint,
                "runtime": runtime,
                "split": str(split_path),
                "local_work_items": len(local_items),
                "samples_completed": samples_completed,
                "batch_shuffle": batch_shuffle,
                "scheduler_trace": probe.scheduler_trace,
                "scheduler_trace_sha256": schedule_hash,
                "elapsed_seconds": time.time() - start_time,
            }
    except Exception:
        writer.close()
        raise
    else:
        writer.close()
    os.replace(temporary_metrics, final_metrics)
    atomic_write_json(final_meta, metadata)
    atomic_write_json(
        done_path,
        {
            "rank": rank,
            "world_size": world_size,
            "stage": args.stage,
            "samples_completed": samples_completed,
            "metrics": str(final_metrics),
            "metadata": str(final_meta),
            "elapsed_seconds": time.time() - start_time,
        },
    )
    print(f"rank {rank}: complete -> {done_path}")


if __name__ == "__main__":
    main()
