#!/usr/bin/env python3
"""Freeze document-disjoint sanity/discovery/confirmation probe splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from docgrid_flow.analysis.mmdit_correspondence import (  # noqa: E402
    ManifestSample,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_backward_map,
    load_config,
    load_mask,
    manifest_asset_paths,
    read_manifest,
    runtime_environment,
    stable_sha256,
    stat_fingerprint,
    write_manifest,
)


SEVERITIES = ("mild", "moderate", "hard", "extreme")
SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}
SEVERITY_ALIASES = {
    "mild": "mild",
    "light": "mild",
    "easy": "mild",
    "low": "mild",
    "moderate": "moderate",
    "medium": "moderate",
    "mid": "moderate",
    "hard": "hard",
    "heavy": "hard",
    "high": "hard",
    "extreme": "extreme",
    "very_hard": "extreme",
    "severe": "extreme",
}


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.height), int(image.width)


def displacement_d90(sample: ManifestSample) -> float:
    backward = load_backward_map(sample.backward_map)
    target_h, target_w = backward.shape[-2:]
    source_h, source_w = sample.input_size or _image_size(sample.warped_image)
    valid = load_mask(sample.valid_mask, (target_h, target_w))[0]
    y, x = torch.meshgrid(
        torch.arange(target_h, dtype=torch.float32),
        torch.arange(target_w, dtype=torch.float32),
        indexing="ij",
    )
    canonical_x = (x + 0.5) * (source_w / target_w) - 0.5
    canonical_y = (y + 0.5) * (source_h / target_h) - 0.5
    finite = torch.isfinite(backward).all(dim=0)
    inside = (
        (backward[0] >= -0.5)
        & (backward[0] <= source_w - 0.5)
        & (backward[1] >= -0.5)
        & (backward[1] <= source_h - 0.5)
    )
    keep = valid & finite & inside
    if not bool(keep.any()):
        raise ValueError(f"{sample.sample_id}: no finite in-source GT map pixels")
    displacement = torch.sqrt(
        (backward[0] - canonical_x).square() + (backward[1] - canonical_y).square()
    )
    return float(torch.quantile(displacement[keep].float(), 0.90).item())


def validate_sample(sample: ManifestSample) -> None:
    if not any(
        sample.source_record.get(name) not in (None, "")
        for name in ("document_id", "doc_id", "document")
    ):
        raise ValueError(
            f"{sample.sample_id}: document_id is required; falling back to sample_id "
            "would not prove document-disjoint Discovery/Confirmation splits"
        )
    paths = (sample.warped_image, sample.backward_map)
    if sample.rectified_image is None:
        raise ValueError(f"{sample.sample_id}: rectified image is required for audit/visualization")
    paths += (sample.rectified_image,)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{sample.sample_id}: missing asset {path}")
    if sample.valid_mask is not None and not sample.valid_mask.is_file():
        raise FileNotFoundError(f"{sample.sample_id}: missing valid mask {sample.valid_mask}")
    structure_paths = (
        sample.horizontal_structure,
        sample.vertical_structure,
        sample.boundary_structure,
    )
    if any(path is not None for path in structure_paths) and not all(
        path is not None and path.is_file() for path in structure_paths
    ):
        raise ValueError(
            f"{sample.sample_id}: H/V/boundary labels must all exist or all be absent"
        )


def assign_severities(samples: list[ManifestSample]) -> list[ManifestSample]:
    d90 = {sample.sample_id: displacement_d90(sample) for sample in samples}
    sorted_values = sorted(d90.values())

    def quantile(fraction: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = fraction * (len(sorted_values) - 1)
        lower = int(position)
        upper = min(len(sorted_values) - 1, lower + 1)
        weight = position - lower
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    thresholds = quantile(0.25), quantile(0.50), quantile(0.75)
    result: list[ManifestSample] = []
    for sample in samples:
        declared = SEVERITY_ALIASES.get(sample.warp_severity)
        value = d90[sample.sample_id]
        computed_index = sum(value > threshold for threshold in thresholds)
        severity = declared or SEVERITIES[computed_index]
        source_record = dict(sample.source_record)
        source_record["warp_severity"] = severity
        source_record["warp_displacement_d90_px"] = value
        source_record["warp_severity_source"] = "manifest" if declared else "gt_map_quartile"
        result.append(
            replace(
                sample,
                warp_severity=severity,
                source_record=source_record,
            )
        )
    return result


def one_per_document(samples: list[ManifestSample], seed: int) -> list[ManifestSample]:
    buckets: dict[str, list[ManifestSample]] = {}
    for sample in samples:
        buckets.setdefault(sample.document_id, []).append(sample)
    rng = random.Random(seed)
    chosen: list[ManifestSample] = []
    for document_id in sorted(buckets):
        values = sorted(buckets[document_id], key=lambda item: item.sample_id)
        chosen.append(values[rng.randrange(len(values))])
    return chosen


def _page_type(sample: ManifestSample) -> str:
    for key in ("page_type", "content_type", "layout_type", "category"):
        if sample.subset_tags.get(key):
            return sample.subset_tags[key]
        if sample.source_record.get(key):
            return str(sample.source_record[key])
    return "unknown"


def stratified_take(
    pool: list[ManifestSample], count: int, rng: random.Random
) -> tuple[list[ManifestSample], list[ManifestSample]]:
    buckets: dict[tuple[str, str], list[ManifestSample]] = {}
    for sample in pool:
        buckets.setdefault((sample.warp_severity, _page_type(sample)), []).append(sample)
    for values in buckets.values():
        rng.shuffle(values)
    keys = sorted(buckets)
    rng.shuffle(keys)
    selected: list[ManifestSample] = []
    while len(selected) < count and keys:
        remaining_keys: list[tuple[str, str]] = []
        for key in keys:
            values = buckets[key]
            if values and len(selected) < count:
                selected.append(values.pop())
            if values:
                remaining_keys.append(key)
        keys = remaining_keys
    if len(selected) != count:
        raise ValueError(f"requested {count} samples but only selected {len(selected)}")
    selected_ids = {sample.sample_id for sample in selected}
    remaining = [sample for sample in pool if sample.sample_id not in selected_ids]
    return selected, remaining


def tag_samples(
    samples: list[ManifestSample], subset: str, *, qualitative: int = 0, seed_repeat: int = 0
) -> list[ManifestSample]:
    result: list[ManifestSample] = []
    for index, sample in enumerate(samples):
        tags = dict(sample.subset_tags)
        tags["mmdit_probe_subset"] = subset
        if index < qualitative:
            tags["mmdit_qualitative"] = "true"
        if index < seed_repeat:
            tags["mmdit_seed_repeat"] = "true"
        source_record = dict(sample.source_record)
        source_record["subset_tags"] = tags
        result.append(replace(sample, subset_tags=tags, source_record=source_record))
    return result


def freeze_config(
    config: dict[str, Any],
    *,
    manifest: Path,
    manifest_role: str,
    profile: str,
    run_dir: Path,
    model_id: str | None,
    model_revision: str | None,
    pipeline_class: str | None,
    lora_checkpoint: str | None,
    lora_alpha: float | None,
    experiment_name: str | None = None,
    experiment_mode: str | None = None,
    gpu_type: str | None = None,
    reference_run: str | None = None,
) -> dict[str, Any]:
    frozen = json.loads(json.dumps(config))
    if experiment_name:
        frozen.setdefault("experiment", {})["name"] = experiment_name
    if experiment_mode:
        frozen.setdefault("experiment", {})["mode"] = experiment_mode
    if reference_run:
        frozen.setdefault("experiment", {})["reference_run"] = str(
            Path(reference_run).expanduser().resolve()
        )
    if gpu_type:
        frozen.setdefault("resources", {})["gpu_type"] = gpu_type
    frozen.setdefault("data", {})["manifest"] = str(manifest.resolve())
    frozen["data"]["manifest_role"] = manifest_role
    frozen.setdefault("split", {})["profile"] = profile
    frozen.setdefault("output", {})["run_dir"] = str(run_dir.resolve())
    if model_id:
        frozen.setdefault("model", {})["model_id"] = model_id
        frozen["model"]["revision"] = model_revision or None
        frozen["model"]["local_files_only"] = Path(model_id).is_absolute()
    elif model_revision:
        frozen.setdefault("model", {})["revision"] = model_revision
    if pipeline_class:
        frozen.setdefault("model", {})["pipeline_class"] = pipeline_class
    if lora_checkpoint:
        checkpoint_path = Path(lora_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")
        if lora_alpha is not None and not float(lora_alpha) > 0:
            raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")
        frozen.setdefault("model", {})["lora_checkpoint"] = str(checkpoint_path)
        frozen["model"]["lora_checkpoint_sha256"] = file_sha256(checkpoint_path)
        frozen["model"]["lora_alpha"] = (
            float(lora_alpha) if lora_alpha is not None else None
        )
        frozen["model"]["adapter_type"] = "peft_lora"
    elif lora_alpha is not None:
        raise ValueError("--lora-alpha requires --lora-checkpoint")
    else:
        for key in (
            "lora_checkpoint",
            "lora_checkpoint_sha256",
            "lora_alpha",
            "adapter_type",
        ):
            frozen.setdefault("model", {}).pop(key, None)
    frozen.setdefault("experiment", {})["frozen"] = True
    frozen["experiment"]["config_sha256"] = stable_sha256(
        {key: value for key, value in frozen.items() if key != "experiment"}
    )
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-role", required=True, choices=("validation", "test"))
    parser.add_argument("--profile", choices=("full", "pilot"), default="full")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--pipeline-class",
        choices=("QwenImageEditPipeline", "QwenImageEditPlusPipeline"),
    )
    parser.add_argument("--lora-checkpoint")
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--experiment-name")
    parser.add_argument("--gpu-type")
    parser.add_argument("--reference-run")
    parser.add_argument(
        "--experiment-mode",
        choices=("formal_zero_shot", "legacy_base_zero_shot", "lora_ablation"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    manifest_path = Path(args.manifest).resolve()
    run_dir = Path(args.run_dir).resolve()
    frozen_path = run_dir / "frozen_config.yaml"
    if frozen_path.exists() and not args.force:
        raise FileExistsError(
            f"frozen experiment already exists: {frozen_path}; use a new run dir "
            "or pass --force only before any probe stage has run"
        )
    config = load_config(config_path, require_resolved=False)
    frozen = freeze_config(
        config,
        manifest=manifest_path,
        manifest_role=args.manifest_role,
        profile=args.profile,
        run_dir=run_dir,
        model_id=args.model_id,
        model_revision=args.model_revision,
        pipeline_class=args.pipeline_class,
        lora_checkpoint=args.lora_checkpoint,
        lora_alpha=args.lora_alpha,
        experiment_name=args.experiment_name,
        experiment_mode=args.experiment_mode,
        gpu_type=args.gpu_type,
        reference_run=args.reference_run,
    )
    sizes = frozen["split"][args.profile]
    required = int(sizes["sanity"]) + int(sizes["discovery"]) + int(sizes["confirmation"])
    samples = read_manifest(manifest_path)
    for sample in samples:
        validate_sample(sample)
        normalized_split = SPLIT_ALIASES.get(str(sample.split).strip().lower())
        if normalized_split is None:
            raise ValueError(
                f"{sample.sample_id}: every record must declare an explicit validation/test "
                f"split, got {sample.split!r}; training or unverified records are forbidden"
            )
        if normalized_split != args.manifest_role:
            raise ValueError(
                f"{sample.sample_id}: record split={sample.split!r} resolves to "
                f"{normalized_split!r}, not requested manifest_role={args.manifest_role!r}"
            )
    samples = assign_severities(samples)
    samples = one_per_document(samples, int(frozen["split"].get("seed", 20260803)))
    if len(samples) < required:
        raise ValueError(
            f"profile={args.profile} requires {required} unique document_id values, "
            f"but the manifest provides {len(samples)}"
        )
    rng = random.Random(int(frozen["split"].get("seed", 20260803)))
    sanity, pool = stratified_take(samples, int(sizes["sanity"]), rng)
    discovery, pool = stratified_take(pool, int(sizes["discovery"]), rng)
    confirmation, _pool = stratified_take(pool, int(sizes["confirmation"]), rng)
    sanity = tag_samples(sanity, "sanity_v1", qualitative=min(8, len(sanity)))
    discovery = tag_samples(discovery, "discovery_v1")
    confirmation = tag_samples(
        confirmation,
        "confirmation_v1",
        qualitative=min(int(frozen["visualization"].get("qualitative_samples", 24)), len(confirmation)),
        seed_repeat=min(int(frozen["seed_stability"].get("sample_count", 32)), len(confirmation)),
    )
    split_dir = run_dir / "splits"
    write_manifest(split_dir / "sanity_v1.jsonl", sanity)
    write_manifest(split_dir / "discovery_v1.jsonl", discovery)
    write_manifest(split_dir / "confirmation_v1.jsonl", confirmation)
    frozen["data"]["splits"] = {
        "sanity": str((split_dir / "sanity_v1.jsonl").resolve()),
        "discovery": str((split_dir / "discovery_v1.jsonl").resolve()),
        "confirmation": str((split_dir / "confirmation_v1.jsonl").resolve()),
    }
    frozen["data"]["split_sha256"] = {
        name: file_sha256(path) for name, path in frozen["data"]["splits"].items()
    }
    # Hash the final frozen payload after all generated split paths/digests have
    # been added. Only the digest field itself is excluded from its preimage.
    frozen.setdefault("experiment", {}).pop("config_sha256", None)
    frozen["experiment"]["config_sha256"] = stable_sha256(frozen)
    frozen_text = yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True)
    atomic_write_text(frozen_path, frozen_text)
    selected_samples = [*sanity, *discovery, *confirmation]
    asset_stat_fingerprint = stat_fingerprint(manifest_asset_paths(selected_samples))
    model_id = str(frozen.get("model", {}).get("model_id", ""))
    model_stat_fingerprint = (
        stat_fingerprint([model_id]) if Path(model_id).is_absolute() else None
    )
    environment = runtime_environment()
    environment.update(
        {
            "source_config": str(config_path),
            "source_config_sha256": file_sha256(config_path),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": file_sha256(manifest_path),
            "manifest_role_assertion": args.manifest_role,
            "profile": args.profile,
            "asset_stat_fingerprint": asset_stat_fingerprint,
            "model_stat_fingerprint": model_stat_fingerprint,
            "unique_documents_available": len(samples),
            "split_counts": {
                "sanity": len(sanity),
                "discovery": len(discovery),
                "confirmation": len(confirmation),
            },
            "split_document_sha256": {
                "sanity": stable_sha256(sorted(sample.document_id for sample in sanity)),
                "discovery": stable_sha256(sorted(sample.document_id for sample in discovery)),
                "confirmation": stable_sha256(sorted(sample.document_id for sample in confirmation)),
            },
        }
    )
    atomic_write_json(run_dir / "environment.json", environment)
    atomic_write_json(
        run_dir / "splits" / "split_summary.json",
        {
            "profile": args.profile,
            "counts": environment["split_counts"],
            "severity_counts": {
                name: {
                    severity: sum(sample.warp_severity == severity for sample in values)
                    for severity in SEVERITIES
                }
                for name, values in (
                    ("sanity", sanity),
                    ("discovery", discovery),
                    ("confirmation", confirmation),
                )
            },
        },
    )
    print(f"frozen experiment: {frozen_path}")
    print(json.dumps(environment["split_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
