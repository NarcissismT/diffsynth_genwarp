"""Render document warps with exact analytic absolute backward maps.

This creates Gate-eligible *analytic* geometry labels from a corpus of flat
rectified documents.  It does not claim to reproduce the real camera domain;
the source document identity is hash-split before variants are rendered so no
document can leak across train/validation/test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .checkpoint import file_sha256
from .data import COORDINATE_CONVENTION, MAP_DIRECTION
from .geometry import (
    backward_map_valid_mask,
    canonical_backward_map,
    warp_with_backward_map,
)
from .metrics import cell_valid_mask, jacobian_determinant


@dataclass(frozen=True)
class WarpProfile:
    severity: str
    perspective_fraction: float
    bend_fraction: float


_PROFILES = (
    WarpProfile("light", 0.015, 0.010),
    WarpProfile("medium", 0.035, 0.022),
    WarpProfile("heavy", 0.060, 0.035),
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _load_rgb(path: Path, size: tuple[int, int] | None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None:
            image = image.resize((size[1], size[0]), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()[None]


def _save_rgb_atomic(value: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite rendered image: {destination}")
    array = (
        value.detach().float().cpu()[0].permute(1, 2, 0).clamp(0.0, 1.0)
        .mul(255.0).round().byte().numpy()
    )
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            Image.fromarray(array).save(handle, format="PNG")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_label_atomic(
    destination: Path,
    *,
    backward_map: torch.Tensor,
    valid_mask: torch.Tensor,
    horizontal: torch.Tensor,
    vertical: torch.Tensor,
    boundary: torch.Tensor,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite analytic label: {destination}")
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                backward_map=(
                    backward_map[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                ),
                valid_mask=valid_mask[0, 0].cpu().numpy().astype(np.uint8),
                horizontal_structure=horizontal[0, 0].cpu().numpy().astype(np.float32),
                vertical_structure=vertical[0, 0].cpu().numpy().astype(np.float32),
                boundary_structure=boundary[0, 0].cpu().numpy().astype(np.float32),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fit_corner_homography(
    size: tuple[int, int], destination_offsets: torch.Tensor
) -> torch.Tensor:
    height, width = size
    dtype = torch.float64
    source = torch.tensor(
        ((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)),
        dtype=dtype,
    )
    destination = source + destination_offsets.to(dtype=dtype, device="cpu")
    rows: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    zero, one = source.new_zeros(()), source.new_ones(())
    for (x, y), (u, v) in zip(source, destination):
        rows.append(torch.stack((x, y, one, zero, zero, zero, -u * x, -u * y)))
        rows.append(torch.stack((zero, zero, zero, x, y, one, -v * x, -v * y)))
        values.extend((u, v))
    coefficients = torch.linalg.solve(torch.stack(rows), torch.stack(values))
    return torch.cat((coefficients, one.reshape(1))).reshape(3, 3).float()


def _apply_homography(coordinates: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = coordinates.shape
    flat = coordinates.reshape(batch, 2, -1)
    homogeneous = torch.cat((flat, torch.ones_like(flat[:, :1])), dim=1)
    transformed = homography.to(coordinates)[None] @ homogeneous
    denominator = transformed[:, 2:3]
    safe = torch.where(denominator.abs() > 1.0e-8, denominator, torch.ones_like(denominator))
    return (transformed[:, :2] / safe).reshape(batch, 2, height, width)


def _smooth_displacement(
    coordinates: torch.Tensor,
    size: tuple[int, int],
    *,
    amplitude_x: float,
    amplitude_y: float,
    phase_x: float,
    phase_y: float,
) -> torch.Tensor:
    height, width = size
    x = 2.0 * (coordinates[:, 0:1] + 0.5) / width - 1.0
    y = 2.0 * (coordinates[:, 1:2] + 0.5) / height - 1.0
    envelope = (1.0 - x.square()).clamp_min(0.0) * (1.0 - y.square()).clamp_min(0.0)
    dx = amplitude_x * width * envelope * torch.sin(math.pi * y + phase_x)
    dy = amplitude_y * height * envelope * torch.sin(math.pi * x + phase_y)
    return torch.cat((dx, dy), dim=1)


def _map_from_parameters(
    coordinates: torch.Tensor,
    homography: torch.Tensor,
    size: tuple[int, int],
    parameters: dict[str, float],
) -> torch.Tensor:
    return _apply_homography(coordinates, homography) + _smooth_displacement(
        coordinates, size, **parameters
    )


def _invert_map(
    source_grid: torch.Tensor,
    homography: torch.Tensor,
    size: tuple[int, int],
    parameters: dict[str, float],
    *,
    iterations: int = 20,
) -> torch.Tensor:
    inverse_h = torch.linalg.inv(homography).to(source_grid)
    target = _apply_homography(source_grid, inverse_h)
    for _ in range(iterations):
        displacement = _smooth_displacement(target, size, **parameters)
        target = _apply_homography(source_grid - displacement, inverse_h)
    return target


def _normalized_response(value: torch.Tensor) -> torch.Tensor:
    scale = value.flatten(1).mean(dim=1).clamp_min(1.0e-4)[:, None, None, None]
    return (value / (4.0 * scale)).clamp(0.0, 1.0)


def _structure_labels(
    rectified: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gray = rectified.float().mean(dim=1, keepdim=True)
    horizontal = F.pad(torch.abs(gray[..., 1:, :] - gray[..., :-1, :]), (0, 0, 0, 1))
    vertical = F.pad(torch.abs(gray[..., 1:] - gray[..., :-1]), (0, 1, 0, 0))
    horizontal = F.avg_pool2d(horizontal, (3, 9), stride=1, padding=(1, 4))
    vertical = F.avg_pool2d(vertical, (9, 3), stride=1, padding=(4, 1))
    eroded = -F.max_pool2d(-valid.float(), 3, stride=1, padding=1)
    boundary = (valid.float() - eroded).clamp(0.0, 1.0)
    return _normalized_response(horizontal), _normalized_response(vertical), boundary


def _sample_parameters(
    profile: WarpProfile, size: tuple[int, int], generator: torch.Generator
) -> tuple[torch.Tensor, dict[str, float]]:
    height, width = size
    random = torch.rand((12,), generator=generator)
    offsets = (random[:8].reshape(4, 2) * 2.0 - 1.0)
    offsets = offsets * torch.tensor((width, height)) * profile.perspective_fraction
    # Keep the transform orientation preserving by biasing corners toward a
    # coherent photographed quadrilateral rather than independent extremes.
    homography = _fit_corner_homography(size, offsets)
    parameters = {
        "amplitude_x": profile.bend_fraction * (0.65 + 0.70 * float(random[8])),
        "amplitude_y": profile.bend_fraction * (0.65 + 0.70 * float(random[9])),
        "phase_x": 2.0 * math.pi * float(random[10]),
        "phase_y": 2.0 * math.pi * float(random[11]),
    }
    return homography, parameters


@torch.no_grad()
def render_analytic_sample(
    rectified: torch.Tensor,
    *,
    profile: WarpProfile,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor | float]:
    if rectified.ndim != 4 or rectified.shape[0] != 1 or rectified.shape[1] != 3:
        raise ValueError("rectified must be [1,3,H,W]")
    size = tuple(int(value) for value in rectified.shape[-2:])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    grid = canonical_backward_map(1, size, size, device=device, dtype=torch.float32)
    rectified = rectified.to(device)
    accepted: tuple[torch.Tensor, torch.Tensor, dict[str, float], torch.Tensor] | None = None
    for _ in range(32):
        homography, parameters = _sample_parameters(profile, size, generator)
        homography = homography.to(device)
        backward_map = _map_from_parameters(grid, homography, size, parameters)
        valid = backward_map_valid_mask(backward_map, size)
        determinant = jacobian_determinant(backward_map)
        cells = cell_valid_mask(valid)
        fold_rate = float(((determinant <= 0.0) & cells).sum()) / max(int(cells.sum()), 1)
        if fold_rate == 0.0 and float(valid.float().mean()) >= 0.75:
            accepted = backward_map, homography, parameters, valid
            break
    if accepted is None:
        raise RuntimeError("could not sample a non-folding analytic warp")
    backward_map, homography, parameters, valid = accepted
    inverse_map = _invert_map(grid, homography, size, parameters)
    warped, inverse_valid = warp_with_backward_map(
        rectified, inverse_map, padding_mode="border", return_valid=True
    )
    warped = torch.where(inverse_valid.expand_as(warped), warped, torch.ones_like(warped))
    horizontal, vertical, boundary = _structure_labels(rectified, valid)
    oracle = warp_with_backward_map(warped, backward_map, padding_mode="border")
    rgb_valid = valid.expand_as(oracle)
    oracle_l1 = float(torch.abs(oracle - rectified)[rgb_valid].mean())
    return {
        "warped_image": warped.cpu(),
        "backward_map": backward_map.cpu(),
        "valid_mask": valid.cpu(),
        "horizontal_structure": horizontal.cpu(),
        "vertical_structure": vertical.cpu(),
        "boundary_structure": boundary.cpu(),
        "oracle_rgb_l1": oracle_l1,
    }


def _document_split(document_id: str, seed: int, ratios: tuple[float, float, float]) -> str:
    digest = hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _sample_seed(document_id: str, variant: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{document_id}:{variant}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _write_json_atomic(destination: Path, payload: Any, *, jsonl: bool = False) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite renderer output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            if jsonl:
                for item in payload:
                    handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
            else:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_analytic_dataset(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    image_column: str = "image",
    category_column: str = "category",
    variants_per_document: int = 3,
    seed: int = 1337,
    split_ratios: tuple[float, float, float] = (0.90, 0.05, 0.05),
    output_size: tuple[int, int] | None = (512, 512),
    max_documents: int | None = None,
    device_name: str = "auto",
    shard_index: int = 0,
    num_shards: int = 1,
    cycle_profile_by_document: bool = False,
) -> dict[str, Any]:
    source_csv = Path(input_csv).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to use non-empty renderer output: {destination}")
    if not source_csv.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {source_csv}")
    if variants_per_document < 1:
        raise ValueError("variants_per_document must be positive")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if len(split_ratios) != 3 or any(value < 0.0 for value in split_ratios):
        raise ValueError("split_ratios must be three non-negative values")
    if abs(sum(split_ratios) - 1.0) > 1.0e-6 or any(value == 0.0 for value in split_ratios):
        raise ValueError("train/val/test split ratios must be positive and sum to 1")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    documents: dict[str, tuple[Path, str]] = {}
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if image_column not in (reader.fieldnames or ()):
            raise ValueError(f"input CSV lacks image column {image_column!r}")
        for index, row in enumerate(reader):
            rectified = _resolve(source_csv.parent, str(row[image_column]))
            category = str(row.get(category_column) or "uncategorized").strip()
            document_id = f"{category}:{rectified.stem}"
            previous = documents.setdefault(document_id, (rectified, category))
            if previous[0] != rectified:
                raise ValueError(f"document_id collision for {document_id!r}")
    ordered = sorted(
        documents.items(),
        key=lambda item: hashlib.sha256(
            f"selection:{seed}:{item[0]}".encode("utf-8")
        ).digest(),
    )
    if max_documents is not None:
        ordered = ordered[:max_documents]
    if len(ordered) < 3:
        raise ValueError("analytic rendering needs at least three source documents")
    selected_document_count = len(ordered)
    # Stable round-robin assignment after deterministic global ordering keeps
    # every document and all its variants in exactly one Slurm shard.
    ordered = ordered[shard_index::num_shards]
    if not ordered:
        raise ValueError(
            f"analytic shard {shard_index}/{num_shards} contains no documents"
        )
    # Apply deterministic document selection before remote-path checks.  This
    # makes --max-documents a real bounded probe even when the source CSV has
    # hundreds of thousands of rows on a network filesystem.
    for document_id, (rectified, _) in ordered:
        if not rectified.is_file():
            raise FileNotFoundError(
                f"selected document {document_id!r} is missing: {rectified}"
            )

    assignments = [
        (document_id, payload, _document_split(document_id, seed, split_ratios))
        for document_id, payload in ordered
    ]
    document_counts = {
        split: sum(assigned_split == split for _, _, assigned_split in assignments)
        for split in ("train", "val", "test")
    }
    empty = [name for name, count in document_counts.items() if count == 0]
    if empty and num_shards == 1:
        raise ValueError(
            "hash split produced empty splits; increase max_documents or change seed: "
            + ", ".join(empty)
        )

    generator_sha = file_sha256(Path(__file__))
    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    oracle_values: list[float] = []
    for document_id, (rectified_path, category), split in assignments:
        rectified = _load_rgb(rectified_path, output_size)
        size = tuple(int(value) for value in rectified.shape[-2:])
        safe_document = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:24]
        target_path = destination / "rectified" / f"{safe_document}.png"
        _save_rgb_atomic(rectified, target_path)
        profile_offset = (
            int.from_bytes(
                hashlib.sha256(f"profile:{seed}:{document_id}".encode("utf-8")).digest()[:8],
                "big",
            )
            % len(_PROFILES)
            if cycle_profile_by_document
            else 0
        )
        for variant in range(variants_per_document):
            profile = _PROFILES[(profile_offset + variant) % len(_PROFILES)]
            variant_seed = _sample_seed(document_id, variant, seed)
            rendered = render_analytic_sample(
                rectified, profile=profile, seed=variant_seed, device=device
            )
            sample_id = f"{safe_document}-v{variant:02d}-{profile.severity}"
            image_path = destination / "warped" / split / f"{sample_id}.png"
            label_path = destination / "labels" / split / f"{sample_id}.npz"
            _save_rgb_atomic(rendered["warped_image"], image_path)
            _save_label_atomic(
                label_path,
                backward_map=rendered["backward_map"],
                valid_mask=rendered["valid_mask"],
                horizontal=rendered["horizontal_structure"],
                vertical=rendered["vertical_structure"],
                boundary=rendered["boundary_structure"],
            )
            oracle_values.append(float(rendered["oracle_rgb_l1"]))
            records[split].append(
                {
                    "sample_id": sample_id,
                    "document_id": document_id,
                    "warp_severity": profile.severity,
                    "label_provenance": "analytic_gt",
                    "label_source": (
                        f"cp_docflow.render_analytic_gt.v2;generator_sha256={generator_sha};"
                        f"dataset_seed={seed};sample_seed={variant_seed}"
                    ),
                    "map_direction": MAP_DIRECTION,
                    "coordinate_convention": COORDINATE_CONVENTION,
                    "warped_image": str(image_path),
                    # The copied target is exactly the tensor used by the
                    # renderer, including its fixed output resolution.  The
                    # original path is retained separately for provenance.
                    "rectified_image": str(target_path),
                    "source_rectified_image": str(rectified_path),
                    "backward_map": str(label_path),
                    "valid_mask": str(label_path),
                    "horizontal_structure": str(label_path),
                    "vertical_structure": str(label_path),
                    "boundary_structure": str(label_path),
                    "input_size": list(size),
                    "output_size": list(size),
                    "subset_tags": {
                        "category": category,
                        "domain": "analytic_renderer",
                        "difficulty": profile.severity,
                    },
                }
            )
    manifests: dict[str, str] = {}
    for split, values in records.items():
        path = destination / "manifests" / f"{split}.jsonl"
        _write_json_atomic(path, values, jsonl=True)
        manifests[split] = str(path)
    report = {
        "schema": "docgrid_flow.analytic_renderer.v3",
        "gate_eligible_provenance": "analytic_gt",
        "domain_limit": "analytic geometry; real-camera generalization requires separate evaluation",
        "source_csv": str(source_csv),
        "source_csv_sha256": file_sha256(source_csv),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": generator_sha,
        "seed": seed,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "selected_document_count_before_sharding": selected_document_count,
        "variants_per_document": variants_per_document,
        "cycle_profile_by_document": cycle_profile_by_document,
        "split_ratios": list(split_ratios),
        "output_size": None if output_size is None else list(output_size),
        "documents": document_counts,
        "samples": {name: len(values) for name, values in records.items()},
        "manifests": manifests,
        "manifest_sha256": {
            name: file_sha256(path) for name, path in manifests.items()
        },
        "oracle_rgb_l1_mean": sum(oracle_values) / len(oracle_values),
        "profiles": [profile.__dict__ for profile in _PROFILES],
    }
    _write_json_atomic(destination / "renderer_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--category-column", default="category")
    parser.add_argument("--variants-per-document", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--output-height", type=int, default=512)
    parser.add_argument("--output-width", type=int, default=512)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--cycle-profile-by-document",
        action="store_true",
        help="offset the light/medium/heavy profile cycle deterministically per document",
    )
    args = parser.parse_args()
    result = render_analytic_dataset(
        args.input_csv,
        args.output_dir,
        image_column=args.image_column,
        category_column=args.category_column,
        variants_per_document=args.variants_per_document,
        seed=args.seed,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        output_size=(args.output_height, args.output_width),
        max_documents=args.max_documents,
        device_name=args.device,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        cycle_profile_by_document=args.cycle_profile_by_document,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
