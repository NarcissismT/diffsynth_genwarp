"""Migrate legacy RAFT displacement CSVs without misrepresenting them as GT.

The historical ``metadata_*`` files point at 1024x1024 torchvision-RAFT
displacements even though the stored document images are commonly 512x512.
This tool materializes an absolute backward map in the native image coordinate
system and emits a strict DocGrid-Flow manifest whose provenance is permanently
``raft_pseudo``.  It cannot create Gate-eligible analytic/renderer labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .checkpoint import file_sha256
from .data import COORDINATE_CONVENTION, MAP_DIRECTION
from .geometry import (
    backward_map_valid_mask,
    canonical_backward_map,
    resize_backward_map,
)

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_REQUIRED_COLUMNS = {"image", "edit_image", "category", "flow_gt_path"}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if min(height, width) < 1:
        raise ValueError(f"invalid image size for {path}: {(height, width)}")
    return height, width


def _flow_tensor(path: Path) -> torch.Tensor:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            if "flow" in payload:
                array = payload["flow"]
            elif len(payload.files) == 1:
                array = payload[payload.files[0]]
            else:
                raise ValueError(f"ambiguous flow keys in {path}: {payload.files}")
        finally:
            payload.close()
    else:
        array = payload
    if array.ndim != 3:
        raise ValueError(f"legacy flow must be rank three, got {array.shape} in {path}")
    if array.shape[-1] == 2:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != 2:
        raise ValueError(f"legacy flow must be [2,H,W] or [H,W,2], got {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None]
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"legacy flow contains non-finite values: {path}")
    return tensor


def displacement_to_native_absolute_map(
    displacement: torch.Tensor,
    *,
    flow_source_size: tuple[int, int],
    native_source_size: tuple[int, int],
    native_output_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a legacy displacement to native absolute source coordinates."""

    if displacement.ndim != 4 or displacement.shape[1] != 2:
        raise ValueError("displacement must be [B,2,H,W]")
    flow_grid_size = tuple(int(value) for value in displacement.shape[-2:])
    absolute = canonical_backward_map(
        displacement.shape[0],
        flow_grid_size,
        flow_source_size,
        device=displacement.device,
        dtype=torch.float32,
    ) + displacement.float()
    native = resize_backward_map(
        absolute,
        native_output_size,
        source_size_from=flow_source_size,
        source_size_to=native_source_size,
    )
    valid = backward_map_valid_mask(native, native_source_size)
    return native, valid


def _safe_sample_id(category: str, stem: str) -> str:
    value = _SAFE_ID.sub("-", f"{category}-{stem}").strip("-.")
    if not value:
        raise ValueError("cannot derive sample_id from category/image stem")
    return value


def _write_npz_atomic(
    destination: Path,
    backward_map: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite derived map: {destination}")
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                backward_map=(
                    backward_map.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                ),
                valid_mask=valid_mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(destination: Path, payload: Any, *, jsonl: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            if jsonl:
                for value in payload:
                    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            else:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def migrate_legacy_raft_csv(
    csv_path: str | Path,
    output_manifest: str | Path,
    derived_map_dir: str | Path,
    *,
    label_checkpoint: str | Path,
    generator_script: str | Path,
    flow_source_size: tuple[int, int] = (1024, 1024),
    limit: int | None = None,
) -> dict[str, Any]:
    """Materialize one legacy split and return its immutable migration report."""

    source_csv = Path(csv_path).resolve()
    manifest = Path(output_manifest).resolve()
    map_root = Path(derived_map_dir).resolve()
    checkpoint = Path(label_checkpoint).resolve()
    generator = Path(generator_script).resolve()
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest}")
    report_path = manifest.with_suffix(manifest.suffix + ".migration.json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite migration report: {report_path}")
    for path, role in (
        (source_csv, "legacy CSV"),
        (checkpoint, "RAFT checkpoint"),
        (generator, "flow generator"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {role}: {path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if min(flow_source_size) < 1:
        raise ValueError("flow_source_size must be positive")

    checkpoint_sha = file_sha256(checkpoint)
    generator_sha = file_sha256(generator)
    records: list[dict[str, Any]] = []
    source_flow_digest = hashlib.sha256()
    seen: set[str] = set()
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(_REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"legacy CSV lacks required columns: {missing}")
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            rectified = _resolve(source_csv.parent, row["image"])
            warped = _resolve(source_csv.parent, row["edit_image"])
            source_flow = _resolve(source_csv.parent, row["flow_gt_path"])
            for path, role in (
                (rectified, "rectified image"),
                (warped, "warped image"),
                (source_flow, "legacy RAFT flow"),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"row {index}: missing {role}: {path}")
            category = str(row.get("category") or "uncategorized").strip()
            sample_id = _safe_sample_id(category, warped.stem)
            if sample_id in seen:
                raise ValueError(f"duplicate derived sample_id={sample_id!r}")
            seen.add(sample_id)
            native_source_size = _image_size(warped)
            native_output_size = _image_size(rectified)
            displacement = _flow_tensor(source_flow)
            backward_map, valid = displacement_to_native_absolute_map(
                displacement,
                flow_source_size=flow_source_size,
                native_source_size=native_source_size,
                native_output_size=native_output_size,
            )
            relative_map = Path(category) / f"{warped.stem}.npz"
            derived_map = map_root / relative_map
            _write_npz_atomic(derived_map, backward_map, valid)
            flow_sha = file_sha256(source_flow)
            source_flow_digest.update(sample_id.encode("utf-8"))
            source_flow_digest.update(b"\0")
            source_flow_digest.update(bytes.fromhex(flow_sha))
            difficulty = str(row.get("difficulty") or "unknown").strip().lower()
            records.append(
                {
                    "sample_id": sample_id,
                    # No claim is made that multiple legacy variants have been
                    # grouped. Formal document-level splitting must be rebuilt
                    # from an authoritative source identity before any GT Gate.
                    "document_id": f"legacy:{category}:{warped.stem}",
                    "warp_severity": difficulty,
                    "label_provenance": "raft_pseudo",
                    "label_source": (
                        "torchvision_raft_large_default;num_flow_updates=20;"
                        f"generator_sha256={generator_sha}"
                    ),
                    "label_checkpoint_sha256": checkpoint_sha,
                    "map_direction": MAP_DIRECTION,
                    "coordinate_convention": COORDINATE_CONVENTION,
                    "warped_image": str(warped),
                    "rectified_image": str(rectified),
                    "backward_map": str(derived_map),
                    "valid_mask": str(derived_map),
                    "input_size": list(native_source_size),
                    "output_size": list(native_output_size),
                    "subset_tags": {
                        "category": category,
                        "difficulty": difficulty,
                        "label_domain": "raft_pseudo_exploratory_only",
                    },
                    "legacy_source_flow": str(source_flow),
                    "legacy_source_flow_sha256": flow_sha,
                    "legacy_flow_format": "backward_displacement_xy",
                    "legacy_flow_source_size": list(flow_source_size),
                    "conversion": "absolute_map_align_corners_false_native_v1",
                }
            )
    if not records:
        raise ValueError(f"legacy CSV contains no rows: {source_csv}")
    _write_json_atomic(manifest, records, jsonl=True)
    report = {
        "schema": "docgrid_flow.legacy_raft_migration.v2",
        "gate_eligible": False,
        "verified_gt_only": False,
        "label_provenance": "raft_pseudo",
        "warning": "RAFT pseudo labels are exploratory-only and cannot unlock any Gate",
        "source_csv": str(source_csv),
        "source_csv_sha256": file_sha256(source_csv),
        "source_flow_set_sha256": source_flow_digest.hexdigest(),
        "label_checkpoint": str(checkpoint),
        "label_checkpoint_sha256": checkpoint_sha,
        "generator_script": str(generator),
        "generator_script_sha256": generator_sha,
        "flow_source_size": list(flow_source_size),
        "output_manifest": str(manifest),
        "output_manifest_sha256": file_sha256(manifest),
        "derived_map_dir": str(map_root),
        "samples": len(records),
        "limit": limit,
    }
    _write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--derived-map-dir", required=True)
    parser.add_argument("--label-checkpoint", required=True)
    parser.add_argument("--generator-script", required=True)
    parser.add_argument("--flow-source-height", type=int, default=1024)
    parser.add_argument("--flow-source-width", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = migrate_legacy_raft_csv(
        args.csv,
        args.output_manifest,
        args.derived_map_dir,
        label_checkpoint=args.label_checkpoint,
        generator_script=args.generator_script,
        flow_source_size=(args.flow_source_height, args.flow_source_width),
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
