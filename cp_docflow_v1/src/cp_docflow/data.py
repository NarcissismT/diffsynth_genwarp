"""Strict manifest and dataset contract for absolute backward maps."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .geometry import (
    backward_map_valid_mask,
    canonical_backward_map,
    resize_backward_map_with_mask,
)

MAP_DIRECTION = "output_to_warped_source"
COORDINATE_CONVENTION = "absolute_source_pixel_xy"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _parse_size(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [height,width], got {value!r}")
    result = (int(value[0]), int(value[1]))
    if min(result) < 1:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_np(path: Path, preferred_key: str) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            if preferred_key in payload:
                return payload[preferred_key]
            if len(payload.files) == 1:
                return payload[payload.files[0]]
            raise ValueError(
                f"{path} has keys {payload.files}; expected {preferred_key!r}"
            )
        finally:
            payload.close()
    return payload


def _as_map(array: np.ndarray) -> Tensor:
    if array.ndim != 3:
        raise ValueError(f"backward_map must be rank 3, got {array.shape}")
    if array.shape[-1] == 2:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != 2:
        raise ValueError(
            f"backward_map must be [H,W,2] or [2,H,W], got {array.shape}"
        )
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def _as_valid(array: np.ndarray, size: tuple[int, int]) -> Tensor:
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"valid_mask must be [H,W] or [1,H,W], got {array.shape}")
    valid = torch.from_numpy(np.ascontiguousarray(array)).bool()
    if tuple(valid.shape[-2:]) != size:
        raise ValueError(
            f"valid_mask grid {tuple(valid.shape[-2:])} differs from output_size={size}"
        )
    return valid


def _as_structure(array: np.ndarray, size: tuple[int, int], name: str) -> Tensor:
    """Normalize an optional H/V/boundary supervision raster to [1,H,W]."""

    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.moveaxis(array, -1, 0)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"{name} must be [H,W], [1,H,W], or [H,W,1], got {array.shape}")
    if tuple(array.shape[-2:]) != size:
        raise ValueError(f"{name} grid {tuple(array.shape[-2:])} differs from output_size={size}")
    value = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value.clamp(0.0, 1.0)


@dataclass(frozen=True)
class ManifestRecord:
    """One sample under the CP-DocFlow v1 schema."""

    sample_id: str
    document_id: str
    warp_severity: str
    label_provenance: str
    label_source: str
    label_checkpoint_sha256: str | None
    map_direction: str
    coordinate_convention: str
    warped_image: Path
    rectified_image: Path
    backward_map: Path
    valid_mask: Path | None
    horizontal_structure: Path | None
    vertical_structure: Path | None
    boundary_structure: Path | None
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    subset_tags: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], root: Path) -> "ManifestRecord":
        required = {
            "sample_id",
            "document_id",
            "warp_severity",
            "label_provenance",
            "label_source",
            "map_direction",
            "coordinate_convention",
            "warped_image",
            "rectified_image",
            "backward_map",
            "input_size",
            "output_size",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"manifest record is missing required keys: {missing}")
        sample_id = str(value["sample_id"]).strip()
        document_id = str(value["document_id"]).strip()
        severity = str(value["warp_severity"]).strip()
        label_provenance = str(value["label_provenance"]).strip().lower()
        label_source = str(value["label_source"]).strip()
        map_direction = str(value["map_direction"]).strip().lower()
        coordinate_convention = str(value["coordinate_convention"]).strip().lower()
        checkpoint_value = value.get("label_checkpoint_sha256")
        checkpoint_sha256 = (
            None
            if checkpoint_value in (None, "")
            else str(checkpoint_value).strip().lower()
        )
        if (
            not sample_id
            or not document_id
            or not severity
            or not label_provenance
            or not label_source
        ):
            raise ValueError(
                "sample_id, document_id, warp_severity, label_provenance, and "
                "label_source must be non-empty"
            )
        if map_direction != MAP_DIRECTION:
            raise ValueError(
                f"map_direction must be {MAP_DIRECTION!r}, got {map_direction!r}"
            )
        if coordinate_convention != COORDINATE_CONVENTION:
            raise ValueError(
                "coordinate_convention must be "
                f"{COORDINATE_CONVENTION!r}, got {coordinate_convention!r}"
            )
        if checkpoint_sha256 is not None and not _SHA256.fullmatch(checkpoint_sha256):
            raise ValueError("label_checkpoint_sha256 must be a 64-character hex digest")
        if label_provenance == "raft_pseudo" and checkpoint_sha256 is None:
            raise ValueError("raft_pseudo records require label_checkpoint_sha256")
        valid_value = value.get("valid_mask")
        raw_tags = value.get("subset_tags", {})
        if not isinstance(raw_tags, Mapping):
            raise ValueError("subset_tags must be a string-to-string mapping")
        subset_tags = tuple(
            sorted(
                (str(key).strip().lower(), str(item).strip().lower())
                for key, item in raw_tags.items()
            )
        )
        if any(not key or not item for key, item in subset_tags):
            raise ValueError("subset_tags keys and values must be non-empty")
        structure_values = {
            name: value.get(name)
            for name in (
                "horizontal_structure",
                "vertical_structure",
                "boundary_structure",
            )
        }
        present = {name for name, item in structure_values.items() if item not in (None, "")}
        if present and len(present) != len(structure_values):
            raise ValueError(
                "horizontal_structure, vertical_structure, and boundary_structure "
                "must be supplied together"
            )
        return cls(
            sample_id=sample_id,
            document_id=document_id,
            warp_severity=severity,
            label_provenance=label_provenance,
            label_source=label_source,
            label_checkpoint_sha256=checkpoint_sha256,
            map_direction=map_direction,
            coordinate_convention=coordinate_convention,
            warped_image=_resolve(root, str(value["warped_image"])),
            rectified_image=_resolve(root, str(value["rectified_image"])),
            backward_map=_resolve(root, str(value["backward_map"])),
            valid_mask=(
                None if valid_value in (None, "") else _resolve(root, str(valid_value))
            ),
            horizontal_structure=(
                None
                if structure_values["horizontal_structure"] in (None, "")
                else _resolve(root, str(structure_values["horizontal_structure"]))
            ),
            vertical_structure=(
                None
                if structure_values["vertical_structure"] in (None, "")
                else _resolve(root, str(structure_values["vertical_structure"]))
            ),
            boundary_structure=(
                None
                if structure_values["boundary_structure"] in (None, "")
                else _resolve(root, str(structure_values["boundary_structure"]))
            ),
            input_size=_parse_size(value["input_size"], "input_size"),
            output_size=_parse_size(value["output_size"], "output_size"),
            subset_tags=subset_tags,
        )


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest = Path(path)
    records: list[ManifestRecord] = []
    seen: set[str] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = ManifestRecord.from_mapping(raw, manifest.parent)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid {manifest}:{line_number}: {exc}") from exc
            if record.sample_id in seen:
                raise ValueError(f"duplicate sample_id={record.sample_id!r} in {manifest}")
            seen.add(record.sample_id)
            records.append(record)
    if not records:
        raise ValueError(f"manifest contains no records: {manifest}")
    return records


def assert_document_disjoint(*splits: Iterable[ManifestRecord]) -> None:
    """Fail when a document identity appears in more than one data split."""

    owners: dict[str, int] = {}
    for split_index, records in enumerate(splits):
        for record in records:
            previous = owners.setdefault(record.document_id, split_index)
            if previous != split_index:
                raise ValueError(
                    f"document_id={record.document_id!r} leaks across splits "
                    f"{previous} and {split_index}"
                )


def dataset_payload_sha256(records: Iterable[ManifestRecord]) -> str:
    """Digest all image/map/mask payloads referenced by a frozen split."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.sample_id):
        digest.update(record.sample_id.encode("utf-8"))
        digest.update(b"\0")
        assets = (
            ("warped_image", record.warped_image),
            ("rectified_image", record.rectified_image),
            ("backward_map", record.backward_map),
            ("valid_mask", record.valid_mask),
            ("horizontal_structure", record.horizontal_structure),
            ("vertical_structure", record.vertical_structure),
            ("boundary_structure", record.boundary_structure),
        )
        for name, path in assets:
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            if path is None:
                digest.update(b"<none>\0")
                continue
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(file_digest.digest())
            digest.update(b"\0")
    return digest.hexdigest()


# torch 2.0's Dataset is not runtime-subscriptable on the Python 3.8 Slurm
# image.  Keep the item type on ``__getitem__`` instead of the base class.
class DocumentMapDataset(Dataset):
    """Load images and an absolute backward map under one coordinate contract."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        input_work_size: Sequence[int] | None = None,
        output_work_size: Sequence[int] | None = None,
    ) -> None:
        self.manifest = Path(manifest)
        self.records = read_manifest(self.manifest)
        structure_modes = {record.horizontal_structure is not None for record in self.records}
        if len(structure_modes) != 1:
            raise ValueError(
                "a manifest must use explicit H/V/boundary labels for every record "
                "or for none, so DataLoader collation remains deterministic"
            )
        self.has_explicit_structure = structure_modes.pop()
        self.input_work_size = (
            None
            if input_work_size is None
            else _parse_size(input_work_size, "input_work_size")
        )
        self.output_work_size = (
            None
            if output_work_size is None
            else _parse_size(output_work_size, "output_work_size")
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        warped = _load_rgb(record.warped_image)
        rectified = _load_rgb(record.rectified_image)
        if tuple(warped.shape[-2:]) != record.input_size:
            raise ValueError(
                f"{record.sample_id}: warped image size {tuple(warped.shape[-2:])} "
                f"!= declared input_size {record.input_size}"
            )
        if tuple(rectified.shape[-2:]) != record.output_size:
            raise ValueError(
                f"{record.sample_id}: rectified image size {tuple(rectified.shape[-2:])} "
                f"!= declared output_size {record.output_size}"
            )

        backward_map = _as_map(_load_np(record.backward_map, "backward_map"))
        if tuple(backward_map.shape[-2:]) != record.output_size:
            raise ValueError(
                f"{record.sample_id}: backward_map grid "
                f"{tuple(backward_map.shape[-2:])} != output_size {record.output_size}"
            )
        if record.valid_mask is None:
            valid = torch.ones((1, *record.output_size), dtype=torch.bool)
        else:
            valid = _as_valid(
                _load_np(record.valid_mask, "valid_mask"),
                record.output_size,
            )
        finite = torch.isfinite(backward_map).all(dim=0, keepdim=True)
        valid &= finite
        # Replace invalid non-finite coordinates before interpolation; validity
        # continues to prevent them from contributing to a loss or metric.
        canonical = canonical_backward_map(
            1,
            record.output_size,
            record.input_size,
            dtype=backward_map.dtype,
        ).squeeze(0)
        backward_map = torch.where(finite.expand_as(backward_map), backward_map, canonical)

        input_size = self.input_work_size or record.input_size
        output_size = self.output_work_size or record.output_size
        if input_size != record.input_size:
            warped = F.interpolate(
                warped.unsqueeze(0),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if output_size != record.output_size:
            rectified = F.interpolate(
                rectified.unsqueeze(0),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if input_size != record.input_size or output_size != record.output_size:
            backward_map, valid = resize_backward_map_with_mask(
                backward_map.unsqueeze(0),
                valid.unsqueeze(0),
                output_size,
                source_size_from=record.input_size,
                source_size_to=input_size,
            )
            backward_map = backward_map.squeeze(0)
            valid = valid.squeeze(0)
        valid &= backward_map_valid_mask(
            backward_map.unsqueeze(0), input_size
        ).squeeze(0)
        if not bool(valid.any()):
            raise ValueError(
                f"{record.sample_id}: no valid map pixels remain after work-size transform"
            )

        sample = {
            "sample_id": record.sample_id,
            "document_id": record.document_id,
            "warp_severity": record.warp_severity,
            "label_provenance": record.label_provenance,
            "label_source": record.label_source,
            "label_checkpoint_sha256": record.label_checkpoint_sha256 or "",
            "map_direction": record.map_direction,
            "coordinate_convention": record.coordinate_convention,
            "warped_image": warped,
            "rectified_image": rectified,
            "backward_map": backward_map,
            "valid_mask": valid,
            "input_size": torch.tensor(input_size, dtype=torch.int64),
            "output_size": torch.tensor(output_size, dtype=torch.int64),
            "native_input_size": torch.tensor(record.input_size, dtype=torch.int64),
            "native_output_size": torch.tensor(record.output_size, dtype=torch.int64),
            "target_canvas_size": torch.tensor(output_size, dtype=torch.int64),
            "target_window": torch.tensor(
                (0.0, 0.0, float(output_size[1]), float(output_size[0])),
                dtype=torch.float32,
            ),
            "training_view": "full_page",
            "subset_tags_json": json.dumps(
                dict(record.subset_tags), sort_keys=True, ensure_ascii=False
            ),
        }
        if self.has_explicit_structure:
            structures = {
                name: _as_structure(
                    _load_np(getattr(record, name), name),
                    record.output_size,
                    name,
                )
                for name in (
                    "horizontal_structure",
                    "vertical_structure",
                    "boundary_structure",
                )
            }
            if output_size != record.output_size:
                structures = {
                    name: F.interpolate(
                        value.unsqueeze(0),
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    for name, value in structures.items()
                }
            sample.update(structures)
        return sample
