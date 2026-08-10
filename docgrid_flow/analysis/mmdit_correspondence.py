"""Geometry and metric primitives for the MMDiT correspondence experiment.

The public coordinate contract is deliberately small and strict:

* backward maps are absolute source-pixel coordinates;
* the last coordinate axis is ``(x, y)``;
* all resize conversions use the ``align_corners=False`` half-pixel rule;
* target/noise tokens are queries and warped-condition tokens are keys.

This module has no dependency on Diffusers.  The Qwen-specific online hook is
implemented in :mod:`docgrid_flow.providers.qwen_diffusers`, while this file is
unit-testable with small synthetic tensors.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import Tensor


FORMAT_VERSION = 1
MAP_DIRECTION = "target_to_warped_source"
COORDINATE_CONVENTION = "absolute_source_pixel_xy_align_corners_false"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def substitute(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if fallback is not None:
            return fallback
        return match.group(0)

    return _ENV_PATTERN.sub(substitute, value)


def load_config(path: str | Path, *, require_resolved: bool = True) -> dict[str, Any]:
    """Load YAML and expand ``${NAME}`` / ``${NAME:-fallback}`` values."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    result = _expand_environment(raw)
    if require_resolved:
        serialized = json.dumps(result, ensure_ascii=False)
        unresolved = sorted({match.group(1) for match in _ENV_PATTERN.finditer(serialized)})
        if unresolved:
            raise ValueError(
                "unresolved environment variables in configuration: "
                + ", ".join(unresolved)
            )
        experiment = result.get("experiment", {})
        if bool(experiment.get("frozen", False)):
            expected = experiment.get("config_sha256")
            payload = json.loads(json.dumps(result))
            payload.setdefault("experiment", {}).pop("config_sha256", None)
            actual = stable_sha256(payload)
            if expected != actual:
                raise ValueError(
                    f"frozen config SHA-256 mismatch: expected={expected}, actual={actual}"
                )
            for name, split_path in result.get("data", {}).get("splits", {}).items():
                expected_split = result.get("data", {}).get("split_sha256", {}).get(name)
                if expected_split is None or file_sha256(split_path) != expected_split:
                    raise ValueError(f"frozen split SHA-256 mismatch: {name}={split_path}")
    return result


def stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_fingerprint(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Build a reproducible content manifest for model or dataset assets."""

    files: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.update(item.resolve() for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"fingerprinted asset does not exist: {path}")
    entries = []
    for path in sorted(files, key=str):
        stat = path.stat()
        entries.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "content_sha256": file_sha256(path),
            }
        )
    return {
        "file_count": len(entries),
        "total_size": sum(entry["size"] for entry in entries),
        "sha256": stable_sha256(entries),
    }


def manifest_asset_paths(samples: Iterable["ManifestSample"]) -> list[Path]:
    paths: set[Path] = set()
    for sample in samples:
        for path in (
            sample.warped_image,
            sample.rectified_image,
            sample.backward_map,
            sample.valid_mask,
            sample.horizontal_structure,
            sample.vertical_structure,
            sample.boundary_structure,
        ):
            if path is not None:
                paths.add(path.resolve())
    return sorted(paths, key=str)


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_write_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, destination)


def runtime_environment() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    for package in ("diffusers", "transformers", "accelerate", "numpy", "PIL", "yaml"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - runtime diagnostic
            versions[package] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        distribution = importlib.metadata.distribution("diffusers")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            versions["diffusers_direct_url"] = json.loads(direct_url)
    except Exception:
        pass
    if torch.cuda.is_available():
        versions["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return versions


def _first(mapping: Mapping[str, Any], names: Sequence[str], *, required: bool) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    if required:
        raise ValueError(f"manifest record needs one of {list(names)}")
    return None


def _path(root: Path, value: Any, *, required: bool = True) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError("required path is empty")
        return None
    result = Path(str(value)).expanduser()
    return result.resolve() if result.is_absolute() else (root / result).resolve()


def _pair(value: Any, name: str) -> tuple[int, int] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [height,width], got {value!r}")
    result = int(value[0]), int(value[1])
    if min(result) < 1:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    document_id: str
    warped_image: Path
    rectified_image: Path | None
    backward_map: Path
    valid_mask: Path | None
    horizontal_structure: Path | None
    vertical_structure: Path | None
    boundary_structure: Path | None
    input_size: tuple[int, int] | None
    output_size: tuple[int, int] | None
    warp_severity: str
    split: str
    subset_tags: dict[str, str]
    source_record: dict[str, Any]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], root: Path, line_number: int
    ) -> "ManifestSample":
        warped_value = _first(
            value,
            ("warped_image", "warped", "image", "image_path", "source"),
            required=True,
        )
        warped = _path(root, warped_value)
        sample_id = str(
            _first(value, ("sample_id", "id", "name"), required=False)
            or warped.stem
        ).strip()
        document_id = str(
            _first(value, ("document_id", "doc_id", "document"), required=False)
            or sample_id
        ).strip()
        if not sample_id or not document_id:
            raise ValueError("sample_id and document_id must be non-empty")
        tags_value = value.get("subset_tags", {})
        if tags_value is None:
            tags_value = {}
        if not isinstance(tags_value, Mapping):
            raise ValueError("subset_tags must be a mapping")
        tags = {str(key): str(item) for key, item in tags_value.items()}
        split = str(
            _first(value, ("split", "subset", "partition"), required=False)
            or tags.get("split", "")
        ).strip().lower()
        map_value = _first(
            value,
            ("backward_map", "map", "backward", "bm"),
            required=False,
        )
        if map_value is None and value.get("flow") not in (None, ""):
            flow_format = str(value.get("flow_format", "")).strip().lower()
            if flow_format not in {
                "absolute",
                "absolute_map",
                "backward_map",
                "absolute_source_pixel_xy",
            }:
                raise ValueError(
                    "a generic 'flow' field is accepted only when flow_format explicitly "
                    "declares an absolute backward map; displacement/normalized flow must "
                    "be converted before this experiment"
                )
            map_value = value["flow"]
        if map_value is None:
            raise ValueError("manifest record needs backward_map (absolute source-pixel xy)")
        declared_direction = str(value.get("map_direction", "")).strip().lower()
        if declared_direction and declared_direction not in {
            "target_to_warped_source",
            "output_to_warped_source",
        }:
            raise ValueError(
                f"map_direction={declared_direction!r} is not target/output to warped source"
            )
        declared_coordinates = str(value.get("coordinate_convention", "")).strip().lower()
        if declared_coordinates and declared_coordinates not in {
            "absolute_source_pixel_xy",
            "absolute_source_pixel_xy_align_corners_false",
        }:
            raise ValueError(
                f"coordinate_convention={declared_coordinates!r} is not absolute source-pixel xy"
            )
        return cls(
            sample_id=sample_id,
            document_id=document_id,
            warped_image=warped,
            rectified_image=_path(
                root,
                _first(
                    value,
                    ("rectified_image", "rectified", "target", "target_image"),
                    required=False,
                ),
                required=False,
            ),
            backward_map=_path(
                root,
                map_value,
            ),
            valid_mask=_path(
                root,
                _first(value, ("valid_mask", "valid", "mask"), required=False),
                required=False,
            ),
            horizontal_structure=_path(
                root, value.get("horizontal_structure"), required=False
            ),
            vertical_structure=_path(
                root, value.get("vertical_structure"), required=False
            ),
            boundary_structure=_path(
                root, value.get("boundary_structure"), required=False
            ),
            input_size=_pair(value.get("input_size"), "input_size"),
            output_size=_pair(value.get("output_size"), "output_size"),
            warp_severity=str(value.get("warp_severity", "unknown")).strip().lower()
            or "unknown",
            split=split,
            subset_tags=tags,
            source_record={**dict(value), "_manifest_line": int(line_number)},
        )

    def frozen_mapping(self) -> dict[str, Any]:
        result = dict(self.source_record)
        result.pop("_manifest_line", None)
        result.update(
            {
                "sample_id": self.sample_id,
                "document_id": self.document_id,
                "warped_image": str(self.warped_image),
                "rectified_image": (
                    None if self.rectified_image is None else str(self.rectified_image)
                ),
                "backward_map": str(self.backward_map),
                "valid_mask": None if self.valid_mask is None else str(self.valid_mask),
                "horizontal_structure": (
                    None
                    if self.horizontal_structure is None
                    else str(self.horizontal_structure)
                ),
                "vertical_structure": (
                    None
                    if self.vertical_structure is None
                    else str(self.vertical_structure)
                ),
                "boundary_structure": (
                    None
                    if self.boundary_structure is None
                    else str(self.boundary_structure)
                ),
                "input_size": None if self.input_size is None else list(self.input_size),
                "output_size": None if self.output_size is None else list(self.output_size),
                "warp_severity": self.warp_severity,
                "split": self.split,
                "subset_tags": self.subset_tags,
                "map_direction": MAP_DIRECTION,
                "coordinate_convention": COORDINATE_CONVENTION,
            }
        )
        return result


def read_manifest(path: str | Path) -> list[ManifestSample]:
    manifest = Path(path).resolve()
    samples: list[ManifestSample] = []
    seen: set[str] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError("record must be a JSON object")
                sample = ManifestSample.from_mapping(raw, manifest.parent, line_number)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid {manifest}:{line_number}: {exc}") from exc
            if sample.sample_id in seen:
                raise ValueError(f"duplicate sample_id={sample.sample_id!r}")
            seen.add(sample.sample_id)
            samples.append(sample)
    if not samples:
        raise ValueError(f"manifest contains no samples: {manifest}")
    return samples


def write_manifest(path: str | Path, samples: Iterable[ManifestSample]) -> None:
    text = "".join(
        json.dumps(sample.frozen_mapping(), sort_keys=True, ensure_ascii=False) + "\n"
        for sample in samples
    )
    atomic_write_text(path, text)


def _load_numpy(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            for key in preferred_keys:
                if key in payload:
                    return np.asarray(payload[key])
            if len(payload.files) == 1:
                return np.asarray(payload[payload.files[0]])
            raise ValueError(
                f"{path} has keys {payload.files}; expected one of {list(preferred_keys)}"
            )
        finally:
            payload.close()
    return np.asarray(payload)


def load_backward_map(path: str | Path) -> Tensor:
    array = _load_numpy(Path(path), ("backward_map", "map", "bm", "flow"))
    if array.ndim != 3:
        raise ValueError(f"backward map must be rank 3, got {array.shape}")
    if array.shape[-1] == 2:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != 2:
        raise ValueError(f"backward map must be [2,H,W] or [H,W,2], got {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def load_mask(path: str | Path | None, size: tuple[int, int]) -> Tensor:
    if path is None:
        return torch.ones((1, *size), dtype=torch.bool)
    array = _load_numpy(Path(path), ("valid_mask", "valid", "mask"))
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.moveaxis(array, -1, 0)
    if array.shape != (1, *size):
        raise ValueError(f"valid mask must be [1,{size[0]},{size[1]}], got {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array)).bool()


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.height), int(image.width)


@dataclass(frozen=True)
class SourceResizeTransform:
    """Pixel-coordinate transform matching source image preprocessing."""

    source_size: tuple[int, int]
    work_size: tuple[int, int]
    mode: str = "stretch"
    resized_size: tuple[int, int] | None = None
    padding_xy: tuple[int, int] = (0, 0)

    @classmethod
    def create(
        cls,
        source_size: Sequence[int],
        work_size: Sequence[int],
        mode: str = "stretch",
    ) -> "SourceResizeTransform":
        source_h, source_w = int(source_size[0]), int(source_size[1])
        work_h, work_w = int(work_size[0]), int(work_size[1])
        normalized = str(mode).lower()
        if min(source_h, source_w, work_h, work_w) < 1:
            raise ValueError("source/work sizes must be positive")
        if normalized == "stretch":
            return cls((source_h, source_w), (work_h, work_w), normalized, (work_h, work_w), (0, 0))
        if normalized != "letterbox":
            raise ValueError("resize mode must be 'stretch' or 'letterbox'")
        scale = min(work_w / source_w, work_h / source_h)
        resized_w = max(1, min(work_w, int(round(source_w * scale))))
        resized_h = max(1, min(work_h, int(round(source_h * scale))))
        pad_x = (work_w - resized_w) // 2
        pad_y = (work_h - resized_h) // 2
        return cls(
            (source_h, source_w),
            (work_h, work_w),
            normalized,
            (resized_h, resized_w),
            (pad_x, pad_y),
        )

    def apply_coords(self, xy: Tensor) -> Tensor:
        if xy.shape[-1] != 2:
            raise ValueError("coordinates must end in (x,y)")
        source_h, source_w = self.source_size
        resized_h, resized_w = self.resized_size or self.work_size
        pad_x, pad_y = self.padding_xy
        result = xy.float().clone()
        result[..., 0] = (result[..., 0] + 0.5) * (resized_w / source_w) - 0.5 + pad_x
        result[..., 1] = (result[..., 1] + 0.5) * (resized_h / source_h) - 0.5 + pad_y
        return result

    def invert_coords(self, xy: Tensor) -> Tensor:
        if xy.shape[-1] != 2:
            raise ValueError("coordinates must end in (x,y)")
        source_h, source_w = self.source_size
        resized_h, resized_w = self.resized_size or self.work_size
        pad_x, pad_y = self.padding_xy
        result = xy.float().clone()
        result[..., 0] = ((result[..., 0] - pad_x) + 0.5) * (source_w / resized_w) - 0.5
        result[..., 1] = ((result[..., 1] - pad_y) + 0.5) * (source_h / resized_h) - 0.5
        return result

    def apply_image(self, image: Image.Image, fill: int = 255) -> Image.Image:
        work_h, work_w = self.work_size
        resized_h, resized_w = self.resized_size or self.work_size
        resized = image.convert("RGB").resize(
            (resized_w, resized_h), Image.Resampling.LANCZOS
        )
        if self.mode == "stretch":
            return resized
        result = Image.new("RGB", (work_w, work_h), color=(fill, fill, fill))
        result.paste(resized, self.padding_xy)
        return result


def pixel_to_token_coordinates(
    xy_pixel: Tensor,
    pixel_size: Sequence[int],
    token_grid: Sequence[int],
) -> Tensor:
    pixel_h, pixel_w = int(pixel_size[0]), int(pixel_size[1])
    token_h, token_w = int(token_grid[0]), int(token_grid[1])
    result = xy_pixel.float().clone()
    result[..., 0] = (result[..., 0] + 0.5) * (token_w / pixel_w) - 0.5
    result[..., 1] = (result[..., 1] + 0.5) * (token_h / pixel_h) - 0.5
    return result


def token_to_pixel_coordinates(
    xy_token: Tensor,
    token_grid: Sequence[int],
    pixel_size: Sequence[int],
) -> Tensor:
    token_h, token_w = int(token_grid[0]), int(token_grid[1])
    pixel_h, pixel_w = int(pixel_size[0]), int(pixel_size[1])
    result = xy_token.float().clone()
    result[..., 0] = (result[..., 0] + 0.5) * (pixel_w / token_w) - 0.5
    result[..., 1] = (result[..., 1] + 0.5) * (pixel_h / token_h) - 0.5
    return result


def make_token_grid(grid: Sequence[int], *, device: torch.device | str = "cpu") -> Tensor:
    height, width = int(grid[0]), int(grid[1])
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def identity_source_token_coordinates(
    target_grid: Sequence[int], source_grid: Sequence[int], *, device: torch.device | str = "cpu"
) -> Tensor:
    target_h, target_w = int(target_grid[0]), int(target_grid[1])
    source_h, source_w = int(source_grid[0]), int(source_grid[1])
    result = make_token_grid((target_h, target_w), device=device)
    result[:, 0] = (result[:, 0] + 0.5) * (source_w / target_w) - 0.5
    result[:, 1] = (result[:, 1] + 0.5) * (source_h / target_h) - 0.5
    return result


def _target_token_centers_native(
    target_grid: Sequence[int], native_target_size: Sequence[int], device: torch.device
) -> Tensor:
    target_h, target_w = int(target_grid[0]), int(target_grid[1])
    native_h, native_w = int(native_target_size[0]), int(native_target_size[1])
    result = make_token_grid((target_h, target_w), device=device)
    result[:, 0] = (result[:, 0] + 0.5) * (native_w / target_w) - 0.5
    result[:, 1] = (result[:, 1] + 0.5) * (native_h / target_h) - 0.5
    return result


def sample_gt_at_target_tokens(
    backward_map: Tensor,
    valid_mask: Tensor,
    *,
    target_grid: Sequence[int],
    source_transform: SourceResizeTransform,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample GT at target-token centres and convert it to work/key units.

    Returns ``(source_pixel_xy_work, valid, target_centres_native)`` flattened
    over the target lattice.
    """

    if backward_map.ndim != 3 or backward_map.shape[0] != 2:
        raise ValueError("backward_map must be [2,H,W]")
    if valid_mask.shape != (1, *backward_map.shape[-2:]):
        raise ValueError("valid_mask must be [1,H,W] and match backward_map")
    target_h, target_w = int(target_grid[0]), int(target_grid[1])
    native_h, native_w = backward_map.shape[-2:]
    device = backward_map.device
    centres = _target_token_centers_native((target_h, target_w), (native_h, native_w), device)
    normalized = centres.clone()
    normalized[:, 0] = 2.0 * (normalized[:, 0] + 0.5) / native_w - 1.0
    normalized[:, 1] = 2.0 * (normalized[:, 1] + 0.5) / native_h - 1.0
    grid = normalized.reshape(1, target_h, target_w, 2)
    finite = torch.isfinite(backward_map).all(dim=0, keepdim=True)
    support = valid_mask.bool() & finite
    safe_map = torch.where(support.expand_as(backward_map), backward_map, 0.0)
    sampled_numerator = F.grid_sample(
        safe_map.unsqueeze(0).float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].permute(1, 2, 0).reshape(-1, 2)
    sampled_support = F.grid_sample(
        support.float().unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0].reshape(-1)
    sampled = sampled_numerator / sampled_support.clamp_min(1.0e-12)[:, None]
    # The plan specifies nearest-neighbour validity at the token centre.  The
    # bilinear support check additionally prevents invalid/NaN neighbours from
    # being replaced by zeros and biasing an otherwise valid GT coordinate.
    valid_nearest = F.grid_sample(
        support.float().unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0].reshape(-1) > 0.5
    valid = valid_nearest & (sampled_support > 1.0e-6)
    sampled_work = source_transform.apply_coords(sampled)
    work_h, work_w = source_transform.work_size
    finite_work = torch.isfinite(sampled_work).all(dim=-1)
    inside = (
        (sampled_work[:, 0] >= -0.5)
        & (sampled_work[:, 0] <= work_w - 0.5)
        & (sampled_work[:, 1] >= -0.5)
        & (sampled_work[:, 1] <= work_h - 0.5)
    )
    return sampled_work, valid & finite_work & inside, centres


STRUCTURE_NAMES = (
    "text_or_horizontal",
    "vertical",
    "boundary",
    "blank",
    "pseudo_edge",
    "pseudo_boundary",
    "pseudo_non_edge",
)
STRUCTURE_TO_ID = {name: index for index, name in enumerate(STRUCTURE_NAMES)}


def _sample_scalar_raster(raster: Tensor, target_grid: tuple[int, int]) -> Tensor:
    if raster.ndim == 2:
        raster = raster[None]
    if raster.ndim != 3 or raster.shape[0] != 1:
        raise ValueError("structure raster must be [H,W] or [1,H,W]")
    height, width = raster.shape[-2:]
    centres = _target_token_centers_native(target_grid, (height, width), raster.device)
    normalized = centres.clone()
    normalized[:, 0] = 2.0 * (normalized[:, 0] + 0.5) / width - 1.0
    normalized[:, 1] = 2.0 * (normalized[:, 1] + 0.5) / height - 1.0
    grid = normalized.reshape(1, target_grid[0], target_grid[1], 2)
    return F.grid_sample(
        raster.float().unsqueeze(0),
        grid,
        mode="bilinear",
        align_corners=False,
    )[0, 0].reshape(-1)


def _load_structure(path: Path, name: str, size: tuple[int, int]) -> Tensor:
    array = _load_numpy(path, (name, "mask"))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.moveaxis(array, -1, 0)
    if array.ndim == 2:
        array = array[None]
    if array.shape != (1, *size):
        raise ValueError(f"{name} must have shape [1,{size[0]},{size[1]}], got {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def explicit_structure_labels(sample: ManifestSample, target_grid: tuple[int, int]) -> Tensor | None:
    paths = (
        sample.horizontal_structure,
        sample.vertical_structure,
        sample.boundary_structure,
    )
    if all(path is None for path in paths):
        return None
    if any(path is None for path in paths):
        raise ValueError(
            f"{sample.sample_id}: H/V/boundary structure paths must be all present or all absent"
        )
    native_size = sample.output_size or tuple(load_backward_map(sample.backward_map).shape[-2:])
    horizontal = _sample_scalar_raster(
        _load_structure(paths[0], "horizontal_structure", native_size), target_grid
    )
    vertical = _sample_scalar_raster(
        _load_structure(paths[1], "vertical_structure", native_size), target_grid
    )
    boundary = _sample_scalar_raster(
        _load_structure(paths[2], "boundary_structure", native_size), target_grid
    )
    labels = torch.full(
        (target_grid[0] * target_grid[1],),
        STRUCTURE_TO_ID["blank"],
        dtype=torch.long,
    )
    labels[horizontal >= 0.5] = STRUCTURE_TO_ID["text_or_horizontal"]
    labels[vertical >= 0.5] = STRUCTURE_TO_ID["vertical"]
    labels[boundary >= 0.5] = STRUCTURE_TO_ID["boundary"]
    return labels


def load_warped_tensor(
    sample: ManifestSample, transform: SourceResizeTransform
) -> tuple[Tensor, Image.Image]:
    with Image.open(sample.warped_image) as opened:
        processed = transform.apply_image(opened)
    array = np.asarray(processed, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, processed


def pseudo_structure_labels(
    warped: Tensor,
    gt_source_pixel_xy: Tensor,
    valid: Tensor,
    target_grid: tuple[int, int],
) -> Tensor:
    """Build a clearly-labelled edge/non-edge fallback from the input image."""

    if warped.ndim != 3 or warped.shape[0] != 3:
        raise ValueError("warped input must be [3,H,W]")
    gray = (
        0.2989 * warped[0:1] + 0.5870 * warped[1:2] + 0.1140 * warped[2:3]
    ).unsqueeze(0)
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=gray.dtype,
    ).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1.0e-12)
    height, width = warped.shape[-2:]
    normalized = gt_source_pixel_xy.clone().float()
    normalized[:, 0] = 2.0 * (normalized[:, 0] + 0.5) / width - 1.0
    normalized[:, 1] = 2.0 * (normalized[:, 1] + 0.5) / height - 1.0
    sampled = F.grid_sample(
        magnitude,
        normalized.reshape(1, target_grid[0], target_grid[1], 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0].reshape(-1)
    threshold = torch.quantile(sampled[valid], 0.70) if bool(valid.any()) else sampled.new_tensor(0.0)
    labels = torch.full(
        (sampled.numel(),), STRUCTURE_TO_ID["pseudo_non_edge"], dtype=torch.long
    )
    labels[sampled >= threshold] = STRUCTURE_TO_ID["pseudo_edge"]
    yy, xx = torch.meshgrid(
        torch.arange(target_grid[0]), torch.arange(target_grid[1]), indexing="ij"
    )
    boundary_width = max(1, int(round(min(target_grid) * 0.06)))
    boundary = (
        (xx < boundary_width)
        | (xx >= target_grid[1] - boundary_width)
        | (yy < boundary_width)
        | (yy >= target_grid[0] - boundary_width)
    ).reshape(-1)
    labels[boundary] = STRUCTURE_TO_ID["pseudo_boundary"]
    return labels


def displacement_bin_ids(displacement_px: Tensor) -> Tensor:
    bins = torch.zeros_like(displacement_px, dtype=torch.long)
    bins[displacement_px >= 16.0] = 1
    bins[displacement_px >= 48.0] = 2
    bins[displacement_px >= 96.0] = 3
    return bins


DISPLACEMENT_NAMES = ("0-16px", "16-48px", "48-96px", ">96px")


def select_target_indices(
    valid: Tensor,
    structure_ids: Tensor,
    displacement_ids: Tensor,
    count: int | None,
    *,
    seed: int,
) -> Tensor:
    candidates = torch.nonzero(valid, as_tuple=False).flatten().cpu().tolist()
    if not candidates:
        raise ValueError("sample has no valid target tokens")
    if count is None or count <= 0 or count >= len(candidates):
        return torch.tensor(candidates, dtype=torch.long)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index in candidates:
        key = int(structure_ids[index]), int(displacement_ids[index])
        buckets.setdefault(key, []).append(index)
    generator = random.Random(int(seed))
    for values in buckets.values():
        generator.shuffle(values)
    keys = sorted(buckets)
    generator.shuffle(keys)
    selected: list[int] = []
    while len(selected) < count and keys:
        remaining: list[tuple[int, int]] = []
        for key in keys:
            values = buckets[key]
            if values and len(selected) < count:
                selected.append(values.pop())
            if values:
                remaining.append(key)
        keys = remaining
    return torch.tensor(sorted(selected), dtype=torch.long)


@dataclass(frozen=True)
class EvaluationContext:
    sample: ManifestSample
    target_grid: tuple[int, int]
    source_grid: tuple[int, int]
    source_work_size: tuple[int, int]
    gt_source_pixel_xy_all: Tensor
    gt_source_token_xy_all: Tensor
    valid_all: Tensor
    identity_source_token_xy_all: Tensor
    displacement_px_all: Tensor
    displacement_ids_all: Tensor
    structure_ids_all: Tensor
    target_indices: Tensor
    structure_is_pseudo: bool
    warped_pil: Image.Image | None = None

    @property
    def count(self) -> int:
        return int(self.target_indices.numel())

    def selected(self, value: Tensor) -> Tensor:
        return value[self.target_indices.to(value.device)]

    def with_source_grid(self, source_grid: tuple[int, int]) -> "EvaluationContext":
        if tuple(source_grid) == self.source_grid:
            return self
        gt_token = pixel_to_token_coordinates(
            self.gt_source_pixel_xy_all, self.source_work_size, source_grid
        )
        identity = identity_source_token_coordinates(self.target_grid, source_grid)
        return replace(
            self,
            source_grid=tuple(source_grid),
            gt_source_token_xy_all=gt_token,
            identity_source_token_xy_all=identity,
        )


def build_evaluation_context(
    sample: ManifestSample,
    *,
    target_grid: tuple[int, int],
    source_grid: tuple[int, int],
    work_size: tuple[int, int],
    resize_mode: str,
    sample_count: int | None,
    sample_seed: int,
    keep_pil: bool = False,
) -> EvaluationContext:
    backward_map = load_backward_map(sample.backward_map)
    native_target_size = tuple(int(value) for value in backward_map.shape[-2:])
    if sample.output_size is not None and sample.output_size != native_target_size:
        raise ValueError(
            f"{sample.sample_id}: declared output_size={sample.output_size} but map is {native_target_size}"
        )
    native_source_size = sample.input_size or image_size(sample.warped_image)
    actual_source_size = image_size(sample.warped_image)
    if native_source_size != actual_source_size:
        raise ValueError(
            f"{sample.sample_id}: declared input_size={native_source_size} but image is {actual_source_size}"
        )
    valid_mask = load_mask(sample.valid_mask, native_target_size)
    transform = SourceResizeTransform.create(native_source_size, work_size, resize_mode)
    warped_tensor, warped_pil = load_warped_tensor(sample, transform)
    gt_pixel, valid, _ = sample_gt_at_target_tokens(
        backward_map,
        valid_mask,
        target_grid=target_grid,
        source_transform=transform,
    )
    gt_token = pixel_to_token_coordinates(gt_pixel, work_size, source_grid)
    identity_token = identity_source_token_coordinates(target_grid, source_grid)
    identity_pixel = token_to_pixel_coordinates(identity_token, source_grid, work_size)
    displacement_px = torch.linalg.vector_norm(gt_pixel - identity_pixel, dim=-1)
    displacement_ids = displacement_bin_ids(displacement_px)
    structure = explicit_structure_labels(sample, target_grid)
    pseudo = structure is None
    if structure is None:
        structure = pseudo_structure_labels(
            warped_tensor, gt_pixel, valid, target_grid
        )
    sample_hash = int(
        hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()[:8], 16
    )
    target_indices = select_target_indices(
        valid,
        structure,
        displacement_ids,
        sample_count,
        seed=int(sample_seed) ^ sample_hash,
    )
    return EvaluationContext(
        sample=sample,
        target_grid=tuple(target_grid),
        source_grid=tuple(source_grid),
        source_work_size=tuple(work_size),
        gt_source_pixel_xy_all=gt_pixel,
        gt_source_token_xy_all=gt_token,
        valid_all=valid,
        identity_source_token_xy_all=identity_token,
        displacement_px_all=displacement_px,
        displacement_ids_all=displacement_ids,
        structure_ids_all=structure,
        target_indices=target_indices,
        structure_is_pseudo=pseudo,
        warped_pil=warped_pil if keep_pil else None,
    )


def _quantile(values: Tensor, fraction: float) -> Tensor:
    if values.numel() == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=torch.float32)
    return torch.quantile(values.float(), float(fraction))


def _mean(values: Tensor) -> Tensor:
    if values.numel() == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=torch.float32)
    return values.float().mean()


def _finite_float(value: Tensor | float | int) -> float | None:
    number = float(value.detach().float().cpu().item()) if isinstance(value, Tensor) else float(value)
    return number if math.isfinite(number) else None


def _quantile_sketch(values: Tensor, max_centroids: int = 32) -> dict[str, Any]:
    """Build a compact mergeable sketch for token-micro EPE quantiles.

    Averaging per-sample medians is not a pooled median.  The evaluator therefore
    stores equal-mass centroids; the report merges them across samples/ranks and
    computes approximate pooled quantiles without serializing every token EPE.
    """

    finite = values.float()[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"means": [], "weights": []}
    ordered = finite.sort().values
    chunks = torch.tensor_split(ordered, min(int(max_centroids), ordered.numel()))
    return {
        "means": [chunk.mean() for chunk in chunks],
        "weights": [int(chunk.numel()) for chunk in chunks],
    }


def _materialize_scalar_trees(value: Any) -> Any:
    """Copy many scalar tensors to CPU with one synchronization."""

    tensors: list[Tensor] = []
    integer_flags: list[bool] = []

    def encode(item: Any) -> Any:
        if isinstance(item, Tensor):
            if item.numel() != 1:
                raise ValueError(f"metric tensor must be scalar, got {tuple(item.shape)}")
            index = len(tensors)
            tensors.append(item.reshape(()))
            integer_flags.append(not item.dtype.is_floating_point)
            return {"__metric_tensor__": index}
        if isinstance(item, dict):
            return {key: encode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [encode(child) for child in item]
        if isinstance(item, tuple):
            return tuple(encode(child) for child in item)
        return item

    encoded = encode(value)
    if not tensors:
        return encoded
    numbers = torch.stack([tensor.float() for tensor in tensors]).detach().cpu().tolist()

    def decode(item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {"__metric_tensor__"}:
            index = int(item["__metric_tensor__"])
            number = float(numbers[index])
            if integer_flags[index]:
                return int(round(number))
            return number if math.isfinite(number) else None
        if isinstance(item, dict):
            return {key: decode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [decode(child) for child in item]
        if isinstance(item, tuple):
            return tuple(decode(child) for child in item)
        return item

    return decode(encoded)


def _metric_summary(
    *,
    mask: Tensor,
    hard_token_xy: Tensor,
    soft_token_xy: Tensor,
    top_indices: Tensor,
    topks: Sequence[int],
    radii: Sequence[int],
    gt_token_xy: Tensor,
    gt_pixel_xy: Tensor,
    identity_token_xy: Tensor,
    source_grid: tuple[int, int],
    work_size: tuple[int, int],
    mass: dict[int, Tensor],
    margin: dict[int, Tensor],
    entropy: Tensor,
) -> dict[str, Any]:
    selected = mask.bool()
    result: dict[str, Any] = {"valid_tokens": selected.sum()}
    gt_round = gt_token_xy.round()
    source_h, source_w = source_grid
    gt_round[:, 0].clamp_(0, source_w - 1)
    gt_round[:, 1].clamp_(0, source_h - 1)
    top_x = (top_indices % source_w).float()
    top_y = torch.div(top_indices, source_w, rounding_mode="floor").float()
    for k in topks:
        limit = min(int(k), top_indices.shape[1])
        for radius in radii:
            hit = (
                torch.maximum(
                    (top_x[:, :limit] - gt_round[:, 0:1]).abs(),
                    (top_y[:, :limit] - gt_round[:, 1:2]).abs(),
                )
                <= float(radius)
            ).any(dim=1)
            result[f"recall_at_{k}_r{radius}"] = _mean(hit[selected])

    hard_pixel = token_to_pixel_coordinates(hard_token_xy, source_grid, work_size)
    soft_pixel = token_to_pixel_coordinates(soft_token_xy, source_grid, work_size)
    hard_epe_px = torch.linalg.vector_norm(hard_pixel - gt_pixel_xy, dim=-1)
    soft_epe_px = torch.linalg.vector_norm(soft_pixel - gt_pixel_xy, dim=-1)
    hard_epe_token = torch.linalg.vector_norm(hard_token_xy - gt_token_xy, dim=-1)
    soft_epe_token = torch.linalg.vector_norm(soft_token_xy - gt_token_xy, dim=-1)
    for prefix, pixel_epe, token_epe in (
        ("hard", hard_epe_px, hard_epe_token),
        ("soft", soft_epe_px, soft_epe_token),
    ):
        values_px = pixel_epe[selected]
        values_token = token_epe[selected]
        result[f"{prefix}_epe_mean_px"] = _mean(values_px)
        result[f"{prefix}_epe_median_px"] = _quantile(values_px, 0.5)
        result[f"{prefix}_epe_p95_px"] = _quantile(values_px, 0.95)
        result[f"{prefix}_epe_px_sketch"] = _quantile_sketch(values_px)
        for threshold in (1, 3, 5):
            result[f"{prefix}_pck_px_{threshold}"] = _mean(
                values_px <= float(threshold)
            )
        for threshold in (0.5, 1.0, 2.0):
            label = str(threshold).replace(".", "p")
            result[f"{prefix}_pck_token_{label}"] = _mean(
                values_token <= threshold
            )
    result["normalized_entropy"] = _mean(entropy[selected])
    for radius, values in mass.items():
        chosen = values[selected]
        result[f"gt_mass_r{radius}"] = _mean(chosen)
        result[f"gt_nll_r{radius}"] = _mean(
            -torch.log(chosen.clamp_min(1.0e-12))
        )
        result[f"margin_r{radius}"] = _mean(margin[radius][selected])
    large_motion = (
        torch.maximum(
            (gt_token_xy[:, 0] - identity_token_xy[:, 0]).abs(),
            (gt_token_xy[:, 1] - identity_token_xy[:, 1]).abs(),
        )
        > 2.0
    ) & selected
    identity_prediction = (
        torch.maximum(
            (hard_token_xy[:, 0] - identity_token_xy[:, 0]).abs(),
            (hard_token_xy[:, 1] - identity_token_xy[:, 1]).abs(),
        )
        <= 1.0
    )
    result["false_identity_tokens"] = large_motion.sum()
    result["false_identity_rate"] = _mean(identity_prediction[large_motion])
    return result


def _subgroup_masks(context: EvaluationContext, device: torch.device) -> dict[str, Tensor]:
    indices = context.target_indices
    displacement = context.displacement_ids_all[indices].to(device)
    structure_cpu = context.structure_ids_all[indices]
    structure = structure_cpu.to(device)
    result: dict[str, Tensor] = {}
    for index, name in enumerate(DISPLACEMENT_NAMES):
        result[f"displacement/{name}"] = displacement == index
    present_structure = sorted(set(int(value) for value in structure_cpu.tolist()))
    for index in present_structure:
        result[f"structure/{STRUCTURE_NAMES[index]}"] = structure == index
    return result


def evaluate_similarity(
    query: Tensor,
    key: Tensor,
    context: EvaluationContext,
    *,
    temperatures: Sequence[float],
    topks: Sequence[int] = (1, 5, 10),
    radii: Sequence[int] = (0, 1, 2),
    source_chunk_size: int = 2048,
    artifact_query_positions: Tensor | None = None,
    return_token_details: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None]:
    """Evaluate a normalized all-to-all Q/K cost without retaining raw features."""

    if query.ndim != 2 or key.ndim != 2 or query.shape[1] != key.shape[1]:
        raise ValueError(
            f"query/key must be [P,D]/[S,D] with shared D, got {query.shape}/{key.shape}"
        )
    if query.shape[0] != context.count:
        raise ValueError(f"query count {query.shape[0]} != selected GT count {context.count}")
    if not bool(torch.isfinite(query).all()):
        raise FloatingPointError("query features contain NaN or Inf")
    if not bool(torch.isfinite(key).all()):
        raise FloatingPointError("key features contain NaN or Inf")
    source_h, source_w = context.source_grid
    source_count = source_h * source_w
    if key.shape[0] != source_count:
        raise ValueError(f"key count {key.shape[0]} != source grid {context.source_grid}")
    if not temperatures or any(float(value) <= 0 for value in temperatures):
        raise ValueError("temperatures must be non-empty and positive")
    device = query.device
    key = key.to(device)
    query = F.normalize(query.float(), p=2, dim=-1, eps=1.0e-8).to(key.dtype)
    key = F.normalize(key.float(), p=2, dim=-1, eps=1.0e-8).to(query.dtype)
    gt_token = context.selected(context.gt_source_token_xy_all).to(device).float()
    gt_pixel = context.selected(context.gt_source_pixel_xy_all).to(device).float()
    identity_token = context.selected(context.identity_source_token_xy_all).to(device).float()
    valid = context.selected(context.valid_all).to(device).bool()
    gt_round = gt_token.round()
    gt_round[:, 0].clamp_(0, source_w - 1)
    gt_round[:, 1].clamp_(0, source_h - 1)
    max_k = min(max(int(value) for value in topks), source_count)
    top_values = torch.full((query.shape[0], max_k), -float("inf"), device=device)
    top_indices = torch.full((query.shape[0], max_k), -1, dtype=torch.long, device=device)
    temperatures = tuple(float(value) for value in temperatures)
    state = {
        tau: {
            "maximum": torch.full((query.shape[0],), -float("inf"), device=device),
            "partition": torch.zeros((query.shape[0],), device=device),
            "coord_sum": torch.zeros((query.shape[0], 2), device=device),
            "logit_sum": torch.zeros((query.shape[0],), device=device),
            "mass": {radius: torch.zeros((query.shape[0],), device=device) for radius in radii},
        }
        for tau in temperatures
    }
    gt_max = {radius: torch.full((query.shape[0],), -float("inf"), device=device) for radius in radii}
    nongt_max = {radius: torch.full((query.shape[0],), -float("inf"), device=device) for radius in radii}
    artifact_parts: list[Tensor] = []
    artifact_positions = (
        None if artifact_query_positions is None else artifact_query_positions.to(device).long()
    )
    chunk_size = max(1, int(source_chunk_size))
    for start in range(0, source_count, chunk_size):
        stop = min(source_count, start + chunk_size)
        score = torch.matmul(query, key[start:stop].transpose(0, 1)).float()
        local_values, local_indices = torch.topk(score, min(max_k, stop - start), dim=1)
        local_indices += start
        merged_values = torch.cat((top_values, local_values), dim=1)
        merged_indices = torch.cat((top_indices, local_indices), dim=1)
        top_values, positions = torch.topk(merged_values, max_k, dim=1)
        top_indices = torch.gather(merged_indices, 1, positions)
        flat = torch.arange(start, stop, device=device)
        candidate_x = (flat % source_w).float()
        candidate_y = torch.div(flat, source_w, rounding_mode="floor").float()
        coords = torch.stack((candidate_x, candidate_y), dim=1)
        chebyshev = torch.maximum(
            (candidate_x[None] - gt_round[:, 0:1]).abs(),
            (candidate_y[None] - gt_round[:, 1:2]).abs(),
        )
        neighborhood = {radius: chebyshev <= float(radius) for radius in radii}
        for radius in radii:
            inside = neighborhood[radius]
            gt_max[radius] = torch.maximum(
                gt_max[radius], score.masked_fill(~inside, -float("inf")).max(dim=1).values
            )
            nongt_max[radius] = torch.maximum(
                nongt_max[radius], score.masked_fill(inside, -float("inf")).max(dim=1).values
            )
        for tau in temperatures:
            current = state[tau]
            logits = score / tau
            chunk_max = logits.max(dim=1).values
            new_max = torch.maximum(current["maximum"], chunk_max)
            old_scale = torch.exp(current["maximum"] - new_max)
            exponent = torch.exp(logits - new_max[:, None])
            current["partition"] = current["partition"] * old_scale + exponent.sum(dim=1)
            current["coord_sum"] = current["coord_sum"] * old_scale[:, None] + exponent @ coords
            current["logit_sum"] = current["logit_sum"] * old_scale + (exponent * logits).sum(dim=1)
            for radius in radii:
                current["mass"][radius] = (
                    current["mass"][radius] * old_scale
                    + (exponent * neighborhood[radius]).sum(dim=1)
                )
            current["maximum"] = new_max
        if artifact_positions is not None and artifact_positions.numel() > 0:
            artifact_parts.append(score[artifact_positions].to(torch.float16).cpu())

    hard_indices = top_indices[:, 0]
    hard_token = torch.stack(
        (
            (hard_indices % source_w).float(),
            torch.div(hard_indices, source_w, rounding_mode="floor").float(),
        ),
        dim=1,
    )
    subgroup_masks = _subgroup_masks(context, device)
    tensor_rows: list[dict[str, Any]] = []
    for tau in temperatures:
        current = state[tau]
        partition = current["partition"].clamp_min(1.0e-30)
        soft_token = current["coord_sum"] / partition[:, None]
        expected_logit = current["logit_sum"] / partition
        entropy = torch.log(partition) + current["maximum"] - expected_logit
        entropy = entropy / math.log(max(source_count, 2))
        mass = {radius: current["mass"][radius] / partition for radius in radii}
        margin = {radius: gt_max[radius] - nongt_max[radius] for radius in radii}
        overall = _metric_summary(
            mask=valid,
            hard_token_xy=hard_token,
            soft_token_xy=soft_token,
            top_indices=top_indices,
            topks=topks,
            radii=radii,
            gt_token_xy=gt_token,
            gt_pixel_xy=gt_pixel,
            identity_token_xy=identity_token,
            source_grid=context.source_grid,
            work_size=context.source_work_size,
            mass=mass,
            margin=margin,
            entropy=entropy,
        )
        subgroups = {
            name: _metric_summary(
                mask=valid & group_mask,
                hard_token_xy=hard_token,
                soft_token_xy=soft_token,
                top_indices=top_indices,
                topks=topks,
                radii=radii,
                gt_token_xy=gt_token,
                gt_pixel_xy=gt_pixel,
                identity_token_xy=identity_token,
                source_grid=context.source_grid,
                work_size=context.source_work_size,
                mass=mass,
                margin=margin,
                entropy=entropy,
            )
            for name, group_mask in subgroup_masks.items()
        }
        tensor_rows.append(
            {
                "temperature": tau,
                "metrics": overall,
                "subgroups": subgroups,
            }
        )
    rows = _materialize_scalar_trees(tensor_rows)
    if return_token_details:
        top1_cpu = hard_token.detach().cpu().numpy().astype(np.float32)
        target_indices_cpu = context.target_indices.cpu().numpy().astype(np.int32)
        for row in rows:
            row["top1_xy"] = top1_cpu
            row["target_indices"] = target_indices_cpu
    artifact = None
    if artifact_positions is not None and artifact_positions.numel() > 0:
        artifact = {
            "cost": torch.cat(artifact_parts, dim=1).numpy(),
            "artifact_query_positions": artifact_positions.cpu().numpy().astype(np.int32),
            "target_indices": context.target_indices.cpu().numpy().astype(np.int32),
            "gt_source_token_xy": gt_token.detach().cpu().numpy().astype(np.float32),
            "identity_source_token_xy": identity_token.detach().cpu().numpy().astype(np.float32),
            "top_indices": top_indices.detach().cpu().numpy().astype(np.int32),
            "top_scores": top_values.detach().cpu().numpy().astype(np.float32),
        }
    return rows, artifact


def _random_recall_probability(population: int, success: int, draws: int) -> float:
    if success <= 0 or draws <= 0:
        return 0.0
    if success >= population or draws >= population - success + 1:
        return 1.0
    probability_miss = 1.0
    for index in range(min(draws, population)):
        probability_miss *= (population - success - index) / (population - index)
    return float(1.0 - probability_miss)


def evaluate_baselines(
    context: EvaluationContext,
    *,
    topks: Sequence[int] = (1, 5, 10),
    radii: Sequence[int] = (0, 1, 2),
    random_seed: int = 0,
) -> list[dict[str, Any]]:
    indices = context.target_indices
    valid = context.valid_all[indices].bool()
    gt_token = context.gt_source_token_xy_all[indices].float()
    gt_pixel = context.gt_source_pixel_xy_all[indices].float()
    identity = context.identity_source_token_xy_all[indices].float()
    identity_pixel = token_to_pixel_coordinates(
        identity, context.source_grid, context.source_work_size
    )
    identity_epe = torch.linalg.vector_norm(identity_pixel - gt_pixel, dim=-1)
    identity_metrics: dict[str, Any] = {
        "valid_tokens": int(valid.sum().item()),
        "hard_epe_mean_px": _finite_float(_mean(identity_epe[valid])),
        "hard_epe_median_px": _finite_float(_quantile(identity_epe[valid], 0.5)),
        "hard_epe_p95_px": _finite_float(_quantile(identity_epe[valid], 0.95)),
        "hard_epe_px_sketch": _materialize_scalar_trees(
            _quantile_sketch(identity_epe[valid])
        ),
    }
    source_h, source_w = context.source_grid
    gt_round = gt_token.round()
    gt_round[:, 0].clamp_(0, source_w - 1)
    gt_round[:, 1].clamp_(0, source_h - 1)
    for radius in radii:
        hit = (
            torch.maximum(
                (identity[:, 0] - gt_round[:, 0]).abs(),
                (identity[:, 1] - gt_round[:, 1]).abs(),
            )
            <= float(radius)
        )
        identity_metrics[f"hit_r{radius}"] = _finite_float(_mean(hit[valid]))
    population = source_h * source_w
    random_metrics: dict[str, Any] = {"valid_tokens": int(valid.sum().item())}
    for radius in radii:
        rounded = gt_round.clone()
        rounded[:, 0].clamp_(0, source_w - 1)
        rounded[:, 1].clamp_(0, source_h - 1)
        left = (rounded[:, 0] - radius).clamp_min(0)
        right = (rounded[:, 0] + radius).clamp_max(source_w - 1)
        top = (rounded[:, 1] - radius).clamp_min(0)
        bottom = (rounded[:, 1] + radius).clamp_max(source_h - 1)
        successes = ((right - left + 1) * (bottom - top + 1)).long()
        for k in topks:
            chances = torch.tensor(
                [
                    _random_recall_probability(population, int(success), int(k))
                    for success in successes.tolist()
                ]
            )
            random_metrics[f"recall_at_{k}_r{radius}"] = _finite_float(_mean(chances[valid]))
    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    random_index = torch.randint(population, (indices.numel(),), generator=generator)
    random_token = torch.stack(
        (
            (random_index % source_w).float(),
            torch.div(random_index, source_w, rounding_mode="floor").float(),
        ),
        dim=1,
    )
    random_pixel = token_to_pixel_coordinates(
        random_token, context.source_grid, context.source_work_size
    )
    random_epe = torch.linalg.vector_norm(random_pixel - gt_pixel, dim=-1)
    random_metrics.update(
        {
            "hard_epe_mean_px": _finite_float(_mean(random_epe[valid])),
            "hard_epe_median_px": _finite_float(_quantile(random_epe[valid], 0.5)),
            "hard_epe_p95_px": _finite_float(_quantile(random_epe[valid], 0.95)),
            "hard_epe_px_sketch": _materialize_scalar_trees(
                _quantile_sketch(random_epe[valid])
            ),
        }
    )
    base = {
        "format_version": FORMAT_VERSION,
        "sample_id": context.sample.sample_id,
        "document_id": context.sample.document_id,
        "warp_severity": context.sample.warp_severity,
        "target_grid": list(context.target_grid),
        "source_grid": list(context.source_grid),
        "structure_labels": "pseudo" if context.structure_is_pseudo else "explicit",
    }
    return [
        {**base, "record_type": "baseline", "baseline": "identity", "metrics": identity_metrics},
        {**base, "record_type": "baseline", "baseline": "random_candidate", "metrics": random_metrics},
    ]


class JsonlWriter:
    """Line-buffered writer for a single rank-owned shard."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Rank-owned shards do not need line-by-line fsync.  A large userspace
        # buffer avoids hundreds of thousands of tiny writes to JuiceFS during
        # the all-layer discovery scan.
        self._handle = self.path.open("a", encoding="utf-8", buffering=1024 * 1024)

    def write(self, value: Mapping[str, Any]) -> None:
        serializable = {
            key: item
            for key, item in value.items()
            if key not in {"top1_xy", "target_indices"}
        }
        self._handle.write(
            json.dumps(serializable, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        )

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
