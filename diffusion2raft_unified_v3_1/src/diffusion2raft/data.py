"""Dataset and guide-artifact augmentation for rectification flow training."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .geometry import (
    backward_flow_to_map,
    backward_warp,
    flow_valid_mask,
    make_pixel_grid,
    map_to_backward_flow,
    pixel_map_valid_mask,
    resize_backward_flow,
)


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _resize_image(image: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)


def _load_array(path: Path, preferred_key: str) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if preferred_key in payload:
            return payload[preferred_key]
        if len(payload.files) != 1:
            raise ValueError(f"{path} has keys {payload.files}; expected {preferred_key!r}")
        return payload[payload.files[0]]
    return payload


def _as_flow_tensor(array: np.ndarray) -> Tensor:
    if array.ndim != 3:
        raise ValueError(f"flow array must be rank 3, got {array.shape}")
    if array.shape[-1] == 2:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != 2:
        raise ValueError(f"flow must be [H,W,2] or [2,H,W], got {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unsqueeze(0)


def _convert_flow_format(
    flow_or_map: Tensor,
    flow_format: str,
    *,
    source_size: tuple[int, int],
) -> Tensor:
    flow_format = flow_format.lower()
    if flow_format in {"displacement", "backward_displacement", "flow"}:
        return flow_or_map
    if flow_format in {"absolute_map", "backward_map", "map"}:
        return map_to_backward_flow(flow_or_map)
    if flow_format in {"normalized_grid", "grid_sample"}:
        source_h, source_w = source_size
        pixel_map = flow_or_map.clone()
        pixel_map[:, 0] = (pixel_map[:, 0] + 1.0) * (source_w - 1) / 2.0
        pixel_map[:, 1] = (pixel_map[:, 1] + 1.0) * (source_h - 1) / 2.0
        return map_to_backward_flow(pixel_map)
    raise ValueError(
        f"unknown flow_format={flow_format!r}; use displacement, absolute_map, or normalized_grid"
    )


def _size_from_record(record: dict[str, Any], key: str) -> tuple[int, int] | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be [height, width], got {value!r}")
    size = (int(value[0]), int(value[1]))
    if min(size) <= 1:
        raise ValueError(f"{key} must contain dimensions > 1, got {size}")
    return size


def _resolve_flow_canvases(
    record: dict[str, Any],
    *,
    flow_grid_size: tuple[int, int],
    image_source_size: tuple[int, int],
    image_target_size: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve GT source coordinates without a silent resize assumption."""

    declared_target = _size_from_record(record, "flow_target_size")
    if declared_target is not None and declared_target != flow_grid_size:
        raise ValueError(
            "flow_target_size must match the flow array grid; got "
            f"flow_target_size={declared_target}, flow_grid={flow_grid_size}"
        )
    declared_source = _size_from_record(record, "flow_source_size")
    if declared_source is not None:
        return declared_source, flow_grid_size
    if flow_grid_size == image_target_size:
        return image_source_size, flow_grid_size
    raise ValueError(
        "ambiguous GT-flow coordinate system: flow grid "
        f"{flow_grid_size} differs from target image {image_target_size}, and "
        "flow_source_size is missing. Add e.g. "
        '"flow_source_size":[1024,1024] to the manifest.'
    )


def source_affine_homography(
    source_size: Sequence[int],
    *,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    translation: Sequence[float] = (0.0, 0.0),
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return an old-source -> augmented-source pixel homography.

    Rotation and scaling are about the exact pixel center ``((W-1)/2,
    (H-1)/2)``. ``translation`` is ``(x, y)`` in pixels. Positive angles use
    image coordinates and therefore move the top edge toward the right (a
    visually clockwise rotation). This explicit convention is useful both for
    tests and for composing a source-only augmentation with backward flow.
    """

    source_h, source_w = int(source_size[0]), int(source_size[1])
    if source_h <= 1 or source_w <= 1:
        raise ValueError(
            f"source_size must contain dimensions > 1, got {source_size}"
        )
    if len(translation) != 2:
        raise ValueError(f"translation must be (x, y), got {translation!r}")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and > 0, got {scale}")
    angle = math.radians(float(angle_deg))
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    center_x = (source_w - 1) / 2.0
    center_y = (source_h - 1) / 2.0
    translate_x, translate_y = float(translation[0]), float(translation[1])
    return torch.tensor(
        (
            (
                cosine,
                -sine,
                center_x - cosine * center_x + sine * center_y + translate_x,
            ),
            (
                sine,
                cosine,
                center_y - sine * center_x - cosine * center_y + translate_y,
            ),
            (0.0, 0.0, 1.0),
        ),
        dtype=dtype,
        device=device,
    )


def _corner_perspective_homography(
    source_size: Sequence[int],
    destination_offsets: Tensor,
) -> Tensor:
    """Fit a canvas-corner homography from four destination offsets."""

    source_h, source_w = int(source_size[0]), int(source_size[1])
    if destination_offsets.shape != (4, 2):
        raise ValueError(
            "destination_offsets must be [4,2] ordered TL,TR,BR,BL, got "
            f"{tuple(destination_offsets.shape)}"
        )
    # Solve in float64: pixel coordinates around 512 make the unnormalised DLT
    # system needlessly inaccurate in float32, while this is only a 8x8 solve.
    solve_dtype = torch.float64
    corners = torch.tensor(
        (
            (0.0, 0.0),
            (source_w - 1.0, 0.0),
            (source_w - 1.0, source_h - 1.0),
            (0.0, source_h - 1.0),
        ),
        dtype=solve_dtype,
        device=destination_offsets.device,
    )
    destination = corners + destination_offsets.to(dtype=solve_dtype)
    rows: list[Tensor] = []
    values: list[Tensor] = []
    zero = corners.new_zeros(())
    one = corners.new_ones(())
    for (x, y), (u, v) in zip(corners, destination):
        rows.append(torch.stack((x, y, one, zero, zero, zero, -u * x, -u * y)))
        rows.append(torch.stack((zero, zero, zero, x, y, one, -v * x, -v * y)))
        values.extend((u, v))
    coefficients = torch.linalg.solve(torch.stack(rows), torch.stack(values))
    homography = torch.cat((coefficients, one.reshape(1))).reshape(3, 3)
    return homography.to(dtype=destination_offsets.dtype)


def _transform_pixel_map(pixel_map: Tensor, homography: Tensor) -> tuple[Tensor, Tensor]:
    """Apply a pixel homography and return map plus finite-denominator mask."""

    if pixel_map.ndim != 4 or pixel_map.shape[1] != 2:
        raise ValueError(f"pixel_map must be [B,2,H,W], got {tuple(pixel_map.shape)}")
    if homography.shape != (3, 3):
        raise ValueError(f"homography must be [3,3], got {tuple(homography.shape)}")
    batch, _, height, width = pixel_map.shape
    flat = pixel_map.reshape(batch, 2, -1)
    homogeneous = torch.cat((flat, torch.ones_like(flat[:, :1])), dim=1)
    transformed = torch.matmul(homography.unsqueeze(0), homogeneous)
    denominator = transformed[:, 2:3]
    projective_valid = torch.isfinite(transformed).all(dim=1, keepdim=True)
    projective_valid &= denominator.abs() > 1.0e-8
    safe_denominator = torch.where(
        projective_valid,
        denominator,
        torch.ones_like(denominator),
    )
    result = transformed[:, :2] / safe_denominator
    result = torch.where(
        projective_valid.expand_as(result),
        result,
        torch.zeros_like(result),
    )
    return (
        result.reshape(batch, 2, height, width),
        projective_valid.reshape(batch, 1, height, width),
    )


def _snap_map_to_canvas_edges(pixel_map: Tensor, source_size: Sequence[int]) -> Tensor:
    """Remove tiny trigonometric error at exact 90-degree canvas edges."""

    source_h, source_w = int(source_size[0]), int(source_size[1])
    result = pixel_map.clone()
    for channel, upper in ((0, source_w - 1.0), (1, source_h - 1.0)):
        coordinate = result[:, channel : channel + 1]
        coordinate = torch.where(coordinate.abs() < 1.0e-5, 0.0, coordinate)
        coordinate = torch.where((coordinate - upper).abs() < 1.0e-5, upper, coordinate)
        result[:, channel : channel + 1] = coordinate
    return result


def apply_source_homography(
    source: Tensor,
    backward_flow: Tensor,
    valid: Tensor,
    homography: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Warp only ``source`` and compose the matching GT backward flow.

    ``homography`` maps coordinates in the old source image to coordinates in
    the augmented source image. If the original GT absolute map is ``M(x)``,
    the returned flow represents ``H M(x) - x``. The target image is not an
    argument and therefore cannot accidentally be transformed. Validity keeps
    the original mask and additionally requires both old and new source
    coordinates to be finite and in their respective canvases.
    """

    if source.ndim != 3:
        raise ValueError(f"source must be [C,H,W], got {tuple(source.shape)}")
    if backward_flow.ndim != 3 or backward_flow.shape[0] != 2:
        raise ValueError(
            f"backward_flow must be [2,H,W], got {tuple(backward_flow.shape)}"
        )
    if valid.shape != (1, *backward_flow.shape[-2:]):
        raise ValueError(
            f"valid must be [1,H,W] matching flow, got {tuple(valid.shape)}"
        )
    source_size = tuple(int(value) for value in source.shape[-2:])
    if tuple(backward_flow.shape[-2:]) != source_size:
        raise ValueError(
            "source-only augmentation currently requires source and target work "
            "canvases to match, got "
            f"source={source_size}, target={tuple(backward_flow.shape[-2:])}"
        )

    homography = homography.to(
        device=backward_flow.device,
        dtype=backward_flow.dtype,
    )
    determinant = torch.linalg.det(homography)
    if not bool(torch.isfinite(determinant)) or abs(float(determinant)) < 1.0e-8:
        raise ValueError("source homography must be finite and invertible")

    old_map = backward_flow_to_map(backward_flow.unsqueeze(0))
    old_finite = torch.isfinite(old_map).all(dim=1, keepdim=True)
    old_in_bounds = pixel_map_valid_mask(old_map, source_size)
    safe_old_map = torch.nan_to_num(old_map, nan=0.0, posinf=0.0, neginf=0.0)
    new_map, forward_projective_valid = _transform_pixel_map(
        safe_old_map,
        homography,
    )
    new_map = _snap_map_to_canvas_edges(new_map, source_size)
    new_in_bounds = pixel_map_valid_mask(new_map, source_size)
    new_flow = map_to_backward_flow(new_map).squeeze(0)
    new_valid = (
        valid.unsqueeze(0).bool()
        & old_finite
        & old_in_bounds
        & forward_projective_valid
        & new_in_bounds
    ).squeeze(0)

    # Rendering the augmented source uses the inverse transform: every output
    # source pixel asks where it came from in the old source. Border padding
    # avoids synthetic black wedges; such extrapolated pixels never become GT
    # valid because validity above follows transformed old coordinates.
    output_grid = make_pixel_grid(
        1,
        *source_size,
        device=backward_flow.device,
        dtype=backward_flow.dtype,
    )
    inverse_map, _ = _transform_pixel_map(output_grid, torch.linalg.inv(homography))
    inverse_flow = map_to_backward_flow(inverse_map)
    augmented_source = backward_warp(
        source.unsqueeze(0),
        inverse_flow,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0)
    return augmented_source, new_flow, new_valid


def _parse_scale_range(value: Any) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        result = (float(value), float(value))
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        result = (float(value[0]), float(value[1]))
    else:
        raise ValueError(f"scale must be a number or [min,max], got {value!r}")
    if (
        not all(math.isfinite(item) and item > 0.0 for item in result)
        or result[0] > result[1]
    ):
        raise ValueError(f"scale must satisfy 0 < min <= max, got {result}")
    return result


def _parse_translation(value: Any) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        result = (float(value), float(value))
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        result = (float(value[0]), float(value[1]))
    else:
        raise ValueError(
            f"translation must be a number or [max_x,max_y], got {value!r}"
        )
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in result):
        raise ValueError(f"translation fractions must be in [0,1], got {result}")
    return result


class SourceGeometryAugment:
    """Random source-only rotation/scale/translation/light perspective."""

    _CONFIG_KEYS = {
        "probability",
        "max_rotation_deg",
        "scale",
        "translation",
        "perspective",
    }

    def __init__(
        self,
        *,
        probability: float = 0.0,
        max_rotation_deg: float = 0.0,
        scale: float | Sequence[float] = (1.0, 1.0),
        translation: float | Sequence[float] = (0.0, 0.0),
        perspective: float = 0.0,
    ) -> None:
        self.probability = float(probability)
        self.max_rotation_deg = float(max_rotation_deg)
        self.scale = _parse_scale_range(scale)
        self.translation = _parse_translation(translation)
        self.perspective = float(perspective)
        if (
            not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError(f"probability must be in [0,1], got {self.probability}")
        if (
            not math.isfinite(self.max_rotation_deg)
            or not 0.0 <= self.max_rotation_deg <= 180.0
        ):
            raise ValueError(
                f"max_rotation_deg must be in [0,180], got {self.max_rotation_deg}"
            )
        if (
            not math.isfinite(self.perspective)
            or not 0.0 <= self.perspective <= 0.25
        ):
            raise ValueError(f"perspective must be in [0,0.25], got {self.perspective}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SourceGeometryAugment":
        unknown = set(config) - cls._CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown source_geometry_augment keys: {sorted(unknown)}")
        return cls(**dict(config))

    @staticmethod
    def _sample_symmetric(maximum: float) -> float:
        return (2.0 * float(torch.rand(())) - 1.0) * maximum

    def sample_homography(self, source: Tensor) -> Tensor:
        source_size = tuple(int(value) for value in source.shape[-2:])
        source_h, source_w = source_size
        angle = self._sample_symmetric(self.max_rotation_deg)
        scale = self.scale[0] + float(torch.rand(())) * (
            self.scale[1] - self.scale[0]
        )
        translation = (
            self._sample_symmetric(self.translation[0]) * (source_w - 1),
            self._sample_symmetric(self.translation[1]) * (source_h - 1),
        )
        affine = source_affine_homography(
            source_size,
            angle_deg=angle,
            scale=scale,
            translation=translation,
            dtype=source.dtype,
            device=source.device,
        )
        if self.perspective == 0.0:
            return affine
        offset_scale = source.new_tensor((source_w - 1.0, source_h - 1.0))
        offsets = (2.0 * torch.rand((4, 2), device=source.device) - 1.0)
        offsets = offsets.to(dtype=source.dtype) * offset_scale * self.perspective
        perspective = _corner_perspective_homography(source_size, offsets)
        return perspective @ affine

    def __call__(
        self,
        source: Tensor,
        backward_flow: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.probability == 0.0 or float(torch.rand(())) >= self.probability:
            return source, backward_flow, valid
        return apply_source_homography(
            source,
            backward_flow,
            valid,
            self.sample_homography(source),
        )


class GuideArtifactAugment:
    """Simulate local redraws and text mismatches in a generated guide."""

    def __init__(self, probability: float = 0.65) -> None:
        self.probability = float(probability)

    def __call__(self, guide: Tensor) -> Tensor:
        if torch.rand(()) >= self.probability:
            return guide
        result = guide.clone()
        _, h, w = result.shape

        # Mild global appearance mismatch.
        contrast = 0.85 + 0.30 * torch.rand(())
        brightness = -0.05 + 0.10 * torch.rand(())
        result = ((result - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)
        if torch.rand(()) < 0.35:
            kernel = 3 if torch.rand(()) < 0.7 else 5
            result = F.avg_pool2d(
                result.unsqueeze(0), kernel, stride=1, padding=kernel // 2
            ).squeeze(0)

        # Cut/shift or erase small text-like regions. Geometry remains mostly
        # intact while local correspondence becomes deliberately unreliable.
        count = int(torch.randint(1, 5, ()).item())
        for _ in range(count):
            patch_h = int(torch.randint(max(3, h // 80), max(4, h // 15), ()).item())
            patch_w = int(torch.randint(max(8, w // 30), max(9, w // 6), ()).item())
            y0 = int(torch.randint(0, max(1, h - patch_h + 1), ()).item())
            x0 = int(torch.randint(0, max(1, w - patch_w + 1), ()).item())
            patch = result[:, y0 : y0 + patch_h, x0 : x0 + patch_w]
            if torch.rand(()) < 0.65:
                shift = int(torch.randint(-max(2, patch_w // 6), max(3, patch_w // 6), ()).item())
                result[:, y0 : y0 + patch_h, x0 : x0 + patch_w] = torch.roll(
                    patch, shifts=shift, dims=-1
                )
            else:
                fill = 0.90 + 0.10 * torch.rand((3, 1, 1), dtype=result.dtype)
                result[:, y0 : y0 + patch_h, x0 : x0 + patch_w] = fill
        return result.clamp(0.0, 1.0)


class DocumentFlowDataset(Dataset[dict[str, Any]]):
    """JSONL dataset with explicit backward-flow convention.

    Each record uses paths relative to the manifest directory:

    ``{"id":"001", "warped":"...", "target":"...", "guide":"...",
    "flow":"...npy", "valid":"...npy", "flow_format":"displacement"}``
    """

    def __init__(
        self,
        manifest: str | Path,
        work_size: tuple[int, int],
        *,
        augment_guide: bool = False,
        guide_artifact_prob: float = 0.65,
        source_geometry_augment: Mapping[str, Any] | None = None,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.work_size = (int(work_size[0]), int(work_size[1]))
        if self.work_size[0] % 8 or self.work_size[1] % 8:
            raise ValueError(f"RAFT work_size must be divisible by 8, got {self.work_size}")
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"manifest contains no samples: {self.manifest}")
        self.guide_augment = (
            GuideArtifactAugment(guide_artifact_prob) if augment_guide else None
        )
        if source_geometry_augment is not None and not isinstance(
            source_geometry_augment, Mapping
        ):
            raise ValueError("source_geometry_augment must be a configuration mapping")
        self.source_geometry_augment = (
            SourceGeometryAugment.from_config(source_geometry_augment)
            if source_geometry_augment is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        warped_path = _resolve(self.root, record["warped"])
        target_path = _resolve(self.root, record["target"])
        flow_path = _resolve(self.root, record["flow"])
        guide_path = _resolve(self.root, record.get("guide"))
        valid_path = _resolve(self.root, record.get("valid"))
        assert warped_path is not None and target_path is not None and flow_path is not None

        warped_native = _load_rgb(warped_path)
        target_native = _load_rgb(target_path)
        source_size = tuple(int(v) for v in warped_native.shape[-2:])
        target_size = tuple(int(v) for v in target_native.shape[-2:])

        flow = _as_flow_tensor(_load_array(flow_path, "flow"))
        flow_grid_size = tuple(int(v) for v in flow.shape[-2:])
        flow_source_native, flow_target_native = _resolve_flow_canvases(
            record,
            flow_grid_size=flow_grid_size,
            image_source_size=source_size,
            image_target_size=target_size,
        )
        flow = _convert_flow_format(
            flow,
            str(record.get("flow_format", "displacement")),
            source_size=flow_source_native,
        )
        flow = resize_backward_flow(
            flow,
            self.work_size,
            source_size_from=flow_source_native,
            source_size_to=self.work_size,
        ).squeeze(0)

        warped = _resize_image(warped_native, self.work_size)
        target = _resize_image(target_native, self.work_size)
        if guide_path is not None:
            guide = _resize_image(_load_rgb(guide_path), self.work_size)
            guide_available = True
            if self.guide_augment is not None:
                guide = self.guide_augment(guide)
        else:
            guide = torch.zeros_like(target)
            guide_available = False

        if valid_path is not None:
            valid_np = _load_array(valid_path, "valid").astype(np.float32)
            if valid_np.ndim == 2:
                valid_np = valid_np[None]
            valid = torch.from_numpy(np.ascontiguousarray(valid_np)).unsqueeze(0)
            valid = F.interpolate(valid, size=self.work_size, mode="nearest").squeeze(0) > 0.5
        else:
            valid = torch.ones((1, *self.work_size), dtype=torch.bool)
        valid &= flow_valid_mask(flow.unsqueeze(0), self.work_size).squeeze(0)
        valid &= torch.isfinite(flow).all(dim=0, keepdim=True)
        if self.source_geometry_augment is not None:
            warped, flow, valid = self.source_geometry_augment(warped, flow, valid)

        return {
            "id": str(record.get("id", index)),
            "warped": warped,
            "target": target,
            "guide": guide,
            "guide_available": torch.tensor(guide_available),
            "flow": flow,
            "valid": valid,
            "native_source_size": torch.tensor(source_size),
            "native_target_size": torch.tensor(target_size),
            "flow_source_size": torch.tensor(flow_source_native),
            "flow_target_size": torch.tensor(flow_target_native),
        }


def write_manifest(records: list[dict[str, Any]], destination: str | Path) -> None:
    """Write records in the format consumed by :class:`DocumentFlowDataset`."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
