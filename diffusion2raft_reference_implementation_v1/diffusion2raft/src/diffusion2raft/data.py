"""Dataset and guide-artifact augmentation for rectification flow training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .geometry import (
    flow_valid_mask,
    map_to_backward_flow,
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
        # GT flow may be authored on its own canvas, independent of the image
        # resolution (e.g. 1024 flow with 512 images). Use the flow's native
        # target grid, and its native *source* canvas for coordinate scaling.
        flow_target_native = tuple(int(v) for v in flow.shape[-2:])
        flow_source_native = tuple(
            int(v) for v in record.get("flow_source_size", flow_target_native)
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
        }


def write_manifest(records: list[dict[str, Any]], destination: str | Path) -> None:
    """Write records in the format consumed by :class:`DocumentFlowDataset`."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
