"""Stage-5 low-page, structure-patch, and deployment-page view mixture."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .data import DocumentMapDataset
from .geometry import resize_backward_map_with_mask


def _size(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must be [height,width]")
    result = (int(value[0]), int(value[1]))
    if min(result) < 4:
        raise ValueError(f"{name} dimensions must be at least four")
    return result


def _normalized_response(value: Tensor) -> Tensor:
    scale = value.mean().clamp_min(1.0e-4)
    return (value / (4.0 * scale)).clamp(0.0, 1.0)


def _derived_structure(rectified: Tensor, valid: Tensor) -> dict[str, Tensor]:
    gray = rectified.float().mean(dim=0, keepdim=True)
    horizontal = F.pad(torch.abs(gray[:, 1:] - gray[:, :-1]), (0, 0, 0, 1))
    vertical = F.pad(torch.abs(gray[..., 1:] - gray[..., :-1]), (0, 1, 0, 0))
    horizontal = F.avg_pool2d(
        horizontal[None], (3, 9), stride=1, padding=(1, 4)
    )[0]
    vertical = F.avg_pool2d(
        vertical[None], (9, 3), stride=1, padding=(4, 1)
    )[0]
    valid_float = valid.float()[None]
    eroded = -F.max_pool2d(-valid_float, 3, stride=1, padding=1)[0]
    boundary = (valid.float() - eroded).clamp(0.0, 1.0)
    # Explicit canvas-edge emphasis ensures margins/corners remain represented
    # even when the synthetic valid mask covers the complete image.
    band = max(1, min(rectified.shape[-2:]) // 64)
    boundary = boundary.clone()
    boundary[..., :band, :] = 1.0
    boundary[..., -band:, :] = 1.0
    boundary[..., :, :band] = 1.0
    boundary[..., :, -band:] = 1.0
    return {
        "horizontal_structure": _normalized_response(horizontal),
        "vertical_structure": _normalized_response(vertical),
        "boundary_structure": boundary,
    }


class FullPageMixedViewDataset(Dataset):
    """Deterministic per-epoch Stage-5 view sampling.

    A structure crop only crops the *target* tensors. The warped source remains
    the complete high-resolution page and map values remain absolute pixels in
    that complete source. ``target_window`` records where the crop lives in the
    complete target canvas so the model can construct the correct canonical
    coordinate tokens.
    """

    VIEW_NAMES = ("low_page", "structure_patch", "full_page")

    def __init__(
        self,
        base: DocumentMapDataset,
        config: Mapping[str, Any],
        *,
        seed: int,
    ) -> None:
        self.base = base
        self.records = base.records
        self.manifest = base.manifest
        self.seed = int(seed)
        self.epoch = 0
        self.low_input_size = _size(
            config.get("low_input_size", [512, 512]), "low_input_size"
        )
        self.low_output_size = _size(
            config.get("low_output_size", self.low_input_size), "low_output_size"
        )
        self.patch_size = _size(
            config.get("structure_patch_size", [512, 512]),
            "structure_patch_size",
        )
        probabilities = config.get(
            "probabilities",
            {"low_page": 0.25, "structure_patch": 0.50, "full_page": 0.25},
        )
        if not isinstance(probabilities, Mapping):
            raise ValueError("stage5_mix.probabilities must be a mapping")
        unknown = set(probabilities) - set(self.VIEW_NAMES)
        if unknown:
            raise ValueError(f"unknown Stage-5 views: {sorted(unknown)}")
        weights = torch.tensor(
            [float(probabilities.get(name, 0.0)) for name in self.VIEW_NAMES]
        )
        if bool((weights < 0).any()) or float(weights.sum()) <= 0.0:
            raise ValueError("Stage-5 view probabilities must be non-negative and non-zero")
        self.probabilities = weights / weights.sum()
        multiplier = float(config.get("samples_per_epoch_multiplier", 3.0))
        if multiplier <= 0.0:
            raise ValueError("samples_per_epoch_multiplier must be positive")
        self.length = max(1, int(math.ceil(len(base) * multiplier)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def _generator(self, index: int) -> torch.Generator:
        # Prime multipliers keep view/crop draws stable across worker counts.
        value = self.seed + 1_000_003 * self.epoch + 9_973 * int(index)
        return torch.Generator().manual_seed(value)

    @staticmethod
    def _ensure_structure(sample: dict[str, Any]) -> dict[str, Tensor]:
        keys = (
            "horizontal_structure",
            "vertical_structure",
            "boundary_structure",
        )
        if all(name in sample for name in keys):
            return {name: sample[name].float() for name in keys}
        return _derived_structure(sample["rectified_image"], sample["valid_mask"])

    def _low_page(
        self, sample: dict[str, Any], structures: dict[str, Tensor]
    ) -> dict[str, Any]:
        source_size = tuple(int(value) for value in sample["input_size"])
        target_size = tuple(int(value) for value in sample["output_size"])
        result = dict(sample)
        result["warped_image"] = F.interpolate(
            sample["warped_image"][None],
            self.low_input_size,
            mode="bilinear",
            align_corners=False,
        )[0]
        result["rectified_image"] = F.interpolate(
            sample["rectified_image"][None],
            self.low_output_size,
            mode="bilinear",
            align_corners=False,
        )[0]
        backward_map, valid = resize_backward_map_with_mask(
            sample["backward_map"][None],
            sample["valid_mask"][None],
            self.low_output_size,
            source_size_from=source_size,
            source_size_to=self.low_input_size,
        )
        result["backward_map"] = backward_map[0]
        result["valid_mask"] = valid[0]
        for name, value in structures.items():
            result[name] = F.interpolate(
                value[None], self.low_output_size, mode="bilinear", align_corners=False
            )[0]
        result["input_size"] = torch.tensor(self.low_input_size, dtype=torch.int64)
        result["output_size"] = torch.tensor(self.low_output_size, dtype=torch.int64)
        result["target_canvas_size"] = torch.tensor(
            self.low_output_size, dtype=torch.int64
        )
        result["target_window"] = torch.tensor(
            (0.0, 0.0, float(self.low_output_size[1]), float(self.low_output_size[0]))
        )
        result["training_view"] = "low_page"
        return result

    def _structure_patch(
        self,
        sample: dict[str, Any],
        structures: dict[str, Tensor],
        generator: torch.Generator,
    ) -> dict[str, Any]:
        canvas_h, canvas_w = (int(value) for value in sample["output_size"])
        patch_h = min(self.patch_size[0], canvas_h)
        patch_w = min(self.patch_size[1], canvas_w)
        score = (
            structures["horizontal_structure"]
            + structures["vertical_structure"]
            + 2.0 * structures["boundary_structure"]
        ) * sample["valid_mask"].float()
        flat = score.flatten().clamp_min(0.0)
        if float(flat.sum()) <= 0.0:
            flat = sample["valid_mask"].float().flatten()
        selected = int(torch.multinomial(flat + 1.0e-8, 1, generator=generator))
        center_y, center_x = divmod(selected, canvas_w)
        y0 = min(max(center_y - patch_h // 2, 0), canvas_h - patch_h)
        x0 = min(max(center_x - patch_w // 2, 0), canvas_w - patch_w)
        slices = (..., slice(y0, y0 + patch_h), slice(x0, x0 + patch_w))
        result = dict(sample)
        for name in ("rectified_image", "backward_map", "valid_mask"):
            result[name] = sample[name][slices].contiguous()
        for name, value in structures.items():
            result[name] = value[slices].contiguous()
        if not bool(result["valid_mask"].any()):
            raise ValueError("structure patch contains no valid GT map pixels")
        result["output_size"] = torch.tensor((patch_h, patch_w), dtype=torch.int64)
        result["target_canvas_size"] = torch.tensor(
            (canvas_h, canvas_w), dtype=torch.int64
        )
        result["target_window"] = torch.tensor(
            (float(x0), float(y0), float(patch_w), float(patch_h)),
            dtype=torch.float32,
        )
        result["training_view"] = "structure_patch"
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = self._generator(index)
        sample = self.base[index % len(self.base)]
        structures = self._ensure_structure(sample)
        view_index = int(torch.multinomial(self.probabilities, 1, generator=generator))
        view = self.VIEW_NAMES[view_index]
        if view == "low_page":
            return self._low_page(sample, structures)
        if view == "structure_patch":
            return self._structure_patch(sample, structures, generator)
        result = dict(sample)
        result.update(structures)
        result["training_view"] = "full_page"
        return result

