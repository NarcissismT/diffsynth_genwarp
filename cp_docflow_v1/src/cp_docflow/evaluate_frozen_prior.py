"""Evaluate the frozen supervised TorchScript prior on a DocGrid manifest.

This adapter exists only to produce a reproducible comparison baseline.  The
teacher predicts a backward *displacement* on a stretched 512-square canvas;
the evaluator converts it to DocGrid-Flow's absolute source-pixel map before
computing any metric.  It is never registered as part of the trainable model.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from .checkpoint import COORDINATE_CONTRACT, file_sha256
from .evaluate_full import evaluate_full
from .geometry import canonical_backward_map, resize_backward_map

BASELINE_SCHEMA = "docgrid_flow.frozen_supervised_prior_baseline.v1"


def _pair(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain height and width")
    result = (int(value[0]), int(value[1]))
    if min(result) < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _autocast_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower().replace("torch.", "")
    choices = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError("autocast_dtype must be float16, bfloat16, or float32")
    return choices[normalized]


def _gaussian_blur(value: Tensor, kernel_size: int) -> Tensor:
    """Match torchvision's default-sigma Gaussian blur without a dependency."""

    if kernel_size <= 1:
        return value
    if kernel_size % 2 != 1:
        raise ValueError("blur_kernel must be a positive odd integer")
    sigma = 0.15 * kernel_size + 0.35
    coordinate = torch.arange(
        kernel_size, device=value.device, dtype=value.dtype
    ) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    weight = kernel_2d.expand(value.shape[1], 1, -1, -1)
    padding = kernel_size // 2
    padded = F.pad(value, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, weight, groups=value.shape[1])


class FrozenSupervisedPriorAdapter(nn.Module):
    """Expose the archived teacher through the evaluator's map-output API."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: torch.device | str,
        expected_sha256: str,
        expected_size_bytes: int,
        input_size: int,
        flow_size: int,
        blur_kernel: int,
        autocast_dtype: str,
        requires_logical_cuda0: bool,
    ) -> None:
        super().__init__()
        path = Path(checkpoint).resolve(strict=True)
        self.teacher_device = torch.device(device)
        if self.teacher_device.type == "cuda" and self.teacher_device.index is None:
            self.teacher_device = torch.device("cuda", torch.cuda.current_device())
        if requires_logical_cuda0 and self.teacher_device != torch.device("cuda", 0):
            raise RuntimeError(
                "the archived teacher requires process-local logical cuda:0"
            )
        if int(input_size) <= 1 or int(flow_size) <= 1:
            raise ValueError("input_size and flow_size must be greater than one")
        if int(blur_kernel) < 1 or int(blur_kernel) % 2 != 1:
            raise ValueError("blur_kernel must be a positive odd integer")
        stat_before = path.stat()
        if stat_before.st_size != int(expected_size_bytes):
            raise ValueError(
                "frozen prior size mismatch: "
                f"expected {expected_size_bytes}, got {stat_before.st_size}"
            )
        actual_sha256 = file_sha256(path)
        if actual_sha256.lower() != str(expected_sha256).lower():
            raise ValueError(
                "frozen prior SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        stat_after_hash = path.stat()
        if (
            stat_before.st_size,
            stat_before.st_mtime_ns,
            stat_before.st_ino,
        ) != (
            stat_after_hash.st_size,
            stat_after_hash.st_mtime_ns,
            stat_after_hash.st_ino,
        ):
            raise RuntimeError("frozen prior changed while it was being hashed")
        teacher = torch.jit.load(str(path), map_location=self.teacher_device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
        stat_after_load = path.stat()
        if (
            stat_before.st_size,
            stat_before.st_mtime_ns,
            stat_before.st_ino,
        ) != (
            stat_after_load.st_size,
            stat_after_load.st_mtime_ns,
            stat_after_load.st_ino,
        ):
            raise RuntimeError("frozen prior changed while it was being loaded")
        object.__setattr__(self, "_teacher", teacher)
        self.checkpoint_path = str(path)
        self.checkpoint_sha256 = actual_sha256
        self.checkpoint_size_bytes = int(stat_before.st_size)
        self.input_size = int(input_size)
        self.flow_size = int(flow_size)
        self.blur_kernel = int(blur_kernel)
        self.autocast_dtype_name = str(autocast_dtype)
        self.autocast_dtype = _autocast_dtype(autocast_dtype)
        self.requires_logical_cuda0 = bool(requires_logical_cuda0)
        self.external_teacher_parameter_count = sum(
            parameter.numel() for parameter in teacher.parameters()
        )

    @property
    def teacher(self) -> torch.jit.ScriptModule:
        return object.__getattribute__(self, "_teacher")

    def train(self, mode: bool = True) -> "FrozenSupervisedPriorAdapter":
        super().train(mode)
        self.teacher.eval()
        return self

    def _teacher_flow(self, warped: Tensor) -> Tensor:
        if warped.ndim != 4 or warped.shape[1] != 3:
            raise ValueError(f"warped must be [B,3,H,W], got {tuple(warped.shape)}")
        if warped.device != self.teacher_device:
            raise RuntimeError(
                f"teacher was loaded on {self.teacher_device}, input is on {warped.device}"
            )
        teacher_input = F.interpolate(
            warped.float(),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )[:, (2, 1, 0)].contiguous()
        autocast_enabled = self.teacher_device.type == "cuda" and self.autocast_dtype in {
            torch.float16,
            torch.bfloat16,
        }
        self.teacher.eval()
        autocast_context = (
            torch.autocast(
                device_type=self.teacher_device.type,
                dtype=self.autocast_dtype,
            )
            if autocast_enabled
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            output = self.teacher(teacher_input, teacher_input)
        if not isinstance(output, (list, tuple)) or not output:
            raise TypeError("frozen prior must return a non-empty Tensor list")
        flow = output[-1]
        expected = (warped.shape[0], 2, self.flow_size, self.flow_size)
        if not isinstance(flow, Tensor) or tuple(flow.shape) != expected:
            actual = type(flow).__name__ if not isinstance(flow, Tensor) else tuple(flow.shape)
            raise ValueError(f"frozen prior flow must be {expected}, got {actual}")
        flow = _gaussian_blur(flow.float(), self.blur_kernel)
        if not bool(torch.isfinite(flow).all()):
            raise ValueError("frozen prior returned non-finite flow")
        return flow

    def forward(
        self,
        warped: Tensor,
        *,
        output_size: Sequence[int] | None = None,
        render: bool = False,
        profile: bool = False,
    ) -> dict[str, Any]:
        del render, profile
        native_size = tuple(int(value) for value in warped.shape[-2:])
        target_size = native_size if output_size is None else _pair(output_size, "output_size")
        flow = self._teacher_flow(warped)
        teacher_map = canonical_backward_map(
            warped.shape[0],
            (self.flow_size, self.flow_size),
            (self.input_size, self.input_size),
            device=flow.device,
            dtype=flow.dtype,
        ) + flow
        backward_map = resize_backward_map(
            teacher_map,
            target_size,
            source_size_from=(self.input_size, self.input_size),
            source_size_to=native_size,
        )
        canonical = canonical_backward_map(
            warped.shape[0],
            target_size,
            native_size,
            device=backward_map.device,
            dtype=backward_map.dtype,
        )
        confidence = torch.ones(
            warped.shape[0],
            1,
            *target_size,
            device=backward_map.device,
            dtype=backward_map.dtype,
        )
        zeros_xy = torch.zeros_like(backward_map)
        zeros_gate = torch.zeros_like(confidence)
        return {
            "backward_map": backward_map,
            "coarse_backward_map": backward_map,
            "coarse_low": backward_map,
            "confidence": confidence,
            "composition_gate": zeros_gate,
            "residual_proposal": zeros_xy,
            "qwen_gate": zeros_gate,
            "refiner_sequence": [],
            "canonical_map": canonical,
            "runtime_breakdown": {},
        }


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("baseline config must contain a mapping")
    if config.get("schema") != BASELINE_SCHEMA:
        raise ValueError(
            f"baseline config schema must be {BASELINE_SCHEMA!r}, got {config.get('schema')!r}"
        )
    return config_path, config


@torch.no_grad()
def evaluate_frozen_prior(
    config_path: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "cuda",
    max_visualizations: int = 32,
) -> dict[str, Any]:
    config_file, config = _load_config(config_path)
    checkpoint = Path(str(config["checkpoint"]))
    if not checkpoint.is_absolute():
        checkpoint = (config_file.parent / checkpoint).resolve()
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    adapter = FrozenSupervisedPriorAdapter(
        checkpoint,
        device=device,
        expected_sha256=str(config["checkpoint_sha256"]),
        expected_size_bytes=int(config["checkpoint_size_bytes"]),
        input_size=int(config["input_size"]),
        flow_size=int(config["flow_size"]),
        blur_kernel=int(config["blur_kernel"]),
        autocast_dtype=str(config["autocast_dtype"]),
        requires_logical_cuda0=bool(config["requires_logical_cuda0"]),
    )
    input_work_size = _pair(config["evaluation_input_work_size"], "evaluation_input_work_size")
    output_work_size = _pair(
        config["evaluation_output_work_size"], "evaluation_output_work_size"
    )
    prepared_payload = {
        "model_config": {
            "qwen_backend": "none",
            "baseline_backend": "frozen_supervised_torchscript",
        },
        "training_stage": "frozen_prior",
        "training_seed": None,
        "input_work_size": input_work_size,
        "output_work_size": output_work_size,
        "data_contract": None,
        "gate_receipts": None,
    }
    identity = {
        "schema": BASELINE_SCHEMA,
        "config": str(config_file),
        "config_sha256": file_sha256(config_file),
        "checkpoint": adapter.checkpoint_path,
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "checkpoint_size_bytes": adapter.checkpoint_size_bytes,
        "input_contract": "RGB_[0,1]_resized_512_square_then_BGR",
        "teacher_output_contract": "backward_displacement_xy_pixel_512_square",
        "evaluation_output_contract": COORDINATE_CONTRACT,
        "blur_kernel": adapter.blur_kernel,
        "autocast_dtype": adapter.autocast_dtype_name,
        "requires_logical_cuda0": adapter.requires_logical_cuda0,
        "external_teacher_parameter_count": adapter.external_teacher_parameter_count,
    }
    return evaluate_full(
        adapter.checkpoint_path,
        manifest,
        output_dir,
        device_name=str(device),
        allowed_label_provenance={"analytic_gt", "renderer_gt"},
        gate=False,
        max_visualizations=max_visualizations,
        prepared_model=adapter,
        prepared_payload=prepared_payload,
        checkpoint_sha256=adapter.checkpoint_sha256,
        report_metadata={"baseline_identity": identity},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-visualizations", type=int, default=32)
    args = parser.parse_args()
    report = evaluate_frozen_prior(
        args.config,
        args.manifest,
        args.output_dir,
        device_name=args.device,
        max_visualizations=args.max_visualizations,
    )
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
