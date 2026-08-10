#!/usr/bin/env python3
"""Single-GPU 512/518 audit against historical teacher deployment outputs.

This diagnostic deliberately reproduces the old displacement scaling and
``align_corners`` mismatch.  It is not the corrected project inference path;
its sole purpose is to identify which historical input size most likely made
an existing reference directory.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from diffusion2raft.postprocess import TorchScriptLamaInpainter


def _teacher_input(
    image_bgr: np.ndarray, size: int, device: torch.device
) -> torch.Tensor:
    height, width = image_bgr.shape[:2]
    if min(height, width) > 2048:
        scale = 1024.0 / min(height, width)
        middle = cv2.resize(
            image_bgr,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        resized = cv2.resize(
            middle, (size, size), interpolation=cv2.INTER_AREA
        )
    elif min(height, width) > size:
        resized = cv2.resize(
            image_bgr, (size, size), interpolation=cv2.INTER_AREA
        )
    else:
        resized = cv2.resize(
            image_bgr, (size, size), interpolation=cv2.INTER_LINEAR
        )
    return (
        torch.from_numpy(np.ascontiguousarray(resized))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
        .to(device)
    )


def _run_teacher(
    teacher: torch.jit.ScriptModule,
    image_bgr: np.ndarray,
    input_size: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = _teacher_input(image_bgr, input_size, device)
    dtype = next(teacher.parameters()).dtype
    if dtype in {torch.float16, torch.bfloat16}:
        tensor = tensor.to(dtype)
    elif dtype != torch.float32:
        raise ValueError(f"unsupported teacher parameter dtype: {dtype}")
    autocast_enabled = device.type == "cuda" and dtype == torch.float32
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=autocast_enabled,
    ):
        output = teacher(tensor, tensor)
    if not isinstance(output, (list, tuple)) or not output:
        raise TypeError("teacher must return a non-empty flow sequence")
    flow = output[-1].float()
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"invalid teacher flow shape: {tuple(flow.shape)}")
    return flow


def _historical_unwarp(
    image_bgr: np.ndarray,
    flow: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_bgr.shape[:2]
    _, _, flow_height, flow_width = flow.shape
    image = (
        torch.from_numpy(np.ascontiguousarray(image_bgr))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
        .to(flow.device)
    )
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    pixel_grid = torch.stack((x, y), dim=0).unsqueeze(0).to(flow.device)
    restored = gaussian_blur(flow.float(), [39, 39])
    restored = F.interpolate(
        restored,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    restored[:, 0] *= width / flow_width
    restored[:, 1] *= height / flow_height
    pixel_map = restored + pixel_grid
    normalized = pixel_map.clone()
    normalized[:, 0] = 2.0 * normalized[:, 0] / (width - 1) - 1.0
    normalized[:, 1] = 2.0 * normalized[:, 1] / (height - 1) - 1.0
    invalid = ((normalized > 1.0) | (normalized < -1.0)).any(dim=1, keepdim=True)
    sampled = F.grid_sample(
        image,
        normalized.permute(0, 2, 3, 1),
        padding_mode="zeros",
        align_corners=False,
    )
    raw_bgr = (
        sampled[0]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return raw_bgr, invalid


def _inpaint_bgr(
    inpainter: TorchScriptLamaInpainter,
    raw_bgr: np.ndarray,
    invalid: torch.Tensor,
) -> np.ndarray:
    raw_rgb = (
        torch.from_numpy(np.ascontiguousarray(raw_bgr[:, :, ::-1]))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
        .to(invalid.device)
    )
    output_rgb, _ = inpainter.forward_with_mask(raw_rgb, invalid)
    return (
        output_rgb[0]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .cpu()
        .numpy()[:, :, ::-1]
        .copy()
    )


def _comparison(actual: np.ndarray, reference_path: Path) -> dict[str, Any]:
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise ValueError(f"OpenCV failed to read reference: {reference_path}")
    if reference.shape != actual.shape:
        raise ValueError(
            f"reference shape {reference.shape} != output shape {actual.shape}"
        )
    difference = actual.astype(np.float32) - reference.astype(np.float32)
    mae = float(np.abs(difference).mean())
    mse = float(np.square(difference).mean())
    return {
        "reference": str(reference_path.resolve()),
        "mae_u8": mae,
        "rmse_u8": math.sqrt(mse),
        "psnr_db": None if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse),
        "exact_pixel_fraction": float(np.all(difference == 0.0, axis=2).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--lama", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 518])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "this traced teacher hard-codes cuda:0 inside its RoPE graph; "
            "the audit requires --device cuda:0"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this audit on a GPU node")
    if device.index not in {None, 0}:
        raise RuntimeError("the historical trace requires visible device cuda:0")
    torch.cuda.set_device(0)
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV failed to read input: {args.image}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    teacher = torch.jit.load(
        str(args.teacher.resolve()), map_location=device
    ).eval()
    inpainter = TorchScriptLamaInpainter(
        args.lama,
        device=device,
        inpaint_size=512,
        dilation_kernel=11,
    )
    report: dict[str, Any] = {
        "note": "historical CUDA FP16-autocast deployment diagnostic",
        "device": str(device),
        "teacher": str(args.teacher.resolve()),
        "lama": inpainter.identity,
        "image": str(args.image.resolve()),
        "runs": [],
    }
    for size in args.sizes:
        run_started = time.monotonic()
        flow = _run_teacher(teacher, image_bgr, int(size), device)
        raw_bgr, invalid = _historical_unwarp(image_bgr, flow)
        final_bgr = _inpaint_bgr(inpainter, raw_bgr, invalid)
        output_path = args.output_dir / f"historical_input_{int(size)}.jpg"
        if not cv2.imwrite(str(output_path), final_bgr):
            raise RuntimeError(f"failed to save {output_path}")
        encoded_bgr = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if encoded_bgr is None:
            raise RuntimeError(f"failed to re-read {output_path}")
        report["runs"].append(
            {
                "input_size": int(size),
                "flow_shape": list(flow.shape),
                "invalid_fraction": float(invalid.float().mean()),
                "seconds": time.monotonic() - run_started,
                "output": str(output_path.resolve()),
                "comparisons": [
                    _comparison(encoded_bgr, reference)
                    for reference in args.reference
                ],
            }
        )
    report["total_seconds"] = time.monotonic() - started
    report_path = args.output_dir / "audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
