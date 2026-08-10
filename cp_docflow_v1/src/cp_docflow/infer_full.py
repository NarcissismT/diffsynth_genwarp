"""Infer a final map and sample the native warped image exactly once."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .checkpoint import COORDINATE_CONTRACT, file_sha256, load_full_checkpoint
from .config import build_full_model
from .geometry import resize_backward_map, warp_with_backward_map
from .infer import _device, _load_rgb, _save_rgb


@torch.no_grad()
def infer_full(
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    output_size: tuple[int, int] | None = None,
    device_name: str = "auto",
) -> dict[str, object]:
    device = _device(device_name)
    payload = load_full_checkpoint(checkpoint_path, map_location=device)
    model = build_full_model(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.set_execution_stage(payload["training_stage"])
    model.eval()
    native_source = _load_rgb(Path(image_path)).to(device)
    native_source_size = tuple(int(value) for value in native_source.shape[-2:])
    native_output_size = output_size or native_source_size
    work_source_size = tuple(int(value) for value in payload["input_work_size"])
    work_output_size = tuple(int(value) for value in payload["output_work_size"])
    model_input = F.interpolate(
        native_source,
        size=work_source_size,
        mode="bilinear",
        align_corners=False,
    )
    # No work-size preview is rendered.  The only RGB sampling operation that
    # produces an image happens below against native_source.
    started = time.perf_counter()
    prediction = model(
        model_input, output_size=work_output_size, render=False, profile=True
    )
    native_map = resize_backward_map(
        prediction["backward_map"],
        native_output_size,
        source_size_from=work_source_size,
        source_size_to=native_source_size,
    )
    native_confidence = F.interpolate(
        prediction["confidence"],
        native_output_size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    rectified, valid = warp_with_backward_map(
        native_source.float(),
        native_map,
        padding_mode="border",
        return_valid=True,
    )
    runtime_seconds = time.perf_counter() - started
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    paths = {
        "rectified_image": destination / f"{stem}_rectified.png",
        "backward_map": destination / f"{stem}_backward_map.npy",
        "coarse_backward_map": destination / f"{stem}_coarse_backward_map.npy",
        "confidence": destination / f"{stem}_confidence.npy",
        "confidence_preview": destination / f"{stem}_confidence.png",
        "fold_mask": destination / f"{stem}_fold_mask.png",
        "valid": destination / f"{stem}_valid.png",
        "metadata": destination / f"{stem}_metadata.json",
    }
    _save_rgb(rectified, paths["rectified_image"])
    np.save(
        paths["backward_map"],
        native_map.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32),
    )
    native_coarse = resize_backward_map(
        prediction["coarse_backward_map"],
        native_output_size,
        source_size_from=work_source_size,
        source_size_to=native_source_size,
    )
    np.save(
        paths["coarse_backward_map"],
        native_coarse.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32),
    )
    np.save(
        paths["confidence"],
        native_confidence.detach().cpu().squeeze().numpy().astype(np.float32),
    )
    Image.fromarray(valid.detach().cpu().squeeze().numpy().astype(np.uint8) * 255).save(
        paths["valid"]
    )
    Image.fromarray(
        (native_confidence.detach().cpu().squeeze().numpy() * 255.0)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    ).save(paths["confidence_preview"])
    from .metrics import jacobian_determinant

    determinant = jacobian_determinant(native_map)
    fold_mask = determinant.le(0.0).detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    Image.fromarray(fold_mask).save(paths["fold_mask"])
    metadata: dict[str, object] = {
        # v2 names the two iterative stages explicitly. ``fm_steps`` is kept
        # as a v1 compatibility alias for ``flow_matching_steps``.
        "schema": "docgrid_flow.inference.v2",
        "architecture_name": "DocGrid-Flow",
        "coordinate_contract": COORDINATE_CONTRACT,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_stage": payload["training_stage"],
        "input": str(Path(image_path).resolve()),
        "native_source_size": list(native_source_size),
        "native_output_size": list(native_output_size),
        "work_source_size": list(work_source_size),
        "work_output_size": list(work_output_size),
        "qwen_backend": prediction["qwen_backend"],
        "qwen_vae_decoder_used": False,
        "qwen_latent_output_discarded": prediction["qwen_backend"] == "qwen",
        "fm_steps": len(prediction["flow_matching_sequence"]),
        "refiner_iterations": len(prediction["refiner_sequence"]),
        "final_rgb_source": "single_grid_sample_from_native_warped_image",
        "valid_ratio": float(valid.float().mean().item()),
        "confidence_mean": float(native_confidence.mean().item()),
        "runtime_seconds": runtime_seconds,
        "runtime_breakdown": prediction.get("runtime_breakdown", {}),
        "backward_map": str(paths["backward_map"]),
        "coarse_backward_map": str(paths["coarse_backward_map"]),
        "confidence": str(paths["confidence"]),
        "confidence_preview": str(paths["confidence_preview"]),
        "fold_mask": str(paths["fold_mask"]),
        "rectified_image": str(paths["rectified_image"]),
    }
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-height", type=int)
    parser.add_argument("--output-width", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (args.output_height is None) != (args.output_width is None):
        parser.error("--output-height and --output-width must be provided together")
    requested = (
        None
        if args.output_height is None
        else (args.output_height, args.output_width)
    )
    result = infer_full(
        args.checkpoint,
        args.image,
        args.output_dir,
        output_size=requested,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
