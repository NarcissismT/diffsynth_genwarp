"""Run Stage-1 inference and sample the native source image exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .checkpoint import COORDINATE_CONTRACT, file_sha256, load_checkpoint
from .config import build_coarse_model
from .geometry import resize_backward_map, warp_with_backward_map


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
    Image.fromarray((array.numpy() * 255.0 + 0.5).astype(np.uint8)).save(path)


@torch.no_grad()
def infer(
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    output_size: tuple[int, int] | None = None,
    device_name: str = "auto",
) -> dict[str, object]:
    device = _device(device_name)
    payload = load_checkpoint(checkpoint_path, map_location=device)
    model = build_coarse_model(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    native_source = _load_rgb(Path(image_path)).to(device)
    native_source_size = tuple(int(v) for v in native_source.shape[-2:])
    native_output_size = output_size or native_source_size
    work_source_size = tuple(int(v) for v in payload["input_work_size"])
    work_output_size = tuple(int(v) for v in payload["output_work_size"])
    model_input = F.interpolate(
        native_source,
        size=work_source_size,
        mode="bilinear",
        align_corners=False,
    )
    # render=False is important: the deliverable image below is produced by
    # exactly one grid_sample from native_source, never by a work-size preview.
    prediction = model(
        model_input,
        output_size=work_output_size,
        render=False,
    )
    native_map = resize_backward_map(
        prediction["backward_map"],
        native_output_size,
        source_size_from=work_source_size,
        source_size_to=native_source_size,
    )
    native_confidence = F.interpolate(
        prediction["confidence"],
        size=native_output_size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    rectified, valid = warp_with_backward_map(
        native_source.float(),
        native_map,
        padding_mode="border",
        return_valid=True,
    )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    image_output = destination / f"{stem}_rectified.png"
    map_output = destination / f"{stem}_backward_map.npy"
    valid_output = destination / f"{stem}_valid.png"
    confidence_output = destination / f"{stem}_confidence.npy"
    metadata_output = destination / f"{stem}_metadata.json"
    _save_rgb(rectified, image_output)
    np.save(
        map_output,
        native_map.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32),
    )
    np.save(
        confidence_output,
        native_confidence.detach().cpu().squeeze(0).squeeze(0).numpy().astype(np.float32),
    )
    valid_array = valid.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    Image.fromarray(valid_array).save(valid_output)
    metadata: dict[str, object] = {
        "schema": "cp_docflow.inference.v1",
        "coordinate_contract": COORDINATE_CONTRACT,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "input": str(Path(image_path).resolve()),
        "native_source_size": list(native_source_size),
        "native_output_size": list(native_output_size),
        "work_source_size": list(work_source_size),
        "work_output_size": list(work_output_size),
        "valid_ratio": float(valid.float().mean().item()),
        "confidence_mean": float(native_confidence.mean().item()),
        "final_rgb_source": "single_grid_sample_from_native_warped_image",
        "qwen_vae_decoder_used": False,
        "backward_map": str(map_output),
        "confidence": str(confidence_output),
        "rectified_image": str(image_output),
    }
    with metadata_output.open("w", encoding="utf-8") as handle:
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
    requested_size = (
        None
        if args.output_height is None
        else (args.output_height, args.output_width)
    )
    metadata = infer(
        args.checkpoint,
        args.image,
        args.output_dir,
        output_size=requested_size,
        device_name=args.device,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
