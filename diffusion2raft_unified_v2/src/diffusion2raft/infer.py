"""Run document rectification and sample pixels only from the original image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import load_config
from .geometry import backward_warp, flow_valid_mask, resize_backward_flow
from .losses import jacobian_determinant
from .models import build_rectifier


def _load_image(path: str | Path) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor, image


def _resize(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=True)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach()
        .squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)


def build_inference_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    stage: str,
    device: torch.device,
) -> Any:
    """Build the rectifier once and load a trained checkpoint.

    Extracted so directory-mode inference can load the 39GB Qwen backbone a
    single time and reuse it across every image instead of per file.
    """

    model_config = dict(config["model"])
    if stage == "joint":
        model_config["raft_pretrained"] = False
    model = build_rectifier(
        model_config,
        dict(config.get("qwen", {})),
        stage=stage,
        device=device,
    )
    # Trusted local checkpoint written by this trainer (carries a config dict),
    # so weights_only must stay False on newer torch defaults.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch; missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
    return model.to(device).eval()


@torch.inference_mode()
def rectify(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    warped_path: str | Path,
    guide_path: str | Path | None,
    output_dir: str | Path,
    *,
    stage: str = "unified",
    output_size: tuple[int, int] | None = None,
    model: Any | None = None,
) -> dict[str, Path]:
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    work_size = tuple(int(v) for v in config["data"]["work_size"])
    if work_size[0] % 8 or work_size[1] % 8:
        raise ValueError(f"work_size must be divisible by 8, got {work_size}")

    warped_native, _ = _load_image(warped_path)
    native_source_size = tuple(int(v) for v in warped_native.shape[-2:])
    target_size = output_size or native_source_size
    warped_work = _resize(warped_native, work_size).to(device)

    if stage == "joint":
        if guide_path is None:
            raise ValueError("legacy joint inference needs --guide")
        guide_native, _ = _load_image(guide_path)
        guide_work = _resize(guide_native, work_size).to(device)
    else:
        guide_work = None

    model_config = dict(config["model"])
    if stage == "joint":
        model_config["raft_pretrained"] = False
    if model is None:
        model = build_inference_model(
            config, checkpoint_path, stage=stage, device=device
        )
    outputs = model(warped_work, guide_work, stage=stage)
    flow_work = outputs["final_flow"].float()
    prior_flow_work = outputs["prior_flow"].float()

    # Transform the absolute source-coordinate map. This remains correct when
    # the low-resolution canvas was anisotropically resized.
    flow_native = resize_backward_flow(
        flow_work,
        target_size,
        source_size_from=work_size,
        source_size_to=native_source_size,
    )
    rectified = backward_warp(
        warped_native.to(device),
        flow_native,
        padding_mode="border",
    )
    prior_flow_native = resize_backward_flow(
        prior_flow_work,
        target_size,
        source_size_from=work_size,
        source_size_to=native_source_size,
    )
    prior_rectified = backward_warp(
        warped_native.to(device),
        prior_flow_native,
        padding_mode="border",
    )
    valid = flow_valid_mask(flow_native, native_source_size)
    determinant = jacobian_determinant(flow_native)
    valid_bool = valid[:, 0]
    valid_cells = (
        valid_bool[:, :-1, :-1]
        & valid_bool[:, 1:, :-1]
        & valid_bool[:, :-1, 1:]
        & valid_bool[:, 1:, 1:]
    )
    if valid_cells.any():
        valid_determinant = determinant[valid_cells]
        fold_rate = float((valid_determinant <= 0).float().mean().cpu())
        jacobian_p01 = float(torch.quantile(valid_determinant.float(), 0.01).cpu())
    else:
        fold_rate = float("nan")
        jacobian_p01 = float("nan")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(warped_path).stem
    image_path = output_dir / f"{stem}_rectified.png"
    prior_image_path = output_dir / f"{stem}_prior_rectified.png"
    flow_path = output_dir / f"{stem}_backward_flow.npy"
    valid_path = output_dir / f"{stem}_valid.png"
    metadata_path = output_dir / f"{stem}_metadata.json"
    _save_image(rectified, image_path)
    _save_image(prior_rectified, prior_image_path)
    np.save(
        flow_path,
        flow_native.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32),
    )
    _save_image(valid.expand(-1, 3, -1, -1).float(), valid_path)

    residual_p50 = None
    residual_p95 = None
    if outputs.get("residuals"):
        residual_native = resize_backward_flow(
            outputs["residuals"][-1].float(),
            target_size,
            source_size_from=work_size,
            source_size_to=target_size,
        )
        residual_norm = torch.linalg.vector_norm(residual_native, dim=1)
        residual_p50 = float(torch.quantile(residual_norm, 0.50).cpu())
        residual_p95 = float(torch.quantile(residual_norm, 0.95).cpu())

    confidence_path = None
    confidence_mean = None
    if "feature_confidence" in outputs:
        confidence = F.interpolate(
            outputs["feature_confidence"].float(),
            size=target_size,
            mode="bilinear",
            align_corners=True,
        ).clamp(0.0, 1.0)
        confidence_path = output_dir / f"{stem}_feature_confidence.png"
        _save_image(confidence.expand(-1, 3, -1, -1), confidence_path)
        confidence_mean = float(confidence.mean().cpu())

    metadata = {
        "flow_convention": "backward displacement: output(x,y) samples source((x,y)+flow(x,y)); channels=(x,y)",
        "warped": str(Path(warped_path).resolve()),
        "guide": str(Path(guide_path).resolve()) if guide_path else None,
        "feature_backend": model_config.get("feature_backend") if stage == "unified" else None,
        "uses_decoded_qwen_rgb": stage == "joint",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "stage": stage,
        "work_size_hw": list(work_size),
        "source_size_hw": list(native_source_size),
        "output_size_hw": list(target_size),
        "valid_fraction": float(valid.float().mean().cpu()),
        "fold_rate": fold_rate,
        "jacobian_p01": jacobian_p01,
        "residual_p50_px": residual_p50,
        "residual_p95_px": residual_p95,
        "feature_confidence_mean": confidence_mean,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "image": image_path,
        "prior_image": prior_image_path,
        "flow": flow_path,
        "valid": valid_path,
        "metadata": metadata_path,
        **({"confidence": confidence_path} if confidence_path is not None else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warped", help="single warped image")
    parser.add_argument(
        "--input-dir", help="directory of warped images (batch mode; Qwen loads once)"
    )
    parser.add_argument("--guide")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stage", choices=("prior", "joint", "unified"), default="unified"
    )
    parser.add_argument("--output-height", type=int)
    parser.add_argument("--output-width", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if (args.output_height is None) != (args.output_width is None):
        parser.error("--output-height and --output-width must be supplied together")
    if bool(args.warped) == bool(args.input_dir):
        parser.error("supply exactly one of --warped or --input-dir")
    output_size = (
        (args.output_height, args.output_width) if args.output_height is not None else None
    )

    if args.warped:
        paths = rectify(
            config,
            args.checkpoint,
            args.warped,
            args.guide,
            args.output_dir,
            stage=args.stage,
            output_size=output_size,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    # Batch mode: build the model (and 39GB Qwen) once, reuse per image.
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    model = build_inference_model(
        config, args.checkpoint, stage=args.stage, device=device
    )
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = sorted(
        p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in exts
    )
    if not images:
        parser.error(f"no images found in {args.input_dir}")
    print(f"batch inference: {len(images)} images -> {args.output_dir}", flush=True)
    for index, image_path in enumerate(images, start=1):
        rectify(
            config,
            args.checkpoint,
            image_path,
            args.guide,
            args.output_dir,
            stage=args.stage,
            output_size=output_size,
            model=model,
        )
        print(f"  [{index}/{len(images)}] {image_path.name}", flush=True)


if __name__ == "__main__":
    main()
