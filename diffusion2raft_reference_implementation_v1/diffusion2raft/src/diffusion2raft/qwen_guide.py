"""Offline Qwen-Image-Edit guide generation.

The 20B edit model is intentionally kept out of the RAFT training loop. Guides
are deterministic, cached artifacts; gradients never pass through Qwen/VAE.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .config import load_config


class QwenGuideGenerator:
    def __init__(self, config: dict[str, Any], device: str = "cuda") -> None:
        try:
            from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Qwen guide generation requires a recent diffusers build. Install the qwen extra; "
                "for Qwen-Image-Edit-2511 use the latest diffusers Git revision if necessary."
            ) from exc

        self.config = config
        self.device = device
        model_id = str(config.get("model_id", "Qwen/Qwen-Image-Edit-2511"))
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        pipeline_cls = (
            QwenImageEditPlusPipeline
            if any(tag in model_id.lower() for tag in ("2509", "2511", "plus"))
            else QwenImageEditPipeline
        )
        self.pipeline = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype)
        if bool(config.get("cpu_offload", False)) and device.startswith("cuda"):
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(device)
        self.is_plus = pipeline_cls is QwenImageEditPlusPipeline

    @torch.inference_mode()
    def __call__(self, image: Image.Image, size: tuple[int, int], seed: int) -> Image.Image:
        height, width = size
        if height % 32 or width % 32:
            raise ValueError(f"Qwen output size must be divisible by 32, got {size}")
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        kwargs = {
            "image": [image] if self.is_plus else image,
            "prompt": str(self.config["prompt"]),
            "height": height,
            "width": width,
            "generator": generator,
            "num_inference_steps": int(self.config.get("num_inference_steps", 40)),
            "true_cfg_scale": float(self.config.get("true_cfg_scale", 4.0)),
            "guidance_scale": float(self.config.get("guidance_scale", 1.0)),
            "num_images_per_prompt": 1,
        }
        result = self.pipeline(**kwargs).images[0]
        return result.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def generate_manifest_guides(
    config: dict[str, Any],
    manifest_path: str | Path,
    output_dir: str | Path,
    output_manifest: str | Path,
    *,
    device: str,
    overwrite: bool = False,
) -> Path:
    manifest_path = Path(manifest_path)
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    size = tuple(int(v) for v in config["data"]["work_size"])
    generator = QwenGuideGenerator(config["qwen"], device=device)
    base_seed = int(config["qwen"].get("seed", 0))

    updated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        source = Path(record["warped"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        identifier = str(record.get("id", index)).replace(os.sep, "_")
        destination = output_dir / f"{identifier}.png"
        if overwrite or not destination.exists():
            with Image.open(source) as image:
                guide = generator(image, size, base_seed + index)
            guide.save(destination)
        copied = dict(record)
        copied["guide"] = os.path.relpath(destination.resolve(), output_manifest.parent.resolve())
        updated.append(copied)

    with output_manifest.open("w", encoding="utf-8") as handle:
        for record in updated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = generate_manifest_guides(
        load_config(args.config),
        args.manifest,
        args.output_dir,
        args.output_manifest,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"updated manifest: {result}")


if __name__ == "__main__":
    main()

