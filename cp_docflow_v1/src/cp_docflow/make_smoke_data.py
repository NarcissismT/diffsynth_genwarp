"""Generate a tiny analytic backward-map dataset for CPU smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .geometry import canonical_backward_map, warp_with_backward_map


def _source_image(height: int, width: int, index: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin = max(2, min(height, width) // 16)
    draw.rectangle(
        (margin, margin, width - margin - 1, height - margin - 1),
        outline=(15, 15, 15),
        width=max(1, margin // 3),
    )
    spacing = max(4, height // 8)
    for y in range(2 * margin, height - margin, spacing):
        draw.line((2 * margin, y, width - 2 * margin, y), fill=(30, 30, 30), width=1)
    draw.text((2 * margin, margin + 1), f"DOC {index:03d}", fill=(0, 0, 0))
    return image


def _write_split(
    root: Path,
    split: str,
    count: int,
    size: tuple[int, int],
    offset: int,
) -> Path:
    height, width = size
    image_dir = root / "images"
    map_dir = root / "maps"
    mask_dir = root / "masks"
    for directory in (image_dir, map_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = root / f"{split}.jsonl"
    records: list[dict[str, object]] = []
    for local_index in range(count):
        index = offset + local_index
        sample_id = f"{split}-{index:04d}"
        source_pil = _source_image(height, width, index)
        source_path = image_dir / f"{sample_id}-warped.png"
        target_path = image_dir / f"{sample_id}-rectified.png"
        map_path = map_dir / f"{sample_id}.npy"
        mask_path = mask_dir / f"{sample_id}.npy"
        source_pil.save(source_path)
        source_array = np.asarray(source_pil, dtype=np.float32) / 255.0
        source = torch.from_numpy(source_array).permute(2, 0, 1).unsqueeze(0)
        backward_map = canonical_backward_map(1, size)
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        amplitude = 0.5 + 0.25 * (index % 3)
        backward_map = backward_map.clone()
        backward_map[:, 0] += amplitude * torch.sin(2.0 * torch.pi * y / max(height, 1))
        backward_map[:, 1] += amplitude * torch.sin(2.0 * torch.pi * x / max(width, 1))
        target, valid = warp_with_backward_map(source, backward_map, return_valid=True)
        target_array = (
            target.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0 + 0.5
        ).astype(np.uint8)
        Image.fromarray(target_array).save(target_path)
        np.save(map_path, backward_map.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32))
        np.save(mask_path, valid.squeeze(0).squeeze(0).numpy())
        records.append(
            {
                "sample_id": sample_id,
                "document_id": f"document-{index:04d}",
                "warp_severity": "smoke",
                "label_provenance": "synthetic_analytic",
                "label_source": "cp_docflow.make_smoke_data.v1",
                "map_direction": "output_to_warped_source",
                "coordinate_convention": "absolute_source_pixel_xy",
                "warped_image": str(source_path.relative_to(root)),
                "rectified_image": str(target_path.relative_to(root)),
                "backward_map": str(map_path.relative_to(root)),
                "valid_mask": str(mask_path.relative_to(root)),
                "input_size": [height, width],
                "output_size": [height, width],
            }
        )
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    if min(args.train_count, args.val_count, args.height, args.width) < 1:
        parser.error("counts and sizes must be positive")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_manifest = _write_split(
        root,
        "train",
        args.train_count,
        (args.height, args.width),
        0,
    )
    val_manifest = _write_split(
        root,
        "val",
        args.val_count,
        (args.height, args.width),
        args.train_count,
    )
    print(json.dumps({"train": str(train_manifest), "val": str(val_manifest)}))


if __name__ == "__main__":
    main()
