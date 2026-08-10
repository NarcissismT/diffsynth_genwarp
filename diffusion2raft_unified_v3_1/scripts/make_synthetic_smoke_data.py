#!/usr/bin/env python3
"""Create a tiny exact-flow dataset for a CUDA end-to-end smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def grid(height: int, width: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    return np.stack((x, y), axis=-1)


def homography_from_points(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    rows = []
    values = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.extend(((x, y, 1, 0, 0, 0, -u * x, -u * y), (0, 0, 0, x, y, 1, -v * x, -v * y)))
        values.extend((u, v))
    solution = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(values))
    return np.append(solution, 1.0).reshape(3, 3)


def transform(coordinates: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (coordinates, np.ones((*coordinates.shape[:-1], 1), dtype=np.float64)), axis=-1
    )
    mapped = homogeneous @ homography.T
    return mapped[..., :2] / mapped[..., 2:3]


def sample(image: np.ndarray, coordinates: np.ndarray, fill: float = 1.0) -> np.ndarray:
    h, w, _ = image.shape
    x, y = coordinates[..., 0], coordinates[..., 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x_clip, y_clip = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x_clip).astype(int), np.floor(y_clip).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (x_clip - x0)[..., None], (y_clip - y0)[..., None]
    top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
    result = (1 - wy) * top + wy * bottom
    result[~valid] = fill
    return result


def document(height: int, width: int, index: int) -> np.ndarray:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    margin = width // 10
    draw.rectangle((margin, height // 12, width - margin, height - height // 12), outline=(25, 25, 25), width=2)
    draw.text((margin + 12, height // 10), f"Synthetic document {index:03d}", fill="black", font=font)
    for row in range(12):
        y = height // 6 + row * max(10, height // 18)
        line_width = width - 2 * margin - 24 - (row % 4) * width // 18
        draw.line((margin + 12, y, margin + 12 + line_width, y), fill=(35, 35, 35), width=2)
    draw.rectangle((margin + 20, height * 2 // 3, width // 2, height * 5 // 6), outline=(50, 50, 50), width=2)
    return np.asarray(image, dtype=np.float64) / 255.0


def make_sample(root: Path, split: str, index: int, height: int, width: int, rng: np.random.Generator) -> dict[str, str]:
    target = document(height, width, index)
    corners = np.array(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)), dtype=np.float64)
    jitter = np.stack(
        (rng.uniform(-0.05, 0.05, 4) * width, rng.uniform(-0.05, 0.05, 4) * height), axis=-1
    )
    source_corners = corners + jitter
    h_target_to_source = homography_from_points(corners, source_corners)
    h_source_to_target = np.linalg.inv(h_target_to_source)
    target_grid = grid(height, width)
    source_grid = grid(height, width)
    source = sample(target, transform(source_grid, h_source_to_target))
    source_map = transform(target_grid, h_target_to_source)
    flow = (source_map - target_grid).astype(np.float32)
    valid = (
        (source_map[..., 0] >= 0)
        & (source_map[..., 0] <= width - 1)
        & (source_map[..., 1] >= 0)
        & (source_map[..., 1] <= height - 1)
    )

    guide = target.copy()
    y0, y1 = height // 3, height // 3 + max(8, height // 20)
    x0, x1 = width // 4, width * 3 // 4
    guide[y0:y1, x0:x1] = np.roll(guide[y0:y1, x0:x1], shift=7, axis=1)

    stem = f"{split}_{index:04d}"
    for folder in ("images", "targets", "guides", "flows", "masks"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(source * 255, 0, 255))).save(root / "images" / f"{stem}.png")
    Image.fromarray(np.uint8(np.clip(target * 255, 0, 255))).save(root / "targets" / f"{stem}.png")
    Image.fromarray(np.uint8(np.clip(guide * 255, 0, 255))).save(root / "guides" / f"{stem}.png")
    np.save(root / "flows" / f"{stem}.npy", flow)
    np.save(root / "masks" / f"{stem}.npy", valid)
    return {
        "id": stem,
        "warped": f"images/{stem}.png",
        "target": f"targets/{stem}.png",
        "guide": f"guides/{stem}.png",
        "flow": f"flows/{stem}.npy",
        "valid": f"masks/{stem}.npy",
        "flow_format": "displacement",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/smoke_data")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    root = Path(args.output)
    rng = np.random.default_rng(args.seed)
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        records = [make_sample(root, split, i, args.height, args.width, rng) for i in range(count)]
        with (root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    print(f"synthetic smoke dataset written to {root}")


if __name__ == "__main__":
    main()

