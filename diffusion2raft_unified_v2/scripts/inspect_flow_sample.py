#!/usr/bin/env python3
"""Visualize one manifest GT flow to catch direction/coordinate mistakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def grid(height: int, width: int) -> np.ndarray:
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack((x, y), axis=-1).astype(np.float64)


def sample(image: np.ndarray, pixel_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = image.shape
    x, y = pixel_map[..., 0], pixel_map[..., 1]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x, y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = (x - x0)[..., None], (y - y0)[..., None]
    top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
    return (1 - wy) * top + wy * bottom, valid


def load_array(path: Path, key: str) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        return payload[key] if key in payload else payload[payload.files[0]]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="tmp/flow_inspection")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    record = records[args.index]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else manifest.parent / path

    warped = np.asarray(Image.open(resolve(record["warped"])).convert("RGB"), dtype=np.float64) / 255.0
    target = np.asarray(Image.open(resolve(record["target"])).convert("RGB"), dtype=np.float64) / 255.0
    flow = load_array(resolve(record["flow"]), "flow").astype(np.float64)
    if flow.shape[0] == 2 and flow.shape[-1] != 2:
        flow = np.moveaxis(flow, 0, -1)
    if flow.shape[:2] != target.shape[:2]:
        raise ValueError(f"flow shape {flow.shape} does not match target {target.shape}")
    target_grid = grid(*target.shape[:2])
    flow_format = str(record.get("flow_format", "displacement")).lower()
    if flow_format in {"displacement", "backward_displacement", "flow"}:
        pixel_map = target_grid + flow
    elif flow_format in {"absolute_map", "backward_map", "map"}:
        pixel_map = flow
    elif flow_format in {"normalized_grid", "grid_sample"}:
        pixel_map = flow.copy()
        pixel_map[..., 0] = (pixel_map[..., 0] + 1) * (warped.shape[1] - 1) / 2
        pixel_map[..., 1] = (pixel_map[..., 1] + 1) * (warped.shape[0] - 1) / 2
    else:
        raise ValueError(f"unknown flow_format: {flow_format}")
    reconstruction, valid = sample(warped, pixel_map)
    if record.get("valid"):
        supplied = load_array(resolve(record["valid"]), "valid").astype(bool).squeeze()
        valid &= supplied
    difference = np.abs(reconstruction - target)
    mae = float(difference[valid].mean()) if valid.any() else float("nan")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(reconstruction * 255, 0, 255))).save(output / "gt_rectified.png")
    Image.fromarray(np.uint8(np.clip(difference * 4 * 255, 0, 255))).save(output / "difference_x4.png")
    Image.fromarray(np.uint8(valid * 255)).save(output / "valid.png")
    print(f"sample={record.get('id', args.index)} valid_fraction={valid.mean():.4f} rgb_mae={mae:.6f}")
    print(f"wrote inspection to {output}")


if __name__ == "__main__":
    main()

