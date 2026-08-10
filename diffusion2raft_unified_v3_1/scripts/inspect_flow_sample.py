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


def parse_size(record: dict[str, object], key: str) -> tuple[int, int] | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be [height, width], got {value!r}")
    size = (int(value[0]), int(value[1]))
    if min(size) <= 1:
        raise ValueError(f"invalid {key}: {size}")
    return size


def resize_absolute_map(pixel_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    out_h, out_w = size
    in_h, in_w = pixel_map.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return pixel_map.copy()
    y = np.linspace(0.0, in_h - 1.0, out_h)
    x = np.linspace(0.0, in_w - 1.0, out_w)
    y0, x0 = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, in_h - 1), np.minimum(x0 + 1, in_w - 1)
    wy, wx = (y - y0)[:, None, None], (x - x0)[None, :, None]
    top = (1.0 - wx) * pixel_map[y0[:, None], x0[None, :]] + wx * pixel_map[
        y0[:, None], x1[None, :]
    ]
    bottom = (1.0 - wx) * pixel_map[y1[:, None], x0[None, :]] + wx * pixel_map[
        y1[:, None], x1[None, :]
    ]
    return (1.0 - wy) * top + wy * bottom


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    out_h, out_w = size
    in_h, in_w = mask.shape
    y = np.rint(np.linspace(0.0, in_h - 1.0, out_h)).astype(np.int64)
    x = np.rint(np.linspace(0.0, in_w - 1.0, out_w)).astype(np.int64)
    return mask[y[:, None], x[None, :]]


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
    parser.add_argument("--max-mae", type=float)
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
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"flow must be [H,W,2] or [2,H,W], got {flow.shape}")
    flow_grid_size = tuple(int(value) for value in flow.shape[:2])
    declared_target = parse_size(record, "flow_target_size")
    if declared_target is not None and declared_target != flow_grid_size:
        raise ValueError(
            f"flow_target_size={declared_target} does not match flow grid {flow_grid_size}"
        )
    flow_source_size = parse_size(record, "flow_source_size")
    if flow_source_size is None:
        if flow_grid_size != target.shape[:2]:
            raise ValueError(
                f"ambiguous flow grid {flow_grid_size} for target {target.shape[:2]}; "
                "add flow_source_size to the manifest"
            )
        flow_source_size = tuple(int(value) for value in warped.shape[:2])
    flow_grid = grid(*flow_grid_size)
    flow_format = str(record.get("flow_format", "displacement")).lower()
    if flow_format in {"displacement", "backward_displacement", "flow"}:
        pixel_map = flow_grid + flow
    elif flow_format in {"absolute_map", "backward_map", "map"}:
        pixel_map = flow
    elif flow_format in {"normalized_grid", "grid_sample"}:
        pixel_map = flow.copy()
        pixel_map[..., 0] = (pixel_map[..., 0] + 1) * (flow_source_size[1] - 1) / 2
        pixel_map[..., 1] = (pixel_map[..., 1] + 1) * (flow_source_size[0] - 1) / 2
    else:
        raise ValueError(f"unknown flow_format: {flow_format}")
    pixel_map = resize_absolute_map(pixel_map, tuple(target.shape[:2]))
    pixel_map[..., 0] *= (warped.shape[1] - 1) / max(flow_source_size[1] - 1, 1)
    pixel_map[..., 1] *= (warped.shape[0] - 1) / max(flow_source_size[0] - 1, 1)
    reconstruction, valid = sample(warped, pixel_map)
    if record.get("valid"):
        supplied = load_array(resolve(record["valid"]), "valid").astype(bool).squeeze()
        if supplied.ndim != 2:
            raise ValueError(f"valid mask must reduce to [H,W], got {supplied.shape}")
        valid &= resize_mask(supplied, tuple(target.shape[:2]))
    difference = np.abs(reconstruction - target)
    mae = float(difference[valid].mean()) if valid.any() else float("nan")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(reconstruction * 255, 0, 255))).save(output / "gt_rectified.png")
    Image.fromarray(np.uint8(np.clip(difference * 4 * 255, 0, 255))).save(output / "difference_x4.png")
    Image.fromarray(np.uint8(valid * 255)).save(output / "valid.png")
    Image.fromarray(np.uint8(np.clip(target * 255, 0, 255))).save(output / "target.png")
    print(f"sample={record.get('id', args.index)} valid_fraction={valid.mean():.4f} rgb_mae={mae:.6f}")
    print(
        f"flow_grid={flow_grid_size} flow_source_canvas={flow_source_size} "
        f"warped_image={warped.shape[:2]} target_image={target.shape[:2]}"
    )
    print(f"wrote inspection to {output}")
    if args.max_mae is not None and (not np.isfinite(mae) or mae > args.max_mae):
        raise SystemExit(
            f"GT-flow inspection failed: rgb_mae={mae:.6f} > max_mae={args.max_mae:.6f}"
        )


if __name__ == "__main__":
    main()
