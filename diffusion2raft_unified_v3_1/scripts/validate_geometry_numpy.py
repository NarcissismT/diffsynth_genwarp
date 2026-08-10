#!/usr/bin/env python3
"""Dependency-light validation of the coordinate contract used by the project."""

from __future__ import annotations

import numpy as np


def grid(height: int, width: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    return np.stack((x, y), axis=-1)


def bilinear_sample(field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    h, w, _ = field.shape
    x = np.clip(coordinates[..., 0], 0, w - 1)
    y = np.clip(coordinates[..., 1], 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    top = (1.0 - wx) * field[y0, x0] + wx * field[y0, x1]
    bottom = (1.0 - wx) * field[y1, x0] + wx * field[y1, x1]
    return (1.0 - wy) * top + wy * bottom


def resize_backward_flow(
    flow: np.ndarray,
    target_size: tuple[int, int],
    source_size_from: tuple[int, int],
    source_size_to: tuple[int, int],
) -> np.ndarray:
    h0, w0, _ = flow.shape
    h1, w1 = target_size
    old_map = flow + grid(h0, w0)
    new_grid = grid(h1, w1)
    old_coordinates = new_grid.copy()
    old_coordinates[..., 0] *= (w0 - 1) / max(w1 - 1, 1)
    old_coordinates[..., 1] *= (h0 - 1) / max(h1 - 1, 1)
    new_map = bilinear_sample(old_map, old_coordinates)
    new_map[..., 0] *= (source_size_to[1] - 1) / max(source_size_from[1] - 1, 1)
    new_map[..., 1] *= (source_size_to[0] - 1) / max(source_size_from[0] - 1, 1)
    return new_map - new_grid


def compose(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return residual + bilinear_sample(base, grid(*residual.shape[:2]) + residual)


def recover_residual(
    base: np.ndarray, composed: np.ndarray, iterations: int = 8
) -> np.ndarray:
    residual = composed - base
    coordinates = grid(*residual.shape[:2])
    for _ in range(iterations):
        residual = composed - bilinear_sample(base, coordinates + residual)
    return residual


def main() -> None:
    # Exact affine map under different target/source anisotropic scales.
    target0 = (17, 23)
    target1 = (41, 67)
    source0 = (31, 37)
    source1 = (83, 109)
    g0 = grid(*target0)
    absolute0 = np.empty_like(g0)
    absolute0[..., 0] = 1.15 * g0[..., 0] + 0.07 * g0[..., 1] + 2.0
    absolute0[..., 1] = -0.04 * g0[..., 0] + 1.10 * g0[..., 1] + 3.0
    flow0 = absolute0 - g0
    resized = resize_backward_flow(flow0, target1, source0, source1)

    g1 = grid(*target1)
    x0 = g1[..., 0] * (target0[1] - 1) / (target1[1] - 1)
    y0 = g1[..., 1] * (target0[0] - 1) / (target1[0] - 1)
    expected_map = np.empty_like(g1)
    expected_map[..., 0] = (1.15 * x0 + 0.07 * y0 + 2.0) * (source1[1] - 1) / (source0[1] - 1)
    expected_map[..., 1] = (-0.04 * x0 + 1.10 * y0 + 3.0) * (source1[0] - 1) / (source0[0] - 1)
    resize_error = np.max(np.abs((resized + g1) - expected_map))
    assert resize_error < 1e-10, resize_error

    # Composition is not equivalent to direct addition for a varying base map.
    h, w = 25, 29
    g = grid(h, w)
    base = np.empty_like(g)
    base[..., 0] = 0.10 * g[..., 0] + 0.03 * g[..., 1]
    base[..., 1] = -0.02 * g[..., 0] + 0.08 * g[..., 1]
    residual = np.zeros_like(g)
    residual[..., 0] = 1.25
    residual[..., 1] = 0.75
    composed = compose(base, residual)
    interior = (slice(0, -2), slice(0, -2))
    expected = residual + np.stack(
        (
            0.10 * (g[..., 0] + 1.25) + 0.03 * (g[..., 1] + 0.75),
            -0.02 * (g[..., 0] + 1.25) + 0.08 * (g[..., 1] + 0.75),
        ),
        axis=-1,
    )
    compose_error = np.max(np.abs(composed[interior] - expected[interior]))
    assert compose_error < 1e-10, compose_error
    assert np.max(np.abs(composed[interior] - (base + residual)[interior])) > 1e-2

    recovered = recover_residual(base, composed)
    inverse_error = np.max(np.abs(recovered[interior] - residual[interior]))
    assert inverse_error < 1e-8, inverse_error

    print(
        "geometry validation passed: "
        f"resize_max_error={resize_error:.3e}, compose_max_error={compose_error:.3e}, "
        f"residual_inverse_max_error={inverse_error:.3e}"
    )


if __name__ == "__main__":
    main()
