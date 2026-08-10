#!/usr/bin/env python3
"""Evaluate axis alignment with OpenCV LSD as a no-reference proxy.

This script never compares an image with geometric ground truth.  Straight,
axis-aligned document lines are useful for regression triage, but these scores
cannot prove rectification quality, content fidelity, crop completeness, or
absence of local artifacts.  Always inspect images and use reference metrics
when ground truth is available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CANDIDATE_SUFFIXES = ("_rectified", "-rectified", ".rectified")
MASK_SUFFIXES = ("_evaluation_valid", "_eval_valid", "_valid", "-valid", ".valid")
PROXY_NOTICE = (
    "OpenCV LSD axis alignment is a no-reference structural proxy, not proof "
    "of overall rectification quality or content fidelity."
)


def _normalized_key(path: Path, suffixes: Sequence[str]) -> str:
    stem = path.stem
    folded = stem.casefold()
    for suffix in suffixes:
        if folded.endswith(suffix.casefold()) and len(stem) > len(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.casefold()


def _suffix_priority(path: Path, suffixes: Sequence[str]) -> int:
    folded = path.stem.casefold()
    for index, suffix in enumerate(suffixes):
        if folded.endswith(suffix.casefold()) and len(path.stem) > len(suffix):
            return index
    return len(suffixes)


def build_image_index(
    directory: str | Path,
    *,
    suffixes: Sequence[str] = (),
) -> dict[str, Path]:
    """Index non-recursive image files by a normalized basename."""

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    result: dict[str, Path] = {}
    priorities: dict[str, int] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        key = _normalized_key(path, suffixes)
        priority = _suffix_priority(path, suffixes)
        previous = result.get(key)
        if previous is not None:
            previous_priority = priorities[key]
            if priority < previous_priority:
                result[key] = path
                priorities[key] = priority
                continue
            if priority > previous_priority:
                continue
            raise ValueError(
                "ambiguous basename pairing in "
                f"{directory}: {previous.name!r} and {path.name!r} both map to {key!r}"
            )
        result[key] = path
        priorities[key] = priority
    if not result:
        raise ValueError(f"no supported images found in {directory}")
    return result


def metrics_from_segments(
    segments: np.ndarray,
    *,
    axis_threshold_deg: float = 5.0,
) -> dict[str, int | float | None]:
    """Compute exact axis-relative metrics for ``[N,4]`` LSD segments."""

    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    if not math.isfinite(axis_threshold_deg) or not 0.0 <= axis_threshold_deg <= 45.0:
        raise ValueError("axis_threshold_deg must be finite and in [0,45]")
    if segments.size == 0:
        return {
            "line_count": 0,
            "total_line_length_px": 0.0,
            "orientation_error_deg": None,
            "orientation_error_deg_length_weighted": None,
            "axis_fraction": None,
            "axis_fraction_length_weighted": None,
            "orientation_error_sum_deg": 0.0,
            "orientation_error_length_sum_deg_px": 0.0,
            "axis_line_count": 0,
            "axis_line_length_px": 0.0,
        }

    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    lengths = np.hypot(dx, dy)
    keep = np.isfinite(lengths) & (lengths > 0.0)
    segments = segments[keep]
    lengths = lengths[keep]
    if not len(segments):
        return metrics_from_segments(
            np.empty((0, 4), dtype=np.float64),
            axis_threshold_deg=axis_threshold_deg,
        )

    angles = np.degrees(np.arctan2(
        segments[:, 3] - segments[:, 1],
        segments[:, 2] - segments[:, 0],
    ))
    modulo = np.abs(angles) % 90.0
    errors = np.minimum(modulo, 90.0 - modulo)
    near_axis = errors < float(axis_threshold_deg)
    length_sum = float(lengths.sum())
    error_sum = float(errors.sum())
    weighted_error_sum = float(np.dot(errors, lengths))
    axis_count = int(near_axis.sum())
    axis_length = float(lengths[near_axis].sum())
    count = int(len(errors))
    return {
        "line_count": count,
        "total_line_length_px": length_sum,
        "orientation_error_deg": error_sum / count,
        "orientation_error_deg_length_weighted": weighted_error_sum / length_sum,
        "axis_fraction": axis_count / count,
        "axis_fraction_length_weighted": axis_length / length_sum,
        "orientation_error_sum_deg": error_sum,
        "orientation_error_length_sum_deg_px": weighted_error_sum,
        "axis_line_count": axis_count,
        "axis_line_length_px": axis_length,
    }


def _line_mask_coverage(segments: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    coverages: list[float] = []
    for x1, y1, x2, y2 in segments:
        length = float(math.hypot(float(x2 - x1), float(y2 - y1)))
        sample_count = max(2, min(64, int(math.ceil(length / 4.0)) + 1))
        fraction = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        xs = np.rint(x1 + fraction * (x2 - x1)).astype(np.int64)
        ys = np.rint(y1 + fraction * (y2 - y1)).astype(np.int64)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        samples = np.zeros(sample_count, dtype=bool)
        samples[inside] = mask[ys[inside], xs[inside]]
        coverages.append(float(samples.mean()))
    return np.asarray(coverages, dtype=np.float64)


def evaluate_image(
    image_path: str | Path,
    *,
    mask_path: str | Path | None = None,
    max_dimension: int = 1600,
    min_length_fraction: float = 0.03,
    axis_threshold_deg: float = 5.0,
    mask_min_coverage: float = 0.90,
) -> dict[str, int | float | None]:
    """Detect LSD segments and return axis-alignment proxy metrics."""

    if max_dimension < 2:
        raise ValueError("max_dimension must be at least 2")
    if not math.isfinite(min_length_fraction) or not 0.0 <= min_length_fraction <= 1.0:
        raise ValueError("min_length_fraction must be finite and in [0,1]")
    if not math.isfinite(mask_min_coverage) or not 0.0 <= mask_min_coverage <= 1.0:
        raise ValueError("mask_min_coverage must be finite and in [0,1]")

    image_path = Path(image_path)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"failed to read image: {image_path}")
    native_height, native_width = gray.shape

    valid_mask: np.ndarray | None = None
    if mask_path is not None:
        mask_path = Path(mask_path)
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise ValueError(f"failed to read evaluation-valid mask: {mask_path}")
        if raw_mask.shape != gray.shape:
            raw_mask = cv2.resize(
                raw_mask,
                (native_width, native_height),
                interpolation=cv2.INTER_NEAREST,
            )
        valid_mask = raw_mask > 127

    scale = min(1.0, float(max_dimension) / max(native_height, native_width))
    if scale < 1.0:
        # Keep the historical diagnostic's floor rounding so old typical-set
        # scores remain directly comparable.
        resized_width = max(2, int(native_width * scale))
        resized_height = max(2, int(native_height * scale))
        gray = cv2.resize(
            gray,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        if valid_mask is not None:
            valid_mask = cv2.resize(
                valid_mask.astype(np.uint8),
                (resized_width, resized_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)[0]
    if detected is None:
        segments = np.empty((0, 4), dtype=np.float64)
    else:
        segments = np.asarray(detected, dtype=np.float64).reshape(-1, 4)

    if len(segments) and valid_mask is not None:
        coverage = _line_mask_coverage(segments, valid_mask)
        segments = segments[coverage >= float(mask_min_coverage)]

    if len(segments):
        lengths = np.hypot(
            segments[:, 2] - segments[:, 0],
            segments[:, 3] - segments[:, 1],
        )
        positive = np.isfinite(lengths) & (lengths > 0.0)
        long_enough = positive & (
            lengths > float(min_length_fraction) * max(gray.shape)
        )
        # Preserve the original diagnostic's fallback for sparse pages.
        segments = segments[long_enough if long_enough.any() else positive]

    return metrics_from_segments(
        segments,
        axis_threshold_deg=axis_threshold_deg,
    )


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return None if not len(array) else float(array.mean())


def _median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return None if not len(array) else float(np.median(array))


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float | None]:
    metrics = [
        row["metrics"]
        for row in rows
        if row.get("status") == "ok" and row.get("metrics", {}).get("line_count", 0) > 0
    ]
    paired = sum(row.get("status") != "missing_image" for row in rows)
    missing_images = sum(row.get("status") == "missing_image" for row in rows)
    missing_masks = sum(row.get("status") == "missing_mask" for row in rows)
    no_lines = sum(row.get("status") == "no_lines" for row in rows)
    line_count = sum(int(metric["line_count"]) for metric in metrics)
    length_sum = sum(float(metric["total_line_length_px"]) for metric in metrics)
    error_sum = sum(float(metric["orientation_error_sum_deg"]) for metric in metrics)
    weighted_error_sum = sum(
        float(metric["orientation_error_length_sum_deg_px"]) for metric in metrics
    )
    axis_count = sum(int(metric["axis_line_count"]) for metric in metrics)
    axis_length = sum(float(metric["axis_line_length_px"]) for metric in metrics)
    return {
        "indexed_images": len(rows),
        "paired_images": int(paired),
        "evaluated_images": len(metrics),
        "missing_images": int(missing_images),
        "missing_masks": int(missing_masks),
        "no_line_images": int(no_lines),
        "total_line_count": int(line_count),
        "total_line_length_px": float(length_sum),
        "pooled_orientation_error_deg": error_sum / line_count if line_count else None,
        "pooled_orientation_error_deg_length_weighted": (
            weighted_error_sum / length_sum if length_sum else None
        ),
        "pooled_axis_fraction": axis_count / line_count if line_count else None,
        "pooled_axis_fraction_length_weighted": (
            axis_length / length_sum if length_sum else None
        ),
        "image_mean_orientation_error_deg": _mean(
            float(metric["orientation_error_deg"]) for metric in metrics
        ),
        "image_median_orientation_error_deg": _median(
            float(metric["orientation_error_deg"]) for metric in metrics
        ),
        "image_mean_orientation_error_deg_length_weighted": _mean(
            float(metric["orientation_error_deg_length_weighted"])
            for metric in metrics
        ),
        "image_median_orientation_error_deg_length_weighted": _median(
            float(metric["orientation_error_deg_length_weighted"])
            for metric in metrics
        ),
        "image_mean_axis_fraction": _mean(
            float(metric["axis_fraction"]) for metric in metrics
        ),
        "image_mean_axis_fraction_length_weighted": _mean(
            float(metric["axis_fraction_length_weighted"]) for metric in metrics
        ),
    }


def evaluate_dataset(
    images_dir: str | Path,
    candidates: Mapping[str, str | Path],
    *,
    valid_masks: Mapping[str, str | Path] | None = None,
    max_dimension: int = 1600,
    min_length_fraction: float = 0.03,
    axis_threshold_deg: float = 5.0,
    mask_min_coverage: float = 0.90,
) -> dict[str, Any]:
    """Pair and evaluate multiple candidate directories on one image index."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    valid_masks = dict(valid_masks or {})
    unknown_masks = set(valid_masks) - set(candidates)
    if unknown_masks:
        raise ValueError(
            f"valid masks reference unknown candidates: {sorted(unknown_masks)}"
        )
    image_index = build_image_index(images_dir)
    candidate_results: dict[str, Any] = {}
    for name, directory in candidates.items():
        candidate_index = build_image_index(directory, suffixes=CANDIDATE_SUFFIXES)
        mask_index = (
            build_image_index(valid_masks[name], suffixes=MASK_SUFFIXES)
            if name in valid_masks
            else None
        )
        rows: list[dict[str, Any]] = []
        for key, indexed_path in image_index.items():
            candidate_path = candidate_index.get(key)
            row: dict[str, Any] = {
                "basename": indexed_path.stem,
                "indexed_image": str(indexed_path.resolve()),
                "candidate_image": (
                    str(candidate_path.resolve()) if candidate_path is not None else None
                ),
                "evaluation_valid_mask": None,
            }
            if candidate_path is None:
                row["status"] = "missing_image"
                rows.append(row)
                continue
            mask_path = mask_index.get(key) if mask_index is not None else None
            if mask_index is not None and mask_path is None:
                row["status"] = "missing_mask"
                rows.append(row)
                continue
            if mask_path is not None:
                row["evaluation_valid_mask"] = str(mask_path.resolve())
            metrics = evaluate_image(
                candidate_path,
                mask_path=mask_path,
                max_dimension=max_dimension,
                min_length_fraction=min_length_fraction,
                axis_threshold_deg=axis_threshold_deg,
                mask_min_coverage=mask_min_coverage,
            )
            row["status"] = "ok" if metrics["line_count"] else "no_lines"
            row["metrics"] = metrics
            rows.append(row)
        extras = sorted(
            str(path.resolve())
            for key, path in candidate_index.items()
            if key not in image_index
        )
        candidate_results[name] = {
            "directory": str(Path(directory).resolve()),
            "valid_mask_directory": (
                str(Path(valid_masks[name]).resolve()) if name in valid_masks else None
            ),
            "summary": summarize_rows(rows),
            "per_image": rows,
            "unpaired_candidate_images": extras,
        }

    return {
        "schema_version": 1,
        "metric_family": "opencv_lsd_axis_alignment",
        "proxy_notice": PROXY_NOTICE,
        "images_directory": str(Path(images_dir).resolve()),
        "config": {
            "max_dimension": int(max_dimension),
            "min_length_fraction": float(min_length_fraction),
            "axis_threshold_deg": float(axis_threshold_deg),
            "mask_min_coverage": float(mask_min_coverage),
            "candidate_suffixes": list(CANDIDATE_SUFFIXES),
            "mask_suffixes": list(MASK_SUFFIXES),
        },
        "candidates": candidate_results,
    }


def _parse_named_directories(values: Sequence[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, directory = value.partition("=")
        name = name.strip()
        directory = directory.strip()
        if not separator or not name or not directory:
            raise ValueError(f"{option} expects NAME=DIR, got {value!r}")
        if name in result:
            raise ValueError(f"duplicate {option} name {name!r}")
        result[name] = Path(directory)
    return result


def _atomic_write_json(report: Mapping[str, Any], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        required=True,
        help="Directory defining the canonical image basenames to evaluate.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Candidate directory; repeat for multiple methods.",
    )
    parser.add_argument(
        "--valid-mask",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help=(
            "Optional candidate-specific evaluation-valid mask directory; "
            "repeat as needed. Files such as BASENAME_valid.png are paired."
        ),
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--max-dimension", type=int, default=1600)
    parser.add_argument("--min-length-fraction", type=float, default=0.03)
    parser.add_argument("--axis-threshold-deg", type=float, default=5.0)
    parser.add_argument("--mask-min-coverage", type=float, default=0.90)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        candidates = _parse_named_directories(args.candidate, "--candidate")
        valid_masks = _parse_named_directories(args.valid_mask, "--valid-mask")
        report = evaluate_dataset(
            args.images,
            candidates,
            valid_masks=valid_masks,
            max_dimension=args.max_dimension,
            min_length_fraction=args.min_length_fraction,
            axis_threshold_deg=args.axis_threshold_deg,
            mask_min_coverage=args.mask_min_coverage,
        )
        output = _atomic_write_json(report, args.output)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(PROXY_NOTICE)
    print(
        f"{'candidate':<20} {'images':>8} {'LSD err':>10} "
        f"{'length-wtd':>12} {'axis frac':>10}"
    )
    for name, candidate in report["candidates"].items():
        summary = candidate["summary"]
        error = summary["image_mean_orientation_error_deg"]
        weighted = summary["image_mean_orientation_error_deg_length_weighted"]
        axis_fraction = summary["image_mean_axis_fraction"]
        format_value = lambda value: "n/a" if value is None else f"{value:.4f}"
        print(
            f"{name:<20} {summary['evaluated_images']:>8d} "
            f"{format_value(error):>10} {format_value(weighted):>12} "
            f"{format_value(axis_fraction):>10}"
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
