"""Run native inference twice and record exact fixed-seed repeatability."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import file_sha256
from .infer_full import infer_full


def verify_repeatability(
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    image = Path(image_path).resolve()
    with tempfile.TemporaryDirectory(prefix="docgrid-repeatability-") as directory:
        root = Path(directory)
        first = infer_full(checkpoint, image, root / "first", device_name=device_name)
        second = infer_full(checkpoint, image, root / "second", device_name=device_name)
        comparisons = {
            "backward_map_exact": np.array_equal(
                np.load(first["backward_map"]), np.load(second["backward_map"])
            ),
            "coarse_map_exact": np.array_equal(
                np.load(first["coarse_backward_map"]),
                np.load(second["coarse_backward_map"]),
            ),
            "confidence_exact": np.array_equal(
                np.load(first["confidence"]), np.load(second["confidence"])
            ),
            "rectified_png_exact": file_sha256(first["rectified_image"])
            == file_sha256(second["rectified_image"]),
        }
    result: dict[str, Any] = {
        "schema": "docgrid_flow.repeatability_evidence.v2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "input": str(image),
        "input_sha256": file_sha256(image),
        "device": device_name,
        "comparisons": comparisons,
        "fixed_seed_repeatable": all(comparisons.values()),
    }
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite repeatability evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = verify_repeatability(
        args.checkpoint, args.image, args.output, device_name=args.device
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

