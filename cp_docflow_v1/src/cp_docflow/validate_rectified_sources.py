"""Fail-closed source-resolution/aspect audit for analytic full-page rendering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def validate_rectified_sources(
    input_csv: str | Path,
    *,
    image_column: str = "image",
    category_column: str = "category",
    seed: int = 1337,
    max_documents: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    target_size: tuple[int, int] = (1024, 768),
    minimum_source_size: tuple[int, int] = (1024, 768),
    aspect_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Validate the exact deterministic document subset used by one render shard.

    Sizes are ``(height, width)``.  Requiring both adequate native resolution
    and a target-compatible aspect ratio prevents the analytic renderer from
    silently stretching 512x512 crops into a nominal 1024x768 Stage-5 corpus.
    """

    source_csv = Path(input_csv).resolve()
    if not source_csv.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {source_csv}")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if min(*target_size, *minimum_source_size) < 1:
        raise ValueError("target and minimum source dimensions must be positive")
    if not 0.0 <= aspect_tolerance <= 0.25:
        raise ValueError("aspect_tolerance must be in [0,0.25]")

    documents: dict[str, tuple[Path, str]] = {}
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if image_column not in (reader.fieldnames or ()):
            raise ValueError(f"input CSV lacks image column {image_column!r}")
        for row in reader:
            rectified = _resolve(source_csv.parent, str(row[image_column]))
            category = str(row.get(category_column) or "uncategorized").strip()
            document_id = f"{category}:{rectified.stem}"
            previous = documents.setdefault(document_id, (rectified, category))
            if previous[0] != rectified:
                raise ValueError(f"document_id collision for {document_id!r}")

    ordered = sorted(
        documents.items(),
        key=lambda item: hashlib.sha256(
            f"selection:{seed}:{item[0]}".encode("utf-8")
        ).digest(),
    )
    if max_documents is not None:
        ordered = ordered[:max_documents]
    selected_document_count = len(ordered)
    selected = ordered[shard_index::num_shards]
    if not selected:
        raise ValueError(
            f"source-validation shard {shard_index}/{num_shards} contains no documents"
        )

    target_height, target_width = target_size
    minimum_height, minimum_width = minimum_source_size
    target_aspect = target_width / target_height
    invalid: list[dict[str, Any]] = []
    source_pixels = 0
    minimum_seen = [None, None]
    maximum_seen = [0, 0]
    for document_id, (path, _) in selected:
        if not path.is_file():
            invalid.append({"document_id": document_id, "reason": "missing", "path": str(path)})
            continue
        with Image.open(path) as image:
            width, height = image.size
        source_pixels += height * width
        minimum_seen[0] = height if minimum_seen[0] is None else min(minimum_seen[0], height)
        minimum_seen[1] = width if minimum_seen[1] is None else min(minimum_seen[1], width)
        maximum_seen[0] = max(maximum_seen[0], height)
        maximum_seen[1] = max(maximum_seen[1], width)
        relative_aspect_error = abs((width / height) - target_aspect) / target_aspect
        reasons: list[str] = []
        if height < minimum_height or width < minimum_width:
            reasons.append("source_resolution_below_minimum")
        if relative_aspect_error > aspect_tolerance:
            reasons.append("source_aspect_incompatible_with_target")
        if reasons and len(invalid) < 20:
            invalid.append(
                {
                    "document_id": document_id,
                    "path": str(path),
                    "source_size": [height, width],
                    "relative_aspect_error": relative_aspect_error,
                    "reason": ",".join(reasons),
                }
            )
    if invalid:
        raise ValueError(
            "full-page source audit failed; refusing resolution/aspect stretching: "
            + json.dumps(invalid, ensure_ascii=False)
        )
    return {
        "schema": "docgrid_flow.rectified_source_validation.v1",
        "source_csv": str(source_csv),
        "seed": int(seed),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "selected_document_count_before_sharding": selected_document_count,
        "validated_documents": len(selected),
        "target_size": list(target_size),
        "minimum_source_size": list(minimum_source_size),
        "aspect_tolerance": float(aspect_tolerance),
        "minimum_seen_size": minimum_seen,
        "maximum_seen_size": maximum_seen,
        "mean_source_pixels": source_pixels / len(selected),
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--category-column", default="category")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--target-height", type=int, default=1024)
    parser.add_argument("--target-width", type=int, default=768)
    parser.add_argument("--minimum-source-height", type=int, default=1024)
    parser.add_argument("--minimum-source-width", type=int, default=768)
    parser.add_argument("--aspect-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    result = validate_rectified_sources(
        args.input_csv,
        image_column=args.image_column,
        category_column=args.category_column,
        seed=args.seed,
        max_documents=args.max_documents,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        target_size=(args.target_height, args.target_width),
        minimum_source_size=(args.minimum_source_height, args.minimum_source_width),
        aspect_tolerance=args.aspect_tolerance,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
