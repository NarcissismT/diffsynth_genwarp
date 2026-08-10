#!/usr/bin/env python3
"""Convert an upwarp metadata CSV into the JSONL manifest this project consumes.

CSV column mapping (upwarp_img_1in10_white_0511):
    edit_image         -> warped   (distorted input, the source we sample pixels from)
    image              -> target   (rectified ground-truth page)
    flow_gt_path       -> flow     (backward displacement, [2,Hf,Wf], (x,y), pixels)
    corrected_vae_path -> guide    (existing VAE rectification, used as the guide)
    category           -> category (kept for per-category evaluation)

The GT flow here is authored on a 1024x1024 canvas while the images are 512x512,
so ``flow_source_size`` is written explicitly; the data loader rescales the flow
coordinate map to ``work_size`` correctly. Paths in these CSVs are absolute and
are emitted verbatim, so the manifest can live anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# The GT flow canvas. Both the target grid and the source coordinates of the
# flow are in this scale for this dataset (verified by warping).
FLOW_SOURCE_SIZE = [1024, 1024]

COLUMNS = {
    "warped": "edit_image",
    "target": "image",
    "flow": "flow_gt_path",
    "guide": "corrected_vae_path",
}


def build(
    csv_path: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    categories: set[str] | None = None,
    with_guide: bool = True,
    flow_source_size: list[int] | None = None,
    require_exists: bool = False,
) -> int:
    flow_source_size = flow_source_size or FLOW_SOURCE_SIZE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_missing = 0
    with csv_path.open("r", encoding="utf-8") as handle, out_path.open(
        "w", encoding="utf-8"
    ) as sink:
        reader = csv.DictReader(handle)
        for column in COLUMNS.values():
            if column not in (reader.fieldnames or []):
                raise SystemExit(f"CSV missing required column {column!r}: {csv_path}")
        for row in reader:
            if categories and row.get("category") not in categories:
                continue
            paths = {key: row[column] for key, column in COLUMNS.items()}
            if require_exists and not all(Path(p).exists() for p in paths.values()):
                skipped_missing += 1
                continue
            identifier = Path(row[COLUMNS["warped"]]).stem
            record: dict[str, object] = {
                "id": identifier,
                "warped": paths["warped"],
                "target": paths["target"],
                "flow": paths["flow"],
                "flow_format": "displacement",
                "flow_source_size": flow_source_size,
                "category": row.get("category", ""),
            }
            if with_guide:
                record["guide"] = paths["guide"]
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    if skipped_missing:
        print(f"  skipped {skipped_missing} rows with missing files", file=sys.stderr)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="source metadata CSV")
    parser.add_argument("--output", required=True, help="destination .jsonl")
    parser.add_argument("--limit", type=int, help="cap number of records (for smoke runs)")
    parser.add_argument(
        "--categories",
        nargs="*",
        help="keep only these categories (e.g. Doc3d_crop UVDoc_crop)",
    )
    parser.add_argument(
        "--no-guide",
        action="store_true",
        help="omit the guide column (prior-stage-only manifest)",
    )
    parser.add_argument(
        "--flow-source-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        help=f"flow source canvas; default {FLOW_SOURCE_SIZE}",
    )
    parser.add_argument(
        "--require-exists",
        action="store_true",
        help="skip rows whose files are missing (slower; verifies every path)",
    )
    args = parser.parse_args()
    count = build(
        Path(args.csv),
        Path(args.output),
        limit=args.limit,
        categories=set(args.categories) if args.categories else None,
        with_guide=not args.no_guide,
        flow_source_size=args.flow_source_size,
        require_exists=args.require_exists,
    )
    print(f"wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
