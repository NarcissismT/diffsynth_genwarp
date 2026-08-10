#!/usr/bin/env python3
"""Freeze a renderer validation JSONL for the MMDiT correspondence probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_PATHS = ("warped_image", "rectified_image", "backward_map")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {path}")
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for record in records:
                handle.write(
                    json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-documents", type=int, default=328)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    records: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    sample_ids: set[str] = set()
    severities: dict[str, int] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{source}:{line_number}: record must be an object")
            sample_id = str(raw.get("sample_id", "")).strip()
            document_id = str(raw.get("document_id", "")).strip()
            if not sample_id or not document_id:
                raise ValueError(
                    f"{source}:{line_number}: sample_id/document_id are required"
                )
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id={sample_id!r}")
            sample_ids.add(sample_id)
            document_ids.add(document_id)
            if raw.get("label_provenance") != "analytic_gt":
                raise ValueError(f"{sample_id}: expected label_provenance=analytic_gt")
            if raw.get("map_direction") != "output_to_warped_source":
                raise ValueError(f"{sample_id}: invalid map_direction")
            if raw.get("coordinate_convention") != "absolute_source_pixel_xy":
                raise ValueError(f"{sample_id}: invalid coordinate_convention")
            for name in REQUIRED_PATHS:
                asset = Path(str(raw.get(name, "")))
                if not asset.is_absolute() or not asset.is_file():
                    raise FileNotFoundError(f"{sample_id}: missing absolute {name}: {asset}")
            raw["split"] = "validation"
            tags = dict(raw.get("subset_tags") or {})
            tags["split"] = "validation"
            raw["subset_tags"] = tags
            severity = str(raw.get("warp_severity", "unknown"))
            severities[severity] = severities.get(severity, 0) + 1
            records.append(raw)

    if len(document_ids) < args.minimum_documents:
        raise ValueError(
            f"formal full profile needs {args.minimum_documents} unique documents, "
            f"but renderer validation provides {len(document_ids)}"
        )
    atomic_jsonl(output, records)
    report = {
        "schema": "mmdit_correspondence.source_manifest.v1",
        "source": str(source),
        "source_sha256": file_sha256(source),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "records": len(records),
        "unique_documents": len(document_ids),
        "severities": dict(sorted(severities.items())),
        "split": "validation",
        "minimum_documents": args.minimum_documents,
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
