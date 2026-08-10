"""Merge a complete set of immutable analytic-renderer shards."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .checkpoint import file_sha256
from .data import COORDINATE_CONVENTION, MAP_DIRECTION


_SHARD_DIRECTORY = re.compile(r"^shard-(\d+)-of-(\d+)$")


def _require_within(path: Path, parent: Path, *, role: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as error:
        raise ValueError(f"{role} escapes its analytic shard: {path}") from error


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            values.append(value)
    return values


def _write_atomic(path: Path, value: Any, *, jsonl: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite merged analytic data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            if jsonl:
                for item in value:
                    handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
            else:
                json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def merge_analytic_shards(
    shard_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(shard_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to use non-empty merge output: {destination}")
    report_paths = sorted(root.glob("shard-*-of-*/renderer_report.json"))
    if not report_paths:
        raise FileNotFoundError(f"no analytic shard reports under {root}")
    reports = [(path, _load_json(path)) for path in report_paths]
    expected_count = int(reports[0][1].get("num_shards", 0))
    if expected_count < 2:
        raise ValueError("shard merge requires reports with num_shards >= 2")
    indices: list[int] = []
    for report_path, report in reports:
        match = _SHARD_DIRECTORY.fullmatch(report_path.parent.name)
        if match is None:
            raise ValueError(f"invalid analytic shard directory: {report_path.parent}")
        directory_index, directory_count = (int(value) for value in match.groups())
        report_index = int(report.get("shard_index", -1))
        report_count = int(report.get("num_shards", -1))
        if (directory_index, directory_count) != (report_index, report_count):
            raise ValueError(
                f"analytic shard directory/report identity mismatch: {report_path}"
            )
        indices.append(report_index)
    if sorted(indices) != list(range(expected_count)):
        raise ValueError(
            f"analytic shards are incomplete/duplicated: got {sorted(indices)}, "
            f"expected 0..{expected_count - 1}"
        )
    invariant_keys = (
        "schema",
        "gate_eligible_provenance",
        "domain_limit",
        "source_csv",
        "source_csv_sha256",
        "generator_sha256",
        "seed",
        "num_shards",
        "variants_per_document",
        "split_ratios",
        "output_size",
        "selected_document_count_before_sharding",
        "profiles",
    )
    reference = reports[0][1]
    if reference.get("schema") != "docgrid_flow.analytic_renderer.v3":
        raise ValueError(f"unsupported analytic renderer schema: {reference.get('schema')!r}")
    if reference.get("gate_eligible_provenance") != "analytic_gt":
        raise ValueError("analytic shards are not declared Gate-eligible analytic_gt")
    for path, report in reports[1:]:
        differing = [key for key in invariant_keys if report.get(key) != reference.get(key)]
        if differing:
            raise ValueError(f"analytic shard contract differs in {path}: {differing}")

    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    sample_ids: set[str] = set()
    document_splits: dict[str, str] = {}
    document_samples: dict[str, int] = {}
    oracle_weighted_sum = 0.0
    oracle_sample_count = 0
    for report_path, report in sorted(reports, key=lambda item: int(item[1]["shard_index"])):
        manifests = report.get("manifests")
        if not isinstance(manifests, dict):
            raise ValueError(f"analytic shard report lacks manifests: {report_path}")
        declared_samples = report.get("samples")
        if not isinstance(declared_samples, dict):
            raise ValueError(f"analytic shard report lacks sample counts: {report_path}")
        for split in records:
            manifest = Path(str(manifests.get(split, ""))).resolve()
            _require_within(
                manifest,
                report_path.parent.resolve(),
                role=f"shard {split} manifest",
            )
            if not manifest.is_file():
                raise FileNotFoundError(f"missing shard {split} manifest: {manifest}")
            declared_sha = report.get("manifest_sha256", {}).get(split)
            if declared_sha != file_sha256(manifest):
                raise ValueError(f"shard manifest SHA mismatch: {manifest}")
            split_items = _load_jsonl(manifest)
            if len(split_items) != int(declared_samples.get(split, -1)):
                raise ValueError(f"shard sample count mismatch: {manifest}")
            for item in split_items:
                sample_id = str(item.get("sample_id", ""))
                document_id = str(item.get("document_id", ""))
                if not sample_id or sample_id in sample_ids:
                    raise ValueError(f"empty/duplicate analytic sample_id={sample_id!r}")
                if item.get("label_provenance") != "analytic_gt":
                    raise ValueError(f"non-analytic label in shard sample {sample_id!r}")
                if item.get("map_direction") != MAP_DIRECTION:
                    raise ValueError(f"map direction mismatch in shard sample {sample_id!r}")
                if item.get("coordinate_convention") != COORDINATE_CONVENTION:
                    raise ValueError(
                        f"coordinate convention mismatch in shard sample {sample_id!r}"
                    )
                previous = document_splits.setdefault(document_id, split)
                if not document_id or previous != split:
                    raise ValueError(
                        f"document {document_id!r} crosses analytic splits"
                    )
                for field in (
                    "warped_image",
                    "rectified_image",
                    "backward_map",
                    "valid_mask",
                    "horizontal_structure",
                    "vertical_structure",
                    "boundary_structure",
                ):
                    asset = Path(str(item.get(field, ""))).resolve()
                    _require_within(
                        asset,
                        report_path.parent.resolve(),
                        role=f"sample {sample_id!r} field {field!r}",
                    )
                    if not asset.is_file():
                        raise FileNotFoundError(
                            f"missing shard asset for sample {sample_id!r}: {asset}"
                        )
                sample_ids.add(sample_id)
                document_samples[document_id] = document_samples.get(document_id, 0) + 1
                records[split].append(item)
        shard_sample_count = sum(int(declared_samples[split]) for split in records)
        oracle_weighted_sum += float(report["oracle_rgb_l1_mean"]) * shard_sample_count
        oracle_sample_count += shard_sample_count
    empty = [split for split, values in records.items() if not values]
    if empty:
        raise ValueError(f"merged analytic dataset has empty splits: {empty}")
    expected_documents = int(reference["selected_document_count_before_sharding"])
    if len(document_splits) != expected_documents:
        raise ValueError(
            f"merged document count {len(document_splits)} != expected {expected_documents}"
        )
    expected_variants = int(reference["variants_per_document"])
    malformed_documents = {
        document_id: count
        for document_id, count in document_samples.items()
        if count != expected_variants
    }
    if malformed_documents:
        preview = list(sorted(malformed_documents.items()))[:5]
        raise ValueError(
            "analytic documents do not have exactly "
            f"{expected_variants} variants; examples={preview}"
        )
    if len(sample_ids) != expected_documents * expected_variants:
        raise ValueError("merged analytic sample total violates document/variant contract")

    manifests: dict[str, str] = {}
    for split, values in records.items():
        values.sort(key=lambda item: str(item["sample_id"]))
        path = destination / "manifests" / f"{split}.jsonl"
        _write_atomic(path, values, jsonl=True)
        manifests[split] = str(path)
    result = {
        "schema": "docgrid_flow.analytic_shard_merge.v1",
        "gate_eligible_provenance": "analytic_gt",
        "shard_root": str(root),
        "num_shards": expected_count,
        "source_csv": reference["source_csv"],
        "source_csv_sha256": reference["source_csv_sha256"],
        "generator_sha256": reference["generator_sha256"],
        "seed": reference["seed"],
        "variants_per_document": reference["variants_per_document"],
        "split_ratios": reference["split_ratios"],
        "output_size": reference["output_size"],
        "documents": {
            split: sum(value == split for value in document_splits.values())
            for split in records
        },
        "samples": {split: len(values) for split, values in records.items()},
        "oracle_rgb_l1_mean": oracle_weighted_sum / oracle_sample_count,
        "manifests": manifests,
        "manifest_sha256": {
            split: file_sha256(path) for split, path in manifests.items()
        },
        "shard_reports": [
            {"path": str(path), "sha256": file_sha256(path)} for path, _ in reports
        ],
    }
    _write_atomic(destination / "merge_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = merge_analytic_shards(args.shard_root, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
