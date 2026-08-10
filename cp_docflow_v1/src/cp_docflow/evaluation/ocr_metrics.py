"""Score auditable model/oracle OCR transcripts for Gate-5 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..checkpoint import file_sha256

_REQUIRED_FIELDS = ("sample_id", "reference_text", "model_text", "oracle_text")
_IMAGE_HASH_FIELDS = ("model_image_sha256", "oracle_image_sha256")
_IMAGE_MANIFEST_FIELDS = (
    "sample_id",
    "model_image",
    "model_image_sha256",
    "oracle_image",
    "oracle_image_sha256",
)


def normalize_ocr_text(value: str) -> str:
    """Use one reproducible Unicode/whitespace policy for CER and WER."""

    normalized = unicodedata.normalize("NFKC", str(value)).replace("\r", "\n")
    return " ".join(normalized.split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance using O(min(N,M)) memory."""

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def _load_transcripts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"OCR transcript file does not exist: {path}")
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"OCR JSONL line {line_number} is not an object")
                rows.append(value)
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("OCR transcript file contains no samples")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        missing = [field for field in _REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"OCR row {index} lacks fields: {missing}")
        sample_id = str(row["sample_id"]).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"OCR sample_id is empty or duplicated: {sample_id!r}")
        seen.add(sample_id)
        parsed = {
            "sample_id": sample_id,
            "reference_text": str(row["reference_text"]),
            "model_text": str(row["model_text"]),
            "oracle_text": str(row["oracle_text"]),
        }
        for field in _IMAGE_HASH_FIELDS:
            if field in row and row[field] is not None:
                parsed[field] = str(row[field]).strip()
        result.append(parsed)
    return result


def _load_ocr_image_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"OCR image manifest does not exist: {path}")
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"OCR image manifest line {line_number} is not an object")
            missing = [field for field in _IMAGE_MANIFEST_FIELDS if field not in value]
            if missing:
                raise ValueError(
                    f"OCR image manifest line {line_number} lacks fields: {missing}"
                )
            sample_id = str(value["sample_id"]).strip()
            if not sample_id or sample_id in result:
                raise ValueError(
                    f"OCR image manifest sample_id is empty or duplicated: {sample_id!r}"
                )
            parsed = {field: str(value[field]).strip() for field in _IMAGE_MANIFEST_FIELDS}
            for path_field, hash_field in (
                ("model_image", "model_image_sha256"),
                ("oracle_image", "oracle_image_sha256"),
            ):
                image_path = Path(parsed[path_field])
                if not image_path.is_absolute():
                    image_path = (path.parent / image_path).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"OCR image manifest references missing image: {image_path}"
                    )
                actual = file_sha256(image_path)
                if parsed[hash_field] != actual:
                    raise ValueError(
                        f"OCR image SHA-256 mismatch for sample {sample_id!r}: {path_field}"
                    )
                parsed[path_field] = str(image_path)
            result[sample_id] = parsed
    if not result:
        raise ValueError("OCR image manifest contains no samples")
    return result


def _geometry_sample_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"geometry per-sample CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "sample_id" not in rows[0]:
        raise ValueError("geometry per-sample CSV lacks sample_id rows")
    values = [str(row["sample_id"]).strip() for row in rows]
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError("geometry per-sample CSV has empty or duplicate sample_id")
    return set(values)


def _write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OCR evidence: {path}")
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
            json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OCR evidence: {path}")
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def score_ocr_transcripts(
    transcript_path: str | Path,
    output_dir: str | Path,
    *,
    ocr_engine: str,
    ocr_engine_version: str,
    geometry_evaluation: str | Path | None = None,
    geometry_per_sample: str | Path | None = None,
    ocr_image_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate corpus CER/WER and freeze the source/evaluation identities."""

    source = Path(transcript_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to use non-empty OCR output: {destination}")
    engine = str(ocr_engine).strip()
    version = str(ocr_engine_version).strip()
    if not engine or not version:
        raise ValueError("ocr_engine and ocr_engine_version must be non-empty")
    if version.lower() in {"unknown", "unset", "none", "n/a"}:
        raise ValueError("ocr_engine_version must identify the exact fixed engine")
    geometry_values = (
        geometry_evaluation,
        geometry_per_sample,
        ocr_image_manifest,
    )
    if any(value is not None for value in geometry_values) and not all(
        value is not None for value in geometry_values
    ):
        raise ValueError(
            "geometry_evaluation, geometry_per_sample, and ocr_image_manifest "
            "must be supplied together"
        )
    transcripts = _load_transcripts(source)
    transcript_ids = {row["sample_id"] for row in transcripts}

    geometry_identity: dict[str, Any] | None = None
    if all(value is not None for value in geometry_values):
        geometry_report_path = Path(geometry_evaluation).resolve()
        geometry_rows_path = Path(geometry_per_sample).resolve()
        image_manifest_path = Path(ocr_image_manifest).resolve()
        with geometry_report_path.open("r", encoding="utf-8") as handle:
            geometry_report = json.load(handle)
        if not isinstance(geometry_report, dict):
            raise ValueError("geometry evaluation must be a JSON object")
        expected = _geometry_sample_ids(geometry_rows_path)
        image_rows = _load_ocr_image_manifest(image_manifest_path)
        image_ids = set(image_rows)
        if transcript_ids != expected or image_ids != expected:
            missing = sorted(expected - transcript_ids)[:10]
            extra = sorted(transcript_ids - expected)[:10]
            raise ValueError(
                "OCR, exported-image, and geometry sample sets differ; "
                f"transcript_missing={missing}, transcript_extra={extra}, "
                f"image_missing={sorted(expected - image_ids)[:10]}, "
                f"image_extra={sorted(image_ids - expected)[:10]}"
            )
        export_identity = geometry_report.get("ocr_image_export")
        image_manifest_sha256 = file_sha256(image_manifest_path)
        if not isinstance(export_identity, dict) or export_identity.get(
            "manifest_sha256"
        ) != image_manifest_sha256:
            raise ValueError(
                "OCR image manifest is not bound to the geometry evaluation"
            )
        for row in transcripts:
            exported = image_rows[row["sample_id"]]
            for field in _IMAGE_HASH_FIELDS:
                if row.get(field) != exported[field]:
                    raise ValueError(
                        f"OCR transcript {field} differs from exported image for "
                        f"sample {row['sample_id']!r}"
                    )
        geometry_identity = {
            "evaluation": str(geometry_report_path),
            "evaluation_sha256": file_sha256(geometry_report_path),
            "per_sample": str(geometry_rows_path),
            "per_sample_sha256": file_sha256(geometry_rows_path),
            "checkpoint_sha256": geometry_report.get("checkpoint_sha256"),
            "manifest_sha256": geometry_report.get("manifest_sha256"),
            "dataset_payload_sha256": geometry_report.get(
                "evaluation_dataset_payload_sha256"
            ),
            "ocr_image_manifest": str(image_manifest_path),
            "ocr_image_manifest_sha256": image_manifest_sha256,
        }

    per_sample: list[dict[str, Any]] = []
    totals = {
        "reference_characters": 0,
        "reference_words": 0,
        "model_character_edits": 0,
        "oracle_character_edits": 0,
        "model_word_edits": 0,
        "oracle_word_edits": 0,
    }
    for row in transcripts:
        reference = normalize_ocr_text(row["reference_text"])
        model = normalize_ocr_text(row["model_text"])
        oracle = normalize_ocr_text(row["oracle_text"])
        reference_words = reference.split()
        model_words = model.split()
        oracle_words = oracle.split()
        model_character_edits = edit_distance(reference, model)
        oracle_character_edits = edit_distance(reference, oracle)
        model_word_edits = edit_distance(reference_words, model_words)
        oracle_word_edits = edit_distance(reference_words, oracle_words)
        reference_characters = len(reference)
        word_count = len(reference_words)
        totals["reference_characters"] += reference_characters
        totals["reference_words"] += word_count
        totals["model_character_edits"] += model_character_edits
        totals["oracle_character_edits"] += oracle_character_edits
        totals["model_word_edits"] += model_word_edits
        totals["oracle_word_edits"] += oracle_word_edits
        per_sample.append(
            {
                "sample_id": row["sample_id"],
                "reference_characters": reference_characters,
                "reference_words": word_count,
                "model_character_edits": model_character_edits,
                "oracle_character_edits": oracle_character_edits,
                "model_word_edits": model_word_edits,
                "oracle_word_edits": oracle_word_edits,
                "model_cer": (
                    model_character_edits / reference_characters
                    if reference_characters
                    else None
                ),
                "oracle_cer": (
                    oracle_character_edits / reference_characters
                    if reference_characters
                    else None
                ),
                "model_wer": model_word_edits / word_count if word_count else None,
                "oracle_wer": oracle_word_edits / word_count if word_count else None,
                "model_image_sha256": row.get("model_image_sha256"),
                "oracle_image_sha256": row.get("oracle_image_sha256"),
            }
        )
    if totals["reference_characters"] == 0 or totals["reference_words"] == 0:
        raise ValueError("OCR corpus must contain non-empty reference text and words")

    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "docgrid_flow.ocr_evaluation.v2",
        "normalization": "Unicode NFKC; collapse all whitespace to one ASCII space",
        "aggregation": "corpus edit distance divided by corpus reference length",
        "ocr_engine": engine,
        "ocr_engine_version": version,
        "transcripts": str(source),
        "transcripts_sha256": file_sha256(source),
        "samples": len(per_sample),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(transcript_ids)).encode("utf-8")
        ).hexdigest(),
        "ocr_cer": totals["model_character_edits"]
        / totals["reference_characters"],
        "oracle_ocr_cer": totals["oracle_character_edits"]
        / totals["reference_characters"],
        "ocr_wer": totals["model_word_edits"] / totals["reference_words"],
        "oracle_ocr_wer": totals["oracle_word_edits"] / totals["reference_words"],
        "totals": totals,
        "geometry_identity": geometry_identity,
    }
    report_path = destination / "metrics.json"
    _write_json_atomic(report_path, report)
    _write_csv_atomic(destination / "per_sample.csv", per_sample)
    fragment = {
        "ocr_cer": report["ocr_cer"],
        "oracle_ocr_cer": report["oracle_ocr_cer"],
        "ocr_wer": report["ocr_wer"],
        "oracle_ocr_wer": report["oracle_ocr_wer"],
        "ocr_evaluation": str(report_path),
        "ocr_evaluation_sha256": file_sha256(report_path),
    }
    _write_json_atomic(destination / "gate_evidence.fragment.json", fragment)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ocr-engine", required=True)
    parser.add_argument("--ocr-engine-version", required=True)
    parser.add_argument("--geometry-evaluation")
    parser.add_argument("--geometry-per-sample")
    parser.add_argument("--ocr-image-manifest")
    args = parser.parse_args()
    report = score_ocr_transcripts(
        args.transcripts,
        args.output_dir,
        ocr_engine=args.ocr_engine,
        ocr_engine_version=args.ocr_engine_version,
        geometry_evaluation=args.geometry_evaluation,
        geometry_per_sample=args.geometry_per_sample,
        ocr_image_manifest=args.ocr_image_manifest,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
