"""Fail-closed evidence binding for the manual ``typical`` all-40 review.

The structural and line-proxy reports used by the v3.3 finalizer cannot prove
that a page is front-facing, readable, uncropped, or free from local waves.
This module provides the deliberately separate human-review contract:

* inventory the exact 40 source and reference images;
* inventory the six reviewable artifacts for both ``v33_anchor`` and
  ``v33_best``;
* bind every file's resolved path, stat metadata, and SHA-256 into one
  canonical evidence digest;
* create a per-image review template; and
* accept the review only when all 40 rows and all criteria explicitly pass
  against evidence that is still present and byte-for-byte unchanged.

The functions return ordinary JSON-compatible dictionaries.  They do not
write reports or mutate input directories, so a caller can decide where and
how to persist the evidence and review documents.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat as stat_module
from pathlib import Path
from typing import Any, Mapping, Sequence


TYPICAL_IMAGE_COUNT = 40
CANDIDATE_NAME = "v33_best"

EVIDENCE_SCHEMA = "diffusion2raft.typical_review_evidence"
EVIDENCE_SCHEMA_VERSION = 1
REVIEW_SCHEMA = "diffusion2raft.typical_manual_review"
REVIEW_SCHEMA_VERSION = 2
SUMMARY_SCHEMA = "diffusion2raft.typical_review_summary"
SUMMARY_SCHEMA_VERSION = 2

REVIEW_CRITERIA = (
    "full_resolution_reviewed",
    "rectangular_front_facing",
    "line_straightness",
    "content_preserved",
    "crop_and_margins_complete",
    "no_local_waves_or_folds",
    "inpainting_seams_acceptable",
    "at_least_target_first",
    "at_least_target_second",
)

# Order is part of the schema and therefore also part of the evidence digest.
INVENTORY_NAMES = (
    "source",
    "target_first",
    "target_second",
    "v33_anchor_final",
    "v33_anchor_raw",
    "v33_anchor_prior",
    "v33_anchor_valid",
    "v33_anchor_inpaint",
    "v33_anchor_evaluation_valid",
    "v33_best_final",
    "v33_best_raw",
    "v33_best_prior",
    "v33_best_valid",
    "v33_best_inpaint",
    "v33_best_evaluation_valid",
)

_OUTPUT_KINDS = (
    "final",
    "raw",
    "prior",
    "valid",
    "inpaint",
    "evaluation_valid",
)

# More specific suffixes must precede suffixes that they contain.
_OUTPUT_SUFFIXES = (
    ("raw", "_rectified_raw.png"),
    ("prior", "_prior_rectified.png"),
    ("inpaint", "_inpaint_mask.png"),
    ("evaluation_valid", "_evaluation_valid.png"),
    ("final", "_rectified.png"),
    ("valid", "_valid.png"),
)

_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "candidate",
        "expected_count",
        "roots",
        "stems",
        "inventories",
        "evidence_sha256",
    }
)
_ROOT_FIELDS = frozenset(
    {"source", "target_first", "target_second", "v33_anchor", "v33_best"}
)
_RECORD_FIELDS = frozenset({"stem", "path", "stat", "sha256"})
_STAT_FIELDS = frozenset({"size_bytes", "mtime_ns"})
_REVIEW_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "candidate",
        "expected_count",
        "evidence_sha256",
        "stems",
        "reviews",
    }
)
_REVIEW_ROW_FIELDS = frozenset({"stem", "notes", *REVIEW_CRITERIA})


class TypicalReviewError(RuntimeError):
    """The evidence or manual-review contract was violated."""


def _field_names(value: Mapping[Any, Any], *, label: str) -> set[str]:
    if not all(isinstance(key, str) for key in value):
        raise TypicalReviewError(f"{label} contains a non-string field name")
    return set(value)


def _require_exact_fields(
    value: Mapping[Any, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = _field_names(value, label=label)
    if actual != expected:
        raise TypicalReviewError(
            f"{label} fields do not match the schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_exact_int(value: Any, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:
        raise TypicalReviewError(
            f"{label} must be integer {expected}, got {value!r}"
        )


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypicalReviewError(f"value is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def canonical_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 of an evidence mapping without its digest."""

    if not isinstance(evidence, Mapping):
        raise TypicalReviewError("evidence must be a mapping")
    unsigned = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _resolved_directory(directory: str | Path, *, label: str) -> Path:
    try:
        resolved = Path(directory).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise TypicalReviewError(f"cannot resolve {label} directory {directory}: {error}") from error
    if not resolved.is_dir():
        raise TypicalReviewError(f"{label} is not a directory: {resolved}")
    return resolved


def _directory_entries(directory: Path, *, label: str) -> list[Path]:
    try:
        return sorted(directory.iterdir(), key=lambda path: (path.name.casefold(), path.name))
    except OSError as error:
        raise TypicalReviewError(f"cannot list {label} directory {directory}: {error}") from error


def _reference_index(directory: str | Path, *, label: str) -> dict[str, Path]:
    root = _resolved_directory(directory, label=label)
    entries = _directory_entries(root, label=label)
    incompatible_images = [
        path.name
        for path in entries
        if path.suffix.casefold() in _IMAGE_SUFFIXES and path.suffix != ".jpg"
    ]
    if incompatible_images:
        raise TypicalReviewError(
            f"{label} must use lowercase .jpg files only; "
            f"incompatible={incompatible_images[:8]}"
        )
    paths = [path for path in entries if path.suffix == ".jpg"]
    index: dict[str, Path] = {}
    casefolded: dict[str, str] = {}
    for path in paths:
        stem = path.stem
        key = stem.casefold()
        if key in casefolded:
            raise TypicalReviewError(
                f"{label} has a case-insensitive duplicate stem: "
                f"{casefolded[key]!r}, {stem!r}"
            )
        casefolded[key] = stem
        index[stem] = path
    if len(paths) != TYPICAL_IMAGE_COUNT:
        raise TypicalReviewError(
            f"{label} must contain exactly {TYPICAL_IMAGE_COUNT} lowercase .jpg files; "
            f"found={len(paths)}"
        )
    return index


def _classify_output_name(name: str) -> tuple[str, str] | None:
    folded_name = name.casefold()
    for kind, suffix in _OUTPUT_SUFFIXES:
        if folded_name.endswith(suffix):
            if not name.endswith(suffix):
                raise TypicalReviewError(
                    f"v3.3 output suffix must use exact lowercase spelling: {name!r}"
                )
            stem = name[: -len(suffix)]
            if not stem:
                raise TypicalReviewError(f"v3.3 output has an empty stem: {name!r}")
            return kind, stem
    return None


def _output_indexes(directory: str | Path, *, label: str) -> dict[str, dict[str, Path]]:
    root = _resolved_directory(directory, label=label)
    indexes: dict[str, dict[str, Path]] = {kind: {} for kind in _OUTPUT_KINDS}
    folded: dict[str, dict[str, str]] = {kind: {} for kind in _OUTPUT_KINDS}
    for path in _directory_entries(root, label=label):
        classified = _classify_output_name(path.name)
        if classified is None:
            # Flow arrays, metadata, confidence images, and inference_report.json
            # belong to the inference contract but are not visual-review inputs.
            continue
        kind, stem = classified
        key = stem.casefold()
        if key in folded[kind]:
            raise TypicalReviewError(
                f"{label}/{kind} has a case-insensitive duplicate stem: "
                f"{folded[kind][key]!r}, {stem!r}"
            )
        folded[kind][key] = stem
        indexes[kind][stem] = path
    for kind, index in indexes.items():
        if len(index) != TYPICAL_IMAGE_COUNT:
            raise TypicalReviewError(
                f"{label}/{kind} must contain exactly {TYPICAL_IMAGE_COUNT} artifacts; "
                f"found={len(index)}"
            )
    return indexes


def _stable_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _artifact_record(path: Path, *, stem: str) -> dict[str, Any]:
    """Hash a regular file while detecting replacement or mutation during the read."""

    try:
        before = path.lstat()
        if stat_module.S_ISLNK(before.st_mode) or not stat_module.S_ISREG(before.st_mode):
            raise TypicalReviewError(f"review artifact is not a regular non-symlink file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            finished = os.fstat(handle.fileno())
        after = path.lstat()
    except TypicalReviewError:
        raise
    except OSError as error:
        raise TypicalReviewError(f"cannot read review artifact {path}: {error}") from error

    identities = {
        _stable_stat_identity(before),
        _stable_stat_identity(opened),
        _stable_stat_identity(finished),
        _stable_stat_identity(after),
    }
    if len(identities) != 1:
        raise TypicalReviewError(f"review artifact changed while hashing: {path}")
    if before.st_size <= 0:
        raise TypicalReviewError(f"review artifact is empty: {path}")
    return {
        "stem": stem,
        "path": str(path.resolve(strict=True)),
        "stat": {
            "size_bytes": int(before.st_size),
            "mtime_ns": int(before.st_mtime_ns),
        },
        "sha256": digest.hexdigest(),
    }


def _assert_exact_stems(
    index: Mapping[str, Path], stems: Sequence[str], *, label: str
) -> None:
    actual = set(index)
    expected = set(stems)
    if actual != expected:
        raise TypicalReviewError(
            f"{label} stems do not exactly match source; "
            f"missing={sorted(expected - actual)[:8]}, "
            f"extra={sorted(actual - expected)[:8]}"
        )


def build_typical_evidence(
    source_dir: str | Path,
    target_first_dir: str | Path,
    target_second_dir: str | Path,
    anchor_dir: str | Path,
    best_dir: str | Path,
) -> dict[str, Any]:
    """Inventory and hash the complete all-40 manual-review evidence set."""

    roots = {
        "source": str(_resolved_directory(source_dir, label="source")),
        "target_first": str(
            _resolved_directory(target_first_dir, label="target_first")
        ),
        "target_second": str(
            _resolved_directory(target_second_dir, label="target_second")
        ),
        "v33_anchor": str(
            _resolved_directory(anchor_dir, label="v33_anchor")
        ),
        "v33_best": str(_resolved_directory(best_dir, label="v33_best")),
    }
    source = _reference_index(roots["source"], label="source")
    target_first = _reference_index(roots["target_first"], label="target_first")
    target_second = _reference_index(roots["target_second"], label="target_second")
    anchor = _output_indexes(roots["v33_anchor"], label="v33_anchor")
    best = _output_indexes(roots["v33_best"], label="v33_best")

    stems = sorted(source, key=lambda value: (value.casefold(), value))
    indexes: dict[str, Mapping[str, Path]] = {
        "source": source,
        "target_first": target_first,
        "target_second": target_second,
    }
    indexes.update({f"v33_anchor_{kind}": anchor[kind] for kind in _OUTPUT_KINDS})
    indexes.update({f"v33_best_{kind}": best[kind] for kind in _OUTPUT_KINDS})
    if tuple(indexes) != INVENTORY_NAMES:
        raise AssertionError("internal inventory order drifted from the evidence schema")

    for name, index in indexes.items():
        _assert_exact_stems(index, stems, label=name)

    inventories = {
        name: [_artifact_record(index[stem], stem=stem) for stem in stems]
        for name, index in indexes.items()
    }
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "candidate": CANDIDATE_NAME,
        "expected_count": TYPICAL_IMAGE_COUNT,
        "roots": roots,
        "stems": stems,
        "inventories": inventories,
    }
    evidence["evidence_sha256"] = canonical_evidence_sha256(evidence)
    # This also detects accidental path reuse between logically distinct inputs.
    _validate_evidence_structure(evidence)
    return evidence


def _validate_stems(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != TYPICAL_IMAGE_COUNT:
        raise TypicalReviewError(
            f"{label} must be a list of exactly {TYPICAL_IMAGE_COUNT} stems"
        )
    if not all(isinstance(stem, str) and stem for stem in value):
        raise TypicalReviewError(f"{label} contains an invalid stem")
    folded = [stem.casefold() for stem in value]
    if len(set(folded)) != TYPICAL_IMAGE_COUNT:
        raise TypicalReviewError(f"{label} contains a duplicate stem")
    expected_order = sorted(value, key=lambda stem: (stem.casefold(), stem))
    if value != expected_order:
        raise TypicalReviewError(f"{label} is not in canonical stem order")
    return list(value)


def _validate_evidence_structure(evidence: Mapping[str, Any]) -> list[str]:
    if not isinstance(evidence, Mapping):
        raise TypicalReviewError("evidence must be a mapping")
    _require_exact_fields(evidence, _EVIDENCE_FIELDS, label="evidence")
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise TypicalReviewError(f"evidence schema must be {EVIDENCE_SCHEMA!r}")
    _require_exact_int(
        evidence.get("schema_version"),
        EVIDENCE_SCHEMA_VERSION,
        label="evidence.schema_version",
    )
    if evidence.get("candidate") != CANDIDATE_NAME:
        raise TypicalReviewError(f"evidence candidate must be {CANDIDATE_NAME!r}")
    _require_exact_int(
        evidence.get("expected_count"),
        TYPICAL_IMAGE_COUNT,
        label="evidence.expected_count",
    )
    roots = evidence.get("roots")
    if not isinstance(roots, Mapping):
        raise TypicalReviewError("evidence.roots must be a mapping")
    _require_exact_fields(roots, _ROOT_FIELDS, label="evidence.roots")
    root_paths: list[str] = []
    for name in _ROOT_FIELDS:
        value = roots.get(name)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise TypicalReviewError(f"evidence root is not absolute: {name}")
        root_paths.append(value)
    if len(set(root_paths)) != len(root_paths):
        raise TypicalReviewError("evidence roots must be distinct directories")
    stems = _validate_stems(evidence.get("stems"), label="evidence.stems")

    inventories = evidence.get("inventories")
    if not isinstance(inventories, Mapping):
        raise TypicalReviewError("evidence.inventories must be a mapping")
    if _field_names(inventories, label="evidence.inventories") != set(INVENTORY_NAMES):
        raise TypicalReviewError(
            "evidence inventory names do not exactly match the all-40 schema"
        )

    seen_paths: set[str] = set()
    for name in INVENTORY_NAMES:
        records = inventories[name]
        if not isinstance(records, list) or len(records) != TYPICAL_IMAGE_COUNT:
            raise TypicalReviewError(
                f"evidence inventory {name!r} must have {TYPICAL_IMAGE_COUNT} records"
            )
        for expected_stem, record in zip(stems, records, strict=True):
            if not isinstance(record, Mapping):
                raise TypicalReviewError(f"evidence record {name}/{expected_stem} is not a mapping")
            _require_exact_fields(
                record, _RECORD_FIELDS, label=f"evidence record {name}/{expected_stem}"
            )
            if record.get("stem") != expected_stem:
                raise TypicalReviewError(
                    f"evidence record stem mismatch in {name}: "
                    f"expected={expected_stem!r}, actual={record.get('stem')!r}"
                )
            path = record.get("path")
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise TypicalReviewError(f"evidence path is not absolute: {name}/{expected_stem}")
            if path in seen_paths:
                raise TypicalReviewError(f"evidence path is reused by multiple inventories: {path}")
            seen_paths.add(path)
            stat_value = record.get("stat")
            if not isinstance(stat_value, Mapping):
                raise TypicalReviewError(f"evidence stat is missing: {name}/{expected_stem}")
            _require_exact_fields(
                stat_value, _STAT_FIELDS, label=f"evidence stat {name}/{expected_stem}"
            )
            size = stat_value.get("size_bytes")
            mtime = stat_value.get("mtime_ns")
            if type(size) is not int or size <= 0:
                raise TypicalReviewError(f"invalid evidence size: {name}/{expected_stem}")
            if type(mtime) is not int or mtime < 0:
                raise TypicalReviewError(f"invalid evidence mtime: {name}/{expected_stem}")
            digest = record.get("sha256")
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise TypicalReviewError(f"invalid evidence SHA-256: {name}/{expected_stem}")

    digest = evidence.get("evidence_sha256")
    expected_digest = canonical_evidence_sha256(evidence)
    if not isinstance(digest, str) or digest != expected_digest:
        raise TypicalReviewError(
            f"evidence digest mismatch; expected={expected_digest}, actual={digest!r}"
        )
    return stems


def validate_typical_evidence(
    evidence: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Validate evidence structure/digest and, by default, every current file."""

    stems = _validate_evidence_structure(evidence)
    if verify_files:
        roots = evidence["roots"]
        current = build_typical_evidence(
            roots["source"],
            roots["target_first"],
            roots["target_second"],
            roots["v33_anchor"],
            roots["v33_best"],
        )
        if current != dict(evidence):
            raise TypicalReviewError(
                "review evidence file changed after inventory or exact directory "
                "inventory changed"
            )
    return {
        "status": "valid",
        "candidate": CANDIDATE_NAME,
        "expected_count": TYPICAL_IMAGE_COUNT,
        "inventory_count": len(INVENTORY_NAMES),
        "file_count": len(INVENTORY_NAMES) * TYPICAL_IMAGE_COUNT,
        "evidence_sha256": evidence["evidence_sha256"],
    }


def make_typical_review_template(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Create the editable all-40 review document with every criterion pending."""

    stems = _validate_evidence_structure(evidence)
    return {
        "schema": REVIEW_SCHEMA,
        "schema_version": REVIEW_SCHEMA_VERSION,
        "candidate": CANDIDATE_NAME,
        "expected_count": TYPICAL_IMAGE_COUNT,
        "evidence_sha256": evidence["evidence_sha256"],
        "stems": stems,
        "reviews": [
            {
                "stem": stem,
                **{criterion: "pending" for criterion in REVIEW_CRITERIA},
                "notes": "",
            }
            for stem in stems
        ],
    }


def validate_completed_typical_review(
    review: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless the unchanged ``v33_best`` evidence passes all checks."""

    # Re-hashing here is intentional: it prevents review approval from being
    # retained after any source, reference, anchor, best image, or mask changes.
    evidence_status = validate_typical_evidence(evidence, verify_files=True)
    evidence_stems = list(evidence["stems"])

    if not isinstance(review, Mapping):
        raise TypicalReviewError("review must be a mapping")
    _require_exact_fields(review, _REVIEW_FIELDS, label="review")
    if review.get("schema") != REVIEW_SCHEMA:
        raise TypicalReviewError(f"review schema must be {REVIEW_SCHEMA!r}")
    _require_exact_int(
        review.get("schema_version"),
        REVIEW_SCHEMA_VERSION,
        label="review.schema_version",
    )
    if review.get("candidate") != CANDIDATE_NAME:
        raise TypicalReviewError(f"review candidate must be {CANDIDATE_NAME!r}")
    _require_exact_int(
        review.get("expected_count"),
        TYPICAL_IMAGE_COUNT,
        label="review.expected_count",
    )
    if review.get("evidence_sha256") != evidence["evidence_sha256"]:
        raise TypicalReviewError("review evidence digest does not match current evidence")
    review_stems = _validate_stems(review.get("stems"), label="review.stems")
    if review_stems != evidence_stems:
        raise TypicalReviewError("review stems do not exactly match evidence stems")

    rows = review.get("reviews")
    if not isinstance(rows, list) or len(rows) != TYPICAL_IMAGE_COUNT:
        raise TypicalReviewError(
            f"review must contain exactly {TYPICAL_IMAGE_COUNT} per-image rows"
        )
    row_stems: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypicalReviewError(f"review row {index} is not a mapping")
        _require_exact_fields(row, _REVIEW_ROW_FIELDS, label=f"review row {index}")
        stem = row.get("stem")
        if not isinstance(stem, str) or not stem:
            raise TypicalReviewError(f"review row {index} has an invalid stem")
        row_stems.append(stem)
    if len(set(stem.casefold() for stem in row_stems)) != TYPICAL_IMAGE_COUNT:
        raise TypicalReviewError("review rows contain a duplicate stem")
    missing = set(evidence_stems) - set(row_stems)
    extra = set(row_stems) - set(evidence_stems)
    if missing or extra:
        raise TypicalReviewError(
            f"review row stems do not match evidence; "
            f"missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}"
        )
    if row_stems != evidence_stems:
        raise TypicalReviewError("review rows are not in exact evidence stem order")

    criterion_counts = {criterion: 0 for criterion in REVIEW_CRITERIA}
    for row in rows:
        stem = row["stem"]
        if not isinstance(row.get("notes"), str):
            raise TypicalReviewError(f"review notes must be a string: {stem}")
        for criterion in REVIEW_CRITERIA:
            status = row.get(criterion)
            if status != "pass":
                raise TypicalReviewError(
                    f"review is not fully passed: {stem}/{criterion}={status!r}"
                )
            criterion_counts[criterion] += 1

    return {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "passed",
        "candidate": CANDIDATE_NAME,
        "expected_count": TYPICAL_IMAGE_COUNT,
        "reviewed_count": TYPICAL_IMAGE_COUNT,
        "passed_image_count": TYPICAL_IMAGE_COUNT,
        "inventory_count": evidence_status["inventory_count"],
        "evidence_file_count": evidence_status["file_count"],
        "evidence_sha256": evidence["evidence_sha256"],
        "stems": evidence_stems,
        "criterion_pass_counts": criterion_counts,
    }


__all__ = [
    "CANDIDATE_NAME",
    "EVIDENCE_SCHEMA",
    "EVIDENCE_SCHEMA_VERSION",
    "INVENTORY_NAMES",
    "REVIEW_CRITERIA",
    "REVIEW_SCHEMA",
    "REVIEW_SCHEMA_VERSION",
    "SUMMARY_SCHEMA",
    "SUMMARY_SCHEMA_VERSION",
    "TYPICAL_IMAGE_COUNT",
    "TypicalReviewError",
    "build_typical_evidence",
    "canonical_evidence_sha256",
    "make_typical_review_template",
    "validate_completed_typical_review",
    "validate_typical_evidence",
]
