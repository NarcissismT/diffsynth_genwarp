from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion2raft.typical_review import (
    CANDIDATE_NAME,
    INVENTORY_NAMES,
    REVIEW_CRITERIA,
    TYPICAL_IMAGE_COUNT,
    TypicalReviewError,
    build_typical_evidence,
    canonical_evidence_sha256,
    make_typical_review_template,
    validate_completed_typical_review,
    validate_typical_evidence,
)


OUTPUT_SUFFIXES = {
    "final": "_rectified.png",
    "raw": "_rectified_raw.png",
    "prior": "_prior_rectified.png",
    "valid": "_valid.png",
    "inpaint": "_inpaint_mask.png",
    "evaluation_valid": "_evaluation_valid.png",
}


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_evidence_tree(root: Path) -> tuple[list[str], dict[str, Path]]:
    stems = [f"Page_{index:02d}" for index in range(TYPICAL_IMAGE_COUNT)]
    directories = {
        "source": root / "source",
        "target_first": root / "target_first",
        "target_second": root / "target_second",
        "anchor": root / "v33_anchor",
        "best": root / "v33_best",
    }
    for directory in directories.values():
        directory.mkdir(parents=True)
    for index, stem in enumerate(stems):
        for label in ("source", "target_first", "target_second"):
            _write(
                directories[label] / f"{stem}.jpg",
                f"{label}:{index}".encode("ascii"),
            )
        for candidate in ("anchor", "best"):
            for kind, suffix in OUTPUT_SUFFIXES.items():
                _write(
                    directories[candidate] / f"{stem}{suffix}",
                    f"{candidate}:{kind}:{index}".encode("ascii"),
                )
        # Non-review inference sidecars are intentionally allowed and ignored.
        _write(directories["anchor"] / f"{stem}_metadata.json", b"{}")
        _write(directories["best"] / f"{stem}_feature_confidence.png", b"confidence")
    _write(directories["anchor"] / "inference_report.json", b"{}")
    _write(directories["best"] / "inference_report.json", b"{}")
    return stems, directories


def _build(directories: dict[str, Path]) -> dict[str, object]:
    return build_typical_evidence(
        directories["source"],
        directories["target_first"],
        directories["target_second"],
        directories["anchor"],
        directories["best"],
    )


def _mark_all_pass(review: dict[str, object]) -> None:
    for row in review["reviews"]:  # type: ignore[index, union-attr]
        for criterion in REVIEW_CRITERIA:
            row[criterion] = "pass"


class TypicalReviewTest(unittest.TestCase):
    def test_builds_complete_anchor_and_best_evidence_and_pass_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stems, directories = _make_evidence_tree(root)
            evidence = _build(directories)

            self.assertEqual(evidence["candidate"], CANDIDATE_NAME)
            self.assertEqual(evidence["expected_count"], TYPICAL_IMAGE_COUNT)
            self.assertEqual(evidence["stems"], stems)
            self.assertEqual(tuple(evidence["inventories"]), INVENTORY_NAMES)
            self.assertEqual(
                evidence["evidence_sha256"], canonical_evidence_sha256(evidence)
            )
            self.assertEqual(
                validate_typical_evidence(evidence)["file_count"],
                len(INVENTORY_NAMES) * TYPICAL_IMAGE_COUNT,
            )

            first = evidence["inventories"]["v33_best_final"][0]
            expected_path = directories["best"] / f"{stems[0]}_rectified.png"
            self.assertEqual(first["path"], str(expected_path.resolve()))
            self.assertEqual(first["stat"]["size_bytes"], expected_path.stat().st_size)
            self.assertEqual(first["stat"]["mtime_ns"], expected_path.stat().st_mtime_ns)
            self.assertEqual(
                first["sha256"], hashlib.sha256(expected_path.read_bytes()).hexdigest()
            )

            review = make_typical_review_template(evidence)
            self.assertIn("at_least_target_second", REVIEW_CRITERIA)
            self.assertEqual(len(review["reviews"]), TYPICAL_IMAGE_COUNT)
            self.assertTrue(
                all(
                    row[criterion] == "pending"
                    for row in review["reviews"]
                    for criterion in REVIEW_CRITERIA
                )
            )
            _mark_all_pass(review)
            summary = validate_completed_typical_review(review, evidence)
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["candidate"], CANDIDATE_NAME)
            self.assertEqual(summary["reviewed_count"], TYPICAL_IMAGE_COUNT)
            self.assertEqual(summary["passed_image_count"], TYPICAL_IMAGE_COUNT)
            self.assertEqual(summary["evidence_file_count"], 600)
            self.assertEqual(
                summary["criterion_pass_counts"],
                {criterion: TYPICAL_IMAGE_COUNT for criterion in REVIEW_CRITERIA},
            )

    def test_inventory_rejects_missing_extra_duplicate_and_stem_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, directories = _make_evidence_tree(Path(temporary))

            missing = directories["best"] / "Page_00_rectified_raw.png"
            payload = missing.read_bytes()
            missing.unlink()
            with self.assertRaisesRegex(TypicalReviewError, "v33_best/raw.*exactly 40"):
                _build(directories)
            _write(missing, payload)

            extra = directories["anchor"] / "not_a_source_rectified.png"
            _write(extra, b"extra")
            with self.assertRaisesRegex(TypicalReviewError, "v33_anchor/final.*found=41"):
                _build(directories)
            extra.unlink()

            duplicate = directories["source"] / "page_00.jpg"
            _write(duplicate, b"duplicate")
            with self.assertRaisesRegex(TypicalReviewError, "case-insensitive duplicate"):
                _build(directories)
            duplicate.unlink()

            old_target = directories["target_second"] / "Page_00.jpg"
            new_target = directories["target_second"] / "Other.jpg"
            old_target.rename(new_target)
            with self.assertRaisesRegex(TypicalReviewError, "stems do not exactly match"):
                _build(directories)

    def test_rejects_wrong_reference_extension_and_output_suffix_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, directories = _make_evidence_tree(Path(temporary))
            original = directories["target_first"] / "Page_00.jpg"
            wrong = directories["target_first"] / "Page_00.PNG"
            original.rename(wrong)
            with self.assertRaisesRegex(TypicalReviewError, "lowercase .jpg"):
                _build(directories)
            wrong.rename(original)

            output = directories["best"] / "Page_00_valid.png"
            uppercase = directories["best"] / "Page_00_valid.PNG"
            output.rename(uppercase)
            with self.assertRaisesRegex(TypicalReviewError, "exact lowercase spelling"):
                _build(directories)

    def test_evidence_digest_and_current_files_are_both_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, directories = _make_evidence_tree(Path(temporary))
            evidence = _build(directories)

            tampered = copy.deepcopy(evidence)
            tampered["inventories"]["source"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(TypicalReviewError, "evidence digest mismatch"):
                validate_typical_evidence(tampered, verify_files=False)

            changed = Path(evidence["inventories"]["v33_anchor_prior"][0]["path"])
            changed.write_bytes(changed.read_bytes() + b"changed-after-review")
            with self.assertRaisesRegex(TypicalReviewError, "file changed after inventory"):
                validate_typical_evidence(evidence)

    def test_post_inventory_extra_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, directories = _make_evidence_tree(Path(temporary))
            evidence = _build(directories)
            _write(directories["source"] / "Extra.jpg", b"late source")
            with self.assertRaisesRegex(TypicalReviewError, "exactly 40"):
                validate_typical_evidence(evidence)

        with tempfile.TemporaryDirectory() as temporary:
            _, directories = _make_evidence_tree(Path(temporary))
            evidence = _build(directories)
            _write(
                directories["best"] / "Extra_rectified.png",
                b"late candidate",
            )
            with self.assertRaisesRegex(TypicalReviewError, "exactly 40"):
                validate_typical_evidence(evidence)

    def test_review_is_strict_about_schema_candidate_rows_digest_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, directories = _make_evidence_tree(Path(temporary))
            evidence = _build(directories)
            passed = make_typical_review_template(evidence)
            _mark_all_pass(passed)
            validate_completed_typical_review(passed, evidence)

            cases: list[tuple[str, dict[str, object], str]] = []

            pending = copy.deepcopy(passed)
            pending["reviews"][0]["full_resolution_reviewed"] = "pending"
            cases.append(("pending", pending, "not fully passed"))

            failed = copy.deepcopy(passed)
            failed["reviews"][0]["content_preserved"] = "fail"
            cases.append(("fail", failed, "not fully passed"))

            second_failed = copy.deepcopy(passed)
            second_failed["reviews"][0]["at_least_target_second"] = "fail"
            cases.append(("target-second", second_failed, "not fully passed"))

            wrong_candidate = copy.deepcopy(passed)
            wrong_candidate["candidate"] = "v33_anchor"
            cases.append(("candidate", wrong_candidate, "candidate must be 'v33_best'"))

            wrong_schema = copy.deepcopy(passed)
            wrong_schema["schema_version"] = 1
            cases.append(("schema", wrong_schema, "schema_version"))

            wrong_count = copy.deepcopy(passed)
            wrong_count["expected_count"] = 39
            cases.append(("count", wrong_count, "expected_count"))

            wrong_digest = copy.deepcopy(passed)
            wrong_digest["evidence_sha256"] = "0" * 64
            cases.append(("digest", wrong_digest, "digest does not match"))

            wrong_stems = copy.deepcopy(passed)
            wrong_stems["stems"][-1] = "Wrong"
            cases.append(("stems", wrong_stems, "stems"))

            missing = copy.deepcopy(passed)
            missing["reviews"].pop()
            cases.append(("missing", missing, "exactly 40 per-image rows"))

            duplicate = copy.deepcopy(passed)
            duplicate["reviews"][1]["stem"] = duplicate["reviews"][0]["stem"]
            cases.append(("duplicate", duplicate, "duplicate stem"))

            bool_status = copy.deepcopy(passed)
            bool_status["reviews"][0]["line_straightness"] = True
            cases.append(("boolean", bool_status, "not fully passed"))

            for label, review, error_pattern in cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(TypicalReviewError, error_pattern):
                        validate_completed_typical_review(review, evidence)


if __name__ == "__main__":
    unittest.main()
