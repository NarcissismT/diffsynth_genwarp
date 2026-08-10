from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diffusion2raft import teacher_capacity_assets as assets


class TeacherCapacityAssetsTest(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path, *, count: int = 300) -> Path:
        (root / "warped.bin").write_bytes(b"warped-v1")
        (root / "target.bin").write_bytes(b"target-v1")
        (root / "flow.bin").write_bytes(b"flow-v1")
        (root / "valid.bin").write_bytes(b"valid-v1")
        # A deliberately nonexistent guide proves it is not authenticated by
        # this audit. DocumentFlowDataset's audit path does not load guides.
        records = []
        for index in range(count):
            record = {
                "id": f"sample-{index}",
                "warped": "warped.bin",
                "target": "target.bin",
                "flow": "flow.bin",
                "guide": "guide-is-outside-this-audit.bin",
            }
            if index % 2 == 0:
                record["valid"] = "valid.bin"
            records.append(record)
        manifest = root / "val.jsonl"
        manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return manifest

    @staticmethod
    def _replace_same_size(path: Path, payload: bytes, mtime_ns: int) -> None:
        if len(payload) != path.stat().st_size:
            raise AssertionError("replacement must preserve size")
        replacement = path.with_name(path.name + ".replacement")
        replacement.write_bytes(payload)
        os.utime(replacement, ns=(mtime_ns, mtime_ns))
        os.replace(replacement, path)

    def test_builds_canonical_deduplicated_identity_and_fast_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_fixture(root)
            identity = assets.build_teacher_capacity_asset_manifest(manifest)

            self.assertEqual(
                set(identity),
                {
                    "schema_version",
                    "kind",
                    "manifest",
                    "record_count",
                    "unique_assets",
                    "record_refs",
                    "aggregate_sha256",
                },
            )
            self.assertEqual(identity["record_count"], 300)
            self.assertEqual(len(identity["record_refs"]), 300)
            self.assertEqual(len(identity["unique_assets"]), 4)
            self.assertEqual(
                [entry["asset_index"] for entry in identity["unique_assets"]],
                list(range(4)),
            )
            self.assertEqual(
                len({entry["path"] for entry in identity["unique_assets"]}), 4
            )
            first, second = identity["record_refs"][:2]
            self.assertEqual(first["warped"], second["warped"])
            self.assertIsNotNone(first["valid"])
            self.assertIsNone(second["valid"])
            self.assertTrue(
                all(
                    ref["guide_excluded_reason"] == assets.GUIDE_EXCLUDED_REASON
                    for ref in identity["record_refs"]
                )
            )
            manual_payload = {
                key: value
                for key, value in identity.items()
                if key != "aggregate_sha256"
            }
            expected = hashlib.sha256(
                json.dumps(
                    manual_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(identity["aggregate_sha256"], expected)
            self.assertEqual(
                assets.canonical_teacher_capacity_asset_digest(identity), expected
            )

            # The fast path hashes only the manifest, through its separate
            # stable-read function. It must never call the asset byte hasher.
            with mock.patch.object(
                assets, "_hash_fd", wraps=assets._hash_fd
            ) as asset_hasher:
                verified = assets.fast_verify_teacher_capacity_asset_manifest(
                    manifest,
                    identity,
                    expected_aggregate_sha256=identity["aggregate_sha256"],
                )
            self.assertEqual(asset_hasher.call_count, 0)
            self.assertEqual(verified, identity)
            self.assertIsNot(verified, identity)

    def test_same_size_mtime_replacement_has_new_hash_and_fast_rejects_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_fixture(root)
            before = assets.build_teacher_capacity_asset_manifest(manifest)
            path = root / "warped.bin"
            original_mtime = path.stat().st_mtime_ns
            old_inode = path.stat().st_ino
            self._replace_same_size(path, b"changed-1", original_mtime)
            self.assertNotEqual(path.stat().st_ino, old_inode)
            self.assertEqual(path.stat().st_mtime_ns, original_mtime)

            after = assets.build_teacher_capacity_asset_manifest(manifest)
            before_asset = next(
                item for item in before["unique_assets"] if item["path"] == str(path)
            )
            after_asset = next(
                item for item in after["unique_assets"] if item["path"] == str(path)
            )
            self.assertEqual(before_asset["size"], after_asset["size"])
            self.assertEqual(before_asset["mtime_ns"], after_asset["mtime_ns"])
            self.assertNotEqual(before_asset["inode"], after_asset["inode"])
            self.assertNotEqual(before_asset["sha256"], after_asset["sha256"])
            with self.assertRaisesRegex(
                assets.TeacherCapacityAssetsError, "inode differs"
            ):
                assets.fast_verify_teacher_capacity_asset_manifest(manifest, before)

    def test_manifest_is_stably_read_and_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_fixture(root)
            original_reader = assets._read_and_hash_fd
            original = manifest.read_bytes()
            replacement = bytearray(original)
            replacement[replacement.index(b"sample-0")] ^= 1
            original_mtime = manifest.stat().st_mtime_ns

            def replace_after_read(fd: int):
                result = original_reader(fd)
                self._replace_same_size(manifest, bytes(replacement), original_mtime)
                return result

            with mock.patch.object(
                assets, "_read_and_hash_fd", side_effect=replace_after_read
            ):
                with self.assertRaisesRegex(
                    assets.TeacherCapacityAssetsError, "changed while hashing"
                ):
                    assets.build_teacher_capacity_asset_manifest(manifest)

    def test_asset_replacement_during_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_fixture(root)
            warped = root / "warped.bin"
            original_mtime = warped.stat().st_mtime_ns
            original_hasher = assets._hash_fd
            replaced = False

            def replace_warped_after_hash(fd: int) -> str:
                nonlocal replaced
                digest = original_hasher(fd)
                if not replaced:
                    replaced = True
                    self._replace_same_size(warped, b"changed-1", original_mtime)
                return digest

            with mock.patch.object(
                assets, "_hash_fd", side_effect=replace_warped_after_hash
            ):
                with self.assertRaisesRegex(
                    assets.TeacherCapacityAssetsError, "changed while hashing"
                ):
                    assets.build_teacher_capacity_asset_manifest(manifest)

    def test_manifest_count_symlink_missing_and_nonregular_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short = self._write_fixture(root, count=299)
            with self.assertRaisesRegex(
                assets.TeacherCapacityAssetsError, "exactly 300"
            ):
                assets.build_teacher_capacity_asset_manifest(short)

        for kind in ("missing", "directory", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self._write_fixture(root)
                path = root / "flow.bin"
                path.unlink()
                if kind == "directory":
                    path.mkdir()
                elif kind == "symlink":
                    target = root / "flow-target.bin"
                    target.write_bytes(b"flow-v1")
                    path.symlink_to(target)
                with self.assertRaises(assets.TeacherCapacityAssetsError):
                    assets.build_teacher_capacity_asset_manifest(manifest)

    def test_strict_schema_duplicate_indexes_and_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = assets.build_teacher_capacity_asset_manifest(
                self._write_fixture(root)
            )

        cases: list[tuple[str, dict, str]] = []
        tampered = copy.deepcopy(identity)
        tampered["unique_assets"][0]["size"] += 1
        cases.append(("tamper", tampered, "aggregate_sha256"))

        duplicate_record = copy.deepcopy(identity)
        duplicate_record["record_refs"][1]["dataset_index"] = 0
        cases.append(("duplicate record", duplicate_record, "duplicate dataset_index"))

        duplicate_asset = copy.deepcopy(identity)
        duplicate_asset["unique_assets"][1]["asset_index"] = 0
        cases.append(("duplicate asset", duplicate_asset, "duplicate asset_index"))

        duplicate_path = copy.deepcopy(identity)
        duplicate_path["unique_assets"][1]["path"] = duplicate_path["unique_assets"][0]["path"]
        cases.append(("duplicate path", duplicate_path, "duplicate resolved path"))

        bad_schema = copy.deepcopy(identity)
        bad_schema["comment"] = "not allowed"
        cases.append(("extra field", bad_schema, "fields differ"))

        bad_guide_reason = copy.deepcopy(identity)
        bad_guide_reason["record_refs"][0]["guide_excluded_reason"] = "ignored"
        cases.append(("guide reason", bad_guide_reason, "guide_excluded_reason"))

        for label, candidate, error in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(assets.TeacherCapacityAssetsError, error):
                    assets.validate_teacher_capacity_asset_identity(candidate)

    def test_fast_verify_rejects_manifest_path_retarget_and_manifest_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_fixture(root)
            identity = assets.build_teacher_capacity_asset_manifest(manifest)

            # A retarget through a symlink is rejected even when it reaches a
            # regular file with the expected bytes.
            warped = root / "warped.bin"
            target = root / "warped-target.bin"
            target.write_bytes(warped.read_bytes())
            warped.unlink()
            warped.symlink_to(target)
            with self.assertRaisesRegex(
                assets.TeacherCapacityAssetsError, "symlink"
            ):
                assets.fast_verify_teacher_capacity_asset_manifest(manifest, identity)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_fixture(root)
            identity = assets.build_teacher_capacity_asset_manifest(manifest)
            text = manifest.read_text(encoding="utf-8")
            manifest.write_text(text.replace("sample-0", "sample-x", 1), encoding="utf-8")
            with self.assertRaisesRegex(
                assets.TeacherCapacityAssetsError,
                "validation manifest (size|mtime_ns|sha256) differs",
            ):
                assets.fast_verify_teacher_capacity_asset_manifest(manifest, identity)


if __name__ == "__main__":
    unittest.main()
