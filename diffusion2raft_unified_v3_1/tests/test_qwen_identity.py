from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diffusion2raft import qwen_identity


class QwenIdentityTest(unittest.TestCase):
    @staticmethod
    def _make_model_tree(root: Path) -> dict[str, bytes]:
        payloads: dict[str, bytes] = {}
        for index, relative_path in enumerate(
            qwen_identity.QWEN_MODEL_RELATIVE_PATHS
        ):
            path = root.joinpath(*relative_path.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"qwen-v1:{index:02d}:{relative_path}".encode("utf-8")
            path.write_bytes(payload)
            payloads[relative_path] = payload
        return payloads

    @staticmethod
    def _path(root: Path, relative_path: str) -> Path:
        return root.joinpath(*relative_path.split("/"))

    @staticmethod
    def _replace_same_size(path: Path, payload: bytes, mtime_ns: int) -> None:
        if len(payload) != path.stat().st_size:
            raise AssertionError("replacement payload must preserve size")
        replacement = path.with_name(path.name + ".replacement")
        replacement.write_bytes(payload)
        os.utime(replacement, ns=(mtime_ns, mtime_ns))
        os.replace(replacement, path)

    @staticmethod
    def _manual_manifest_digest(manifest: dict) -> str:
        payload = {
            "schema_version": manifest["schema_version"],
            "files": manifest["files"],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def test_fixed_list_and_canonical_manifest_digest(self) -> None:
        self.assertEqual(len(qwen_identity.QWEN_MODEL_RELATIVE_PATHS), 37)
        self.assertEqual(
            len(set(qwen_identity.QWEN_MODEL_RELATIVE_PATHS)), 37
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_model_tree(root)
            manifest = qwen_identity.build_qwen_manifest(root)

        self.assertEqual(
            set(manifest), {"schema_version", "files", "manifest_sha256"}
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            tuple(entry["path"] for entry in manifest["files"]),
            qwen_identity.QWEN_MODEL_RELATIVE_PATHS,
        )
        self.assertTrue(
            all(set(entry) == {"path", "size", "sha256"} for entry in manifest["files"])
        )
        expected = self._manual_manifest_digest(manifest)
        self.assertEqual(manifest["manifest_sha256"], expected)
        self.assertEqual(
            qwen_identity.canonical_qwen_manifest_digest(manifest), expected
        )
        self.assertEqual(qwen_identity.validate_qwen_manifest(manifest), manifest)

    def test_same_size_replacement_with_restored_mtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = self._make_model_tree(root)
            manifest = qwen_identity.build_qwen_manifest(root)
            relative_path = qwen_identity.QWEN_MODEL_RELATIVE_PATHS[0]
            path = self._path(root, relative_path)
            original_mtime = path.stat().st_mtime_ns
            replacement = bytes(
                byte ^ 0x01 for byte in payloads[relative_path]
            )
            self._replace_same_size(path, replacement, original_mtime)

            with self.assertRaisesRegex(
                qwen_identity.QwenIdentityError, "sha256 differs"
            ):
                with qwen_identity.open_verified_qwen_tree(root, manifest):
                    self.fail("a changed Qwen file must not be exposed")

    def test_in_place_mutation_during_use_is_detected_after_mtime_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = self._make_model_tree(root)
            manifest = qwen_identity.build_qwen_manifest(root)
            relative_path = qwen_identity.QWEN_MODEL_RELATIVE_PATHS[-1]
            path = self._path(root, relative_path)
            original_mtime = path.stat().st_mtime_ns
            changed = bytes(byte ^ 0x20 for byte in payloads[relative_path])

            with self.assertRaisesRegex(
                qwen_identity.QwenIdentityError,
                rf"Qwen file {relative_path} (changed|content changed)",
            ):
                with qwen_identity.open_verified_qwen_tree(root, manifest):
                    path.write_bytes(changed)
                    os.utime(path, ns=(original_mtime, original_mtime))

    def test_procfd_mirror_reads_authenticated_inode_after_path_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = self._make_model_tree(root)
            manifest = qwen_identity.build_qwen_manifest(root)
            relative_path = qwen_identity.QWEN_MODEL_RELATIVE_PATHS[1]
            path = self._path(root, relative_path)
            old_bytes = payloads[relative_path]
            new_bytes = bytes(byte ^ 0x04 for byte in old_bytes)
            load_root: Path | None = None

            with self.assertRaisesRegex(
                qwen_identity.QwenIdentityError,
                rf"Qwen file {relative_path} changed while in use",
            ):
                with qwen_identity.open_verified_qwen_tree(
                    root, manifest
                ) as opened:
                    load_root = Path(opened.load_path)
                    mirrored = self._path(load_root, relative_path)
                    self.assertTrue(mirrored.is_symlink())
                    self.assertRegex(os.readlink(mirrored), r"^/proc/self/fd/[0-9]+$")

                    self._replace_same_size(
                        path, new_bytes, path.stat().st_mtime_ns
                    )
                    self.assertEqual(mirrored.read_bytes(), old_bytes)
                    self.assertEqual(path.read_bytes(), new_bytes)

            self.assertIsNotNone(load_root)
            self.assertFalse(load_root.exists())

    def test_manifest_tampering_and_digest_pin_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_model_tree(root)
            manifest = qwen_identity.build_qwen_manifest(root)

        tampered = copy.deepcopy(manifest)
        tampered["files"][0]["size"] += 1
        with self.assertRaisesRegex(
            qwen_identity.QwenIdentityError, "does not match its contents"
        ):
            qwen_identity.validate_qwen_manifest(tampered)

        wrong_pin = "0" * 64
        if wrong_pin == manifest["manifest_sha256"]:
            wrong_pin = "1" * 64
        with self.assertRaisesRegex(
            qwen_identity.QwenIdentityError, "configured expected digest"
        ):
            qwen_identity.validate_qwen_manifest(
                manifest, expected_manifest_sha256=wrong_pin
            )

    def test_verified_context_rehashes_every_fd_without_a_stat_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_model_tree(root)
            manifest = qwen_identity.build_qwen_manifest(root)
            real_hash_fd = qwen_identity._hash_fd
            with mock.patch.object(
                qwen_identity, "_hash_fd", wraps=real_hash_fd
            ) as hash_fd:
                with qwen_identity.open_verified_qwen_tree(root, manifest):
                    pass
        self.assertEqual(
            hash_fd.call_count,
            2 * len(qwen_identity.QWEN_MODEL_RELATIVE_PATHS),
        )

    def test_unrelated_files_do_not_change_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_model_tree(root)
            before = qwen_identity.build_qwen_manifest(root)
            (root / "README.md").write_text("not a runtime dependency")
            (root / ".download-marker").write_text("ignored")
            (root / "cache").mkdir()
            (root / "cache" / "unrelated.bin").write_bytes(b"ignored")
            after = qwen_identity.build_qwen_manifest(root)
        self.assertEqual(after, before)

    def test_strict_schema_rejects_bad_paths_duplicates_and_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_model_tree(root)
            manifest = qwen_identity.build_qwen_manifest(root)

        cases: list[tuple[str, dict, str]] = []
        absolute = copy.deepcopy(manifest)
        absolute["files"][0]["path"] = "/model_index.json"
        cases.append(("absolute", absolute, "must not be absolute"))

        traversal = copy.deepcopy(manifest)
        traversal["files"][0]["path"] = "../model_index.json"
        cases.append(("parent traversal", traversal, "contain '.'/'..'"))

        duplicate = copy.deepcopy(manifest)
        duplicate["files"][1]["path"] = duplicate["files"][0]["path"]
        cases.append(("duplicate", duplicate, "duplicate relative paths"))

        missing_path = copy.deepcopy(manifest)
        missing_path["files"].pop()
        cases.append(("missing dependency", missing_path, "fixed v1 dependency list"))

        extra_path = copy.deepcopy(manifest)
        extra_path["files"].append(
            {"path": "unrelated.bin", "size": 0, "sha256": "0" * 64}
        )
        cases.append(("extra dependency", extra_path, "fixed v1 dependency list"))

        wrong_order = copy.deepcopy(manifest)
        wrong_order["files"][0], wrong_order["files"][1] = (
            wrong_order["files"][1],
            wrong_order["files"][0],
        )
        cases.append(("wrong order", wrong_order, "fixed v1 dependency list"))

        float_version = copy.deepcopy(manifest)
        float_version["schema_version"] = 1.0
        cases.append(("float version", float_version, "must be integer 1"))

        missing_top = copy.deepcopy(manifest)
        del missing_top["schema_version"]
        cases.append(("missing top field", missing_top, "fields differ"))

        extra_top = copy.deepcopy(manifest)
        extra_top["comment"] = "not permitted"
        cases.append(("extra top field", extra_top, "fields differ"))

        missing_entry = copy.deepcopy(manifest)
        del missing_entry["files"][0]["size"]
        cases.append(("missing entry field", missing_entry, "fields differ"))

        extra_entry = copy.deepcopy(manifest)
        extra_entry["files"][0]["mtime_ns"] = 0
        cases.append(("extra entry field", extra_entry, "fields differ"))

        tuple_files = copy.deepcopy(manifest)
        tuple_files["files"] = tuple(tuple_files["files"])
        cases.append(("non-list files", tuple_files, "must be a list"))

        for label, candidate, expected_error in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    qwen_identity.QwenIdentityError, expected_error
                ):
                    qwen_identity.validate_qwen_manifest(candidate)

    def test_missing_symlink_and_non_regular_dependencies_are_rejected(self) -> None:
        relative_path = qwen_identity.QWEN_MODEL_RELATIVE_PATHS[0]
        for kind in ("missing", "symlink", "directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._make_model_tree(root)
                path = self._path(root, relative_path)
                path.unlink()
                if kind == "symlink":
                    target = root / "unrelated-target"
                    target.write_bytes(b"target")
                    path.symlink_to(target)
                elif kind == "directory":
                    path.mkdir()
                with self.assertRaises(qwen_identity.QwenIdentityError):
                    qwen_identity.build_qwen_manifest(root)

    def test_all_model_nodes_are_opened_with_nofollow_nonblock_and_openat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_model_tree(root)
            real_open = os.open
            calls: list[tuple[object, int, object]] = []

            def recording_open(path, flags, *args, **kwargs):
                calls.append((path, flags, kwargs.get("dir_fd")))
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(
                qwen_identity.os, "open", side_effect=recording_open
            ):
                qwen_identity.build_qwen_manifest(root)

        self.assertGreater(len(calls), len(qwen_identity.QWEN_MODEL_RELATIVE_PATHS))
        for _path, flags, _dir_fd in calls:
            self.assertTrue(flags & os.O_NOFOLLOW)
            self.assertTrue(flags & os.O_NONBLOCK)
        self.assertIsNone(calls[0][2])
        self.assertTrue(all(dir_fd is not None for _, _, dir_fd in calls[1:]))


if __name__ == "__main__":
    unittest.main()
