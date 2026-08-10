from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diffusion2raft import external_file


class ExternalFileIdentityTest(unittest.TestCase):
    @staticmethod
    def _replace_same_size(path: Path, data: bytes, mtime_ns: int) -> None:
        if len(data) != path.stat().st_size:
            raise AssertionError("replacement must preserve file size")
        replacement = path.with_name(path.name + ".replacement")
        replacement.write_bytes(data)
        os.utime(replacement, ns=(mtime_ns, mtime_ns))
        os.replace(replacement, path)

    def test_every_validation_rehashes_and_catches_restored_stat_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pt"
            path.write_bytes(b"original-bytes")
            original_mtime = path.stat().st_mtime_ns

            real_hash_fd = external_file._hash_fd
            with mock.patch.object(
                external_file, "_hash_fd", wraps=real_hash_fd
            ) as hash_fd:
                first = external_file.stable_external_file_identity(path)
                second = external_file.stable_external_file_identity(path)
                self.assertEqual(first, second)
                self.assertEqual(hash_fd.call_count, 2)

                self._replace_same_size(path, b"replaced-bytes", original_mtime)
                third = external_file.stable_external_file_identity(path)
                self.assertEqual(third["file_size"], first["file_size"])
                self.assertEqual(third["mtime_ns"], first["mtime_ns"])
                self.assertNotEqual(third["sha256"], first["sha256"])
                self.assertEqual(hash_fd.call_count, 3)

                # The shared filesystem may preserve ctime_ns for a rapid
                # same-inode rewrite, so stat-keyed digest memoization is not
                # a safe substitute for reading the bytes again.
                path.write_bytes(b"original-bytes")
                os.utime(path, ns=(original_mtime, original_mtime))
                fourth = external_file.stable_external_file_identity(path)
                self.assertEqual(fourth["sha256"], first["sha256"])
                self.assertEqual(hash_fd.call_count, 4)

    def test_open_flags_and_path_identity_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pt"
            path.write_bytes(b"stable")
            real_open = os.open
            observed_flags: list[int] = []

            def recording_open(name, flags, *args, **kwargs):
                observed_flags.append(flags)
                return real_open(name, flags, *args, **kwargs)

            with mock.patch.object(external_file.os, "open", side_effect=recording_open):
                external_file.stable_external_file_identity(path)
            self.assertTrue(observed_flags)
            self.assertTrue(observed_flags[-1] & os.O_NOFOLLOW)
            self.assertTrue(observed_flags[-1] & os.O_NONBLOCK)

            race_path = Path(directory) / "race.pt"
            race_path.write_bytes(b"stable")
            original_hash_fd = external_file._hash_fd

            def replace_during_hash(fd: int) -> str:
                digest = original_hash_fd(fd)
                self._replace_same_size(
                    race_path, b"change", race_path.stat().st_mtime_ns
                )
                return digest

            with mock.patch.object(
                external_file, "_hash_fd", side_effect=replace_during_hash
            ):
                with self.assertRaisesRegex(
                    external_file.ExternalFileIdentityError,
                    "changed while hashing",
                ):
                    external_file.stable_external_file_identity(race_path)

    def test_sha256_must_be_canonical_lowercase(self) -> None:
        with self.assertRaisesRegex(
            external_file.ExternalFileIdentityError, "64 lowercase"
        ):
            external_file.canonical_sha256("A" * 64)


if __name__ == "__main__":
    unittest.main()
