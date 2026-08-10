from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion2raft.infer import (
    CheckpointProvenanceError,
    load_checkpoint_with_provenance,
)


def _artifact(path: Path) -> dict[str, object]:
    value = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class CheckpointProvenanceTest(unittest.TestCase):
    def test_hashes_compares_and_loads_the_same_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            torch.save({"identity": "expected"}, checkpoint)
            expected = _artifact(checkpoint)

            payload, actual = load_checkpoint_with_provenance(
                checkpoint, expected_artifact=expected
            )
            self.assertEqual(payload, {"identity": "expected"})
            self.assertEqual(actual, expected)

    def test_swapped_same_stat_path_is_rejected_by_sha_before_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"A" * 4096)
            expected = _artifact(checkpoint)
            original_stat = checkpoint.stat()

            replacement = root / "replacement.pt"
            replacement.write_bytes(b"B" * 4096)
            os.utime(
                replacement,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            os.replace(replacement, checkpoint)

            with mock.patch("diffusion2raft.infer.torch.load") as torch_load:
                with self.assertRaisesRegex(
                    CheckpointProvenanceError, "does not match finalizer attestation"
                ):
                    load_checkpoint_with_provenance(
                        checkpoint, expected_artifact=expected
                    )
            torch_load.assert_not_called()

    def test_swap_and_restore_cannot_change_the_loaded_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            restore_link = root / "restore-original.pt"
            replacement = root / "replacement.pt"
            torch.save({"identity": "expected"}, checkpoint)
            os.link(checkpoint, restore_link)
            torch.save({"identity": "wrong"}, replacement)
            expected = _artifact(checkpoint)
            real_torch_load = torch.load
            loaded_from_descriptor: list[bool] = []
            loaded_identities: list[str] = []

            def swap_restore_and_load(source: object, *args: object, **kwargs: object):
                loaded_from_descriptor.append(
                    hasattr(source, "fileno") and not isinstance(source, (str, Path))
                )
                os.replace(replacement, checkpoint)
                try:
                    payload = real_torch_load(source, *args, **kwargs)
                    loaded_identities.append(payload["identity"])
                    return payload
                finally:
                    # Restore the exact original inode before the final lstat.
                    os.replace(restore_link, checkpoint)

            with mock.patch(
                "diffusion2raft.infer.torch.load", side_effect=swap_restore_and_load
            ):
                try:
                    _, actual = load_checkpoint_with_provenance(
                        checkpoint, expected_artifact=expected
                    )
                except CheckpointProvenanceError as error:
                    # Some filesystems update ctime when the hard-link count
                    # changes and therefore reject even this harmless swap.
                    self.assertIn("changed or its pathname was replaced", str(error))
                else:
                    self.assertEqual(actual, expected)
            self.assertEqual(loaded_from_descriptor, [True])
            self.assertEqual(loaded_identities, ["expected"])

    def test_path_replacement_during_load_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            replacement = root / "replacement.pt"
            torch.save({"identity": "expected"}, checkpoint)
            torch.save({"identity": "wrong"}, replacement)
            expected = _artifact(checkpoint)
            real_torch_load = torch.load

            def replace_after_load(source: object, *args: object, **kwargs: object):
                payload = real_torch_load(source, *args, **kwargs)
                os.replace(replacement, checkpoint)
                return payload

            with mock.patch(
                "diffusion2raft.infer.torch.load", side_effect=replace_after_load
            ):
                with self.assertRaisesRegex(
                    CheckpointProvenanceError, "pathname was replaced"
                ):
                    load_checkpoint_with_provenance(
                        checkpoint, expected_artifact=expected
                    )

    def test_in_place_mutation_during_load_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            torch.save({"identity": "expected"}, checkpoint)
            expected = _artifact(checkpoint)
            real_torch_load = torch.load

            def mutate_after_load(source: object, *args: object, **kwargs: object):
                payload = real_torch_load(source, *args, **kwargs)
                with checkpoint.open("ab") as handle:
                    handle.write(b"mutation")
                return payload

            with mock.patch(
                "diffusion2raft.infer.torch.load", side_effect=mutate_after_load
            ):
                with self.assertRaisesRegex(CheckpointProvenanceError, "changed"):
                    load_checkpoint_with_provenance(
                        checkpoint, expected_artifact=expected
                    )

    def test_rejects_symlink_and_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.pt"
            target.write_bytes(b"checkpoint")
            symlink = root / "symlink.pt"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(
                CheckpointProvenanceError, "safely opened"
            ):
                load_checkpoint_with_provenance(symlink)

            fifo = root / "fifo.pt"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(
                CheckpointProvenanceError, "not a regular file"
            ):
                load_checkpoint_with_provenance(fifo)

    def test_legacy_torch_retry_rewinds_the_authenticated_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            torch.save({"identity": "expected"}, checkpoint)
            real_torch_load = torch.load
            second_positions: list[int] = []

            def emulate_legacy(source: object, *args: object, **kwargs: object):
                if "weights_only" in kwargs:
                    source.read(17)  # type: ignore[attr-defined]
                    raise TypeError("legacy torch")
                second_positions.append(source.tell())  # type: ignore[attr-defined]
                return real_torch_load(source, *args, **kwargs)

            with mock.patch(
                "diffusion2raft.infer.torch.load", side_effect=emulate_legacy
            ):
                payload, _ = load_checkpoint_with_provenance(checkpoint)
            self.assertEqual(second_positions, [0])
            self.assertEqual(payload, {"identity": "expected"})


if __name__ == "__main__":
    unittest.main()
