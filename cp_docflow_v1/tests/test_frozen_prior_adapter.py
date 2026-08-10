from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cp_docflow.checkpoint import file_sha256
from cp_docflow.evaluate_frozen_prior import FrozenSupervisedPriorAdapter
from cp_docflow.geometry import canonical_backward_map


class _ZeroFlowTeacher(nn.Module):
    def forward(self, first: torch.Tensor, second: torch.Tensor) -> list[torch.Tensor]:
        del second
        return [first[:, :2] * 0.0]


class FrozenPriorAdapterTest(unittest.TestCase):
    def test_zero_displacement_becomes_absolute_identity_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "teacher.pt"
            example = torch.zeros(1, 3, 8, 8)
            teacher = torch.jit.trace(
                _ZeroFlowTeacher(), (example, example), strict=False
            )
            torch.jit.save(teacher, checkpoint)
            adapter = FrozenSupervisedPriorAdapter(
                checkpoint,
                device="cpu",
                expected_sha256=file_sha256(checkpoint),
                expected_size_bytes=checkpoint.stat().st_size,
                input_size=8,
                flow_size=8,
                blur_kernel=1,
                autocast_dtype="float32",
                requires_logical_cuda0=False,
            )
            warped = torch.rand(1, 3, 10, 6)
            output = adapter(warped, output_size=(10, 6))
            expected = canonical_backward_map(1, (10, 6), (10, 6))
            torch.testing.assert_close(output["backward_map"], expected)
            self.assertEqual(output["refiner_sequence"], [])

    def test_checkpoint_hash_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "teacher.pt"
            example = torch.zeros(1, 3, 8, 8)
            torch.jit.save(
                torch.jit.trace(
                    _ZeroFlowTeacher(), (example, example), strict=False
                ),
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                FrozenSupervisedPriorAdapter(
                    checkpoint,
                    device="cpu",
                    expected_sha256="0" * 64,
                    expected_size_bytes=checkpoint.stat().st_size,
                    input_size=8,
                    flow_size=8,
                    blur_kernel=1,
                    autocast_dtype="float32",
                    requires_logical_cuda0=False,
                )


if __name__ == "__main__":
    unittest.main()
