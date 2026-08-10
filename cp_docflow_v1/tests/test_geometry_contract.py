from __future__ import annotations

import unittest

import torch

from cp_docflow.geometry import (
    ALIGN_CORNERS,
    canonical_backward_map,
    canonical_backward_map_window,
    crop_backward_map,
    flip_backward_map,
    normalized_grid_to_pixel_map,
    pad_backward_map,
    pixel_map_to_normalized_grid,
    resize_backward_map,
    resize_backward_map_with_mask,
    warp_with_backward_map,
)


class GeometryContractTest(unittest.TestCase):
    def test_align_corners_is_globally_false(self) -> None:
        self.assertIs(ALIGN_CORNERS, False)

    def test_xy_normalization_round_trip(self) -> None:
        backward_map = canonical_backward_map(2, (7, 11))
        backward_map[:, 0] += 0.25
        backward_map[:, 1] -= 0.50
        normalized = pixel_map_to_normalized_grid(backward_map, (7, 11))
        restored = normalized_grid_to_pixel_map(normalized, (7, 11))
        torch.testing.assert_close(restored, backward_map, atol=1.0e-6, rtol=0.0)
        # Last dimension of a grid_sample grid must still be x, y.
        self.assertAlmostEqual(float(normalized[0, 0, 0, 0]), (2 * 0.25 + 1) / 11 - 1)
        self.assertAlmostEqual(float(normalized[0, 0, 0, 1]), (2 * -0.50 + 1) / 7 - 1)

    def test_identity_map_is_exact_identity_warp(self) -> None:
        source = torch.rand(2, 3, 9, 13)
        identity = canonical_backward_map(2, source.shape[-2:])
        warped = warp_with_backward_map(source, identity)
        torch.testing.assert_close(warped, source, atol=1.0e-6, rtol=0.0)

    def test_target_window_keeps_complete_source_coordinates(self) -> None:
        window = torch.tensor([[5.0, 7.0, 20.0, 16.0]])
        backward_map = canonical_backward_map_window(
            1, (16, 20), (32, 40), (32, 40), window
        )
        expected = canonical_backward_map(1, (16, 20)).clone()
        expected[:, 0] += 5.0
        expected[:, 1] += 7.0
        torch.testing.assert_close(backward_map, expected, atol=1.0e-6, rtol=0.0)
        source = torch.rand(1, 3, 32, 40)
        crop = warp_with_backward_map(source, backward_map)
        torch.testing.assert_close(
            crop, source[..., 7:23, 5:25], atol=2.0e-6, rtol=0.0
        )

    def test_integer_translation_matches_manual_oracle(self) -> None:
        source = torch.arange(6 * 8, dtype=torch.float32).reshape(1, 1, 6, 8)
        backward_map = canonical_backward_map(1, (6, 8))
        backward_map[:, 0] += 1.0
        result, valid = warp_with_backward_map(
            source,
            backward_map,
            padding_mode="zeros",
            return_valid=True,
        )
        torch.testing.assert_close(result[..., :-1], source[..., 1:], atol=1.0e-6, rtol=0.0)
        self.assertTrue(bool(valid[..., :-1].all()))
        self.assertFalse(bool(valid[..., -1:].any()))

    def test_resize_preserves_canonical_map(self) -> None:
        original = canonical_backward_map(1, (5, 7), (9, 13))
        resized = resize_backward_map(
            original,
            (11, 17),
            source_size_from=(9, 13),
            source_size_to=(19, 23),
        )
        expected = canonical_backward_map(1, (11, 17), (19, 23))
        torch.testing.assert_close(resized, expected, atol=1.0e-6, rtol=0.0)

    def test_resize_scales_source_pixel_residual(self) -> None:
        original = canonical_backward_map(1, (5, 7), (9, 13))
        original = original.clone()
        original[:, 0] += 2.0
        original[:, 1] -= 1.0
        resized = resize_backward_map(
            original,
            (10, 14),
            source_size_from=(9, 13),
            source_size_to=(18, 26),
        )
        residual = resized - canonical_backward_map(1, (10, 14), (18, 26))
        torch.testing.assert_close(
            residual[:, 0],
            torch.full_like(residual[:, 0], 4.0),
            atol=1.0e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            residual[:, 1],
            torch.full_like(residual[:, 1], -2.0),
            atol=1.0e-5,
            rtol=0.0,
        )

    def test_masked_resize_never_marks_mixed_invalid_boundary_valid(self) -> None:
        backward_map = canonical_backward_map(1, (4, 4)).clone()
        backward_map[:, 0, :, :3] += 20.0
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        valid[..., -1] = False
        resized, resized_valid = resize_backward_map_with_mask(
            backward_map,
            valid,
            (8, 8),
            source_size_from=(4, 4),
            source_size_to=(8, 8),
        )
        residual = resized - canonical_backward_map(1, (8, 8))
        # Any supervised output retains the exact 2x-scaled residual. Pixels
        # whose bilinear footprint touches the invalid column are invalid.
        supervised_x = residual[:, 0:1][resized_valid]
        torch.testing.assert_close(
            supervised_x,
            torch.full_like(supervised_x, 40.0),
            atol=1.0e-5,
            rtol=0.0,
        )
        self.assertFalse(bool(resized_valid[..., -2:].any()))

    def test_synchronized_flip_preserves_identity(self) -> None:
        identity = canonical_backward_map(1, (8, 12))
        flipped = flip_backward_map(identity, (8, 12), horizontal=True, vertical=True)
        torch.testing.assert_close(flipped, identity, atol=1.0e-6, rtol=0.0)

    def test_crop_rebases_source_coordinates(self) -> None:
        identity = canonical_backward_map(1, (8, 10))
        cropped = crop_backward_map(
            identity,
            target_box=(2, 1, 8, 7),
            source_offset=(2, 1),
        )
        expected = canonical_backward_map(1, (6, 6))
        torch.testing.assert_close(cropped, expected, atol=1.0e-6, rtol=0.0)

    def test_padding_shifts_source_and_invalidates_target_border(self) -> None:
        identity = canonical_backward_map(1, (4, 5))
        valid = torch.ones(1, 1, 4, 5, dtype=torch.bool)
        padded, padded_valid = pad_backward_map(
            identity,
            valid,
            target_padding=(1, 2, 3, 1),
            source_padding=(2, 0, 4, 0),
        )
        self.assertEqual(tuple(padded.shape), (1, 2, 8, 8))
        self.assertEqual(int(padded_valid.sum()), 20)
        pasted = padded[..., 3:7, 1:6]
        expected = identity.clone()
        expected[:, 0] += 2
        expected[:, 1] += 4
        torch.testing.assert_close(pasted, expected)


if __name__ == "__main__":
    unittest.main()
