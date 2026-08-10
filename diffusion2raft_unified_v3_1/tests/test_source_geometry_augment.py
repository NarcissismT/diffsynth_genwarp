from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - source-only environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SourceGeometryAugmentTest(unittest.TestCase):
    @staticmethod
    def _identity_sample(size: int = 41):
        y, x = torch.meshgrid(
            torch.arange(size),
            torch.arange(size),
            indexing="ij",
        )
        source = torch.stack((x, y, x + y)).float() / (2.0 * (size - 1))
        flow = torch.zeros(2, size, size)
        valid = torch.ones(1, size, size, dtype=torch.bool)
        return source, flow, valid, x, y

    def test_identity_and_probability_zero_are_exact_noops(self) -> None:
        from diffusion2raft.data import (
            SourceGeometryAugment,
            apply_source_homography,
            source_affine_homography,
        )

        source, flow, valid, _, _ = self._identity_sample()
        identity = source_affine_homography(source.shape[-2:])
        transformed = apply_source_homography(source, flow, valid, identity)
        torch.testing.assert_close(transformed[0], source, rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(transformed[1], flow, rtol=0.0, atol=0.0)
        self.assertTrue(torch.equal(transformed[2], valid))

        disabled = SourceGeometryAugment(
            probability=0.0,
            max_rotation_deg=180.0,
            scale=(0.8, 1.1),
            translation=(0.1, 0.1),
            perspective=0.05,
        )
        no_op = disabled(source, flow, valid)
        self.assertIs(no_op[0], source)
        self.assertIs(no_op[1], flow)
        self.assertIs(no_op[2], valid)

    def test_90_degree_rotation_updates_absolute_source_coordinates(self) -> None:
        from diffusion2raft.data import apply_source_homography, source_affine_homography
        from diffusion2raft.geometry import backward_flow_to_map

        size = 5
        source, flow, valid, x, y = self._identity_sample(size)
        homography = source_affine_homography((size, size), angle_deg=90.0)
        rotated_source, rotated_flow, rotated_valid = apply_source_homography(
            source,
            flow,
            valid,
            homography,
        )

        absolute_map = backward_flow_to_map(rotated_flow.unsqueeze(0)).squeeze(0)
        expected_map = torch.stack(((size - 1) - y, x)).float()
        torch.testing.assert_close(absolute_map, expected_map, rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(
            rotated_source,
            torch.rot90(source, k=-1, dims=(-2, -1)),
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertTrue(bool(rotated_valid.all()))

    def test_30_degree_coordinates_and_gt_warp_remain_consistent(self) -> None:
        from diffusion2raft.data import apply_source_homography, source_affine_homography
        from diffusion2raft.geometry import backward_flow_to_map, backward_warp

        size = 41
        center = (size - 1) / 2.0
        source, flow, valid, x, y = self._identity_sample(size)
        homography = source_affine_homography((size, size), angle_deg=30.0)
        rotated_source, rotated_flow, rotated_valid = apply_source_homography(
            source,
            flow,
            valid,
            homography,
        )
        absolute_map = backward_flow_to_map(rotated_flow.unsqueeze(0)).squeeze(0)

        # A point ten pixels to the right of center follows the explicit
        # old-source -> new-source image-coordinate rotation convention.
        expected = torch.tensor(
            (
                center + 10.0 * math.cos(math.radians(30.0)),
                center + 10.0 * math.sin(math.radians(30.0)),
            )
        )
        torch.testing.assert_close(
            absolute_map[:, int(center), int(center + 10)],
            expected,
            rtol=0.0,
            atol=2.0e-5,
        )
        self.assertTrue(bool(rotated_valid[:, int(center), int(center)]))
        self.assertFalse(bool(rotated_valid.all()))

        reconstructed = backward_warp(
            rotated_source.unsqueeze(0),
            rotated_flow.unsqueeze(0),
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        # Exclude a two-pixel interpolation halo at both old and transformed
        # source boundaries. On this affine RGB ramp, the remaining GT warp is
        # analytically exact apart from floating-point roundoff.
        interior = (
            rotated_valid
            & (x.unsqueeze(0) > 4)
            & (x.unsqueeze(0) < size - 5)
            & (y.unsqueeze(0) > 4)
            & (y.unsqueeze(0) < size - 5)
            & (absolute_map[0:1] > 2)
            & (absolute_map[0:1] < size - 3)
            & (absolute_map[1:2] > 2)
            & (absolute_map[1:2] < size - 3)
        )
        self.assertGreater(int(interior.sum()), 500)
        torch.testing.assert_close(
            reconstructed[:, interior[0]],
            source[:, interior[0]],
            rtol=0.0,
            atol=2.0e-5,
        )

    def test_random_light_perspective_is_finite_and_respects_input_valid(self) -> None:
        from diffusion2raft.data import SourceGeometryAugment

        source, flow, valid, _, _ = self._identity_sample()
        valid[:, 20, 20] = False
        augment = SourceGeometryAugment.from_config(
            {
                "probability": 1.0,
                "max_rotation_deg": 180.0,
                "scale": [0.85, 1.05],
                "translation": [0.04, 0.04],
                "perspective": 0.025,
            }
        )
        torch.manual_seed(17)
        augmented_source, augmented_flow, augmented_valid = augment(source, flow, valid)
        self.assertTrue(bool(torch.isfinite(augmented_source).all()))
        self.assertTrue(bool(torch.isfinite(augmented_flow).all()))
        self.assertFalse(bool(augmented_valid[:, 20, 20]))
        self.assertGreater(int(augmented_valid.sum()), 100)

    def test_dataset_augmentation_changes_source_but_not_target(self) -> None:
        from diffusion2raft.data import (
            DocumentFlowDataset,
            source_affine_homography,
            write_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            size = 8
            y, x = np.meshgrid(
                np.arange(size, dtype=np.uint8),
                np.arange(size, dtype=np.uint8),
                indexing="ij",
            )
            source_u8 = np.stack((x * 20, y * 20, (x + y) * 9), axis=-1)
            target_u8 = np.stack(
                (
                    np.full_like(x, 31),
                    x * 17,
                    np.full_like(x, 223),
                ),
                axis=-1,
            )
            Image.fromarray(source_u8).save(root / "source.png")
            Image.fromarray(target_u8).save(root / "target.png")
            np.save(root / "flow.npy", np.zeros((size, size, 2), dtype=np.float32))
            manifest = root / "manifest.jsonl"
            write_manifest(
                [
                    {
                        "id": "source-only",
                        "warped": "source.png",
                        "target": "target.png",
                        "flow": "flow.npy",
                        "flow_format": "displacement",
                    }
                ],
                manifest,
            )
            baseline = DocumentFlowDataset(manifest, (size, size))[0]
            augmented_dataset = DocumentFlowDataset(
                manifest,
                (size, size),
                source_geometry_augment={
                    "probability": 1.0,
                    "max_rotation_deg": 90.0,
                    "scale": [1.0, 1.0],
                    "translation": [0.0, 0.0],
                    "perspective": 0.0,
                },
            )
            fixed_90 = source_affine_homography((size, size), angle_deg=90.0)
            assert augmented_dataset.source_geometry_augment is not None
            with patch.object(
                augmented_dataset.source_geometry_augment,
                "sample_homography",
                return_value=fixed_90,
            ):
                augmented = augmented_dataset[0]

            torch.testing.assert_close(
                augmented["target"], baseline["target"], rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                augmented["guide"], baseline["guide"], rtol=0.0, atol=0.0
            )
            self.assertFalse(torch.equal(augmented["warped"], baseline["warped"]))
            self.assertFalse(torch.equal(augmented["flow"], baseline["flow"]))
            self.assertTrue(bool(augmented["valid"].all()))

    def test_training_loader_never_enables_augmentation_for_validation(self) -> None:
        from diffusion2raft import train

        geometry_config = {
            "probability": 0.7,
            "max_rotation_deg": 180.0,
            "scale": [0.85, 1.05],
            "translation": [0.04, 0.04],
            "perspective": 0.025,
        }
        config = {
            "data": {
                "work_size": [32, 32],
                "batch_size": 1,
                "num_workers": 0,
                "source_geometry_augment": geometry_config,
            }
        }
        with patch.object(train, "DocumentFlowDataset") as dataset_type, patch.object(
            train, "DataLoader"
        ):
            dataset_type.return_value = object()
            train._make_loader(
                config,
                "train.jsonl",
                stage="unified",
                training=True,
                rank=0,
                world_size=1,
            )
            self.assertEqual(
                dataset_type.call_args.kwargs["source_geometry_augment"],
                geometry_config,
            )
            train._make_loader(
                config,
                "val.jsonl",
                stage="unified",
                training=False,
                rank=0,
                world_size=1,
            )
            self.assertIsNone(
                dataset_type.call_args.kwargs["source_geometry_augment"]
            )

    def test_big_rotation_recipe_is_opt_in_and_v31_stays_disabled(self) -> None:
        from diffusion2raft.config import load_config

        root = Path(__file__).resolve().parents[1]
        v31 = load_config(root / "configs" / "unified.yaml")
        v32_data = load_config(
            root / "configs" / "unified_v3_2_bigrot_data.yaml"
        )
        self.assertNotIn("source_geometry_augment", v31["data"])
        self.assertEqual(
            v32_data["data"]["source_geometry_augment"],
            {
                "probability": 0.70,
                "max_rotation_deg": 180.0,
                "scale": [0.85, 1.05],
                "translation": [0.04, 0.04],
                "perspective": 0.025,
            },
        )


if __name__ == "__main__":
    unittest.main()
