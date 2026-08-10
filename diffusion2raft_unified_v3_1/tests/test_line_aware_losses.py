from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - source-only environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class LineAwareLossTest(unittest.TestCase):
    def test_structure_detector_finds_long_rules(self) -> None:
        from diffusion2raft.losses import target_structure_maps

        target = torch.ones(1, 3, 64, 64)
        target[:, :, 20, 6:58] = 0.0
        target[:, :, 6:58, 44] = 0.0
        maps = target_structure_maps(
            target,
            line_kernel=9,
            line_threshold=0.10,
            line_temperature=0.03,
        )
        horizontal_score = maps["horizontal"][:, :, 18:23, 10:36].mean()
        vertical_score = maps["vertical"][:, :, 10:36, 42:47].mean()
        self.assertGreater(float(horizontal_score), 0.02)
        self.assertGreater(float(vertical_score), 0.02)

    def test_straightness_penalizes_tangent_variation(self) -> None:
        from diffusion2raft.losses import line_straightness_loss

        target = torch.zeros(1, 2, 32, 48)
        valid = torch.ones(1, 1, 32, 48, dtype=torch.bool)
        horizontal = torch.ones(1, 1, 32, 48)
        vertical = torch.zeros_like(horizontal)
        flat = line_straightness_loss(
            target,
            target,
            valid,
            horizontal,
            vertical,
            robust=False,
        )
        prediction = target.clone()
        x = torch.linspace(0.0, 4.0 * torch.pi, 48)
        prediction[:, 1] = 2.0 * torch.sin(x).view(1, 1, 48)
        wavy = line_straightness_loss(
            prediction,
            target,
            valid,
            horizontal,
            vertical,
            robust=False,
        )
        self.assertEqual(float(flat), 0.0)
        self.assertGreater(float(wavy), 0.1)

    @staticmethod
    def _rectification_metrics(
        target_image: "torch.Tensor",
        final_flow: "torch.Tensor",
        prior_flow: "torch.Tensor",
        *,
        suppress_structures: bool = False,
    ) -> dict[str, "torch.Tensor"]:
        from diffusion2raft.losses import RectificationLoss

        config = {
            "flow": 1.0,
            "prior_flow": 0.0,
            "reconstruction": 0.0,
            "gradient": 0.0,
            "structure_flow": 0.0,
            "line_reconstruction": 0.0,
            "flow_gradient": 0.0,
            "curvature": 0.0,
            "line_straightness": 0.0,
            "bending": 0.0,
            "anti_fold": 0.0,
            "residual": 0.0,
            "residual_flow": 0.0,
            "qwen_match": 0.0,
            "confidence": 0.0,
            "structure_line_kernel": 9,
            "structure_line_threshold": 0.10,
            "structure_line_temperature": 0.03,
        }
        if suppress_structures:
            config.update(
                {
                    "structure_edge_threshold": 10.0,
                    "structure_edge_temperature": 0.01,
                    "structure_line_threshold": 10.0,
                    "structure_line_temperature": 0.01,
                }
            )
        target_flow = torch.zeros_like(final_flow)
        return RectificationLoss(config)(
            {
                "stage": "prior",
                "flows": [final_flow],
                "final_flow": final_flow,
                "prior_flow": prior_flow,
                "residuals": [],
            },
            {
                "warped": target_image.clone(),
                "target": target_image,
                "flow": target_flow,
                "valid": torch.ones_like(target_flow[:, :1], dtype=torch.bool),
            },
        )

    def test_prior_line_metrics_prove_final_improvement(self) -> None:
        height, width = 32, 48
        target = torch.ones(1, 3, height, width)
        target[:, :, 16, 4:44] = 0.0
        final_flow = torch.zeros(
            1, 2, height, width, requires_grad=True
        )
        prior_flow = torch.zeros_like(final_flow.detach())
        x = torch.linspace(0.0, 4.0 * torch.pi, width)
        prior_flow[:, 1] = 2.0 * torch.sin(x).view(1, 1, width)

        metrics = self._rectification_metrics(target, final_flow, prior_flow)
        self.assertGreater(float(metrics["prior_line_epe"]), 0.0)
        self.assertGreater(float(metrics["line_epe_gain"]), 0.0)
        self.assertGreater(
            float(metrics["prior_line_straightness_error"]), 0.0
        )
        self.assertGreater(float(metrics["line_straightness_gain"]), 0.0)
        self.assertTrue(
            torch.allclose(
                metrics["line_epe_gain"],
                metrics["prior_line_epe"] - metrics["line_epe"],
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["line_straightness_gain"],
                metrics["prior_line_straightness_error"]
                - metrics["line_straightness_error"],
            )
        )
        for key in (
            "prior_line_epe",
            "line_epe_gain",
            "prior_line_straightness_error",
            "line_straightness_gain",
        ):
            self.assertFalse(metrics[key].requires_grad)
        metrics["total"].backward()
        self.assertIsNotNone(final_flow.grad)

    def test_empty_line_maps_are_finite_and_fail_closed(self) -> None:
        height, width = 16, 20
        target = torch.ones(1, 3, height, width)
        final_flow = torch.zeros(
            1, 2, height, width, requires_grad=True
        )
        prior_flow = torch.ones_like(final_flow.detach())
        metrics = self._rectification_metrics(
            target,
            final_flow,
            prior_flow,
            suppress_structures=True,
        )
        self.assertTrue(torch.isfinite(metrics["prior_line_epe"]))
        self.assertTrue(
            torch.allclose(metrics["prior_line_epe"], metrics["prior_epe"])
        )
        self.assertTrue(torch.allclose(metrics["line_epe"], metrics["epe"]))
        self.assertEqual(float(metrics["prior_line_straightness_error"]), 0.0)
        self.assertEqual(float(metrics["line_straightness_error"]), 0.0)
        self.assertEqual(float(metrics["line_straightness_gain"]), 0.0)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class IndependentFlowCanvasTest(unittest.TestCase):
    @staticmethod
    def _write_case(root: Path, *, declare_source: bool) -> Path:
        from diffusion2raft.data import write_manifest

        height = width = 32
        y, x = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )
        image = np.stack(
            (
                x / (width - 1),
                y / (height - 1),
                (x + y) / (height + width - 2),
            ),
            axis=-1,
        )
        image_u8 = np.uint8(np.round(image * 255.0))
        Image.fromarray(image_u8).save(root / "warped.png")
        Image.fromarray(image_u8).save(root / "target.png")
        np.save(root / "flow.npy", np.zeros((2, 64, 64), dtype=np.float32))
        record: dict[str, object] = {
            "id": "identity",
            "warped": "warped.png",
            "target": "target.png",
            "flow": "flow.npy",
            "flow_format": "displacement",
        }
        if declare_source:
            record["flow_source_size"] = [64, 64]
        manifest = root / "manifest.jsonl"
        write_manifest([record], manifest)
        return manifest

    def test_explicit_64_flow_to_32_images_stays_identity(self) -> None:
        from diffusion2raft.data import DocumentFlowDataset

        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_case(Path(directory), declare_source=True)
            sample = DocumentFlowDataset(manifest, (32, 32))[0]
            self.assertLess(float(sample["flow"].abs().max()), 1e-5)
            self.assertTrue(bool(sample["valid"].all()))
            self.assertEqual(tuple(sample["flow_source_size"].tolist()), (64, 64))

    def test_ambiguous_independent_grid_is_rejected(self) -> None:
        from diffusion2raft.data import DocumentFlowDataset

        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_case(Path(directory), declare_source=False)
            dataset = DocumentFlowDataset(manifest, (32, 32))
            with self.assertRaisesRegex(ValueError, "flow_source_size"):
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
