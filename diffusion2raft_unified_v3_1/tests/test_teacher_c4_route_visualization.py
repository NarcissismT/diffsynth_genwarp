from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from diffusion2raft.teacher_c4_route_visualization import (
    RouteRender,
    _capacity_drift,
    project_residual_to_bounded_feature_grid,
    render_comparison_figure,
)


class TeacherC4RouteVisualizationTest(unittest.TestCase):
    def test_feature_grid_projection_enforces_full_resolution_cap(self) -> None:
        residual = torch.zeros(1, 2, 64, 64)
        residual[:, 0] = 100.0
        residual[:, 1] = -80.0

        low, restored, limits = project_residual_to_bounded_feature_grid(
            residual,
            feature_stride=8,
            max_residual_px=40.0,
        )

        self.assertEqual(tuple(low.shape), (1, 2, 8, 8))
        self.assertEqual(tuple(restored.shape), tuple(residual.shape))
        self.assertAlmostEqual(limits[0], 40.0 * 7.0 / 63.0, places=7)
        self.assertAlmostEqual(limits[1], 40.0 * 7.0 / 63.0, places=7)
        self.assertLessEqual(float(restored[:, 0].abs().max()), 40.0001)
        self.assertLessEqual(float(restored[:, 1].abs().max()), 40.0001)
        self.assertAlmostEqual(float(restored[:, 0].mean()), 40.0, places=4)
        self.assertAlmostEqual(float(restored[:, 1].mean()), -40.0, places=4)

    @staticmethod
    def _route(
        role: str,
        turn: int,
        *,
        teacher_epe: float,
        trainable: float,
        overflow: float,
        axis_max: tuple[float, float],
    ) -> RouteRender:
        image = Image.new("RGB", (64, 64), "white")
        return RouteRender(
            role=role,
            quarter_turn_deg=turn,
            residual_rotation_deg=0.0,
            candidate=image,
            teacher=image,
            capped_oracle=image,
            reference=image,
            metrics={
                "teacher_epe_px": teacher_epe,
                "trainable_coverage": trainable,
                "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": overflow,
                "oracle_residual_axis_absmax_px": {
                    "x": axis_max[0],
                    "y": axis_max[1],
                },
            },
        )

    def test_renderer_writes_self_contained_png(self) -> None:
        wrong = self._route(
            "wrong",
            0,
            teacher_epe=250.3,
            trainable=0.015,
            overflow=0.984,
            axis_max=(383.7, 358.7),
        )
        correct = self._route(
            "correct",
            90,
            teacher_epe=9.87,
            trainable=0.994,
            overflow=0.0,
            axis_max=(17.3, 7.95),
        )
        figure = render_comparison_figure(
            wrong,
            correct,
            sample_id="Pers_NoAug_0010947",
            injected_rotation_deg=-44.89151,
            max_residual_px=40.0,
        )
        self.assertEqual(figure.size, (2140, 1480))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            figure.save(path, format="PNG")
            header = path.read_bytes()[:24]
            self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 10_000)

    def test_small_gpu_reduction_drift_is_reported_but_tolerated(self) -> None:
        expected = {
            "teacher_epe_px": 9.871,
            "oracle_solver_coverage": 0.9938,
            "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": 0.0,
            "trainable_coverage": 0.9938,
            "stride_trainable_oracle_reconstruction_epe_px": 0.0337,
            "oracle_residual_axis_absmax_px": {"x": 17.29, "y": 7.95},
        }
        actual = {
            **expected,
            "teacher_epe_px": 9.873,
            "oracle_residual_axis_absmax_px": {"x": 17.31, "y": 7.94},
        }
        drift = _capacity_drift(actual, expected)
        self.assertTrue(drift["teacher_epe_px"]["within_tolerance"])
        self.assertAlmostEqual(drift["teacher_epe_px"]["difference"], 0.002)

        divergent = {**actual, "teacher_epe_px": 20.0}
        with self.assertRaisesRegex(RuntimeError, "materially diverged"):
            _capacity_drift(divergent, expected)


if __name__ == "__main__":
    unittest.main()
