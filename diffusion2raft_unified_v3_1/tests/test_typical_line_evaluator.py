from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_typical_lines.py"
SPEC = importlib.util.spec_from_file_location("evaluate_typical_lines", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def _line_image(angle_deg: float, size: int = 256) -> np.ndarray:
    image = np.full((size, size), 255, dtype=np.uint8)
    angle = math.radians(angle_deg)
    half_length = 92.0
    center = size / 2.0
    dx = half_length * math.cos(angle)
    dy = half_length * math.sin(angle)
    cv2.line(
        image,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        color=0,
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    return image


class TypicalLineEvaluatorTest(unittest.TestCase):
    def test_exact_segment_metrics_include_equal_and_length_weighted_forms(self) -> None:
        angle = math.radians(10.0)
        segments = np.asarray(
            [
                [0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0 * math.cos(angle), 10.0 * math.sin(angle)],
                [0.0, 0.0, 0.0, 20.0],
            ],
            dtype=np.float64,
        )
        metrics = evaluator.metrics_from_segments(segments, axis_threshold_deg=5.0)
        self.assertEqual(metrics["line_count"], 3)
        self.assertAlmostEqual(metrics["orientation_error_deg"], 10.0 / 3.0)
        self.assertAlmostEqual(
            metrics["orientation_error_deg_length_weighted"], 2.5
        )
        self.assertAlmostEqual(metrics["axis_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(
            metrics["axis_fraction_length_weighted"], 0.75
        )

    def test_mask_coverage_filters_segments_by_their_sampled_support(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[:50] = True
        segments = np.asarray(
            [[10.0, 20.0, 90.0, 20.0], [10.0, 80.0, 90.0, 80.0]],
            dtype=np.float64,
        )
        coverage = evaluator._line_mask_coverage(segments, mask)
        self.assertEqual(tuple(coverage.shape), (2,))
        self.assertEqual(float(coverage[0]), 1.0)
        self.assertEqual(float(coverage[1]), 0.0)

    def test_multiple_candidates_suffix_pairing_masks_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            flat = root / "flat"
            tilted = root / "tilted"
            masks = root / "masks"
            for path in (images, flat, tilted, masks):
                path.mkdir()

            blank = np.full((256, 256), 255, dtype=np.uint8)
            cv2.imwrite(str(images / "page_one.jpg"), blank)
            cv2.imwrite(str(images / "page_two.png"), blank)
            cv2.imwrite(str(flat / "page_one_rectified.png"), _line_image(0.0))
            cv2.imwrite(str(flat / "page_two-rectified.jpg"), _line_image(0.0))
            cv2.imwrite(str(tilted / "page_one.jpg"), _line_image(15.0))
            cv2.imwrite(str(tilted / "page_two.png"), _line_image(15.0))
            full_mask = np.full((128, 128), 255, dtype=np.uint8)
            cv2.imwrite(str(masks / "page_one_valid.png"), full_mask)
            cv2.imwrite(str(masks / "page_one_evaluation_valid.png"), full_mask)
            cv2.imwrite(str(masks / "page_two_eval_valid.png"), full_mask)

            report = evaluator.evaluate_dataset(
                images,
                {"flat": flat, "tilted": tilted},
                valid_masks={"tilted": masks},
                max_dimension=256,
                min_length_fraction=0.03,
            )
            self.assertIn("no-reference structural proxy", report["proxy_notice"])
            flat_result = report["candidates"]["flat"]
            tilted_result = report["candidates"]["tilted"]
            self.assertEqual(flat_result["summary"]["evaluated_images"], 2)
            self.assertEqual(tilted_result["summary"]["evaluated_images"], 2)
            self.assertTrue(
                flat_result["per_image"][0]["candidate_image"].endswith(
                    "page_one_rectified.png"
                )
            )
            self.assertIsNotNone(
                tilted_result["per_image"][0]["evaluation_valid_mask"]
            )
            self.assertTrue(
                tilted_result["per_image"][0]["evaluation_valid_mask"].endswith(
                    "page_one_evaluation_valid.png"
                )
            )
            flat_error = flat_result["summary"][
                "image_mean_orientation_error_deg_length_weighted"
            ]
            tilted_error = tilted_result["summary"][
                "image_mean_orientation_error_deg_length_weighted"
            ]
            self.assertLess(flat_error, 2.0)
            self.assertGreater(tilted_error, 10.0)
            self.assertLess(flat_error, tilted_error)

            output = evaluator._atomic_write_json(report, root / "report.json")
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema_version"], 1)
            self.assertEqual(set(loaded["candidates"]), {"flat", "tilted"})

    def test_missing_requested_mask_is_reported_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            candidate = root / "candidate"
            masks = root / "masks"
            for path in (images, candidate, masks):
                path.mkdir()
            image = _line_image(0.0)
            for name in ("one.png", "two.png"):
                cv2.imwrite(str(images / name), image)
                cv2.imwrite(str(candidate / name), image)
            cv2.imwrite(str(masks / "one_valid.png"), np.full_like(image, 255))

            result = evaluator.evaluate_dataset(
                images,
                {"candidate": candidate},
                valid_masks={"candidate": masks},
                max_dimension=256,
            )["candidates"]["candidate"]
            self.assertEqual(result["summary"]["missing_masks"], 1)
            statuses = {row["basename"]: row["status"] for row in result["per_image"]}
            self.assertEqual(statuses["one"], "ok")
            self.assertEqual(statuses["two"], "missing_mask")


if __name__ == "__main__":
    unittest.main()
