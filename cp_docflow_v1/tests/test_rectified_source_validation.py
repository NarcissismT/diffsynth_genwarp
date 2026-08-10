from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cp_docflow.validate_rectified_sources import validate_rectified_sources


def _fixture_csv(root: Path, size: tuple[int, int], count: int = 4) -> Path:
    height, width = size
    csv_path = root / "pages.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "category"))
        writer.writeheader()
        for index in range(count):
            path = root / f"page-{index}.png"
            Image.fromarray(np.full((height, width, 3), 255, dtype=np.uint8)).save(path)
            writer.writerow({"image": str(path), "category": "fixture"})
    return csv_path


class RectifiedSourceValidationTest(unittest.TestCase):
    def test_accepts_high_resolution_target_aspect_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = _fixture_csv(Path(directory), (128, 96))
            result = validate_rectified_sources(
                csv_path,
                target_size=(64, 48),
                minimum_source_size=(64, 48),
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["validated_documents"], 4)

    def test_rejects_square_low_resolution_pages_for_portrait_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = _fixture_csv(Path(directory), (32, 32))
            with self.assertRaisesRegex(ValueError, "resolution/aspect stretching"):
                validate_rectified_sources(
                    csv_path,
                    target_size=(64, 48),
                    minimum_source_size=(64, 48),
                )


if __name__ == "__main__":
    unittest.main()
