from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "make_typical_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("make_typical_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


class TypicalComparisonTest(unittest.TestCase):
    @staticmethod
    def _save_image(path: Path, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (5, 4), color).save(path)

    def test_suffix_pairing_missing_cells_relative_links_and_json_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            first = root / "candidate one"
            second = root / "candidate-two"
            output = root / "report" / "comparison.html"
            source.mkdir()
            first.mkdir()
            second.mkdir()

            self._save_image(source / "a.jpg", (255, 0, 0))
            self._save_image(source / "b space.png", (0, 255, 0))
            self._save_image(first / "a_rectified.png", (0, 0, 255))
            self._save_image(first / "b space.jpg", (20, 30, 40))
            self._save_image(first / "extra.png", (50, 60, 70))
            self._save_image(second / "a.png", (80, 90, 100))

            report = COMPARISON.generate_comparison(
                source,
                [("Model <A>", first), ("Model B", second)],
                output,
                title="Typical & review",
                thumbnail_width=128,
            )

            self.assertEqual(report["pairing"]["source_count"], 2)
            self.assertEqual(report["pairing"]["total_pairings"], 3)
            self.assertEqual(report["pairing"]["complete_row_count"], 1)
            self.assertEqual(report["pairing"]["rows_with_any_candidate"], 2)
            self.assertEqual(report["candidates"]["Model <A>"]["matched_count"], 2)
            self.assertEqual(report["candidates"]["Model <A>"]["extra_keys"], ["extra"])
            self.assertEqual(report["candidates"]["Model B"]["missing_keys"], ["b space"])

            rows = {row["key"]: row for row in report["rows"]}
            self.assertTrue(rows["a"]["candidates"]["Model <A>"].endswith(
                "a_rectified.png"
            ))
            self.assertIsNone(rows["b space"]["candidates"]["Model B"])

            json_path = output.with_suffix(".json")
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["pairing"], report["pairing"])
            page = output.read_text(encoding="utf-8")
            self.assertIn("Typical &amp; review", page)
            self.assertIn("Model &lt;A&gt;", page)
            self.assertIn("缺失 / missing", page)
            self.assertIn("b%20space", page)
            self.assertNotIn(f'src="{source.resolve()}', page)
            self.assertIn("target=\"_blank\"", page)

    def test_duplicate_normalized_candidate_basename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            candidate = root / "candidate"
            source.mkdir()
            candidate.mkdir()
            self._save_image(source / "a.jpg", (1, 2, 3))
            self._save_image(candidate / "a.png", (4, 5, 6))
            self._save_image(candidate / "a_rectified.jpg", (7, 8, 9))

            with self.assertRaisesRegex(ValueError, "duplicate candidate basename 'a'"):
                COMPARISON.generate_comparison(
                    source,
                    [("candidate", candidate)],
                    root / "comparison.html",
                )


if __name__ == "__main__":
    unittest.main()
