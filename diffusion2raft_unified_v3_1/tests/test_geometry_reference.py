from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class GeometryReferenceTest(unittest.TestCase):
    def test_reference_validator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(root / "scripts" / "validate_geometry_numpy.py")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("geometry validation passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()

