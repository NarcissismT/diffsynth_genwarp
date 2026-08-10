from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cp_docflow.aggregate_seeds import aggregate_seed_reports
from cp_docflow.evaluate_full import evaluate_full


class ReproducibilityToolsTest(unittest.TestCase):
    @staticmethod
    def _evaluation(seed: int, epe: float) -> dict[str, object]:
        return {
            "schema": "docgrid_flow.full_evaluation.v2",
            "manifest_sha256": "a" * 64,
            "training_stage": "full_page",
            "training_seed": seed,
            "checkpoint": f"/checkpoint/seed-{seed}/best.pt",
            "checkpoint_sha256": str(seed).zfill(64),
            "aggregate": {
                "epe": epe,
                "fold_rate": 0.001,
                "final_win_rate": 0.7,
            },
        }

    def test_three_distinct_seed_reports_create_immutable_stability_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths: list[Path] = []
            for seed, epe in ((1337, 5.0), (2027, 5.1), (3407, 4.9)):
                path = root / f"seed-{seed}.json"
                path.write_text(
                    json.dumps(self._evaluation(seed, epe)), encoding="utf-8"
                )
                paths.append(path)
            output = root / "multi_seed_evidence.json"
            result = aggregate_seed_reports(paths, output)
            self.assertEqual(result["seeds"], [1337, 2027, 3407])
            self.assertTrue(result["multi_seed_stable"])
            self.assertAlmostEqual(result["metrics"]["epe"]["mean"], 5.0)
            with self.assertRaises(FileExistsError):
                aggregate_seed_reports(paths, output)

    def test_duplicate_training_seeds_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths: list[Path] = []
            for index in range(3):
                path = root / f"evaluation-{index}.json"
                path.write_text(
                    json.dumps(self._evaluation(1337, 5.0)), encoding="utf-8"
                )
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "distinct checkpoint"):
                aggregate_seed_reports(paths, root / "evidence.json")

    def test_gate_evaluation_rejects_runtime_ablation_before_loading_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot use runtime ablation"):
            evaluate_full(
                "/does/not/exist.pt",
                "/does/not/exist.jsonl",
                "/does/not/matter",
                allowed_label_provenance={"renderer_gt"},
                gate=True,
                runtime_overrides={"fm_steps": 1},
            )


if __name__ == "__main__":
    unittest.main()
