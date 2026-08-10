from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOBS = PROJECT_ROOT / "slurm" / "docgrid_v2" / "container_jobs"


def _expand_job(name: str, environment: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    command = (
        "srun(){ env; printf '__DOCGRID_ARG__=%s\\n' \"$@\"; }; "
        "export -f srun; "
        f"source {JOBS / name}"
    )
    merged = os.environ.copy()
    merged.update(environment)
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=PROJECT_ROOT,
        env=merged,
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[str, str] = {}
    arguments: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("__DOCGRID_ARG__="):
            arguments.append(line.split("=", 1)[1])
        elif "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values, arguments


class ContainerJobContractTest(unittest.TestCase):
    def test_launcher_runs_worker_directly_when_portal_already_entered_container(self) -> None:
        python = os.environ.get("DOCGRID_TEST_PYTHON") or os.sys.executable
        command = (
            f"source {JOBS / 'common_container.sh'}; "
            "docgrid_run_container 1 'printf \"__LOCAL_WORKER__\\n\"'"
        )
        environment = os.environ.copy()
        environment.update(
            {
                "SLURM_JOB_ID": "fixture-job",
                "DOCGRID_PYTHON": python,
                "DOCGRID_CONTAINER_WORKDIR": str(PROJECT_ROOT.parent),
            }
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("without nested srun", result.stdout)
        self.assertIn("__LOCAL_WORKER__", result.stdout)

    def test_full_page_render_is_isolated_and_passes_all_shape_controls(self) -> None:
        values, arguments = _expand_job(
            "00_render_full_page_array_a800.sh",
            {
                "SLURM_ARRAY_TASK_ID": "7",
                "DOCGRID_DATA_ROOT": "/immutable/data",
                "DOCGRID_FULL_PAGE_INPUT_CSV": "/inputs/full-page.csv",
            },
        )
        self.assertEqual(values["DOCGRID_SHARD_INDEX"], "7")
        self.assertEqual(values["DOCGRID_RENDER_HEIGHT"], "1024")
        self.assertEqual(values["DOCGRID_RENDER_WIDTH"], "768")
        self.assertEqual(values["DOCGRID_ANALYTIC_VARIANTS"], "1")
        self.assertEqual(values["DOCGRID_ANALYTIC_MAX_DOCUMENTS"], "20000")
        self.assertEqual(values["DOCGRID_ANALYTIC_INPUT_CSV"], "/inputs/full-page.csv")
        self.assertEqual(
            values["DOCGRID_ANALYTIC_SHARD_ROOT"],
            "/immutable/data/analytic_full_page_shards_seed1337_1024x768",
        )
        container_env = next(
            value for value in arguments if value.startswith("--container-env=")
        )
        self.assertIn("DOCGRID_ANALYTIC_MAX_DOCUMENTS", container_env)
        self.assertIn("DOCGRID_RENDER_HEIGHT", container_env)
        self.assertIn("DOCGRID_RENDER_WIDTH", container_env)
        self.assertIn("DOCGRID_FULL_PAGE_ASPECT_TOLERANCE", container_env)
        worker = arguments[-1]
        self.assertIn("render_full_page_gt_shard.sbatch", worker)

    def test_gate5_uses_full_page_validation_manifest(self) -> None:
        values, _ = _expand_job(
            "06_evaluate_gate_a800.sh",
            {"DOCGRID_GATE": "gate5", "DOCGRID_DATA_ROOT": "/immutable/data"},
        )
        self.assertEqual(
            values["DOCGRID_EVAL_MANIFEST"],
            "/immutable/data/full_page/manifests/val.jsonl",
        )

    def test_gate1_keeps_512_validation_manifest(self) -> None:
        values, _ = _expand_job(
            "06_evaluate_gate_a800.sh",
            {"DOCGRID_GATE": "gate1", "DOCGRID_DATA_ROOT": "/immutable/data"},
        )
        self.assertEqual(
            values["DOCGRID_EVAL_MANIFEST"],
            "/immutable/data/analytic_merged_seed1337_512/manifests/val.jsonl",
        )

    def test_frozen_prior_baseline_is_bound_before_stage0_audit(self) -> None:
        values, arguments = _expand_job(
            "00_evaluate_prior_baseline_a800.sh",
            {"DOCGRID_DATA_ROOT": "/immutable/data"},
        )
        self.assertEqual(
            values["DOCGRID_EVAL_MANIFEST"],
            "/immutable/data/analytic_merged_seed1337_512/manifests/val.jsonl",
        )
        self.assertEqual(
            values["DOCGRID_BASELINE_METRICS"],
            "/immutable/data/runs/baselines/frozen_supervised_prior/metrics.json",
        )
        self.assertIn("evaluate_frozen_prior_baseline.sbatch", arguments[-1])

        audit_values, _ = _expand_job(
            "00_audit_cpu.sh", {"DOCGRID_DATA_ROOT": "/immutable/data"}
        )
        self.assertEqual(
            audit_values["DOCGRID_BASELINE_METRICS"],
            values["DOCGRID_BASELINE_METRICS"],
        )
        self.assertEqual(
            audit_values["DOCGRID_BASELINE_CHECKPOINT"],
            values["DOCGRID_BASELINE_CHECKPOINT"],
        )

    def test_gate5_deterministic_baseline_runs_at_full_page_work_size(self) -> None:
        values, arguments = _expand_job(
            "05_evaluate_deterministic_baseline_a800.sh",
            {"DOCGRID_DATA_ROOT": "/immutable/data", "DOCGRID_SEED": "2027"},
        )
        self.assertEqual(values["DOCGRID_EVAL_INPUT_HEIGHT"], "1024")
        self.assertEqual(values["DOCGRID_EVAL_INPUT_WIDTH"], "768")
        self.assertEqual(values["DOCGRID_EVAL_OUTPUT_HEIGHT"], "1024")
        self.assertEqual(values["DOCGRID_EVAL_OUTPUT_WIDTH"], "768")
        self.assertEqual(
            values["DOCGRID_CHECKPOINT"],
            "/immutable/data/runs/stage2_warr/seed-2027/best.pt",
        )
        self.assertIn(
            "evaluate_deterministic_full_page_baseline.sbatch", arguments[-1]
        )

    def test_gate_receipts_choose_the_canonical_baseline_automatically(self) -> None:
        common = {
            "DOCGRID_DATA_ROOT": "/immutable/data",
            "DOCGRID_GATE_REVIEWER": "unit-test",
            "DOCGRID_GATE_REVIEW_NOTE": "reviewed",
            "DOCGRID_GATE_DECISION": "failed",
        }
        gate1, _ = _expand_job(
            "06_write_gate_receipt_cpu.sh", {**common, "DOCGRID_GATE": "gate1"}
        )
        self.assertEqual(
            gate1["DOCGRID_BASELINE_EVALUATION"],
            "/immutable/data/runs/baselines/frozen_supervised_prior/metrics.json",
        )
        gate5, _ = _expand_job(
            "06_write_gate_receipt_cpu.sh", {**common, "DOCGRID_GATE": "gate5"}
        )
        self.assertEqual(
            gate5["DOCGRID_BASELINE_EVALUATION"],
            "/immutable/data/runs/baselines/full_page_deterministic/seed-1337/metrics.json",
        )

    def test_ocr_scoring_defaults_to_the_exact_gate5_image_export(self) -> None:
        values, arguments = _expand_job(
            "07_score_ocr_cpu.sh",
            {
                "DOCGRID_DATA_ROOT": "/immutable/data",
                "DOCGRID_SEED": "2027",
                "DOCGRID_OCR_TRANSCRIPTS": "/ocr/transcripts.jsonl",
                "DOCGRID_OCR_ENGINE": "PaddleOCR",
                "DOCGRID_OCR_ENGINE_VERSION": "3.0.0",
            },
        )
        gate5_eval = "/immutable/data/runs/stage5_full_page/seed-2027/gate5_eval"
        self.assertEqual(values["DOCGRID_GEOMETRY_EVALUATION"], f"{gate5_eval}/metrics.json")
        self.assertEqual(values["DOCGRID_GEOMETRY_PER_SAMPLE"], f"{gate5_eval}/per_sample.csv")
        self.assertEqual(values["DOCGRID_OCR_IMAGE_MANIFEST"], f"{gate5_eval}/ocr_images.jsonl")
        self.assertEqual(values["DOCGRID_OCR_ENGINE_VERSION"], "3.0.0")
        self.assertIn("score_ocr.sbatch", arguments[-1])


if __name__ == "__main__":
    unittest.main()
