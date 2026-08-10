from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_ROOT = PROJECT_ROOT.parent
PIPELINE_ROOT = TOP_ROOT / "slurm" / "v33_pipeline"
COMMON = PIPELINE_ROOT / "common.sh"
CANDIDATE = PIPELINE_ROOT / "00_teacher_candidate_399999_diagnostic_1gpu.sh"
ORACLE_C4 = PIPELINE_ROOT / "diagnostic_teacher_399999_oracle_c4_1gpu.sh"
CANONICAL_C4 = (
    PIPELINE_ROOT
    / "diagnostic_teacher_399999_oracle_c4_canonical_frame_v2_1gpu.sh"
)
SOLVER_SWEEP_C4 = (
    PIPELINE_ROOT
    / "diagnostic_teacher_399999_oracle_c4_canonical_solver_sweep_v3_1gpu.sh"
)
FULL_GEOMETRY_C4 = (
    PIPELINE_ROOT
    / "diagnostic_teacher_399999_oracle_c4_canonical_full_geometry_v4_1gpu.sh"
)
FULL_GEOMETRY_GRID_C4 = (
    PIPELINE_ROOT
    / "diagnostic_teacher_399999_oracle_c4_canonical_full_geometry_capacity_grid_v5_1gpu.sh"
)
BEST_OF_C4 = (
    PIPELINE_ROOT
    / "diagnostic_teacher_399999_oracle_c4_canonical_full_geometry_best_of_c4_v6_1gpu.sh"
)
STRUCTURAL_SAFETY_C4 = (
    PIPELINE_ROOT
    / "diagnostic_teacher_399999_best_of_c4_structural_safety_v7_1gpu.sh"
)
VISUALIZE_C4 = (
    PIPELINE_ROOT / "visualize_teacher_399999_c4_route_failure_1gpu.sh"
)
CAPACITY = PIPELINE_ROOT / "01_teacher_capacity_1gpu.sh"
SMOKE = PIPELINE_ROOT / "02_teacher_smoke_8gpu.sh"
TRAIN = PIPELINE_ROOT / "03_teacher_train_8gpu.sh"
VERIFIER_PATH = PROJECT_ROOT / "scripts" / "verify_v33_formal_smoke_report.py"
CAPACITY_EVALUATOR_PATH = (
    PROJECT_ROOT / "scripts" / "evaluate_teacher_capacity_report.py"
)
CANONICAL_C4_CLI = (
    PROJECT_ROOT
    / "scripts"
    / "diagnose_teacher_quarter_turn_oracle_canonical_frame.py"
)
SOLVER_SWEEP_C4_CLI = (
    PROJECT_ROOT
    / "scripts"
    / "diagnose_teacher_quarter_turn_oracle_solver_sweep.py"
)
FULL_GEOMETRY_C4_CLI = (
    PROJECT_ROOT
    / "scripts"
    / "diagnose_teacher_full_geometry_oracle_canonical.py"
)
FULL_GEOMETRY_GRID_C4_CLI = (
    PROJECT_ROOT
    / "scripts"
    / "diagnose_teacher_full_geometry_oracle_capacity_grid.py"
)
BEST_OF_C4_CLI = (
    PROJECT_ROOT
    / "scripts"
    / "diagnose_teacher_full_geometry_oracle_best_of_c4.py"
)
STRUCTURAL_SAFETY_C4_CLI = (
    PROJECT_ROOT
    / "scripts"
    / "diagnose_teacher_best_of_c4_structural_safety.py"
)
VISUALIZE_C4_CLI = (
    PROJECT_ROOT / "scripts" / "visualize_teacher_c4_route_failure.py"
)
OVERVIEW_RENDERER = PROJECT_ROOT / "scripts" / "render_v33_architecture_overview.py"
OVERVIEW_ROOT = PROJECT_ROOT / "docs" / "figures"

SPEC = importlib.util.spec_from_file_location("verify_v33_formal_smoke_report", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)

EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "evaluate_teacher_capacity_report", CAPACITY_EVALUATOR_PATH
)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
CAPACITY_EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(CAPACITY_EVALUATOR)


SEED_SHA = "7f5f743de8994907b916182fd3d1c1e81c0015b5b53fb8bf5038ff4c2ad17fe5"
TEACHER_SHA = "3d079e19445168169144f2af741362f673289b6510df4a4c1af348449ae045b9"
CANDIDATE_SHA = "e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5"


class V33SlurmPipelineTest(unittest.TestCase):
    def test_job_bodies_are_foreground_fixed_and_syntax_valid(self) -> None:
        scripts = (
            COMMON,
            CANDIDATE,
            ORACLE_C4,
            CANONICAL_C4,
            SOLVER_SWEEP_C4,
            FULL_GEOMETRY_C4,
            FULL_GEOMETRY_GRID_C4,
            BEST_OF_C4,
            STRUCTURAL_SAFETY_C4,
            VISUALIZE_C4,
            CAPACITY,
            SMOKE,
            TRAIN,
        )
        for script in scripts:
            self.assertTrue(script.is_file(), script)
            subprocess.run(["bash", "-n", str(script)], check=True)
        common = COMMON.read_text(encoding="utf-8")
        candidate = CANDIDATE.read_text(encoding="utf-8")
        oracle_c4 = ORACLE_C4.read_text(encoding="utf-8")
        canonical_c4 = CANONICAL_C4.read_text(encoding="utf-8")
        canonical_cli = CANONICAL_C4_CLI.read_text(encoding="utf-8")
        solver_sweep_c4 = SOLVER_SWEEP_C4.read_text(encoding="utf-8")
        solver_sweep_cli = SOLVER_SWEEP_C4_CLI.read_text(encoding="utf-8")
        full_geometry_c4 = FULL_GEOMETRY_C4.read_text(encoding="utf-8")
        full_geometry_cli = FULL_GEOMETRY_C4_CLI.read_text(encoding="utf-8")
        full_geometry_grid_c4 = FULL_GEOMETRY_GRID_C4.read_text(encoding="utf-8")
        full_geometry_grid_cli = FULL_GEOMETRY_GRID_C4_CLI.read_text(
            encoding="utf-8"
        )
        best_of_c4 = BEST_OF_C4.read_text(encoding="utf-8")
        best_of_c4_cli = BEST_OF_C4_CLI.read_text(encoding="utf-8")
        structural_safety_c4 = STRUCTURAL_SAFETY_C4.read_text(encoding="utf-8")
        structural_safety_cli = STRUCTURAL_SAFETY_C4_CLI.read_text(
            encoding="utf-8"
        )
        visualize_c4 = VISUALIZE_C4.read_text(encoding="utf-8")
        visualize_c4_cli = VISUALIZE_C4_CLI.read_text(encoding="utf-8")
        capacity = CAPACITY.read_text(encoding="utf-8")
        smoke = SMOKE.read_text(encoding="utf-8")
        train = TRAIN.read_text(encoding="utf-8")

        self.assertIn(f'readonly D2R_EXPECTED_SEED_SHA256="{SEED_SHA}"', common)
        self.assertIn(f'readonly D2R_EXPECTED_TEACHER_SHA256="{TEACHER_SHA}"', common)
        self.assertIn('readonly D2R_SEED_CHECKPOINT="runs/d2r_v3_1/unified/epoch_0020.pt"', common)
        self.assertIn('readonly D2R_CAPACITY_POINTER="${D2R_CAPACITY_DIR}/approved.json"', common)
        self.assertIn('readonly D2R_PYTHON="/usr/bin/python"', common)

        joined = "\n".join(
            (
                candidate,
                oracle_c4,
                canonical_c4,
                solver_sweep_c4,
                full_geometry_c4,
                full_geometry_grid_c4,
                best_of_c4,
                structural_safety_c4,
                visualize_c4,
                capacity,
                smoke,
                train,
            )
        )
        for forbidden in (
            "nohup",
            "setsid",
            "runs/d2r_v3_1/unified/latest.pt",
            "_background.sh",
            "ALLOW_PARTIAL_SMOKE=1",
        ):
            self.assertNotIn(forbidden, joined)
        self.assertIsNone(re.search(r"(?m)(?<!&)&\s*(?:#.*)?$", joined))

        self.assertNotIn("export CUDA_VISIBLE_DEVICES=", capacity)
        self.assertIn("d2r_require_visible_gpus 1", capacity)
        self.assertIn("d2r_require_visible_gpus 1", candidate)
        self.assertIn("399999_raft_unwarp.pt", candidate)
        self.assertIn(f'readonly D2R_CANDIDATE_SHA256="{CANDIDATE_SHA}"', candidate)
        self.assertIn("scripts/preflight_teacher_capacity.py", candidate)
        self.assertIn("--sample-count 300", candidate)
        self.assertIn("--full-geometry-per-sample 1", candidate)
        self.assertIn("scripts/evaluate_teacher_capacity_report.py", candidate)
        self.assertIn("--require-pass", candidate)
        self.assertIn("D2R_V33_CANDIDATE_399999_CAPACITY_PASS", candidate)
        self.assertIn("D2R_V33_CANDIDATE_399999_CAPACITY_REJECT", candidate)
        self.assertNotIn("teacher_capacity_production.py generate", candidate)
        self.assertNotIn("approved.json", candidate)
        self.assertIn("d2r_require_visible_gpus 1", oracle_c4)
        self.assertIn("399999_raft_unwarp.pt", oracle_c4)
        self.assertIn(
            f'readonly D2R_ORACLE_TEACHER_SHA256="{CANDIDATE_SHA}"', oracle_c4
        )
        self.assertIn("scripts/diagnose_teacher_quarter_turn_oracle.py", oracle_c4)
        self.assertIn("--expected-teacher-sha256", oracle_c4)
        self.assertIn("--sample-count 300", oracle_c4)
        self.assertIn("D2R_V33_QUARTER_TURN_DIAGNOSTIC_COMPLETE", oracle_c4)
        self.assertIn("runs/v33_diagnostics/teacher_quarter_turn_oracle", oracle_c4)
        self.assertNotIn("teacher_capacity_production.py", oracle_c4)
        self.assertNotIn("approved.json", oracle_c4)
        self.assertIn("d2r_require_visible_gpus 1", canonical_c4)
        self.assertIn("399999_raft_unwarp.pt", canonical_c4)
        self.assertIn(
            f'readonly D2R_CANONICAL_TEACHER_SHA256="{CANDIDATE_SHA}"',
            canonical_c4,
        )
        self.assertIn(
            "scripts/diagnose_teacher_quarter_turn_oracle_canonical_frame.py",
            canonical_c4,
        )
        self.assertIn('main(["--canonical-frame-v2", *sys.argv[1:]])', canonical_cli)
        self.assertIn("--sample-count 300", canonical_c4)
        self.assertIn(
            "runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_frame_v2",
            canonical_c4,
        )
        self.assertIn(
            "D2R_V33_QUARTER_TURN_CANONICAL_FRAME_DIAGNOSTIC_COMPLETE",
            canonical_c4,
        )
        for forbidden in (
            "teacher_capacity_production.py",
            "approved.json",
            "D2R_CAPACITY_POINTER",
            "D2R_CAPACITY_DIR",
            "D2R_V33_CAPACITY_PASS",
            "02_teacher_smoke_8gpu.sh",
            "03_teacher_train_8gpu.sh",
            "train_unified",
        ):
            self.assertNotIn(forbidden, canonical_c4)
            self.assertNotIn(forbidden, solver_sweep_c4)
            self.assertNotIn(forbidden, full_geometry_c4)
            self.assertNotIn(forbidden, full_geometry_grid_c4)
            self.assertNotIn(forbidden, best_of_c4)
            self.assertNotIn(forbidden, structural_safety_c4)
        self.assertIn("d2r_require_visible_gpus 1", solver_sweep_c4)
        self.assertIn("399999_raft_unwarp.pt", solver_sweep_c4)
        self.assertIn(CANDIDATE_SHA, solver_sweep_c4)
        self.assertIn(
            "scripts/diagnose_teacher_quarter_turn_oracle_solver_sweep.py",
            solver_sweep_c4,
        )
        self.assertIn("--residual-target-iteration-sweep", solver_sweep_cli)
        for iterations in ("6", "12", "24"):
            self.assertIn(f'"{iterations}"', solver_sweep_cli)
        self.assertIn("--sample-count 300", solver_sweep_c4)
        self.assertIn(
            "runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_solver_sweep_v3",
            solver_sweep_c4,
        )
        self.assertIn(
            "teacher_quarter_turn_oracle_canonical_solver_sweep_diagnostic",
            solver_sweep_c4,
        )
        self.assertIn('"$report_version" == "3"', solver_sweep_c4)
        self.assertIn('"$baseline_iterations" == "6"', solver_sweep_c4)
        self.assertIn('"$protocol_iterations" == "6,12,24"', solver_sweep_c4)
        self.assertIn('"$solver_iterations" == "6,12,24"', solver_sweep_c4)
        self.assertIn('"$sample_counts" == "300,300,300"', solver_sweep_c4)
        self.assertIn(
            "D2R_V33_QUARTER_TURN_CANONICAL_SOLVER_SWEEP_COMPLETE",
            solver_sweep_c4,
        )
        self.assertIn("d2r_require_visible_gpus 1", full_geometry_c4)
        self.assertIn("399999_raft_unwarp.pt", full_geometry_c4)
        self.assertIn(CANDIDATE_SHA, full_geometry_c4)
        self.assertIn(
            "scripts/diagnose_teacher_full_geometry_oracle_canonical.py",
            full_geometry_c4,
        )
        self.assertIn("--full-geometry-per-sample", full_geometry_cli)
        self.assertIn('"1"', full_geometry_cli)
        self.assertIn("--residual-target-iterations-override", full_geometry_cli)
        self.assertIn('"12"', full_geometry_cli)
        self.assertIn("--sample-count 300", full_geometry_c4)
        self.assertIn(
            "runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_v4",
            full_geometry_c4,
        )
        self.assertIn(
            "teacher_quarter_turn_oracle_canonical_full_geometry_diagnostic",
            full_geometry_c4,
        )
        self.assertIn('"$report_version" == "4"', full_geometry_c4)
        self.assertIn('"$configured_iterations" == "6"', full_geometry_c4)
        self.assertIn('"$effective_iterations" == "12"', full_geometry_c4)
        self.assertIn('"$aggregate_count" == "300"', full_geometry_c4)
        self.assertIn(
            "D2R_V33_QUARTER_TURN_CANONICAL_FULL_GEOMETRY_COMPLETE",
            full_geometry_c4,
        )
        self.assertIn("d2r_require_visible_gpus 1", full_geometry_grid_c4)
        self.assertIn("399999_raft_unwarp.pt", full_geometry_grid_c4)
        self.assertIn(CANDIDATE_SHA, full_geometry_grid_c4)
        self.assertIn(
            "scripts/diagnose_teacher_full_geometry_oracle_capacity_grid.py",
            full_geometry_grid_c4,
        )
        self.assertIn(
            "--full-geometry-solver-iteration-sweep", full_geometry_grid_cli
        )
        self.assertIn("--full-geometry-residual-cap-sweep", full_geometry_grid_cli)
        for value in ("12", "24", "32", "40"):
            self.assertIn(f'"{value}"', full_geometry_grid_cli)
        self.assertIn("--sample-count 300", full_geometry_grid_c4)
        self.assertIn(
            "runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_capacity_grid_v5",
            full_geometry_grid_c4,
        )
        self.assertIn(
            "teacher_quarter_turn_oracle_canonical_full_geometry_capacity_grid_diagnostic",
            full_geometry_grid_c4,
        )
        self.assertIn('"$report_version" == "5"', full_geometry_grid_c4)
        self.assertIn('"$protocol_iterations" == "12,24"', full_geometry_grid_c4)
        self.assertIn('"$protocol_caps" == "24,32,40"', full_geometry_grid_c4)
        self.assertIn(
            '"$grid_cells" == "12:24,12:32,12:40,24:24,24:32,24:40"',
            full_geometry_grid_c4,
        )
        self.assertIn(
            "D2R_V33_QUARTER_TURN_CANONICAL_FULL_GEOMETRY_CAPACITY_GRID_COMPLETE",
            full_geometry_grid_c4,
        )
        self.assertIn("d2r_require_visible_gpus 1", best_of_c4)
        self.assertIn("399999_raft_unwarp.pt", best_of_c4)
        self.assertIn(CANDIDATE_SHA, best_of_c4)
        self.assertIn(
            "scripts/diagnose_teacher_full_geometry_oracle_best_of_c4.py",
            best_of_c4,
        )
        self.assertIn("--full-geometry-best-of-c4", best_of_c4_cli)
        self.assertIn("--c4-candidate-batch-size", best_of_c4_cli)
        self.assertIn("--sample-count 300", best_of_c4)
        self.assertIn(
            "runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_best_of_c4_v6",
            best_of_c4,
        )
        self.assertIn(
            "teacher_quarter_turn_oracle_canonical_full_geometry_best_of_c4_diagnostic",
            best_of_c4,
        )
        self.assertIn('"$report_version" == "6"', best_of_c4)
        self.assertIn('"$candidate_order" == "0,-90,90,180"', best_of_c4)
        self.assertIn('"$nearest_cells" == "6"', best_of_c4)
        self.assertIn('"$min_candidates" == "4"', best_of_c4)
        self.assertIn(
            "D2R_V33_QUARTER_TURN_CANONICAL_FULL_GEOMETRY_BEST_OF_C4_COMPLETE",
            best_of_c4,
        )
        self.assertIn("d2r_require_visible_gpus 1", structural_safety_c4)
        self.assertIn("399999_raft_unwarp.pt", structural_safety_c4)
        self.assertIn(CANDIDATE_SHA, structural_safety_c4)
        self.assertIn(
            "scripts/diagnose_teacher_best_of_c4_structural_safety.py",
            structural_safety_c4,
        )
        self.assertIn(
            "teacher_structural_safety_diagnostic import main",
            structural_safety_cli,
        )
        self.assertIn("--best-of-c4-report", structural_safety_c4)
        self.assertIn("--residual-target-iterations 24", structural_safety_c4)
        self.assertIn("--residual-cap-sweep 24 32 40", structural_safety_c4)
        self.assertIn("--selected-max-residual-px 40", structural_safety_c4)
        self.assertIn("--sample-count 300", structural_safety_c4)
        self.assertIn(
            "runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_structural_safety_v7",
            structural_safety_c4,
        )
        self.assertIn(
            "teacher_quarter_turn_oracle_canonical_full_geometry_structural_safety_diagnostic",
            structural_safety_c4,
        )
        self.assertIn('"$report_version" == "7"', structural_safety_c4)
        self.assertIn('"$protocol_iterations" == "24"', structural_safety_c4)
        self.assertIn('"$protocol_caps" == "24.0,32.0,40.0"', structural_safety_c4)
        self.assertIn('"$cell_cap" == "40.0"', structural_safety_c4)
        self.assertIn(
            "D2R_V33_BEST_OF_C4_STRUCTURAL_SAFETY_COMPLETE",
            structural_safety_c4,
        )
        self.assertIn("d2r_require_visible_gpus 1", visualize_c4)
        self.assertIn("399999_raft_unwarp.pt", visualize_c4)
        self.assertIn(CANDIDATE_SHA, visualize_c4)
        self.assertIn(
            "scripts/visualize_teacher_c4_route_failure.py", visualize_c4
        )
        self.assertIn(
            "teacher_c4_route_visualization import main", visualize_c4_cli
        )
        self.assertIn("--expected-v6-report-sha256", visualize_c4)
        self.assertIn("--sample-id \"$D2R_C4_VIS_SAMPLE_ID\"", visualize_c4)
        self.assertIn("--residual-target-iterations 24", visualize_c4)
        self.assertIn("--max-residual-px 40", visualize_c4)
        self.assertIn("--feature-stride 8", visualize_c4)
        self.assertIn(
            "runs/v33_diagnostics/teacher_c4_route_failure_visualization_v1",
            visualize_c4,
        )
        self.assertIn(
            "D2R_V33_C4_ROUTE_VISUALIZATION_COMPLETE", visualize_c4
        )
        self.assertNotIn("approved.json", visualize_c4)
        self.assertNotIn("teacher_capacity_production.py", visualize_c4)
        for token in (
            "scripts/teacher_capacity_production.py generate",
            '--config "$D2R_CONFIG"',
            '--pointer "$D2R_CAPACITY_POINTER"',
            '--resume "$D2R_SEED_CHECKPOINT"',
            '--output-dir "$D2R_CAPACITY_DIR"',
            "--seed 42",
            "--batch-size 1",
            "--device cuda:0",
            "--threads 16",
            "scripts/teacher_capacity_production.py verify",
            "D2R_V33_CAPACITY_PASS",
        ):
            self.assertIn(token, capacity)

        self.assertIn("d2r_require_visible_gpus 8", smoke)
        self.assertIn("SMOKE_NPROC=8", smoke)
        self.assertIn('FAILURE_WORLD_SIZES="2 8"', smoke)
        self.assertIn("ALLOW_INCOMPLETE_SEED=0", smoke)
        self.assertIn("ALLOW_PARTIAL_SMOKE=0", smoke)
        self.assertIn("bash scripts/smoke_unified_v33_teacher.sh", smoke)
        self.assertIn("scripts/verify_v33_formal_smoke_report.py", smoke)
        self.assertIn("d2r_acquire_job_lock production_pipeline", capacity)
        self.assertIn("d2r_acquire_job_lock production_pipeline", smoke)
        self.assertIn("d2r_acquire_job_lock production_pipeline", train)

        self.assertIn("d2r_require_visible_gpus 8", train)
        self.assertIn("NUM_GPUS=8", train)
        self.assertIn("EPOCHS=32", train)
        self.assertIn("RUN_PREFLIGHT=1", train)
        self.assertIn("PREFLIGHT_SAMPLES=3", train)
        self.assertIn("MAX_GT_MAE=0.08", train)
        self.assertIn("CHECK_GPU_ONLY=0", train)
        self.assertIn("CHECK_LAUNCH_ONLY=0", train)
        self.assertIn("TRAIN_LOCK_HELD=0", train)
        self.assertIn("-u TRAIN_LOCK_FD", train)
        self.assertIn("ALLOW_RESUME_BELOW_MIN_EPOCHS=0", train)
        self.assertIn("bash scripts/train_unified_v33_teacher.sh", train)
        self.assertNotIn("bash scripts/train_unified_v3.sh", train)
        self.assertIn("epoch_0032.pt", train)
        self.assertIn('d2r_checkpoint_status "$anchor_checkpoint" 20 21', train)
        self.assertIn("immutable anchor.pt 的 capacity receipt 复验失败", train)
        self.assertIn("latest.pt 的 capacity receipt 复验失败", train)
        self.assertIn("anchor/latest/final checkpoint", train)
        self.assertIn("D2R_V33_TRAIN_COMPLETE", train)

    def test_sync_allowlist_contains_pipeline_jobs(self) -> None:
        sync_script = TOP_ROOT / "scripts" / "sync_code_mirror.sh"
        subprocess.run(["bash", "-n", str(sync_script)], check=True)
        sync_text = sync_script.read_text(encoding="utf-8")
        self.assertIn('"PROGRESS_PLAN.md"', sync_text)
        self.assertIn('"slurm/v33_pipeline"', sync_text)
        self.assertIn('"$V31_PROJECT/docs"', sync_text)
        main_readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for name in (
            "00_teacher_candidate_399999_diagnostic_1gpu.sh",
            "diagnostic_teacher_399999_oracle_c4_1gpu.sh",
            "diagnostic_teacher_399999_oracle_c4_canonical_frame_v2_1gpu.sh",
            "diagnostic_teacher_399999_oracle_c4_canonical_solver_sweep_v3_1gpu.sh",
            "diagnostic_teacher_399999_oracle_c4_canonical_full_geometry_v4_1gpu.sh",
            "diagnostic_teacher_399999_oracle_c4_canonical_full_geometry_capacity_grid_v5_1gpu.sh",
            "diagnostic_teacher_399999_oracle_c4_canonical_full_geometry_best_of_c4_v6_1gpu.sh",
            "diagnostic_teacher_399999_best_of_c4_structural_safety_v7_1gpu.sh",
            "visualize_teacher_399999_c4_route_failure_1gpu.sh",
            "01_teacher_capacity_1gpu.sh",
            "02_teacher_smoke_8gpu.sh",
            "03_teacher_train_8gpu.sh",
        ):
            self.assertIn(name, main_readme)
        self.assertIn("docs/figures/v33_unified_model_overview.svg", main_readme)
        self.assertNotIn("1_capacity_v33.sbatch", main_readme)

    def test_capacity_report_evaluator_renders_actionable_failure(self) -> None:
        metrics = {
            "sample_count": 300,
            "oracle_solver_coverage": 0.72,
            "oracle_residual_overflow_given_solvable_any_axis_pixel_rate": 0.66,
            "trainable_coverage": 0.25,
            "stride_trainable_oracle_reconstruction_epe_px": 0.10,
        }
        report = {
            "results": {
                "original": dict(metrics),
                "rotation_augmented": dict(metrics),
                "full_geometry_augmented": dict(metrics),
                "rotation_bins": [],
            }
        }
        decision = {
            "passed": False,
            "policy_id": "teacher_capacity_production_v1",
            "failures": [
                {
                    "code": "aggregate.rotation_augmented.trainable_coverage",
                    "actual": 0.25,
                    "operator": ">=",
                    "threshold": 0.985,
                }
            ],
        }
        summary = CAPACITY_EVALUATOR.render_summary(report, decision)
        self.assertIn("TEACHER_CAPACITY_DECISION=REJECT", summary)
        self.assertIn("aggregate.rotation_augmented", summary)
        self.assertIn("solver=0.720000", summary)
        self.assertIn("trainable=0.250000", summary)
        self.assertIn("failed_check_count=1", summary)
        self.assertIn(
            "FAIL aggregate.rotation_augmented.trainable_coverage", summary
        )

    def test_architecture_overview_is_editable_and_has_expected_exports(self) -> None:
        renderer = OVERVIEW_RENDERER.read_text(encoding="utf-8")
        for token in (
            '"svg.fonttype": "none"',
            r"$F(x)=\alpha R(x)+B(x+\alpha R(x))$",
            "feature gate ≠ residual gate",
            "Features only · no RGB decode",
        ):
            self.assertIn(token, renderer)

        svg = OVERVIEW_ROOT / "v33_unified_model_overview.svg"
        pdf = OVERVIEW_ROOT / "v33_unified_model_overview.pdf"
        png = OVERVIEW_ROOT / "v33_unified_model_overview.png"
        for artifact in (svg, pdf, png):
            self.assertTrue(artifact.is_file(), artifact)
            self.assertGreater(artifact.stat().st_size, 1024, artifact)

        svg_text = svg.read_text(encoding="utf-8")
        for label in (
            "Qwen-Image-Edit",
            "Reliability-gated fusion",
            "RAFT-like residual refiner",
            "Backward-flow composition",
            "TRAINING SUPERVISION ONLY",
        ):
            self.assertIn(label, svg_text)
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
        png_header = png.read_bytes()[:24]
        self.assertEqual(png_header[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(png_header[12:16], b"IHDR")
        self.assertEqual(struct.unpack(">II", png_header[16:24]), (2700, 1500))

    def test_formal_smoke_report_reverification_binds_seed_config_and_functional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "epoch_0020.pt"
            config = root / "unified_v3_3_teacher_anchor.yaml"
            seed.write_bytes(b"immutable-seed")
            config.write_bytes(b"formal-config")
            seed_sha = hashlib.sha256(seed.read_bytes()).hexdigest()
            config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
            functional_path = root / "functional_report.json"
            functional = {
                "schema_version": 1,
                "kind": "v33_real_teacher_qwen_ddp_smoke",
                "scope": "functional_substage_only",
                "passed": True,
                "invoked_world_size": 8,
                "seed_completed_epochs": 20,
                "output_completed_epochs": 21,
                "teacher_sha256": TEACHER_SHA,
                "verified_checkpoints": {
                    name: {
                        "output_completed_epochs": 21,
                        "optimizer_state_count": 4,
                        "optimizer_step_min": 1.0,
                        "optimizer_step_max": 1.0,
                        "teacher_sha256": TEACHER_SHA,
                    }
                    for name in ("anchor", "best", "latest")
                },
                "artifacts": {
                    "seed": {
                        "source_path": str(seed.resolve()),
                        "sha256": seed_sha,
                    },
                    "config": {
                        "path": str(config.resolve()),
                        "sha256": config_sha,
                    },
                },
            }
            functional_path.write_text(json.dumps(functional), encoding="utf-8")
            functional_bytes = functional_path.read_bytes()
            overall_path = root / "overall_report.json"
            overall = {
                "schema_version": 1,
                "kind": "v33_teacher_qwen_ddp_smoke_overall",
                "decision": "pass",
                "passed": True,
                "formal": True,
                "functional_world_size": 8,
                "seed_completed_epochs": 20,
                "teacher_sha256": TEACHER_SHA,
                "failure_world_sizes": [2, 8],
                "failure_evidence": [
                    {
                        "world_size": world_size,
                        "exit_status": 1,
                        "rank_isolation": [
                            [rank, rank, 1, 0] for rank in range(world_size)
                        ],
                        "log_sha256": str(world_size) * 64,
                    }
                    for world_size in (2, 8)
                ],
                "functional_report": {
                    "path": str(functional_path.resolve()),
                    "size_bytes": len(functional_bytes),
                    "sha256": hashlib.sha256(functional_bytes).hexdigest(),
                },
            }
            overall_path.write_text(json.dumps(overall), encoding="utf-8")

            result = VERIFIER.verify_formal_smoke_report(
                overall_report=overall_path,
                expected_seed=seed,
                expected_seed_sha256=seed_sha,
                expected_config=config,
                expected_teacher_sha256=TEACHER_SHA,
            )
            self.assertEqual(result["seed_sha256"], seed_sha)
            self.assertEqual(result["config_sha256"], config_sha)

            functional["invoked_world_size"] = 2
            functional_path.write_text(json.dumps(functional), encoding="utf-8")
            with self.assertRaisesRegex(
                VERIFIER.SmokeReportVerificationError,
                "functional report content differs",
            ):
                VERIFIER.verify_formal_smoke_report(
                    overall_report=overall_path,
                    expected_seed=seed,
                    expected_seed_sha256=seed_sha,
                    expected_config=config,
                    expected_teacher_sha256=TEACHER_SHA,
                )


if __name__ == "__main__":
    unittest.main()
