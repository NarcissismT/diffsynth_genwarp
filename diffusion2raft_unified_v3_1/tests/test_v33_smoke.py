from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FORMAL_TEACHER_SHA256 = (
    "3d079e19445168169144f2af741362f673289b6510df4a4c1af348449ae045b9"
)


def _fixture():
    from verify_v33_smoke_checkpoint import REQUIRED_METRICS

    teacher_sha = "a" * 64
    seed = {
        "stage": "unified",
        "prior_backend": "learned",
        "epoch": 19,
        "model": {"prior.weight": torch.zeros(1)},
    }
    identity = {"sha256": teacher_sha}
    config = {
        "model": {
            "feature_backend": "qwen",
            "prior_backend": "torchscript",
            "prior_torchscript_sha256": teacher_sha,
        },
        "qwen": {"model_id": "/models/qwen"},
        "train": {"stage": "unified", "epochs": 32},
    }
    effective_config = copy.deepcopy(config)
    effective_config["train"].update(
        epochs=21,
        max_train_steps=1,
        max_val_batches=1,
        preview_every=0,
    )
    output = {
        "stage": "unified",
        "prior_backend": "torchscript",
        "epoch": 20,
        "model": {"prior._teacher_backend_marker": torch.zeros(1)},
        "teacher_prior_identity": identity,
        "deployment_contract": {"teacher_prior_identity": identity},
        "residual_application": {"origin_epoch": 20, "scale": 0.0},
        "config": effective_config,
        "metrics": {name: 0.0 for name in REQUIRED_METRICS},
        "optimizer": {
            "state": {
                0: {"step": torch.tensor(1.0)},
                1: {"step": torch.tensor(1.0)},
            }
        },
    }
    return seed, output, config


def _isolation_rows(world_size: int) -> list[tuple[int, int]]:
    return [(rank, rank) for rank in range(world_size)]


def _failure_evidence(world_size: int, status: int = 1) -> dict:
    ranks = [[rank, rank, 1, 0] for rank in range(world_size)]
    token = f"token-w{world_size}"
    return {
        "world_size": world_size,
        "exit_status": status,
        "failure_token": token,
        "log_text": "\n".join(
            (
                "D2R_DDP_ISOLATION_OK "
                + json.dumps(
                    {"world_size": world_size, "ranks": ranks}, sort_keys=True
                ),
                f"D2R_EXPECTED_RANK_FAILURE {token}",
            )
        ),
        "log_sha256": "a" * 64,
        "log_path": f"/tmp/failure-w{world_size}.log",
    }


def _overall_functional_report(
    world_size: int = 8,
    *,
    seed_completed_epochs: int = 20,
    config_path: str = "/repo/configs/unified_v3_3_teacher_anchor.yaml",
    config_sha256: str = "c" * 64,
) -> dict:
    output_completed_epochs = seed_completed_epochs + 1
    artifacts = {
        "config": {
            "path": config_path,
            "size_bytes": 100,
            "sha256": config_sha256,
        },
        "seed": {"path": "/tmp/seed.pt", "size_bytes": 100, "sha256": "1" * 64},
        "functional_log": {
            "path": "/tmp/functional.log",
            "size_bytes": 100,
            "sha256": "2" * 64,
        },
    }
    for index, name in enumerate(("anchor", "best", "latest"), start=3):
        artifacts[name] = {
            "path": f"/tmp/{name}.pt",
            "size_bytes": 100,
            "sha256": str(index) * 64,
            "ephemeral": True,
        }
    checkpoint_semantics = {
        name: {
            "output_completed_epochs": output_completed_epochs,
            "optimizer_state_count": 2,
            "optimizer_step_min": 1.0,
            "optimizer_step_max": 1.0,
            "teacher_sha256": FORMAL_TEACHER_SHA256,
        }
        for name in ("anchor", "best", "latest")
    }
    return {
        "schema_version": 1,
        "kind": "v33_real_teacher_qwen_ddp_smoke",
        "scope": "functional_substage_only",
        "passed": True,
        "invoked_world_size": world_size,
        "verified_rank_isolation": [[rank, rank] for rank in range(world_size)],
        "seed_completed_epochs": seed_completed_epochs,
        "output_completed_epochs": output_completed_epochs,
        "optimizer_state_count": 2,
        "optimizer_step_min": 1.0,
        "optimizer_step_max": 1.0,
        "teacher_sha256": FORMAL_TEACHER_SHA256,
        "validation_metrics": {"epe": 1.0},
        "verified_checkpoints": checkpoint_semantics,
        "artifacts": artifacts,
        "_artifact_path": "/tmp/functional_report.json",
        "_artifact_size_bytes": 100,
        "_artifact_sha256": "f" * 64,
    }


def _preflight_evidence() -> dict:
    return {
        "path": "/tmp/preflight_report.json",
        "size_bytes": 100,
        "sha256": "e" * 64,
        "errors": [],
    }


class V33SmokeContractTest(unittest.TestCase):
    def test_one_step_teacher_migration_contract_passes(self) -> None:
        from verify_v33_smoke_checkpoint import verify_smoke_payloads

        seed, output, config = _fixture()
        report = verify_smoke_payloads(
            seed,
            output,
            config,
            expected_seed_completed_epochs=20,
            invoked_world_size=8,
            observed_rank_isolation=_isolation_rows(8),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["optimizer_step_min"], 1.0)
        self.assertEqual(report["optimizer_step_max"], 1.0)
        self.assertEqual(report["invoked_world_size"], 8)

    def test_empty_or_wrong_step_optimizer_state_is_rejected(self) -> None:
        from verify_v33_smoke_checkpoint import verify_smoke_payloads

        seed, output, config = _fixture()
        output["optimizer"]["state"] = {}
        with self.assertRaisesRegex(ValueError, "optimizer.state is empty"):
            verify_smoke_payloads(
                seed,
                output,
                config,
                expected_seed_completed_epochs=20,
                invoked_world_size=2,
                observed_rank_isolation=_isolation_rows(2),
            )
        _, output, _ = _fixture()
        output["optimizer"]["state"][0]["step"] = torch.tensor(2.0)
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            verify_smoke_payloads(
                seed,
                output,
                config,
                expected_seed_completed_epochs=20,
                invoked_world_size=2,
                observed_rank_isolation=_isolation_rows(2),
            )

    def test_backend_epoch_teacher_and_alpha_mismatches_are_rejected(self) -> None:
        from verify_v33_smoke_checkpoint import verify_smoke_payloads

        mutations = (
            (lambda value: value.update(prior_backend="learned"), "torchscript"),
            (lambda value: value.update(epoch=21), "exactly one epoch"),
            (
                lambda value: value["teacher_prior_identity"].update(sha256="b" * 64),
                "configured SHA-256",
            ),
            (
                lambda value: value["residual_application"].update(scale=0.1),
                "scale at zero",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                seed, output, config = _fixture()
                mutation(output)
                with self.assertRaisesRegex(ValueError, message):
                    verify_smoke_payloads(
                        seed,
                        output,
                        config,
                        expected_seed_completed_epochs=20,
                        invoked_world_size=2,
                        observed_rank_isolation=_isolation_rows(2),
                    )

    def test_saved_config_and_rank_evidence_are_enforced(self) -> None:
        from verify_v33_smoke_checkpoint import verify_smoke_payloads

        seed, output, config = _fixture()
        output["config"]["model"]["feature_backend"] = "cnn"
        with self.assertRaisesRegex(ValueError, "output model config"):
            verify_smoke_payloads(
                seed,
                output,
                config,
                expected_seed_completed_epochs=20,
                invoked_world_size=2,
                observed_rank_isolation=_isolation_rows(2),
            )
        seed, output, config = _fixture()
        with self.assertRaisesRegex(ValueError, "rank-isolation evidence"):
            verify_smoke_payloads(
                seed,
                output,
                config,
                expected_seed_completed_epochs=20,
                invoked_world_size=2,
                observed_rank_isolation=[(0, 0)],
            )

        seed, output, config = _fixture()
        output["config"]["train"]["max_train_steps"] = 2
        with self.assertRaisesRegex(ValueError, "bounded one-step run"):
            verify_smoke_payloads(
                seed,
                output,
                config,
                expected_seed_completed_epochs=20,
                invoked_world_size=2,
                observed_rank_isolation=_isolation_rows(2),
            )

    def test_functional_log_must_bind_every_rank_exactly_once(self) -> None:
        from verify_v33_smoke_checkpoint import parse_functional_rank_isolation

        lines = "\n".join(
            f"[info] global_rank={rank} physical_local_rank={rank} "
            f"device='{rank}' -> logical cuda:0"
            for rank in range(2)
        )
        self.assertEqual(
            parse_functional_rank_isolation(lines, expected_world_size=2),
            [(0, 0), (1, 1)],
        )
        with self.assertRaisesRegex(ValueError, "exactly one isolated worker"):
            parse_functional_rank_isolation(lines + "\n" + lines, expected_world_size=2)

    def test_overall_report_requires_formal_eight_and_two_plus_eight_failures(self) -> None:
        from finalize_v33_smoke_report import _atomic_write, build_overall_report

        config_identity = {
            "path": "/repo/configs/unified_v3_3_teacher_anchor.yaml",
            "size_bytes": 100,
            "sha256": "c" * 64,
        }
        functional = _overall_functional_report()
        report = build_overall_report(
            functional,
            [_failure_evidence(2), _failure_evidence(8)],
            formal=True,
            formal_config_identity=config_identity,
            preflight_evidence=_preflight_evidence(),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "pass")
        with self.assertRaisesRegex(ValueError, "two- and eight-rank"):
            build_overall_report(
                functional,
                [_failure_evidence(2)],
                formal=True,
                formal_config_identity=config_identity,
                preflight_evidence=_preflight_evidence(),
            )
        with self.assertRaisesRegex(ValueError, "timeout"):
            build_overall_report(
                functional,
                [_failure_evidence(2, status=124), _failure_evidence(8)],
                formal=True,
                formal_config_identity=config_identity,
                preflight_evidence=_preflight_evidence(),
            )
        partial = build_overall_report(
            _overall_functional_report(world_size=2),
            [_failure_evidence(2)],
            formal=False,
            preflight_evidence=_preflight_evidence(),
        )
        self.assertFalse(partial["passed"])
        self.assertEqual(partial["decision"], "partial_only")
        with self.assertRaisesRegex(ValueError, "schema version"):
            build_overall_report(
                {"passed": True, "invoked_world_size": 8},
                [_failure_evidence(2), _failure_evidence(8)],
                formal=True,
                formal_config_identity=config_identity,
                preflight_evidence=_preflight_evidence(),
            )
        incomplete = _overall_functional_report(seed_completed_epochs=19)
        with self.assertRaisesRegex(ValueError, "at least 20"):
            build_overall_report(
                incomplete,
                [_failure_evidence(2), _failure_evidence(8)],
                formal=True,
                formal_config_identity=config_identity,
                preflight_evidence=_preflight_evidence(),
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overall.json"
            _atomic_write(path, {"passed": False})
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _atomic_write(path, {"passed": True})
            self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["passed"])

    def test_shell_entry_is_syntax_valid_and_uses_isolated_temporary_training(self) -> None:
        script = ROOT / "scripts" / "smoke_unified_v33_teacher.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text(encoding="utf-8")
        for fragment in (
            "mktemp -d",
            "--max_restarts=0",
            "--no_python",
            "isolate_cuda_rank.sh",
            "--max-train-steps 1",
            "--max-val-batches 1",
            "verify_v33_smoke_checkpoint.py",
            "probe_v33_ddp_isolation.py",
            "ALLOW_PARTIAL_SMOKE",
            "--kill-after=10s",
            "--kill-after=30s",
            "--functional-log",
            "finalize_v33_smoke_report.py",
            "overall_report.json",
            "teacher_capacity_production.py verify",
            "D2R_TEACHER_CAPACITY_RECEIPT_B64",
            "TEACHER_CAPACITY_POINTER",
        ):
            self.assertIn(fragment, text)
        self.assertIn('FUNCTIONAL_ROOT="$SMOKE_TMP/functional"', text)
        freeze_index = text.index('SEED_CHECKPOINT="$FROZEN_SEED_CHECKPOINT"')
        minimum_index = text.index("if (( completed_epochs < MIN_SEED_COMPLETED_EPOCHS ))")
        capacity_index = text.index(
            '"$PYTHON" scripts/teacher_capacity_production.py verify'
        )
        report_index = text.index('mkdir -p "$REPORT_ROOT"')
        train_index = text.index('"$PYTHON" -m torch.distributed.run')
        self.assertLess(freeze_index, minimum_index)
        self.assertLess(minimum_index, capacity_index)
        self.assertLess(capacity_index, report_index)
        self.assertLess(report_index, train_index)
        capacity_block = text[capacity_index:report_index]
        self.assertIn('--pointer "$TEACHER_CAPACITY_POINTER"', capacity_block)
        self.assertIn('--resume "$SEED_CHECKPOINT"', capacity_block)
        self.assertNotIn("D2R_V33_FUNCTIONAL_SMOKE_PASS", text)
        self.assertNotIn("D2R_DDP_FAILURE_PROPAGATION_PASS", text)

    def test_missing_capacity_pointer_has_no_formal_or_partial_report_side_effect(
        self,
    ) -> None:
        script = ROOT / "scripts" / "smoke_unified_v33_teacher.sh"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_python = temporary / "fake-python"
            fake_python.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    -c)
        printf '1 8\\n'
        ;;
    scripts/checkpoint_status.py)
        printf 'unified\\t19\\t20\\tline_epe\\t4.0\\n'
        ;;
    *)
        printf 'unexpected fake-python args: %s\\n' "$*" >&2
        exit 99
        ;;
esac
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            seed = temporary / "seed.pt"
            seed.write_bytes(b"seed fixture\n")
            missing_pointer = temporary / "missing-approved.json"

            for partial in (False, True):
                with self.subTest(partial=partial):
                    report = temporary / f"report-{'partial' if partial else 'formal'}"
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "PYTHON": str(fake_python),
                            "SEED_CHECKPOINT": str(seed),
                            "TEACHER_CAPACITY_POINTER": str(missing_pointer),
                            "REPORT_ROOT": str(report),
                            "SLURM_TMPDIR": str(temporary),
                            "ALLOW_PARTIAL_SMOKE": "1" if partial else "0",
                            "SMOKE_NPROC": "2" if partial else "8",
                            "FAILURE_WORLD_SIZES": "2" if partial else "2 8",
                            "RUN_ID": "capacity-partial" if partial else "capacity-formal",
                        }
                    )
                    completed = subprocess.run(
                        ["bash", str(script)],
                        cwd=ROOT,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 64, completed.stderr)
                    self.assertIn("teacher-capacity evidence pointer", completed.stderr)
                    self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
