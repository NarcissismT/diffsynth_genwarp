from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EvaluationProtocolTest(unittest.TestCase):
    @staticmethod
    def _payload(epoch: int, *, runtime: float | None = None):
        payload = {
            "epoch": epoch,
            "config": {
                "model": {
                    "correlation_temperature": 1.0,
                    "correlation_temperature_start": 9.797959,
                    "correlation_ramp_start_epoch": 9,
                    "correlation_ramp_epochs": 3,
                }
            },
        }
        if runtime is not None:
            payload["correlation_temperature"] = runtime
        return payload

    @staticmethod
    def _ablation_metrics(
        epe: float,
        *,
        mode: str,
        scale: float,
        prior_epe: float = 5.75,
    ):
        from diffusion2raft.ablate_residual_qwen import METRIC_KEYS

        gain = prior_epe - epe
        better = gain > 0.0
        values = {metric: 0.0 for metric in METRIC_KEYS}
        values.update(
            epe=epe,
            epe_p95=epe * 2.0,
            line_epe=5.5 - gain * 0.2,
            line_straightness_error=0.12 - gain * 0.01,
            edge_epe=epe,
            prior_epe=prior_epe,
            epe_gain=gain,
            relative_epe_gain=gain / prior_epe,
            final_win_rate=0.62 if better else (0.0 if scale == 0.0 else 0.40),
            fold_rate=0.0002 if better else (0.0001 if scale == 0.0 else 0.0005),
            jacobian_p01=0.1,
            residual_epe=abs(gain),
            residual_p95=abs(gain) * 2.0,
            applied_residual_p95=0.0 if scale == 0.0 else scale,
            feature_confidence=0.5,
            matching_feature_confidence=(
                0.0 if mode in {"none", "context_only"} else 0.5
            ),
            context_feature_confidence=(
                0.0 if mode in {"none", "matching_only"} else 0.5
            ),
            qwen_match_epe=29.0,
            qwen_advantage=-20.0,
            qwen_win_rate=0.02,
        )
        if scale == 0.0:
            values.update(
                epe=prior_epe,
                line_epe=5.5,
                line_straightness_error=0.12,
                epe_gain=0.0,
                relative_epe_gain=0.0,
                final_win_rate=0.0,
                applied_residual_p95=0.0,
            )
        return values

    def _formal_ablation_report(self):
        from diffusion2raft.ablate_residual_qwen import (
            _commit_cell,
            _load_or_initialize_report,
            _number_key,
        )

        temperature = 9.797959
        scales = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
        epes = {
            "both": [5.75, 5.30, 4.90, 5.00, 5.30, 5.80],
            "none": [5.75, 6.00, 5.90, 6.10, 6.20, 6.30],
            "matching_only": [5.75, 6.50, 6.40, 6.30, 6.20, 6.00],
            "context_only": [5.75, 5.50, 5.20, 5.00, 4.90, 4.80],
        }
        protocol = {
            "version": 3,
            "training_correlation_temperature": temperature,
            "cells": [
                {
                    "temperature": temperature,
                    "temperature_key": _number_key(temperature),
                    "qwen_mode": mode,
                    "is_training_temperature": True,
                }
                for mode in ("both", "none", "matching_only", "context_only")
            ],
            "residual_scales": scales,
            "validation_samples": 300,
            "max_batches": None,
        }
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "formal-report.json"
        report = _load_or_initialize_report(path, protocol=protocol, resume=False)
        for mode, mode_epes in epes.items():
            result = {
                "temperature": temperature,
                "qwen_mode": mode,
                "evaluated_batches": 300,
                "evaluated_samples": 300,
                "metrics": {
                    _number_key(scale): self._ablation_metrics(
                        epe, mode=mode, scale=scale
                    )
                    for scale, epe in zip(scales, mode_epes, strict=True)
                },
            }
            _commit_cell(
                report,
                path=path,
                temperature_key=_number_key(temperature),
                mode=mode,
                result=result,
            )
        return report

    def test_checkpoint_temperature_is_reconstructed_and_cross_checked(self) -> None:
        from diffusion2raft.train import _checkpoint_correlation_temperature

        expected = (9.797959, 3.1301691647577132, 1.0)
        for epoch, temperature in zip((8, 9, 10), expected):
            with self.subTest(epoch=epoch):
                self.assertAlmostEqual(
                    _checkpoint_correlation_temperature(
                        self._payload(epoch), required=True
                    ),
                    temperature,
                )
                self.assertAlmostEqual(
                    _checkpoint_correlation_temperature(
                        self._payload(epoch, runtime=temperature), required=True
                    ),
                    temperature,
                )

        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            _checkpoint_correlation_temperature(
                self._payload(8, runtime=1.0), required=True
            )
        with self.assertRaisesRegex(RuntimeError, "requires config"):
            _checkpoint_correlation_temperature(
                {"epoch": 8, "correlation_temperature": 9.797959},
                required=True,
            )
        with self.assertRaisesRegex(RuntimeError, "finite and positive"):
            _checkpoint_correlation_temperature(
                self._payload(8, runtime=float("nan")), required=True
            )

    def test_distributed_eval_sampler_has_no_padding_or_duplicates(self) -> None:
        from diffusion2raft.train import ExactDistributedEvalSampler

        shards = [
            list(ExactDistributedEvalSampler(300, num_replicas=8, rank=rank))
            for rank in range(8)
        ]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(300)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(shard) for shard in shards], [38] * 4 + [37] * 4)

    def test_default_staged_cell_plan_has_exactly_eight_cells(self) -> None:
        from diffusion2raft.ablate_residual_qwen import build_cell_plan

        training_temperature = 9.797959
        cells = build_cell_plan(
            [training_temperature, 3.1301691647577132, 1.0],
            training_temperature=training_temperature,
            qwen_modes=["both", "none"],
            training_temperature_qwen_modes=["matching_only", "context_only"],
        )
        self.assertEqual(
            cells,
            [
                (training_temperature, "both"),
                (training_temperature, "none"),
                (3.1301691647577132, "both"),
                (3.1301691647577132, "none"),
                (1.0, "both"),
                (1.0, "none"),
                (training_temperature, "matching_only"),
                (training_temperature, "context_only"),
            ],
        )

    def test_qwen_manifest_identity_tracks_recursive_metadata_not_contents(self) -> None:
        from diffusion2raft.ablate_residual_qwen import qwen_model_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "qwen"
            nested = root / "transformer"
            nested.mkdir(parents=True)
            config_path = root / "config.json"
            shard_path = nested / "model-00001.safetensors"
            config_path.write_text("{}", encoding="utf-8")
            shard_path.write_bytes(b"not-a-real-weight")
            config = {
                "model": {"feature_backend": "qwen"},
                "qwen": {"model_id": str(root), "local_files_only": True},
            }
            first = qwen_model_identity(config)
            second = qwen_model_identity(config)
            self.assertEqual(first, second)
            self.assertEqual(first["file_count"], 2)
            self.assertEqual(
                [item["relative_path"] for item in first["files"]],
                ["config.json", "transformer/model-00001.safetensors"],
            )
            self.assertTrue(all("sha256" not in item for item in first["files"]))

            stat = shard_path.stat()
            os.utime(
                shard_path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
            )
            changed = qwen_model_identity(config)
            self.assertNotEqual(first["manifest_sha256"], changed["manifest_sha256"])

    def test_partial_report_is_atomic_and_strictly_resumable(self) -> None:
        from diffusion2raft.ablate_residual_qwen import (
            METRIC_KEYS,
            _commit_cell,
            _load_or_initialize_report,
        )

        protocol = {
            "version": 2,
            "cells": [
                {
                    "temperature": 1.0,
                    "temperature_key": "1.0",
                    "qwen_mode": "both",
                    "is_training_temperature": True,
                }
            ],
            "residual_scales": [0.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = _load_or_initialize_report(
                path, protocol=protocol, resume=False
            )
            self.assertTrue(path.is_file())
            self.assertIsNone(report["decision"])
            cell = {
                "temperature": 1.0,
                "qwen_mode": "both",
                "evaluated_batches": 1,
                "evaluated_samples": 1,
                "metrics": {
                    "0.0": {metric: 0.0 for metric in METRIC_KEYS}
                },
            }
            _commit_cell(
                report,
                path=path,
                temperature_key="1.0",
                mode="both",
                result=cell,
            )
            resumed = _load_or_initialize_report(
                path, protocol=protocol, resume=True
            )
            self.assertTrue(resumed["progress"]["complete"])
            self.assertEqual(resumed["decision"]["status"], "insufficient")
            with self.assertRaisesRegex(RuntimeError, "protocol"):
                _load_or_initialize_report(
                    path,
                    protocol={**protocol, "version": 3},
                    resume=True,
                )

    def test_formal_decision_classifies_residual_and_qwen_paths(self) -> None:
        from diffusion2raft.ablate_residual_qwen import analyze_complete_report

        decision = analyze_complete_report(self._formal_ablation_report())
        self.assertEqual(decision["status"], "ready")
        verdicts = {
            cell["qwen_mode"]: cell["residual_verdict"]
            for cell in decision["cells"]
        }
        self.assertEqual(
            verdicts,
            {
                "both": "over_correction",
                "none": "residual_direction_wrong",
                "matching_only": "needs_pixel_gate",
                "context_only": "full_residual_best",
            },
        )
        comparison = decision["training_temperature_qwen_comparison"]
        self.assertEqual(comparison["best_mode"], "context_only")
        self.assertEqual(
            [row["qwen_mode"] for row in comparison["ranking"]],
            ["context_only", "both", "none", "matching_only"],
        )
        self.assertEqual(decision["global_best"]["qwen_mode"], "context_only")
        self.assertEqual(decision["global_best"]["residual_scale"], 1.0)
        self.assertEqual(decision["global_best"]["quality_gate"]["status"], "pass")
        self.assertIn("Qwen@train=context_only", decision["summary_text"])

    def test_complete_decision_is_recomputed_during_strict_validation(self) -> None:
        from diffusion2raft.ablate_residual_qwen import analyze_complete_report

        report = copy.deepcopy(self._formal_ablation_report())
        report["decision"]["global_best"]["metrics"]["epe"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "decision"):
            analyze_complete_report(report)

    def test_incomplete_report_cannot_be_analyzed(self) -> None:
        from diffusion2raft.ablate_residual_qwen import (
            _load_or_initialize_report,
            analyze_complete_report,
        )

        protocol = {
            "version": 3,
            "cells": [
                {
                    "temperature": 1.0,
                    "temperature_key": "1.0",
                    "qwen_mode": "both",
                    "is_training_temperature": True,
                }
            ],
            "residual_scales": [0.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            report = _load_or_initialize_report(
                Path(directory) / "partial.json", protocol=protocol, resume=False
            )
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                analyze_complete_report(report)

    def test_quality_gate_marks_missing_prior_line_metrics_insufficient(self) -> None:
        from diffusion2raft.ablate_residual_qwen import _quality_gate

        candidate = {
            "epe_gain": 1.0,
            "final_win_rate": 0.7,
            "fold_rate": 0.0001,
            "line_epe": 4.0,
            "line_straightness_error": 0.08,
        }
        gate = _quality_gate(candidate, {})
        self.assertEqual(gate["status"], "insufficient")
        self.assertIsNone(gate["passed"])
        self.assertFalse(
            gate["checks"]["line_epe_better_than_prior"]["available"]
        )

    def test_offline_residual_scaling_matches_model_composition(self) -> None:
        from diffusion2raft.ablate_residual_qwen import outputs_at_residual_scale
        from diffusion2raft.models.unified import build_unified_rectifier

        torch.manual_seed(19)
        config = {
            "feature_backend": "lite",
            "feature_channels": 8,
            "feature_stride": 8,
            "cnn_feature_channels": 8,
            "refiner_hidden_channels": 8,
            "refiner_iterations": 2,
            "correlation_radius": 1,
            "correlation_temperature": 1.0,
            "feature_dropout_prob": 0.0,
            "prior_base_channels": 4,
            "prior_control_stride": 8,
            "max_residual_px": 8.0,
        }
        model = build_unified_rectifier(config, {}, device="cpu").eval()
        warped = torch.rand(1, 3, 48, 56)
        with torch.inference_mode():
            model.set_residual_application_scale(1.0)
            full = model(warped, stage="unified")
            full_reused = outputs_at_residual_scale(
                full, 1.0, source_size=(48, 56)
            )
            self.assertIs(full_reused["flows"], full["flows"])
            self.assertIs(full_reused["residuals"], full["residuals"])
            offline = outputs_at_residual_scale(
                full, 0.25, source_size=(48, 56)
            )
            model.set_residual_application_scale(0.25)
            direct = model(warped, stage="unified")
            self.assertTrue(
                torch.allclose(
                    offline["final_flow"], direct["final_flow"], atol=1e-6
                )
            )
            for offline_residual, direct_residual in zip(
                offline["residuals"], direct["residuals"]
            ):
                self.assertTrue(
                    torch.allclose(
                        offline_residual, direct_residual, atol=1e-7
                    )
                )

            zero = outputs_at_residual_scale(
                full, 0.0, source_size=(48, 56)
            )
            self.assertTrue(torch.equal(zero["final_flow"], full["prior_flow"]))
            self.assertEqual(
                int(torch.count_nonzero(zero["residuals"][-1])), 0
            )


if __name__ == "__main__":
    unittest.main()
