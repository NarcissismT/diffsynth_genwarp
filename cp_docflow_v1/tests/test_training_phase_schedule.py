from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import torch

from cp_docflow.checkpoint import full_checkpoint_payload
from cp_docflow.make_smoke_data import _write_split
from cp_docflow.models.docgrid_flow import CPDocFlow
from cp_docflow.train_full import (
    configure_training_profile,
    resolve_training_phases,
    train,
    training_phase_for_step,
)


class TrainingPhaseScheduleTest(unittest.TestCase):
    @staticmethod
    def _model() -> CPDocFlow:
        return CPDocFlow(
            coarse={"base_channels": 8, "feature_channels": 16},
            qwen_backend="none",
            qwen_feature_channels=12,
            fusion_channels=16,
            hv_channels=8,
            velocity_hidden_channels=16,
            velocity_time_channels=16,
            flow_blocks=2,
            flow_heads=4,
            refiner_hidden_channels=16,
        )

    def test_warr_warmup_freezes_coarse_then_joint_profile_unfreezes_it(self) -> None:
        model = self._model()
        configure_training_profile(model, "warr", "warr")
        self.assertFalse(any(parameter.requires_grad for parameter in model.coarse.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.refiner.parameters()))
        configure_training_profile(model, "warr", "warr_joint")
        self.assertTrue(any(parameter.requires_grad for parameter in model.coarse.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.refiner.parameters()))

    def test_coordinate_warmup_and_joint_profiles_match_the_plan(self) -> None:
        model = self._model()
        configure_training_profile(model, "coord_fm", "coord_fm")
        self.assertTrue(any(parameter.requires_grad for parameter in model.velocity.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.refiner.parameters()))
        configure_training_profile(model, "coord_fm", "coord_fm_joint")
        self.assertTrue(any(parameter.requires_grad for parameter in model.velocity.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.refiner.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.coarse.parameters()))

    def test_coordinate_fm_velocity_has_real_gradient_and_frozen_parent_does_not(self) -> None:
        model = self._model()
        configure_training_profile(model, "coord_fm", "coord_fm")
        image = torch.rand(1, 3, 32, 32)
        target = torch.zeros(1, 2, 32, 32)
        yy, xx = torch.meshgrid(
            torch.arange(32, dtype=torch.float32),
            torch.arange(32, dtype=torch.float32),
            indexing="ij",
        )
        target[:, 0] = xx + 1.5
        target[:, 1] = yy - 0.75
        output = model(
            image,
            target_map=target,
            valid_mask=torch.ones(1, 1, 32, 32, dtype=torch.bool),
            render=False,
        )
        objective = (
            output["velocity_prediction"] - output["velocity_target"]
        ).square().mean()
        objective.backward()
        velocity_gradients = [
            parameter.grad
            for parameter in model.velocity.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(
            any(
                gradient is not None and bool((gradient.abs() > 0).any())
                for gradient in velocity_gradients
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.coarse.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.refiner.parameters()))

    def test_qwen_stage_backpropagates_only_through_adapter_and_fusion(self) -> None:
        model = CPDocFlow(
            coarse={"base_channels": 8, "feature_channels": 16},
            qwen_backend="lite",
            qwen={"hidden_channels": 12, "feature_layers": [-3, -2, -1]},
            qwen_feature_channels=12,
            fusion_channels=16,
            hv_channels=8,
            velocity_hidden_channels=16,
            velocity_time_channels=16,
            flow_blocks=1,
            flow_heads=4,
            fm_steps=1,
            refiner_hidden_channels=16,
            refiner_iterations=1,
        )
        with torch.no_grad():
            model.coarse.map_head[-1].weight.normal_(mean=0.0, std=0.02)
        configure_training_profile(model, "qwen", "qwen")
        output = model(torch.rand(1, 3, 32, 32), render=False)
        output["backward_map"].square().mean().backward()
        adapter_gradients = [
            parameter.grad
            for parameter in model.qwen_adapter.parameters()
            if parameter.requires_grad
        ]
        fusion_gradients = [
            parameter.grad
            for parameter in model.fusion.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(
            any(
                gradient is not None and bool((gradient.abs() > 0).any())
                for gradient in adapter_gradients
            )
        )
        self.assertTrue(
            any(
                gradient is not None and bool((gradient.abs() > 0).any())
                for gradient in fusion_gradients
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.qwen_source.parameters())
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.velocity.parameters()))

    def test_fraction_schedule_switches_at_the_exact_boundary(self) -> None:
        phases = resolve_training_phases(
            {
                "learning_rate": 1.0e-4,
                "phase_schedule": [
                    {
                        "name": "frozen",
                        "profile": "coord_fm",
                        "duration_fraction": 0.7,
                    },
                    {
                        "name": "joint",
                        "profile": "coord_fm_joint",
                        "duration_fraction": 0.3,
                        "learning_rate": 1.0e-5,
                    },
                ],
            },
            "coord_fm",
        )
        self.assertEqual(training_phase_for_step(phases, 69, 100).name, "frozen")
        self.assertEqual(training_phase_for_step(phases, 70, 100).name, "joint")
        self.assertEqual(training_phase_for_step(phases, 99, 100).learning_rate, 1.0e-5)

    def test_invalid_schedule_or_cross_stage_profile_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            resolve_training_phases(
                {
                    "phase_schedule": [
                        {
                            "profile": "warr",
                            "duration_fraction": 0.5,
                        }
                    ]
                },
                "warr",
            )
        with self.assertRaisesRegex(ValueError, "invalid for stage"):
            configure_training_profile(self._model(), "warr", "coord_fm")

    def test_optimizer_covers_both_coordinate_training_phases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = _write_split(root, "train", 2, (32, 32), 0)
            val_manifest = _write_split(root, "val", 1, (32, 32), 2)
            model_config = {
                "coarse": {"base_channels": 8, "feature_channels": 16},
                "qwen_backend": "none",
                "qwen_feature_channels": 12,
                "fusion_channels": 16,
                "hv_channels": 8,
                "velocity_hidden_channels": 16,
                "velocity_time_channels": 16,
                "flow_blocks": 2,
                "flow_heads": 4,
                "flow_window_size": 4,
                "flow_global_pool_size": 2,
                "fm_steps": 1,
                "refiner_hidden_channels": 16,
                "refiner_iterations": 1,
            }
            parent = root / "parent.pt"
            torch.save(
                full_checkpoint_payload(
                    CPDocFlow(**model_config),
                    model_config=model_config,
                    input_work_size=(32, 32),
                    output_work_size=(32, 32),
                    epoch=0,
                    training_stage="warr",
                ),
                parent,
            )
            source_config = root / "phase_smoke.yaml"
            source_config.write_text("phase schedule test\n", encoding="utf-8")
            output = root / "run"
            config = {
                "_project_root": str(root),
                "_config_path": str(source_config),
                "seed": 7,
                "data": {
                    "train_manifest": str(train_manifest),
                    "val_manifest": str(val_manifest),
                    "input_work_size": [32, 32],
                    "output_work_size": [32, 32],
                    "allowed_label_provenance": ["synthetic_analytic"],
                },
                "model": model_config,
                "train": {
                    "stage": "coord_fm",
                    "enforce_stage_gates": False,
                    "parent_checkpoint": str(parent),
                    "output_dir": str(output),
                    "device": "cpu",
                    "epochs": 1,
                    "batch_size": 1,
                    "eval_batch_size": 1,
                    "num_workers": 0,
                    "learning_rate": 1.0e-3,
                    "weight_decay": 0.0,
                    "gradient_clip": 1.0,
                    "mixed_precision": False,
                    "phase_schedule": [
                        {
                            "name": "fm_only",
                            "profile": "coord_fm",
                            "duration_fraction": 0.5,
                            "learning_rate": 1.0e-3,
                        },
                        {
                            "name": "joint",
                            "profile": "coord_fm_joint",
                            "duration_fraction": 0.5,
                            "learning_rate": 1.0e-4,
                        },
                    ],
                },
            }
            train(config)
            history = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(history[-1]["training_phase_counts"], {"fm_only": 1, "joint": 1})
            self.assertEqual(history[-1]["training_profile"], "coord_fm_joint")
            self.assertAlmostEqual(history[-1]["learning_rate"], 1.0e-4)
            run_manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreater(
                run_manifest["trainable_parameters_by_phase"]["joint"],
                run_manifest["trainable_parameters_by_phase"]["fm_only"],
            )


if __name__ == "__main__":
    unittest.main()
