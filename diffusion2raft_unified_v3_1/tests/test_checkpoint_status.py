from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "checkpoint_status.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_status", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKPOINT_STATUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKPOINT_STATUS)

from diffusion2raft.teacher_capacity_policy import (  # noqa: E402
    CANONICAL_POLICY_SHA256,
    POLICY_ID,
    POLICY_SCHEMA_VERSION,
)
from diffusion2raft.teacher_capacity_receipt import (  # noqa: E402
    build_teacher_capacity_receipt,
)


_RECEIPT_TEACHER_KEYS = (
    "sha256",
    "file_size",
    "input_size",
    "flow_size",
    "blur_kernel",
    "autocast_dtype",
    "requires_logical_cuda0",
)


def _capacity_receipt(
    identity: dict[str, object], *, teacher_overrides: dict[str, object] | None = None
) -> dict[str, object]:
    teacher = {key: identity[key] for key in _RECEIPT_TEACHER_KEYS}
    if teacher_overrides is not None:
        teacher.update(teacher_overrides)
    return build_teacher_capacity_receipt(
        evidence_report_sha256="1" * 64,
        capacity_contract_sha256="2" * 64,
        capacity_config_projection_sha256="3" * 64,
        protocol_sha256="4" * 64,
        implementation_sha256="5" * 64,
        policy={
            "id": POLICY_ID,
            "version": POLICY_SCHEMA_VERSION,
            "sha256": CANONICAL_POLICY_SHA256,
        },
        migration_seed={
            "sha256": "6" * 64,
            "stage": "unified",
            "epoch_index": 19,
            "completed_epochs": 20,
        },
        teacher=teacher,
        manifest={
            "sha256": "8" * 64,
            "split": "val",
            "record_count": 300,
        },
    )


@unittest.skipIf(torch is None, "PyTorch is required")
class CheckpointStatusTest(unittest.TestCase):
    def test_teacher_metadata_is_rejected_before_model_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher.pt"
            lama = root / "lama.pt"
            teacher.write_bytes(b"teacher")
            lama.write_bytes(b"lama")
            teacher_stat = teacher.stat()
            lama_stat = lama.stat()
            identity = {
                "version": 2,
                "resolved_path": str(teacher.resolve()),
                "file_size": teacher_stat.st_size,
                "mtime_ns": teacher_stat.st_mtime_ns,
                "sha256": hashlib.sha256(teacher.read_bytes()).hexdigest(),
                "input_size": 16,
                "flow_size": 16,
                "blur_kernel": 1,
                "autocast_dtype": "float16",
                "requires_logical_cuda0": True,
            }
            contract = {
                "version": 2,
                "teacher_prior_identity": identity,
                "inpaint": {
                    "enabled": True,
                    "path": str(lama.resolve()),
                    "size_bytes": lama_stat.st_size,
                    "mtime_ns": lama_stat.st_mtime_ns,
                    "sha256": hashlib.sha256(lama.read_bytes()).hexdigest(),
                },
            }
            payload = {
                "model": {
                    "prior._teacher_backend_marker": torch.tensor(
                        [0x54, 0x53, 0x01], dtype=torch.uint8
                    )
                },
                "optimizer": {"param_groups": [{}]},
                "stage": "unified",
                "epoch": 3,
                "prior_backend": "torchscript",
                "teacher_prior_identity": identity,
                "capacity_evidence_receipt": _capacity_receipt(identity),
                "config": {
                    "model": {
                        "prior_torchscript_sha256": identity["sha256"],
                    },
                    "inference": {
                        "inpaint": {
                            "enabled": True,
                            "sha256": contract["inpaint"]["sha256"],
                        }
                    },
                },
                "residual_application": {
                    "version": 1,
                    "origin_epoch": 0,
                    "warmup_epochs": 1,
                    "ramp_epochs": 6,
                    "max_scale": 1.0,
                    "scale": 0.5,
                },
                "deployment_contract": contract,
                "best_metric": {"name": "line_epe", "mode": "min", "value": 1.0},
            }
            checkpoint = root / "checkpoint.pt"
            torch.save(payload, checkpoint)
            summary = CHECKPOINT_STATUS.inspect_checkpoint(
                checkpoint,
                expect_stage="unified",
                require_optimizer=True,
            )
            self.assertEqual(summary[2], 4)

            missing_receipt = {
                key: value
                for key, value in payload.items()
                if key != "capacity_evidence_receipt"
            }
            torch.save(missing_receipt, checkpoint)
            with self.assertRaisesRegex(
                ValueError, "no capacity_evidence_receipt"
            ):
                CHECKPOINT_STATUS.inspect_checkpoint(
                    checkpoint,
                    expect_stage="unified",
                    require_optimizer=True,
                )

            malformed_receipt = {
                **payload,
                "capacity_evidence_receipt": {},
            }
            torch.save(malformed_receipt, checkpoint)
            with self.assertRaisesRegex(
                ValueError, "invalid capacity_evidence_receipt"
            ):
                CHECKPOINT_STATUS.inspect_checkpoint(
                    checkpoint,
                    expect_stage="unified",
                    require_optimizer=True,
                )

            binding_mismatches = {
                "sha256": "9" * 64,
                "file_size": identity["file_size"] + 1,
                "input_size": identity["input_size"] + 1,
                "flow_size": identity["flow_size"] + 1,
                "blur_kernel": identity["blur_kernel"] + 2,
                "autocast_dtype": "bfloat16",
                "requires_logical_cuda0": False,
            }
            for field, replacement in binding_mismatches.items():
                with self.subTest(binding_field=field):
                    mismatched_receipt = {
                        **payload,
                        "capacity_evidence_receipt": _capacity_receipt(
                            identity,
                            teacher_overrides={field: replacement},
                        ),
                    }
                    torch.save(mismatched_receipt, checkpoint)
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"capacity_evidence_receipt.teacher differs.*{field}",
                    ):
                        CHECKPOINT_STATUS.inspect_checkpoint(
                            checkpoint,
                            expect_stage="unified",
                            require_optimizer=True,
                        )

            torch.save(payload, checkpoint)

            legacy_identity = {**identity, "version": 1}
            legacy = {
                **payload,
                "teacher_prior_identity": legacy_identity,
                "deployment_contract": {
                    **contract,
                    "version": 1,
                    "teacher_prior_identity": legacy_identity,
                },
            }
            torch.save(legacy, checkpoint)
            with self.assertRaisesRegex(ValueError, "strict version 2"):
                CHECKPOINT_STATUS.inspect_checkpoint(
                    checkpoint,
                    expect_stage="unified",
                    require_optimizer=True,
                )

            torch.save(payload, checkpoint)

            def replace_same_size(path: Path, data: bytes, original_stat) -> None:
                self.assertEqual(len(data), original_stat.st_size)
                replacement = path.with_name(path.name + ".replacement")
                replacement.write_bytes(data)
                os.utime(
                    replacement,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
                os.replace(replacement, path)
                self.assertEqual(path.stat().st_size, original_stat.st_size)
                self.assertEqual(path.stat().st_mtime_ns, original_stat.st_mtime_ns)

            replace_same_size(teacher, b"TEACHER", teacher_stat)
            with self.assertRaisesRegex(ValueError, "teacher external file sha256"):
                CHECKPOINT_STATUS.inspect_checkpoint(
                    checkpoint,
                    expect_stage="unified",
                    require_optimizer=True,
                )
            replace_same_size(teacher, b"teacher", teacher_stat)

            replace_same_size(lama, b"LAMA", lama_stat)
            with self.assertRaisesRegex(ValueError, "LAMA external file sha256"):
                CHECKPOINT_STATUS.inspect_checkpoint(
                    checkpoint,
                    expect_stage="unified",
                    require_optimizer=True,
                )
            replace_same_size(lama, b"lama", lama_stat)

            bad = {
                **payload,
                "residual_application": {
                    **payload["residual_application"],
                    "scale": 0.75,
                },
            }
            torch.save(bad, checkpoint)
            with self.assertRaisesRegex(ValueError, "scale disagrees"):
                CHECKPOINT_STATUS.inspect_checkpoint(
                    checkpoint,
                    expect_stage="unified",
                    require_optimizer=True,
                )

            bad_identity = {**identity, "requires_logical_cuda0": 1}
            bad_contract = {
                **contract,
                "teacher_prior_identity": bad_identity,
            }
            bad = {
                **payload,
                "teacher_prior_identity": bad_identity,
                "deployment_contract": bad_contract,
            }
            torch.save(bad, checkpoint)
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                CHECKPOINT_STATUS.inspect_checkpoint(
                    checkpoint,
                    expect_stage="unified",
                    require_optimizer=True,
                )

    def test_learned_checkpoint_rejects_capacity_receipt(self) -> None:
        learned = {
            "model": {"prior.weight": torch.tensor([1.0])},
            "optimizer": {"param_groups": [{}]},
            "stage": "unified",
            "epoch": 3,
            "prior_backend": "learned",
        }
        self.assertEqual(
            CHECKPOINT_STATUS.inspect_checkpoint_payload(
                learned,
                expect_stage="unified",
                require_optimizer=True,
            ),
            ("unified", 3, 4, "-", "nan"),
        )

        identity = {
            "sha256": "7" * 64,
            "file_size": 1,
            "input_size": 16,
            "flow_size": 16,
            "blur_kernel": 1,
            "autocast_dtype": "float16",
            "requires_logical_cuda0": True,
        }
        learned_with_receipt = {
            **learned,
            "capacity_evidence_receipt": _capacity_receipt(identity),
        }
        with self.assertRaisesRegex(
            ValueError,
            "learned checkpoint contains capacity_evidence_receipt",
        ):
            CHECKPOINT_STATUS.inspect_checkpoint_payload(
                learned_with_receipt,
                expect_stage="unified",
                require_optimizer=True,
            )


if __name__ == "__main__":
    unittest.main()
