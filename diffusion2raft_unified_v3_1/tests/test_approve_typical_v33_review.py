from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import approve_typical_v33_review as approval


def _write(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path.resolve()


def _placeholder_artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": 1,
        "mtime_ns": 1,
        "sha256": "a" * 64,
    }


def _manifest_tree(
    root: Path,
) -> tuple[
    Path,
    Path,
    Path,
    dict[str, object],
    approval.ReleaseContract,
]:
    inputs = {
        name: root / name for name in ("source", "target_first", "target_second")
    }
    outputs = {
        "v33_anchor": root / "run" / "typical_final" / "run-id" / "v33_anchor",
        "v33_best": root / "run" / "typical_final" / "run-id" / "v33_best",
    }
    for directory in (*inputs.values(), *outputs.values()):
        directory.mkdir(parents=True)

    checkpoint_dir = root / "run" / "unified"
    checkpoint_paths = {
        name: _write(checkpoint_dir / f"{name}.pt", name.encode("ascii"))
        for name in ("anchor", "best", "latest")
    }
    checkpoints = {
        name: {"path": str(path)} for name, path in checkpoint_paths.items()
    }

    report_dir = root / "reports" / "run-id"
    report_dir.mkdir(parents=True)
    evaluation_path = report_dir / "evaluation_manifest.json"
    review_path = _write(report_dir / "completed_review.json", b"{}")
    report_records = {
        name: _placeholder_artifact(report_dir / filename)
        for name, filename in approval._REPORT_NAMES.items()
    }
    config_path = _write(root / "config.yaml", b"train: {}\n")
    policy_path = _write(root / "policy.yaml", b"schema_version: 1\n")
    config_record = _placeholder_artifact(config_path)
    policy_record = _placeholder_artifact(policy_path)
    inference_records = {
        candidate: _placeholder_artifact(
            outputs[candidate] / "inference_report.json"
        )
        for candidate in ("v33_anchor", "v33_best")
    }
    manifest: dict[str, object] = {
        "schema": approval.EVALUATION_SCHEMA,
        "schema_version": 1,
        "created_utc": "2026-07-23T00:00:00+00:00",
        "pipeline_status": "evaluation_complete",
        "selected_candidate": "v33_best",
        "release_ready": False,
        "run_id": "run-id",
        "expected_count": 40,
        "config": config_record,
        "inputs": {
            "warped": str(inputs["source"].resolve()),
            "target_first": str(inputs["target_first"].resolve()),
            "target_second": str(inputs["target_second"].resolve()),
        },
        "checkpoints": checkpoints,
        "outputs": {
            name: str(path.resolve()) for name, path in outputs.items()
        },
        "inference_reports": {
            "v33_anchor": {
                "checkpoint": str(checkpoint_paths["anchor"]),
                "input_count": 40,
                "success_count": 40,
                "error_count": 0,
                "artifact": inference_records["v33_anchor"],
            },
            "v33_best": {
                "checkpoint": str(checkpoint_paths["best"]),
                "input_count": 40,
                "success_count": 40,
                "error_count": 0,
                "artifact": inference_records["v33_best"],
            },
        },
        "reports": report_records,
        "automatic_quality_gate": {
            "status": "passed",
            "policy": policy_record,
            "failure_count": 0,
            "failures": [],
            "summary": {"passed_gate_count": 10},
        },
        "manual_quality_review": {
            "status": "pending",
            "candidate": "v33_best",
            "evidence_sha256": "b" * 64,
            "required_reviewed_count": 40,
        },
        "quality_proxy": {"metric": "proxy"},
    }
    evaluation_path.write_text(json.dumps(manifest), encoding="utf-8")
    contract = approval.ReleaseContract(
        config_path=config_path,
        quality_policy_path=policy_path,
        source_dir=inputs["source"].resolve(),
        target_first_dir=inputs["target_first"].resolve(),
        target_second_dir=inputs["target_second"].resolve(),
        run_root=(root / "run").resolve(),
        report_root=(root / "reports").resolve(),
    )
    return (
        evaluation_path.resolve(),
        review_path,
        (root / "run").resolve(),
        manifest,
        contract,
    )


class ApproveTypicalV33ReviewTest(unittest.TestCase):
    def test_script_has_no_duplicate_constant_dict_literal_keys(self) -> None:
        tree = ast.parse(
            (SCRIPTS / "approve_typical_v33_review.py").read_text(
                encoding="utf-8"
            )
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, (str, int, float, bytes))
            ]
            self.assertEqual(
                len(keys),
                len(set(keys)),
                f"duplicate dict literal key at line {node.lineno}",
            )

    def test_evaluation_v1_envelope_is_strict_and_fixed_to_passed_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluation, _, run_root, manifest, contract = _manifest_tree(Path(temporary))
            self.assertEqual(
                approval.validate_evaluation_manifest(
                    manifest, manifest_path=evaluation, contract=contract
                ),
                run_root,
            )

            cases: list[tuple[str, dict[str, object], str]] = []
            wrong_version = copy.deepcopy(manifest)
            wrong_version["schema_version"] = 2
            cases.append(("version", wrong_version, "schema_version"))
            bool_version = copy.deepcopy(manifest)
            bool_version["schema_version"] = True
            cases.append(("bool-version", bool_version, "schema_version"))
            wrong_candidate = copy.deepcopy(manifest)
            wrong_candidate["selected_candidate"] = "v33_anchor"
            cases.append(("candidate", wrong_candidate, "selected_candidate"))
            already_ready = copy.deepcopy(manifest)
            already_ready["release_ready"] = True
            cases.append(("ready", already_ready, "release_ready"))
            failed_gate = copy.deepcopy(manifest)
            failed_gate["automatic_quality_gate"]["status"] = "failed"
            cases.append(("failed", failed_gate, "status"))
            hidden_failure = copy.deepcopy(manifest)
            hidden_failure["automatic_quality_gate"]["failures"] = [{"code": "x"}]
            cases.append(("failure-row", hidden_failure, "failures"))
            changed_checkpoint = copy.deepcopy(manifest)
            changed_checkpoint["inference_reports"]["v33_best"]["checkpoint"] = str(
                Path(temporary) / "wrong.pt"
            )
            cases.append(("checkpoint", changed_checkpoint, "checkpoint"))
            extra_field = copy.deepcopy(manifest)
            extra_field["unexpected"] = True
            cases.append(("extra", extra_field, "extra"))
            substituted_policy = copy.deepcopy(manifest)
            substituted_policy["automatic_quality_gate"]["policy"]["path"] = str(
                _write(Path(temporary) / "loose_policy.yaml", b"loose: true\n")
            )
            cases.append(("policy-substitution", substituted_policy, "policy 非 canonical"))
            alternate_input = Path(temporary) / "easy_inputs"
            alternate_input.mkdir()
            substituted_input = copy.deepcopy(manifest)
            substituted_input["inputs"]["warped"] = str(alternate_input.resolve())
            cases.append(("input-substitution", substituted_input, "canonical typical roots"))
            unsafe_run_id = copy.deepcopy(manifest)
            unsafe_run_id["run_id"] = " ../escape "
            cases.append(("unsafe-run-id", unsafe_run_id, "安全路径组件"))
            non_utc = copy.deepcopy(manifest)
            non_utc["created_utc"] = "2026-07-23T08:00:00+08:00"
            cases.append(("non-utc", non_utc, "使用 UTC"))

            for label, candidate, pattern in cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(approval.ApprovalError, pattern):
                        approval.validate_evaluation_manifest(
                            candidate,
                            manifest_path=evaluation,
                            contract=contract,
                        )

    def test_artifact_record_rehashes_stat_and_sha_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _write(root / "artifact.json", b"first")
            record = approval._current_artifact(artifact, label="artifact")
            self.assertEqual(
                approval.verify_artifact_record(record, label="artifact"),
                artifact,
            )

            artifact.write_bytes(b"other")
            with self.assertRaisesRegex(approval.ApprovalError, "stat/SHA"):
                approval.verify_artifact_record(record, label="artifact")

            link = root / "link.json"
            link.symlink_to(artifact)
            with self.assertRaisesRegex(approval.ApprovalError, "规范化真实路径"):
                approval.verify_artifact_record(
                    {**record, "path": str(link)}, label="link"
                )

    def test_bound_json_uses_one_fd_and_detects_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = _write(root / "document.json", b'{"value": 1}')
            replacement = _write(root / "replacement.json", b'{"value": 2}')
            original_read = approval.os.read
            swapped = False

            def swap_after_read(descriptor: int, size: int) -> bytes:
                nonlocal swapped
                payload = original_read(descriptor, size)
                if payload and not swapped:
                    swapped = True
                    document.rename(root / "old.json")
                    replacement.rename(document)
                return payload

            with mock.patch.object(approval.os, "read", side_effect=swap_after_read):
                with self.assertRaisesRegex(
                    approval.ApprovalError,
                    "同一文件描述符读取期间|文件描述符与当前路径实体",
                ):
                    approval._load_bound_json(document, label="document")

            duplicate = _write(root / "duplicate.json", b'{"x": 1, "x": 1}')
            with self.assertRaisesRegex(approval.ApprovalError, "duplicate JSON field"):
                approval._load_bound_json(duplicate, label="duplicate")

    def test_checkpoint_final_sha_rejects_same_size_with_restored_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = _write(root / "best.pt", b"first")
            original_stat = checkpoint.stat()
            original = approval._current_artifact(
                checkpoint, label="best checkpoint"
            )
            state = approval.ApprovalState(
                manifest={},
                evaluation_path=checkpoint,
                evaluation_artifact={},
                review_path=checkpoint,
                review_artifact={},
                review={},
                review_summary={},
                evidence={},
                evidence_status={},
                checkpoint_summaries={"best": dict(original)},
                bound_artifacts=(),
                output_snapshots={},
                quality_report={},
                run_root=root,
            )
            approval._verify_checkpoint_stats(state)

            checkpoint.write_bytes(b"other")
            os.utime(
                checkpoint,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            self.assertEqual(checkpoint.stat().st_size, original["size_bytes"])
            self.assertEqual(checkpoint.stat().st_mtime_ns, original["mtime_ns"])
            with self.assertRaisesRegex(approval.ApprovalError, "artifact"):
                approval._verify_checkpoint_stats(state)

    def test_final_external_recheck_rejects_teacher_and_lama_replacement(self) -> None:
        def identity(path: Path, *, teacher: bool) -> dict[str, object]:
            item: dict[str, object] = {
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if teacher:
                item.update(
                    version=2,
                    resolved_path=str(path.resolve()),
                    file_size=path.stat().st_size,
                )
            else:
                item.update(
                    enabled=True,
                    path=str(path.resolve()),
                    size_bytes=path.stat().st_size,
                )
            return item

        def replace_same_stat(path: Path, payload: bytes) -> None:
            original = path.stat()
            replacement = path.with_suffix(path.suffix + ".replacement")
            replacement.write_bytes(payload)
            os.utime(
                replacement,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )
            os.replace(replacement, path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = _write(root / "teacher.pt", b"teacher")
            lama = _write(root / "lama.pt", b"lama")
            teacher_identity = identity(teacher, teacher=True)
            lama_identity = identity(lama, teacher=False)
            state = approval.ApprovalState(
                manifest={},
                evaluation_path=teacher,
                evaluation_artifact={},
                review_path=teacher,
                review_artifact={},
                review={},
                review_summary={},
                evidence={},
                evidence_status={},
                checkpoint_summaries={
                    "best": {
                        "teacher_prior_identity": teacher_identity,
                        "deployment_contract": {
                            "version": 2,
                            "inpaint": lama_identity,
                        },
                    }
                },
                bound_artifacts=(),
                output_snapshots={},
                quality_report={},
                run_root=root,
            )
            approval._verify_external_model_identities(state)

            with self.subTest("teacher"):
                replace_same_stat(teacher, b"TEACHER")
                with self.assertRaisesRegex(approval.ApprovalError, "teacher.*sha256"):
                    approval._verify_external_model_identities(state)

            teacher.write_bytes(b"teacher")
            os.utime(
                teacher,
                ns=(
                    teacher.stat().st_atime_ns,
                    int(teacher_identity["mtime_ns"]),
                ),
            )
            approval._verify_external_model_identities(state)
            with self.subTest("LAMA"):
                replace_same_stat(lama, b"LAMA")
                with self.assertRaisesRegex(approval.ApprovalError, "LAMA.*sha256"):
                    approval._verify_external_model_identities(state)

    def test_line_report_is_recomputed_and_complete_typed_report_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / name for name in ("source", "first", "second", "anchor", "best")]
            for directory in directories:
                directory.mkdir()
            stored = {
                "schema_version": 1,
                "metric_family": "opencv_lsd_axis_alignment",
                "config": {"max_dimension": 1600},
                "candidates": {"sentinel": {"directory": "/fixed"}},
            }
            weighted = {"v33_best_full": 1.0}
            with mock.patch.object(
                approval, "evaluate_dataset", return_value=copy.deepcopy(stored)
            ) as evaluator, mock.patch.object(
                approval, "validate_line_report", return_value=weighted
            ):
                current, result = approval.recompute_line_report(
                    stored,
                    input_dir=directories[0],
                    target_first_dir=directories[1],
                    target_second_dir=directories[2],
                    anchor_output=directories[3],
                    best_output=directories[4],
                )
            self.assertEqual(current, stored)
            self.assertEqual(result, weighted)
            self.assertEqual(evaluator.call_count, 1)
            candidates = evaluator.call_args.args[1]
            self.assertEqual(
                tuple(candidates),
                (
                    "warped",
                    "target_first",
                    "target_second",
                    "v33_anchor_full",
                    "v33_best_full",
                    "v33_anchor_valid",
                    "v33_best_valid",
                ),
            )
            self.assertEqual(
                evaluator.call_args.kwargs["valid_masks"],
                {
                    "v33_anchor_valid": directories[3],
                    "v33_best_valid": directories[4],
                },
            )

            changed = copy.deepcopy(stored)
            changed["config"]["max_dimension"] = 999
            with mock.patch.object(
                approval, "evaluate_dataset", return_value=changed
            ), mock.patch.object(
                approval, "validate_line_report", return_value=weighted
            ):
                with self.assertRaisesRegex(approval.ApprovalError, "重算结果"):
                    approval.recompute_line_report(
                        stored,
                        input_dir=directories[0],
                        target_first_dir=directories[1],
                        target_second_dir=directories[2],
                        anchor_output=directories[3],
                        best_output=directories[4],
                    )

    def test_quality_gate_is_recomputed_and_must_match_report_and_manifest(self) -> None:
        current = {
            "passed": True,
            "failures": [],
            "summary": {"passed_gate_count": 12},
        }
        policy_record = _placeholder_artifact(Path("/policy.yaml"))
        manifest_gate = {
            "status": "passed",
            "policy": policy_record,
            "failure_count": 0,
            "failures": [],
            "summary": current["summary"],
        }
        inference = {
            candidate: {"_finalizer_validation": {"per_image": {}}}
            for candidate in ("v33_anchor", "v33_best")
        }
        with mock.patch.object(
            approval, "evaluate_typical_quality", return_value=current
        ) as evaluator, mock.patch.object(
            approval, "_quality_validation_inputs", return_value={}
        ):
            result = approval.recompute_quality_gate(
                policy=object(),
                checkpoints={},
                line_report={},
                inference=inference,
                stored_report=copy.deepcopy(current),
                manifest_gate=manifest_gate,
            )
        self.assertEqual(result, current)
        self.assertEqual(evaluator.call_count, 1)

        stale = copy.deepcopy(current)
        stale["summary"]["passed_gate_count"] = 11
        with mock.patch.object(
            approval, "evaluate_typical_quality", return_value=current
        ), mock.patch.object(
            approval, "_quality_validation_inputs", return_value={}
        ):
            with self.assertRaisesRegex(approval.ApprovalError, "重算结果"):
                approval.recompute_quality_gate(
                    policy=object(),
                    checkpoints={},
                    line_report={},
                    inference=inference,
                    stored_report=stale,
                    manifest_gate=manifest_gate,
                )

        failed = {"passed": False, "failures": [{"code": "x"}], "summary": {}}
        failed_manifest = {
            "status": "passed",
            "policy": policy_record,
            "failure_count": 0,
            "failures": [],
            "summary": {},
        }
        with mock.patch.object(
            approval, "evaluate_typical_quality", return_value=failed
        ), mock.patch.object(
            approval, "_quality_validation_inputs", return_value={}
        ):
            with self.assertRaisesRegex(approval.ApprovalError, "未通过"):
                approval.recompute_quality_gate(
                    policy=object(),
                    checkpoints={},
                    line_report={},
                    inference=inference,
                    stored_report=failed,
                    manifest_gate=failed_manifest,
                )

    def test_evidence_roots_are_bound_to_this_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, manifest, _ = _manifest_tree(root)
            expected = {
                "source": manifest["inputs"]["warped"],
                "target_first": manifest["inputs"]["target_first"],
                "target_second": manifest["inputs"]["target_second"],
                "v33_anchor": manifest["outputs"]["v33_anchor"],
                "v33_best": manifest["outputs"]["v33_best"],
            }
            inventories: dict[str, list[dict[str, str]]] = {}
            for name in approval.INVENTORY_NAMES:
                if name == "source":
                    directory = Path(expected["source"])
                elif name in {"target_first", "target_second"}:
                    directory = Path(expected[name])
                elif name.startswith("v33_anchor_"):
                    directory = Path(expected["v33_anchor"])
                else:
                    directory = Path(expected["v33_best"])
                file_path = _write(directory / f"{name}.png")
                inventories[name] = [{"path": str(file_path)}]
            evidence = {"roots": expected, "inventories": inventories}
            approval._validate_evidence_roots(evidence, manifest)

            changed = copy.deepcopy(evidence)
            changed["roots"]["v33_best"] = changed["roots"]["v33_anchor"]
            with self.assertRaisesRegex(approval.ApprovalError, "evidence roots"):
                approval._validate_evidence_roots(changed, manifest)

    def test_final_check_revalidates_files_and_rejects_pending_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = _write(root / "evaluation_manifest.json", b"evaluation")
            review = _write(root / "review.json", b"review")
            state = approval.ApprovalState(
                manifest={"outputs": {}},
                evaluation_path=evaluation,
                evaluation_artifact=approval._current_artifact(evaluation, label="evaluation"),
                review_path=review,
                review_artifact=approval._current_artifact(review, label="review"),
                review={"status": "pending"},
                review_summary={},
                evidence={"evidence_sha256": "a" * 64},
                evidence_status={},
                checkpoint_summaries={},
                bound_artifacts=(),
                output_snapshots={},
                quality_report={},
                run_root=root,
            )
            with mock.patch.object(
                approval,
                "validate_typical_evidence",
                return_value={"status": "valid"},
            ) as evidence_validator, mock.patch.object(
                approval,
                "validate_completed_typical_review",
                side_effect=approval.TypicalReviewError("pending"),
            ):
                with self.assertRaisesRegex(approval.ApprovalError, "最终复核"):
                    approval.final_toctou_check(state)
            evidence_validator.assert_called_once_with(
                state.evidence, verify_files=True
            )

    def test_final_check_rejects_changed_nonvisual_inference_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = _write(root / "evaluation_manifest.json", b"evaluation")
            review = _write(root / "review.json", b"review")
            output = root / "v33_best"
            sidecar = _write(output / "Page_metadata.json", b'{"fold_rate": 0}')
            snapshot = approval._snapshot_output_directory(
                output, label="v33_best"
            )
            sidecar.write_bytes(b'{"fold_rate": 1}')
            state = approval.ApprovalState(
                manifest={"outputs": {"v33_best": str(output.resolve())}},
                evaluation_path=evaluation,
                evaluation_artifact=approval._current_artifact(evaluation, label="evaluation"),
                review_path=review,
                review_artifact=approval._current_artifact(review, label="review"),
                review={},
                review_summary={},
                evidence={},
                evidence_status={},
                checkpoint_summaries={},
                bound_artifacts=(),
                output_snapshots={"v33_best": snapshot},
                quality_report={},
                run_root=root,
            )
            with self.assertRaisesRegex(approval.ApprovalError, "approval snapshot"):
                approval.final_toctou_check(state)

    def test_persisted_output_binding_verifier_catches_post_release_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "v33_best"
            sidecar = _write(output / "Page_residual_backward_flow.npy", b"flow-v1")
            files = approval._snapshot_output_directory(output, label="v33_best")
            binding = {
                "directory": str(output.resolve()),
                "file_count": len(files),
                "inventory_sha256": approval._inventory_sha256(files),
                "files": files,
            }
            status = approval.verify_inference_output_binding(
                binding, label="v33_best"
            )
            self.assertEqual(status["status"], "valid")
            sidecar.write_bytes(b"flow-v2")
            with self.assertRaisesRegex(approval.ApprovalError, "stat/SHA"):
                approval.verify_inference_output_binding(
                    binding, label="v33_best"
                )

    def test_approval_atomically_creates_once_and_binds_all_digests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, review, run_root, manifest, contract = _manifest_tree(root)

            def make_state(
                current_manifest: dict[str, object], **kwargs: object
            ) -> approval.ApprovalState:
                return approval.ApprovalState(
                    manifest=current_manifest,
                    evaluation_path=evaluation,
                    evaluation_artifact=kwargs["evaluation_artifact"],
                    review_path=review,
                    review_artifact=approval._current_artifact(review, label="review"),
                    review={"candidate": "v33_best"},
                    review_summary={"status": "passed"},
                    evidence={"evidence_sha256": "b" * 64},
                    evidence_status={"status": "valid"},
                    checkpoint_summaries={},
                    bound_artifacts=(),
                    output_snapshots={"v33_anchor": {}, "v33_best": {}},
                    quality_report={"summary": {"passed_gate_count": 10}},
                    run_root=run_root,
                )

            with mock.patch.object(
                approval, "validate_release_inputs", side_effect=make_state
            ) as validator, mock.patch.object(
                approval,
                "final_toctou_check",
                return_value=(
                    {"status": "valid"},
                    {"status": "passed", "reviewed_count": 40},
                ),
            ):
                payload, output = approval.approve_typical_review(
                    evaluation, review, contract=contract
                )

            self.assertEqual(output, evaluation.parent / "final_manifest.json")
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertIs(payload["release_ready"], True)
            self.assertEqual(payload["selected_candidate"], "v33_best")
            self.assertEqual(
                payload["bindings"]["evaluation_manifest_sha256"],
                approval._current_artifact(evaluation, label="evaluation")["sha256"],
            )
            self.assertEqual(
                payload["bindings"]["completed_review_sha256"],
                approval._current_artifact(review, label="review")["sha256"],
            )
            self.assertEqual(payload["bindings"]["evidence_sha256"], "b" * 64)
            self.assertEqual(
                set(payload["inference_output_bindings"]),
                {"v33_anchor", "v33_best"},
            )
            self.assertTrue((run_root / approval.RUN_LOCK_NAME).is_file())
            self.assertEqual(validator.call_count, 1)

            original = output.read_bytes()
            with mock.patch.object(
                approval, "validate_release_inputs", side_effect=AssertionError("must not run")
            ):
                with self.assertRaisesRegex(approval.ApprovalError, "拒绝覆盖"):
                    approval.approve_typical_review(
                        evaluation, review, contract=contract
                    )
            self.assertEqual(output.read_bytes(), original)

    def test_atomic_create_never_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "final_manifest.json"
            approval._atomic_create_json({"release_ready": True}, output)
            first = output.read_bytes()
            with self.assertRaisesRegex(approval.ApprovalError, "拒绝覆盖"):
                approval._atomic_create_json({"release_ready": False}, output)
            self.assertEqual(output.read_bytes(), first)

    def test_real_lock_bound_read_and_durable_publish_primitives_compose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            evaluation = _write(
                root / "evaluation_manifest.json",
                b'{"schema_version": 1, "release_ready": false}',
            )
            expected = approval._current_artifact(
                evaluation, label="evaluation"
            )
            with approval._exclusive_lock(root / approval.RUN_LOCK_NAME):
                payload, artifact = approval._load_bound_json(
                    evaluation,
                    label="evaluation",
                    expected_artifact=expected,
                )
                output = approval._atomic_create_json(payload, root / "final_manifest.json")
            published, _ = approval._load_bound_json(
                output, label="final manifest"
            )
            self.assertEqual(artifact, expected)
            self.assertEqual(published, payload)

    def test_atomic_create_rejects_dangling_symlink_race_and_fsyncs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "final_manifest.json"
            redirected = root / "redirected" / "release.json"
            real_link = approval.os.link

            def race_link(*args: object, **kwargs: object) -> None:
                output.symlink_to(redirected)
                real_link(*args, **kwargs)

            with mock.patch.object(approval.os, "link", side_effect=race_link):
                with self.assertRaisesRegex(approval.ApprovalError, "拒绝覆盖"):
                    approval._atomic_create_json({"release_ready": True}, output)
            self.assertTrue(output.is_symlink())
            self.assertFalse(redirected.exists())

            output.unlink()
            fsync_kinds: list[str] = []
            real_fsync = approval.os.fsync

            def record_fsync(descriptor: int) -> None:
                mode = os.fstat(descriptor).st_mode
                fsync_kinds.append(
                    "file" if stat.S_ISREG(mode) else "directory"
                )
                real_fsync(descriptor)

            with mock.patch.object(approval.os, "fsync", side_effect=record_fsync):
                approval._atomic_create_json({"release_ready": True}, output)
            self.assertGreaterEqual(len(fsync_kinds), 2)
            self.assertEqual(fsync_kinds[0], "file")
            self.assertEqual(fsync_kinds[-1], "directory")

    def test_report_logical_leaf_symlink_is_not_hidden_by_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = _write(root / "target.json", b"{}")
            logical = root / "all40_line_proxy.json"
            logical.symlink_to(target)
            record = approval._current_artifact(target, label="target")
            with self.assertRaisesRegex(approval.ApprovalError, "logical leaf"):
                approval.verify_artifact_record(
                    record,
                    label="line report",
                    expected_path=logical,
                )


if __name__ == "__main__":
    unittest.main()
