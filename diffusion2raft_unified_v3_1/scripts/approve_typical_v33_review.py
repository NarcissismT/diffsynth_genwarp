#!/usr/bin/env python3
"""Approve a completed v3.3 typical review without trusting stale artifacts.

``finalize_typical_v33.py`` deliberately stops after producing an automatic
evaluation and a pending human-review template.  This is the second, release
approval stage.  It re-validates the complete evaluation against the current
files, re-runs the LSD evaluator and the frozen automatic policy, and accepts
only a fully passed review bound to unchanged all-40 evidence.

The sole durable output is a new ``final_manifest.json`` next to the supplied
``evaluation_manifest.json``.  An existing final manifest is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat as stat_module
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for import_root in (str(SCRIPT_DIR), str(SRC_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from checkpoint_status import _typed_equal
from diffusion2raft.external_file import (
    ExternalFileIdentityError,
    validate_external_file_identity,
)
from diffusion2raft.typical_quality_v2 import (
    QualityGateInputError,
    evaluate_typical_quality,
    parse_quality_policy_yaml,
)
from diffusion2raft.typical_review import (
    CANDIDATE_NAME,
    INVENTORY_NAMES,
    TYPICAL_IMAGE_COUNT,
    TypicalReviewError,
    make_typical_review_template,
    validate_completed_typical_review,
    validate_typical_evidence,
)
from evaluate_typical_lines import evaluate_dataset
from finalize_typical_v33 import (
    FinalizationError,
    _exclusive_lock,
    _quality_validation_inputs,
    checkpoint_summary,
    inference_critical_config,
    validate_checkpoint_set,
    validate_comparison_report,
    validate_image_sets,
    validate_inference_output,
    validate_line_report,
)


EVALUATION_SCHEMA = "diffusion2raft.typical_evaluation_manifest"
EVALUATION_SCHEMA_VERSION = 1
FINAL_SCHEMA = "diffusion2raft.typical_release_manifest"
FINAL_SCHEMA_VERSION = 1
FINAL_MANIFEST_NAME = "final_manifest.json"
RUN_LOCK_NAME = ".finalize_typical_v33.lock"

CANONICAL_CONFIG = REPO_ROOT / "configs" / "unified_v3_3_teacher_anchor.yaml"
CANONICAL_QUALITY_POLICY = REPO_ROOT / "configs" / "typical_v33_quality_v2.yaml"
CANONICAL_RUN_ROOT = REPO_ROOT / "runs" / "d2r_v3_3_teacher_anchor"
CANONICAL_REPORT_ROOT = REPO_ROOT / "reports" / "typical_v33_teacher_anchor"
CANONICAL_SOURCE = Path(
    "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/"
    "test_silver_bullet_imgs/typical"
)
CANONICAL_TARGET_FIRST = Path(
    "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/"
    "test_silver_bullet_imgs/"
    "typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999"
)
CANONICAL_TARGET_SECOND = Path(
    "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/"
    "test_silver_bullet_imgs/"
    "typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999-2nd"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ARTIFACT_FIELDS = frozenset({"path", "size_bytes", "mtime_ns", "sha256"})
_EVALUATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "created_utc",
        "pipeline_status",
        "selected_candidate",
        "release_ready",
        "run_id",
        "expected_count",
        "config",
        "inputs",
        "checkpoints",
        "outputs",
        "inference_reports",
        "reports",
        "automatic_quality_gate",
        "manual_quality_review",
        "quality_proxy",
    }
)
_REPORT_NAMES = {
    "line_proxy": "all40_line_proxy.json",
    "comparison_json": "all40_comparison.json",
    "comparison_html": "all40_comparison.html",
    "automatic_quality_gate": "automatic_quality_gate.json",
    "review_evidence": "review_evidence.json",
    "quality_review_template": "quality_review_template.json",
}
_QUALITY_PROXY_NOTICE = (
    "LSD axis alignment is a no-reference proxy and cannot prove "
    "rectification quality or content fidelity. Direct target comparisons "
    "below use full-frame v3.3 scores; valid-mask scores are separate "
    "diagnostics on a smaller support."
)


class ApprovalError(RuntimeError):
    """A fail-closed release-approval contract violation."""


@dataclass(frozen=True)
class ReleaseContract:
    """Independent trust roots for the one production v3.3 release layout."""

    config_path: Path
    quality_policy_path: Path
    source_dir: Path
    target_first_dir: Path
    target_second_dir: Path
    run_root: Path
    report_root: Path

    @classmethod
    def canonical(cls) -> "ReleaseContract":
        return cls(
            config_path=CANONICAL_CONFIG.resolve(strict=False),
            quality_policy_path=CANONICAL_QUALITY_POLICY.resolve(strict=False),
            source_dir=CANONICAL_SOURCE.resolve(strict=False),
            target_first_dir=CANONICAL_TARGET_FIRST.resolve(strict=False),
            target_second_dir=CANONICAL_TARGET_SECOND.resolve(strict=False),
            run_root=CANONICAL_RUN_ROOT.resolve(strict=False),
            report_root=CANONICAL_REPORT_ROOT.resolve(strict=False),
        )

    def checkpoints(self) -> dict[str, Path]:
        return {
            name: self.run_root / "unified" / f"{name}.pt"
            for name in ("anchor", "best", "latest")
        }

    def report_dir(self, run_id: str) -> Path:
        return self.report_root / run_id

    def output_dirs(self, run_id: str) -> dict[str, Path]:
        root = self.run_root / "typical_final" / run_id
        return {
            "v33_anchor": root / "v33_anchor",
            "v33_best": root / "v33_best",
        }


@dataclass(frozen=True)
class ApprovalState:
    """Validated state retained for the final, under-lock TOCTOU check."""

    manifest: Mapping[str, Any]
    evaluation_path: Path
    evaluation_artifact: Mapping[str, Any]
    review_path: Path
    review_artifact: Mapping[str, Any]
    review: Mapping[str, Any]
    review_summary: Mapping[str, Any]
    evidence: Mapping[str, Any]
    evidence_status: Mapping[str, Any]
    checkpoint_summaries: Mapping[str, Mapping[str, Any]]
    bound_artifacts: tuple[tuple[str, Mapping[str, Any]], ...]
    output_snapshots: Mapping[str, Mapping[str, Mapping[str, Any]]]
    quality_report: Mapping[str, Any]
    run_root: Path


def _field_names(value: Mapping[Any, Any], *, label: str) -> set[str]:
    if not all(isinstance(key, str) for key in value):
        raise ApprovalError(f"{label} 含非字符串字段名")
    return set(value)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ApprovalError(f"{label} 必须是 mapping")
    return value


def _exact_fields(
    value: Mapping[Any, Any], expected: frozenset[str] | set[str], *, label: str
) -> None:
    actual = _field_names(value, label=label)
    expected_set = set(expected)
    if actual != expected_set:
        raise ApprovalError(
            f"{label} 字段不符合 schema；"
            f"missing={sorted(expected_set - actual)}, "
            f"extra={sorted(actual - expected_set)}"
        )


def _exact_int(value: Any, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ApprovalError(f"{label} 必须严格为整数 {expected}，实际为 {value!r}")


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ApprovalError(f"{label} 必须是非空字符串")
    return value


def _normalized_absolute_path(value: Any, *, label: str) -> Path:
    raw = _nonempty_string(value, label=label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ApprovalError(f"{label} 必须是绝对路径：{raw!r}")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ApprovalError(f"{label} 无法解析：{path}: {error}") from error
    if str(resolved) != raw:
        raise ApprovalError(
            f"{label} 必须是规范化真实路径；recorded={raw!r}, resolved={str(resolved)!r}"
        )
    return resolved


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable_file(
    path: Path, *, label: str, capture_bytes: bool
) -> tuple[dict[str, Any], bytes | None]:
    """Hash bytes from one no-follow fd and bind that fd to the path entry."""

    path = Path(path)
    if not path.is_absolute():
        raise ApprovalError(f"{label} 路径必须是绝对路径：{path}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise ApprovalError("当前平台不支持安全读取所需的 O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ApprovalError(f"无法安全打开 {label}：{path}: {error}") from error
    chunks: list[bytes] | None = [] if capture_bytes else None
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode):
            raise ApprovalError(f"{label} 必须是普通文件：{path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    except ApprovalError:
        raise
    except OSError as error:
        raise ApprovalError(f"读取 {label} 失败：{path}: {error}") from error
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise ApprovalError(f"{label} 在同一文件描述符读取期间发生变化：{path}")
    if before.st_size <= 0:
        raise ApprovalError(f"{label} 为空：{path}")
    try:
        path_stat = path.lstat()
    except OSError as error:
        raise ApprovalError(f"读取后无法检查 {label} 路径：{path}: {error}") from error
    if stat_module.S_ISLNK(path_stat.st_mode) or not stat_module.S_ISREG(path_stat.st_mode):
        raise ApprovalError(f"{label} 路径不是普通非符号链接文件：{path}")
    if _stat_identity(path_stat) != _stat_identity(after):
        raise ApprovalError(f"{label} 文件描述符与当前路径实体不一致：{path}")
    try:
        canonical = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ApprovalError(f"无法解析 {label}：{path}: {error}") from error
    if canonical != path:
        raise ApprovalError(f"{label} 路径不是规范真实路径：{path}")
    record = {
        "path": str(path),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }
    payload = b"".join(chunks) if chunks is not None else None
    return record, payload


def _current_artifact(path: Path, *, label: str) -> dict[str, Any]:
    record, _ = _read_stable_file(path, label=label, capture_bytes=False)
    return record


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _load_bound_json(
    path: Path,
    *,
    label: str,
    expected_artifact: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load JSON only when its before/after stat and digest are identical."""

    before, raw = _read_stable_file(path, label=label, capture_bytes=True)
    if expected_artifact is not None:
        _assert_typed_equal(
            before,
            dict(expected_artifact),
            label=f"{label} artifact",
        )
    assert raw is not None
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_fields,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ApprovalError(f"{label} 不是严格 JSON：{error}") from error
    return payload, before


def verify_artifact_record(
    record: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    """Re-hash a finalizer artifact record and require typed exact equality."""

    artifact = _mapping(record, label=label)
    _exact_fields(artifact, _ARTIFACT_FIELDS, label=label)
    path = _normalized_absolute_path(artifact.get("path"), label=f"{label}.path")
    if expected_path is not None:
        expected_parent = expected_path.parent.resolve(strict=True)
        expected_literal = expected_parent / expected_path.name
        if path != expected_literal:
            raise ApprovalError(
                f"{label}.path 错误或 logical leaf 是 symlink；"
                f"expected={expected_literal}, actual={path}"
            )
    size = artifact.get("size_bytes")
    mtime = artifact.get("mtime_ns")
    digest = artifact.get("sha256")
    if type(size) is not int or size <= 0:
        raise ApprovalError(f"{label}.size_bytes 必须是正整数")
    if type(mtime) is not int or mtime < 0:
        raise ApprovalError(f"{label}.mtime_ns 必须是非负整数")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ApprovalError(f"{label}.sha256 不是规范 SHA-256")
    current = _current_artifact(path, label=label)
    if not _typed_equal(dict(artifact), current):
        raise ApprovalError(
            f"{label} 文件 stat/SHA 已变化；recorded={dict(artifact)!r}, current={current!r}"
        )
    return path


def _derive_run_root(checkpoints: Mapping[str, Any]) -> Path:
    parents: set[Path] = set()
    paths: list[Path] = []
    for name in ("anchor", "best", "latest"):
        summary = _mapping(checkpoints.get(name), label=f"checkpoints.{name}")
        path = _normalized_absolute_path(
            summary.get("path"), label=f"checkpoints.{name}.path"
        )
        if not path.is_file():
            raise ApprovalError(f"checkpoint 不存在：{path}")
        paths.append(path)
        parents.add(path.parent)
    if len(parents) != 1:
        raise ApprovalError(
            "anchor/best/latest checkpoint 必须位于同一个 run checkpoint 目录"
        )
    checkpoint_dir = next(iter(parents))
    run_root = checkpoint_dir.parent.resolve()
    if run_root == Path(run_root.anchor) or not run_root.is_dir():
        raise ApprovalError(f"无法安全确定 run-root：{run_root}")
    if len(set(paths)) != 3:
        raise ApprovalError("anchor/best/latest checkpoint 路径必须互不相同")
    return run_root


def validate_evaluation_manifest(
    manifest: Any,
    *,
    manifest_path: Path,
    contract: ReleaseContract | None = None,
) -> Path:
    """Validate the exact v1 release-candidate envelope and derive run-root."""

    contract = contract or ReleaseContract.canonical()
    root = _mapping(manifest, label="evaluation manifest")
    _exact_fields(root, _EVALUATION_FIELDS, label="evaluation manifest")
    if root.get("schema") != EVALUATION_SCHEMA:
        raise ApprovalError(f"evaluation schema 必须为 {EVALUATION_SCHEMA!r}")
    _exact_int(
        root.get("schema_version"),
        EVALUATION_SCHEMA_VERSION,
        label="evaluation.schema_version",
    )
    if root.get("pipeline_status") != "evaluation_complete":
        raise ApprovalError("evaluation.pipeline_status 必须为 evaluation_complete")
    if root.get("selected_candidate") != CANDIDATE_NAME:
        raise ApprovalError(f"selected_candidate 必须为 {CANDIDATE_NAME!r}")
    if root.get("release_ready") is not False:
        raise ApprovalError("evaluation.release_ready 必须严格为 false")
    _exact_int(
        root.get("expected_count"),
        TYPICAL_IMAGE_COUNT,
        label="evaluation.expected_count",
    )
    created_utc = _nonempty_string(
        root.get("created_utc"), label="evaluation.created_utc"
    )
    try:
        parsed_created = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    except ValueError as error:
        raise ApprovalError("evaluation.created_utc 必须是 RFC3339 时间") from error
    if (
        parsed_created.tzinfo is None
        or parsed_created.utcoffset() != timezone.utc.utcoffset(parsed_created)
    ):
        raise ApprovalError("evaluation.created_utc 必须明确使用 UTC")
    run_id = _nonempty_string(root.get("run_id"), label="evaluation.run_id")
    if _RUN_ID_RE.fullmatch(run_id) is None or run_id in {".", ".."}:
        raise ApprovalError("evaluation.run_id 必须是单个安全路径组件")

    inputs = _mapping(root.get("inputs"), label="evaluation.inputs")
    _exact_fields(inputs, {"warped", "target_first", "target_second"}, label="evaluation.inputs")
    actual_inputs = {
        name: _normalized_absolute_path(
            inputs[name], label=f"evaluation.inputs.{name}"
        )
        for name in ("warped", "target_first", "target_second")
    }
    expected_inputs = {
        "warped": contract.source_dir,
        "target_first": contract.target_first_dir,
        "target_second": contract.target_second_dir,
    }
    if actual_inputs != expected_inputs:
        raise ApprovalError(
            f"evaluation inputs 未使用 canonical typical roots；"
            f"expected={expected_inputs!r}, actual={actual_inputs!r}"
        )

    checkpoints = _mapping(root.get("checkpoints"), label="evaluation.checkpoints")
    _exact_fields(checkpoints, {"anchor", "best", "latest"}, label="evaluation.checkpoints")
    run_root = _derive_run_root(checkpoints)
    if run_root != contract.run_root:
        raise ApprovalError(
            f"evaluation run-root 非 canonical；expected={contract.run_root}, actual={run_root}"
        )
    for name, expected_path in contract.checkpoints().items():
        actual_path = Path(checkpoints[name]["path"])
        if actual_path != expected_path:
            raise ApprovalError(
                f"checkpoints.{name}.path 非 canonical；"
                f"expected={expected_path}, actual={actual_path}"
            )

    outputs = _mapping(root.get("outputs"), label="evaluation.outputs")
    _exact_fields(outputs, {"v33_anchor", "v33_best"}, label="evaluation.outputs")
    expected_outputs = contract.output_dirs(run_id)
    for name in ("v33_anchor", "v33_best"):
        path = _normalized_absolute_path(outputs[name], label=f"evaluation.outputs.{name}")
        if not path.is_dir():
            raise ApprovalError(f"evaluation.outputs.{name} 不是目录：{path}")
        if path != expected_outputs[name]:
            raise ApprovalError(
                f"evaluation.outputs.{name} 非 canonical run-id 目录；"
                f"expected={expected_outputs[name]}, actual={path}"
            )
    if outputs["v33_anchor"] == outputs["v33_best"]:
        raise ApprovalError("anchor/best 输出目录不得相同")

    inference_reports = _mapping(
        root.get("inference_reports"), label="evaluation.inference_reports"
    )
    _exact_fields(
        inference_reports,
        {"v33_anchor", "v33_best"},
        label="evaluation.inference_reports",
    )
    for candidate in ("v33_anchor", "v33_best"):
        item = _mapping(
            inference_reports[candidate],
            label=f"evaluation.inference_reports.{candidate}",
        )
        _exact_fields(
            item,
            {"checkpoint", "input_count", "success_count", "error_count", "artifact"},
            label=f"evaluation.inference_reports.{candidate}",
        )
        _exact_int(item.get("input_count"), TYPICAL_IMAGE_COUNT, label=f"{candidate}.input_count")
        _exact_int(item.get("success_count"), TYPICAL_IMAGE_COUNT, label=f"{candidate}.success_count")
        _exact_int(item.get("error_count"), 0, label=f"{candidate}.error_count")
        _mapping(item.get("artifact"), label=f"{candidate}.artifact")
        checkpoint_name = "anchor" if candidate == "v33_anchor" else "best"
        if item.get("checkpoint") != checkpoints[checkpoint_name].get("path"):
            raise ApprovalError(
                f"evaluation.inference_reports.{candidate}.checkpoint "
                f"与 checkpoints.{checkpoint_name}.path 不一致"
            )

    reports = _mapping(root.get("reports"), label="evaluation.reports")
    _exact_fields(reports, set(_REPORT_NAMES), label="evaluation.reports")
    for name in _REPORT_NAMES:
        _mapping(reports[name], label=f"evaluation.reports.{name}")

    automatic = _mapping(
        root.get("automatic_quality_gate"),
        label="evaluation.automatic_quality_gate",
    )
    _exact_fields(
        automatic,
        {"status", "policy", "failure_count", "failures", "summary"},
        label="evaluation.automatic_quality_gate",
    )
    if automatic.get("status") != "passed":
        raise ApprovalError("automatic quality status 必须为 passed")
    _exact_int(
        automatic.get("failure_count"),
        0,
        label="automatic_quality_gate.failure_count",
    )
    if automatic.get("failures") != []:
        raise ApprovalError("automatic_quality_gate.failures 必须为空列表")
    _mapping(automatic.get("summary"), label="automatic_quality_gate.summary")
    _mapping(automatic.get("policy"), label="automatic_quality_gate.policy")

    manual = _mapping(
        root.get("manual_quality_review"), label="evaluation.manual_quality_review"
    )
    _exact_fields(
        manual,
        {"status", "candidate", "evidence_sha256", "required_reviewed_count"},
        label="evaluation.manual_quality_review",
    )
    # The first stage is intentionally pending; the separately supplied review
    # must be complete.  Any pending criterion in that document is rejected by
    # validate_completed_typical_review below.
    if manual.get("status") != "pending":
        raise ApprovalError("evaluation manual review 初始状态必须为 pending")
    if manual.get("candidate") != CANDIDATE_NAME:
        raise ApprovalError(f"manual review candidate 必须为 {CANDIDATE_NAME!r}")
    digest = manual.get("evidence_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ApprovalError("manual review evidence_sha256 无效")
    _exact_int(
        manual.get("required_reviewed_count"),
        TYPICAL_IMAGE_COUNT,
        label="manual_quality_review.required_reviewed_count",
    )
    _mapping(root.get("quality_proxy"), label="evaluation.quality_proxy")
    config_record = _mapping(root.get("config"), label="evaluation.config")
    if config_record.get("path") != str(contract.config_path):
        raise ApprovalError(
            f"evaluation config 非 canonical；expected={contract.config_path}, "
            f"actual={config_record.get('path')!r}"
        )
    policy_record = automatic["policy"]
    if policy_record.get("path") != str(contract.quality_policy_path):
        raise ApprovalError(
            f"quality policy 非 canonical；expected={contract.quality_policy_path}, "
            f"actual={policy_record.get('path')!r}"
        )

    expected_manifest_path = manifest_path.expanduser().resolve(strict=True)
    if expected_manifest_path.name != "evaluation_manifest.json":
        raise ApprovalError("输入评估清单文件名必须为 evaluation_manifest.json")
    canonical_manifest_path = contract.report_dir(run_id) / "evaluation_manifest.json"
    if expected_manifest_path != canonical_manifest_path:
        raise ApprovalError(
            f"evaluation manifest 非 canonical run-id report 目录；"
            f"expected={canonical_manifest_path}, actual={expected_manifest_path}"
        )
    return run_root


def _snapshot_output_directory(
    directory: Path, *, label: str
) -> dict[str, Mapping[str, Any]]:
    """Hash the entire flat inference directory for an approval-time snapshot."""

    if not directory.is_dir():
        raise ApprovalError(f"{label} 目录不存在：{directory}")
    result: dict[str, Mapping[str, Any]] = {}
    try:
        entries = sorted(directory.iterdir(), key=lambda item: (item.name.casefold(), item.name))
    except OSError as error:
        raise ApprovalError(f"无法列出 {label}：{directory}: {error}") from error
    for path in entries:
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise ApprovalError(f"无法检查 {label}/{path.name}: {error}") from error
        if stat_module.S_ISLNK(mode) or not stat_module.S_ISREG(mode):
            raise ApprovalError(f"{label} 只能包含普通非符号链接文件：{path}")
        result[path.name] = _current_artifact(path, label=f"{label}/{path.name}")
    if not result:
        raise ApprovalError(f"{label} 目录为空：{directory}")
    return result


def _inventory_sha256(files: Mapping[str, Any]) -> str:
    try:
        canonical = json.dumps(
            files,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ApprovalError(f"output inventory 不是 canonical JSON：{error}") from error
    return hashlib.sha256(canonical).hexdigest()


def verify_inference_output_binding(
    binding: Any, *, label: str
) -> dict[str, Any]:
    """Verify a persisted final-manifest inventory against its current directory."""

    item = _mapping(binding, label=label)
    _exact_fields(
        item,
        {"directory", "file_count", "inventory_sha256", "files"},
        label=label,
    )
    directory = _normalized_absolute_path(
        item.get("directory"), label=f"{label}.directory"
    )
    if not directory.is_dir():
        raise ApprovalError(f"{label}.directory 不是目录：{directory}")
    files = _mapping(item.get("files"), label=f"{label}.files")
    count = item.get("file_count")
    if type(count) is not int or count != len(files):
        raise ApprovalError(f"{label}.file_count 与 files 数量不一致")
    recorded_digest = item.get("inventory_sha256")
    if (
        not isinstance(recorded_digest, str)
        or _SHA256_RE.fullmatch(recorded_digest) is None
        or recorded_digest != _inventory_sha256(files)
    ):
        raise ApprovalError(f"{label}.inventory_sha256 无效或不匹配")
    for name, record in files.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ApprovalError(f"{label}.files 含非法 basename：{name!r}")
        verify_artifact_record(
            record,
            label=f"{label}.files.{name}",
            expected_path=directory / name,
        )
    current = _snapshot_output_directory(directory, label=label)
    _assert_typed_equal(current, dict(files), label=f"{label} current inventory")
    return {
        "status": "valid",
        "directory": str(directory),
        "file_count": count,
        "inventory_sha256": recorded_digest,
    }


def _assert_typed_equal(actual: Any, expected: Any, *, label: str) -> None:
    if not _typed_equal(actual, expected):
        raise ApprovalError(f"{label} 与评估阶段保存值不一致")


def _verify_report_artifacts(
    manifest: Mapping[str, Any], *, report_dir: Path
) -> tuple[dict[str, Path], list[tuple[str, Mapping[str, Any]]]]:
    reports = manifest["reports"]
    paths: dict[str, Path] = {}
    bindings: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[Path] = set()
    for name, filename in _REPORT_NAMES.items():
        expected = report_dir / filename
        record = reports[name]
        path = verify_artifact_record(
            record,
            label=f"reports.{name}",
            expected_path=expected,
        )
        if path in seen:
            raise ApprovalError(f"多个 report artifact 复用同一路径：{path}")
        seen.add(path)
        paths[name] = path
        bindings.append((f"reports.{name}", record))
    return paths, bindings


def _current_checkpoint_summaries(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    stored = manifest["checkpoints"]
    current: dict[str, Mapping[str, Any]] = {}
    for name in ("anchor", "best", "latest"):
        path = Path(stored[name]["path"])
        try:
            current[name] = checkpoint_summary(path)
        except FinalizationError as error:
            raise ApprovalError(f"当前 {name} checkpoint 无效：{error}") from error
        _assert_typed_equal(current[name], stored[name], label=f"checkpoints.{name}")
    train = _mapping(config.get("train"), label="config.train")
    epochs = train.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise ApprovalError("config.train.epochs 必须是整数")
    best_metric_name = str(train.get("best_metric", "epe"))
    best_metric_mode = str(train.get("best_metric_mode", "min")).lower()
    try:
        validate_checkpoint_set(
            current["anchor"],
            current["best"],
            current["latest"],
            expected_total_epochs=epochs,
            current_inference_config=inference_critical_config(config),
            expected_best_metric_name=best_metric_name,
            expected_best_metric_mode=best_metric_mode,
        )
    except FinalizationError as error:
        raise ApprovalError(f"当前 checkpoint 集合不再满足发布契约：{error}") from error
    return current


def _validate_current_inference(
    manifest: Mapping[str, Any],
    *,
    checkpoints: Mapping[str, Mapping[str, Any]],
    source_index: Mapping[str, Path],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Mapping[str, Any]]],
    list[tuple[str, Mapping[str, Any]]],
]:
    outputs = {
        name: Path(manifest["outputs"][name]).resolve()
        for name in ("v33_anchor", "v33_best")
    }
    snapshots = {
        name: _snapshot_output_directory(path, label=f"outputs.{name}")
        for name, path in outputs.items()
    }
    validated: dict[str, Mapping[str, Any]] = {}
    bindings: list[tuple[str, Mapping[str, Any]]] = []
    for candidate, checkpoint_name in (
        ("v33_anchor", "anchor"),
        ("v33_best", "best"),
    ):
        checkpoint_path = Path(checkpoints[checkpoint_name]["path"]).resolve()
        report_record = manifest["inference_reports"][candidate]["artifact"]
        verify_artifact_record(
            report_record,
            label=f"inference_reports.{candidate}.artifact",
            expected_path=outputs[candidate] / "inference_report.json",
        )
        bindings.append((f"inference_reports.{candidate}.artifact", report_record))
        try:
            current = validate_inference_output(
                outputs[candidate],
                checkpoint=checkpoint_path,
                checkpoint_metadata=checkpoints[checkpoint_name],
                source_index=source_index,
                expected_count=TYPICAL_IMAGE_COUNT,
            )
        except FinalizationError as error:
            raise ApprovalError(f"当前 {candidate} 推理产物无效：{error}") from error
        expected_entry = manifest["inference_reports"][candidate]
        expected_core = {
            "checkpoint": expected_entry["checkpoint"],
            "input_count": expected_entry["input_count"],
            "success_count": expected_entry["success_count"],
            "error_count": expected_entry["error_count"],
        }
        current_core = {
            key: current[key]
            for key in ("checkpoint", "input_count", "success_count", "error_count")
        }
        _assert_typed_equal(current_core, expected_core, label=f"inference_reports.{candidate}")
        validated[candidate] = current
    return validated, snapshots, bindings


def recompute_line_report(
    stored_report: Mapping[str, Any],
    *,
    input_dir: Path,
    target_first_dir: Path,
    target_second_dir: Path,
    anchor_output: Path,
    best_output: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Re-run the fixed evaluator and require a typed-exact report match."""

    try:
        validate_line_report(stored_report, expected_count=TYPICAL_IMAGE_COUNT)
        current = evaluate_dataset(
            input_dir,
            {
                "warped": input_dir,
                "target_first": target_first_dir,
                "target_second": target_second_dir,
                "v33_anchor_full": anchor_output,
                "v33_best_full": best_output,
                "v33_anchor_valid": anchor_output,
                "v33_best_valid": best_output,
            },
            valid_masks={
                "v33_anchor_valid": anchor_output,
                "v33_best_valid": best_output,
            },
        )
        weighted = validate_line_report(current, expected_count=TYPICAL_IMAGE_COUNT)
    except (FinalizationError, FileNotFoundError, ValueError) as error:
        raise ApprovalError(f"当前 line evaluator 失败：{error}") from error
    # Comparing the complete report also binds evaluator parameters, source and
    # candidate directories, and every indexed/candidate/mask row path.
    _assert_typed_equal(current, dict(stored_report), label="all40 line report 重算结果")
    return current, weighted


def _expected_quality_proxy(
    weighted_proxy: Mapping[str, float],
    inference: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "notice": _QUALITY_PROXY_NOTICE,
        "metric": "image_mean_orientation_error_deg_length_weighted",
        "values": dict(weighted_proxy),
        "evaluation_valid_fraction": {
            candidate: {
                key: value
                for key, value in inference[candidate]["_finalizer_validation"].items()
                if key != "per_image"
            }
            for candidate in ("v33_anchor", "v33_best")
        },
        "v33_anchor_at_most_target_first": (
            weighted_proxy["v33_anchor_full"] <= weighted_proxy["target_first"]
        ),
        "v33_best_at_most_target_first": (
            weighted_proxy["v33_best_full"] <= weighted_proxy["target_first"]
        ),
        "v33_anchor_at_most_target_second": (
            weighted_proxy["v33_anchor_full"] <= weighted_proxy["target_second"]
        ),
        "v33_best_at_most_target_second": (
            weighted_proxy["v33_best_full"] <= weighted_proxy["target_second"]
        ),
    }


def recompute_quality_gate(
    *,
    policy: Any,
    checkpoints: Mapping[str, Mapping[str, Any]],
    line_report: Mapping[str, Any],
    inference: Mapping[str, Mapping[str, Any]],
    stored_report: Mapping[str, Any],
    manifest_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-run the pure gate and bind both saved report and manifest summary."""

    try:
        current = evaluate_typical_quality(
            policy,
            validation=_quality_validation_inputs(checkpoints),
            line_report=line_report,
            inference_metrics={
                candidate: inference[candidate]["_finalizer_validation"]["per_image"]
                for candidate in ("v33_anchor", "v33_best")
            },
        )
    except (QualityGateInputError, FinalizationError) as error:
        raise ApprovalError(f"automatic quality gate 重算失败：{error}") from error
    _assert_typed_equal(current, dict(stored_report), label="automatic quality gate 重算结果")
    if current.get("passed") is not True or current.get("failures") != []:
        raise ApprovalError("重算 automatic quality gate 未通过或含 failure")
    expected_manifest = {
        "status": "passed",
        "policy": manifest_gate["policy"],
        "failure_count": 0,
        "failures": [],
        "summary": current.get("summary"),
    }
    _assert_typed_equal(dict(manifest_gate), expected_manifest, label="manifest automatic quality gate")
    return current


def _validate_evidence_roots(
    evidence: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Bind every review inventory path to this evaluation's input/output roots."""

    inventories = _mapping(evidence.get("inventories"), label="evidence.inventories")
    expected_roots = {
        "source": Path(manifest["inputs"]["warped"]).resolve(),
        "target_first": Path(manifest["inputs"]["target_first"]).resolve(),
        "target_second": Path(manifest["inputs"]["target_second"]).resolve(),
    }
    for name in INVENTORY_NAMES:
        if name.startswith("v33_anchor_"):
            expected_roots[name] = Path(manifest["outputs"]["v33_anchor"]).resolve()
        elif name.startswith("v33_best_"):
            expected_roots[name] = Path(manifest["outputs"]["v33_best"]).resolve()
    if set(expected_roots) != set(INVENTORY_NAMES):
        raise AssertionError("internal evidence-root mapping is incomplete")
    declared_roots = _mapping(evidence.get("roots"), label="evidence.roots")
    expected_declared_roots = {
        "source": str(expected_roots["source"]),
        "target_first": str(expected_roots["target_first"]),
        "target_second": str(expected_roots["target_second"]),
        "v33_anchor": str(Path(manifest["outputs"]["v33_anchor"]).resolve()),
        "v33_best": str(Path(manifest["outputs"]["v33_best"]).resolve()),
    }
    _assert_typed_equal(
        dict(declared_roots),
        expected_declared_roots,
        label="evidence roots 与 evaluation",
    )
    for name in INVENTORY_NAMES:
        records = inventories.get(name)
        if not isinstance(records, list):
            raise ApprovalError(f"evidence inventory 缺失：{name}")
        for record in records:
            item = _mapping(record, label=f"evidence.inventories.{name}[]")
            path = _normalized_absolute_path(item.get("path"), label=f"evidence.{name}.path")
            if path.parent != expected_roots[name]:
                raise ApprovalError(
                    f"evidence {name} 路径未绑定到本次 evaluation；"
                    f"expected_root={expected_roots[name]}, actual={path}"
                )


def validate_release_inputs(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    evaluation_artifact: Mapping[str, Any],
    review_path: Path,
    run_root: Path,
) -> ApprovalState:
    """Perform the complete under-lock evaluation and manual-review audit."""

    report_dir = manifest_path.parent.resolve()
    config_path = verify_artifact_record(manifest["config"], label="config")
    policy_record = manifest["automatic_quality_gate"]["policy"]
    policy_path = verify_artifact_record(policy_record, label="quality policy")
    report_paths, report_bindings = _verify_report_artifacts(manifest, report_dir=report_dir)
    bound_artifacts: list[tuple[str, Mapping[str, Any]]] = [
        ("config", manifest["config"]),
        ("quality policy", policy_record),
        *report_bindings,
    ]

    config_record, config_raw = _read_stable_file(
        config_path, label="config", capture_bytes=True
    )
    _assert_typed_equal(
        config_record, manifest["config"], label="config artifact"
    )
    assert config_raw is not None
    try:
        config = yaml.safe_load(config_raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise ApprovalError(f"当前 config 无法加载：{config_path}: {error}") from error
    if not isinstance(config, Mapping):
        raise ApprovalError(f"当前 config root 不是 mapping：{config_path}")

    current_policy_record, policy_raw = _read_stable_file(
        policy_path, label="quality policy", capture_bytes=True
    )
    _assert_typed_equal(
        current_policy_record, policy_record, label="quality policy artifact"
    )
    assert policy_raw is not None
    try:
        policy = parse_quality_policy_yaml(policy_raw.decode("utf-8"))
    except (UnicodeError, QualityGateInputError) as error:
        raise ApprovalError(f"当前 quality policy 无效：{error}") from error

    inputs = {
        name: Path(manifest["inputs"][name]).resolve()
        for name in ("warped", "target_first", "target_second")
    }
    try:
        indexes = validate_image_sets(
            inputs["warped"],
            inputs["target_first"],
            inputs["target_second"],
            expected_count=TYPICAL_IMAGE_COUNT,
        )
    except FinalizationError as error:
        raise ApprovalError(f"当前 typical 输入集合无效：{error}") from error

    checkpoints = _current_checkpoint_summaries(manifest, config)
    inference, snapshots, inference_bindings = _validate_current_inference(
        manifest,
        checkpoints=checkpoints,
        source_index=indexes["warped"],
    )
    bound_artifacts.extend(inference_bindings)

    stored_line_payload, _ = _load_bound_json(
        report_paths["line_proxy"],
        label="line report",
        expected_artifact=manifest["reports"]["line_proxy"],
    )
    stored_line = _mapping(
        stored_line_payload,
        label="line report",
    )
    outputs = {
        name: Path(manifest["outputs"][name]).resolve()
        for name in ("v33_anchor", "v33_best")
    }
    line_report, weighted = recompute_line_report(
        stored_line,
        input_dir=inputs["warped"],
        target_first_dir=inputs["target_first"],
        target_second_dir=inputs["target_second"],
        anchor_output=outputs["v33_anchor"],
        best_output=outputs["v33_best"],
    )

    comparison_payload, _ = _load_bound_json(
        report_paths["comparison_json"],
        label="comparison report",
        expected_artifact=manifest["reports"]["comparison_json"],
    )
    comparison = _mapping(
        comparison_payload,
        label="comparison report",
    )
    try:
        validate_comparison_report(
            comparison,
            source_keys=tuple(path.stem for path in indexes["warped"].values()),
            expected_count=TYPICAL_IMAGE_COUNT,
        )
    except FinalizationError as error:
        raise ApprovalError(f"当前 comparison report 无效：{error}") from error

    stored_gate_payload, _ = _load_bound_json(
        report_paths["automatic_quality_gate"],
        label="automatic gate report",
        expected_artifact=manifest["reports"]["automatic_quality_gate"],
    )
    stored_gate = _mapping(
        stored_gate_payload,
        label="automatic gate report",
    )
    quality_report = recompute_quality_gate(
        policy=policy,
        checkpoints=checkpoints,
        line_report=line_report,
        inference=inference,
        stored_report=stored_gate,
        manifest_gate=manifest["automatic_quality_gate"],
    )
    _assert_typed_equal(
        manifest["quality_proxy"],
        _expected_quality_proxy(weighted, inference),
        label="manifest quality_proxy",
    )

    evidence_payload, _ = _load_bound_json(
        report_paths["review_evidence"],
        label="review evidence",
        expected_artifact=manifest["reports"]["review_evidence"],
    )
    evidence = _mapping(
        evidence_payload,
        label="review evidence",
    )
    try:
        evidence_status = validate_typical_evidence(evidence, verify_files=True)
    except TypicalReviewError as error:
        raise ApprovalError(f"review evidence 无效或文件已变化：{error}") from error
    _validate_evidence_roots(evidence, manifest)
    if evidence.get("evidence_sha256") != manifest["manual_quality_review"]["evidence_sha256"]:
        raise ApprovalError("evaluation manifest 与 review evidence digest 不一致")

    template_payload, _ = _load_bound_json(
        report_paths["quality_review_template"],
        label="review template",
        expected_artifact=manifest["reports"]["quality_review_template"],
    )
    template = _mapping(
        template_payload,
        label="review template",
    )
    try:
        expected_template = make_typical_review_template(evidence)
    except TypicalReviewError as error:
        raise ApprovalError(f"无法重建 review template：{error}") from error
    _assert_typed_equal(template, expected_template, label="quality review template")

    review_path = review_path.expanduser().resolve(strict=True)
    review_payload, review_artifact = _load_bound_json(
        review_path, label="completed review"
    )
    review = _mapping(review_payload, label="completed review")
    try:
        review_summary = validate_completed_typical_review(review, evidence)
    except TypicalReviewError as error:
        raise ApprovalError(f"人工 review 尚未全部通过或 evidence 已变化：{error}") from error

    return ApprovalState(
        manifest=manifest,
        evaluation_path=manifest_path,
        evaluation_artifact=evaluation_artifact,
        review_path=review_path,
        review_artifact=review_artifact,
        review=review,
        review_summary=review_summary,
        evidence=evidence,
        evidence_status=evidence_status,
        checkpoint_summaries=checkpoints,
        bound_artifacts=tuple(bound_artifacts),
        output_snapshots=snapshots,
        quality_report=quality_report,
        run_root=run_root,
    )


def _verify_checkpoint_stats(state: ApprovalState) -> None:
    for name, summary in state.checkpoint_summaries.items():
        path = Path(summary["path"])
        digest = summary.get("sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ApprovalError(f"{name} checkpoint summary 缺少规范 SHA-256")
        current = _current_artifact(path, label=f"{name} checkpoint")
        expected = {
            "path": summary["path"],
            "size_bytes": summary["size_bytes"],
            "mtime_ns": summary["mtime_ns"],
            "sha256": digest,
        }
        _assert_typed_equal(
            current,
            expected,
            label=f"{name} checkpoint artifact",
        )


def _verify_external_model_identities(state: ApprovalState) -> None:
    """Re-hash every teacher/LAMA binding immediately before publication."""

    for name, summary in state.checkpoint_summaries.items():
        teacher = summary.get("teacher_prior_identity")
        if not isinstance(teacher, Mapping) or teacher.get("version") != 2:
            raise ApprovalError(f"{name} teacher identity 不是 strict v2")
        try:
            validate_external_file_identity(
                teacher,
                path_key="resolved_path",
                size_key="file_size",
                label=f"{name} teacher",
            )
        except ExternalFileIdentityError as error:
            raise ApprovalError(
                f"发布前 {name} teacher 外部模型复核失败：{error}"
            ) from error

        deployment = summary.get("deployment_contract")
        if not isinstance(deployment, Mapping) or deployment.get("version") != 2:
            raise ApprovalError(f"{name} deployment contract 不是 strict v2")
        inpaint = deployment.get("inpaint")
        if not isinstance(inpaint, Mapping) or not isinstance(
            inpaint.get("enabled"), bool
        ):
            raise ApprovalError(f"{name} LAMA identity 无效")
        if not inpaint["enabled"]:
            continue
        try:
            validate_external_file_identity(
                inpaint,
                path_key="path",
                size_key="size_bytes",
                label=f"{name} LAMA",
            )
        except ExternalFileIdentityError as error:
            raise ApprovalError(
                f"发布前 {name} LAMA 外部模型复核失败：{error}"
            ) from error


def final_toctou_check(state: ApprovalState) -> tuple[dict[str, Any], dict[str, Any]]:
    """Last operation before publication: re-hash every release binding."""

    current_evaluation = _current_artifact(
        state.evaluation_path, label="evaluation manifest"
    )
    _assert_typed_equal(
        current_evaluation,
        dict(state.evaluation_artifact),
        label="evaluation manifest artifact",
    )
    current_review = _current_artifact(state.review_path, label="completed review")
    _assert_typed_equal(
        current_review,
        dict(state.review_artifact),
        label="completed review artifact",
    )
    for label, record in state.bound_artifacts:
        verify_artifact_record(record, label=label)
    for name, expected in state.output_snapshots.items():
        current = _snapshot_output_directory(
            Path(state.manifest["outputs"][name]), label=f"outputs.{name}"
        )
        _assert_typed_equal(current, expected, label=f"outputs.{name} approval snapshot")

    # Keep these as the final evidence operations.  validate_completed performs
    # its own second verify_files=True pass; no pending/fail row can survive it.
    try:
        evidence_status = validate_typical_evidence(
            state.evidence, verify_files=True
        )
        review_summary = validate_completed_typical_review(
            state.review, state.evidence
        )
    except TypicalReviewError as error:
        raise ApprovalError(f"发布前 evidence/review 最终复核失败：{error}") from error

    # These are deliberately the last filesystem checks before the immutable
    # release manifest is assembled: neither a checkpoint nor one of its
    # external teacher/LAMA dependencies may change during the slower all-40
    # output/evidence revalidation above.
    _verify_checkpoint_stats(state)
    _verify_external_model_identities(state)
    return evidence_status, review_summary


def build_final_manifest(
    state: ApprovalState,
    *,
    evidence_status: Mapping[str, Any],
    review_summary: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_record = state.manifest["reports"]["review_evidence"]
    gate_record = state.manifest["reports"]["automatic_quality_gate"]
    output_bindings: dict[str, Any] = {}
    for candidate in ("v33_anchor", "v33_best"):
        files = {
            name: dict(record)
            for name, record in state.output_snapshots[candidate].items()
        }
        output_bindings[candidate] = {
            "directory": state.manifest["outputs"][candidate],
            "file_count": len(files),
            "inventory_sha256": _inventory_sha256(files),
            "files": files,
        }
    return {
        "schema": FINAL_SCHEMA,
        "schema_version": FINAL_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_status": "release_approved",
        "selected_candidate": CANDIDATE_NAME,
        "release_ready": True,
        "run_id": state.manifest["run_id"],
        "expected_count": TYPICAL_IMAGE_COUNT,
        "bindings": {
            "evaluation_manifest": dict(state.evaluation_artifact),
            "completed_review": dict(state.review_artifact),
            "review_evidence": dict(evidence_record),
            "evaluation_manifest_sha256": state.evaluation_artifact["sha256"],
            "completed_review_sha256": state.review_artifact["sha256"],
            "review_evidence_artifact_sha256": evidence_record["sha256"],
            "evidence_sha256": state.evidence["evidence_sha256"],
        },
        "automatic_quality_gate": {
            "status": "passed",
            "report": dict(gate_record),
            "policy": dict(state.manifest["automatic_quality_gate"]["policy"]),
            "summary": state.quality_report["summary"],
        },
        "manual_quality_review": {
            "status": "passed",
            "summary": dict(review_summary),
            "evidence_validation": dict(evidence_status),
        },
        "checkpoints": dict(state.manifest["checkpoints"]),
        "outputs": dict(state.manifest["outputs"]),
        "inference_output_bindings": output_bindings,
    }


def _atomic_create_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Durably create the fixed leaf via one no-follow parent directory fd."""

    path = Path(path)
    if not path.is_absolute() or path.name != FINAL_MANIFEST_NAME:
        raise ApprovalError(
            f"final manifest 必须是绝对固定 leaf {FINAL_MANIFEST_NAME!r}：{path}"
        )
    try:
        parent = path.parent.resolve(strict=True)
        parent_lstat = parent.lstat()
    except (OSError, RuntimeError) as error:
        raise ApprovalError(f"final manifest parent 无效：{path.parent}: {error}") from error
    if not stat_module.S_ISDIR(parent_lstat.st_mode) or stat_module.S_ISLNK(
        parent_lstat.st_mode
    ):
        raise ApprovalError(f"final manifest parent 必须是真实目录：{parent}")
    if path.parent != parent:
        raise ApprovalError(f"final manifest parent 不是规范真实路径：{path.parent}")
    try:
        encoded = (
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ApprovalError(f"final manifest 无法编码为严格 JSON：{error}") from error

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise ApprovalError("当前平台不支持安全发布所需的 O_NOFOLLOW")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | nofollow
    try:
        directory_fd = os.open(parent, directory_flags)
    except OSError as error:
        raise ApprovalError(f"无法安全打开 final manifest parent：{parent}: {error}") from error
    opened_parent = os.fstat(directory_fd)
    if (
        not stat_module.S_ISDIR(opened_parent.st_mode)
        or (opened_parent.st_dev, opened_parent.st_ino)
        != (parent_lstat.st_dev, parent_lstat.st_ino)
    ):
        os.close(directory_fd)
        raise ApprovalError("final manifest parent 在打开期间被替换")

    temporary_name = (
        f".{FINAL_MANIFEST_NAME}.approval-{os.getpid()}-{secrets.token_hex(8)}"
    )
    temporary_fd: int | None = None
    temporary_exists = False
    try:
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | nofollow,
                0o644,
                dir_fd=directory_fd,
            )
            temporary_exists = True
            offset = 0
            while offset < len(encoded):
                written = os.write(temporary_fd, encoded[offset:])
                if written <= 0:
                    raise OSError("short write while creating final manifest")
                offset += written
            os.fsync(temporary_fd)
            os.close(temporary_fd)
            temporary_fd = None
            os.link(
                temporary_name,
                FINAL_MANIFEST_NAME,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ApprovalError(f"final manifest 已存在，拒绝覆盖：{path}") from error
        except OSError as error:
            raise ApprovalError(f"无法原子创建 final manifest：{path}: {error}") from error
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_exists = False
        os.fsync(directory_fd)

        try:
            current_parent = parent.lstat()
            final_lstat = os.stat(
                FINAL_MANIFEST_NAME,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ApprovalError(f"发布后无法复核 final manifest：{error}") from error
        if (
            (current_parent.st_dev, current_parent.st_ino)
            != (opened_parent.st_dev, opened_parent.st_ino)
            or not stat_module.S_ISREG(final_lstat.st_mode)
        ):
            raise ApprovalError("final manifest parent/leaf 在发布期间被替换")
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)
    current_record, current_bytes = _read_stable_file(
        parent / FINAL_MANIFEST_NAME,
        label="final manifest",
        capture_bytes=True,
    )
    assert current_bytes is not None
    if current_bytes != encoded or current_record["path"] != str(parent / FINAL_MANIFEST_NAME):
        raise ApprovalError("final manifest 发布后 bytes/path 校验失败")
    return parent / FINAL_MANIFEST_NAME


def approve_typical_review(
    evaluation_manifest: str | Path,
    completed_review: str | Path,
    *,
    contract: ReleaseContract | None = None,
) -> tuple[dict[str, Any], Path]:
    """Approve a release and atomically create its non-overwritable manifest."""

    contract = contract or ReleaseContract.canonical()
    try:
        evaluation_path = Path(evaluation_manifest).expanduser().resolve(strict=True)
        review_path = Path(completed_review).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ApprovalError(f"evaluation/review 输入路径无效：{error}") from error
    preliminary, _ = _load_bound_json(
        evaluation_path, label="evaluation manifest"
    )
    preliminary_run_root = validate_evaluation_manifest(
        preliminary, manifest_path=evaluation_path, contract=contract
    )

    lock_path = preliminary_run_root / RUN_LOCK_NAME
    try:
        lock_context = _exclusive_lock(lock_path)
        with lock_context:
            # Re-load only after the run lock is held, and reject a manifest
            # that tried to redirect approval to a different run-root.
            manifest_payload, evaluation_artifact = _load_bound_json(
                evaluation_path, label="evaluation manifest"
            )
            manifest = _mapping(
                manifest_payload, label="evaluation manifest"
            )
            run_root = validate_evaluation_manifest(
                manifest,
                manifest_path=evaluation_path,
                contract=contract,
            )
            if run_root != preliminary_run_root:
                raise ApprovalError("evaluation manifest 在获取 run-root 锁前后发生变化")
            final_path = evaluation_path.parent / FINAL_MANIFEST_NAME
            if os.path.lexists(final_path):
                raise ApprovalError(f"final manifest 已存在，拒绝覆盖：{final_path}")
            state = validate_release_inputs(
                manifest,
                manifest_path=evaluation_path,
                evaluation_artifact=evaluation_artifact,
                review_path=review_path,
                run_root=run_root,
            )
            evidence_status, review_summary = final_toctou_check(state)
            final_manifest = build_final_manifest(
                state,
                evidence_status=evidence_status,
                review_summary=review_summary,
            )
            output = _atomic_create_json(final_manifest, final_path)
            return final_manifest, output
    except FinalizationError as error:
        raise ApprovalError(f"无法获取 run-root 独占锁：{error}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-manifest",
        required=True,
        help="Path to the first-stage evaluation_manifest.json.",
    )
    parser.add_argument(
        "--review",
        required=True,
        help="Path to the completed v1 all-40 manual review JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _, output = approve_typical_review(
            args.evaluation_manifest,
            args.review,
        )
    except ApprovalError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 65
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[error] unexpected approval failure: {error}", file=sys.stderr)
        return 70
    print(f"[ok] v3.3 typical release approved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
