#!/usr/bin/env python3
"""Prepare the fail-closed v3.3 anchor/best all-40 quality review.

This stage validates inference artifacts, applies the frozen automatic quality
policy, hashes every reviewable image, and writes an evaluation manifest plus a
40-image review template.  It never writes ``final_manifest.json``: release is
a separate approval step that re-hashes the evidence and requires every manual
content-fidelity criterion to pass for the fixed ``v33_best`` candidate.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for import_root in (str(SCRIPT_DIR), str(SRC_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from checkpoint_status import _load, _typed_equal, inspect_checkpoint_payload
from diffusion2raft.config import load_config
from diffusion2raft.typical_quality_v2 import (
    QualityGateInputError,
    evaluate_typical_quality,
    parse_quality_policy_yaml,
)
from diffusion2raft.typical_review import (
    CANDIDATE_NAME as REVIEW_CANDIDATE_NAME,
    TypicalReviewError,
    build_typical_evidence,
    make_typical_review_template,
    validate_typical_evidence,
)
from evaluate_typical_lines import _atomic_write_json, evaluate_dataset
from make_typical_comparison import generate_comparison


TRAINING_REVISION = "v3_3_teacher_anchor_residual_warmup"
CHECKPOINT_ARTIFACT_FIELDS = (
    "path",
    "size_bytes",
    "mtime_ns",
    "sha256",
)
LINE_CANDIDATES = (
    "warped",
    "target_first",
    "target_second",
    "v33_anchor_full",
    "v33_best_full",
    "v33_anchor_valid",
    "v33_best_valid",
)
COMPARISON_CANDIDATES = (
    "target_first",
    "target_second",
    "v33_anchor",
    "v33_best",
)
REQUIRED_OUTPUTS = {
    "image",
    "raw_image",
    "prior_image",
    "flow",
    "prior_flow",
    "valid",
    "inpaint_mask",
    "evaluation_valid",
    "metadata",
    "residual_flow",
    "confidence",
}
V33_COMPARISON_EXTRA_SUFFIXES = (
    "_rectified_raw",
    # make_typical_comparison strips the terminal "_rectified" from
    # BASENAME_prior_rectified.png, leaving BASENAME_prior.
    "_prior",
    "_valid",
    "_inpaint_mask",
    "_evaluation_valid",
    "_feature_confidence",
)


class FinalizationError(RuntimeError):
    """A fail-closed finalization contract violation."""


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve the parent while preserving the leaf for O_NOFOLLOW opening."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.parent.resolve() / candidate.name


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_strict_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise FinalizationError(f"JSON 文件不存在或为空：{path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise FinalizationError(f"无效严格 JSON：{path}: {error}") from error


def _atomic_json(payload: Mapping[str, Any], path: Path) -> Path:
    try:
        return _atomic_write_json(payload, path)
    except (OSError, TypeError, ValueError) as error:
        raise FinalizationError(f"无法原子写入 JSON {path}: {error}") from error


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizationError(f"{label} 必须是有限数值，实际为 {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise FinalizationError(f"{label} 不是有限数值：{value!r}")
    return result


def _jpg_index(directory: Path, *, expected_count: int, label: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FinalizationError(f"{label} 目录不存在：{directory}")
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in supported
    )
    incompatible = [path.name for path in image_files if path.suffix != ".jpg"]
    if incompatible:
        raise FinalizationError(
            f"{label} 必须只含能被固定 '*.jpg' glob 命中的小写 .jpg；"
            f"不兼容文件={incompatible[:8]}"
        )
    if len(image_files) != expected_count:
        raise FinalizationError(
            f"{label} 图像数必须为 {expected_count}，实际为 {len(image_files)}"
        )
    index: dict[str, Path] = {}
    for path in image_files:
        key = path.stem.casefold()
        if key in index:
            raise FinalizationError(
                f"{label} 存在大小写归一化后的重复 basename："
                f"{index[key].name!r}, {path.name!r}"
            )
        index[key] = path.resolve()
    return index


def validate_image_sets(
    input_dir: Path,
    target_first_dir: Path,
    target_second_dir: Path,
    *,
    expected_count: int,
) -> dict[str, dict[str, Path]]:
    indexes = {
        "warped": _jpg_index(
            input_dir, expected_count=expected_count, label="warped/source"
        ),
        "target_first": _jpg_index(
            target_first_dir, expected_count=expected_count, label="target_first"
        ),
        "target_second": _jpg_index(
            target_second_dir, expected_count=expected_count, label="target_second"
        ),
    }
    source_keys = set(indexes["warped"])
    for name in ("target_first", "target_second"):
        keys = set(indexes[name])
        if keys != source_keys:
            raise FinalizationError(
                f"{name} basename 与 source 不一致；"
                f"missing={sorted(source_keys - keys)[:8]}, "
                f"extra={sorted(keys - source_keys)[:8]}"
            )
    for key in sorted(source_keys):
        source_shape = _read_image(
            indexes["warped"][key], label=f"warped/source:{key}"
        ).shape[:2]
        for name in ("target_first", "target_second"):
            target_shape = _read_image(
                indexes[name][key], label=f"{name}:{key}"
            ).shape[:2]
            if tuple(target_shape) != tuple(source_shape):
                raise FinalizationError(
                    f"{name}/{key} 尺寸与 source 不一致；"
                    f"source={tuple(source_shape)}, target={tuple(target_shape)}"
                )
    return indexes


def _mapping_copy(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalizationError(f"checkpoint 缺少 mapping 字段：{label}")
    return copy.deepcopy(dict(value))


def inference_critical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze every config field that can affect the v3.3 forward/deployment."""

    data = _mapping_copy(config.get("data"), label="config.data")
    if "work_size" not in data:
        raise FinalizationError("config.data 缺少 work_size")
    return {
        "data": {"work_size": copy.deepcopy(data["work_size"])},
        "model": _mapping_copy(config.get("model"), label="config.model"),
        "qwen": _mapping_copy(config.get("qwen"), label="config.qwen"),
        "inference": _mapping_copy(
            config.get("inference"), label="config.inference"
        ),
    }


_CHECKPOINT_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _checkpoint_identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(value, field)) for field in _CHECKPOINT_IDENTITY_FIELDS)


def _stable_checkpoint_payload(
    path: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Load and hash one immutable checkpoint identity through one descriptor."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            initial_fd_stat = os.fstat(handle.fileno())
            initial_path_stat = os.lstat(path)
            if not stat.S_ISREG(initial_fd_stat.st_mode):
                raise FinalizationError(f"checkpoint 不是普通文件：{path}")
            if initial_fd_stat.st_size == 0:
                raise FinalizationError(f"checkpoint 为空：{path}")
            if _checkpoint_identity(initial_fd_stat) != _checkpoint_identity(
                initial_path_stat
            ):
                raise FinalizationError(
                    f"checkpoint 路径与已打开文件身份不一致：{path}"
                )

            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            handle.seek(0)
            payload = _load(handle)

            final_fd_stat = os.fstat(handle.fileno())
            final_path_stat = os.lstat(path)
            initial_identity = _checkpoint_identity(initial_fd_stat)
            if (
                _checkpoint_identity(final_fd_stat) != initial_identity
                or _checkpoint_identity(final_path_stat) != initial_identity
            ):
                raise FinalizationError(
                    f"checkpoint 在读取期间发生变化或路径被替换：{path}"
                )
    except FinalizationError:
        raise
    except OSError as error:
        raise FinalizationError(
            f"checkpoint 无法安全打开或稳定读取 {path}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)

    artifact = {
        "path": str(path),
        "size_bytes": int(initial_fd_stat.st_size),
        "mtime_ns": int(initial_fd_stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }
    return payload, artifact


def checkpoint_summary(path: Path) -> dict[str, Any]:
    path = _resolve_checkpoint_path(path)
    try:
        payload, artifact = _stable_checkpoint_payload(path)
        stage, epoch, completed, best_name, best_value = inspect_checkpoint_payload(
            payload,
            expect_stage="unified",
            require_optimizer=True,
        )
    except Exception as error:
        raise FinalizationError(f"checkpoint 校验失败 {path}: {error}") from error
    try:
        saved_config = _mapping_copy(payload.get("config"), label="config")
        train_config = _mapping_copy(saved_config.get("train"), label="config.train")
        configured_epochs = train_config.get("epochs")
        if isinstance(configured_epochs, bool) or not isinstance(configured_epochs, int):
            raise FinalizationError("checkpoint config.train.epochs 必须是整数")
        summary = {
            "path": str(path),
            "size_bytes": artifact["size_bytes"],
            "mtime_ns": artifact["mtime_ns"],
            "sha256": artifact["sha256"],
            "stage": stage,
            "epoch_index": int(epoch),
            "completed_epochs": int(completed),
            "configured_epochs": int(configured_epochs),
            "inference_critical_config": inference_critical_config(saved_config),
            "training_revision": payload.get("training_revision"),
            "prior_backend": payload.get("prior_backend"),
            "teacher_prior_identity": _mapping_copy(
                payload.get("teacher_prior_identity"),
                label="teacher_prior_identity",
            ),
            "residual_application": _mapping_copy(
                payload.get("residual_application"),
                label="residual_application",
            ),
            "deployment_contract": _mapping_copy(
                payload.get("deployment_contract"),
                label="deployment_contract",
            ),
            "best_metric": _mapping_copy(
                payload.get("best_metric"),
                label="best_metric",
            ),
            "metrics": _mapping_copy(
                payload.get("metrics"),
                label="metrics",
            ),
            "best_metric_status": {
                "name": best_name,
                "value": best_value,
            },
        }
        return summary
    finally:
        del payload
        gc.collect()


def _checkpoint_artifact_from_summary(
    summary: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    artifact = {field: summary.get(field) for field in CHECKPOINT_ARTIFACT_FIELDS}
    if not isinstance(artifact["path"], str) or not artifact["path"]:
        raise FinalizationError(f"{label} checkpoint artifact path 无效")
    if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] <= 0:
        raise FinalizationError(f"{label} checkpoint artifact size_bytes 无效")
    if type(artifact["mtime_ns"]) is not int or artifact["mtime_ns"] < 0:
        raise FinalizationError(f"{label} checkpoint artifact mtime_ns 无效")
    digest = artifact["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise FinalizationError(f"{label} checkpoint artifact sha256 无效")
    return artifact


def _residual_schedule(summary: Mapping[str, Any]) -> dict[str, Any]:
    residual = summary["residual_application"]
    return {
        key: residual[key]
        for key in ("version", "origin_epoch", "warmup_epochs", "ramp_epochs", "max_scale")
    }


def validate_checkpoint_set(
    anchor: Mapping[str, Any],
    best: Mapping[str, Any],
    latest: Mapping[str, Any],
    *,
    expected_total_epochs: int,
    current_inference_config: Mapping[str, Any],
    expected_best_metric_name: str = "line_epe",
    expected_best_metric_mode: str = "min",
) -> None:
    summaries = {"anchor": anchor, "best": best, "latest": latest}
    for name, summary in summaries.items():
        if summary.get("training_revision") != TRAINING_REVISION:
            raise FinalizationError(
                f"{name} training_revision 错误："
                f"{summary.get('training_revision')!r}"
            )
        if summary.get("prior_backend") != "torchscript":
            raise FinalizationError(
                f"{name} 必须声明 prior_backend=torchscript"
            )
        teacher_identity = summary.get("teacher_prior_identity")
        deployment_contract = summary.get("deployment_contract")
        if not isinstance(teacher_identity, Mapping) or teacher_identity.get(
            "version"
        ) != 2:
            raise FinalizationError(f"{name} teacher identity 必须是 strict v2")
        if not isinstance(deployment_contract, Mapping) or deployment_contract.get(
            "version"
        ) != 2:
            raise FinalizationError(f"{name} deployment contract 必须是 strict v2")
        if int(summary["configured_epochs"]) != int(expected_total_epochs):
            raise FinalizationError(
                f"{name} checkpoint 的 epochs={summary['configured_epochs']}，"
                f"当前配置要求 {expected_total_epochs}"
            )
        best_metric = summary.get("best_metric")
        if not isinstance(best_metric, Mapping):
            raise FinalizationError(f"{name} 缺少 best_metric mapping")
        if best_metric.get("name") != expected_best_metric_name:
            raise FinalizationError(
                f"{name} best_metric.name={best_metric.get('name')!r}，"
                f"expected={expected_best_metric_name!r}"
            )
        if best_metric.get("mode") != expected_best_metric_mode:
            raise FinalizationError(
                f"{name} best_metric.mode={best_metric.get('mode')!r}，"
                f"expected={expected_best_metric_mode!r}"
            )

    anchor_inference_config = anchor.get("inference_critical_config")
    if not isinstance(anchor_inference_config, Mapping):
        raise FinalizationError("anchor 缺少 inference_critical_config")
    if not _typed_equal(anchor_inference_config, current_inference_config):
        raise FinalizationError(
            "当前推理 config 与 checkpoint 保存的 inference-critical config 不一致"
        )
    for name in ("best", "latest"):
        if not _typed_equal(
            anchor_inference_config,
            summaries[name].get("inference_critical_config"),
        ):
            raise FinalizationError(
                f"{name} 与 anchor 的 inference-critical config 不一致"
            )

    for field in ("teacher_prior_identity", "deployment_contract"):
        for name in ("best", "latest"):
            if not _typed_equal(anchor[field], summaries[name][field]):
                raise FinalizationError(f"{name} 与 anchor 的 {field} 不一致")
    anchor_schedule = _residual_schedule(anchor)
    for name in ("best", "latest"):
        if not _typed_equal(anchor_schedule, _residual_schedule(summaries[name])):
            raise FinalizationError(f"{name} 与 anchor 的 residual schedule 不一致")

    # best.pt must be the exact checkpoint that produced the final best value
    # carried by latest.pt, not merely any checkpoint from a compatible run.
    if not _typed_equal(best["best_metric"], latest["best_metric"]):
        raise FinalizationError("best.pt 与 latest.pt 的最终 best_metric 不一致")
    best_value = _finite_number(
        best["best_metric"].get("value"), label="best.best_metric.value"
    )
    best_metrics = best.get("metrics")
    if not isinstance(best_metrics, Mapping):
        raise FinalizationError("best checkpoint 缺少 metrics mapping")
    best_observed = _finite_number(
        best_metrics.get(expected_best_metric_name),
        label=f"best.metrics.{expected_best_metric_name}",
    )
    if best_observed != best_value:
        raise FinalizationError(
            "best checkpoint 自身 validation metric 与最终 best_metric.value 不一致；"
            f"observed={best_observed}, recorded={best_value}"
        )

    anchor_residual = anchor["residual_application"]
    if int(anchor["epoch_index"]) != int(anchor_residual["origin_epoch"]):
        raise FinalizationError("anchor epoch 必须严格等于 residual origin_epoch")
    if float(anchor_residual["scale"]) != 0.0:
        raise FinalizationError("anchor residual scale 必须严格为 0")
    if int(best["epoch_index"]) < int(anchor["epoch_index"]):
        raise FinalizationError("best checkpoint 早于 immutable anchor")
    if int(best["epoch_index"]) > int(latest["epoch_index"]):
        raise FinalizationError("best checkpoint 晚于 latest checkpoint")
    if int(latest["completed_epochs"]) != int(expected_total_epochs):
        raise FinalizationError(
            f"v3.3 completed epoch 必须严格等于配置总轮数：latest completed="
            f"{latest['completed_epochs']}/{expected_total_epochs}"
        )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Acquire a reusable lock without following or modifying its leaf entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    if path.name in {"", ".", ".."}:
        raise FinalizationError(f"非法锁文件名：{path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise FinalizationError("当前平台不支持安全的 O_NOFOLLOW 锁")
    file_flags |= nofollow
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        try:
            directory_fd = os.open(parent, directory_flags)
            file_fd = os.open(path.name, file_flags, 0o600, dir_fd=directory_fd)
        except OSError as error:
            raise FinalizationError(
                f"无法安全打开 all-40 锁 {path}: {error}"
            ) from error
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise FinalizationError(f"锁路径不是普通文件：{path}")
        try:
            fcntl.flock(file_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FinalizationError(f"已有 all-40 收尾任务持有锁：{path}") from error
        yield
    finally:
        if file_fd is not None:
            try:
                fcntl.flock(file_fd, fcntl.LOCK_UN)
            finally:
                os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _require_fresh_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FinalizationError(f"输出路径存在但不是目录：{path}")
        if any(path.iterdir()):
            raise FinalizationError(
                f"输出目录非空；为防止混入旧结果，拒绝复用：{path}"
            )
    else:
        path.mkdir(parents=True)


def _run_inference(
    *,
    checkpoint: Path,
    checkpoint_artifact: Mapping[str, Any],
    output_dir: Path,
    input_dir: Path,
    config: Path,
) -> None:
    expected_artifact = _checkpoint_artifact_from_summary(
        checkpoint_artifact, label="inference"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": sys.executable,
            "CONFIG": str(config),
            "CHECKPOINT": str(checkpoint),
            "EXPECTED_CHECKPOINT_PATH": expected_artifact["path"],
            "EXPECTED_CHECKPOINT_SIZE_BYTES": str(
                expected_artifact["size_bytes"]
            ),
            "EXPECTED_CHECKPOINT_MTIME_NS": str(expected_artifact["mtime_ns"]),
            "EXPECTED_CHECKPOINT_SHA256": expected_artifact["sha256"],
            "INPUT_DIR": str(input_dir),
            "OUTPUT_DIR": str(output_dir),
        }
    )
    command = ["bash", str(SCRIPT_DIR / "infer_typical_v33_teacher.sh")]
    try:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise FinalizationError(
            f"推理失败 checkpoint={checkpoint}, exit={error.returncode}"
        ) from error


def _ensure_under(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FinalizationError(f"{label} 跑出本次输出目录：{path}") from error


def _read_image(path: Path, *, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise FinalizationError(f"无法读取 {label}：{path}")
    return image


def _expected_artifact_paths(output_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "image": output_dir / f"{stem}_rectified.png",
        "raw_image": output_dir / f"{stem}_rectified_raw.png",
        "prior_image": output_dir / f"{stem}_prior_rectified.png",
        "flow": output_dir / f"{stem}_backward_flow.npy",
        "prior_flow": output_dir / f"{stem}_prior_backward_flow.npy",
        "valid": output_dir / f"{stem}_valid.png",
        "inpaint_mask": output_dir / f"{stem}_inpaint_mask.png",
        "evaluation_valid": output_dir / f"{stem}_evaluation_valid.png",
        "metadata": output_dir / f"{stem}_metadata.json",
        "residual_flow": output_dir / f"{stem}_residual_backward_flow.npy",
        "confidence": output_dir / f"{stem}_feature_confidence.png",
    }


def _validate_flow(
    path: Path, shape_hw: tuple[int, int], *, label: str
) -> np.ndarray:
    try:
        flow = np.load(path, allow_pickle=False)
    except Exception as error:
        raise FinalizationError(f"无法读取 {label}：{path}: {error}") from error
    if flow.shape != (*shape_hw, 2):
        raise FinalizationError(
            f"{label} shape 错误：expected={(*shape_hw, 2)}, actual={flow.shape}"
        )
    if flow.dtype != np.float32:
        raise FinalizationError(
            f"{label} dtype 错误：expected=float32, actual={flow.dtype}"
        )
    if not np.isfinite(flow).all():
        raise FinalizationError(f"{label} 含 NaN/Inf：{path}")
    return flow


def _flow_topology_metrics(
    flow: np.ndarray, valid: np.ndarray
) -> tuple[float | None, float | None]:
    """Recompute infer.py's fold/Jacobian metrics from persisted artifacts."""

    if flow.shape[:2] != valid.shape or flow.shape[2:] != (2,):
        raise FinalizationError("flow/valid shape 不一致，无法重算 topology")
    if min(valid.shape) < 2:
        return None, None
    u = flow[..., 0]
    v = flow[..., 1]
    du_dx = u[:-1, 1:] - u[:-1, :-1]
    du_dy = u[1:, :-1] - u[:-1, :-1]
    dv_dx = v[:-1, 1:] - v[:-1, :-1]
    dv_dy = v[1:, :-1] - v[:-1, :-1]
    determinant = (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx
    valid_cells = (
        valid[:-1, :-1]
        & valid[1:, :-1]
        & valid[:-1, 1:]
        & valid[1:, 1:]
    )
    values = determinant[valid_cells]
    if values.size == 0:
        return None, None
    return (
        float(np.mean(values <= 0.0)),
        float(np.quantile(values.astype(np.float32, copy=False), 0.01)),
    )


def validate_inference_output(
    output_dir: Path,
    *,
    checkpoint: Path,
    checkpoint_metadata: Mapping[str, Any],
    source_index: Mapping[str, Path],
    expected_count: int,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    report_path = output_dir / "inference_report.json"
    report = _load_strict_json(report_path)
    if not isinstance(report, Mapping):
        raise FinalizationError(f"inference report 不是 mapping：{report_path}")
    expected_checkpoint_artifact = _checkpoint_artifact_from_summary(
        checkpoint_metadata, label="expected inference"
    )
    canonical_checkpoint = str(_resolve_checkpoint_path(checkpoint))
    if expected_checkpoint_artifact["path"] != canonical_checkpoint:
        raise FinalizationError(
            "checkpoint 参数与 summary canonical path 不一致；"
            f"checkpoint={canonical_checkpoint!r}, "
            f"summary={expected_checkpoint_artifact['path']!r}"
        )
    expected_checkpoint = expected_checkpoint_artifact["path"]
    expected_source_dir = str(next(iter(source_index.values())).parent.resolve())
    exact_fields = {
        "checkpoint": expected_checkpoint,
        "source_dir": expected_source_dir,
        "input_count": expected_count,
        "success_count": expected_count,
        "error_count": 0,
        "resize_policy": "stretch",
        "image_decoder": "opencv",
        "resize_interpolation": "opencv_baseline",
    }
    for key, expected in exact_fields.items():
        if report.get(key) != expected:
            raise FinalizationError(
                f"inference report {key} 错误："
                f"expected={expected!r}, actual={report.get(key)!r}"
            )
    if not _typed_equal(
        report.get("checkpoint_artifact"), expected_checkpoint_artifact
    ):
        raise FinalizationError(
            "inference report checkpoint_artifact 与 finalizer summary 不一致"
        )
    if report.get("errors") != []:
        raise FinalizationError("inference report errors 必须为空")
    successes = report.get("successes")
    if not isinstance(successes, list) or len(successes) != expected_count:
        raise FinalizationError("inference report successes 数量错误")

    expected_inputs = {str(path.resolve()) for path in source_index.values()}
    seen_inputs: set[str] = set()
    seen_artifacts: set[Path] = set()
    evaluation_valid_fractions: list[float] = []
    per_image_metrics: dict[str, dict[str, float | None]] = {}
    for success in successes:
        if not isinstance(success, Mapping):
            raise FinalizationError("inference success row 必须是 mapping")
        source_path = Path(str(success.get("input"))).resolve()
        source_string = str(source_path)
        if source_string not in expected_inputs or source_string in seen_inputs:
            raise FinalizationError(f"inference input 缺失、重复或越界：{source_path}")
        seen_inputs.add(source_string)
        source_image = _read_image(source_path, label="source image")
        source_hw = tuple(int(value) for value in source_image.shape[:2])

        outputs = success.get("outputs")
        if not isinstance(outputs, Mapping):
            raise FinalizationError(f"outputs 不是 mapping：{source_path}")
        if set(outputs) != REQUIRED_OUTPUTS:
            raise FinalizationError(
                f"{source_path.name} artifact keys 错误；"
                f"missing={sorted(REQUIRED_OUTPUTS - set(outputs))}, "
                f"extra={sorted(set(outputs) - REQUIRED_OUTPUTS)}"
            )
        artifact_paths: dict[str, Path] = {}
        expected_artifacts = _expected_artifact_paths(output_dir, source_path.stem)
        for name, raw_path in outputs.items():
            artifact = Path(str(raw_path)).resolve()
            _ensure_under(artifact, output_dir, label=f"{source_path.name}:{name}")
            expected_artifact = expected_artifacts[name].resolve()
            if artifact != expected_artifact:
                raise FinalizationError(
                    f"{source_path.name}:{name} 路径/后缀错误；"
                    f"expected={expected_artifact}, actual={artifact}"
                )
            if artifact in seen_artifacts:
                raise FinalizationError(f"artifact 被多个输出重复引用：{artifact}")
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise FinalizationError(f"artifact 不存在或为空：{artifact}")
            artifact_paths[name] = artifact
            seen_artifacts.add(artifact)

        for name in (
            "image",
            "raw_image",
            "prior_image",
            "valid",
            "inpaint_mask",
            "evaluation_valid",
            "confidence",
        ):
            artifact_image = _read_image(artifact_paths[name], label=name)
            if tuple(artifact_image.shape[:2]) != source_hw:
                raise FinalizationError(
                    f"{source_path.name}:{name} 尺寸不一致；"
                    f"source={source_hw}, artifact={artifact_image.shape[:2]}"
                )
            if artifact_image.dtype != np.uint8:
                raise FinalizationError(
                    f"{source_path.name}:{name} dtype 必须为 uint8，"
                    f"实际为 {artifact_image.dtype}"
                )
            if artifact_image.ndim != 3 or artifact_image.shape[2] != 3:
                raise FinalizationError(
                    f"{source_path.name}:{name} 必须是三通道 RGB PNG"
                )
        valid = _read_image(artifact_paths["valid"], label="valid")
        inpaint = _read_image(artifact_paths["inpaint_mask"], label="inpaint_mask")
        evaluation_valid = _read_image(
            artifact_paths["evaluation_valid"], label="evaluation_valid"
        )
        for mask_name, mask_image in (
            ("valid", valid),
            ("inpaint_mask", inpaint),
            ("evaluation_valid", evaluation_valid),
        ):
            if not (
                np.array_equal(mask_image[..., 0], mask_image[..., 1])
                and np.array_equal(mask_image[..., 0], mask_image[..., 2])
            ):
                raise FinalizationError(
                    f"{source_path.name}:{mask_name} 三通道必须完全相同"
                )
            values = set(int(value) for value in np.unique(mask_image))
            if not values.issubset({0, 255}):
                raise FinalizationError(
                    f"{source_path.name}:{mask_name} 必须是 0/255 二值 mask；"
                    f"actual={sorted(values)[:8]}"
                )
        valid_bool = valid[..., 0] > 127 if valid.ndim == 3 else valid > 127
        inpaint_bool = inpaint[..., 0] > 127 if inpaint.ndim == 3 else inpaint > 127
        evaluation_bool = (
            evaluation_valid[..., 0] > 127
            if evaluation_valid.ndim == 3
            else evaluation_valid > 127
        )
        if not np.array_equal(evaluation_bool, valid_bool & ~inpaint_bool):
            raise FinalizationError(
                f"{source_path.name} evaluation_valid != valid & ~inpaint_mask"
            )
        observed_valid_fraction = float(valid_bool.mean())
        observed_inpaint_fraction = float(inpaint_bool.mean())
        observed_evaluation_fraction = float(evaluation_bool.mean())
        evaluation_valid_fractions.append(observed_evaluation_fraction)

        final_flow = _validate_flow(
            artifact_paths["flow"], source_hw, label="final flow"
        )
        _validate_flow(artifact_paths["prior_flow"], source_hw, label="prior flow")
        _validate_flow(
            artifact_paths["residual_flow"], source_hw, label="residual flow"
        )

        metadata = _load_strict_json(artifact_paths["metadata"])
        if not isinstance(metadata, Mapping):
            raise FinalizationError(f"metadata 不是 mapping：{artifact_paths['metadata']}")
        if metadata.get("checkpoint") != expected_checkpoint:
            raise FinalizationError(f"metadata checkpoint 不匹配：{source_path.name}")
        if not _typed_equal(
            metadata.get("checkpoint_artifact"), expected_checkpoint_artifact
        ):
            raise FinalizationError(
                f"metadata checkpoint_artifact 不匹配：{source_path.name}"
            )
        if metadata.get("warped") != source_string:
            raise FinalizationError(f"metadata warped 不匹配：{source_path.name}")
        if metadata.get("final_image_inpainted") is not True:
            raise FinalizationError(f"metadata 未声明 LAMA final：{source_path.name}")
        for field, observed in (
            ("valid_fraction", observed_valid_fraction),
            ("inpaint_fraction", observed_inpaint_fraction),
            ("evaluation_valid_fraction", observed_evaluation_fraction),
        ):
            recorded = _finite_number(
                metadata.get(field), label=f"{source_path.name}:metadata.{field}"
            )
            if not math.isclose(recorded, observed, rel_tol=0.0, abs_tol=1.0e-7):
                raise FinalizationError(
                    f"{source_path.name}:metadata.{field} 与实际 mask 不一致；"
                    f"recorded={recorded}, observed={observed}"
                )
        if not _typed_equal(
            metadata.get("teacher_prior_identity"),
            checkpoint_metadata["teacher_prior_identity"],
        ):
            raise FinalizationError(f"metadata teacher identity 不匹配：{source_path.name}")
        if not _typed_equal(
            metadata.get("residual_application"),
            checkpoint_metadata["residual_application"],
        ):
            raise FinalizationError(f"metadata residual contract 不匹配：{source_path.name}")
        if not _typed_equal(
            metadata.get("deployment_contract"),
            checkpoint_metadata["deployment_contract"],
        ):
            raise FinalizationError(f"metadata deployment contract 不匹配：{source_path.name}")
        observed_fold, observed_jacobian = _flow_topology_metrics(
            final_flow, valid_bool
        )
        fold_value = metadata.get("fold_rate")
        jacobian_value = metadata.get("jacobian_p01")
        for field, recorded_raw, observed, tolerance in (
            ("fold_rate", fold_value, observed_fold, 1.0e-7),
            ("jacobian_p01", jacobian_value, observed_jacobian, 1.0e-5),
        ):
            if observed is None:
                if recorded_raw is not None:
                    raise FinalizationError(
                        f"{source_path.name}:metadata.{field} 应为 null"
                    )
                continue
            recorded = _finite_number(
                recorded_raw,
                label=f"{source_path.name}:metadata.{field}",
            )
            if not math.isclose(
                recorded, observed, rel_tol=0.0, abs_tol=tolerance
            ):
                raise FinalizationError(
                    f"{source_path.name}:metadata.{field} 与持久化 flow 不一致；"
                    f"recorded={recorded}, observed={observed}"
                )
        per_image_metrics[source_path.stem] = {
            "valid_fraction": observed_valid_fraction,
            "inpaint_fraction": observed_inpaint_fraction,
            "evaluation_valid_fraction": observed_evaluation_fraction,
            # Keep None here so the policy gate, rather than this structural
            # parser, emits the stable non-finite quality failure code.
            "fold_rate": observed_fold,
            "jacobian_p01": observed_jacobian,
        }
    if seen_inputs != expected_inputs:
        raise FinalizationError("inference report 未覆盖完整 source 集合")
    directories = [path for path in output_dir.iterdir() if path.is_dir()]
    if directories:
        raise FinalizationError(f"推理输出含意外子目录：{directories[:4]}")
    actual_files = {
        path.resolve() for path in output_dir.iterdir() if path.is_file()
    }
    expected_files = seen_artifacts | {report_path.resolve()}
    if actual_files != expected_files:
        raise FinalizationError(
            "推理输出文件集合不精确；"
            f"missing={sorted(map(str, expected_files - actual_files))[:4]}, "
            f"extra={sorted(map(str, actual_files - expected_files))[:4]}"
        )
    validated = dict(report)
    validated["_finalizer_validation"] = {
        "mean_evaluation_valid_fraction": float(
            np.mean(evaluation_valid_fractions)
        ),
        "min_evaluation_valid_fraction": float(
            np.min(evaluation_valid_fractions)
        ),
        "max_evaluation_valid_fraction": float(
            np.max(evaluation_valid_fractions)
        ),
        "per_image": per_image_metrics,
    }
    return validated


def validate_line_report(
    report: Mapping[str, Any], *, expected_count: int
) -> dict[str, float]:
    candidates = report.get("candidates")
    if not isinstance(candidates, Mapping) or tuple(candidates) != LINE_CANDIDATES:
        raise FinalizationError(
            f"line evaluator candidate 顺序/集合错误："
            f"{tuple(candidates) if isinstance(candidates, Mapping) else candidates!r}"
        )
    weighted: dict[str, float] = {}
    for name in LINE_CANDIDATES:
        candidate = candidates[name]
        if not isinstance(candidate, Mapping):
            raise FinalizationError(f"line candidate 不是 mapping：{name}")
        summary = candidate.get("summary")
        if not isinstance(summary, Mapping):
            raise FinalizationError(f"line summary 缺失：{name}")
        expected_fields = {
            "indexed_images": expected_count,
            "paired_images": expected_count,
            "evaluated_images": expected_count,
            "missing_images": 0,
            "missing_masks": 0,
            "no_line_images": 0,
        }
        for field, expected in expected_fields.items():
            if summary.get(field) != expected:
                raise FinalizationError(
                    f"line summary {name}.{field}="
                    f"{summary.get(field)!r}, expected={expected}"
                )
        weighted[name] = _finite_number(
            summary.get("image_mean_orientation_error_deg_length_weighted"),
            label=f"{name}.image_mean_orientation_error_deg_length_weighted",
        )
        rows = candidate.get("per_image")
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise FinalizationError(f"line evaluator per_image 数量错误：{name}")
        for row in rows:
            if row.get("status") != "ok":
                raise FinalizationError(
                    f"line evaluator 非 ok row：{name}/{row.get('basename')}: "
                    f"{row.get('status')}"
                )
    return weighted


def expected_v33_extra_keys(source_keys: Sequence[str]) -> set[str]:
    return {
        f"{key}{suffix}"
        for key in source_keys
        for suffix in V33_COMPARISON_EXTRA_SUFFIXES
    }


def validate_comparison_report(
    report: Mapping[str, Any],
    *,
    source_keys: Sequence[str],
    expected_count: int,
) -> None:
    expected_names = list(COMPARISON_CANDIDATES)
    if report.get("candidate_order") != expected_names:
        raise FinalizationError("comparison candidate_order 错误")
    pairing = report.get("pairing")
    if not isinstance(pairing, Mapping):
        raise FinalizationError("comparison pairing 缺失")
    if pairing.get("source_count") != expected_count:
        raise FinalizationError("comparison source_count 错误")
    if pairing.get("complete_row_count") != expected_count:
        raise FinalizationError("comparison 存在不完整行")
    candidates = report.get("candidates")
    if not isinstance(candidates, Mapping):
        raise FinalizationError("comparison candidates 缺失")
    allowed_v33_extras = expected_v33_extra_keys(source_keys)
    for name in expected_names:
        item = candidates.get(name)
        if not isinstance(item, Mapping):
            raise FinalizationError(f"comparison candidate 缺失：{name}")
        if item.get("matched_count") != expected_count or item.get("missing_count") != 0:
            raise FinalizationError(f"comparison {name} 配对不完整")
        extras = set(item.get("extra_keys", []))
        if name in {"target_first", "target_second"}:
            if extras:
                raise FinalizationError(f"comparison {name} 含意外 extra：{sorted(extras)[:8]}")
        elif extras != allowed_v33_extras:
            raise FinalizationError(
                f"comparison {name} sidecar 集合错误；"
                f"missing={sorted(allowed_v33_extras - extras)[:8]}, "
                f"extra={sorted(extras - allowed_v33_extras)[:8]}"
            )


def _artifact_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _verify_checkpoint_artifacts(
    checkpoints: Mapping[str, Path],
    summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    """Ensure all checkpoint bytes stayed fixed throughout all-40 evaluation."""

    for name, path in checkpoints.items():
        current = _artifact_record(path)
        expected = {
            key: summaries[name].get(key)
            for key in ("path", "size_bytes", "mtime_ns", "sha256")
        }
        if not _typed_equal(current, expected):
            raise FinalizationError(
                f"{name} checkpoint 在 all40 运行期间发生变化"
            )


def _quality_validation_inputs(
    checkpoint_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Adapt already-validated checkpoint summaries to the pure gate schema."""

    result: dict[str, dict[str, Any]] = {}
    for source_name, quality_name in (
        ("anchor", "v33_anchor"),
        ("best", "v33_best"),
    ):
        summary = checkpoint_summaries[source_name]
        inference_config = summary.get("inference_critical_config")
        if not isinstance(inference_config, Mapping):
            raise FinalizationError(f"{source_name} 缺少 inference-critical config")
        model_config = inference_config.get("model")
        if not isinstance(model_config, Mapping):
            raise FinalizationError(
                f"{source_name} inference-critical config 缺少 model"
            )
        residual = summary.get("residual_application")
        if not isinstance(residual, Mapping):
            raise FinalizationError(f"{source_name} 缺少 residual_application")
        metrics = summary.get("metrics")
        if not isinstance(metrics, Mapping):
            raise FinalizationError(f"{source_name} 缺少 validation metrics")
        result[quality_name] = {
            "stage": summary.get("stage"),
            "feature_backend": model_config.get("feature_backend"),
            "epoch_index": summary.get("epoch_index"),
            "residual_scale": residual.get("scale"),
            "metrics": dict(metrics),
        }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified_v3_3_teacher_anchor.yaml")
    parser.add_argument(
        "--quality-policy",
        default="configs/typical_v33_quality_v2.yaml",
    )
    parser.add_argument("--run-root", default="runs/d2r_v3_3_teacher_anchor")
    parser.add_argument(
        "--anchor-checkpoint",
        default="runs/d2r_v3_3_teacher_anchor/unified/anchor.pt",
    )
    parser.add_argument(
        "--best-checkpoint",
        default="runs/d2r_v3_3_teacher_anchor/unified/best.pt",
    )
    parser.add_argument(
        "--latest-checkpoint",
        default="runs/d2r_v3_3_teacher_anchor/unified/latest.pt",
    )
    parser.add_argument(
        "--input-dir",
        default=(
            "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/"
            "test_silver_bullet_imgs/typical"
        ),
    )
    parser.add_argument(
        "--target-first-dir",
        default=(
            "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/"
            "test_silver_bullet_imgs/"
            "typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999"
        ),
    )
    parser.add_argument(
        "--target-second-dir",
        default=(
            "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/"
            "test_silver_bullet_imgs/"
            "typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999-2nd"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--output-root")
    parser.add_argument("--report-dir")
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    return parser


def _require_fresh_release_inference(skip_inference: bool) -> None:
    if skip_inference:
        raise FinalizationError(
            "production all-40 release 禁止 --skip-inference；"
            "必须由本次 finalizer 启动带 checkpoint provenance 的全新推理"
        )


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.expected_count <= 0:
        raise FinalizationError("--expected-count 必须为正整数")
    if not args.run_id or "/" in args.run_id or args.run_id in {".", ".."}:
        raise FinalizationError("--run-id 必须是单个非空路径组件")

    config_path = _resolve(args.config)
    quality_policy_path = _resolve(args.quality_policy)
    run_root = _resolve(args.run_root)
    anchor_checkpoint = _resolve_checkpoint_path(args.anchor_checkpoint)
    best_checkpoint = _resolve_checkpoint_path(args.best_checkpoint)
    latest_checkpoint = _resolve_checkpoint_path(args.latest_checkpoint)
    input_dir = _resolve(args.input_dir)
    target_first_dir = _resolve(args.target_first_dir)
    target_second_dir = _resolve(args.target_second_dir)
    output_root = (
        _resolve(args.output_root)
        if args.output_root
        else run_root / "typical_final" / args.run_id
    )
    report_dir = (
        _resolve(args.report_dir)
        if args.report_dir
        else REPO_ROOT / "reports" / "typical_v33_teacher_anchor" / args.run_id
    )
    anchor_output = output_root / "v33_anchor"
    best_output = output_root / "v33_best"

    if not config_path.is_file():
        raise FinalizationError(f"config 不存在：{config_path}")
    if not quality_policy_path.is_file():
        raise FinalizationError(f"quality policy 不存在：{quality_policy_path}")
    try:
        quality_policy = parse_quality_policy_yaml(
            quality_policy_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, QualityGateInputError) as error:
        raise FinalizationError(f"quality policy 无效：{error}") from error
    config_artifact = _artifact_record(config_path)
    quality_policy_artifact = _artifact_record(quality_policy_path)
    config = load_config(config_path)
    expected_total_epochs = int(config["train"]["epochs"])
    expected_best_metric_name = str(config["train"].get("best_metric", "epe"))
    expected_best_metric_mode = str(
        config["train"].get("best_metric_mode", "min")
    ).lower()
    current_inference_config = inference_critical_config(config)
    indexes = validate_image_sets(
        input_dir,
        target_first_dir,
        target_second_dir,
        expected_count=args.expected_count,
    )
    checkpoint_summaries = {
        "anchor": checkpoint_summary(anchor_checkpoint),
        "best": checkpoint_summary(best_checkpoint),
        "latest": checkpoint_summary(latest_checkpoint),
    }
    validate_checkpoint_set(
        checkpoint_summaries["anchor"],
        checkpoint_summaries["best"],
        checkpoint_summaries["latest"],
        expected_total_epochs=expected_total_epochs,
        expected_best_metric_name=expected_best_metric_name,
        expected_best_metric_mode=expected_best_metric_mode,
        current_inference_config=current_inference_config,
    )
    print(
        "[ok] preflight: "
        f"images={args.expected_count}, latest="
        f"{checkpoint_summaries['latest']['completed_epochs']}/"
        f"{expected_total_epochs}"
    )
    if args.preflight_only:
        return None

    _require_fresh_release_inference(args.skip_inference)

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise FinalizationError("完整 all-40 推理要求至少一张可用 CUDA GPU")

    lock_path = run_root / ".finalize_typical_v33.lock"
    with _exclusive_lock(lock_path):
        if report_dir.exists() and any(report_dir.iterdir()):
            raise FinalizationError(
                f"report 目录非空；拒绝覆盖已有收尾结果：{report_dir}"
            )
        _require_fresh_directory(anchor_output)
        _require_fresh_directory(best_output)
        _run_inference(
            checkpoint=anchor_checkpoint,
            checkpoint_artifact=checkpoint_summaries["anchor"],
            output_dir=anchor_output,
            input_dir=input_dir,
            config=config_path,
        )
        _run_inference(
            checkpoint=best_checkpoint,
            checkpoint_artifact=checkpoint_summaries["best"],
            output_dir=best_output,
            input_dir=input_dir,
            config=config_path,
        )

        anchor_inference = validate_inference_output(
            anchor_output,
            checkpoint=anchor_checkpoint,
            checkpoint_metadata=checkpoint_summaries["anchor"],
            source_index=indexes["warped"],
            expected_count=args.expected_count,
        )
        best_inference = validate_inference_output(
            best_output,
            checkpoint=best_checkpoint,
            checkpoint_metadata=checkpoint_summaries["best"],
            source_index=indexes["warped"],
            expected_count=args.expected_count,
        )

        # Hash the exact visual evidence before computing any image-derived
        # quality metric.  A complete re-enumeration/re-hash below then proves
        # that the LSD report and the review template refer to these bytes,
        # rather than to files replaced between evaluation and publication.
        try:
            evidence = build_typical_evidence(
                input_dir,
                target_first_dir,
                target_second_dir,
                anchor_output,
                best_output,
            )
            review_template = make_typical_review_template(evidence)
        except TypicalReviewError as error:
            raise FinalizationError(f"manual review evidence 无效：{error}") from error

        report_dir.mkdir(parents=True, exist_ok=True)
        line_path = report_dir / "all40_line_proxy.json"
        line_report = evaluate_dataset(
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
        weighted_proxy = validate_line_report(
            line_report, expected_count=args.expected_count
        )
        _atomic_json(line_report, line_path)

        comparison_html = report_dir / "all40_comparison.html"
        comparison_json = report_dir / "all40_comparison.json"
        comparison_report = generate_comparison(
            input_dir,
            [
                ("target_first", target_first_dir),
                ("target_second", target_second_dir),
                ("v33_anchor", anchor_output),
                ("v33_best", best_output),
            ],
            comparison_html,
            output_json=comparison_json,
            title="Typical all-40: targets vs v3.3 anchor/best",
        )
        validate_comparison_report(
            comparison_report,
            source_keys=tuple(path.stem for path in indexes["warped"].values()),
            expected_count=args.expected_count,
        )
        if not comparison_html.is_file() or comparison_html.stat().st_size == 0:
            raise FinalizationError("comparison HTML 未生成")
        if not comparison_json.is_file() or comparison_json.stat().st_size == 0:
            raise FinalizationError("comparison JSON 未生成")

        quality_proxy = {
            "notice": (
                "LSD axis alignment is a no-reference proxy and cannot prove "
                "rectification quality or content fidelity. Direct target "
                "comparisons below use full-frame v3.3 scores; valid-mask "
                "scores are separate diagnostics on a smaller support."
            ),
            "metric": "image_mean_orientation_error_deg_length_weighted",
            "values": weighted_proxy,
            "evaluation_valid_fraction": {
                "v33_anchor": {
                    key: value
                    for key, value in anchor_inference[
                        "_finalizer_validation"
                    ].items()
                    if key != "per_image"
                },
                "v33_best": {
                    key: value
                    for key, value in best_inference[
                        "_finalizer_validation"
                    ].items()
                    if key != "per_image"
                },
            },
            "v33_anchor_at_most_target_first": (
                weighted_proxy["v33_anchor_full"]
                <= weighted_proxy["target_first"]
            ),
            "v33_best_at_most_target_first": (
                weighted_proxy["v33_best_full"]
                <= weighted_proxy["target_first"]
            ),
            "v33_anchor_at_most_target_second": (
                weighted_proxy["v33_anchor_full"]
                <= weighted_proxy["target_second"]
            ),
            "v33_best_at_most_target_second": (
                weighted_proxy["v33_best_full"]
                <= weighted_proxy["target_second"]
            ),
        }

        try:
            quality_report = evaluate_typical_quality(
                quality_policy,
                validation=_quality_validation_inputs(checkpoint_summaries),
                line_report=line_report,
                inference_metrics={
                    "v33_anchor": anchor_inference["_finalizer_validation"][
                        "per_image"
                    ],
                    "v33_best": best_inference["_finalizer_validation"][
                        "per_image"
                    ],
                },
            )
        except QualityGateInputError as error:
            raise FinalizationError(
                f"automatic quality gate 输入契约无效：{error}"
            ) from error
        quality_report_path = report_dir / "automatic_quality_gate.json"
        _atomic_json(quality_report, quality_report_path)

        try:
            validate_typical_evidence(evidence, verify_files=True)
        except TypicalReviewError as error:
            raise FinalizationError(
                f"manual review evidence 在评估期间发生变化：{error}"
            ) from error
        evidence_path = report_dir / "review_evidence.json"
        review_template_path = report_dir / "quality_review_template.json"
        _atomic_json(evidence, evidence_path)
        _atomic_json(review_template, review_template_path)

        # Config and policy govern every inference/acceptance decision.  Abort
        # rather than publish a mixed-protocol manifest if either changed
        # during the potentially long two-checkpoint inference run.
        if _artifact_record(config_path) != config_artifact:
            raise FinalizationError("config 在 all40 运行期间发生变化")
        if _artifact_record(quality_policy_path) != quality_policy_artifact:
            raise FinalizationError("quality policy 在 all40 运行期间发生变化")
        _verify_checkpoint_artifacts(
            {
                "anchor": anchor_checkpoint,
                "best": best_checkpoint,
                "latest": latest_checkpoint,
            },
            checkpoint_summaries,
        )

        evaluation_manifest_path = report_dir / "evaluation_manifest.json"
        automatic_status = "passed" if quality_report["passed"] else "failed"
        manifest = {
            "schema": "diffusion2raft.typical_evaluation_manifest",
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_status": "evaluation_complete",
            "selected_candidate": REVIEW_CANDIDATE_NAME,
            "release_ready": False,
            "run_id": args.run_id,
            "expected_count": args.expected_count,
            "config": config_artifact,
            "inputs": {
                "warped": str(input_dir),
                "target_first": str(target_first_dir),
                "target_second": str(target_second_dir),
            },
            "checkpoints": checkpoint_summaries,
            "outputs": {
                "v33_anchor": str(anchor_output),
                "v33_best": str(best_output),
            },
            "inference_reports": {
                "v33_anchor": {
                    "checkpoint": anchor_inference["checkpoint"],
                    "input_count": anchor_inference["input_count"],
                    "success_count": anchor_inference["success_count"],
                    "error_count": anchor_inference["error_count"],
                    "artifact": _artifact_record(
                        anchor_output / "inference_report.json"
                    ),
                },
                "v33_best": {
                    "checkpoint": best_inference["checkpoint"],
                    "input_count": best_inference["input_count"],
                    "success_count": best_inference["success_count"],
                    "error_count": best_inference["error_count"],
                    "artifact": _artifact_record(best_output / "inference_report.json"),
                },
            },
            "reports": {
                "line_proxy": _artifact_record(line_path),
                "comparison_json": _artifact_record(comparison_json),
                "comparison_html": _artifact_record(comparison_html),
                "automatic_quality_gate": _artifact_record(
                    quality_report_path
                ),
                "review_evidence": _artifact_record(evidence_path),
                "quality_review_template": _artifact_record(
                    review_template_path
                ),
            },
            "automatic_quality_gate": {
                "status": automatic_status,
                "policy": quality_policy_artifact,
                "failure_count": len(quality_report["failures"]),
                "failures": quality_report["failures"],
                "summary": quality_report["summary"],
            },
            "manual_quality_review": {
                "status": "pending",
                "candidate": REVIEW_CANDIDATE_NAME,
                "evidence_sha256": evidence["evidence_sha256"],
                "required_reviewed_count": args.expected_count,
            },
            "quality_proxy": quality_proxy,
        }
        _atomic_json(manifest, evaluation_manifest_path)
        print(f"[ok] all-40 evaluation complete: {evaluation_manifest_path}")
        print(f"[review] visual comparison: {comparison_html}")
        print(f"[review] pending template: {review_template_path}")
        if not quality_report["passed"]:
            codes = sorted(
                {str(item["code"]) for item in quality_report["failures"]}
            )
            raise FinalizationError(
                "automatic quality gate 未通过；"
                f"failure_codes={codes}, report={quality_report_path}"
            )
        return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run(args)
    except FinalizationError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 65
    except Exception as error:
        print(f"[error] unexpected finalization failure: {error}", file=sys.stderr)
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
