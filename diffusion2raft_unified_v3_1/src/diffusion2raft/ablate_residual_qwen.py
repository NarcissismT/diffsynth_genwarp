"""Controlled correlation/Qwen-path/residual-scale validation sweep.

The checkpoint's saved config is authoritative.  For each temperature and
Qwen path mode the expensive model (including Qwen) runs once per batch.  Its
raw recurrent residuals are then recomposed offline at every requested scale:

    F_lambda(x) = lambda R(x) + B(x + lambda R(x))

This is exactly the model's backward-flow composition, not a linear blend of
the already-composed final flow.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
import sys
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .data import DocumentFlowDataset
from .geometry import compose_backward_flows, flow_valid_mask
from .losses import RectificationLoss
from .models import build_rectifier
from .train import (
    _checkpoint_correlation_temperature,
    _raw_model,
    _to_device,
)


METRIC_KEYS = (
    "epe",
    "epe_p95",
    "line_epe",
    "line_straightness_error",
    "edge_epe",
    "prior_epe",
    "epe_gain",
    "relative_epe_gain",
    "final_win_rate",
    "fold_rate",
    "jacobian_p01",
    "residual_epe",
    "residual_p95",
    "applied_residual_p95",
    "feature_confidence",
    "matching_feature_confidence",
    "context_feature_confidence",
    "qwen_match_epe",
    "qwen_advantage",
    "qwen_win_rate",
)
QWEN_MODES = ("both", "none", "matching_only", "context_only")
DEFAULT_TEMPERATURES = (9.797959, 3.1301691647577132, 1.0)
DEFAULT_QWEN_MODES = ("both", "none")
DEFAULT_TRAINING_TEMPERATURE_QWEN_MODES = (
    "matching_only",
    "context_only",
)
REPORT_VERSION = 3
DECISION_VERSION = 1
DECISION_RESIDUAL_SCALES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
POSITIVE_DECISION_RESIDUAL_SCALES = DECISION_RESIDUAL_SCALES[1:]
TARGET_FOLD_RATE = 4e-4


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _number_key(value: float) -> str:
    """Use Python's shortest round-trippable float representation as a key."""

    return repr(float(value))


def build_cell_plan(
    temperatures: list[float],
    *,
    training_temperature: float,
    qwen_modes: list[str],
    training_temperature_qwen_modes: list[str],
) -> list[tuple[float, str]]:
    """Build the staged protocol without the wasteful full Cartesian product."""

    cells: list[tuple[float, str]] = []
    seen: set[tuple[float, str]] = set()

    def add(temperature: float, mode: str) -> None:
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"correlation temperature must be finite and positive, got {temperature}"
            )
        if mode not in QWEN_MODES:
            raise ValueError(f"unknown Qwen mode {mode!r}; choose from {QWEN_MODES}")
        cell = (temperature, mode)
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)

    for temperature in temperatures:
        for mode in qwen_modes:
            add(temperature, mode)
    for mode in training_temperature_qwen_modes:
        add(training_temperature, mode)
    if not cells:
        raise ValueError("ablation cell plan must not be empty")

    # Nested JSON result keys must retain a one-to-one mapping to numeric cells.
    keyed: dict[tuple[str, str], float] = {}
    for temperature, mode in cells:
        key = (_number_key(temperature), mode)
        previous = keyed.get(key)
        if previous is not None and previous != temperature:
            raise ValueError(
                "distinct temperatures collide after JSON key formatting: "
                f"{previous!r} and {temperature!r}"
            )
        keyed[key] = temperature
    return cells


def _checkpoint_payload_and_identity(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash and load the same open checkpoint inode, even if the path is replaced."""

    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(0)
        try:
            payload = torch.load(handle, map_location="cpu", weights_only=False)
        except TypeError:
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu")
        after = os.fstat(handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"checkpoint changed while hashing/loading: {resolved}")
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload must be a mapping")
    return payload, {
        "resolved_path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
    }


def _file_metadata(path: Path, *, relative_to: Path) -> dict[str, Any]:
    lstat = path.lstat()
    record: dict[str, Any] = {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "size_bytes": int(lstat.st_size),
        "mtime_ns": int(lstat.st_mtime_ns),
        "kind": "symlink" if path.is_symlink() else "file",
    }
    if path.is_symlink():
        record["symlink_target"] = os.readlink(path)
        try:
            target_stat = path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Qwen model contains a broken symlink: {path}") from exc
        record["target_size_bytes"] = int(target_stat.st_size)
        record["target_mtime_ns"] = int(target_stat.st_mtime_ns)
    return record


def qwen_model_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Describe external Qwen files without hashing multi-gigabyte contents."""

    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise RuntimeError("checkpoint config.model must be a mapping")
    if str(model_config.get("feature_backend", "qwen")).lower() != "qwen":
        return {"kind": "not_applicable", "feature_backend": "lite"}

    qwen_config = config.get("qwen")
    if not isinstance(qwen_config, dict):
        raise RuntimeError("Qwen checkpoint config.qwen must be a mapping")
    configured_id = qwen_config.get("model_id")
    if not isinstance(configured_id, str) or not configured_id:
        raise RuntimeError("Qwen checkpoint config requires a non-empty model_id")
    candidate = Path(configured_id).expanduser()
    if not candidate.exists():
        return {
            "kind": "remote_model_id",
            "configured_model_id": configured_id,
            "revision": qwen_config.get("revision"),
            "local_files_only": bool(qwen_config.get("local_files_only", False)),
        }

    resolved = candidate.resolve(strict=True)
    if resolved.is_file():
        files = [_file_metadata(resolved, relative_to=resolved.parent)]
    elif resolved.is_dir():
        files = [
            _file_metadata(path, relative_to=resolved)
            for path in sorted(
                (item for item in resolved.rglob("*") if item.is_file() or item.is_symlink()),
                key=lambda item: item.relative_to(resolved).as_posix(),
            )
        ]
    else:
        raise RuntimeError(f"Qwen model_id is not a regular file/directory: {resolved}")
    root_stat = resolved.stat()
    manifest_sha256 = _json_identity(files)
    return {
        "kind": "local_file_manifest",
        "configured_model_id": configured_id,
        "resolved_model_id": str(resolved),
        "root_mtime_ns": int(root_stat.st_mtime_ns),
        "file_count": len(files),
        "total_file_size_bytes": sum(
            int(item.get("target_size_bytes", item["size_bytes"])) for item in files
        ),
        "manifest_sha256": manifest_sha256,
        "files": files,
    }


def _source_code_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    files: list[dict[str, Any]] = []
    for path in sorted(package_root.rglob("*.py")):
        content = path.read_bytes()
        files.append(
            {
                "relative_path": path.relative_to(package_root).as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "package_root": str(package_root),
        "manifest_sha256": _json_identity(files),
        "files": files,
    }


def _manifest_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    content = resolved.read_bytes()
    stat = resolved.stat()
    return {
        "resolved_path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        identity.update(
            cuda_device_name=torch.cuda.get_device_name(index),
            cuda_capability=list(torch.cuda.get_device_capability(index)),
            cudnn_version=torch.backends.cudnn.version(),
        )
    return identity


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            # Some network/FUSE filesystems provide atomic rename but do not
            # implement directory fsync.  The report file itself was fsynced.
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
                raise
    finally:
        os.close(directory_fd)


@contextlib.contextmanager
def _exclusive_report_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another ablation owns report lock: {lock_path}") from exc
        yield


def outputs_at_residual_scale(
    outputs: dict[str, Any], scale: float, *, source_size: tuple[int, int]
) -> dict[str, Any]:
    """Shallow-copy model outputs and apply raw residuals at ``scale``."""

    scale = float(scale)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError(f"residual scale must be finite and in [0,1], got {scale}")
    prior_flow = outputs["prior_flow"]
    raw_residuals = outputs.get("raw_residuals")
    if not isinstance(raw_residuals, list) or not raw_residuals:
        raise ValueError("unified outputs require a non-empty raw_residuals list")

    recomposed = dict(outputs)
    if scale == 1.0 and float(outputs.get("residual_application_scale", -1.0)) == 1.0:
        flows = outputs.get("flows")
        residuals = outputs.get("residuals")
        if (
            isinstance(flows, list)
            and len(flows) == len(raw_residuals)
            and isinstance(residuals, list)
            and len(residuals) == len(raw_residuals)
            and isinstance(outputs.get("final_valid"), Tensor)
        ):
            # The expensive online forward has already produced this exact cell.
            # Reuse it instead of repeating every grid_sample composition.
            return recomposed
    if scale == 0.0:
        residuals = [residual * 0.0 for residual in raw_residuals]
        flows = [prior_flow for _ in raw_residuals]
        final_valid = outputs.get("prior_valid")
        if not isinstance(final_valid, Tensor):
            final_valid = flow_valid_mask(prior_flow, source_size)
    else:
        residuals = (
            raw_residuals
            if scale == 1.0
            else [residual * scale for residual in raw_residuals]
        )
        flows = [
            compose_backward_flows(prior_flow, residual)
            for residual in residuals
        ]
        final_valid = flow_valid_mask(flows[-1], source_size)
    recomposed.update(
        residuals=residuals,
        flows=flows,
        final_flow=flows[-1],
        final_valid=final_valid,
        residual_application_scale=scale,
    )
    return recomposed


def _set_qwen_mode(model: torch.nn.Module, mode: str) -> None:
    if mode not in QWEN_MODES:
        raise ValueError(f"unknown Qwen mode {mode!r}; choose from {QWEN_MODES}")
    raw = _raw_model(model)
    raw.qwen_off = mode == "none"
    raw.qwen_matching_off = mode == "context_only"
    raw.qwen_context_off = mode == "matching_only"


def _authoritative_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint is missing its authoritative config")
    if str(payload.get("stage")) != "unified":
        raise RuntimeError("ablation requires checkpoint stage='unified'")
    if str(config.get("train", {}).get("stage")) != "unified":
        raise RuntimeError("checkpoint config.train.stage must be 'unified'")
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise RuntimeError("checkpoint config.model must be a mapping")
    configured_prior = str(model_config.get("prior_backend", "learned")).lower()
    declared_prior = str(payload.get("prior_backend", configured_prior)).lower()
    if configured_prior != "learned" or declared_prior != "learned":
        raise RuntimeError(
            "this formal sweep currently accepts learned-prior checkpoints only; "
            "TorchScript teachers require identity/deployment-contract validation"
        )
    return config


def _load_model(
    payload: dict[str, Any], *, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = _authoritative_config(payload)
    model_config = config["model"]
    model = build_rectifier(
        dict(model_config),
        dict(config.get("qwen", {})),
        stage="unified",
        device=device,
    ).to(device)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint model state is missing")
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [
        key for key in missing
        if not key.startswith("diffusion_encoder._pipeline")
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch; missing={bad_missing[:12]}, "
            f"unexpected={unexpected[:12]}"
        )
    return model.eval(), config


def _build_loader(config: dict[str, Any]) -> DataLoader:
    manifest = Path(config["data"]["val_manifest"]).resolve(strict=True)
    dataset = DocumentFlowDataset(
        manifest,
        tuple(config["data"]["work_size"]),
        augment_guide=False,
    )
    num_workers = int(config["data"].get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(config["data"].get("batch_size", 1)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in ablation report: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read resume report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("resume report root must be a mapping")
    return value


def _cell_coordinates(protocol: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(cell["temperature_key"]), str(cell["qwen_mode"]))
        for cell in protocol["cells"]
    }


def _validate_cell_result(
    value: Any,
    *,
    temperature: float,
    mode: str,
    scale_keys: list[str],
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "temperature",
        "qwen_mode",
        "evaluated_batches",
        "evaluated_samples",
        "metrics",
    }:
        raise RuntimeError("resume report contains an invalid cell schema")
    if type(value["temperature"]) is not float or value["temperature"] != temperature:
        raise RuntimeError("resume cell temperature disagrees with protocol")
    if value["qwen_mode"] != mode:
        raise RuntimeError("resume cell Qwen mode disagrees with protocol")
    for count_key in ("evaluated_batches", "evaluated_samples"):
        count = value[count_key]
        if isinstance(count, bool) or not isinstance(count, Integral) or int(count) <= 0:
            raise RuntimeError(f"resume cell {count_key} must be a positive integer")
    metrics = value["metrics"]
    if not isinstance(metrics, dict) or list(metrics) != scale_keys:
        raise RuntimeError("resume cell residual-scale keys disagree with protocol")
    for scale_key, scale_metrics in metrics.items():
        if not isinstance(scale_metrics, dict) or tuple(scale_metrics) != METRIC_KEYS:
            raise RuntimeError(
                f"resume metrics schema is invalid at residual scale {scale_key}"
            )
        for metric, number in scale_metrics.items():
            if isinstance(number, bool) or not isinstance(number, Real):
                raise RuntimeError(f"resume metric {metric} must be a real number")
            if not math.isfinite(float(number)):
                raise RuntimeError(f"resume metric {metric} must be finite")


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    """Rank candidates by EPE, then by conservative geometric tie-breakers."""

    metrics = candidate["metrics"]
    return (
        float(metrics["epe"]),
        float(metrics["fold_rate"]),
        -float(metrics["final_win_rate"]),
        float(candidate["residual_scale"]),
    )


def _decision_candidate(
    cell: dict[str, Any], *, temperature_key: str, scale: float
) -> dict[str, Any] | None:
    metrics = cell["metrics"].get(_number_key(scale))
    if not isinstance(metrics, dict):
        return None
    return {
        "temperature": float(cell["temperature"]),
        "temperature_key": temperature_key,
        "qwen_mode": str(cell["qwen_mode"]),
        "residual_scale": float(scale),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }


def _quality_gate(
    metrics: dict[str, Any], prior_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the user-specified safety/quality target without inventing data."""

    definitions = (
        ("epe_gain_positive", "epe_gain", ">", 0.0, metrics),
        ("majority_pixel_win", "final_win_rate", ">", 0.5, metrics),
        ("fold_rate_safe", "fold_rate", "<", TARGET_FOLD_RATE, metrics),
        (
            "line_epe_better_than_prior",
            "line_epe",
            "<",
            prior_metrics.get("line_epe"),
            metrics,
        ),
        (
            "line_straightness_better_than_prior",
            "line_straightness_error",
            "<",
            prior_metrics.get("line_straightness_error"),
            metrics,
        ),
    )
    checks: dict[str, dict[str, Any]] = {}
    insufficient: list[str] = []
    for name, metric_name, operator, threshold, source in definitions:
        value = source.get(metric_name)
        available = (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(float(value))
            and not isinstance(threshold, bool)
            and isinstance(threshold, Real)
            and math.isfinite(float(threshold))
        )
        if not available:
            checks[name] = {
                "available": False,
                "metric": metric_name,
                "operator": operator,
                "threshold": None,
                "value": None,
                "passed": None,
            }
            insufficient.append(
                f"{name}: candidate or prior metric is unavailable"
            )
            continue
        numeric_value = float(value)
        numeric_threshold = float(threshold)
        passed = (
            numeric_value > numeric_threshold
            if operator == ">"
            else numeric_value < numeric_threshold
        )
        checks[name] = {
            "available": True,
            "metric": metric_name,
            "operator": operator,
            "threshold": numeric_threshold,
            "value": numeric_value,
            "passed": bool(passed),
        }
    if insufficient:
        status = "insufficient"
        passed: bool | None = None
    else:
        passed = all(bool(check["passed"]) for check in checks.values())
        status = "pass" if passed else "fail"
    return {
        "status": status,
        "passed": passed,
        "checks": checks,
        "insufficient_reasons": insufficient,
        "prior_line_metrics": {
            "source": "lambda_zero_exact_prior",
            "line_epe": (
                float(prior_metrics["line_epe"])
                if isinstance(prior_metrics.get("line_epe"), Real)
                and not isinstance(prior_metrics.get("line_epe"), bool)
                and math.isfinite(float(prior_metrics["line_epe"]))
                else None
            ),
            "line_straightness_error": (
                float(prior_metrics["line_straightness_error"])
                if isinstance(prior_metrics.get("line_straightness_error"), Real)
                and not isinstance(
                    prior_metrics.get("line_straightness_error"), bool
                )
                and math.isfinite(float(prior_metrics["line_straightness_error"]))
                else None
            ),
        },
    }


def _residual_verdict(
    prior: dict[str, Any] | None,
    best_positive: dict[str, Any] | None,
    *,
    has_all_required_scales: bool,
) -> str:
    if prior is None or best_positive is None or not has_all_required_scales:
        return "insufficient"
    prior_epe = float(prior["metrics"]["epe"])
    positive_epe = float(best_positive["metrics"]["epe"])
    scale = float(best_positive["residual_scale"])
    if positive_epe < prior_epe:
        return "full_residual_best" if scale == 1.0 else "over_correction"
    if scale == 1.0:
        return "needs_pixel_gate"
    return "residual_direction_wrong"


def _decision_summary_text(decision: dict[str, Any]) -> str:
    if decision["status"] != "ready":
        reasons = decision["insufficient_reasons"]
        detail = "; ".join(reasons[:3])
        if len(reasons) > 3:
            detail += f"; and {len(reasons) - 3} more"
        return f"Formal sweep decision: INSUFFICIENT ({detail})"

    best = decision["global_best"]
    comparison = decision["training_temperature_qwen_comparison"]
    gate = best["quality_gate"]
    failed = [
        name
        for name, check in gate["checks"].items()
        if check["passed"] is False
    ]
    gate_text = gate["status"].upper()
    if failed:
        gate_text += "[" + ",".join(failed) + "]"
    return (
        "Formal sweep decision: "
        f"best T={best['temperature']:.6g}, qwen={best['qwen_mode']}, "
        f"lambda={best['residual_scale']:.2f}, "
        f"EPE={best['metrics']['epe']:.6f}; "
        f"residual={best['residual_verdict']}; "
        f"Qwen@train={comparison['best_mode']}; target={gate_text}"
    )


def _decision_from_validated_cells(report: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic decision after cell/progress validation."""

    protocol = report["protocol"]
    results = report["results"]
    reasons: list[str] = []
    protocol_scales = {
        float(scale): _number_key(float(scale))
        for scale in protocol.get("residual_scales", [])
        if isinstance(scale, Real) and not isinstance(scale, bool)
    }
    missing_scales = [
        scale for scale in DECISION_RESIDUAL_SCALES if scale not in protocol_scales
    ]
    if missing_scales:
        reasons.append(
            "missing required residual scales: "
            + ", ".join(_number_key(scale) for scale in missing_scales)
        )

    training_value = protocol.get("training_correlation_temperature")
    if (
        isinstance(training_value, bool)
        or not isinstance(training_value, Real)
        or not math.isfinite(float(training_value))
    ):
        training_temperature: float | None = None
        reasons.append("training correlation temperature is unavailable")
    else:
        training_temperature = float(training_value)

    if protocol.get("max_batches", "missing") is not None:
        reasons.append("max_batches is set; the report is not a full validation sweep")
    validation_samples = protocol.get("validation_samples")
    if (
        isinstance(validation_samples, bool)
        or not isinstance(validation_samples, Integral)
        or int(validation_samples) <= 0
    ):
        expected_samples: int | None = None
        reasons.append("protocol validation_samples is unavailable")
    else:
        expected_samples = int(validation_samples)

    cell_decisions: list[dict[str, Any]] = []
    training_cells: dict[str, dict[str, Any]] = {}
    for cell_spec in protocol["cells"]:
        temperature_key = str(cell_spec["temperature_key"])
        mode = str(cell_spec["qwen_mode"])
        cell = results[temperature_key][mode]
        if expected_samples is not None and int(cell["evaluated_samples"]) != expected_samples:
            reasons.append(
                f"cell T={temperature_key},qwen={mode} evaluated "
                f"{cell['evaluated_samples']}/{expected_samples} samples"
            )

        candidates = [
            candidate
            for scale in DECISION_RESIDUAL_SCALES
            if (
                candidate := _decision_candidate(
                    cell, temperature_key=temperature_key, scale=scale
                )
            )
            is not None
        ]
        candidate_by_scale = {
            float(candidate["residual_scale"]): candidate
            for candidate in candidates
        }
        prior = candidate_by_scale.get(0.0)
        positive = [
            candidate_by_scale[scale]
            for scale in POSITIVE_DECISION_RESIDUAL_SCALES
            if scale in candidate_by_scale
        ]
        all_scales = len(candidate_by_scale) == len(DECISION_RESIDUAL_SCALES)
        best_positive = min(positive, key=_candidate_sort_key) if positive else None
        best_overall = min(candidates, key=_candidate_sort_key) if candidates else None
        verdict = _residual_verdict(
            prior,
            best_positive,
            has_all_required_scales=all_scales,
        )
        quality_gate = (
            _quality_gate(best_overall["metrics"], prior["metrics"])
            if best_overall is not None and prior is not None
            else {
                "status": "insufficient",
                "passed": None,
                "checks": {},
                "insufficient_reasons": ["lambda=0 or candidate metrics unavailable"],
                "prior_line_metrics": {
                    "source": "lambda_zero_exact_prior",
                    "line_epe": None,
                    "line_straightness_error": None,
                },
            }
        )
        cell_decision = {
            "temperature": float(cell["temperature"]),
            "temperature_key": temperature_key,
            "qwen_mode": mode,
            "prior": prior,
            "best_positive": best_positive,
            "best_positive_lambda": (
                float(best_positive["residual_scale"])
                if best_positive is not None
                else None
            ),
            "best_overall": best_overall,
            "best_lambda": (
                float(best_overall["residual_scale"])
                if best_overall is not None
                else None
            ),
            "full_residual": candidate_by_scale.get(1.0),
            "residual_verdict": verdict,
            "quality_gate": quality_gate,
        }
        cell_decisions.append(cell_decision)
        if (
            training_temperature is not None
            and float(cell["temperature"]) == training_temperature
        ):
            if mode in training_cells:
                reasons.append(f"duplicate training-temperature Qwen mode: {mode}")
            training_cells[mode] = cell_decision

    missing_modes = [mode for mode in QWEN_MODES if mode not in training_cells]
    if missing_modes:
        reasons.append(
            "missing training-temperature Qwen modes: " + ", ".join(missing_modes)
        )

    qwen_rows: list[dict[str, Any]] = []
    if not missing_modes:
        none_best = training_cells["none"]["best_positive"]
        if none_best is None:
            reasons.append("training-temperature qwen=none has no positive lambda")
        else:
            none_epe = float(none_best["metrics"]["epe"])
            sortable: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for mode in QWEN_MODES:
                cell_decision = training_cells[mode]
                best_positive = cell_decision["best_positive"]
                full_residual = cell_decision["full_residual"]
                if best_positive is None or full_residual is None:
                    reasons.append(
                        f"training-temperature qwen={mode} lacks positive/full residual metrics"
                    )
                    continue
                row = {
                    "qwen_mode": mode,
                    "best_positive_lambda": float(
                        best_positive["residual_scale"]
                    ),
                    "best_positive_epe": float(best_positive["metrics"]["epe"]),
                    "full_residual_epe": float(full_residual["metrics"]["epe"]),
                    "epe_improvement_vs_none_best_positive": (
                        none_epe - float(best_positive["metrics"]["epe"])
                    ),
                    "residual_verdict": cell_decision["residual_verdict"],
                }
                sortable.append((row, best_positive))
            sortable.sort(key=lambda item: _candidate_sort_key(item[1]))
            qwen_rows = [row for row, _ in sortable]
            for rank, row in enumerate(qwen_rows, 1):
                row["rank"] = rank

    global_candidates = [
        cell["best_overall"]
        for cell in cell_decisions
        if cell["best_overall"] is not None
    ]
    global_candidate = (
        min(global_candidates, key=_candidate_sort_key)
        if global_candidates
        else None
    )
    global_best: dict[str, Any] | None = None
    if global_candidate is None:
        reasons.append("no residual candidate is available")
    else:
        matching_cell = next(
            cell
            for cell in cell_decisions
            if cell["temperature_key"] == global_candidate["temperature_key"]
            and cell["qwen_mode"] == global_candidate["qwen_mode"]
        )
        global_best = {
            **global_candidate,
            "residual_verdict": matching_cell["residual_verdict"],
            "quality_gate": matching_cell["quality_gate"],
        }

    decision: dict[str, Any] = {
        "version": DECISION_VERSION,
        "status": "ready" if not reasons else "insufficient",
        "required_residual_scales": list(DECISION_RESIDUAL_SCALES),
        "thresholds": {
            "epe_gain": {"operator": ">", "value": 0.0},
            "final_win_rate": {"operator": ">", "value": 0.5},
            "fold_rate": {"operator": "<", "value": TARGET_FOLD_RATE},
            "line_epe": {"operator": "<", "reference": "lambda_zero_prior"},
            "line_straightness_error": {
                "operator": "<",
                "reference": "lambda_zero_prior",
            },
        },
        "insufficient_reasons": reasons,
        "cells": cell_decisions,
        "training_temperature_qwen_comparison": {
            "temperature": training_temperature,
            "ranking": qwen_rows,
            "best_mode": qwen_rows[0]["qwen_mode"] if qwen_rows else None,
        },
        "global_best": global_best,
        "summary_text": "",
    }
    decision["summary_text"] = _decision_summary_text(decision)
    return decision


def _validate_resume_report(
    report: dict[str, Any], protocol: dict[str, Any]
) -> int:
    if set(report) != {
        "protocol",
        "protocol_sha256",
        "progress",
        "results",
        "decision",
    }:
        raise RuntimeError("resume report has unexpected or missing top-level fields")
    expected_digest = _json_identity(protocol)
    if report["protocol_sha256"] != expected_digest:
        raise RuntimeError("resume report protocol digest disagrees with this run")
    if _canonical_json(report["protocol"]) != _canonical_json(protocol):
        raise RuntimeError(
            "resume report protocol/checkpoint/Qwen identity differs from this run"
        )

    results = report["results"]
    if not isinstance(results, dict):
        raise RuntimeError("resume report results must be a mapping")
    allowed = _cell_coordinates(protocol)
    scale_keys = [_number_key(value) for value in protocol["residual_scales"]]
    completed = 0
    for temperature_key, modes in results.items():
        if not isinstance(modes, dict):
            raise RuntimeError("resume temperature result must be a mapping")
        for mode, cell_result in modes.items():
            if (temperature_key, mode) not in allowed:
                raise RuntimeError(
                    "resume report contains a cell outside the current protocol: "
                    f"temperature={temperature_key}, qwen={mode}"
                )
            cell = next(
                item
                for item in protocol["cells"]
                if item["temperature_key"] == temperature_key
                and item["qwen_mode"] == mode
            )
            _validate_cell_result(
                cell_result,
                temperature=float(cell["temperature"]),
                mode=mode,
                scale_keys=scale_keys,
            )
            _verify_zero_and_gate_invariants(cell_result["metrics"], mode=mode)
            completed += 1
    progress = report["progress"]
    expected_progress = {
        "completed_cells": completed,
        "total_cells": len(protocol["cells"]),
        "complete": completed == len(protocol["cells"]),
    }
    if progress != expected_progress:
        raise RuntimeError(
            f"resume progress is inconsistent: saved={progress}, "
            f"expected={expected_progress}"
        )
    if expected_progress["complete"]:
        expected_decision = _decision_from_validated_cells(report)
        if _canonical_json(report["decision"]) != _canonical_json(expected_decision):
            raise RuntimeError(
                "resume decision is missing or disagrees with validated metrics"
            )
    elif report["decision"] is not None:
        raise RuntimeError("partial ablation report must not contain a decision")
    return completed


def analyze_complete_report(report: dict[str, Any]) -> dict[str, Any]:
    """Strictly validate a complete report and return its bound decision."""

    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("ablation report protocol must be a mapping")
    _validate_resume_report(report, protocol)
    if not bool(report["progress"]["complete"]):
        raise RuntimeError("ablation report is not complete")
    decision = report["decision"]
    if not isinstance(decision, dict):  # pragma: no cover - checked above.
        raise RuntimeError("complete ablation report has no decision")
    return decision


def _load_or_initialize_report(
    path: Path,
    *,
    protocol: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"output report already exists; pass --resume or choose a fresh path: {path}"
            )
        report = _strict_json_load(path)
        completed = _validate_resume_report(report, protocol)
        print(
            f"[resume] validated {completed}/{len(protocol['cells'])} completed cells",
            flush=True,
        )
        return report
    report = {
        "protocol": protocol,
        "protocol_sha256": _json_identity(protocol),
        "progress": {
            "completed_cells": 0,
            "total_cells": len(protocol["cells"]),
            "complete": False,
        },
        "results": {},
        "decision": None,
    }
    _atomic_write_json(path, report)
    return report


def _cell_is_complete(report: dict[str, Any], temperature_key: str, mode: str) -> bool:
    modes = report["results"].get(temperature_key)
    return isinstance(modes, dict) and mode in modes


def _commit_cell(
    report: dict[str, Any],
    *,
    path: Path,
    temperature_key: str,
    mode: str,
    result: dict[str, Any],
) -> None:
    if _cell_is_complete(report, temperature_key, mode):
        raise RuntimeError("refusing to overwrite an already completed ablation cell")
    cell_spec = next(
        (
            cell
            for cell in report["protocol"]["cells"]
            if str(cell["temperature_key"]) == temperature_key
            and str(cell["qwen_mode"]) == mode
        ),
        None,
    )
    if cell_spec is None:
        raise RuntimeError("refusing to commit a cell outside the ablation protocol")
    _validate_cell_result(
        result,
        temperature=float(cell_spec["temperature"]),
        mode=mode,
        scale_keys=[
            _number_key(value) for value in report["protocol"]["residual_scales"]
        ],
    )
    _verify_zero_and_gate_invariants(result["metrics"], mode=mode)
    report["results"].setdefault(temperature_key, {})[mode] = result
    completed = sum(
        len(modes) for modes in report["results"].values()
        if isinstance(modes, dict)
    )
    report["progress"] = {
        "completed_cells": completed,
        "total_cells": len(report["protocol"]["cells"]),
        "complete": completed == len(report["protocol"]["cells"]),
    }
    report["decision"] = (
        _decision_from_validated_cells(report)
        if report["progress"]["complete"]
        else None
    )
    _validate_resume_report(report, report["protocol"])
    _atomic_write_json(path, report)


def _verify_zero_and_gate_invariants(
    metrics: dict[str, dict[str, float]], *, mode: str
) -> None:
    zero = metrics.get(_number_key(0.0))
    if zero is not None:
        checks = {
            "epe-prior_epe": zero["epe"] - zero["prior_epe"],
            "epe_gain": zero["epe_gain"],
            "relative_epe_gain": zero["relative_epe_gain"],
            "final_win_rate": zero["final_win_rate"],
            "applied_residual_p95": zero["applied_residual_p95"],
        }
        failed = {key: value for key, value in checks.items() if abs(value) > 1e-10}
        if failed:
            raise RuntimeError(f"lambda=0 is not the exact prior baseline: {failed}")

    representative = next(iter(metrics.values()))
    if mode in {"none", "context_only"} and abs(
        representative["matching_feature_confidence"]
    ) > 1e-12:
        raise RuntimeError(f"Qwen mode {mode} failed to disable matching")
    if mode in {"none", "matching_only"} and abs(
        representative["context_feature_confidence"]
    ) > 1e-12:
        raise RuntimeError(f"Qwen mode {mode} failed to disable context")


def _run_cell(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: RectificationLoss,
    *,
    device: torch.device,
    temperature: float,
    mode: str,
    residual_scales: list[float],
    max_batches: int | None,
) -> dict[str, Any]:
    raw = _raw_model(model)
    raw.set_correlation_temperature(float(temperature))
    _set_qwen_mode(model, mode)
    scale_keys = [_number_key(scale) for scale in residual_scales]
    totals = {
        scale_key: torch.zeros(
            len(METRIC_KEYS), device=device, dtype=torch.float64
        )
        for scale_key in scale_keys
    }
    batch_count = 0
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _to_device(batch, device)
        current_batch_size = int(batch["warped"].shape[0])
        outputs = model(batch["warped"], None, stage="unified")
        source_size = tuple(int(value) for value in batch["warped"].shape[-2:])
        for scale, scale_key in zip(residual_scales, scale_keys, strict=True):
            scaled_outputs = (
                outputs
                if scale == 1.0
                else outputs_at_residual_scale(
                    outputs, scale, source_size=source_size
                )
            )
            losses = criterion(scaled_outputs, batch)
            metric_tensors: list[Tensor] = []
            for key in METRIC_KEYS:
                value = losses.get(key)
                if not isinstance(value, Tensor) or value.ndim != 0:
                    raise RuntimeError(f"required metric {key!r} is unavailable")
                metric_tensors.append(value.detach().to(dtype=torch.float64))
            # Stay on device for the whole cell; this avoids one CUDA synchronize
            # per scalar metric and transfers only a tiny matrix at cell end.
            totals[scale_key].add_(
                torch.stack(metric_tensors) * current_batch_size
            )
        batch_count += 1
        sample_count += current_batch_size
    if batch_count <= 0 or sample_count <= 0:
        raise RuntimeError("validation sweep evaluated zero samples")

    averaged = torch.stack(
        [totals[scale_key] / sample_count for scale_key in scale_keys]
    ).cpu()
    if not bool(torch.isfinite(averaged).all()):
        raise RuntimeError(
            f"non-finite metric at temperature={temperature}, qwen={mode}"
        )
    metrics = {
        scale_key: {
            metric: float(averaged[scale_index, metric_index])
            for metric_index, metric in enumerate(METRIC_KEYS)
        }
        for scale_index, scale_key in enumerate(scale_keys)
    }
    _verify_zero_and_gate_invariants(metrics, mode=mode)
    return {
        "temperature": float(temperature),
        "qwen_mode": mode,
        "evaluated_batches": batch_count,
        "evaluated_samples": sample_count,
        "metrics": metrics,
    }


def _assert_external_identities(
    protocol: dict[str, Any], config: dict[str, Any]
) -> None:
    if qwen_model_identity(config) != protocol["qwen_model_identity"]:
        raise RuntimeError("external Qwen model identity changed during the sweep")
    manifest_path = Path(config["data"]["val_manifest"])
    if _manifest_identity(manifest_path) != protocol["validation_manifest"]:
        raise RuntimeError("validation manifest identity changed during the sweep")
    if _source_code_identity() != protocol["source_code_identity"]:
        raise RuntimeError("evaluator source code changed during the sweep")


@torch.inference_mode()
def run_sweep(
    checkpoint: str | Path,
    *,
    output_json: str | Path,
    resume: bool,
    device: torch.device,
    temperatures: list[float] | None,
    qwen_modes: list[str],
    training_temperature_qwen_modes: list[str],
    residual_scales: list[float],
    max_batches: int | None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    output_path = Path(output_json).resolve()
    payload, checkpoint_identity = _checkpoint_payload_and_identity(checkpoint_path)
    config = _authoritative_config(payload)
    training_temperature = _checkpoint_correlation_temperature(payload, required=True)
    if training_temperature is None:  # pragma: no cover
        raise RuntimeError("checkpoint training temperature is unavailable")
    if temperatures is None:
        temperatures = [float(training_temperature)]
    cells = build_cell_plan(
        temperatures,
        training_temperature=float(training_temperature),
        qwen_modes=qwen_modes,
        training_temperature_qwen_modes=training_temperature_qwen_modes,
    )
    if not residual_scales:
        raise ValueError("residual scale list must not be empty")
    scale_keys = [_number_key(value) for value in residual_scales]
    if len(scale_keys) != len(set(scale_keys)):
        raise ValueError("residual scales collide after JSON key formatting")

    loader = _build_loader(config)
    feature_backend = str(config["model"].get("feature_backend", "qwen")).lower()
    configured_batch_size = int(config["data"].get("batch_size", 1))
    if feature_backend == "qwen" and configured_batch_size != 1:
        raise RuntimeError("formal Qwen ablation requires data.batch_size=1")
    protocol = {
        "version": REPORT_VERSION,
        "checkpoint": checkpoint_identity,
        "checkpoint_epoch_index": int(payload["epoch"]),
        "checkpoint_display_epoch": int(payload["epoch"]) + 1,
        "training_correlation_temperature": float(training_temperature),
        "cells": [
            {
                "temperature": float(temperature),
                "temperature_key": _number_key(temperature),
                "qwen_mode": mode,
                "is_training_temperature": float(temperature)
                == float(training_temperature),
            }
            for temperature, mode in cells
        ],
        "residual_scales": [float(value) for value in residual_scales],
        "validation_manifest": _manifest_identity(
            Path(config["data"]["val_manifest"])
        ),
        "validation_samples": len(loader.dataset),
        "configured_batch_size": configured_batch_size,
        "max_batches": max_batches,
        "qwen_model_identity": qwen_model_identity(config),
        "source_code_identity": _source_code_identity(),
        "runtime_identity": _runtime_identity(device),
        "composition": "lambda*R(x) + B(x + lambda*R(x))",
        "lambda_zero_note": (
            "EPE equals prior_epe; strict final_error < prior_error gives win_rate=0"
        ),
    }
    report = _load_or_initialize_report(
        output_path, protocol=protocol, resume=resume
    )
    if report["progress"]["complete"]:
        print(f"[resume] report already complete: {output_path}", flush=True)
        return report

    model, loaded_config = _load_model(payload, device=device)
    if loaded_config is not config:
        raise RuntimeError("internal checkpoint config identity changed")
    raw = _raw_model(model)
    raw.set_residual_application_scale(1.0)
    criterion = RectificationLoss(config["loss"]).to(device)

    for temperature, mode in cells:
        temperature_key = _number_key(temperature)
        if _cell_is_complete(report, temperature_key, mode):
            print(
                f"[resume] skip temperature={temperature_key} qwen={mode}",
                flush=True,
            )
            continue
        cell_result = _run_cell(
            model,
            loader,
            criterion,
            device=device,
            temperature=temperature,
            mode=mode,
            residual_scales=residual_scales,
            max_batches=max_batches,
        )
        # Refuse to combine cells if an external model, manifest, or source file
        # changed during a long-running job.
        _assert_external_identities(protocol, config)
        _commit_cell(
            report,
            path=output_path,
            temperature_key=temperature_key,
            mode=mode,
            result=cell_result,
        )
        scale_one = cell_result["metrics"].get(_number_key(1.0))
        metric_note = (
            ""
            if scale_one is None
            else f" epe@1={scale_one['epe']:.6f} gain@1={scale_one['epe_gain']:.6f}"
        )
        print(
            f"[done] temperature={temperature_key} qwen={mode} "
            f"batches={cell_result['evaluated_batches']} "
            f"samples={cell_result['evaluated_samples']} "
            f"report={output_path}{metric_note}",
            flush=True,
        )
    _set_qwen_mode(model, "both")
    raw.set_correlation_temperature(float(training_temperature))
    _validate_resume_report(report, protocol)
    return report


def _finite_unique(values: list[float], *, label: str) -> list[float]:
    normalized: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} values must be finite")
        if number not in normalized:
            normalized.append(number)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=list(DEFAULT_TEMPERATURES),
        help="temperatures scanned for --qwen-modes",
    )
    parser.add_argument(
        "--qwen-modes", nargs="+", choices=QWEN_MODES,
        default=list(DEFAULT_QWEN_MODES),
        help="modes evaluated at every requested temperature",
    )
    parser.add_argument(
        "--training-temperature-qwen-modes",
        nargs="+",
        choices=QWEN_MODES,
        default=list(DEFAULT_TRAINING_TEMPERATURE_QWEN_MODES),
        help="additional modes evaluated only at the checkpoint temperature",
    )
    parser.add_argument(
        "--residual-scales", type=float, nargs="+",
        default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="strictly validate and continue an existing per-cell report",
    )
    args = parser.parse_args()

    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    scales = _finite_unique(args.residual_scales, label="residual scale")
    if any(not 0.0 <= value <= 1.0 for value in scales):
        parser.error("all residual scales must be in [0,1]")

    temperatures = _finite_unique(args.temperatures, label="temperature")
    if any(value <= 0.0 for value in temperatures):
        parser.error("all temperatures must be positive")

    output_path = args.output_json.resolve()
    with _exclusive_report_lock(output_path):
        report = run_sweep(
            args.checkpoint,
            output_json=output_path,
            resume=bool(args.resume),
            device=device,
            temperatures=temperatures,
            qwen_modes=list(dict.fromkeys(args.qwen_modes)),
            training_temperature_qwen_modes=list(
                dict.fromkeys(args.training_temperature_qwen_modes)
            ),
            residual_scales=scales,
            max_batches=args.max_batches,
        )
    print(
        f"report={output_path} "
        f"complete={report['progress']['complete']} "
        f"cells={report['progress']['completed_cells']}/"
        f"{report['progress']['total_cells']}",
        flush=True,
    )
    if report["progress"]["complete"]:
        print(analyze_complete_report(report)["summary_text"], flush=True)


if __name__ == "__main__":
    main()
