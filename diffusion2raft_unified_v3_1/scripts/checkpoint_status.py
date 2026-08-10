#!/usr/bin/env python3
"""Validate and summarize a trusted local training checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from diffusion2raft.external_file import (  # noqa: E402
    ExternalFileIdentityError,
    validate_external_file_identity,
)
from diffusion2raft.teacher_capacity_receipt import (  # noqa: E402
    strict_validate_teacher_capacity_receipt,
)


_CAPACITY_EVIDENCE_RECEIPT_KEY = "capacity_evidence_receipt"
_TEACHER_RECEIPT_BINDING_KEYS = (
    "sha256",
    "file_size",
    "input_size",
    "flow_size",
    "blur_kernel",
    "autocast_dtype",
    "requires_logical_cuda0",
)


def _load(source: Any) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        seek = getattr(source, "seek", None)
        if callable(seek):
            seek(0)
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a mapping, got {type(payload)!r}")
    return payload


def _field(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _validate_external_stat(
    identity: Mapping[str, Any],
    *,
    path_key: str,
    size_key: str,
    label: str,
) -> None:
    try:
        validate_external_file_identity(
            identity,
            path_key=path_key,
            size_key=size_key,
            label=label,
        )
    except ExternalFileIdentityError as exc:
        raise ValueError(str(exc)) from exc


def _validate_teacher_checkpoint(
    payload: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    epoch: int,
) -> None:
    declared = payload.get("prior_backend")
    if declared is None:
        saved_config = payload.get("config")
        if isinstance(saved_config, Mapping):
            saved_model = saved_config.get("model")
            if isinstance(saved_model, Mapping):
                declared = saved_model.get("prior_backend")
    marker_present = "prior._teacher_backend_marker" in model
    backend = (
        str(declared).lower()
        if declared is not None
        else ("torchscript" if marker_present else "learned")
    )
    if backend not in {"learned", "torchscript"}:
        raise ValueError(f"checkpoint has unknown prior_backend={declared!r}")
    if backend == "learned":
        if marker_present:
            raise ValueError("learned checkpoint contains a teacher marker")
        if _CAPACITY_EVIDENCE_RECEIPT_KEY in payload:
            raise ValueError("learned checkpoint contains capacity_evidence_receipt")
        return
    if not marker_present:
        raise ValueError("torchscript checkpoint has no teacher marker")

    if _CAPACITY_EVIDENCE_RECEIPT_KEY not in payload:
        raise ValueError("teacher checkpoint has no capacity_evidence_receipt")
    try:
        capacity_receipt = strict_validate_teacher_capacity_receipt(
            payload[_CAPACITY_EVIDENCE_RECEIPT_KEY]
        )
    except ValueError as exc:
        raise ValueError(
            "teacher checkpoint has invalid capacity_evidence_receipt: "
            f"{exc}"
        ) from exc

    identity = payload.get("teacher_prior_identity")
    required_identity_keys = {
        "version",
        "resolved_path",
        "file_size",
        "mtime_ns",
        "sha256",
        "input_size",
        "flow_size",
        "blur_kernel",
        "autocast_dtype",
    }
    optional_identity_keys = {"requires_logical_cuda0"}
    if not isinstance(identity, Mapping) or not (
        required_identity_keys <= set(identity)
        and set(identity) <= required_identity_keys | optional_identity_keys
    ):
        raise ValueError("teacher checkpoint has invalid teacher_prior_identity schema")
    for key in ("version", "file_size", "mtime_ns", "input_size", "flow_size", "blur_kernel"):
        value = identity[key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"teacher_prior_identity.{key} must be an integer")
    if identity["version"] != 2:
        raise ValueError("teacher_prior_identity is not strict version 2")
    if "requires_logical_cuda0" in identity and not isinstance(
        identity["requires_logical_cuda0"], bool
    ):
        raise ValueError(
            "teacher_prior_identity.requires_logical_cuda0 must be a boolean"
        )
    receipt_teacher = capacity_receipt["teacher"]
    differing_receipt_fields = [
        key
        for key in _TEACHER_RECEIPT_BINDING_KEYS
        if key not in identity
        or not _typed_equal(receipt_teacher[key], identity[key])
    ]
    if differing_receipt_fields:
        raise ValueError(
            "capacity_evidence_receipt.teacher differs from "
            "teacher_prior_identity; "
            f"differing_fields={differing_receipt_fields}"
        )
    _validate_external_stat(
        identity,
        path_key="resolved_path",
        size_key="file_size",
        label="teacher",
    )
    saved_config = payload.get("config")
    saved_model = saved_config.get("model") if isinstance(saved_config, Mapping) else None
    if not isinstance(saved_model, Mapping):
        raise ValueError("strict teacher checkpoint has no config.model")
    if saved_model.get("prior_torchscript_sha256") != identity["sha256"]:
        raise ValueError(
            "config.model.prior_torchscript_sha256 differs from teacher identity"
        )

    residual = payload.get("residual_application")
    residual_keys = {
        "version",
        "origin_epoch",
        "warmup_epochs",
        "ramp_epochs",
        "max_scale",
        "scale",
    }
    if not isinstance(residual, Mapping) or set(residual) != residual_keys:
        raise ValueError("teacher checkpoint has invalid residual_application schema")
    for key in ("version", "origin_epoch", "warmup_epochs", "ramp_epochs"):
        value = residual[key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"residual_application.{key} must be an integer")
    for key in ("max_scale", "scale"):
        value = residual[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"residual_application.{key} must be real")
    origin = int(residual["origin_epoch"])
    warmup = int(residual["warmup_epochs"])
    ramp = int(residual["ramp_epochs"])
    maximum = float(residual["max_scale"])
    if residual["version"] != 1 or origin < 0 or warmup < 1 or ramp < 0:
        raise ValueError("teacher checkpoint has invalid residual schedule")
    if not math.isfinite(maximum) or not 0.0 <= maximum <= 1.0:
        raise ValueError("teacher checkpoint has invalid residual max_scale")
    if epoch < origin:
        raise ValueError("teacher checkpoint epoch predates residual origin")
    age = epoch - origin
    if age < warmup:
        expected_scale = 0.0
    elif ramp == 0:
        expected_scale = maximum
    else:
        progress = min(max((age - warmup + 1) / ramp, 0.0), 1.0)
        expected_scale = maximum * progress
    scale = float(residual["scale"])
    if not math.isfinite(scale) or not math.isclose(
        scale, expected_scale, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("teacher residual scale disagrees with epoch/schedule")

    contract = payload.get("deployment_contract")
    if not isinstance(contract, Mapping) or contract.get("version") != 2:
        raise ValueError("teacher checkpoint has no valid deployment_contract")
    if not _typed_equal(contract.get("teacher_prior_identity"), identity):
        raise ValueError("deployment contract teacher identity differs")
    inpaint = contract.get("inpaint")
    if not isinstance(inpaint, Mapping) or not isinstance(
        inpaint.get("enabled"), bool
    ):
        raise ValueError("deployment contract has invalid inpaint identity")
    if inpaint["enabled"]:
        _validate_external_stat(
            inpaint,
            path_key="path",
            size_key="size_bytes",
            label="LAMA",
        )
    saved_inference = (
        saved_config.get("inference") if isinstance(saved_config, Mapping) else None
    )
    if saved_inference is None:
        saved_inference = {}
    if not isinstance(saved_inference, Mapping):
        raise ValueError("strict teacher checkpoint has invalid config.inference")
    configured_inpaint = saved_inference.get("inpaint", {})
    if configured_inpaint is None:
        configured_inpaint = {}
    if not isinstance(configured_inpaint, Mapping):
        raise ValueError("strict teacher checkpoint has invalid config inference.inpaint")
    if bool(configured_inpaint.get("enabled", False)) != inpaint["enabled"]:
        raise ValueError("config inference.inpaint.enabled differs from deployment contract")
    if inpaint["enabled"] and configured_inpaint.get("sha256") != inpaint.get(
        "sha256"
    ):
        raise ValueError(
            "config inference.inpaint.sha256 differs from LAMA identity"
        )

    best = payload.get("best_metric")
    if not isinstance(best, Mapping) or set(best) != {"name", "mode", "value"}:
        raise ValueError("teacher checkpoint has invalid best_metric schema")
    if not isinstance(best["name"], str) or best["mode"] not in {"min", "max"}:
        raise ValueError("teacher checkpoint has invalid best_metric contract")
    value = best["value"]
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(float(value)):
        raise ValueError("teacher checkpoint has invalid best_metric value")


def inspect_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expect_stage: str | None,
    require_optimizer: bool,
) -> tuple[str, int, int, str, str]:
    """Validate an already-loaded checkpoint using the canonical strict rules."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a mapping, got {type(payload)!r}")
    model = payload.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("checkpoint has no non-empty model state")
    if require_optimizer:
        optimizer = payload.get("optimizer")
        if not isinstance(optimizer, Mapping) or not optimizer:
            raise ValueError("checkpoint has no optimizer state; exact continuation is unsafe")

    stage = str(payload.get("stage", ""))
    if not stage:
        raise ValueError("checkpoint has no stage metadata")
    if expect_stage is not None and stage != expect_stage:
        raise ValueError(f"checkpoint stage is {stage!r}, expected {expect_stage!r}")

    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < -1:
        raise ValueError(f"checkpoint epoch must be an integer >= -1, got {epoch!r}")
    completed_epochs = epoch + 1
    _validate_teacher_checkpoint(payload, model, epoch=epoch)

    best = payload.get("best_metric")
    if isinstance(best, Mapping) and best.get("name") is not None:
        best_name = _field(best["name"])
        raw_value = best.get("value")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            best_value = "nan"
        else:
            best_value = format(value, ".12g") if math.isfinite(value) else str(value)
    else:
        best_name = "-"
        best_value = "nan"
    return _field(stage), epoch, completed_epochs, best_name, best_value


def inspect_checkpoint(
    path: Path, *, expect_stage: str | None, require_optimizer: bool
) -> tuple[str, int, int, str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"checkpoint does not exist or is empty: {path}")
    payload = _load(path)
    return inspect_checkpoint_payload(
        payload,
        expect_stage=expect_stage,
        require_optimizer=require_optimizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expect-stage")
    parser.add_argument("--require-optimizer", action="store_true")
    parser.add_argument("--format", choices=("tsv", "human"), default="human")
    args = parser.parse_args()
    try:
        stage, epoch, completed, best_name, best_value = inspect_checkpoint(
            args.checkpoint,
            expect_stage=args.expect_stage,
            require_optimizer=args.require_optimizer,
        )
    except Exception as error:
        print(f"checkpoint validation error: {error}", file=sys.stderr)
        raise SystemExit(65) from error

    if args.format == "tsv":
        print("\t".join((stage, str(epoch), str(completed), best_name, best_value)))
    else:
        print(
            f"checkpoint={args.checkpoint} stage={stage} epoch_index={epoch} "
            f"completed_epochs={completed} best={best_name}:{best_value}"
        )


if __name__ == "__main__":
    main()
