"""Strict, serializable deployment contracts for teacher-backed inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .external_file import (
    canonical_sha256,
    stable_external_file_identity,
    validate_external_file_identity,
)


TEACHER_DEPLOYMENT_CONTRACT_VERSION = 2


def _external_file_identity(
    path_value: str | Path, *, expected_sha256: str, label: str
) -> dict[str, Any]:
    return stable_external_file_identity(
        path_value,
        expected_sha256=expected_sha256,
        label=label,
    )


def _validate_contract_external_files(
    contract: Mapping[str, Any], *, label: str, verify_files: bool
) -> None:
    if contract.get("version") != TEACHER_DEPLOYMENT_CONTRACT_VERSION:
        raise RuntimeError(
            f"{label} deployment contract is not strict version "
            f"{TEACHER_DEPLOYMENT_CONTRACT_VERSION}"
        )
    teacher = contract.get("teacher_prior_identity")
    if not isinstance(teacher, Mapping):
        raise RuntimeError(f"{label} deployment contract has no teacher identity")
    if teacher.get("version") != 2:
        raise RuntimeError(f"{label} teacher identity is not strict version 2")
    _validate_external_identity_shape(
        teacher,
        path_key="resolved_path",
        size_key="file_size",
        label=f"{label} teacher",
    )
    if verify_files:
        validate_external_file_identity(
            teacher,
            path_key="resolved_path",
            size_key="file_size",
            label=f"{label} teacher",
        )

    inpaint = contract.get("inpaint")
    if not isinstance(inpaint, Mapping) or not isinstance(
        inpaint.get("enabled"), bool
    ):
        raise RuntimeError(f"{label} deployment contract has invalid inpaint identity")
    if inpaint["enabled"]:
        _validate_external_identity_shape(
            inpaint,
            path_key="path",
            size_key="size_bytes",
            label=f"{label} LAMA",
        )
        if verify_files:
            validate_external_file_identity(
                inpaint,
                path_key="path",
                size_key="size_bytes",
                label=f"{label} LAMA",
            )


def _validate_external_identity_shape(
    identity: Mapping[str, Any],
    *,
    path_key: str,
    size_key: str,
    label: str,
) -> None:
    path = identity.get(path_key)
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise RuntimeError(f"{label} identity has no absolute path")
    size = identity.get(size_key)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RuntimeError(f"{label} identity has invalid size")
    mtime = identity.get("mtime_ns")
    if isinstance(mtime, bool) or not isinstance(mtime, int) or mtime < 0:
        raise RuntimeError(f"{label} identity has invalid mtime_ns")
    canonical_sha256(identity.get("sha256"), label=f"{label} identity sha256")


def build_teacher_deployment_contract(
    config: dict[str, Any],
    *,
    teacher_identity: dict[str, Any],
    inpaint_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze every preprocessing/sampling choice that defines corrected-512.

    The external teacher identity is supplied by the already-loaded model so a
    later filesystem replacement cannot make a checkpoint claim it trained
    against different bytes.  LAMA identity may likewise be supplied by the
    loaded inference wrapper; training otherwise computes the same stable
    SHA-256 identity directly from the configured artifact.
    """

    data = config.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("teacher deployment contract requires config.data")
    raw_work_size = data.get("work_size")
    if not isinstance(raw_work_size, (list, tuple)) or len(raw_work_size) != 2:
        raise RuntimeError(
            "teacher deployment contract requires data.work_size=[height,width]"
        )
    work_size = [int(raw_work_size[0]), int(raw_work_size[1])]
    if min(work_size) <= 1:
        raise RuntimeError(f"invalid teacher deployment work_size: {work_size}")

    model = config.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("teacher deployment contract requires config.model")
    expected_teacher_sha256 = canonical_sha256(
        model.get("prior_torchscript_sha256"),
        label="model.prior_torchscript_sha256",
    )
    if teacher_identity.get("version") != 2:
        raise RuntimeError("teacher deployment requires teacher identity version 2")
    if teacher_identity.get("sha256") != expected_teacher_sha256:
        raise RuntimeError(
            "teacher identity sha256 differs from model.prior_torchscript_sha256"
        )

    inference = config.get("inference", {})
    if inference is None:
        inference = {}
    if not isinstance(inference, dict):
        raise RuntimeError("teacher deployment contract requires mapping inference")
    decoder = str(inference.get("image_decoder", "pil")).lower()
    resize_policy = str(inference.get("resize_policy", "stretch")).lower()
    interpolation = str(
        inference.get("resize_interpolation", "bilinear")
    ).lower()
    padding_mode = str(inference.get("padding_mode", "replicate")).lower()

    raw_inpaint = inference.get("inpaint", {})
    if raw_inpaint is None:
        raw_inpaint = {}
    if not isinstance(raw_inpaint, dict):
        raise RuntimeError("teacher deployment inference.inpaint must be a mapping")
    inpaint_enabled = bool(raw_inpaint.get("enabled", False))
    if inpaint_enabled:
        configured_path = raw_inpaint.get("path")
        if not configured_path:
            raise RuntimeError(
                "teacher deployment requires inference.inpaint.path when enabled"
            )
        expected_inpaint_sha256 = canonical_sha256(
            raw_inpaint.get("sha256"), label="inference.inpaint.sha256"
        )
        if inpaint_identity is None:
            file_identity = _external_file_identity(
                str(configured_path),
                expected_sha256=expected_inpaint_sha256,
                label="TorchScript LAMA",
            )
            normalized_inpaint_identity = {
                "enabled": True,
                "backend": "torchscript_lama",
                "path": file_identity["resolved_path"],
                "size_bytes": file_identity["file_size"],
                "mtime_ns": file_identity["mtime_ns"],
                "sha256": file_identity["sha256"],
                "input_size": int(raw_inpaint.get("size", 512)),
                "dilation_kernel": int(raw_inpaint.get("dilation", 11)),
            }
        else:
            normalized_inpaint_identity = dict(inpaint_identity)
            if normalized_inpaint_identity.get("sha256") != expected_inpaint_sha256:
                raise RuntimeError(
                    "LAMA identity sha256 differs from inference.inpaint.sha256"
                )
    else:
        normalized_inpaint_identity = {"enabled": False}

    contract = {
        "version": TEACHER_DEPLOYMENT_CONTRACT_VERSION,
        "work_size_hw": work_size,
        "source_preprocess": {
            "image_decoder": decoder,
            "public_tensor_channel_order": "RGB",
            "teacher_channel_order": "BGR",
            "value_range": "0_to_1",
            "normalization": "none",
            "resize_policy": resize_policy,
            "resize_interpolation": interpolation,
            "padding_mode": padding_mode,
            "opencv_recipe": (
                "short_gt_2048_area_to_short_1024_then_area_square;"
                "else_short_gt_target_area;else_linear"
                if interpolation == "opencv_baseline"
                else None
            ),
        },
        "teacher_prior_identity": dict(teacher_identity),
        "flow_restore": {
            "convention": "backward_displacement_xy",
            "method": "absolute_source_coordinate_map",
            "flow_resize_mode": "bilinear",
            "flow_resize_align_corners": True,
        },
        "final_warp": {
            "mode": "bilinear",
            "align_corners": True,
            "padding_mode": "zeros" if inpaint_enabled else "border",
        },
        "inpaint": {
            **normalized_inpaint_identity,
            "preprocess": (
                {
                    "version": "legacy_uint8_opencv_v1",
                    "warp_quantization": "uint8_truncate",
                    "image_resize": "opencv_inter_linear",
                    "mask_resize": "opencv_inter_linear_then_gt_100",
                    "composite_mask": "opencv_dilate_binary_then_gt_0_4",
                    "channel_order": "BGR",
                    "final_output_quantization": "uint8_truncate",
                }
                if inpaint_enabled
                else None
            ),
        },
    }
    # The identities above were produced either by the authenticated loaded
    # components or by ``stable_external_file_identity`` in this function.
    # Re-hashing them here would duplicate multi-GB I/O without strengthening
    # the load binding; publication boundaries perform a fresh byte check.
    _validate_contract_external_files(
        contract, label="current", verify_files=False
    )
    return contract


def deployment_contract_differences(
    saved: Any,
    current: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return paths whose values or exact Python scalar types differ."""

    if type(saved) is not type(current):
        return [prefix or "<root>"]
    if isinstance(current, dict):
        differences: list[str] = []
        for key in sorted(set(saved) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved or key not in current:
                differences.append(path)
            else:
                differences.extend(
                    deployment_contract_differences(
                        saved[key], current[key], prefix=path
                    )
                )
        return differences
    if isinstance(current, list):
        if len(saved) != len(current):
            return [prefix or "<root>"]
        differences = []
        for index, (saved_value, current_value) in enumerate(zip(saved, current)):
            differences.extend(
                deployment_contract_differences(
                    saved_value,
                    current_value,
                    prefix=f"{prefix}[{index}]",
                )
            )
        return differences
    return [] if saved == current else [prefix or "<root>"]


def validate_teacher_deployment_contract(
    saved: Any,
    current: dict[str, Any],
    *,
    external_files_authenticated: bool = False,
) -> dict[str, Any]:
    if not isinstance(saved, dict):
        raise RuntimeError("teacher checkpoint is missing deployment_contract")
    # Production callers set ``external_files_authenticated`` only after their
    # current contract was built from the same-fd teacher/LAMA loaders.  The
    # public default remains self-contained and re-hashes both contracts.
    _validate_contract_external_files(
        saved,
        label="saved",
        verify_files=not external_files_authenticated,
    )
    _validate_contract_external_files(
        current,
        label="current",
        verify_files=not external_files_authenticated,
    )
    differences = deployment_contract_differences(saved, current)
    if differences:
        raise RuntimeError(
            "teacher deployment contract mismatch; "
            f"differing_fields={differences[:24]}"
        )
    return dict(current)


__all__ = [
    "TEACHER_DEPLOYMENT_CONTRACT_VERSION",
    "build_teacher_deployment_contract",
    "deployment_contract_differences",
    "validate_teacher_deployment_contract",
]
