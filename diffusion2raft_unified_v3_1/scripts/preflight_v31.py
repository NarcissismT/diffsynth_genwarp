#!/usr/bin/env python3
"""Fail-fast checks before launching the expensive unified v3.1 job."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from diffusion2raft.external_file import stable_external_file_identity  # noqa: E402


REQUIRED_RECORD_KEYS = ("warped", "target", "flow")


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def _record_size(record: dict[str, Any], key: str) -> tuple[int, int] | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be [height,width], got {value!r}")
    size = (int(value[0]), int(value[1]))
    if min(size) <= 1:
        raise ValueError(f"invalid {key}: {size}")
    return size


def _flow_grid_size(path: Path) -> tuple[int, int]:
    payload = np.load(path, mmap_mode="r")
    try:
        if isinstance(payload, np.lib.npyio.NpzFile):
            array = payload["flow"] if "flow" in payload else payload[payload.files[0]]
        else:
            array = payload
        shape = tuple(int(value) for value in array.shape)
    finally:
        if isinstance(payload, np.lib.npyio.NpzFile):
            payload.close()
    if len(shape) != 3:
        raise ValueError(f"flow must be rank 3, got {shape}")
    if shape[0] == 2:
        return shape[1], shape[2]
    if shape[-1] == 2:
        return shape[0], shape[1]
    raise ValueError(f"flow must be [2,H,W] or [H,W,2], got {shape}")


def _sample_indices(length: int, count: int) -> list[int]:
    count = max(1, min(int(count), length))
    if count == 1:
        return [0]
    return sorted(
        {round(index * (length - 1) / (count - 1)) for index in range(count)}
    )


def _validate_sample(
    manifest: Path,
    record: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    paths = {key: _resolve(manifest, str(record[key])) for key in REQUIRED_RECORD_KEYS}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"record {index} {key} is missing: {path}")
    with Image.open(paths["warped"]) as image:
        source_image_size = (image.height, image.width)
    with Image.open(paths["target"]) as image:
        target_image_size = (image.height, image.width)
    flow_grid_size = _flow_grid_size(paths["flow"])
    flow_source_size = _record_size(record, "flow_source_size")
    flow_target_size = _record_size(record, "flow_target_size")
    if flow_target_size is not None and flow_target_size != flow_grid_size:
        raise ValueError(
            f"record {index}: flow_target_size={flow_target_size} != grid={flow_grid_size}"
        )
    if flow_source_size is None and flow_grid_size != target_image_size:
        raise ValueError(
            f"record {index}: flow grid {flow_grid_size} differs from target image "
            f"{target_image_size}; flow_source_size is required"
        )
    return {
        "index": index,
        "id": str(record.get("id", index)),
        "source_image_size": source_image_size,
        "target_image_size": target_image_size,
        "flow_grid_size": flow_grid_size,
        "flow_source_size": flow_source_size or source_image_size,
        "flow_format": str(record.get("flow_format", "displacement")),
    }


def _checkpoint_summary(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        import torch
    except ImportError:
        return {}, "PyTorch is unavailable; checkpoint internals were not inspected"
    # PyTorch 2.6 changed the default of weights_only.  This is a trusted local
    # training checkpoint and also contains optimizer/config metadata, so make
    # the intended full checkpoint load explicit while remaining compatible
    # with older PyTorch releases.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    keys = tuple(state)
    required_prefixes = ("prior.", "fusion.", "refiner.")
    missing_prefixes = [prefix for prefix in required_prefixes if not any(k.startswith(prefix) for k in keys)]
    if missing_prefixes:
        raise ValueError(f"checkpoint is missing model branches: {missing_prefixes}")
    if str(payload.get("stage", "")) != "unified":
        raise ValueError(f"expected unified checkpoint, got stage={payload.get('stage')!r}")
    return {
        "stage": payload.get("stage"),
        "completed_epoch": int(payload.get("epoch", -1)) + 1,
        "training_revision": payload.get("training_revision"),
        "model_tensor_count": len(keys),
        "has_optimizer": "optimizer" in payload,
    }, None


def _inference_dependency_summary(
    config: dict[str, Any],
    root: Path,
    work_size: tuple[int, int],
) -> dict[str, Any]:
    """Validate deployment-only decoder/resize/inpainting dependencies."""

    inference = config.get("inference", {})
    if inference is None:
        inference = {}
    if not isinstance(inference, dict):
        raise TypeError("inference config must be a mapping")
    resize_policy = str(inference.get("resize_policy", "stretch")).lower()
    image_decoder = str(inference.get("image_decoder", "pil")).lower()
    interpolation = str(inference.get("resize_interpolation", "bilinear")).lower()
    if image_decoder not in {"pil", "opencv"}:
        raise ValueError(f"unknown inference.image_decoder={image_decoder!r}")
    if interpolation not in {"bilinear", "opencv_baseline"}:
        raise ValueError(
            f"unknown inference.resize_interpolation={interpolation!r}"
        )
    if interpolation == "opencv_baseline":
        if image_decoder != "opencv":
            raise ValueError(
                "opencv_baseline interpolation requires image_decoder=opencv"
            )
        if resize_policy != "stretch" or work_size[0] != work_size[1]:
            raise ValueError(
                "opencv_baseline interpolation requires stretch and square work_size"
            )
    if image_decoder == "opencv" or interpolation == "opencv_baseline":
        if importlib.util.find_spec("cv2") is None:
            raise RuntimeError("OpenCV is required by the inference decoder/resize config")

    raw_inpaint = inference.get("inpaint", {})
    if raw_inpaint is None:
        raw_inpaint = {}
    if not isinstance(raw_inpaint, dict):
        raise TypeError("inference.inpaint must be a mapping")
    enabled = bool(raw_inpaint.get("enabled", False))
    inpaint_summary: dict[str, Any] = {"enabled": enabled}
    if enabled:
        configured_path = raw_inpaint.get("path")
        if not configured_path:
            raise ValueError("inference.inpaint.path is required when enabled")
        inpaint_path = Path(str(configured_path))
        if not inpaint_path.is_absolute():
            inpaint_path = root / inpaint_path
        if not inpaint_path.is_file():
            raise FileNotFoundError(f"TorchScript LAMA model does not exist: {inpaint_path}")
        size = int(raw_inpaint.get("size", 512))
        dilation = int(raw_inpaint.get("dilation", 11))
        if size <= 1:
            raise ValueError("inference.inpaint.size must be greater than one")
        if dilation < 1 or dilation % 2 != 1:
            raise ValueError(
                "inference.inpaint.dilation must be a positive odd integer"
            )
        inpaint_file_identity = stable_external_file_identity(
            inpaint_path,
            expected_sha256=raw_inpaint.get("sha256"),
            label="TorchScript LAMA",
        )
        inpaint_summary.update(
            backend="torchscript_lama",
            path=inpaint_file_identity["resolved_path"],
            size_bytes=inpaint_file_identity["file_size"],
            mtime_ns=inpaint_file_identity["mtime_ns"],
            sha256=inpaint_file_identity["sha256"],
            input_size=size,
            dilation_kernel=dilation,
        )
    return {
        "resize_policy": resize_policy,
        "image_decoder": image_decoder,
        "resize_interpolation": interpolation,
        "inpaint_identity": inpaint_summary,
    }


def _torchscript_prior_dependency_summary(
    config: dict[str, Any],
    root: Path,
    inpaint_identity: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from diffusion2raft.deployment import build_teacher_deployment_contract

    model_config = dict(config.get("model", {}))
    configured_teacher = model_config.get("prior_torchscript_path")
    if not configured_teacher:
        raise ValueError(
            "model.prior_torchscript_path is required for the TorchScript prior"
        )
    teacher_path = Path(str(configured_teacher))
    if not teacher_path.is_absolute():
        teacher_path = root / teacher_path
    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"TorchScript geometry teacher does not exist: {teacher_path}"
        )
    file_identity = stable_external_file_identity(
        teacher_path,
        expected_sha256=model_config.get("prior_torchscript_sha256"),
        label="TorchScript geometry teacher",
    )
    summary = {
        "path": file_identity["resolved_path"],
        "size_bytes": file_identity["file_size"],
        "mtime_ns": file_identity["mtime_ns"],
        "sha256": file_identity["sha256"],
        "input_size": int(model_config.get("prior_torchscript_size", 512)),
        "flow_size": int(
            model_config.get(
                "prior_torchscript_flow_size",
                model_config.get("prior_torchscript_size", 512),
            )
        ),
        "blur_kernel": int(model_config.get("prior_torchscript_blur_kernel", 39)),
        "autocast_dtype": str(
            model_config.get("prior_torchscript_autocast_dtype", "float16")
        ),
        "requires_logical_cuda0": bool(
            model_config.get("prior_torchscript_requires_logical_cuda0", False)
        ),
    }
    if min(summary["input_size"], summary["flow_size"]) <= 1:
        raise ValueError("TorchScript prior input/flow sizes must be greater than one")
    if summary["blur_kernel"] < 1 or summary["blur_kernel"] % 2 != 1:
        raise ValueError(
            "model.prior_torchscript_blur_kernel must be a positive odd integer"
        )
    teacher_identity = {
        "version": 2,
        "resolved_path": summary["path"],
        "file_size": summary["size_bytes"],
        "mtime_ns": summary["mtime_ns"],
        "sha256": summary["sha256"],
        "input_size": summary["input_size"],
        "flow_size": summary["flow_size"],
        "blur_kernel": summary["blur_kernel"],
        "autocast_dtype": summary["autocast_dtype"].removeprefix("torch."),
    }
    if summary["requires_logical_cuda0"]:
        teacher_identity["requires_logical_cuda0"] = True
        if config.get("qwen", {}).get("device_map") is not None:
            raise ValueError(
                "a logical-cuda:0 TorchScript teacher requires "
                "qwen.device_map: null for per-rank GPU isolation"
            )
    contract = build_teacher_deployment_contract(
        config,
        teacher_identity=teacher_identity,
        inpaint_identity=inpaint_identity,
    )
    return summary, contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--max-mae", type=float, default=0.08)
    parser.add_argument("--output-dir", default="runs/preflight_v31")
    parser.add_argument("--skip-reconstruction", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    work_size = tuple(int(value) for value in config["data"]["work_size"])
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"config": str(config_path), "work_size": work_size}
    if any(value % 8 for value in work_size):
        errors.append(f"work_size must be divisible by 8, got {work_size}")
    try:
        report["inference"] = _inference_dependency_summary(
            config, root, work_size
        )
    except Exception as error:
        errors.append(f"inference dependency validation failed: {error}")

    feature_backend = str(config.get("model", {}).get("feature_backend", "qwen"))
    model_id = str(config.get("qwen", {}).get("model_id", ""))
    if feature_backend == "qwen" and model_id.startswith("/") and not Path(model_id).exists():
        errors.append(f"local Qwen model path does not exist: {model_id}")

    model_config = dict(config.get("model", {}))
    prior_backend = str(model_config.get("prior_backend", "learned")).lower()
    if prior_backend not in {"learned", "torchscript"}:
        errors.append(f"unknown model.prior_backend: {prior_backend!r}")
    elif prior_backend == "torchscript":
        try:
            prior_summary, deployment_contract = (
                _torchscript_prior_dependency_summary(
                    config,
                    root,
                    report.get("inference", {}).get("inpaint_identity"),
                )
            )
            report["torchscript_prior"] = prior_summary
            report["deployment_contract"] = deployment_contract
        except Exception as error:
            errors.append(f"teacher dependency validation failed: {error}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_file():
        errors.append(f"checkpoint does not exist: {checkpoint}")
    else:
        try:
            summary, warning = _checkpoint_summary(checkpoint)
            report["checkpoint"] = summary
            if warning:
                warnings.append(warning)
        except Exception as error:  # fail with a consolidated report
            errors.append(f"checkpoint inspection failed: {error}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_reports: dict[str, Any] = {}
    inspect_script = Path(__file__).with_name("inspect_flow_sample.py")

    for split_key in ("train_manifest", "val_manifest"):
        configured = config["data"].get(split_key)
        if not configured:
            continue
        manifest = Path(str(configured))
        if not manifest.is_absolute():
            manifest = root / manifest
        if not manifest.is_file():
            errors.append(f"{split_key} does not exist: {manifest}")
            continue
        records: list[dict[str, Any]] = []
        try:
            for line_number, line in enumerate(
                manifest.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                record = json.loads(line)
                missing = [key for key in REQUIRED_RECORD_KEYS if not record.get(key)]
                if missing:
                    raise ValueError(f"line {line_number} misses {missing}")
                records.append(record)
        except Exception as error:
            errors.append(f"failed to parse {manifest}: {error}")
            continue
        if not records:
            errors.append(f"manifest is empty: {manifest}")
            continue

        samples: list[dict[str, Any]] = []
        for index in _sample_indices(len(records), args.sample_count):
            try:
                sample_report = _validate_sample(manifest, records[index], index)
                samples.append(sample_report)
                if not args.skip_reconstruction:
                    destination = output_dir / split_key.replace("_manifest", "") / f"{index:06d}"
                    command = [
                        sys.executable,
                        str(inspect_script),
                        "--manifest",
                        str(manifest),
                        "--index",
                        str(index),
                        "--output-dir",
                        str(destination),
                        "--max-mae",
                        str(args.max_mae),
                    ]
                    completed = subprocess.run(command, capture_output=True, text=True)
                    sample_report["reconstruction_log"] = completed.stdout.strip()
                    if completed.returncode:
                        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            except Exception as error:
                errors.append(f"{split_key} sample {index} failed: {error}")
        manifest_reports[split_key] = {
            "path": str(manifest),
            "record_count": len(records),
            "samples": samples,
        }

    report["manifests"] = manifest_reports
    report["warnings"] = warnings
    report["errors"] = errors
    report_path = output_dir / "preflight_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for warning in warnings:
        print(f"[warn] {warning}")
    if errors:
        for error in errors:
            print(f"[error] {error}", file=sys.stderr)
        print(f"preflight failed; report: {report_path}", file=sys.stderr)
        raise SystemExit(1)
    print(f"preflight passed; report: {report_path}")
    for key, item in manifest_reports.items():
        print(f"  {key}: {item['record_count']} records, {len(item['samples'])} checked")


if __name__ == "__main__":
    main()
