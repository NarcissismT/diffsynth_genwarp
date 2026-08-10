"""Validate the real frozen Qwen feature hook before Stage-4 training."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .checkpoint import file_sha256
from .models.qwen_feature_probe import FrozenQwenImageEditFeatureSource

_SCHEMA = "docgrid_flow.qwen_runtime_validation.v2"


def _load_image(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (size[1], size[0]), Image.Resampling.LANCZOS
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()[None]


def _component_parameter_count(pipeline: Any) -> int:
    seen: set[int] = set()
    total = 0
    for name in ("transformer", "vae", "text_encoder"):
        component = getattr(pipeline, name, None)
        if component is None:
            continue
        for parameter in component.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
    return total


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Qwen validation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_qwen_runtime(
    image_path: str | Path,
    model_id: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "cuda",
    input_size: tuple[int, int] = (512, 512),
    feature_type: str = "hidden",
    feature_layers: tuple[int, ...] = (-24, -12, -1),
    feature_dtype: str = "bfloat16",
    cpu_offload: bool = True,
) -> dict[str, Any]:
    image_file = Path(image_path).resolve()
    model_root = Path(model_id).resolve()
    destination = Path(output_path).resolve()
    if not image_file.is_file():
        raise FileNotFoundError(f"Qwen probe image does not exist: {image_file}")
    model_index = model_root / "model_index.json"
    transformer_config = model_root / "transformer" / "config.json"
    for path, role in (
        (model_index, "Qwen model index"),
        (transformer_config, "Qwen transformer config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {role}: {path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Qwen validation requested but CUDA is unavailable")
    source = FrozenQwenImageEditFeatureSource(
        {
            "model_id": str(model_root),
            "feature_type": feature_type,
            "feature_layers": list(feature_layers),
            "feature_num_inference_steps": 1,
            "guidance_scale": 1.0,
            "feature_seed": 0,
            "hidden_channels": 3072,
            "feature_dtype": feature_dtype,
            "feature_quantization": "none",
            "local_files_only": True,
            "cpu_offload": cpu_offload,
            "device_map": None,
        }
    )
    image = _load_image(image_file, input_size).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pipeline = source._ensure_pipeline(device)
    decoder = pipeline.vae.decode

    def forbidden_decoder(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Qwen VAE decoder was called during feature validation")

    pipeline.vae.decode = forbidden_decoder
    try:
        with torch.no_grad():
            features = source(image)
    finally:
        pipeline.vae.decode = decoder
    shapes = [
        {
            "layer": int(layer),
            "target": list(target.shape),
            "source": list(condition.shape),
            "dtype": str(target.dtype),
            "finite": bool(torch.isfinite(target).all() and torch.isfinite(condition).all()),
        }
        for layer, (target, condition) in zip(feature_layers, features)
    ]
    if len(shapes) != len(feature_layers) or not all(item["finite"] for item in shapes):
        raise RuntimeError("Qwen feature validation produced missing/non-finite maps")
    report = {
        "schema": _SCHEMA,
        "validated": True,
        "qwen_vae_decoder_called": False,
        "model_id": str(model_root),
        "model_index_sha256": file_sha256(model_index),
        "transformer_config_sha256": file_sha256(transformer_config),
        "pipeline_class": pipeline.__class__.__name__,
        "diffusers_version": package_version("diffusers"),
        "torch_version": torch.__version__,
        "device": str(device),
        "input_image": str(image_file),
        "input_image_sha256": file_sha256(image_file),
        "input_size": list(input_size),
        "feature_type": feature_type,
        "feature_layers": list(feature_layers),
        "probe_contract": {
            "input_size": list(input_size),
            "feature_type": feature_type,
            "feature_layers": list(feature_layers),
            "feature_dtype": feature_dtype,
            "feature_quantization": "none",
            "feature_num_inference_steps": 1,
            "guidance_scale": 1.0,
            "feature_seed": 0,
            "hidden_channels": 3072,
            "cpu_offload": bool(cpu_offload),
            "local_files_only": True,
            "output_type": "latent",
            "vae_decoder_forbidden": True,
        },
        "features": shapes,
        "external_parameter_count": _component_parameter_count(pipeline),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    _write_json_atomic(destination, report)
    return report


def check_qwen_runtime_report(
    report_path: str | Path,
    model_id: str | Path,
    *,
    feature_type: str | None = None,
    feature_layers: tuple[int, ...] | None = None,
    input_size: tuple[int, int] | None = None,
    feature_dtype: str | None = None,
    cpu_offload: bool | None = None,
    feature_num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    feature_seed: int | None = None,
    hidden_channels: int | None = None,
) -> dict[str, Any]:
    path = Path(report_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    model_root = Path(model_id).resolve()
    if not isinstance(report, dict) or report.get("schema") != _SCHEMA:
        raise ValueError("invalid Qwen runtime validation report")
    if report.get("validated") is not True or report.get("qwen_vae_decoder_called") is not False:
        raise ValueError("Qwen runtime report did not validate the decoder-free feature path")
    if report.get("model_id") != str(model_root):
        raise ValueError("Qwen runtime report model path differs")
    contract = report.get("probe_contract")
    if not isinstance(contract, dict):
        raise ValueError("Qwen runtime report lacks the immutable probe contract")
    for key in ("input_size", "feature_type", "feature_layers"):
        if report.get(key) != contract.get(key):
            raise ValueError(f"Qwen runtime report contradicts its probe contract: {key}")
    expected_contract: dict[str, Any] = {
        "feature_type": feature_type,
        "feature_layers": None if feature_layers is None else list(feature_layers),
        "input_size": None if input_size is None else list(input_size),
        "feature_dtype": feature_dtype,
        "cpu_offload": cpu_offload,
        "feature_num_inference_steps": feature_num_inference_steps,
        "guidance_scale": guidance_scale,
        "feature_seed": feature_seed,
        "hidden_channels": hidden_channels,
    }
    for key, expected in expected_contract.items():
        if expected is not None and contract.get(key) != expected:
            raise ValueError(f"Qwen runtime report probe contract differs: {key}")
    fixed_contract = {
        "feature_quantization": "none",
        "local_files_only": True,
        "output_type": "latent",
        "vae_decoder_forbidden": True,
    }
    for key, expected in fixed_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"Qwen runtime report unsafe probe contract: {key}")
    for relative, key in (
        ("model_index.json", "model_index_sha256"),
        ("transformer/config.json", "transformer_config_sha256"),
    ):
        current = file_sha256(model_root / relative)
        if report.get(key) != current:
            raise ValueError(f"Qwen runtime report {key} is stale")
    features = report.get("features")
    if not isinstance(features, list) or not features or not all(
        isinstance(value, dict) and value.get("finite") is True for value in features
    ):
        raise ValueError("Qwen runtime report lacks finite feature shapes")
    declared_layers = contract.get("feature_layers")
    if (
        not isinstance(declared_layers, list)
        or [item.get("layer") for item in features] != declared_layers
    ):
        raise ValueError("Qwen runtime report feature entries differ from declared layers")
    declared_channels = contract.get("hidden_channels")
    for item in features:
        target_shape = item.get("target")
        source_shape = item.get("source")
        if (
            not isinstance(target_shape, list)
            or not isinstance(source_shape, list)
            or len(target_shape) != 4
            or len(source_shape) != 4
            or target_shape[0] != 1
            or source_shape[0] != 1
            or target_shape[1] != declared_channels
            or source_shape[1] != declared_channels
        ):
            raise ValueError("Qwen runtime report feature shape violates probe contract")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--feature-type", choices=("hidden", "qk"), default="hidden")
    parser.add_argument("--feature-layers", type=int, nargs="+", default=(-24, -12, -1))
    parser.add_argument("--feature-dtype", default="bfloat16")
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--check-report")
    args = parser.parse_args()
    if args.check_report:
        report = check_qwen_runtime_report(
            args.check_report,
            args.model_id,
            feature_type=args.feature_type,
            feature_layers=tuple(args.feature_layers),
            input_size=(args.input_height, args.input_width),
            feature_dtype=args.feature_dtype,
            cpu_offload=not args.no_cpu_offload,
            feature_num_inference_steps=1,
            guidance_scale=1.0,
            feature_seed=0,
            hidden_channels=3072,
        )
    else:
        if not args.image or not args.output:
            parser.error("validation requires --image and --output")
        report = validate_qwen_runtime(
            args.image,
            args.model_id,
            args.output,
            device_name=args.device,
            input_size=(args.input_height, args.input_width),
            feature_type=args.feature_type,
            feature_layers=tuple(args.feature_layers),
            feature_dtype=args.feature_dtype,
            cpu_offload=not args.no_cpu_offload,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
