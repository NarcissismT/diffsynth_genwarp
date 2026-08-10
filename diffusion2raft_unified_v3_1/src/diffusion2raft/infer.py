"""Run document rectification with optional invalid-border inpainting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import load_config
from .deployment import (
    build_teacher_deployment_contract,
    validate_teacher_deployment_contract,
)
from .geometry import (
    backward_flow_to_map,
    backward_warp,
    flow_valid_mask,
    make_pixel_grid,
    resize_backward_flow,
)
from .losses import jacobian_determinant
from .models import build_rectifier
from .postprocess import TorchScriptLamaInpainter


CHECKPOINT_ARTIFACT_FIELDS = (
    "path",
    "size_bytes",
    "mtime_ns",
    "sha256",
)
_CHECKPOINT_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CheckpointProvenanceError(RuntimeError):
    """The checkpoint cannot be proven to be the requested immutable artifact."""


def _canonical_checkpoint_path(path: str | Path) -> Path:
    """Resolve the parent but preserve the leaf for the O_NOFOLLOW open."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.name in {"", ".", ".."}:
        raise CheckpointProvenanceError(f"invalid checkpoint leaf: {path!r}")
    try:
        parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CheckpointProvenanceError(
            f"cannot resolve checkpoint parent {candidate.parent}: {error}"
        ) from error
    return parent / candidate.name


def _checkpoint_identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(value, field)) for field in _CHECKPOINT_IDENTITY_FIELDS)


def _validate_expected_checkpoint_artifact(
    expected: Mapping[str, Any], *, canonical_path: Path
) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise CheckpointProvenanceError("expected checkpoint artifact must be a mapping")
    actual_fields = set(expected)
    required_fields = set(CHECKPOINT_ARTIFACT_FIELDS)
    if actual_fields != required_fields:
        raise CheckpointProvenanceError(
            "expected checkpoint artifact fields are incomplete; "
            f"missing={sorted(required_fields - actual_fields)}, "
            f"extra={sorted(actual_fields - required_fields)}"
        )
    path = expected.get("path")
    if not isinstance(path, str) or path != str(canonical_path):
        raise CheckpointProvenanceError(
            "expected checkpoint canonical path mismatch; "
            f"expected={str(canonical_path)!r}, supplied={path!r}"
        )
    size = expected.get("size_bytes")
    mtime = expected.get("mtime_ns")
    digest = expected.get("sha256")
    if type(size) is not int or size <= 0:
        raise CheckpointProvenanceError(
            f"expected checkpoint size_bytes must be a positive integer, got {size!r}"
        )
    if type(mtime) is not int or mtime < 0:
        raise CheckpointProvenanceError(
            f"expected checkpoint mtime_ns must be a non-negative integer, got {mtime!r}"
        )
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise CheckpointProvenanceError(
            f"expected checkpoint sha256 is invalid: {digest!r}"
        )
    return {
        "path": path,
        "size_bytes": size,
        "mtime_ns": mtime,
        "sha256": digest,
    }


def _torch_load_checkpoint_handle(handle: Any) -> Any:
    """Load from a rewound descriptor on both current and legacy torch."""

    handle.seek(0)
    try:
        return torch.load(handle, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch may consume bytes before rejecting ``weights_only``.
        handle.seek(0)
        return torch.load(handle, map_location="cpu")


def load_checkpoint_with_provenance(
    path: str | Path,
    *,
    expected_artifact: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Hash, attest, and load a checkpoint through one stable descriptor.

    The expected artifact, when supplied, is compared after hashing and before
    deserialization.  ``torch.load`` receives the already-authenticated file
    object rather than the pathname, closing the summary-to-load swap window.
    """

    canonical_path = _canonical_checkpoint_path(path)
    normalized_expected = (
        _validate_expected_checkpoint_artifact(
            expected_artifact, canonical_path=canonical_path
        )
        if expected_artifact is not None
        else None
    )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if nofollow == 0 or nonblock == 0:
        raise CheckpointProvenanceError(
            "platform lacks O_NOFOLLOW/O_NONBLOCK required for safe checkpoint loading"
        )
    flags = os.O_RDONLY | nofollow | nonblock | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(canonical_path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            initial_fd_stat = os.fstat(handle.fileno())
            initial_path_stat = os.lstat(canonical_path)
            if not stat.S_ISREG(initial_fd_stat.st_mode):
                raise CheckpointProvenanceError(
                    f"checkpoint is not a regular file: {canonical_path}"
                )
            if initial_fd_stat.st_size <= 0:
                raise CheckpointProvenanceError(f"checkpoint is empty: {canonical_path}")
            initial_identity = _checkpoint_identity(initial_fd_stat)
            if _checkpoint_identity(initial_path_stat) != initial_identity:
                raise CheckpointProvenanceError(
                    "checkpoint pathname and opened descriptor identities differ: "
                    f"{canonical_path}"
                )

            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            artifact = {
                "path": str(canonical_path),
                "size_bytes": int(initial_fd_stat.st_size),
                "mtime_ns": int(initial_fd_stat.st_mtime_ns),
                "sha256": digest.hexdigest(),
            }
            if normalized_expected is not None and artifact != normalized_expected:
                raise CheckpointProvenanceError(
                    "checkpoint artifact does not match finalizer attestation; "
                    f"expected={normalized_expected!r}, actual={artifact!r}"
                )

            try:
                payload = _torch_load_checkpoint_handle(handle)
            finally:
                # Run the postcondition even when deserialization itself raises;
                # a concurrent replacement must never be hidden by a load error.
                final_fd_stat = os.fstat(handle.fileno())
                final_path_stat = os.lstat(canonical_path)
                if (
                    _checkpoint_identity(final_fd_stat) != initial_identity
                    or _checkpoint_identity(final_path_stat) != initial_identity
                ):
                    raise CheckpointProvenanceError(
                        "checkpoint changed or its pathname was replaced during load: "
                        f"{canonical_path}"
                    )
    except CheckpointProvenanceError:
        raise
    except OSError as error:
        raise CheckpointProvenanceError(
            f"checkpoint cannot be safely opened or read {canonical_path}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return payload, artifact


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "inference.image_decoder=opencv or "
            "resize_interpolation=opencv_baseline requires OpenCV"
        ) from error
    return cv2


def _load_image(
    path: str | Path,
    *,
    decoder: str = "pil",
) -> tuple[torch.Tensor, Image.Image]:
    normalized = str(decoder).lower()
    if normalized == "pil":
        # Keep the original default path unchanged for existing configs.
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor, image
    if normalized != "opencv":
        raise ValueError(
            f"unknown inference image decoder {decoder!r}; use pil or opencv"
        )

    cv2 = _require_cv2()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV failed to decode image: {path}")
    image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    tensor = (
        torch.from_numpy(image_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
    )
    return tensor, Image.fromarray(image_rgb, mode="RGB")


def _resize(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=True)


def _opencv_baseline_resize(
    image: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Reproduce the corrected-512 deployment's OpenCV resize branches.

    The source deployment resized uint8 images before dividing by 255.  Images
    whose short side exceeds 2048 first go through an aspect-preserving AREA
    resize to short-side 1024, followed by an AREA stretch to the square model
    canvas.  Ordinary downsampling uses AREA and upsampling uses LINEAR.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be [B,3,H,W], got {tuple(image.shape)}")
    if image.device.type != "cpu":
        raise ValueError("opencv_baseline resize requires a CPU image tensor")
    target_h, target_w = int(size[0]), int(size[1])
    if target_h != target_w:
        raise ValueError(
            "opencv_baseline resize requires a square model canvas, got "
            f"{(target_h, target_w)}"
        )
    if min(target_h, target_w) <= 1:
        raise ValueError(f"resize dimensions must be greater than one, got {size}")

    cv2 = _require_cv2()
    arrays = (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )
    resized_tensors: list[torch.Tensor] = []
    for array in arrays:
        height, width = int(array.shape[0]), int(array.shape[1])
        if min(height, width) > 2048:
            scale = 1024.0 / min(height, width)
            middle_width = int(width * scale)
            middle_height = int(height * scale)
            array = cv2.resize(
                array,
                (middle_width, middle_height),
                interpolation=cv2.INTER_AREA,
            )
            array = cv2.resize(
                array,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )
        elif min(height, width) > target_h:
            array = cv2.resize(
                array,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            array = cv2.resize(
                array,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
        resized_tensors.append(
            torch.from_numpy(np.ascontiguousarray(array))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )
    return torch.stack(resized_tensors, dim=0)


@dataclass(frozen=True)
class CanvasTransform:
    """Map a native image into the fixed model canvas without losing aspect ratio."""

    native_size: tuple[int, int]
    canvas_size: tuple[int, int]
    content_top: int
    content_left: int
    content_size: tuple[int, int]
    policy: str

    @property
    def content_slices(self) -> tuple[slice, slice]:
        height, width = self.content_size
        return (
            slice(self.content_top, self.content_top + height),
            slice(self.content_left, self.content_left + width),
        )


def compute_canvas_transform(
    native_size: tuple[int, int],
    canvas_size: tuple[int, int],
    policy: str,
) -> CanvasTransform:
    """Return the exact source/target canvas transform used during inference."""

    native_h, native_w = (int(native_size[0]), int(native_size[1]))
    canvas_h, canvas_w = (int(canvas_size[0]), int(canvas_size[1]))
    if min(native_h, native_w, canvas_h, canvas_w) <= 1:
        raise ValueError(
            f"native/canvas dimensions must be greater than one, got "
            f"native={native_size}, canvas={canvas_size}"
        )
    normalized = str(policy).lower()
    if normalized == "stretch":
        content_h, content_w = canvas_h, canvas_w
    elif normalized == "letterbox":
        scale = min(canvas_h / native_h, canvas_w / native_w)
        content_h = min(canvas_h, max(2, int(round(native_h * scale))))
        content_w = min(canvas_w, max(2, int(round(native_w * scale))))
    else:
        raise ValueError(
            f"unknown inference resize policy {policy!r}; use stretch or letterbox"
        )
    return CanvasTransform(
        native_size=(native_h, native_w),
        canvas_size=(canvas_h, canvas_w),
        content_top=(canvas_h - content_h) // 2,
        content_left=(canvas_w - content_w) // 2,
        content_size=(content_h, content_w),
        policy=normalized,
    )


def image_to_model_canvas(
    image: torch.Tensor,
    transform: CanvasTransform,
    *,
    padding_mode: str = "replicate",
    resize_interpolation: str = "bilinear",
) -> torch.Tensor:
    """Apply ``transform`` to an ``[B,C,H,W]`` native image tensor."""

    if tuple(int(value) for value in image.shape[-2:]) != transform.native_size:
        raise ValueError(
            f"image size {tuple(image.shape[-2:])} does not match transform "
            f"native size {transform.native_size}"
        )
    interpolation = str(resize_interpolation).lower()
    if interpolation == "bilinear":
        resized = _resize(image, transform.content_size)
    elif interpolation == "opencv_baseline":
        if transform.policy != "stretch":
            raise ValueError(
                "opencv_baseline resize is only supported with resize_policy=stretch"
            )
        if transform.canvas_size[0] != transform.canvas_size[1]:
            raise ValueError(
                "opencv_baseline resize requires a square model canvas, got "
                f"{transform.canvas_size}"
            )
        resized = _opencv_baseline_resize(image, transform.content_size)
    else:
        raise ValueError(
            "unknown inference resize interpolation "
            f"{resize_interpolation!r}; use bilinear or opencv_baseline"
        )
    content_h, content_w = transform.content_size
    canvas_h, canvas_w = transform.canvas_size
    top = transform.content_top
    left = transform.content_left
    padding = (left, canvas_w - left - content_w, top, canvas_h - top - content_h)
    if not any(padding):
        return resized
    if padding_mode == "replicate":
        return F.pad(resized, padding, mode="replicate")
    if padding_mode == "white":
        return F.pad(resized, padding, mode="constant", value=1.0)
    if padding_mode == "black":
        return F.pad(resized, padding, mode="constant", value=0.0)
    raise ValueError(
        f"unknown inference padding mode {padding_mode!r}; "
        "use replicate, white, or black"
    )


def flow_from_model_canvas(
    flow: torch.Tensor,
    transform: CanvasTransform,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Restore a model-canvas flow through an exact absolute-map transform.

    Both the target crop and the source-coordinate canvas are transformed.  In
    particular, source coordinates that land in letterbox padding remain
    outside the native image and are subsequently marked invalid instead of
    being silently squeezed onto a real border pixel.
    """

    if tuple(int(value) for value in flow.shape[-2:]) != transform.canvas_size:
        raise ValueError(
            f"flow canvas {tuple(flow.shape[-2:])} does not match "
            f"transform canvas {transform.canvas_size}"
        )
    y_slice, x_slice = transform.content_slices
    pixel_map = backward_flow_to_map(flow)[..., y_slice, x_slice]
    pixel_map = F.interpolate(
        pixel_map,
        size=target_size,
        mode="bilinear",
        align_corners=True,
    )
    native_h, native_w = transform.native_size
    content_h, content_w = transform.content_size
    pixel_map = pixel_map.clone()
    pixel_map[:, 0] = (
        (pixel_map[:, 0] - transform.content_left)
        * (native_w - 1)
        / max(content_w - 1, 1)
    )
    pixel_map[:, 1] = (
        (pixel_map[:, 1] - transform.content_top)
        * (native_h - 1)
        / max(content_h - 1, 1)
    )
    target_grid = make_pixel_grid(
        flow.shape[0],
        int(target_size[0]),
        int(target_size[1]),
        device=flow.device,
        dtype=flow.dtype,
    )
    return pixel_map - target_grid


def residual_from_model_canvas(
    residual: torch.Tensor,
    transform: CanvasTransform,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Restore a target-to-intermediate residual after removing letterbox padding."""

    y_slice, x_slice = transform.content_slices
    cropped = residual[..., y_slice, x_slice]
    return resize_backward_flow(
        cropped,
        target_size,
        source_size_from=transform.content_size,
        source_size_to=target_size,
    )


def _save_image(
    tensor: torch.Tensor,
    path: Path,
    *,
    quantization: str = "round",
) -> None:
    scaled = (
        tensor.detach()
        .squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
    )
    if quantization == "round":
        scaled = scaled.round()
    elif quantization != "truncate":
        raise ValueError(f"unknown image quantization mode: {quantization!r}")
    array = scaled.byte().cpu().numpy()
    Image.fromarray(array).save(path)


def _strict_json_value(value: Any) -> Any:
    """Replace non-finite floats recursively so emitted metadata is strict JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def _safe_quantile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    """torch.quantile rejects tensors above ~16M elements; subsample if needed."""

    flat = values.reshape(-1).float()
    limit = 16_000_000
    if flat.numel() > limit:
        # Deterministic evenly-spaced subsample preserves the distribution shape
        # for a percentile estimate without a huge allocation.
        step = (flat.numel() + limit - 1) // limit
        flat = flat[::step]
    return torch.quantile(flat, quantile)


def _curvature_p95(flow: torch.Tensor) -> float:
    values: list[torch.Tensor] = []
    if flow.shape[-1] >= 3:
        dxx = flow[..., 2:] - 2.0 * flow[..., 1:-1] + flow[..., :-2]
        values.append(torch.linalg.vector_norm(dxx, dim=1).reshape(-1))
    if flow.shape[-2] >= 3:
        dyy = flow[..., 2:, :] - 2.0 * flow[..., 1:-1, :] + flow[..., :-2, :]
        values.append(torch.linalg.vector_norm(dyy, dim=1).reshape(-1))
    return 0.0 if not values else float(_safe_quantile(torch.cat(values), 0.95).cpu())


class RectificationSession:
    """Load Qwen and the rectifier once, then process one or many images."""

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint_path: str | Path,
        *,
        stage: str = "unified",
        expected_checkpoint_artifact: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        requested_device = str(config.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(requested_device)
        self.work_size = tuple(int(v) for v in config["data"]["work_size"])
        if self.work_size[0] % 8 or self.work_size[1] % 8:
            raise ValueError(f"work_size must be divisible by 8, got {self.work_size}")
        inference_config = dict(config.get("inference", {}))
        self.resize_policy = str(
            inference_config.get("resize_policy", "stretch")
        ).lower()
        self.padding_mode = str(
            inference_config.get("padding_mode", "replicate")
        ).lower()
        self.image_decoder = str(
            inference_config.get("image_decoder", "pil")
        ).lower()
        self.resize_interpolation = str(
            inference_config.get("resize_interpolation", "bilinear")
        ).lower()
        # Validate early, before loading a 20B checkpoint.
        compute_canvas_transform((2, 2), self.work_size, self.resize_policy)
        if self.padding_mode not in {"replicate", "white", "black"}:
            raise ValueError(
                f"unknown inference padding mode {self.padding_mode!r}; "
                "use replicate, white, or black"
            )
        if self.image_decoder not in {"pil", "opencv"}:
            raise ValueError(
                f"unknown inference image decoder {self.image_decoder!r}; "
                "use pil or opencv"
            )
        if self.resize_interpolation not in {"bilinear", "opencv_baseline"}:
            raise ValueError(
                "unknown inference resize interpolation "
                f"{self.resize_interpolation!r}; use bilinear or opencv_baseline"
            )
        if self.resize_interpolation == "opencv_baseline":
            if self.image_decoder != "opencv":
                raise ValueError(
                    "resize_interpolation=opencv_baseline requires "
                    "image_decoder=opencv"
                )
            if self.resize_policy != "stretch":
                raise ValueError(
                    "resize_interpolation=opencv_baseline requires "
                    "resize_policy=stretch"
                )
            if self.work_size[0] != self.work_size[1]:
                raise ValueError(
                    "resize_interpolation=opencv_baseline requires a square "
                    f"work_size, got {self.work_size}"
                )
            _require_cv2()

        raw_inpaint_config = inference_config.get("inpaint", {})
        if raw_inpaint_config is None:
            raw_inpaint_config = {}
        if not isinstance(raw_inpaint_config, dict):
            raise TypeError("inference.inpaint must be a mapping")
        self.inpaint_enabled = bool(raw_inpaint_config.get("enabled", False))
        self.inpainter: TorchScriptLamaInpainter | None = None
        self.inpaint_identity: dict[str, Any] = {"enabled": False}
        if self.inpaint_enabled:
            configured_path = raw_inpaint_config.get("path")
            if not configured_path:
                raise ValueError(
                    "inference.inpaint.path is required when inpainting is enabled"
                )
            configured_sha256 = raw_inpaint_config.get("sha256")
            if not isinstance(configured_sha256, str) or not configured_sha256:
                raise ValueError(
                    "inference.inpaint.sha256 is required when inpainting is enabled"
                )
            inpaint_path = Path(str(configured_path))
            inpaint_size = int(raw_inpaint_config.get("size", 512))
            dilation = int(raw_inpaint_config.get("dilation", 11))
            self.inpainter = TorchScriptLamaInpainter(
                inpaint_path,
                device=self.device,
                inpaint_size=inpaint_size,
                dilation_kernel=dilation,
                expected_sha256=configured_sha256,
            )
            self.inpaint_identity = dict(self.inpainter.identity)
        self.stage = str(stage)
        self.payload, self.checkpoint_artifact = load_checkpoint_with_provenance(
            checkpoint_path,
            expected_artifact=expected_checkpoint_artifact,
        )
        self.checkpoint_path = Path(self.checkpoint_artifact["path"])
        self.model_config = dict(config["model"])
        if self.stage == "joint":
            self.model_config["raft_pretrained"] = False
        self.model = build_rectifier(
            self.model_config,
            dict(config.get("qwen", {})),
            stage=self.stage,
            device=self.device,
        )
        state = self.payload.get("model", self.payload)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # A unified checkpoint carries extra unified-only branches (Qwen
        # encoder, CNN encoder, reliability fusion, recurrent refiner).  Loading
        # it into a prior/joint model to inspect the shared Stage-A geometry is
        # legitimate, so ignore those known-extra keys; any other mismatch is
        # still a real error.
        if self.stage in {"prior", "joint"}:
            unified_only_prefixes = (
                "diffusion_encoder.",
                "cnn_encoder.",
                "fusion.",
                "refiner.",
            )
            unexpected = [
                key for key in unexpected
                if not key.startswith(unified_only_prefixes)
            ]
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch; missing={missing[:12]}, "
                f"unexpected={unexpected[:12]}"
            )
        self._restore_teacher_checkpoint_contract()
        self.model.to(self.device).eval()
        self._restore_correlation_temperature()

    def _restore_teacher_checkpoint_contract(self) -> None:
        """Bind an external teacher and its exact residual deployment scale."""

        if (
            self.stage != "unified"
            or str(getattr(self.model, "prior_backend", "learned"))
            != "torchscript"
        ):
            return
        from .train import (
            _restore_residual_application_from_checkpoint,
            _validate_teacher_prior_identity,
        )

        _validate_teacher_prior_identity(
            self.model, self.payload, required=True
        )
        _restore_residual_application_from_checkpoint(
            self.model, self.payload, required=True
        )
        teacher_identity = getattr(self.model, "teacher_prior_identity", None)
        if teacher_identity is None:
            raise RuntimeError("TorchScript model does not expose teacher identity")
        current_contract = build_teacher_deployment_contract(
            self.config,
            teacher_identity=teacher_identity,
            inpaint_identity=self.inpaint_identity,
        )
        validate_teacher_deployment_contract(
            self.payload.get("deployment_contract"),
            current_contract,
            external_files_authenticated=True,
        )
        self.deployment_contract = current_contract

    def _restore_correlation_temperature(self) -> None:
        """Match the correlation temperature the checkpoint was trained with.

        The refiner's cost volume is scaled by ``1 / correlation_temperature``.
        The unified config ramps this value across the first epochs, so a
        checkpoint saved mid-ramp expects a temperature different from the
        config target.  Building the model at the config default and never
        restoring it feeds the refiner an input distribution it never saw and
        silently destroys the residual (unified then looks worse than prior).
        The checkpoint stores its own ``config`` and ``epoch``; use them to
        recover the exact training-time temperature.
        """

        if self.stage != "unified":
            return
        set_temperature = getattr(self.model, "set_correlation_temperature", None)
        if set_temperature is None:
            raise RuntimeError(
                "unified checkpoint model cannot restore correlation temperature"
            )
        from .train import _checkpoint_correlation_temperature

        temperature = _checkpoint_correlation_temperature(
            self.payload, required=True
        )
        if temperature is None:  # pragma: no cover - required=True is strict.
            raise RuntimeError("checkpoint correlation_temperature is unavailable")
        set_temperature(float(temperature))
        self.correlation_temperature = float(temperature)

    @torch.inference_mode()
    def rectify(
        self,
        warped_path: str | Path,
        guide_path: str | Path | None,
        output_dir: str | Path,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, Path]:
        warped_native, _ = _load_image(
            warped_path,
            decoder=self.image_decoder,
        )
        native_source_size = tuple(int(v) for v in warped_native.shape[-2:])
        target_size = output_size or native_source_size
        transform = compute_canvas_transform(
            native_source_size,
            self.work_size,
            self.resize_policy,
        )
        warped_work = image_to_model_canvas(
            warped_native,
            transform,
            padding_mode=self.padding_mode,
            resize_interpolation=self.resize_interpolation,
        ).to(self.device)

        if self.stage == "joint":
            if guide_path is None:
                raise ValueError("legacy joint inference needs --guide")
            guide_native, _ = _load_image(
                guide_path,
                decoder=self.image_decoder,
            )
            if tuple(int(v) for v in guide_native.shape[-2:]) != native_source_size:
                raise ValueError(
                    "letterbox-safe joint inference requires guide and warped images "
                    f"to share a canvas; got guide={tuple(guide_native.shape[-2:])}, "
                    f"warped={native_source_size}"
                )
            guide_work = image_to_model_canvas(
                guide_native,
                transform,
                padding_mode=self.padding_mode,
                resize_interpolation=self.resize_interpolation,
            ).to(self.device)
        else:
            guide_work = None

        outputs = self.model(warped_work, guide_work, stage=self.stage)
        flow_work = outputs["final_flow"].float()
        prior_flow_work = outputs["prior_flow"].float()
        flow_native = flow_from_model_canvas(flow_work, transform, target_size)
        prior_flow_native = flow_from_model_canvas(
            prior_flow_work,
            transform,
            target_size,
        )
        valid = flow_valid_mask(flow_native, native_source_size)
        warped_native_device = warped_native.to(self.device)
        final_raw_padding_mode = "zeros" if self.inpainter is not None else "border"
        rectified_raw = backward_warp(
            warped_native_device,
            flow_native,
            padding_mode=final_raw_padding_mode,
        )
        if self.inpainter is not None:
            rectified, inpaint_mask = self.inpainter.forward_with_mask(
                rectified_raw, ~valid
            )
        else:
            rectified = rectified_raw
            inpaint_mask = torch.zeros_like(valid)
        evaluation_valid = valid & ~inpaint_mask
        prior_rectified = backward_warp(
            warped_native_device,
            prior_flow_native,
            padding_mode="border",
        )
        determinant = jacobian_determinant(flow_native)
        valid_bool = valid[:, 0]
        valid_cells = (
            valid_bool[:, :-1, :-1]
            & valid_bool[:, 1:, :-1]
            & valid_bool[:, :-1, 1:]
            & valid_bool[:, 1:, 1:]
        )
        if valid_cells.any():
            valid_determinant = determinant[valid_cells]
            fold_rate = float((valid_determinant <= 0).float().mean().cpu())
            jacobian_p01 = float(
                _safe_quantile(valid_determinant, 0.01).cpu()
            )
        else:
            fold_rate = None
            jacobian_p01 = None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(warped_path).stem
        image_path = output_dir / f"{stem}_rectified.png"
        raw_image_path = (
            output_dir / f"{stem}_rectified_raw.png"
            if self.inpainter is not None
            else None
        )
        prior_image_path = output_dir / f"{stem}_prior_rectified.png"
        flow_path = output_dir / f"{stem}_backward_flow.npy"
        prior_flow_path = output_dir / f"{stem}_prior_backward_flow.npy"
        valid_path = output_dir / f"{stem}_valid.png"
        inpaint_mask_path = (
            output_dir / f"{stem}_inpaint_mask.png"
            if self.inpainter is not None
            else None
        )
        evaluation_valid_path = (
            output_dir / f"{stem}_evaluation_valid.png"
            if self.inpainter is not None
            else None
        )
        metadata_path = output_dir / f"{stem}_metadata.json"
        _save_image(
            rectified,
            image_path,
            quantization="truncate" if self.inpainter is not None else "round",
        )
        if raw_image_path is not None:
            _save_image(rectified_raw, raw_image_path)
        _save_image(prior_rectified, prior_image_path)
        np.save(
            flow_path,
            flow_native.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32),
        )
        np.save(
            prior_flow_path,
            prior_flow_native.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.float32),
        )
        _save_image(valid.expand(-1, 3, -1, -1).float(), valid_path)
        if inpaint_mask_path is not None and evaluation_valid_path is not None:
            _save_image(
                inpaint_mask.expand(-1, 3, -1, -1).float(),
                inpaint_mask_path,
            )
            _save_image(
                evaluation_valid.expand(-1, 3, -1, -1).float(),
                evaluation_valid_path,
            )

        residual_p50 = None
        residual_p95 = None
        residual_curvature_p95 = None
        residual_flow_path = None
        if outputs.get("residuals"):
            residual_native = residual_from_model_canvas(
                outputs["residuals"][-1].float(),
                transform,
                target_size,
            )
            residual_norm = torch.linalg.vector_norm(residual_native, dim=1)
            residual_p50 = float(_safe_quantile(residual_norm, 0.50).cpu())
            residual_p95 = float(_safe_quantile(residual_norm, 0.95).cpu())
            residual_curvature_p95 = _curvature_p95(residual_native)
            residual_flow_path = output_dir / f"{stem}_residual_backward_flow.npy"
            np.save(
                residual_flow_path,
                residual_native.squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.float32),
            )

        confidence_path = None
        confidence_mean = None
        confidence_p10 = None
        confidence_p50 = None
        confidence_p90 = None
        if "feature_confidence" in outputs:
            confidence_canvas = F.interpolate(
                outputs["feature_confidence"].float(),
                size=self.work_size,
                mode="bilinear",
                align_corners=True,
            )
            y_slice, x_slice = transform.content_slices
            confidence = F.interpolate(
                confidence_canvas[..., y_slice, x_slice],
                size=target_size,
                mode="bilinear",
                align_corners=True,
            ).clamp(0.0, 1.0)
            confidence_path = output_dir / f"{stem}_feature_confidence.png"
            _save_image(confidence.expand(-1, 3, -1, -1), confidence_path)
            confidence_mean = float(confidence.mean().cpu())
            confidence_p10 = float(_safe_quantile(confidence, 0.10).cpu())
            confidence_p50 = float(_safe_quantile(confidence, 0.50).cpu())
            confidence_p90 = float(_safe_quantile(confidence, 0.90).cpu())

        content_h, content_w = transform.content_size
        source_scale_y = (native_source_size[0] - 1) / max(content_h - 1, 1)
        source_scale_x = (native_source_size[1] - 1) / max(content_w - 1, 1)
        scale_min = max(min(source_scale_y, source_scale_x), 1e-12)
        metadata = {
            "flow_convention": "backward displacement: output(x,y) samples source((x,y)+flow(x,y)); channels=(x,y)",
            "warped": str(Path(warped_path).resolve()),
            "guide": str(Path(guide_path).resolve()) if guide_path else None,
            "feature_backend": self.model_config.get("feature_backend")
            if self.stage == "unified"
            else None,
            "uses_decoded_qwen_rgb": self.stage == "joint",
            "checkpoint": self.checkpoint_artifact["path"],
            "checkpoint_artifact": dict(self.checkpoint_artifact),
            "checkpoint_training_revision": self.payload.get("training_revision"),
            "checkpoint_best_metric": self.payload.get("best_metric"),
            "checkpoint_epoch_index": self.payload.get("epoch"),
            "correlation_temperature": getattr(
                self, "correlation_temperature", None
            ),
            "teacher_prior_identity": self.payload.get("teacher_prior_identity"),
            "residual_application": self.payload.get("residual_application"),
            "deployment_contract": self.payload.get("deployment_contract"),
            "stage": self.stage,
            "resize_policy": transform.policy,
            "padding_mode": self.padding_mode,
            "image_decoder": self.image_decoder,
            "resize_interpolation": self.resize_interpolation,
            "inpaint_identity": self.inpaint_identity,
            "final_raw_padding_mode": final_raw_padding_mode,
            "final_image_inpainted": self.inpainter is not None,
            "final_image_quantization": (
                "uint8_truncate" if self.inpainter is not None else "uint8_round"
            ),
            "raw_geometry_image": str(raw_image_path.resolve())
            if raw_image_path is not None
            else None,
            "prior_diagnostic": {
                "padding_mode": "border",
                "inpainted": False,
            },
            "mask_semantics": {
                "valid": "flow_valid_before_inpainting",
                "inpaint_mask": (
                    "pixels_replaced_by_lama_after_11x11_dilation"
                    if self.inpainter is not None
                    else None
                ),
                "evaluation_valid": (
                    "flow_valid_and_not_replaced_by_lama"
                    if self.inpainter is not None
                    else "same_as_valid"
                ),
            },
            "work_size_hw": list(self.work_size),
            "work_content_box_yxhw": [
                transform.content_top,
                transform.content_left,
                content_h,
                content_w,
            ],
            "source_size_hw": list(native_source_size),
            "output_size_hw": list(target_size),
            "work_content_to_native_source_scale_yx": [
                source_scale_y,
                source_scale_x,
            ],
            "source_scale_anisotropy_ratio": max(source_scale_y, source_scale_x)
            / scale_min,
            "valid_fraction": float(valid.float().mean().cpu()),
            "inpaint_fraction": float(inpaint_mask.float().mean().cpu()),
            "evaluation_valid_fraction": float(
                evaluation_valid.float().mean().cpu()
            ),
            "fold_rate": fold_rate,
            "jacobian_p01": jacobian_p01,
            "residual_p50_px": residual_p50,
            "residual_p95_px": residual_p95,
            "final_flow_curvature_p95_px": _curvature_p95(flow_native),
            "prior_flow_curvature_p95_px": _curvature_p95(prior_flow_native),
            "residual_curvature_p95_px": residual_curvature_p95,
            "feature_confidence_mean": confidence_mean,
            "feature_confidence_p10": confidence_p10,
            "feature_confidence_p50": confidence_p50,
            "feature_confidence_p90": confidence_p90,
        }
        metadata_path.write_text(
            json.dumps(
                _strict_json_value(metadata),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return {
            "image": image_path,
            **({"raw_image": raw_image_path} if raw_image_path is not None else {}),
            "prior_image": prior_image_path,
            "flow": flow_path,
            "prior_flow": prior_flow_path,
            "valid": valid_path,
            **(
                {
                    "inpaint_mask": inpaint_mask_path,
                    "evaluation_valid": evaluation_valid_path,
                }
                if inpaint_mask_path is not None
                and evaluation_valid_path is not None
                else {}
            ),
            "metadata": metadata_path,
            **(
                {"residual_flow": residual_flow_path}
                if residual_flow_path is not None
                else {}
            ),
            **({"confidence": confidence_path} if confidence_path is not None else {}),
        }


def rectify(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    warped_path: str | Path,
    guide_path: str | Path | None,
    output_dir: str | Path,
    *,
    stage: str = "unified",
    output_size: tuple[int, int] | None = None,
    expected_checkpoint_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Backward-compatible one-image API."""

    session = RectificationSession(
        config,
        checkpoint_path,
        stage=stage,
        expected_checkpoint_artifact=expected_checkpoint_artifact,
    )
    return session.rectify(
        warped_path,
        guide_path,
        output_dir,
        output_size=output_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-path")
    parser.add_argument("--expected-checkpoint-size-bytes", type=int)
    parser.add_argument("--expected-checkpoint-mtime-ns", type=int)
    parser.add_argument("--expected-checkpoint-sha256")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--warped")
    source.add_argument("--warped-dir")
    parser.add_argument("--guide")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stage", choices=("prior", "joint", "unified"), default="unified"
    )
    parser.add_argument("--output-height", type=int)
    parser.add_argument("--output-width", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resize-policy", choices=("stretch", "letterbox"))
    parser.add_argument("--padding-mode", choices=("replicate", "white", "black"))
    parser.add_argument("--image-decoder", choices=("pil", "opencv"))
    parser.add_argument(
        "--resize-interpolation", choices=("bilinear", "opencv_baseline")
    )
    parser.add_argument("--glob", default="*")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    inference_config = config.setdefault("inference", {})
    if args.resize_policy:
        inference_config["resize_policy"] = args.resize_policy
    if args.padding_mode:
        inference_config["padding_mode"] = args.padding_mode
    if args.image_decoder:
        inference_config["image_decoder"] = args.image_decoder
    if args.resize_interpolation:
        inference_config["resize_interpolation"] = args.resize_interpolation
    if (args.output_height is None) != (args.output_width is None):
        parser.error("--output-height and --output-width must be supplied together")
    output_size = (
        (args.output_height, args.output_width) if args.output_height is not None else None
    )
    if args.warped_dir and args.guide:
        parser.error("--guide is only supported with single-image --warped")
    expected_values = (
        args.expected_checkpoint_path,
        args.expected_checkpoint_size_bytes,
        args.expected_checkpoint_mtime_ns,
        args.expected_checkpoint_sha256,
    )
    if any(value is not None for value in expected_values) and not all(
        value is not None for value in expected_values
    ):
        parser.error(
            "expected checkpoint provenance arguments must be supplied together"
        )
    expected_checkpoint_artifact = (
        {
            "path": args.expected_checkpoint_path,
            "size_bytes": args.expected_checkpoint_size_bytes,
            "mtime_ns": args.expected_checkpoint_mtime_ns,
            "sha256": args.expected_checkpoint_sha256,
        }
        if all(value is not None for value in expected_values)
        else None
    )
    session = RectificationSession(
        config,
        args.checkpoint,
        stage=args.stage,
        expected_checkpoint_artifact=expected_checkpoint_artifact,
    )
    if args.warped:
        paths = session.rectify(
            args.warped,
            args.guide,
            args.output_dir,
            output_size=output_size,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    source_dir = Path(args.warped_dir)
    iterator = source_dir.rglob(args.glob) if args.recursive else source_dir.glob(args.glob)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    inputs = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in extensions
    )
    if not inputs:
        parser.error(f"no input images matched {args.glob!r} under {source_dir}")
    output_root = Path(args.output_dir)
    successes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, warped_path in enumerate(inputs, start=1):
        relative = warped_path.relative_to(source_dir)
        destination = output_root / relative.parent
        try:
            paths = session.rectify(
                warped_path,
                None,
                destination,
                output_size=output_size,
            )
            successes.append(
                {
                    "input": str(warped_path.resolve()),
                    "outputs": {name: str(path.resolve()) for name, path in paths.items()},
                }
            )
            print(f"[{index}/{len(inputs)}] {relative} -> {paths['image']}")
        except Exception as error:
            errors.append({"input": str(warped_path.resolve()), "error": str(error)})
            print(f"[{index}/{len(inputs)}] failed {relative}: {error}")
            if not args.continue_on_error:
                raise
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "inference_report.json"
    report_path.write_text(
        json.dumps(
            {
                "checkpoint": session.checkpoint_artifact["path"],
                "checkpoint_artifact": dict(session.checkpoint_artifact),
                "source_dir": str(source_dir.resolve()),
                "resize_policy": session.resize_policy,
                "padding_mode": session.padding_mode,
                "image_decoder": session.image_decoder,
                "resize_interpolation": session.resize_interpolation,
                "inpaint_identity": session.inpaint_identity,
                "input_count": len(inputs),
                "success_count": len(successes),
                "error_count": len(errors),
                "successes": successes,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"completed {len(successes)}/{len(inputs)} images; "
        f"errors={len(errors)} report={report_path}"
    )


if __name__ == "__main__":
    main()
