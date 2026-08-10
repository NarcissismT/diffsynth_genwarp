"""Frozen TorchScript geometry prior used as a strong baseline anchor.

The production teacher is the exact single-image flow regressor used to make
the comparison baseline.  It is intentionally kept outside the owning
``nn.Module`` registry: otherwise every unified checkpoint would duplicate a
multi-gigabyte frozen model.  A tiny persistent marker is the only teacher
state saved with the trainable unified heads.

The teacher's coordinate contract is a backward displacement on a stretched
square canvas.  Inputs are RGB tensors in ``[0, 1]``; the deployed model was
trained with BGR tensors and no mean/std normalization.  Its final flow is
Gaussian-smoothed before being restored to the caller's target/source canvas
through absolute coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms.functional import gaussian_blur

from ..external_file import open_stable_external_file
from ..geometry import resize_backward_flow


def _autocast_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower().replace("torch.", "")
    choices = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError(
            "teacher prior autocast dtype must be fp16, bf16, or fp32, "
            f"got {name!r}"
        )
    return choices[normalized]


class TorchScriptGeometryPrior(nn.Module):
    """Run a frozen external TorchScript backward-flow predictor.

    ``torch.jit.load`` receives ``map_location`` at construction time because
    the scripted module is deliberately not registered as a child module and
    therefore will not be moved by a later ``rectifier.to(device)`` call.
    """

    backend_name = "torchscript"
    marker_name = "_teacher_backend_marker"
    _MARKER = (0x54, 0x53, 0x01)  # ``TS`` + serialization contract revision.

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str,
        input_size: int = 512,
        flow_size: int | None = None,
        blur_kernel: int = 39,
        autocast_dtype: str = "float16",
        requires_logical_cuda0: bool = False,
        expected_sha256: str | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = str(checkpoint_path)
        self.input_size = int(input_size)
        self.flow_size = int(input_size if flow_size is None else flow_size)
        self.blur_kernel = int(blur_kernel)
        self.autocast_dtype = _autocast_dtype(autocast_dtype)
        self.requires_logical_cuda0 = bool(requires_logical_cuda0)
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())
        self.teacher_device = requested_device
        if self.requires_logical_cuda0 and self.teacher_device != torch.device(
            "cuda", 0
        ):
            raise RuntimeError(
                "this TorchScript teacher requires process-local logical cuda:0; "
                f"got {self.teacher_device}. Use the isolated v3.3 launcher or "
                "run single-GPU inference with the selected card exposed as cuda:0."
            )
        if self.input_size <= 1:
            raise ValueError("teacher prior input_size must be greater than one")
        if self.flow_size <= 1:
            raise ValueError("teacher prior flow_size must be greater than one")
        if self.blur_kernel < 1 or self.blur_kernel % 2 != 1:
            raise ValueError("teacher prior blur_kernel must be a positive odd integer")
        with open_stable_external_file(
            self.checkpoint_path,
            expected_sha256=expected_sha256,
            label="TorchScript geometry teacher",
        ) as opened:
            teacher = torch.jit.load(
                opened.load_path, map_location=self.teacher_device
            )
            identity = dict(opened.identity)
        self.resolved_checkpoint_path = str(identity["resolved_path"])
        self.checkpoint_size_bytes = int(identity["file_size"])
        self.checkpoint_mtime_ns = int(identity["mtime_ns"])
        self.checkpoint_sha256 = str(identity["sha256"])
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
        # Bypass nn.Module.__setattr__: the 3.5GB teacher must not appear in
        # parameters(), state_dict(), DDP broadcasts, or unified checkpoints.
        object.__setattr__(self, "_teacher", teacher)
        self.register_buffer(
            self.marker_name,
            torch.tensor(self._MARKER, dtype=torch.uint8),
            persistent=True,
        )

    @property
    def teacher(self) -> torch.jit.ScriptModule:
        return object.__getattribute__(self, "_teacher")

    def train(self, mode: bool = True) -> "TorchScriptGeometryPrior":
        # The wrapper may follow the unified heads into train mode, but the
        # external teacher itself must remain deterministic and frozen.
        super().train(mode)
        self.teacher.eval()
        return self

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        marker_key = prefix + self.marker_name
        incoming = state_dict.get(marker_key)
        expected = getattr(self, self.marker_name)
        if incoming is not None and (
            incoming.dtype != expected.dtype
            or tuple(incoming.shape) != tuple(expected.shape)
            or not torch.equal(incoming.detach().cpu(), expected.detach().cpu())
        ):
            error_msgs.append(
                f"invalid TorchScript prior backend marker at {marker_key!r}"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _check_input(warped: Tensor) -> None:
        if warped.ndim != 4 or warped.shape[1] != 3:
            raise ValueError(f"warped must be [B,3,H,W], got {tuple(warped.shape)}")
        if not warped.is_floating_point():
            raise TypeError(f"warped must be floating point, got {warped.dtype}")
        if min(int(warped.shape[-2]), int(warped.shape[-1])) <= 1:
            raise ValueError(
                "teacher prior requires spatial dimensions greater than one, "
                f"got {tuple(warped.shape[-2:])}"
            )
        if not bool(torch.isfinite(warped).all()):
            raise ValueError("warped contains NaN or infinite values")

    def _extract_flow(self, output: Any, batch: int) -> Tensor:
        if not isinstance(output, (list, tuple)) or not output:
            raise TypeError(
                "TorchScript geometry teacher must return a non-empty Tensor list"
            )
        flow = output[-1]
        expected = (batch, 2, self.flow_size, self.flow_size)
        if not isinstance(flow, Tensor):
            raise TypeError(
                "last TorchScript geometry teacher output must be a Tensor, "
                f"got {type(flow).__name__}"
            )
        if tuple(flow.shape) != expected:
            raise ValueError(
                "TorchScript geometry teacher returned the wrong flow shape; "
                f"expected {expected}, got {tuple(flow.shape)}"
            )
        if not flow.is_floating_point():
            raise TypeError(f"teacher flow must be floating point, got {flow.dtype}")
        if not bool(torch.isfinite(flow).all()):
            raise ValueError("TorchScript geometry teacher returned NaN or infinite flow")
        return flow

    def forward(self, warped: Tensor) -> Tensor:
        self._check_input(warped)
        if warped.device != self.teacher_device:
            raise RuntimeError(
                "teacher prior input/device mismatch: scripted teacher was loaded on "
                f"{self.teacher_device}, input is on {warped.device}"
            )
        native_size = tuple(int(value) for value in warped.shape[-2:])
        with torch.autocast(device_type=warped.device.type, enabled=False):
            teacher_input = F.interpolate(
                warped.float(),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                # Match cv2.resize used by the baseline deployment.
                align_corners=False,
            )
            # The deployed baseline consumes BGR [0,1], without mean/std scaling.
            teacher_input = teacher_input[:, (2, 1, 0)].contiguous()
        autocast_enabled = (
            self.teacher_device.type == "cuda"
            and self.autocast_dtype in {torch.float16, torch.bfloat16}
        )
        self.teacher.eval()
        with torch.no_grad(), torch.autocast(
            device_type=self.teacher_device.type,
            dtype=self.autocast_dtype,
            enabled=autocast_enabled,
        ):
            output = self.teacher(teacher_input, teacher_input)
        flow = self._extract_flow(output, int(warped.shape[0]))
        # Blur and coordinate conversion are geometry math. Keep them FP32 even
        # when the surrounding unified training step runs under BF16 autocast.
        with torch.autocast(device_type=warped.device.type, enabled=False):
            flow = flow.float()
            if self.blur_kernel > 1:
                flow = gaussian_blur(
                    flow,
                    kernel_size=[self.blur_kernel, self.blur_kernel],
                )
            if native_size != (self.flow_size, self.flow_size):
                flow = resize_backward_flow(
                    flow,
                    native_size,
                    source_size_from=(self.flow_size, self.flow_size),
                    source_size_to=native_size,
                )
        if tuple(flow.shape) != (warped.shape[0], 2, *native_size):
            raise RuntimeError(
                "teacher prior post-processing produced an invalid shape: "
                f"{tuple(flow.shape)}"
            )
        if not bool(torch.isfinite(flow).all()):
            raise RuntimeError("teacher prior post-processing produced non-finite flow")
        return flow


__all__ = ["TorchScriptGeometryPrior"]
