"""Optional image post-processing for invalid rectification borders.

The historical deployment filled out-of-bounds pixels with a frozen LAMA
TorchScript model after dilating the invalid-flow mask.  This module keeps that
large model external to the owning :class:`torch.nn.Module`, so adding the
post-processor does not copy its weights into checkpoints or DDP state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .external_file import open_stable_external_file


class TorchScriptLamaInpainter(nn.Module):
    """Fill invalid rectification borders with a frozen TorchScript LAMA.

    The public image contract is RGB ``[0, 1]``.  The deployed LAMA consumes
    BGR image tensors and a binary mask on a 512x512 canvas.  Its output is
    converted back to RGB and used only inside the dilated native-resolution
    invalid mask; valid source pixels are returned unchanged.

    The scripted model is intentionally not a registered child module.  It is
    loaded directly on ``device`` and therefore cannot follow a later
    ``inpainter.to(other_device)`` call.  Inputs on any other device fail
    explicitly instead of causing an implicit transfer.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str,
        inpaint_size: int = 512,
        dilation_kernel: int = 11,
        mask_threshold: float = 100.0 / 255.0,
        expected_sha256: str | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = str(checkpoint_path)
        self.inpaint_size = int(inpaint_size)
        self.dilation_kernel = int(dilation_kernel)
        self.mask_threshold = float(mask_threshold)

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())
        self.lama_device = requested_device

        if self.inpaint_size <= 1:
            raise ValueError("LAMA inpaint_size must be greater than one")
        if self.dilation_kernel < 1 or self.dilation_kernel % 2 != 1:
            raise ValueError("LAMA dilation_kernel must be a positive odd integer")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("LAMA mask_threshold must be in [0, 1]")

        with open_stable_external_file(
            self.checkpoint_path,
            expected_sha256=expected_sha256,
            label="TorchScript LAMA",
        ) as opened:
            lama = torch.jit.load(
                opened.load_path, map_location=self.lama_device
            )
            identity = dict(opened.identity)
        self.resolved_checkpoint_path = str(identity["resolved_path"])
        self.checkpoint_size_bytes = int(identity["file_size"])
        self.checkpoint_mtime_ns = int(identity["mtime_ns"])
        self.checkpoint_sha256 = str(identity["sha256"])
        for parameter in lama.parameters():
            parameter.requires_grad_(False)
        lama.eval()
        # Bypass nn.Module.__setattr__: the external model must not appear in
        # parameters(), named_modules(), state_dict(), or DDP broadcasts.
        object.__setattr__(self, "_lama", lama)

    @property
    def lama(self) -> torch.jit.ScriptModule:
        return object.__getattribute__(self, "_lama")

    @property
    def identity(self) -> dict[str, object]:
        return {
            "enabled": True,
            "backend": "torchscript_lama",
            "path": self.resolved_checkpoint_path,
            "size_bytes": self.checkpoint_size_bytes,
            "mtime_ns": self.checkpoint_mtime_ns,
            "sha256": self.checkpoint_sha256,
            "input_size": self.inpaint_size,
            "dilation_kernel": self.dilation_kernel,
        }

    def train(self, mode: bool = True) -> "TorchScriptLamaInpainter":
        super().train(mode)
        self.lama.eval()
        return self

    def _validate_inputs(self, rectified: Tensor, invalid_mask: Tensor) -> None:
        if rectified.ndim != 4 or rectified.shape[1] != 3:
            raise ValueError(
                f"rectified must be [B,3,H,W], got {tuple(rectified.shape)}"
            )
        if not rectified.is_floating_point():
            raise TypeError(f"rectified must be floating point, got {rectified.dtype}")
        if min(int(rectified.shape[-2]), int(rectified.shape[-1])) <= 1:
            raise ValueError(
                "LAMA requires spatial dimensions greater than one, got "
                f"{tuple(rectified.shape[-2:])}"
            )
        if invalid_mask.ndim != 4 or invalid_mask.shape[1] != 1:
            raise ValueError(
                f"invalid_mask must be [B,1,H,W], got {tuple(invalid_mask.shape)}"
            )
        if (
            invalid_mask.shape[0] != rectified.shape[0]
            or invalid_mask.shape[-2:] != rectified.shape[-2:]
        ):
            raise ValueError(
                "invalid_mask must share rectified batch/spatial dimensions; got "
                f"rectified={tuple(rectified.shape)}, mask={tuple(invalid_mask.shape)}"
            )
        if rectified.device != self.lama_device:
            raise RuntimeError(
                "LAMA image/device mismatch: scripted model was loaded on "
                f"{self.lama_device}, image is on {rectified.device}"
            )
        if invalid_mask.device != self.lama_device:
            raise RuntimeError(
                "LAMA mask/device mismatch: scripted model was loaded on "
                f"{self.lama_device}, mask is on {invalid_mask.device}"
            )
        if invalid_mask.is_complex():
            raise TypeError("invalid_mask must be a real-valued or boolean tensor")
        if not bool(torch.isfinite(rectified).all()):
            raise ValueError("rectified contains NaN or infinite values")
        if not bool(torch.isfinite(invalid_mask).all()):
            raise ValueError("invalid_mask contains NaN or infinite values")
        if bool((rectified < 0.0).any()) or bool((rectified > 1.0).any()):
            raise ValueError("rectified must contain values in [0, 1]")
        if bool((invalid_mask < 0).any()) or bool((invalid_mask > 1).any()):
            raise ValueError("invalid_mask must contain values in [0, 1]")

    def _validate_output(self, output: object, batch: int) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError(
                "TorchScript LAMA must return a Tensor, "
                f"got {type(output).__name__}"
            )
        expected = (batch, 3, self.inpaint_size, self.inpaint_size)
        if tuple(output.shape) != expected:
            raise ValueError(
                "TorchScript LAMA returned the wrong output shape; "
                f"expected {expected}, got {tuple(output.shape)}"
            )
        if not output.is_floating_point():
            raise TypeError(f"TorchScript LAMA output must be floating point, got {output.dtype}")
        if output.device != self.lama_device:
            raise RuntimeError(
                "TorchScript LAMA returned output on the wrong device; "
                f"expected {self.lama_device}, got {output.device}"
            )
        if not bool(torch.isfinite(output).all()):
            raise ValueError("TorchScript LAMA returned NaN or infinite values")
        return output

    @staticmethod
    def _require_cv2():
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError(
                "historical LAMA preprocessing requires OpenCV"
            ) from exc
        return cv2

    def forward_with_mask(
        self, rectified: Tensor, invalid_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the inpainted RGB image and exact native composite mask.

        This intentionally reproduces the deployed uint8/OpenCV path: the
        zero-padded warp is truncated to uint8 before OpenCV LINEAR resizing,
        and the resized uint8 mask is thresholded with ``>100``.
        """

        self._validate_inputs(rectified, invalid_mask)
        native_size = tuple(int(value) for value in rectified.shape[-2:])
        cv2 = self._require_cv2()

        # All image and mask preparation is FP32/uint8 even if the caller is
        # inside an AMP region. This matches the historical deployment.
        with torch.no_grad(), torch.autocast(
            device_type=self.lama_device.type, enabled=False
        ):
            image_rgb = rectified.float().clamp(0.0, 1.0)
            binary_invalid = (invalid_mask.float() > 0.5).float()
            dilated_invalid = F.max_pool2d(
                binary_invalid,
                kernel_size=self.dilation_kernel,
                stride=1,
                padding=self.dilation_kernel // 2,
            )

            # np.uint8(float * 255) truncates rather than rounds in the source
            # deployment. A torch uint8 cast has the same non-negative rule.
            native_rgb_u8 = image_rgb.mul(255.0).to(torch.uint8).cpu()
            native_bgr_u8 = native_rgb_u8[:, (2, 1, 0)].contiguous()
            native_mask_u8 = (
                dilated_invalid.mul(255.0).to(torch.uint8).cpu()
            )
            lama_images: list[Tensor] = []
            lama_masks: list[Tensor] = []
            threshold_u8 = int(round(self.mask_threshold * 255.0))
            for image_chw, mask_chw in zip(native_bgr_u8, native_mask_u8):
                image_hwc = image_chw.permute(1, 2, 0).numpy()
                resized_image = cv2.resize(
                    image_hwc,
                    (self.inpaint_size, self.inpaint_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized_mask = cv2.resize(
                    mask_chw[0].numpy(),
                    (self.inpaint_size, self.inpaint_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                lama_images.append(
                    torch.from_numpy(np.ascontiguousarray(resized_image))
                    .permute(2, 0, 1)
                    .float()
                    .div(255.0)
                )
                lama_masks.append(
                    torch.from_numpy(
                        np.ascontiguousarray(resized_mask > threshold_u8)
                    ).unsqueeze(0).float()
                )
            lama_image = torch.stack(lama_images).to(self.lama_device)
            lama_mask = torch.stack(lama_masks).to(self.lama_device)

            self.lama.eval()
            output = self._validate_output(
                self.lama(lama_image, lama_mask), int(rectified.shape[0])
            )
            output = output.detach().float().clamp(0.0, 1.0)
            output = F.interpolate(
                output,
                size=native_size,
                mode="bilinear",
                align_corners=False,
            )
            output_rgb = output[:, (2, 1, 0)].contiguous()
            composite_mask = dilated_invalid.bool()
            quantized_rgb = native_rgb_u8.to(self.lama_device).float().div(255.0)
            composed = torch.where(composite_mask, output_rgb, quantized_rgb)
            return composed.clamp(0.0, 1.0), composite_mask

    def forward(self, rectified: Tensor, invalid_mask: Tensor) -> Tensor:
        composed, _ = self.forward_with_mask(rectified, invalid_mask)
        return composed


__all__ = ["TorchScriptLamaInpainter"]
