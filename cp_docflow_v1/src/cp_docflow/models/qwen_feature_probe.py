"""Canonical frozen Qwen feature probe and trainable geometry adapter.

The pretrained pipeline is deliberately kept outside ``nn.Module`` state: a
CP-DocFlow checkpoint contains only the small DPT/FPN adapter, never another
copy of the 20B Qwen weights.  The pipeline is always requested with
``output_type='latent'`` and its latent result is discarded, so the VAE
decoder cannot become a source of document pixels.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from .coarse import ConvNormAct, ResidualBlock


def _torch_dtype(name: str) -> torch.dtype:
    choices = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    normalized = str(name).lower().replace("torch.", "")
    if normalized not in choices:
        raise ValueError(f"unsupported Qwen dtype {name!r}")
    return choices[normalized]


def _factor_grid(token_count: int, aspect_ratio: float) -> tuple[int, int]:
    if token_count < 1:
        raise ValueError("Qwen source token count must be positive")
    best = (1, token_count)
    error_best = float("inf")
    for factor in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % factor:
            continue
        other = token_count // factor
        for height, width in ((factor, other), (other, factor)):
            error = abs(width / max(height, 1) - aspect_ratio)
            if error < error_best:
                best, error_best = (height, width), error
    return best


class LiteQwenFeatureSource(nn.Module):
    """Small source-only stand-in that preserves the production interface."""

    def __init__(self, hidden_channels: int = 32, layers: int = 3) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        channels = int(hidden_channels)
        self.hidden_channels = channels
        self.layer_count = int(layers)
        self.stem = nn.Sequential(
            ConvNormAct(3, channels, kernel_size=5, stride=2),
            ResidualBlock(channels),
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(channels, channels, stride=2),
                    ResidualBlock(channels),
                )
                for _ in range(max(2, layers))
            ]
        )

    def forward(self, warped_image: Tensor) -> list[tuple[Tensor, Tensor]]:
        feature = self.stem(warped_image)
        levels: list[Tensor] = []
        for block in self.down:
            feature = block(feature)
            levels.append(feature)
        selected = levels[-self.layer_count :]
        # Qwen has target-denoising and source-condition tokens.  The lite
        # backend derives both from the warped image solely to exercise the
        # exact same adapter/fusion graph in unit tests.
        return [(item, item) for item in selected]


class FrozenQwenImageEditFeatureSource(nn.Module):
    """Capture frozen Qwen hidden/QK tokens without decoding an edited image."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.model_id = str(self.config["model_id"])
        self.prompt = str(
            self.config.get(
                "prompt",
                "Rectify the photographed document geometry only. Preserve all content.",
            )
        )
        self.feature_type = str(self.config.get("feature_type", "hidden")).lower()
        if self.feature_type not in {"hidden", "qk"}:
            raise ValueError("qwen.feature_type must be hidden or qk")
        requested = self.config.get("feature_layers", [-24, -12, -1])
        self.requested_layers = tuple(int(value) for value in requested)
        if not self.requested_layers:
            raise ValueError("qwen.feature_layers must not be empty")
        self.num_inference_steps = int(
            self.config.get("feature_num_inference_steps", 4)
        )
        if self.num_inference_steps < 1:
            raise ValueError("qwen.feature_num_inference_steps must be positive")
        self.guidance_scale = float(self.config.get("guidance_scale", 1.0))
        self.seed = int(self.config.get("feature_seed", 0))
        self.hidden_channels = int(self.config.get("hidden_channels", 3072))
        self.layer_count = len(self.requested_layers)
        object.__setattr__(self, "_pipeline", None)
        object.__setattr__(self, "_layer_indices", None)

    @property
    def pipeline(self) -> Any | None:
        return object.__getattribute__(self, "_pipeline")

    def _ensure_pipeline(self, device: torch.device) -> Any:
        existing = self.pipeline
        if existing is not None:
            return existing
        try:
            from diffusers import QwenImageEditPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Qwen conditioning needs a diffusers build containing "
                "QwenImageEditPipeline"
            ) from exc
        try:
            from diffusers import QwenImageEditPlusPipeline
        except ImportError:
            QwenImageEditPlusPipeline = None

        wants_plus = any(
            tag in self.model_id.lower() for tag in ("2509", "2511", "plus")
        )
        if wants_plus and QwenImageEditPlusPipeline is None:
            raise RuntimeError(
                f"{self.model_id!r} requires QwenImageEditPlusPipeline; update diffusers"
            )
        pipeline_class = (
            QwenImageEditPlusPipeline if wants_plus else QwenImageEditPipeline
        )
        dtype_default = "bfloat16" if device.type == "cuda" else "float32"
        load_kwargs: dict[str, Any] = {
            "torch_dtype": _torch_dtype(
                str(self.config.get("feature_dtype", dtype_default))
            )
        }
        if bool(self.config.get("local_files_only", True)):
            load_kwargs["local_files_only"] = True
        if self.config.get("revision") is not None:
            load_kwargs["revision"] = str(self.config["revision"])
        quantization = str(self.config.get("feature_quantization", "none")).lower()
        if quantization not in {"none", "4bit", "8bit"}:
            raise ValueError("qwen.feature_quantization must be none, 4bit, or 8bit")
        if quantization != "none":
            try:
                from diffusers.quantizers import PipelineQuantizationConfig
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen 4/8-bit loading needs current diffusers and bitsandbytes"
                ) from exc
            load_kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend=f"bitsandbytes_{quantization}",
                quant_kwargs={f"load_in_{quantization}": True},
                components_to_quantize=list(
                    self.config.get("quantize_components", ["transformer"])
                ),
            )
        device_map = self.config.get("device_map")
        if device_map is not None:
            load_kwargs["device_map"] = device_map
            if self.config.get("device_map_max_memory") is not None:
                load_kwargs["max_memory"] = self.config["device_map_max_memory"]
        pipeline = pipeline_class.from_pretrained(self.model_id, **load_kwargs)
        if device_map is None:
            if bool(self.config.get("cpu_offload", False)) and device.type == "cuda":
                pipeline.enable_model_cpu_offload(gpu_id=device.index or 0)
            else:
                pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        for name in ("transformer", "vae", "text_encoder"):
            component = getattr(pipeline, name, None)
            if component is not None:
                component.requires_grad_(False)
                component.eval()
        block_count = len(pipeline.transformer.transformer_blocks)
        indices = tuple(
            sorted(
                {
                    value if value >= 0 else block_count + value
                    for value in self.requested_layers
                }
            )
        )
        if len(indices) != self.layer_count or any(
            value < 0 or value >= block_count for value in indices
        ):
            raise ValueError(
                f"feature_layers={self.requested_layers} invalid for {block_count} blocks"
            )
        inner_dim = int(getattr(pipeline.transformer, "inner_dim"))
        if inner_dim != self.hidden_channels:
            raise RuntimeError(
                f"configured qwen.hidden_channels={self.hidden_channels}, "
                f"but transformer.inner_dim={inner_dim}"
            )
        object.__setattr__(self, "_pipeline", pipeline)
        object.__setattr__(self, "_layer_indices", indices)
        return pipeline

    def train(self, mode: bool = True) -> "FrozenQwenImageEditFeatureSource":
        super().train(mode)
        if self.pipeline is not None:
            for name in ("transformer", "vae", "text_encoder"):
                component = getattr(self.pipeline, name, None)
                if component is not None:
                    component.eval()
        return self

    @staticmethod
    def _to_pil(image: Tensor) -> Image.Image:
        array = (
            image.detach()
            .float()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .byte()
            .cpu()
            .numpy()
        )
        return Image.fromarray(array, mode="RGB")

    def _capture_one(self, image: Tensor, sample_seed: int) -> list[tuple[Tensor, Tensor]]:
        height, width = (int(value) for value in image.shape[-2:])
        if height % 32 or width % 32:
            raise ValueError(
                f"Qwen feature input must be divisible by 32, got {(height, width)}"
            )
        pipeline = self._ensure_pipeline(image.device)
        indices: tuple[int, ...] = object.__getattribute__(self, "_layer_indices")
        captured: dict[tuple[int, str], Tensor] = {}
        handles: list[Any] = []

        def hidden_hook(index: int):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                if not isinstance(output, (tuple, list)) or len(output) < 2:
                    raise RuntimeError("unexpected Qwen block output schema")
                captured[(index, "hidden")] = output[1]

            return hook

        def qk_hook(index: int, kind: str):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                if not isinstance(output, Tensor) or output.ndim not in {3, 4}:
                    raise RuntimeError(f"unexpected Qwen {kind} tensor")
                captured[(index, kind)] = (
                    output.flatten(2) if output.ndim == 4 else output
                )

            return hook

        for index in indices:
            block = pipeline.transformer.transformer_blocks[index]
            if self.feature_type == "hidden":
                handles.append(block.register_forward_hook(hidden_hook(index)))
            else:
                handles.append(
                    (block.attn.norm_q or block.attn.to_q).register_forward_hook(
                        qk_hook(index, "query")
                    )
                )
                handles.append(
                    (block.attn.norm_k or block.attn.to_k).register_forward_hook(
                        qk_hook(index, "key")
                    )
                )
        execution_device = getattr(pipeline, "_execution_device", image.device)
        generator_device = (
            execution_device if str(execution_device).startswith("cuda") else "cpu"
        )
        generator = torch.Generator(device=generator_device).manual_seed(sample_seed)
        is_plus = pipeline.__class__.__name__.endswith("PlusPipeline")
        kwargs: dict[str, Any] = {
            "image": [self._to_pil(image)] if is_plus else self._to_pil(image),
            "prompt": self.prompt,
            "height": height,
            "width": width,
            "generator": generator,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": 1.0,
            "num_images_per_prompt": 1,
            "output_type": "latent",
        }
        try:
            with torch.no_grad():
                latent_output = pipeline(**kwargs)
                del latent_output
        finally:
            for handle in handles:
                handle.remove()

        vae_scale = int(pipeline.vae_scale_factor)
        target_h, target_w = height // vae_scale // 2, width // vae_scale // 2
        target_count = target_h * target_w
        result: list[tuple[Tensor, Tensor]] = []
        for index in indices:
            if self.feature_type == "hidden":
                packed = captured.get((index, "hidden"))
                if packed is None:
                    raise RuntimeError(f"Qwen block {index} hidden hook did not fire")
                if packed.ndim != 3:
                    raise RuntimeError(
                        f"Qwen block {index} image hidden must be [B,N,C], "
                        f"got {tuple(packed.shape)}"
                    )
                target_tokens, source_tokens = packed[:, :target_count], packed[:, target_count:]
            else:
                query = captured.get((index, "query"))
                key = captured.get((index, "key"))
                if query is None or key is None:
                    raise RuntimeError(f"Qwen block {index} Q/K hooks did not both fire")
                if query.ndim != 3 or key.ndim != 3 or query.shape != key.shape:
                    raise RuntimeError(
                        f"Qwen block {index} flattened Q/K must share [B,N,C], "
                        f"got {tuple(query.shape)} and {tuple(key.shape)}"
                    )
                target_tokens, source_tokens = query[:, :target_count], key[:, target_count:]
            if (
                target_tokens.shape[0] != 1
                or target_tokens.shape[1] != target_count
                or target_tokens.shape[2] != self.hidden_channels
            ):
                raise RuntimeError(
                    f"Qwen block {index} target-token contract mismatch: "
                    f"got {tuple(target_tokens.shape)}, expected "
                    f"(1,{target_count},{self.hidden_channels})"
                )
            if source_tokens.shape[1] < 1:
                raise RuntimeError("Qwen capture contains no source-condition tokens")
            if source_tokens.shape[0] != 1 or source_tokens.shape[2] != self.hidden_channels:
                raise RuntimeError(
                    f"Qwen block {index} source-token contract mismatch: "
                    f"got {tuple(source_tokens.shape)}"
                )
            source_h, source_w = _factor_grid(
                int(source_tokens.shape[1]), width / max(height, 1)
            )
            if source_h * source_w != source_tokens.shape[1]:
                raise RuntimeError("Qwen source-condition tokens cannot form a spatial grid")
            target_map = target_tokens.transpose(1, 2).reshape(
                1, target_tokens.shape[-1], target_h, target_w
            )
            source_map = source_tokens.transpose(1, 2).reshape(
                1, source_tokens.shape[-1], source_h, source_w
            )
            result.append((target_map, source_map))
        return result

    def forward(self, warped_image: Tensor) -> list[tuple[Tensor, Tensor]]:
        batches = [
            self._capture_one(warped_image[index], self.seed + index)
            for index in range(warped_image.shape[0])
        ]
        return [
            (
                torch.cat([batch[layer][0] for batch in batches], dim=0),
                torch.cat([batch[layer][1] for batch in batches], dim=0),
            )
            for layer in range(self.layer_count)
        ]


class QwenDPTFPNAdapter(nn.Module):
    """Project target/source tokens and fuse selected Qwen layers at 1/8."""

    def __init__(
        self,
        input_channels: int,
        feature_channels: int,
        layer_count: int,
    ) -> None:
        super().__init__()
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        self.layer_count = int(layer_count)
        self.target_projections = nn.ModuleList(
            [ConvNormAct(input_channels, feature_channels, kernel_size=1) for _ in range(layer_count)]
        )
        self.source_projections = nn.ModuleList(
            [ConvNormAct(input_channels, feature_channels, kernel_size=1) for _ in range(layer_count)]
        )
        self.pair_fusions = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(2 * feature_channels, feature_channels, kernel_size=1),
                    ResidualBlock(feature_channels),
                )
                for _ in range(layer_count)
            ]
        )
        self.top_down = nn.ModuleList(
            [ResidualBlock(feature_channels) for _ in range(layer_count - 1)]
        )
        self.output = nn.Sequential(
            ConvNormAct(layer_count * feature_channels, feature_channels, kernel_size=1),
            ResidualBlock(feature_channels),
        )

    def forward(
        self,
        features: list[tuple[Tensor, Tensor]],
        output_size: tuple[int, int],
    ) -> Tensor:
        if len(features) != self.layer_count:
            raise ValueError(
                f"expected {self.layer_count} Qwen layers, got {len(features)}"
            )
        projected: list[Tensor] = []
        for index, ((target, source), target_proj, source_proj, fusion) in enumerate(
            zip(
                features,
                self.target_projections,
                self.source_projections,
                self.pair_fusions,
            )
        ):
            parameter = next(target_proj.parameters())
            target = target.to(parameter.device, parameter.dtype)
            source = source.to(parameter.device, parameter.dtype)
            target = F.interpolate(
                target_proj(target), output_size, mode="bilinear", align_corners=False
            )
            source = F.interpolate(
                source_proj(source), output_size, mode="bilinear", align_corners=False
            )
            projected.append(fusion(torch.cat((target, source), dim=1)))
        # Deep-to-shallow top-down propagation followed by DPT-style concat.
        propagated = list(projected)
        for reverse_index, block in zip(
            range(self.layer_count - 2, -1, -1), self.top_down
        ):
            propagated[reverse_index] = block(
                propagated[reverse_index] + propagated[reverse_index + 1]
            )
        return self.output(torch.cat(propagated, dim=1))


def build_qwen_feature_source(
    backend: str,
    config: dict[str, Any] | None,
) -> LiteQwenFeatureSource | FrozenQwenImageEditFeatureSource | None:
    normalized = str(backend).lower()
    values = dict(config or {})
    if normalized == "none":
        return None
    if normalized == "lite":
        return LiteQwenFeatureSource(
            hidden_channels=int(values.get("hidden_channels", 32)),
            layers=len(values.get("feature_layers", [-3, -2, -1])),
        )
    if normalized == "qwen":
        return FrozenQwenImageEditFeatureSource(values)
    raise ValueError("qwen_backend must be one of: none, lite, qwen")
