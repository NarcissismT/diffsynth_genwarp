"""Unified diffusion-feature/flow model for document rectification.

The important distinction from :mod:`guided_raft` is that no decoded edit RGB
is produced or consumed.  Qwen-Image-Edit runs inside the model and exposes
target (denoising) tokens and source-condition tokens.  A small RAFT-like
recurrent decoder turns those features into a bounded residual backward flow.
The final RGB is always sampled from the original warped image.

Qwen itself is frozen by default.  This is intentional for the first unified
stage: it keeps the 20B backbone stable while the trainable token projectors,
fusion gate, recurrent refiner, and Stage-A prior are optimized in one graph.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from ..geometry import (
    backward_warp,
    compose_backward_flows,
    flow_valid_mask,
    resize_backward_flow,
)
from .prior import DocumentGeometryPrior


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels),
            ConvNormAct(channels, channels, activation=False),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.body(x))


class ImageFeatureEncoder(nn.Module):
    """Shared high-frequency CNN branch at 1/8 resolution."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        c = int(channels)
        self.output_channels = c
        self.net = nn.Sequential(
            ConvNormAct(3, c // 2, kernel_size=5, stride=2),
            ResidualBlock(c // 2),
            ConvNormAct(c // 2, c, stride=2),
            ResidualBlock(c),
            ConvNormAct(c, c, stride=2),
            ResidualBlock(c),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)


class LiteEditFeatureEncoder(nn.Module):
    """Trainable stand-in used only for unit/smoke tests.

    Production runs should set ``feature_backend: qwen``.  Keeping a light
    backend makes it possible to verify the full unified graph without loading
    a 20B model.
    """

    def __init__(self, feature_channels: int = 96) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.source_encoder = ImageFeatureEncoder(self.feature_channels)
        self.target_encoder = ImageFeatureEncoder(self.feature_channels)

    def forward(self, warped: Tensor, prior_rectified: Tensor) -> dict[str, Tensor]:
        return {
            "source": self.source_encoder(warped),
            "target": self.target_encoder(prior_rectified),
        }


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    choices = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in choices:
        raise ValueError(f"unsupported Qwen dtype {name!r}")
    return choices[normalized]


def _factor_grid(token_count: int, aspect_ratio: float) -> tuple[int, int]:
    """Factor a packed token count into the grid closest to an aspect ratio."""

    if token_count <= 0:
        raise ValueError(f"token_count must be positive, got {token_count}")
    best = (1, token_count)
    best_error = float("inf")
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height:
            continue
        width = token_count // height
        for candidate_h, candidate_w in ((height, width), (width, height)):
            error = abs(candidate_w / max(candidate_h, 1) - aspect_ratio)
            if error < best_error:
                best = (candidate_h, candidate_w)
                best_error = error
    return best


class QwenEditFeatureEncoder(nn.Module):
    """Expose Qwen edit tokens without VAE decoding an edited RGB image.

    Diffusers concatenates target denoising latents before condition-image
    latents.  Hooks on selected transformer blocks therefore let us split the
    hidden state into a target grid and a source grid.  Only the small
    projection/fusion layers are registered in this module; the frozen Qwen
    pipeline is an external pretrained component and is not duplicated in each
    rectifier checkpoint.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        feature_channels: int,
        output_stride: int,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        try:
            from diffusers import QwenImageEditPipeline
        except ImportError as exc:
            raise RuntimeError(
                "unified Qwen features require a diffusers build with QwenImageEditPipeline "
                "(diffusers>=0.35); install the project with `pip install -e '.[qwen]'`"
            ) from exc
        # The Plus pipeline only exists in newer diffusers releases. It is only
        # needed for 2509/2511/Plus models; the base Qwen-Image-Edit uses the
        # base pipeline, so a missing Plus class must not break the base path.
        try:
            from diffusers import QwenImageEditPlusPipeline
        except ImportError:
            QwenImageEditPlusPipeline = None

        self.config = dict(config)
        self.feature_channels = int(feature_channels)
        self.output_stride = int(output_stride)
        self.device_name = str(device)
        self.prompt = str(config["prompt"])
        self.num_inference_steps = int(config.get("feature_num_inference_steps", 4))
        self.guidance_scale = float(config.get("guidance_scale", 1.0))
        self.seed = int(config.get("feature_seed", config.get("seed", 0)))
        self.feature_type = str(config.get("feature_type", "hidden")).lower()
        if self.feature_type not in {"hidden", "qk"}:
            raise ValueError("qwen.feature_type must be 'hidden' or 'qk'")

        model_id = str(config.get("model_id", "Qwen/Qwen-Image-Edit-2511"))
        wants_plus = any(tag in model_id.lower() for tag in ("2509", "2511", "plus"))
        if wants_plus and QwenImageEditPlusPipeline is None:
            raise RuntimeError(
                f"model_id={model_id!r} needs QwenImageEditPlusPipeline, which is absent "
                "from the installed diffusers. Install a newer diffusers, or point "
                "qwen.model_id at a base Qwen-Image-Edit model."
            )
        pipeline_cls = QwenImageEditPlusPipeline if wants_plus else QwenImageEditPipeline
        dtype_name = str(
            config.get(
                "feature_dtype",
                "bfloat16" if str(device).startswith("cuda") else "float32",
            )
        )
        load_kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype(dtype_name)}
        quantization = str(config.get("feature_quantization", "none")).lower()
        if quantization not in {"none", "4bit", "8bit"}:
            raise ValueError("qwen.feature_quantization must be none, 4bit, or 8bit")
        if quantization != "none":
            try:
                from diffusers.quantizers import PipelineQuantizationConfig
            except ImportError as exc:
                raise RuntimeError(
                    "pipeline-level Qwen quantization requires a current diffusers "
                    "source build and bitsandbytes"
                ) from exc
            load_kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend=f"bitsandbytes_{quantization}",
                quant_kwargs={
                    f"load_in_{quantization}": True,
                    **(
                        {
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_compute_dtype": _torch_dtype(dtype_name),
                        }
                        if quantization == "4bit"
                        else {}
                    ),
                },
                components_to_quantize=list(
                    config.get("quantize_components", ["transformer"])
                ),
            )
        if config.get("revision") is not None:
            load_kwargs["revision"] = str(config["revision"])
        if bool(config.get("local_files_only", False)):
            load_kwargs["local_files_only"] = True
        pipeline = pipeline_cls.from_pretrained(model_id, **load_kwargs)
        if bool(config.get("cpu_offload", False)) and str(device).startswith("cuda"):
            device_index = torch.device(device).index
            pipeline.enable_model_cpu_offload(gpu_id=0 if device_index is None else device_index)
        else:
            pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        for component_name in ("transformer", "vae", "text_encoder"):
            component = getattr(pipeline, component_name, None)
            if component is not None:
                component.requires_grad_(False)
                component.eval()

        # DiffusionPipeline is not an nn.Module.  Storing it outside Module's
        # registry avoids 20B frozen weights being written into every checkpoint.
        object.__setattr__(self, "_pipeline", pipeline)
        self.is_plus = pipeline_cls is QwenImageEditPlusPipeline
        block_count = len(pipeline.transformer.transformer_blocks)
        requested_layers = config.get("feature_layers", [-24, -12, -1])
        self.layer_indices = tuple(
            sorted(
                {
                    int(index) if int(index) >= 0 else block_count + int(index)
                    for index in requested_layers
                }
            )
        )
        if not self.layer_indices or any(index < 0 or index >= block_count for index in self.layer_indices):
            raise ValueError(
                f"Qwen feature_layers={requested_layers} are invalid for {block_count} blocks"
            )
        hidden_dim = int(getattr(pipeline.transformer, "inner_dim"))
        self.projections = nn.ModuleList(
            [ConvNormAct(hidden_dim, self.feature_channels, kernel_size=1) for _ in self.layer_indices]
        )
        fused_channels = self.feature_channels * len(self.layer_indices)
        self.layer_fusion = nn.Sequential(
            ConvNormAct(fused_channels, self.feature_channels, kernel_size=1),
            ResidualBlock(self.feature_channels),
        )

    @property
    def pipeline(self) -> Any:
        return object.__getattribute__(self, "_pipeline")

    def train(self, mode: bool = True) -> "QwenEditFeatureEncoder":
        super().train(mode)
        # The projection heads follow ``mode``; the pretrained backbone never
        # leaves eval mode, which also makes repeated feature extraction stable.
        for component_name in ("transformer", "vae", "text_encoder"):
            component = getattr(self.pipeline, component_name, None)
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

    def _capture_one(
        self,
        image: Tensor,
        output_size: tuple[int, int],
        *,
        sample_seed: int,
    ) -> tuple[Tensor, Tensor]:
        pipeline = self.pipeline
        height, width = int(image.shape[-2]), int(image.shape[-1])
        if height % 32 or width % 32:
            raise ValueError(
                f"Qwen unified feature input must be divisible by 32, got {(height, width)}"
            )
        captured: dict[tuple[int, str], Tensor] = {}
        handles: list[Any] = []

        def make_hidden_hook(index: int):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                if not isinstance(output, (tuple, list)) or len(output) < 2:
                    raise RuntimeError(
                        "Qwen transformer block output changed; expected "
                        "(text_hidden, image_hidden)"
                    )
                # The hook is called at every denoising step.  Overwriting keeps
                # the last (most rectified) feature without retaining all steps.
                captured[(index, "hidden")] = output[1]

            return hook

        def make_qk_hook(index: int, kind: str):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                if not isinstance(output, Tensor) or output.ndim not in {3, 4}:
                    raise RuntimeError(
                        f"unexpected Qwen {kind} output at layer {index}: "
                        f"{type(output)!r}"
                    )
                # norm_q/norm_k return [B,T,heads,head_dim].  Flattening heads
                # recovers the transformer's inner dimension while retaining
                # Q/K features that are intrinsically correspondence-oriented.
                captured[(index, kind)] = output.flatten(2) if output.ndim == 4 else output

            return hook

        for index in self.layer_indices:
            block = pipeline.transformer.transformer_blocks[index]
            if self.feature_type == "hidden":
                handles.append(block.register_forward_hook(make_hidden_hook(index)))
            else:
                query_module = block.attn.norm_q or block.attn.to_q
                key_module = block.attn.norm_k or block.attn.to_k
                handles.append(
                    query_module.register_forward_hook(make_qk_hook(index, "query"))
                )
                handles.append(
                    key_module.register_forward_hook(make_qk_hook(index, "key"))
                )

        execution_device = getattr(pipeline, "_execution_device", self.device_name)
        generator_device = execution_device if str(execution_device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(sample_seed)
        pil_image = self._to_pil(image)
        kwargs = {
            "image": [pil_image] if self.is_plus else pil_image,
            "prompt": self.prompt,
            "height": height,
            "width": width,
            "generator": generator,
            "num_inference_steps": self.num_inference_steps,
            # No negative prompt is supplied, so true CFG would only add cost.
            "true_cfg_scale": 1.0,
            "guidance_scale": self.guidance_scale,
            "num_images_per_prompt": 1,
            "output_type": "latent",
        }
        try:
            # Qwen stays frozen, but the captured tensors remain ordinary
            # no-grad tensors that can feed trainable projection heads.
            with torch.no_grad():
                pipeline(**kwargs)
        finally:
            for handle in handles:
                handle.remove()

        expected = {
            (index, kind)
            for index in self.layer_indices
            for kind in (("hidden",) if self.feature_type == "hidden" else ("query", "key"))
        }
        if set(captured) != expected:
            missing = sorted(expected - set(captured))
            raise RuntimeError(f"failed to capture Qwen features {missing}")

        vae_scale = int(pipeline.vae_scale_factor)
        target_h = height // vae_scale // 2
        target_w = width // vae_scale // 2
        target_count = target_h * target_w
        projected_target: list[Tensor] = []
        projected_source: list[Tensor] = []
        for projection, index in zip(self.projections, self.layer_indices, strict=True):
            if self.feature_type == "hidden":
                target_source_tokens = captured[(index, "hidden")]
                target_tokens = target_source_tokens[:, :target_count]
                source_tokens = target_source_tokens[:, target_count:]
            else:
                # Target denoising tokens act as queries; source-condition
                # tokens act as keys.  This follows the feature choice that
                # DA-Flow found more correspondence-ready than block outputs.
                query = captured[(index, "query")]
                key = captured[(index, "key")]
                target_tokens = query[:, :target_count]
                source_tokens = key[:, target_count:]
            if target_tokens.ndim != 3 or target_tokens.shape[0] != 1:
                raise RuntimeError(
                    f"unexpected Qwen {self.feature_type} target shape "
                    f"{tuple(target_tokens.shape)}"
                )
            if source_tokens.shape[1] <= 0:
                raise RuntimeError(
                    "Qwen hidden state has no source-condition tokens: "
                    f"target={target_tokens.shape[1]}, source={source_tokens.shape[1]}"
                )
            source_h, source_w = _factor_grid(
                int(source_tokens.shape[1]), width / max(height, 1)
            )
            target_map = target_tokens.transpose(1, 2).reshape(
                1, target_tokens.shape[-1], target_h, target_w
            )
            source_map = source_tokens.transpose(1, 2).reshape(
                1, source_tokens.shape[-1], source_h, source_w
            )
            projection_dtype = next(projection.parameters()).dtype
            target_map = target_map.to(dtype=projection_dtype)
            source_map = source_map.to(dtype=projection_dtype)
            target_map = projection(target_map)
            source_map = projection(source_map)
            target_map = F.interpolate(
                target_map, size=output_size, mode="bilinear", align_corners=True
            )
            source_map = F.interpolate(
                source_map, size=output_size, mode="bilinear", align_corners=True
            )
            projected_target.append(target_map)
            projected_source.append(source_map)

        return (
            self.layer_fusion(torch.cat(projected_target, dim=1)),
            self.layer_fusion(torch.cat(projected_source, dim=1)),
        )

    def forward(self, warped: Tensor, prior_rectified: Tensor) -> dict[str, Tensor]:
        del prior_rectified  # Qwen receives the original source, never a decoded guide.
        output_size = (
            max(1, warped.shape[-2] // self.output_stride),
            max(1, warped.shape[-1] // self.output_stride),
        )
        targets: list[Tensor] = []
        sources: list[Tensor] = []
        # QwenImageEditPlusPipeline currently supports one prompt/image per
        # invocation.  Under torchrun each rank should therefore use batch=1.
        for batch_index in range(warped.shape[0]):
            target, source = self._capture_one(
                warped[batch_index],
                output_size,
                sample_seed=self.seed,
            )
            targets.append(target)
            sources.append(source)
        return {"target": torch.cat(targets), "source": torch.cat(sources)}


class FeatureReliabilityFusion(nn.Module):
    """Fuse semantic Qwen geometry with source-faithful CNN features.

    The confidence gate implements a safe fallback: where Qwen target tokens
    disagree strongly with the source, the reference becomes the current
    pre-rectified CNN feature, encouraging a near-zero residual instead of a
    text-scale hallucination-following flow.
    """

    def __init__(
        self,
        qwen_channels: int,
        cnn_channels: int,
        match_channels: int,
        *,
        shared_qwen_projection: bool = True,
    ) -> None:
        super().__init__()
        self.shared_qwen_projection = bool(shared_qwen_projection)
        self.qwen_source = ConvNormAct(qwen_channels, match_channels, kernel_size=1)
        self.qwen_target = ConvNormAct(qwen_channels, match_channels, kernel_size=1)
        self.cnn_match = ConvNormAct(cnn_channels, match_channels, kernel_size=1)
        self.confidence = nn.Sequential(
            ConvNormAct(3 * match_channels, match_channels, kernel_size=3),
            nn.Conv2d(match_channels, 1, kernel_size=3, padding=1),
        )
        self.context = nn.Sequential(
            ConvNormAct(3 * match_channels + cnn_channels + 2, match_channels),
            ResidualBlock(match_channels),
        )

    def forward(
        self,
        qwen_target: Tensor,
        qwen_source: Tensor,
        cnn_source: Tensor,
        prior_flow_low: Tensor,
        force_fallback: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.shared_qwen_projection:
            # Applying the same effective function to target and source avoids
            # destroying their pretrained similarity with two independently
            # initialized heads.  Both legacy parameter sets are retained and
            # averaged, so v2 unified checkpoints remain exactly loadable.
            target = 0.5 * (
                self.qwen_target(qwen_target) + self.qwen_source(qwen_target)
            )
            source_qwen = 0.5 * (
                self.qwen_target(qwen_source) + self.qwen_source(qwen_source)
            )
        else:
            # Query and key features naturally occupy different domains and
            # can use separate task heads, as in DA-Flow.
            target = self.qwen_target(qwen_target)
            source_qwen = self.qwen_source(qwen_source)
        source_cnn = self.cnn_match(cnn_source)
        qwen_enhanced_source = source_qwen + source_cnn
        difference = torch.abs(target - qwen_enhanced_source)
        raw_confidence = torch.sigmoid(
            self.confidence(torch.cat((target, qwen_enhanced_source, difference), dim=1))
        )
        confidence = raw_confidence
        if force_fallback is not None:
            confidence = confidence * (1.0 - force_fallback.to(confidence.dtype))
        # At zero confidence both sides become the same source-faithful CNN
        # feature, so local correlation has an identity/zero-residual fallback.
        reference = confidence * target + (1.0 - confidence) * source_cnn
        source = confidence * qwen_enhanced_source + (1.0 - confidence) * source_cnn
        reference = F.normalize(reference.float(), dim=1, eps=1e-6).to(reference.dtype)
        source = F.normalize(source.float(), dim=1, eps=1e-6).to(source.dtype)
        qwen_reference = F.normalize(target.float(), dim=1, eps=1e-6).to(target.dtype)
        qwen_source_match = F.normalize(
            source_qwen.float(), dim=1, eps=1e-6
        ).to(source_qwen.dtype)
        context = self.context(
            torch.cat(
                (
                    confidence * target,
                    confidence * source_qwen,
                    source_cnn,
                    cnn_source,
                    prior_flow_low,
                ),
                dim=1,
            )
        )
        return (
            reference,
            source,
            context,
            raw_confidence,
            confidence,
            qwen_reference,
            qwen_source_match,
        )


class ConvGRU(nn.Module):
    def __init__(self, hidden_channels: int, input_channels: int) -> None:
        super().__init__()
        merged = hidden_channels + input_channels
        self.update_gate = nn.Conv2d(merged, hidden_channels, 3, padding=1)
        self.reset_gate = nn.Conv2d(merged, hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(merged, hidden_channels, 3, padding=1)

    def forward(self, hidden: Tensor, value: Tensor) -> Tensor:
        merged = torch.cat((hidden, value), dim=1)
        update = torch.sigmoid(self.update_gate(merged))
        reset = torch.sigmoid(self.reset_gate(merged))
        candidate = torch.tanh(self.candidate(torch.cat((reset * hidden, value), dim=1)))
        return (1.0 - update) * hidden + update * candidate


def _local_correlation(
    reference: Tensor,
    source_at_flow: Tensor,
    radius: int,
    *,
    temperature: float = 0.10,
) -> Tensor:
    batch, channels, height, width = reference.shape
    kernel = 2 * radius + 1
    patches = F.unfold(source_at_flow, kernel_size=kernel, padding=radius)
    patches = patches.view(batch, channels, kernel * kernel, height, width)
    # reference/source are L2-normalized, so their dot product is already a
    # cosine similarity.  Dividing it again by sqrt(C) makes a 96-channel cost
    # volume almost flat.  Temperature scaling restores usable contrast.
    return (patches * reference.unsqueeze(2)).sum(dim=1) / max(float(temperature), 1e-4)


class RecurrentResidualRefiner(nn.Module):
    """Small RAFT-like local-correlation decoder for a bounded residual map."""

    def __init__(
        self,
        feature_channels: int,
        *,
        hidden_channels: int = 128,
        correlation_radius: int = 4,
        correlation_temperature: float = 0.10,
        iterations: int = 6,
        max_residual_px: float = 24.0,
    ) -> None:
        super().__init__()
        self.radius = int(correlation_radius)
        self.correlation_temperature = float(correlation_temperature)
        self.iterations = int(iterations)
        self.max_residual_px = float(max_residual_px)
        correlation_channels = (2 * self.radius + 1) ** 2
        self.hidden_init = nn.Conv2d(feature_channels, hidden_channels, 3, padding=1)
        self.motion = nn.Sequential(
            ConvNormAct(correlation_channels + 2, hidden_channels),
            ResidualBlock(hidden_channels),
        )
        self.gru = ConvGRU(hidden_channels, hidden_channels + feature_channels + 2)
        self.flow_head = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def set_correlation_temperature(self, temperature: float) -> None:
        """Adjust cost-volume contrast without changing checkpoint parameters."""

        self.correlation_temperature = float(temperature)

    def forward(
        self,
        reference: Tensor,
        source: Tensor,
        context: Tensor,
        full_size: tuple[int, int],
    ) -> list[Tensor]:
        batch, _, height, width = reference.shape
        flow = reference.new_zeros((batch, 2, height, width))
        hidden = torch.tanh(self.hidden_init(context))
        full_h, full_w = int(full_size[0]), int(full_size[1])
        limit_x = self.max_residual_px * (width - 1) / max(full_w - 1, 1)
        limit_y = self.max_residual_px * (height - 1) / max(full_h - 1, 1)
        limit = flow.new_tensor((limit_x, limit_y)).view(1, 2, 1, 1).clamp_min(1e-4)
        predictions: list[Tensor] = []
        for _ in range(self.iterations):
            source_at_flow = backward_warp(
                source,
                flow,
                padding_mode="border",
                align_corners=True,
            )
            correlation = _local_correlation(
                reference,
                source_at_flow,
                self.radius,
                temperature=self.correlation_temperature,
            )
            motion = self.motion(torch.cat((correlation, flow), dim=1))
            hidden = self.gru(hidden, torch.cat((motion, context, flow), dim=1))
            candidate = flow + self.flow_head(hidden)
            flow = limit * torch.tanh(candidate / limit)
            predictions.append(
                resize_backward_flow(
                    flow,
                    full_size,
                    source_size_from=(height, width),
                    source_size_to=full_size,
                )
            )
        return predictions


class UnifiedDocumentRectifier(nn.Module):
    """One-input model: warped image -> final backward flow."""

    def __init__(
        self,
        prior: DocumentGeometryPrior,
        diffusion_encoder: nn.Module,
        *,
        feature_channels: int = 96,
        cnn_channels: int = 64,
        hidden_channels: int = 128,
        correlation_radius: int = 4,
        correlation_temperature: float = 0.10,
        match_temperature: float = 0.10,
        iterations: int = 6,
        max_residual_px: float = 24.0,
        feature_stride: int = 8,
        feature_dropout_prob: float = 0.10,
        shared_qwen_projection: bool = True,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.diffusion_encoder = diffusion_encoder
        self.cnn_encoder = ImageFeatureEncoder(cnn_channels)
        self.fusion = FeatureReliabilityFusion(
            feature_channels,
            cnn_channels,
            feature_channels,
            shared_qwen_projection=shared_qwen_projection,
        )
        self.refiner = RecurrentResidualRefiner(
            feature_channels,
            hidden_channels=hidden_channels,
            correlation_radius=correlation_radius,
            correlation_temperature=correlation_temperature,
            iterations=iterations,
            max_residual_px=max_residual_px,
        )
        self.feature_stride = int(feature_stride)
        self.feature_dropout_prob = float(feature_dropout_prob)
        self.correlation_temperature = float(correlation_temperature)
        self.match_temperature = float(match_temperature)

    def set_prior_trainable(self, trainable: bool) -> None:
        self.prior.requires_grad_(trainable)
        self.prior.train(trainable and self.training)

    def set_correlation_temperature(self, temperature: float) -> None:
        self.correlation_temperature = float(temperature)
        self.refiner.set_correlation_temperature(temperature)

    def forward(
        self,
        warped: Tensor,
        guide: Tensor | None = None,
        *,
        stage: str = "unified",
    ) -> dict[str, Any]:
        del guide
        # Coordinate arithmetic stays in FP32 even when feature convolutions
        # run under BF16 autocast. BF16 cannot represent every pixel coordinate
        # on a 768px grid and otherwise introduces quantized flow ripples.
        prior_flow = self.prior(warped).float()
        with torch.autocast(device_type=warped.device.type, enabled=False):
            prior_rectified, prior_valid = backward_warp(
                warped.float(),
                prior_flow,
                padding_mode="border",
                return_valid=True,
            )
        output: dict[str, Any] = {
            "stage": stage,
            "prior_flow": prior_flow,
            "prior_rectified": prior_rectified,
            "prior_valid": prior_valid,
        }
        if stage == "prior":
            output.update(
                flows=[prior_flow],
                residuals=[],
                final_flow=prior_flow,
                final_valid=prior_valid,
            )
            return output
        if stage != "unified":
            raise ValueError(f"UnifiedDocumentRectifier requires stage='unified', got {stage!r}")

        diffusion_features = self.diffusion_encoder(warped, prior_rectified)
        qwen_target = diffusion_features["target"]
        qwen_source = diffusion_features["source"]
        feature_size = tuple(int(v) for v in qwen_target.shape[-2:])
        if qwen_source.shape[-2:] != feature_size:
            qwen_source = F.interpolate(
                qwen_source, size=feature_size, mode="bilinear", align_corners=True
            )

        # Move source-condition tokens into the pre-rectified coordinate frame
        # before matching against Qwen's target denoising tokens.
        with torch.autocast(device_type=warped.device.type, enabled=False):
            prior_flow_low = resize_backward_flow(
                prior_flow,
                feature_size,
                source_size_from=warped.shape[-2:],
                source_size_to=feature_size,
            )
            qwen_source_rectified = backward_warp(
                qwen_source.float(),
                prior_flow_low.float(),
                padding_mode="border",
            )
        cnn_source = self.cnn_encoder(prior_rectified)
        if cnn_source.shape[-2:] != feature_size:
            cnn_source = F.interpolate(
                cnn_source, size=feature_size, mode="bilinear", align_corners=True
            )
        force_fallback = None
        if self.training and self.feature_dropout_prob > 0:
            force_fallback = (
                torch.rand(
                    (warped.shape[0], 1, 1, 1),
                    device=warped.device,
                )
                < self.feature_dropout_prob
            ).to(qwen_target.dtype)
        (
            reference,
            source,
            context,
            raw_confidence,
            effective_confidence,
            qwen_reference,
            qwen_source_match,
        ) = self.fusion(
            qwen_target,
            qwen_source_rectified,
            cnn_source,
            prior_flow_low,
            force_fallback,
        )
        with torch.autocast(device_type=warped.device.type, enabled=False):
            qwen_match_logits = _local_correlation(
                qwen_reference.float(),
                qwen_source_match.float(),
                self.refiner.radius,
                temperature=self.match_temperature,
            )
            residuals = self.refiner(
                reference.float(),
                source.float(),
                context.float(),
                warped.shape[-2:],
            )
            flows = [compose_backward_flows(prior_flow, residual) for residual in residuals]
        final_flow = flows[-1]
        output.update(
            flows=flows,
            residuals=residuals,
            final_flow=final_flow,
            final_valid=flow_valid_mask(final_flow, warped.shape[-2:]),
            # Raw confidence is supervised/calibrated.  Effective confidence
            # additionally includes sample-level Qwen dropout during training.
            feature_confidence=raw_confidence,
            effective_feature_confidence=effective_confidence,
            qwen_match_logits=qwen_match_logits,
            qwen_match_radius=self.refiner.radius,
        )
        return output


def build_unified_rectifier(
    model_config: dict[str, Any],
    qwen_config: dict[str, Any],
    *,
    device: torch.device | str,
) -> UnifiedDocumentRectifier:
    prior = DocumentGeometryPrior(
        base_channels=int(model_config.get("prior_base_channels", 32)),
        max_displacement_ratio=float(model_config.get("prior_max_displacement_ratio", 0.35)),
        control_stride=int(model_config.get("prior_control_stride", 8)),
    )
    feature_channels = int(model_config.get("feature_channels", 96))
    feature_stride = int(model_config.get("feature_stride", 8))
    backend = str(model_config.get("feature_backend", "qwen")).lower()
    if backend == "qwen":
        diffusion_encoder: nn.Module = QwenEditFeatureEncoder(
            qwen_config,
            feature_channels=feature_channels,
            output_stride=feature_stride,
            device=device,
        )
    elif backend == "lite":
        diffusion_encoder = LiteEditFeatureEncoder(feature_channels)
    else:
        raise ValueError(f"feature_backend must be 'qwen' or 'lite', got {backend!r}")
    return UnifiedDocumentRectifier(
        prior,
        diffusion_encoder,
        feature_channels=feature_channels,
        cnn_channels=int(model_config.get("cnn_feature_channels", 64)),
        hidden_channels=int(model_config.get("refiner_hidden_channels", 128)),
        correlation_radius=int(model_config.get("correlation_radius", 4)),
        correlation_temperature=float(model_config.get("correlation_temperature", 0.10)),
        match_temperature=float(model_config.get("match_temperature", 0.10)),
        iterations=int(model_config.get("refiner_iterations", 6)),
        max_residual_px=float(model_config.get("max_residual_px", 24.0)),
        feature_stride=feature_stride,
        feature_dropout_prob=float(model_config.get("feature_dropout_prob", 0.10)),
        shared_qwen_projection=bool(model_config.get("shared_qwen_projection", True)),
    )
