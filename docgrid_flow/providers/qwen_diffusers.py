"""Online multi-step Q/K probe for Diffusers Qwen-Image-Edit.

Unlike the historical feature source in this repository, this provider never
caches all layers or lets later denoising steps overwrite earlier ones.  A
selected block calls a consumer immediately after its normalized Q and K have
been produced.  The consumer computes metrics synchronously, after which the
only references to the high-dimensional tensors are released.

The implementation intentionally hooks public ``nn.Module`` boundaries
(``transformer``, ``pos_embed``, ``norm_q`` and ``norm_k``) rather than
replacing the Diffusers attention processor.  This preserves the exact model
forward while still allowing Qwen's RoPE to be replayed for the pre/post
comparison.
"""

from __future__ import annotations

import contextlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from PIL import Image
from torch import Tensor, nn

from docgrid_flow.analysis.mmdit_correspondence import stable_sha256


_SUPPORTED_PIPELINES = {
    "QwenImageEditPipeline",
    "QwenImageEditPlusPipeline",
}

# Keep this identical to the module set used by
# scripts/train_exp_A_layer12.sh -> train_flow_head_v4_layer_probe.py.
_LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_out.0",
    "to_add_out",
    "img_mlp.net.2",
    "img_mod.1",
    "txt_mlp.net.2",
    "txt_mod.1",
)


def torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower().replace("torch.", "")
    choices = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError(f"unsupported dtype {name!r}")
    return choices[normalized]


@dataclass(frozen=True)
class LoadedLora:
    path: str
    rank: int
    alpha: float
    tensor_count: int
    target_modules: tuple[str, ...]


def _normalize_lora_state_dict(
    state_dict: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], int, tuple[str, ...]]:
    """Validate a DiffSynth/PEFT LoRA checkpoint and infer its rank/targets."""

    if not state_dict:
        raise ValueError("LoRA checkpoint contains no tensors")
    normalized: dict[str, Tensor] = {}
    ranks: set[int] = set()
    targets: set[str] = set()
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        for prefix in ("module.", "pipe.dit.", "dit.", "transformer."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        key = key.replace(".lora_A.weight", ".lora_A.default.weight")
        key = key.replace(".lora_B.weight", ".lora_B.default.weight")
        if ".lora_A.default.weight" in key:
            marker = ".lora_A.default.weight"
            if value.ndim != 2:
                raise ValueError(f"{raw_key}: LoRA A tensor must be 2-D")
            ranks.add(int(value.shape[0]))
        elif ".lora_B.default.weight" in key:
            marker = ".lora_B.default.weight"
            if value.ndim != 2:
                raise ValueError(f"{raw_key}: LoRA B tensor must be 2-D")
            ranks.add(int(value.shape[1]))
        else:
            raise ValueError(
                f"{raw_key}: checkpoint is not a pure PEFT LoRA state dict"
            )
        module_path = key[: -len(marker)]
        matched = tuple(
            target for target in _LORA_TARGET_MODULES if module_path.endswith(target)
        )
        if len(matched) != 1:
            raise ValueError(
                f"{raw_key}: module does not match the frozen Qwen LoRA targets"
            )
        targets.add(matched[0])
        if key in normalized:
            raise ValueError(f"duplicate normalized LoRA key: {key}")
        normalized[key] = value
    if len(ranks) != 1 or next(iter(ranks)) < 1:
        raise ValueError(f"LoRA checkpoint has inconsistent ranks: {sorted(ranks)}")
    return normalized, next(iter(ranks)), tuple(sorted(targets))


def _load_lora_checkpoint(
    transformer: nn.Module,
    checkpoint: str | Path,
    *,
    alpha: float | None,
) -> tuple[nn.Module, LoadedLora]:
    """Inject and strictly load the same PEFT LoRA used by DiffSynth training."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")
    try:
        from peft import LoraConfig, inject_adapter_in_model
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError(
            "loading a DiffSynth LoRA checkpoint requires peft and safetensors"
        ) from exc

    raw_state = load_file(str(checkpoint_path), device="cpu")
    state_dict, rank, targets = _normalize_lora_state_dict(raw_state)
    if set(targets) != set(_LORA_TARGET_MODULES):
        missing = sorted(set(_LORA_TARGET_MODULES) - set(targets))
        extra = sorted(set(targets) - set(_LORA_TARGET_MODULES))
        raise ValueError(
            "LoRA checkpoint does not cover the complete training target set: "
            f"missing={missing}, extra={extra}"
        )
    resolved_alpha = float(rank if alpha is None else alpha)
    if not resolved_alpha > 0:
        raise ValueError(f"LoRA alpha must be positive, got {resolved_alpha}")
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=resolved_alpha,
        target_modules=list(targets),
        bias="none",
    )
    adapted = inject_adapter_in_model(lora_config, transformer)
    incompatible = adapted.load_state_dict(state_dict, strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    missing_lora = sorted(
        key
        for key in incompatible.missing_keys
        if ".lora_A." in key or ".lora_B." in key
    )
    injected_keys = {
        key
        for key, _parameter in adapted.named_parameters()
        if ".lora_A." in key or ".lora_B." in key
    }
    absent = sorted(injected_keys - set(state_dict))
    if unexpected or missing_lora or absent or len(injected_keys) != len(state_dict):
        raise RuntimeError(
            "LoRA checkpoint did not load exactly: "
            f"checkpoint={len(state_dict)}, injected={len(injected_keys)}, "
            f"unexpected={unexpected[:8]}, missing={sorted(set(missing_lora + absent))[:8]}"
        )
    info = LoadedLora(
        path=str(checkpoint_path),
        rank=rank,
        alpha=resolved_alpha,
        tensor_count=len(state_dict),
        target_modules=targets,
    )
    return adapted, info


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected scalar tensor, got {tuple(value.shape)}")
        return float(value.detach().float().cpu().item())
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


@dataclass(frozen=True)
class ImageTokenLayout:
    shapes: tuple[tuple[int, int, int], ...]
    offsets: tuple[int, ...]

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return self.shapes[0]

    @property
    def source_shape(self) -> tuple[int, int, int]:
        if len(self.shapes) != 2:
            raise RuntimeError(
                f"the experiment requires exactly one warped condition, got {len(self.shapes) - 1}"
            )
        return self.shapes[1]

    @property
    def target_grid(self) -> tuple[int, int]:
        frames, height, width = self.target_shape
        if frames != 1:
            raise RuntimeError(f"expected one target frame, got {frames}")
        return height, width

    @property
    def source_grid(self) -> tuple[int, int]:
        frames, height, width = self.source_shape
        if frames != 1:
            raise RuntimeError(f"expected one source frame, got {frames}")
        return height, width

    @property
    def total_tokens(self) -> int:
        return self.offsets[-1]

    def segment(self, tensor: Tensor, index: int) -> Tensor:
        return tensor[:, self.offsets[index] : self.offsets[index + 1]]

    def frequencies(self, tensor: Tensor, index: int) -> Tensor:
        return tensor[self.offsets[index] : self.offsets[index + 1]]

    @classmethod
    def parse(cls, value: Any) -> "ImageTokenLayout":
        # Diffusers uses [[target_shape, source_shape]] for batch size one.
        shapes_value = value
        if isinstance(shapes_value, (list, tuple)) and len(shapes_value) == 1:
            first = shapes_value[0]
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
                shapes_value = first
        if not isinstance(shapes_value, (list, tuple)) or not shapes_value:
            raise RuntimeError(f"unexpected img_shapes={value!r}")
        shapes: list[tuple[int, int, int]] = []
        for shape in shapes_value:
            if not isinstance(shape, (list, tuple)) or len(shape) != 3:
                raise RuntimeError(f"invalid image token shape {shape!r}")
            normalized = tuple(int(item) for item in shape)
            if min(normalized) < 1:
                raise RuntimeError(f"image token shape must be positive: {normalized}")
            shapes.append(normalized)
        if len(shapes) != 2:
            raise RuntimeError(
                "MMDiT correspondence exp1 accepts one target and exactly one warped-source "
                f"segment, but img_shapes={shapes}"
            )
        offsets = [0]
        for frames, height, width in shapes:
            offsets.append(offsets[-1] + frames * height * width)
        return cls(tuple(shapes), tuple(offsets))


@dataclass(frozen=True)
class FeatureMetadata:
    layer: int
    step_index: int
    branch: str
    scheduler_timestep: float | None
    transformer_timestep: float | None
    sigma: float | None
    rope_state: str
    layout: ImageTokenLayout


class FeatureConsumer(Protocol):
    capture_current_query: bool

    def on_layout(self, layout: ImageTokenLayout) -> None:
        ...

    def on_feature(
        self,
        metadata: FeatureMetadata,
        query_target: Tensor | None,
        key_source: Tensor,
    ) -> None:
        ...

    def on_step_end(self, trace: Mapping[str, Any]) -> None:
        ...


def apply_qwen_rotary(x: Tensor, frequencies: Tensor) -> Tensor:
    """Replay Qwen's complex RoPE on ``[B,S,H,D]`` Q/K tensors."""

    if x.ndim != 4:
        raise ValueError(f"Qwen Q/K must be [B,S,H,D], got {tuple(x.shape)}")
    if x.shape[-1] % 2:
        raise ValueError("Qwen attention head dimension must be even for complex RoPE")
    if frequencies.ndim != 2 or frequencies.shape[0] != x.shape[1]:
        raise ValueError(
            f"RoPE frequencies {tuple(frequencies.shape)} do not match Q/K {tuple(x.shape)}"
        )
    complex_x = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    )
    frequencies = frequencies.to(device=x.device)
    if not torch.is_complex(frequencies):
        # A small number of Diffusers revisions expose (cos,sin) elsewhere,
        # but Qwen's pos_embed output is normally complex.  Supporting a final
        # real/imag axis here makes the failure mode explicit and testable.
        if frequencies.shape[-1] == x.shape[-1]:
            frequencies = torch.view_as_complex(
                frequencies.float().reshape(frequencies.shape[0], -1, 2)
            )
        else:
            raise ValueError("Qwen RoPE frequencies must be complex")
    if frequencies.shape[-1] != complex_x.shape[-1]:
        raise ValueError(
            f"RoPE feature width {frequencies.shape[-1]} != Q/K complex width {complex_x.shape[-1]}"
        )
    rotated = complex_x * frequencies.unsqueeze(1)
    return torch.view_as_real(rotated).flatten(-2).to(dtype=x.dtype)


def _standardize_qk(output: Tensor, total_tokens: int, heads: int) -> Tensor:
    """Normalize supported hook layouts to ``[B,S,H,D]``."""

    if not isinstance(output, Tensor):
        raise RuntimeError(f"Q/K norm hook returned {type(output)!r}, expected Tensor")
    if output.ndim == 4:
        if output.shape[1] == total_tokens:
            result = output
        elif output.shape[2] == total_tokens:
            result = output.transpose(1, 2)
        else:
            raise RuntimeError(
                f"cannot locate token axis ({total_tokens}) in Q/K shape {tuple(output.shape)}"
            )
        if result.shape[2] != heads:
            raise RuntimeError(
                f"Q/K head axis {result.shape[2]} differs from attention.heads={heads}"
            )
        return result
    if output.ndim == 3 and output.shape[1] == total_tokens:
        if output.shape[2] % heads:
            raise RuntimeError(
                f"flattened Q/K width {output.shape[2]} is not divisible by heads={heads}"
            )
        return output.unflatten(-1, (heads, output.shape[2] // heads))
    raise RuntimeError(
        f"unexpected Q/K hook shape {tuple(output.shape)} for {total_tokens} tokens"
    )


class DiffusersQwenCorrespondenceProbe:
    """Load one frozen Qwen pipeline and stream selected Q/K to a consumer."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        selections: Mapping[int, Mapping[int, Sequence[str]]] | None = None,
        selection_builder: Callable[
            [int], Mapping[int, Mapping[int, Sequence[str]]]
        ]
        | None = None,
    ) -> None:
        self.config = dict(config)
        self.device = torch.device(device)
        if (selections is None) == (selection_builder is None):
            raise ValueError("provide exactly one of selections or selection_builder")
        self._loaded_lora: LoadedLora | None = None
        self.pipeline = self._load_pipeline()
        self.block_count = len(self.pipeline.transformer.transformer_blocks)
        raw_selections = (
            selection_builder(self.block_count)
            if selection_builder is not None
            else selections
        )
        assert raw_selections is not None
        self.selections = {
            int(step): {
                int(layer): frozenset(str(state).lower() for state in states)
                for layer, states in layers.items()
            }
            for step, layers in raw_selections.items()
        }
        invalid_states = {
            state
            for layers in self.selections.values()
            for states in layers.values()
            for state in states
            if state not in {"pre", "post"}
        }
        if invalid_states:
            raise ValueError(f"invalid RoPE states: {sorted(invalid_states)}")
        requested_layers = {
            layer for layers in self.selections.values() for layer in layers
        }
        if any(layer < 0 or layer >= self.block_count for layer in requested_layers):
            raise ValueError(
                f"selected layers {sorted(requested_layers)} invalid for {self.block_count} blocks"
            )
        self._consumer: FeatureConsumer | None = None
        self._step_index = 0
        self._calls_in_step = 0
        self._branch = "unknown"
        self._active = False
        self._layout: ImageTokenLayout | None = None
        self._image_frequencies: Tensor | None = None
        self._transformer_timestep: float | None = None
        self._queries: dict[int, Tensor] = {}
        self._handles: list[Any] = []
        self._scheduler_trace: list[dict[str, Any]] = []
        self._register_hooks(requested_layers)

    def _load_pipeline(self) -> Any:
        pipeline_class = str(
            self.config.get("pipeline_class", "QwenImageEditPlusPipeline")
        )
        if pipeline_class not in _SUPPORTED_PIPELINES:
            raise ValueError(
                f"unsupported Qwen pipeline_class={pipeline_class!r}; "
                f"choose one of {sorted(_SUPPORTED_PIPELINES)}"
            )
        try:
            import diffusers

            pipeline_type = getattr(diffusers, pipeline_class)
        except (ImportError, AttributeError) as exc:  # pragma: no cover - production image
            raise RuntimeError(
                f"the runtime Diffusers build does not contain {pipeline_class}"
            ) from exc
        model_id = str(self.config.get("model_id", "Qwen/Qwen-Image-Edit-2511"))
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype(str(self.config.get("dtype", "bfloat16")))
        }
        revision = self.config.get("revision")
        if revision not in (None, ""):
            load_kwargs["revision"] = str(revision)
        if bool(self.config.get("local_files_only", False)):
            load_kwargs["local_files_only"] = True
        pipeline = pipeline_type.from_pretrained(model_id, **load_kwargs)
        lora_checkpoint = self.config.get("lora_checkpoint")
        if lora_checkpoint not in (None, ""):
            pipeline.transformer, self._loaded_lora = _load_lora_checkpoint(
                pipeline.transformer,
                str(lora_checkpoint),
                alpha=(
                    None
                    if self.config.get("lora_alpha") in (None, "")
                    else float(self.config["lora_alpha"])
                ),
            )
        if bool(self.config.get("cpu_offload", False)):
            if self.device.type != "cuda":
                raise ValueError("cpu_offload is only meaningful for a CUDA probe")
            pipeline.enable_model_cpu_offload(gpu_id=self.device.index or 0)
        else:
            pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=bool(self.config.get("disable_progress", True)))
        for name in ("transformer", "vae", "text_encoder"):
            component = getattr(pipeline, name, None)
            if component is not None:
                component.requires_grad_(False)
                component.eval()
        if not hasattr(pipeline, "_callback_tensor_inputs"):
            raise RuntimeError("installed Qwen pipeline lacks callback_on_step_end support")
        return pipeline

    @property
    def scheduler_trace(self) -> list[dict[str, Any]]:
        return list(self._scheduler_trace)

    def fingerprint(self) -> dict[str, Any]:
        transformer = self.pipeline.transformer
        scheduler_config = _jsonable(getattr(self.pipeline.scheduler, "config", {}))
        model_config = _jsonable(getattr(transformer, "config", {}))
        commit_candidates = []
        for component in (self.pipeline, transformer, getattr(self.pipeline, "text_encoder", None)):
            component_config = getattr(component, "config", None)
            if component_config is None:
                continue
            commit = (
                component_config.get("_commit_hash")
                if hasattr(component_config, "get")
                else getattr(component_config, "_commit_hash", None)
            )
            if commit:
                commit_candidates.append(str(commit))
        return {
            "model_id": str(self.config.get("model_id")),
            "revision": self.config.get("revision"),
            "lora_checkpoint": (
                None if self._loaded_lora is None else self._loaded_lora.path
            ),
            "lora_checkpoint_sha256": self.config.get("lora_checkpoint_sha256"),
            "lora_rank": None if self._loaded_lora is None else self._loaded_lora.rank,
            "lora_alpha": None if self._loaded_lora is None else self._loaded_lora.alpha,
            "lora_tensor_count": (
                None if self._loaded_lora is None else self._loaded_lora.tensor_count
            ),
            "lora_target_modules": (
                [] if self._loaded_lora is None else list(self._loaded_lora.target_modules)
            ),
            "resolved_model_commit_candidates": sorted(set(commit_candidates)),
            "pipeline_class": self.pipeline.__class__.__name__,
            "transformer_class": transformer.__class__.__name__,
            "block_count": self.block_count,
            "inner_dim": int(getattr(transformer, "inner_dim", -1)),
            "scheduler_class": self.pipeline.scheduler.__class__.__name__,
            "scheduler_config": scheduler_config,
            "scheduler_config_sha256": stable_sha256(scheduler_config),
            "transformer_config": model_config,
            "transformer_config_sha256": stable_sha256(model_config),
            "dtype": str(getattr(transformer, "dtype", "unknown")),
            "device": str(self.device),
            "qk_capture": "post-QK-Norm; pre/post-Qwen-RoPE",
            "token_segmentation": "runtime img_shapes",
        }

    def _register_hooks(self, requested_layers: set[int]) -> None:
        transformer = self.pipeline.transformer
        try:
            handle = transformer.register_forward_pre_hook(
                self._transformer_pre_hook, with_kwargs=True
            )
        except TypeError as exc:  # pragma: no cover - obsolete torch
            raise RuntimeError(
                "the probe requires torch register_forward_pre_hook(..., with_kwargs=True)"
            ) from exc
        self._handles.append(handle)
        self._handles.append(transformer.pos_embed.register_forward_hook(self._rope_hook))
        for layer in sorted(requested_layers):
            block = transformer.transformer_blocks[layer]
            attention = block.attn
            if attention.norm_q is None or attention.norm_k is None:
                raise RuntimeError(
                    f"block {layer} has no QK norm modules; the preregistered "
                    "post-QK-Norm probe cannot be reproduced"
                )
            self._handles.append(
                attention.norm_q.register_forward_hook(self._make_q_hook(layer, int(attention.heads)))
            )
            self._handles.append(
                attention.norm_k.register_forward_hook(self._make_k_hook(layer, int(attention.heads)))
            )

    def _transformer_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del args
        self._branch = "conditional" if self._calls_in_step == 0 else "negative"
        self._calls_in_step += 1
        self._active = (
            self._branch == "conditional" and self._step_index in self.selections
        )
        self._queries.clear()
        self._image_frequencies = None
        timestep_value = kwargs.get("timestep")
        if isinstance(timestep_value, Tensor) and timestep_value.numel() > 0:
            self._transformer_timestep = float(
                timestep_value.detach().reshape(-1)[0].float().cpu().item()
            )
        else:
            self._transformer_timestep = None
        img_shapes = kwargs.get("img_shapes")
        if img_shapes is None:
            raise RuntimeError(
                "Qwen transformer forward did not expose img_shapes; exact target/source "
                "token segmentation is impossible"
            )
        layout = ImageTokenLayout.parse(img_shapes)
        self._layout = layout
        if self._consumer is not None and self._active:
            self._consumer.on_layout(layout)

    def _rope_hook(
        self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        if not self._active:
            return
        if not isinstance(output, (tuple, list)) or not output:
            raise RuntimeError("Qwen pos_embed hook did not return (image_freqs,text_freqs)")
        image_frequencies = output[0]
        if not isinstance(image_frequencies, Tensor):
            raise RuntimeError("Qwen image RoPE frequencies are not a tensor")
        if self._layout is None or image_frequencies.shape[0] != self._layout.total_tokens:
            raise RuntimeError(
                f"RoPE token count {image_frequencies.shape[0]} != img_shapes total "
                f"{None if self._layout is None else self._layout.total_tokens}"
            )
        self._image_frequencies = image_frequencies

    def _make_q_hook(self, layer: int, heads: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
            if not self._active or layer not in self.selections.get(self._step_index, {}):
                return
            if self._consumer is None or not self._consumer.capture_current_query:
                return
            if self._layout is None:
                raise RuntimeError("Q hook fired before token layout was captured")
            query = _standardize_qk(output, self._layout.total_tokens, heads)
            if query.shape[0] != 1:
                raise RuntimeError(f"Qwen edit probe requires batch=1, got {query.shape[0]}")
            # Keep only the target segment between the immediately adjacent Q/K
            # module calls.  The full all-image query tensor is not retained.
            self._queries[layer] = self._layout.segment(query, 0)

        return hook

    def _make_k_hook(self, layer: int, heads: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
            states = self.selections.get(self._step_index, {}).get(layer)
            if not self._active or not states:
                return
            if self._consumer is None or self._layout is None:
                raise RuntimeError("K hook fired without an active consumer/layout")
            key = _standardize_qk(output, self._layout.total_tokens, heads)
            if key.shape[0] != 1:
                raise RuntimeError(f"Qwen edit probe requires batch=1, got {key.shape[0]}")
            query_target = self._queries.pop(layer, None)
            if self._consumer.capture_current_query and query_target is None:
                raise RuntimeError(f"layer {layer} K hook fired without its target Q")
            key_source = self._layout.segment(key, 1)
            transformer_timestep = self._transformer_timestep
            scheduler_timestep = _scalar(getattr(self.pipeline, "_current_timestep", None))
            sigma = self._sigma_at(self._step_index)
            if "pre" in states:
                metadata = FeatureMetadata(
                    layer=layer,
                    step_index=self._step_index,
                    branch=self._branch,
                    scheduler_timestep=scheduler_timestep,
                    transformer_timestep=transformer_timestep,
                    sigma=sigma,
                    rope_state="pre",
                    layout=self._layout,
                )
                self._consumer.on_feature(
                    metadata,
                    None
                    if query_target is None
                    else query_target.reshape(query_target.shape[1], -1),
                    key_source.reshape(key_source.shape[1], -1),
                )
            if "post" in states:
                if self._image_frequencies is None:
                    raise RuntimeError("post-RoPE requested but pos_embed frequencies were not captured")
                query_post = (
                    None
                    if query_target is None
                    else apply_qwen_rotary(
                        query_target, self._layout.frequencies(self._image_frequencies, 0)
                    )
                )
                key_post = apply_qwen_rotary(
                    key_source, self._layout.frequencies(self._image_frequencies, 1)
                )
                metadata = FeatureMetadata(
                    layer=layer,
                    step_index=self._step_index,
                    branch=self._branch,
                    scheduler_timestep=scheduler_timestep,
                    transformer_timestep=transformer_timestep,
                    sigma=sigma,
                    rope_state="post",
                    layout=self._layout,
                )
                self._consumer.on_feature(
                    metadata,
                    None if query_post is None else query_post.reshape(query_post.shape[1], -1),
                    key_post.reshape(key_post.shape[1], -1),
                )
            del key_source, key, query_target

        return hook

    def _sigma_at(self, step_index: int) -> float | None:
        sigmas = getattr(self.pipeline.scheduler, "sigmas", None)
        if sigmas is None or step_index >= len(sigmas):
            return None
        return _scalar(sigmas[step_index])

    def _step_callback(
        self, _pipeline: Any, step_index: int, timestep: Tensor, callback_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        if int(step_index) != self._step_index:
            raise RuntimeError(
                f"callback step={step_index} but hook state expected {self._step_index}"
            )
        trace = {
            "step_index": int(step_index),
            "scheduler_timestep": _scalar(timestep),
            "sigma": self._sigma_at(int(step_index)),
            "transformer_forwards": int(self._calls_in_step),
            "branches": ["conditional"]
            + (["negative"] if self._calls_in_step > 1 else []),
        }
        self._scheduler_trace.append(trace)
        if self._consumer is not None:
            self._consumer.on_step_end(trace)
        self._queries.clear()
        self._image_frequencies = None
        self._transformer_timestep = None
        self._layout = None
        self._active = False
        self._calls_in_step = 0
        self._step_index = int(step_index) + 1
        return callback_kwargs

    def run_sample(
        self,
        image: Image.Image,
        *,
        seed: int,
        consumer: FeatureConsumer,
    ) -> None:
        if self._consumer is not None:
            raise RuntimeError("probe is already processing a sample")
        self._consumer = consumer
        self._step_index = 0
        self._calls_in_step = 0
        self._scheduler_trace = []
        execution_device = getattr(self.pipeline, "_execution_device", self.device)
        generator_device = execution_device if str(execution_device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        height = int(self.config.get("height", 512))
        width = int(self.config.get("width", 512))
        if image.size != (width, height):
            raise ValueError(
                f"probe input must already match {(width, height)}, got {image.size}; "
                "GT and image preprocessing must share one transform"
            )
        kwargs: dict[str, Any] = {
            "image": (
                [image.convert("RGB")]
                if self.pipeline.__class__.__name__ == "QwenImageEditPlusPipeline"
                else image.convert("RGB")
            ),
            "prompt": str(self.config["prompt"]),
            "height": height,
            "width": width,
            "generator": generator,
            "num_inference_steps": int(self.config.get("num_inference_steps", 50)),
            "true_cfg_scale": float(self.config.get("true_cfg_scale", 1.0)),
            "num_images_per_prompt": 1,
            "output_type": "latent",
            "callback_on_step_end": self._step_callback,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
        negative_prompt = self.config.get("negative_prompt")
        if negative_prompt is not None:
            kwargs["negative_prompt"] = str(negative_prompt)
        guidance_scale = self.config.get("guidance_scale")
        if guidance_scale is not None:
            kwargs["guidance_scale"] = float(guidance_scale)
        max_length = self.config.get("max_sequence_length")
        if max_length is not None:
            kwargs["max_sequence_length"] = int(max_length)
        signature = inspect.signature(self.pipeline.__call__).parameters
        unsupported = sorted(key for key in kwargs if key not in signature)
        if unsupported:
            raise RuntimeError(
                f"installed {self.pipeline.__class__.__name__} lacks required arguments {unsupported}"
            )
        try:
            with torch.inference_mode():
                output = self.pipeline(**kwargs)
            latent = getattr(output, "images", None)
            if latent is not None and not isinstance(latent, Tensor):
                raise RuntimeError(
                    "output_type='latent' unexpectedly returned decoded/non-tensor images; "
                    "the experiment forbids VAE Decoder output"
                )
            if len(self._scheduler_trace) != int(kwargs["num_inference_steps"]):
                raise RuntimeError(
                    f"captured {len(self._scheduler_trace)} step callbacks, expected "
                    f"{kwargs['num_inference_steps']}"
                )
            del output, latent
        finally:
            self._queries.clear()
            self._image_frequencies = None
            self._transformer_timestep = None
            self._layout = None
            self._active = False
            self._consumer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self) -> None:
        for handle in self._handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self._handles.clear()

    def __enter__(self) -> "DiffusersQwenCorrespondenceProbe":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def build_selections(
    *,
    block_count: int,
    steps: Sequence[int],
    layers: Sequence[int] | str,
    rope_states: Sequence[str],
) -> dict[int, dict[int, tuple[str, ...]]]:
    if layers == "all":
        resolved_layers = tuple(range(int(block_count)))
    else:
        resolved_layers = []
        for value in layers:
            index = int(value)
            resolved_layers.append(index if index >= 0 else int(block_count) + index)
        resolved_layers = tuple(sorted(set(resolved_layers)))
    return {
        int(step): {int(layer): tuple(rope_states) for layer in resolved_layers}
        for step in steps
    }
