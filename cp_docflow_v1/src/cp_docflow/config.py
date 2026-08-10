"""Configuration helpers shared by Stage-1 commands."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models.coarse import DeterministicCoarseRectifier
from .models.docgrid_flow import CPDocFlow


_ENV_REFERENCE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_environment(value: Any) -> Any:
    """Resolve explicit ``${DOCGRID_NAME:-default}`` YAML references.

    Environment reads are deliberately opt-in and only occur for values using
    this syntax.  This lets Slurm jobs point the fixed stage configs at an
    immutable audited dataset without editing the source YAML for every run.
    """

    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        resolved = os.environ.get(name)
        if resolved is not None:
            return resolved
        if default is not None:
            return default
        raise ValueError(
            f"configuration requires environment variable {name}; "
            "provide it or use ${NAME:-default}"
        )

    return _ENV_REFERENCE.sub(replace, value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_config_tree(config_path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if config_path in stack:
        raise ValueError(f"cyclic config inheritance: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    value = dict(value)
    parent_value = value.pop("extends", None)
    if parent_value is None:
        return value
    parent_path = Path(str(parent_value))
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent = _read_config_tree(parent_path.resolve(), (*stack, config_path))
    return _deep_merge(parent, value)


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Compatibility fallback for the original flat ``configs/*.yaml`` layout.
    return config_path.parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = _expand_environment(_read_config_tree(config_path))
    value["_config_path"] = str(config_path)
    value["_project_root"] = str(_find_project_root(config_path))
    return value


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def build_coarse_model(model_config: dict[str, Any]) -> DeterministicCoarseRectifier:
    allowed = {
        "base_channels",
        "feature_channels",
        "max_displacement_ratio",
        "min_log_variance",
        "max_log_variance",
    }
    unknown = set(model_config) - allowed
    if unknown:
        raise ValueError(f"unknown deterministic coarse model keys: {sorted(unknown)}")
    return DeterministicCoarseRectifier(**model_config)


# Explicitly grouped to keep YAML compatibility reviewable against
# ``CPDocFlow.__init__``. Compatibility-only constructor keys remain accepted.
_FULL_MODEL_CONDITIONING_KEYS = {
    "coarse",
    "qwen_backend",
    "qwen",
    "qwen_feature_channels",
    "instantiate_qwen_adapter",
    "fusion_channels",
    "fusion_mode",
    "hv_channels",
    "enable_hv_condition",
    "use_qwen_condition",
}
_FULL_MODEL_FLOW_KEYS = {
    "velocity_hidden_channels",
    "velocity_time_channels",
    "flow_blocks",
    "flow_heads",
    "flow_window_size",
    "flow_global_pool_size",
    "max_velocity_px",
    "minimum_residual_gate",
    "residual_clip_px",
    "sigma_min",
    "sigma_max",
    "composition_uses_confidence",
    "detach_confidence_for_flow",
    "fm_steps",
    "enable_flow_matching",
    "inference_seed",
}
_FULL_MODEL_REFINER_KEYS = {
    "enable_refiner",
    "refiner_hidden_channels",
    "refiner_iterations",
    "refiner_max_step_px",
    "convex_hidden_channels",
    "upsampling_mode",
}
_FULL_MODEL_COMPATIBILITY_KEYS = {
    "anchor_strength",
    "confidence_preserve_strength",
}
FULL_MODEL_ALLOWED_KEYS = frozenset(
    _FULL_MODEL_CONDITIONING_KEYS
    | _FULL_MODEL_FLOW_KEYS
    | _FULL_MODEL_REFINER_KEYS
    | _FULL_MODEL_COMPATIBILITY_KEYS
)


def build_full_model(model_config: dict[str, Any]) -> CPDocFlow:
    unknown = set(model_config) - FULL_MODEL_ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown full CP-DocFlow model keys: {sorted(unknown)}")
    return CPDocFlow(**model_config)
