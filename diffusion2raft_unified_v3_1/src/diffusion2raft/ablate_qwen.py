"""Evaluate one checkpoint with both Qwen paths enabled versus disabled.

This is a controlled diagnostic, not a recommendation to remove Qwen.  The
checkpoint's saved config and training-time correlation temperature are
authoritative; the CLI config contributes only the requested runtime device.
Use :mod:`diffusion2raft.ablate_residual_qwen` for the full matching/context
split and residual-strength sweep.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import DocumentFlowDataset
from .losses import RectificationLoss
from .models import build_rectifier
from .train import _evaluate, _raw_model


def _build_loader(config: dict[str, Any], manifest: str) -> DataLoader:
    dataset = DocumentFlowDataset(
        manifest, tuple(config["data"]["work_size"]), augment_guide=False
    )
    return DataLoader(
        dataset,
        batch_size=int(config["data"].get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )


# Metrics worth comparing across the on/off runs (keys must match losses.py).
_KEYS = (
    "epe",
    "epe_p95",
    "line_epe",
    "line_straightness_error",
    "edge_epe",
    "prior_epe",
    "epe_gain",
    "final_win_rate",
    "fold_rate",
    "residual_epe",
)


def run(config: dict[str, Any], checkpoint: str, *, max_batches: int | None) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve(strict=True)
    before = checkpoint_path.stat()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError("checkpoint is missing its authoritative config")
    runtime_device = str(
        config.get("device", checkpoint_config.get("device", "cuda"))
    )
    config = copy.deepcopy(checkpoint_config)
    config["device"] = runtime_device
    device = torch.device(runtime_device)
    model = build_rectifier(
        dict(config["model"]),
        dict(config.get("qwen", {})),
        stage="unified",
        device=device,
    ).to(device)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only Qwen-backbone (non-registered) keys may be absent; heads must match.
    bad = [k for k in missing if not k.startswith("diffusion_encoder._pipeline")]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch; missing={bad[:8]}, unexpected={unexpected[:8]}")
    # The refiner cost volume is scaled by 1/correlation_temperature, and the
    # unified config ramps that value over the first epochs.  Evaluating a
    # mid-ramp checkpoint at the config target (1.0) feeds the refiner an
    # unseen input scale and makes the Qwen/residual path look far worse than
    # it is.  Recover the exact training-time temperature from the checkpoint.
    from .train import _checkpoint_correlation_temperature

    raw = _raw_model(model)
    temperature = _checkpoint_correlation_temperature(payload, required=True)
    if temperature is None:  # pragma: no cover - required=True is strict.
        raise RuntimeError("checkpoint correlation_temperature is unavailable")
    raw.set_correlation_temperature(float(temperature))
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = checkpoint_path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise RuntimeError(f"checkpoint changed while hashing: {checkpoint_path}")
    print(f"[info] restored training-time correlation_temperature={temperature:.4f} "
          f"(checkpoint epoch={int(payload['epoch']) + 1})")
    criterion = RectificationLoss(config["loss"]).to(device)
    loader = _build_loader(config, config["data"]["val_manifest"])

    results: dict[str, Any] = {}
    for label, qwen_off in (("qwen_on", False), ("qwen_off", True)):
        raw.qwen_off = qwen_off
        metrics = _evaluate(
            model, loader, criterion, device, "unified",
            distributed=False, max_batches=max_batches,
        )
        results[label] = {k: metrics.get(k, float("nan")) for k in _KEYS}
    raw.qwen_off = False
    results["_protocol"] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_epoch_index": int(payload["epoch"]),
        "checkpoint_display_epoch": int(payload["epoch"]) + 1,
        "correlation_temperature": float(temperature),
        "validation_manifest": str(
            Path(config["data"]["val_manifest"]).resolve(strict=True)
        ),
        "validation_samples": len(loader.dataset),
        "max_batches": max_batches,
        "qwen_modes": ["qwen_on", "qwen_off"],
        "config_source": "checkpoint",
        "note": "same single-GPU evaluator; qwen_off disables matching and context",
    }
    return results


def _fmt(results: dict[str, Any]) -> str:
    on, off = results["qwen_on"], results["qwen_off"]
    lines = [f"{'metric':<14}{'qwen_on':>12}{'qwen_off':>12}{'delta(off-on)':>16}"]
    for k in _KEYS:
        d = off[k] - on[k]
        lines.append(f"{k:<14}{on[k]:>12.4f}{off[k]:>12.4f}{d:>+16.4f}")
    verdict = (
        "Qwen-off is not worse: inspect path split/features before changing Qwen."
        if off["epe"] <= on["epe"] + 1e-3
        else "Qwen-on is better: retain Qwen and inspect which path provides the gain."
    )
    lines.append("verdict: " + verdict)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--max-batches", type=int, help="limit val batches for a quick check")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if str(config.get("device", "cuda")).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    results = run(config, args.checkpoint, max_batches=args.max_batches)
    text = _fmt(results)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.name}.tmp.{os.getpid()}"
        )
        temporary.write_text(
            json.dumps(
                results, indent=2, ensure_ascii=False, allow_nan=False
            ),
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
