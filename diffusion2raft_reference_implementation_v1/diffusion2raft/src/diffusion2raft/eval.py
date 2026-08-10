"""Evaluate a checkpoint over a manifest with README's primary metrics.

Reports flow EPE, Jacobian fold rate / near-fold tail, invalid-sampling ratio,
straight-line bending and text-edge reprojection error -- not just PSNR/SSIM.
OCR CER is opt-in (``--ocr``) and only runs when the manifest carries ground
truth text and an OCR engine is installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .config import load_config
from .data import DocumentFlowDataset
from .metrics import (
    aggregate_metrics,
    compute_geometry_metrics,
    ocr_character_error_rate,
)
from .models import build_guided_rectifier


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def evaluate_manifest(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    manifest: str | Path,
    *,
    stage: str = "joint",
    device: str | None = None,
    run_ocr: bool = False,
    output_json: str | Path | None = None,
) -> dict[str, float]:
    requested_device = str(device or config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch_device = torch.device(requested_device)

    work_size = tuple(int(v) for v in config["data"]["work_size"])
    dataset = DocumentFlowDataset(manifest, work_size, augment_guide=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"].get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )

    model_config = dict(config["model"])
    model_config["raft_pretrained"] = False
    model = build_guided_rectifier(model_config, stage=stage)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch; missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
    model.to(torch_device).eval()

    min_jacobian = float(config.get("loss", {}).get("min_jacobian", 0.05))
    per_sample: list[dict[str, float]] = []
    for batch in loader:
        batch = _to_device(batch, torch_device)
        if stage == "joint" and not bool(batch["guide_available"].all()):
            raise RuntimeError("joint-stage evaluation sample is missing a cached guide")
        guide = batch["guide"] if stage == "joint" else None
        outputs = model(batch["warped"], guide, stage=stage)
        final_flow = outputs["final_flow"].float()
        valid = batch["valid"].bool()
        source_size = work_size  # flow already lives on the work canvas
        metrics = compute_geometry_metrics(
            final_flow,
            batch["flow"],
            valid,
            source_size=source_size,
            target_image=batch["target"],
            min_jacobian=min_jacobian,
        )
        if run_ocr:
            from .geometry import backward_warp

            rectified = backward_warp(batch["warped"], final_flow, padding_mode="border")
            cer = ocr_character_error_rate(rectified, batch.get("text"))
            if cer is not None:
                metrics["ocr_cer"] = cer
        per_sample.append(metrics)

    summary = aggregate_metrics(per_sample)
    summary["num_samples"] = float(len(per_sample))
    if output_json is not None:
        Path(output_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", choices=("prior", "joint"), default="joint")
    parser.add_argument("--device")
    parser.add_argument("--ocr", action="store_true", help="run optional OCR CER")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    summary = evaluate_manifest(
        load_config(args.config),
        args.checkpoint,
        args.manifest,
        stage=args.stage,
        device=args.device,
        run_ocr=args.ocr,
        output_json=args.output_json,
    )
    width = max(len(key) for key in summary)
    for key in sorted(summary):
        print(f"{key:<{width}} : {summary[key]:.6f}")


if __name__ == "__main__":
    main()
