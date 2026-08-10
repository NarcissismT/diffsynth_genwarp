"""Train the Stage-1 deterministic map/confidence baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .checkpoint import checkpoint_payload, file_sha256, load_checkpoint
from .config import build_coarse_model, load_config, project_path
from .data import DocumentMapDataset, assert_document_disjoint
from .losses import CoarseLossWeights, CoarseRectificationLoss
from .metrics import cell_valid_mask, endpoint_error_map, jacobian_determinant


def _parse_size(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [height,width]")
    result = (int(value[0]), int(value[1]))
    if min(result) < 16:
        raise ValueError(f"{name} dimensions must be >=16, got {result}")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _identity_set_sha256(values: set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _assert_provenance(
    dataset: DocumentMapDataset,
    allowed: set[str],
    split_name: str,
) -> None:
    if not allowed:
        raise ValueError(
            "data.allowed_label_provenance must explicitly list trusted map sources"
        )
    found = {record.label_provenance for record in dataset.records}
    rejected = sorted(found - allowed)
    if rejected:
        raise ValueError(
            f"{split_name} contains unapproved label_provenance={rejected}; "
            f"allowed={sorted(allowed)}"
        )


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: CoarseRectificationLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    errors: list[Tensor] = []
    folds = 0
    cells = 0
    loss_sum = 0.0
    samples = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        output = model(
            batch["warped_image"],
            output_size=batch["backward_map"].shape[-2:],
        )
        losses = criterion(output, batch)
        valid = batch["valid_mask"].bool()
        error = endpoint_error_map(output["backward_map"], batch["backward_map"].float())
        errors.append(error[valid].detach().cpu())
        determinant = jacobian_determinant(output["backward_map"])
        cell_mask = cell_valid_mask(valid)
        folds += int(((determinant <= 0.0) & cell_mask).sum().item())
        cells += int(cell_mask.sum().item())
        batch_size = int(batch["warped_image"].shape[0])
        loss_sum += float(losses["total"].item()) * batch_size
        samples += batch_size
    all_errors = torch.cat(errors).float() if errors else torch.empty(0)
    return {
        "loss": loss_sum / max(samples, 1),
        "epe": float(all_errors.mean()) if all_errors.numel() else float("nan"),
        "epe_p95": (
            float(torch.quantile(all_errors, 0.95))
            if all_errors.numel()
            else float("nan")
        ),
        "fold_rate": folds / max(cells, 1),
        "samples": float(samples),
    }


def train(
    config: dict[str, Any],
    *,
    resume: str | None = None,
    output_dir_override: str | None = None,
) -> Path:
    seed = int(config.get("seed", 1337))
    _seed_everything(seed)
    data_config = dict(config.get("data", {}))
    train_config = dict(config.get("train", {}))
    model_config = dict(config.get("model", {}))
    output_dir = project_path(
        config,
        output_dir_override
        if output_dir_override is not None
        else train_config.get("output_dir", "runs/det_coarse"),
    )
    resume_path = None if resume is None else Path(resume).resolve()
    if resume_path is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to mix a new run into non-empty output_dir={output_dir}; "
                "choose a new run/seed directory or use --resume"
            )
    else:
        if not output_dir.exists():
            raise FileNotFoundError(
                f"resume output_dir does not exist: {output_dir}"
            )
        if resume_path.parent != output_dir.resolve():
            raise ValueError(
                "resume checkpoint must be inside the configured output_dir; "
                f"got checkpoint={resume_path}, output_dir={output_dir.resolve()}"
            )
    input_size = _parse_size(data_config["input_work_size"], "input_work_size")
    output_size = _parse_size(data_config["output_work_size"], "output_work_size")
    train_manifest = project_path(config, data_config["train_manifest"])
    val_manifest = project_path(config, data_config["val_manifest"])
    train_dataset = DocumentMapDataset(
        train_manifest,
        input_work_size=input_size,
        output_work_size=output_size,
    )
    val_dataset = DocumentMapDataset(
        val_manifest,
        input_work_size=input_size,
        output_work_size=output_size,
    )
    assert_document_disjoint(train_dataset.records, val_dataset.records)
    allowed_provenance = {
        str(value).strip().lower()
        for value in data_config.get("allowed_label_provenance", [])
    }
    _assert_provenance(train_dataset, allowed_provenance, "train")
    _assert_provenance(val_dataset, allowed_provenance, "val")
    data_contract = {
        "train_manifest": str(train_manifest.resolve()),
        "train_manifest_sha256": file_sha256(train_manifest),
        "val_manifest": str(val_manifest.resolve()),
        "val_manifest_sha256": file_sha256(val_manifest),
        "allowed_label_provenance": sorted(allowed_provenance),
        "train_document_count": len({record.document_id for record in train_dataset.records}),
        "train_document_ids_sha256": _identity_set_sha256(
            {record.document_id for record in train_dataset.records}
        ),
        "val_document_count": len({record.document_id for record in val_dataset.records}),
        "val_document_ids_sha256": _identity_set_sha256(
            {record.document_id for record in val_dataset.records}
        ),
        "document_disjoint_verified": True,
    }

    device = _device(str(train_config.get("device", "auto")))
    model = build_coarse_model(model_config).to(device)
    criterion = CoarseRectificationLoss(
        CoarseLossWeights.from_mapping(config.get("loss"))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 2.0e-4)),
        weight_decay=float(train_config.get("weight_decay", 1.0e-4)),
    )
    start_epoch = 0
    best_epe = float("inf")
    if resume is not None:
        payload = load_checkpoint(resume_path, map_location=device)
        if payload["model_config"] != model_config:
            raise ValueError("resume checkpoint model_config differs from current config")
        if tuple(payload["input_work_size"]) != input_size:
            raise ValueError("resume checkpoint input_work_size differs from current config")
        if tuple(payload["output_work_size"]) != output_size:
            raise ValueError("resume checkpoint output_work_size differs from current config")
        if payload.get("data_contract") != data_contract:
            raise ValueError("resume checkpoint data contract differs from current manifests")
        model.load_state_dict(payload["model_state"], strict=True)
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_epe = float(
            payload.get("best_epe", payload.get("metrics", {}).get("epe", float("inf")))
        )

    batch_size = int(train_config.get("batch_size", 2))
    workers = int(train_config.get("num_workers", 4))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_config.get("eval_batch_size", batch_size)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["_config_path"], output_dir / "config.yaml")
    epochs = int(train_config.get("epochs", 20))
    use_amp = bool(train_config.get("mixed_precision", True)) and device.type == "cuda"
    gradient_clip = float(train_config.get("gradient_clip", 1.0))
    metrics_path = output_dir / "metrics.json"
    history: list[dict[str, Any]] = []
    if resume is not None and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            existing_history = json.load(handle)
        if not isinstance(existing_history, list):
            raise ValueError(f"existing metrics history is not a list: {metrics_path}")
        history = existing_history
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for raw_batch in train_loader:
            batch = _tensor_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                output = model(
                    batch["warped_image"],
                    output_size=batch["backward_map"].shape[-2:],
                )
                losses = criterion(output, batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            count = int(batch["warped_image"].shape[0])
            running_loss += float(losses["total"].detach().item()) * count
            seen += count
        metrics = evaluate_loader(model, val_loader, criterion, device)
        metrics["train_loss"] = running_loss / max(seen, 1)
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        is_best = metrics["epe"] < best_epe
        if is_best:
            best_epe = metrics["epe"]
        payload = checkpoint_payload(
            model,
            model_config=model_config,
            input_work_size=input_size,
            output_work_size=output_size,
            epoch=epoch,
            optimizer=optimizer,
            metrics=metrics,
            data_contract=data_contract,
            config_sha256=file_sha256(config["_config_path"]),
            best_epe=best_epe,
        )
        epoch_path = output_dir / f"epoch_{epoch:04d}.pt"
        torch.save(payload, epoch_path)
        torch.save(payload, output_dir / "latest.pt")
        if is_best:
            torch.save(payload, output_dir / "best.pt")
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, sort_keys=True), flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", help="Override train.output_dir from the config")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, resume=args.resume, output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
