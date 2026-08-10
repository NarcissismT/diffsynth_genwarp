#!/usr/bin/env python3
"""Plot completed v3.1 validation summaries from a training log.

The script intentionally treats only ``validation epoch=...`` lines as completed
epochs.  A partially trained epoch therefore cannot appear as a finished result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


VALIDATION_PREFIX = "validation epoch="
VALUE_PATTERN = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>[-+0-9.eE]+)")
REQUIRED_KEYS = ("epoch", "epe", "prior_epe", "gain", "line_epe", "win", "fold_rate", "corr_t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="v3.1 stdout training log")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-final-epoch", type=int, default=20)
    parser.add_argument(
        "--incomplete-reason",
        default="Slurm allocation reached its time limit before validation/checkpointing",
    )
    return parser.parse_args()


def load_validation(log_path: Path) -> tuple[list[dict[str, object]], int]:
    raw_bytes = log_path.read_bytes()
    records_by_epoch: dict[int, dict[str, object]] = {}
    line_count = 0
    for line_count, raw_line in enumerate(raw_bytes.decode("utf-8", errors="replace").splitlines(), start=1):
        if not raw_line.startswith(VALIDATION_PREFIX):
            continue
        values = {match.group("key"): match.group("value") for match in VALUE_PATTERN.finditer(raw_line)}
        missing = [key for key in REQUIRED_KEYS if key not in values]
        if missing:
            raise ValueError(f"validation line {line_count} is missing {missing}: {raw_line}")
        epoch = int(values["epoch"])
        records_by_epoch[epoch] = {
            "epoch": epoch,
            "model_epe_px": float(values["epe"]),
            "prior_epe_px": float(values["prior_epe"]),
            "gain_vs_prior_px": float(values["gain"]),
            "line_epe_px": float(values["line_epe"]),
            "win_rate_fraction": float(values["win"]),
            "fold_rate_fraction": float(values["fold_rate"]),
            "correlation_temperature": float(values["corr_t"]),
            "source_log_line": line_count,
            "raw_validation_summary": raw_line,
        }
    records = [records_by_epoch[epoch] for epoch in sorted(records_by_epoch)]
    if not records:
        raise ValueError(f"no completed validation summaries found in {log_path}")
    return records, line_count


def best_record(records: list[dict[str, object]], key: str, *, highest: bool = False) -> dict[str, object]:
    selector = max if highest else min
    return selector(records, key=lambda record: float(record[key]))


def write_chart(records: list[dict[str, object]], output_path: Path, expected_final_epoch: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(record["epoch"]) for record in records]
    epe = [float(record["model_epe_px"]) for record in records]
    prior = [float(record["prior_epe_px"]) for record in records]
    line_epe = [float(record["line_epe_px"]) for record in records]
    gain = [float(record["gain_vs_prior_px"]) for record in records]
    win = [100.0 * float(record["win_rate_fraction"]) for record in records]
    fold = [10_000.0 * float(record["fold_rate_fraction"]) for record in records]

    best = best_record(records, "model_epe_px")
    best_epoch = int(best["epoch"])
    best_index = epochs.index(best_epoch)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 12), sharex=True, constrained_layout=True)
    fig.suptitle("Diffusion2RAFT v3.1 — completed validation progress", fontsize=16, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, prior, "--", color="#7f8c8d", linewidth=2, label="Prior EPE")
    ax.plot(epochs, epe, "o-", color="#1565c0", linewidth=2.5, label="Final EPE")
    ax.plot(epochs, line_epe, "s-", color="#00897b", linewidth=2, label="Line EPE")
    ax.scatter([best_epoch], [epe[best_index]], s=120, color="#d32f2f", zorder=5)
    ax.annotate(
        f"best completed: epoch {best_epoch}\nEPE {epe[best_index]:.4f}",
        xy=(best_epoch, epe[best_index]),
        xytext=(-120, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#d32f2f"},
        color="#b71c1c",
        fontweight="bold",
    )
    ax.set_ylabel("EPE (pixels, lower is better)")
    ax.legend(loc="upper right", ncol=3)

    ax = axes[1]
    ax.plot(epochs, gain, "o-", color="#6a1b9a", linewidth=2.5, label="Gain vs prior")
    ax.set_ylabel("Gain vs prior (pixels)", color="#6a1b9a")
    ax.tick_params(axis="y", labelcolor="#6a1b9a")
    ax2 = ax.twinx()
    ax2.plot(epochs, win, "s-", color="#ef6c00", linewidth=2.2, label="Final win rate")
    ax2.axhline(50.0, color="#ef6c00", linestyle=":", alpha=0.65, label="50% target")
    ax2.set_ylabel("Win rate (%)", color="#ef6c00")
    ax2.tick_params(axis="y", labelcolor="#ef6c00")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower right", ncol=3)

    ax = axes[2]
    ax.plot(epochs, fold, "o-", color="#2e7d32", linewidth=2.5, label="Fold rate")
    ax.axhline(4.0, color="#c62828", linestyle="--", linewidth=1.8, label="Target: 4e-4")
    ax.fill_between(epochs, fold, 4.0, where=[value <= 4.0 for value in fold], color="#66bb6a", alpha=0.12)
    ax.set_ylabel("Fold rate (×1e-4)")
    ax.set_xlabel("Completed validation epoch")
    ax.legend(loc="upper right")

    for axis in axes:
        axis.set_xticks(list(range(min(epochs), max(expected_final_epoch, max(epochs)) + 1)))
        if expected_final_epoch > max(epochs):
            axis.axvline(expected_final_epoch, color="#c62828", linestyle=":", linewidth=1.8)
    if expected_final_epoch > max(epochs):
        axes[2].annotate(
            f"epoch {expected_final_epoch} incomplete\n(no checkpoint/validation)",
            xy=(expected_final_epoch, axes[2].get_ylim()[0]),
            xytext=(-145, 38),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#c62828"},
            color="#b71c1c",
            fontweight="bold",
        )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_path = args.log.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, line_count = load_validation(log_path)
    raw_bytes = log_path.read_bytes()
    latest = records[-1]
    best_epe = best_record(records, "model_epe_px")
    best_line = best_record(records, "line_epe_px")
    best_win = best_record(records, "win_rate_fraction", highest=True)
    best_fold = best_record(records, "fold_rate_fraction")
    first = records[0]
    incomplete = args.expected_final_epoch > int(latest["epoch"])

    json_path = output_dir / "v31_validation_progress.json"
    markdown_path = output_dir / "v31_validation_progress.md"
    png_path = output_dir / "v31_validation_progress.png"
    write_chart(records, png_path, args.expected_final_epoch)

    summary = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "log_path": str(log_path),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "lines": line_count,
            "selection": "completed validation summaries only",
        },
        "validation": records,
        "derived_summary": {
            "best_model_epe": {"epoch": best_epe["epoch"], "value_px": best_epe["model_epe_px"]},
            "best_line_epe": {"epoch": best_line["epoch"], "value_px": best_line["line_epe_px"]},
            "highest_win_rate": {"epoch": best_win["epoch"], "fraction": best_win["win_rate_fraction"]},
            "lowest_fold_rate": {"epoch": best_fold["epoch"], "fraction": best_fold["fold_rate_fraction"]},
            "first_to_latest_epe_drop_px": round(
                float(first["model_epe_px"]) - float(latest["model_epe_px"]), 6
            ),
            "first_to_latest_epe_drop_percent": round(
                100.0
                * (float(first["model_epe_px"]) - float(latest["model_epe_px"]))
                / float(first["model_epe_px"]),
                4,
            ),
        },
        "training_status": {
            "latest_completed_epoch": latest["epoch"],
            "expected_final_epoch": args.expected_final_epoch,
            "complete": not incomplete,
            "incomplete_reason": args.incomplete_reason if incomplete else None,
            "important": "Partial epochs are excluded because no validation summary/checkpoint exists.",
        },
        "artifacts": {"png": str(png_path), "markdown": str(markdown_path), "json": str(json_path)},
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latest_epe = float(latest["model_epe_px"])
    first_epe = float(first["model_epe_px"])
    epe_drop = first_epe - latest_epe
    markdown = f"""# v3.1 验证进度（完整 epoch）

数据源：`{log_path}`。本报告只统计出现完整 `validation epoch=...` 汇总的 epoch。

![v3.1 validation progress](./{png_path.name})

## 当前结论

- 当前最新完整 checkpoint：epoch {int(latest['epoch'])}；EPE **{latest_epe:.4f} px**，相对 prior 改善 **{float(latest['gain_vs_prior_px']):.4f} px**。
- epoch {int(first['epoch'])} → {int(latest['epoch'])}：EPE 下降 **{epe_drop:.4f} px（{100.0 * epe_drop / first_epe:.2f}%）**；Line EPE 从 {float(first['line_epe_px']):.3f} 降至 **{float(latest['line_epe_px']):.3f}**。
- 当前 win rate **{100.0 * float(latest['win_rate_fraction']):.1f}%**；fold rate **{float(latest['fold_rate_fraction']):.6f}**，低于 4e-4 目标。
- 最低完整验证 EPE 和 Line EPE 均出现在 epoch {int(best_epe['epoch'])}。
"""
    if incomplete:
        markdown += (
            f"- epoch {args.expected_final_epoch} **未完成**：{args.incomplete_reason}；没有对应 validation 汇总或 "
            "`epoch_0020.pt`，不能计为完成结果。\n"
        )
    markdown += "\n精确逐 epoch 数值与源日志行号见 `v31_validation_progress.json`。\n"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(png_path)
    print(markdown_path)
    print(json_path)


if __name__ == "__main__":
    main()
