#!/usr/bin/env python3
"""Render the paper-style v3.3 unified-model architecture overview.

The figure is intentionally generated from code so labels, tensor shapes, and
visual conventions stay reviewable together with the model implementation.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "diffusion2raft-matplotlib")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "figures"

BG = "#F8FAFC"
INK = "#172033"
MUTED = "#5C667A"
GRID = "#DCE2EA"

FROZEN_BLUE = "#DCEFF8"
FROZEN_BLUE_EDGE = "#0072B2"
FROZEN_PURPLE = "#ECE8FA"
FROZEN_PURPLE_EDGE = "#6F55B5"
TRAINABLE = "#FDE7C4"
TRAINABLE_EDGE = "#D98200"
GEOMETRY = "#E9E4F3"
GEOMETRY_EDGE = "#76549B"
IMAGE = "#F0F2F5"
IMAGE_EDGE = "#4B5563"
OUTPUT = "#E2F4EA"
OUTPUT_EDGE = "#218A55"
TRAINING = "#FBE7EC"
TRAINING_EDGE = "#B84A62"


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    badge: str | None = None,
    title_size: float = 10.0,
    body_size: float = 7.8,
    linewidth: float = 1.7,
    zorder: int = 4,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.12,
        y + height - 0.19,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        zorder=zorder + 1,
    )
    if body:
        ax.text(
            x + 0.12,
            y + height - 0.52,
            body,
            ha="left",
            va="top",
            fontsize=body_size,
            color=MUTED,
            linespacing=1.28,
            zorder=zorder + 1,
        )
    if badge:
        ax.text(
            x + width - 0.08,
            y + height - 0.08,
            badge,
            ha="right",
            va="top",
            fontsize=6.4,
            fontweight="bold",
            color=edgecolor,
            bbox={
                "boxstyle": "round,pad=0.22,rounding_size=0.4",
                "facecolor": "white",
                "edgecolor": edgecolor,
                "linewidth": 0.8,
            },
            zorder=zorder + 2,
        )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = IMAGE_EDGE,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    rad: float = 0.0,
    dashed: bool = False,
    linewidth: float = 1.55,
    mutation_scale: float = 12.0,
    zorder: int = 3,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": BG,
                "edgecolor": "none",
                "alpha": 0.94,
            },
            zorder=zorder + 1,
        )
    return patch


def document_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    warped: bool,
    edgecolor: str,
) -> None:
    page = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=1.0,
        zorder=6,
    )
    ax.add_patch(page)
    for index, fraction in enumerate((0.76, 0.58, 0.40, 0.23)):
        yy = y + height * fraction
        x0 = x + width * 0.14
        x1 = x + width * (0.82 if index != 2 else 0.69)
        if warped:
            verts = [
                (x0, yy),
                (x0 + width * 0.18, yy + height * 0.05),
                (x1 - width * 0.20, yy - height * 0.055),
                (x1, yy + height * 0.015),
            ]
            path = MplPath(
                verts,
                [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
            )
            line = PathPatch(
                path,
                facecolor="none",
                edgecolor=edgecolor,
                linewidth=0.85,
                zorder=7,
            )
            ax.add_patch(line)
        else:
            ax.plot([x0, x1], [yy, yy], color=edgecolor, linewidth=0.85, zorder=7)


def section_header(ax: plt.Axes, x: float, width: float, label: str) -> None:
    ax.text(
        x,
        8.72,
        label,
        ha="left",
        va="bottom",
        fontsize=9.6,
        fontweight="bold",
        color=INK,
    )
    ax.plot([x, x + width], [8.64, 8.64], color=GRID, linewidth=1.2, zorder=1)


def legend_item(
    ax: plt.Axes,
    x: float,
    label: str,
    facecolor: str,
    edgecolor: str,
    *,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, 9.12),
            0.22,
            0.16,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.1,
            linestyle=(0, (3, 2)) if dashed else "solid",
            zorder=4,
        )
    )
    ax.text(x + 0.29, 9.20, label, ha="left", va="center", fontsize=7.0, color=MUTED)


def build_figure() -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.unicode_minus": True,
            "svg.hashsalt": "diffusion2raft-v33-overview-v1",
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(BG)

    ax.text(
        0.35,
        9.67,
        "Diffusion2RAFT v3.3 — Teacher-Anchored Unified Rectification",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.37,
        9.31,
        "Frozen diffusion and flow backbones provide geometry cues; trainable heads predict a bounded residual backward flow.",
        ha="left",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )

    legend_item(ax, 11.75, "Frozen external", FROZEN_BLUE, FROZEN_BLUE_EDGE)
    legend_item(ax, 13.37, "Trainable / saved", TRAINABLE, TRAINABLE_EDGE)
    legend_item(ax, 15.15, "Geometry", GEOMETRY, GEOMETRY_EDGE)
    legend_item(ax, 16.32, "Training only", TRAINING, TRAINING_EDGE, dashed=True)

    section_header(ax, 0.35, 4.45, "A   INPUT & FROZEN BACKBONES")
    section_header(ax, 5.15, 4.45, "B   DIFFUSION FEATURES & PRIOR ALIGNMENT")
    section_header(ax, 9.95, 5.05, "C   SAFE FUSION & RESIDUAL FLOW")
    section_header(ax, 15.35, 2.30, "D   GEOMETRIC OUTPUT")

    for x in (5.00, 9.80, 15.20):
        ax.plot([x, x], [2.48, 8.50], color=GRID, linewidth=0.9, linestyle=(0, (2, 5)))

    rounded_box(
        ax,
        0.45,
        4.62,
        1.55,
        1.72,
        title="Warped document",
        body=r"$I_w$   [N, 3, H, W]",
        facecolor=IMAGE,
        edgecolor=IMAGE_EDGE,
        title_size=9.2,
    )
    document_icon(ax, 0.82, 4.82, 0.78, 0.88, warped=True, edgecolor=IMAGE_EDGE)

    rounded_box(
        ax,
        2.42,
        6.63,
        2.38,
        1.20,
        title="259999 geometry teacher",
        body="Corrected-512 backward flow\nStrong rotation & perspective",
        facecolor=FROZEN_BLUE,
        edgecolor=FROZEN_BLUE_EDGE,
        badge="FROZEN",
        title_size=9.4,
    )
    rounded_box(
        ax,
        2.42,
        3.58,
        2.38,
        1.72,
        title="Qwen-Image-Edit",
        body="4 denoising steps\nLast-step hidden: −24 / −12 / −1\nFeatures only · no RGB decode",
        facecolor=FROZEN_BLUE,
        edgecolor=FROZEN_BLUE_EDGE,
        badge="FROZEN",
        title_size=10.0,
    )

    rounded_box(
        ax,
        5.35,
        6.84,
        1.67,
        0.91,
        title="Coarse prior flow",
        body=r"$B$   [N, 2, H, W]",
        facecolor=GEOMETRY,
        edgecolor=GEOMETRY_EDGE,
        title_size=9.1,
        body_size=7.3,
    )
    rounded_box(
        ax,
        7.48,
        6.56,
        2.05,
        1.36,
        title="Prior image warp",
        body=r"$I_p(x)=I_w(x+B(x))$" "\nCoarsely rectified RGB",
        facecolor=GEOMETRY,
        edgecolor=GEOMETRY_EDGE,
        title_size=9.5,
    )
    rounded_box(
        ax,
        10.12,
        6.57,
        1.96,
        1.34,
        title="CNN feature encoder",
        body=r"$C$  [N, 64, H/8, W/8]" "\nSource-faithful detail",
        facecolor=TRAINABLE,
        edgecolor=TRAINABLE_EDGE,
        badge="TRAINABLE",
        title_size=9.4,
        body_size=7.4,
    )

    rounded_box(
        ax,
        5.28,
        3.47,
        2.02,
        1.82,
        title="Layer projection & fusion",
        body=(
            r"$Q_t$  target denoising tokens" "\n"
            r"$Q_s$  source-condition tokens" "\n"
            "96 channels · 1/8 grid"
        ),
        facecolor=TRAINABLE,
        edgecolor=TRAINABLE_EDGE,
        badge="TRAINABLE",
        title_size=9.2,
        body_size=7.0,
    )
    rounded_box(
        ax,
        7.70,
        3.60,
        1.90,
        1.55,
        title="Prior feature alignment",
        body=(r"$Q_s^B=\mathrm{Warp}(Q_s,B_{\downarrow})$" "\n"
              r"$Q_t$ remains in target frame"),
        facecolor=GEOMETRY,
        edgecolor=GEOMETRY_EDGE,
        title_size=9.0,
        body_size=7.0,
    )
    rounded_box(
        ax,
        10.10,
        3.63,
        2.25,
        1.73,
        title="Reliability-gated fusion",
        body=(
            r"Match: $m=\min(\mathrm{sg}(g)^2,0.05)$" "\n"
            r"Context: $c=\mathrm{sg}(g)$" "\n"
            "CNN fallback · feature gate ≠ residual gate"
        ),
        facecolor=TRAINABLE,
        edgecolor=TRAINABLE_EDGE,
        badge="TRAINABLE",
        title_size=9.5,
        body_size=7.2,
    )
    rounded_box(
        ax,
        12.88,
        3.78,
        2.28,
        2.02,
        title="RAFT-like residual refiner",
        body=(
            "9×9 local correlation\n"
            "Motion encoder + ConvGRU ×6\n"
            r"Flow head → $R$" "\n"
            r"Per-axis: $|R_x|,|R_y|\leq24$ px"
        ),
        facecolor=TRAINABLE,
        edgecolor=TRAINABLE_EDGE,
        badge="TRAINABLE",
        title_size=9.5,
        body_size=7.0,
    )

    rounded_box(
        ax,
        15.52,
        5.39,
        2.00,
        1.42,
        title="Backward-flow composition",
        body=(r"$F(x)=\alpha R(x)+B(x+\alpha R(x))$" "\n"
              r"Global $\alpha$: 0→1 warm-up"),
        facecolor=GEOMETRY,
        edgecolor=GEOMETRY_EDGE,
        title_size=9.1,
        body_size=7.0,
    )
    rounded_box(
        ax,
        15.52,
        3.17,
        2.00,
        1.57,
        title="Flow-preserving output",
        body=(r"Final backward flow $F$" "\n"
              r"Rectified $=\mathrm{Warp}(I_w,F)$" "\n"
              "Sample original RGB · no Qwen synthesis"),
        facecolor=OUTPUT,
        edgecolor=OUTPUT_EDGE,
        title_size=9.3,
        body_size=6.8,
    )
    document_icon(ax, 17.10, 3.31, 0.27, 0.43, warped=False, edgecolor=OUTPUT_EDGE)

    # Main inference graph.
    arrow(ax, (2.00, 5.74), (2.42, 7.08), color=FROZEN_BLUE_EDGE, rad=-0.13)
    arrow(ax, (2.00, 5.18), (2.42, 4.45), color=FROZEN_BLUE_EDGE, rad=0.12)
    arrow(ax, (4.80, 7.23), (5.35, 7.23), color=GEOMETRY_EDGE)
    arrow(ax, (7.02, 7.29), (7.48, 7.25), color=GEOMETRY_EDGE)
    arrow(
        ax,
        (1.96, 5.92),
        (7.49, 6.91),
        color=IMAGE_EDGE,
        label=r"original $I_w$",
        label_xy=(4.58, 6.15),
        rad=-0.11,
        linewidth=1.25,
    )
    arrow(ax, (9.53, 7.22), (10.12, 7.22), color=TRAINABLE_EDGE)

    arrow(ax, (4.80, 4.43), (5.28, 4.43), color=FROZEN_BLUE_EDGE)
    arrow(ax, (7.30, 4.38), (7.70, 4.38), color=GEOMETRY_EDGE)
    arrow(ax, (9.60, 4.38), (10.10, 4.45), color=TRAINABLE_EDGE)
    arrow(
        ax,
        (6.20, 6.84),
        (8.18, 5.15),
        color=GEOMETRY_EDGE,
        label=r"resize $B\rightarrow B_{\downarrow}$",
        label_xy=(7.42, 5.88),
        rad=0.18,
        linewidth=1.25,
    )
    arrow(
        ax,
        (11.10, 6.57),
        (11.22, 5.36),
        color=TRAINABLE_EDGE,
        label="CNN detail",
        label_xy=(11.58, 5.93),
        rad=0.04,
    )
    arrow(
        ax,
        (6.55, 6.84),
        (10.28, 5.13),
        color=GEOMETRY_EDGE,
        label=r"$B_{\downarrow}$ context",
        label_xy=(8.92, 5.70),
        rad=-0.14,
        linewidth=1.15,
    )
    arrow(
        ax,
        (12.35, 4.52),
        (12.88, 4.70),
        color=TRAINABLE_EDGE,
        label="Ref · Src · Ctx",
        label_xy=(12.23, 5.62),
    )
    arrow(ax, (15.16, 4.89), (15.52, 5.88), color=GEOMETRY_EDGE, rad=-0.20)
    arrow(
        ax,
        (6.55, 7.75),
        (16.25, 6.81),
        color=GEOMETRY_EDGE,
        label="base flow B",
        label_xy=(11.95, 8.12),
        rad=0.13,
        linewidth=1.65,
    )
    arrow(
        ax,
        (16.52, 5.39),
        (16.52, 4.74),
        color=OUTPUT_EDGE,
        label="final F",
        label_xy=(16.90, 5.05),
    )
    ax.plot(
        [1.23, 1.23, 14.95],
        [4.62, 2.62, 2.62],
        color=IMAGE_EDGE,
        linewidth=1.20,
        solid_capstyle="round",
        zorder=1,
    )
    arrow(
        ax,
        (14.95, 2.62),
        (15.52, 3.48),
        color=IMAGE_EDGE,
        label="original $I_w$ pixels · no Qwen RGB",
        label_xy=(8.55, 2.70),
        rad=0.06,
        linewidth=1.20,
    )
    # Recurrent loop.
    loop = FancyArrowPatch(
        (14.93, 5.66),
        (13.13, 5.75),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color=TRAINABLE_EDGE,
        connectionstyle="arc3,rad=0.52",
        zorder=5,
    )
    ax.add_patch(loop)
    ax.text(14.03, 6.08, "iterate ×6", ha="center", va="center", fontsize=6.8, color=TRAINABLE_EDGE)

    # Training-only band.
    band = FancyBboxPatch(
        (0.27, 0.25),
        17.46,
        1.95,
        boxstyle="round,pad=0.025,rounding_size=0.14",
        facecolor="#FFF8FA",
        edgecolor=TRAINING_EDGE,
        linewidth=1.15,
        linestyle=(0, (5, 3)),
        zorder=1,
    )
    ax.add_patch(band)
    ax.text(
        0.52,
        1.95,
        "TRAINING SUPERVISION ONLY",
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=TRAINING_EDGE,
    )
    ax.text(
        2.68,
        1.95,
        "dashed paths do not run during inference · gradients stop at frozen Qwen and teacher",
        ha="left",
        va="center",
        fontsize=7.2,
        color=MUTED,
    )
    ax.text(
        17.42,
        1.95,
        "best.pt = orange modules + metadata · frozen backbones stay external",
        ha="right",
        va="center",
        fontsize=7.0,
        color=MUTED,
        style="italic",
    )

    rounded_box(
        ax,
        0.58,
        0.68,
        1.38,
        0.78,
        title="GT flow",
        body=r"$F_{gt}$",
        facecolor=TRAINING,
        edgecolor=TRAINING_EDGE,
        title_size=8.4,
        body_size=7.2,
        linewidth=1.15,
    )
    rounded_box(
        ax,
        2.42,
        0.58,
        2.18,
        0.98,
        title="Fixed-point residual target",
        body=r"$F_{gt}(x)=R^*(x)+B(x+R^*(x))$",
        facecolor=TRAINING,
        edgecolor=TRAINING_EDGE,
        title_size=8.2,
        body_size=6.8,
        linewidth=1.15,
    )
    rounded_box(
        ax,
        5.08,
        0.58,
        2.25,
        0.98,
        title="Feature-match auxiliaries",
        body="Qwen local-match CE\nConfidence oracle BCE",
        facecolor=TRAINING,
        edgecolor=TRAINING_EDGE,
        title_size=8.3,
        body_size=6.9,
        linewidth=1.15,
    )
    rounded_box(
        ax,
        7.83,
        0.58,
        2.62,
        0.98,
        title="Geometric supervision",
        body="Sequence flow + residual\nStructure / line / anti-fold",
        facecolor=TRAINING,
        edgecolor=TRAINING_EDGE,
        title_size=8.3,
        body_size=6.9,
        linewidth=1.15,
    )
    rounded_box(
        ax,
        10.94,
        0.68,
        1.50,
        0.78,
        title="GT flat image",
        body=r"$I_{gt}$",
        facecolor=TRAINING,
        edgecolor=TRAINING_EDGE,
        title_size=8.2,
        body_size=7.0,
        linewidth=1.15,
    )
    rounded_box(
        ax,
        12.90,
        0.58,
        2.22,
        0.98,
        title="Appearance supervision",
        body="Photo + gradient\nLine reconstruction",
        facecolor=TRAINING,
        edgecolor=TRAINING_EDGE,
        title_size=8.3,
        body_size=6.9,
        linewidth=1.15,
    )
    rounded_box(
        ax,
        15.60,
        0.68,
        1.50,
        0.78,
        title="Total loss",
        body="Update orange modules",
        facecolor=TRAINABLE,
        edgecolor=TRAINABLE_EDGE,
        title_size=8.3,
        body_size=6.5,
        linewidth=1.25,
    )

    arrow(ax, (1.96, 1.07), (2.42, 1.07), color=TRAINING_EDGE, dashed=True, mutation_scale=9)
    arrow(ax, (4.60, 1.07), (5.08, 1.07), color=TRAINING_EDGE, dashed=True, mutation_scale=9)
    # F_gt and its fixed-point residual target jointly supervise the recurrent
    # composed-flow and raw-residual predictions. Route them above the loss
    # boxes so the three loss families remain visibly parallel.
    ax.plot(
        [4.16, 4.16, 8.25],
        [1.56, 1.72, 1.72],
        color=TRAINING_EDGE,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    arrow(
        ax,
        (8.25, 1.72),
        (8.25, 1.56),
        color=TRAINING_EDGE,
        dashed=True,
        label=r"$F_{gt}$ and $R^*$",
        label_xy=(6.40, 1.73),
        mutation_scale=9,
        linewidth=1.0,
    )
    # I_gt supplies both appearance targets and the structure/line masks.
    arrow(ax, (10.94, 1.07), (10.45, 1.07), color=TRAINING_EDGE, dashed=True, mutation_scale=9)
    arrow(ax, (12.44, 1.07), (12.90, 1.07), color=TRAINING_EDGE, dashed=True, mutation_scale=9)

    # Parallel loss aggregation bus below the boxes; this avoids implying that
    # diffusion, geometry, and appearance losses execute serially.
    for loss_x in (6.205, 9.14, 14.01):
        ax.plot(
            [loss_x, loss_x],
            [0.58, 0.40],
            color=TRAINING_EDGE,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=2,
        )
    ax.plot(
        [6.205, 15.34],
        [0.40, 0.40],
        color=TRAINING_EDGE,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.text(
        15.34,
        0.40,
        "Σ",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=TRAINING_EDGE,
        bbox={
            "boxstyle": "circle,pad=0.18",
            "facecolor": "white",
            "edgecolor": TRAINING_EDGE,
            "linewidth": 1.0,
        },
        zorder=4,
    )
    arrow(
        ax,
        (15.44, 0.48),
        (15.68, 0.72),
        color=TRAINING_EDGE,
        dashed=True,
        mutation_scale=9,
        linewidth=1.0,
    )

    arrow(
        ax,
        (6.18, 1.56),
        (6.12, 3.47),
        color=TRAINING_EDGE,
        dashed=True,
        label="projection gradients",
        label_xy=(6.80, 2.58),
        mutation_scale=10,
        linewidth=1.05,
    )
    arrow(
        ax,
        (6.72, 1.56),
        (10.42, 3.63),
        color=TRAINING_EDGE,
        dashed=True,
        label="match / confidence gradients",
        label_xy=(8.86, 2.38),
        rad=-0.09,
        mutation_scale=10,
        linewidth=1.05,
    )
    arrow(
        ax,
        (9.15, 1.56),
        (13.63, 3.78),
        color=TRAINING_EDGE,
        dashed=True,
        label="flow gradients",
        label_xy=(11.60, 2.48),
        rad=-0.11,
        mutation_scale=10,
        linewidth=1.05,
    )
    arrow(
        ax,
        (13.98, 1.56),
        (16.02, 3.17),
        color=TRAINING_EDGE,
        dashed=True,
        label="image gradients",
        label_xy=(14.82, 2.40),
        rad=-0.09,
        mutation_scale=10,
        linewidth=1.05,
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the v3.3 unified rectification architecture overview."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "v33_unified_model_overview"
    fig = build_figure()
    common = {
        "Creator": "diffusion2raft_unified_v3_1/scripts/render_v33_architecture_overview.py",
        "Title": "Diffusion2RAFT v3.3 unified model overview",
        "Description": "Teacher-anchored diffusion-feature and residual-flow architecture.",
    }
    fig.savefig(
        stem.with_suffix(".svg"),
        format="svg",
        facecolor=BG,
        metadata={**common, "Date": None},
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        format="pdf",
        facecolor=BG,
        metadata={
            "Creator": common["Creator"],
            "Title": common["Title"],
            "Subject": common["Description"],
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        stem.with_suffix(".png"),
        format="png",
        dpi=150,
        facecolor=BG,
        metadata={"Software": common["Creator"], "Title": common["Title"]},
    )
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.svg')}")
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
