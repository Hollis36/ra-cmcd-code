#!/usr/bin/env python3
"""Accuracy-vs-efficiency tradeoff figure — accuracy-vs-efficiency figure of the manuscript.

v3 (2026-09-17), derived from v2 which reproduces the published figure pixel for pixel. Only the reading aids change:
  * the early and mid-to-late fusion baselines are drawn as circles like the base model, matching the legend entry
    "YOLO11-RGBT variants" (v2 drew them as a square and a triangle, the legend shapes of DMAF and CFT);
  * panel (c) placed the "early" and "mid-late" labels next to each other's markers; they are re-placed;
  * the base model sits under the RA-CMCD star, so its "base" label now carries a leader line to its marker;
  * panel (c) also placed the ICAFusion label beside the DMAF square and the C2Former label beside the CFT triangle;
    both now sit next to their own markers;
  * label placement is checked by label_qa.py (inside the panel, clear of every other marker and of the callout
    arrows, and closer to its own marker than 0.8x the distance to any other); the labels that failed it were moved
    to the nearest admissible offset: in (a) and (b) "early" sat 34 pt left of its marker (in (b) across the y axis),
    "DMAF" in (b) 34 pt away, and "ICAFusion"/"C2Former" crossed the right border of the panel.
Data, axes, colours, callouts and layout are unchanged.

Three panels (a) Params vs mAP  (b) FLOPs vs mAP  (c) Latency vs mAP.
Each panel has a panel_header band, log-scale x-axis, RA-CMCD highlighted as a
large accent-coloured star with a callout, all other methods labelled by name,
and a Pareto-advantage shading in the low-x/high-y corner.

Data sources (immutable):
  outputs/efficiency_metrics.json  — params_M / flops_G / latency_ms per model
  ACC dict below                   — night-160 single-model mAP@0.5:0.95

Output: <outdir>/fig_efficiency_tradeoff.{png,pdf,svg}
        Default outdir: paper/<SUBMISSION>/figures/redraw_aigcstyle_20260612

The canonical paper/<SUBMISSION>/figures/fig_efficiency_tradeoff.{png,pdf}
is NOT touched; only the redraw directory is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Locate repo root (two levels up from this script: scripts/sota → scripts → repo)
REPO = Path("<YOLO11_ROOT>")

# Add scripts/sota to sys.path so we can import the sibling style library
_SOTA_DIR = Path("<YOLO11_ROOT>/scripts/sota")
if str(_SOTA_DIR) not in sys.path:
    sys.path.insert(0, str(_SOTA_DIR))

from tgrs_aigc_style import (
    SEMANTIC,
    SEMANTIC_FILL,
    INK,
    MUTED,
    GRID,
    FS_BASE,
    COL_DOUBLE,
    apply_tgrs_style,
    panel_header,
    check_text_overlaps,
    save_figure,
)

EFF = REPO / "outputs" / "efficiency_metrics.json"
_DEFAULT_OUTDIR = (
    REPO / "paper" / "<SUBMISSION>" / "figures" / "redraw_aigcstyle_20260612"
)

# ---------------------------------------------------------------------------
# night-160 single-model mAP@0.5:0.95, 4-seed mean
# Source: outputs/sota_runs/sota_summary_current.json + Table racmcd-single of the manuscript
# DO NOT change these values.
# ---------------------------------------------------------------------------
ACC = {
    "RA-CMCD (ours)": 0.6930,
    "YOLO11-RGBT base": 0.6852,
    "YOLO11 early": 0.6699,   # N1 AdamW retrain, four-seed mean (2026-09-07)
    "YOLO11 mid-late": 0.6682,  # N1 AdamW retrain, four-seed mean (2026-09-07)
    "C2Former": 0.6405,
    "CFT": 0.6358,
    "ICAFusion": 0.6040,
    "DMAF": 0.6060,
    "Self-Guided CBEG (diagnostic)": 0.6037,
}

# Colour assignments per role
_ROLE_COLOR = {
    "ours":      SEMANTIC["accent"],
    "baseline":  SEMANTIC["baseline"],
    "teacher":   "#7A609D",    # PURPLE
    "efficient": SEMANTIC["student"],
}

_SHORT_LABEL = {
    "YOLO11-RGBT base": "base",
    "YOLO11 early":     "early",
    "YOLO11 mid-late":  "mid-late",
}


def _build_points(eff: dict) -> list[dict]:
    """Assemble per-method scatter point descriptors from JSON + ACC dict."""
    s = eff.get("RA-CMCD student (YOLO11-RGBT, ours)", {})
    cbeg = eff.get("Self-Guided CBEG (diagnostic, ours)", {})
    pts = [
        # ── Ours (accent star) ──────────────────────────────────────────────
        {
            "label": "RA-CMCD (ours)",
            "params": s.get("params_M"),
            "flops": s.get("flops_G"),
            "lat": s.get("latency_ms"),
            "acc": ACC["RA-CMCD (ours)"],
            "role": "ours",
            "marker": "*",
        },
        # ── YOLO11-RGBT variants (baseline) ─────────────────────────────────
        # early/mid-late/DMAF 的 params/flops/lat 来自 efficiency_table.tex 实测口径，
        # 尚未录入 efficiency_metrics.json；数据更新时两处需同步。
        {
            "label": "YOLO11-RGBT base",
            "params": s.get("params_M"),   # same backbone as ours
            "flops": s.get("flops_G"),
            "lat": s.get("latency_ms"),
            "acc": ACC["YOLO11-RGBT base"],
            "role": "baseline",
            "marker": "o",
        },
        {
            "label": "YOLO11 early",
            "params": 2.6,
            "flops": 6.3,
            "lat": 6.0,
            "acc": ACC["YOLO11 early"],
            "role": "baseline",
            "marker": "o",
        },
        {
            "label": "YOLO11 mid-late",
            "params": 4.3,
            "flops": 11.2,
            "lat": 9.4,
            "acc": ACC["YOLO11 mid-late"],
            "role": "baseline",
            "marker": "o",
        },
        # ── DMAF (efficient) ─────────────────────────────────────────────────
        {
            "label": "DMAF",
            "params": 27.4,
            "flops": 54.3,
            "lat": 6.7,
            "acc": ACC["DMAF"],
            "role": "efficient",
            "marker": "s",
        },
    ]
    # ── Published RGBT teachers (from JSON) ────────────────────────────────
    for name, mshape in [("C2Former", "p"), ("CFT", "^"), ("ICAFusion", "v")]:
        r = eff.get(name, {})
        pts.append({
            "label": name,
            "params": r.get("params_M"),
            "flops": r.get("flops_G"),
            "lat": r.get("latency_ms"),
            "acc": ACC[name],
            "role": "teacher",
            "marker": mshape,
        })
    # ── Diagnostic student (ghost point) ────────────────────────────────────
    _cbeg_removed_F1 = ({
        "label": "CBEG (diag.)",
        "params": cbeg.get("params_M"),
        "flops": cbeg.get("flops_G"),
        "lat": cbeg.get("latency_ms"),
        "acc": ACC["Self-Guided CBEG (diagnostic)"],
        "role": "efficient",
        "marker": "d",
        "ghost": True,
    })
    return pts


def _scatter_style(p: dict) -> dict:
    """Return scatter kwargs (facecolor, edgecolor, s, alpha, zorder)."""
    ghost = p.get("ghost", False)
    role = p["role"]
    if ghost:
        return dict(facecolor="white", edgecolor=SEMANTIC["student"],
                    s=44, alpha=0.85, zorder=2)
    if role == "ours":
        return dict(facecolor=SEMANTIC["accent"], edgecolor=INK,
                    s=190, alpha=1.0, zorder=5)
    color = _ROLE_COLOR[role]
    return dict(facecolor=color, edgecolor=INK,
                s=52, alpha=0.88, zorder=3)


# Per-panel, per-label annotation offsets (dx pts, dy pts, ha). "ours" has no label; its callout box serves that role.
# Point positions:
#   params: early=2.6M, mid-late=4.3M, RA-CMCD=RGBT-base=5.14M, DMAF=27.4M, C2Former=81M, ICAFusion=120M, CFT=208M
#   flops:  early=6.3G, mid-late=11.2G, RA-CMCD=RGBT-base=12.45G, DMAF=54.3G, C2Former=94G, ICAFusion=96G, CFT=112G
#   lat:    early=6.0ms, DMAF=6.7ms, mid-late=9.4ms, ICAFusion=15.1ms, RA-CMCD=RGBT-base=17.08ms, CFT=26ms, C2Former=33ms
# Entries marked label_qa were set on 2026-09-17 by label_qa.py's search (nearest offset that keeps the label inside
# its panel, clear of other markers and of the callout arrows, and clearly closer to its own marker); the other
# entries already passed that check and keep their earlier values. The CBEG entries are unused (point removed).
_OFFSETS: dict[tuple[str, str], tuple[int, int, str]] = {
    # ---- params panel ----
    ("YOLO11 early",     "params"): ( -4,  -4, "right"),   # label_qa
    ("YOLO11 mid-late",  "params"): (  4, -12, "left"),    # label_qa
    ("YOLO11-RGBT base", "params"): ( 18, -16, "left"),    # v3: leader line to the marker under the star
    ("DMAF",             "params"): ( -4,   2, "right"),   # label_qa
    ("CBEG (diag.)",     "params"): (-34,  -8, "right"),
    ("C2Former",         "params"): (  4,   6, "left"),    # label_qa
    ("ICAFusion",        "params"): (  0, -10, "center"),  # label_qa
    ("CFT",              "params"): (  5, -12, "left"),
    # ---- flops panel ----
    ("YOLO11 early",     "flops"):  (  0, -10, "center"),  # label_qa
    ("YOLO11 mid-late",  "flops"):  (  4, -12, "left"),    # label_qa
    ("YOLO11-RGBT base", "flops"):  ( 18, -16, "left"),
    ("DMAF",             "flops"):  ( -4,   2, "right"),   # label_qa
    ("CBEG (diag.)",     "flops"):  (-34,  -8, "right"),
    ("C2Former",         "flops"):  ( -4,   0, "right"),   # label_qa
    ("ICAFusion",        "flops"):  (  0, -10, "center"),  # label_qa
    ("CFT",              "flops"):  (  4,  -8, "left"),    # label_qa
    # ---- latency panel ----
    ("YOLO11 early",     "lat"):    ( -4,  -8, "right"),   # label_qa
    ("DMAF",             "lat"):    ( -4,   2, "right"),   # label_qa
    ("YOLO11 mid-late",  "lat"):    (  0, -10, "center"),  # label_qa
    ("CBEG (diag.)",     "lat"):    (  5, -14, "left"),
    ("ICAFusion",        "lat"):    (  6, -12, "left"),    # v3: v2 put it left of the DMAF square
    ("YOLO11-RGBT base", "lat"):    ( 16, -16, "left"),
    ("CFT",              "lat"):    ( -7,  -2, "right"),   # v3: left of its own marker, level with it
    ("C2Former",         "lat"):    (  0,   9, "center"),  # v3: above its own marker (v2 put it beside the CFT triangle)
}


def _pareto_shade(ax, pts_valid: list[dict], key: str, y_lo: float, y_hi: float) -> None:
    """Shade the Pareto-advantage region with a subtle fill_betweenx.

    Region: x ∈ [x_min_data, x_ours] and y ∈ [y_ours_acc, y_hi].
    Uses SEMANTIC_FILL['student'] at low alpha so it doesn't dominate.
    """
    ours = next((p for p in pts_valid if p["role"] == "ours"), None)
    if ours is None:
        return
    x_all = [p[key] for p in pts_valid if p[key] is not None]
    if not x_all:
        return
    x_ours = ours[key]
    y_ours_pct = ours["acc"] * 100.0
    # Fill between x=x_min and x=x_ours, vertically from y_ours to y_hi
    # fill_between on a log-scale axes fills by x values (log-interpolated)
    ax.fill_betweenx(
        [y_ours_pct, y_hi],
        min(x_all) * 0.45,
        x_ours,
        color=SEMANTIC_FILL["student"],
        alpha=0.50,
        linewidth=0,
        zorder=0,
    )


def _draw_panel(
    ax,
    pts: list[dict],
    key: str,
    xlabel: str,
    ours_callout_xy_data: tuple[float, float],
    ours_callout_xytext_frac: tuple[float, float],
) -> None:
    """Draw one efficiency-tradeoff scatter panel.

    ours_callout_xy_data: (x, y_pct) data coords for arrow tip on the star.
    ours_callout_xytext_frac: (cx, cy) axes-fraction for the callout text box.
    """
    pts_valid = [p for p in pts if p[key] is not None]

    # ── Set scale + axes style first ────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=FS_BASE)
    ax.grid(True, color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)

    # Fixed ylim: covers all points with room for labels and callout
    y_lo, y_hi = 57.5, 73.5
    ax.set_ylim(y_lo, y_hi)
    # Explicit yticks at 4-unit steps: [60, 64, 68, 72]
    # These avoid landing on or near any data point acc values
    ax.set_yticks([60, 64, 68, 72])
    # x margins handled by set_xlim after plotting
    ax.margins(x=0.32, y=0.0)  # y margin already applied via set_ylim

    # ── Pareto shading ───────────────────────────────────────────────────────
    _pareto_shade(ax, pts_valid, key, y_lo, y_hi)

    # ── Scatter points ───────────────────────────────────────────────────────
    for p in pts_valid:
        kw = _scatter_style(p)
        ax.scatter(
            p[key], p["acc"] * 100.0,
            marker=p["marker"],
            linewidth=0.5,
            **kw
        )

    # ── Name labels for every non-ours point ─────────────────────────────────
    for p in pts_valid:
        if p["role"] == "ours":
            continue
        ghost = p.get("ghost", False)
        role = p["role"]
        color = _ROLE_COLOR[role] if not ghost else SEMANTIC["student"]
        text = _SHORT_LABEL.get(p["label"], p["label"])
        dx, dy, ha = _OFFSETS.get((p["label"], key), (5, 5, "left"))
        ax.annotate(
            text,
            (p[key], p["acc"] * 100.0),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=FS_BASE,
            color=color,
            fontweight="normal",
            style="italic" if ghost else "normal",
            ha=ha,
            annotation_clip=False,
            # v3: the base marker is covered by the RA-CMCD star; a thin leader identifies it
            arrowprops=(dict(arrowstyle="-", color=color, lw=0.5, shrinkA=0, shrinkB=2.5)
                        if p["label"] == "YOLO11-RGBT base" else None),
        )

    # ── Callout for ours: text box (split from arrow to keep bbox manageable) ──
    # Following the Fig.9 pattern: text-only artist + empty-text arrow artist.
    # Empty-text annotations are excluded from check_text_overlaps (empty strip()).
    ours_pt = next((p for p in pts_valid if p["role"] == "ours"), None)
    if ours_pt is not None:
        x_val = ours_pt[key]
        y_val = ours_pt["acc"]
        # Build callout text from actual data values (no hardcoding)
        if key == "params":
            callout_text = f"{x_val:.1f}M,\n{y_val:.3f}"
        elif key == "flops":
            callout_text = f"{x_val:.1f}G,\n{y_val:.3f}"
        else:
            callout_text = f"{x_val:.1f}ms,\n{y_val:.3f}"

        cx_frac, cy_frac = ours_callout_xytext_frac

        # Text box — in axes fraction, NO arrowprops here (checked by QA)
        ax.text(
            cx_frac, cy_frac, callout_text,
            transform=ax.transAxes,
            fontsize=FS_BASE, fontweight="bold", color=SEMANTIC["accent"],
            va="center", ha="center", zorder=10, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.32", fc="white",
                      ec=SEMANTIC["accent"], lw=0.9),
        )
        # Arrow only — text="" → stripped to "" → excluded from overlap check
        ax.annotate(
            "",
            xy=(x_val, y_val * 100.0),
            xytext=(cx_frac, cy_frac),
            xycoords="data",
            textcoords="axes fraction",
            zorder=9,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-|>",
                color=SEMANTIC["accent"],
                lw=0.9,
                connectionstyle="arc3,rad=-0.20",
                shrinkA=4,
                shrinkB=5,
            ),
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Fig. 6 efficiency tradeoff (AIGC-composition style)"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=_DEFAULT_OUTDIR,
        help=(
            "Output directory "
            "(default: paper/<SUBMISSION>/figures/redraw_aigcstyle_20260612)"
        ),
    )
    args = parser.parse_args(argv)
    outdir = args.outdir

    # ── Load data ───────────────────────────────────────────────────────────
    eff = {r["name"]: r for r in json.loads(EFF.read_text())["results"]}
    pts = _build_points(eff)

    # Determine which panels are available
    specs = [
        ("params", "Parameters (M)"),
        ("flops",  "FLOPs (G)"),
    ]
    if any(p["lat"] for p in pts):
        specs.append(("lat", "Latency (ms)"))

    n_panels = len(specs)

    # ── Apply unified style ──────────────────────────────────────────────────
    apply_tgrs_style()

    # Canvas: COL_DOUBLE × 2.55 in
    fig_h = 2.55
    fig = plt.figure(figsize=(COL_DOUBLE, fig_h))

    # Axes rects (figure fraction):
    #   left margin: 0.068 (y-axis label)
    #   right margin: 0.008
    #   bottom: 0.20 (xlabel + legend space)
    #   top: 0.855 (leaves ~17pt for header band at fig_h=2.55in)
    left0 = 0.068
    right_edge = 0.992
    gap = 0.026        # inter-panel gap
    total_w = right_edge - left0 - gap * (n_panels - 1)
    ax_w = total_w / n_panels
    ax_bot = 0.20
    ax_top = 0.855
    ax_h = ax_top - ax_bot

    axs = []
    for i in range(n_panels):
        left = left0 + i * (ax_w + gap)
        ax = fig.add_axes((left, ax_bot, ax_w, ax_h))
        axs.append(ax)

    # ── Get ours data values for callout text (read from pts, not hardcoded) ─
    ours_pt = next(p for p in pts if p["role"] == "ours")

    # Callout text boxes: upper-right corner, above teacher cluster.
    # Now using the split text+arrow approach (text box alone, arrow empty-text).
    # This means the callout box bbox is just the small text box; no arm.
    # Geometry (params panel, xlim~0.64-844, ylim=57.5-73.5):
    #   Teacher cluster labels top out at y_frac~0.52 (C2Former+8pt)
    #   RA-CMCD star at y_frac~0.74; "base" label at y_frac~0.58
    #   Callout at y_frac=0.82 -> y=70.6 (above all labels, below ytick 72 at 0.906)
    #   x_frac=0.85 -> x~280M (right of all teacher points, clear of all labels)
    _callout_pos = {
        "params": (0.85, 0.82),
        "flops":  (0.85, 0.82),
        "lat":    (0.85, 0.82),
    }

    # Panel titles for header
    _panel_titles = {
        "params": "Params vs mAP",
        "flops":  "FLOPs vs mAP",
        "lat":    "Latency vs mAP",
    }

    for ax, (key, xlabel) in zip(axs, specs):
        cpos = _callout_pos[key]
        _draw_panel(
            ax, pts, key, xlabel,
            ours_callout_xy_data=(ours_pt[key], ours_pt["acc"]),
            ours_callout_xytext_frac=cpos,
        )

    # ── Shared y-label ───────────────────────────────────────────────────────
    axs[0].set_ylabel("Night-160 mAP@0.5:0.95 (%)", fontsize=FS_BASE, fontweight="bold")

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=SEMANTIC["baseline"], markeredgecolor=INK,
                   markersize=5, label="YOLO11-RGBT variants"),
        plt.Line2D([0], [0], marker="s", linestyle="None",
                   markerfacecolor=SEMANTIC["student"], markeredgecolor=INK,
                   markersize=5, label="DMAF"),
        plt.Line2D([0], [0], marker="p", linestyle="None",
                   markerfacecolor="#7A609D", markeredgecolor=INK,
                   markersize=5, label="C2Former"),
        plt.Line2D([0], [0], marker="^", linestyle="None",
                   markerfacecolor="#7A609D", markeredgecolor=INK,
                   markersize=5, label="CFT"),
        plt.Line2D([0], [0], marker="v", linestyle="None",
                   markerfacecolor="#7A609D", markeredgecolor=INK,
                   markersize=5, label="ICAFusion"),
        plt.Line2D([0], [0], marker="*", linestyle="None",
                   markerfacecolor=SEMANTIC["accent"], markeredgecolor=INK,
                   markersize=9, label="RA-CMCD student"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),   # relayout: legend below the x-axis titles
        ncol=4,
        frameon=True,
        edgecolor="#9EACBD",
        framealpha=1.0,
        fontsize=FS_BASE,
        columnspacing=0.85,
        handletextpad=0.40,
    )

    # ── Panel headers (MUST come after final layout) ──────────────────────────
    letters = list("abc")
    for ax, (key, _), letter in zip(axs, specs, letters):
        panel_header(ax, letter, _panel_titles[key])

    # ── Overlap QA ───────────────────────────────────────────────────────────
    overlaps = check_text_overlaps(fig)
    if overlaps:
        raise RuntimeError(
            "Fig.6 label collision(s): "
            + str([(a.get_text(), b.get_text()) for a, b in overlaps])
        )
    from label_qa import qa as label_placement_qa  # sibling module; checks labels against markers and panel borders
    placement = label_placement_qa(fig)
    if placement:
        raise RuntimeError("label placement: " + "; ".join(placement))

    # ── Save ─────────────────────────────────────────────────────────────────
    save_figure(fig, "fig_efficiency_tradeoff", outdir, max_width_in=COL_DOUBLE)
    print(f"latency panel: {'yes' if len(specs) == 3 else 'no'}")


if __name__ == "__main__":
    main()
