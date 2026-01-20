#!/usr/bin/env python3
"""
Paper-ready ablation plots for V1~V4.

You already printed the aggregate results in terminal:
V1: MAE/RMSE/MAPE mean±std + per-axis MAE
V2: ...
V3: ...
V4: ...

This script can:
1) Directly plot publication-quality bar charts from the hardcoded aggregate numbers (no JSON needed)
2) Optionally (recommended) read `comparison_report_*.json` to draw per-sample MAE boxplots

Outputs (PNG+PDF):
- ablation_bars_metrics.(png/pdf): MAE/RMSE/MAPE mean±std (error bars)
- ablation_axis_mae.(png/pdf): X/Y/Z axis MAE (mean)
- ablation_trend.(png/pdf): trend lines across versions
- ablation_mae_boxplot.(png/pdf): per-sample MAE distribution (requires report_json)
Also writes:
- ablation_summary.csv
- ablation_table.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Agg:
    avg_mae: float
    std_mae: float
    min_mae: float
    max_mae: float
    avg_rmse: float
    std_rmse: float
    avg_mape: float
    std_mape: float
    min_mape: float
    max_mape: float
    avg_mae_x: float
    avg_mae_y: float
    avg_mae_z: float


# Hardcoded from your terminal summary @cmd (959-988)
# (Values correspond to the 2000-sample run, seed=42)
HARDCODED_AGG: Dict[str, Agg] = {
    "V1": Agg(
        avg_mae=0.168475,
        std_mae=0.188721,
        min_mae=0.013369,
        max_mae=1.399419,
        avg_rmse=0.205151,
        std_rmse=0.231838,
        avg_mape=0.4543,
        std_mape=0.4264,
        min_mape=0.0295,
        max_mape=5.6246,
        avg_mae_x=0.095744,
        avg_mae_y=0.089353,
        avg_mae_z=0.067366,
    ),
    "V2": Agg(
        avg_mae=0.145182,
        std_mae=0.107383,
        min_mae=0.013970,
        max_mae=0.692214,
        avg_rmse=0.176619,
        std_rmse=0.131710,
        avg_mape=0.4162,
        std_mape=0.3212,
        min_mape=0.0308,
        max_mape=3.5386,
        avg_mae_x=0.081754,
        avg_mae_y=0.078985,
        avg_mae_z=0.055208,
    ),
    "V3": Agg(
        avg_mae=0.127759,
        std_mae=0.099616,
        min_mae=0.013772,
        max_mae=0.722944,
        avg_rmse=0.156810,
        std_rmse=0.123776,
        avg_mape=0.3647,
        std_mape=0.2959,
        min_mape=0.0307,
        max_mape=3.9872,
        avg_mae_x=0.071283,
        avg_mae_y=0.067177,
        avg_mae_z=0.051845,
    ),
    "V4": Agg(
        avg_mae=0.115093,
        std_mae=0.090647,
        min_mae=0.013758,
        max_mae=0.523439,
        avg_rmse=0.141226,
        std_rmse=0.112431,
        avg_mape=0.3286,
        std_mape=0.2973,
        min_mape=0.0408,
        max_mape=5.0103,
        avg_mae_x=0.067247,
        avg_mae_y=0.050021,
        avg_mae_z=0.053213,
    ),
}


COLOR = {
    "V1": "#E74C3C",  # red
    "V2": "#3498DB",  # blue
    "V3": "#9B59B6",  # purple
    "V4": "#E67E22",  # orange
}


META = {
    # Keep each line short to avoid x-tick overlap in paper figures
    "V1": "v1\n16D\nBiGRU\n+CA",
    "V2": "v2\n24D\nBiGRU\n+CA",
    "V3": "v3\n24D\nGNN+BiGRU\n+CA",
    "V4": "v4\n32D\nGNN+BiGRU\n+CA",
}


def paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "savefig.dpi": 300,
            # Avoid "invisible character" / missing glyph issues in PDF viewers
            # by embedding TrueType fonts (Type 42) instead of Type 3.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_bars_metrics(agg: Dict[str, Agg], order: List[str], out_dir: Path):
    paper_style()
    x = np.arange(len(order))
    labels = [META.get(k, k) for k in order]
    colors = [COLOR.get(k, "#666666") for k in order]

    mae = np.array([agg[k].avg_mae for k in order])
    mae_s = np.array([agg[k].std_mae for k in order])
    rmse = np.array([agg[k].avg_rmse for k in order])
    rmse_s = np.array([agg[k].std_rmse for k in order])
    mape = np.array([agg[k].avg_mape for k in order])
    mape_s = np.array([agg[k].std_mape for k in order])

    # Slightly wider + extra bottom padding to prevent multi-line tick overlap
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), constrained_layout=True)
    for ax, y, yerr, title, ylabel in [
        (axes[0], mae, mae_s, "MAE (mean +/- std)", "Error (m)"),
        (axes[1], rmse, rmse_s, "RMSE (mean +/- std)", "Error (m)"),
        (axes[2], mape, mape_s, "MAPE (mean +/- std)", "Error (%)"),
    ]:
        ax.bar(x, y, color=colors, edgecolor="black", linewidth=1.0, zorder=2)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", labelsize=8, pad=6)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)

    fig.savefig(out_dir / "ablation_bars_metrics.png", bbox_inches="tight")
    fig.savefig(out_dir / "ablation_bars_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_axis_mae(agg: Dict[str, Agg], order: List[str], out_dir: Path):
    paper_style()
    x = np.arange(len(order))
    labels = [META.get(k, k) for k in order]
    colors = [COLOR.get(k, "#666666") for k in order]

    mae_x = np.array([agg[k].avg_mae_x for k in order])
    mae_y = np.array([agg[k].avg_mae_y for k in order])
    mae_z = np.array([agg[k].avg_mae_z for k in order])

    w = 0.25
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.bar(x - w, mae_x, width=w, color=colors, edgecolor="black", linewidth=1.0, label="MAE-X")
    ax.bar(x, mae_y, width=w, color=colors, edgecolor="black", linewidth=1.0, alpha=0.85, label="MAE-Y")
    ax.bar(x + w, mae_z, width=w, color=colors, edgecolor="black", linewidth=1.0, alpha=0.70, label="MAE-Z")
    ax.set_title("Per-axis MAE (mean)")
    ax.set_ylabel("Error (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=3, frameon=True)

    fig.savefig(out_dir / "ablation_axis_mae.png", bbox_inches="tight")
    fig.savefig(out_dir / "ablation_axis_mae.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_trend(agg: Dict[str, Agg], order: List[str], out_dir: Path):
    paper_style()
    x = np.arange(len(order))
    labels = [META.get(k, k) for k in order]

    mae = np.array([agg[k].avg_mae for k in order])
    rmse = np.array([agg[k].avg_rmse for k in order])
    mape = np.array([agg[k].avg_mape for k in order])

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.plot(x, mae, "-o", color="#2C3E50", linewidth=2.0, markersize=5, label="MAE (m)")
    ax.plot(x, rmse, "-s", color="#16A085", linewidth=2.0, markersize=5, label="RMSE (m)")
    ax2 = ax.twinx()
    ax2.plot(x, mape, "-^", color="#8E44AD", linewidth=2.0, markersize=5, label="MAPE (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (m)")
    ax2.set_ylabel("Error (%)")
    ax.set_title("Ablation trend across versions")
    ax.grid(True, axis="y", alpha=0.25)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=True)

    fig.savefig(out_dir / "ablation_trend.png", bbox_inches="tight")
    fig.savefig(out_dir / "ablation_trend.pdf", bbox_inches="tight")
    plt.close(fig)


def write_tables(agg: Dict[str, Agg], order: List[str], out_dir: Path):
    rows = []
    for k in order:
        a = agg[k]
        rows.append(
            {
                "model": k,
                "avg_mae": a.avg_mae,
                "std_mae": a.std_mae,
                "avg_rmse": a.avg_rmse,
                "std_rmse": a.std_rmse,
                "avg_mape": a.avg_mape,
                "std_mape": a.std_mape,
                "avg_mae_x": a.avg_mae_x,
                "avg_mae_y": a.avg_mae_y,
                "avg_mae_z": a.avg_mae_z,
                "min_mae": a.min_mae,
                "max_mae": a.max_mae,
                "min_mape": a.min_mape,
                "max_mape": a.max_mape,
            }
        )

    csv_path = out_dir / "ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = out_dir / "ablation_table.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated ablation table\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Model & MAE$\\downarrow$ & RMSE$\\downarrow$ & MAPE$\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        for k in order:
            a = agg[k]
            f.write(
                f"{k} & {a.avg_mae:.4f} $\\pm$ {a.std_mae:.4f} & "
                f"{a.avg_rmse:.4f} $\\pm$ {a.std_rmse:.4f} & "
                f"{a.avg_mape:.3f} $\\pm$ {a.std_mape:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def _find_latest_report(reports_dir: Path) -> Optional[Path]:
    cands = sorted(reports_dir.glob("comparison_report_*.json"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _load_per_sample_mae_from_report(report_path: Path, models: List[str]) -> Dict[str, List[float]]:
    """
    Streaming extraction: looks for:
      "metrics": { "V1": { "mae": ... }, "V2": { "mae": ... }, ... }
    Avoids loading full JSON into memory.
    """
    want = set(models)
    out: Dict[str, List[float]] = {m: [] for m in models}

    # match: "V1": {  then later  "mae": number
    re_model = re.compile(r'^\s*"(?P<m>V[1-4])"\s*:\s*\{\s*$')
    re_mae = re.compile(r'^\s*"mae"\s*:\s*(?P<v>[-+0-9.eE]+)\s*,?\s*$')
    in_samples = False
    current_model: Optional[str] = None
    got_mae = False

    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not in_samples:
                if '"samples"' in line and "[" in line:
                    in_samples = True
                continue

            mm = re_model.match(line)
            if mm:
                m = mm.group("m")
                if m in want:
                    current_model = m
                    got_mae = False
                else:
                    current_model = None
                    got_mae = False
                continue

            if current_model and not got_mae:
                mv = re_mae.match(line)
                if mv:
                    out[current_model].append(float(mv.group("v")))
                    got_mae = True

    return out


def plot_mae_boxplot(mae_per_sample: Dict[str, List[float]], order: List[str], out_dir: Path):
    paper_style()
    data = [mae_per_sample.get(k, []) for k in order]
    present = [k for k, d in zip(order, data) if len(d) > 0]
    data = [d for d in data if len(d) > 0]
    if not data:
        return

    labels = [META.get(k, k) for k in present]
    colors = [COLOR.get(k, "#666666") for k in present]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    ax.set_title("Per-sample MAE distribution (boxplot)")
    ax.set_ylabel("MAE (m)")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(out_dir / "ablation_mae_boxplot.png", bbox_inches="tight")
    fig.savefig(out_dir / "ablation_mae_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default=None, help="Where to save figures (default: current dir)")
    ap.add_argument("--order", type=str, default="V1,V2,V3,V4", help="Model order, e.g. V1,V2,V3,V4")
    ap.add_argument(
        "--report_json",
        type=str,
        default=None,
        help="Optional: comparison_report_*.json to draw boxplot using per-sample MAE",
    )
    ap.add_argument(
        "--reports_dir",
        type=str,
        default=str(Path(__file__).parent / "comparison_figures_allmodels"),
        help="Used to auto-pick latest report if report_json not set",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    order = [x.strip().upper() for x in args.order.split(",") if x.strip()]
    agg = {k: HARDCODED_AGG[k] for k in order if k in HARDCODED_AGG}
    order = [k for k in order if k in agg]

    plot_bars_metrics(agg, order, out_dir)
    plot_axis_mae(agg, order, out_dir)
    plot_trend(agg, order, out_dir)
    write_tables(agg, order, out_dir)

    # Optional boxplot (requires per-sample MAE list from report)
    report_path: Optional[Path] = Path(args.report_json) if args.report_json else _find_latest_report(Path(args.reports_dir))
    if report_path and report_path.exists():
        mae_per_sample = _load_per_sample_mae_from_report(report_path, order)
        plot_mae_boxplot(mae_per_sample, order, out_dir)

    print(f"[OK] Output dir: {out_dir}")
    if report_path and report_path.exists():
        print(f"[OK] Boxplot report: {report_path}")
    print("[OK] Generated:")
    print("  - ablation_bars_metrics.(png/pdf)")
    print("  - ablation_axis_mae.(png/pdf)")
    print("  - ablation_trend.(png/pdf)")
    print("  - ablation_mae_boxplot.(png/pdf)  (if report_json exists)")
    print("  - ablation_summary.csv")
    print("  - ablation_table.tex")


if __name__ == "__main__":
    main()

