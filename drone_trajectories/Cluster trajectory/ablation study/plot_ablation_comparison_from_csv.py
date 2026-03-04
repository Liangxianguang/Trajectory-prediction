#!/usr/bin/env python3
"""
Ablation comparison plots from CSV/JSON (paper-ready).

Inputs:
- CSV with aggregate metrics per experiment (e.g., ablation_results.csv)
- Optional JSON with per-step MAE (e.g., ablation_summary.json)

Outputs (PNG+PDF):
- ablation_overall_metrics.(png/pdf): MAE/RMSE/MAPE (grouped bars, mean±std)
- ablation_ade_fde.(png/pdf): ADE/FDE (grouped bars, mean±std)
- ablation_axis_mae.(png/pdf): MAE_X/MAE_Y/MAE_Z (grouped bars)
- ablation_trend.(png/pdf): MAE/RMSE/MAPE trend across experiments
- ablation_per_step_mae_boxplot.(png/pdf): per-step MAE distribution (requires JSON)
Also writes:
- ablation_summary_table.csv
- ablation_summary_table.tex
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ExpRow:
    experiment_id: int
    experiment_name: str
    MAE_mean: float
    MAE_std: float
    RMSE_mean: float
    RMSE_std: float
    ADE_mean: float
    ADE_std: float
    FDE_mean: float
    FDE_std: float
    MAPE_mean: float
    MAPE_std: float
    MAE_X_mean: float
    MAE_Y_mean: float
    MAE_Z_mean: float


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_csv(csv_path: Path) -> List[ExpRow]:
    rows: List[ExpRow] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ExpRow(
                    experiment_id=int(r["experiment_id"]),
                    experiment_name=r["experiment_name"],
                    MAE_mean=float(r["MAE_mean"]),
                    MAE_std=float(r["MAE_std"]),
                    RMSE_mean=float(r["RMSE_mean"]),
                    RMSE_std=float(r["RMSE_std"]),
                    ADE_mean=float(r["ADE_mean"]),
                    ADE_std=float(r["ADE_std"]),
                    FDE_mean=float(r["FDE_mean"]),
                    FDE_std=float(r["FDE_std"]),
                    MAPE_mean=float(r["MAPE_mean"]),
                    MAPE_std=float(r["MAPE_std"]),
                    MAE_X_mean=float(r["MAE_X_mean"]),
                    MAE_Y_mean=float(r["MAE_Y_mean"]),
                    MAE_Z_mean=float(r["MAE_Z_mean"]),
                )
            )
    rows.sort(key=lambda x: x.experiment_id)
    return rows


def _label_short(name: str) -> str:
    # Split like "Exp1: XXX" to short multi-line tick
    if ":" in name:
        prefix, title = name.split(":", 1)
        return f"{prefix.strip()}\n{title.strip()}"
    return name


def _palette(n: int) -> List[str]:
    base = ["#E74C3C", "#3498DB", "#9B59B6", "#E67E22", "#27AE60", "#2C3E50"]
    if n <= len(base):
        return base[:n]
    return (base * ((n // len(base)) + 1))[:n]


def plot_grouped_bars(
    title: str,
    ylabel: str,
    metrics: List[str],
    means: Dict[str, List[float]],
    stds: Dict[str, List[float]],
    labels: List[str],
    out_path: Path,
):
    paper_style()
    x = np.arange(len(metrics))
    n = len(labels)
    width = 0.75 / n
    colors = _palette(n)

    fig, ax = plt.subplots(figsize=(10.8, 3.8), constrained_layout=True)
    for i, label in enumerate(labels):
        y = np.array(means[label])
        yerr = np.array(stds[label])
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            y,
            width=width,
            label=label,
            color=colors[i],
            edgecolor="black",
            linewidth=0.9,
            zorder=2,
        )
        ax.errorbar(
            x + (i - (n - 1) / 2) * width,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            zorder=3,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper left", ncol=2, frameon=True)

    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_axis_mae(rows: List[ExpRow], out_path: Path):
    paper_style()
    labels = [_label_short(r.experiment_name) for r in rows]
    colors = _palette(len(rows))

    x = np.arange(len(rows))
    w = 0.25

    mae_x = np.array([r.MAE_X_mean for r in rows])
    mae_y = np.array([r.MAE_Y_mean for r in rows])
    mae_z = np.array([r.MAE_Z_mean for r in rows])

    fig, ax = plt.subplots(figsize=(10.8, 3.8), constrained_layout=True)
    ax.bar(x - w, mae_x, width=w, color=colors, edgecolor="black", linewidth=0.9, label="MAE-X")
    ax.bar(x, mae_y, width=w, color=colors, edgecolor="black", linewidth=0.9, alpha=0.85, label="MAE-Y")
    ax.bar(x + w, mae_z, width=w, color=colors, edgecolor="black", linewidth=0.9, alpha=0.70, label="MAE-Z")

    ax.set_title("Per-axis MAE (mean)")
    ax.set_ylabel("Error (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=8, pad=6)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=3, frameon=True)

    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_trend(rows: List[ExpRow], out_path: Path):
    paper_style()
    labels = [_label_short(r.experiment_name) for r in rows]
    x = np.arange(len(rows))

    mae = np.array([r.MAE_mean for r in rows])
    rmse = np.array([r.RMSE_mean for r in rows])
    mape = np.array([r.MAPE_mean for r in rows])

    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    ax.plot(x, mae, "-o", color="#2C3E50", linewidth=2.0, markersize=5, label="MAE (m)")
    ax.plot(x, rmse, "-s", color="#16A085", linewidth=2.0, markersize=5, label="RMSE (m)")
    ax2 = ax.twinx()
    ax2.plot(x, mape, "-^", color="#8E44AD", linewidth=2.0, markersize=5, label="MAPE (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (m)")
    ax2.set_ylabel("Error (%)")
    ax.set_title("Ablation trend across experiments")
    ax.grid(True, axis="y", alpha=0.25)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", frameon=True)

    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_per_step_boxplot(summary_json: Path, rows: List[ExpRow], out_path: Path):
    if not summary_json or not summary_json.exists():
        return

    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    # build per-step MAE lists for each experiment id
    exp_map = {r.experiment_id: r for r in rows}
    data = []
    labels = []

    for key, exp in summary.get("experiments", {}).items():
        agg = exp.get("aggregate_stats", {})
        per_step = agg.get("MAE_per_step_mean")
        if not per_step:
            continue
        exp_id = exp.get("experiment_id")
        if exp_id not in exp_map:
            continue
        data.append(per_step)
        labels.append(_label_short(exp_map[exp_id].experiment_name))

    if not data:
        return

    paper_style()
    fig, ax = plt.subplots(figsize=(10.2, 3.8), constrained_layout=True)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55, showfliers=False)
    colors = _palette(len(data))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.9)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.1)

    ax.set_title("Per-step MAE distribution (boxplot)")
    ax.set_ylabel("MAE (m)")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_tables(rows: List[ExpRow], out_dir: Path):
    csv_path = out_dir / "ablation_summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment_id",
                "experiment_name",
                "MAE_mean",
                "MAE_std",
                "RMSE_mean",
                "RMSE_std",
                "ADE_mean",
                "ADE_std",
                "FDE_mean",
                "FDE_std",
                "MAPE_mean",
                "MAPE_std",
                "MAE_X_mean",
                "MAE_Y_mean",
                "MAE_Z_mean",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.experiment_id,
                    r.experiment_name,
                    r.MAE_mean,
                    r.MAE_std,
                    r.RMSE_mean,
                    r.RMSE_std,
                    r.ADE_mean,
                    r.ADE_std,
                    r.FDE_mean,
                    r.FDE_std,
                    r.MAPE_mean,
                    r.MAPE_std,
                    r.MAE_X_mean,
                    r.MAE_Y_mean,
                    r.MAE_Z_mean,
                ]
            )

    tex_path = out_dir / "ablation_summary_table.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated ablation table\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Model & MAE$\\downarrow$ & RMSE$\\downarrow$ & MAPE$\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r.experiment_name} & {r.MAE_mean:.4f} $\\pm$ {r.MAE_std:.4f} & "
                f"{r.RMSE_mean:.4f} $\\pm$ {r.RMSE_std:.4f} & "
                f"{r.MAPE_mean:.3f} $\\pm$ {r.MAPE_std:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to ablation_results.csv")
    ap.add_argument("--summary_json", default=None, help="Optional ablation_summary.json")
    ap.add_argument("--output_dir", default=None, help="Output directory")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    summary_json = Path(args.summary_json) if args.summary_json else None
    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    labels = [_label_short(r.experiment_name) for r in rows]

    metrics_main = ["MAE", "RMSE", "MAPE"]
    means_main = {
        labels[i]: [rows[i].MAE_mean, rows[i].RMSE_mean, rows[i].MAPE_mean]
        for i in range(len(rows))
    }
    stds_main = {
        labels[i]: [rows[i].MAE_std, rows[i].RMSE_std, rows[i].MAPE_std]
        for i in range(len(rows))
    }
    plot_grouped_bars(
        "Overall metrics (mean ± std)",
        "Error (m or %)",
        metrics_main,
        means_main,
        stds_main,
        labels,
        out_dir / "ablation_overall_metrics",
    )

    metrics_ade = ["ADE", "FDE"]
    means_ade = {
        labels[i]: [rows[i].ADE_mean, rows[i].FDE_mean] for i in range(len(rows))
    }
    stds_ade = {
        labels[i]: [rows[i].ADE_std, rows[i].FDE_std] for i in range(len(rows))
    }
    plot_grouped_bars(
        "ADE/FDE comparison (mean ± std)",
        "Error (m)",
        metrics_ade,
        means_ade,
        stds_ade,
        labels,
        out_dir / "ablation_ade_fde",
    )

    plot_axis_mae(rows, out_dir / "ablation_axis_mae")
    plot_trend(rows, out_dir / "ablation_trend")
    plot_per_step_boxplot(summary_json, rows, out_dir / "ablation_per_step_mae_boxplot")
    write_tables(rows, out_dir)

    print(f"[OK] Output dir: {out_dir}")


if __name__ == "__main__":
    main()
