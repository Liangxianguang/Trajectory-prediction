#!/usr/bin/env python3
"""
Plot comparison between LBEBM3D and Exp5_Full models - Publication Ready
Generate paper-quality figures with professional styling
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

matplotlib.use('Agg')

# ============================================================================
# Paper-style configuration
# ============================================================================
def paper_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.alpha': 0.25,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

def palette(n: int):
    """Professional color palette"""
    base = ['#E74C3C', '#3498DB', '#9B59B6', '#E67E22', '#27AE60', '#2C3E50']
    if n <= len(base):
        return base[:n]
    return (base * ((n // len(base)) + 1))[:n]

# ============================================================================
# Load JSON data
# ============================================================================
json_file = Path(__file__).parent / 'ablation_summary.json'
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract experiment data
exps = data['experiments']
lbebm_stats = exps['LBEBM3D']['aggregate_stats']
exp5_stats = exps['Exp5_Full']['aggregate_stats']

output_dir = Path(__file__).parent / 'comparison_plots_v2'
output_dir.mkdir(exist_ok=True)

print(f"Output directory: {output_dir}")
print("Generating paper-quality figures...")

# Model labels
MODEL_NAMES = ['3DMoTraj', 'DG32-BCAT']
colors = palette(2)
COLOR_BASELINE = colors[0]  # 3DMoTraj (baseline)
COLOR_OURS = colors[1]      # DG32-BCAT (our method)

# ============================================================================
# 1. Overall Metrics Comparison (MAE, RMSE, MAPE)
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(10.8, 3.8), constrained_layout=True)

metrics = ['MAE', 'RMSE', 'MAPE']
x = np.arange(len(metrics))
width = 0.35

# Prepare data for 2 models x 3 metrics
lbebm_values = [lbebm_stats['MAE_mean'], lbebm_stats['RMSE_mean'], lbebm_stats['MAPE_mean']]
lbebm_errors = [lbebm_stats['MAE_std'], lbebm_stats['RMSE_std'], lbebm_stats['MAPE_std']]
exp5_values = [exp5_stats['MAE_mean'], exp5_stats['RMSE_mean'], exp5_stats['MAPE_mean']]
exp5_errors = [exp5_stats['MAE_std'], exp5_stats['RMSE_std'], exp5_stats['MAPE_std']]

bars1 = ax.bar(x - width/2, lbebm_values, width, label='3DMoTraj',
               color=COLOR_BASELINE, edgecolor='black', linewidth=0.9, zorder=2)
ax.errorbar(x - width/2, lbebm_values, lbebm_errors, fmt='none',
            ecolor='black', elinewidth=1.0, capsize=3, zorder=3)

bars2 = ax.bar(x + width/2, exp5_values, width, label='DG32-BCAT',
               color=COLOR_OURS, edgecolor='black', linewidth=0.9, zorder=2)
ax.errorbar(x + width/2, exp5_values, exp5_errors, fmt='none',
            ecolor='black', elinewidth=1.0, capsize=3, zorder=3)

ax.set_ylabel('Error (m or %)', fontsize=10)
ax.set_title('Overall Metrics Comparison', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.legend(loc='upper left', frameon=True, fontsize=9)
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)

plt.savefig(output_dir / 'overall_metrics.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'overall_metrics.pdf', bbox_inches='tight')
plt.close()
print("✓ overall_metrics.png/pdf")

# ============================================================================
# 2. ADE/FDE Comparison
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(10.8, 3.8), constrained_layout=True)

ade_means = [lbebm_stats['ADE_mean'], exp5_stats['ADE_mean']]
ade_stds = [lbebm_stats['ADE_std'], exp5_stats['ADE_std']]
fde_means = [lbebm_stats['FDE_mean'], exp5_stats['FDE_mean']]
fde_stds = [lbebm_stats['FDE_std'], exp5_stats['FDE_std']]

# Create compact grouped bars for tight comparison
metrics = ['ADE', 'FDE']
x = np.arange(len(metrics))
width = 0.35

# 3DMoTraj
bars1 = ax.bar(x - width/2, ade_means, width, label='3DMoTraj',
               color=COLOR_BASELINE, edgecolor='black', linewidth=0.9, zorder=2)
ax.errorbar(x - width/2, ade_means, ade_stds, fmt='none',
            ecolor='black', elinewidth=1.0, capsize=3, zorder=3)

# DG32-BCAT  
bars2 = ax.bar(x + width/2, fde_means, width, label='DG32-BCAT',
               color=COLOR_OURS, edgecolor='black', linewidth=0.9, zorder=2)
ax.errorbar(x + width/2, fde_means, fde_stds, fmt='none',
            ecolor='black', elinewidth=1.0, capsize=3, zorder=3)

# Add value labels on bars for better visibility
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Error (m)', fontsize=10)
ax.set_title('ADE/FDE Comparison: 3DMoTraj vs DG32-BCAT', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fontsize=9)
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)

plt.savefig(output_dir / 'ade_fde_comparison.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'ade_fde_comparison.pdf', bbox_inches='tight')
plt.close()
print("✓ ade_fde_comparison.png/pdf")

# ============================================================================
# 3. Per-Axis MAE Comparison (X, Y, Z)
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(10.8, 3.8), constrained_layout=True)

mae_x = [lbebm_stats['MAE_X_mean'], exp5_stats['MAE_X_mean']]
mae_y = [lbebm_stats['MAE_Y_mean'], exp5_stats['MAE_Y_mean']]
mae_z = [lbebm_stats['MAE_Z_mean'], exp5_stats['MAE_Z_mean']]

x = np.arange(len(MODEL_NAMES))
width = 0.25

ax.bar(x - width, mae_x, width, color=COLOR_BASELINE, edgecolor='black', 
       linewidth=0.9, label='MAE-X', zorder=2)
ax.bar(x, mae_y, width, color=COLOR_BASELINE, edgecolor='black', 
       linewidth=0.9, alpha=0.85, label='MAE-Y', zorder=2)
ax.bar(x + width, mae_z, width, color=COLOR_BASELINE, edgecolor='black', 
       linewidth=0.9, alpha=0.70, label='MAE-Z', zorder=2)

# Add comparison for DG32-BCAT
ax.bar(x - width + 0.05, mae_x, width, color=COLOR_OURS, edgecolor='black', 
       linewidth=0.9, alpha=0.6, zorder=1)
ax.bar(x + 0.05, mae_y, width, color=COLOR_OURS, edgecolor='black', 
       linewidth=0.9, alpha=0.5, zorder=1)
ax.bar(x + width + 0.05, mae_z, width, color=COLOR_OURS, edgecolor='black', 
       linewidth=0.9, alpha=0.4, zorder=1)

ax.set_ylabel('Error (m)', fontsize=10)
ax.set_title('Per-Axis Error Analysis', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, fontsize=9)
ax.legend(loc='upper left', ncol=3, frameon=True, fontsize=9)
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)

plt.savefig(output_dir / 'per_axis_mae.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'per_axis_mae.pdf', bbox_inches='tight')
plt.close()
print("✓ per_axis_mae.png/pdf")

# ============================================================================
# 4. Per-Agent MAE Comparison
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(10.8, 3.8), constrained_layout=True)

agent_ids = ['Agent 1', 'Agent 2', 'Agent 3']
lbebm_per_agent = lbebm_stats['MAE_per_agent_mean']
exp5_per_agent = exp5_stats['MAE_per_agent_mean']

x = np.arange(len(agent_ids))
width = 0.35

bars1 = ax.bar(x - width/2, lbebm_per_agent, width, label='3DMoTraj',
               color=COLOR_BASELINE, edgecolor='black', linewidth=0.9, zorder=2)
bars2 = ax.bar(x + width/2, exp5_per_agent, width, label='DG32-BCAT',
               color=COLOR_OURS, edgecolor='black', linewidth=0.9, zorder=2)

ax.set_ylabel('MAE (m)', fontsize=10)
ax.set_title('Per-Agent Error Distribution', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(agent_ids, fontsize=9)
ax.legend(loc='upper left', frameon=True, fontsize=9)
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)

plt.savefig(output_dir / 'per_agent_mae.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'per_agent_mae.pdf', bbox_inches='tight')
plt.close()
print("✓ per_agent_mae.png/pdf")

# ============================================================================
# 5. Per-Step Error Progression (Line Plot)
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(10.8, 3.8), constrained_layout=True)

steps = np.arange(1, 11)
lbebm_per_step = lbebm_stats['MAE_per_step_mean']
exp5_per_step = exp5_stats['MAE_per_step_mean']
lbebm_per_step_std = lbebm_stats['MAE_per_step_std']
exp5_per_step_std = exp5_stats['MAE_per_step_std']

ax.plot(steps, lbebm_per_step, 'o-', linewidth=1.8, markersize=5,
        label='3DMoTraj', color=COLOR_BASELINE, zorder=3)
ax.fill_between(steps,
                np.array(lbebm_per_step) - np.array(lbebm_per_step_std),
                np.array(lbebm_per_step) + np.array(lbebm_per_step_std),
                alpha=0.15, color=COLOR_BASELINE, zorder=1)

ax.plot(steps, exp5_per_step, 's-', linewidth=1.8, markersize=5,
        label='DG32-BCAT', color=COLOR_OURS, zorder=3)
ax.fill_between(steps,
                np.array(exp5_per_step) - np.array(exp5_per_step_std),
                np.array(exp5_per_step) + np.array(exp5_per_step_std),
                alpha=0.15, color=COLOR_OURS, zorder=1)

ax.set_xlabel('Prediction Step', fontsize=10)
ax.set_ylabel('MAE (m)', fontsize=10)
ax.set_title('Per-Step Error Progression (Mean ± Std)', fontsize=11)
ax.set_xticks(steps)
ax.set_xticklabels([f'{i}' for i in range(1, 11)], fontsize=9)
ax.legend(loc='upper left', frameon=True, fontsize=9)
ax.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)

plt.savefig(output_dir / 'per_step_error_progression.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'per_step_error_progression.pdf', bbox_inches='tight')
plt.close()
print("✓ per_step_error_progression.png/pdf")

# ============================================================================
# 6. Error Distribution Comparison (Box Plot)
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(10.8, 3.8), constrained_layout=True)

# MAE distribution by step
bp = ax.boxplot([lbebm_stats['MAE_per_step_mean'], exp5_stats['MAE_per_step_mean']],
                 labels=MODEL_NAMES, patch_artist=True, widths=0.55, showfliers=False)
for patch, color in zip(bp['boxes'], [COLOR_BASELINE, COLOR_OURS]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
    patch.set_edgecolor('black')
    patch.set_linewidth(0.9)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.1)
for whisker in bp['whiskers']:
    whisker.set_linewidth(0.9)

ax.set_ylabel('MAE (m)', fontsize=10)
ax.set_title('Per-Step MAE Distribution', fontsize=11)
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)

plt.savefig(output_dir / 'error_distribution_boxplot.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'error_distribution_boxplot.pdf', bbox_inches='tight')
plt.close()
print("✓ error_distribution_boxplot.png/pdf")

# ============================================================================
# 7. Comprehensive Metrics Table
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(12.0, 5.0), constrained_layout=True)
ax.axis('tight')
ax.axis('off')

metrics_data = [
    ['Metric', '3DMoTraj', 'DG32-BCAT', 'Improvement (%)', 'Better'],
    ['MAE (m)',
     f"{lbebm_stats['MAE_mean']:.4f}±{lbebm_stats['MAE_std']:.4f}",
     f"{exp5_stats['MAE_mean']:.4f}±{exp5_stats['MAE_std']:.4f}",
     f"{(1 - exp5_stats['MAE_mean']/lbebm_stats['MAE_mean'])*100:.1f}%",
     '↓' if exp5_stats['MAE_mean'] < lbebm_stats['MAE_mean'] else '↑'],
    ['RMSE (m)',
     f"{lbebm_stats['RMSE_mean']:.4f}±{lbebm_stats['RMSE_std']:.4f}",
     f"{exp5_stats['RMSE_mean']:.4f}±{exp5_stats['RMSE_std']:.4f}",
     f"{(1 - exp5_stats['RMSE_mean']/lbebm_stats['RMSE_mean'])*100:.1f}%",
     '↓' if exp5_stats['RMSE_mean'] < lbebm_stats['RMSE_mean'] else '↑'],
    ['ADE (m)',
     f"{lbebm_stats['ADE_mean']:.4f}±{lbebm_stats['ADE_std']:.4f}",
     f"{exp5_stats['ADE_mean']:.4f}±{exp5_stats['ADE_std']:.4f}",
     f"{(1 - exp5_stats['ADE_mean']/lbebm_stats['ADE_mean'])*100:.1f}%",
     '↓' if exp5_stats['ADE_mean'] < lbebm_stats['ADE_mean'] else '↑'],
    ['FDE (m)',
     f"{lbebm_stats['FDE_mean']:.4f}±{lbebm_stats['FDE_std']:.4f}",
     f"{exp5_stats['FDE_mean']:.4f}±{exp5_stats['FDE_std']:.4f}",
     f"{(1 - exp5_stats['FDE_mean']/lbebm_stats['FDE_mean'])*100:.1f}%",
     '↓' if exp5_stats['FDE_mean'] < lbebm_stats['FDE_mean'] else '↑'],
    ['MAPE (%)',
     f"{lbebm_stats['MAPE_mean']:.4f}±{lbebm_stats['MAPE_std']:.4f}",
     f"{exp5_stats['MAPE_mean']:.4f}±{exp5_stats['MAPE_std']:.4f}",
     f"{(1 - exp5_stats['MAPE_mean']/lbebm_stats['MAPE_mean'])*100:.1f}%",
     '↓' if exp5_stats['MAPE_mean'] < lbebm_stats['MAPE_mean'] else '↑'],
    ['MAE-X (m)',
     f"{lbebm_stats['MAE_X_mean']:.4f}",
     f"{exp5_stats['MAE_X_mean']:.4f}",
     f"{(1 - exp5_stats['MAE_X_mean']/lbebm_stats['MAE_X_mean'])*100:.1f}%",
     '↓' if exp5_stats['MAE_X_mean'] < lbebm_stats['MAE_X_mean'] else '↑'],
    ['MAE-Y (m)',
     f"{lbebm_stats['MAE_Y_mean']:.4f}",
     f"{exp5_stats['MAE_Y_mean']:.4f}",
     f"{(1 - exp5_stats['MAE_Y_mean']/lbebm_stats['MAE_Y_mean'])*100:.1f}%",
     '↓' if exp5_stats['MAE_Y_mean'] < lbebm_stats['MAE_Y_mean'] else '↑'],
    ['MAE-Z (m)',
     f"{lbebm_stats['MAE_Z_mean']:.4f}",
     f"{exp5_stats['MAE_Z_mean']:.4f}",
     f"{(1 - exp5_stats['MAE_Z_mean']/lbebm_stats['MAE_Z_mean'])*100:.1f}%",
     '↓' if exp5_stats['MAE_Z_mean'] < lbebm_stats['MAE_Z_mean'] else '↑'],
]

table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                colWidths=[0.18, 0.22, 0.22, 0.18, 0.10])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Header row styling
for i in range(5):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Data row styling
for i in range(1, len(metrics_data)):
    color = '#ECF0F1' if i % 2 == 0 else 'white'
    for j in range(5):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_edgecolor('#95A5A6')
        table[(i, j)].set_linewidth(0.5)
        if j == 3:  # Improvement column
            table[(i, j)].set_text_props(weight='bold')

plt.suptitle('Comprehensive Metrics Comparison Table (Lower is Better)',
             fontsize=11, fontweight='bold', y=0.98)
plt.savefig(output_dir / 'metrics_comparison_table.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'metrics_comparison_table.pdf', bbox_inches='tight')
plt.close()
print("✓ metrics_comparison_table.png/pdf")

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "="*75)
print("COMPARISON SUMMARY - 3DMoTraj vs DG32-BCAT")
print("="*75)
print(f"\nModel: 3DMoTraj (Baseline)")
print(f"  MAE:  {lbebm_stats['MAE_mean']:.4f} ± {lbebm_stats['MAE_std']:.4f} m")
print(f"  RMSE: {lbebm_stats['RMSE_mean']:.4f} ± {lbebm_stats['RMSE_std']:.4f} m")
print(f"  ADE:  {lbebm_stats['ADE_mean']:.4f} ± {lbebm_stats['ADE_std']:.4f} m")
print(f"  FDE:  {lbebm_stats['FDE_mean']:.4f} ± {lbebm_stats['FDE_std']:.4f} m")
print(f"  MAPE: {lbebm_stats['MAPE_mean']:.4f} ± {lbebm_stats['MAPE_std']:.4f} %")

print(f"\nModel: DG32-BCAT (Our Method)")
print(f"  MAE:  {exp5_stats['MAE_mean']:.4f} ± {exp5_stats['MAE_std']:.4f} m")
print(f"  RMSE: {exp5_stats['RMSE_mean']:.4f} ± {exp5_stats['RMSE_std']:.4f} m")
print(f"  ADE:  {exp5_stats['ADE_mean']:.4f} ± {exp5_stats['ADE_std']:.4f} m")
print(f"  FDE:  {exp5_stats['FDE_mean']:.4f} ± {exp5_stats['FDE_std']:.4f} m")
print(f"  MAPE: {exp5_stats['MAPE_mean']:.4f} ± {exp5_stats['MAPE_std']:.4f} %")

mae_improve = (1 - exp5_stats['MAE_mean']/lbebm_stats['MAE_mean'])*100
ade_improve = (1 - exp5_stats['ADE_mean']/lbebm_stats['ADE_mean'])*100
fde_improve = (1 - exp5_stats['FDE_mean']/lbebm_stats['FDE_mean'])*100

print(f"\nPerformance Improvement (DG32-BCAT vs 3DMoTraj):")
print(f"  MAE:  {mae_improve:+.1f}% {'✓ Better' if mae_improve > 0 else '✗ Worse'}")
print(f"  ADE:  {ade_improve:+.1f}% {'✓ Better' if ade_improve > 0 else '✗ Worse'}")
print(f"  FDE:  {fde_improve:+.1f}% {'✓ Better' if fde_improve > 0 else '✗ Worse'}")
print("="*75)

# ============================================================================
# Generate CSV Summary Table
# ============================================================================
import csv

csv_path = output_dir / 'metrics_comparison.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Header
    writer.writerow(['Metric', '3DMoTraj (Mean)', '3DMoTraj (Std)', 'DG32-BCAT (Mean)', 'DG32-BCAT (Std)', 'Improvement (%)', 'Better'])
    
    # Data rows
    metrics_list = [
        ('MAE (m)', lbebm_stats['MAE_mean'], lbebm_stats['MAE_std'], exp5_stats['MAE_mean'], exp5_stats['MAE_std']),
        ('RMSE (m)', lbebm_stats['RMSE_mean'], lbebm_stats['RMSE_std'], exp5_stats['RMSE_mean'], exp5_stats['RMSE_std']),
        ('ADE (m)', lbebm_stats['ADE_mean'], lbebm_stats['ADE_std'], exp5_stats['ADE_mean'], exp5_stats['ADE_std']),
        ('FDE (m)', lbebm_stats['FDE_mean'], lbebm_stats['FDE_std'], exp5_stats['FDE_mean'], exp5_stats['FDE_std']),
        ('MAPE (%)', lbebm_stats['MAPE_mean'], lbebm_stats['MAPE_std'], exp5_stats['MAPE_mean'], exp5_stats['MAPE_std']),
        ('MAE-X (m)', lbebm_stats['MAE_X_mean'], 0, exp5_stats['MAE_X_mean'], 0),
        ('MAE-Y (m)', lbebm_stats['MAE_Y_mean'], 0, exp5_stats['MAE_Y_mean'], 0),
        ('MAE-Z (m)', lbebm_stats['MAE_Z_mean'], 0, exp5_stats['MAE_Z_mean'], 0),
    ]
    
    for metric_name, baseline_mean, baseline_std, our_mean, our_std in metrics_list:
        improvement = (1 - our_mean / baseline_mean) * 100 if baseline_mean != 0 else 0
        better = '↓' if our_mean < baseline_mean else '↑'
        writer.writerow([
            metric_name,
            f'{baseline_mean:.6f}',
            f'{baseline_std:.6f}',
            f'{our_mean:.6f}',
            f'{our_std:.6f}',
            f'{improvement:.2f}%',
            better
        ])

print(f"\n✓ All figures saved to: {output_dir}")
print("\nGenerated Publication-Ready Files:")
print("  ✓ overall_metrics.png/pdf")
print("  ✓ ade_fde_comparison.png/pdf")
print("  ✓ per_axis_mae.png/pdf")
print("  ✓ per_agent_mae.png/pdf")
print("  ✓ per_step_error_progression.png/pdf")
print("  ✓ error_distribution_boxplot.png/pdf")
print("  ✓ metrics_comparison_table.png/pdf")
print("  ✓ metrics_comparison.csv")
print("="*75)
