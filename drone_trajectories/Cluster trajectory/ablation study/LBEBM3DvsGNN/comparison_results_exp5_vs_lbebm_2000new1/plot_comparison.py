#!/usr/bin/env python3
"""
Plot comparison between 3DMoTraj and Exp5 models
Generate publication-ready figures
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']

# Load JSON data
json_file = Path(__file__).parent / 'ablation_summary.json'
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract experiment data
exps = data['experiments']
lbebm_stats = exps['LBEBM3D']['aggregate_stats']
exp5_stats = exps['Exp5_Full']['aggregate_stats']

output_dir = Path(__file__).parent / 'comparison_plots'
output_dir.mkdir(exist_ok=True)

print(f"Output directory: {output_dir}")

# ============================================================================
# 1. Overall Metrics Comparison (MAE, RMSE, MAPE)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

models = ['3DMoTraj', 'DG32-BCAT']
mae_means = [lbebm_stats['MAE_mean'], exp5_stats['MAE_mean']]
mae_stds = [lbebm_stats['MAE_std'], exp5_stats['MAE_std']]
rmse_means = [lbebm_stats['RMSE_mean'], exp5_stats['RMSE_mean']]
rmse_stds = [lbebm_stats['RMSE_std'], exp5_stats['RMSE_std']]
mape_means = [lbebm_stats['MAPE_mean'], exp5_stats['MAPE_mean']]
mape_stds = [lbebm_stats['MAPE_std'], exp5_stats['MAPE_std']]

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, mae_means, width, label='MAE', color='#E74C3C', alpha=0.8, edgecolor='black')
ax.errorbar(x - width, mae_means, mae_stds, fmt='none', color='black', capsize=5, linewidth=1.5)

bars2 = ax.bar(x, rmse_means, width, label='RMSE', color='#3498DB', alpha=0.8, edgecolor='black')
ax.errorbar(x, rmse_means, rmse_stds, fmt='none', color='black', capsize=5, linewidth=1.5)

bars3 = ax.bar(x + width, mape_means, width, label='MAPE (%)', color='#9B59B6', alpha=0.8, edgecolor='black')
ax.errorbar(x + width, mape_means, mape_stds, fmt='none', color='black', capsize=5, linewidth=1.5)

ax.set_ylabel('Error (m or %)', fontsize=12, fontweight='bold')
ax.set_title('Overall Metrics Comparison: MAE, RMSE, MAPE', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'overall_metrics.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'overall_metrics.pdf', bbox_inches='tight')
plt.close()
print("✓ overall_metrics.png/pdf")

# ============================================================================
# 2. ADE/FDE Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

ade_means = [lbebm_stats['ADE_mean'], exp5_stats['ADE_mean']]
ade_stds = [lbebm_stats['ADE_std'], exp5_stats['ADE_std']]
fde_means = [lbebm_stats['FDE_mean'], exp5_stats['FDE_mean']]
fde_stds = [lbebm_stats['FDE_std'], exp5_stats['FDE_std']]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, ade_means, width, label='ADE (avg)', color='#27AE60', alpha=0.8, edgecolor='black')
ax.errorbar(x - width/2, ade_means, ade_stds, fmt='none', color='black', capsize=5, linewidth=1.5)

bars2 = ax.bar(x + width/2, fde_means, width, label='FDE (final)', color='#E67E22', alpha=0.8, edgecolor='black')
ax.errorbar(x + width/2, fde_means, fde_stds, fmt='none', color='black', capsize=5, linewidth=1.5)

ax.set_ylabel('Error (m)', fontsize=12, fontweight='bold')
ax.set_title('Trajectory Metrics: ADE vs FDE', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'ade_fde_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'ade_fde_comparison.pdf', bbox_inches='tight')
plt.close()
print("✓ ade_fde_comparison.png/pdf")

# ============================================================================
# 3. Per-Axis MAE Comparison (X, Y, Z)
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

axes = ['X (Longitudinal)', 'Y (Lateral)', 'Z (Vertical)']
mae_x = [lbebm_stats['MAE_X_mean'], exp5_stats['MAE_X_mean']]
mae_y = [lbebm_stats['MAE_Y_mean'], exp5_stats['MAE_Y_mean']]
mae_z = [lbebm_stats['MAE_Z_mean'], exp5_stats['MAE_Z_mean']]

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, mae_x, width, label='MAE-X', color='#E74C3C', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, mae_y, width, label='MAE-Y', color='#3498DB', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, mae_z, width, label='MAE-Z', color='#9B59B6', alpha=0.8, edgecolor='black')

ax.set_ylabel('Error (m)', fontsize=12, fontweight='bold')
ax.set_title('Per-Axis Error Analysis: MAE-X, MAE-Y, MAE-Z', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'per_axis_mae.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'per_axis_mae.pdf', bbox_inches='tight')
plt.close()
print("✓ per_axis_mae.png/pdf")

# ============================================================================
# 4. Per-Agent MAE Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

agent_ids = ['Agent 1', 'Agent 2', 'Agent 3']
lbebm_per_agent = lbebm_stats['MAE_per_agent_mean']
exp5_per_agent = exp5_stats['MAE_per_agent_mean']

x = np.arange(len(agent_ids))
width = 0.35

bars1 = ax.bar(x - width/2, lbebm_per_agent, width, label='3DMoTraj', color='#E74C3C', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, exp5_per_agent, width, label='DG32-BCAT', color='#3498DB', alpha=0.8, edgecolor='black')

ax.set_ylabel('MAE (m)', fontsize=12, fontweight='bold')
ax.set_title('Per-Agent Error Distribution', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(agent_ids, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'per_agent_mae.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'per_agent_mae.pdf', bbox_inches='tight')
plt.close()
print("✓ per_agent_mae.png/pdf")

# ============================================================================
# 5. Per-Step Error Progression (Line Plot)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

steps = np.arange(1, 11)
lbebm_per_step = lbebm_stats['MAE_per_step_mean']
exp5_per_step = exp5_stats['MAE_per_step_mean']
lbebm_per_step_std = lbebm_stats['MAE_per_step_std']
exp5_per_step_std = exp5_stats['MAE_per_step_std']

ax.plot(steps, lbebm_per_step, 'o-', linewidth=2.5, markersize=8, 
        label='3DMoTraj', color='#E74C3C')
ax.fill_between(steps, 
                np.array(lbebm_per_step) - np.array(lbebm_per_step_std),
                np.array(lbebm_per_step) + np.array(lbebm_per_step_std),
                alpha=0.2, color='#E74C3C')

ax.plot(steps, exp5_per_step, 's-', linewidth=2.5, markersize=8,
        label='DG32-BCAT', color='#3498DB')
ax.fill_between(steps,
                np.array(exp5_per_step) - np.array(exp5_per_step_std),
                np.array(exp5_per_step) + np.array(exp5_per_step_std),
                alpha=0.2, color='#3498DB')

ax.set_xlabel('Prediction Step (0.1s per step)', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE (m)', fontsize=12, fontweight='bold')
ax.set_title('Per-Step Error Progression (Mean ± Std)', fontsize=14, fontweight='bold')
ax.set_xticks(steps)
ax.set_xticklabels([f'Step {i}' for i in range(1, 11)], fontsize=10)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'per_step_error_progression.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'per_step_error_progression.pdf', bbox_inches='tight')
plt.close()
print("✓ per_step_error_progression.png/pdf")

# ============================================================================
# 6. Error Distribution Comparison (Box Plot)
# ============================================================================
fig, axes_plot = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# MAE distribution by step
ax = axes_plot[0]
bp1 = ax.boxplot([lbebm_stats['MAE_per_step_mean'], exp5_stats['MAE_per_step_mean']],
                   tick_labels=['3DMoTraj', 'DG32-BCAT'],
                   patch_artist=True,
                   widths=0.6)
colors = ['#E74C3C', '#3498DB']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.0)
for median in bp1['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)
ax.set_ylabel('MAE per step (m)', fontsize=11, fontweight='bold')
ax.set_title('MAE per Step Distribution', fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

# Per-agent distribution
ax = axes_plot[1]
bp2 = ax.boxplot([lbebm_stats['MAE_per_agent_mean'], exp5_stats['MAE_per_agent_mean']],
                   tick_labels=['3DMoTraj', 'DG32-BCAT'],
                   patch_artist=True,
                   widths=0.6)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.0)
for median in bp2['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)
ax.set_ylabel('MAE per agent (m)', fontsize=11, fontweight='bold')
ax.set_title('MAE per Agent Distribution', fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution_boxplot.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'error_distribution_boxplot.pdf', bbox_inches='tight')
plt.close()
print("✓ error_distribution_boxplot.png/pdf")

# ============================================================================
# 7. Comprehensive Metrics Table
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
ax.axis('tight')
ax.axis('off')

metrics_data = [
    ['Metric', '3DMoTraj', 'DG32-BCAT', 'Improvement %'],
    ['MAE (m)', 
     f"{lbebm_stats['MAE_mean']:.4f} ± {lbebm_stats['MAE_std']:.4f}",
     f"{exp5_stats['MAE_mean']:.4f} ± {exp5_stats['MAE_std']:.4f}",
     f"{(1 - exp5_stats['MAE_mean']/lbebm_stats['MAE_mean'])*100:.1f}%"],
    ['RMSE (m)',
     f"{lbebm_stats['RMSE_mean']:.4f} ± {lbebm_stats['RMSE_std']:.4f}",
     f"{exp5_stats['RMSE_mean']:.4f} ± {exp5_stats['RMSE_std']:.4f}",
     f"{(1 - exp5_stats['RMSE_mean']/lbebm_stats['RMSE_mean'])*100:.1f}%"],
    ['ADE (m)',
     f"{lbebm_stats['ADE_mean']:.4f} ± {lbebm_stats['ADE_std']:.4f}",
     f"{exp5_stats['ADE_mean']:.4f} ± {exp5_stats['ADE_std']:.4f}",
     f"{(1 - exp5_stats['ADE_mean']/lbebm_stats['ADE_mean'])*100:.1f}%"],
    ['FDE (m)',
     f"{lbebm_stats['FDE_mean']:.4f} ± {lbebm_stats['FDE_std']:.4f}",
     f"{exp5_stats['FDE_mean']:.4f} ± {exp5_stats['FDE_std']:.4f}",
     f"{(1 - exp5_stats['FDE_mean']/lbebm_stats['FDE_mean'])*100:.1f}%"],
    ['MAPE (%)',
     f"{lbebm_stats['MAPE_mean']:.4f} ± {lbebm_stats['MAPE_std']:.4f}",
     f"{exp5_stats['MAPE_mean']:.4f} ± {exp5_stats['MAPE_std']:.4f}",
     f"{(1 - exp5_stats['MAPE_mean']/lbebm_stats['MAPE_mean'])*100:.1f}%"],
    ['MAE-X (m)',
     f"{lbebm_stats['MAE_X_mean']:.4f}",
     f"{exp5_stats['MAE_X_mean']:.4f}",
     f"{(1 - exp5_stats['MAE_X_mean']/lbebm_stats['MAE_X_mean'])*100:.1f}%"],
    ['MAE-Y (m)',
     f"{lbebm_stats['MAE_Y_mean']:.4f}",
     f"{exp5_stats['MAE_Y_mean']:.4f}",
     f"{(1 - exp5_stats['MAE_Y_mean']/lbebm_stats['MAE_Y_mean'])*100:.1f}%"],
    ['MAE-Z (m)',
     f"{lbebm_stats['MAE_Z_mean']:.4f}",
     f"{exp5_stats['MAE_Z_mean']:.4f}",
     f"{(1 - exp5_stats['MAE_Z_mean']/lbebm_stats['MAE_Z_mean'])*100:.1f}%"],
]

table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Color header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(metrics_data)):
    color = '#ECF0F1' if i % 2 == 0 else 'white'
    for j in range(4):
        table[(i, j)].set_facecolor(color)
        if j == 3:  # Improvement column
            table[(i, j)].set_text_props(weight='bold', color='#27AE60')

plt.title('Comprehensive Metrics Comparison Table\n(Lower is Better)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison_table.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'metrics_comparison_table.pdf', bbox_inches='tight')
plt.close()
print("✓ metrics_comparison_table.png/pdf")

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"\nModel: 3DMoTraj")
print(f"  MAE:  {lbebm_stats['MAE_mean']:.4f} ± {lbebm_stats['MAE_std']:.4f} m")
print(f"  RMSE: {lbebm_stats['RMSE_mean']:.4f} ± {lbebm_stats['RMSE_std']:.4f} m")
print(f"  ADE:  {lbebm_stats['ADE_mean']:.4f} ± {lbebm_stats['ADE_std']:.4f} m")
print(f"  FDE:  {lbebm_stats['FDE_mean']:.4f} ± {lbebm_stats['FDE_std']:.4f} m")
print(f"  MAPE: {lbebm_stats['MAPE_mean']:.4f} ± {lbebm_stats['MAPE_std']:.4f} %")

print(f"\nModel: Exp5 (Full Model)")
print(f"  MAE:  {exp5_stats['MAE_mean']:.4f} ± {exp5_stats['MAE_std']:.4f} m")
print(f"  RMSE: {exp5_stats['RMSE_mean']:.4f} ± {exp5_stats['RMSE_std']:.4f} m")
print(f"  ADE:  {exp5_stats['ADE_mean']:.4f} ± {exp5_stats['ADE_std']:.4f} m")
print(f"  FDE:  {exp5_stats['FDE_mean']:.4f} ± {exp5_stats['FDE_std']:.4f} m")
print(f"  MAPE: {exp5_stats['MAPE_mean']:.4f} ± {exp5_stats['MAPE_std']:.4f} %")

mae_improve = (1 - exp5_stats['MAE_mean']/lbebm_stats['MAE_mean'])*100
ade_improve = (1 - exp5_stats['ADE_mean']/lbebm_stats['ADE_mean'])*100
fde_improve = (1 - exp5_stats['FDE_mean']/lbebm_stats['FDE_mean'])*100

print(f"\nImprovement (Exp5 vs 3DMoTraj):")
print(f"  MAE:  {mae_improve:+.1f}% {'↑ Better' if mae_improve > 0 else '↓ Worse'}")
print(f"  ADE:  {ade_improve:+.1f}% {'↑ Better' if ade_improve > 0 else '↓ Worse'}")
print(f"  FDE:  {fde_improve:+.1f}% {'↑ Better' if fde_improve > 0 else '↓ Worse'}")
print("\n" + "="*70)

print(f"\nAll figures saved to: {output_dir}")
print("\nGenerated files:")
print("  ✓ overall_metrics.png/pdf")
print("  ✓ ade_fde_comparison.png/pdf")
print("  ✓ per_axis_mae.png/pdf")
print("  ✓ per_agent_mae.png/pdf")
print("  ✓ per_step_error_progression.png/pdf")
print("  ✓ error_distribution_boxplot.png/pdf")
print("  ✓ metrics_comparison_table.png/pdf")
