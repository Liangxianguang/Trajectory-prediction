#!/usr/bin/env python3
"""
消融实验对比可视化脚本 - Publication Ready
Ablation Study Comparison with Professional Styling
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import csv

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
    base = ['#E74C3C', '#3498DB', '#F39C12', '#27AE60', '#9B59B6', '#2C3E50']
    if n <= len(base):
        return base[:n]
    return (base * ((n // len(base)) + 1))[:n]

# ============================================================================
# Data Definition
# ============================================================================
experiments = {
    'Exp1': {
        'id': 1,
        'name': 'Baseline GRU\n(No Enhancements)',
        'GAT': False,
        'Features': False,
        'BiCA': False,
        'MAE': 0.18227435286529362,
        'MAE_std': 0.21126856403793037,
        'RMSE': 0.22490701648127287,
        'RMSE_std': 0.2613114226932803,
        'ADE': 0.18227435286529362,
        'ADE_std': 0.21126856403793037,
        'FDE': 0.4134756118413061,
        'FDE_std': 0.4928940254139765,
        'MAE_X': 0.09732065942359623,
        'MAE_Y': 0.09626485842594411,
        'MAE_Z': 0.08222829529689625,
    },
    'Exp2': {
        'id': 2,
        'name': 'Features + BiCA\n(No GAT)',
        'GAT': False,
        'Features': True,
        'BiCA': True,
        'MAE': 0.11897107543237508,
        'MAE_std': 0.13055362474613794,
        'RMSE': 0.1441525536607951,
        'RMSE_std': 0.16016179227589203,
        'ADE': 0.11897107543237508,
        'ADE_std': 0.13055362474613794,
        'FDE': 0.2549774806108326,
        'FDE_std': 0.3006202939318239,
        'MAE_X': 0.06638090324529912,
        'MAE_Y': 0.06416763166931924,
        'MAE_Z': 0.05012260203086771,
    },
    'Exp3': {
        'id': 3,
        'name': 'GAT + BiCA\n(No Features)',
        'GAT': True,
        'Features': False,
        'BiCA': True,
        'MAE': 0.17848871530313046,
        'MAE_std': 0.2090005208977654,
        'RMSE': 0.21998588318936527,
        'RMSE_std': 0.2582886463908101,
        'ADE': 0.17848871530313046,
        'ADE_std': 0.2090005208977654,
        'FDE': 0.40325969784706833,
        'FDE_std': 0.48659858056285676,
        'MAE_X': 0.0950571438963525,
        'MAE_Y': 0.09429576209117659,
        'MAE_Z': 0.08130525422410574,
    },
    'Exp4': {
        'id': 4,
        'name': 'GAT + Features\n(No BiCA)',
        'GAT': True,
        'Features': True,
        'BiCA': False,
        'MAE': 0.11541079707955941,
        'MAE_std': 0.09325642685792335,
        'RMSE': 0.13967633253056555,
        'RMSE_std': 0.11282006145991645,
        'ADE': 0.11541079707955941,
        'ADE_std': 0.09325642685792335,
        'FDE': 0.24760575945582242,
        'FDE_std': 0.2098059202237851,
        'MAE_X': 0.06329807130072732,
        'MAE_Y': 0.0647764295306988,
        'MAE_Z': 0.044877685566316355,
    },
    'Exp5': {
        'id': 5,
        'name': 'Full Model\n(DG32-BCAT)',
        'GAT': True,
        'Features': True,
        'BiCA': True,
        'MAE': 0.09903461142117158,
        'MAE_std': 0.07611374855980217,
        'RMSE': 0.11923696516919881,
        'RMSE_std': 0.09217083986042751,
        'ADE': 0.09903461142117158,
        'ADE_std': 0.07611374855980217,
        'FDE': 0.2063685186188668,
        'FDE_std': 0.17320386269822066,
        'MAE_X': 0.05386405574693345,
        'MAE_Y': 0.05452390086674132,
        'MAE_Z': 0.04014760680485051,
    }
}

output_dir = Path(__file__).parent / 'ablation_analysis_plots'
output_dir.mkdir(exist_ok=True)

print(f"Output directory: {output_dir}")
print("Generating ablation study comparison figures...\n")

# Calculate improvements relative to Exp1 (baseline)
baseline = experiments['Exp1']
colors = palette(5)

# ============================================================================
# 1. Overall Metrics Comparison (MAE, RMSE, ADE, FDE)
# ============================================================================
paper_style()
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

metrics_configs = [
    ('MAE (m)', [exp['MAE'] for exp in experiments.values()], 
     [exp['MAE_std'] for exp in experiments.values()], 0),
    ('RMSE (m)', [exp['RMSE'] for exp in experiments.values()],
     [exp['RMSE_std'] for exp in experiments.values()], 1),
    ('ADE (m)', [exp['ADE'] for exp in experiments.values()],
     [exp['ADE_std'] for exp in experiments.values()], 2),
    ('FDE (m)', [exp['FDE'] for exp in experiments.values()],
     [exp['FDE_std'] for exp in experiments.values()], 3),
]

exp_names = ['Exp1\n(Base)', 'Exp2\n(Feat+BiCA)', 'Exp3\n(GAT+BiCA)', 'Exp4\n(GAT+Feat)', 'Exp5\n(Full)']
x = np.arange(len(experiments))

for metric_name, values, stds, idx in metrics_configs:
    ax = axes[idx // 2, idx % 2]
    
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.9, zorder=2, alpha=0.85)
    ax.errorbar(x, values, stds, fmt='none', ecolor='black', elinewidth=1.0, capsize=3, zorder=3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=9)
    ax.grid(True, axis='y', alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    
    # Highlight the best (lowest for error metrics)
    best_idx = np.argmin(values)
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(2.0)

plt.suptitle('Ablation Study: Overall Metrics Comparison', fontsize=12, fontweight='bold', y=0.995)
plt.savefig(output_dir / 'ablation_overall_metrics.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'ablation_overall_metrics.pdf', bbox_inches='tight')
plt.close()
print("✓ ablation_overall_metrics.png/pdf")

# ============================================================================
# 2. Improvement over Baseline (Exp1)
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

metrics = ['MAE', 'RMSE', 'ADE', 'FDE']
x = np.arange(len(metrics))
width = 0.15

baseline_vals = [baseline['MAE'], baseline['RMSE'], baseline['ADE'], baseline['FDE']]

for i, (exp_key, exp) in enumerate(list(experiments.items())[1:], 1):  # Skip baseline
    exp_vals = [exp['MAE'], exp['RMSE'], exp['ADE'], exp['FDE']]
    improvements = [(1 - v / b) * 100 for v, b in zip(exp_vals, baseline_vals)]
    
    offset = (i - 1) * width - 1.5 * width
    bars = ax.bar(x + offset, improvements, width, label=exp_key, 
                 color=colors[i], edgecolor='black', linewidth=0.8, alpha=0.85, zorder=2)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Improvement over Baseline (%)', fontsize=10, fontweight='bold')
ax.set_title('Performance Improvement Relative to Exp1 (Baseline GRU)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10, fontweight='bold')
ax.legend(loc='upper left', ncol=4, frameon=True, fontsize=9)
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)

plt.savefig(output_dir / 'ablation_improvement_over_baseline.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'ablation_improvement_over_baseline.pdf', bbox_inches='tight')
plt.close()
print("✓ ablation_improvement_over_baseline.png/pdf")

# ============================================================================
# 3. Per-Axis Error (MAE-X, MAE-Y, MAE-Z)
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

mae_x = [exp['MAE_X'] for exp in experiments.values()]
mae_y = [exp['MAE_Y'] for exp in experiments.values()]
mae_z = [exp['MAE_Z'] for exp in experiments.values()]

x = np.arange(len(experiments))
width = 0.25

ax.bar(x - width, mae_x, width, label='MAE-X', color='#E74C3C', 
       edgecolor='black', linewidth=0.8, alpha=0.85, zorder=2)
ax.bar(x, mae_y, width, label='MAE-Y', color='#3498DB',
       edgecolor='black', linewidth=0.8, alpha=0.85, zorder=2)
ax.bar(x + width, mae_z, width, label='MAE-Z', color='#F39C12',
       edgecolor='black', linewidth=0.8, alpha=0.85, zorder=2)

ax.set_ylabel('Error (m)', fontsize=10, fontweight='bold')
ax.set_title('Per-Axis Error Analysis (X, Y, Z Coordinates)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, fontsize=9)
ax.legend(loc='upper right', ncol=3, frameon=True, fontsize=9)
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)

plt.savefig(output_dir / 'ablation_per_axis_mae.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'ablation_per_axis_mae.pdf', bbox_inches='tight')
plt.close()
print("✓ ablation_per_axis_mae.png/pdf")

# ============================================================================
# 4. Component Contribution Analysis (Stacked Bar)
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

# Calculate contribution relative to baseline
contributions = []
for exp in experiments.values():
    improvement = (1 - exp['MAE'] / baseline['MAE']) * 100
    contributions.append(improvement)

colors_contrib = ['#E74C3C' if c < 0 else '#27AE60' for c in contributions]

bars = ax.bar(range(len(experiments)), contributions, color=colors_contrib, 
              edgecolor='black', linewidth=0.9, alpha=0.85, zorder=2)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, contributions)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
           fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('MAE Improvement (%)', fontsize=10, fontweight='bold')
ax.set_title('Component Contribution to MAE Reduction (Relative to Baseline)', fontsize=11, fontweight='bold')
ax.set_xticks(range(len(experiments)))
ax.set_xticklabels(exp_names, fontsize=9)
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)

plt.savefig(output_dir / 'ablation_contribution_analysis.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'ablation_contribution_analysis.pdf', bbox_inches='tight')
plt.close()
print("✓ ablation_contribution_analysis.png/pdf")

# ============================================================================
# 5. Comprehensive Metrics Table
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
ax.axis('tight')
ax.axis('off')

# Build table data
table_data = [
    ['Experiment', 'GAT', 'Features', 'BiCA', 'MAE (m)', 'RMSE (m)', 'ADE (m)', 'FDE (m)', 
     'RMSE Improve', 'ADE Improve', 'FDE Improve']
]

for exp_key, exp in experiments.items():
    gat = '✓' if exp['GAT'] else '×'
    feat = '✓' if exp['Features'] else '×'
    bica = '✓' if exp['BiCA'] else '×'
    
    rmse_improve = (1 - exp['RMSE'] / baseline['RMSE']) * 100
    ade_improve = (1 - exp['ADE'] / baseline['ADE']) * 100
    fde_improve = (1 - exp['FDE'] / baseline['FDE']) * 100
    
    table_data.append([
        exp['name'].replace('\n', ' '),
        gat,
        feat,
        bica,
        f"{exp['MAE']:.4f}",
        f"{exp['RMSE']:.4f}",
        f"{exp['ADE']:.4f}",
        f"{exp['FDE']:.4f}",
        f"{rmse_improve:+.1f}%",
        f"{ade_improve:+.1f}%",
        f"{fde_improve:+.1f}%",
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.06, 0.08, 0.07, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Header row styling
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Data row styling
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        # Alternate row colors
        bg_color = '#ECF0F1' if i % 2 == 0 else 'white'
        table[(i, j)].set_facecolor(bg_color)
        table[(i, j)].set_edgecolor('#95A5A6')
        table[(i, j)].set_linewidth(0.5)
        
        # Highlight best values in improvement columns
        if j >= 8 and i > 1:  # Improvement columns
            val = table_data[i][j]
            if '+' in str(val):
                try:
                    pct = float(val.split('%')[0].replace('+', ''))
                    if pct == max([float(table_data[k][j].split('%')[0].replace('+', '')) 
                                 for k in range(2, len(table_data))]):
                        table[(i, j)].set_facecolor('#D5F4E6')
                except:
                    pass

# 标题位置下移，在图表下方
fig.suptitle('Ablation Study: Comprehensive Metrics Comparison', 
             fontsize=12, fontweight='bold', y=0.02, x=0.5)
plt.savefig(output_dir / 'ablation_metrics_table.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'ablation_metrics_table.pdf', bbox_inches='tight')
plt.close()
print("✓ ablation_metrics_table.png/pdf")

# ============================================================================
# 6. Feature/Component Heatmap
# ============================================================================
paper_style()
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

# Create heatmap data: improvements for each feature combination
feature_impact = []
exp_names_short = []
for exp_key, exp in experiments.items():
    mae_improve = (1 - exp['MAE'] / baseline['MAE']) * 100
    feature_impact.append([mae_improve])
    exp_names_short.append(exp_key)

feature_impact = np.array(feature_impact).T

im = ax.imshow(feature_impact, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=50)

ax.set_xticks(range(len(exp_names_short)))
ax.set_xticklabels(exp_names_short, fontsize=10, fontweight='bold')
ax.set_yticks([0])
ax.set_yticklabels(['MAE Improvement (%)'], fontsize=10, fontweight='bold')

# Add text annotations
for i in range(len(exp_names_short)):
    val = feature_impact[0, i]
    ax.text(i, 0, f'{val:.1f}%', ha='center', va='center',
           color='white' if abs(val) > 25 else 'black', fontsize=11, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('MAE Improvement (%)', fontsize=10, fontweight='bold')

ax.set_title('Feature Component Contribution Heatmap', fontsize=11, fontweight='bold', pad=10)
plt.savefig(output_dir / 'ablation_component_heatmap.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'ablation_component_heatmap.pdf', bbox_inches='tight')
plt.close()
print("✓ ablation_component_heatmap.png/pdf")

# ============================================================================
# 7. Export CSV Summary
# ============================================================================
csv_path = output_dir / 'ablation_summary.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Write header
    writer.writerow(['Experiment', 'GAT', 'Features', 'BiCA', 
                    'MAE (m)', 'MAE_std', 'RMSE (m)', 'RMSE_std',
                    'ADE (m)', 'ADE_std', 'FDE (m)', 'FDE_std',
                    'MAE_X', 'MAE_Y', 'MAE_Z',
                    'MAE vs Baseline (%)', 'ADE vs Baseline (%)', 'FDE vs Baseline (%)'])
    
    # Write data
    for exp_key, exp in experiments.items():
        gat = 'Yes' if exp['GAT'] else 'No'
        feat = 'Yes' if exp['Features'] else 'No'
        bica = 'Yes' if exp['BiCA'] else 'No'
        
        mae_improve = (1 - exp['MAE'] / baseline['MAE']) * 100
        ade_improve = (1 - exp['ADE'] / baseline['ADE']) * 100
        fde_improve = (1 - exp['FDE'] / baseline['FDE']) * 100
        
        writer.writerow([
            exp_key,
            gat, feat, bica,
            f"{exp['MAE']:.6f}", f"{exp['MAE_std']:.6f}",
            f"{exp['RMSE']:.6f}", f"{exp['RMSE_std']:.6f}",
            f"{exp['ADE']:.6f}", f"{exp['ADE_std']:.6f}",
            f"{exp['FDE']:.6f}", f"{exp['FDE_std']:.6f}",
            f"{exp['MAE_X']:.6f}", f"{exp['MAE_Y']:.6f}", f"{exp['MAE_Z']:.6f}",
            f"{mae_improve:.2f}", f"{ade_improve:.2f}", f"{fde_improve:.2f}",
        ])

print(f"✓ ablation_summary.csv")

# ============================================================================
# Print Summary Report
# ============================================================================
print("\n" + "="*90)
print("ABLATION STUDY SUMMARY REPORT")
print("="*90)

print("\n📊 BASELINE PERFORMANCE (Exp1: GRU Only)")
print(f"  MAE:  {baseline['MAE']:.6f} ± {baseline['MAE_std']:.6f} m")
print(f"  RMSE: {baseline['RMSE']:.6f} ± {baseline['RMSE_std']:.6f} m")
print(f"  ADE:  {baseline['ADE']:.6f} ± {baseline['ADE_std']:.6f} m")
print(f"  FDE:  {baseline['FDE']:.6f} ± {baseline['FDE_std']:.6f} m")

print("\n📈 COMPONENT CONTRIBUTION ANALYSIS")
for exp_key, exp in list(experiments.items())[1:]:
    gat_contrib = " [+GAT]" if exp['GAT'] else ""
    feat_contrib = " [+Features]" if exp['Features'] else ""
    bica_contrib = " [+BiCA]" if exp['BiCA'] else ""
    
    mae_improve = (1 - exp['MAE'] / baseline['MAE']) * 100
    ade_improve = (1 - exp['ADE'] / baseline['ADE']) * 100
    fde_improve = (1 - exp['FDE'] / baseline['FDE']) * 100
    
    print(f"\n{exp_key}:{gat_contrib}{feat_contrib}{bica_contrib}")
    print(f"  MAE:  {exp['MAE']:.6f} ({mae_improve:+.1f}% vs baseline)")
    print(f"  ADE:  {exp['ADE']:.6f} ({ade_improve:+.1f}% vs baseline)")
    print(f"  FDE:  {exp['FDE']:.6f} ({fde_improve:+.1f}% vs baseline)")

print("\n🏆 BEST OVERALL MODEL")
best_exp = max(list(experiments.items())[1:], key=lambda x: (1 - x[1]['MAE'] / baseline['MAE']))
exp_key, exp = best_exp
mae_improve = (1 - exp['MAE'] / baseline['MAE']) * 100
ade_improve = (1 - exp['ADE'] / baseline['ADE']) * 100
fde_improve = (1 - exp['FDE'] / baseline['FDE']) * 100

print(f"  Model: {exp_key}")
print(f"  MAE Improvement:  {mae_improve:.1f}%")
print(f"  ADE Improvement:  {ade_improve:.1f}%")
print(f"  FDE Improvement:  {fde_improve:.1f}%")

print("\n📁 Output Files Generated:")
print(f"  ✓ {output_dir / 'ablation_overall_metrics.png'}")
print(f"  ✓ {output_dir / 'ablation_improvement_over_baseline.png'}")
print(f"  ✓ {output_dir / 'ablation_per_axis_mae.png'}")
print(f"  ✓ {output_dir / 'ablation_contribution_analysis.png'}")
print(f"  ✓ {output_dir / 'ablation_metrics_table.png'}")
print(f"  ✓ {output_dir / 'ablation_component_heatmap.png'}")
print(f"  ✓ {output_dir / 'ablation_summary.csv'}")
print("\n" + "="*90)
