#!/usr/bin/env python3
"""
LBEBM3D vs GNN+BiGRU Comparison - Paper-ready Visualization Script
===================================================================

生成出版级别的对比图表，包括：
1. 整体性能对比（MAE/RMSE/FDE/MAPE）
2. 轴向误差对比（X/Y/Z）
3. 样本分布箱线图
4. Per-agent 误差对比
5. 误差趋势对比

使用漂亮的配色和排版：
- LBEBM3D: 红色 (#E74C3C)
- GNN+BiGRU: 橙色 (#E67E22)

用法:
    python plot_lbebm_vs_gnn_comparison.py \
        --summary_json comparison_results/comparison_summary.json \
        --output_dir comparison_results/paper_figures \
        --dpi 300
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 配置绘图风格
def setup_paper_style():
    """设置论文出版级别的绘图风格"""
    rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'figure.figsize': (10, 6),
    })
    sns.set_palette("husl")

# 颜色定义
COLORS = {
    'lbebm': '#E74C3C',   # 红色
    'gnn': '#E67E22',      # 橙色
    'truth': '#27AE60',    # 绿色
}

MODEL_NAMES = {
    'lbebm': 'LBEBM3D',
    'gnn': 'GNN+BiGRU',
}

@dataclass
class MetricsStats:
    """指标统计"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float


def extract_metrics_stats(summary_json: dict, model_name: str) -> Dict[str, MetricsStats]:
    """从汇总统计信息提取指标"""
    stats_dict = {}
    
    aggregate = summary_json[model_name]['aggregate_stats']
    
    for metric_name in ['MAE', 'RMSE', 'ADE', 'FDE', 'MAPE', 'MAE_X', 'MAE_Y', 'MAE_Z']:
        if metric_name in aggregate:
            agg = aggregate[metric_name]
            stats_dict[metric_name] = MetricsStats(
                mean=agg['mean'],
                std=agg['std'],
                min=agg['min'],
                max=agg['max'],
                median=agg['median'],
                q25=agg['mean'] - 0.67 * agg['std'],  # 近似下四分位
                q75=agg['mean'] + 0.67 * agg['std'],  # 近似上四分位
            )
    
    return stats_dict


def plot_overall_comparison(summary_json: Path, output_dir: Path):
    """绘制整体性能对比图"""
    setup_paper_style()
    
    with open(summary_json, 'r') as f:
        summary = json.load(f)
    
    lbebm_stats = extract_metrics_stats(summary, 'LBEBM3D')
    gnn_stats = extract_metrics_stats(summary, 'GNN_BiGRU')
    
    # 主要指标
    metrics_to_plot = ['ADE', 'FDE', 'RMSE', 'MAPE']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lbebm_values = [lbebm_stats[m].mean for m in metrics_to_plot]
    lbebm_errors = [lbebm_stats[m].std for m in metrics_to_plot]
    gnn_values = [gnn_stats[m].mean for m in metrics_to_plot]
    gnn_errors = [gnn_stats[m].std for m in metrics_to_plot]
    
    bars1 = ax.bar(x - width/2, lbebm_values, width, 
                   label='LBEBM3D', color=COLORS['lbebm'], 
                   edgecolor='black', linewidth=1.5, alpha=0.85,
                   yerr=lbebm_errors, capsize=5, error_kw={'elinewidth': 2})
    
    bars2 = ax.bar(x + width/2, gnn_values, width,
                   label='GNN+BiGRU', color=COLORS['gnn'],
                   edgecolor='black', linewidth=1.5, alpha=0.85,
                   yerr=gnn_errors, capsize=5, error_kw={'elinewidth': 2})
    
    # 在柱子上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Error (m or %)', fontsize=12, fontweight='bold')
    ax.set_title('LBEBM3D vs GNN+BiGRU - Overall Performance Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 添加样本数信息
    num_samples = summary.get('num_samples', 'N/A')
    ax.text(0.99, 0.02, f'Samples: {num_samples}', transform=ax.transAxes,
           fontsize=10, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'overall_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info("✓ Saved: overall_comparison.png/pdf")


def plot_axis_mae_comparison(summary_json: Path, output_dir: Path):
    """绘制轴向MAE对比图"""
    setup_paper_style()
    
    with open(summary_json, 'r') as f:
        summary = json.load(f)
    
    lbebm_stats = extract_metrics_stats(summary, 'LBEBM3D')
    gnn_stats = extract_metrics_stats(summary, 'GNN_BiGRU')
    
    axes_to_plot = ['MAE_X', 'MAE_Y', 'MAE_Z']
    x = np.arange(len(axes_to_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lbebm_values = [lbebm_stats[m].mean for m in axes_to_plot]
    lbebm_errors = [lbebm_stats[m].std for m in axes_to_plot]
    gnn_values = [gnn_stats[m].mean for m in axes_to_plot]
    gnn_errors = [gnn_stats[m].std for m in axes_to_plot]
    
    bars1 = ax.bar(x - width/2, lbebm_values, width,
                   label='LBEBM3D', color=COLORS['lbebm'],
                   edgecolor='black', linewidth=1.5, alpha=0.85,
                   yerr=lbebm_errors, capsize=5, error_kw={'elinewidth': 2})
    
    bars2 = ax.bar(x + width/2, gnn_values, width,
                   label='GNN+BiGRU', color=COLORS['gnn'],
                   edgecolor='black', linewidth=1.5, alpha=0.85,
                   yerr=gnn_errors, capsize=5, error_kw={'elinewidth': 2})
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Error (m)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Axis MAE Comparison (X/Y/Z)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['X-Axis', 'Y-Axis', 'Z-Axis'], fontsize=11)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.savefig(output_dir / 'axis_mae_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'axis_mae_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info("✓ Saved: axis_mae_comparison.png/pdf")


def plot_boxplot_comparison(summary_json: Path, output_dir: Path):
    """绘制样本分布箱线图"""
    setup_paper_style()
    
    with open(summary_json, 'r') as f:
        summary = json.load(f)
    
    lbebm_metrics = summary['LBEBM3D']['all_metrics']
    gnn_metrics = summary['GNN_BiGRU']['all_metrics']
    
    lbebm_mae = [m['MAE'] for m in lbebm_metrics]
    gnn_mae = [m['MAE'] for m in gnn_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot([lbebm_mae, gnn_mae],
                     labels=['LBEBM3D', 'GNN+BiGRU'],
                     patch_artist=True,
                     widths=0.6,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8))
    
    # 设置箱线图颜色
    colors = [COLORS['lbebm'], COLORS['gnn']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 美化其他元素
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    plt.setp(bp['medians'], color='darkblue', linewidth=2)
    
    ax.set_ylabel('MAE (m)', fontsize=12, fontweight='bold')
    num_samples = summary.get('num_samples', len(lbebm_mae))
    ax.set_title(f'MAE Distribution Across All Samples (n={num_samples})', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 添加统计信息
    stats_text = (
        f"LBEBM3D: μ={np.mean(lbebm_mae):.4f}, σ={np.std(lbebm_mae):.4f}\n"
        f"GNN+BiGRU: μ={np.mean(gnn_mae):.4f}, σ={np.std(gnn_mae):.4f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.savefig(output_dir / 'mae_boxplot_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'mae_boxplot_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info("✓ Saved: mae_boxplot_comparison.png/pdf")


def plot_per_agent_comparison(summary_json: Path, output_dir: Path):
    """绘制Per-Agent误差对比"""
    setup_paper_style()
    
    with open(summary_json, 'r') as f:
        summary = json.load(f)
    
    lbebm_agg = summary['LBEBM3D'].get('aggregate_stats', {})
    gnn_agg = summary['GNN_BiGRU'].get('aggregate_stats', {})
    
    if 'MAE_per_agent_mean' not in lbebm_agg or 'MAE_per_agent_mean' not in gnn_agg:
        logger.warning("⚠ Per-agent statistics not available in aggregate_stats")
        return
    
    num_agents = len(lbebm_agg['MAE_per_agent_mean'])
    lbebm_means = lbebm_agg['MAE_per_agent_mean']
    gnn_means = gnn_agg['MAE_per_agent_mean']
    lbebm_stds = lbebm_agg['MAE_per_agent_std']
    gnn_stds = gnn_agg['MAE_per_agent_std']
    
    x = np.arange(num_agents)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, lbebm_means, width,
                   label='LBEBM3D', color=COLORS['lbebm'],
                   edgecolor='black', linewidth=1.5, alpha=0.85,
                   yerr=lbebm_stds, capsize=5, error_kw={'elinewidth': 2})
    
    bars2 = ax.bar(x + width/2, gnn_means, width,
                   label='GNN+BiGRU', color=COLORS['gnn'],
                   edgecolor='black', linewidth=1.5, alpha=0.85,
                   yerr=gnn_stds, capsize=5, error_kw={'elinewidth': 2})
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('MAE (m)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Agent ID', fontsize=12, fontweight='bold')
    ax.set_title('Per-Agent MAE Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Agent {i}' for i in range(num_agents)], fontsize=11)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.savefig(output_dir / 'per_agent_mae_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'per_agent_mae_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info("✓ Saved: per_agent_mae_comparison.png/pdf")


def plot_error_trend(summary_json: Path, output_dir: Path):
    """绘制误差随时间步的变化趋势"""
    setup_paper_style()
    
    with open(summary_json, 'r') as f:
        summary = json.load(f)
    
    lbebm_metrics = summary['LBEBM3D']['all_metrics']
    gnn_metrics = summary['GNN_BiGRU']['all_metrics']
    
    # 计算每个时间步的平均误差
    lbebm_per_step = []
    gnn_per_step = []
    
    for m in lbebm_metrics:
        if 'MAE_per_step' in m:
            lbebm_per_step.append(m['MAE_per_step'])
    
    for m in gnn_metrics:
        if 'MAE_per_step' in m:
            gnn_per_step.append(m['MAE_per_step'])
    
    if lbebm_per_step and gnn_per_step:
        lbebm_per_step = np.mean(lbebm_per_step, axis=0)
        gnn_per_step = np.mean(gnn_per_step, axis=0)
        
        steps = np.arange(len(lbebm_per_step))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(steps, lbebm_per_step, 'o-', color=COLORS['lbebm'],
               linewidth=2.5, markersize=8, label='LBEBM3D', alpha=0.85)
        ax.plot(steps, gnn_per_step, 's-', color=COLORS['gnn'],
               linewidth=2.5, markersize=8, label='GNN+BiGRU', alpha=0.85)
        
        ax.set_xlabel('Prediction Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean MAE (m)', fontsize=12, fontweight='bold')
        ax.set_title('Error Trend Over Prediction Horizon', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(steps)
        
        fig.savefig(output_dir / 'error_trend_comparison.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'error_trend_comparison.pdf', bbox_inches='tight')
        plt.close(fig)
        logger.info("✓ Saved: error_trend_comparison.png/pdf")


def generate_comparison_table(summary_json: Path, output_dir: Path):
    """生成对比表格"""
    setup_paper_style()
    
    with open(summary_json, 'r') as f:
        summary = json.load(f)
    
    lbebm_stats = extract_metrics_stats(summary, 'LBEBM3D')
    gnn_stats = extract_metrics_stats(summary, 'GNN_BiGRU')
    
    # 创建对比表格
    table_data = []
    metrics_list = ['ADE', 'FDE', 'RMSE', 'MAPE', 'MAE_X', 'MAE_Y', 'MAE_Z']
    
    for metric in metrics_list:
        if metric in lbebm_stats and metric in gnn_stats:
            lbebm_mean = lbebm_stats[metric].mean
            gnn_mean = gnn_stats[metric].mean
            improvement = ((lbebm_mean - gnn_mean) / lbebm_mean * 100) if lbebm_mean != 0 else 0
            
            table_data.append([
                metric,
                f"{lbebm_mean:.4f} ± {lbebm_stats[metric].std:.4f}",
                f"{gnn_mean:.4f} ± {gnn_stats[metric].std:.4f}",
                f"{improvement:+.2f}%"
            ])
    
    # 保存为文本表格
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = summary.get('num_samples', 'N/A')
    with open(output_dir / 'comparison_table.txt', 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"LBEBM3D vs GNN+BiGRU Comparison - Detailed Metrics (n={num_samples})\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Metric':<15} {'LBEBM3D (mean ± std)':<30} {'GNN+BiGRU (mean ± std)':<30} {'Improvement':<15}\n")
        f.write("-" * 100 + "\n")
        for row in table_data:
            f.write(f"{row[0]:<15} {row[1]:<30} {row[2]:<30} {row[3]:<15}\n")
        f.write("=" * 100 + "\n\n")
        f.write("说明:\n")
        f.write("- Improvement 为正表示 GNN+BiGRU 性能更好\n")
        f.write(f"- 所有指标基于 {num_samples} 样本评估\n")
    
    logger.info("✓ Saved: comparison_table.txt")
    
    # 打印到控制台
    print("\n" + "=" * 100)
    print(f"LBEBM3D vs GNN+BiGRU 详细对比 (n={num_samples})")
    print("=" * 100)
    print(f"{'Metric':<15} {'LBEBM3D':<30} {'GNN+BiGRU':<30} {'Improvement':<15}")
    print("-" * 100)
    for row in table_data:
        print(f"{row[0]:<15} {row[1]:<30} {row[2]:<30} {row[3]:<15}")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description='生成论文级对比图表')
    parser.add_argument('--summary_json', type=Path, required=True,
                       help='Comparison summary JSON file path')
    parser.add_argument('--output_dir', type=Path, default=Path('paper_figures'),
                       help='Output directory for figures')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI')
    
    args = parser.parse_args()
    
    if not args.summary_json.exists():
        logger.error(f"❌ Summary JSON not found: {args.summary_json}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📊 正在生成论文级对比图表...")
    logger.info(f"📖 输入文件: {args.summary_json}")
    logger.info(f"📁 输出目录: {args.output_dir}\n")
    
    # 生成所有图表
    plot_overall_comparison(args.summary_json, args.output_dir)
    plot_axis_mae_comparison(args.summary_json, args.output_dir)
    plot_boxplot_comparison(args.summary_json, args.output_dir)
    plot_per_agent_comparison(args.summary_json, args.output_dir)
    plot_error_trend(args.summary_json, args.output_dir)
    generate_comparison_table(args.summary_json, args.output_dir)
    
    logger.info(f"\n✅ 所有图表已生成！")
    logger.info(f"📁 输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()
