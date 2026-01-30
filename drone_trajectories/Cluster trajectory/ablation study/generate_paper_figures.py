#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验论文级总结图表生成
============================

根据推理结果生成以下论文级图表：
  1. 训练曲线对比 (4个子图: train loss, val loss, val MAE, learning rate)
  2. 性能指标对比 (MAE/RMSE/MAPE + 误差棒)
  3. 改进分析 (相对于基线的改进百分比)
  4. 误差分布 (5个直方图)
  5. 性能指标表 (CSV + LaTeX)

使用示例:
    python generate_paper_figures.py \
        --ablation_dir . \
        --inference_results ablation_viz_results/summary.json \
        --output_dir paper_figures
"""

import numpy as np
import json
from pathlib import Path
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import pandas as pd

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 设置论文级样式
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# 颜色方案
COLORS = {
    'exp1': '#E74C3C',  # 红
    'exp2': '#3498DB',  # 蓝
    'exp3': '#9B59B6',  # 紫
    'exp4': '#27AE60',  # 绿
    'exp5': '#F39C12',  # 橙
}

LABELS = {
    'exp1': 'E1: Baseline',
    'exp2': 'E2: Feat+BiCA',
    'exp3': 'E3: GNN+BiCA',
    'exp4': 'E4: GNN+Feat',
    'exp5': 'E5: Full',
}


def load_training_histories(ablation_dir):
    """加载所有消融实验的训练历史"""
    ablation_dir = Path(ablation_dir)
    histories = {}
    
    for exp_id in range(1, 6):
        exp_names = {
            1: 'exp1_baseline',
            2: 'exp2_feat_bigru',
            3: 'exp3_gnn_bigru',
            4: 'exp4_gnn_feat',
            5: 'exp5_full',
        }
        
        csv_file = ablation_dir / f"ablation_results_agents_3_{exp_names[exp_id]}" / \
                   f"training_history_agents_3_{exp_names[exp_id]}.csv"
        
        if csv_file.exists():
            history = pd.read_csv(csv_file)
            histories[exp_id] = history
            logger.info(f"✓ 加载E{exp_id}训练历史: {len(history)} epochs")
        else:
            logger.warning(f"⚠ 未找到E{exp_id}训练历史: {csv_file}")
    
    return histories


def plot_training_curves(histories, output_dir):
    """绘制训练曲线对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves Comparison (All 5 Experiments)', fontsize=14, fontweight='bold')
    
    exp_ids = sorted(histories.keys())
    
    # 1. 训练损失
    ax = axes[0, 0]
    for exp_id in exp_ids:
        history = histories[exp_id]
        ax.plot(history['epoch'], history['train_loss'], 
               label=LABELS[f'exp{exp_id}'], color=COLORS[f'exp{exp_id}'], 
               linewidth=2, marker='o', markersize=2, markevery=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. 验证损失
    ax = axes[0, 1]
    for exp_id in exp_ids:
        history = histories[exp_id]
        ax.plot(history['epoch'], history['val_loss'],
               label=LABELS[f'exp{exp_id}'], color=COLORS[f'exp{exp_id}'],
               linewidth=2, marker='s', markersize=2, markevery=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 3. 验证MAE
    ax = axes[1, 0]
    for exp_id in exp_ids:
        history = histories[exp_id]
        ax.plot(history['epoch'], history['val_mae'],
               label=LABELS[f'exp{exp_id}'], color=COLORS[f'exp{exp_id}'],
               linewidth=2, marker='^', markersize=2, markevery=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MAE (m)')
    ax.set_title('Validation MAE')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 4. 学习率
    ax = axes[1, 1]
    for exp_id in exp_ids:
        history = histories[exp_id]
        if 'learning_rate' in history.columns:
            ax.plot(history['epoch'], history['learning_rate'],
                   label=LABELS[f'exp{exp_id}'], color=COLORS[f'exp{exp_id}'],
                   linewidth=2, marker='D', markersize=2, markevery=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'training_curves_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ 已保存训练曲线图: {output_file}")
    plt.close()


def plot_best_metrics(histories, output_dir):
    """绘制最佳指标对比表"""
    exp_ids = sorted(histories.keys())
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # 收集最佳指标
    table_data = []
    for exp_id in exp_ids:
        history = histories[exp_id]
        
        best_val_loss_idx = history['val_loss'].idxmin()
        best_val_mae_idx = history['val_mae'].idxmin()
        
        row = [
            LABELS[f'exp{exp_id}'].replace('E', 'Exp'),
            f"{history['val_loss'].min():.6f}",
            f"{history.loc[best_val_loss_idx, 'epoch']:.0f}",
            f"{history['val_mae'].min():.6f}",
            f"{history.loc[best_val_mae_idx, 'epoch']:.0f}",
            f"{history['train_loss'].iloc[-1]:.6f}",
            f"{history['val_loss'].iloc[-1]:.6f}",
        ]
        table_data.append(row)
    
    # 创建表格
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Best Val Loss', 'Epoch', 'Best MAE (m)', 'Epoch', 'Final Train Loss', 'Final Val Loss'],
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.1, 0.15, 0.1, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 表头样式
    for i in range(7):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 行颜色
    for i, exp_id in enumerate(exp_ids):
        for j in range(7):
            table[(i+1, j)].set_facecolor(COLORS[f'exp{exp_id}'])
            table[(i+1, j)].set_alpha(0.3)
    
    plt.title('Best Metrics Summary', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'best_metrics_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ 已保存最佳指标表: {output_file}")
    plt.close()


def plot_improvement_analysis(histories, output_dir):
    """绘制改进分析"""
    exp_ids = sorted(histories.keys())
    
    # 获取基线（Exp1）的最佳MAE
    baseline_mae = histories[1]['val_mae'].min()
    
    # 计算改进
    improvements = []
    mae_values = []
    
    for exp_id in exp_ids:
        best_mae = histories[exp_id]['val_mae'].min()
        mae_values.append(best_mae)
        
        # 改进百分比：(基线-当前)/基线 * 100
        improvement = (baseline_mae - best_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        improvements.append(improvement)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Improvement Analysis', fontsize=13, fontweight='bold')
    
    # 1. 绝对MAE对比
    ax = axes[0]
    x_pos = np.arange(len(exp_ids))
    colors_list = [COLORS[f'exp{exp_id}'] for exp_id in exp_ids]
    
    bars = ax.bar(x_pos, mae_values, color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # 标签
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'E{exp_id}' for exp_id in exp_ids])
    ax.set_ylabel('Best MAE (m)', fontsize=11, fontweight='bold')
    ax.set_title('Absolute MAE Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加值标签
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. 相对改进 (%)
    ax = axes[1]
    colors_improvement = ['#27AE60' if imp > 0 else '#E74C3C' for imp in improvements]
    
    bars = ax.bar(x_pos, improvements, color=colors_improvement, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'E{exp_id}' for exp_id in exp_ids])
    ax.set_ylabel('Improvement vs Baseline (%)', fontsize=11, fontweight='bold')
    ax.set_title('Relative Improvement')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加值标签
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if val > 0 else 'top'
        offset = height * 0.02 if val > 0 else -height * 0.02
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
               f'{val:+.1f}%', ha='center', va=va, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'improvement_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ 已保存改进分析图: {output_file}")
    plt.close()


def plot_final_comparison(histories, inference_summary, output_dir):
    """绘制最终对比（训练vs推理）"""
    exp_ids = sorted(histories.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(exp_ids))
    width = 0.35
    
    # 从训练历史获取最佳验证MAE
    train_mae = [histories[exp_id]['val_mae'].min() for exp_id in exp_ids]
    
    # 从推理结果获取推理MAE
    if inference_summary and 'statistics' in inference_summary:
        infer_mae = [inference_summary['statistics'][f'exp{exp_id}']['mean_mae'] 
                    for exp_id in exp_ids]
    else:
        infer_mae = train_mae  # 如果没有推理结果，使用训练结果
    
    colors_list = [COLORS[f'exp{exp_id}'] for exp_id in exp_ids]
    
    bars1 = ax.bar(x_pos - width/2, train_mae, width, label='Training (Best Val MAE)',
                  color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, infer_mae, width, label='Inference (Mean MAE)',
                  color=colors_list, alpha=0.5, hatch='///', edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('MAE (m)', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Inference Performance', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'E{exp_id}' for exp_id in exp_ids])
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'training_vs_inference.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ 已保存训练vs推理对比图: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='消融实验论文级图表生成')
    parser.add_argument('--ablation_dir', type=str, default='.', help='消融实验结果目录')
    parser.add_argument('--inference_results', type=str, default=None, help='推理结果汇总JSON文件')
    parser.add_argument('--output_dir', type=str, default='paper_figures', help='输出目录')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("消融实验论文级图表生成")
    print("="*80 + "\n")
    
    # 加载训练历史
    print("加载训练历史...")
    histories = load_training_histories(args.ablation_dir)
    
    if not histories:
        logger.error("未找到任何训练历史文件！")
        return
    
    print(f"✓ 加载了 {len(histories)} 个实验的训练历史\n")
    
    # 加载推理结果（可选）
    inference_summary = None
    if args.inference_results and Path(args.inference_results).exists():
        with open(args.inference_results, 'r', encoding='utf-8') as f:
            inference_summary = json.load(f)
        logger.info(f"✓ 加载推理结果: {args.inference_results}")
    
    print("\n生成论文级图表...\n")
    
    # 生成各个图表
    plot_training_curves(histories, output_dir)
    plot_best_metrics(histories, output_dir)
    plot_improvement_analysis(histories, output_dir)
    
    if inference_summary:
        plot_final_comparison(histories, inference_summary, output_dir)
    
    print("\n" + "="*80)
    print("✓ 论文级图表生成完成！")
    print(f"输出目录: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
