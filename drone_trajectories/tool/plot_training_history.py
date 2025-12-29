#!/usr/bin/env python3
"""
训练历史绘图脚本
从 CSV 文件读取训练历史，生成美观的 loss 收敛图

使用方法：
  python plot_training_history.py --csv_file path/to/history.csv
  python plot_training_history.py --csv_file path/to/history.csv --output_dir path/to/output
  python plot_training_history.py --csv_file path/to/history.csv --show  # 显示图表

cd /d "D:\Trajectory prediction\drone_trajectories\tool"

python plot_training_history.py ^
  --csv_file newdata1_short_gru_models\short_enhanced_gru_model_history.csv ^
  --output_dir newdata1_short_gru_models ^
  --show
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_history(csv_file, output_dir=None, show=False):
    """
    绘制训练历史
    
    Args:
        csv_file: 训练历史 CSV 文件路径
        output_dir: 输出目录，如果为 None，则与 csv 文件同目录
        show: 是否显示图表
    """
    if not os.path.exists(csv_file):
        print(f"❌ 文件不存在: {csv_file}")
        return
    
    # 读取数据
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return
    
    if len(df) == 0:
        print(f"❌ CSV 文件为空")
        return
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
        if not output_dir:
            output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取模型名称（从 csv 文件名提取）
    csv_name = Path(csv_file).stem
    model_name = csv_name.replace('_history', '')
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # 创建图表 (1x3 子图)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Loss 曲线
    ax = axes[0]
    ax.plot(df['Epoch'], df['Train Loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
    ax.plot(df['Epoch'], df['Val Loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
    
    # 找到最小 val loss 的位置
    best_epoch = df.loc[df['Val Loss'].idxmin(), 'Epoch']
    best_loss = df['Val Loss'].min()
    ax.scatter([best_epoch], [best_loss], color='red', s=100, marker='*', zorder=5, label=f'Best (Epoch {int(best_epoch)})')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Loss Convergence', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 学习率变化
    ax = axes[1]
    ax.plot(df['Epoch'], df['Learning Rate'], 'g-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. Teacher Forcing Ratio 变化
    ax = axes[2]
    ax.plot(df['Epoch'], df['Teacher Forcing Ratio'], 'purple', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Teacher Forcing Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Teacher Forcing Decay', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(df['Teacher Forcing Ratio']) * 1.1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, f'{model_name}_training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图表到: {output_path}")
    
    # 显示图表（如果需要）
    if show:
        plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print(f"📊 训练统计信息 - {model_name}")
    print("="*60)
    print(f"  总 Epochs:          {int(df['Epoch'].max())}")
    print(f"  最初 Train Loss:    {df['Train Loss'].iloc[0]:.6f}")
    print(f"  最终 Train Loss:    {df['Train Loss'].iloc[-1]:.6f}")
    print(f"  最小 Val Loss:      {df['Val Loss'].min():.6f} (Epoch {int(best_epoch)})")
    print(f"  最终 Val Loss:      {df['Val Loss'].iloc[-1]:.6f}")
    print(f"  初始学习率:         {df['Learning Rate'].iloc[0]:.6e}")
    print(f"  最终学习率:         {df['Learning Rate'].iloc[-1]:.6e}")
    print(f"  平均 Epoch 时间:    {df['Epoch Time (s)'].mean():.2f}s")
    print(f"  总训练时间:         {df['Epoch Time (s)'].sum():.2f}s ({df['Epoch Time (s)'].sum()/3600:.2f}h)")
    print("="*60)
    
    # 返回 dataframe（便于进一步分析）
    return df

def plot_multiple_models(csv_files, output_dir=None):
    """
    比较多个模型的训练曲线
    
    Args:
        csv_files: CSV 文件列表
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))
    
    for csv_file, color in zip(csv_files, colors):
        if not os.path.exists(csv_file):
            print(f"⚠ 跳过不存在的文件: {csv_file}")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            model_name = Path(csv_file).stem.replace('_history', '')
            
            # Train Loss
            axes[0].plot(df['Epoch'], df['Train Loss'], linewidth=2, 
                        label=f'{model_name} (Train)', color=color, linestyle='-', marker='o', markersize=3)
            
            # Val Loss
            axes[1].plot(df['Epoch'], df['Val Loss'], linewidth=2,
                        label=f'{model_name} (Val)', color=color, linestyle='--', marker='s', markersize=3)
        except Exception as e:
            print(f"⚠ 读取文件失败 {csv_file}: {e}")
    
    # 配置子图
    for ax, title in zip(axes, ['Train Loss', 'Val Loss']):
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'models_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存对比图表到: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制训练历史图表')
    parser.add_argument('--csv_file', type=str, help='单个训练历史 CSV 文件')
    parser.add_argument('--csv_files', type=str, nargs='+', help='多个 CSV 文件用于对比')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--show', action='store_true', help='显示图表')
    
    args = parser.parse_args()
    
    if args.csv_file:
        plot_training_history(args.csv_file, args.output_dir, args.show)
    elif args.csv_files:
        plot_multiple_models(args.csv_files, args.output_dir)
    else:
        print("❌ 请指定 --csv_file 或 --csv_files")
        parser.print_help()
