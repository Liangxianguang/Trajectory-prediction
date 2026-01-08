#!/usr/bin/env python3
"""
可视化训练历史 - 观察改进效果

使用方法:
    python plot_training_history.py outputs_improved_v2/training_history_agents_3.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_training_history(csv_file):
    """绘制训练历史图表"""
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"❌ 找不到文件: {csv_file}")
        return
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    print(f"\n✅ 读取训练数据: {csv_file}")
    print(f"   共{len(df)}个epoch")
    print(f"   最终Val MAE: {df['Val MAE (m)'].iloc[-1]:.6f}m")
    print(f"   最优Val MAE: {df['Val MAE (m)'].min():.6f}m (Epoch {df['Val MAE (m)'].idxmin()+1})")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('集群轨迹模型训练过程（融合单机最佳实践）', fontsize=16, fontweight='bold')
    
    # 1. Train Loss vs Val Loss
    ax = axes[0, 0]
    ax.plot(df['Epoch'], df['Train Loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax.plot(df['Epoch'], df['Val Loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss曲线 (目标: 两者逐步改善)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Val MAE（最关键指标）
    ax = axes[0, 1]
    ax.plot(df['Epoch'], df['Val MAE (m)'], 'g-', linewidth=2.5, marker='o', markersize=4)
    ax.axhline(y=0.2, color='orange', linestyle='--', linewidth=2, label='目标: 0.2m')
    ax.axhline(y=df['Val MAE (m)'].iloc[0], color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'初始: {df["Val MAE (m)"].iloc[0]:.3f}m')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MAE (m)', fontsize=11)
    ax.set_title('验证MAE (关键指标)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Teacher Forcing衰减
    ax = axes[1, 0]
    ax.plot(df['Epoch'], df['Teacher Forcing Ratio'], 'purple', linewidth=2.5, marker='^', markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('TF Ratio', fontsize=11)
    ax.set_title('Teacher Forcing衰减 (从0.5到0.0)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 0.55])
    
    # 4. 学习率变化
    ax = axes[1, 1]
    ax.semilogy(df['Epoch'], df['Learning Rate'], 'brown', linewidth=2, marker='D', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate (log scale)', fontsize=11)
    ax.set_title('学习率调整过程', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # 保存和显示
    output_file = csv_path.parent / 'training_progress.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ 已保存图表: {output_file}")
    
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python plot_training_history.py <csv_file>")
        print("\n例如:")
        print("  python plot_training_history.py outputs_improved_v2/training_history_agents_3.csv")
        return
    
    csv_file = sys.argv[1]
    plot_training_history(csv_file)

if __name__ == '__main__':
    main()
