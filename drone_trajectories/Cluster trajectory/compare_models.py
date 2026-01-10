#!/usr/bin/env python3
"""
快速对比脚本：GNN vs BiGRU 模型性能对比

运行方式：
  # 训练GNN模型（推荐）
  python compare_models.py --model gnn --epochs 120 --batch_size 256

  # 训练原始BiGRU（用于对比）
  python compare_models.py --model bigru --epochs 120 --batch_size 256

  # 对比结果
  python compare_models.py --compare
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_training_curves():
    """对比两个模型的训练曲线"""
    
    # 尝试加载GNN模型的训练历史
    gnn_history_path = Path('gru_models_gnn/training_history_agents_3.json')
    bigru_history_path = Path('newloss_swarm_models_enhanced/training_history_agents_3.json')
    
    results = {}
    
    if gnn_history_path.exists():
        with open(gnn_history_path) as f:
            gnn_history = json.load(f)
            results['gnn'] = gnn_history
            logger.info(f"✓ 加载GNN历史: {len(gnn_history['epoch'])} epochs")
    else:
        logger.warning(f"未找到GNN历史: {gnn_history_path}")
    
    if bigru_history_path.exists():
        with open(bigru_history_path) as f:
            bigru_history = json.load(f)
            results['bigru'] = bigru_history
            logger.info(f"✓ 加载BiGRU历史: {len(bigru_history['epoch'])} epochs")
    else:
        logger.warning(f"未找到BiGRU历史: {bigru_history_path}")
    
    if not results:
        logger.error("未找到任何训练历史文件！")
        return
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GNN vs BiGRU 模型对比', fontsize=16, fontweight='bold')
    
    # 1. 训练损失对比
    ax = axes[0, 0]
    if 'gnn' in results and 'train_loss' in results['gnn']:
        ax.plot(results['gnn']['epoch'], results['gnn']['train_loss'], 
               label='GNN', marker='o', markersize=3, linewidth=2)
    if 'bigru' in results and 'train_loss' in results['bigru']:
        ax.plot(results['bigru']['epoch'], results['bigru']['train_loss'],
               label='BiGRU', marker='s', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title('训练损失')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 验证损失对比
    ax = axes[0, 1]
    if 'gnn' in results and 'val_loss' in results['gnn']:
        ax.plot(results['gnn']['epoch'], results['gnn']['val_loss'],
               label='GNN', marker='o', markersize=3, linewidth=2)
    if 'bigru' in results and 'val_loss' in results['bigru']:
        ax.plot(results['bigru']['epoch'], results['bigru']['val_loss'],
               label='BiGRU', marker='s', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss')
    ax.set_title('验证损失（关键指标）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 验证MAE对比
    ax = axes[1, 0]
    if 'gnn' in results and 'val_mae' in results['gnn']:
        ax.plot(results['gnn']['epoch'], results['gnn']['val_mae'],
               label='GNN', marker='o', markersize=3, linewidth=2)
    if 'bigru' in results and 'val_mae' in results['bigru']:
        ax.plot(results['bigru']['epoch'], results['bigru']['val_mae'],
               label='BiGRU', marker='s', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val MAE (m)')
    ax.set_title('验证MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 学习率变化
    ax = axes[1, 1]
    if 'gnn' in results and 'learning_rate' in results['gnn']:
        ax.plot(results['gnn']['epoch'], results['gnn']['learning_rate'],
               label='GNN', marker='o', markersize=3, linewidth=2)
    if 'bigru' in results and 'learning_rate' in results['bigru']:
        ax.plot(results['bigru']['epoch'], results['bigru']['learning_rate'],
               label='BiGRU', marker='s', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('学习率调度')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("✓ 对比图已保存: model_comparison.png")
    plt.close()
    
    # 生成文本总结
    summary = generate_summary(results)
    print("\n" + "="*80)
    print(summary)
    print("="*80)


def generate_summary(results):
    """生成对比总结"""
    summary = "模型性能对比总结\n" + "-"*80 + "\n"
    
    for model_name, history in results.items():
        summary += f"\n【{model_name.upper()}模型】\n"
        
        epochs = history.get('epoch', [])
        val_loss = history.get('val_loss', [])
        val_mae = history.get('val_mae', [])
        train_loss = history.get('train_loss', [])
        
        if epochs and val_loss:
            summary += f"  总轮数: {len(epochs)}\n"
            summary += f"  最低验证损失: {min(val_loss):.6f} (Epoch {np.argmin(val_loss)+1})\n"
            summary += f"  最终验证损失: {val_loss[-1]:.6f}\n"
            summary += f"  损失改进: {(val_loss[0]-min(val_loss))/val_loss[0]*100:.1f}%\n"
        
        if val_mae:
            summary += f"  最低MAE: {min(val_mae):.6f}m\n"
            summary += f"  最终MAE: {val_mae[-1]:.6f}m\n"
        
        if train_loss and val_loss:
            train_val_gap = (train_loss[-1] - val_loss[-1]) / val_loss[-1] * 100
            summary += f"  最终Train-Val差距: {train_val_gap:.1f}% {'(过拟合)' if train_val_gap < -30 else '(正常)'}\n"
        
        # 检查损失曲线是否持续下降
        if val_loss and len(val_loss) > 10:
            recent_loss = np.mean(val_loss[-10:])
            early_loss = np.mean(val_loss[10:20])
            if recent_loss < early_loss:
                summary += f"  ✓ 损失曲线: 持续下降（后期低于前期）\n"
            else:
                summary += f"  ✗ 损失曲线: 停滞或波动（后期≥前期）\n"
    
    # 对比
    if 'gnn' in results and 'bigru' in results:
        summary += "\n【对比分析】\n"
        gnn_val_loss = min(results['gnn'].get('val_loss', [float('inf')]))
        bigru_val_loss = min(results['bigru'].get('val_loss', [float('inf')]))
        
        if gnn_val_loss < float('inf') and bigru_val_loss < float('inf'):
            improvement = (bigru_val_loss - gnn_val_loss) / bigru_val_loss * 100
            summary += f"  GNN vs BiGRU 最优验证损失:\n"
            summary += f"    BiGRU: {bigru_val_loss:.6f}\n"
            summary += f"    GNN:   {gnn_val_loss:.6f}\n"
            summary += f"    改进:  {improvement:+.1f}% {'✓ (GNN更优)' if improvement > 0 else '✗ (BiGRU更优)'}\n"
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='GNN vs BiGRU 模型对比')
    parser.add_argument('--compare', action='store_true', default=True,
                       help='仅进行对比分析（不训练）')
    parser.add_argument('--model', type=str, choices=['gnn', 'bigru'],
                       help='选择训练模型')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=256)
    
    args = parser.parse_args()
    
    if args.model:
        if args.model == 'gnn':
            logger.info(f"启动GNN模型训练 (epochs={args.epochs}, batch_size={args.batch_size})")
            import subprocess
            subprocess.run([
                'python', 'train_swarm_gnn.py',
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--agents', '3'
            ])
        elif args.model == 'bigru':
            logger.info(f"启动BiGRU模型训练 (epochs={args.epochs}, batch_size={args.batch_size})")
            import subprocess
            subprocess.run([
                'python', 'train_swarm_model_enhanced.py',
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--agents', '3'
            ])
    else:
        # 仅进行对比分析
        logger.info("执行模型对比分析...")
        compare_training_curves()


if __name__ == '__main__':
    main()
