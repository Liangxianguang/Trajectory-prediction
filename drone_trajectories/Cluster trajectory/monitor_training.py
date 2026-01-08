#!/usr/bin/env python3
"""
训练进度实时监控脚本
可实时查看所有agent配置的训练状态、MAE、损失等指标

使用示例：
    python monitor_training.py
    python monitor_training.py --agents 3
    python monitor_training.py --output_dir newloss_swarm_models_enhanced
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json


def get_checkpoint_status(output_dir, num_agents):
    """检查检查点文件状态"""
    output_path = Path(output_dir)
    
    ckpt_last = output_path / f'last_checkpoint_agents_{num_agents}.pt'
    ckpt_best = output_path / f'best_model_agents_{num_agents}.pt'
    ckpt_interrupted = output_path / f'interrupted_checkpoint_agents_{num_agents}.pt'
    
    status = {
        'last': ckpt_last.exists(),
        'best': ckpt_best.exists(),
        'interrupted': ckpt_interrupted.exists(),
        'last_size': ckpt_last.stat().st_size / (1024**2) if ckpt_last.exists() else 0,
        'best_size': ckpt_best.stat().st_size / (1024**2) if ckpt_best.exists() else 0,
    }
    return status


def monitor_single_agent(output_dir, num_agents, verbose=False):
    """监控单个agent配置的训练"""
    
    output_path = Path(output_dir)
    csv_path = output_path / f'training_history_agents_{num_agents}.csv'
    config_path = output_path / f'training_config_agents_{num_agents}.json'
    
    if not csv_path.exists():
        return None
    
    # 读取训练历史
    df = pd.read_csv(csv_path)
    
    # 检查点状态
    ckpt_status = get_checkpoint_status(output_dir, num_agents)
    
    result = {
        'agents': num_agents,
        'total_epochs': len(df),
        'best_epoch': df['Val MAE (m)'].idxmin() + 1,
        'best_mae': df['Val MAE (m)'].min(),
        'current_mae': df['Val MAE (m)'].iloc[-1],
        'best_loss': df['Val Loss'].min(),
        'current_loss': df['Val Loss'].iloc[-1],
        'current_lr': df['Learning Rate'].iloc[-1],
        'current_tf': df['Teacher Forcing Ratio'].iloc[-1],
        'train_loss': df['Train Loss'].iloc[-1],
        'val_loss': df['Val Loss'].iloc[-1],
        'improvement_rate': (df['Val Loss'].iloc[0] - df['Val Loss'].iloc[-1]) / df['Val Loss'].iloc[0] * 100,
        'ckpt_status': ckpt_status,
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                result['config_timestamp'] = config.get('timestamp', 'N/A')
        except:
            pass
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"{num_agents} 架无人机详细信息")
        print(f"{'='*70}")
        print(f"总Epoch数: {result['total_epochs']}")
        print(f"\n指标统计:")
        print(f"  最佳MAE: {result['best_mae']:.6f} m @ Epoch {result['best_epoch']}")
        print(f"  当前MAE: {result['current_mae']:.6f} m")
        print(f"  MAE改进: {result['best_mae'] - result['current_mae']:.6f} m")
        print(f"  最低Loss: {result['best_loss']:.6f}")
        print(f"  当前Loss: {result['current_loss']:.6f}")
        print(f"  训练loss: {result['train_loss']:.6f}")
        print(f"  整体改进率: {result['improvement_rate']:.2f}%")
        
        print(f"\n当前状态:")
        print(f"  学习率: {result['current_lr']:.2e}")
        print(f"  TF比率: {result['current_tf']:.4f}")
        
        print(f"\n检查点文件:")
        print(f"  最后检查点: {'✓' if result['ckpt_status']['last'] else '✗'} ({result['ckpt_status']['last_size']:.1f} MB)")
        print(f"  最佳模型: {'✓' if result['ckpt_status']['best'] else '✗'} ({result['ckpt_status']['best_size']:.1f} MB)")
        print(f"  中断检查点: {'✓' if result['ckpt_status']['interrupted'] else '✗'}")
        
        # 最近10个epoch的统计
        print(f"\n最近10个Epoch:")
        recent_df = df.tail(10)[['Epoch', 'Train Loss', 'Val Loss', 'Val MAE (m)', 'Learning Rate']].copy()
        recent_df['Val MAE (m)'] = recent_df['Val MAE (m)'].apply(lambda x: f"{x:.6f}")
        recent_df['Train Loss'] = recent_df['Train Loss'].apply(lambda x: f"{x:.6f}")
        recent_df['Val Loss'] = recent_df['Val Loss'].apply(lambda x: f"{x:.6f}")
        recent_df['Learning Rate'] = recent_df['Learning Rate'].apply(lambda x: f"{x:.2e}")
        print(recent_df.to_string(index=False))
    
    return result


def print_summary_table(results):
    """打印汇总表格"""
    print(f"\n{'='*100}")
    print("训练进度汇总表")
    print(f"{'='*100}")
    
    # 表头
    header = f"{'Agents':<8} {'Total Epochs':<15} {'Best MAE':<15} {'Current MAE':<15} {'Improvement %':<15} {'Ckpts':<15}"
    print(header)
    print("-" * 100)
    
    # 每个agent的一行
    for result in results:
        if result is None:
            continue
        
        agents = result['agents']
        total = result['total_epochs']
        best_mae = result['best_mae']
        curr_mae = result['current_mae']
        improve = result['best_mae'] - result['current_mae']
        improve_pct = (result['best_mae'] - result['current_mae']) / result['best_mae'] * 100 if result['best_mae'] > 0 else 0
        
        ckpt_str = "L" if result['ckpt_status']['last'] else "-"
        ckpt_str += "B" if result['ckpt_status']['best'] else "-"
        ckpt_str += "I" if result['ckpt_status']['interrupted'] else "-"
        
        line = f"{agents:<8} {total:<15} {best_mae:<15.6f} {curr_mae:<15.6f} {improve_pct:<15.2f}% {ckpt_str:<15}"
        print(line)
    
    print(f"{'='*100}\n")
    print("图例: L=最后检查点, B=最佳模型, I=中断检查点")


def monitor_training(output_dir='newloss_swarm_models_enhanced', agents='all', verbose=True):
    """监控所有或指定agent配置的训练"""
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    
    # 确定要监控的agent列表
    if agents == 'all':
        agents_list = [3, 4, 5, 6]
    else:
        agents_list = [int(agents)]
    
    # 收集结果
    results = []
    for num_agents in agents_list:
        result = monitor_single_agent(output_dir, num_agents, verbose=verbose)
        results.append(result)
    
    # 如果有结果，打印汇总表格
    valid_results = [r for r in results if r is not None]
    if valid_results:
        print_summary_table(valid_results)
        
        # 整体统计
        total_epochs = sum(r['total_epochs'] for r in valid_results)
        avg_mae = np.mean([r['current_mae'] for r in valid_results])
        min_mae = min(r['best_mae'] for r in valid_results)
        max_mae = max(r['best_mae'] for r in valid_results)
        
        print(f"总体统计:")
        print(f"  总Epoch数: {total_epochs}")
        print(f"  平均MAE: {avg_mae:.6f} m")
        print(f"  最小MAE: {min_mae:.6f} m")
        print(f"  最大MAE: {max_mae:.6f} m")
        print(f"  最优配置: {valid_results[[r['best_mae'] for r in valid_results].index(min_mae)]['agents']} 架")
    else:
        print("⚠ 未找到任何训练历史文件")


def main():
    parser = argparse.ArgumentParser(
        description='训练进度实时监控工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 监控所有agent配置
  python monitor_training.py

  # 只监控3架无人机
  python monitor_training.py --agents 3

  # 指定输出目录
  python monitor_training.py --output_dir my_models_dir

  # 仅看汇总（不显示详细信息）
  python monitor_training.py --verbose 0
        """
    )
    
    parser.add_argument('--output_dir', type=str, default='newloss_swarm_models_enhanced',
                        help='模型输出目录')
    parser.add_argument('--agents', type=str, default='all',
                        help='无人机数量 (3|4|5|6|all)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='是否显示详细信息 (0=仅汇总, 1=详细)')
    
    args = parser.parse_args()
    
    print(f"\n🕐 监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🎯 Agent配置: {args.agents}")
    
    monitor_training(
        output_dir=args.output_dir,
        agents=args.agents,
        verbose=bool(args.verbose)
    )


if __name__ == '__main__':
    main()
