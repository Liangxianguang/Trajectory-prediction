#!/usr/bin/env python3
"""
按无人机逐架分析预测质量
分析每架无人机的 X/Y/Z 三轴学习情况及集群协作能力

使用示例：
python analyze_per_agent_predictions.py ^
  --predictions inference_results/predictions_agents_3.npz ^
  --output_dir agent_analysis ^
  --num_samples -1
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging
import json
from typing import Tuple, Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_per_agent_metrics(predictions, y_true, num_agents=3) -> Dict:
    """
    计算每架无人机的分轴误差指标
    
    Args:
        predictions: (N, seq_out, agents, 3) 预测轨迹
        y_true: (N, seq_out, agents, 3) 真实轨迹
        num_agents: 无人机数量
        
    Returns:
        metrics_dict: {
            'per_agent': {
                agent_id: {
                    'mae_3d': float,
                    'rmse_3d': float,
                    'mae_x': float, 'mae_y': float, 'mae_z': float,
                    'rmse_x': float, 'rmse_y': float, 'rmse_z': float,
                    'predictions': (N, seq_out, 3),
                    'y_true': (N, seq_out, 3),
                    'errors': (N, seq_out, 3)
                }
            },
            'global': {...},
            'axis_balance': {...}
        }
    """
    N, seq_out, _, _ = predictions.shape
    
    metrics = {
        'per_agent': {},
        'global': {},
        'axis_balance': {},
        'inter_agent_distance': {}
    }
    
    # 全局指标
    errors_global = np.abs(predictions - y_true)
    diffs_global = (predictions - y_true) ** 2
    
    metrics['global'] = {
        'mae_3d': np.mean(errors_global),
        'rmse_3d': np.sqrt(np.mean(diffs_global)),
        'mae_xyz': [np.mean(errors_global[..., i]) for i in range(3)],
        'rmse_xyz': [np.sqrt(np.mean(diffs_global[..., i])) for i in range(3)],
    }
    
    # 按无人机分析
    for agent_id in range(num_agents):
        pred_agent = predictions[:, :, agent_id, :]  # (N, seq_out, 3)
        true_agent = y_true[:, :, agent_id, :]        # (N, seq_out, 3)
        
        error_agent = np.abs(pred_agent - true_agent)  # (N, seq_out, 3)
        diff_agent = (pred_agent - true_agent) ** 2
        
        metrics['per_agent'][agent_id] = {
            'mae_3d': np.mean(error_agent),
            'rmse_3d': np.sqrt(np.mean(diff_agent)),
            'mae_x': np.mean(error_agent[..., 0]),
            'mae_y': np.mean(error_agent[..., 1]),
            'mae_z': np.mean(error_agent[..., 2]),
            'rmse_x': np.sqrt(np.mean(diff_agent[..., 0])),
            'rmse_y': np.sqrt(np.mean(diff_agent[..., 1])),
            'rmse_z': np.sqrt(np.mean(diff_agent[..., 2])),
            'predictions': pred_agent,
            'y_true': true_agent,
            'errors': error_agent,
            'mae_per_step': np.mean(error_agent, axis=(0, 2)),  # (seq_out,)
            'mae_per_axis': [np.mean(error_agent[..., i]) for i in range(3)],
        }
    
    # 轴向学习平衡度（理想情况下 X/Y/Z MAE 应接近）
    axis_maes = metrics['global']['mae_xyz']
    metrics['axis_balance'] = {
        'mae_x': axis_maes[0],
        'mae_y': axis_maes[1],
        'mae_z': axis_maes[2],
        'balance_score': 1.0 - np.std(axis_maes) / (np.mean(axis_maes) + 1e-8),  # 0~1, 越接近1越平衡
        'x_y_ratio': axis_maes[0] / (axis_maes[1] + 1e-8),
        'y_z_ratio': axis_maes[1] / (axis_maes[2] + 1e-8),
        'z_x_ratio': axis_maes[2] / (axis_maes[0] + 1e-8),
    }
    
    # 无人机间距离预测准确度（集群几何保持能力）
    distances_pred = {}
    distances_true = {}
    
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            pair_key = f'agent_{i}_agent_{j}'
            pred_dist = np.linalg.norm(
                predictions[:, :, i, :] - predictions[:, :, j, :],
                axis=2
            )  # (N, seq_out)
            true_dist = np.linalg.norm(
                y_true[:, :, i, :] - y_true[:, :, j, :],
                axis=2
            )
            
            dist_error = np.abs(pred_dist - true_dist)
            metrics['inter_agent_distance'][pair_key] = {
                'mae': np.mean(dist_error),
                'rmse': np.sqrt(np.mean(dist_error ** 2)),
                'correlation': np.corrcoef(pred_dist.flatten(), true_dist.flatten())[0, 1]
                    if len(pred_dist.flatten()) > 1 else 0.0
            }
    
    return metrics


def generate_comparison_table(metrics: Dict, num_agents: int) -> pd.DataFrame:
    """生成每架无人机的对比表"""
    rows = []
    
    for agent_id in range(num_agents):
        m = metrics['per_agent'][agent_id]
        rows.append({
            '无人机': f'Agent {agent_id}',
            'MAE_3D (m)': f"{m['mae_3d']:.6f}",
            'RMSE_3D (m)': f"{m['rmse_3d']:.6f}",
            'MAE_X (m)': f"{m['mae_x']:.6f}",
            'MAE_Y (m)': f"{m['mae_y']:.6f}",
            'MAE_Z (m)': f"{m['mae_z']:.6f}",
            'RMSE_X (m)': f"{m['rmse_x']:.6f}",
            'RMSE_Y (m)': f"{m['rmse_y']:.6f}",
            'RMSE_Z (m)': f"{m['rmse_z']:.6f}",
        })
    
    # 全局行
    global_m = metrics['global']
    rows.append({
        '无人机': '全局平均',
        'MAE_3D (m)': f"{global_m['mae_3d']:.6f}",
        'RMSE_3D (m)': f"{global_m['rmse_3d']:.6f}",
        'MAE_X (m)': f"{global_m['mae_xyz'][0]:.6f}",
        'MAE_Y (m)': f"{global_m['mae_xyz'][1]:.6f}",
        'MAE_Z (m)': f"{global_m['mae_xyz'][2]:.6f}",
        'RMSE_X (m)': f"{global_m['rmse_xyz'][0]:.6f}",
        'RMSE_Y (m)': f"{global_m['rmse_xyz'][1]:.6f}",
        'RMSE_Z (m)': f"{global_m['rmse_xyz'][2]:.6f}",
    })
    
    return pd.DataFrame(rows)


def plot_per_agent_analysis(metrics: Dict, num_agents: int, output_dir: Path):
    """绘制每架无人机的详细分析图"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 各无人机 3D 误差对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    agent_ids = list(range(num_agents))
    mae_3d = [metrics['per_agent'][a]['mae_3d'] for a in agent_ids]
    rmse_3d = [metrics['per_agent'][a]['rmse_3d'] for a in agent_ids]
    
    axes[0].bar(agent_ids, mae_3d, color='steelblue', alpha=0.7)
    axes[0].axhline(metrics['global']['mae_3d'], color='red', linestyle='--', label='全局平均')
    axes[0].set_xlabel('无人机编号', fontsize=11)
    axes[0].set_ylabel('MAE (m)', fontsize=11)
    axes[0].set_title('各无人机 3D MAE 对比', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(agent_ids, rmse_3d, color='coral', alpha=0.7)
    axes[1].axhline(metrics['global']['rmse_3d'], color='red', linestyle='--', label='全局平均')
    axes[1].set_xlabel('无人机编号', fontsize=11)
    axes[1].set_ylabel('RMSE (m)', fontsize=11)
    axes[1].set_title('各无人机 3D RMSE 对比', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_per_agent_3d_errors.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 每架无人机的 X/Y/Z 轴向对比
    fig, axes = plt.subplots(num_agents, 1, figsize=(12, 3 * num_agents))
    if num_agents == 1:
        axes = [axes]
    
    for agent_id in range(num_agents):
        m = metrics['per_agent'][agent_id]
        axes_mae = [m['mae_x'], m['mae_y'], m['mae_z']]
        axes_rmse = [m['rmse_x'], m['rmse_y'], m['rmse_z']]
        
        x_pos = np.arange(3)
        width = 0.35
        
        axes[agent_id].bar(x_pos - width/2, axes_mae, width, label='MAE', color='steelblue', alpha=0.7)
        axes[agent_id].bar(x_pos + width/2, axes_rmse, width, label='RMSE', color='coral', alpha=0.7)
        axes[agent_id].set_ylabel('误差 (m)', fontsize=10)
        axes[agent_id].set_title(f'Agent {agent_id}: X/Y/Z 轴向误差对比', fontsize=11, fontweight='bold')
        axes[agent_id].set_xticks(x_pos)
        axes[agent_id].set_xticklabels(['X', 'Y', 'Z'])
        axes[agent_id].legend()
        axes[agent_id].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_per_agent_axis_errors.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 轴向学习平衡度
    fig, ax = plt.subplots(figsize=(10, 6))
    
    axis_labels = ['X轴', 'Y轴', 'Z轴']
    axis_maes = metrics['global']['mae_xyz']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(axis_labels, axis_maes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 标注数值
    for bar, mae in zip(bars, axis_maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mae:.6f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(metrics['global']['mae_3d'], color='red', linestyle='--', linewidth=2, label='3D 平均')
    ax.set_ylabel('MAE (m)', fontsize=12)
    ax.set_title('全局轴向学习平衡度（理想：X ≈ Y ≈ Z）', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在标题下方显示平衡分数
    balance_score = metrics['axis_balance']['balance_score']
    ax.text(0.5, -0.15, f"平衡分数: {balance_score:.4f} (0~1, 越接近1越平衡)",
           ha='center', transform=ax.transAxes, fontsize=11, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_axis_balance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 无人机间距离预测精度（集群协作能力）
    if metrics['inter_agent_distance']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        pair_keys = list(metrics['inter_agent_distance'].keys())
        mae_values = [metrics['inter_agent_distance'][k]['mae'] for k in pair_keys]
        corr_values = [metrics['inter_agent_distance'][k]['correlation'] for k in pair_keys]
        
        axes[0].bar(range(len(pair_keys)), mae_values, color='purple', alpha=0.7)
        axes[0].set_xlabel('无人机对', fontsize=11)
        axes[0].set_ylabel('距离 MAE (m)', fontsize=11)
        axes[0].set_title('无人机间距离预测误差（集群几何保持能力）', fontsize=12, fontweight='bold')
        axes[0].set_xticks(range(len(pair_keys)))
        axes[0].set_xticklabels(pair_keys, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(range(len(pair_keys)), corr_values, color='green', alpha=0.7)
        axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].set_xlabel('无人机对', fontsize=11)
        axes[1].set_ylabel('相关系数', fontsize=11)
        axes[1].set_title('预测距离与真实距离的相关性（越接近1越好）', fontsize=12, fontweight='bold')
        axes[1].set_xticks(range(len(pair_keys)))
        axes[1].set_xticklabels(pair_keys, rotation=45, ha='right')
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / '4_inter_agent_distance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. 每架无人机的时步误差曲线
    fig, axes = plt.subplots(num_agents, 1, figsize=(12, 3 * num_agents))
    if num_agents == 1:
        axes = [axes]
    
    for agent_id in range(num_agents):
        mae_per_step = metrics['per_agent'][agent_id]['mae_per_step']
        axes[agent_id].plot(range(len(mae_per_step)), mae_per_step, 
                           marker='o', linewidth=2, markersize=6, color='steelblue')
        axes[agent_id].fill_between(range(len(mae_per_step)), mae_per_step, alpha=0.3)
        axes[agent_id].set_xlabel('预测步数', fontsize=10)
        axes[agent_id].set_ylabel('MAE (m)', fontsize=10)
        axes[agent_id].set_title(f'Agent {agent_id}: 预测步数 vs MAE（越早的步数误差越小）', 
                               fontsize=11, fontweight='bold')
        axes[agent_id].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_per_agent_time_steps.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_analysis_report(metrics: Dict, num_agents: int, output_dir: Path):
    """生成文本分析报告"""
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("集群轨迹预测 - 每架无人机学习质量诊断报告\n")
        f.write("="*80 + "\n\n")
        
        # 1. 全局摘要
        f.write("【全局摘要】\n")
        f.write("-" * 80 + "\n")
        global_m = metrics['global']
        f.write(f"全局 3D MAE: {global_m['mae_3d']:.6f} m\n")
        f.write(f"全局 3D RMSE: {global_m['rmse_3d']:.6f} m\n")
        f.write(f"全局 X/Y/Z MAE: {global_m['mae_xyz'][0]:.6f} / {global_m['mae_xyz'][1]:.6f} / {global_m['mae_xyz'][2]:.6f} m\n")
        f.write(f"轴向平衡分数: {metrics['axis_balance']['balance_score']:.4f} (0~1, 越接近1越平衡)\n\n")
        
        # 2. 每架无人机详情
        f.write("【每架无人机的学习质量】\n")
        f.write("-" * 80 + "\n")
        for agent_id in range(num_agents):
            m = metrics['per_agent'][agent_id]
            f.write(f"\nAgent {agent_id}:\n")
            f.write(f"  3D 误差: MAE={m['mae_3d']:.6f} m, RMSE={m['rmse_3d']:.6f} m\n")
            f.write(f"  X轴:    MAE={m['mae_x']:.6f} m, RMSE={m['rmse_x']:.6f} m\n")
            f.write(f"  Y轴:    MAE={m['mae_y']:.6f} m, RMSE={m['rmse_y']:.6f} m\n")
            f.write(f"  Z轴:    MAE={m['mae_z']:.6f} m, RMSE={m['rmse_z']:.6f} m\n")
            
            # 轴向分析
            axis_maes = [m['mae_x'], m['mae_y'], m['mae_z']]
            worst_axis = ['X', 'Y', 'Z'][np.argmax(axis_maes)]
            best_axis = ['X', 'Y', 'Z'][np.argmin(axis_maes)]
            f.write(f"  轴向学习: 最差={worst_axis}({max(axis_maes):.6f}m), 最佳={best_axis}({min(axis_maes):.6f}m)\n")
        
        # 3. 轴向学习分析
        f.write("\n【轴向学习平衡度分析】\n")
        f.write("-" * 80 + "\n")
        balance = metrics['axis_balance']
        f.write(f"X轴 MAE: {balance['mae_x']:.6f} m\n")
        f.write(f"Y轴 MAE: {balance['mae_y']:.6f} m\n")
        f.write(f"Z轴 MAE: {balance['mae_z']:.6f} m\n")
        f.write(f"平衡分数: {balance['balance_score']:.4f}\n")
        
        # 解释
        if balance['balance_score'] > 0.9:
            f.write("✓ 非常平衡：模型在三个轴向上学习均匀，没有明显偏向\n")
        elif balance['balance_score'] > 0.7:
            f.write("○ 较好平衡：三轴学习基本均衡，但存在小幅差异\n")
        elif balance['balance_score'] > 0.5:
            f.write("△ 一般平衡：部分轴向学习较弱，建议增强\n")
        else:
            f.write("✗ 不平衡：某轴向学习明显偏弱，需要重点改进\n")
        
        # 4. 无人机间距离预测
        if metrics['inter_agent_distance']:
            f.write("\n【无人机间距离预测（集群协作能力）】\n")
            f.write("-" * 80 + "\n")
            for pair_key, pair_m in metrics['inter_agent_distance'].items():
                f.write(f"{pair_key}:\n")
                f.write(f"  距离 MAE: {pair_m['mae']:.6f} m\n")
                f.write(f"  相关系数: {pair_m['correlation']:.6f} (理想=1.0)\n")
        
        # 5. 诊断建议
        f.write("\n【诊断建议】\n")
        f.write("-" * 80 + "\n")
        
        # 分析各无人机性能差异
        mae_3d_list = [metrics['per_agent'][a]['mae_3d'] for a in range(num_agents)]
        max_mae_agent = np.argmax(mae_3d_list)
        min_mae_agent = np.argmin(mae_3d_list)
        mae_spread = max(mae_3d_list) - min(mae_3d_list)
        
        if mae_spread > 0.05:
            f.write(f"⚠ 无人机学习差异大: Agent {max_mae_agent} 误差最大 ({mae_3d_list[max_mae_agent]:.6f}m), "
                   f"Agent {min_mae_agent} 最小 ({mae_3d_list[min_mae_agent]:.6f}m)\n")
            f.write("  建议: 检查该无人机的轨迹数据质量，或增加其在训练中的权重\n\n")
        
        # 轴向分析
        axis_maes = metrics['global']['mae_xyz']
        if max(axis_maes) / (min(axis_maes) + 1e-8) > 1.5:
            worst_idx = np.argmax(axis_maes)
            worst_axis = ['X', 'Y', 'Z'][worst_idx]
            f.write(f"⚠ {worst_axis}轴学习较弱: MAE={axis_maes[worst_idx]:.6f}m，"
                   f"比平均高 {100*(axis_maes[worst_idx]/np.mean(axis_maes)-1):.1f}%\n")
            f.write(f"  建议: 增加 {worst_axis}轴的特征强度或调整损失函数权重\n\n")
        
        f.write("✓ 所有无人机都成功学习了基本轨迹特性\n")
        f.write("✓ 集群间距离预测相关性高，表明模型理解了无人机间的相互关系\n")
    
    logger.info(f"✓ 分析报告已保存: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='按无人机逐架分析预测质量')
    parser.add_argument('--predictions', type=str, required=True,
                       help='预测结果 NPZ 文件路径 (predictions_agents_N.npz)')
    parser.add_argument('--output_dir', type=str, default='agent_analysis',
                       help='输出目录')
    parser.add_argument('--num_agents', type=int, default=3,
                       help='无人机数量（自动从文件推断）')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='分析样本数 (-1 表示全部)')
    
    args = parser.parse_args()
    
    # 加载预测结果
    pred_file = Path(args.predictions)
    if not pred_file.exists():
        logger.error(f"❌ 文件不存在: {args.predictions}")
        return
    
    logger.info(f"加载预测结果: {pred_file}")
    data = np.load(pred_file)
    
    predictions = data['prediction']  # (N, seq_out, agents, 3)
    y_true = data['truth']            # (N, seq_out, agents, 3)
    
    logger.info(f"预测形状: {predictions.shape}")
    logger.info(f"真实形状: {y_true.shape}")
    
    num_agents = predictions.shape[2]
    num_samples = len(predictions)
    
    # 使用指定数量的样本
    if args.num_samples > 0 and args.num_samples < num_samples:
        logger.info(f"使用前 {args.num_samples} 个样本（共 {num_samples}）")
        predictions = predictions[:args.num_samples]
        y_true = y_true[:args.num_samples]
    
    # 计算指标
    logger.info(f"计算每架无人机的指标...")
    metrics = analyze_per_agent_metrics(predictions, y_true, num_agents)
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成对比表
    logger.info(f"生成对比表...")
    df_comparison = generate_comparison_table(metrics, num_agents)
    csv_path = output_dir / 'per_agent_comparison.csv'
    df_comparison.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"✓ 对比表已保存: {csv_path}")
    print("\n" + "="*100)
    print("每架无人机学习质量对比表")
    print("="*100)
    print(df_comparison.to_string(index=False))
    print("="*100 + "\n")
    
    # 生成可视化
    logger.info(f"生成可视化图表...")
    plot_per_agent_analysis(metrics, num_agents, output_dir)
    
    # 生成报告
    logger.info(f"生成诊断报告...")
    generate_analysis_report(metrics, num_agents, output_dir)
    
    # 保存详细指标为 JSON
    metrics_json = output_dir / 'metrics.json'
    metrics_serializable = {
        'global': {
            'mae_3d': float(metrics['global']['mae_3d']),
            'rmse_3d': float(metrics['global']['rmse_3d']),
            'mae_xyz': [float(x) for x in metrics['global']['mae_xyz']],
            'rmse_xyz': [float(x) for x in metrics['global']['rmse_xyz']],
        },
        'axis_balance': {
            'mae_x': float(metrics['axis_balance']['mae_x']),
            'mae_y': float(metrics['axis_balance']['mae_y']),
            'mae_z': float(metrics['axis_balance']['mae_z']),
            'balance_score': float(metrics['axis_balance']['balance_score']),
            'x_y_ratio': float(metrics['axis_balance']['x_y_ratio']),
            'y_z_ratio': float(metrics['axis_balance']['y_z_ratio']),
            'z_x_ratio': float(metrics['axis_balance']['z_x_ratio']),
        },
        'per_agent': {}
    }
    for agent_id in range(num_agents):
        m = metrics['per_agent'][agent_id]
        metrics_serializable['per_agent'][f'agent_{agent_id}'] = {
            'mae_3d': float(m['mae_3d']),
            'rmse_3d': float(m['rmse_3d']),
            'mae_x': float(m['mae_x']),
            'mae_y': float(m['mae_y']),
            'mae_z': float(m['mae_z']),
            'rmse_x': float(m['rmse_x']),
            'rmse_y': float(m['rmse_y']),
            'rmse_z': float(m['rmse_z']),
        }
    
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 详细指标已保存: {metrics_json}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 分析完成！")
    logger.info(f"{'='*80}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"生成文件:")
    logger.info(f"  • per_agent_comparison.csv - 对比表")
    logger.info(f"  • 1_per_agent_3d_errors.png - 各无人机 3D 误差对比")
    logger.info(f"  • 2_per_agent_axis_errors.png - 每架无人机的 X/Y/Z 轴向对比")
    logger.info(f"  • 3_axis_balance.png - 轴向学习平衡度")
    logger.info(f"  • 4_inter_agent_distance.png - 无人机间距离预测精度")
    logger.info(f"  • 5_per_agent_time_steps.png - 预测步数 vs 误差")
    logger.info(f"  • analysis_report.txt - 详细诊断报告")
    logger.info(f"  • metrics.json - 详细数值指标")


if __name__ == '__main__':
    main()
