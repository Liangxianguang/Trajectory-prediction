#!/usr/bin/env python3
"""
v3_new 推断结果可视化脚本 - 验证末端动力学增强效果
==================================================

功能：
✅ 加载 v3_new 推断结果 (.npz 文件)
✅ 与真实轨迹对比，生成 6 视图对比图
✅ 显示误差演化和末端动力学特征
✅ 同时支持对比增强 vs 无增强的预测结果
✅ 输出详细评估报告

使用示例：
    # 可视化单个模型推断结果
    python visualize_v3_new_inference.py ^
        --result_file infer_results_v3_new/predictions_agents_3_v3.npz ^
        --num_samples 5 ^
        --output_dir visualization_v3_new

    # 对比增强 vs 无增强
    python visualize_v3_new_inference.py ^
        --result_file enhanced/predictions_agents_3_v3.npz ^
        --baseline_file baseline/predictions_agents_3_v3.npz ^
        --num_samples 5 ^
        --output_dir visualization_comparison
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from datetime import datetime
import json
import logging
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_result_file(result_file):
    """加载推断结果 .npz 文件
    
    Returns:
        dict with keys: input, truth, prediction, mae, rmse, mae_per_step, mae_per_agent
    """
    if not Path(result_file).exists():
        raise FileNotFoundError(f"结果文件不存在: {result_file}")
    
    data = np.load(result_file)
    result = {
        'input': data['input'],           # (samples, seq_in, agents, 3)
        'truth': data['truth'],           # (samples, seq_out, agents, 3)
        'prediction': data['prediction'], # (samples, seq_out, agents, 3)
    }
    
    # 从 .npz 中加载标量指标（如果存在）
    if 'mae' in data:
        result['mae'] = float(data['mae'])
    if 'rmse' in data:
        result['rmse'] = float(data['rmse'])
    if 'mae_per_step' in data:
        result['mae_per_step'] = data['mae_per_step']
    if 'mae_per_agent' in data:
        result['mae_per_agent'] = data['mae_per_agent']
    
    return result


def compute_endpoint_dynamics(trajectory, window_size=5, dt=0.1):
    """计算轨迹末端的动力学特征
    
    Args:
        trajectory: (seq, agents, 3) 轨迹
        window_size: 分析窗口大小
        dt: 时间步长
    
    Returns:
        dict with: curvature_seq, velocity_direction_change, angular_velocity
    """
    seq_len, num_agents, _ = trajectory.shape
    
    # 提取末尾窗口
    start_idx = max(0, seq_len - window_size)
    endpoint_traj = trajectory[start_idx:, :, :]  # (N, agents, 3)
    
    # 计算速度和加速度
    vel = np.diff(endpoint_traj, axis=0) / dt  # (N-1, agents, 3)
    acc = np.diff(vel, axis=0) / dt if vel.shape[0] > 1 else np.zeros_like(vel)
    
    # 计算曲率 κ = |v × a| / |v|^3
    curvatures = []
    for i in range(vel.shape[0] - 1):
        v = vel[i]  # (agents, 3)
        a = acc[i]  # (agents, 3)
        
        cross = np.cross(v, a)  # (agents, 3)
        cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)  # (agents, 1)
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)  # (agents, 1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            curv = cross_norm[:, 0] / (v_norm[:, 0] ** 3 + 1e-8)
        curvatures.append(curv)
    
    if curvatures:
        curvatures = np.array(curvatures)  # (N-2, agents)
    else:
        curvatures = np.zeros((vel.shape[0], num_agents))
    
    # 速度方向变化
    vel_direction_changes = []
    for i in range(vel.shape[0] - 1):
        v1 = vel[i] / (np.linalg.norm(vel[i], axis=1, keepdims=True) + 1e-8)
        v2 = vel[i+1] / (np.linalg.norm(vel[i+1], axis=1, keepdims=True) + 1e-8)
        
        dot_prod = np.sum(v1 * v2, axis=1)
        angle = np.arccos(np.clip(dot_prod, -1.0, 1.0))
        vel_direction_changes.append(angle)
    
    if vel_direction_changes:
        vel_direction_changes = np.array(vel_direction_changes)  # (N-1, agents)
    else:
        vel_direction_changes = np.zeros((vel.shape[0], num_agents))
    
    # 角速度 ω = |v × a| / |v|^2
    angular_velocities = []
    for i in range(vel.shape[0] - 1):
        v = vel[i]
        a = acc[i]
        
        cross = np.cross(v, a)
        cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            omega = cross_norm[:, 0] / (v_norm[:, 0] ** 2 + 1e-8)
        angular_velocities.append(omega)
    
    if angular_velocities:
        angular_velocities = np.array(angular_velocities)  # (N-2, agents)
    else:
        angular_velocities = np.zeros((vel.shape[0], num_agents))
    
    return {
        'curvature_seq': curvatures,
        'velocity_direction_change': vel_direction_changes,
        'angular_velocity': angular_velocities,
    }


def plot_prediction_comparison(history, truth, pred, sample_idx, agents, output_dir, 
                               baseline_pred=None, tail_window=5):
    """绘制预测对比图（增强 vs 真实，可选对比基线）
    
    Args:
        history: (seq_in, agents, 3)
        truth: (seq_out, agents, 3)
        pred: (seq_out, agents, 3)
        baseline_pred: (seq_out, agents, 3) optional
        sample_idx: 样本索引
        agents: agent 数量
        output_dir: 输出目录
        tail_window: 末端窗口大小
    """
    
    fig = plt.figure(figsize=(20, 13))
    
    # 1. 3D 轨迹对比
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, agents))
    
    for agent_id in range(agents):
        color = colors[agent_id]
        
        # 历史轨迹
        ax3d.plot(history[:, agent_id, 0], history[:, agent_id, 1], history[:, agent_id, 2],
                 'o-', color=color, linewidth=2, markersize=4, alpha=0.6,
                 label='History' if agent_id == 0 else '')
        
        # 连接历史最后一点
        last_point = history[-1, agent_id, :]
        
        # 真实未来轨迹（绿色）
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        ax3d.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
                 's-', color='green', linewidth=2.5, markersize=6, alpha=0.85,
                 label='True' if agent_id == 0 else '')
        
        # 增强预测（蓝色虚线）
        pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
        ax3d.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
                 '^--', color='blue', linewidth=2.5, markersize=5, alpha=0.75,
                 label='Enhanced Pred' if agent_id == 0 else '')
        
        # 基线预测（红色虚线，如果提供）
        if baseline_pred is not None:
            baseline_traj = np.vstack([last_point, baseline_pred[:, agent_id, :]])
            ax3d.plot(baseline_traj[:, 0], baseline_traj[:, 1], baseline_traj[:, 2],
                     'x--', color='red', linewidth=2, markersize=4, alpha=0.6,
                     label='Baseline' if agent_id == 0 else '')
    
    ax3d.set_xlabel('X (m)', fontsize=10, fontweight='bold')
    ax3d.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
    ax3d.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
    title_str = f'Sample {sample_idx}: 3D Trajectories\n'
    if baseline_pred is not None:
        title_str += 'Enhanced (blue) vs Baseline (red) vs True (green)'
    else:
        title_str += 'Enhanced (blue) vs True (green)'
    ax3d.set_title(title_str, fontsize=11, fontweight='bold')
    ax3d.legend(fontsize=9, loc='upper left')
    ax3d.grid(True, alpha=0.3)
    
    # 2. XY 平面
    ax_xy = fig.add_subplot(2, 3, 2)
    for agent_id in range(agents):
        color = colors[agent_id]
        last_point = history[-1, agent_id, :]
        
        ax_xy.plot(history[:, agent_id, 0], history[:, agent_id, 1],
                  'o-', color=color, linewidth=2, markersize=4, alpha=0.6)
        
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        ax_xy.plot(true_traj[:, 0], true_traj[:, 1],
                  's-', color='green', linewidth=2.5, markersize=6, alpha=0.85,
                  label='True' if agent_id == 0 else '')
        
        pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
        ax_xy.plot(pred_traj[:, 0], pred_traj[:, 1],
                  '^--', color='blue', linewidth=2.5, markersize=5, alpha=0.75,
                  label='Enhanced' if agent_id == 0 else '')
        
        if baseline_pred is not None:
            baseline_traj = np.vstack([last_point, baseline_pred[:, agent_id, :]])
            ax_xy.plot(baseline_traj[:, 0], baseline_traj[:, 1],
                      'x--', color='red', linewidth=2, markersize=4, alpha=0.6,
                      label='Baseline' if agent_id == 0 else '')
    
    ax_xy.set_xlabel('X (m)', fontsize=10, fontweight='bold')
    ax_xy.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
    ax_xy.set_title('XY 平面', fontsize=11, fontweight='bold')
    ax_xy.legend(fontsize=9, loc='best')
    ax_xy.grid(True, alpha=0.3)
    
    # 3. XZ 平面
    ax_xz = fig.add_subplot(2, 3, 3)
    for agent_id in range(agents):
        color = colors[agent_id]
        last_point = history[-1, agent_id, :]
        
        ax_xz.plot(history[:, agent_id, 0], history[:, agent_id, 2],
                  'o-', color=color, linewidth=2, markersize=4, alpha=0.6)
        
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        ax_xz.plot(true_traj[:, 0], true_traj[:, 2],
                  's-', color='green', linewidth=2.5, markersize=6, alpha=0.85)
        
        pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
        ax_xz.plot(pred_traj[:, 0], pred_traj[:, 2],
                  '^--', color='blue', linewidth=2.5, markersize=5, alpha=0.75)
        
        if baseline_pred is not None:
            baseline_traj = np.vstack([last_point, baseline_pred[:, agent_id, :]])
            ax_xz.plot(baseline_traj[:, 0], baseline_traj[:, 2],
                      'x--', color='red', linewidth=2, markersize=4, alpha=0.6)
    
    ax_xz.set_xlabel('X (m)', fontsize=10, fontweight='bold')
    ax_xz.set_ylabel('Z (m)', fontsize=10, fontweight='bold')
    ax_xz.set_title('XZ 平面', fontsize=11, fontweight='bold')
    ax_xz.grid(True, alpha=0.3)
    
    # 4. YZ 平面
    ax_yz = fig.add_subplot(2, 3, 4)
    for agent_id in range(agents):
        color = colors[agent_id]
        last_point = history[-1, agent_id, :]
        
        ax_yz.plot(history[:, agent_id, 1], history[:, agent_id, 2],
                  'o-', color=color, linewidth=2, markersize=4, alpha=0.6)
        
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        ax_yz.plot(true_traj[:, 1], true_traj[:, 2],
                  's-', color='green', linewidth=2.5, markersize=6, alpha=0.85)
        
        pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
        ax_yz.plot(pred_traj[:, 1], pred_traj[:, 2],
                  '^--', color='blue', linewidth=2.5, markersize=5, alpha=0.75)
        
        if baseline_pred is not None:
            baseline_traj = np.vstack([last_point, baseline_pred[:, agent_id, :]])
            ax_yz.plot(baseline_traj[:, 1], baseline_traj[:, 2],
                      'x--', color='red', linewidth=2, markersize=4, alpha=0.6)
    
    ax_yz.set_xlabel('Y (m)', fontsize=10, fontweight='bold')
    ax_yz.set_ylabel('Z (m)', fontsize=10, fontweight='bold')
    ax_yz.set_title('YZ 平面', fontsize=11, fontweight='bold')
    ax_yz.grid(True, alpha=0.3)
    
    # 5. 误差对比（沿时间步）
    ax_err = fig.add_subplot(2, 3, 5)
    
    steps = np.arange(truth.shape[0])
    
    # 增强版误差
    err_enhanced = np.linalg.norm(pred - truth, axis=2).mean(axis=1)  # (seq_out,)
    ax_err.plot(steps, err_enhanced, 'b-o', linewidth=2.5, markersize=6,
               label='Enhanced', alpha=0.8)
    
    # 基线版误差
    if baseline_pred is not None:
        err_baseline = np.linalg.norm(baseline_pred - truth, axis=2).mean(axis=1)
        ax_err.plot(steps, err_baseline, 'r-x', linewidth=2.5, markersize=5,
                   label='Baseline', alpha=0.8)
        
        # 显示改善百分比
        improvement = (err_baseline - err_enhanced) / (err_baseline + 1e-8) * 100
        mean_improvement = improvement.mean()
        ax_err.text(0.98, 0.97, f'Avg Improvement: {mean_improvement:.2f}%',
                   transform=ax_err.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax_err.set_xlabel('Prediction Step', fontsize=10, fontweight='bold')
    ax_err.set_ylabel('Mean Position Error (m)', fontsize=10, fontweight='bold')
    ax_err.set_title('误差演化', fontsize=11, fontweight='bold')
    ax_err.legend(fontsize=10, loc='best')
    ax_err.grid(True, alpha=0.3)
    
    # 6. 末端动力学特征
    ax_dyn = fig.add_subplot(2, 3, 6)
    
    # 计算末端动力学
    endpoint_dyn = compute_endpoint_dynamics(history, window_size=tail_window)
    
    # 绘制末端曲率和速度方向变化
    if endpoint_dyn['curvature_seq'].shape[0] > 0:
        curv_mean = endpoint_dyn['curvature_seq'].mean(axis=1)
        tail_steps = np.arange(len(curv_mean))
        
        ax_dyn_twin = ax_dyn.twinx()
        
        ln1 = ax_dyn.plot(tail_steps, curv_mean, 'g-o', linewidth=2.5, markersize=6,
                         label='Curvature (末端)', alpha=0.8)
        
        vel_change_mean = endpoint_dyn['velocity_direction_change'].mean(axis=1)
        if len(vel_change_mean) > 0:
            ln2 = ax_dyn_twin.plot(tail_steps[:len(vel_change_mean)], vel_change_mean,
                                   'purple', marker='s', linewidth=2.5, markersize=5,
                                   label='Velocity Dir Change', alpha=0.8)
        
        ax_dyn.set_xlabel('Tail Step Index', fontsize=10, fontweight='bold')
        ax_dyn.set_ylabel('Curvature', color='g', fontsize=10, fontweight='bold')
        ax_dyn_twin.set_ylabel('Velocity Direction Change (rad)', color='purple', 
                              fontsize=10, fontweight='bold')
        ax_dyn.set_title('末端动力学特征', fontsize=11, fontweight='bold')
        ax_dyn.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    png_file = output_path / f'prediction_sample_{sample_idx:06d}.png'
    fig.savefig(png_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ 图表已保存: {png_file.name}")
    
    plt.close(fig)


def compute_sample_metrics(truth, pred, baseline_pred=None):
    """计算单个样本的评估指标"""
    # 增强版指标
    err_enhanced = np.linalg.norm(pred - truth, axis=2)  # (seq_out, agents)
    mae_enhanced = err_enhanced.mean()
    rmse_enhanced = np.sqrt((err_enhanced ** 2).mean())
    
    # 轴向误差
    mae_x = np.abs(pred[..., 0] - truth[..., 0]).mean()
    mae_y = np.abs(pred[..., 1] - truth[..., 1]).mean()
    mae_z = np.abs(pred[..., 2] - truth[..., 2]).mean()
    
    metrics = {
        'mae_enhanced': float(mae_enhanced),
        'rmse_enhanced': float(rmse_enhanced),
        'mae_x': float(mae_x),
        'mae_y': float(mae_y),
        'mae_z': float(mae_z),
    }
    
    # 基线版本指标（如果提供）
    if baseline_pred is not None:
        err_baseline = np.linalg.norm(baseline_pred - truth, axis=2)
        mae_baseline = err_baseline.mean()
        rmse_baseline = np.sqrt((err_baseline ** 2).mean())
        
        improvement_mae = (mae_baseline - mae_enhanced) / (mae_baseline + 1e-8) * 100
        improvement_rmse = (rmse_baseline - rmse_enhanced) / (rmse_baseline + 1e-8) * 100
        
        metrics['mae_baseline'] = float(mae_baseline)
        metrics['rmse_baseline'] = float(rmse_baseline)
        metrics['improvement_mae_percent'] = float(improvement_mae)
        metrics['improvement_rmse_percent'] = float(improvement_rmse)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='v3_new 推断结果可视化')
    
    parser.add_argument('--result_file', required=True, help='增强后的推断结果 .npz 文件')
    parser.add_argument('--baseline_file', help='基线推断结果 .npz 文件（用于对比）')
    parser.add_argument('--num_samples', type=int, default=5, help='可视化样本数')
    parser.add_argument('--output_dir', default='visualization_v3_new', help='输出目录')
    parser.add_argument('--tail_window', type=int, default=5, help='末端窗口大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # 加载结果
    logger.info(f"加载增强后的结果: {args.result_file}")
    result_enhanced = load_result_file(args.result_file)
    
    result_baseline = None
    if args.baseline_file:
        logger.info(f"加载基线结果: {args.baseline_file}")
        result_baseline = load_result_file(args.baseline_file)
    
    # 样本选择
    num_available = result_enhanced['input'].shape[0]
    num_samples = min(args.num_samples, num_available)
    sample_indices = np.random.choice(num_available, num_samples, replace=False)
    
    logger.info(f"总样本数: {num_available}, 可视化: {num_samples}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化样本
    all_metrics = []
    
    for i, sample_idx in enumerate(sample_indices):
        logger.info(f"\n样本 {i+1}/{num_samples} (索引 {sample_idx}):")
        
        history = result_enhanced['input'][sample_idx]          # (seq_in, agents, 3)
        truth = result_enhanced['truth'][sample_idx]            # (seq_out, agents, 3)
        pred_enhanced = result_enhanced['prediction'][sample_idx]  # (seq_out, agents, 3)
        
        pred_baseline = None
        if result_baseline is not None:
            pred_baseline = result_baseline['prediction'][sample_idx]
        
        # 绘制对比图
        plot_prediction_comparison(
            history, truth, pred_enhanced, sample_idx, 
            result_enhanced['input'].shape[2],
            args.output_dir,
            baseline_pred=pred_baseline,
            tail_window=args.tail_window
        )
        
        # 计算指标
        metrics = compute_sample_metrics(truth, pred_enhanced, pred_baseline)
        metrics['sample_idx'] = int(sample_idx)
        all_metrics.append(metrics)
        
        logger.info(f"  增强 MAE: {metrics['mae_enhanced']:.6f} m")
        if 'mae_baseline' in metrics:
            logger.info(f"  基线 MAE: {metrics['mae_baseline']:.6f} m")
            logger.info(f"  改善: {metrics['improvement_mae_percent']:.2f}%")
    
    # 生成报告
    logger.info("\n生成总结报告...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'result_file': str(args.result_file),
        'baseline_file': str(args.baseline_file) if args.baseline_file else None,
        'num_samples_visualized': num_samples,
        'sample_indices': [int(idx) for idx in sample_indices],
        'samples': all_metrics,
    }
    
    if all_metrics:
        avg_mae_enhanced = np.mean([m['mae_enhanced'] for m in all_metrics])
        report['average_mae_enhanced'] = float(avg_mae_enhanced)
        
        if 'mae_baseline' in all_metrics[0]:
            avg_mae_baseline = np.mean([m['mae_baseline'] for m in all_metrics])
            avg_improvement = np.mean([m['improvement_mae_percent'] for m in all_metrics])
            
            report['average_mae_baseline'] = float(avg_mae_baseline)
            report['average_improvement_percent'] = float(avg_improvement)
            
            logger.info(f"\n=== 总体对比结果 ===")
            logger.info(f"基线平均 MAE: {avg_mae_baseline:.6f} m")
            logger.info(f"增强平均 MAE: {avg_mae_enhanced:.6f} m")
            logger.info(f"平均改善: {avg_improvement:.2f}%")
        else:
            logger.info(f"\n=== 总体结果 ===")
            logger.info(f"增强平均 MAE: {avg_mae_enhanced:.6f} m")
    
    report_file = output_dir / f'visualization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 报告已保存: {report_file.name}")
    logger.info(f"\n✓ 可视化完成！输出目录: {output_dir}")


if __name__ == '__main__':
    main()
