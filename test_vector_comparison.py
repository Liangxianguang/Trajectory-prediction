#!/usr/bin/env python3
"""
对比测试：VECTOR 增强预测 vs 原始预测方法
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))


def load_trajectory(csv_path):
    """加载轨迹 CSV"""
    df = pd.read_csv(csv_path)
    traj = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    return traj


def compute_velocity(trajectory, dt=0.1):
    """计算速度"""
    velocity = np.diff(trajectory, axis=0) / dt
    return velocity


def test_single_trajectory(trajectory_path, predictor_direct, predictor_velocity,
                          inp_len=20, out_len=10, dt=0.1):
    """测试单个轨迹"""
    logger.info(f"\n{'='*80}")
    logger.info(f"测试轨迹: {Path(trajectory_path).name}")
    logger.info(f"{'='*80}")
    
    # 加载轨迹
    trajectory = load_trajectory(trajectory_path)
    logger.info(f"轨迹形状: {trajectory.shape}")
    
    if len(trajectory) < inp_len + out_len:
        logger.warning(f"轨迹太短，跳过")
        return None
    
    # 分割：输入 + 真实后续
    inp_pos = trajectory[-(inp_len + out_len):-out_len, :]
    actual_future = trajectory[-out_len:, :]
    full_trajectory = trajectory[-(inp_len + out_len):, :]
    
    # 方法 1：直接位置预测
    logger.info("\n【方法 1】直接位置预测")
    try:
        pred_direct = predictor_direct.predict_enhanced(full_trajectory, method='position', 
                                                        input_length=inp_len, dt=dt)
        shift = actual_future[0] - pred_direct[0]
        pred_direct_aligned = pred_direct + shift
        metrics_direct = predictor_direct.evaluate_prediction_quality(actual_future, pred_direct_aligned)
        logger.info(f"  MAE: {metrics_direct['mae']:.4f}m")
        logger.info(f"  RMSE: {metrics_direct['rmse']:.4f}m")
        logger.info(f"  R²: {metrics_direct['r_squared']:.4f}")
    except Exception as e:
        logger.error(f"  失败: {e}")
        pred_direct_aligned = None
        metrics_direct = None
    
    # 方法 2：速度积分预测（VECTOR 方法）
    logger.info("\n【方法 2】速度积分预测（VECTOR）")
    try:
        pred_velocity = predictor_velocity.predict_enhanced(full_trajectory, method='velocity',
                                                           input_length=inp_len, dt=dt)
        shift = actual_future[0] - pred_velocity[0]
        pred_velocity_aligned = pred_velocity + shift
        metrics_velocity = predictor_velocity.evaluate_prediction_quality(actual_future, pred_velocity_aligned)
        logger.info(f"  MAE: {metrics_velocity['mae']:.4f}m")
        logger.info(f"  RMSE: {metrics_velocity['rmse']:.4f}m")
        logger.info(f"  R²: {metrics_velocity['r_squared']:.4f}")
    except Exception as e:
        logger.error(f"  失败: {e}")
        pred_velocity_aligned = None
        metrics_velocity = None
    
    # 对比结果
    if metrics_direct and metrics_velocity:
        logger.info(f"\n【对比】")
        mae_improvement = (metrics_direct['mae'] - metrics_velocity['mae']) / metrics_direct['mae'] * 100
        rmse_improvement = (metrics_direct['rmse'] - metrics_velocity['rmse']) / metrics_direct['rmse'] * 100
        
        logger.info(f"MAE 改进: {mae_improvement:+.2f}% {'✓' if mae_improvement > 0 else '✗'}")
        logger.info(f"RMSE 改进: {rmse_improvement:+.2f}% {'✓' if rmse_improvement > 0 else '✗'}")
    
    return {
        'trajectory': Path(trajectory_path).name,
        'inp_pos': inp_pos,
        'actual_future': actual_future,
        'pred_direct': pred_direct_aligned,
        'pred_velocity': pred_velocity_aligned,
        'metrics_direct': metrics_direct,
        'metrics_velocity': metrics_velocity
    }


def visualize_comparison(results, output_dir):
    """可视化对比结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('VECTOR 增强预测 vs 直接位置预测对比', fontsize=16, fontweight='bold')
    
    colors_methods = {'直接位置预测': '#FF6B6B', 'VECTOR(速度积分)': '#4ECDC4'}
    
    for idx, result in enumerate(results):
        if result is None or result['pred_direct'] is None or result['pred_velocity'] is None:
            continue
        
        inp_pos = result['inp_pos']
        actual = result['actual_future']
        pred_direct = result['pred_direct']
        pred_velocity = result['pred_velocity']
        
        # 3D 轨迹对比（第 1 行）
        ax1 = axes[0, 0] if idx == 0 else axes[0, 0]
        ax1.plot(inp_pos[:, 0], inp_pos[:, 1], 'b-o', linewidth=2, markersize=4, label='输入(观测)', alpha=0.7)
        ax1.plot(actual[:, 0], actual[:, 1], 'g-s', linewidth=3, markersize=8, label='真实', alpha=0.9, zorder=10)
        ax1.plot(pred_direct[:, 0], pred_direct[:, 1], linestyle='--', marker='^', 
                linewidth=2, color=colors_methods['直接位置预测'], label='直接预测', alpha=0.8)
        ax1.plot(pred_velocity[:, 0], pred_velocity[:, 1], linestyle=':', marker='v',
                linewidth=2, color=colors_methods['VECTOR(速度积分)'], label='VECTOR', alpha=0.8)
        ax1.set_xlabel('X (m)', fontweight='bold')
        ax1.set_ylabel('Y (m)', fontweight='bold')
        ax1.set_title('XY 平面对比', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        ax1.axis('equal')
        
        break  # 只画第一个结果以避免重叠
    
    # MAE 对比（第 2 行左）
    ax_mae = axes[1, 0]
    methods = ['直接位置预测', 'VECTOR(速度积分)']
    maes = []
    for result in results:
        if result is None or result['metrics_direct'] is None or result['metrics_velocity'] is None:
            continue
        maes.append([result['metrics_direct']['mae'], result['metrics_velocity']['mae']])
    
    if maes:
        maes = np.array(maes).mean(axis=0)
        bars = ax_mae.bar(methods, maes, color=[colors_methods[m] for m in methods], alpha=0.8, edgecolor='black', linewidth=2)
        ax_mae.set_ylabel('MAE (m)', fontweight='bold')
        ax_mae.set_title('平均绝对误差对比', fontweight='bold')
        ax_mae.grid(True, alpha=0.3, axis='y')
        for bar, mae in zip(bars, maes):
            ax_mae.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{mae:.3f}m', ha='center', va='bottom', fontweight='bold')
    
    # RMSE 对比（第 2 行中）
    ax_rmse = axes[1, 1]
    rmses = []
    for result in results:
        if result is None or result['metrics_direct'] is None or result['metrics_velocity'] is None:
            continue
        rmses.append([result['metrics_direct']['rmse'], result['metrics_velocity']['rmse']])
    
    if rmses:
        rmses = np.array(rmses).mean(axis=0)
        bars = ax_rmse.bar(methods, rmses, color=[colors_methods[m] for m in methods], alpha=0.8, edgecolor='black', linewidth=2)
        ax_rmse.set_ylabel('RMSE (m)', fontweight='bold')
        ax_rmse.set_title('均方根误差对比', fontweight='bold')
        ax_rmse.grid(True, alpha=0.3, axis='y')
        for bar, rmse in zip(bars, rmses):
            ax_rmse.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{rmse:.3f}m', ha='center', va='bottom', fontweight='bold')
    
    # R² 对比（第 2 行右）
    ax_r2 = axes[1, 2]
    r2s = []
    for result in results:
        if result is None or result['metrics_direct'] is None or result['metrics_velocity'] is None:
            continue
        r2s.append([result['metrics_direct']['r_squared'], result['metrics_velocity']['r_squared']])
    
    if r2s:
        r2s = np.array(r2s).mean(axis=0)
        bars = ax_r2.bar(methods, r2s, color=[colors_methods[m] for m in methods], alpha=0.8, edgecolor='black', linewidth=2)
        ax_r2.set_ylabel('R² 系数', fontweight='bold')
        ax_r2.set_title('R² 对比', fontweight='bold')
        ax_r2.set_ylim([0, 1.0])
        ax_r2.grid(True, alpha=0.3, axis='y')
        for bar, r2 in zip(bars, r2s):
            ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    png_path = output_dir / 'vector_comparison.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n✓ 对比图表已保存: {png_path}")
    
    return png_path


def main():
    # 初始化预测器
    logger.info("初始化预测器...")
    
    # 模型和统计量路径
    models_dir = Path(r'D:\Trajectory prediction\drone_path_predictor_ros-main\config\mixed_dataset')
    
    # 位置模型（两个预测器共用）
    pos_model_path = models_dir / 'mix_pos_max_norm_128_3_0p5.pth'
    pos_stats_path = models_dir / 'pos_stats.npz'
    
    # 速度模型（仅用于速度积分预测器）
    vel_model_path = models_dir / 'mix_vel_max_norm_128_3_0p5.pth'
    vel_stats_path = models_dir / 'vel_stats.npz'
    
    # 预测器 1：直接位置预测（不用速度模型）
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'drone_path_predictor_ros-main' / 'drone_path_predictor_ros'))
        from vector_predictor_enhanced import EnhancedPredictorGRU
        predictor_direct = EnhancedPredictorGRU(
            str(pos_model_path), str(pos_stats_path),
            use_velocity_integration=False
        )
        logger.info("✓ 直接预测器初始化完成")
    except Exception as e:
        logger.error(f"初始化直接预测器失败: {e}")
        return
    
    # 预测器 2：速度积分预测（VECTOR 方法）
    try:
        predictor_velocity = EnhancedPredictorGRU(
            str(pos_model_path), str(pos_stats_path),
            str(vel_model_path), str(vel_stats_path),
            use_velocity_integration=True,
            enforce_first_step_continuity=True
        )
        logger.info("✓ VECTOR 预测器初始化完成")
    except Exception as e:
        logger.error(f"初始化 VECTOR 预测器失败: {e}")
        return
    
    # 测试轨迹列表
    traj_dir = Path(r'D:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories')
    test_trajectories = list(traj_dir.glob('gazebo_trajectory_*.csv'))[:10]  # 测试前 10 个
    
    logger.info(f"\n找到 {len(test_trajectories)} 个测试轨迹")
    
    # 测试所有轨迹
    results = []
    for traj_path in test_trajectories:
        result = test_single_trajectory(traj_path, predictor_direct, predictor_velocity)
        results.append(result)
    
    # 统计汇总
    logger.info(f"\n{'='*80}")
    logger.info("【总体统计】")
    logger.info(f"{'='*80}")
    
    valid_results = [r for r in results if r is not None and r['metrics_direct'] and r['metrics_velocity']]
    
    if valid_results:
        avg_mae_direct = np.mean([r['metrics_direct']['mae'] for r in valid_results])
        avg_mae_velocity = np.mean([r['metrics_velocity']['mae'] for r in valid_results])
        
        avg_rmse_direct = np.mean([r['metrics_direct']['rmse'] for r in valid_results])
        avg_rmse_velocity = np.mean([r['metrics_velocity']['rmse'] for r in valid_results])
        
        logger.info(f"\n平均 MAE:")
        logger.info(f"  直接预测: {avg_mae_direct:.4f}m")
        logger.info(f"  VECTOR:   {avg_mae_velocity:.4f}m")
        logger.info(f"  改进:     {(avg_mae_direct - avg_mae_velocity) / avg_mae_direct * 100:+.2f}%")
        
        logger.info(f"\n平均 RMSE:")
        logger.info(f"  直接预测: {avg_rmse_direct:.4f}m")
        logger.info(f"  VECTOR:   {avg_rmse_velocity:.4f}m")
        logger.info(f"  改进:     {(avg_rmse_direct - avg_rmse_velocity) / avg_rmse_direct * 100:+.2f}%")
        
        # 生成对比图表
        output_dir = Path(r'D:\Trajectory prediction\evaluation_results')
        visualize_comparison(results, output_dir)
    
    logger.info(f"\n✓ 测试完成！")


if __name__ == '__main__':
    main()
