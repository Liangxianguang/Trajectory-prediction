#!/usr/bin/env python3
"""
GNN 模型推理脚本 - 适配 train_swarm_gnn.py
==============================================

功能：
1. 加载 train_swarm_gnn.py 训练的最佳模型
2. 支持 GCN 和 非 GCN 模式推理
3. 支持三种位置重建方法：
   - direct: 直接反归一化
   - simple: 简单积分 + 速度约束
   - physics_constrained: 物理约束 + 加速度平滑（推荐）
4. 自动反归一化到物理空间
5. 计算 MAE/RMSE/MAPE 等多维指标
6. 生成 3D 轨迹对比可视化
7. 保存推理结果为 NPZ 文件

用法：
  # 推理无 GCN 模型（物理约束重建，推荐）
  python infer_gnn_model.py ^
    --model gru_models_subset_nogcn1/best_model_agents_3.pt ^
    --data_dir swarm_segments ^
    --agents 3 ^
    --batch_size 256 ^
    --num_samples 100 ^
    --use_gcn 0 ^
    --reconstruction_method physics_constrained ^
    --smoothing_weight 0.3 ^
    --visualize ^
    --num_vis 10 ^
    --output_dir infer_results_gnn_nogcn

  # 对比三种重建方法
  python infer_gnn_model.py --model ... --reconstruction_method direct
  python infer_gnn_model.py --model ... --reconstruction_method simple
  python infer_gnn_model.py --model ... --reconstruction_method physics_constrained

  # 推理有 GCN 模型
  python infer_gnn_model.py ^
    --model gru_models_gnn/best_model_agents_3.pt ^
    --data_dir swarm_segments ^
    --agents 3 ^
    --use_gcn 1 ^
    --reconstruction_method physics_constrained ^
    --visualize
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
from datetime import datetime

# 配置日志和中文字体
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============= 导入模型和工具函数 =============

from train_swarm_gnn import (
    DynamicGraphSwarmGRUModel,
    compute_multi_scale_velocity,
    compute_curvature,
    compute_plane_curvatures,
    SwarmTrajectoryDatasetGNN,
)


# ============= 推理函数 =============

def compute_swarm_metrics(pred_abs, target_abs):
    """
    计算多维度集群指标：
    - MAE: 平均绝对误差
    - RMSE: 均方根误差
    - MAPE: 平均绝对百分比误差
    - MaxError: 最大偏差
    """
    error = np.abs(pred_abs - target_abs)
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))
    
    # MAPE（避免除零）
    target_norm = np.abs(target_abs)
    target_norm[target_norm < 0.1] = 0.1
    mape = np.mean(error / target_norm)
    
    # 最大误差
    max_error = np.max(error)
    
    return mae, rmse, mape, max_error


def load_test_data(data_dir, num_agents, num_samples=-1, random_sample=False):
    """
    加载测试数据
    
    Args:
        data_dir: 数据目录
        num_agents: 无人机数量
        num_samples: 要加载的样本数（-1=全部）
        random_sample: 是否随机采样
    
    Returns:
        X, Y: (num_samples, seq, agents, 3) 格式的数据
    """
    data_path = Path(data_dir)
    X_file = data_path / f'input_agents_{num_agents}.npz'
    Y_file = data_path / f'output_agents_{num_agents}.npz'
    
    if not X_file.exists() or not Y_file.exists():
        raise FileNotFoundError(f"找不到数据文件: {X_file}, {Y_file}")
    
    logger.info(f"加载 {num_agents} 架无人机数据...")
    X = np.load(X_file)['data']  # 形状: (seq_len, samples, agents, 3)
    Y = np.load(Y_file)['data']  # 形状: (seq_out, samples, agents, 3)
    
    # 转置为 (samples, seq, agents, 3) 格式
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    logger.info(f"  输入形状: {X.shape}, 输出形状: {Y.shape}")
    
    # 采样
    total_samples = len(X)
    if num_samples > 0 and num_samples < total_samples:
        if random_sample:
            indices = np.random.choice(total_samples, num_samples, replace=False)
        else:
            indices = np.arange(num_samples)
        X = X[indices]
        Y = Y[indices]
        logger.info(f"  采样 {num_samples} 个样本")
    
    return X, Y


def reconstruct_positions_simple(pred_delta, x_orig, dt=0.1, verbose=False):
    """
    简单位置重建（改进版）：直接积分位移增量 + 速度平滑
    
    Args:
        pred_delta: (batch, seq_out, agents, 3) 预测的位移增量
        x_orig: (batch, seq_in, agents, 3) 输入序列
        dt: 时间步长
        verbose: 是否打印诊断信息
    
    Returns:
        pred_positions: (batch, seq_out, agents, 3) 重建的绝对位置
    """
    batch_size, seq_out, num_agents, _ = pred_delta.shape
    seq_in = x_orig.shape[1]
    
    last_pos = x_orig[:, -1, :, :]  # (batch, agents, 3)
    
    # 计算历史速度和最大速度
    input_vel = np.diff(x_orig, axis=1) / dt  # (batch, seq_in-1, agents, 3)
    max_vel = np.max(np.linalg.norm(input_vel, axis=-1), axis=(1, 2), keepdims=True) * 1.5  # (batch, 1, 1)
    max_vel = np.maximum(max_vel, 0.1)  # 防止为0
    
    # 速度约束：检查每个增量是否过大
    pred_delta_safe = pred_delta.copy()
    for b in range(batch_size):
        for a in range(num_agents):
            for t in range(seq_out):
                step_vel_norm = np.linalg.norm(pred_delta[b, t, a, :]) / dt
                if step_vel_norm > max_vel[b, 0, 0]:
                    pred_delta_safe[b, t, a, :] *= (max_vel[b, 0, 0] / (step_vel_norm + 1e-8)) * dt
    
    # 直接积分
    pred_positions = last_pos[:, np.newaxis, :, :] + np.cumsum(pred_delta_safe, axis=1)
    
    return pred_positions


def reconstruct_positions_physics_constrained(pred_delta, x_orig, dt=0.1, smoothing_weight=0.3):
    """
    物理约束位置重建（改进版）：加入加速度平滑约束 + 速度约束
    
    Args:
        pred_delta: (batch, seq_out, agents, 3) 预测的位移增量
        x_orig: (batch, seq_in, agents, 3) 输入序列
        dt: 时间步长
        smoothing_weight: 平滑权重，范围 [0, 1]
    
    Returns:
        pred_positions: (batch, seq_out, agents, 3) 重建的绝对位置
    """
    batch_size, seq_out, num_agents, _ = pred_delta.shape
    seq_in = x_orig.shape[1]
    
    # 计算历史加速度和速度
    input_vel = np.diff(x_orig, axis=1) / dt  # (batch, seq_in-1, agents, 3)
    input_acc = np.diff(input_vel, axis=1) / dt if seq_in > 2 else np.zeros((batch_size, 1, num_agents, 3))
    
    avg_acc = np.mean(input_acc, axis=1, keepdims=True)  # (batch, 1, agents, 3)
    max_vel = np.max(np.linalg.norm(input_vel, axis=-1), axis=1, keepdims=True) * 2.0  # (batch, 1, agents)
    max_vel = np.maximum(max_vel, 0.1)
    max_acc_norm = np.max(np.linalg.norm(input_acc, axis=-1)) * 1.5 if seq_in > 2 else 5.0
    max_acc_norm = max(max_acc_norm, 5.0)
    
    last_pos = x_orig[:, -1, :, :]  # (batch, agents, 3)
    last_vel = input_vel[:, -1, :, :] if seq_in > 1 else np.zeros((batch_size, num_agents, 3))
    
    # 预测原始位置
    pred_positions_raw = last_pos[:, np.newaxis, :, :] + np.cumsum(pred_delta, axis=1)
    
    # 计算期望速度
    desired_vel = np.diff(np.vstack([last_pos[0:1], pred_positions_raw[0]]), axis=0) / dt
    for b in range(batch_size):
        desired_vel_b = np.diff(np.vstack([last_pos[b:b+1], pred_positions_raw[b]]), axis=0) / dt
        if b == 0:
            desired_vel_all = desired_vel_b[np.newaxis, :, :, :]
        else:
            desired_vel_all = np.vstack([desired_vel_all, desired_vel_b[np.newaxis, :, :, :]])
    
    # 逐步构建预测
    pred_positions = np.zeros_like(pred_positions_raw)
    current_pos = last_pos.copy()
    current_vel = last_vel.copy()
    
    for t in range(seq_out):
        # 计算原始加速度
        raw_accel = (desired_vel_all[:, t, :, :] - current_vel) / dt
        
        # 约束1：加速度平滑
        constrained_accel = (1 - smoothing_weight) * raw_accel + smoothing_weight * avg_acc[:, 0, :, :]
        
        # 约束2：最大加速度限制
        accel_norms = np.linalg.norm(constrained_accel, axis=-1, keepdims=True)  # (batch, agents, 1)
        accel_scale = np.minimum(1.0, max_acc_norm / (accel_norms + 1e-8))
        constrained_accel = constrained_accel * accel_scale
        
        # 约束3：速度更新
        new_vel = current_vel + constrained_accel * dt
        
        # 限制速度
        vel_norms = np.linalg.norm(new_vel, axis=-1, keepdims=True)  # (batch, agents, 1)
        vel_scale = np.minimum(1.0, max_vel / (vel_norms + 1e-8))
        new_vel = new_vel * vel_scale
        
        current_vel = new_vel
        
        # 更新位置
        next_pos = current_pos + current_vel * dt
        pred_positions[:, t, :, :] = next_pos
        current_pos = next_pos
    
    return pred_positions


def reconstruct_positions_trajectory_smoothing(pred_positions, window_size=3):
    """
    轨迹平滑：对预测结果进行滑动平均
    
    Args:
        pred_positions: (batch, seq_out, agents, 3) 预测位置
        window_size: 平滑窗口大小
    
    Returns:
        smoothed: (batch, seq_out, agents, 3) 平滑后的位置
    """
    if window_size <= 1:
        return pred_positions
    
    batch_size, seq_out, num_agents, coords = pred_positions.shape
    smoothed = np.zeros_like(pred_positions)
    
    for t in range(seq_out):
        start = max(0, t - window_size // 2)
        end = min(seq_out, t + window_size // 2 + 1)
        smoothed[:, t, :, :] = np.mean(pred_positions[:, start:end, :, :], axis=1)
    
    return smoothed


def infer_batch(model, features, x_orig, device, output_std, output_mean=None, 
                reconstruction_method='physics_constrained', dt=0.1, smoothing_weight=0.3):
    """
    推理一个批次，支持多种位置重建方法
    
    Args:
        model: 模型
        features: (batch, seq_in, agents, 16) 特征
        x_orig: (batch, seq_in, agents, 3) 原始位置
        device: 设备
        output_std: 反归一化因子
        output_mean: 反归一化均值（通常为0）
        reconstruction_method: 位置重建方法
            - 'direct': 直接反归一化，y = x[-1] + delta
            - 'simple': 简单积分，加入速度约束
            - 'physics_constrained': 物理约束（加速度、速度限制）
        dt: 时间步长
        smoothing_weight: 物理约束中的平滑权重
    
    Returns:
        pred_abs: (batch, seq_out, agents, 3) 绝对位置预测
    """
    model.eval()
    with torch.no_grad():
        features = torch.tensor(features, device=device, dtype=torch.float32)
        x_orig = torch.tensor(x_orig, device=device, dtype=torch.float32)
        
        # 推理得到归一化增量
        pred_norm = model(features, x_orig, teacher_forcing_ratio=0.0)
        
        # 反归一化：y_norm = (y_delta - output_mean) / output_std
        # 反解：y_delta = y_norm * output_std + output_mean
        if output_mean is None:
            output_mean = 0.0
        
        output_std_tensor = torch.tensor(output_std, dtype=torch.float32, device=device)
        output_mean_tensor = torch.tensor(output_mean, dtype=torch.float32, device=device)
        
        # 反归一化得到增量
        pred_delta = pred_norm * output_std_tensor + output_mean_tensor
        pred_delta_np = pred_delta.cpu().numpy()  # (batch, seq_out, agents, 3)
        
        # 位置重建
        if reconstruction_method == 'direct':
            # 直接方法：y = x[-1] + delta
            last_pos = x_orig[:, -1:, :, :]  # (batch, 1, agents, 3)
            pred_abs = (last_pos + pred_delta).cpu().numpy()
            
        elif reconstruction_method == 'simple':
            # 简单积分 + 速度约束
            pred_abs = reconstruct_positions_simple(pred_delta_np, x_orig.cpu().numpy(), dt=dt)
            
        elif reconstruction_method == 'physics_constrained':
            # 物理约束 + 加速度平滑
            pred_abs = reconstruct_positions_physics_constrained(
                pred_delta_np, x_orig.cpu().numpy(), dt=dt, smoothing_weight=smoothing_weight
            )
            # 再进行轨迹平滑
            pred_abs = reconstruct_positions_trajectory_smoothing(pred_abs, window_size=3)
            
        else:
            raise ValueError(f"Unknown reconstruction method: {reconstruction_method}")
        
        return pred_abs


def visualize_sample_predictions(X, Y_true, Y_pred, sample_idx=0, save_path=None):
    """
    可视化单个样本的预测结果
    
    Args:
        X: (seq_in, agents, 3) 输入轨迹
        Y_true: (seq_out, agents, 3) 真实输出
        Y_pred: (seq_out, agents, 3) 预测输出
        sample_idx: 样本索引
        save_path: 保存路径（None=不保存）
    """
    num_agents = X.shape[1]
    
    fig = plt.figure(figsize=(16, 5))
    
    for agent_id in range(num_agents):
        # 3D 轨迹对比
        ax = fig.add_subplot(1, num_agents, agent_id + 1, projection='3d')
        
        X_agent = X[:, agent_id, :]
        Y_true_agent = Y_true[:, agent_id, :]
        Y_pred_agent = Y_pred[:, agent_id, :]
        
        # 输入轨迹（虚线）
        ax.plot(X_agent[:, 0], X_agent[:, 1], X_agent[:, 2], 
               'g--', label='输入', linewidth=2, alpha=0.7)
        
        # 真实轨迹（红色实线）
        ax.plot(Y_true_agent[:, 0], Y_true_agent[:, 1], Y_true_agent[:, 2], 
               'r-', label='真实', linewidth=2, marker='x', markersize=4)
        
        # 预测轨迹（蓝色实线）
        ax.plot(Y_pred_agent[:, 0], Y_pred_agent[:, 1], Y_pred_agent[:, 2], 
               'b-', label='预测', linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'无人机 {agent_id} (样本 {sample_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ 可视化已保存: {save_path}")
    
    plt.close()


def generate_inference_report(X_all, Y_all, Y_pred, config, output_dir):
    """
    生成推理报告
    
    Args:
        X_all: 所有输入
        Y_all: 所有真实输出
        Y_pred: 所有预测输出
        config: 模型配置
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    num_agents = X_all.shape[2]
    num_samples = len(X_all)
    
    # 计算全局指标
    mae_global, rmse_global, mape_global, max_err_global = compute_swarm_metrics(Y_pred, Y_all)
    
    # 按无人机计算指标
    mae_per_agent = []
    rmse_per_agent = []
    for agent_id in range(num_agents):
        mae, rmse, _, _ = compute_swarm_metrics(
            Y_pred[:, :, agent_id, :], 
            Y_all[:, :, agent_id, :]
        )
        mae_per_agent.append(mae)
        rmse_per_agent.append(rmse)
    
    # 按轴计算指标
    mae_x = np.mean(np.abs(Y_pred[..., 0] - Y_all[..., 0]))
    mae_y = np.mean(np.abs(Y_pred[..., 1] - Y_all[..., 1]))
    mae_z = np.mean(np.abs(Y_pred[..., 2] - Y_all[..., 2]))
    
    # 按时步计算指标
    mae_per_step = []
    for t in range(Y_pred.shape[1]):
        mae = np.mean(np.abs(Y_pred[:, t, :, :] - Y_all[:, t, :, :]))
        mae_per_step.append(mae)
    
    # 生成报告
    report = f"""
{'='*80}
GNN 集群轨迹预测模型 - 推理评估报告
{'='*80}

【基本信息】
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
无人机数量: {num_agents}
评估样本数: {num_samples}
模型配置:
  hidden_size: {config.get('hidden_size', 'N/A')}
  num_layers: {config.get('num_layers', 'N/A')}
  dropout: {config.get('dropout', 'N/A')}
  use_gcn: {'是' if config.get('use_gcn') else '否'}

【全局指标】
{'='*80}
平均绝对误差 (MAE):  {mae_global:.6f} m
均方根误差 (RMSE):   {rmse_global:.6f} m
平均百分比误差 (MAPE): {mape_global:.6f}
最大误差:           {max_err_global:.6f} m

【按无人机分解】
{'='*80}
"""
    for agent_id in range(num_agents):
        report += f"无人机 {agent_id}:\n"
        report += f"  MAE:  {mae_per_agent[agent_id]:.6f} m\n"
        report += f"  RMSE: {rmse_per_agent[agent_id]:.6f} m\n"
    
    report += f"\n【按轴分解】\n{'='*80}\n"
    report += f"X 轴 MAE: {mae_x:.6f} m\n"
    report += f"Y 轴 MAE: {mae_y:.6f} m\n"
    report += f"Z 轴 MAE: {mae_z:.6f} m\n"
    
    report += f"\n【按时步分解】\n{'='*80}\n"
    for t, mae in enumerate(mae_per_step, 1):
        report += f"时步 {t:2d}: {mae:.6f} m\n"
    
    report += f"\n{'='*80}\n"
    
    # 保存报告
    report_path = output_dir / f'inference_report_agents_{num_agents}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"✓ 报告已保存: {report_path}")
    logger.info(report)
    
    # 保存指标为 JSON
    metrics = {
        'global': {
            'mae': float(mae_global),
            'rmse': float(rmse_global),
            'mape': float(mape_global),
            'max_error': float(max_err_global),
        },
        'per_agent': {
            f'agent_{i}': {
                'mae': float(mae_per_agent[i]),
                'rmse': float(rmse_per_agent[i]),
            }
            for i in range(num_agents)
        },
        'per_axis': {
            'X': float(mae_x),
            'Y': float(mae_y),
            'Z': float(mae_z),
        },
        'per_step': [float(m) for m in mae_per_step],
    }
    
    metrics_path = output_dir / f'inference_metrics_agents_{num_agents}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✓ 指标已保存: {metrics_path}")


# ============= 主函数 =============

def main():
    parser = argparse.ArgumentParser(description='GNN 模型推理脚本 (train_swarm_gnn.py)')
    parser.add_argument('--model', type=str, required=True, 
                       help='最佳模型文件路径')
    parser.add_argument('--data_dir', type=str, default='swarm_segments',
                       help='数据目录')
    parser.add_argument('--agents', type=int, default=3,
                       help='无人机数量')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='推理批大小')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='推理样本数 (-1=全部)')
    parser.add_argument('--use_gcn', type=int, default=0,
                       help='是否使用 GCN (0/1)')
    parser.add_argument('--random_sample', action='store_true',
                       help='是否随机采样')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化')
    parser.add_argument('--num_vis', type=int, default=5,
                       help='可视化样本数')
    parser.add_argument('--output_dir', type=str, default='infer_results_gnn',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--reconstruction_method', type=str, default='physics_constrained',
                       choices=['direct', 'simple', 'physics_constrained'],
                       help='位置重建方法')
    parser.add_argument('--smoothing_weight', type=float, default=0.3,
                       help='物理约束中的平滑权重 [0, 1]')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='时间步长')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"位置重建方法: {args.reconstruction_method}")
    
    # ============= 1. 加载模型 =============
    logger.info("\n" + "="*80)
    logger.info("阶段 1/4: 加载模型")
    logger.info("="*80)
    
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    logger.info(f"加载 checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    stats = checkpoint.get('stats', {})
    
    # 创建模型
    model = DynamicGraphSwarmGRUModel(
        input_size=16,
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        num_agents=args.agents,
        output_size=3,
        dropout=config.get('dropout', 0.2),
        use_gcn=bool(args.use_gcn)  # ✅ 支持无 GCN 推理
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✓ 模型加载成功")
    logger.info(f"  使用 GCN: {'是' if args.use_gcn else '否'}")
    logger.info(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 获取统计量
    output_mean = np.array(stats.get('output_mean', 0.0))
    output_std = np.array(stats.get('output_std', 1.0))
    
    # ============= 2. 加载数据 =============
    logger.info("\n" + "="*80)
    logger.info("阶段 2/4: 加载数据")
    logger.info("="*80)
    
    X_all, Y_all = load_test_data(
        args.data_dir, args.agents, 
        num_samples=args.num_samples,
        random_sample=args.random_sample
    )
    
    logger.info(f"✓ 数据加载成功: {len(X_all)} 样本")
    
    # ============= 3. 推理 =============
    logger.info("\n" + "="*80)
    logger.info("阶段 3/4: 推理")
    logger.info("="*80)
    
    # 计算特征（同训练过程）
    dataset = SwarmTrajectoryDatasetGNN(
        X_all, Y_all,
        input_mean=stats.get('input_mean'),
        input_std=stats.get('input_std'),
        output_mean=stats.get('output_mean'),
        output_std=stats.get('output_std'),
        feature_mean=stats.get('input_mean_all'),
        feature_std=stats.get('input_std_all'),
    )
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=0
    )
    
    predictions = []
    for features, x_orig, y_norm, y_orig in tqdm(dataloader, desc="推理进度"):
        pred = infer_batch(
            model, features.numpy(), x_orig.numpy(), 
            device, 
            torch.tensor(output_std, device=device, dtype=torch.float32),
            torch.tensor(output_mean, device=device, dtype=torch.float32) if isinstance(output_mean, np.ndarray) else output_mean,
            reconstruction_method=args.reconstruction_method,
            dt=args.dt,
            smoothing_weight=args.smoothing_weight
        )
        predictions.append(pred)
    
    Y_pred = np.concatenate(predictions, axis=0)
    
    logger.info(f"✓ 推理完成")
    logger.info(f"  预测形状: {Y_pred.shape}")
    
    # ============= 4. 评估与可视化 =============
    logger.info("\n" + "="*80)
    logger.info("阶段 4/4: 评估与可视化")
    logger.info("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成报告
    generate_inference_report(X_all, Y_all, Y_pred, config, output_dir)
    
    # 保存预测结果
    result_file = output_dir / f'predictions_agents_{args.agents}.npz'
    np.savez(
        result_file,
        input=X_all,
        target=Y_all,
        prediction=Y_pred,
    )
    logger.info(f"✓ 预测结果已保存: {result_file}")
    
    # 生成可视化
    if args.visualize:
        logger.info(f"生成 {args.num_vis} 个可视化样本...")
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        sample_indices = np.random.choice(len(X_all), min(args.num_vis, len(X_all)), replace=False)
        
        for idx in sample_indices:
            vis_path = vis_dir / f'sample_{idx:04d}.png'
            visualize_sample_predictions(
                X_all[idx], Y_all[idx], Y_pred[idx],
                sample_idx=idx,
                save_path=vis_path
            )
        
        logger.info(f"✓ 可视化完成: {len(sample_indices)} 张图片")
    
    logger.info("\n" + "="*80)
    logger.info("✓ 推理完成！")
    logger.info("="*80)


if __name__ == '__main__':
    main()
