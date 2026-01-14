#!/usr/bin/env python3
"""
集群轨迹预测可视化脚本 v3（支持 v2 和 v3 GNN 模型）

📊 功能：
    ✅ 评估和可视化集群无人机轨迹预测模型 (v2 和 v3 GNN)
    ✅ 自动检测模型版本（v2 或 v3）
    ✅ 输出静态 PNG 图表（6 视图对比）
    ✅ 输出交互式 3D 图表（Matplotlib，鼠标拖拽旋转）
    ✅ 生成详细 JSON 评估报告（MAE/RMSE 等）

🔄 统计量管理（无需手动管理）：
    ✅ 24D 特征统计：从 norm_stats_agents_N_v3.npz/v2.npz 自动加载
    ✅ 输出统计：从 checkpoint 自动加载
    ✅ 完全复现训练时的分布一致性

📁 输入文件：
    - 模型：gru_models_v3_agents_3/.../best_model.pt （v3）或 gru_models_enhanced/.../best_model.pt（v2）
    - 统计：norm_stats_agents_3_v3.npz 或 norm_stats_agents_3_v2.npz（自动加载）
    - 数据：swarm_segments/input_agents_3.npz, output_agents_3.npz

📤 输出文件：
    - PNG：swarm_prediction_sample_XXXXXX.png（6 视图对比）
    - PNG：swarm_prediction_interactive_XXXXXX.png（3D 交互式图快照）
    - JSON：evaluation_report_YYYYMMDD_HHMMSS.json（详细指标）

💡 使用示例：
    # v3 with GNN
    python visualize_swarm_prediction_v3_gnn.py ^
        --model_path gru_models_v3_agents_3/best_model_agents_3_v3.pt ^
        --agents 3 ^
        --num_samples 5

    # v2 (auto-detect)
    python visualize_swarm_prediction_v3_gnn.py ^
        --model_path gru_models_enhanced/best_model_agents_3.pt ^
        --agents 3 ^
        --num_samples 5
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import sys

# 导入模型和推理函数
sys.path.insert(0, str(Path(__file__).parent))
from infer_swarm_model_v3_gnn import (
    DynamicsAwareSwarmGRUModel_with_GNN,
    DynamicsAwareSwarmGRUModel,
    infer_batch_v2,
    infer_batch_v3,
    compute_features_for_inference,
    estimate_feature_stats_from_data,
    load_data_robust,
    detect_model_version,
    _extract_array_from_npz_field,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def plot_swarm_prediction(history, true_future, pred_future, sample_idx, num_agents, 
                         output_dir, model_version='v2', interactive=False, fast_mode=False):
    """
    绘制集群轨迹对比图
    
    Args:
        history: 输入历史轨迹 (seq_in, num_agents, 3)
        true_future: 真实未来轨迹 (seq_out, num_agents, 3)
        pred_future: 预测未来轨迹 (seq_out, num_agents, 3)
        sample_idx: 样本索引
        num_agents: 无人机数量
        output_dir: 输出目录
        model_version: 模型版本（'v2' 或 'v3'）
        interactive: 是否显示交互窗口
        fast_mode: 快速模式（跳过交互式 3D 图）
    """
    
    # 创建大图
    fig = plt.figure(figsize=(20, 13))
    
    # 1. 3D 轨迹对比
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    
    # 绘制所有无人机的历史轨迹（蓝色）
    for agent_id in range(num_agents):
        if agent_id == 0:
            ax3d.plot(history[:, agent_id, 0], history[:, agent_id, 1], history[:, agent_id, 2],
                     'b-o', linewidth=2.5, markersize=5, label='History', alpha=0.8)
        else:
            ax3d.plot(history[:, agent_id, 0], history[:, agent_id, 1], history[:, agent_id, 2],
                     'bo', linewidth=2.5, markersize=5, alpha=0.8)
    
    # 绘制真实和预测的未来轨迹
    for agent_id in range(num_agents):
        # 连接历史最后一点到预测起点
        last_point = history[-1:, agent_id, :]  # (1, 3)
        
        # 真实轨迹（绿色）
        true_traj = np.vstack([last_point, true_future[:, agent_id, :]])
        ax3d.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
                 'gs-', linewidth=2.8, markersize=7, alpha=0.9,
                 label='True Future' if agent_id == 0 else '')
        
        # 预测轨迹（红色虚线）
        pred_traj = np.vstack([last_point, pred_future[:, agent_id, :]])
        ax3d.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
                 'r^--', linewidth=2.5, markersize=6, alpha=0.75,
                 label='Predicted Future' if agent_id == 0 else '')
    
    ax3d.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax3d.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax3d.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax3d.set_title(f'Sample {sample_idx}: 3D Swarm Trajectories ({model_version.upper()})\nAll {num_agents} Agents', 
                  fontsize=12, fontweight='bold')
    ax3d.legend(fontsize=10, loc='upper left')
    ax3d.grid(True, alpha=0.3)
    
    # 2. XY 平面
    ax_xy = fig.add_subplot(2, 3, 2)
    for agent_id in range(num_agents):
        last_point = history[-1:, agent_id, :]
        true_traj = np.vstack([last_point, true_future[:, agent_id, :]])
        pred_traj = np.vstack([last_point, pred_future[:, agent_id, :]])
        
        if agent_id == 0:
            ax_xy.plot(history[:, agent_id, 0], history[:, agent_id, 1],
                      'b-o', linewidth=2.5, markersize=5, label='History', alpha=0.8)
        
        ax_xy.plot(true_traj[:, 0], true_traj[:, 1],
                  'gs-', linewidth=2.8, markersize=7, label='True' if agent_id == 0 else '', alpha=0.9)
        ax_xy.plot(pred_traj[:, 0], pred_traj[:, 1],
                  'r^--', linewidth=2.5, markersize=6, label='Predicted' if agent_id == 0 else '', alpha=0.75)
    
    ax_xy.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax_xy.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax_xy.set_title('XY Plane Projection', fontsize=12, fontweight='bold')
    ax_xy.legend(fontsize=10, loc='best')
    ax_xy.grid(True, alpha=0.3)
    
    # 3. XZ 平面
    ax_xz = fig.add_subplot(2, 3, 3)
    for agent_id in range(num_agents):
        last_point = history[-1:, agent_id, :]
        true_traj = np.vstack([last_point, true_future[:, agent_id, :]])
        pred_traj = np.vstack([last_point, pred_future[:, agent_id, :]])
        
        if agent_id == 0:
            ax_xz.plot(history[:, agent_id, 0], history[:, agent_id, 2],
                      'b-o', linewidth=2.5, markersize=5, label='History', alpha=0.8)
        
        ax_xz.plot(true_traj[:, 0], true_traj[:, 2],
                  'gs-', linewidth=2.8, markersize=7, label='True' if agent_id == 0 else '', alpha=0.9)
        ax_xz.plot(pred_traj[:, 0], pred_traj[:, 2],
                  'r^--', linewidth=2.5, markersize=6, label='Predicted' if agent_id == 0 else '', alpha=0.75)
    
    ax_xz.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax_xz.set_ylabel('Z (m)', fontsize=11, fontweight='bold')
    ax_xz.set_title('XZ Plane Projection', fontsize=12, fontweight='bold')
    ax_xz.legend(fontsize=10, loc='best')
    ax_xz.grid(True, alpha=0.3)
    
    # 4. YZ 平面
    ax_yz = fig.add_subplot(2, 3, 4)
    for agent_id in range(num_agents):
        last_point = history[-1:, agent_id, :]
        true_traj = np.vstack([last_point, true_future[:, agent_id, :]])
        pred_traj = np.vstack([last_point, pred_future[:, agent_id, :]])
        
        if agent_id == 0:
            ax_yz.plot(history[:, agent_id, 1], history[:, agent_id, 2],
                      'b-o', linewidth=2.5, markersize=5, label='History', alpha=0.8)
        
        ax_yz.plot(true_traj[:, 1], true_traj[:, 2],
                  'gs-', linewidth=2.8, markersize=7, label='True' if agent_id == 0 else '', alpha=0.9)
        ax_yz.plot(pred_traj[:, 1], pred_traj[:, 2],
                  'r^--', linewidth=2.5, markersize=6, label='Predicted' if agent_id == 0 else '', alpha=0.75)
    
    ax_yz.set_xlabel('Y (m)', fontsize=11, fontweight='bold')
    ax_yz.set_ylabel('Z (m)', fontsize=11, fontweight='bold')
    ax_yz.set_title('YZ Plane Projection', fontsize=12, fontweight='bold')
    ax_yz.legend(fontsize=10, loc='best')
    ax_yz.grid(True, alpha=0.3)
    
    # 5. 轴向误差
    ax_error_ts = fig.add_subplot(2, 3, 5)
    
    steps = np.arange(true_future.shape[0])
    
    error_x = np.abs(pred_future[:, :, 0] - true_future[:, :, 0]).mean(axis=1)
    error_y = np.abs(pred_future[:, :, 1] - true_future[:, :, 1]).mean(axis=1)
    error_z = np.abs(pred_future[:, :, 2] - true_future[:, :, 2]).mean(axis=1)
    
    ax_error_ts.plot(steps, error_x, 'rs-', linewidth=2.5, markersize=7, label='X Axis Error', alpha=0.8)
    ax_error_ts.plot(steps, error_y, 'bo-', linewidth=2.5, markersize=7, label='Y Axis Error', alpha=0.8)
    ax_error_ts.plot(steps, error_z, 'g^-', linewidth=2.5, markersize=7, label='Z Axis Error', alpha=0.8)
    
    ax_error_ts.set_xlabel('Prediction Step', fontsize=11, fontweight='bold')
    ax_error_ts.set_ylabel('Mean Absolute Error (m)', fontsize=11, fontweight='bold')
    ax_error_ts.set_title('Per-Step Axis-wise Error', fontsize=12, fontweight='bold')
    ax_error_ts.legend(fontsize=10, loc='best')
    ax_error_ts.grid(True, alpha=0.3)
    
    # 6. 总体误差分布
    ax_error_dist = fig.add_subplot(2, 3, 6)
    
    errors_per_step = np.linalg.norm(pred_future - true_future, axis=2).mean(axis=1)
    
    bars = ax_error_dist.bar(steps, errors_per_step, color='tab:red', alpha=0.7, edgecolor='darkred', linewidth=1.5)
    
    for i, (step, err) in enumerate(zip(steps, errors_per_step)):
        ax_error_dist.text(step, err + 0.01, f'{err:.3f}', ha='center', va='bottom', fontsize=9)
    
    mean_error = errors_per_step.mean()
    ax_error_dist.axhline(y=mean_error, color='darkred', linestyle='--', linewidth=2, 
                          label=f'Mean Error: {mean_error:.4f}m')
    
    ax_error_dist.set_xlabel('Prediction Step', fontsize=11, fontweight='bold')
    ax_error_dist.set_ylabel('Position Error (m)', fontsize=11, fontweight='bold')
    ax_error_dist.set_title('Position Error per Step (All Agents Avg)', fontsize=12, fontweight='bold')
    ax_error_dist.legend(fontsize=10, loc='best')
    ax_error_dist.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()

    # 保存 PNG 图表
    logger.info(f"    保存 PNG 图表...")
    png_path = Path(output_dir) / f'swarm_prediction_sample_{sample_idx:06d}_{model_version}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    logger.info(f"    ✓ PNG 保存完成: {png_path.name}")

    if interactive:
        plt.show()

    plt.close(fig)
    
    # 生成交互式 3D 图表（Matplotlib，可在窗口中直接旋转）
    if not fast_mode:
        interactive_path = save_rotatable_plot(history, true_future, pred_future, sample_idx, num_agents, output_dir, model_version)
        if interactive_path is not None:
            logger.info(f"    ✓ 交互式 3D 图已保存: {interactive_path.name}")

    return png_path


def save_rotatable_plot(history, true_future, pred_future, sample_idx, num_agents, output_dir, model_version='v2'):
    """生成交互式 3D 图表（使用 Matplotlib，可在窗口中直接旋转）"""
    
    logger.info(f"    生成交互式 3D 轨迹图...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    for agent_id in range(num_agents):
        color = colors[agent_id]
        last_point = history[-1, agent_id, :]
        
        # 历史轨迹（蓝色虚线 + 点）
        ax.plot(history[:, agent_id, 0], history[:, agent_id, 1], history[:, agent_id, 2],
                'o-', color=color, linewidth=2, markersize=4, alpha=0.7,
                label=f'Agent {agent_id} History' if agent_id == 0 else '')
        
        # 真实未来轨迹（绿色方形标记）
        true_traj = np.vstack([last_point, true_future[:, agent_id, :]])
        ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
                's--', color='green', linewidth=2.5, markersize=6, alpha=0.8,
                label=f'True Future' if agent_id == 0 else '')
        
        # 预测未来轨迹（红色三角形标记）
        pred_traj = np.vstack([last_point, pred_future[:, agent_id, :]])
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
                '^--', color='red', linewidth=2.5, markersize=6, alpha=0.75,
                label=f'Predicted Future' if agent_id == 0 else '')
    
    # 轴标签和标题
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax.set_title(f'Sample {sample_idx:06d} - 3D Rotatable Trajectories ({model_version.upper()})\n'
                 f'(Drag mouse to rotate, Scroll to zoom)',
                 fontsize=12, fontweight='bold', pad=20)
    
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存为 PNG（用于记录）
    png_path = Path(output_dir) / f'swarm_prediction_interactive_{sample_idx:06d}_{model_version}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    logger.info(f"    ✓ 交互式 3D 图已显示（可用鼠标拖拽旋转）")
    
    plt.show()  # 显示交互窗口
    plt.close(fig)
    
    return png_path


def compute_metrics(true_future, pred_future):
    """计算评估指标"""
    errors = np.linalg.norm(pred_future - true_future, axis=2)  # (seq_out, num_agents)
    
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # 轴向误差
    mae_x = float(np.mean(np.abs(pred_future[..., 0] - true_future[..., 0])))
    mae_y = float(np.mean(np.abs(pred_future[..., 1] - true_future[..., 1])))
    mae_z = float(np.mean(np.abs(pred_future[..., 2] - true_future[..., 2])))
    
    # 按时步的误差
    mae_per_step = np.mean(errors, axis=1).tolist()
    
    # 按代理的误差
    mae_per_agent = np.mean(errors, axis=0).tolist()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_z': mae_z,
        'mae_per_step': mae_per_step,
        'mae_per_agent': mae_per_agent,
    }


def main():
    parser = argparse.ArgumentParser(description='集群轨迹预测可视化 (v2/v3)')
    
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径 (.pt)')
    parser.add_argument('--data_dir', type=str, default='swarm_segments', help='数据目录')
    parser.add_argument('--agents', type=int, default=3, help='Agent 数量')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_samples', type=int, default=5, help='可视化样本数')
    parser.add_argument('--output_dir', type=str, default='visualization_output_v3', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--force_v2', action='store_true', help='强制使用 v2 模型')
    parser.add_argument('--force_v3', action='store_true', help='强制使用 v3 模型（GNN）')
    parser.add_argument('--fast_mode', action='store_true', help='快速模式（跳过交互式 3D 图）')
    parser.add_argument('--interactive', action='store_true', help='显示交互窗口')
    parser.add_argument('--use_subset', action='store_true', help='使用 _subset 数据')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)
    
    config = checkpoint.get("config", {})
    
    # 检测模型版本
    if args.force_v3:
        model_version = 'v3'
    elif args.force_v2:
        model_version = 'v2'
    else:
        model_version = detect_model_version(checkpoint)
    
    logger.info(f"检测到模型版本: {model_version}")
    
    # 创建模型
    if model_version == 'v3':
        logger.info("加载 v3 (GNN) 模型...")
        model = DynamicsAwareSwarmGRUModel_with_GNN(
            input_size=24,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            output_size=3,
            dropout=0.0,
            use_attention=config.get("use_attention", True),
            gnn_hidden=config.get("gnn_hidden", 64),
            num_gnn_heads=config.get("gnn_heads", 4),
            edge_threshold=config.get("edge_threshold", 5.5),
            fusion_mode=config.get("gnn_fusion_mode", "concat"),
        ).to(device)
        infer_fn = lambda **kwargs: infer_batch_v3(**kwargs)
    else:
        logger.info("加载 v2 模型...")
        model = DynamicsAwareSwarmGRUModel(
            input_size=24,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            output_size=3,
            dropout=0.0,
            use_attention=config.get("use_attention", True),
        ).to(device)
        infer_fn = lambda **kwargs: infer_batch_v2(**kwargs)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("✓ 模型加载完成")
    
    # 加载统计量
    output_mean = np.array(checkpoint["output_mean"], dtype=np.float32)
    output_std = np.array(checkpoint["output_std"], dtype=np.float32)
    logger.info(f"✓ 输出统计量: mean={output_mean}, std={output_std}")
    
    # 加载数据
    logger.info(f"加载数据: {args.data_dir} (use_subset={args.use_subset})")
    X_all, Y_all = load_data_robust(args.data_dir, args.agents, use_subset=args.use_subset)
    logger.info(f"✓ 数据加载完成: {len(X_all)} 个样本")
    
    # 【优化】随机选择样本（像 v2 一样）
    num_vis_samples = min(args.num_samples if args.num_samples > 0 else len(X_all), len(X_all))
    sample_indices = np.random.choice(len(X_all), num_vis_samples, replace=False)
    X_viz = X_all[sample_indices]
    Y_viz = Y_all[sample_indices]
    logger.info(f"随机选择 {num_vis_samples} 个样本用于可视化")
    
    # 加载特征统计量
    feature_mean_all = None
    feature_std_all = None
    
    stats_paths = [
        Path(args.model_path).parent / f'norm_stats_agents_{args.agents}_v3.npz',
        Path(args.model_path).parent / f'norm_stats_agents_{args.agents}_v2.npz',
        Path(args.model_path).parent / f'norm_stats_agents_{args.agents}.npz',
    ]
    
    for stats_path in stats_paths:
        if stats_path.exists():
            logger.info(f"加载特征统计: {stats_path.name}")
            try:
                stats_file = np.load(stats_path, allow_pickle=True)
                raw_mean = stats_file.get('feature_mean', stats_file.get('input_mean_all', None))
                raw_std = stats_file.get('feature_std', stats_file.get('input_std_all', None))
                
                feature_mean_all = _extract_array_from_npz_field(raw_mean, expected_len=24)
                feature_std_all = _extract_array_from_npz_field(raw_std, expected_len=24)
                
                if feature_mean_all is not None and feature_std_all is not None:
                    logger.info("✓ 特征统计加载成功")
                    break
            except Exception as e:
                logger.debug(f"未能加载 {stats_path.name}: {e}")
    
    # 备选加载方式
    if feature_mean_all is None or feature_std_all is None:
        if 'feature_mean' in checkpoint and 'feature_std' in checkpoint:
            feature_mean_all = np.array(checkpoint['feature_mean'])
            feature_std_all = np.array(checkpoint['feature_std'])
            logger.info("✓ 从 checkpoint 加载特征统计")
        else:
            feature_mean_all = np.zeros(24, dtype=np.float32)
            feature_std_all = np.ones(24, dtype=np.float32)
            logger.warning("⚠️ 使用默认零均值单位方差")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 推理
    logger.info(f"推理 {num_vis_samples} 个样本...")
    predictions = []
    
    num_batches = (num_vis_samples + args.batch_size - 1) // args.batch_size
    with tqdm(total=num_batches, desc="推理批次") as pbar:
        for start in range(0, num_vis_samples, args.batch_size):
            end = min(start + args.batch_size, num_vis_samples)
            batch_X = X_viz[start:end]
            
            # 向量化特征计算
            batch_feats = np.stack([
                compute_features_for_inference(x, feature_mean_all, feature_std_all)
                for x in batch_X
            ], axis=0)
            
            if model_version == 'v3':
                batch_pred = infer_fn(
                    model=model,
                    features_batch=batch_feats,
                    x_orig_batch=batch_X,
                    device=device,
                    output_mean=output_mean,
                    output_std=output_std,
                    edge_threshold=config.get("edge_threshold", 5.5),
                    debug=False
                )
            else:
                batch_pred = infer_fn(
                    model=model,
                    features_batch=batch_feats,
                    x_orig_batch=batch_X,
                    device=device,
                    output_mean=output_mean,
                    output_std=output_std,
                    debug=False
                )
            
            predictions.append(batch_pred)
            pbar.update(1)
    
    predictions = np.concatenate(predictions, axis=0)
    logger.info(f"✓ 推理完成: 形状 {predictions.shape}")
    
    # 可视化样本
    all_metrics = []
    
    logger.info(f"可视化 {num_vis_samples} 个样本...")
    for i, sample_idx in enumerate(sample_indices):
        logger.info(f"\n  样本 {i + 1}/{num_vis_samples} (索引 {sample_idx}):")
        
        history = X_viz[i]  # (seq_in, agents, 3)
        true_future = Y_viz[i]  # (seq_out, agents, 3)
        pred_future = predictions[i]  # (seq_out, agents, 3)
        
        # 绘制并计算指标
        plot_swarm_prediction(history, true_future, pred_future, sample_idx, args.agents,
                            output_dir, model_version=model_version, 
                            interactive=args.interactive, fast_mode=args.fast_mode)
        
        metrics = compute_metrics(true_future, pred_future)
        metrics['sample_idx'] = int(sample_idx)
        all_metrics.append(metrics)
        
        logger.info(f"    MAE: {metrics['mae']:.6f} m, RMSE: {metrics['rmse']:.6f} m")
    
    # 生成评估报告
    logger.info("\n生成评估报告...")
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_version': model_version,
        'model_path': str(args.model_path),
        'num_agents': args.agents,
        'num_samples_visualized': num_vis_samples,
        'sample_indices': [int(idx) for idx in sample_indices],
        'total_samples_available': len(X_all),
        'samples': all_metrics,
    }
    
    # 计算全局统计
    if all_metrics:
        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_rmse = np.mean([m['rmse'] for m in all_metrics])
        report['average_mae'] = float(avg_mae)
        report['average_rmse'] = float(avg_rmse)
    
    report_file = output_dir / f'evaluation_report_{model_version}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 评估报告已保存: {report_file.name}")
    logger.info(f"\n=== 可视化完成 ===")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"模型版本: {model_version}")
    logger.info(f"可视化样本数: {num_vis_samples}")


if __name__ == '__main__':
    main()
