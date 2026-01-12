#!/usr/bin/env python3
"""
集群轨迹预测可视化脚本 v2（24D 特征版本）

📊 功能：
    ✅ 评估和可视化集群无人机轨迹预测模型 (v2)
    ✅ 输出静态 PNG 图表（6 视图对比）
    ✅ 输出交互式 3D 图表（Matplotlib，鼠标拖拽旋转）
    ✅ 生成详细 JSON 评估报告（MAE/RMSE 等）

🔄 统计量管理（无需手动管理）：
    ✅ 24D 特征统计：从 norm_stats_agents_N_v2.npz 自动加载（训练时保存）
    ✅ 输出统计：从 checkpoint 自动加载（训练时保存）
    ✅ 完全复现训练时的分布一致性

📁 输入文件：
    - 模型：24dmodel/.../best_model_agents_3_v2.pt
    - 统计：24dmodel/.../norm_stats_agents_3_v2.npz（自动加载）
    - 数据：swarm_segments/input_agents_3.npz, output_agents_3.npz

📤 输出文件：
    - PNG：swarm_prediction_sample_XXXXXX.png（6 视图对比）
    - PNG：swarm_prediction_interactive_XXXXXX.png（3D 交互式图快照）
    - JSON：evaluation_report_YYYYMMDD_HHMMSS.json（详细指标）

💡 使用示例：
    python visualize_swarm_prediction_v2.py ^
        --model_path 24dmodel/best_model_agents_3_v2.pt ^
        --agents 3 ^
        --num_samples 1
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

# 导入推理脚本中的函数和类
sys.path.insert(0, str(Path(__file__).parent))
from infer_swarm_model_v2 import (
    DynamicsAwareSwarmGRUModel,
    infer_batch,
    compute_features_for_inference,
    estimate_feature_stats_from_data,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def plot_swarm_prediction(history, true_future, pred_future, sample_idx, num_agents, 
                         output_dir, interactive=False, fast_mode=False):
    """
    绘制集群轨迹对比图
    
    Args:
        history: 输入历史轨迹 (seq_in, num_agents, 3)
        true_future: 真实未来轨迹 (seq_out, num_agents, 3)
        pred_future: 预测未来轨迹 (seq_out, num_agents, 3)
        sample_idx: 样本索引
        num_agents: 无人机数量
        output_dir: 输出目录
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
    ax3d.set_title(f'Sample {sample_idx}: 3D Swarm Trajectories (All {num_agents} Agents)', 
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
    png_path = Path(output_dir) / f'swarm_prediction_sample_{sample_idx:06d}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    logger.info(f"    ✓ PNG 保存完成: {png_path.name}")

    if interactive:
        plt.show()

    plt.close(fig)
    
    # 生成交互式 3D 图表（Matplotlib，可在窗口中直接旋转）
    if not fast_mode:
        interactive_path = save_rotatable_plot(history, true_future, pred_future, sample_idx, num_agents, output_dir)
        if interactive_path is not None:
            logger.info(f"    ✓ 交互式 3D 图已保存: {interactive_path.name}")

    return png_path


def save_rotatable_plot(history, true_future, pred_future, sample_idx, num_agents, output_dir):
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
    ax.set_title(f'Sample {sample_idx:06d} - 3D Rotatable Trajectories\n'
                 f'(Drag mouse to rotate, Scroll to zoom)',
                 fontsize=12, fontweight='bold', pad=20)
    
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存为 PNG（用于记录）
    png_path = Path(output_dir) / f'swarm_prediction_interactive_{sample_idx:06d}.png'
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
    max_error = float(np.max(errors))
    
    return {'mae': mae, 'rmse': rmse, 'max_error': max_error}


def load_data_robust(data_dir, num_agents, use_subset=False):
    """加载输入/输出数据对"""
    data_path = Path(data_dir)
    
    # 根据 use_subset 选择数据文件
    suffix = '_subset' if use_subset else ''
    X_file = data_path / f'input_agents_{num_agents}{suffix}.npz'
    Y_file = data_path / f'output_agents_{num_agents}{suffix}.npz'
    
    if not X_file.exists():
        raise FileNotFoundError(f"找不到输入文件: {X_file}")
    if not Y_file.exists():
        raise FileNotFoundError(f"找不到输出文件: {Y_file}")
    
    logger.info(f"  加载数据 (use_subset={use_subset}): {X_file.name}, {Y_file.name}")
    
    X_raw = np.load(X_file)['data']
    Y_raw = np.load(Y_file)['data']
    
    logger.info(f"  X_raw 原始形状: {X_raw.shape}, Y_raw 原始形状: {Y_raw.shape}")
    
    X = np.transpose(X_raw, (1, 0, 2, 3))  # (samples, seq_in, agents, 3)
    Y = np.transpose(Y_raw, (1, 0, 2, 3))  # (samples, seq_out, agents, 3)
    
    logger.info(f"  转置后 X: {X.shape}, Y: {Y.shape}")
    
    return X, Y


def main():
    parser = argparse.ArgumentParser(description='集群轨迹预测可视化 (v2 24D特征)')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data_dir', type=str, default='swarm_segments', help='数据目录')
    parser.add_argument('--agents', type=int, required=True, help='无人机数量')
    parser.add_argument('--num_samples', type=int, default=5, help='可视化样本数')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                       help='指定要可视化的样本索引')
    parser.add_argument('--output_dir', type=str, default='visualization_results_v2', help='输出目录')
    parser.add_argument('--interactive', action='store_true', help='显示交互窗口')
    parser.add_argument('--use_subset', action='store_true', help='使用 subset 数据（需与训练时一致）')
    parser.add_argument('--fast', action='store_true', help='快速模式：只生成 PNG，跳过交互式 3D（节省时间）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)
    
    config = checkpoint.get('config', {})
    
    # 从 checkpoint 直接读取统计量
    if 'output_mean' in checkpoint and 'output_std' in checkpoint:
        output_mean = checkpoint['output_mean']
        output_std = checkpoint['output_std']
        logger.info(f"✓ 从 checkpoint 加载输出统计量: output_mean={output_mean}, output_std={output_std}")
    else:
        logger.warning(f"⚠️ checkpoint 中未找到统计量")
        output_mean = None
        output_std = None
    
    model = DynamicsAwareSwarmGRUModel(
        input_size=24,
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        use_attention=config.get('use_attention', True)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载数据
    logger.info(f"加载数据...")
    X, Y = load_data_robust(args.data_dir, args.agents, use_subset=args.use_subset)
    
    # 安全提取函数：在第一次使用之前确保存在，兼容多种 npz 存储格式
    def _extract_array_from_npz_field(field, expected_len=None):
        """从 npz 字段中安全提取 numpy 数组。

        处理常见存储格式：
        - 直接的 ndarray
        - dtype=object 的包裹（需要 .item() 或索引解包）
        - 单元素数组内嵌一个 list/ndarray
        返回 np.ndarray 或 None（无法解析时）。
        """
        if field is None:
            return None
        try:
            arr = np.asarray(field)
        except Exception:
            try:
                arr = field.item()
                arr = np.asarray(arr)
            except Exception:
                return None

        # 如果是 0-d 或 1 元素且内部是可迭代对象，尝试解包
        if arr.ndim == 0:
            try:
                val = arr.item()
                arr = np.asarray(val)
            except Exception:
                pass
        elif arr.ndim == 1 and arr.size == 1:
            inner = arr[0]
            if isinstance(inner, (list, tuple, np.ndarray)):
                arr = np.asarray(inner)

        # 最终确保是一维数组
        if arr.ndim > 1:
            arr = arr.ravel()

        if expected_len is not None:
            try:
                if arr.size != expected_len:
                    return None
            except Exception:
                return None

        return arr.astype(np.float32)

    # 加载特征统计量（兼容旧格式/多种键名，并验证长度 24）
    logger.info(f"\n【特征统计量加载】")
    stats_path = Path(args.model_path).parent / f'norm_stats_agents_{args.agents}_v2.npz'
    if not stats_path.exists():
        stats_path = Path(args.model_path).parent / f'norm_stats_agents_{args.agents}.npz'

    feature_mean_all = None
    feature_std_all = None

    if stats_path.exists():
        logger.info(f"  优先：从 {stats_path.name} 加载...")
        try:
            stats_file = np.load(stats_path, allow_pickle=True)

            candidate_mean = None
            for k in ('feature_mean', 'input_mean_all', 'input_mean'):
                if k in stats_file:
                    candidate_mean = stats_file[k]
                    break

            candidate_std = None
            for k in ('feature_std', 'input_std_all', 'input_std'):
                if k in stats_file:
                    candidate_std = stats_file[k]
                    break

            feature_mean_all = _extract_array_from_npz_field(candidate_mean, expected_len=24)
            feature_std_all = _extract_array_from_npz_field(candidate_std, expected_len=24)

            if feature_mean_all is not None and feature_std_all is not None:
                logger.info(f"✓ 特征统计加载成功 (来自 .npz): mean shape: {feature_mean_all.shape}, std shape: {feature_std_all.shape}")
            else:
                logger.warning(f"⚠️ .npz 中未找到有效的 24D 特征统计量 (尝试回退到 checkpoint)")
                feature_mean_all = None
                feature_std_all = None
        except Exception as e:
            logger.warning(f"⚠️ 加载 .npz 文件失败: {e}")
            feature_mean_all = None
            feature_std_all = None

    if feature_mean_all is None or feature_std_all is None:
        if 'feature_mean' in checkpoint and 'feature_std' in checkpoint:
            logger.info(f"  备选：从 checkpoint 加载特征统计量...")
            feature_mean_all = _extract_array_from_npz_field(checkpoint.get('feature_mean'), expected_len=24)
            feature_std_all = _extract_array_from_npz_field(checkpoint.get('feature_std'), expected_len=24)
            if feature_mean_all is not None and feature_std_all is not None:
                logger.info(f"✓ 特征统计加载成功 (来自 checkpoint): mean shape: {feature_mean_all.shape}")
        else:
            sample_count = min(200, len(X))
            if sample_count <= 0:
                sample_count = 1
            logger.info(f"  备选：从数据估算 24D 特征统计量（采样 {sample_count} 个轨迹）")
            feature_mean_all, feature_std_all = estimate_feature_stats_from_data(
                X,
                dt=0.1,
                num_samples=sample_count,
                seed=args.seed,
            )
            logger.info(f"  ✓ 估算特征统计 shape: mean {feature_mean_all.shape}, std {feature_std_all.shape}")
    
    # 选择样本
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        sample_indices = np.random.choice(len(X), min(args.num_samples, len(X)), replace=False)
    
    logger.info(f"\n可视化 {len(sample_indices)} 个样本...")
    
    all_metrics = []
    
    for sample_idx in sample_indices:
        logger.info(f"\n处理样本 {sample_idx}...")
        
        # 推理
        X_sample = X[sample_idx:sample_idx+1]
        Y_sample = Y[sample_idx:sample_idx+1]

        # 构建 24D 特征输入
        feature = compute_features_for_inference(X_sample[0], feature_mean_all, feature_std_all)
        feature = feature[np.newaxis, ...]

        debug_flag = sample_idx == sample_indices[0]
        preds = infer_batch(
            model,
            feature,
            X_sample,
            device,
            output_mean if isinstance(output_mean, np.ndarray) else np.array(output_mean),
            output_std if isinstance(output_std, np.ndarray) else np.array(output_std),
            debug=debug_flag,
        )
        
        # 获取数据
        history = X_sample[0]  # (seq_in, agents, 3)
        true_future = Y_sample[0]  # (seq_out, agents, 3)
        pred_future = preds[0]  # (seq_out, agents, 3)
        
        # 计算指标
        metrics = compute_metrics(true_future, pred_future)
        metrics['sample_idx'] = int(sample_idx)
        all_metrics.append(metrics)
        
        logger.info(f"  MAE: {metrics['mae']:.6f}m ({metrics['mae']*100:.2f}cm)")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}m ({metrics['rmse']*100:.2f}cm)")
        logger.info(f"  Max Error: {metrics['max_error']:.6f}m")
        
        # 绘制图表
        logger.info(f"  生成可视化图表...")
        plot_swarm_prediction(history, true_future, pred_future, sample_idx, args.agents,
                             output_dir, interactive=args.interactive, fast_mode=args.fast)
        logger.info(f"  ✓ 样本 {sample_idx} 处理完成")
    
    # 生成总体报告
    logger.info(f"\n生成评估报告...")
    
    # 处理统计量的转换
    if isinstance(output_mean, np.ndarray):
        output_mean_list = output_mean.tolist()
    else:
        output_mean_list = float(output_mean) if output_mean is not None else None
    
    if isinstance(output_std, np.ndarray):
        output_std_list = output_std.tolist()
    else:
        output_std_list = float(output_std) if output_std is not None else None
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'config': config,
        'num_agents': args.agents,
        'num_samples_visualized': len(sample_indices),
        'sample_indices': [int(idx) for idx in sample_indices],
        'data_statistics': {
            'output_mean': output_mean_list,
            'output_std': output_std_list,
        },
        'metrics': all_metrics,
        'summary': {
            'avg_mae': float(np.mean([m['mae'] for m in all_metrics])),
            'avg_rmse': float(np.mean([m['rmse'] for m in all_metrics])),
            'max_mae': float(np.max([m['mae'] for m in all_metrics])),
            'min_mae': float(np.min([m['mae'] for m in all_metrics])),
        }
    }
    
    report_path = output_dir / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n✓ 评估完成！")
    logger.info(f"  平均 MAE: {report['summary']['avg_mae']:.6f}m ({report['summary']['avg_mae']*100:.2f}cm)")
    logger.info(f"  报告已保存: {report_path}")


if __name__ == '__main__':
    main()