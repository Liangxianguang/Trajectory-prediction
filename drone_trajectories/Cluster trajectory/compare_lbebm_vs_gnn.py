#!/usr/bin/env python3
"""
LBEBM3D vs v4 GNN+BiGRU+CrossAttention 对比脚本（精简版，仅对比两种方法）
输出：
  - comparison_lbebm_v4/sample_xxx_comparison.png
  - comparison_lbebm_v4/comparison_summary.json
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
import sys
import importlib.util

# Add paths
cluster_traj_dir = Path(__file__).parent
project_root = cluster_traj_dir.parent.parent
tool_dir = project_root / "drone_trajectories" /"3DMoTraj" / "tool"

sys.path.insert(0, str(cluster_traj_dir))
sys.path.insert(0, str(tool_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Import LBEBM3D（优先标准导入，失败则手动加载）
LBEBM_AVAILABLE = False
try:
    from infer_lbebm3d_baseline import LBEBM3DInfer, infer_model_params_from_state_dict  # type: ignore
    LBEBM_AVAILABLE = True
    logger.info("✓ LBEBM3D 导入成功 (标准方式)")
except Exception as e:
    logger.warning(f"✗ 标准导入失败: {e}")
    try:
        spec = importlib.util.spec_from_file_location("infer_lbebm3d_baseline", str(tool_dir / "infer_lbebm3d_baseline.py"))
        if spec and spec.loader:
            lbebm_module = importlib.util.module_from_spec(spec)
            sys.modules["infer_lbebm3d_baseline"] = lbebm_module
            spec.loader.exec_module(lbebm_module)
            LBEBM3DInfer = lbebm_module.LBEBM3DInfer
            infer_model_params_from_state_dict = lbebm_module.infer_model_params_from_state_dict
            LBEBM_AVAILABLE = True
            logger.info("✓ LBEBM3D 导入成功 (手动加载)")
        else:
            logger.error("✗ 无法创建 spec")
    except Exception as e2:
        logger.error(f"✗ 手动加载失败: {e2}")
        logger.error(f"   尝试路径: {tool_dir / 'infer_lbebm3d_baseline.py'}")

# Import GNN+BiGRU (v4 with enhanced tail dynamics)
try:
    from infer_swarm_model_v4_enhanced_tail import (
        load_data_robust as load_data_v4,
        load_all_32d_features,
        compute_feature_statistics,
        normalize_features,
        EnhancedTailDynamicsAnalyzer,
        infer_batch_v4_enhanced,
        enhance_prediction_with_tail_dynamics,
        apply_physical_constraints,
    )
    from train_swarm_model_v3_with_gnn import (
        DynamicsAwareSwarmGRUModel_with_GNN,
    )
    from train_swarm_model_v2_dynamics_aware import (
        DynamicsAwareSwarmGRUModel,
    )
    V4_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    V4_AVAILABLE = False
    logging.warning(f"v4 GNN+BiGRU not available: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'


def compute_metrics(pred, gt):
    """
    计算详细的评估指标（兼容 visualize_model_comparison.py 的格式）
    
    Args:
        pred: (seq_out, num_agents, 3) 预测
        gt: (seq_out, num_agents, 3) 真值
    
    Returns:
        dict 包含各种指标
    """
    errors = np.linalg.norm(pred - gt, axis=2)  # (seq_out, num_agents)
    
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # ADE: Average Displacement Error (同 MAE)
    ade = mae
    
    # FDE: Final Displacement Error
    fde = float(np.mean(errors[-1]))
    
    # MAPE: Mean Absolute Percentage Error
    true_distances = np.linalg.norm(gt, axis=2)  # (seq_out, num_agents)
    epsilon = 1e-6
    valid_mask = true_distances > epsilon
    if np.any(valid_mask):
        mape = float(np.mean(np.abs(errors[valid_mask] / true_distances[valid_mask]) * 100.0))
    else:
        mape = 0.0
    
    # Per-axis MAE
    mae_x = float(np.mean(np.abs(pred[..., 0] - gt[..., 0])))
    mae_y = float(np.mean(np.abs(pred[..., 1] - gt[..., 1])))
    mae_z = float(np.mean(np.abs(pred[..., 2] - gt[..., 2])))
    
    # Per-step MAE (average across agents)
    mae_per_step = np.mean(errors, axis=1).tolist()  # (seq_out,)
    
    # Per-agent MAE (average across steps)
    mae_per_agent = np.mean(errors, axis=0).tolist()  # (num_agents,)
    
    # Per-step FDE (final position error per step is not standard, but we can compute cumulative)
    fde_per_agent = errors[-1].tolist()  # FDE for each agent at final step
    
    # RMSE per agent
    rmse_per_agent = (np.sqrt(np.mean(errors ** 2, axis=0))).tolist()  # (num_agents,)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "ADE": ade,
        "FDE": fde,
        "MAPE": mape,
        "MAE_X": mae_x,
        "MAE_Y": mae_y,
        "MAE_Z": mae_z,
        "MAE_per_step": mae_per_step,
        "MAE_per_agent": mae_per_agent,
        "FDE_per_agent": fde_per_agent,
        "RMSE_per_agent": rmse_per_agent,
    }


def predict_lbebm_multi_agent(
    model,
    X_sample,
    device,
    data_scale,
    e_init_sig,
    e_prior_sig,
    e_l_steps,
    e_l_step_size,
    e_l_with_noise,
):
    """
    使用 LBEBM3D 模型预测多个 agent 的轨迹
    注意：LBEBM3D 是单个 agent 的模型，所以逐个预测每个 agent
    
    Args:
        model: LBEBM3DInfer 模型
        X_sample: (seq_in, num_agents, 3) 输入
        device: torch device
        data_scale, e_init_sig, e_prior_sig, e_l_steps, e_l_step_size, e_l_with_noise: Langevin 参数
    
    Returns:
        pred_abs: (seq_out, num_agents, 3) 预测
    """
    num_agents = X_sample.shape[1]
    future_length = model.future_length
    pred_abs_all = np.zeros((future_length, num_agents, 3), dtype=np.float32)
    
    # 逐个预测每个 agent
    for agent_idx in range(num_agents):
        past_abs = X_sample[:, agent_idx, :]  # (seq_in, 3)
        
        # Preprocess: 匹配训练时的预处理
        last_obs = past_abs[-1]
        past_rel = (past_abs - last_obs) * data_scale
        
        # 转换为张量并 flatten
        past_flat = torch.from_numpy(past_rel.reshape(1, -1)).to(device=device, dtype=torch.double)  # (1, past_length*3)
        
        # 构建 Langevin 配置
        langevin_cfg = {
            "e_init_sig": e_init_sig,
            "e_prior_sig": e_prior_sig,
            "e_l_steps": e_l_steps,
            "e_l_step_size": e_l_step_size,
            "e_l_with_noise": e_l_with_noise,
        }

        try:
            plan_flat = model.sample_plan(past_flat, langevin_cfg)  # (1, num_subgoals*3)
            pred_rel = model.predict(past_flat, plan_flat).cpu().numpy()  # (1, future_length, 3)

            # 反归一化：除以 data_scale 并加上 last_obs
            pred_abs_agent = pred_rel[0] / data_scale + last_obs  # (future_length, 3)
        except Exception as e:
            logger.warning(f"Failed to predict for agent {agent_idx}: {e}, using zero prediction")
            pred_abs_agent = np.zeros((future_length, 3), dtype=np.float32)
        
        pred_abs_all[:, agent_idx, :] = pred_abs_agent
    
    return pred_abs_all


def predict_gnn_bigru(
    model,
    X_sample,
    features,
    device,
    output_mean,
    output_std,
    feature_means,
    feature_stds,
    tail_analyzer=None,
    use_tail_enhancement=False,
    edge_threshold=5.0,
    use_physical_constraints_flag=True,
    use_enhanced_infer=False,
    pc_dt: float = 0.1,
    pc_smoothing_weight: float = 0.2,
    tail_decay: float = 0.1,
):
    """
    使用 GNN+BiGRU 模型预测单个样本（v4_enhanced 风格）
    
    Args:
        model: v4 GNN+BiGRU 模型
        X_sample: (seq_in, num_agents, 3)
        features: (seq_in, num_agents, 32)
        device: torch device
        output_mean, output_std: 输出归一化参数
        feature_means, feature_stds: 特征归一化参数
        tail_analyzer: EnhancedTailDynamicsAnalyzer instance (optional)
        use_tail_enhancement: Whether to use tail enhancement
        edge_threshold: GNN edge threshold
        use_physical_constraints_flag: Whether to apply physical constraints (default: False for fair comparison)
    
    Returns:
        pred_abs: (seq_out, num_agents, 3)
    """
    # Normalize features
    if feature_means is not None and feature_stds is not None:
        safe_std = np.where(feature_stds < 1e-8, 1.0, feature_stds)
        features_norm = (features - feature_means) / safe_std
        features_norm = np.clip(features_norm, -5.0, 5.0)
    else:
        features_norm = features
    
    X_batch = X_sample[np.newaxis, ...]  # (1, seq_in, agents, 3)
    F_batch = features_norm[np.newaxis, ...]  # (1, seq_in, agents, 32)
    
    model.eval()
    with torch.no_grad():
        features_t = torch.from_numpy(F_batch).float().to(device)
        x_orig_t = torch.from_numpy(X_batch).float().to(device)
        
        # 模型输出归一化的 delta：(1, seq_out, agents, 3)
        pred_delta_norm, _, _ = model(
            features_t, x_orig_t,
            y=None, y_velocity=None, y_accel=None,
            teacher_forcing_ratio=0.0
        )
        
        # 反归一化到物理单位
        output_mean_t = torch.tensor(output_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        output_std_t = torch.tensor(output_std, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        
        pred_delta_phys = (pred_delta_norm * output_std_t + output_mean_t).cpu().numpy()  # (1, seq_out, agents, 3)
        
        # 尾部动力学增强（可选）
        if use_tail_enhancement and tail_analyzer is not None:
            pred_delta_phys = enhance_prediction_with_tail_dynamics(
                pred_delta_phys, X_batch, tail_analyzer, decay_factor=tail_decay
            )
        
        if use_enhanced_infer and use_physical_constraints_flag:
            # 使用物理约束重建平滑轨迹（增强推理）
            pred_abs = apply_physical_constraints(
                X_batch,
                pred_delta_phys,
                dt=pc_dt,
                smoothing_weight=pc_smoothing_weight
            )  # (1, seq_out, agents, 3)
        else:
            # 直接从 delta 重建位置（保持原始预测）
            last_pos = X_batch[:, -1:, :, :]  # (1, 1, num_agents, 3)
            pred_abs = last_pos + np.cumsum(pred_delta_phys, axis=1)  # (1, seq_out, num_agents, 3)
        
        pred_abs = pred_abs[0]  # (seq_out, agents, 3)
    
    return pred_abs


def visualize_comparison(X_sample, Y_sample, pred_lbebm, pred_gnn, sample_idx, output_path):
    """
    对比 LBEBM3D（红） vs v4 GNN（橙）
    现在 LBEBM 也预测所有 agents
    """
    num_agents = X_sample.shape[1]
    fig = plt.figure(figsize=(20, 13))

    colors = {
        "history": "b",
        "gt": "#27AE60",  # green
        "lbebm": "#E74C3C",  # red
        "gnn": "#E67E22",  # orange
    }

    # 1. 3D - 绘制所有 agent
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    for aid in range(num_agents):
        if aid == 0:
            ax3d.plot(X_sample[:, aid, 0], X_sample[:, aid, 1], X_sample[:, aid, 2], "b-o", linewidth=2.5, markersize=5, alpha=0.8, label="History")
        else:
            ax3d.plot(X_sample[:, aid, 0], X_sample[:, aid, 1], X_sample[:, aid, 2], "b-o", linewidth=2.5, markersize=5, alpha=0.8)
        last = X_sample[-1:, aid, :]
        gt_traj = np.vstack([last, Y_sample[:, aid, :]])
        lb_traj = np.vstack([last, pred_lbebm[:, aid, :]])
        gnn_traj = np.vstack([last, pred_gnn[:, aid, :]])
        
        # Ground truth - 所有 agent
        ax3d.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], "s-", color=colors["gt"], linewidth=2.8, markersize=7, alpha=0.9, label="Ground Truth" if aid == 0 else "")
        
        # LBEBM - 所有 agent
        ax3d.plot(lb_traj[:, 0], lb_traj[:, 1], lb_traj[:, 2], "^--", color=colors["lbebm"], linewidth=2.5, markersize=6, alpha=0.85, label="LBEBM3D" if aid == 0 else "")
        
        # GNN - 所有 agent
        ax3d.plot(gnn_traj[:, 0], gnn_traj[:, 1], gnn_traj[:, 2], "D--", color=colors["gnn"], linewidth=2.5, markersize=6, alpha=0.85, label="GNN+BiGRU" if aid == 0 else "")
    
    ax3d.set_xlabel("X (m)", fontsize=11, fontweight="bold")
    ax3d.set_ylabel("Y (m)", fontsize=11, fontweight="bold")
    ax3d.set_zlabel("Z (m)", fontsize=11, fontweight="bold")
    ax3d.set_title(f"Sample {sample_idx}: 3D Trajectories (LBEBM vs GNN)", fontsize=12, fontweight="bold")
    ax3d.legend(fontsize=10, loc="upper left")
    ax3d.grid(True, alpha=0.3)

    def plot_2d(ax, ax1, ax2, title):
        for aid in range(num_agents):
            last = X_sample[-1:, aid, :]
            gt_traj = np.vstack([last, Y_sample[:, aid, :]])
            lb_traj = np.vstack([last, pred_lbebm[:, aid, :]])
            gnn_traj = np.vstack([last, pred_gnn[:, aid, :]])
            
            if aid == 0:
                ax.plot(X_sample[:, aid, ax1], X_sample[:, aid, ax2], "b-o", linewidth=2.5, markersize=5, alpha=0.8, label="History")
            else:
                ax.plot(X_sample[:, aid, ax1], X_sample[:, aid, ax2], "b-o", linewidth=2.5, markersize=5, alpha=0.8)
            
            # Ground truth - 所有 agent
            ax.plot(gt_traj[:, ax1], gt_traj[:, ax2], "s-", color=colors["gt"], linewidth=2.8, markersize=7, alpha=0.9, label="True" if aid == 0 else "")
            
            # LBEBM - 所有 agent
            ax.plot(lb_traj[:, ax1], lb_traj[:, ax2], "^--", color=colors["lbebm"], linewidth=2.5, markersize=6, alpha=0.85, label="LBEBM3D" if aid == 0 else "")
            
            # GNN - 所有 agent
            ax.plot(gnn_traj[:, ax1], gnn_traj[:, ax2], "D--", color=colors["gnn"], linewidth=2.5, markersize=6, alpha=0.85, label="GNN+BiGRU" if aid == 0 else "")
        ax.set_xlabel(["X", "Y", "Z"][ax1] + " (m)", fontsize=11, fontweight="bold")
        ax.set_ylabel(["X", "Y", "Z"][ax2] + " (m)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plot_2d(fig.add_subplot(2, 3, 2), 0, 1, "XY Plane Projection")
    plot_2d(fig.add_subplot(2, 3, 3), 0, 2, "XZ Plane Projection")
    plot_2d(fig.add_subplot(2, 3, 4), 1, 2, "YZ Plane Projection")

    # per-step axis MAE
    ax_err = fig.add_subplot(2, 3, 5)
    steps = np.arange(pred_lbebm.shape[0])
    err_x_lb = np.mean(np.abs(pred_lbebm[:, :, 0] - Y_sample[:, :, 0]), axis=1)
    err_y_lb = np.mean(np.abs(pred_lbebm[:, :, 1] - Y_sample[:, :, 1]), axis=1)
    err_z_lb = np.mean(np.abs(pred_lbebm[:, :, 2] - Y_sample[:, :, 2]), axis=1)
    err_x_gn = np.abs(pred_gnn[:, :, 0] - Y_sample[:, :, 0]).mean(axis=1)
    err_y_gn = np.abs(pred_gnn[:, :, 1] - Y_sample[:, :, 1]).mean(axis=1)
    err_z_gn = np.abs(pred_gnn[:, :, 2] - Y_sample[:, :, 2]).mean(axis=1)
    ax_err.plot(steps, err_x_lb, "r^-", linewidth=2.5, markersize=7, alpha=0.8, label="LBEBM |X|")
    ax_err.plot(steps, err_y_lb, "g^-", linewidth=2.5, markersize=7, alpha=0.8, label="LBEBM |Y|")
    ax_err.plot(steps, err_z_lb, "b^-", linewidth=2.5, markersize=7, alpha=0.8, label="LBEBM |Z|")
    ax_err.plot(steps, err_x_gn, "D-", color=colors["gnn"], linewidth=2.5, markersize=7, alpha=0.8, label="GNN |X|")
    ax_err.plot(steps, err_y_gn, "D--", color=colors["gnn"], linewidth=2.5, markersize=7, alpha=0.8, label="GNN |Y|")
    ax_err.plot(steps, err_z_gn, "D:", color=colors["gnn"], linewidth=2.5, markersize=7, alpha=0.8, label="GNN |Z|")
    ax_err.set_xlabel("Prediction Step", fontsize=11, fontweight="bold")
    ax_err.set_ylabel("Mean Absolute Error (m)", fontsize=11, fontweight="bold")
    ax_err.set_title("Per-Step Axis-wise Error", fontsize=12, fontweight="bold")
    ax_err.legend(fontsize=9, loc="best")
    ax_err.grid(True, alpha=0.3)

    # overall per-step L2
    ax_bar = fig.add_subplot(2, 3, 6)
    l2_lb = np.mean(np.linalg.norm(pred_lbebm - Y_sample, axis=2), axis=1)
    l2_gn = np.mean(np.linalg.norm(pred_gnn - Y_sample, axis=2), axis=1)
    width = 0.35
    x = np.arange(len(steps))
    bars1 = ax_bar.bar(x - width / 2, l2_lb, width, color=colors["lbebm"], alpha=0.7, edgecolor="darkred", linewidth=1.5, label="LBEBM3D")
    bars2 = ax_bar.bar(x + width / 2, l2_gn, width, color=colors["gnn"], alpha=0.7, edgecolor="darkred", linewidth=1.5, label="GNN+BiGRU")
    for b in bars1:
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for b in bars2:
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_xlabel("Prediction Step", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("Position Error (m)", fontsize=11, fontweight="bold")
    ax_bar.set_title("Position Error per Step (All Agents Avg)", fontsize=12, fontweight="bold")
    ax_bar.legend(fontsize=10, loc="best")
    ax_bar.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LBEBM3D vs GNN+BiGRU Comparison')
    
    parser.add_argument('--data_dir', required=True, help='Data directory')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--use_subset', action='store_true', help='Use subset data')
    
    # Model paths
    parser.add_argument('--lbebm_model', required=True, help='LBEBM3D model path')
    parser.add_argument('--gnn_model', required=True, help='GNN+BiGRU model path')
    
    # LBEBM parameters
    parser.add_argument('--data_scale', type=float, default=1.0)
    parser.add_argument('--e_init_sig', type=float, default=2.0)
    parser.add_argument('--e_prior_sig', type=float, default=2.0)
    parser.add_argument('--e_l_steps', type=int, default=20)
    parser.add_argument('--e_l_step_size', type=float, default=0.4)
    parser.add_argument('--e_l_with_noise', action='store_true')
    
    # GNN parameters
    parser.add_argument('--features_32d_dir', default='features_32d', help='32D features dir')
    parser.add_argument('--edge_threshold', type=float, default=5.0)
    parser.add_argument('--no_gnn', action='store_true')
    parser.add_argument('--no_tail_enhancement', action='store_true')
    parser.add_argument('--use_physical_constraints', action='store_true', help='Apply physical constraints (default: off for fair comparison)')
    parser.add_argument('--gnn_use_enhanced_infer', action='store_true',
                       help='Use enhanced v4 inference (tail + physical constraints) for GNN')
    parser.add_argument('--gnn_pc_dt', type=float, default=0.1, help='dt for physical constraints when enhanced infer is used')
    parser.add_argument('--gnn_pc_smoothing', type=float, default=0.3, help='smoothing weight for physical constraints')
    parser.add_argument('--gnn_tail_decay', type=float, default=0.15, help='tail enhancement decay factor')
    
    # Sample selection
    parser.add_argument('--sample_indices', type=str, default=None,
                       help='Comma-separated sample indices')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of random samples')
    parser.add_argument('--seed', type=int, default=42)
    
    # Outlier handling
    parser.add_argument('--remove_gnn_outliers', action='store_true',
                       help='Remove a small fraction of worst GNN samples (by metric) from both models for robust stats')
    parser.add_argument('--gnn_outlier_metric', type=str, default='MAE',
                       choices=['MAE', 'FDE'],
                       help='Metric used to detect GNN outliers (default: MAE)')
    parser.add_argument('--gnn_outlier_percent', type=float, default=1.0,
                       help='Percentage of worst GNN samples (by chosen metric) to drop (0-50, default: 1.0)')
    
    # Output
    parser.add_argument('--output_dir', default='comparison_lbebm_v4', help='Output directory')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not LBEBM_AVAILABLE:
        logger.error("❌ LBEBM3D 模块无法导入，无法进行对比")
        logger.error(f"   确认文件存在: {tool_dir / 'infer_lbebm3d_baseline.py'}")
        sys.exit(1)
    
    if not V4_AVAILABLE:
        logger.error("❌ GNN+BiGRU 模块无法导入")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # === Load Data ===
    logger.info(f"Loading data from: {args.data_dir}")
    X_all, Y_all = load_data_v4(args.data_dir, args.agents, use_subset=args.use_subset)
    logger.info(f"Data shape: X={X_all.shape}, Y={Y_all.shape}")
    
    # === Load LBEBM Model ===
    logger.info(f"Loading LBEBM3D model: {args.lbebm_model}")
    ckpt = torch.load(args.lbebm_model, map_location='cpu',weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    
    params = infer_model_params_from_state_dict(state_dict)
    sub_goal_indexes = [2, 5, 7, 9] if params['future_length'] >= 10 else \
                       list(np.linspace(0, params['future_length']-1, params['num_subgoals'], dtype=int))
    
    lbebm_model = LBEBM3DInfer(
        enc_past_size=params['enc_past_size'],
        enc_dest_size=params['enc_dest_size'],
        enc_latent_size=params['enc_latent_size'],
        dec_size=params['dec_size'],
        predictor_size=params['predictor_size'],
        fdim=params['fdim'],
        zdim=params['zdim'],
        ny=params['ny'],
        past_length=params['past_length'],
        future_length=params['future_length'],
        sub_goal_indexes=sub_goal_indexes,
    ).double()
    lbebm_model.load_state_dict(state_dict, strict=True)
    lbebm_model.to(device)
    lbebm_model.eval()
    logger.info("LBEBM3D model loaded")
    
    # === Load GNN+BiGRU Model ===
    logger.info(f"Loading GNN+BiGRU model: {args.gnn_model}")
    try:
        checkpoint = torch.load(args.gnn_model, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.gnn_model, map_location='cpu')
    
    config = checkpoint.get('config', {})
    use_gnn = config.get('use_gnn', True) and not args.no_gnn
    
    if use_gnn:
        gnn_model = DynamicsAwareSwarmGRUModel_with_GNN(
            input_size=config.get('input_size', 32),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 3),
            output_size=3,
            dropout=config.get('dropout', 0.2),
            use_attention=config.get('use_attention', True),
            gnn_hidden=config.get('gnn_hidden', 64),
            num_gnn_heads=config.get('gnn_heads', 4),
            edge_threshold=args.edge_threshold,
            fusion_mode=config.get('gnn_fusion_mode', 'concat'),
        )
    else:
        gnn_model = DynamicsAwareSwarmGRUModel(
            input_size=config.get('input_size', 32),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 3),
            output_size=3,
            dropout=config.get('dropout', 0.2),
            use_attention=config.get('use_attention', True),
        )
    
    gnn_model.load_state_dict(checkpoint['model_state_dict'])
    gnn_model.to(device)
    gnn_model.eval()
    logger.info("GNN+BiGRU model loaded")
    
    # 加载 GNN 模型的输出统计量
    gnn_output_mean = np.array(checkpoint.get('output_mean', np.zeros(3)), dtype=np.float32)
    gnn_output_std = np.array(checkpoint.get('output_std', np.ones(3)), dtype=np.float32)
    logger.info(f"GNN output stats: mean={gnn_output_mean}, std={gnn_output_std}")
    
    # === Initialize Tail Dynamics Analyzer (Optional) ===
    tail_analyzer = None
    if not args.no_tail_enhancement:
        logger.info("Initializing tail dynamics analyzer for GNN...")
        tail_analyzer = EnhancedTailDynamicsAnalyzer(
            short_window=3,
            medium_window=5,
            long_window=8,
            dt=0.1
        )
        logger.info("✓ Tail analyzer initialized")
    else:
        logger.info("Tail enhancement disabled")
    
    # === Load Features for GNN ===
    logger.info("Loading 32D features for GNN...")
    features_all, feature_means, feature_stds = load_32d_features_with_stats(
        args.features_32d_dir, args.agents, args.use_subset
    )
    if features_all is None:
        raise RuntimeError("Failed to load 32D features")
    logger.info(f"Features shape: {features_all.shape}")
    
    # === Select Samples ===
    np.random.seed(args.seed)
    total_samples = len(X_all)
    
    if args.sample_indices:
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(',')]
    else:
        num_samples = min(args.num_samples, total_samples)
        sample_indices = np.random.choice(total_samples, num_samples, replace=False).tolist()
    
    logger.info(f"Selected samples: {sample_indices}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Run Comparison ===
    all_metrics_lbebm = []
    all_metrics_gnn = []
    
    for sample_idx in sample_indices:
        logger.info(f"\n=== Processing sample {sample_idx} ===")
        
        X_sample = X_all[sample_idx]  # (seq_in, agents, 3)
        Y_sample = Y_all[sample_idx]  # (seq_out, agents, 3)
        features_sample = features_all[sample_idx]  # (seq_in, agents, 32)
        
        # LBEBM prediction
        logger.info("Running LBEBM3D prediction...")
        pred_lbebm = predict_lbebm_multi_agent(
            lbebm_model, X_sample, device, args.data_scale,
            args.e_init_sig, args.e_prior_sig, args.e_l_steps,
            args.e_l_step_size, args.e_l_with_noise
        )
        
        # GNN prediction
        logger.info("Running GNN+BiGRU prediction...")
        pred_gnn = predict_gnn_bigru(
            gnn_model, X_sample, features_sample, device,
            gnn_output_mean, gnn_output_std,
            feature_means, feature_stds,
            tail_analyzer=tail_analyzer,
            use_tail_enhancement=not args.no_tail_enhancement,
            edge_threshold=args.edge_threshold,
            use_physical_constraints_flag=(args.use_physical_constraints or args.gnn_use_enhanced_infer),
            use_enhanced_infer=args.gnn_use_enhanced_infer,
            pc_dt=args.gnn_pc_dt,
            pc_smoothing_weight=args.gnn_pc_smoothing,
            tail_decay=args.gnn_tail_decay,
        )
        
        # Compute metrics (for all agents)
        metrics_lbebm = compute_metrics(pred_lbebm, Y_sample)
        metrics_gnn = compute_metrics(pred_gnn, Y_sample)
        
        logger.info(f"LBEBM3D - ADE: {metrics_lbebm['ADE']:.4f}, FDE: {metrics_lbebm['FDE']:.4f}")
        logger.info(f"GNN+BiGRU - ADE: {metrics_gnn['ADE']:.4f}, FDE: {metrics_gnn['FDE']:.4f}")
        
        all_metrics_lbebm.append(metrics_lbebm)
        all_metrics_gnn.append(metrics_gnn)
        
        # Visualize
        output_path = output_dir / f"sample_{sample_idx}_comparison.png"
        visualize_comparison(
            X_sample, Y_sample, pred_lbebm, pred_gnn,
            sample_idx, output_path
        )
    
    # === Optional: remove extreme GNN outliers (and corresponding LBEBM samples) ===
    used_metrics_lbebm = all_metrics_lbebm
    used_metrics_gnn = all_metrics_gnn
    used_sample_indices = sample_indices
    removed_sample_indices = []
    
    if args.remove_gnn_outliers:
        metric_name = args.gnn_outlier_metric
        if not all_metrics_gnn or metric_name not in all_metrics_gnn[0]:
            logger.warning(f"Outlier removal requested, but metric '{metric_name}' not found. Skipping outlier removal.")
        else:
            perc = max(0.0, min(args.gnn_outlier_percent, 50.0))
            if perc <= 0.0:
                logger.info("Outlier percentage <= 0, skipping outlier removal.")
            else:
                gnn_values = np.array([m[metric_name] for m in all_metrics_gnn], dtype=np.float32)
                n = len(gnn_values)
                k = int(np.ceil(n * perc / 100.0))
                if k <= 0:
                    logger.info("Computed outlier count k=0, skipping outlier removal.")
                elif k >= n:
                    logger.warning("Outlier count >= sample count, skipping outlier removal.")
                else:
                    # threshold: keep the best (n-k) samples
                    threshold = np.partition(gnn_values, n - k - 1)[n - k - 1]
                    keep_mask = gnn_values <= threshold
                    
                    used_metrics_lbebm = [m for m, keep in zip(all_metrics_lbebm, keep_mask) if keep]
                    used_metrics_gnn = [m for m, keep in zip(all_metrics_gnn, keep_mask) if keep]
                    removed_sample_indices = [idx for idx, keep in zip(sample_indices, keep_mask) if not keep]
                    used_sample_indices = [idx for idx, keep in zip(sample_indices, keep_mask) if keep]
                    
                    logger.info(
                        f"Outlier removal enabled: metric={metric_name}, "
                        f"percent={perc}%, removed={len(removed_sample_indices)}/{n} samples."
                    )
                    if removed_sample_indices:
                        logger.info(f"Removed sample indices (GNN worst cases): {removed_sample_indices}")
                    logger.info(f"Remaining samples for statistics: {len(used_sample_indices)}")
    
    # === Aggregate Results === (on possibly filtered samples)
    logger.info("\n=== Overall Results ===")
    
    # 计算所有指标的汇总统计
    def compute_aggregate_stats(metrics_list):
        """计算所有指标的汇总统计"""
        stats = {}
        
        # 主要指标
        for metric in ['MAE', 'ADE', 'FDE', 'RMSE', 'MAPE', 'MAE_X', 'MAE_Y', 'MAE_Z']:
            if metrics_list and metric in metrics_list[0]:
                values = np.array([m[metric] for m in metrics_list])
                stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                }
        
        # Per-step 误差
        if metrics_list and 'MAE_per_step' in metrics_list[0]:
            per_step_values = np.array([m['MAE_per_step'] for m in metrics_list])
            stats['MAE_per_step_avg'] = float(np.mean(per_step_values))
            stats['MAE_per_step_std'] = float(np.std(per_step_values))
        
        # Per-agent 误差
        if metrics_list and 'MAE_per_agent' in metrics_list[0]:
            per_agent_values = np.array([m['MAE_per_agent'] for m in metrics_list])
            stats['MAE_per_agent_mean'] = [float(np.mean(per_agent_values[:, i])) for i in range(per_agent_values.shape[1])]
            stats['MAE_per_agent_std'] = [float(np.std(per_agent_values[:, i])) for i in range(per_agent_values.shape[1])]
        
        if metrics_list and 'FDE_per_agent' in metrics_list[0]:
            per_agent_values = np.array([m['FDE_per_agent'] for m in metrics_list])
            stats['FDE_per_agent_mean'] = [float(np.mean(per_agent_values[:, i])) for i in range(per_agent_values.shape[1])]
            stats['FDE_per_agent_std'] = [float(np.std(per_agent_values[:, i])) for i in range(per_agent_values.shape[1])]
        
        return stats
    
    lbebm_aggregate = compute_aggregate_stats(used_metrics_lbebm)
    gnn_aggregate = compute_aggregate_stats(used_metrics_gnn)
    
    logger.info(f"\n✓ LBEBM3D Statistics:")
    logger.info(f"  ADE: {lbebm_aggregate['ADE']['mean']:.4f} ± {lbebm_aggregate['ADE']['std']:.4f}")
    logger.info(f"  FDE: {lbebm_aggregate['FDE']['mean']:.4f} ± {lbebm_aggregate['FDE']['std']:.4f}")
    logger.info(f"  RMSE: {lbebm_aggregate['RMSE']['mean']:.4f} ± {lbebm_aggregate['RMSE']['std']:.4f}")
    logger.info(f"  MAPE: {lbebm_aggregate['MAPE']['mean']:.2f}% ± {lbebm_aggregate['MAPE']['std']:.2f}%")
    
    logger.info(f"\n✓ GNN+BiGRU Statistics:")
    logger.info(f"  ADE: {gnn_aggregate['ADE']['mean']:.4f} ± {gnn_aggregate['ADE']['std']:.4f}")
    logger.info(f"  FDE: {gnn_aggregate['FDE']['mean']:.4f} ± {gnn_aggregate['FDE']['std']:.4f}")
    logger.info(f"  RMSE: {gnn_aggregate['RMSE']['mean']:.4f} ± {gnn_aggregate['RMSE']['std']:.4f}")
    logger.info(f"  MAPE: {gnn_aggregate['MAPE']['mean']:.2f}% ± {gnn_aggregate['MAPE']['std']:.2f}%")
    
    # 性能对比
    logger.info(f"\n✓ Performance Comparison (improvement %):")
    ade_improve = ((lbebm_aggregate['ADE']['mean'] - gnn_aggregate['ADE']['mean']) / lbebm_aggregate['ADE']['mean'] * 100)
    fde_improve = ((lbebm_aggregate['FDE']['mean'] - gnn_aggregate['FDE']['mean']) / lbebm_aggregate['FDE']['mean'] * 100)
    logger.info(f"  ADE improvement: {ade_improve:+.2f}% (GNN better)" if ade_improve > 0 else f"  ADE improvement: {ade_improve:+.2f}% (LBEBM better)")
    logger.info(f"  FDE improvement: {fde_improve:+.2f}% (GNN better)" if fde_improve > 0 else f"  FDE improvement: {fde_improve:+.2f}% (LBEBM better)")
    
    # Save summary
    summary = {
        "num_samples": len(used_metrics_lbebm),
        "sample_indices": used_sample_indices,
        "removed_sample_indices": removed_sample_indices,
        "outlier_filter": {
            "enabled": bool(args.remove_gnn_outliers),
            "metric": args.gnn_outlier_metric,
            "percent": args.gnn_outlier_percent,
        },
        "LBEBM3D": {
            "aggregate_stats": lbebm_aggregate,
            "all_metrics": used_metrics_lbebm,
        },
        "GNN_BiGRU": {
            "aggregate_stats": gnn_aggregate,
            "all_metrics": used_metrics_gnn,
        },
    }
    
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"All visualizations saved to: {output_dir}")


def load_32d_features_with_stats(features_dir, num_agents, use_subset=False):
    """加载 32D 特征和统计量"""
    features_dir = Path(features_dir)
    subset_suffix = '_subset' if use_subset else ''
    
    candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_32d.npz',
        features_dir / f'features_agents_{num_agents}_32d{subset_suffix}.npz',
    ]
    
    for path in candidates:
        if path.exists():
            try:
                logger.info(f"Loading 32D features: {path}")
                data = np.load(path)
                features = np.asarray(data['features'])
                
                # 尝试从文件加载统计量
                means = data.get('means', None)
                stds = data.get('stds', None)
                
                if means is not None:
                    means = np.asarray(means)
                else:
                    # 计算统计量
                    logger.info("Computing feature statistics from data...")
                    subset = features[:min(1000, len(features))].reshape(-1, 32)
                    means = np.mean(subset, axis=0)
                
                if stds is not None:
                    stds = np.asarray(stds)
                else:
                    # 计算统计量
                    subset = features[:min(1000, len(features))].reshape(-1, 32)
                    stds = np.std(subset, axis=0)
                    stds = np.where(stds < 1e-8, 1.0, stds)
                
                logger.info(f"Features loaded: {features.shape}, stats: mean={means.shape}, std={stds.shape}")
                return features, means, stds
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    
    return None, None, None


if __name__ == '__main__':
    main()
