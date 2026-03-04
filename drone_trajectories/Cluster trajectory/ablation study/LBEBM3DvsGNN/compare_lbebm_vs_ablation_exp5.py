#!/usr/bin/env python3
"""
LBEBM3D vs Ablation Exp5 (DG32-BCAT) 对比脚本
输出：
  - comparison_exp5_vs_lbebm/sample_xxx_comparison.png
  - comparison_exp5_vs_lbebm/comparison_summary.json
"""
import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Resolve paths
current_dir = Path(__file__).resolve().parent
cluster_traj_dir = current_dir.parent.parent
project_root = cluster_traj_dir.parent.parent
tool_dir = project_root / "drone_trajectories" / "3DMoTraj" / "tool"

sys.path.insert(0, str(cluster_traj_dir))
sys.path.insert(0, str(tool_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Import LBEBM3D
LBEBM_AVAILABLE = False
try:
    from infer_lbebm3d_baseline import LBEBM3DInfer, infer_model_params_from_state_dict  # type: ignore
    LBEBM_AVAILABLE = True
    logger.info("✓ LBEBM3D 导入成功 (标准方式)")
except Exception as e:
    logger.warning(f"✗ 标准导入失败: {e}")
    try:
        spec = __import__("importlib.util").util.spec_from_file_location(
            "infer_lbebm3d_baseline", str(tool_dir / "infer_lbebm3d_baseline.py")
        )
        if spec and spec.loader:
            lbebm_module = __import__("importlib.util").util.module_from_spec(spec)
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

# Import Exp5 model (v3 with GNN)
try:
    from train_swarm_model_v3_with_gnn import DynamicsAwareSwarmGRUModel_with_GNN
    EXP5_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    EXP5_AVAILABLE = False
    logger.warning(f"Exp5 模型不可用: {e}")

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_data_ablation(data_dir: str, num_agents: int, use_subset: bool = False):
    data_path = Path(data_dir)
    subset_suffix = '_subset' if use_subset else ''
    x_file = data_path / f'input_agents_{num_agents}{subset_suffix}.npz'
    y_file = data_path / f'output_agents_{num_agents}{subset_suffix}.npz'

    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {x_file}, {y_file}")

    x = np.load(x_file)['data']
    y = np.load(y_file)['data']

    # (seq, samples, agents, 3) -> (samples, seq, agents, 3)
    x = np.transpose(x, (1, 0, 2, 3))
    y = np.transpose(y, (1, 0, 2, 3))

    return x, y


def load_32d_features_with_stats(features_dir, num_agents, use_subset=False):
    features_dir = Path(features_dir)
    subset_suffix = '_subset' if use_subset else ''

    candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_32d.npz',
        features_dir / f'features_agents_{num_agents}_32d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
        features_dir / f'features_agents_{num_agents}_32d.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}.npz',
    ]

    for path in candidates:
        if path.exists():
            try:
                logger.info(f"Loading 32D features: {path}")
                data = np.load(path)
                features = np.asarray(data['features'])

                means = data.get('means', None)
                stds = data.get('stds', None)

                if means is not None:
                    means = np.asarray(means)
                else:
                    subset = features[:min(1000, len(features))].reshape(-1, 32)
                    means = np.mean(subset, axis=0)

                if stds is not None:
                    stds = np.asarray(stds)
                else:
                    subset = features[:min(1000, len(features))].reshape(-1, 32)
                    stds = np.std(subset, axis=0)
                    stds = np.where(stds < 1e-8, 1.0, stds)

                logger.info(f"Features loaded: {features.shape}, stats: mean={means.shape}, std={stds.shape}")
                return features, means, stds
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    return None, None, None


def apply_physical_constraints(history, pred_delta, dt=0.1, smoothing_weight=0.2, constraint_relaxation=1.0):
    """
    Apply physical constraints with velocity-aware reconstruction.
    (Copied from v4-style ablation inference for consistency)
    """
    history = np.array(history, dtype=np.float32)
    seq_in, num_agents, _ = history.shape

    if seq_in < 2:
        return history[-1:, :, :] + pred_delta

    history_vel = np.diff(history, axis=0) / dt

    if history_vel.shape[0] == 0:
        return history[-1:, :, :] + pred_delta

    if history_vel.shape[0] >= 5:
        last_vel = np.mean(history_vel[-5:, :, :], axis=0)
    else:
        last_vel = np.mean(history_vel, axis=0)

    history_acc = np.diff(history_vel, axis=0) / dt if history_vel.shape[0] > 1 else np.zeros((1, num_agents, 3), dtype=np.float32)
    avg_acc = history_acc.mean(axis=0) if history_acc.shape[0] > 0 else np.zeros((num_agents, 3), dtype=np.float32)

    if history_acc.shape[0] >= 3:
        recent_acc = history_acc[-3:].mean(axis=0)
    elif history_acc.shape[0] > 0:
        recent_acc = history_acc[-1, :, :]
    else:
        recent_acc = np.zeros((num_agents, 3), dtype=np.float32)

    vel_norms = np.linalg.norm(history_vel, axis=2)
    max_vel = np.maximum(np.max(vel_norms, axis=0), 1e-3)

    acc_norms = np.linalg.norm(history_acc, axis=2)
    max_acc = np.maximum(np.max(acc_norms, axis=0), 1e-3)

    relaxation = constraint_relaxation if constraint_relaxation > 0 else 1.0
    max_vel = max_vel * relaxation
    max_acc = max_acc * relaxation

    current_pos = history[-1:, :, :].copy()
    current_vel = last_vel.copy()
    seq_out = pred_delta.shape[0]

    reconstructed = np.zeros((seq_out, num_agents, 3), dtype=np.float32)

    last_vel_mag = np.linalg.norm(last_vel, axis=1, keepdims=True)
    last_vel_mag = np.maximum(last_vel_mag, 1e-3)

    for step in range(seq_out):
        if step == 0:
            step_delta = pred_delta[step, :, :]
        else:
            step_delta = pred_delta[step, :, :] - pred_delta[step - 1, :, :]

        step_delta_norm = np.linalg.norm(step_delta, axis=1, keepdims=True) + 1e-8
        desired_vel_dir = step_delta / step_delta_norm

        last_vel_norm = np.linalg.norm(last_vel, axis=1, keepdims=True) + 1e-8
        last_vel_unit = last_vel / last_vel_norm
        accel_tangent = np.sum(recent_acc * last_vel_unit, axis=1, keepdims=True)

        accel_factor = 1.0 + np.tanh(accel_tangent * dt * 2.0) * 0.25
        target_vel_mag = last_vel_mag * accel_factor

        desired_vel = desired_vel_dir * target_vel_mag

        raw_accel = (desired_vel - current_vel) / dt
        accel_weight = 0.4
        constrained_accel = (
            (1 - smoothing_weight) * raw_accel +
            smoothing_weight * (1 - accel_weight) * avg_acc +
            smoothing_weight * accel_weight * recent_acc
        )

        accel_norm = np.linalg.norm(constrained_accel, axis=1, keepdims=True)
        accel_scale = np.minimum(1.0, max_acc[:, np.newaxis] / (accel_norm + 1e-8))
        constrained_accel = constrained_accel * accel_scale

        new_vel = current_vel + constrained_accel * dt

        vel_norm = np.linalg.norm(new_vel, axis=1, keepdims=True)
        max_allowed_vel = max_vel[:, np.newaxis] * 1.5
        comfort_vel = max_vel[:, np.newaxis] * 1.2

        vel_scale = np.where(
            vel_norm <= comfort_vel,
            1.0,
            np.minimum(1.0, max_allowed_vel / (vel_norm + 1e-8))
        )

        current_vel = new_vel * vel_scale

        current_pos = current_pos + current_vel[np.newaxis, :, :] * dt
        reconstructed[step] = current_pos[0, :, :]

    return reconstructed


def compute_metrics(pred, gt):
    errors = np.linalg.norm(pred - gt, axis=2)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ade = mae
    fde = float(np.mean(errors[-1]))

    true_distances = np.linalg.norm(gt, axis=2)
    epsilon = 1e-6
    valid_mask = true_distances > epsilon
    if np.any(valid_mask):
        mape = float(np.mean(np.abs(errors[valid_mask] / true_distances[valid_mask]) * 100.0))
    else:
        mape = 0.0

    mae_x = float(np.mean(np.abs(pred[..., 0] - gt[..., 0])))
    mae_y = float(np.mean(np.abs(pred[..., 1] - gt[..., 1])))
    mae_z = float(np.mean(np.abs(pred[..., 2] - gt[..., 2])))

    mae_per_step = np.mean(errors, axis=1).tolist()
    mae_per_agent = np.mean(errors, axis=0).tolist()
    fde_per_agent = errors[-1].tolist()
    rmse_per_agent = (np.sqrt(np.mean(errors ** 2, axis=0))).tolist()

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
    x_sample,
    device,
    data_scale,
    e_init_sig,
    e_prior_sig,
    e_l_steps,
    e_l_step_size,
    e_l_with_noise,
):
    num_agents = x_sample.shape[1]
    future_length = model.future_length
    pred_abs_all = np.zeros((future_length, num_agents, 3), dtype=np.float32)

    for agent_idx in range(num_agents):
        past_abs = x_sample[:, agent_idx, :]
        last_obs = past_abs[-1]
        past_rel = (past_abs - last_obs) * data_scale
        past_flat = torch.from_numpy(past_rel.reshape(1, -1)).to(device=device, dtype=torch.double)

        langevin_cfg = {
            "e_init_sig": e_init_sig,
            "e_prior_sig": e_prior_sig,
            "e_l_steps": e_l_steps,
            "e_l_step_size": e_l_step_size,
            "e_l_with_noise": e_l_with_noise,
        }

        try:
            plan_flat = model.sample_plan(past_flat, langevin_cfg)
            pred_rel = model.predict(past_flat, plan_flat).cpu().numpy()
            pred_abs_agent = pred_rel[0] / data_scale + last_obs
        except Exception as e:
            logger.warning(f"Failed to predict for agent {agent_idx}: {e}, using zero prediction")
            pred_abs_agent = np.zeros((future_length, 3), dtype=np.float32)

        pred_abs_all[:, agent_idx, :] = pred_abs_agent

    return pred_abs_all


def predict_exp5(
    model,
    x_sample,
    features,
    device,
    output_mean,
    output_std,
    feature_means,
    feature_stds,
    use_physical_constraints=True,
    pc_dt=0.1,
    pc_smoothing_weight=0.3,
    pc_constraint_relaxation=1.0,
):
    if feature_means is not None and feature_stds is not None:
        safe_std = np.where(feature_stds < 1e-8, 1.0, feature_stds)
        features_norm = (features - feature_means) / safe_std
        features_norm = np.clip(features_norm, -5.0, 5.0)
    else:
        features_norm = features

    x_batch = x_sample[np.newaxis, ...]
    f_batch = features_norm[np.newaxis, ...]

    model.eval()
    with torch.no_grad():
        features_t = torch.from_numpy(f_batch).float().to(device)
        x_orig_t = torch.from_numpy(x_batch).float().to(device)

        pred_delta_norm, _, _ = model(
            features_t, x_orig_t,
            y=None, y_velocity=None, y_accel=None,
            teacher_forcing_ratio=0.0
        )

        output_mean_t = torch.tensor(output_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        output_std_t = torch.tensor(output_std, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        pred_delta_phys = (pred_delta_norm * output_std_t + output_mean_t).cpu().numpy()
        pred_delta_phys = pred_delta_phys[0]

        if use_physical_constraints:
            pred_abs = apply_physical_constraints(
                x_sample,
                pred_delta_phys,
                dt=pc_dt,
                smoothing_weight=pc_smoothing_weight,
                constraint_relaxation=pc_constraint_relaxation
            )
        else:
            last_pos = x_batch[:, -1:, :, :]
            pred_abs = last_pos + np.cumsum(pred_delta_phys[np.newaxis, ...], axis=1)
            pred_abs = pred_abs[0]

    return pred_abs


def visualize_comparison(x_sample, y_sample, pred_lbebm, pred_exp5, sample_idx, output_path):
    num_agents = x_sample.shape[1]
    fig = plt.figure(figsize=(20, 13))

    colors = {
        "history": "b",
        "gt": "#27AE60",
        "lbebm": "#E74C3C",
        "exp5": "#E67E22",
    }

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    for aid in range(num_agents):
        if aid == 0:
            ax3d.plot(x_sample[:, aid, 0], x_sample[:, aid, 1], x_sample[:, aid, 2],
                      "b-o", linewidth=2.5, markersize=5, alpha=0.8, label="History")
        else:
            ax3d.plot(x_sample[:, aid, 0], x_sample[:, aid, 1], x_sample[:, aid, 2],
                      "b-o", linewidth=2.5, markersize=5, alpha=0.8)

        last = x_sample[-1:, aid, :]
        gt_traj = np.vstack([last, y_sample[:, aid, :]])
        lb_traj = np.vstack([last, pred_lbebm[:, aid, :]])
        ex_traj = np.vstack([last, pred_exp5[:, aid, :]])

        ax3d.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2],
                  "s-", color=colors["gt"], linewidth=2.8, markersize=7, alpha=0.9,
                  label="Ground Truth" if aid == 0 else "")
        ax3d.plot(lb_traj[:, 0], lb_traj[:, 1], lb_traj[:, 2],
                  "^--", color=colors["lbebm"], linewidth=2.5, markersize=6, alpha=0.85,
                  label="LBEBM3D" if aid == 0 else "")
        ax3d.plot(ex_traj[:, 0], ex_traj[:, 1], ex_traj[:, 2],
                  "D--", color=colors["exp5"], linewidth=2.5, markersize=6, alpha=0.85,
                  label="Exp5 Full" if aid == 0 else "")

    ax3d.set_xlabel("X (m)", fontsize=11, fontweight="bold")
    ax3d.set_ylabel("Y (m)", fontsize=11, fontweight="bold")
    ax3d.set_zlabel("Z (m)", fontsize=11, fontweight="bold")
    ax3d.set_title(f"Sample {sample_idx}: 3D Trajectories (LBEBM vs Exp5)", fontsize=12, fontweight="bold")
    ax3d.legend(fontsize=10, loc="upper left")
    ax3d.grid(True, alpha=0.3)

    def plot_2d(ax, ax1, ax2, title):
        for aid in range(num_agents):
            last = x_sample[-1:, aid, :]
            gt_traj = np.vstack([last, y_sample[:, aid, :]])
            lb_traj = np.vstack([last, pred_lbebm[:, aid, :]])
            ex_traj = np.vstack([last, pred_exp5[:, aid, :]])

            if aid == 0:
                ax.plot(x_sample[:, aid, ax1], x_sample[:, aid, ax2],
                        "b-o", linewidth=2.5, markersize=5, alpha=0.8, label="History")
            else:
                ax.plot(x_sample[:, aid, ax1], x_sample[:, aid, ax2],
                        "b-o", linewidth=2.5, markersize=5, alpha=0.8)

            ax.plot(gt_traj[:, ax1], gt_traj[:, ax2],
                    "s-", color=colors["gt"], linewidth=2.8, markersize=7, alpha=0.9,
                    label="True" if aid == 0 else "")
            ax.plot(lb_traj[:, ax1], lb_traj[:, ax2],
                    "^--", color=colors["lbebm"], linewidth=2.5, markersize=6, alpha=0.85,
                    label="LBEBM3D" if aid == 0 else "")
            ax.plot(ex_traj[:, ax1], ex_traj[:, ax2],
                    "D--", color=colors["exp5"], linewidth=2.5, markersize=6, alpha=0.85,
                    label="Exp5 Full" if aid == 0 else "")

        ax.set_xlabel(["X", "Y", "Z"][ax1] + " (m)", fontsize=11, fontweight="bold")
        ax.set_ylabel(["X", "Y", "Z"][ax2] + " (m)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plot_2d(fig.add_subplot(2, 3, 2), 0, 1, "XY Plane Projection")
    plot_2d(fig.add_subplot(2, 3, 3), 0, 2, "XZ Plane Projection")
    plot_2d(fig.add_subplot(2, 3, 4), 1, 2, "YZ Plane Projection")

    ax_err = fig.add_subplot(2, 3, 5)
    steps = np.arange(pred_lbebm.shape[0])
    err_x_lb = np.mean(np.abs(pred_lbebm[:, :, 0] - y_sample[:, :, 0]), axis=1)
    err_y_lb = np.mean(np.abs(pred_lbebm[:, :, 1] - y_sample[:, :, 1]), axis=1)
    err_z_lb = np.mean(np.abs(pred_lbebm[:, :, 2] - y_sample[:, :, 2]), axis=1)
    err_x_ex = np.abs(pred_exp5[:, :, 0] - y_sample[:, :, 0]).mean(axis=1)
    err_y_ex = np.abs(pred_exp5[:, :, 1] - y_sample[:, :, 1]).mean(axis=1)
    err_z_ex = np.abs(pred_exp5[:, :, 2] - y_sample[:, :, 2]).mean(axis=1)
    ax_err.plot(steps, err_x_lb, "r^-", linewidth=2.5, markersize=7, alpha=0.8, label="LBEBM |X|")
    ax_err.plot(steps, err_y_lb, "g^-", linewidth=2.5, markersize=7, alpha=0.8, label="LBEBM |Y|")
    ax_err.plot(steps, err_z_lb, "b^-", linewidth=2.5, markersize=7, alpha=0.8, label="LBEBM |Z|")
    ax_err.plot(steps, err_x_ex, "D-", color=colors["exp5"], linewidth=2.5, markersize=7, alpha=0.8, label="Exp5 |X|")
    ax_err.plot(steps, err_y_ex, "D--", color=colors["exp5"], linewidth=2.5, markersize=7, alpha=0.8, label="Exp5 |Y|")
    ax_err.plot(steps, err_z_ex, "D:", color=colors["exp5"], linewidth=2.5, markersize=7, alpha=0.8, label="Exp5 |Z|")
    ax_err.set_xlabel("Prediction Step", fontsize=11, fontweight="bold")
    ax_err.set_ylabel("Mean Absolute Error (m)", fontsize=11, fontweight="bold")
    ax_err.set_title("Per-Step Axis-wise Error", fontsize=12, fontweight="bold")
    ax_err.legend(fontsize=9, loc="best")
    ax_err.grid(True, alpha=0.3)

    ax_bar = fig.add_subplot(2, 3, 6)
    l2_lb = np.mean(np.linalg.norm(pred_lbebm - y_sample, axis=2), axis=1)
    l2_ex = np.mean(np.linalg.norm(pred_exp5 - y_sample, axis=2), axis=1)
    width = 0.35
    x = np.arange(len(steps))
    bars1 = ax_bar.bar(x - width / 2, l2_lb, width, color=colors["lbebm"], alpha=0.7,
                       edgecolor="darkred", linewidth=1.5, label="LBEBM3D")
    bars2 = ax_bar.bar(x + width / 2, l2_ex, width, color=colors["exp5"], alpha=0.7,
                       edgecolor="darkred", linewidth=1.5, label="Exp5 Full")
    for b in bars1:
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for b in bars2:
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
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
    parser = argparse.ArgumentParser(description='LBEBM3D vs Ablation Exp5 (DG32-BCAT) Comparison')

    parser.add_argument('--data_dir', required=True, help='Data directory')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--use_subset', action='store_true', help='Use subset data')

    parser.add_argument('--lbebm_model', required=True, help='LBEBM3D model path')
    parser.add_argument('--exp5_dir', required=True, help='Ablation exp5 results directory')

    parser.add_argument('--features_32d_dir', default='features_32d', help='32D features dir')

    parser.add_argument('--data_scale', type=float, default=1.0)
    parser.add_argument('--e_init_sig', type=float, default=2.0)
    parser.add_argument('--e_prior_sig', type=float, default=2.0)
    parser.add_argument('--e_l_steps', type=int, default=20)
    parser.add_argument('--e_l_step_size', type=float, default=0.4)
    parser.add_argument('--e_l_with_noise', action='store_true')

    parser.add_argument('--no_physical_constraints', action='store_true',
                        help='Disable physical constraints for Exp5 reconstruction')
    parser.add_argument('--pc_dt', type=float, default=0.1, help='dt for physical constraints')
    parser.add_argument('--pc_smoothing_weight', type=float, default=0.3, help='smoothing weight for physical constraints')
    parser.add_argument('--pc_constraint_relaxation', type=float, default=1.0, help='constraint relaxation factor')

    parser.add_argument('--remove_exp5_outliers', action='store_true',
                        help='Remove a fraction of worst Exp5 samples (by metric) from both models for robust stats')
    parser.add_argument('--exp5_outlier_metric', type=str, default='MAE',
                        choices=['MAE', 'FDE'],
                        help='Metric used to detect Exp5 outliers (default: MAE)')
    parser.add_argument('--exp5_outlier_percent', type=float, default=1.0,
                        help='Percentage of worst Exp5 samples to drop (0-50, default: 1.0)')

    parser.add_argument('--sample_indices', type=str, default=None, help='Comma-separated sample indices')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of random samples')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--use_val_split', action='store_true',
                        help='Select samples from validation split (same as infer_and_visualize_ablation.py)')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')

    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable per-sample visualization for faster evaluation')

    parser.add_argument('--output_dir', default='comparison_exp5_vs_lbebm', help='Output directory')

    args = parser.parse_args()

    if not LBEBM_AVAILABLE:
        logger.error("❌ LBEBM3D 模块无法导入，无法进行对比")
        logger.error(f"   确认文件存在: {tool_dir / 'infer_lbebm3d_baseline.py'}")
        sys.exit(1)

    if not EXP5_AVAILABLE:
        logger.error("❌ Exp5 模型模块无法导入")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === Load Data ===
    logger.info(f"Loading data from: {args.data_dir}")
    x_all, y_all = load_data_ablation(args.data_dir, args.agents, use_subset=args.use_subset)
    logger.info(f"Data shape: X={x_all.shape}, Y={y_all.shape}")

    # === Load LBEBM Model ===
    logger.info(f"Loading LBEBM3D model: {args.lbebm_model}")
    ckpt = torch.load(args.lbebm_model, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    params = infer_model_params_from_state_dict(state_dict)
    sub_goal_indexes = [2, 5, 7, 9] if params['future_length'] >= 10 else \
        list(np.linspace(0, params['future_length'] - 1, params['num_subgoals'], dtype=int))

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

    # === Load Exp5 Model ===
    exp5_dir = Path(args.exp5_dir)
    config_path = exp5_dir / f"config_agents_{args.agents}_exp5_full.json"
    stats_path = exp5_dir / f"stats_agents_{args.agents}_exp5_full.npz"
    model_path = exp5_dir / f"best_model_agents_{args.agents}_exp5_full.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"统计文件不存在: {stats_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    stats = np.load(stats_path)
    exp5_output_mean = stats['output_mean']
    exp5_output_std = stats['output_std']

    logger.info(f"Loading Exp5 model: {model_path}")
    exp5_ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    exp5_model = DynamicsAwareSwarmGRUModel_with_GNN(
        input_size=config.get('input_features', 32),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 3),
        output_size=3,
        dropout=0.3,
        use_attention=True,
        gnn_hidden=config.get('gnn_hidden', 64),
        num_gnn_heads=config.get('gnn_heads', 4),
        edge_threshold=config.get('edge_threshold', 5.0),
        fusion_mode='concat',
    )
    exp5_model.load_state_dict(exp5_ckpt['model_state_dict'])
    exp5_model.to(device)
    exp5_model.eval()
    logger.info("Exp5 model loaded")

    # === Load 32D Features ===
    logger.info("Loading 32D features for Exp5...")
    features_all, feature_means, feature_stds = load_32d_features_with_stats(
        args.features_32d_dir, args.agents, args.use_subset
    )
    if features_all is None:
        raise RuntimeError("Failed to load 32D features")
    logger.info(f"Features shape: {features_all.shape}")

    # Align sample counts if needed
    total_samples = min(len(x_all), len(features_all))
    if len(x_all) != len(features_all):
        logger.warning(f"Sample count mismatch: X={len(x_all)}, features={len(features_all)}. Using {total_samples} samples.")
        x_all = x_all[:total_samples]
        y_all = y_all[:total_samples]
        features_all = features_all[:total_samples]

    # === Optional: use validation split like infer_and_visualize_ablation.py ===
    val_indices = None
    if args.use_val_split:
        indices = np.arange(total_samples)
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        num_val = max(1, int(total_samples * args.val_split)) if args.val_split > 0 else 0
        if num_val <= 0:
            logger.warning("val_split <= 0, skipping validation split.")
        else:
            val_indices = indices[:num_val]
            x_all = x_all[val_indices]
            y_all = y_all[val_indices]
            features_all = features_all[val_indices]
            total_samples = len(x_all)
            logger.info(f"Using validation split: {total_samples} samples")

    # === Select Samples ===
    np.random.seed(args.seed)
    if args.sample_indices:
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(',')]
    else:
        num_samples = min(args.num_samples, total_samples)
        sample_indices = np.random.choice(total_samples, num_samples, replace=False).tolist()
        sample_indices = sorted(sample_indices)

    if val_indices is not None:
        sample_index_labels = [int(val_indices[i]) for i in sample_indices]
    else:
        sample_index_labels = sample_indices

    logger.info(f"Selected samples: {sample_index_labels}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Run Comparison ===
    all_metrics_lbebm = []
    all_metrics_exp5 = []

    for i, sample_idx in enumerate(sample_indices):
        logger.info(f"\n=== Processing sample {sample_idx} ===")

        x_sample = x_all[sample_idx]
        y_sample = y_all[sample_idx]
        features_sample = features_all[sample_idx]

        logger.info("Running LBEBM3D prediction...")
        pred_lbebm = predict_lbebm_multi_agent(
            lbebm_model, x_sample, device, args.data_scale,
            args.e_init_sig, args.e_prior_sig, args.e_l_steps,
            args.e_l_step_size, args.e_l_with_noise
        )

        logger.info("Running Exp5 prediction...")
        pred_exp5 = predict_exp5(
            exp5_model, x_sample, features_sample, device,
            exp5_output_mean, exp5_output_std,
            feature_means, feature_stds,
            use_physical_constraints=not args.no_physical_constraints,
            pc_dt=args.pc_dt,
            pc_smoothing_weight=args.pc_smoothing_weight,
            pc_constraint_relaxation=args.pc_constraint_relaxation,
        )

        metrics_lbebm = compute_metrics(pred_lbebm, y_sample)
        metrics_exp5 = compute_metrics(pred_exp5, y_sample)

        logger.info(f"LBEBM3D - ADE: {metrics_lbebm['ADE']:.4f}, FDE: {metrics_lbebm['FDE']:.4f}")
        logger.info(f"Exp5 Full - ADE: {metrics_exp5['ADE']:.4f}, FDE: {metrics_exp5['FDE']:.4f}")

        all_metrics_lbebm.append(metrics_lbebm)
        all_metrics_exp5.append(metrics_exp5)

        if not args.no_visualize:
            output_label = sample_index_labels[i]
            output_path = output_dir / f"sample_{output_label}_comparison.png"
            visualize_comparison(
                x_sample, y_sample, pred_lbebm, pred_exp5,
                sample_idx, output_path
            )

    # === Optional: remove extreme Exp5 outliers (and corresponding LBEBM samples) ===
    used_metrics_lbebm = all_metrics_lbebm
    used_metrics_exp5 = all_metrics_exp5
    used_sample_indices = sample_index_labels
    removed_sample_indices = []

    if args.remove_exp5_outliers:
        metric_name = args.exp5_outlier_metric
        if not all_metrics_exp5 or metric_name not in all_metrics_exp5[0]:
            logger.warning(f"Outlier removal requested, but metric '{metric_name}' not found. Skipping outlier removal.")
        else:
            perc = max(0.0, min(args.exp5_outlier_percent, 50.0))
            if perc <= 0.0:
                logger.info("Outlier percentage <= 0, skipping outlier removal.")
            else:
                exp5_values = np.array([m[metric_name] for m in all_metrics_exp5], dtype=np.float32)
                n = len(exp5_values)
                k = int(np.ceil(n * perc / 100.0))
                if k <= 0:
                    logger.info("Computed outlier count k=0, skipping outlier removal.")
                elif k >= n:
                    logger.warning("Outlier count >= sample count, skipping outlier removal.")
                else:
                    threshold = np.partition(exp5_values, n - k - 1)[n - k - 1]
                    keep_mask = exp5_values <= threshold

                    used_metrics_lbebm = [m for m, keep in zip(all_metrics_lbebm, keep_mask) if keep]
                    used_metrics_exp5 = [m for m, keep in zip(all_metrics_exp5, keep_mask) if keep]
                    removed_sample_indices = [idx for idx, keep in zip(sample_index_labels, keep_mask) if not keep]
                    used_sample_indices = [idx for idx, keep in zip(sample_index_labels, keep_mask) if keep]

                    logger.info(
                        f"Outlier removal enabled: metric={metric_name}, "
                        f"percent={perc}%, removed={len(removed_sample_indices)}/{n} samples."
                    )
                    if removed_sample_indices:
                        logger.info(f"Removed sample indices (Exp5 worst cases): {removed_sample_indices}")
                    logger.info(f"Remaining samples for statistics: {len(used_sample_indices)}")

    # === Aggregate Results ===
    logger.info("\n=== Overall Results ===")

    def compute_aggregate_stats(metrics_list):
        stats = {}
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

        if metrics_list and 'MAE_per_step' in metrics_list[0]:
            per_step_values = np.array([m['MAE_per_step'] for m in metrics_list])
            stats['MAE_per_step_avg'] = float(np.mean(per_step_values))
            stats['MAE_per_step_std'] = float(np.std(per_step_values))

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
    exp5_aggregate = compute_aggregate_stats(used_metrics_exp5)

    logger.info("\n✓ LBEBM3D Statistics:")
    logger.info(f"  ADE: {lbebm_aggregate['ADE']['mean']:.4f} ± {lbebm_aggregate['ADE']['std']:.4f}")
    logger.info(f"  FDE: {lbebm_aggregate['FDE']['mean']:.4f} ± {lbebm_aggregate['FDE']['std']:.4f}")
    logger.info(f"  RMSE: {lbebm_aggregate['RMSE']['mean']:.4f} ± {lbebm_aggregate['RMSE']['std']:.4f}")
    logger.info(f"  MAPE: {lbebm_aggregate['MAPE']['mean']:.2f}% ± {lbebm_aggregate['MAPE']['std']:.2f}%")

    logger.info("\n✓ Exp5 Full Statistics:")
    logger.info(f"  ADE: {exp5_aggregate['ADE']['mean']:.4f} ± {exp5_aggregate['ADE']['std']:.4f}")
    logger.info(f"  FDE: {exp5_aggregate['FDE']['mean']:.4f} ± {exp5_aggregate['FDE']['std']:.4f}")
    logger.info(f"  RMSE: {exp5_aggregate['RMSE']['mean']:.4f} ± {exp5_aggregate['RMSE']['std']:.4f}")
    logger.info(f"  MAPE: {exp5_aggregate['MAPE']['mean']:.2f}% ± {exp5_aggregate['MAPE']['std']:.2f}%")

    ade_improve = ((lbebm_aggregate['ADE']['mean'] - exp5_aggregate['ADE']['mean']) / lbebm_aggregate['ADE']['mean'] * 100)
    fde_improve = ((lbebm_aggregate['FDE']['mean'] - exp5_aggregate['FDE']['mean']) / lbebm_aggregate['FDE']['mean'] * 100)
    logger.info("\n✓ Performance Comparison (improvement %):")
    logger.info(f"  ADE improvement: {ade_improve:+.2f}% (Exp5 better)" if ade_improve > 0 else f"  ADE improvement: {ade_improve:+.2f}% (LBEBM better)")
    logger.info(f"  FDE improvement: {fde_improve:+.2f}% (Exp5 better)" if fde_improve > 0 else f"  FDE improvement: {fde_improve:+.2f}% (LBEBM better)")

    # ===== v4-style comprehensive summary (2 models) =====
    print("\n" + "=" * 80)
    print("Inference Complete - Comprehensive Statistics Summary")
    print("=" * 80 + "\n")

    exp_names = {
        "LBEBM3D": "LBEBM3D",
        "Exp5_Full": "Exp5: Full Model (32D)",
    }

    def extract_aggregate_stats(metrics_list):
        if not metrics_list:
            return {}

        key_metrics = [
            'MAE', 'RMSE', 'ADE', 'FDE', 'MAPE', 'MAE_X', 'MAE_Y', 'MAE_Z',
            'Velocity_MAE', 'Velocity_RMSE', 'Speed_error_MAE'
        ]

        stats = {}
        for metric in key_metrics:
            values = [m[metric] for m in metrics_list if metric in m]
            if values:
                stats[f'{metric}_mean'] = float(np.mean(values))
                stats[f'{metric}_std'] = float(np.std(values))
                stats[f'{metric}_min'] = float(np.min(values))
                stats[f'{metric}_max'] = float(np.max(values))

        mae_per_step_all = [m['MAE_per_step'] for m in metrics_list if 'MAE_per_step' in m and m['MAE_per_step']]
        if mae_per_step_all:
            try:
                values_array = np.array(mae_per_step_all)
                stats['MAE_per_step_mean'] = values_array.mean(axis=0).tolist()
                stats['MAE_per_step_std'] = values_array.std(axis=0).tolist()
            except Exception:
                pass

        mae_per_agent_all = [m['MAE_per_agent'] for m in metrics_list if 'MAE_per_agent' in m and m['MAE_per_agent']]
        if mae_per_agent_all:
            try:
                values_array = np.array(mae_per_agent_all)
                stats['MAE_per_agent_mean'] = values_array.mean(axis=0).tolist()
                stats['MAE_per_agent_std'] = values_array.std(axis=0).tolist()
            except Exception:
                pass

        return stats

    summary_v4 = {
        'num_samples': len(used_sample_indices),
        'configuration': {
            'pc_dt': args.pc_dt,
            'pc_smoothing_weight': args.pc_smoothing_weight,
            'pc_constraint_relaxation': args.pc_constraint_relaxation,
            'use_subset': args.use_subset,
            'use_val_split': args.use_val_split,
            'val_split': args.val_split,
            'outlier_filter': {
                'enabled': bool(args.remove_exp5_outliers),
                'metric': args.exp5_outlier_metric,
                'percent': args.exp5_outlier_percent,
                'removed_sample_indices': removed_sample_indices,
            },
        },
        'experiments': {}
    }

    summary_v4['experiments']['LBEBM3D'] = {
        'name': exp_names['LBEBM3D'],
        'num_samples': len(used_metrics_lbebm),
        'aggregate_stats': extract_aggregate_stats(used_metrics_lbebm)
    }
    summary_v4['experiments']['Exp5_Full'] = {
        'name': exp_names['Exp5_Full'],
        'num_samples': len(used_metrics_exp5),
        'aggregate_stats': extract_aggregate_stats(used_metrics_exp5)
    }

    print("📊 PERFORMANCE SUMMARY\n")
    print("=" * 90)
    print(f"{'Experiment':<30} {'MAE':<10} {'RMSE':<10} {'FDE':<10} {'MAPE':<10} {'Vel_MAE':<10}")
    print("=" * 90)

    for key in ['LBEBM3D', 'Exp5_Full']:
        stats = summary_v4['experiments'][key]['aggregate_stats']
        name = exp_names[key][:28]
        mae = stats.get('MAE_mean', 0)
        rmse = stats.get('RMSE_mean', 0)
        fde = stats.get('FDE_mean', 0)
        mape = stats.get('MAPE_mean', 0)
        vel_mae = stats.get('Velocity_MAE_mean', 0)
        print(f"{name:<30} {mae:<10.4f} {rmse:<10.4f} {fde:<10.4f} {mape:<10.4f} {vel_mae:<10.4f}")

    print("=" * 90)

    # Save streamlined summary as JSON (optimized for plotting)
    summary_v4_file = output_dir / 'ablation_summary.json'
    with open(summary_v4_file, 'w', encoding='utf-8') as f:
        json.dump(summary_v4, f, indent=2, ensure_ascii=False)

    # Save CSV format for easy analysis
    csv_v4_file = output_dir / 'ablation_results.csv'
    with open(csv_v4_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'experiment_name', 'MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std',
            'ADE_mean', 'ADE_std', 'FDE_mean', 'FDE_std', 'MAPE_mean', 'MAPE_std',
            'MAE_X_mean', 'MAE_Y_mean', 'MAE_Z_mean', 'Velocity_MAE_mean', 'Speed_error_MAE_mean',
            'Min_mean', 'Median_mean', 'Max_mean', 'P95_mean'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in ['LBEBM3D', 'Exp5_Full']:
            stats = summary_v4['experiments'][key]['aggregate_stats']
            row = {
                'experiment_name': exp_names[key],
                'MAE_mean': stats.get('MAE_mean', 0),
                'MAE_std': stats.get('MAE_std', 0),
                'RMSE_mean': stats.get('RMSE_mean', 0),
                'RMSE_std': stats.get('RMSE_std', 0),
                'ADE_mean': stats.get('ADE_mean', 0),
                'ADE_std': stats.get('ADE_std', 0),
                'FDE_mean': stats.get('FDE_mean', 0),
                'FDE_std': stats.get('FDE_std', 0),
                'MAPE_mean': stats.get('MAPE_mean', 0),
                'MAPE_std': stats.get('MAPE_std', 0),
                'MAE_X_mean': stats.get('MAE_X_mean', 0),
                'MAE_Y_mean': stats.get('MAE_Y_mean', 0),
                'MAE_Z_mean': stats.get('MAE_Z_mean', 0),
                'Velocity_MAE_mean': stats.get('Velocity_MAE_mean', 0),
                'Speed_error_MAE_mean': stats.get('Speed_error_MAE_mean', 0),
                'Min_mean': stats.get('Min_mean', 0),
                'Median_mean': stats.get('Median_mean', 0),
                'Max_mean': stats.get('Max_mean', 0),
                'P95_mean': stats.get('P95_mean', 0),
            }
            writer.writerow(row)

    logger.info(f"v4-style summary saved: {summary_v4_file}")
    logger.info(f"v4-style CSV saved: {csv_v4_file}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(used_metrics_lbebm),
        "sample_indices": used_sample_indices,
        "removed_sample_indices": removed_sample_indices,
        "outlier_filter": {
            "enabled": bool(args.remove_exp5_outliers),
            "metric": args.exp5_outlier_metric,
            "percent": args.exp5_outlier_percent,
        },
        "LBEBM3D": {
            "aggregate_stats": lbebm_aggregate,
            "all_metrics": used_metrics_lbebm,
        },
        "Exp5_Full": {
            "aggregate_stats": exp5_aggregate,
            "all_metrics": used_metrics_exp5,
            "config": config,
        },
    }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save aggregate CSV for paper plotting
    csv_summary_path = output_dir / "comparison_summary.csv"
    with open(csv_summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Metric", "Mean", "Std", "Min", "Max", "Median"])
        for model_name, agg in [("LBEBM3D", lbebm_aggregate), ("Exp5_Full", exp5_aggregate)]:
            for metric, stats in agg.items():
                if isinstance(stats, dict) and all(k in stats for k in ("mean", "std", "min", "max", "median")):
                    writer.writerow([
                        model_name,
                        metric,
                        stats["mean"],
                        stats["std"],
                        stats["min"],
                        stats["max"],
                        stats["median"],
                    ])

    # Save per-sample metrics CSV
    csv_samples_path = output_dir / "comparison_samples.csv"
    with open(csv_samples_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx",
            "model",
            "MAE",
            "RMSE",
            "ADE",
            "FDE",
            "MAPE",
            "MAE_X",
            "MAE_Y",
            "MAE_Z",
        ])
        for idx, (m_lbebm, m_exp5) in zip(used_sample_indices, zip(used_metrics_lbebm, used_metrics_exp5)):
            writer.writerow([
                idx, "LBEBM3D",
                m_lbebm.get("MAE"), m_lbebm.get("RMSE"), m_lbebm.get("ADE"),
                m_lbebm.get("FDE"), m_lbebm.get("MAPE"),
                m_lbebm.get("MAE_X"), m_lbebm.get("MAE_Y"), m_lbebm.get("MAE_Z"),
            ])
            writer.writerow([
                idx, "Exp5_Full",
                m_exp5.get("MAE"), m_exp5.get("RMSE"), m_exp5.get("ADE"),
                m_exp5.get("FDE"), m_exp5.get("MAPE"),
                m_exp5.get("MAE_X"), m_exp5.get("MAE_Y"), m_exp5.get("MAE_Z"),
            ])

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"CSV summary saved to: {csv_summary_path}")
    logger.info(f"Per-sample CSV saved to: {csv_samples_path}")
    logger.info(f"All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
