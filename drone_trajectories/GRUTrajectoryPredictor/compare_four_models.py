#!/usr/bin/env python3
"""
四模型对比脚本: 3DMoTraj (LBEBM3D) vs DG32-BCAT (Exp5) vs MRGTraj vs SwarmGRU v3
=================================================================================

输出：
  - comparison_four_models/sample_xxx_comparison_v4.png
  - comparison_four_models/comparison_summary.json
  - comparison_four_models/comparison_summary.csv
  - comparison_four_models/comparison_samples.csv
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
workspace_root = Path("D:\\Trajectory prediction")
cluster_traj_dir = workspace_root / "drone_trajectories" / "Cluster trajectory"
tool_dir = workspace_root / "drone_trajectories" / "3DMoTraj" / "tool"
mrgraj_dir = workspace_root / "drone_trajectories" / "MRGTraj-main"

ablation_study_dir = cluster_traj_dir / "ablation study"

sys.path.insert(0, str(cluster_traj_dir))
sys.path.insert(0, str(ablation_study_dir))
sys.path.insert(0, str(tool_dir))
sys.path.insert(0, str(mrgraj_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Import LBEBM3D
LBEBM_AVAILABLE = False
LBEBM3DInfer = None
infer_model_params_from_state_dict = None
try:
    from infer_lbebm3d_baseline import LBEBM3DInfer, infer_model_params_from_state_dict
    LBEBM_AVAILABLE = True
    logger.info("✓ LBEBM3D 导入成功")
except Exception as e:
    logger.warning(f"✗ LBEBM3D 导入失败 (标准方式): {e}")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "infer_lbebm3d_baseline", str(tool_dir / "infer_lbebm3d_baseline.py")
        )
        if spec and spec.loader:
            lbebm_module = importlib.util.module_from_spec(spec)
            sys.modules["infer_lbebm3d_baseline"] = lbebm_module
            spec.loader.exec_module(lbebm_module)
            LBEBM3DInfer = lbebm_module.LBEBM3DInfer
            infer_model_params_from_state_dict = lbebm_module.infer_model_params_from_state_dict
            LBEBM_AVAILABLE = True
            logger.info("✓ LBEBM3D 导入成功 (动态加载)")
    except Exception as e2:
        logger.error(f"✗ LBEBM3D 导入失败 (动态加载): {e2}")

# Import Exp5 model (v3 with GNN)
EXP5_AVAILABLE = False
DynamicsAwareSwarmGRUModel_with_GNN = None
try:
    from train_swarm_model_v3_with_gnn import DynamicsAwareSwarmGRUModel_with_GNN
    EXP5_AVAILABLE = True
    logger.info("✓ Exp5 (DG32-BCAT) 导入成功")
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"✗ Exp5 导入失败 (标准方式): {e}")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_swarm_model_v3_with_gnn",
            str(cluster_traj_dir / "train_swarm_model_v3_with_gnn.py")
        )
        if spec and spec.loader:
            exp5_module = importlib.util.module_from_spec(spec)
            sys.modules["train_swarm_model_v3_with_gnn"] = exp5_module
            spec.loader.exec_module(exp5_module)
            DynamicsAwareSwarmGRUModel_with_GNN = exp5_module.DynamicsAwareSwarmGRUModel_with_GNN
            EXP5_AVAILABLE = True
            logger.info("✓ Exp5 (DG32-BCAT) 导入成功 (动态加载)")
    except Exception as e2:
        logger.error(f"✗ Exp5 导入失败 (动态加载): {e2}")

# Import AblationDataset — 与消融实验训练管道完全一致的特征归一化
ABLATION_DATASET_AVAILABLE = False
AblationDataset = None
try:
    from ablation_train_utils import AblationDataset
    ABLATION_DATASET_AVAILABLE = True
    logger.info("✓ AblationDataset 导入成功")
except Exception as e:
    logger.warning(f"✗ AblationDataset 导入失败，将使用备用归一化方式: {e}")

# Import MRGTraj
MRGRAJ_AVAILABLE = False
try:
    from model_swarm import MRGTrajSwarm
    MRGRAJ_AVAILABLE = True
    logger.info("✓ MRGTraj 导入成功")
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"✗ MRGTraj 导入失败: {e}")

plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 2.2
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
try:
    plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
except Exception:
    plt.rcParams['font.family'] = 'DejaVu Sans'

PAPER_STYLE = {
    "palette": {
        "history": "#020FFDFF",
        "gt": "#07C054A7",
        "mrgraj": "#EC0C0C",
        "3dmotraj": "#C508FEFD",
        "exp5": "#00C3F9FF",
        "swarm_gru": "#8A2BE2",
    },
    "linestyles": {
        "history": "-",
        "gt": "-",
        "mrgraj": "--",
        "3dmotraj": "-.",
        "exp5": ":",
        "swarm_gru": (0, (3, 1, 1, 1)),
    },
    "markers": {
        "history": ".",
        "gt": "s",
        "mrgraj": "o",
        "3dmotraj": "^",
        "exp5": "D",
        "swarm_gru": "v",
    },
    "linewidth": 3.0,
    "markersize": 6,
}


def load_data_ablation(data_dir: str, num_agents: int, use_subset: bool = False):
    """加载无人机轨迹数据"""
    data_path = Path(data_dir)
    subset_suffix = '_subset' if use_subset else ''
    x_file = data_path / f'input_agents_{num_agents}{subset_suffix}.npz'
    y_file = data_path / f'output_agents_{num_agents}{subset_suffix}.npz'

    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {x_file}, {y_file}")

    x = np.load(x_file)['data']
    y = np.load(y_file)['data']

    x = np.transpose(x, (1, 0, 2, 3))
    y = np.transpose(y, (1, 0, 2, 3))

    return x, y


def load_32d_features_with_stats(features_dir, num_agents, use_subset=False):
    """加载32D特征"""
    features_dir = Path(features_dir)
    subset_suffix = '_subset' if use_subset else ''

    candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_32d.npz',
        features_dir / f'features_agents_{num_agents}_32d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
        features_dir / f'features_agents_{num_agents}_32d.npz',
    ]

    for path in candidates:
        if path.exists():
            logger.info(f"加载特征: {path}")
            data = np.load(path)
            features = data['features']
            feature_means = data.get('feature_means', None)
            feature_stds = data.get('feature_stds', None)
            if feature_means is None and 'means' in data:
                feature_means = data.get('means', None)
            if feature_stds is None and 'stds' in data:
                feature_stds = data.get('stds', None)
            if feature_means is not None:
                feature_means = feature_means.astype(np.float32)
            if feature_stds is not None:
                feature_stds = feature_stds.astype(np.float32)
            if feature_means is None or feature_stds is None:
                feature_dim = features.shape[-1]
                flat_features = features.reshape(-1, feature_dim).astype(np.float32)
                feature_means = np.mean(flat_features, axis=0).astype(np.float32)
                feature_stds = np.std(flat_features, axis=0).astype(np.float32)
                feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
                logger.info("32D feature stats missing in file, fallback to computed means/stds")
            return features.astype(np.float32), feature_means, feature_stds

    logger.warning(f"特征文件未找到，候选位置: {candidates}")
    return None, None, None


def apply_physical_constraints(history, pred_delta, dt=0.1, smoothing_weight=0.2, constraint_relaxation=1.0):
    """Apply physical constraints with velocity-aware reconstruction."""
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


class SwarmGRUModel(torch.nn.Module):
    """GRU encoder-decoder for swarm trajectory prediction (v3)"""

    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3,
                 num_layers=2, dropout=0.3, num_agents=3, seq_out=10):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_agents = num_agents
        self.seq_out = seq_out

        self.encoder = torch.nn.GRU(
            input_size=input_dim * num_agents,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.decoder = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = torch.nn.Linear(hidden_dim, input_dim * num_agents)

    def forward(self, x):
        batch_size = x.size(0)
        seq_in = x.size(1)

        x_flat = x.view(batch_size, seq_in, -1)
        _, h_n = self.encoder(x_flat)

        decoder_in = torch.zeros(batch_size, self.seq_out, self.hidden_dim,
                                device=x.device, dtype=x.dtype)

        decoder_out, _ = self.decoder(decoder_in, h_n)
        y_flat = self.fc(decoder_out)
        y = y_flat.view(batch_size, self.seq_out, self.num_agents, self.output_dim)

        return y


def load_swarm_gru_model(model_path, device):
    """Load SwarmGRU v3 model and stats"""
    if not Path(model_path).exists():
        logger.error(f"SwarmGRU 模型不存在: {model_path}")
        return None, None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    args = checkpoint.get('args', {})
    stats = checkpoint.get('dataset_stats', None)

    if not stats:
        logger.error("SwarmGRU checkpoint 缺少 dataset_stats")
        return None, None

    seq_out = 10
    model = SwarmGRUModel(
        input_dim=3,
        hidden_dim=args.get('hidden_dim', 64),
        output_dim=3,
        num_layers=args.get('num_layers', 2),
        dropout=args.get('dropout', 0.3),
        num_agents=args.get('num_agents', 3),
        seq_out=seq_out
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, stats


def predict_swarm_gru(model, x_sample, stats, device):
    """SwarmGRU v3 预测 (输出绝对坐标)"""
    input_mean = np.array(stats['input_mean'], dtype=np.float32)
    input_std = np.array(stats['input_std'], dtype=np.float32)
    output_mean = np.array(stats['output_mean'], dtype=np.float32)
    output_std = np.array(stats['output_std'], dtype=np.float32)

    x_norm = (x_sample - input_mean) / input_std
    x_batch = torch.from_numpy(x_norm[np.newaxis, ...]).float().to(device)

    with torch.no_grad():
        pred_delta_norm = model(x_batch).cpu().numpy()[0]

    pred_delta = pred_delta_norm * output_std + output_mean
    last_pos = x_sample[-1:, :, :]
    pred_abs = last_pos + pred_delta

    return pred_abs


def compute_metrics(pred, gt):
    """计算预测指标 (MAE, RMSE, ADE, FDE, MAPE)"""
    errors = np.linalg.norm(pred - gt, axis=2)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ade = mae
    fde = float(np.mean(errors[-1]))

    true_distances = np.linalg.norm(gt, axis=2)
    epsilon = 1e-6
    valid_mask = true_distances > epsilon
    if np.any(valid_mask):
        mape = float(np.mean(errors[valid_mask] / true_distances[valid_mask]) * 100.0)
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


def predict_lbebm_multi_agent(model, x_sample, device, data_scale, e_init_sig, e_prior_sig, e_l_steps, e_l_step_size, e_l_with_noise):
    """LBEBM3D 多智能体预测"""
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
            logger.warning(f"LBEBM3D 预测 agent {agent_idx} 失败: {e}")
            pred_abs_agent = np.zeros((future_length, 3), dtype=np.float32)

        pred_abs_all[:, agent_idx, :] = pred_abs_agent

    return pred_abs_all


def predict_exp5(model, x_sample, features, device, output_mean, output_std, feature_means, feature_stds, use_physical_constraints=True, pc_dt=0.1, pc_smoothing_weight=0.3, pc_constraint_relaxation=1.0):
    """Exp5 (DG32-BCAT) 多智能体预测"""
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
            last_pos = x_sample[-1:, :, :]
            pred_abs = last_pos + np.cumsum(pred_delta_phys, axis=0)

    return pred_abs


def predict_mrgraj(model, x_sample, device, use_physical_constraints=True, pc_dt=0.1, pc_smoothing_weight=0.3, pc_constraint_relaxation=1.0):
    """MRGTraj 多智能体预测"""
    x_batch = torch.from_numpy(x_sample[np.newaxis, ...]).float().to(device)

    model.eval()
    with torch.no_grad():
        pred_traj, _, _ = model(x_batch)
        pred_output = pred_traj.squeeze(0).cpu().numpy().astype(np.float32)

    last_obs = x_sample[-1:, :, :]
    first_pred = pred_output[0:1, :, :]
    distance_from_last = np.mean(np.linalg.norm(first_pred - last_obs, axis=2))

    if distance_from_last > 1.0:
        pred_delta = np.diff(np.vstack([last_obs, pred_output]), axis=0)
        if use_physical_constraints:
            pred_delta_cum = np.cumsum(pred_delta, axis=0)
            pred_abs = apply_physical_constraints(
                x_sample,
                pred_delta_cum,
                dt=pc_dt,
                smoothing_weight=pc_smoothing_weight,
                constraint_relaxation=pc_constraint_relaxation
            )
        else:
            pred_abs = last_obs + np.cumsum(pred_delta, axis=0)
    else:
        if use_physical_constraints:
            pred_delta = np.diff(np.vstack([last_obs, pred_output]), axis=0)
            pred_delta_cum = np.cumsum(pred_delta, axis=0)
            pred_abs = apply_physical_constraints(
                x_sample,
                pred_delta_cum,
                dt=pc_dt,
                smoothing_weight=pc_smoothing_weight,
                constraint_relaxation=pc_constraint_relaxation
            )
        else:
            pred_abs = pred_output

    return pred_abs


def visualize_comparison(x_sample, y_sample, pred_lbebm, pred_exp5, pred_mrgraj, pred_swarm_gru, sample_idx, output_path):
    """绘制四模型对比可视化"""
    num_agents = x_sample.shape[1]

    colors = PAPER_STYLE["palette"]
    ls = PAPER_STYLE["linestyles"]
    mk = PAPER_STYLE["markers"]
    lw = PAPER_STYLE["linewidth"]
    ms = PAPER_STYLE["markersize"]

    fig = plt.figure(figsize=(30, 16), facecolor='white')
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.95, top=0.93, bottom=0.08,
                          hspace=0.35, wspace=0.30)
    fig.suptitle(f'Four-Model Comparison - Sample #{sample_idx} | {num_agents} Agents',
                 fontsize=16, fontweight='bold', y=0.97)

    models = [
        ('MRGTraj', pred_mrgraj, 'mrgraj'),
        ('3DMoTraj', pred_lbebm, '3dmotraj'),
        ('DG32-BCAT(Ours)', pred_exp5, 'exp5'),
        ('SwarmGRU v3', pred_swarm_gru, 'swarm_gru'),
    ]

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    for aid in range(num_agents):
        ax3d.plot(x_sample[:, aid, 0], x_sample[:, aid, 1], x_sample[:, aid, 2],
                  color=colors['history'], linestyle=ls['history'], linewidth=lw, alpha=0.8,
                  label='History' if aid == 0 else '', zorder=1)
        ax3d.scatter(x_sample[:, aid, 0], x_sample[:, aid, 1], x_sample[:, aid, 2],
                     color=colors['history'], marker=mk['history'], s=ms*8, alpha=0.6,
                     edgecolors='white', linewidths=0.5, zorder=1)

        last_pt = x_sample[-1:, aid, :]
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        ax3d.plot(gt_full[:, 0], gt_full[:, 1], gt_full[:, 2],
                  color=colors['gt'], linestyle=ls['gt'], linewidth=lw, alpha=0.95,
                  label='Ground Truth' if aid == 0 else '', zorder=10)
        ax3d.scatter(gt_full[:, 0], gt_full[:, 1], gt_full[:, 2],
                     color=colors['gt'], marker=mk['gt'], s=ms*8, alpha=0.75,
                     edgecolors='white', linewidths=0.5, zorder=10)

        for model_name, pred, color_key in models:
            pred_full = np.vstack([last_pt, pred[:, aid, :]])
            ax3d.plot(pred_full[:, 0], pred_full[:, 1], pred_full[:, 2],
                      color=colors[color_key], linestyle=ls[color_key], linewidth=lw, alpha=0.85,
                      label=model_name if aid == 0 else '', zorder=5)
            ax3d.scatter(pred_full[:, 0], pred_full[:, 1], pred_full[:, 2],
                         color=colors[color_key], marker=mk[color_key], s=ms*8, alpha=0.65,
                         edgecolors='white', linewidths=0.5, zorder=5)

    ax3d.set_xlabel('X (m)', fontweight='bold', fontsize=12)
    ax3d.set_ylabel('Y (m)', fontweight='bold', fontsize=12)
    ax3d.set_zlabel('Z (m)', fontweight='bold', fontsize=12)
    ax3d.set_title('3D Trajectory Comparison\nAll Agents', fontweight='bold', fontsize=13)
    ax3d.legend(fontsize=9, loc='upper left', ncol=2)
    ax3d.grid(True, linestyle='--', alpha=0.3)
    ax3d.view_init(elev=30, azim=-60)

    ax_xy = fig.add_subplot(gs[0, 1])
    for aid in range(num_agents):
        ax_xy.plot(x_sample[:, aid, 0], x_sample[:, aid, 1],
                   color=colors['history'], linestyle=ls['history'], linewidth=lw, alpha=0.8,
                   label='History' if aid == 0 else '', zorder=1)
        ax_xy.scatter(x_sample[:, aid, 0], x_sample[:, aid, 1],
                      color=colors['history'], marker=mk['history'], s=ms*8, alpha=0.6,
                      edgecolors='white', linewidths=0.5, zorder=1)

        last_pt = x_sample[-1:, aid, :]
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        ax_xy.plot(gt_full[:, 0], gt_full[:, 1],
                   color=colors['gt'], linestyle=ls['gt'], linewidth=lw, alpha=0.95,
                   label='Ground Truth' if aid == 0 else '', zorder=10)
        ax_xy.scatter(gt_full[:, 0], gt_full[:, 1],
                      color=colors['gt'], marker=mk['gt'], s=ms*8, alpha=0.75,
                      edgecolors='white', linewidths=0.5, zorder=10)

        for model_name, pred, color_key in models:
            pred_full = np.vstack([last_pt, pred[:, aid, :]])
            ax_xy.plot(pred_full[:, 0], pred_full[:, 1],
                       color=colors[color_key], linestyle=ls[color_key], linewidth=lw, alpha=0.85,
                       label=model_name if aid == 0 else '', zorder=5)
            ax_xy.scatter(pred_full[:, 0], pred_full[:, 1],
                          color=colors[color_key], marker=mk[color_key], s=ms*8, alpha=0.65,
                          edgecolors='white', linewidths=0.5, zorder=5)

    ax_xy.set_xlabel('X (m)', fontweight='bold', fontsize=12)
    ax_xy.set_ylabel('Y (m)', fontweight='bold', fontsize=12)
    ax_xy.set_title('XY Plane (Top View)', fontweight='bold', fontsize=13)
    ax_xy.legend(fontsize=9, loc='best', ncol=2)
    ax_xy.grid(True, linestyle='--', alpha=0.3)
    ax_xy.set_aspect('equal', adjustable='box')
    ax_xy.locator_params(axis='x', nbins=5)
    ax_xy.locator_params(axis='y', nbins=5)
    ax_xy.tick_params(axis='both', which='major', labelsize=10)

    ax_xz = fig.add_subplot(gs[0, 2])
    for aid in range(num_agents):
        ax_xz.plot(x_sample[:, aid, 0], x_sample[:, aid, 2],
                   color=colors['history'], linestyle=ls['history'], linewidth=lw, alpha=0.8,
                   label='History' if aid == 0 else '', zorder=1)
        ax_xz.scatter(x_sample[:, aid, 0], x_sample[:, aid, 2],
                      color=colors['history'], marker=mk['history'], s=ms*8, alpha=0.6,
                      edgecolors='white', linewidths=0.5, zorder=1)

        last_pt = x_sample[-1:, aid, :]
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        ax_xz.plot(gt_full[:, 0], gt_full[:, 2],
                   color=colors['gt'], linestyle=ls['gt'], linewidth=lw, alpha=0.95, zorder=10)
        ax_xz.scatter(gt_full[:, 0], gt_full[:, 2],
                      color=colors['gt'], marker=mk['gt'], s=ms*8, alpha=0.75,
                      edgecolors='white', linewidths=0.5, zorder=10)

        for model_name, pred, color_key in models:
            pred_full = np.vstack([last_pt, pred[:, aid, :]])
            ax_xz.plot(pred_full[:, 0], pred_full[:, 2],
                       color=colors[color_key], linestyle=ls[color_key], linewidth=lw, alpha=0.85, zorder=5)
            ax_xz.scatter(pred_full[:, 0], pred_full[:, 2],
                          color=colors[color_key], marker=mk[color_key], s=ms*8, alpha=0.65,
                          edgecolors='white', linewidths=0.5, zorder=5)

    ax_xz.set_xlabel('X (m)', fontweight='bold', fontsize=12)
    ax_xz.set_ylabel('Z (m)', fontweight='bold', fontsize=12)
    ax_xz.set_title('XZ Plane (Side View)', fontweight='bold', fontsize=13)
    ax_xz.grid(True, linestyle='--', alpha=0.3)
    ax_xz.set_aspect('equal', adjustable='box')
    ax_xz.locator_params(axis='x', nbins=5)
    ax_xz.locator_params(axis='y', nbins=5)
    ax_xz.tick_params(axis='both', which='major', labelsize=10)

    ax_yz = fig.add_subplot(gs[1, 0])
    for aid in range(num_agents):
        ax_yz.plot(x_sample[:, aid, 1], x_sample[:, aid, 2],
                   color=colors['history'], linestyle=ls['history'], linewidth=lw, alpha=0.8,
                   label='History' if aid == 0 else '', zorder=1)
        ax_yz.scatter(x_sample[:, aid, 1], x_sample[:, aid, 2],
                      color=colors['history'], marker=mk['history'], s=ms*8, alpha=0.6,
                      edgecolors='white', linewidths=0.5, zorder=1)

        last_pt = x_sample[-1:, aid, :]
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        ax_yz.plot(gt_full[:, 1], gt_full[:, 2],
                   color=colors['gt'], linestyle=ls['gt'], linewidth=lw, alpha=0.95, zorder=10)
        ax_yz.scatter(gt_full[:, 1], gt_full[:, 2],
                      color=colors['gt'], marker=mk['gt'], s=ms*8, alpha=0.75,
                      edgecolors='white', linewidths=0.5, zorder=10)

        for model_name, pred, color_key in models:
            pred_full = np.vstack([last_pt, pred[:, aid, :]])
            ax_yz.plot(pred_full[:, 1], pred_full[:, 2],
                       color=colors[color_key], linestyle=ls[color_key], linewidth=lw, alpha=0.85, zorder=5)
            ax_yz.scatter(pred_full[:, 1], pred_full[:, 2],
                          color=colors[color_key], marker=mk[color_key], s=ms*8, alpha=0.65,
                          edgecolors='white', linewidths=0.5, zorder=5)

    ax_yz.set_xlabel('Y (m)', fontweight='bold', fontsize=12)
    ax_yz.set_ylabel('Z (m)', fontweight='bold', fontsize=12)
    ax_yz.set_title('YZ Plane (Front View)', fontweight='bold', fontsize=13)
    ax_yz.grid(True, linestyle='--', alpha=0.3)
    ax_yz.set_aspect('equal', adjustable='box')
    ax_yz.locator_params(axis='x', nbins=5)
    ax_yz.locator_params(axis='y', nbins=5)
    ax_yz.tick_params(axis='both', which='major', labelsize=10)

    ax_err = fig.add_subplot(gs[1, 1])
    steps = np.arange(pred_lbebm.shape[0])
    err_mrgraj = np.mean(np.linalg.norm(pred_mrgraj - y_sample, axis=2), axis=1)
    err_lbebm = np.mean(np.linalg.norm(pred_lbebm - y_sample, axis=2), axis=1)
    err_exp5 = np.mean(np.linalg.norm(pred_exp5 - y_sample, axis=2), axis=1)
    err_swarm = np.mean(np.linalg.norm(pred_swarm_gru - y_sample, axis=2), axis=1)

    ax_err.plot(steps, err_mrgraj, color=colors['mrgraj'], linestyle=ls['mrgraj'], marker=mk['mrgraj'],
                markersize=7, label='MRGTraj', linewidth=lw, alpha=0.9)
    ax_err.plot(steps, err_lbebm, color=colors['3dmotraj'], linestyle=ls['3dmotraj'], marker=mk['3dmotraj'],
                markersize=7, label='3DMoTraj', linewidth=lw, alpha=0.9)
    ax_err.plot(steps, err_exp5, color=colors['exp5'], linestyle=ls['exp5'], marker=mk['exp5'],
                markersize=7, label='DG32-BCAT(Ours)', linewidth=lw, alpha=0.9)
    ax_err.plot(steps, err_swarm, color=colors['swarm_gru'], linestyle=ls['swarm_gru'], marker=mk['swarm_gru'],
                markersize=7, label='SwarmGRU v3', linewidth=lw, alpha=0.9)

    ax_err.set_xlabel('Prediction Step', fontweight='bold', fontsize=12)
    ax_err.set_ylabel('Mean Position Error (m)', fontweight='bold', fontsize=12)
    ax_err.set_title('Per-Step Error (All Agents Avg)', fontweight='bold', fontsize=13)
    ax_err.legend(fontsize=10, loc='best')
    ax_err.grid(True, linestyle='--', alpha=0.3)

    ax_bar = fig.add_subplot(gs[1, 2])
    metrics_lbebm = compute_metrics(pred_lbebm, y_sample)
    metrics_exp5 = compute_metrics(pred_exp5, y_sample)
    metrics_mrgraj = compute_metrics(pred_mrgraj, y_sample)
    metrics_swarm = compute_metrics(pred_swarm_gru, y_sample)

    x_pos = np.arange(2)
    width = 0.2

    bars1 = ax_bar.bar(x_pos - 1.5 * width, [metrics_mrgraj['ADE'], metrics_mrgraj['FDE']], width,
                       color=colors['mrgraj'], alpha=0.85, edgecolor='black', linewidth=1.5, label='MRGTraj')
    bars2 = ax_bar.bar(x_pos - 0.5 * width, [metrics_lbebm['ADE'], metrics_lbebm['FDE']], width,
                       color=colors['3dmotraj'], alpha=0.85, edgecolor='black', linewidth=1.5, label='3DMoTraj')
    bars3 = ax_bar.bar(x_pos + 0.5 * width, [metrics_exp5['ADE'], metrics_exp5['FDE']], width,
                       color=colors['exp5'], alpha=0.85, edgecolor='black', linewidth=1.5, label='DG32-BCAT')
    bars4 = ax_bar.bar(x_pos + 1.5 * width, [metrics_swarm['ADE'], metrics_swarm['FDE']], width,
                       color=colors['swarm_gru'], alpha=0.85, edgecolor='black', linewidth=1.5, label='SwarmGRU v3')

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(['ADE', 'FDE'], fontweight='bold', fontsize=11)
    ax_bar.set_ylabel('Error (m)', fontweight='bold', fontsize=12)
    ax_bar.set_title('Overall Performance Metrics', fontweight='bold', fontsize=13)
    ax_bar.legend(fontsize=9, loc='upper left', ncol=2)
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3)

    png_path = str(Path(output_path).parent / f'sample_{sample_idx}_comparison_v4.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    logger.info(f"  ✓ 可视化保存: {png_path}")
    plt.close()
    plt.rcdefaults()


def main():
    parser = argparse.ArgumentParser(description='四模型对比: 3DMoTraj vs DG32-BCAT vs MRGTraj vs SwarmGRU v3')

    parser.add_argument('--data_dir', default=str(workspace_root / "drone_trajectories" / "Cluster trajectory" / "swarm_segments"), help='数据目录')
    parser.add_argument('--agents', type=int, default=3, help='无人机数量')
    parser.add_argument('--use_subset', action='store_true', help='使用子集数据')

    parser.add_argument('--lbebm_model', required=True, help='LBEBM3D 模型路径')
    parser.add_argument('--exp5_dir', required=True, help='Exp5 结果目录')
    parser.add_argument('--mrgraj_model', required=True, help='MRGTraj 模型路径')
    parser.add_argument('--mrgraj_ckpt_dir', default='checkpoints_lbebm3d', help='MRGTraj 检查点目录')
    parser.add_argument('--gru_model', default=str(current_dir / 'Models' / 'swarm_gru_agents_3_best.pth'), help='SwarmGRU v3 模型路径')

    parser.add_argument('--features_32d_dir', default='features_32d', help='32D特征目录')

    parser.add_argument('--data_scale', type=float, default=1.0)
    parser.add_argument('--e_init_sig', type=float, default=2.0)
    parser.add_argument('--e_prior_sig', type=float, default=2.0)
    parser.add_argument('--e_l_steps', type=int, default=20)
    parser.add_argument('--e_l_step_size', type=float, default=0.4)
    parser.add_argument('--e_l_with_noise', action='store_true')

    parser.add_argument('--no_physical_constraints', action='store_true', help='禁用 Exp5 物理约束')
    parser.add_argument('--pc_dt', type=float, default=0.1, help='物理约束时间步')
    parser.add_argument('--pc_smoothing_weight', type=float, default=0.3, help='物理约束平滑权重')
    parser.add_argument('--pc_constraint_relaxation', type=float, default=1.0, help='物理约束松弛因子')

    parser.add_argument('--no_mrgraj_physical_constraints', action='store_true', help='禁用 MRGTraj 物理约束')
    parser.add_argument('--mrgraj_pc_dt', type=float, default=0.1, help='MRGTraj 物理约束时间步')
    parser.add_argument('--mrgraj_pc_smoothing_weight', type=float, default=0.2, help='MRGTraj 物理约束平滑权重')
    parser.add_argument('--mrgraj_pc_constraint_relaxation', type=float, default=0.8, help='MRGTraj 物理约束松弛因子')

    parser.add_argument('--sample_indices', type=str, default=None, help='逗号分隔的样本索引')
    parser.add_argument('--num_samples', type=int, default=10, help='随机样本数')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--use_val_split', action='store_true', help='使用验证集划分')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--no_visualize', action='store_true', help='禁用可视化')
    parser.add_argument('--output_dir', default='comparison_four_models', help='输出目录')

    args = parser.parse_args()

    if not LBEBM_AVAILABLE:
        logger.error("✗ LBEBM3D 不可用")
        sys.exit(1)

    if not EXP5_AVAILABLE:
        logger.error("✗ Exp5 (DG32-BCAT) 不可用")
        sys.exit(1)

    if not MRGRAJ_AVAILABLE:
        logger.error("✗ MRGTraj 不可用")
        sys.exit(1)

    if not Path(args.gru_model).exists():
        logger.error(f"✗ SwarmGRU 模型不存在: {args.gru_model}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    logger.info(f"加载数据: {args.data_dir}")
    x_all, y_all = load_data_ablation(args.data_dir, args.agents, use_subset=args.use_subset)
    logger.info(f"数据形状: X={x_all.shape}, Y={y_all.shape}")

    logger.info(f"加载 LBEBM3D 模型: {args.lbebm_model}")
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
    logger.info("✓ LBEBM3D 模型加载成功")

    logger.info(f"加载 Exp5 模型 (消融实验一致方式): {args.exp5_dir}")
    exp5_dir = Path(args.exp5_dir)
    config_path = exp5_dir / f"config_agents_{args.agents}_exp5_full.json"
    stats_path = exp5_dir / f"stats_agents_{args.agents}_exp5_full.npz"
    model_path = exp5_dir / f"best_model_agents_{args.agents}_exp5_full.pt"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    stats = np.load(stats_path)
    exp5_output_mean = stats['output_mean']
    exp5_output_std = stats['output_std']

    exp5_ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    # 与消融实验 AblationModelEnsemble.load_experiment(5,...) 完全一致：input_size 固定为 32
    exp5_model = DynamicsAwareSwarmGRUModel_with_GNN(
        input_size=32,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
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
    logger.info("✓ Exp5 (DG32-BCAT) 模型加载成功")

    logger.info(f"加载 MRGTraj 模型: {args.mrgraj_model}")
    import argparse as arg_module
    mrgraj_args = arg_module.Namespace(
        d_model=256,
        n_heads=4,
        n_layers=2,
        noise_dim=64,
        agent_dim=3,
        obs_len=20,
        pred_len=10,
        num_agents=args.agents,
    )
    mrgraj_model = MRGTrajSwarm(mrgraj_args)
    mrgraj_ckpt = torch.load(args.mrgraj_model, map_location='cpu', weights_only=False)
    mrgraj_model.load_state_dict(mrgraj_ckpt['model_state_dict'])
    mrgraj_model.to(device)
    mrgraj_model.eval()
    logger.info("✓ MRGTraj 模型加载成功")

    logger.info(f"加载 SwarmGRU v3 模型: {args.gru_model}")
    swarm_gru_model, swarm_stats = load_swarm_gru_model(args.gru_model, device)
    if swarm_gru_model is None:
        logger.error("✗ SwarmGRU v3 模型加载失败")
        sys.exit(1)
    logger.info("✓ SwarmGRU v3 模型加载成功")

    logger.info("加载 32D 特征 (消融实验一致方式)...")
    # 只取原始特征数组，忽略文件中可能存储的统计量键
    # 原因：ablation_train_utils.load_ablation_data 始终从全量数据重新计算统计量
    features_all_raw, _, _ = load_32d_features_with_stats(
        args.features_32d_dir, args.agents, args.use_subset
    )
    if features_all_raw is None:
        logger.warning("⚠ 32D 特征加载失败，Exp5 可能无法运行")
        features_all = None
        feature_means = None
        feature_stds = None
    else:
        features_all = features_all_raw
        # 与 load_ablation_data 完全一致：从全量特征重新计算 mean/std
        feat_dim = features_all_raw.shape[-1]
        feature_means = np.mean(features_all_raw.reshape(-1, feat_dim), axis=0).astype(np.float32)
        feature_stds = np.std(features_all_raw.reshape(-1, feat_dim), axis=0).astype(np.float32)
        feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
        logger.info(f"特征形状: {features_all_raw.shape} (统计量已从全量数据重新计算)")

    total_samples = min(len(x_all), len(features_all) if features_all is not None else len(x_all))
    if features_all is not None and len(x_all) != len(features_all):
        logger.warning(f"样本数不匹配，截断至 {total_samples}")
        x_all = x_all[:total_samples]
        y_all = y_all[:total_samples]
        features_all = features_all[:total_samples]

    # 创建与消融实验训练完全一致的 AblationDataset，供 EXP5 推理使用
    ablation_dataset = None
    if features_all is not None and ABLATION_DATASET_AVAILABLE:
        # 统计量计算方式与 load_ablation_data 完全一致
        input_mean_ds = np.mean(x_all.reshape(-1, 3), axis=0).astype(np.float32)
        input_std_ds = np.std(x_all.reshape(-1, 3), axis=0).astype(np.float32)
        input_std_ds = np.where(input_std_ds < 1e-8, 1.0, input_std_ds)
        y_delta_ds = y_all - x_all[:, -1:, :, :]
        output_mean_ds = np.mean(y_delta_ds.reshape(-1, 3), axis=0).astype(np.float32)
        output_std_ds = np.std(y_delta_ds.reshape(-1, 3), axis=0).astype(np.float32)
        output_std_ds = np.where(output_std_ds < 1e-8, 1.0, output_std_ds)
        ablation_dataset = AblationDataset(
            x_all, y_all,
            feature_dim=32, normalize=True, dt=0.1,
            input_mean=input_mean_ds, input_std=input_std_ds,
            output_mean=output_mean_ds, output_std=output_std_ds,
            features_precomputed=features_all,
            feature_mean=feature_means, feature_std=feature_stds
        )
        logger.info(f"✓ AblationDataset 创建成功 ({len(ablation_dataset)} 样本)")

    np.random.seed(args.seed)
    if args.sample_indices:
        sample_indices = [int(i) for i in args.sample_indices.split(',')]
    else:
        sample_indices = np.random.choice(total_samples, args.num_samples, replace=False).tolist()

    logger.info(f"选定样本: {sample_indices}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics_lbebm = []
    all_metrics_exp5 = []
    all_metrics_mrgraj = []
    all_metrics_swarm = []

    for i, sample_idx in enumerate(sample_indices):
        logger.info(f"\n[{i+1}/{len(sample_indices)}] 处理样本 {sample_idx}")

        x_sample = x_all[sample_idx]
        y_sample = y_all[sample_idx]
        features = features_all[sample_idx] if features_all is not None else None

        pred_lbebm = predict_lbebm_multi_agent(
            lbebm_model, x_sample, device, args.data_scale,
            args.e_init_sig, args.e_prior_sig, args.e_l_steps, args.e_l_step_size, args.e_l_with_noise
        )

        pred_exp5 = predict_exp5(
            exp5_model, x_sample, features, device,
            exp5_output_mean, exp5_output_std, feature_means, feature_stds,
            use_physical_constraints=not args.no_physical_constraints,
            pc_dt=args.pc_dt,
            pc_smoothing_weight=args.pc_smoothing_weight,
            pc_constraint_relaxation=args.pc_constraint_relaxation
        ) if features is not None else np.zeros_like(y_sample)

        pred_mrgraj = predict_mrgraj(
            mrgraj_model, x_sample, device,
            use_physical_constraints=not args.no_mrgraj_physical_constraints,
            pc_dt=args.mrgraj_pc_dt,
            pc_smoothing_weight=args.mrgraj_pc_smoothing_weight,
            pc_constraint_relaxation=args.mrgraj_pc_constraint_relaxation
        )

        pred_swarm = predict_swarm_gru(swarm_gru_model, x_sample, swarm_stats, device)

        metrics_lbebm = compute_metrics(pred_lbebm, y_sample)
        metrics_exp5 = compute_metrics(pred_exp5, y_sample) if features is not None else {k: 0 for k in metrics_lbebm.keys()}
        metrics_mrgraj = compute_metrics(pred_mrgraj, y_sample)
        metrics_swarm = compute_metrics(pred_swarm, y_sample)

        all_metrics_lbebm.append(metrics_lbebm)
        all_metrics_exp5.append(metrics_exp5)
        all_metrics_mrgraj.append(metrics_mrgraj)
        all_metrics_swarm.append(metrics_swarm)

        logger.info(f"  LBEBM3D:   ADE={metrics_lbebm['ADE']:.4f}m, FDE={metrics_lbebm['FDE']:.4f}m")
        logger.info(f"  Exp5:      ADE={metrics_exp5['ADE']:.4f}m, FDE={metrics_exp5['FDE']:.4f}m")
        logger.info(f"  MRGTraj:   ADE={metrics_mrgraj['ADE']:.4f}m, FDE={metrics_mrgraj['FDE']:.4f}m")
        logger.info(f"  SwarmGRU:  ADE={metrics_swarm['ADE']:.4f}m, FDE={metrics_swarm['FDE']:.4f}m")

        if not args.no_visualize:
            viz_path = output_dir / f"sample_{sample_idx:06d}_comparison.png"
            visualize_comparison(x_sample, y_sample, pred_lbebm, pred_exp5, pred_mrgraj, pred_swarm, sample_idx, str(viz_path))

    logger.info("\n" + "=" * 100)
    logger.info("=== 整体结果 ===")

    def aggregate_metrics(metrics_list):
        if not metrics_list or all(m.get('ADE', 0) == 0 for m in metrics_list):
            return {k: {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0} for k in metrics_list[0].keys()} if metrics_list else {}

        result = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                values = np.array([m[key] for m in metrics_list], dtype=np.float32)
                result[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                }
        return result

    agg_lbebm = aggregate_metrics(all_metrics_lbebm)
    agg_exp5 = aggregate_metrics(all_metrics_exp5)
    agg_mrgraj = aggregate_metrics(all_metrics_mrgraj)
    agg_swarm = aggregate_metrics(all_metrics_swarm)

    logger.info("\n✓ LBEBM3D 统计:")
    logger.info(f"  ADE: {agg_lbebm['ADE']['mean']:.4f} ± {agg_lbebm['ADE']['std']:.4f}m")
    logger.info(f"  FDE: {agg_lbebm['FDE']['mean']:.4f} ± {agg_lbebm['FDE']['std']:.4f}m")
    logger.info(f"  RMSE: {agg_lbebm['RMSE']['mean']:.4f} ± {agg_lbebm['RMSE']['std']:.4f}m")

    logger.info("\n✓ Exp5 (DG32-BCAT) 统计:")
    logger.info(f"  ADE: {agg_exp5['ADE']['mean']:.4f} ± {agg_exp5['ADE']['std']:.4f}m")
    logger.info(f"  FDE: {agg_exp5['FDE']['mean']:.4f} ± {agg_exp5['FDE']['std']:.4f}m")
    logger.info(f"  RMSE: {agg_exp5['RMSE']['mean']:.4f} ± {agg_exp5['RMSE']['std']:.4f}m")

    logger.info("\n✓ MRGTraj 统计:")
    logger.info(f"  ADE: {agg_mrgraj['ADE']['mean']:.4f} ± {agg_mrgraj['ADE']['std']:.4f}m")
    logger.info(f"  FDE: {agg_mrgraj['FDE']['mean']:.4f} ± {agg_mrgraj['FDE']['std']:.4f}m")
    logger.info(f"  RMSE: {agg_mrgraj['RMSE']['mean']:.4f} ± {agg_mrgraj['RMSE']['std']:.4f}m")

    logger.info("\n✓ SwarmGRU v3 统计:")
    logger.info(f"  ADE: {agg_swarm['ADE']['mean']:.4f} ± {agg_swarm['ADE']['std']:.4f}m")
    logger.info(f"  FDE: {agg_swarm['FDE']['mean']:.4f} ± {agg_swarm['FDE']['std']:.4f}m")
    logger.info(f"  RMSE: {agg_swarm['RMSE']['mean']:.4f} ± {agg_swarm['RMSE']['std']:.4f}m")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(sample_indices),
        "sample_indices": sample_indices,
        "metrics": {
            "lbebm3d": agg_lbebm,
            "exp5": agg_exp5,
            "mrgraj": agg_mrgraj,
            "swarm_gru_v3": agg_swarm,
        }
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n保存汇总: {summary_path}")

    csv_summary_path = output_dir / "comparison_summary.csv"
    with open(csv_summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Metric", "Mean", "Std", "Min", "Max", "Median"])
        for model_name, agg in [
            ("LBEBM3D", agg_lbebm),
            ("DG32-BCAT", agg_exp5),
            ("MRGTraj", agg_mrgraj),
            ("SwarmGRU v3", agg_swarm),
        ]:
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
    logger.info(f"CSV 汇总保存: {csv_summary_path}")

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
        for idx, (m_lbebm, m_exp5, m_mrgraj, m_swarm) in zip(sample_indices, zip(all_metrics_lbebm, all_metrics_exp5, all_metrics_mrgraj, all_metrics_swarm)):
            for name, metrics in (
                ("LBEBM3D", m_lbebm),
                ("DG32-BCAT", m_exp5),
                ("MRGTraj", m_mrgraj),
                ("SwarmGRU v3", m_swarm),
            ):
                writer.writerow([
                    idx,
                    name,
                    metrics.get("MAE"),
                    metrics.get("RMSE"),
                    metrics.get("ADE"),
                    metrics.get("FDE"),
                    metrics.get("MAPE"),
                    metrics.get("MAE_X"),
                    metrics.get("MAE_Y"),
                    metrics.get("MAE_Z"),
                ])
    logger.info(f"每样本指标保存: {csv_samples_path}")

    logger.info("\n" + "=" * 100)
    logger.info(f"完成！结果保存至: {output_dir}")
    logger.info("=" * 100)


if __name__ == '__main__':
    main()
