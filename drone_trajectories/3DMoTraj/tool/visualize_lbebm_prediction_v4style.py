#!/usr/bin/env python3
"""
LBEBM3D 可视化脚本（v4 风格，6 视图）

输入：Cluster trajectory 的 swarm_segments NPZ 格式数据
  - input_agents_X(_subset).npz: (seq_in, samples, agents, 3)
  - output_agents_X(_subset).npz: (seq_out, samples, agents, 3)

输出：
  - PNG：lbebm_prediction_sample_XXXXXX.png（6视图对比：3D/XY/XZ/YZ/逐步误差/柱状误差）

注意：
  - LBEBM3D 本身是单轨迹模型，这里对每个 agent 独立运行一次推理，再合并到同一张图中。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer_lbebm3d_baseline import LBEBM3DInfer, infer_model_params_from_state_dict

# 配置中文字体与负号显示（与 visualize_swarm_prediction_v4.py 一致）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_npz_pair(data_dir: Path, agents: int, use_subset: bool):
    suffix = "_subset" if use_subset else ""
    X_file = data_dir / f"input_agents_{agents}{suffix}.npz"
    Y_file = data_dir / f"output_agents_{agents}{suffix}.npz"
    X_raw = np.load(X_file)["data"]  # (seq_in, samples, agents, 3)
    Y_raw = np.load(Y_file)["data"]  # (seq_out, samples, agents, 3)
    X = np.transpose(X_raw, (1, 0, 2, 3))  # (samples, seq_in, agents, 3)
    Y = np.transpose(Y_raw, (1, 0, 2, 3))  # (samples, seq_out, agents, 3)
    return X, Y


def _predict_one_agent(
    model: LBEBM3DInfer,
    past_abs: np.ndarray,  # (seq_in, 3)
    device: torch.device,
    data_scale: float,
    e_init_sig: float,
    e_prior_sig: float,
    e_l_steps: int,
    e_l_step_size: float,
    e_l_with_noise: bool,
) -> np.ndarray:
    last = past_abs[-1:].copy()  # (1,3)
    past_rel = (past_abs - last) * data_scale
    past_t = torch.from_numpy(past_rel.reshape(1, -1)).to(device=device, dtype=torch.double)
    langevin_cfg = {
        "e_init_sig": e_init_sig,
        "e_prior_sig": e_prior_sig,
        "e_l_steps": e_l_steps,
        "e_l_step_size": e_l_step_size,
        "e_l_with_noise": e_l_with_noise,
    }
    plan = model.sample_plan(past_t, langevin_cfg)
    fut_rel = model.predict(past_t, plan).detach().cpu().numpy()[0]  # (T,3) rel*data_scale
    fut_abs = fut_rel / data_scale + last[0]
    return fut_abs


def _plot_6_views(history, truth, pred, sample_idx: int, agents: int, out_png: Path):
    """
    history: (seq_in, agents, 3)
    truth/pred: (seq_out, agents, 3)
    """
    # 创建大图（与 v4 一致）
    fig = plt.figure(figsize=(20, 13))

    # 1. 3D 轨迹对比（与 v4 完全一致的样式）
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")

    # 历史轨迹（蓝色）
    for agent_id in range(agents):
        if agent_id == 0:
            ax3d.plot(
                history[:, agent_id, 0],
                history[:, agent_id, 1],
                history[:, agent_id, 2],
                "b-o",
                linewidth=2.5,
                markersize=5,
                label="History",
                alpha=0.8,
            )
        else:
            ax3d.plot(
                history[:, agent_id, 0],
                history[:, agent_id, 1],
                history[:, agent_id, 2],
                "b-o",
                linewidth=2.5,
                markersize=5,
                alpha=0.8,
            )

    # 真实与预测（绿色/红色）
    for agent_id in range(agents):
        last_point = history[-1:, agent_id, :]
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        pred_traj = np.vstack([last_point, pred[:, agent_id, :]])

        ax3d.plot(
            true_traj[:, 0],
            true_traj[:, 1],
            true_traj[:, 2],
            "gs-",
            linewidth=2.8,
            markersize=7,
            alpha=0.9,
            label="True Future" if agent_id == 0 else "",
        )
        ax3d.plot(
            pred_traj[:, 0],
            pred_traj[:, 1],
            pred_traj[:, 2],
            "r^--",
            linewidth=2.5,
            markersize=6,
            alpha=0.75,
            label="Predicted Future" if agent_id == 0 else "",
        )

    ax3d.set_xlabel("X (m)", fontsize=11, fontweight="bold")
    ax3d.set_ylabel("Y (m)", fontsize=11, fontweight="bold")
    ax3d.set_zlabel("Z (m)", fontsize=11, fontweight="bold")
    ax3d.set_title(
        f"Sample {sample_idx}: 3D Swarm Trajectories (LBEBM3D)\nAll {agents} Agents",
        fontsize=12,
        fontweight="bold",
    )
    ax3d.legend(fontsize=10, loc="upper left")
    ax3d.grid(True, alpha=0.3)

    # 2. XY 平面
    ax_xy = fig.add_subplot(2, 3, 2)
    for agent_id in range(agents):
        last_point = history[-1:, agent_id, :]
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        pred_traj = np.vstack([last_point, pred[:, agent_id, :]])

        if agent_id == 0:
            ax_xy.plot(
                history[:, agent_id, 0],
                history[:, agent_id, 1],
                "b-o",
                linewidth=2.5,
                markersize=5,
                label="History",
                alpha=0.8,
            )
        else:
            ax_xy.plot(
                history[:, agent_id, 0],
                history[:, agent_id, 1],
                "b-o",
                linewidth=2.5,
                markersize=5,
                alpha=0.8,
            )

        ax_xy.plot(true_traj[:, 0], true_traj[:, 1], "gs-", linewidth=2.8, markersize=7, label="True" if agent_id == 0 else "", alpha=0.9)
        ax_xy.plot(pred_traj[:, 0], pred_traj[:, 1], "r^--", linewidth=2.5, markersize=6, label="Predicted" if agent_id == 0 else "", alpha=0.75)

    ax_xy.set_xlabel("X (m)", fontsize=11, fontweight="bold")
    ax_xy.set_ylabel("Y (m)", fontsize=11, fontweight="bold")
    ax_xy.set_title("XY Plane Projection", fontsize=12, fontweight="bold")
    ax_xy.legend(fontsize=10, loc="best")
    ax_xy.grid(True, alpha=0.3)

    # 3. XZ 平面
    ax_xz = fig.add_subplot(2, 3, 3)
    for agent_id in range(agents):
        last_point = history[-1:, agent_id, :]
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        pred_traj = np.vstack([last_point, pred[:, agent_id, :]])

        if agent_id == 0:
            ax_xz.plot(history[:, agent_id, 0], history[:, agent_id, 2], "b-o", linewidth=2.5, markersize=5, label="History", alpha=0.8)
        else:
            ax_xz.plot(history[:, agent_id, 0], history[:, agent_id, 2], "b-o", linewidth=2.5, markersize=5, alpha=0.8)

        ax_xz.plot(true_traj[:, 0], true_traj[:, 2], "gs-", linewidth=2.8, markersize=7, label="True" if agent_id == 0 else "", alpha=0.9)
        ax_xz.plot(pred_traj[:, 0], pred_traj[:, 2], "r^--", linewidth=2.5, markersize=6, label="Predicted" if agent_id == 0 else "", alpha=0.75)

    ax_xz.set_xlabel("X (m)", fontsize=11, fontweight="bold")
    ax_xz.set_ylabel("Z (m)", fontsize=11, fontweight="bold")
    ax_xz.set_title("XZ Plane Projection", fontsize=12, fontweight="bold")
    ax_xz.legend(fontsize=10, loc="best")
    ax_xz.grid(True, alpha=0.3)

    # 4. YZ 平面
    ax_yz = fig.add_subplot(2, 3, 4)
    for agent_id in range(agents):
        last_point = history[-1:, agent_id, :]
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        pred_traj = np.vstack([last_point, pred[:, agent_id, :]])

        if agent_id == 0:
            ax_yz.plot(history[:, agent_id, 1], history[:, agent_id, 2], "b-o", linewidth=2.5, markersize=5, label="History", alpha=0.8)
        else:
            ax_yz.plot(history[:, agent_id, 1], history[:, agent_id, 2], "b-o", linewidth=2.5, markersize=5, alpha=0.8)

        ax_yz.plot(true_traj[:, 1], true_traj[:, 2], "gs-", linewidth=2.8, markersize=7, label="True" if agent_id == 0 else "", alpha=0.9)
        ax_yz.plot(pred_traj[:, 1], pred_traj[:, 2], "r^--", linewidth=2.5, markersize=6, label="Predicted" if agent_id == 0 else "", alpha=0.75)

    ax_yz.set_xlabel("Y (m)", fontsize=11, fontweight="bold")
    ax_yz.set_ylabel("Z (m)", fontsize=11, fontweight="bold")
    ax_yz.set_title("YZ Plane Projection", fontsize=12, fontweight="bold")
    ax_yz.legend(fontsize=10, loc="best")
    ax_yz.grid(True, alpha=0.3)

    # 5. 轴向误差（与 v4 完全一致）
    ax_error_ts = fig.add_subplot(2, 3, 5)
    steps = np.arange(truth.shape[0])
    error_x = np.abs(pred[:, :, 0] - truth[:, :, 0]).mean(axis=1)
    error_y = np.abs(pred[:, :, 1] - truth[:, :, 1]).mean(axis=1)
    error_z = np.abs(pred[:, :, 2] - truth[:, :, 2]).mean(axis=1)

    ax_error_ts.plot(steps, error_x, "rs-", linewidth=2.5, markersize=7, label="X Axis Error", alpha=0.8)
    ax_error_ts.plot(steps, error_y, "bo-", linewidth=2.5, markersize=7, label="Y Axis Error", alpha=0.8)
    ax_error_ts.plot(steps, error_z, "g^-", linewidth=2.5, markersize=7, label="Z Axis Error", alpha=0.8)

    ax_error_ts.set_xlabel("Prediction Step", fontsize=11, fontweight="bold")
    ax_error_ts.set_ylabel("Mean Absolute Error (m)", fontsize=11, fontweight="bold")
    ax_error_ts.set_title("Per-Step Axis-wise Error", fontsize=12, fontweight="bold")
    ax_error_ts.legend(fontsize=10, loc="best")
    ax_error_ts.grid(True, alpha=0.3)

    # 6. 总体误差分布（与 v4 完全一致）
    ax_error_dist = fig.add_subplot(2, 3, 6)
    errors_per_step = np.linalg.norm(pred - truth, axis=2).mean(axis=1)
    bars = ax_error_dist.bar(steps, errors_per_step, color="tab:red", alpha=0.7, edgecolor="darkred", linewidth=1.5)
    for step, err in enumerate(errors_per_step):
        ax_error_dist.text(step, err + 0.005, f"{err:.3f}", ha="center", va="bottom", fontsize=8)

    mean_error = errors_per_step.mean()
    ax_error_dist.axhline(y=mean_error, color="darkred", linestyle="--", linewidth=2, label=f"Mean Error: {mean_error:.4f}m")
    ax_error_dist.set_xlabel("Prediction Step", fontsize=11, fontweight="bold")
    ax_error_dist.set_ylabel("Position Error (m)", fontsize=11, fontweight="bold")
    ax_error_dist.set_title("Position Error per Step (All Agents Avg)", fontsize=12, fontweight="bold")
    ax_error_dist.legend(fontsize=10, loc="best")
    ax_error_dist.grid(True, axis="y", alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    # 与 v4 一致：dpi=150
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="LBEBM3D visualization (v4 style)")
    ap.add_argument("--model_path", required=True, type=str, help="LBEBM3D checkpoint (.pt)")
    ap.add_argument("--data_dir", required=True, type=str, help="swarm_segments directory")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--use_subset", action="store_true")
    ap.add_argument("--output_dir", type=str, default="lbebm_visualization_output")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--data_scale", type=float, default=1.0)

    # Langevin cfg
    ap.add_argument("--e_init_sig", type=float, default=2.0)
    ap.add_argument("--e_prior_sig", type=float, default=2.0)
    ap.add_argument("--e_l_steps", type=int, default=20)
    ap.add_argument("--e_l_step_size", type=float, default=0.4)
    ap.add_argument("--e_l_with_noise", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    ckpt = torch.load(args.model_path, map_location="cpu",weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    params = infer_model_params_from_state_dict(state_dict)

    # subgoals
    if params["num_subgoals"] == 4 and params["future_length"] >= 10:
        sub_goal_indexes = [2, 5, 7, 9]
    else:
        sub_goal_indexes = list(np.linspace(0, params["future_length"] - 1, params["num_subgoals"], dtype=int))

    model = LBEBM3DInfer(
        enc_past_size=params["enc_past_size"],
        enc_dest_size=params["enc_dest_size"],
        enc_latent_size=params["enc_latent_size"],
        dec_size=params["dec_size"],
        predictor_size=params["predictor_size"],
        fdim=params["fdim"],
        zdim=params["zdim"],
        ny=params["ny"],
        past_length=params["past_length"],
        future_length=params["future_length"],
        sub_goal_indexes=sub_goal_indexes,
    ).double()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    X_all, Y_all = _load_npz_pair(Path(args.data_dir), args.agents, args.use_subset)
    total = X_all.shape[0]
    np.random.seed(args.seed)
    idxs = np.random.choice(total, min(args.num_samples, total), replace=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in idxs:
        history = X_all[sample_idx]  # (seq_in, agents, 3)
        truth = Y_all[sample_idx]    # (seq_out, agents, 3)

        pred = np.zeros_like(truth)
        for a in range(args.agents):
            pred[:, a, :] = _predict_one_agent(
                model,
                history[:, a, :],
                device=device,
                data_scale=args.data_scale,
                e_init_sig=args.e_init_sig,
                e_prior_sig=args.e_prior_sig,
                e_l_steps=args.e_l_steps,
                e_l_step_size=args.e_l_step_size,
                e_l_with_noise=args.e_l_with_noise,
            )

        png = out_dir / f"lbebm_prediction_sample_{sample_idx:06d}.png"
        _plot_6_views(history, truth, pred, int(sample_idx), args.agents, png)


if __name__ == "__main__":
    main()

