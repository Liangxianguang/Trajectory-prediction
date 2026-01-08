#!/usr/bin/env python3
"""infer_swarm_model_clean.py

干净的推理脚本（排版与可读性优化），功能与训练脚本保持一致：
- 使用与训练相同的特征计算
- 使用训练保存的统计量完成反归一化
- 自回归推理（teacher_forcing_ratio=0.0）

用法示例：
python infer_swarm_model_clean.py --model path/to/checkpoint.pt --data_dir path/to/data --agents 3 --batch_size 32 --visualize
"""

from pathlib import Path
import sys
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 将项目根加入路径，确保可以导入 train_swarm_model_enhanced
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_swarm_model_enhanced import (
        EnhancedSwarmGRUModel,
        compute_multi_scale_velocity,
        compute_curvature,
        compute_plane_curvatures,
    )
except ImportError as e:
    logger.error("无法导入 train_swarm_model_enhanced.py，请确保该文件在同一目录或 PYTHONPATH 中：%s", e)
    raise


def segment_trajectory(trajectory, seq_in=20, seq_out=10):
    """将连续轨迹分段为输入/输出对（滑动窗口）。
    
    Args:
        trajectory: (seq, agents, 3) 连续轨迹
        seq_in: 输入序列长度
        seq_out: 输出序列长度
    
    Returns:
        X: (num_segments, seq_in, agents, 3)
        Y: (num_segments, seq_out, agents, 3)
    """
    seq_len = trajectory.shape[0]
    step = seq_in + seq_out
    
    segments_X = []
    segments_Y = []
    
    for i in range(0, seq_len - step + 1):
        x_seg = trajectory[i : i + seq_in]
        y_seg = trajectory[i + seq_in : i + seq_in + seq_out]
        segments_X.append(x_seg)
        segments_Y.append(y_seg)
    
    return np.array(segments_X), np.array(segments_Y)


def compute_features_for_inference(
    trajectory,
    input_mean_all=None,
    input_std_all=None,
    input_mean=None,
    input_std=None,
    dt=0.1,
):
    """Compute 16-D features for inference, matching training preprocessing.

    Args:
        trajectory: np.ndarray, (seq_in, agents, 3)
        input_mean_all: optional (16,) global mean for features
        input_std_all: optional (16,) global std for features
        dt: float, sampling interval

    Returns:
        features: np.float32 array, shape (seq_in, agents, 16)
    """

    vel = compute_multi_scale_velocity(trajectory, dt=dt)  # (seq_in, agents, 9)
    curv_3d = compute_curvature(trajectory, dt=dt)  # (seq_in, agents, 1)
    curv_plane = compute_plane_curvatures(trajectory, dt=dt)  # (seq_in, agents, 3)

    # Local max-norm (training used this per-sample normalization)
    vel_norm = np.abs(vel).max()
    if vel_norm > 1e-8:
        vel = vel / vel_norm

    curv_3d = curv_3d / (np.abs(curv_3d).max() + 1e-8)
    curv_plane = curv_plane / (np.abs(curv_plane).max() + 1e-8)

    features = np.concatenate([trajectory, vel, curv_3d, curv_plane], axis=-1)

    # 分层归一化
    if input_mean is not None and input_std is not None:
        mean_vec = np.array(input_mean, dtype=np.float32).reshape(1, 1, 3)
        std_vec = np.array(input_std, dtype=np.float32).reshape(1, 1, 3)
        features[..., :3] = (features[..., :3] - mean_vec) / (std_vec + 1e-8)
    if input_mean_all is not None and input_std_all is not None:
        mean_vec_all = np.array(input_mean_all, dtype=np.float32).reshape(1, 1, 16)
        std_vec_all = np.array(input_std_all, dtype=np.float32).reshape(1, 1, 16)
        features[..., 3:] = (features[..., 3:] - mean_vec_all[..., 3:]) / (std_vec_all[..., 3:] + 1e-8)

    features = np.clip(features, -5.0, 5.0)
    return features.astype(np.float32)


def apply_physical_constraints(history, pred_delta, dt=0.1, smoothing_weight=0.3):
    """Apply velocity/acceleration constraints for smoother reconstruction."""
    history = np.array(history, dtype=np.float32)
    B, seq, agents, _ = history.shape
    if seq < 2:
        history_vel = np.zeros((B, 1, agents, 3), dtype=np.float32)
    else:
        history_vel = np.diff(history, axis=1) / dt

    if history_vel.shape[1] == 0:
        history_vel = np.zeros((B, 1, agents, 3), dtype=np.float32)

    last_vel = (
        history_vel[:, -5:, :, :].mean(axis=1)
        if history_vel.shape[1] > 0
        else np.zeros((B, agents, 3), dtype=np.float32)
    )

    history_acc = (
        np.diff(history_vel, axis=1) / dt if history_vel.shape[1] > 1 else np.zeros((B, 1, agents, 3), dtype=np.float32)
    )
    avg_acc = (
        history_acc.mean(axis=1, keepdims=True)
        if history_acc.shape[1] > 0
        else np.zeros((B, 1, agents, 3), dtype=np.float32)
    )

    max_vel = np.maximum(
        np.max(np.linalg.norm(history_vel, axis=3), axis=1, keepdims=True),
        1e-3,
    )
    max_acc = np.maximum(
        np.max(np.linalg.norm(history_acc, axis=3), axis=1, keepdims=True),
        1e-3,
    )
    max_vel = max_vel[:, 0, :, np.newaxis]
    max_acc = max_acc[:, 0, :, np.newaxis]

    current_pos = history[:, -1, :, :].copy()
    current_vel = last_vel.copy()
    steps = pred_delta.shape[1]
    reconstructed = np.zeros((B, steps, agents, 3), dtype=np.float32)

    for step in range(steps):
        desired_vel = pred_delta[:, step, :, :] / dt
        raw_accel = (desired_vel - current_vel) / dt
        constrained_accel = (
            (1 - smoothing_weight) * raw_accel + smoothing_weight * avg_acc[:, 0, :, :]
        )

        accel_norm = np.linalg.norm(constrained_accel, axis=2, keepdims=True)
        accel_scale = np.minimum(1.0, (max_acc / (accel_norm + 1e-8)))
        constrained_accel = constrained_accel * accel_scale

        new_vel = current_vel + constrained_accel * dt
        vel_norm = np.linalg.norm(new_vel, axis=2, keepdims=True)
        vel_scale = np.minimum(1.0, (max_vel / (vel_norm + 1e-8)))
        current_vel = new_vel * vel_scale

        current_pos = current_pos + current_vel * dt
        reconstructed[:, step, :, :] = current_pos

    return reconstructed


def load_data_robust(data_dir, num_agents):
    """加载输入/输出数据对，返回 X, Y 均为 (samples, seq, agents, 3) 格式。
    
    Args:
        data_dir: 目录路径，包含 input_agents_N.npz 和 output_agents_N.npz
        num_agents: Agent 数量
    
    Returns:
        X: (num_samples, seq_in, num_agents, 3)
        Y: (num_samples, seq_out, num_agents, 3)
    """
    data_path = Path(data_dir)
    
    if not data_path.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 加载输入文件
    input_file = data_path / f"input_agents_{num_agents}.npz"
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    logger.info("加载输入数据: %s", input_file)
    input_data = np.load(input_file)
    X_raw = input_data["data"]  # (seq_in=20, agents, coords=3, num_samples)
    logger.info("输入原始形状: %s", X_raw.shape)
    
    # 加载输出文件
    output_file = data_path / f"output_agents_{num_agents}.npz"
    if not output_file.exists():
        raise FileNotFoundError(f"输出文件不存在: {output_file}")
    
    logger.info("加载输出数据: %s", output_file)
    output_data = np.load(output_file)
    Y_raw = output_data["data"]  # (seq_out=10, agents, coords=3, num_samples)
    logger.info("输出原始形状: %s", Y_raw.shape)
    
    # 转置为 (samples, seq, agents, 3)
    # X_raw: (seq_in, samples, agents, coords) -> (samples, seq_in, agents, coords)
    X = np.transpose(X_raw, (1, 0, 2, 3))
    # Y_raw: (seq_out, samples, agents, coords) -> (samples, seq_out, agents, coords)
    Y = np.transpose(Y_raw, (1, 0, 2, 3))
    
    logger.info("转置后 - 输入: %s, 输出: %s", X.shape, Y.shape)
    
    # 验证 agents 数量
    assert X.shape[2] == num_agents, f"输入 agents 数量 {X.shape[2]} != 期望 {num_agents}"
    assert Y.shape[2] == num_agents, f"输出 agents 数量 {Y.shape[2]} != 期望 {num_agents}"
    
    return X, Y


def infer_batch(model, features_batch, x_orig_batch, device, output_mean, output_std, debug=False):
    """Infer one batch and return absolute position predictions.

    Args:
        model: torch.nn.Module
        features_batch: (B, seq_in, agents, 16)
        x_orig_batch: (B, seq_in, agents, 3)
        output_mean: (3,)
        output_std: (3,)
        debug: 是否打印诊断信息

    Returns:
        pred_absolute: np.array (B, seq_out, agents, 3)
    """

    model.eval()
    with torch.no_grad():
        features_t = torch.from_numpy(features_batch).float().to(device)
        x_orig_t = torch.from_numpy(x_orig_batch).float().to(device)

        # model expected to return normalized delta (B, seq_out, agents, 3)
        pred_delta_norm = model(features_t, x_orig_t, y=None, teacher_forcing_ratio=0.0)

        out_mean = torch.tensor(output_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        out_std = torch.tensor(output_std, dtype=torch.float32, device=device).view(1, 1, 1, 3)

        pred_delta_phys = pred_delta_norm * out_std + out_mean

        pred_absolute = apply_physical_constraints(
            x_orig_batch,
            pred_delta_phys.cpu().numpy(),
            dt=0.1,
            smoothing_weight=0.3,
        )

        # 诊断打印（仅第一个批次的第一个样本）
        if debug and features_batch.shape[0] > 0:
            logger.info("=== 推理诊断信息 ===")
            logger.info(f"输入特征形状: {features_batch.shape}")
            logger.info(f"输入位置形状: {x_orig_batch.shape}")
            logger.info(f"模型输出增量（归一化）形状: {pred_delta_norm.shape}")
            logger.info(f"归一化增量范围: [{pred_delta_norm.min().item():.4f}, {pred_delta_norm.max().item():.4f}]")
            if isinstance(output_mean, torch.Tensor):
                mean_tensor = output_mean.view(-1)
            else:
                mean_tensor = torch.tensor(output_mean, dtype=torch.float32, device=device).view(-1)
            if isinstance(output_std, torch.Tensor):
                std_tensor = output_std.view(-1)
            else:
                std_tensor = torch.tensor(output_std, dtype=torch.float32, device=device).view(-1)
            logger.info(f"反归一化参数: mean={mean_tensor.cpu().numpy()}, std={std_tensor.cpu().numpy()}")
            logger.info(f"物理增量范围: [{pred_delta_phys.min().item():.4f}, {pred_delta_phys.max().item():.4f}]")
            logger.info(f"输入最后位置 (sample 0, agent 0): {x_orig_batch[0, -1, 0, :]}")
            logger.info(f"预测绝对位置范围: [{pred_absolute.min():.4f}, {pred_absolute.max():.4f}]")
            logger.info(f"预测绝对位置 (sample 0, agent 0, step 1): {pred_absolute[0, 0, 0, :]}")

    return pred_absolute


def visualize_predictions(X_all, Y_all, predictions, num_samples_vis=10, save_dir="vis_results"):
    """Save 3D plots with predicted step markers for a subset of samples."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("生成可视化到 %s", save_dir)

    idxs = np.random.choice(len(X_all), min(num_samples_vis, len(X_all)), replace=False)
    for idx in idxs:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        x_input = X_all[idx]
        y_true = Y_all[idx]
        y_pred = predictions[idx]

        n_agents = x_input.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

        for a in range(n_agents):
            # 输入历史轨迹（虚线）
            ax.plot(
                x_input[:, a, 0], x_input[:, a, 1], x_input[:, a, 2],
                color=colors[a], linestyle=":", linewidth=1.3, alpha=0.8
            )
            # 输入终点（圆点标记，较大）
            ax.scatter(x_input[-1, a, 0], x_input[-1, a, 1], x_input[-1, a, 2], 
                      color=colors[a], s=80, marker="o", edgecolors="black", linewidth=2, zorder=5)
            
            # 真实未来轨迹（虚线）
            ax.plot(y_true[:, a, 0], y_true[:, a, 1], y_true[:, a, 2], 
                   color=colors[a], linestyle="--", linewidth=2.0, alpha=0.8)
            # 真实轨迹的每一步点（方形标记）
            ax.scatter(y_true[:, a, 0], y_true[:, a, 1], y_true[:, a, 2], 
                      color=colors[a], s=40, marker="s", alpha=0.6, edgecolors="darkgray", linewidth=0.5)
            
            # 预测未来轨迹（虚线连接各预测点）
            ax.plot(y_pred[:, a, 0], y_pred[:, a, 1], y_pred[:, a, 2], 
                   color=colors[a], linestyle="-", linewidth=1.5, alpha=0.5)
            # 预测轨迹的每一步点（圆形标记，有编号）
            for step in range(y_pred.shape[0]):
                ax.scatter(y_pred[step, a, 0], y_pred[step, a, 1], y_pred[step, a, 2], 
                          color=colors[a], s=100, marker="o", edgecolors=colors[a], linewidth=2, zorder=4)
                # 在点附近标注步数
                ax.text(y_pred[step, a, 0], y_pred[step, a, 1], y_pred[step, a, 2], 
                       f"{step+1}", fontsize=8, color=colors[a], fontweight="bold", zorder=6)

        # 轴标签和标题
        ax.set_xlabel("X (m)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Y (m)", fontsize=11, fontweight="bold")
        ax.set_zlabel("Z (m)", fontsize=11, fontweight="bold")
        
        # 添加标题和说明
        title_text = f"Sample {idx}\n"
        title_text += "虚线(...) = 输入历史 | 虚线(--) + 方形 = 真实未来 | 实线 + 圆形+数字 = 预测未来(标注步数) | 大圆 = 输入终点"
        ax.set_title(title_text, fontsize=11, pad=20)

        # 图例
        legend_elements = [
            plt.Line2D([0], [0], color="gray", linestyle=":", linewidth=2, label="输入历史 (20步)"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=6, 
                      linestyle="--", linewidth=1.5, label="真实未来 (10步)", markeredgecolor="gray"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, 
                      linestyle="-", linewidth=1.5, label="预测未来 (10步,含步数编号)", markeredgecolor="gray"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=8, 
                      label="输入终点 (分界点)", linestyle="none", markeredgecolor="black", markeredgewidth=2)
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.95)

        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"pred_{idx:04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("已保存 %d 张可视化图片（带预测步数标注）", len(idxs))


def main():
    parser = argparse.ArgumentParser(description="集群轨迹模型推理")
    parser.add_argument("--model", required=True, help=".pt 模型文件路径")
    parser.add_argument("--data_dir", default="swarm_segments", help="数据目录")
    parser.add_argument("--agents", type=int, default=3, help="Agent 数量")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_samples", type=int, default=22, help="评估样本数，default=100（快速检验），-1 表示全部")
    parser.add_argument("--random_sample", action="store_true", help="是否随机采样样本（适用于大数据集快速检验）")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化")
    parser.add_argument("--output_dir", default="infer_results", help="输出目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    logger.info("加载 checkpoint: %s", args.model)
    checkpoint = torch.load(args.model, map_location=device)

    config = checkpoint.get("config", {})
    if not config:
        logger.warning("Checkpoint 中未包含 config，使用默认设置")

    model = EnhancedSwarmGRUModel(
        input_size=16,
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        output_size=3,
        dropout=0.0,
        use_attention=config.get("use_attention", False),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("模型加载完成")

    # 必要统计量
    if "output_mean" not in checkpoint or "output_std" not in checkpoint:
        raise ValueError("Checkpoint 缺少 output_mean 或 output_std，无法反归一化")

    output_mean = np.array(checkpoint["output_mean"], dtype=np.float32)
    output_std = np.array(checkpoint["output_std"], dtype=np.float32)
    input_mean_all = checkpoint.get("input_mean_all", None)
    input_std_all = checkpoint.get("input_std_all", None)
    input_mean = checkpoint.get("input_mean")
    input_std = checkpoint.get("input_std")

    if input_mean_all is not None and input_std_all is not None:
        input_mean_all = np.array(input_mean_all, dtype=np.float32)
        input_std_all = np.array(input_std_all, dtype=np.float32)
        logger.info("加载特征统计示例 mean[:4]=%s std[:4]=%s", input_mean_all[:4], input_std_all[:4])
        if input_mean is not None and input_std is not None:
            logger.info("载入位置通道的 mean/std")
        else:
            logger.warning("未找到位置通道的 mean/std，位置通道将保持0均值/1标准差")
    else:
        logger.warning("未找到全局特征统计，推理时将跳过全局 Z-score")
        input_mean_all = np.zeros(16, dtype=np.float32)
        input_std_all = np.ones(16, dtype=np.float32)
    if input_mean is None or input_std is None:
        input_mean = np.zeros(3, dtype=np.float32)
        input_std = np.ones(3, dtype=np.float32)

    X_all, Y_all = load_data_robust(args.data_dir, args.agents)
    
    # 随机采样或按顺序截取
    total_samples = len(X_all)
    if args.num_samples > 0 and args.num_samples < total_samples:
        if args.random_sample:
            sample_indices = np.random.choice(total_samples, args.num_samples, replace=False)
            X_all = X_all[sample_indices]
            Y_all = Y_all[sample_indices]
            logger.info("随机采样 %d 个样本（从 %d 个中）", args.num_samples, total_samples)
        else:
            X_all = X_all[: args.num_samples]
            Y_all = Y_all[: args.num_samples]
            logger.info("截取前 %d 个样本（从 %d 个中）", args.num_samples, total_samples)
    logger.info("待评估样本数: %d", len(X_all))

    predictions = []
    for start in tqdm(range(0, len(X_all), args.batch_size), desc="推理进度"):
        end = min(start + args.batch_size, len(X_all))
        batch_X = X_all[start:end]

        batch_feats = [
            compute_features_for_inference(
                x,
                input_mean_all,
                input_std_all,
                input_mean=input_mean,
                input_std=input_std,
            )
            for x in batch_X
        ]
        batch_feats = np.stack(batch_feats, axis=0)

        # 仅在第一个批次启用诊断
        debug_flag = (start == 0)
        batch_pred = infer_batch(model, batch_feats, batch_X, device, output_mean, output_std, debug=debug_flag)
        predictions.append(batch_pred)

    predictions = np.concatenate(predictions, axis=0)
    logger.info("推理完成，预测形状: %s", predictions.shape)

    mae = np.mean(np.abs(predictions - Y_all))
    rmse = np.sqrt(np.mean((predictions - Y_all) ** 2))
    mae_x = np.mean(np.abs(predictions[..., 0] - Y_all[..., 0]))
    mae_y = np.mean(np.abs(predictions[..., 1] - Y_all[..., 1]))
    mae_z = np.mean(np.abs(predictions[..., 2] - Y_all[..., 2]))

    mae_per_step = np.mean(np.abs(predictions - Y_all), axis=(0, 2, 3))
    mae_per_agent = np.mean(np.abs(predictions - Y_all), axis=(0, 1, 3))

    logger.info("总体 MAE: %.6f m, RMSE: %.6f m", mae, rmse)
    logger.info("MAE (X/Y/Z): %.6f / %.6f / %.6f m", mae_x, mae_y, mae_z)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / f"predictions_agents_{args.agents}.npz"
    np.savez(
        result_file,
        input=X_all,
        truth=Y_all,
        prediction=predictions,
        mae=mae,
        rmse=rmse,
        mae_per_step=mae_per_step,
        mae_per_agent=mae_per_agent,
    )
    logger.info("结果已保存: %s", result_file)

    report_file = out_dir / f"evaluation_report_agents_{args.agents}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("集群轨迹推理评估报告\n")
        f.write(f"模型: {Path(args.model).name}\n")
        f.write(f"Agent 数: {args.agents}\n")
        f.write(f"样本数: {len(X_all)}\n")
        f.write("\n总体指标:\n")
        f.write(f"  MAE: {mae:.6f} m\n")
        f.write(f"  RMSE: {rmse:.6f} m\n")
        f.write("\n分轴 MAE:\n")
        f.write(f"  X: {mae_x:.6f} m\n")
        f.write(f"  Y: {mae_y:.6f} m\n")
        f.write(f"  Z: {mae_z:.6f} m\n")
        f.write("\n分时步 MAE:\n")
        for t, v in enumerate(mae_per_step):
            f.write(f"  Step {t+1}: {v:.6f} m\n")
        f.write("\n分 Agent MAE:\n")
        for a, v in enumerate(mae_per_agent):
            f.write(f"  Agent {a}: {v:.6f} m\n")

    logger.info("评估报告已保存: %s", report_file)

    if args.visualize:
        visualize_predictions(X_all, Y_all, predictions, num_samples_vis=10, save_dir=out_dir / "visualizations")

    logger.info("推理流程完成")


if __name__ == "__main__":
    main()
