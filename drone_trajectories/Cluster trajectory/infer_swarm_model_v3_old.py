#!/usr/bin/env python3
"""
v3 Swarm Trajectory Inference Script
模仿v2的成熟架构，集成GNN图神经网络功能
支持预计算特征和实时特征双模式，确保与训练时的一致性
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

# 将项目根加入路径，确保可以导入 train_swarm_model_v2_dynamics_aware
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from train_swarm_model_v2_dynamics_aware import (
        DynamicsAwareSwarmGRUModel,
        compute_features_enhanced_24d,
    )
except ImportError as e:
    logger.error("无法导入 train_swarm_model_v2_dynamics_aware.py，请确保该文件在同一目录或 PYTHONPATH 中：%s", e)
    raise


def compute_features_for_inference(
    trajectory,
    feature_mean_all=None,
    feature_std_all=None,
    dt=0.1,
):
    """计算 24D 特征以供推理使用，与训练预处理保持一致

    Args:
        trajectory: np.ndarray, (seq_in, agents, 3) 轨迹
        feature_mean_all: optional (24,) 全局均值
        feature_std_all: optional (24,) 全局标准差
        dt: float, 采样间隔

    Returns:
        features: np.float32 array, shape (seq_in, agents, 24)
    """
    # 计算 24D 特征（与训练时完全相同）
    features = compute_features_enhanced_24d(trajectory, dt=dt)  # (seq_in, agents, 24)

    # 全局特征归一化（使用训练时计算的全局统计量）
    if feature_mean_all is not None and feature_std_all is not None:
        mean_vec = np.array(feature_mean_all, dtype=np.float32).reshape(1, 1, 24)
        std_vec = np.array(feature_std_all, dtype=np.float32).reshape(1, 1, 24)
        std_vec = np.where(std_vec < 1e-8, 1.0, std_vec)
        features = (features - mean_vec) / (std_vec + 1e-8)

    features = np.clip(features, -5.0, 5.0)
    return features.astype(np.float32)


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
        # 最后尝试通过 pickle 解包（谨慎）
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


def estimate_feature_stats_from_data(X, dt=0.1, num_samples=100, seed=42):
    """从样本轨迹估算 24D 特征的全局均值与标准差"""
    X = np.asarray(X)
    if X.size == 0:
        return np.zeros(24, dtype=np.float32), np.ones(24, dtype=np.float32)

    rng = np.random.RandomState(seed)
    sample_count = min(num_samples, len(X))
    if sample_count <= 0:
        sample_count = 1
    indices = rng.choice(len(X), sample_count, replace=False)

    feature_chunks = []
    for idx in indices:
        features = compute_features_enhanced_24d(X[idx], dt=dt)  # (seq, agents, 24)
        feature_chunks.append(features.reshape(-1, 24))

    if not feature_chunks:
        return np.zeros(24, dtype=np.float32), np.ones(24, dtype=np.float32)

    stacked = np.concatenate(feature_chunks, axis=0)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def apply_physical_constraints(history, pred_delta, dt=0.1, smoothing_weight=0.3):
    """应用物理约束以实现更平滑的位置重建"""
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
    X_raw = input_data["data"]  # (seq_in=20, agents, coords=3, num_samples) 或其他形状
    logger.info("输入原始形状: %s", X_raw.shape)
    
    # 加载输出文件
    output_file = data_path / f"output_agents_{num_agents}.npz"
    if not output_file.exists():
        raise FileNotFoundError(f"输出文件不存在: {output_file}")
    
    logger.info("加载输出数据: %s", output_file)
    output_data = np.load(output_file)
    Y_raw = output_data["data"]  # (seq_out=10, agents, coords=3, num_samples) 或其他形状
    logger.info("输出原始形状: %s", Y_raw.shape)
    
    # 转置为 (samples, seq, agents, 3)
    # 假设原始格式为：(seq_in, samples, agents, coords) -> (samples, seq_in, agents, coords)
    X = np.transpose(X_raw, (1, 0, 2, 3))
    Y = np.transpose(Y_raw, (1, 0, 2, 3))
    
    logger.info("转置后 - 输入: %s, 输出: %s", X.shape, Y.shape)
    
    # 验证 agents 数量
    assert X.shape[2] == num_agents, f"输入 agents 数量 {X.shape[2]} != 期望 {num_agents}"
    assert Y.shape[2] == num_agents, f"输出 agents 数量 {Y.shape[2]} != 期望 {num_agents}"
    
    return X, Y


def infer_batch(model, features_batch, x_orig_batch, device, output_mean, output_std, debug=False):
    """推理一个批次并返回绝对位置预测

    Args:
        model: torch.nn.Module
        features_batch: (B, seq_in, agents, 24) 24D特征输入
        x_orig_batch: (B, seq_in, agents, 3) 原始位置输入
        output_mean: (3,) 输出增量均值
        output_std: (3,) 输出增量标准差
        debug: 是否打印诊断信息

    Returns:
        pred_absolute: np.array (B, seq_out, agents, 3) 绝对位置预测
    """

    model.eval()
    with torch.no_grad():
        features_t = torch.from_numpy(features_batch).float().to(device)
        x_orig_t = torch.from_numpy(x_orig_batch).float().to(device)

        # model 返回归一化的位置增量（B, seq_out, agents, 3）
        # 以及 pred_vel 和 pred_accel（用于多任务学习，推理时忽略）
        pred_delta_norm, _, _ = model(features_t, x_orig_t, y=None, y_velocity=None, y_accel=None, teacher_forcing_ratio=0.0)

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


def main():
    parser = argparse.ArgumentParser(description="集群轨迹模型推理 (v2 24D特征)")
    parser.add_argument("--model", required=True, help=".pt 模型文件路径")
    parser.add_argument("--data_dir", default="swarm_segments", help="数据目录")
    parser.add_argument("--agents", type=int, default=3, help="Agent 数量")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_samples", type=int, default=22, help="评估样本数，-1 表示全部")
    parser.add_argument("--random_sample", action="store_true", help="是否随机采样样本")
    parser.add_argument("--output_dir", default="infer_results_v2", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    logger.info("加载 checkpoint: %s", args.model)
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    except TypeError:
        # 备选方案：旧版本 PyTorch 不支持 weights_only 参数
        checkpoint = torch.load(args.model, map_location=device)

    config = checkpoint.get("config", {})
    if not config:
        logger.warning("Checkpoint 中未包含 config，使用默认设置")

    model = DynamicsAwareSwarmGRUModel(
        input_size=24,
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        output_size=3,
        dropout=0.0,
        use_attention=config.get("use_attention", True),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("模型加载完成")

    # 必要统计量
    if "output_mean" not in checkpoint or "output_std" not in checkpoint:
        raise ValueError("Checkpoint 缺少 output_mean 或 output_std，无法反归一化")

    output_mean = np.array(checkpoint["output_mean"], dtype=np.float32)
    output_std = np.array(checkpoint["output_std"], dtype=np.float32)
    
    logger.info("✓ 从 checkpoint 加载输出统计量: output_mean=%s, output_std=%s", output_mean, output_std)

    # 加载数据（后续用于特征统计和推理）
    logger.info("加载数据: %s", args.data_dir)
    X_all, Y_all = load_data_robust(args.data_dir, args.agents)
    stats_sample_count = min(200, len(X_all))
    if stats_sample_count <= 0:
        stats_sample_count = 1

    # 加载 24D 特征统计量（与 16D 版本的方式相同，但改为 24 维）
    feature_mean_all = None
    feature_std_all = None

    # 优先尝试从 .npz 文件加载（最准确）
    stats_path = Path(args.model).parent / f'norm_stats_agents_{args.agents}_v2.npz'
    if not stats_path.exists():
        # 尝试没有 _v2 后缀的版本
        stats_path = Path(args.model).parent / f'norm_stats_agents_{args.agents}.npz'

    if stats_path.exists():
        logger.info("优先：从 %s 加载特征统计...", stats_path.name)
        try:
            stats_file = np.load(stats_path, allow_pickle=True)

            # 支持多种键名：优先 feature_mean/feature_std，其次 input_mean_all/input_std_all 或 input_mean/input_std
            raw_mean = None
            raw_std = None
            if 'feature_mean' in stats_file and 'feature_std' in stats_file:
                raw_mean = stats_file['feature_mean']
                raw_std = stats_file['feature_std']
            else:
                raw_mean = stats_file.get('input_mean_all', stats_file.get('input_mean', None))
                raw_std = stats_file.get('input_std_all', stats_file.get('input_std', None))

            feature_mean_all = _extract_array_from_npz_field(raw_mean, expected_len=24)
            feature_std_all = _extract_array_from_npz_field(raw_std, expected_len=24)

            if feature_mean_all is not None and feature_std_all is not None:
                logger.info("✓ 特征统计加载成功 (来自.npz文件):")
                logger.info("  feature_mean shape: %s, 前4个: %s", feature_mean_all.shape, feature_mean_all[:4])
                logger.info("  feature_std shape: %s, 前4个: %s", feature_std_all.shape, feature_std_all[:4])
            else:
                logger.warning("⚠️ .npz 文件中未找到或无法解析 24D 特征统计（期待 shape=(24,)），将尝试从 checkpoint 或退回默认值")
                feature_mean_all = None
                feature_std_all = None
        except Exception as e:
            logger.warning("⚠️ 加载 .npz 文件失败: %s", e)
            feature_mean_all = None
            feature_std_all = None
    
    # 备选：从 checkpoint 加载特征统计量
    if feature_mean_all is None or feature_std_all is None:
        if 'feature_mean' in checkpoint and 'feature_std' in checkpoint:
            logger.info("备选：从 checkpoint 加载特征统计量...")
            feature_mean_all = np.array(checkpoint['feature_mean'])
            feature_std_all = np.array(checkpoint['feature_std'])
            logger.info("✓ 特征统计加载成功 (来自checkpoint):")
            logger.info("  feature_mean shape: %s", feature_mean_all.shape)
            logger.info("  feature_std shape: %s", feature_std_all.shape)
        else:
            if len(X_all) == 0:
                logger.warning("  ⚠️ 未找到特征统计量且数据为空，使用零均值单位方差")
                feature_mean_all = np.zeros(24, dtype=np.float32)
                feature_std_all = np.ones(24, dtype=np.float32)
            else:
                logger.info("  备选：从数据估算 24D 特征统计量（采样 %d 个序列）", stats_sample_count)
                feature_mean_all, feature_std_all = estimate_feature_stats_from_data(
                    X_all,
                    dt=0.1,
                    num_samples=stats_sample_count,
                    seed=args.seed,
                )
                logger.info("  ✓ 估算特征统计完成: mean shape=%s, std shape=%s", feature_mean_all.shape, feature_std_all.shape)

    # 样本选择
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
                feature_mean_all,
                feature_std_all,
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

    logger.info("\n=== 评估结果 ===")
    logger.info("总体 MAE: %.6f m (%.2f cm), RMSE: %.6f m", mae, mae*100, rmse)
    logger.info("MAE (X/Y/Z): %.6f / %.6f / %.6f m", mae_x, mae_y, mae_z)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / f"predictions_agents_{args.agents}_v2.npz"
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

    report_file = out_dir / f"evaluation_report_agents_{args.agents}_v2.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("集群轨迹推理评估报告 (v2 24D 特征)\n")
        f.write(f"模型: {Path(args.model).name}\n")
        f.write(f"Agent 数: {args.agents}\n")
        f.write(f"样本数: {len(X_all)}\n")
        f.write("\n总体指标:\n")
        f.write(f"  MAE: {mae:.6f} m ({mae*100:.2f} cm)\n")
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
    logger.info("推理流程完成")


if __name__ == "__main__":
    main()