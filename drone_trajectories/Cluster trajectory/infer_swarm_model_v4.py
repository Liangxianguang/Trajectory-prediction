#!/usr/bin/env python3
"""
v4 集群轨迹推断脚本（32D特征版本）
=========================================

相比 v3 的改进：
✅ 支持 32D 预计算特征（24D + 8D 曲率）
✅ 自动检测模型版本（v2/v3/v4）
✅ 支持预计算特征推理（从 features_32d 目录加载）
✅ 完整的特征统计量加载管理
✅ 物理约束和多任务目标输出

架构：
    输入位置 → [GNN（可选）] → 32D 特征 → BiGRU 编码 → 多分支解码 → 绝对位置预测

使用示例：
    # v4 with GNN 和 32D 特征
    python infer_swarm_model_v4.py ^
        --model gru_models_v4_fixed_agents_3/best_model_v4_agents_3.pt ^
        --agents 3 --output_dir infer_results_v4 --features_dir features_32d
    
    # v4 不用 GNN
    python infer_swarm_model_v4.py ^
        --model gru_models_v4_fixed_agents_3/best_model_v4_agents_3.pt ^
        --agents 3 --output_dir infer_results_v4 --no_gnn --features_dir features_32d
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

# 将项目根加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from train_swarm_model_v2_dynamics_aware import (
        DynamicsAwareSwarmGRUModel,
    )
    from train_swarm_model_v3_with_gnn import (
        DynamicsAwareSwarmGRUModel_with_GNN,
        build_adjacency_from_positions,
    )
except ImportError as e:
    logger.error("无法导入必要的模块：%s", e)
    raise


def load_all_32d_features(features_dir, num_agents, use_subset=False):
    """
    一次性将所有32D特征加载到内存中，避免重复读取硬盘
    """
    features_dir = Path(features_dir)
    subset_suffix = '_subset' if use_subset else ''
    
    feature_candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_32d.npz',
        features_dir / f'features_agents_{num_agents}_32d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
    ]
    
    for feat_path in feature_candidates:
        if feat_path.exists():
            try:
                logger.info(f"正在预加载特征文件: {feat_path} ...")
                data = np.load(feat_path)
                features_all = np.asarray(data['features'])
                logger.info(f"✓ 特征预加载完成: {features_all.shape}")
                return features_all
            except Exception as e:
                logger.warning(f"预加载特征文件失败 {feat_path}: {e}")
    return None


def load_32d_features_for_sample(features_dir, num_agents, sample_idx, seq_in=20, use_subset=False):
    """
    从预计算的32D特征文件加载单个样本的特征 (由于效率原因，建议使用全局加载方案)
    """
    # 保持原函数签名以备参考，但内部优化为一次性读取
    features_dir = Path(features_dir)
    subset_suffix = '_subset' if use_subset else ''
    
    # 尝试加载预计算特征
    feature_candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_32d.npz',
        features_dir / f'features_agents_{num_agents}_32d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
    ]
    
    for feat_path in feature_candidates:
        if feat_path.exists():
            try:
                data = np.load(feat_path)
                features_all = np.asarray(data['features'])  # (samples, seq_in, agents, 32)
                
                # 确保形状正确
                if features_all.ndim == 4 and sample_idx < features_all.shape[0]:
                    return features_all[sample_idx].astype(np.float32)
                else:
                    logger.warning(f"特征形状不正确: {features_all.shape}, 期望 (samples, seq_in, agents, 32)")
                    continue
            except Exception as e:
                logger.warning(f"加载特征文件失败 {feat_path}: {e}")
                continue
    
    return None


def compute_features_32d_for_inference(
    trajectory,
    feature_mean=None,
    feature_std=None,
    dt=0.1,
):
    """
    计算 32D 特征用于推理
    
    注意：该函数是从预计算特征文件加载，如果需要实时计算，需要实现特征提取函数
    
    Args:
        trajectory: np.ndarray, (seq_in, agents, 3) 轨迹
        feature_mean: optional (32,) 全局均值
        feature_std: optional (32,) 全局标准差
        dt: float, 采样间隔
    
    Returns:
        features: np.float32 array, shape (seq_in, agents, 32)
    """
    # v4 支持预计算特征，这里主要是归一化
    # 实际推理时特征应从预计算文件加载
    
    if feature_mean is not None and feature_std is not None:
        mean_vec = np.array(feature_mean, dtype=np.float32).reshape(1, 1, 32)
        std_vec = np.array(feature_std, dtype=np.float32).reshape(1, 1, 32)
        std_vec = np.where(std_vec < 1e-8, 1.0, std_vec)
        features = (features - mean_vec) / (std_vec + 1e-8)
    
    features = np.clip(features, -5.0, 5.0)
    return features.astype(np.float32)


def _extract_array_from_npz_field(field, expected_len=None):
    """从 npz 字段中安全提取 numpy 数组"""
    if field is None:
        return None
    try:
        arr = np.asarray(field)
    except Exception:
        try:
            arr = np.array(field.tolist())
        except Exception:
            return None
    
    if arr.ndim == 0:
        try:
            arr = np.array([arr.item()])
        except Exception:
            pass
    elif arr.ndim == 1 and arr.size == 1:
        inner = arr[0]
        if isinstance(inner, (list, tuple, np.ndarray)):
            arr = np.asarray(inner)
    
    if arr.ndim > 1:
        arr = arr.ravel()
    
    if expected_len is not None:
        try:
            arr = arr[:expected_len]
        except Exception:
            pass
    
    return arr.astype(np.float32)


def estimate_feature_stats_v4_from_data(features_dir, num_agents, num_samples=100, seed=42, use_subset=False):
    """从预计算特征文件估算统计量
    
    特征文件格式：(num_samples, seq_in, agents, 32)
    """
    features_dir = Path(features_dir)
    subset_suffix = '_subset' if use_subset else ''
    
    feature_candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_32d.npz',
        features_dir / f'features_agents_{num_agents}_32d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
    ]
    
    for feat_path in feature_candidates:
        if feat_path.exists():
            try:
                data = np.load(feat_path)
                features_all = np.asarray(data['features'])  # (num_samples, seq_in, agents, 32)
                
                if features_all.size == 0:
                    continue
                
                # 确保形状为 (num_samples, seq_in, agents, 32)
                if features_all.ndim == 4:
                    num_total_samples = features_all.shape[0]
                elif features_all.ndim == 1:
                    # 可能被 squeeze 了，跳过
                    continue
                else:
                    # 尝试 reshape
                    try:
                        features_all = features_all.reshape(-1, 20, 3, 32)
                        num_total_samples = features_all.shape[0]
                    except ValueError:
                        continue
                
                rng = np.random.RandomState(seed)
                sample_count = min(num_samples, num_total_samples)
                indices = rng.choice(num_total_samples, sample_count, replace=False)
                
                # 提取样本并 reshape 为 (N, 32)
                feature_chunks = []
                for idx in indices:
                    # (seq_in, agents, 32) -> (seq_in * agents, 32)
                    feat = features_all[idx]  # (seq_in, agents, 32)
                    feat = feat.reshape(-1, 32)
                    feature_chunks.append(feat)
                
                stacked = np.concatenate(feature_chunks, axis=0)  # (N*seq_in*agents, 32)
                mean = np.mean(stacked, axis=0)  # (32,)
                std = np.std(stacked, axis=0)   # (32,)
                std = np.where(std < 1e-8, 1.0, std)
                
                logger.info(f"✓ 从 {feat_path.name} 加载特征统计: shape={mean.shape}, samples={sample_count}")
                return mean.astype(np.float32), std.astype(np.float32)
            except Exception as e:
                logger.warning(f"加载特征统计失败 {feat_path}: {e}")
                continue
    
    logger.warning("未找到预计算特征文件，使用默认统计")
    return np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32)


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


def load_data_robust(data_dir, num_agents, use_subset=False):
    """加载输入/输出数据对"""
    data_path = Path(data_dir)
    
    if not data_path.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    subset_suffix = '_subset' if use_subset else ''
    input_file = data_path / f"input_agents_{num_agents}{subset_suffix}.npz"
    output_file = data_path / f"output_agents_{num_agents}{subset_suffix}.npz"
    
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    if not output_file.exists():
        raise FileNotFoundError(f"输出文件不存在: {output_file}")
    
    logger.info("加载输入数据: %s", input_file)
    input_data = np.load(input_file)
    X_raw = input_data["data"]
    logger.info("输入原始形状: %s", X_raw.shape)
    
    logger.info("加载输出数据: %s", output_file)
    output_data = np.load(output_file)
    Y_raw = output_data["data"]
    logger.info("输出原始形状: %s", Y_raw.shape)
    
    # 转置为 (samples, seq, agents, 3)
    X = np.transpose(X_raw, (1, 0, 2, 3))
    Y = np.transpose(Y_raw, (1, 0, 2, 3))
    
    logger.info("转置后 - 输入: %s, 输出: %s", X.shape, Y.shape)
    
    assert X.shape[2] == num_agents, f"输入 agents 数量 {X.shape[2]} != 期望 {num_agents}"
    assert Y.shape[2] == num_agents, f"输出 agents 数量 {Y.shape[2]} != 期望 {num_agents}"
    
    return X, Y


def detect_model_version(checkpoint):
    """检测模型版本（v2/v3/v4）"""
    config = checkpoint.get("config", {})
    
    # 优先检查 config 中的 model_version
    if 'model_version' in config:
        model_version = str(config['model_version']).lower()
        if 'v4' in model_version:
            return 'v4'
        elif 'v3' in model_version:
            return 'v3'
        elif 'v2' in model_version:
            return 'v2'
    
    # 检查特征维度
    if 'input_features' in config:
        input_dim = config['input_features']
        if input_dim == 32:
            return 'v4'
    
    # 检查 use_gnn 标志
    if 'use_gnn' in config:
        return 'v3' if config['use_gnn'] else 'v2'
    
    # 检查 GNN 相关参数
    if any(key in config for key in ['gnn_hidden', 'gnn_heads', 'edge_threshold']):
        return 'v3'
    
    # 默认 v2
    return 'v2'


def infer_batch_v2(model, features_batch, x_orig_batch, device, output_mean, output_std, debug=False):
    """推理一个批次 (v2) 并返回绝对位置预测"""
    model.eval()
    with torch.no_grad():
        features_t = torch.from_numpy(features_batch).float().to(device)
        x_orig_t = torch.from_numpy(x_orig_batch).float().to(device)

        pred_delta_norm, _, _ = model(features_t, x_orig_t, y=None, y_velocity=None, y_accel=None, teacher_forcing_ratio=0.0)
        
        output_mean_t = torch.tensor(output_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        output_std_t = torch.tensor(output_std, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        
        # 反归一化
        pred_delta_phys = pred_delta_norm * output_std_t + output_mean_t
        
        # 应用物理约束重构
        pred_absolute = apply_physical_constraints(
            x_orig_batch,
            pred_delta_phys.cpu().numpy(),
            dt=0.1,
            smoothing_weight=0.3
        )

    return pred_absolute


def infer_batch_v3(model, features_batch, x_orig_batch, device, output_mean, output_std, 
                   edge_threshold=5.0, debug=False):
    """推理一个批次 (v3 with GNN) 并返回绝对位置预测"""
    model.eval()
    with torch.no_grad():
        features_t = torch.from_numpy(features_batch).float().to(device)
        x_orig_t = torch.from_numpy(x_orig_batch).float().to(device)

        # v3/v4 模型在 forward Pass 中会自动计算邻接矩阵，无需外部传入
        pred_delta_norm, _, _ = model(
            features_t, x_orig_t,
            y=None, y_velocity=None, y_accel=None,
            teacher_forcing_ratio=0.0
        )
        
        output_mean_t = torch.tensor(output_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        output_std_t = torch.tensor(output_std, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        
        pred_delta_phys = pred_delta_norm * output_std_t + output_mean_t
        
        # 应用物理约束重构
        pred_absolute = apply_physical_constraints(
            x_orig_batch,
            pred_delta_phys.cpu().numpy(),
            dt=0.1,
            smoothing_weight=0.3
        )

    return pred_absolute


def infer_batch_v4(model, features_batch, x_orig_batch, device, output_mean, output_std, 
                   edge_threshold=5.0, use_gnn=True, debug=False):
    """推理一个批次 (v4 with 32D特征) 并返回绝对位置预测"""
    model.eval()
    with torch.no_grad():
        features_t = torch.from_numpy(features_batch).float().to(device)
        x_orig_t = torch.from_numpy(x_orig_batch).float().to(device)

        # 判断是否真的有 GNN
        has_gnn = use_gnn and hasattr(model, 'gnn')
        
        # 模型在 forward Pass 中会自动计算邻接矩阵
        pred_delta_norm, _, _ = model(
            features_t, x_orig_t,
            y=None, y_velocity=None, y_accel=None,
            teacher_forcing_ratio=0.0
        )
        
        output_mean_t = torch.tensor(output_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        output_std_t = torch.tensor(output_std, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        
        # 反归一化得到物理位移 (B, seq_out, agents, 3)
        # 模型训练时的目标是总位移 y_delta = Y_t - X_last
        pred_delta_phys = (pred_delta_norm * output_std_t + output_mean_t).cpu().numpy()
        
        # 重建绝对位置：last_pos + pred_delta
        last_pos = x_orig_batch[:, -1, :, np.newaxis, :]  # (B, 1, agents, 3)
        pred_absolute = last_pos + pred_delta_phys

    return pred_absolute


def main():
    parser = argparse.ArgumentParser(description="集群轨迹模型推理 (v4 - 32D特征版本)")
    parser.add_argument("--model", required=True, help=".pt 模型文件路径")
    parser.add_argument("--data_dir", default="swarm_segments", help="数据目录")
    parser.add_argument("--agents", type=int, default=3, help="Agent 数量")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_samples", type=int, default=22, help="评估样本数，-1 表示全部")
    parser.add_argument("--random_sample", action="store_true", help="是否随机采样样本")
    parser.add_argument("--output_dir", default="infer_results_v4", help="输出目录")
    parser.add_argument("--features_dir", type=str, default="features_32d", help="32D特征目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--force_v4", action="store_true", help="强制使用 v4 模型")
    parser.add_argument("--no_gnn", action="store_true", help="不使用 GNN")
    parser.add_argument("--use_subset", action="store_true", help="使用 _subset 数据")
    parser.add_argument("--edge_threshold", type=float, default=5.0, help="GNN 邻接阈值")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    logger.info("加载 checkpoint: %s", args.model)
    try:
        checkpoint = torch.load(args.model, map_location='cpu')
    except TypeError:
        checkpoint = torch.load(args.model)

    config = checkpoint.get("config", {})
    if not config:
        logger.warning("Checkpoint 中没有 config，假设为 v4 模型")
        config = {}

    # 检测模型版本
    if args.force_v4:
        model_version = 'v4'
    else:
        model_version = detect_model_version(checkpoint)
    
    logger.info(f"检测到模型版本: {model_version}")

    # 创建模型
    if model_version in ['v3', 'v4'] and not args.no_gnn:
        use_gnn = True
        model = DynamicsAwareSwarmGRUModel_with_GNN(
            input_size=config.get('input_size', 32),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 3),
            gnn_hidden=config.get('gnn_hidden', 64),
            gnn_heads=config.get('gnn_heads', 4),
            fusion_mode=config.get('gnn_fusion_mode', 'concat'),
        )
    else:
        use_gnn = False
        model = DynamicsAwareSwarmGRUModel(
            input_size=config.get('input_size', 32),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 3),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("模型加载完成")

    # 加载统计量
    if "output_mean" not in checkpoint or "output_std" not in checkpoint:
        logger.error("Checkpoint 中缺少 output_mean 或 output_std")
        sys.exit(1)

    output_mean = np.array(checkpoint["output_mean"], dtype=np.float32)
    output_std = np.array(checkpoint["output_std"], dtype=np.float32)
    logger.info("✓ 从 checkpoint 加载输出统计量: output_mean=%s, output_std=%s", output_mean, output_std)

    # 加载数据
    logger.info("加载数据: %s (use_subset=%s)", args.data_dir, args.use_subset)
    X_all, Y_all = load_data_robust(args.data_dir, args.agents, use_subset=args.use_subset)

    # 加载 32D 特征统计量
    feature_mean_all = None
    feature_std_all = None
    
    features_all_cache = None
    if args.features_dir:
        # 优化：一次性加载所有特征文件，避免在循环中重复 np.load
        features_all_cache = load_all_32d_features(args.features_dir, args.agents, use_subset=args.use_subset)
        
        # 既然已经加载了特征，直接用它来计算/获取统计量
        if features_all_cache is not None:
            # 简单粗暴计算均值标准差（或者你之前已经保存了也可以加载，这里为了效率）
            # 注意：如果特征量很大，这里还是建议通过 estimate 函数，但 estimate 函数也应该改用 cache
            subset_for_stats = features_all_cache[:min(1000, len(features_all_cache))].reshape(-1, 32)
            feature_mean_all = np.mean(subset_for_stats, axis=0)
            feature_std_all = np.std(subset_for_stats, axis=0)
            logger.info("✓ 使用预加载特征计算统计量")
        else:
            feature_mean_all, feature_std_all = estimate_feature_stats_v4_from_data(
                args.features_dir, args.agents, use_subset=args.use_subset
            )

    # 样本选择
    total_samples = len(X_all)
    if args.num_samples > 0 and args.num_samples < total_samples:
        if args.random_sample:
            indices = np.random.choice(total_samples, args.num_samples, replace=False)
        else:
            indices = np.arange(args.num_samples)
        X_all = X_all[indices]
        Y_all = Y_all[indices]
        # 同步更新特征缓存索引
        if features_all_cache is not None:
            features_all_cache = features_all_cache[indices]
    
    logger.info("待评估样本数: %d", len(X_all))

    # 推理
    predictions = []
    for start in tqdm(range(0, len(X_all), args.batch_size), desc="推理进度"):
        end = min(start + args.batch_size, len(X_all))
        batch_indices = np.arange(start, end)
        
        X_batch = X_all[start:end]
        
        # 加载 32D 特征 (改用缓存)
        if features_all_cache is not None:
            features_batch = features_all_cache[start:end].astype(np.float32)
        else:
            # 备选方案：逐个加载（虽然慢，但脚本结构更健壮）
            features_batch_list = []
            for idx in batch_indices:
                feat = load_32d_features_for_sample(
                    args.features_dir, args.agents, idx, 
                    seq_in=X_batch.shape[1],
                    use_subset=args.use_subset
                )
                if feat is None:
                    logger.error(f"无法加载样本 {idx} 的特征")
                    sys.exit(1)
                features_batch_list.append(feat)
            features_batch = np.array(features_batch_list)
        
        # 推理
        if model_version == 'v4':
            pred_batch = infer_batch_v4(
                model, features_batch, X_batch, device,
                output_mean, output_std,
                edge_threshold=args.edge_threshold,
                use_gnn=use_gnn
            )
        elif model_version == 'v3':
            pred_batch = infer_batch_v3(
                model, features_batch, X_batch, device,
                output_mean, output_std,
                edge_threshold=args.edge_threshold
            )
        else:
            pred_batch = infer_batch_v2(
                model, features_batch, X_batch, device,
                output_mean, output_std
            )
        
        predictions.append(pred_batch)

    predictions = np.concatenate(predictions, axis=0)
    logger.info("推理完成，预测形状: %s", predictions.shape)

    # 评估
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

    # 保存结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = out_dir / f"predictions_agents_{args.agents}_{model_version}.npz"
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

    report_file = out_dir / f"evaluation_report_agents_{args.agents}_{model_version}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("集群轨迹预测评估报告 (v4 - 32D特征版本)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"模型版本: {model_version}\n")
        f.write(f"模型路径: {args.model}\n")
        f.write(f"Agent 数量: {args.agents}\n")
        f.write(f"样本数: {len(X_all)}\n")
        f.write(f"使用 GNN: {use_gnn}\n")
        f.write(f"特征维度: 32D (24D + 8D 曲率特征)\n\n")
        f.write(f"总体 MAE: {mae:.6f} m ({mae*100:.2f} cm)\n")
        f.write(f"总体 RMSE: {rmse:.6f} m\n")
        f.write(f"MAE (X): {mae_x:.6f} m\n")
        f.write(f"MAE (Y): {mae_y:.6f} m\n")
        f.write(f"MAE (Z): {mae_z:.6f} m\n\n")
        f.write("MAE per step:\n")
        for step, mae_step in enumerate(mae_per_step):
            f.write(f"  步 {step}: {mae_step:.6f} m\n")

    logger.info("评估报告已保存: %s", report_file)
    logger.info("推理流程完成")


if __name__ == "__main__":
    main()
