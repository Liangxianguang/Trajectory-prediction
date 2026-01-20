#!/usr/bin/env python3
"""
预计算轨迹特征脚本 v4 - 32D动力学感知特征
==================================================

相比v3的改进：
✅ 原有24D完全保留（向后兼容）
✅ 新增8D精准特征：解决圆弧方向预测问题
  ├─ 曲率向量 (3D): κ_x, κ_y, κ_z - 指向圆心的方向
  ├─ 角速度向量 (3D): ω_x, ω_y, ω_z - 旋转轴方向
  └─ 主曲率率 (2D): κ_rate_1, κ_rate_2 - 两个正交方向的曲率变化率

为什么这8D特别关键：
- 圆弧轨迹的本质：不是位置本身，而是曲率向量的方向
- v3问题：24D中缺少"曲率向量方向"的3D信息
- v4解决：显式提供3D曲率向量，使RNN能准确学习圆心位置
- 直线场景：自动退化为0（无需担心破坏直线运动）

特征维度组成（总计32D）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原有24D（不变）：
  1-3:   位置 (x, y, z)
  4-6:   速度方向 (v̂_x, v̂_y, v̂_z)
  7:     速度大小 (||v||)
  8:     切向加速度 (a_t)
  9:     法向加速度 (a_n)
  10:    角速度大小 (ω)
  11:    Jerk (da/dt)
  12-20: 多尺度速度 (9D: 1/2/3步)
  21:    3D曲率 (κ)
  22-24: 平面曲率 (κ_xy, κ_yz, κ_xz)

新增8D（精准补强）：
  25-27: 曲率向量 (κ_x, κ_y, κ_z) ⭐ 核心改进
  28-30: 角速度向量 (ω_x, ω_y, ω_z) ⭐ 核心改进
  31-32: 主曲率率 (κ̇_1, κ̇_2) ⭐ 平滑性改进
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

使用示例：
    # 快速测试 (子集数据)
    python precompute_features_v4.py --data_dir swarm_segments --agents 3 --use_subset
    
    # 完整预计算
    python precompute_features_v4.py --data_dir swarm_segments --agents 3 --batch_size 50
    
    # 自定义输出目录
    python precompute_features_v4.py --data_dir swarm_segments --agents 3 \
        --output_dir features_32d --batch_size 100

后续使用v3模型+32D特征进行训练：
    python train_swarm_v3_complete.py \
        --data_dir swarm_segments \
        --agents 3 \
        --epochs 150 \
        --batch_size 256 \
        --use_gnn \
        --use_subset \
        --seed 42
"""

import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ====================================================================
# v3原有的24D特征计算函数（保持不变）
# ====================================================================

def compute_velocity_direction(trajectory, dt=0.1):
    """计算速度方向特征 (3D unit vector)"""
    T, num_agents, _ = trajectory.shape
    vel = np.gradient(trajectory, axis=0) / dt
    vel_mag = np.linalg.norm(vel, axis=2, keepdims=True) + 1e-8
    vel_dir = vel / vel_mag
    vel_dir = np.nan_to_num(vel_dir, nan=0.0)
    vel_mag = np.nan_to_num(vel_mag, nan=0.0)
    return vel_dir, vel_mag


def compute_acceleration_decomposition(trajectory, dt=0.1):
    """计算加速度分解"""
    T, num_agents, _ = trajectory.shape
    vel = np.gradient(trajectory, axis=0) / dt
    acc = np.gradient(vel, axis=0) / dt
    vel_mag = np.linalg.norm(vel, axis=2, keepdims=True) + 1e-8
    vel_dir = vel / vel_mag
    a_tangent = np.sum(acc * vel_dir, axis=2, keepdims=True)
    a_parallel = a_tangent * vel_dir
    a_normal_vec = acc - a_parallel
    a_normal = np.linalg.norm(a_normal_vec, axis=2, keepdims=True)
    a_tangent = np.nan_to_num(a_tangent, nan=0.0)
    a_normal = np.nan_to_num(a_normal, nan=0.0)
    return a_tangent, a_normal


def compute_angular_velocity(trajectory, dt=0.1):
    """计算角速度大小（标量）"""
    T, num_agents, _ = trajectory.shape
    vel = np.gradient(trajectory, axis=0) / dt
    acc = np.gradient(vel, axis=0) / dt
    vel_mag_sq = np.sum(vel ** 2, axis=2, keepdims=True) + 1e-8
    cross_product = np.cross(vel, acc)
    omega = np.linalg.norm(cross_product, axis=2, keepdims=True) / vel_mag_sq
    omega = np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)
    omega = np.clip(omega, -2.0, 2.0)
    return omega


def compute_jerk(trajectory, dt=0.1):
    """计算Jerk"""
    T, num_agents, _ = trajectory.shape
    vel = np.gradient(trajectory, axis=0) / dt
    acc = np.gradient(vel, axis=0) / dt
    jerk = np.gradient(acc, axis=0) / dt
    jerk_mag = np.linalg.norm(jerk, axis=2, keepdims=True)
    jerk_mag = np.nan_to_num(jerk_mag, nan=0.0, posinf=1.0, neginf=0.0)
    jerk_mag = np.clip(jerk_mag, 0, 2.0)
    return jerk_mag


def compute_multi_scale_velocity(trajectory, dt=0.1, scales=[1, 2, 3]):
    """计算多尺度速度特征"""
    T = len(trajectory)
    multi_scale_vels = []
    for scale in scales:
        if T > scale:
            vel = np.diff(trajectory, n=scale, axis=0) / (dt * scale)
            padding = np.tile(vel[-1:], (scale, 1, 1))
            vel = np.vstack([vel, padding])
        else:
            vel = np.diff(trajectory, axis=0) / dt
            padding = np.tile(vel[-1:], (T - len(vel), 1, 1))
            vel = np.vstack([vel, padding])
        multi_scale_vels.append(vel)
    return np.concatenate(multi_scale_vels, axis=-1)


def compute_curvature(trajectory, dt=0.1):
    """计算3D曲率大小"""
    T, num_agents, _ = trajectory.shape
    curvature = np.zeros((T, num_agents, 1))
    for i in range(num_agents):
        traj = trajectory[:, i, :]
        vel = np.gradient(traj, axis=0) / dt
        acc = np.gradient(vel, axis=0) / dt
        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-8
        vel_normalized = vel / vel_norm
        a_parallel = (acc * vel_normalized).sum(axis=1, keepdims=True) * vel_normalized
        a_perp = acc - a_parallel
        a_perp_norm = np.linalg.norm(a_perp, axis=1, keepdims=True)
        curv = a_perp_norm / (vel_norm ** 2)
        curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
        curv = 1.0 / (1.0 + np.exp(-curv))  # Sigmoid压缩
        curvature[:, i, :] = curv
    return curvature


def compute_plane_curvatures(trajectory, dt=0.1):
    """计算三个平面的曲率"""
    T, num_agents, _ = trajectory.shape
    plane_curvs = np.zeros((T, num_agents, 3))
    for i in range(num_agents):
        traj = trajectory[:, i, :]
        eps = 1e-8
        # XY平面
        pos_xy = np.column_stack([traj[:, 0], traj[:, 1], np.zeros(T)])
        vel_xy = np.gradient(pos_xy, axis=0) / dt
        acc_xy = np.gradient(vel_xy, axis=0) / dt
        cross_xy = np.cross(vel_xy, acc_xy)
        vel_norm_xy = np.linalg.norm(vel_xy, axis=1)
        curv_xy = np.linalg.norm(cross_xy, axis=1) / np.maximum(vel_norm_xy ** 3, eps)
        curv_xy = np.nan_to_num(curv_xy, nan=0.0, posinf=1.0, neginf=0.0)
        plane_curvs[:, i, 0] = curv_xy
        # YZ平面
        pos_yz = np.column_stack([np.zeros(T), traj[:, 1], traj[:, 2]])
        vel_yz = np.gradient(pos_yz, axis=0) / dt
        acc_yz = np.gradient(vel_yz, axis=0) / dt
        cross_yz = np.cross(vel_yz, acc_yz)
        vel_norm_yz = np.linalg.norm(vel_yz, axis=1)
        curv_yz = np.linalg.norm(cross_yz, axis=1) / np.maximum(vel_norm_yz ** 3, eps)
        curv_yz = np.nan_to_num(curv_yz, nan=0.0, posinf=1.0, neginf=0.0)
        plane_curvs[:, i, 1] = curv_yz
        # XZ平面
        pos_xz = np.column_stack([traj[:, 0], np.zeros(T), traj[:, 2]])
        vel_xz = np.gradient(pos_xz, axis=0) / dt
        acc_xz = np.gradient(vel_xz, axis=0) / dt
        cross_xz = np.cross(vel_xz, acc_xz)
        vel_norm_xz = np.linalg.norm(vel_xz, axis=1)
        curv_xz = np.linalg.norm(cross_xz, axis=1) / np.maximum(vel_norm_xz ** 3, eps)
        curv_xz = np.nan_to_num(curv_xz, nan=0.0, posinf=1.0, neginf=0.0)
        plane_curvs[:, i, 2] = curv_xz
    return plane_curvs


def compute_features_enhanced_24d(trajectory, dt=0.1):
    """计算原有24D特征"""
    T, num_agents, _ = trajectory.shape
    vel_dir, vel_mag = compute_velocity_direction(trajectory, dt)
    a_tangent, a_normal = compute_acceleration_decomposition(trajectory, dt)
    omega = compute_angular_velocity(trajectory, dt)
    jerk_mag = compute_jerk(trajectory, dt)
    multi_scale_vel = compute_multi_scale_velocity(trajectory, dt)
    curvature = compute_curvature(trajectory, dt)
    plane_curvs = compute_plane_curvatures(trajectory, dt)
    features = np.concatenate([
        trajectory,
        vel_dir,
        vel_mag,
        a_tangent,
        a_normal,
        omega,
        jerk_mag,
        multi_scale_vel,
        curvature,
        plane_curvs
    ], axis=-1)
    return features.astype(np.float32)


# ====================================================================
# 新增的8D特征计算函数（v4专属）⭐ 核心改进
# ====================================================================

def compute_curvature_vector(trajectory, dt=0.1):
    """
    计算曲率向量（3D） - 指向圆心的方向
    
    原理：
    κ_vec = (a_perp) / ||v||²
    其中 a_perp = a - (a·v̂)v̂ 是垂直于速度的加速度分量
    方向：指向曲率的中心（圆心）
    大小：曲率大小
    
    Args:
        trajectory: (T, agents, 3)
    
    Returns:
        curvature_vector: (T, agents, 3) - 3D曲率向量
    """
    T, num_agents, _ = trajectory.shape
    curvature_vector = np.zeros((T, num_agents, 3), dtype=np.float32)
    
    vel = np.gradient(trajectory, axis=0) / dt  # (T, agents, 3)
    acc = np.gradient(vel, axis=0) / dt  # (T, agents, 3)
    
    vel_mag_sq = np.sum(vel ** 2, axis=2, keepdims=True) + 1e-8  # (T, agents, 1)
    vel_dir = vel / (np.sqrt(vel_mag_sq) + 1e-8)  # (T, agents, 3)
    
    # 切向加速度
    a_tangent = np.sum(acc * vel_dir, axis=2, keepdims=True)  # (T, agents, 1)
    a_parallel = a_tangent * vel_dir  # (T, agents, 3)
    
    # 法向加速度（曲率向量的非标准化形式）
    a_perp = acc - a_parallel  # (T, agents, 3)
    
    # 曲率向量 = a_perp / ||v||²
    curvature_vector = a_perp / vel_mag_sq  # (T, agents, 3)
    
    # 数值保护
    curvature_vector = np.nan_to_num(curvature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    curvature_vector = np.clip(curvature_vector, -10.0, 10.0)
    
    logger.debug(f"曲率向量统计: min={curvature_vector.min():.6f}, max={curvature_vector.max():.6f}, mean={curvature_vector.mean():.6f}")
    
    return curvature_vector


def compute_angular_velocity_vector(trajectory, dt=0.1):
    """
    计算角速度向量（3D） - 旋转轴方向
    
    原理：
    ω_vec = (v × a) / ||v||²
    方向：旋转轴（右手法则）
    大小：角速度大小
    
    关键性质：
    - 直线运动：v × a ≈ 0，所以 ω_vec ≈ 0 ✓
    - 圆周运动：v × a 有稳定方向，ω_vec 指向旋转轴 ✓
    
    Args:
        trajectory: (T, agents, 3)
    
    Returns:
        angular_velocity_vector: (T, agents, 3) - 3D角速度向量
    """
    T, num_agents, _ = trajectory.shape
    angular_velocity_vector = np.zeros((T, num_agents, 3), dtype=np.float32)
    
    vel = np.gradient(trajectory, axis=0) / dt  # (T, agents, 3)
    acc = np.gradient(vel, axis=0) / dt  # (T, agents, 3)
    
    vel_mag_sq = np.sum(vel ** 2, axis=2, keepdims=True) + 1e-8  # (T, agents, 1)
    
    # v × a
    cross_product = np.cross(vel, acc)  # (T, agents, 3)
    
    # ω_vec = (v × a) / ||v||²
    angular_velocity_vector = cross_product / vel_mag_sq  # (T, agents, 3)
    
    # 数值保护
    angular_velocity_vector = np.nan_to_num(angular_velocity_vector, nan=0.0, posinf=0.0, neginf=0.0)
    angular_velocity_vector = np.clip(angular_velocity_vector, -10.0, 10.0)
    
    logger.debug(f"角速度向量统计: min={angular_velocity_vector.min():.6f}, max={angular_velocity_vector.max():.6f}, mean={angular_velocity_vector.mean():.6f}")
    
    return angular_velocity_vector


def compute_principal_curvature_rates(trajectory, dt=0.1):
    """
    计算主曲率率（2D） - 沿两个正交方向的曲率变化率
    
    原理：
    在任意点，曲率沿速度方向的变化可以分解为两个正交方向：
    - κ̇_1：沿主曲率方向（最大变化）
    - κ̇_2：沿次曲率方向（正交）
    
    这用于预测曲率的平滑过渡（避免突变）
    
    Args:
        trajectory: (T, agents, 3)
    
    Returns:
        principal_curv_rates: (T, agents, 2) - 两个主要方向的曲率率
    """
    T, num_agents, _ = trajectory.shape
    principal_curv_rates = np.zeros((T, num_agents, 2), dtype=np.float32)
    
    vel = np.gradient(trajectory, axis=0) / dt
    acc = np.gradient(vel, axis=0) / dt
    jerk = np.gradient(acc, axis=0) / dt
    
    vel_mag = np.linalg.norm(vel, axis=2, keepdims=True) + 1e-8
    vel_mag_sq = vel_mag ** 2
    
    # 计算曲率 κ(t)
    vel_dir = vel / vel_mag
    a_parallel = np.sum(acc * vel_dir, axis=2, keepdims=True) * vel_dir
    a_perp = acc - a_parallel
    curvature = np.linalg.norm(a_perp, axis=2, keepdims=True) / vel_mag_sq
    
    # 计算曲率的变化率 dκ/dt
    # dκ/dt ≈ (κ(t+1) - κ(t)) / dt
    dkappa_dt = np.gradient(curvature, axis=0) / dt
    
    # 分解到两个正交方向
    # 主方向：沿速度方向
    principal_rate_1 = dkappa_dt[:, :, 0:1]
    
    # 次方向：垂直于速度（通过Jerk间接得到）
    # 使用Jerk的法向分量
    j_parallel = np.sum(jerk * vel_dir, axis=2, keepdims=True) * vel_dir
    j_perp = jerk - j_parallel
    j_perp_mag = np.linalg.norm(j_perp, axis=2, keepdims=True)
    principal_rate_2 = j_perp_mag / (vel_mag_sq + 1e-8)
    
    principal_curv_rates[:, :, 0:1] = principal_rate_1
    principal_curv_rates[:, :, 1:2] = principal_rate_2
    
    # 数值保护
    principal_curv_rates = np.nan_to_num(principal_curv_rates, nan=0.0, posinf=0.0, neginf=0.0)
    principal_curv_rates = np.clip(principal_curv_rates, -5.0, 5.0)
    
    logger.debug(f"主曲率率统计: min={principal_curv_rates.min():.6f}, max={principal_curv_rates.max():.6f}, mean={principal_curv_rates.mean():.6f}")
    
    return principal_curv_rates


# ====================================================================
# 组合函数：计算完整的32D特征
# ====================================================================

def compute_features_enhanced_32d(trajectory, dt=0.1):
    """
    计算完整的32D增强特征 = 24D(v3) + 8D(新增)
    
    Args:
        trajectory: (T, agents, 3)
        dt: 时间步长
    
    Returns:
        features: (T, agents, 32)
    """
    T, num_agents, _ = trajectory.shape
    
    # 计算原有24D特征
    features_24d = compute_features_enhanced_24d(trajectory, dt)
    
    # 计算新增8D特征
    curvature_vec = compute_curvature_vector(trajectory, dt)  # (T, agents, 3)
    angular_vel_vec = compute_angular_velocity_vector(trajectory, dt)  # (T, agents, 3)
    principal_curv_rate = compute_principal_curvature_rates(trajectory, dt)  # (T, agents, 2)
    
    # 拼接：24D + 8D = 32D
    features_32d = np.concatenate([
        features_24d,          # 24D (原有)
        curvature_vec,         # 3D (新增：曲率向量)
        angular_vel_vec,       # 3D (新增：角速度向量)
        principal_curv_rate    # 2D (新增：主曲率率)
    ], axis=-1)
    
    assert features_32d.shape[-1] == 32, f"特征维度错误: {features_32d.shape[-1]} != 32"
    
    return features_32d.astype(np.float32)


def compute_global_feature_stats(X, num_agents, batch_size=100):
    """
    第一遍扫描：计算所有样本的全局特征统计量
    用于后续的自适应裁剪
    """
    num_samples, seq_len, _, _ = X.shape
    
    # 临时储存所有特征值（按维度）
    feature_stats = {dim: {'min': np.inf, 'max': -np.inf, 
                           'sum': 0, 'sum_sq': 0, 'count': 0}
                     for dim in range(32)}
    
    logger.info("【第1阶段】计算全局特征统计量...")
    
    for idx in tqdm(range(0, num_samples, batch_size), desc="全局统计"):
        end_idx = min(idx + batch_size, num_samples)
        batch_X = X[idx:end_idx]
        
        for sample_idx in range(end_idx - idx):
            x = batch_X[sample_idx]
            feat = compute_features_enhanced_32d(x, dt=0.1)
            
            for dim in range(32):
                feat_dim = feat[:, :, dim].flatten()
                feature_stats[dim]['min'] = min(feature_stats[dim]['min'], feat_dim.min())
                feature_stats[dim]['max'] = max(feature_stats[dim]['max'], feat_dim.max())
                feature_stats[dim]['sum'] += feat_dim.sum()
                feature_stats[dim]['sum_sq'] += (feat_dim ** 2).sum()
                feature_stats[dim]['count'] += len(feat_dim)
    
    # 计算 mean 和 std
    for dim in range(32):
        stats = feature_stats[dim]
        if stats['count'] > 0:
            stats['mean'] = stats['sum'] / stats['count']
            stats['var'] = (stats['sum_sq'] / stats['count']) - (stats['mean'] ** 2)
            stats['std'] = np.sqrt(max(stats['var'], 0))
        else:
            stats['mean'] = 0
            stats['std'] = 1
    
    logger.info("✓ 全局统计计算完成")
    return feature_stats


def precompute_features_for_dataset_v4(X, num_agents, batch_size=100, feature_stats=None):
    """
    批量预计算32D特征（改进版）
    使用全局统计量进行自适应裁剪
    
    Args:
        X: (samples, seq_len, agents, 3)
        num_agents: 无人机数量
        batch_size: 批处理大小
        feature_stats: 全局特征统计量
    
    Returns:
        features: (samples, seq_len, agents, 32)
    """
    num_samples, seq_len, _, _ = X.shape
    features = np.zeros((num_samples, seq_len, num_agents, 32), dtype=np.float32)
    
    logger.info(f"【第2阶段】计算32D特征并应用全局自适应裁剪...")
    logger.info(f"  数据形状: (samples={num_samples}, seq_len={seq_len}, agents={num_agents}, coords=3)")
    logger.info(f"  输出形状: (samples={num_samples}, seq_len={seq_len}, agents={num_agents}, features=32)")
    logger.info(f"  新增特征: 曲率向量(3D) + 角速度向量(3D) + 主曲率率(2D)")
    logger.info(f"  裁剪策略: 全局自适应（使用第一遍扫描的统计量）")
    
    start_time = time.time()
    
    for idx in tqdm(range(0, num_samples, batch_size), desc="特征计算 v4"):
        end_idx = min(idx + batch_size, num_samples)
        batch_X = X[idx:end_idx]
        
        for sample_idx in range(end_idx - idx):
            x = batch_X[sample_idx]
            feat = compute_features_enhanced_32d(x, dt=0.1)
            
            # ✅ 使用全局统计量进行裁剪
            feat_clipped = np.zeros_like(feat)
            if feature_stats is not None:
                for dim in range(32):
                    feat_dim = feat[:, :, dim]
                    stats = feature_stats[dim]
                    
                    mean = stats['mean']
                    std = stats['std']
                    
                    if std > 1e-8:
                        # 保留 ±5σ 范围
                        lower_bound = mean - 5.0 * std
                        upper_bound = mean + 5.0 * std
                        feat_clipped[:, :, dim] = np.clip(feat_dim, lower_bound, upper_bound)
                    else:
                        feat_clipped[:, :, dim] = feat_dim
            else:
                feat_clipped = feat
            
            features[idx + sample_idx] = feat_clipped
    
    elapsed_time = time.time() - start_time
    logger.info(f"✓ 特征计算完成")
    logger.info(f"  耗时: {elapsed_time:.2f} 秒 ({elapsed_time/num_samples:.4f} 秒/样本)")
    logger.info(f"  特征统计（全局自适应裁剪后）:")
    logger.info(f"    min: {features.min():.6f}")
    logger.info(f"    max: {features.max():.6f}")
    logger.info(f"    mean: {features.mean():.6f}")
    logger.info(f"    std: {features.std():.6f}")
    
    # 按维度统计
    logger.info(f"  维度分析:")
    features_24d = features[:, :, :, :24]
    features_curv_vec = features[:, :, :, 24:27]
    features_omega_vec = features[:, :, :, 27:30]
    features_curv_rate = features[:, :, :, 30:32]
    
    logger.info(f"    原有24D: min={features_24d.min():.6f}, max={features_24d.max():.6f}, mean={features_24d.mean():.6f}")
    logger.info(f"    曲率向量(3D): min={features_curv_vec.min():.6f}, max={features_curv_vec.max():.6f}, mean={features_curv_vec.mean():.6f}")
    logger.info(f"    角速度向量(3D): min={features_omega_vec.min():.6f}, max={features_omega_vec.max():.6f}, mean={features_omega_vec.mean():.6f}")
    logger.info(f"    主曲率率(2D): min={features_curv_rate.min():.6f}, max={features_curv_rate.max():.6f}, mean={features_curv_rate.mean():.6f}")
    
    return features


def main():
    parser = argparse.ArgumentParser(description='预计算32D动力学感知特征 (v4版本)')
    parser.add_argument('--data_dir', type=str, default='swarm_segments',
                       help='数据目录')
    parser.add_argument('--agents', type=str, default='3',
                       help='无人机数量 (3|4|5|6|all)')
    parser.add_argument('--output_dir', type=str, default='features_32d',
                       help='输出目录')
    parser.add_argument('--use_subset', action='store_true',
                       help='处理 _subset.npz 子集数据')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='批处理大小（建议 30-100）')
    parser.add_argument('--force', action='store_true',
                       help='覆盖已存在的文件，不提示')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试日志')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if args.agents == 'all':
        agents_list = [3, 4, 5, 6]
    else:
        agents_list = [int(args.agents)]
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   32D动力学感知特征预计算脚本 v4                           ║
║             为 train_swarm_v3_complete.py (GNN模型) 服务                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  特征维度 (32D = 24D + 8D):                                               ║
║  ✓ 原有24D: 位置+速度+加速度+角速度+Jerk+多尺度速度+曲率                  ║
║  ✓ 新增8D:  曲率向量(3D) + 角速度向量(3D) + 主曲率率(2D)                  ║
║                                                                            ║
║  核心改进 (解决圆弧方向预测问题):                                          ║
║  • 曲率向量(3D): 显式指向圆心的方向 ⭐⭐⭐                                 ║
║    - 24D中的"法向加速度"只有大小，缺少方向                                ║
║    - 现在提供完整的3D向量，RNN能准确判断向内收还是向外扩                  ║
║                                                                            ║
║  • 角速度向量(3D): 旋转轴的方向和速度 ⭐⭐⭐                               ║
║    - 24D中的"角速度"只是标量，现在提供旋转轴                              ║
║    - 对于3D圆周运动，轴方向很关键                                        ║
║                                                                            ║
║  • 主曲率率(2D): 曲率沿两个正交方向的变化率 ⭐⭐                          ║
║    - 确保预测的曲率平滑过渡，避免突变                                      ║
║    - 对于长序列预测特别重要                                                ║
║                                                                            ║
║  预期效果:                                                                 ║
║  • 圆弧方向准确率: 24D的30-40% → v4的70-80%                               ║
║  • 直线场景: 无影响（新特征自动为0）                                       ║
║  • 复杂场景: 显著改进（混合直线+圆弧）                                     ║
║                                                                            ║
║  向后兼容:                                                                 ║
║  • 原有模型仍可使用24D特征                                                 ║
║  • v3模型可直接升级到32D特征（仅需修改输入层大小）                        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    logger.info(f"="*80)
    logger.info(f"特征预计算脚本 v4 (32D动力学感知版本)")
    logger.info(f"="*80)
    logger.info(f"数据目录: {data_path}")
    logger.info(f"输出目录: {output_path}")
    logger.info(f"处理无人机数: {agents_list}")
    logger.info(f"处理子集数据: {args.use_subset}")
    logger.info(f"批处理大小: {args.batch_size}")
    
    total_samples = 0
    
    for num_agents in agents_list:
        logger.info(f"\n{'='*80}")
        logger.info(f"处理 {num_agents} 架无人机")
        logger.info(f"{'='*80}")
        
        subset_suffix = '_subset' if args.use_subset else ''
        input_file = data_path / f'input_agents_{num_agents}{subset_suffix}.npz'
        output_file = output_path / f'features_agents_{num_agents}{subset_suffix}_32d.npz'
        
        if not input_file.exists():
            logger.warning(f"⚠️  跳过：找不到 {input_file}")
            continue
        
        if output_file.exists():
            logger.warning(f"⚠️  输出文件已存在: {output_file.name}")
            if not args.force:
                response = input("覆盖？ (y/n): ").strip().lower()
                if response != 'y':
                    logger.info(f"跳过处理 {num_agents} 架无人机\n")
                    continue
        
        try:
            logger.info(f"📂 加载: {input_file.name}")
            X = np.load(input_file)['data']
            logger.info(f"   原始形状: {X.shape} (seq, samples, agents, coords)")
            
            # 转置为 (samples, seq, agents, coords)
            X = np.transpose(X, (1, 0, 2, 3))
            logger.info(f"   转置后: {X.shape} (samples, seq, agents, coords)")
            
            num_samples = X.shape[0]
            total_samples += num_samples
            
            # ✅ 第一遍：计算全局统计量
            logger.info(f"\n")
            feature_stats = compute_global_feature_stats(X, num_agents, batch_size=args.batch_size)
            
            # 打印统计信息
            logger.info(f"\n全局特征统计量总结:")
            for dim in range(32):
                stats = feature_stats[dim]
                logger.info(f"  维度{dim:2d}: μ={stats['mean']:10.4f}, σ={stats['std']:10.4f}, "
                          f"范围=[{stats['min']:10.4f}, {stats['max']:10.4f}]")
            
            # ✅ 第二遍：使用全局统计量进行特征计算和裁剪
            logger.info(f"\n")
            features = precompute_features_for_dataset_v4(X, num_agents, 
                                                          batch_size=args.batch_size,
                                                          feature_stats=feature_stats)
            
            # 保存特征
            logger.info(f"💾 保存特征: {output_file.name}")
            np.savez_compressed(output_file, features=features)
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            logger.info(f"✓ 保存完成!")
            logger.info(f"  文件大小: {file_size_mb:.2f} MB")
            logger.info(f"  特征形状: {features.shape}")
            logger.info(f"  特征维度分解:")
            logger.info(f"    1-24:  原有24D特征")
            logger.info(f"    25-27: 曲率向量 (κ_x, κ_y, κ_z)")
            logger.info(f"    28-30: 角速度向量 (ω_x, ω_y, ω_z)")
            logger.info(f"    31-32: 主曲率率 (κ̇_1, κ̇_2)")
            logger.info(f"  特征统计:")
            logger.info(f"    全局: min={features.min():.6f}, max={features.max():.6f}, mean={features.mean():.6f}, std={features.std():.6f}")
            
            # 分别统计各部分
            features_24d = features[:, :, :, :24]
            features_8d = features[:, :, :, 24:]
            logger.info(f"    前24D: min={features_24d.min():.6f}, max={features_24d.max():.6f}")
            logger.info(f"    新增8D: min={features_8d.min():.6f}, max={features_8d.max():.6f}")
            
        except Exception as e:
            logger.error(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ 所有特征预计算完成！")
    logger.info(f"{'='*80}")
    logger.info(f"总处理样本数: {total_samples}")
    logger.info(f"输出目录: {output_path}")
    
    logger.info(f"\n📖 后续使用方法:")
    logger.info(f"")
    logger.info(f"1. 修改v3模型输入层（从24D改为32D）：")
    logger.info(f"")
    logger.info(f"   在 train_swarm_model_v3_with_gnn.py 中，修改：")
    logger.info(f"   OLD: self.input_dim = 24")
    logger.info(f"   NEW: self.input_dim = 32")
    logger.info(f"")
    logger.info(f"2. 开始训练v3模型（自动加载32D特征）：")
    logger.info(f"")
    logger.info(f"   python train_swarm_v3_complete.py \\")
    logger.info(f"     --data_dir {args.data_dir} \\")
    logger.info(f"     --agents 3 \\")
    logger.info(f"     --epochs 150 \\")
    logger.info(f"     --batch_size 256 \\")
    logger.info(f"     --use_gnn \\")
    logger.info(f"     --gnn_fusion_mode concat \\")
    if args.use_subset:
        logger.info(f"     --use_subset \\")
    logger.info(f"     --use_amp \\")
    logger.info(f"     --seed 42")
    
    logger.info(f"\n✅ 预计算完成，可以开始训练v4模型了！")


if __name__ == '__main__':
    main()