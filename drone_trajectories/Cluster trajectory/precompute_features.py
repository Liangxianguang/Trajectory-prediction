#!/usr/bin/env python3
"""
特征预计算脚本
将所有特征预先计算并保存到磁盘，加速训练速度 10-50 倍
python precompute_features.py ^
    --data_dir swarm_segments ^
    --output_dir swarm_features ^
    --agents all
"""

import numpy as np
import torch
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


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
    """计算 3D 曲率"""
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
        # Sigmoid 压缩
        curv = 1.0 / (1.0 + np.exp(-curv))
        
        curvature[:, i, :] = curv
    
    return curvature


def compute_plane_curvatures(trajectory, dt=0.1):
    """计算 XY/YZ/XZ 三个平面的曲率"""
    T, num_agents, _ = trajectory.shape
    plane_curvs = np.zeros((T, num_agents, 3))
    
    for i in range(num_agents):
        traj = trajectory[:, i, :]
        eps = 1e-8
        
        # XY 平面
        pos_xy = np.column_stack([traj[:, 0], traj[:, 1], np.zeros(T)])
        vel_xy = np.gradient(pos_xy, axis=0) / dt
        acc_xy = np.gradient(vel_xy, axis=0) / dt
        cross_xy = np.cross(vel_xy, acc_xy)
        vel_norm_xy = np.linalg.norm(vel_xy, axis=1)
        curv_xy = np.linalg.norm(cross_xy, axis=1) / np.maximum(vel_norm_xy ** 3, eps)
        curv_xy = np.nan_to_num(curv_xy, nan=0.0, posinf=1.0, neginf=0.0)
        plane_curvs[:, i, 0] = curv_xy
        
        # YZ 平面
        pos_yz = np.column_stack([np.zeros(T), traj[:, 1], traj[:, 2]])
        vel_yz = np.gradient(pos_yz, axis=0) / dt
        acc_yz = np.gradient(vel_yz, axis=0) / dt
        cross_yz = np.cross(vel_yz, acc_yz)
        vel_norm_yz = np.linalg.norm(vel_yz, axis=1)
        curv_yz = np.linalg.norm(cross_yz, axis=1) / np.maximum(vel_norm_yz ** 3, eps)
        curv_yz = np.nan_to_num(curv_yz, nan=0.0, posinf=1.0, neginf=0.0)
        plane_curvs[:, i, 1] = curv_yz
        
        # XZ 平面
        pos_xz = np.column_stack([traj[:, 0], np.zeros(T), traj[:, 2]])
        vel_xz = np.gradient(pos_xz, axis=0) / dt
        acc_xz = np.gradient(vel_xz, axis=0) / dt
        cross_xz = np.cross(vel_xz, acc_xz)
        vel_norm_xz = np.linalg.norm(vel_xz, axis=1)
        curv_xz = np.linalg.norm(cross_xz, axis=1) / np.maximum(vel_norm_xz ** 3, eps)
        curv_xz = np.nan_to_num(curv_xz, nan=0.0, posinf=1.0, neginf=0.0)
        plane_curvs[:, i, 2] = curv_xz
    
    return plane_curvs


def precompute_features(data_dir, output_dir, num_agents):
    """预计算特征"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    X_file = data_path / f'input_agents_{num_agents}.npz'
    Y_file = data_path / f'output_agents_{num_agents}.npz'
    
    if not X_file.exists() or not Y_file.exists():
        logger.warning(f"找不到数据文件 {num_agents} agents，跳过")
        return
    
    logger.info(f"预计算 {num_agents} 架无人机特征...")
    
    # 加载数据
    X = np.load(X_file)['data']
    Y = np.load(Y_file)['data']
    
    # 转换维度：(seq_len, num_samples, agents, 3) -> (num_samples, seq_len, agents, 3)
    # NPZ 格式是 (seq_len, num_samples, agents, 3)
    X = np.transpose(X, (1, 0, 2, 3))  # (20, 242220, N, 3) -> (242220, 20, N, 3)
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    logger.info(f"  数据形状: X={X.shape}, Y={Y.shape}")
    
    num_samples = len(X)
    
    # 预分配特征数组：(samples, seq_in, agents, 16D)
    # 16D = 位置(3) + 多尺度速度(9) + 3D曲率(1) + 平面曲率(3)
    features = np.zeros((num_samples, X.shape[1], X.shape[2], 16), dtype=np.float32)
    
    # 逐样本计算特征
    logger.info(f"  计算 {num_samples} 个样本的特征...")
    for idx in tqdm(range(num_samples), desc=f"样本 {num_agents} agents"):
        x = X[idx]  # (seq_in, agents, 3)
        
        # 计算各种特征
        vel = compute_multi_scale_velocity(x)  # (seq_in, agents, 9)
        curv_3d = compute_curvature(x)  # (seq_in, agents, 1)
        curv_plane = compute_plane_curvatures(x)  # (seq_in, agents, 3)
        
        # 拼接特征
        # ✅ 修复：移除局部的 Max-Normalization，保留原始物理数值
        # 与单机模型保持一致，让 Dataset 通过全局 Mean/Std 进行统一归一化
        # 这样模型能感知速度和曲率的绝对量级
        feat = np.concatenate([x, vel, curv_3d, curv_plane], axis=-1)
        
        # 仅做裁剪防止极端数值溢出 (如除零产生的极大值)
        feat = np.clip(feat, -100, 100)
        
        features[idx] = feat
    
    # 保存特征
    output_file = output_path / f'features_agents_{num_agents}.npz'
    np.savez_compressed(output_file, features=features)
    logger.info(f"✓ 已保存特征到: {output_file}")
    logger.info(f"  文件大小: {output_file.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='预计算集群轨迹特征')
    parser.add_argument('--data_dir', type=str, default='swarm_segments',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default='swarm_features',
                        help='输出目录')
    parser.add_argument('--agents', type=str, default='3',
                        help='无人机数量 (3|4|5|6|all)')
    
    args = parser.parse_args()
    
    if args.agents == 'all':
        agents_list = [3, 4, 5, 6]
    else:
        agents_list = [int(args.agents)]
    
    logger.info(f"开始预计算特征...")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    for num_agents in agents_list:
        try:
            precompute_features(args.data_dir, args.output_dir, num_agents)
        except Exception as e:
            logger.error(f"处理 {num_agents} 架无人机失败: {e}")
    
    logger.info(f"\n✓ 特征预计算完成！")
    logger.info(f"下次训练时使用: python train_swarm_model_enhanced.py --use_precomputed")


if __name__ == '__main__':
    main()
