#!/usr/bin/env python3
"""
预计算轨迹特征脚本 - 用于 train_swarm_model_enhanced.py
===============================================================

目的：
1. 从输入位置数据预计算 16 维特征
2. 保存为 .npz 文件供训练使用
3. 加速训练 10-50 倍（避免每个 epoch 重复计算）

特征维度（16D）：
- 位置 (3D): x, y, z
- 多尺度速度 (9D): 1步、2步、3步速度
- 3D曲率 (1D)
- 平面曲率 (3D): XY、YZ、XZ 平面

使用示例：
    python precompute_features_v2.py --data_dir swarm_segments --agents 3 --use_subset
    python precompute_features_v2.py --data_dir swarm_segments --agents all --batch_size 50
"""

import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def compute_multi_scale_velocity(trajectory, dt=0.1, scales=[1, 2, 3]):
    """计算多尺度速度特征 (seq_len, agents, 9)"""
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
    """计算 3D 曲率 (seq_len, agents, 1)"""
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
        curv = 1.0 / (1.0 + np.exp(-curv))
        
        curvature[:, i, :] = curv
    
    return curvature


def compute_plane_curvatures(trajectory, dt=0.1):
    """计算 XY/YZ/XZ 三个平面的曲率 (seq_len, agents, 3)"""
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


def precompute_features_for_dataset(X, num_agents, batch_size=100):
    """批量预计算特征 (samples, seq_len, agents, 16)"""
    num_samples, seq_len, _, _ = X.shape
    features = np.zeros((num_samples, seq_len, num_agents, 16), dtype=np.float32)
    
    logger.info(f"预计算特征...")
    logger.info(f"  数据形状: (samples={num_samples}, seq_len={seq_len}, agents={num_agents}, coords=3)")
    logger.info(f"  输出形状: (samples={num_samples}, seq_len={seq_len}, agents={num_agents}, features=16)")
    
    for idx in tqdm(range(0, num_samples, batch_size), desc="特征计算"):
        end_idx = min(idx + batch_size, num_samples)
        batch_X = X[idx:end_idx]
        
        for sample_idx in range(end_idx - idx):
            x = batch_X[sample_idx]
            vel = compute_multi_scale_velocity(x, dt=0.1)
            curv_3d = compute_curvature(x, dt=0.1)
            curv_plane = compute_plane_curvatures(x, dt=0.1)
            
            features[idx + sample_idx] = np.concatenate([x, vel, curv_3d, curv_plane], axis=-1)
            features[idx + sample_idx] = np.clip(features[idx + sample_idx], -100, 100)
    
    return features


def main():
    parser = argparse.ArgumentParser(description='预计算轨迹特征')
    parser.add_argument('--data_dir', type=str, default='swarm_segments',
                       help='数据目录')
    parser.add_argument('--agents', type=str, default='3',
                       help='无人机数量 (3|4|5|6|all)')
    parser.add_argument('--output_dir', type=str, default='swarm_segments',
                       help='输出目录（默认覆盖原数据目录）')
    parser.add_argument('--use_subset', action='store_true',
                       help='处理 _subset.npz 子集数据')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='批处理大小（建议 50-200）')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if args.agents == 'all':
        agents_list = [3, 4, 5, 6]
    else:
        agents_list = [int(args.agents)]
    
    logger.info(f"="*70)
    logger.info(f"特征预计算脚本 v2")
    logger.info(f"="*70)
    logger.info(f"数据目录: {data_path}")
    logger.info(f"输出目录: {output_path}")
    logger.info(f"处理无人机数: {agents_list}")
    
    for num_agents in agents_list:
        logger.info(f"\n{'='*70}")
        logger.info(f"处理 {num_agents} 架无人机")
        logger.info(f"{'='*70}")
        
        subset_suffix = '_subset' if args.use_subset else ''
        input_file = data_path / f'input_agents_{num_agents}{subset_suffix}.npz'
        output_file = output_path / f'features_agents_{num_agents}{subset_suffix}.npz'
        
        if not input_file.exists():
            logger.warning(f"⚠️  跳过：找不到 {input_file}")
            continue
        
        if output_file.exists():
            logger.warning(f"⚠️  输出文件已存在: {output_file.name}")
            response = input("覆盖？ (y/n): ").strip().lower()
            if response != 'y':
                logger.info(f"跳过处理 {num_agents} 架无人机\n")
                continue
        
        try:
            logger.info(f"加载: {input_file.name}")
            X = np.load(input_file)['data']
            X = np.transpose(X, (1, 0, 2, 3))
            logger.info(f"✓ 加载完成，形状: {X.shape}")
            
            features = precompute_features_for_dataset(X, num_agents, batch_size=args.batch_size)
            
            np.savez_compressed(output_file, features=features)
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            logger.info(f"✓ 保存完成: {output_file.name}")
            logger.info(f"  文件大小: {file_size_mb:.2f} MB")
            logger.info(f"  特征形状: {features.shape}")
            
        except Exception as e:
            logger.error(f"❌ 处理失败: {e}")
            continue
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ 所有特征预计算完成！")
    logger.info(f"{'='*70}")
    logger.info(f"\n推荐训练命令:")
    logger.info(f"  python train_swarm_model_enhanced.py \\")
    logger.info(f"    --data_dir {args.data_dir} \\")
    logger.info(f"    --agents 3 \\")
    logger.info(f"    --features_dir {args.output_dir} \\")
    if args.use_subset:
        logger.info(f"    --use_subset \\")
    logger.info(f"    --epochs 200 \\")
    logger.info(f"    --batch_size 256 \\")
    logger.info(f"    --use_amp \\")
    logger.info(f"    --use_attention \\")
    logger.info(f"    --seed 42")


if __name__ == '__main__':
    main()
