#!/usr/bin/env python3
"""
消融实验训练工具函数
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 导入特征计算函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_swarm_model_enhanced import (
    compute_multi_scale_velocity,
    compute_curvature,
    compute_plane_curvatures
)
from train_swarm_model_v2_dynamics_aware import (
    compute_features_enhanced_24d
)
from precompute_features_v4 import (
    compute_features_enhanced_32d
)


class AblationDataset(Dataset):
    """消融实验数据集，支持16D和32D特征"""
    
    def __init__(self, X, Y, feature_dim=16, normalize=True, dt=0.1,
                 input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 features_precomputed=None,
                 feature_mean=None, feature_std=None):
        """
        Args:
            X: (samples, seq_in, agents, 3)
            Y: (samples, seq_out, agents, 3)
            feature_dim: 特征维度 (16 或 32)
        """
        self.X_orig = X.copy()
        self.Y_orig = Y.copy()
        self.dt = dt
        self.feature_dim = feature_dim
        self.features_precomputed = features_precomputed
        self.normalize = normalize
        
        # 位置统计
        if input_mean is not None:
            self.input_mean = np.array(input_mean, dtype=np.float32)
            self.input_std = np.array(input_std, dtype=np.float32)
        else:
            self.input_mean = np.mean(X.reshape(-1, 3), axis=0)
            self.input_std = np.std(X.reshape(-1, 3), axis=0)
        
        self.input_std = np.where(self.input_std < 1e-8, 1.0, self.input_std)
        
        # 增量统计
        y_delta = Y - X[:, -1:, :, :]
        if output_mean is not None:
            self.output_mean = np.array(output_mean, dtype=np.float32)
            self.output_std = np.array(output_std, dtype=np.float32)
        else:
            self.output_mean = np.mean(y_delta.reshape(-1, 3), axis=0)
            self.output_std = np.std(y_delta.reshape(-1, 3), axis=0)
        
        self.output_std = np.where(self.output_std < 1e-8, 1.0, self.output_std)
        
        # 特征统计
        self.feature_mean = feature_mean
        self.feature_std = feature_std
    
    def __len__(self):
        return len(self.X_orig)
    
    def __getitem__(self, idx):
        x = self.X_orig[idx].astype(np.float32)  # (seq_in, agents, 3)
        y = self.Y_orig[idx].astype(np.float32)  # (seq_out, agents, 3)
        
        # 位置归一化
        x_normalized = (x - self.input_mean) / self.input_std if self.normalize else x
        
        # 计算目标增量
        y_delta = y - x[-1:, :, :]
        if self.normalize:
            y_delta = (y_delta - self.output_mean) / self.output_std
        
        # 计算速度目标（用于多任务学习）
        vel_input = (x[1:] - x[:-1]) / self.dt
        vel_input = np.vstack([vel_input[0:1], vel_input])
        vel_input_last = vel_input[-1:, :, :]
        
        vel_output = (y[1:] - y[:-1]) / self.dt
        vel_output = np.vstack([vel_output[0:1], vel_output])
        y_velocity = (vel_output - vel_input_last) / self.output_std if self.normalize else (vel_output - vel_input_last)
        
        # 计算加速度目标
        from train_swarm_model_v2_dynamics_aware import compute_acceleration_decomposition
        a_tangent, a_normal = compute_acceleration_decomposition(y, self.dt)
        y_accel = np.concatenate([a_tangent, a_normal], axis=-1)
        
        # 获取或计算特征
        if self.features_precomputed is not None:
            features = self.features_precomputed[idx].astype(np.float32)
        else:
            # 实时计算特征
            if self.feature_dim == 16:
                # 16D特征
                vel = compute_multi_scale_velocity(x, self.dt)
                curv_3d = compute_curvature(x, self.dt)
                curv_plane = compute_plane_curvatures(x, self.dt)
                features = np.concatenate([x, vel, curv_3d, curv_plane], axis=-1)
            elif self.feature_dim == 32:
                # 32D特征
                features = compute_features_enhanced_32d(x, dt=self.dt)
            else:
                raise ValueError(f"不支持的特征维度: {self.feature_dim}")
        
        # 特征归一化
        if self.feature_mean is not None and self.feature_std is not None:
            safe_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
            features = (features - self.feature_mean) / safe_std
            features = np.clip(features, -5.0, 5.0)
        
        return {
            'x': torch.from_numpy(x_normalized.astype(np.float32)),
            'x_orig': torch.from_numpy(x.astype(np.float32)),
            'y_delta': torch.from_numpy(y_delta.astype(np.float32)),
            'y_velocity': torch.from_numpy(y_velocity.astype(np.float32)),
            'y_accel': torch.from_numpy(y_accel.astype(np.float32)),
            'features': torch.from_numpy(features.astype(np.float32)),
        }


def load_ablation_data(data_dir, num_agents, feature_dim=16, batch_size=256, 
                       val_split=0.2, num_workers=0, use_subset=False, 
                       features_dir=None):
    """
    加载消融实验数据
    
    Args:
        feature_dim: 特征维度 (16 或 32)
    """
    data_path = Path(data_dir)
    
    subset_suffix = '_subset' if use_subset else ''
    X_file = data_path / f'input_agents_{num_agents}{subset_suffix}.npz'
    Y_file = data_path / f'output_agents_{num_agents}{subset_suffix}.npz'
    
    if not X_file.exists() or not Y_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {X_file}, {Y_file}")
    
    logger.info(f"加载数据: {X_file.name}, {Y_file.name}")
    X = np.load(X_file)['data']  # (seq_in, samples, agents, 3)
    Y = np.load(Y_file)['data']  # (seq_out, samples, agents, 3)
    
    # 转置
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    logger.info(f"  数据形状: X={X.shape}, Y={Y.shape}")
    
    # 计算统计量
    input_mean = np.mean(X.reshape(-1, 3), axis=0)
    input_std = np.std(X.reshape(-1, 3), axis=0)
    input_std = np.where(input_std < 1e-8, 1.0, input_std)
    
    y_delta = Y - X[:, -1:, :, :]
    output_mean = np.mean(y_delta.reshape(-1, 3), axis=0)
    output_std = np.std(y_delta.reshape(-1, 3), axis=0)
    output_std = np.where(output_std < 1e-8, 1.0, output_std)
    
    logger.info(f"  位置统计: mean={input_mean}, std={input_std}")
    logger.info(f"  增量统计: mean={output_mean}, std={output_std}")
    
    # 加载预计算特征
    features_precomputed = None
    feature_mean = None
    feature_std = None
    
    if features_dir is not None:
        features_dir = Path(features_dir)
        # 支持多种文件名格式
        feature_candidates = [
            # 格式1: features_agents_3_subset_32d.npz (带subset后缀和维度)
            features_dir / f'features_agents_{num_agents}{subset_suffix}_{feature_dim}d.npz',
            # 格式2: features_agents_3_32d_subset.npz (维度在subset前)
            features_dir / f'features_agents_{num_agents}_{feature_dim}d{subset_suffix}.npz',
            # 格式3: features_agents_3_subset_features.npz (无维度后缀)
            features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
            # 格式4: features_agents_3.npz (最简单格式，用于16D)
            features_dir / f'features_agents_{num_agents}.npz',
            # 格式5: features_agents_3_32d.npz (无subset后缀)
            features_dir / f'features_agents_{num_agents}_{feature_dim}d.npz',
        ]
        
        for feat_path in feature_candidates:
            if feat_path.exists():
                logger.info(f"✓ 找到预计算特征: {feat_path}")
                try:
                    feat_data = np.load(feat_path)
                    features_precomputed = feat_data['features']
                    
                    # 计算特征统计量
                    feature_mean = np.mean(features_precomputed.reshape(-1, feature_dim), axis=0)
                    feature_std = np.std(features_precomputed.reshape(-1, feature_dim), axis=0)
                    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
                    
                    logger.info(f"  特征形状: {features_precomputed.shape}")
                    logger.info(f"  特征统计: mean_shape={feature_mean.shape}, std_shape={feature_std.shape}")
                    break
                except Exception as e:
                    logger.warning(f"  加载失败: {e}")
                    continue
    
    if features_precomputed is None:
        logger.warning(f"⚠️  未找到预计算特征文件，将实时计算")
    
    # 分割数据
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    num_val = max(1, int(num_samples * val_split)) if val_split > 0 else 0
    val_sample_idx = indices[:num_val]
    train_sample_idx = indices[num_val:]
    
    train_data_X = X[train_sample_idx]
    train_data_Y = Y[train_sample_idx]
    val_data_X = X[val_sample_idx]
    val_data_Y = Y[val_sample_idx]
    
    # 分割特征
    train_features_precomputed = None
    val_features_precomputed = None
    if features_precomputed is not None:
        train_features_precomputed = features_precomputed[train_sample_idx]
        val_features_precomputed = features_precomputed[val_sample_idx]
    
    logger.info(f"  数据分割: train={len(train_sample_idx)}, val={len(val_sample_idx)}")
    
    # 创建数据集
    train_dataset = AblationDataset(
        train_data_X, train_data_Y,
        feature_dim=feature_dim,
        normalize=True, dt=0.1,
        input_mean=input_mean, input_std=input_std,
        output_mean=output_mean, output_std=output_std,
        features_precomputed=train_features_precomputed,
        feature_mean=feature_mean, feature_std=feature_std
    )
    
    val_dataset = AblationDataset(
        val_data_X, val_data_Y,
        feature_dim=feature_dim,
        normalize=True, dt=0.1,
        input_mean=input_mean, input_std=input_std,
        output_mean=output_mean, output_std=output_std,
        features_precomputed=val_features_precomputed,
        feature_mean=feature_mean, feature_std=feature_std
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(num_workers // 2, 0),
        pin_memory=torch.cuda.is_available()
    )
    
    stats = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
    }
    
    return stats
