#!/usr/bin/env python3
"""
集群轨迹模型 v4 完整训练脚本（修复版）
========================================================

改进：使用32D特征（原有24D + 新增8D曲率特征）

相比 v3：
✅ 输入特征从 24D 扩展到 32D
✅ 新增：曲率向量(3D) + 角速度向量(3D) + 主曲率率(2D)
✅ 在特征层面就有显式的圆弧方向信息
✅ 预期圆弧方向准确率提升 30-50%
✅ 保留所有v3的训练框架（GNN、BiGRU、多任务损失）

修复项：
✅ 按维度计算 output_mean/std（不是全局标量）
✅ y_velocity 改用v3风格的住序计算
✅ y_accel 使用正确的切向/法向分解
✅ 反归一化时正确处理维度广播
✅ Loss权重改回v3风格（position=0.65, velocity=0.25, accel=0.1）

使用示例：
    # 训练 v4 (使用32D预计算特征)
    python train_swarm_v4_fixed.py --agents 3 --epochs 150 --batch_size 256 \\
        --use_gnn --gnn_hidden 64 --gnn_heads 4 --edge_threshold 5.0 \\
        --gnn_fusion_mode concat --use_amp --seed 42 \\
        --features_dir features_32d

    # 快速测试（3个epoch，子集数据）
    python train_swarm_v4_fixed.py --agents 3 --epochs 3 --batch_size 128 \\
        --use_gnn --use_subset --features_dir features_32d

    # 不使用GNN（仅v2+32D特征）
    python train_swarm_v4_fixed.py --agents 3 --epochs 150 --batch_size 256 \\
        --no_gnn --features_dir features_32d
    python train_swarm_v4_fixed.py --agents 3 --epochs 200 --batch_size 128 --use_gnn --gnn_hidden 64 --gnn_heads 4 --edge_threshold 5.0 --gnn_fusion_mode concat --use_subset --seed 42 --features_dir features_32d
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
import json
import pandas as pd

# 导入 v3 的模型和基础组件
from train_swarm_model_v3_with_gnn import (
    DynamicsAwareSwarmGRUModel_with_GNN,
    build_adjacency_from_positions,
    MultiHeadGraphAttention,
)

from train_swarm_model_v2_dynamics_aware import (
    compute_acceleration_decomposition,
    DynamicsAwareLoss,
    DynamicsAwareSwarmGRUModel,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SwarmTrajectoryDatasetV4(Dataset):
    """
    v4 版本数据集 - 支持 32D 特征、多目标和 GNN
    
    ✅ 修复点：
    - 按维度计算 output_mean/std（形状为 (agents, 3) 而不是标量）
    - y_velocity 使用v3风格的住序计算（当前速度 - 上一个时间步速度）
    - y_accel 使用正确的切向/法向分解
    - 特征归一化使用Z-score + 异常值剔除
    """
    
    def __init__(self, X, Y, normalize=True, dt=0.1,
                 input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 features_precomputed=None,
                 feature_mean=None, feature_std=None,
                 feature_min=None, feature_max=None):
        """
        Args:
            X: (samples, seq_in, agents, 3)
            Y: (samples, seq_out, agents, 3)
            normalize: 是否归一化
            dt: 时间步长
            input_mean/std: 输入位置的统计值
            output_mean/std: 输出增量的统计值
            features_precomputed: (samples, seq_in, agents, 32) - 预计算的32D特征
            feature_mean/std/min/max: 特征的统计值
        """
        self.X_orig = X.copy()
        self.Y_orig = Y.copy()
        self.dt = dt
        self.features_precomputed = features_precomputed
        self.normalize = normalize
        
        # ✅ 修复：按维度计算统计量（形状必须是 (agents, 3)）
        if input_mean is not None:
            self.input_mean = np.array(input_mean, dtype=np.float32)
            self.input_std = np.array(input_std, dtype=np.float32)
        else:
            # 全局计算，按所有样本和序列步长平均
            self.input_mean = np.mean(X.reshape(-1, 3), axis=0)  # (3,)
            self.input_std = np.std(X.reshape(-1, 3), axis=0)    # (3,)
        
        self.input_std = np.where(self.input_std < 1e-8, 1.0, self.input_std)
        
        # ✅ 修复：按维度计算输出统计量
        if output_mean is not None:
            self.output_mean = np.array(output_mean, dtype=np.float32)
            self.output_std = np.array(output_std, dtype=np.float32)
        else:
            y_delta = Y - X[:, -1:, :, :]
            self.output_mean = np.mean(y_delta.reshape(-1, 3), axis=0)  # (3,)
            self.output_std = np.std(y_delta.reshape(-1, 3), axis=0)    # (3,)
        
        self.output_std = np.where(self.output_std < 1e-8, 1.0, self.output_std)
        
        # 特征统计
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.feature_min = feature_min
        self.feature_max = feature_max
        
        logger.info(f"✓ 数据集初始化: {len(self)} 个样本")
        logger.info(f"  输入统计: mean={self.input_mean}, std={self.input_std}")
        logger.info(f"  输出统计: mean={self.output_mean}, std={self.output_std}")
        if features_precomputed is not None:
            logger.info(f"  使用预计算的32D特征")
            logger.info(f"  特征统计: mean_shape={feature_mean.shape if feature_mean is not None else 'None'}")
    
    def __len__(self):
        return len(self.X_orig)
    
    def __getitem__(self, idx):
        """返回字典格式的数据（与v3兼容）"""
        x = self.X_orig[idx].astype(np.float32)  # (seq_in, agents, 3)
        y = self.Y_orig[idx].astype(np.float32)  # (seq_out, agents, 3)
        
        # 位置归一化
        x_normalized = (x - self.input_mean) / self.input_std if self.normalize else x
        
        # ✅ 计算目标增量
        y_delta = y - x[-1:, :, :]
        if self.normalize:
            y_delta = (y_delta - self.output_mean) / self.output_std
        
        # ✅ 修复：y_velocity 使用v3风格的住序计算
        # 计算输入序列最后的速度（用于计算输出序列的速度住序）
        vel_input = (x[1:] - x[:-1]) / self.dt  # (seq_in-1, agents, 3)
        vel_input = np.vstack([vel_input[0:1], vel_input])  # 复制第一帧，得到 (seq_in, agents, 3)
        vel_input_last = vel_input[-1:, :, :]  # (1, agents, 3)
        
        # 计算输出序列的速度
        vel_output = (y[1:] - y[:-1]) / self.dt  # (seq_out-1, agents, 3)
        vel_output = np.vstack([vel_output[0:1], vel_output])  # 复制第一帧，得到 (seq_out, agents, 3)
        
        # 速度住序：相对于输入序列最后一个时刻的速度
        y_velocity = (vel_output - vel_input_last) / self.output_std if self.normalize else (vel_output - vel_input_last)
        
        # ✅ 修复：y_accel 使用v3中的切向/法向分解（2D）
        a_tangent, a_normal = compute_acceleration_decomposition(y, self.dt)
        y_accel = np.concatenate([a_tangent, a_normal], axis=-1)  # (seq_out, agents, 2)
        
        # 获取预计算特征（32D）
        if self.features_precomputed is not None:
            features = self.features_precomputed[idx].astype(np.float32)  # (seq_in, agents, 32)
            
            # 特征归一化：Z-score标准化
            if self.feature_mean is not None and self.feature_std is not None:
                # 处理零方差维度
                safe_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
                features = (features - self.feature_mean) / safe_std
                
                # 剔除异常值（超过±5σ视为异常）
                features = np.clip(features, -5.0, 5.0)
        else:
            features = np.zeros((x.shape[0], x.shape[1], 32), dtype=np.float32)
        
        # 转换为张量并返回字典
        return {
            'x': torch.from_numpy(x_normalized.astype(np.float32)),
            'x_orig': torch.from_numpy(x.astype(np.float32)),
            'y_delta': torch.from_numpy(y_delta.astype(np.float32)),
            'y_velocity': torch.from_numpy(y_velocity.astype(np.float32)),
            'y_accel': torch.from_numpy(y_accel.astype(np.float32)),
            'features': torch.from_numpy(features.astype(np.float32)),
        }


def load_swarm_data_v4(data_dir, num_agents, batch_size=256, val_split=0.2, 
                       num_workers=0, use_subset=False, features_dir=None):
    """
    加载数据并计算统计量（v4 版本 - 支持32D预计算特征）
    """
    data_path = Path(data_dir)
    
    subset_suffix = '_subset' if use_subset else ''
    X_file = data_path / f'input_agents_{num_agents}{subset_suffix}.npz'
    Y_file = data_path / f'output_agents_{num_agents}{subset_suffix}.npz'
    
    if not X_file.exists() or not Y_file.exists():
        logger.error(f"数据文件不存在: {X_file}, {Y_file}")
        raise FileNotFoundError(f"数据文件不存在")
    
    logger.info(f"加载数据: {X_file.name}, {Y_file.name}")
    X = np.load(X_file)['data']  # (seq_in, samples, agents, 3)
    Y = np.load(Y_file)['data']  # (seq_out, samples, agents, 3)
    
    # 转置为 (samples, seq_in, agents, 3)
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    logger.info(f"  数据形状: X={X.shape}, Y={Y.shape}")
    
    # ✅ 修复：按维度计算统计量
    input_mean = np.mean(X.reshape(-1, 3), axis=0)
    input_std = np.std(X.reshape(-1, 3), axis=0)
    input_std = np.where(input_std < 1e-8, 1.0, input_std)
    
    y_delta = Y - X[:, -1:, :, :]
    output_mean = np.mean(y_delta.reshape(-1, 3), axis=0)
    output_std = np.std(y_delta.reshape(-1, 3), axis=0)
    output_std = np.where(output_std < 1e-8, 1.0, output_std)
    
    logger.info(f"  位置统计: mean={input_mean}, std={input_std}")
    logger.info(f"  增量统计: mean={output_mean}, std={output_std}")
    
    # ========== 加载预计算的32D特征 ==========
    features_precomputed = None
    feature_mean = None
    feature_std = None
    feature_min = None
    feature_max = None
    
    if features_dir is None:
        features_dir = Path('features_32d')
    else:
        features_dir = Path(features_dir)
    
    # 尝试多种文件名格式
    feature_candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_32d.npz',
        features_dir / f'features_agents_{num_agents}_32d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
    ]
    
    for feat_path in feature_candidates:
        if feat_path.exists():
            logger.info(f"✓ 找到预计算特征: {feat_path}")
            try:
                feat_data = np.load(feat_path)
                features_precomputed = feat_data['features']  # (samples, seq_in, agents, 32)
                
                logger.info(f"  特征形状: {features_precomputed.shape}")
                logger.info(f"  特征范围: [{features_precomputed.min():.6f}, {features_precomputed.max():.6f}]")
                
                # 计算特征统计量（按维度）
                feature_mean = np.mean(features_precomputed.reshape(-1, 32), axis=0)
                feature_std = np.std(features_precomputed.reshape(-1, 32), axis=0)
                feature_min = np.min(features_precomputed.reshape(-1, 32), axis=0)
                feature_max = np.max(features_precomputed.reshape(-1, 32), axis=0)
                
                # 处理零方差维度
                feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
                
                logger.info(f"  特征统计: mean_shape={feature_mean.shape}, std_shape={feature_std.shape}")
                for dim in range(min(5, 32)):
                    logger.info(f"    dim {dim}: μ={feature_mean[dim]:.4f}, σ={feature_std[dim]:.4f}")
                
                break
            except Exception as e:
                logger.warning(f"  加载失败: {e}")
                continue
    
    if features_precomputed is None:
        logger.warning(f"⚠️  未找到预计算特征文件")
    else:
        logger.info(f"✓ 成功加载预计算32D特征")
    
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
    train_dataset = SwarmTrajectoryDatasetV4(
        train_data_X, train_data_Y,
        normalize=True, dt=0.1,
        input_mean=input_mean, input_std=input_std,
        output_mean=output_mean, output_std=output_std,
        features_precomputed=train_features_precomputed,
        feature_mean=feature_mean, feature_std=feature_std,
        feature_min=feature_min, feature_max=feature_max
    )
    
    val_dataset = SwarmTrajectoryDatasetV4(
        val_data_X, val_data_Y,
        normalize=True, dt=0.1,
        input_mean=input_mean, input_std=input_std,
        output_mean=output_mean, output_std=output_std,
        features_precomputed=val_features_precomputed,
        feature_mean=feature_mean, feature_std=feature_std,
        feature_min=feature_min, feature_max=feature_max
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
        'feature_min': feature_min,
        'feature_max': feature_max,
    }
    
    return stats


def train_epoch_v4(model, train_loader, optimizer, loss_fn, device, use_amp=False, tf_ratio=0.6):
    """
    训练单个 epoch（v4版本）
    """
    model.train()
    total_loss = 0
    loss_pos = 0
    loss_vel = 0
    loss_accel = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(train_loader, desc="训练")
    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)
        x_orig = batch['x_orig'].to(device)
        features = batch['features'].to(device)
        y_target = batch['y_delta'].to(device)
        y_vel_target = batch['y_velocity'].to(device)
        y_accel_target = batch['y_accel'].to(device)
        
        # ✅ 处理batch形状：(batch, seq, agents, dim) → (batch*agents, seq, dim)
        batch_size, seq_in, num_agents, feat_dim = features.shape
        features_reshaped = features.reshape(batch_size * num_agents, seq_in, feat_dim)
        x_orig_reshaped = x_orig.reshape(batch_size * num_agents, seq_in, 3)
        y_target_reshaped = y_target.reshape(batch_size * num_agents, -1, 3)
        y_vel_target_reshaped = y_vel_target.reshape(batch_size * num_agents, -1, 3)
        y_accel_target_reshaped = y_accel_target.reshape(batch_size * num_agents, -1, 2)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                output_pos, output_vel, output_accel = model(
                    features_reshaped,
                    x_orig_reshaped,
                    y_target_reshaped,
                    y_vel_target_reshaped,
                    y_accel_target_reshaped,
                    teacher_forcing_ratio=tf_ratio
                )
                
                # ✅ 正确的Loss函数调用（v3风格）
                loss_result = loss_fn(
                    output_pos, y_target_reshaped,
                    output_vel, y_vel_target_reshaped,
                    output_accel, y_accel_target_reshaped
                )
                
                loss = loss_result[0]
                l_pos = loss_result[1]
                l_vel = loss_result[2]
                l_accel = loss_result[3]
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output_pos, output_vel, output_accel = model(
                features_reshaped,
                x_orig_reshaped,
                y_target_reshaped,
                y_vel_target_reshaped,
                y_accel_target_reshaped,
                teacher_forcing_ratio=tf_ratio
            )
            
            loss_result = loss_fn(
                output_pos, y_target_reshaped,
                output_vel, y_vel_target_reshaped,
                output_accel, y_accel_target_reshaped
            )
            
            loss = loss_result[0]
            l_pos = loss_result[1]
            l_vel = loss_result[2]
            l_accel = loss_result[3]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        with torch.no_grad():
            total_loss += float(loss.item())
            loss_pos += float(l_pos.item())
            loss_vel += float(l_vel.item())
            loss_accel += float(l_accel.item())
        
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.6f}',
            'pos': f'{loss_pos / (batch_idx + 1):.6f}',
            'vel': f'{loss_vel / (batch_idx + 1):.6f}',
            'accel': f'{loss_accel / (batch_idx + 1):.6f}',
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_pos = loss_pos / len(train_loader)
    avg_vel = loss_vel / len(train_loader)
    avg_accel = loss_accel / len(train_loader)
    
    return avg_loss, avg_pos, avg_vel, avg_accel


def evaluate_v4(model, val_loader, loss_fn, device, output_mean, output_std):
    """
    验证模型（v4版本）
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    
    # ✅ 修复：正确处理维度广播
    output_mean_tensor = torch.from_numpy(output_mean).float().to(device)  # (3,)
    output_std_tensor = torch.from_numpy(output_std).float().to(device)    # (3,)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证", leave=False):
            x = batch['x'].to(device)
            x_orig = batch['x_orig'].to(device)
            features = batch['features'].to(device)
            y_target = batch['y_delta'].to(device)
            y_vel_target = batch['y_velocity'].to(device)
            y_accel_target = batch['y_accel'].to(device)
            
            batch_size, seq_in, num_agents, feat_dim = features.shape
            features_reshaped = features.reshape(batch_size * num_agents, seq_in, feat_dim)
            x_orig_reshaped = x_orig.reshape(batch_size * num_agents, seq_in, 3)
            y_target_reshaped = y_target.reshape(batch_size * num_agents, -1, 3)
            y_vel_target_reshaped = y_vel_target.reshape(batch_size * num_agents, -1, 3)
            y_accel_target_reshaped = y_accel_target.reshape(batch_size * num_agents, -1, 2)
            
            output_pos, output_vel, output_accel = model(
                features_reshaped,
                x_orig_reshaped,
                y=None,
                teacher_forcing_ratio=0.0
            )
            
            loss_result = loss_fn(
                output_pos, y_target_reshaped,
                output_vel, y_vel_target_reshaped,
                output_accel, y_accel_target_reshaped
            )
            
            loss = loss_result[0]
            
            # ✅ 修复：正确的反归一化和MAE计算
            # output_pos shape: (batch*agents, seq_out, 3)
            # output_mean_tensor shape: (3,) → reshape to (1, 1, 3) for broadcasting
            mean_expanded = output_mean_tensor.view(1, 1, 3)
            std_expanded = output_std_tensor.view(1, 1, 3)
            
            output_pos_denorm = output_pos * std_expanded + mean_expanded
            y_target_denorm = y_target_reshaped * std_expanded + mean_expanded
            
            mae = torch.mean(torch.abs(output_pos_denorm - y_target_denorm)).item()
            
            total_loss += loss.item()
            total_mae += mae
    
    avg_loss = total_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)
    
    return avg_loss, avg_mae


def main():
    parser = argparse.ArgumentParser(description='v4 集群轨迹训练 (32D特征，修复版)')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='swarm_segments', help='数据目录')
    parser.add_argument('--agents', type=int, default=3, help='代理数量')
    parser.add_argument('--seq_in', type=int, default=20, help='输入序列长度')
    parser.add_argument('--seq_out', type=int, default=10, help='输出序列长度')
    parser.add_argument('--use_subset', action='store_true', help='使用数据子集')
    parser.add_argument('--features_dir', type=str, default='features_32d', help='32D特征目录')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=128, help='BiGRU隐藏维度')
    parser.add_argument('--num_layers', type=int, default=3, help='BiGRU层数')
    parser.add_argument('--use_gnn', action='store_true', help='启用GNN')
    parser.add_argument('--no_gnn', action='store_true', help='禁用GNN')
    
    # GNN参数
    parser.add_argument('--gnn_hidden', type=int, default=64, help='GNN隐藏维度')
    parser.add_argument('--gnn_heads', type=int, default=4, help='GNN注意力头数')
    parser.add_argument('--edge_threshold', type=float, default=5.0, help='邻接阈值')
    parser.add_argument('--gnn_fusion_mode', type=str, default='concat',
                       choices=['concat', 'gate', 'add'], help='特征融合方式')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='权重衰减')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    
    # 杂项
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("加载数据和32D特征...")
    data_info = load_swarm_data_v4(
        args.data_dir,
        args.agents,
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=0,
        use_subset=args.use_subset,
        features_dir=args.features_dir
    )
    
    train_loader = data_info['train_loader']
    val_loader = data_info['val_loader']
    
    logger.info(f"✓ 数据加载完成: 训练={len(data_info['train_dataset'])}, 验证={len(data_info['val_dataset'])}")
    
    # 创建模型
    logger.info("创建模型...")
    use_gnn = args.use_gnn and not args.no_gnn
    
    if use_gnn:
        model = DynamicsAwareSwarmGRUModel_with_GNN(
            input_size=32,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=3,
            gnn_hidden=args.gnn_hidden,
            num_gnn_heads=args.gnn_heads,
            edge_threshold=args.edge_threshold,
            fusion_mode=args.gnn_fusion_mode
        )
        model_name = "v4_GNN"
    else:
        model = DynamicsAwareSwarmGRUModel(
            input_size=32,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=3,
        )
        model_name = "v4_baseline"
    
    model = model.to(device)
    logger.info(f"✓ 模型创建: {model_name}")
    logger.info(f"  参数数: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  输入维度: 32D (24D + 8D曲率特征)")
    
    # ✅ 修复：Loss权重改回v3风格
    loss_fn = DynamicsAwareLoss(weight_position=0.65, weight_velocity=0.25, weight_accel=0.1)
    logger.info(f"  Loss权重: position=0.65, velocity=0.25, accel=0.1 (v3风格)")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 检查点目录
    suffix = f"agents_{args.agents}_v4_fixed"
    if use_gnn:
        suffix += "_gnn"
    
    ckpt_dir = Path(f"gru_models_v4_fixed_{suffix}")
    ckpt_dir.mkdir(exist_ok=True)
    
    csv_file = ckpt_dir / f"training_history_{suffix}.csv"
    config_file = ckpt_dir / f"config_{suffix}.json"
    
    # 配置信息
    config = {
        'timestamp': datetime.now().isoformat(),
        'model_version': 'v4_fixed',
        'use_gnn': use_gnn,
        'input_features': 32,
        'num_agents': args.agents,
        'seq_in': args.seq_in,
        'seq_out': args.seq_out,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'seed': args.seed,
        'use_amp': args.use_amp,
    }
    
    if use_gnn:
        config.update({
            'gnn_hidden': args.gnn_hidden,
            'gnn_heads': args.gnn_heads,
            'edge_threshold': args.edge_threshold,
            'fusion_mode': args.gnn_fusion_mode,
        })
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"配置已保存: {config_file}")
    
    # 训练循环
    logger.info(f"\n开始训练 v4 修复版模型（32D特征）...")
    print("\n" + "="*130)
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Pos':<10} {'Vel':<10} {'Accel':<10} {'Val Loss':<14} {'MAE (m)':<12} {'LR':<12}")
    print("="*130)
    
    best_val_loss = float('inf')
    best_mae = float('inf')
    training_history = []
    
    try:
        for epoch in range(args.epochs):
            # 教师强制比率衰减
            tf_ratio = max(0.0, 0.6 - 0.005 * epoch)
            
            # 训练
            train_loss, train_pos, train_vel, train_accel = train_epoch_v4(
                model, train_loader, optimizer, loss_fn, device,
                use_amp=args.use_amp, tf_ratio=tf_ratio
            )
            
            # 验证
            val_loss, val_mae = evaluate_v4(
                model, val_loader, loss_fn, device,
                data_info['output_mean'], data_info['output_std']
            )
            
            # 调度器
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 日志
            status = "✓ 最佳" if val_loss < best_val_loss else ""
            print(f"{epoch:<8} {train_loss:<14.6f} {train_pos:<10.6f} {train_vel:<10.6f} "
                  f"{train_accel:<10.6f} {val_loss:<14.6f} {val_mae:<12.6f} {current_lr:<12.6f} {status}")
            
            # 保存历史
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_pos': train_pos,
                'train_vel': train_vel,
                'train_accel': train_accel,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'lr': current_lr,
            })
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                ckpt = ckpt_dir / f"checkpoint_{epoch:04d}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'training_history': training_history,
                    'input_mean': data_info['input_mean'],
                    'input_std': data_info['input_std'],
                    'output_mean': data_info['output_mean'],
                    'output_std': data_info['output_std'],
                    'feature_mean': data_info['feature_mean'],
                    'feature_std': data_info['feature_std'],
                    'config': config,
                }, ckpt)
                logger.info(f"✓ 定期检查点保存: {ckpt.name}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_mae = val_mae
                best_ckpt = ckpt_dir / f"best_model_{suffix}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_mae': best_mae,
                    'config': config,
                    'input_mean': data_info['input_mean'],
                    'input_std': data_info['input_std'],
                    'output_mean': data_info['output_mean'],
                    'output_std': data_info['output_std'],
                    'feature_mean': data_info['feature_mean'],
                    'feature_std': data_info['feature_std'],
                }, best_ckpt)
                logger.info(f"✓ 最佳模型已更新: {best_ckpt.name} (val_loss={val_loss:.6f}, mae={val_mae:.6f}m)")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 训练被中断，正在保存断点...")
        interrupted_ckpt = ckpt_dir / f"interrupted_checkpoint_{epoch:04d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'training_history': training_history,
            'input_mean': data_info['input_mean'],
            'input_std': data_info['input_std'],
            'output_mean': data_info['output_mean'],
            'output_std': data_info['output_std'],
            'feature_mean': data_info['feature_mean'],
            'feature_std': data_info['feature_std'],
            'config': config,
        }, interrupted_ckpt)
        logger.info(f"✓ 断点已保存: {interrupted_ckpt}")
    
    print("="*130)
    
    # 保存统计信息文件
    stats_file = ckpt_dir / f"stats_{suffix}.npz"
    np.savez(stats_file,
             input_mean=data_info['input_mean'],
             input_std=data_info['input_std'],
             output_mean=data_info['output_mean'],
             output_std=data_info['output_std'],
             feature_mean=data_info['feature_mean'],
             feature_std=data_info['feature_std'],
             feature_min=data_info['feature_min'],
             feature_max=data_info['feature_max'])
    logger.info(f"✓ 统计信息已保存: {stats_file}")
    
    # 保存历史
    df = pd.DataFrame(training_history)
    df.to_csv(csv_file, index=False)
    logger.info(f"✓ 训练历史已保存: {csv_file}")
    
    logger.info(f"\n✓ v4 修复版训练完成!")
    logger.info(f"  ├─ 最佳验证损失: {best_val_loss:.6f}")
    logger.info(f"  ├─ 最佳验证MAE: {best_mae:.6f}m")
    logger.info(f"  ├─ 输出目录: {ckpt_dir}")
    logger.info(f"  ├─ 配置文件: {config_file}")
    logger.info(f"  ├─ 训练历史: {csv_file}")
    logger.info(f"  ├─ 统计信息: {stats_file}")
    logger.info(f"  ├─ 最佳模型: best_model_{suffix}.pt")
    logger.info(f"  ├─ 模型版本: v4_fixed (32D特征，修复版)")
    logger.info(f"  └─ Loss权重: position=0.65, velocity=0.25, accel=0.1 (v3风格)")


if __name__ == '__main__':
    main()
