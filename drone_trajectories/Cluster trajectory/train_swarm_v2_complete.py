#!/usr/bin/env python3
"""
集群轨迹模型 v2 完整训练脚本
动力学感知版本（关注速度方向、加速度变化、周期运动识别）

使用示例：
    python train_swarm_v2_complete.py --agents 3 --epochs 200 --batch_size 256 --use_amp --seed 42

与v1对比：
    v1 (原有): 16D特征 + 位置预测
    v2 (改进): 24D特征 + 多任务学习（位置65% + 速度20% + 加速度15%）
              + 显式速度方向和加速度分解 + 周期特征
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
import csv
from datetime import datetime

# 导入v2模型定义
from train_swarm_model_v2_dynamics_aware import (
    compute_features_enhanced_24d,
    compute_velocity_direction,
    compute_acceleration_decomposition,
    DynamicsAwareSwarmGRUModel,
    DynamicsAwareLoss
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SwarmTrajectoryDatasetV2(Dataset):
    """
    v2版本数据集 - 支持24D特征和多目标
    """
    
    def __init__(self, X, Y, normalize=True, dt=0.1,
                 input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 feature_mean=None, feature_std=None):
        """
        Args:
            X: (samples, seq_in, agents, 3)
            Y: (samples, seq_out, agents, 3)
        """
        self.X_orig = X.copy()
        self.Y_orig = Y.copy()
        self.dt = dt
        
        # 位置统计
        if input_mean is not None:
            self.input_mean = np.array(input_mean, dtype=np.float32).reshape(1, 1, 3)
            self.input_std = np.array(input_std, dtype=np.float32).reshape(1, 1, 3)
        else:
            self.input_mean = np.mean(X.reshape(-1, 3), axis=0).reshape(1, 1, 3)
            self.input_std = np.std(X.reshape(-1, 3), axis=0).reshape(1, 1, 3)
        
        # 增量统计
        y_delta = Y - X[:, -1:, :, :]
        if output_mean is not None:
            self.output_mean = np.array(output_mean, dtype=np.float32).reshape(1, 1, 3)
            self.output_std = np.array(output_std, dtype=np.float32).reshape(1, 1, 3)
        else:
            self.output_mean = np.mean(y_delta.reshape(-1, 3), axis=0).reshape(1, 1, 3)
            self.output_std = np.std(y_delta.reshape(-1, 3), axis=0).reshape(1, 1, 3)
        
        self.feature_mean = feature_mean
        self.feature_std = feature_std
    
    def __len__(self):
        return len(self.X_orig)
    
    def __getitem__(self, idx):
        x = self.X_orig[idx]  # (seq_in, agents, 3)
        y = self.Y_orig[idx]  # (seq_out, agents, 3)
        
        # 计算24D特征
        features = compute_features_enhanced_24d(x, dt=self.dt)  # (seq_in, agents, 24)
        
        # 特征归一化
        if self.feature_mean is not None and self.feature_std is not None:
            feature_mean_vec = self.feature_mean.reshape(1, 1, 24)
            feature_std_vec = self.feature_std.reshape(1, 1, 24)
            features = (features - feature_mean_vec) / (feature_std_vec + 1e-8)
        
        features = np.clip(features, -5, 5)
        
        # 计算目标：位置增量
        y_delta = y - x[-1:, :, :]  # (seq_out, agents, 3)
        
        # 增量归一化
        y_delta_norm = (y_delta - self.output_mean) / (self.output_std + 1e-8)
        y_delta_norm = np.clip(y_delta_norm, -5, 5)
        
        # 计算速度目标（归一化）
        vel = np.gradient(y, axis=0) / self.dt
        vel_norm = (vel - self.input_mean) / (self.input_std + 1e-8)  # 用位置统计近似
        vel_norm = np.clip(vel_norm, -5, 5)
        
        # 计算加速度目标（切向和法向）
        a_tangent, a_normal = compute_acceleration_decomposition(y, self.dt)
        accel = np.concatenate([a_tangent, a_normal], axis=-1)  # (seq_out, agents, 2)
        
        # 保持维度为 (seq, agents, features) 格式
        # DataLoader会在第0维(前面)加上batch_size
        # 最终得到 (batch, seq, agents, features)
        
        return (
            torch.tensor(features, dtype=torch.float32),  # (seq_in, agents, 24)
            torch.tensor(x, dtype=torch.float32),  # (seq_in, agents, 3)
            torch.tensor(y_delta_norm, dtype=torch.float32),  # (seq_out, agents, 3)
            torch.tensor(vel_norm, dtype=torch.float32),  # (seq_out, agents, 3)
            torch.tensor(accel, dtype=torch.float32),  # (seq_out, agents, 2)
            torch.tensor(y, dtype=torch.float32)  # (seq_out, agents, 3)
        )


def load_swarm_data_v2(data_dir, num_agents, batch_size=32, val_split=0.2, 
                       num_workers=0, use_subset=False, features_dir=None):
    """
    加载数据并计算统计量
    
    Args:
        data_dir: 输入/输出数据目录
        num_agents: 无人机数量
        batch_size: 批次大小
        val_split: 验证集比例
        num_workers: DataLoader工作进程数
        use_subset: 是否使用_subset数据
        features_dir: 预计算特征目录 (如果为None则实时计算)
    """
    data_path = Path(data_dir)
    
    # 根据 use_subset 选择文件
    subset_suffix = '_subset' if use_subset else ''
    X_file = data_path / f'input_agents_{num_agents}{subset_suffix}.npz'
    Y_file = data_path / f'output_agents_{num_agents}{subset_suffix}.npz'
    
    if not X_file.exists() or not Y_file.exists():
        raise FileNotFoundError(f"找不到数据文件: {X_file} 或 {Y_file}")
    
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
    
    # ========== 尝试加载预计算特征 ==========
    features_precomputed = None
    feature_mean = None
    feature_std = None
    
    if features_dir is not None:
        # 尝试两种文件名格式
        features_path = Path(features_dir) / f'features_agents_{num_agents}{subset_suffix}_24d.npz'
        if not features_path.exists():
            features_path = Path(features_dir) / f'features_agents_{num_agents}_24d{subset_suffix}.npz'
        
        if features_path.exists():
            logger.info(f"  加载预计算特征: {features_path}")
            try:
                features_data = np.load(features_path)
                features_precomputed = features_data['features']  # (samples, seq_in, agents, 24)
                feature_mean = features_data.get('mean', None)
                feature_std = features_data.get('std', None)
                logger.info(f"  ✓ 预计算特征形状: {features_precomputed.shape}")
                if feature_mean is not None:
                    logger.info(f"  ✓ 特征统计量已加载: mean={feature_mean.shape}, std={feature_std.shape}")
            except Exception as e:
                logger.warning(f"  ⚠ 预计算特征加载失败: {e}，将实时计算")
        else:
            logger.warning(f"  ⚠ 未找到预计算特征文件: {features_path}，将实时计算")
    
    # ========== 如果没有预计算特征，则实时计算 ==========
    if features_precomputed is None:
        logger.info(f"  计算24D特征统计量...")
        sample_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
        sample_features = []
        for idx in sample_indices:
            feat = compute_features_enhanced_24d(X[idx], dt=0.1)
            sample_features.append(feat)
        sample_features = np.concatenate(sample_features, axis=0)  # (采样点数, agents, 24)
        
        feature_mean = np.mean(sample_features, axis=(0, 1))  # (24,)
        feature_std = np.std(sample_features, axis=(0, 1))   # (24,)
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    else:
        # 如果特征统计还没有加载，从预计算特征中计算
        if feature_mean is None:
            logger.info(f"  从预计算特征计算统计量...")
            feature_mean = np.mean(features_precomputed.reshape(-1, 24), axis=0)
            feature_std = np.std(features_precomputed.reshape(-1, 24), axis=0)
            feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    
    logger.info(f"  24D特征统计: mean形状={feature_mean.shape}, std形状={feature_std.shape}")
    
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
    
    # 如果有预计算特征，也要进行相同的分割
    train_features_precomputed = None
    val_features_precomputed = None
    if features_precomputed is not None:
        train_features_precomputed = features_precomputed[train_sample_idx]
        val_features_precomputed = features_precomputed[val_sample_idx]
    
    logger.info(f"  数据分割: train={len(train_sample_idx)}, val={len(val_sample_idx)}")
    if features_precomputed is not None:
        logger.info(f"  ✓ 预计算特征也已分割")
    
    # 创建数据集
    class SwarmTrajectoryDatasetV2WithPrecomputed(SwarmTrajectoryDatasetV2):
        """支持预计算特征的数据集"""
        
        def __init__(self, X, Y, features_precomputed=None, normalize=True, dt=0.1,
                     input_mean=None, input_std=None,
                     output_mean=None, output_std=None,
                     feature_mean=None, feature_std=None):
            super().__init__(X, Y, normalize, dt, input_mean, input_std,
                           output_mean, output_std, feature_mean, feature_std)
            self.features_precomputed = features_precomputed
        
        def __getitem__(self, idx):
            x = self.X_orig[idx]  # (seq_in, agents, 3)
            y = self.Y_orig[idx]  # (seq_out, agents, 3)
            
            # 使用预计算特征或实时计算
            if self.features_precomputed is not None:
                features = self.features_precomputed[idx].copy()  # (seq_in, agents, 24)
            else:
                features = compute_features_enhanced_24d(x, dt=self.dt)  # (seq_in, agents, 24)
            
            # 特征归一化
            if self.feature_mean is not None and self.feature_std is not None:
                feature_mean_vec = self.feature_mean.reshape(1, 1, 24)
                feature_std_vec = self.feature_std.reshape(1, 1, 24)
                features = (features - feature_mean_vec) / (feature_std_vec + 1e-8)
            
            features = np.clip(features, -5, 5)
            
            # 计算目标：位置增量
            y_delta = y - x[-1:, :, :]  # (seq_out, agents, 3)
            
            # 增量归一化
            y_delta_norm = (y_delta - self.output_mean) / (self.output_std + 1e-8)
            y_delta_norm = np.clip(y_delta_norm, -5, 5)
            
            # 计算速度目标（归一化）
            vel = np.gradient(y, axis=0) / self.dt
            vel_norm = (vel - self.input_mean) / (self.input_std + 1e-8)  # 用位置统计近似
            vel_norm = np.clip(vel_norm, -5, 5)
            
            # 计算加速度目标（切向和法向）
            a_tangent, a_normal = compute_acceleration_decomposition(y, self.dt)
            accel = np.concatenate([a_tangent, a_normal], axis=-1)  # (seq_out, agents, 2)
            
            return (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y_delta_norm, dtype=torch.float32),
                torch.tensor(vel_norm, dtype=torch.float32),
                torch.tensor(accel, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )
    
    # 创建数据集
    train_dataset = SwarmTrajectoryDatasetV2WithPrecomputed(
        train_data_X, train_data_Y, 
        features_precomputed=train_features_precomputed,
        normalize=True,
        input_mean=input_mean, input_std=input_std,
        output_mean=output_mean, output_std=output_std,
        feature_mean=feature_mean, feature_std=feature_std
    )
    val_dataset = SwarmTrajectoryDatasetV2WithPrecomputed(
        val_data_X, val_data_Y,
        features_precomputed=val_features_precomputed,
        normalize=True,
        input_mean=input_mean, input_std=input_std,
        output_mean=output_mean, output_std=output_std,
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
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
    }
    
    return train_loader, val_loader, stats


def train_epoch_v2(model, train_loader, optimizer, criterion, device, 
                   scaler=None, use_amp=False, teacher_forcing_ratio=0.5, 
                   epoch=1, total_epochs=200):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_loss_pos = 0
    total_loss_vel = 0
    total_loss_accel = 0
    count = 0
    
    # 自适应TF衰减
    tf_ratio = max(0.0, teacher_forcing_ratio - 0.005 * (epoch - 1))
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [TF={tf_ratio:.4f}]"):
        features, x_orig, y_delta, y_vel, y_accel, y_orig = batch
        features = features.to(device, non_blocking=True)
        x_orig = x_orig.to(device, non_blocking=True)
        y_delta = y_delta.to(device, non_blocking=True)
        y_vel = y_vel.to(device, non_blocking=True)
        y_accel = y_accel.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                pred_pos, pred_vel, pred_accel = model(
                    features, x_orig, y=y_delta, y_velocity=y_vel, 
                    y_accel=y_accel, teacher_forcing_ratio=tf_ratio
                )
                loss, loss_pos, loss_vel, loss_accel = criterion(
                    pred_pos, y_delta,
                    pred_velocity=pred_vel, target_velocity=y_vel,
                    pred_accel=pred_accel, target_accel=y_accel
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_pos, pred_vel, pred_accel = model(
                features, x_orig, y=y_delta, y_velocity=y_vel,
                y_accel=y_accel, teacher_forcing_ratio=tf_ratio
            )
            loss, loss_pos, loss_vel, loss_accel = criterion(
                pred_pos, y_delta,
                pred_velocity=pred_vel, target_velocity=y_vel,
                pred_accel=pred_accel, target_accel=y_accel
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        total_loss_pos += loss_pos.item()
        total_loss_vel += loss_vel.item()
        total_loss_accel += loss_accel.item()
        count += 1
    
    avg_loss = total_loss / max(1, count)
    avg_loss_pos = total_loss_pos / max(1, count)
    avg_loss_vel = total_loss_vel / max(1, count)
    avg_loss_accel = total_loss_accel / max(1, count)
    
    return avg_loss, avg_loss_pos, avg_loss_vel, avg_loss_accel, tf_ratio


def evaluate_v2(model, val_loader, criterion, device, stats=None):
    """评估 v2 版本
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        stats: 统计量字典，包含 output_mean 和 output_std
    
    Returns:
        avg_loss: 平均损失
        avg_mae: 平均绝对误差 (米)
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="评估"):
            features, x_orig, y_delta, y_vel, y_accel, y_orig = batch
            features = features.to(device)
            x_orig = x_orig.to(device)
            y_delta = y_delta.to(device)
            y_vel = y_vel.to(device)
            y_accel = y_accel.to(device)
            y_orig = y_orig.to(device)
            
            # 模型预测（TF比例=0，纯自回归）
            pred_pos, pred_vel, pred_accel = model(
                features, x_orig, y=None, y_velocity=None,
                y_accel=None, teacher_forcing_ratio=0.0
            )
            
            # 计算损失（归一化空间）
            loss, loss_pos, _, _ = criterion(
                pred_pos, y_delta,
                pred_velocity=pred_vel, target_velocity=y_vel,
                pred_accel=pred_accel, target_accel=y_accel
            )
            
            # ✅ 计算MAE（物理空间）
            # 1. 反归一化预测增量到物理空间
            if stats is not None:
                output_mean = stats['output_mean']
                output_std = stats['output_std']
                
                if isinstance(output_mean, np.ndarray):
                    output_mean = torch.tensor(output_mean, device=device, dtype=pred_pos.dtype)
                if isinstance(output_std, np.ndarray):
                    output_std = torch.tensor(output_std, device=device, dtype=pred_pos.dtype)
                
                # 广播到正确形状 (batch, seq_out, agents, 3)
                output_mean = output_mean.view(1, 1, 1, -1)
                output_std = output_std.view(1, 1, 1, -1)
                
                # 反归一化
                pred_delta_physical = pred_pos * output_std + output_mean  # (batch, seq_out, agents, 3)
                
                # 2. 重建绝对位置
                last_pos = x_orig[:, -1:, :, :]  # (batch, 1, agents, 3) 最后一个输入位置
                pred_absolute = last_pos + pred_delta_physical  # (batch, seq_out, agents, 3)
                
                # 3. 计算MAE
                mae = torch.abs(pred_absolute - y_orig).mean().item()
            else:
                # 如果没有统计量，在归一化空间计算MAE（可能不太准确）
                mae = torch.abs(pred_pos - y_delta).mean().item()
            
            total_loss += loss.item()
            total_mae += mae
            count += 1
    
    return total_loss / max(1, count), total_mae / max(1, count)


def main():
    parser = argparse.ArgumentParser(description='训练v2版轨迹模型（动力学感知）')
    parser.add_argument('--data_dir', type=str, default='swarm_segments',
                        help='数据目录')
    parser.add_argument('--agents', type=str, default='3',
                        help='无人机数量')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='GRU层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比例')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=25,
                        help='早停耐心值')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.6,
                        help='TF初始比例')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--output_dir', type=str, default='swarm_models_v2',
                        help='输出目录')
    parser.add_argument('--features_dir', type=str, default=None,
                        help='预计算特征目录 (如果为None则实时计算)')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='使用注意力')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader工作数')
    parser.add_argument('--use_subset', action='store_true',
                        help='使用子集数据')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定检查点恢复训练 (如: last_checkpoint_agents_3_v2.pt)')
    parser.add_argument('--no_resume', action='store_true',
                        help='跳过自动恢复，从头开始训练')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"模型版本: v2 (动力学感知，24D特征，多任务学习)")
    
    # 解析agents参数
    if args.agents == 'all':
        agents_list = [3, 4, 5, 6]
    else:
        agents_list = [int(args.agents)]
    
    for num_agents in agents_list:
        logger.info(f"\n{'='*70}")
        logger.info(f"训练 {num_agents} 架无人机 (v2版本)")
        logger.info(f"{'='*70}")
        
        try:
            train_loader, val_loader, stats = load_swarm_data_v2(
                args.data_dir, num_agents, args.batch_size, 
                args.val_split, args.num_workers, args.use_subset,
                features_dir=args.features_dir
            )
        except FileNotFoundError as e:
            logger.error(f"数据加载失败: {e}")
            continue
        
        model = DynamicsAwareSwarmGRUModel(
            input_size=24,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=3,
            dropout=args.dropout,
            use_attention=args.use_attention
        ).to(device)
        
        logger.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        criterion = DynamicsAwareLoss(
            weight_position=0.80,
            weight_velocity=0.10,
            weight_accel=0.10
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
        
        # ========== 检查点恢复逻辑 ==========
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rate': [],
            'teacher_forcing_ratio': []
        }
        
        ckpt_last = Path(args.output_dir) / f'last_checkpoint_agents_{num_agents}_v2.pt'
        ckpt_interrupted = Path(args.output_dir) / f'interrupted_checkpoint_agents_{num_agents}_v2.pt'
        
        # 自动恢复逻辑
        # 1. 显式指定的 --resume 参数
        if args.resume:
            checkpoint_path = Path(args.output_dir) / args.resume
            if checkpoint_path.exists():
                logger.info(f"从指定检查点恢复: {args.resume}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    patience_counter = checkpoint.get('patience_counter', 0)
                    training_history = checkpoint.get('training_history', training_history)
                    logger.info(f"✓ 已从检查点恢复到 epoch {start_epoch}")
                except Exception as e:
                    logger.error(f"❌ 检查点加载失败: {e}")
                    raise
            else:
                logger.error(f"❌ 检查点不存在: {checkpoint_path}")
                raise FileNotFoundError(f"指定的检查点文件不存在: {args.resume}")
        
        # 2. 自动恢复逻辑（仅当未指定 --resume 时）
        elif not args.no_resume and ckpt_last.exists():
            logger.info(f"检测到最后的检查点，自动恢复...")
            try:
                checkpoint = torch.load(ckpt_last, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                patience_counter = checkpoint.get('patience_counter', 0)
                training_history = checkpoint.get('training_history', training_history)
                logger.info(f"✓ 已自动恢复到 epoch {start_epoch}")
            except RuntimeError as e:
                logger.warning(f"⚠ 检查点加载失败: {e}")
                logger.warning(f"⚠ 从头开始训练")
        
        # 3. 尝试从中断检查点恢复
        elif not args.no_resume and ckpt_interrupted.exists():
            logger.info(f"检测到中断检查点，尝试恢复...")
            try:
                checkpoint = torch.load(ckpt_interrupted, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                patience_counter = checkpoint.get('patience_counter', 0)
                training_history = checkpoint.get('training_history', training_history)
                logger.info(f"✓ 已从中断检查点恢复到 epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"⚠ 中断检查点恢复失败: {e}，从头开始训练")
        else:
            logger.info(f"未找到检查点，从头开始训练")
        
        # 打印训练进度表头
        print("=" * 130)
        print(f"{'Epoch':<8} {'Train Loss':<16} {'Val Loss':<16} {'MAE (m)':<16} {'LR':<14} {'TF Ratio':<12} {'Status':<20}")
        print("=" * 130)
        
        # ========== 训练循环 ==========
        for epoch in range(start_epoch, args.epochs + 1):
            try:
                train_loss, loss_pos, loss_vel, loss_accel, tf_ratio = train_epoch_v2(
                    model, train_loader, optimizer, criterion, device,
                    scaler, args.use_amp, args.teacher_forcing_ratio, epoch, args.epochs
                )
                
                val_loss, val_mae = evaluate_v2(
                    model, val_loader, criterion, device, stats=stats
                )
                
                # 记录训练历史
                training_history['epoch'].append(epoch)
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                training_history['val_mae'].append(val_mae)
                current_lr = optimizer.param_groups[0]['lr']
                training_history['learning_rate'].append(current_lr)
                training_history['teacher_forcing_ratio'].append(tf_ratio)
                
                # ========== 每个epoch保存last checkpoint（用于恢复） ==========
                ckpt_last_path = Path(args.output_dir) / f'last_checkpoint_agents_{num_agents}_v2.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'training_history': training_history,
                    # 保存所有统计量
                    'input_mean': stats['input_mean'],
                    'input_std': stats['input_std'],
                    'input_mean_all': stats.get('input_mean_all'),
                    'input_std_all': stats.get('input_std_all'),
                    'output_mean': stats['output_mean'],
                    'output_std': stats['output_std'],
                    'feature_mean': stats['feature_mean'],  # ✅ 添加特征统计
                    'feature_std': stats['feature_std'],    # ✅ 添加特征统计
                    'config': {
                        'input_size': 24,
                        'hidden_size': args.hidden_size,
                        'num_layers': args.num_layers,
                        'dropout': args.dropout,
                        'use_attention': args.use_attention,
                    }
                }, ckpt_last_path)
                
                # 判断是否是最佳模型
                status = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    status = "✓ BEST"
                    
                    # 额外保存最佳模型
                    best_model_path = Path(args.output_dir) / f'best_model_agents_{num_agents}_v2.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'input_mean': stats['input_mean'],
                        'input_std': stats['input_std'],
                        'input_mean_all': stats.get('input_mean_all'),
                        'input_std_all': stats.get('input_std_all'),
                        'output_mean': stats['output_mean'],
                        'output_std': stats['output_std'],
                        'feature_mean': stats['feature_mean'],  # ✅ 添加特征统计
                        'feature_std': stats['feature_std'],    # ✅ 添加特征统计
                        'config': {
                            'input_size': 24,
                            'hidden_size': args.hidden_size,
                            'num_layers': args.num_layers,
                            'dropout': args.dropout,
                            'use_attention': args.use_attention,
                        }
                    }, best_model_path)
                    
                    # 同时保存统计量到独立的NPZ文件
                    stats_path = Path(args.output_dir) / f'norm_stats_agents_{num_agents}_v2.npz'
                    np.savez(
                        stats_path,
                        input_mean=stats['input_mean'],
                        input_std=stats['input_std'],
                        input_mean_all=stats.get('input_mean_all'),
                        input_std_all=stats.get('input_std_all'),
                        output_mean=stats['output_mean'],
                        output_std=stats['output_std'],
                    )
                else:
                    patience_counter += 1
                    status = f"patience: {patience_counter}/{args.patience}"
                    if patience_counter >= args.patience:
                        print(f"{epoch:<8} {train_loss:<16.6f} {val_loss:<16.6f} {val_mae:<16.6f} {current_lr:<14.2e} {tf_ratio:<12.4f} {'EARLY STOP':<20}")
                        logger.info(f"早停 (patience={args.patience})")
                        break
                
                # 打印进度
                print(f"{epoch:<8} {train_loss:<16.6f} {val_loss:<16.6f} {val_mae:<16.6f} {current_lr:<14.2e} {tf_ratio:<12.4f} {status:<20}")
                
                # 调整学习率
                scheduler.step(val_loss)
                
                # ========== 实时更新训练历史CSV（每个epoch保存一次） ==========
                csv_path = Path(args.output_dir) / f'training_history_agents_{num_agents}_v2.csv'
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val MAE (m)', 'Learning Rate', 'Teacher Forcing Ratio'])
                    for i in range(len(training_history['epoch'])):
                        writer.writerow([
                            training_history['epoch'][i],
                            f"{training_history['train_loss'][i]:.6f}",
                            f"{training_history['val_loss'][i]:.6f}",
                            f"{training_history['val_mae'][i]:.6f}",
                            f"{training_history['learning_rate'][i]:.6e}",
                            f"{training_history['teacher_forcing_ratio'][i]:.4f}"
                        ])
                
                # ========== 实时更新训练配置JSON（每个epoch保存一次） ==========
                config_dict = {
                    'timestamp': datetime.now().isoformat(),
                    'model_version': 'v2 (dynamics-aware)',
                    'num_agents': num_agents,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'weight_decay': args.weight_decay,
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'use_attention': args.use_attention,
                    'use_amp': args.use_amp,
                    'teacher_forcing_ratio': args.teacher_forcing_ratio,
                    'current_epoch': epoch,
                    'best_val_loss': float(best_val_loss),
                    'current_val_loss': float(val_loss),
                    'current_val_mae': float(val_mae),
                    'current_train_loss': float(train_loss),
                    'min_val_loss': float(min(training_history['val_loss'])) if training_history['val_loss'] else 0.0,
                    'min_val_mae': float(min(training_history['val_mae'])) if training_history['val_mae'] else 0.0,
                }
                config_path = Path(args.output_dir) / f'training_config_agents_{num_agents}_v2.json'
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=4, ensure_ascii=False)
                
            except KeyboardInterrupt:
                print(f"\n⚠ 收到中断信号，正在保存检查点...")
                ckpt_interrupt_path = Path(args.output_dir) / f'interrupted_checkpoint_agents_{num_agents}_v2.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'training_history': training_history,
                    'input_mean': stats['input_mean'],
                    'input_std': stats['input_std'],
                    'input_mean_all': stats.get('input_mean_all'),
                    'input_std_all': stats.get('input_std_all'),
                    'output_mean': stats['output_mean'],
                    'output_std': stats['output_std'],
                    'feature_mean': stats['feature_mean'],  # ✅ 添加特征统计
                    'feature_std': stats['feature_std'],    # ✅ 添加特征统计
                    'config': {
                        'input_size': 24,
                        'hidden_size': args.hidden_size,
                        'num_layers': args.num_layers,
                        'dropout': args.dropout,
                        'use_attention': args.use_attention,
                    }
                }, ckpt_interrupt_path)
                logger.info(f"✓ 已保存中断检查点: {ckpt_interrupt_path}")
                logger.info(f"下次运行时会自动恢复（使用 --no_resume 跳过恢复）")
                break
        
        print("=" * 130)
        
        # ========== 训练完成总结 ==========
        best_model_path = Path(args.output_dir) / f'best_model_agents_{num_agents}_v2.pt'
        csv_path = Path(args.output_dir) / f'training_history_agents_{num_agents}_v2.csv'
        config_path = Path(args.output_dir) / f'training_config_agents_{num_agents}_v2.json'
        
        logger.info(f"✓ 训练完成!")
        logger.info(f"  ├─ 最佳模型: {best_model_path}")
        logger.info(f"  ├─ 训练历史: {csv_path}")
        logger.info(f"  └─ 配置文件: {config_path}")
        
        if training_history['val_loss']:
            best_epoch = training_history['epoch'][np.argmin(training_history['val_loss'])]
            min_val_loss = min(training_history['val_loss'])
            min_val_mae = min(training_history['val_mae'])
            logger.info(f"最佳性能: Epoch {best_epoch}")
            logger.info(f"  ├─ Val Loss = {min_val_loss:.6f}")
            logger.info(f"  └─ Val MAE = {min_val_mae:.6f} 米")


if __name__ == '__main__':
    main()
