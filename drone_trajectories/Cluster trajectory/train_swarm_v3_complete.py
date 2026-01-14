#!/usr/bin/env python3
"""
集群轨迹模型 v3 完整训练脚本
========================================================

动力学感知 + GNN 增强版本

相比 v2：
✅ 加入 GAT（图注意力网络）显式建模代理间交互
✅ 基于位置距离动态构建邻接矩阵
✅ 保留原有 24D 特征、BiGRU、多任务损失框架
✅ 完全兼容现有数据加载和训练管线
✅ 支持预计算 24D 特征加速训练
✅ 完整的训练记录、超参数保存、断点续训功能

使用示例：
    # 训练 v3 (启用 GNN，使用预计算特征)
    python train_swarm_v3_complete.py --agents 3 --epochs 150 --batch_size 256 \
        --use_gnn --gnn_hidden 64 --gnn_heads 4 --edge_threshold 5.5 \
        --gnn_fusion_mode concat --use_amp --seed 42
    
    # 快速测试（5个epoch，子集数据）
    python train_swarm_v3_complete.py --agents 3 --epochs 5 --batch_size 64 \
        --use_gnn --gnn_fusion_mode concat --use_subset
    
    # 不使用 GNN（退化为 v2）
    python train_swarm_v3_complete.py --agents 3 --epochs 100 --batch_size 256 --no_gnn

    python train_swarm_v3_complete.py --agents 3 --epochs 200 --batch_size 128 --use_gnn --gnn_hidden 64 --gnn_heads 4 --edge_threshold 5.0 --gnn_fusion_mode concat --use_subset --seed 42 --features_dir features_24d
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
import os

# 导入 v2 基础组件
from train_swarm_model_v2_dynamics_aware import (
    compute_features_enhanced_24d,
    compute_velocity_direction,
    compute_acceleration_decomposition,
    DynamicsAwareLoss,
    DynamicsAwareSwarmGRUModel
)

# 导入 v3 GNN 模型
from train_swarm_model_v3_with_gnn import (
    DynamicsAwareSwarmGRUModel_with_GNN,
    build_adjacency_from_positions
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SwarmTrajectoryDatasetV3(Dataset):
    """
    v3 版本数据集 - 支持 24D 特征、多目标和 GNN
    """
    
    def __init__(self, X, Y, normalize=True, dt=0.1,
                 input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 features_precomputed=None,
                 feature_mean=None, feature_std=None):
        """
        Args:
            X: (samples, seq_in, agents, 3) - 输入轨迹
            Y: (samples, seq_out, agents, 3) - 输出轨迹
            normalize: 是否进行归一化
            dt: 时间步长
            input_mean/std: 输入位置的统计值
            output_mean/std: 输出增量的统计值
            features_precomputed: (samples, seq_in, agents, 24) - 预计算的24D特征
            feature_mean/std: 特征的统计值
        """
        self.X_orig = X.copy()
        self.Y_orig = Y.copy()
        self.dt = dt
        self.features_precomputed = features_precomputed
        
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
        
        # 特征统计
        if feature_mean is not None:
            self.feature_mean = np.array(feature_mean, dtype=np.float32)
            self.feature_std = np.array(feature_std, dtype=np.float32)
        else:
            self.feature_mean = None
            self.feature_std = None
        
        # 归一化
        self.normalize = normalize
        if normalize:
            self.X = (X - self.input_mean) / (self.input_std + 1e-8)
            self.Y = (y_delta) / (self.output_std + 1e-8)
        else:
            self.X = X
            self.Y = y_delta
    
    def __len__(self):
        return len(self.X_orig)
    
    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)  # (seq_in, agents, 3)
        x_orig = self.X_orig[idx].astype(np.float32)
        
        y_delta = self.Y[idx].astype(np.float32)  # (seq_out, agents, 3)
        y_orig = self.Y_orig[idx].astype(np.float32)
        
        # 计算速度和加速度
        vel = (x_orig[1:] - x_orig[:-1]) / self.dt  # (seq_in-1, agents, 3)
        vel = np.vstack([vel[0:1], vel])  # 复制第一帧
        
        accel = (vel[1:] - vel[:-1]) / self.dt  # (seq_in-1, agents, 3)
        accel = np.vstack([accel[0:1], accel])  # 复制第一帧
        
        # 速度和加速度增量
        vel_out = (y_orig[1:] - y_orig[:-1]) / self.dt
        vel_out = np.vstack([vel_out[0:1], vel_out])
        y_vel = (vel_out - vel[-1:]) / (self.output_std + 1e-8)
        
        accel_out = (vel_out[1:] - vel_out[:-1]) / self.dt
        accel_out = np.vstack([accel_out[0:1], accel_out])
        
        # 将加速度分解为切向和法向分量（与模型输出对应）
        a_tangent, a_normal = compute_acceleration_decomposition(y_orig, self.dt)
        y_accel = np.concatenate([a_tangent, a_normal], axis=-1)  # (seq_out, agents, 2)
        
        # 特征
        if self.features_precomputed is not None:
            features = self.features_precomputed[idx].copy().astype(np.float32)  # (seq_in, agents, 24)
            if self.feature_mean is not None and self.feature_std is not None:
                features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        else:
            features = compute_features_enhanced_24d(x_orig, dt=self.dt)  # (seq_in, agents, 24)
            features = np.clip(features, -5, 5)  # 裁剪
            if self.feature_mean is not None and self.feature_std is not None:
                features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        
        return (
            features.astype(np.float32),
            x.astype(np.float32),
            y_delta.astype(np.float32),
            y_vel.astype(np.float32),
            y_accel.astype(np.float32),
            x_orig.astype(np.float32)
        )


def load_swarm_data_v3(data_dir, num_agents, batch_size=256, val_split=0.2, 
                       num_workers=0, use_subset=False, features_dir=None):
    """
    加载数据并计算统计量（v3 版本 - 支持预计算特征）
    
    Args:
        data_dir: 输入/输出数据目录
        num_agents: 无人机数量
        batch_size: 批次大小
        val_split: 验证集比例
        num_workers: DataLoader 工作进程数
        use_subset: 是否使用_subset数据
        features_dir: 预计算特征目录 (如果为None则自动查找)
    
    Returns:
        dict with keys: 'train_loader', 'val_loader', 'train_dataset', 'val_dataset',
                        'input_mean', 'input_std', 'output_mean', 'output_std',
                        'feature_mean', 'feature_std'
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
    
    # 转置为 (samples, seq_in, agents, 3)
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
    
    # 确定特征目录
    if features_dir is None:
        features_dir = data_path
    else:
        features_dir = Path(features_dir)
    
    # 尝试多种文件名格式和位置
    feature_candidates = [
        # 优先级 1: features_24d 子目录 + _subset
        features_dir / 'features_24d' / f'features_agents_{num_agents}{subset_suffix}_24d.npz',
        features_dir / 'features_24d' / f'features_agents_{num_agents}_24d{subset_suffix}.npz',
        
        # 优先级 2: 根目录 + _subset
        features_dir / f'features_agents_{num_agents}{subset_suffix}_24d.npz',
        features_dir / f'features_agents_{num_agents}_24d{subset_suffix}.npz',
        
        # 优先级 3: features_24d 子目录（不含 _subset）
        features_dir / 'features_24d' / f'features_agents_{num_agents}_24d.npz',
        features_dir / 'features_24d' / f'features_agents_{num_agents}.npz',
        
        # 优先级 4: 根目录（不含 _subset）
        features_dir / f'features_agents_{num_agents}_24d.npz',
        features_dir / f'features_agents_{num_agents}.npz',
    ]
    
    for feat_path in feature_candidates:
        if feat_path.exists():
            logger.info(f"  加载预计算特征: {feat_path}")
            try:
                features_data = np.load(feat_path)
                
                # 检查字段名
                if 'features' in features_data.files:
                    features_precomputed = features_data['features']
                elif 'data' in features_data.files:
                    features_precomputed = features_data['data']
                else:
                    # 取第一个 key
                    first_key = list(features_data.files)[0]
                    features_precomputed = features_data[first_key]
                
                # 获取统计量
                feature_mean = features_data.get('mean', None)
                feature_std = features_data.get('std', None)
                
                # 确保形状为 (samples, seq_in, agents, 24)
                if features_precomputed.ndim == 3:
                    # (seq_in, samples, agents, 24) -> 转置
                    features_precomputed = np.transpose(features_precomputed, (1, 0, 2, 3))
                
                # 截断到与 X 相同的长度
                if len(features_precomputed) > len(X):
                    logger.warning(
                        f"特征样本数 ({len(features_precomputed)}) > "
                        f"输入样本数 ({len(X)})，将截断"
                    )
                    features_precomputed = features_precomputed[:len(X)]
                elif len(features_precomputed) < len(X):
                    raise ValueError(
                        f"特征样本数 ({len(features_precomputed)}) < "
                        f"输入样本数 ({len(X)})"
                    )
                
                logger.info(f"  ✓ 预计算特征形状: {features_precomputed.shape}")
                if feature_mean is not None:
                    logger.info(
                        f"  ✓ 特征统计量已加载: "
                        f"mean={feature_mean.shape}, std={feature_std.shape}"
                    )
                break
            
            except Exception as e:
                logger.warning(f"  ⚠ 预计算特征加载失败: {e}，继续查找其他文件...")
                features_precomputed = None
                feature_mean = None
                feature_std = None
                continue
    
    # ========== 如果没有预计算特征，则实时计算 ==========
    if features_precomputed is None:
        logger.info(f"  计算24D特征统计量...")
        sample_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
        sample_features = []
        for idx in sample_indices:
            feat = compute_features_enhanced_24d(X[idx], dt=0.1)
            # ✅ 重要：裁剪特征到 [-5, 5]（与数据集中的处理一致）
            feat = np.clip(feat, -5, 5)
            sample_features.append(feat)
        sample_features = np.concatenate(sample_features, axis=0)  # (采样点数, agents, 24)
        
        feature_mean = np.mean(sample_features, axis=(0, 1))  # (24,)
        feature_std = np.std(sample_features, axis=(0, 1))   # (24,)
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    else:
        # 如果特征统计还没有加载，从预计算特征中计算
        if feature_mean is None:
            logger.info(f"  从预计算特征计算统计量...")
            # ✅ 重要：如果预计算特征可能没有裁剪，需要在计算前裁剪
            features_for_stats = np.clip(features_precomputed, -5, 5)
            feature_mean = np.mean(features_for_stats.reshape(-1, 24), axis=0)
            feature_std = np.std(features_for_stats.reshape(-1, 24), axis=0)
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
    train_dataset = SwarmTrajectoryDatasetV3(
        train_data_X, train_data_Y,
        normalize=True, dt=0.1,
        input_mean=input_mean, input_std=input_std,
        output_mean=output_mean, output_std=output_std,
        features_precomputed=train_features_precomputed,
        feature_mean=feature_mean, feature_std=feature_std
    )
    
    val_dataset = SwarmTrajectoryDatasetV3(
        val_data_X, val_data_Y,
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


def train_epoch_v3(model, train_loader, optimizer, loss_fn, device, use_amp=False,
                   tf_ratio=0.6):
    """
    训练单个 epoch
    """
    model.train()
    total_loss = 0
    loss_pos = 0
    loss_vel = 0
    loss_accel = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(train_loader, desc="训练")
    for batch_idx, batch in enumerate(pbar):
        features, x, y_delta, y_vel, y_accel, x_orig = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                out_delta, out_vel, out_accel = model(
                    features, x_orig, y_delta,
                    teacher_forcing_ratio=tf_ratio
                )
                loss, l_pos, l_vel, l_accel = loss_fn(
                    out_delta, y_delta, out_vel, y_vel, out_accel, y_accel
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out_delta, out_vel, out_accel = model(
                features, x_orig, y_delta,
                teacher_forcing_ratio=tf_ratio
            )
            loss, l_pos, l_vel, l_accel = loss_fn(
                out_delta, y_delta, out_vel, y_vel, out_accel, y_accel
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        loss_pos += l_pos.item()
        loss_vel += l_vel.item()
        loss_accel += l_accel.item()
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'pos': loss_pos / (batch_idx + 1),
            'vel': loss_vel / (batch_idx + 1),
            'accel': loss_accel / (batch_idx + 1)
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_pos = loss_pos / len(train_loader)
    avg_vel = loss_vel / len(train_loader)
    avg_accel = loss_accel / len(train_loader)
    
    return avg_loss, avg_pos, avg_vel, avg_accel


def evaluate_v3(model, val_loader, loss_fn, device, output_mean, output_std):
    """
    验证模型
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    
    # 转换统计量到设备上
    output_mean_tensor = torch.tensor(output_mean, dtype=torch.float32, device=device)
    output_std_tensor = torch.tensor(output_std, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证"):
            features, x, y_delta, y_vel, y_accel, x_orig = [b.to(device) for b in batch]
            
            out_delta, out_vel, out_accel = model(
                features, x_orig, y_delta,
                teacher_forcing_ratio=0.0
            )
            loss, _, _, _ = loss_fn(
                out_delta, y_delta, out_vel, y_vel, out_accel, y_accel
            )
            
            # 计算 MAE（物理空间）
            out_delta_physical = out_delta * output_std_tensor + output_mean_tensor
            y_delta_physical = y_delta * output_std_tensor + output_mean_tensor
            mae = torch.abs(out_delta_physical - y_delta_physical).mean().item()
            
            total_loss += loss.item()
            total_mae += mae
    
    avg_loss = total_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)
    
    return avg_loss, avg_mae


def main():
    parser = argparse.ArgumentParser(description='v3 集群轨迹训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str,
                        default='swarm_segments',
                        help='数据目录')
    parser.add_argument('--agents', type=int, default=3, help='代理数量')
    parser.add_argument('--seq_in', type=int, default=10, help='输入序列长度')
    parser.add_argument('--seq_out', type=int, default=5, help='输出序列长度')
    parser.add_argument('--use_subset', action='store_true', help='使用数据子集')
    parser.add_argument('--features_dir', type=str, default=None, help='预计算特征目录（若为None则自动查找）')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=128, help='BiGRU 隐藏维度')
    parser.add_argument('--num_layers', type=int, default=2, help='BiGRU 层数')
    parser.add_argument('--use_gnn', action='store_true', help='启用 GNN')
    parser.add_argument('--no_gnn', action='store_true', help='禁用 GNN（退化为 v2）')
    
    # GNN 参数
    parser.add_argument('--gnn_hidden', type=int, default=64, help='GNN 隐藏维度')
    parser.add_argument('--gnn_heads', type=int, default=4, help='GNN 注意力头数')
    parser.add_argument('--edge_threshold', type=float, default=5.5, help='邻接阈值（米）')
    parser.add_argument('--gnn_fusion_mode', type=str, default='concat',
                        choices=['concat', 'gate', 'add'],
                        help='GNN 特征融合方式')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='权重衰减')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    
    # 杂项
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--no_resume', action='store_true', help='不从最后的检查点恢复')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='指定检查点路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备：{device}")
    
    # 加载数据
    logger.info("加载数据...")
    data_info = load_swarm_data_v3(
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
    
    logger.info(
        f"✓ 数据加载完成: "
        f"训练={len(data_info['train_dataset'])}, 验证={len(data_info['val_dataset'])}"
    )
    
    # 创建模型
    logger.info("创建模型...")
    if args.no_gnn:
        logger.info("使用 v2（无 GNN）")
        model = DynamicsAwareSwarmGRUModel(
            feature_dim=24,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=3
        )
        use_gnn = False
    else:
        logger.info(f"使用 v3（GNN，fusion={args.gnn_fusion_mode}）")
        model = DynamicsAwareSwarmGRUModel_with_GNN(
            input_size=24,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=3,
            gnn_hidden=args.gnn_hidden,
            num_gnn_heads=args.gnn_heads,
            edge_threshold=args.edge_threshold,
            fusion_mode=args.gnn_fusion_mode
        )
        use_gnn = True
    
    model = model.to(device)
    logger.info(f"模型参数数：{sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    loss_fn = DynamicsAwareLoss(weight_position=0.8, weight_velocity=0.1, weight_accel=0.1)
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 检查点管理
    suffix = f"agents_{args.agents}_v3"
    if use_gnn:
        suffix += f"_gnn_{args.gnn_fusion_mode}"
    
    ckpt_dir = Path(f"gru_models_v3_{suffix}")
    ckpt_dir.mkdir(exist_ok=True)
    
    csv_file = ckpt_dir / f"training_history_{suffix}.csv"
    config_file = ckpt_dir / f"config_{suffix}.json"
    
    # 初始化配置
    config = {
        'timestamp': datetime.now().isoformat(),
        'model_version': 'v3' if use_gnn else 'v2',
        'use_gnn': use_gnn,
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
    }
    
    if use_gnn:
        config.update({
            'gnn_hidden': args.gnn_hidden,
            'gnn_heads': args.gnn_heads,
            'edge_threshold': args.edge_threshold,
            'gnn_fusion_mode': args.gnn_fusion_mode,
        })
    
    # 加载检查点（如果存在）
    start_epoch = 0
    best_val_loss = float('inf')
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_loss_pos': [],
        'train_loss_vel': [],
        'train_loss_accel': [],
        'val_loss': [],
        'val_mae': [],
        'lr': [],
        'tf_ratio': [],
    }
    
    if not args.no_resume and args.checkpoint_path is None:
        ckpt_candidates = sorted(ckpt_dir.glob(f'last_checkpoint_*.pt'))
        if ckpt_candidates:
            ckpt_path = ckpt_candidates[-1]
            logger.info(f"从检查点恢复：{ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            training_history = ckpt.get('training_history', training_history)
    elif args.checkpoint_path:
        logger.info(f"加载检查点：{args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        training_history = ckpt.get('training_history', training_history)
    
    # 训练循环
    logger.info(f"开始训练（从 epoch {start_epoch}）...")
    print("\n" + "=" * 130)
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Pos':<10} {'Vel':<10} {'Accel':<10} {'Val Loss':<14} {'MAE (m)':<12} {'LR':<12} {'TF':<10}")
    print("=" * 130)
    try:
        for epoch in range(start_epoch, args.epochs):
            # 教师强制比率衰减
            tf_ratio = max(0.0, 0.6 - 0.005 * epoch)
            
            # 训练
            train_loss, train_pos, train_vel, train_accel = train_epoch_v3(
                model, train_loader, optimizer, loss_fn, device,
                use_amp=args.use_amp, tf_ratio=tf_ratio
            )
            
            # 验证
            val_loss, val_mae = evaluate_v3(
                model, val_loader, loss_fn, device,
                data_info['output_mean'], data_info['output_std']
            )
            
            # 调度器
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            training_history['epoch'].append(epoch)
            training_history['train_loss'].append(train_loss)
            training_history['train_loss_pos'].append(train_pos)
            training_history['train_loss_vel'].append(train_vel)
            training_history['train_loss_accel'].append(train_accel)
            training_history['val_loss'].append(val_loss)
            training_history['val_mae'].append(val_mae)
            training_history['lr'].append(current_lr)
            training_history['tf_ratio'].append(tf_ratio)
            
            # 保存 CSV
            with open(csv_file, 'a' if epoch > start_epoch else 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['Epoch', 'Train Loss', 'Train Loss (pos)', 'Train Loss (vel)',
                                'Train Loss (accel)', 'Val Loss', 'Val MAE (m)',
                                'Learning Rate', 'Teacher Forcing Ratio']
                )
                if epoch == start_epoch:
                    writer.writeheader()
                writer.writerow({
                    'Epoch': epoch,
                    'Train Loss': f'{train_loss:.6f}',
                    'Train Loss (pos)': f'{train_pos:.6f}',
                    'Train Loss (vel)': f'{train_vel:.6f}',
                    'Train Loss (accel)': f'{train_accel:.6f}',
                    'Val Loss': f'{val_loss:.6f}',
                    'Val MAE (m)': f'{val_mae:.6f}',
                    'Learning Rate': f'{current_lr:.2e}',
                    'Teacher Forcing Ratio': f'{tf_ratio:.4f}',
                })
            
            # 保存配置
            config['current_epoch'] = epoch
            config['best_val_loss'] = best_val_loss
            config['current_val_loss'] = val_loss
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # 打印进度
            status = "✓ 最佳" if val_loss < best_val_loss else ""
            print(f"{epoch:<8} {train_loss:<14.6f} {train_pos:<10.6f} {train_vel:<10.6f} {train_accel:<10.6f} {val_loss:<14.6f} {val_mae:<12.6f} {current_lr:<12.2e} {tf_ratio:<10.4f} {status}")
            
            # 保存最后检查点
            last_ckpt_path = ckpt_dir / f'last_checkpoint_{epoch:04d}.pt'
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
            }, last_ckpt_path)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = ckpt_dir / f'best_model_{epoch:04d}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_mae': val_mae,
                    'config': config,
                    'input_mean': data_info['input_mean'],
                    'input_std': data_info['input_std'],
                    'output_mean': data_info['output_mean'],
                    'output_std': data_info['output_std'],
                    'feature_mean': data_info['feature_mean'],
                    'feature_std': data_info['feature_std'],
                }, best_model_path)
                logger.info(f"✓ 最佳模型已更新: {best_model_path.name} (VAL_LOSS={val_loss:.6f}, MAE={val_mae:.6f}m)")
    
    except KeyboardInterrupt:
        logger.warning("中断训练，保存断点...")
        interrupted_ckpt_path = ckpt_dir / f'interrupted_checkpoint_{epoch:04d}.pt'
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
        }, interrupted_ckpt_path)
        logger.info(f"断点已保存：{interrupted_ckpt_path}")
    
    print("=" * 130)
    logger.info("\n✓ 训练完成！")
    logger.info(f"  ├─ 最佳验证损失: {best_val_loss:.6f}")
    if training_history['val_mae']:
        best_mae_epoch = training_history['val_mae'].index(min(training_history['val_mae']))
        best_mae = min(training_history['val_mae'])
        logger.info(f"  ├─ 最佳 MAE: {best_mae:.6f}m (Epoch {best_mae_epoch})")
    logger.info(f"  ├─ 输出目录: {ckpt_dir}")
    logger.info(f"  ├─ 训练历史: {csv_file.name}")
    logger.info(f"  ├─ 配置文件: {config_file.name}")
    
    # 打印训练总结
    if training_history['epoch']:
        logger.info(f"\n  训练统计:")
        logger.info(f"    - 总 Epoch 数: {len(training_history['epoch'])}")
        logger.info(f"    - 初始学习率: {args.lr:.2e}")
        logger.info(f"    - 最终学习率: {training_history['lr'][-1]:.2e}")
        logger.info(f"    - 初始 Train Loss: {training_history['train_loss'][0]:.6f}")
        logger.info(f"    - 最终 Train Loss: {training_history['train_loss'][-1]:.6f}")
        logger.info(f"    - 初始 Val Loss: {training_history['val_loss'][0]:.6f}")
        logger.info(f"    - 最终 Val Loss: {training_history['val_loss'][-1]:.6f}")


if __name__ == '__main__':
    main()
