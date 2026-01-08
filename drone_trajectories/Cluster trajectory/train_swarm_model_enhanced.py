#!/usr/bin/env python3
"""
增强版集群轨迹模型训练脚本
融合 train_model_enhanced.py 和 train_model_bigru_improved.py 的最佳实践

核心改进：
1. ✓ 增量位移预测（而非绝对坐标） - 更稳定的目标分布
2. ✓ 多平面曲率特征 (XY/YZ/XZ) - 捕捉多维运动特性
3. ✓ BiGRU 编码器 + Cross-Attention 解码器
4. ✓ 平面特征融合机制 - 不同无人机个性化预测
5. ✓ 自适应 Teacher Forcing
6. ✓ 正确的解码器初始化（从原始位置而非特征）

使用示例：
    python train_swarm_model_enhanced.py --agents 3 --epochs 200 --batch_size 2048 --use_attention --use_amp
    python train_swarm_model_enhanced.py --agents all --epochs 100 --use_amp
    python train_swarm_model_enhanced.py --agents 3 --epochs 200 --batch_size 512 --features_dir swarm_features --use_amp --use_attention
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
import csv

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


class SwarmTrajectoryDataset(Dataset):
    """
    增强版集群轨迹数据集
    支持两种模式：
    1. 实时计算特征（功能完整但较慢）
    2. 使用预计算特征（推荐，快 10-50 倍）
    """
    
    def __init__(self, X, Y, normalize=True, dt=0.1, use_delta_target=True, 
                 features_precomputed=None, input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 feature_mean=None, feature_std=None):
        """
        Args:
            X: 输入位置 (samples, seq_in, agents, 3)
            Y: 输出位置 (samples, seq_out, agents, 3)
            normalize: 是否归一化
            dt: 采样间隔
            use_delta_target: 是否使用增量位移作为目标
            features_precomputed: 预计算的特征 (samples, seq_in, agents, 16)，如果提供则不计算
            input_mean: 输入位置的通道级均值 (3,)
            input_std: 输入位置的通道级标准差 (3,)
            output_mean: 增量位移的通道级均值 (3,)
            output_std: 增量位移的通道级标准差 (3,)
            feature_mean: 特征通道级均值 (16,)
            feature_std: 特征通道级标准差 (16,)
        """
        self.X_orig = X.copy()  # 保存原始输入用于计算增量
        self.Y_orig = Y.copy()  # 保存原始输出
        self.dt = dt
        self.use_delta_target = use_delta_target
        self.features_precomputed = features_precomputed
        
        # 保存全局特征统计量
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # ✅ 修复：使用通道级统计量（与单机模型一致）
        if input_mean is not None and input_std is not None:
            self.input_mean = input_mean  # 形状 (3,)
            self.input_std = input_std    # 形状 (3,)
        else:
            # 向后兼容：如果没有提供，计算标量统计
            self.input_mean = X.mean()
            self.input_std = max(X.std(), 1e-8)
        
        if output_mean is not None and output_std is not None:
            self.output_mean = output_mean  # 形状 (3,)
            self.output_std = output_std    # 形状 (3,)
        else:
            # 计算子集统计（仅当未提供时）
            y_delta = Y - X[:, -1:, :, :]
            if hasattr(y_delta, 'reshape'):  # 确保是numpy数组
                self.output_mean = np.mean(y_delta.reshape(-1, 3), axis=0)
                self.output_std = np.std(y_delta.reshape(-1, 3), axis=0)
                self.output_std = np.where(self.output_std < 1e-8, 1.0, self.output_std)
            else:
                self.output_mean = y_delta.mean()
                self.output_std = max(y_delta.std(), 1e-8)
            logger.warning(f"未提供全局统计，使用子集统计: mean={self.output_mean}, std={self.output_std}")
        
        logger.info(f"数据集统计:")
        logger.info(f"  位置_mean: {self.input_mean}, 位置_std: {self.input_std}")
        logger.info(f"  增量_mean: {self.output_mean}, 增量_std: {self.output_std}")
        logger.info(f"  use_delta_target: {self.use_delta_target}")
        logger.info(f"  使用预计算特征: {features_precomputed is not None}")
    
    def __len__(self):
        return len(self.X_orig)
    
    def __getitem__(self, idx):
        x = self.X_orig[idx]  # (seq_in, agents, 3) - 原始未归一化位置
        y = self.Y_orig[idx]  # (seq_out, agents, 3) - 原始位置
        
        # 使用预计算特征或实时计算
        if self.features_precomputed is not None:
            features = self.features_precomputed[idx].copy() # copy为了修改不影响原数组
        else:
            # 计算特征（基于原始位置）
            vel = compute_multi_scale_velocity(x, self.dt)  # (seq_in, agents, 9)
            curv_3d = compute_curvature(x, self.dt)  # (seq_in, agents, 1)
            curv_plane = compute_plane_curvatures(x, self.dt)  # (seq_in, agents, 3)
            
            # 拼接特征: 位置(3) + 多尺度速度(9) + 3D曲率(1) + 平面曲率(3) = 16D
            features = np.concatenate([x, vel, curv_3d, curv_plane], axis=-1)
            features = np.clip(features, -100, 100)

        # -------------------------------------------------------------
        # ✅ 统一归一化逻辑 (修复了之前的双重归一化潜在Bug)
        # -------------------------------------------------------------
        
        # 检查是否提供了有效的全局 16D 特征统计量 (从预计算特征中获得)
        has_global_stats = (self.feature_mean is not None and self.feature_std is not None and 
                           (np.any(self.feature_mean != 0) or np.any(self.feature_std != 1)))
        
        if has_global_stats:
            # 方案 A: 优先使用全局 Z-Score 归一化 (所有 16 个通道)
            # 这会自动处理位置通道的归一化，以及速度/曲率的标准化
            features = (features - self.feature_mean) / self.feature_std
        else:
            # 方案 B: 降级模式（如实时计算且无预计算统计量）
            # 仅归一化已知的位置通道 (0-2)，其他通道保持原始物理数值
            # 注意：我们不再使用 Instance Norm，因为它会破坏物理量级信息
            if self.input_mean is not None and self.input_std is not None:
                features[..., :3] = (features[..., :3] - self.input_mean) / (self.input_std + 1e-8)
                
        # 最终裁剪，防止离群值破坏梯度
        features = np.clip(features, -5, 5)
        
        # **关键**：计算并归一化增量位移目标
        if self.use_delta_target:
            # y_delta = y - x[-1] (从最后一个输入位置的增量)
            y_delta = y - x[-1:, :, :]  # (seq_out, agents, 3)
            # 对增量做全局归一化
            y_target = (y_delta - self.output_mean) / self.output_std
        else:
            y_target = y
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)  # 原始位置用于计算真实MAE
        )


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""
    
    def __init__(self, max_len=256, d_model=128):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EncoderSelfAttnBlock(nn.Module):
    """Transformer 风格自注意力块"""
    
    def __init__(self, d_model=128, num_heads=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 自注意力
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # 前馈网络
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


class EnhancedSwarmGRUModel(nn.Module):
    """
    增强版集群 GRU 模型
    融合单机预测的最佳实践
    """
    
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, 
                 output_size=3, dropout=0.3, use_attention=True):
        super(EnhancedSwarmGRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # ✓ BiGRU特性：推荐开启，单机模型成功经验
        self.use_attention = True  # 强制或者默认为 True
        
        # 特征融合层
        self.feature_fusion = nn.Linear(input_size, hidden_size)
        
        # BiGRU 编码器
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # 可选注意力机制
        if use_attention:
            self.pos_enc = LearnablePositionalEncoding(max_len=256, d_model=hidden_size * 2)
            self.enc_refiner = EncoderSelfAttnBlock(hidden_size * 2, num_heads=4, dropout=dropout)
            # 使用单一的 cross-attention，而非每步都做
            self.decoder_attn = nn.MultiheadAttention(
                hidden_size * 2, num_heads=4, dropout=dropout, batch_first=True
            )
            self.decoder_attn_ln = nn.LayerNorm(hidden_size * 2)
        
        # 解码器输入投影 - 将输出维度投影到隐藏维度
        self.decoder_input_proj = nn.Linear(output_size, hidden_size * 2)
        
        # GRU 解码器
        self.decoder = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 输出投影
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        
        # 平面特异性头（如单机模型）
        head_dim = max(hidden_size, 32)
        self.plane_heads = nn.ModuleDict({
            'xy': nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 2),
            ),
            'yz': nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 2),
            ),
            'xz': nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 2),
            ),
        })
        self.plane_gate = nn.Linear(hidden_size * 2, 3)
        
        self.dropout = nn.Dropout(dropout)
    
    def _merge_bidirectional_hidden(self, h):
        """合并双向隐状态"""
        # h: (num_layers * 2, batch, hidden_size)
        batch_size = h.size(1)
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        h = h.transpose(1, 2).reshape(self.num_layers, batch_size, -1)
        return h
    
    def _compute_plane_predictions(self, hidden_state):
        """计算平面特异性预测"""
        return {
            plane: head(hidden_state)
            for plane, head in self.plane_heads.items()
        }
    
    def _fuse_plane_predictions(self, plane_preds):
        """融合平面预测"""
        xy = plane_preds['xy']
        yz = plane_preds['yz']
        xz = plane_preds['xz']
        
        delta_x = 0.5 * (xy[:, 0] + xz[:, 0])
        delta_y = 0.5 * (xy[:, 1] + yz[:, 0])
        delta_z = 0.5 * (yz[:, 1] + xz[:, 1])
        
        return torch.stack([delta_x, delta_y, delta_z], dim=-1)
    
    def forward(self, x, x_orig, y=None, teacher_forcing_ratio=0.5):
        """
        Args:
            x: 特征张量 (batch, seq_in, agents, 16) 或 (batch*agents, seq_in, 16)
            x_orig: 原始位置 (batch, seq_in, agents, 3) 或 (batch*agents, seq_in, 3)
            y: 目标位置增量或绝对位置
            teacher_forcing_ratio: TF比例
        """
        # 检测输入维度
        if x.dim() == 4:
            batch_size, seq_in, num_agents, feat_dim = x.shape
            x_reshaped = x.reshape(batch_size * num_agents, seq_in, feat_dim)
            x_orig_reshaped = x_orig.reshape(batch_size * num_agents, seq_in, 3)
            if y is not None:
                seq_out = y.shape[1]
                y_reshaped = y.reshape(batch_size * num_agents, seq_out, self.output_size)
            else:
                y_reshaped = None
        else:
            x_reshaped = x
            x_orig_reshaped = x_orig
            y_reshaped = y
            num_agents = None
            batch_size = x.shape[0]
        
        # 特征融合
        x_fused = self.dropout(torch.relu(self.feature_fusion(x_reshaped)))
        
        # BiGRU 编码
        enc_out, h = self.encoder(x_fused)
        h = self._merge_bidirectional_hidden(h)
        
        # 可选注意力
        if self.use_attention:
            enc_out = self.pos_enc(enc_out)
            enc_out = self.enc_refiner(enc_out)
        
        # ✅ 修复：真正的自回归解码（参考单机模型）
        seq_out = y_reshaped.shape[1] if y_reshaped is not None else 10
        predictions = []
        h_t = h
        
        # ------------------------------------------------------------------
        # 初始化：生成初始增量猜测
        # ------------------------------------------------------------------
        q0 = enc_out[:, -1:, :]  # (batch*agents, 1, hidden*2)
        
        # 计算初始 Context
        if self.use_attention:
            ctx0, _ = self.decoder_attn(q0, enc_out, enc_out, need_weights=False)
            ctx0 = self.decoder_attn_ln(q0 + ctx0).squeeze(1)
            plane_source_0 = ctx0
        else:
            ctx0 = h[-1]
            plane_source_0 = ctx0
            
        # 生成初始输出 (Delta 0)
        base_output_0 = self.fc_out(ctx0)
        plane_preds_0 = self._compute_plane_predictions(plane_source_0)
        plane_fused_0 = self._fuse_plane_predictions(plane_preds_0)
        gate_0 = torch.sigmoid(self.plane_gate(plane_source_0))
        
        # 初始 prev_output
        prev_output = plane_fused_0 * gate_0 + base_output_0 * (1.0 - gate_0)
        
        # 逐步解码：每一步都依赖上一步的输出
        for t in range(seq_out):
            # 将上一步输出投影到隐藏维度
            decoder_input = self.decoder_input_proj(prev_output).unsqueeze(1)  # (batch*agents, 1, hidden*2)
            
            # GRU 解码一步
            decoder_out, h_t = self.decoder(decoder_input, h_t)  # decoder_out: (batch*agents, 1, hidden*2)
            
            # ✅ 完整实现Cross-Attention（关键！参考单机模型）
            if self.use_attention:
                h_last = h_t[-1]  # (batch*agents, hidden*2) 最新隐藏状态
                q = h_last.unsqueeze(1)  # (batch*agents, 1, hidden*2) 作为query
                ctx, _ = self.decoder_attn(q, enc_out, enc_out, need_weights=False)  # 查询编码器输出
                ctx = self.decoder_attn_ln(q + ctx).squeeze(1)  # 残差连接 + 层归一化
                plane_source = ctx  # 用于平面头
            else:
                ctx = h_t[-1]
                plane_source = ctx
            
            # 基础输出
            base_output = self.fc_out(ctx)  # (batch*agents, 3)
            
            # 平面特异性预测
            plane_preds = self._compute_plane_predictions(plane_source)
            plane_fused = self._fuse_plane_predictions(plane_preds)
            gate = torch.sigmoid(self.plane_gate(plane_source))
            
            # 融合预测
            y_t = plane_fused * gate + base_output * (1.0 - gate)  # (batch*agents, 3)
            
            predictions.append(y_t.unsqueeze(1))  # (batch*agents, 1, 3)
            
            # ✅ 自适应 Teacher Forcing 决策（参考单机模型最佳实践）
            # 随着时间步增加，减少 Teacher Forcing 的概率，让模型在序列后期更依赖自己
            adaptive_ratio = teacher_forcing_ratio * (1.0 - float(t) / max(1, seq_out))
            
            use_tf = False
            if y_reshaped is not None and adaptive_ratio > 0:
                # 随机决策是否使用 TF
                if torch.rand(1).item() < adaptive_ratio:
                    use_tf = True
            
            if use_tf:
                prev_output = y_reshaped[:, t, :]  # 使用真实增量
            else:
                prev_output = y_t.detach()  # 使用预测结果（断开梯度）
        
        # 拼接所有预测
        output = torch.cat(predictions, dim=1)  # (batch*agents, seq_out, 3)
        
        # 重塑回原始维度
        if num_agents is not None:
            output = output.reshape(batch_size, seq_out, num_agents, self.output_size)
        
        return output


class MultiObjectiveLoss(nn.Module):
    """
    多目标损失函数 - 参考单机模型成功经验的修复版
    
    关键修复：
    1. ✅ 在归一化空间计算损失，避免反归一化数值问题
    2. ✅ 简化损失计算，专注核心目标
    3. ✅ 与单机模型 train_model_bigru_improved.py 完全对齐
    4. ✅ 确保训练loss和MAE一致性
    
    参考：单机模型中的 MultiObjectiveLoss 类
    """
    
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, axis_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 简化轴权重处理（可选特性）
        # ✅ 参考单机模型最佳实践：默认强化 Y 和 Z 轴
        if axis_weights is None:
            axis_weights = [1.0, 1.1, 1.2]  # X, Y, Z
            
        self.register_buffer('axis_weights', torch.tensor(axis_weights, dtype=torch.float32))
        
        logger.info(f"损失函数配置: α={alpha}, β={beta}, γ={gamma}")
        logger.info(f"轴加权: {axis_weights}")
    
    def forward(self, pred, target):
        """
        在归一化空间计算损失 - 与单机模型完全一致
        
        Args:
            pred: 归一化的预测增量 (batch, seq_out, agents, 3)
            target: 归一化的目标增量 (batch, seq_out, agents, 3)
        
        Returns:
            total_loss: 标量损失
        """
        # 1. 位置损失（MSE）- 在归一化空间直接计算
        # ✅ 修复：正确使用轴权重
        axis_w = self.axis_weights.to(pred.device).view(1, 1, 1, 3)
        
        pos_diff = (pred - target) ** 2
        pos_loss = torch.mean(pos_diff * axis_w)
        
        # 2. 加速度损失（二阶差分平滑性）
        if pred.shape[1] > 2:  # seq_out > 2
            pred_acc = torch.diff(torch.diff(pred, dim=1), dim=1)  # (batch, seq-2, agents, 3)
            target_acc = torch.diff(torch.diff(target, dim=1), dim=1)
            
            acc_diff = (pred_acc - target_acc) ** 2
            acc_loss = torch.mean(acc_diff * axis_w)
        else:
            acc_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # 3. 速度损失（一阶差分连续性）
        if pred.shape[1] > 1:  # seq_out > 1
            pred_vel = torch.diff(pred, dim=1)  # (batch, seq-1, agents, 3)
            target_vel = torch.diff(target, dim=1)
            
            vel_diff = (pred_vel - target_vel) ** 2
            vel_loss = torch.mean(vel_diff * axis_w)
        else:
            vel_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # 总损失组合
        total_loss = self.alpha * pos_loss + self.beta * acc_loss + self.gamma * vel_loss
        
        return total_loss


def train_epoch(model, train_loader, optimizer, criterion, device, grad_clip=1.0, 
                scaler=None, use_amp=False, teacher_forcing_ratio=0.5, current_epoch=1, total_epochs=200):
    """
    训练一个 epoch - 简化版（参考单机模型）
    
    ✅ 修复：移除复杂的双目标损失，采用单一TF策略
    """
    model.train()
    total_loss = 0
    count = 0
    
    # 跨Epoch衰减 TF 比率（参考单机模型）
    tf_decay = 0.005  # 每个epoch衰减0.5%
    tf_current = max(0.0, teacher_forcing_ratio - tf_decay * (current_epoch - 1))
    
    for features, x_orig, y, _ in tqdm(train_loader, desc=f"训练 [TF={tf_current:.4f}]"):
        # ✅ 优化：非阻塞数据传输，减少CPU等待时间
        features = features.to(device, non_blocking=True)
        x_orig = x_orig.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # ✅ 简化：单一TF策略（与单机模型一致）
        if use_amp:
            with torch.amp.autocast('cuda'):
                pred = model(features, x_orig, y, teacher_forcing_ratio=tf_current)
                loss = criterion(pred, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(features, x_orig, y, teacher_forcing_ratio=tf_current)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else 0.0, tf_current


def evaluate(model, val_loader, criterion, device):
    """
    评估函数 - 参考单机模型 train_model_bigru_improved.py 的成功经验
    
    关键修复：
    1. ✅ 正确计算MAE：反归一化增量后重建绝对位置
    2. ✅ 与单机模型的evaluate()函数逻辑完全一致
    3. ✅ 确保训练显示的MAE与推理MAE一致
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    count = 0
    
    with torch.no_grad():
        for features, x_orig, y_norm, y_orig in tqdm(val_loader, desc="评估"):
            # ✅ 优化：非阻塞数据传输
            features = features.to(device, non_blocking=True)
            x_orig = x_orig.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)    # 归一化的增量目标
            y_orig = y_orig.to(device, non_blocking=True)     # 原始绝对位置
            
            # 模型预测（归一化增量）
            pred_norm = model(features, x_orig, teacher_forcing_ratio=0.0)
            
            # 损失计算（在归一化空间）
            loss = criterion(pred_norm, y_norm)
            
            # ⭐ MAE计算（在原始物理空间）- 参考单机模型
            # 1. 反归一化增量预测
            if hasattr(val_loader.dataset, 'output_mean') and hasattr(val_loader.dataset, 'output_std'):
                output_mean = val_loader.dataset.output_mean
                output_std = val_loader.dataset.output_std
                
                # ✅ 修复：确保统计量正确广播到设备和形状
                if isinstance(output_mean, np.ndarray):
                    output_mean = torch.tensor(output_mean, device=device, dtype=pred_norm.dtype)
                if isinstance(output_std, np.ndarray):
                    output_std = torch.tensor(output_std, device=device, dtype=pred_norm.dtype)
                
                # 广播到正确形状：(batch, seq_out, agents, 3)
                output_mean = output_mean.view(1, 1, 1, -1)  # (1, 1, 1, 3)
                output_std = output_std.view(1, 1, 1, -1)    # (1, 1, 1, 3)
                
                # 反归一化增量到物理空间
                pred_delta_physical = pred_norm * output_std + output_mean  # (batch, seq_out, agents, 3)
                
                # 重建绝对位置
                last_pos = x_orig[:, -1:, :, :]  # (batch, 1, agents, 3) 最后输入位置
                pred_absolute = last_pos + pred_delta_physical  # (batch, seq_out, agents, 3)
                
                # 计算MAE
                mae = torch.abs(pred_absolute - y_orig).mean().item()
            else:
                # 兜底方案：假设output_std=1（可能不准确）
                logger.warning("⚠️ 无法获取数据集统计量，MAE可能不准确")
                last_pos = x_orig[:, -1:, :, :]
                pred_absolute = last_pos + pred_norm  # 假设增量已是物理单位
                mae = torch.abs(pred_absolute - y_orig).mean().item()
            
            total_loss += loss.item()
            total_mae += mae
            count += 1
    
    avg_loss = total_loss / count if count > 0 else 0.0
    avg_mae = total_mae / count if count > 0 else 0.0
    
    return avg_loss, avg_mae


def load_swarm_data(data_dir, num_agents, batch_size=32, val_split=0.2, 
                   features_dir=None, num_workers=4, prefetch_factor=2):
    """
    加载数据并计算特征统计量（参考单机模型 train_model_bigru_improved.py）
    返回: train_loader, val_loader, 统计量字典
    """
    data_path = Path(data_dir)
    
    X_file = data_path / f'input_agents_{num_agents}.npz'
    Y_file = data_path / f'output_agents_{num_agents}.npz'
    
    if not X_file.exists() or not Y_file.exists():
        raise FileNotFoundError(f"找不到数据文件")
    
    logger.info(f"加载 {num_agents} 架无人机数据...")
    X = np.load(X_file)['data']  # (seq_in, samples, agents, 3)
    Y = np.load(Y_file)['data']  # (seq_out, samples, agents, 3)
    
    # ✅ 修复：NPZ 已经是 (seq_len, samples, agents, 3) 格式，需要转置为 (samples, seq_len, agents, 3)
    X = np.transpose(X, (1, 0, 2, 3))  # (seq_in, samples, agents, 3) → (samples, seq_in, agents, 3)
    Y = np.transpose(Y, (1, 0, 2, 3))  # (seq_out, samples, agents, 3) → (samples, seq_out, agents, 3)
    
    logger.info(f"  输入形状: {X.shape} = (样本数, 输入步长, agents, 坐标维数)")
    logger.info(f"  输出形状: {Y.shape}")
    
    # ========== 计算统计量（参考单机模型） ==========
    logger.info(f"  计算统计量...")
    
    # ✅ 修复：位置数据统计（通道级归一化，参考单机模型）
    input_mean = np.mean(X.reshape(-1, 3), axis=0)  # 形状 (3,) 而非标量
    input_std = np.std(X.reshape(-1, 3), axis=0)    # 形状 (3,) 而非标量
    input_std = np.where(input_std < 1e-8, 1.0, input_std)  # 防止除零
    logger.info(f"    位置统计: mean={input_mean}, std={input_std}")
    
    # ✅ 修复：增量目标统计（通道级归一化）
    y_delta = Y - X[:, -1:, :, :]  # 增量位移 (N, seq_out, agents, 3)
    output_mean = np.mean(y_delta.reshape(-1, 3), axis=0)  # 形状 (3,) 而非标量
    output_std = np.std(y_delta.reshape(-1, 3), axis=0)    # 形状 (3,) 而非标量
    output_std = np.where(output_std < 1e-8, 1.0, output_std)  # 防止除零
    logger.info(f"    增量统计: mean={output_mean}, std={output_std}")
    
    # ========== 尝试加载预计算特征（必须在计算统计之前！）==========
    logger.info(f"  尝试加载预计算特征...")
    features_precomputed = None
    if features_dir is not None:
        features_path = Path(features_dir) / f'features_agents_{num_agents}.npz'
        if features_path.exists():
            logger.info(f"  ✓ 加载预计算特征: {features_path.name}")
            try:
                features_data = np.load(features_path)
                features_precomputed = features_data['features']  # (samples, seq_in, agents, 16)
                logger.info(f"    特征形状: {features_precomputed.shape}")
            except Exception as e:
                logger.warning(f"    ⚠️  加载特征失败: {e}，将使用实时计算")
        else:
            logger.info(f"  ⚠️  未找到预计算特征: {features_path.name}")
    
    # ========== 3. 16维特征统计（优先从预计算特征加载）==========
    # ⭐ 如果预计算特征存在，直接从中计算统计量（快速！）
    # 否则使用默认值（统计量通常影响不大）
    input_mean_all = None
    input_std_all = None
    
    if features_precomputed is not None:
        # 从预计算特征计算统计量
        logger.info(f"  从预计算特征计算16维特征统计量...")
        input_mean_all = np.mean(features_precomputed, axis=(0, 1, 2))  # 平均所有样本、时步、agent
        input_std_all = np.std(features_precomputed, axis=(0, 1, 2))    # (16,)
        input_std_all = np.where(input_std_all < 1e-8, 1.0, input_std_all)
        logger.info(f"    16维特征统计:")
        logger.info(f"      mean_all shape: {input_mean_all.shape}, 前4个: {input_mean_all[:4]}")
        logger.info(f"      std_all shape: {input_std_all.shape}, 前4个: {input_std_all[:4]}")
    else:
        # 未加载预计算特征时使用默认值
        logger.warning(f"  未找到预计算特征，使用默认特征统计量")
        input_mean_all = np.zeros(16)
        input_std_all = np.ones(16)
    
    num_samples = len(X)

    logger.info(f"  使用随机索引打乱样本并切分 train/val (val_split={val_split})")
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    num_val = max(1, int(num_samples * val_split)) if val_split > 0 else 0
    val_sample_idx = indices[:num_val]
    train_sample_idx = indices[num_val:]

    logger.info(f"  分割结果: 训练样本 {len(train_sample_idx)}，验证样本 {len(val_sample_idx)}")

    train_data_X = X[train_sample_idx]
    train_data_Y = Y[train_sample_idx]
    val_data_X = X[val_sample_idx]
    val_data_Y = Y[val_sample_idx]
    
    train_features = features_precomputed[train_sample_idx] if features_precomputed is not None else None
    val_features = features_precomputed[val_sample_idx] if features_precomputed is not None else None
    
    # 🔧 修复 2：使用通道级统计量（已在上面计算），传入 Dataset 中
    # ✅ 修复：传入 feature_mean/std 进行全局归一化
    train_dataset = SwarmTrajectoryDataset(
        train_data_X, train_data_Y, normalize=True, use_delta_target=True,
        features_precomputed=train_features,
        input_mean=input_mean, input_std=input_std,  # ✅ 传入通道级统计量
        output_mean=output_mean, output_std=output_std,
        feature_mean=input_mean_all, feature_std=input_std_all
    )
    val_dataset = SwarmTrajectoryDataset(
        val_data_X, val_data_Y, normalize=True, use_delta_target=True,
        features_precomputed=val_features,
        input_mean=input_mean, input_std=input_std,  # ✅ 传入通道级统计量
        output_mean=output_mean, output_std=output_std,
        feature_mean=input_mean_all, feature_std=input_std_all
    )

    # ✅ 优化：智能DataLoader配置 - 根据数据量和系统配置自适应
    # 大数据集使用更多worker，小数据集减少worker避免开销
    effective_workers = min(num_workers, 8) if len(train_dataset) > 10000 else min(num_workers, 2)
    effective_workers = max(effective_workers, 0)  # 确保非负
    
    use_persistent = effective_workers > 0  # 只有多进程时才启用persistent
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=effective_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if effective_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=max(effective_workers // 2, 1) if effective_workers > 0 else 0,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=use_persistent and effective_workers > 0,
        prefetch_factor=prefetch_factor if effective_workers > 0 else None
    )
    
    logger.info(f"  训练样本: {len(train_dataset)}")
    logger.info(f"  验证样本: {len(val_dataset)}")
    logger.info(f"  数据加载配置:")
    logger.info(f"    训练 workers: {effective_workers}, 验证 workers: {max(effective_workers // 2, 1) if effective_workers > 0 else 0}")
    logger.info(f"    pin_memory: {torch.cuda.is_available()}, persistent_workers: {use_persistent}")
    logger.info(f"    prefetch_factor: {prefetch_factor if effective_workers > 0 else 'N/A'}")
    
    # ========== 返回统计量字典（参考单机模型） ==========
    stats = {
        'input_mean': input_mean,      # ✅ 修复：统一变量名
        'input_std': input_std,        # ✅ 修复：统一变量名
        'input_mean_all': input_mean_all,
        'input_std_all': input_std_all,
        'output_mean': output_mean,
        'output_std': output_std,
    }
    
    return train_loader, val_loader, stats


def main():
    parser = argparse.ArgumentParser(description='训练增强版集群轨迹模型')
    parser.add_argument('--data_dir', type=str, default='swarm_segments',
                        help='数据目录')
    parser.add_argument('--agents', type=str, default='3',
                        help='无人机数量 (3|4|5|6|all)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='GRU 隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='GRU 层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 比例')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪值')
    parser.add_argument('--patience', type=int, default=30,
                        help='早停耐心值')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='Teacher Forcing 初始比例')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--output_dir', type=str, default='newloss_swarm_models_enhanced',
                        help='模型保存目录')
    parser.add_argument('--features_dir', type=str, default=None,
                        help='预计算特征目录 (如果为空则实时计算)')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='使用 Cross-Attention (默认 True)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader进程数 (Windows上必须为0，Linux/Mac可用2-8)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='每个worker的预取批次数')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if args.agents == 'all':
        agents_list = [3, 4, 5, 6]
    else:
        agents_list = [int(args.agents)]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"使用 AMP: {args.use_amp}")
    logger.info(f"使用 Attention: {args.use_attention}")
    
    for num_agents in agents_list:
        logger.info(f"\n{'='*70}")
        logger.info(f"训练 {num_agents} 架无人机模型 (增强版)")
        logger.info(f"{'='*70}")
        
        try:
            train_loader, val_loader, stats = load_swarm_data(
                args.data_dir, num_agents, args.batch_size, args.val_split,
                features_dir=args.features_dir, num_workers=args.num_workers, 
                prefetch_factor=args.prefetch_factor
            )
            # 解包统计量
            input_mean = stats['input_mean']      # ✅ 修复：统一变量名
            input_std = stats['input_std']        # ✅ 修复：统一变量名
            input_mean_all = stats['input_mean_all']
            input_std_all = stats['input_std_all']
            output_mean = stats['output_mean']
            output_std = stats['output_std']
        except FileNotFoundError as e:
            logger.error(f"跳过 {num_agents} 架无人机: {e}")
            continue
        
        model = EnhancedSwarmGRUModel(
            input_size=16,
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
        
        # ✅ 修复：使用简化的损失函数（参考单机模型）
        criterion = MultiObjectiveLoss(
            alpha=0.7, beta=0.2, gamma=0.1,
            # 移除 output_std 参数，在归一化空间直接计算损失
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 使用现代 PyTorch API 避免过时警告
        scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
        
        # 🔄 **改进：智能检查点恢复逻辑**
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
        
        # 尝试恢复逻辑顺序：
        # 1. 显式指定的 --resume 参数
        # 2. 自动检测最后一个检查点（支持意外中断恢复）
        # 3. 中断检查点（优先级低）
        
        ckpt_last = Path(args.output_dir) / f'last_checkpoint_agents_{num_agents}.pt'
        ckpt_interrupted = Path(args.output_dir) / f'interrupted_checkpoint_agents_{num_agents}.pt'
        
        if args.resume:
            # 显式指定检查点
            checkpoint_path = Path(args.output_dir) / args.resume
            if checkpoint_path.exists():
                logger.info(f"从指定检查点恢复: {args.resume}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                patience_counter = checkpoint.get('patience_counter', 0)
                training_history = checkpoint.get('training_history', training_history)
                logger.info(f"✓ 已恢复到 epoch {start_epoch}")
        elif ckpt_last.exists():
            # 自动检测最后的检查点
            logger.info(f"检测到最后的检查点，自动恢复...")
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
        elif ckpt_interrupted.exists():
            # 中断检查点（降级恢复）
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
        
        print("=" * 100)
        print(f"{'Epoch':<8} {'Train Loss':<16} {'Val Loss':<16} {'MAE (m)':<16} {'LR':<12} {'TF Ratio':<12} {'Status':<20}")
        print("=" * 100)
        
        for epoch in range(start_epoch, args.epochs):
            try:
                # ✅ 改进：正确传入epoch信息用于TF衰减
                train_loss, tf_current = train_epoch(
                    model, train_loader, optimizer, criterion, device,
                    grad_clip=args.grad_clip, scaler=scaler, use_amp=args.use_amp,
                    teacher_forcing_ratio=args.teacher_forcing_ratio,
                    current_epoch=epoch+1, total_epochs=args.epochs
                )
                
                val_loss, val_mae = evaluate(model, val_loader, criterion, device)
                
                # ✅ 新增：记录训练历史
                training_history['epoch'].append(epoch + 1)
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                training_history['val_mae'].append(val_mae)
                current_lr = optimizer.param_groups[0]['lr']
                training_history['learning_rate'].append(current_lr)
                training_history['teacher_forcing_ratio'].append(tf_current)
                
                # 🔄 **改进：每个epoch都保存最后一个模型（用于恢复）**
                ckpt_last_path = Path(args.output_dir) / f'last_checkpoint_agents_{num_agents}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'training_history': training_history,
                    # ⭐ 保存所有统计量
                    'input_mean': input_mean.tolist() if hasattr(input_mean, 'tolist') else input_mean,
                    'input_std': input_std.tolist() if hasattr(input_std, 'tolist') else input_std,
                    'input_mean_all': input_mean_all.tolist() if hasattr(input_mean_all, 'tolist') else input_mean_all,
                    'input_std_all': input_std_all.tolist() if hasattr(input_std_all, 'tolist') else input_std_all,
                    'output_mean': output_mean.tolist() if hasattr(output_mean, 'tolist') else output_mean,
                    'output_std': output_std.tolist() if hasattr(output_std, 'tolist') else output_std,
                    'config': {
                        'input_size': 16,
                        'hidden_size': args.hidden_size,
                        'num_layers': args.num_layers,
                        'use_attention': args.use_attention,
                    }
                }, ckpt_last_path)
                
                status = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    status = "✓ BEST"
                    
                    # 🔄 **额外保存最佳模型**
                    best_model_path = Path(args.output_dir) / f'best_model_agents_{num_agents}.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'input_mean': input_mean.tolist() if hasattr(input_mean, 'tolist') else input_mean,
                        'input_std': input_std.tolist() if hasattr(input_std, 'tolist') else input_std,
                        'input_mean_all': input_mean_all.tolist() if hasattr(input_mean_all, 'tolist') else input_mean_all,
                        'input_std_all': input_std_all.tolist() if hasattr(input_std_all, 'tolist') else input_std_all,
                        'output_mean': output_mean.tolist() if hasattr(output_mean, 'tolist') else output_mean,
                        'output_std': output_std.tolist() if hasattr(output_std, 'tolist') else output_std,
                        'config': {
                            'input_size': 16,
                            'hidden_size': args.hidden_size,
                            'num_layers': args.num_layers,
                            'use_attention': args.use_attention,
                        }
                    }, best_model_path)
                    
                    # 同时保存统计量到独立的 .npz 文件
                    stats_path = Path(args.output_dir) / f'norm_stats_agents_{num_agents}.npz'
                    np.savez(
                        stats_path,
                        input_mean=input_mean,
                        input_std=input_std,
                        input_mean_all=input_mean_all,
                        input_std_all=input_std_all,
                        output_mean=output_mean,
                        output_std=output_std,
                    )
                else:
                    patience_counter += 1
                    status = f"patience: {patience_counter}/{args.patience}"
                    if patience_counter >= args.patience:
                        print(f"{epoch+1:<8} {train_loss:<16.6f} {val_loss:<16.6f} {val_mae:<16.6f} {current_lr:<12.2e} {tf_current:<12.4f} {'EARLY STOP':<20}")
                        print(f"早停 (patience={args.patience})")
                        break
                
                print(f"{epoch+1:<8} {train_loss:<16.6f} {val_loss:<16.6f} {val_mae:<16.6f} {current_lr:<12.2e} {tf_current:<12.4f} {status:<20}")
                
                scheduler.step(val_loss)
                
                # 🔄 **实时更新训练历史CSV（每个epoch保存一次）**
                csv_path = Path(args.output_dir) / f'training_history_agents_{num_agents}.csv'
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
                
                # 🔄 **实时更新训练配置JSON（每个epoch保存一次）**
                config_dict = {
                    'timestamp': datetime.now().isoformat(),
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
                    'current_epoch': epoch + 1,
                    'val_min_mae': min(training_history['val_mae']) if training_history['val_mae'] else 0.0,
                    'best_val_loss': best_val_loss,
                    'current_val_loss': val_loss,
                    'current_val_mae': val_mae,
                }
                config_path = Path(args.output_dir) / f'training_config_agents_{num_agents}.json'
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=4, ensure_ascii=False)
                
            except KeyboardInterrupt:
                print(f"\n⚠ 收到中断信号，正在保存检查点...")
                ckpt_interrupt_path = Path(args.output_dir) / f'interrupted_checkpoint_agents_{num_agents}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'training_history': training_history,
                    'input_mean': input_mean.tolist() if hasattr(input_mean, 'tolist') else input_mean,
                    'input_std': input_std.tolist() if hasattr(input_std, 'tolist') else input_std,
                    'input_mean_all': input_mean_all.tolist() if hasattr(input_mean_all, 'tolist') else input_mean_all,
                    'input_std_all': input_std_all.tolist() if hasattr(input_std_all, 'tolist') else input_std_all,
                    'output_mean': output_mean.tolist() if hasattr(output_mean, 'tolist') else output_mean,
                    'output_std': output_std.tolist() if hasattr(output_std, 'tolist') else output_std,
                    'config': {
                        'input_size': 16,
                        'hidden_size': args.hidden_size,
                        'num_layers': args.num_layers,
                        'use_attention': args.use_attention,
                    }
                }, ckpt_interrupt_path)
                logger.info(f"✓ 中断检查点已保存: {ckpt_interrupt_path}")
                break
            except Exception as e:
                logger.error(f"❌ Epoch {epoch+1} 发生异常: {e}")
                logger.error("正在保存紧急检查点...")
                ckpt_error_path = Path(args.output_dir) / f'error_checkpoint_agents_{num_agents}_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'training_history': training_history,
                    'input_mean': input_mean.tolist() if hasattr(input_mean, 'tolist') else input_mean,
                    'input_std': input_std.tolist() if hasattr(input_std, 'tolist') else input_std,
                    'input_mean_all': input_mean_all.tolist() if hasattr(input_mean_all, 'tolist') else input_mean_all,
                    'input_std_all': input_std_all.tolist() if hasattr(input_std_all, 'tolist') else input_std_all,
                    'output_mean': output_mean.tolist() if hasattr(output_mean, 'tolist') else output_mean,
                    'output_std': output_std.tolist() if hasattr(output_std, 'tolist') else output_std,
                    'error': str(e),
                    'config': {
                        'input_size': 16,
                        'hidden_size': args.hidden_size,
                        'num_layers': args.num_layers,
                        'use_attention': args.use_attention,
                    }
                }, ckpt_error_path)
                logger.info(f"✓ 错误检查点已保存: {ckpt_error_path}")
                break
        
        print("=" * 100)
        
        logger.info(f"✓ 训练完成!")
        logger.info(f"最佳模型: best_model_agents_{num_agents}.pt (Loss: {best_val_loss:.6f})")


if __name__ == '__main__':
    main()
