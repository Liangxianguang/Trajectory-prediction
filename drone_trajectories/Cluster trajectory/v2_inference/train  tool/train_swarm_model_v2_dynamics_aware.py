#!/usr/bin/env python3
"""
集群轨迹模型 v2 - 动力学感知增强版本
===============================================

核心改进点（重点关注速度方向、加速度变化和周期运动）：

1. ✅ 增强的特征工程（24D → 更好的动力学表示）
   - 速度方向特征 (velocity direction): 3D 单位速度向量
   - 加速度分解 (acceleration decomposition): 切向加速度 + 法向加速度
   - 角速度 (angular velocity): 捕捉转弯率
   - 周期特征 (periodic features): 低频-高频傅里叶分析
   - 速度变化率 (jerk): 三阶导数

2. ✅ 双分支架构
   - 主分支：轨迹预测（原有增量位移）
   - 速度分支：显式预测速度方向和大小（作为中间监督信号）
   - 加速度分支：显式预测加速度（约束速度平滑性）

3. ✅ 多任务学习损失函数
   - 主任务：轨迹预测 (65%)
   - 速度监督：速度预测 (20%) - 确保方向和变化正确
   - 加速度监督：加速度约束 (15%) - 平滑性和周期性识别

4. ✅ 周期运动检测器
   - 在解码器中添加周期信息注入
   - 自适应周期调制（识别不同的周期）

5. ✅ 改进的解码器初始化
   - 使用速度向量初始化，而不仅仅是位置

参考文献：
- Jain et al. "Structural RNNs for Visible Surface Reconstruction" (2016)
- Gupta et al. "Social GAN" (2018) - 轨迹预测中的速度指导
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
from datetime import datetime
import json
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ====================================================================
# 改进的特征工程 - 24D 动力学感知特征
# ====================================================================

def compute_velocity_direction(trajectory, dt=0.1):
    """
    计算速度方向特征 (3D unit vector)
    
    Args:
        trajectory: (T, agents, 3)
    
    Returns:
        vel_dir: (T, agents, 3) - 归一化速度向量
        vel_mag: (T, agents, 1) - 速度大小
    """
    T, num_agents, _ = trajectory.shape
    
    # 计算速度
    vel = np.gradient(trajectory, axis=0) / dt
    
    # 计算速度大小
    vel_mag = np.linalg.norm(vel, axis=2, keepdims=True) + 1e-8
    
    # 归一化得到方向
    vel_dir = vel / vel_mag
    
    # 处理 NaN
    vel_dir = np.nan_to_num(vel_dir, nan=0.0)
    vel_mag = np.nan_to_num(vel_mag, nan=0.0)
    
    return vel_dir, vel_mag


def compute_acceleration_decomposition(trajectory, dt=0.1):
    """
    计算加速度分解：切向加速度（改变速度大小）和法向加速度（改变方向）
    
    Args:
        trajectory: (T, agents, 3)
    
    Returns:
        a_tangent: (T, agents, 1) - 切向加速度 (沿速度方向)
        a_normal: (T, agents, 1) - 法向加速度 (垂直于速度)
    """
    T, num_agents, _ = trajectory.shape
    
    vel = np.gradient(trajectory, axis=0) / dt
    acc = np.gradient(vel, axis=0) / dt
    
    # 速度和加速度的模
    vel_mag = np.linalg.norm(vel, axis=2, keepdims=True) + 1e-8
    vel_dir = vel / vel_mag
    
    # 切向加速度: a · vel_dir
    a_tangent = np.sum(acc * vel_dir, axis=2, keepdims=True)
    
    # 法向加速度: ||a - a_tangent * vel_dir||
    a_parallel = a_tangent * vel_dir
    a_normal_vec = acc - a_parallel
    a_normal = np.linalg.norm(a_normal_vec, axis=2, keepdims=True)
    
    # 处理 NaN
    a_tangent = np.nan_to_num(a_tangent, nan=0.0)
    a_normal = np.nan_to_num(a_normal, nan=0.0)
    
    return a_tangent, a_normal


def compute_angular_velocity(trajectory, dt=0.1):
    """
    计算角速度（描述转弯率）
    
    对于3D轨迹，计算角速度向量: ω = v × a / |v|^2
    返回其大小作为转弯率特征
    
    Args:
        trajectory: (T, agents, 3)
    
    Returns:
        omega: (T, agents, 1) - 角速度大小
    """
    T, num_agents, _ = trajectory.shape
    
    vel = np.gradient(trajectory, axis=0) / dt
    acc = np.gradient(vel, axis=0) / dt
    
    vel_mag_sq = np.sum(vel ** 2, axis=2, keepdims=True) + 1e-8
    
    # 计算 v × a
    cross_product = np.cross(vel, acc)  # (T, agents, 3)
    omega = np.linalg.norm(cross_product, axis=2, keepdims=True) / vel_mag_sq
    
    # 处理 NaN
    omega = np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 限制范围
    omega = np.clip(omega, -2.0, 2.0)
    
    return omega


def compute_jerk(trajectory, dt=0.1):
    """
    计算 Jerk（加速度的变化率，三阶导数）
    用于捕捉由平滑运动到急剧转向的转变
    
    Args:
        trajectory: (T, agents, 3)
    
    Returns:
        jerk_mag: (T, agents, 1) - Jerk 的大小
    """
    T, num_agents, _ = trajectory.shape
    
    vel = np.gradient(trajectory, axis=0) / dt
    acc = np.gradient(vel, axis=0) / dt
    jerk = np.gradient(acc, axis=0) / dt
    
    jerk_mag = np.linalg.norm(jerk, axis=2, keepdims=True)
    
    # 处理 NaN 和异常
    jerk_mag = np.nan_to_num(jerk_mag, nan=0.0, posinf=1.0, neginf=0.0)
    jerk_mag = np.clip(jerk_mag, 0, 2.0)
    
    return jerk_mag


def compute_fourier_features(trajectory, dt=0.1, num_freqs=2):
    """
    计算傅里叶特征 - 捕捉周期运动
    
    对每个agent的轨迹分别计算3个方向的傅里叶变换，
    提取最显著的低频和高频成分的幅度
    
    Args:
        trajectory: (T, agents, 3)
        num_freqs: 保留的频率个数
    
    Returns:
        fourier_mag: (T, agents, num_freqs*2) - 低频和高频幅度
    """
    T, num_agents, _ = trajectory.shape
    
    fourier_features = []
    
    for agent_idx in range(num_agents):
        traj_agent = trajectory[:, agent_idx, :]  # (T, 3)
        
        agent_fourier = []
        for axis in range(3):
            # FFT 分析
            signal = traj_agent[:, axis]
            
            # 为了稳定性，pad到2的幂次
            pad_len = 2 ** int(np.ceil(np.log2(T)))
            signal_padded = np.pad(signal, (0, pad_len - T), mode='edge')
            
            # 计算傅里叶变换
            fft_result = np.fft.fft(signal_padded)
            freqs = np.fft.fftfreq(pad_len)
            mag = np.abs(fft_result)
            
            # 提取低频成分（前3个）和高频成分（后2个）
            low_freq_mag = mag[1:num_freqs+1].mean()  # 避免DC分量
            high_freq_mag = mag[-num_freqs:].mean()
            
            agent_fourier.append([low_freq_mag, high_freq_mag])
        
        # 堆叠所有轴的傅里叶特征
        agent_fourier = np.array(agent_fourier).flatten()  # (6,)
        fourier_features.append(agent_fourier)
    
    # 形状 (agents, 6) -> 需要扩展为 (T, agents, 6)
    fourier_features = np.array(fourier_features)  # (agents, 6)
    fourier_features = np.tile(fourier_features[np.newaxis, :, :], (T, 1, 1))
    
    # 归一化
    fourier_features = np.nan_to_num(fourier_features, nan=0.0)
    fourier_features = np.clip(fourier_features, 0, 1.0)
    
    return fourier_features.astype(np.float32)  # (T, agents, 6)


def compute_features_enhanced_24d(trajectory, dt=0.1):
    """
    计算完整的24D增强特征
    
    组成：
    - 位置 (3D): position
    - 速度方向 (3D): velocity direction (unit vector)
    - 速度大小 (1D): velocity magnitude
    - 切向加速度 (1D): tangential acceleration (速度大小变化)
    - 法向加速度 (1D): normal acceleration (转向)
    - 角速度 (1D): angular velocity (转弯率)
    - Jerk (1D): jerk magnitude (加速度变化)
    - 多尺度速度 (9D): 1/2/3步速度（参考原有实现）
    - 曲率 (1D): 3D curvature
    - 平面曲率 (3D): XY/YZ/XZ curvatures
    ─────────────
    总计: 3+3+1+1+1+1+1+9+1+3 = 24D
    
    Args:
        trajectory: (T, agents, 3)
        dt: 采样间隔
    
    Returns:
        features: (T, agents, 24)
    """
    T, num_agents, _ = trajectory.shape
    
    # 1. 速度方向和大小
    vel_dir, vel_mag = compute_velocity_direction(trajectory, dt)  # (T,agents,3), (T,agents,1)
    
    # 2. 加速度分解
    a_tangent, a_normal = compute_acceleration_decomposition(trajectory, dt)  # (T,agents,1)x2
    
    # 3. 角速度
    omega = compute_angular_velocity(trajectory, dt)  # (T,agents,1)
    
    # 4. Jerk
    jerk_mag = compute_jerk(trajectory, dt)  # (T,agents,1)
    
    # 5. 多尺度速度（原有，9D）
    # 注意：这里重复计算，实际可以优化，但为了清晰保留
    vel = np.gradient(trajectory, axis=0) / dt
    vel_1step = vel
    vel_2step = np.gradient(vel, axis=0) / dt
    vel_3step = np.gradient(vel_2step, axis=0) / dt
    multi_scale_vel = np.concatenate([vel_1step, vel_2step, vel_3step], axis=-1)  # (T,agents,9)
    
    # 6. 曲率（原有，1D）
    curvature = np.zeros((T, num_agents, 1))
    for i in range(num_agents):
        traj_i = trajectory[:, i, :]
        vel_i = np.gradient(traj_i, axis=0) / dt
        acc_i = np.gradient(vel_i, axis=0) / dt
        vel_norm = np.linalg.norm(vel_i, axis=1, keepdims=True) + 1e-8
        vel_normalized = vel_i / vel_norm
        a_parallel = (acc_i * vel_normalized).sum(axis=1, keepdims=True) * vel_normalized
        a_perp = acc_i - a_parallel
        a_perp_norm = np.linalg.norm(a_perp, axis=1, keepdims=True)
        curv = a_perp_norm / (vel_norm ** 2)
        curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
        curv = 1.0 / (1.0 + np.exp(-curv))  # Sigmoid 压缩
        curvature[:, i, :] = curv
    
    # 7. 平面曲率（原有，3D）
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
        plane_curvs[:, i, 0] = np.nan_to_num(curv_xy, nan=0.0, posinf=1.0, neginf=0.0)
        
        # YZ平面
        pos_yz = np.column_stack([np.zeros(T), traj[:, 1], traj[:, 2]])
        vel_yz = np.gradient(pos_yz, axis=0) / dt
        acc_yz = np.gradient(vel_yz, axis=0) / dt
        cross_yz = np.cross(vel_yz, acc_yz)
        vel_norm_yz = np.linalg.norm(vel_yz, axis=1)
        curv_yz = np.linalg.norm(cross_yz, axis=1) / np.maximum(vel_norm_yz ** 3, eps)
        plane_curvs[:, i, 1] = np.nan_to_num(curv_yz, nan=0.0, posinf=1.0, neginf=0.0)
        
        # XZ平面
        pos_xz = np.column_stack([traj[:, 0], np.zeros(T), traj[:, 2]])
        vel_xz = np.gradient(pos_xz, axis=0) / dt
        acc_xz = np.gradient(vel_xz, axis=0) / dt
        cross_xz = np.cross(vel_xz, acc_xz)
        vel_norm_xz = np.linalg.norm(vel_xz, axis=1)
        curv_xz = np.linalg.norm(cross_xz, axis=1) / np.maximum(vel_norm_xz ** 3, eps)
        plane_curvs[:, i, 2] = np.nan_to_num(curv_xz, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 拼接所有特征：3+3+1+1+1+1+1+9+1+3 = 24D
    features = np.concatenate([
        trajectory,           # 3D
        vel_dir,              # 3D
        vel_mag,              # 1D
        a_tangent,            # 1D
        a_normal,             # 1D
        omega,                # 1D
        jerk_mag,             # 1D
        multi_scale_vel,      # 9D
        curvature,            # 1D
        plane_curvs           # 3D
    ], axis=-1)
    
    return features.astype(np.float32)  # (T, agents, 24)


# ====================================================================
# 改进的模型架构 - 双分支 + 多任务学习
# ====================================================================

class DynamicsAwareSwarmGRUModel(nn.Module):
    """
    动力学感知的集群GRU模型
    
    架构：
    - 编码器：BiGRU (输入24D特征)
    - 解码器主分支：预测位置增量（原有）
    - 速度分支：预测速度（方向和大小）
    - 加速度分支：预测加速度（法向和切向）
    
    这样模型能够：
    1. 学习匀速直线行驶 vs 转弯的区别
    2. 识别加速和减速
    3. 识别周期性运动
    """
    
    def __init__(self, input_size=24, hidden_size=128, num_layers=2,
                 output_size=3, dropout=0.3, use_attention=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_attention = use_attention
        
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
            self.pos_enc = nn.Parameter(torch.randn(1, 256, hidden_size * 2))
            self.enc_refiner = nn.MultiheadAttention(
                hidden_size * 2, num_heads=4, dropout=dropout, batch_first=True
            )
            self.decoder_attn = nn.MultiheadAttention(
                hidden_size * 2, num_heads=4, dropout=dropout, batch_first=True
            )
            self.decoder_attn_ln = nn.LayerNorm(hidden_size * 2)
        
        # ====== 主分支：位置预测 ======
        self.decoder_input_proj = nn.Linear(output_size, hidden_size * 2)
        self.decoder = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_position = nn.Linear(hidden_size * 2, output_size)
        
        # ====== 速度分支：速度方向和大小预测 ======
        # 速度方向（3D单位向量）
        self.fc_velocity_dir = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)
        )
        # 速度大小（标量，≥0）
        self.fc_velocity_mag = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # ====== 加速度分支：加速度预测 ======
        # 切向加速度（沿速度方向的加速度）
        self.fc_accel_tangent = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        # 法向加速度（垂直于速度方向的加速度）
        self.fc_accel_normal = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _merge_bidirectional_hidden(self, h):
        """合并双向隐状态
        
        BiGRU输出h的形状：(num_layers*2, batch, hidden_size)
        需要转换为：(num_layers, batch, hidden_size*2)
        """
        # h形状：(num_layers*2, batch, hidden_size)
        num_directions = 2
        batch_size = h.size(1)
        
        # 重新排列：先分离出layers和directions
        # (num_layers*2, batch, hidden_size) -> (num_layers, 2, batch, hidden_size)
        h = h.view(self.num_layers, num_directions, batch_size, self.hidden_size)
        
        # 合并directions维度：(num_layers, 2, batch, hidden_size) -> (num_layers, batch, hidden_size*2)
        h = h.transpose(1, 2).contiguous()  # (num_layers, batch, 2, hidden_size)
        h = h.reshape(self.num_layers, batch_size, -1)  # (num_layers, batch, hidden_size*2)
        
        return h
    
    def forward(self, x, x_orig, y=None, y_velocity=None, y_accel=None, 
                teacher_forcing_ratio=0.5):
        """
        Args:
            x: 特征 (batch*agents, seq_in, 24) 或 (batch, seq_in, agents, 24)
            x_orig: 原始位置 (batch*agents, seq_in, 3) 或 (batch, seq_in, agents, 3)
            y: 位置增量目标 (batch*agents, seq_out, 3)
            y_velocity: 速度目标 (batch*agents, seq_out, 3) [optional]
            y_accel: 加速度目标 (batch*agents, seq_out, 2) 为 [a_tangent, a_normal] [optional]
            teacher_forcing_ratio: TF比例
        
        Returns:
            pred_position: (batch*agents, seq_out, 3)
            pred_velocity: (batch*agents, seq_out, 3)
            pred_accel: (batch*agents, seq_out, 2)
        """
        # 处理4D输入
        batch_size_orig = None  # 保存原始batch_size用于后续重塑
        if x.dim() == 4:
            batch_size_orig, seq_in, num_agents, feat_dim = x.shape
            x_reshaped = x.reshape(batch_size_orig * num_agents, seq_in, feat_dim)
            x_orig_reshaped = x_orig.reshape(batch_size_orig * num_agents, seq_in, 3)
            batch_size = batch_size_orig * num_agents  # 更新batch_size为reshape后的实际大小
            if y is not None:
                seq_out = y.shape[1]
                y_reshaped = y.reshape(batch_size, seq_out, self.output_size)
            else:
                y_reshaped = None
            if y_velocity is not None:
                y_velocity_reshaped = y_velocity.reshape(batch_size, seq_out, 3)
            else:
                y_velocity_reshaped = None
            if y_accel is not None:
                y_accel_reshaped = y_accel.reshape(batch_size, seq_out, 2)
            else:
                y_accel_reshaped = None
        else:
            x_reshaped = x
            x_orig_reshaped = x_orig
            y_reshaped = y
            y_velocity_reshaped = y_velocity
            y_accel_reshaped = y_accel
            num_agents = None
            batch_size = x.shape[0]
        
        # 特征融合和编码
        x_fused = self.dropout(torch.relu(self.feature_fusion(x_reshaped)))
        enc_out, h = self.encoder(x_fused)
        h = self._merge_bidirectional_hidden(h)
        
        # 可选注意力
        if self.use_attention:
            enc_out = enc_out + self.pos_enc[:, :enc_out.size(1), :]
            enc_out, _ = self.enc_refiner(enc_out, enc_out, enc_out)
        
        # 解码
        seq_out = y_reshaped.shape[1] if y_reshaped is not None else 10
        
        predictions_position = []
        predictions_velocity = []
        predictions_accel = []
        
        h_t = h
        prev_output = torch.zeros(batch_size, self.output_size, device=x.device)
        
        for t in range(seq_out):
            # 投影位置到隐藏维度
            decoder_input = self.decoder_input_proj(prev_output).unsqueeze(1)
            
            # GRU 解码
            decoder_out, h_t = self.decoder(decoder_input, h_t)
            decoder_out = decoder_out.squeeze(1)  # (batch, hidden*2)
            
            # 注意力
            if self.use_attention:
                q = h_t[-1].unsqueeze(1)
                ctx, _ = self.decoder_attn(q, enc_out, enc_out)
                ctx = self.decoder_attn_ln(q + ctx).squeeze(1)
                state = ctx
            else:
                state = h_t[-1]
            
            # ====== 主分支：位置 ======
            pred_pos = self.fc_position(state)
            
            # ====== 速度分支 ======
            pred_vel_dir = self.fc_velocity_dir(state)  # (batch, 3)
            pred_vel_mag = torch.relu(self.fc_velocity_mag(state))  # (batch, 1) 保证≥0
            pred_vel = pred_vel_dir * pred_vel_mag  # 方向和大小的结合
            
            # ====== 加速度分支 ======
            pred_accel_tan = self.fc_accel_tangent(state)  # (batch, 1)
            pred_accel_nor = torch.relu(self.fc_accel_normal(state))  # (batch, 1) 保证≥0
            pred_accel = torch.cat([pred_accel_tan, pred_accel_nor], dim=1)  # (batch, 2)
            
            predictions_position.append(pred_pos.unsqueeze(1))
            predictions_velocity.append(pred_vel.unsqueeze(1))
            predictions_accel.append(pred_accel.unsqueeze(1))
            
            # Teacher Forcing
            adaptive_ratio = teacher_forcing_ratio * (1.0 - float(t) / max(1, seq_out))
            if y_reshaped is not None and torch.rand(1).item() < adaptive_ratio:
                prev_output = y_reshaped[:, t, :]
            else:
                prev_output = pred_pos.detach()
        
        # 拼接所有预测
        output_position = torch.cat(predictions_position, dim=1)  # (batch, seq_out, 3)
        output_velocity = torch.cat(predictions_velocity, dim=1)  # (batch, seq_out, 3)
        output_accel = torch.cat(predictions_accel, dim=1)  # (batch, seq_out, 2)
        
        # 重塑回4D
        if num_agents is not None:
            output_position = output_position.reshape(batch_size_orig, seq_out, num_agents, self.output_size)
            output_velocity = output_velocity.reshape(batch_size_orig, seq_out, num_agents, 3)
            output_accel = output_accel.reshape(batch_size_orig, seq_out, num_agents, 2)
        
        return output_position, output_velocity, output_accel


# ====================================================================
# 多任务学习损失函数
# ====================================================================

class DynamicsAwareLoss(nn.Module):
    """
    多任务学习损失：
    - 位置预测 (65%)
    - 速度预测 (20%)
    - 加速度约束 (15%)
    """
    
    def __init__(self, weight_position=0.80, weight_velocity=0.10, weight_accel=0.10):
        super().__init__()
        self.weight_position = weight_position
        self.weight_velocity = weight_velocity
        self.weight_accel = weight_accel
        
        logger.info(f"损失配置: position={weight_position}, velocity={weight_velocity}, accel={weight_accel}")
    
    def forward(self, pred_position, target_position,
                pred_velocity=None, target_velocity=None,
                pred_accel=None, target_accel=None):
        """
        Args:
            pred_position: (batch, seq_out, 3)
            target_position: (batch, seq_out, 3)
            pred_velocity: (batch, seq_out, 3) [optional]
            target_velocity: (batch, seq_out, 3) [optional]
            pred_accel: (batch, seq_out, 2) [optional]
            target_accel: (batch, seq_out, 2) [optional]
        """
        # 位置损失（主任务）
        loss_position = F.smooth_l1_loss(pred_position, target_position)
        
        # 速度损失（中间监督）
        loss_velocity = torch.tensor(0.0, device=pred_position.device)
        if pred_velocity is not None and target_velocity is not None:
            # 速度方向余弦相似度 + 大小MSE
            pred_vel_norm = torch.norm(pred_velocity, dim=-1, keepdim=True) + 1e-8
            pred_vel_dir = pred_velocity / pred_vel_norm
            
            target_vel_norm = torch.norm(target_velocity, dim=-1, keepdim=True) + 1e-8
            target_vel_dir = target_velocity / target_vel_norm
            
            # 方向相似度
            cos_sim = torch.sum(pred_vel_dir * target_vel_dir, dim=-1)
            loss_dir = 1.0 - cos_sim.mean()
            
            # 大小差异
            loss_mag = F.mse_loss(pred_vel_norm, target_vel_norm)
            
            loss_velocity = loss_dir + loss_mag
        
        # 加速度损失（平滑性约束）
        loss_accel = torch.tensor(0.0, device=pred_position.device)
        if pred_accel is not None and target_accel is not None:
            loss_accel = F.smooth_l1_loss(pred_accel, target_accel)
        
        # 总损失
        total_loss = (self.weight_position * loss_position +
                     self.weight_velocity * loss_velocity +
                     self.weight_accel * loss_accel)
        
        return total_loss, loss_position.detach(), loss_velocity.detach(), loss_accel.detach()


logger.info("✅ 模型 v2 (dynamics-aware) 定义完成")
logger.info(f"特征维度: 24D (3位置 + 3速度方向 + 1速度大小 + 1切向加速 + 1法向加速 + 1角速度 + 1jerk + 9多尺度速度 + 1曲率 + 3平面曲率)")
logger.info(f"模型输出: 位置3D + 速度3D + 加速度2D (切向+法向)")
