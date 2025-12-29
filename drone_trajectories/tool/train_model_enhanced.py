#!/usr/bin/env python3
"""
增强版 GRU 轨迹预测模型训练脚本
核心改进：
1. 增量位移预测（而非绝对坐标）—— 更稳定的分布
2. 多尺度速度 + 曲率特征作为额外输入通道
3. 加速度正则化损失（约束高阶导数）
4. 改进的自适应 Teacher Forcing 策略
5. 物理约束积分（推理时使用）
cd /d "D:\Trajectory prediction\drone_trajectories"
python tool\train_model_enhanced.py ^
  --data_path dataset_position_segments_synth.npz ^
  --output_dir tool\gru_models_enhanced ^
  --model_name enhanced_gru_model ^
  --epochs 120 ^
  --batch_size 512 ^
  --hidden_dim 128 ^
  --num_layers 3 ^
  --use_amp

python tool\train_model_enhanced.py ^
  --data_path dataset_position_segments_synth.npz ^
  --output_dir tool\new_gru_models_enhanced ^
  --model_name new_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 4096 ^
  --hidden_dim 128 ^
  --num_layers 3 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp

  python tool\train_model_enhanced.py ^
  --data_path dataset_position_segments_synth.npz ^
  --output_dir tool\long_gru_models_enhanced ^
  --model_name long_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 4096 ^
  --hidden_dim 256 ^
  --num_layers 5 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp

python tool\train_model_enhanced.py ^
  --data_path dataset_position_segments_synth.npz ^
  --output_dir tool\combined_short_gru_models_enhanced ^
  --model_name short_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 4096 ^
  --hidden_dim 64 ^
  --num_layers 2 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp

python tool\train_model_enhanced.py ^
  --data_path combined_segments.npz ^
  --output_dir tool\new_long_gru_models_enhanced ^
  --model_name long_enhanced_gru_model ^
  --epochs 1 ^
  --batch_size 4096 ^
  --hidden_dim 256 ^
  --num_layers 5 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --loss_lambda_curv 0.02 ^
  --axis_weights 1.0,1.2,1.5
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from datetime import datetime, timedelta
from torch.cuda.amp import autocast
import logging
import json
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def compute_multi_scale_velocity(trajectory, dt=0.1, scales=[1, 2, 3]):
    """
    计算多尺度速度特征
    论文中推荐：不同时间尺度的速度能更好地捕捉轨迹的曲率变化
    
    Args:
        trajectory: (T, 3) 位置序列
        dt: 采样间隔
        scales: 要计算的时间尺度列表
        
    Returns:
        multi_scale_vel: (T, 3*len(scales)) 拼接后的多尺度速度
    """
    multi_scale_vels = []
    
    for scale in scales:
        if len(trajectory) > scale:
            # 计算 scale 倍采样间隔的速度
            vel = (trajectory[scale:] - trajectory[:-scale]) / (scale * dt)
            # 填充到原始长度（用第一个有效速度重复）
            vel_padded = np.vstack([np.tile(vel[0], (scale, 1)), vel])
            multi_scale_vels.append(vel_padded)
    
    if multi_scale_vels:
        return np.concatenate(multi_scale_vels, axis=1)  # (T, 3*len(scales))
    return np.diff(trajectory, axis=0) / dt


def compute_curvature(trajectory, dt=0.1):
    """
    计算轨迹的曲率（高阶几何特征）
    论文中的关键特征：在圆形、8字等复杂轨迹上效果显著
    
    曲率 = ||d²r/dt²|| / ||dr/dt||³（参数方程下）
    
    Args:
        trajectory: (T, 3) 位置序列
        dt: 采样间隔
        
    Returns:
        curvature: (T,) 曲率值
    """
    if len(trajectory) < 2:
        return np.zeros(len(trajectory))
    
    # 一阶导数（速度）
    vel = np.gradient(trajectory, axis=0) / dt
    
    # 二阶导数（加速度）
    acc = np.gradient(vel, axis=0) / dt
    
    # 计算曲率：||v × a|| / ||v||³
    cross_prod = np.cross(vel, acc)
    curvature = np.linalg.norm(cross_prod, axis=1) / (np.linalg.norm(vel, axis=1) ** 3 + 1e-8)
    
    return curvature.reshape(-1, 1)  # (T, 1)


def compute_plane_curvatures(trajectory, dt=0.1):
    """
    ✓ 新增：计算三个平面的单独曲率特征
    XY, YZ, XZ 平面的曲率，帮助模型捕捉每个平面的运动特性
    
    Args:
        trajectory: (T, 3) 位置序列
        dt: 采样间隔
    
    Returns:
        plane_curvs: (T, 3) 三个平面的曲率
    """
    if len(trajectory) < 2:
        return np.zeros((len(trajectory), 3))
    
    curv_list = []
    eps = 1e-8  # 更强的 epsilon 防止 Inf/NaN
    
    # XY 平面曲率：只用 x, y，z 替换为 0
    pos_xy = np.column_stack([trajectory[:, 0], trajectory[:, 1], np.zeros(len(trajectory))])
    vel_xy = np.gradient(pos_xy, axis=0) / dt
    acc_xy = np.gradient(vel_xy, axis=0) / dt
    cross_xy = np.cross(vel_xy, acc_xy)
    vel_norm_xy = np.linalg.norm(vel_xy, axis=1)
    curv_xy = np.linalg.norm(cross_xy, axis=1) / np.maximum(vel_norm_xy ** 3, eps)
    curv_xy = np.nan_to_num(curv_xy, nan=0.0, posinf=1.0, neginf=0.0)  # ← 防止 Inf/NaN
    curv_list.append(curv_xy)
    
    # YZ 平面曲率：只用 y, z，x 替换为 0
    pos_yz = np.column_stack([np.zeros(len(trajectory)), trajectory[:, 1], trajectory[:, 2]])
    vel_yz = np.gradient(pos_yz, axis=0) / dt
    acc_yz = np.gradient(vel_yz, axis=0) / dt
    cross_yz = np.cross(vel_yz, acc_yz)
    vel_norm_yz = np.linalg.norm(vel_yz, axis=1)
    curv_yz = np.linalg.norm(cross_yz, axis=1) / np.maximum(vel_norm_yz ** 3, eps)
    curv_yz = np.nan_to_num(curv_yz, nan=0.0, posinf=1.0, neginf=0.0)
    curv_list.append(curv_yz)
    
    # XZ 平面曲率：只用 x, z，y 替换为 0
    pos_xz = np.column_stack([trajectory[:, 0], np.zeros(len(trajectory)), trajectory[:, 2]])
    vel_xz = np.gradient(pos_xz, axis=0) / dt
    acc_xz = np.gradient(vel_xz, axis=0) / dt
    cross_xz = np.cross(vel_xz, acc_xz)
    vel_norm_xz = np.linalg.norm(vel_xz, axis=1)
    curv_xz = np.linalg.norm(cross_xz, axis=1) / np.maximum(vel_norm_xz ** 3, eps)
    curv_xz = np.nan_to_num(curv_xz, nan=0.0, posinf=1.0, neginf=0.0)
    curv_list.append(curv_xz)
    
    return np.column_stack(curv_list)  # (T, 3)


class EnhancedTrajectoryDataset(Dataset):
    """
    增强版数据集
    特点：
    - 支持增量位移目标（而非绝对坐标）
    - 多尺度速度特征
    - 曲率特征
    - 自动特征拼接
    """
    def __init__(self, position_segs, output_segs, normalize_config=None, 
                 use_delta_target=True, use_multi_scale_vel=True, use_curvature=True,
                 dt=0.1):
        """
        Args:
            position_segs: (3, T, N) 或 (N, T, 3) 位置段
            output_segs: (3, T_out, N) 或 (N, T_out, 3) 输出段
            normalize_config: 归一化参数字典
            use_delta_target: 是否使用增量位移作为目标（推荐 True）
            use_multi_scale_vel: 是否计算多尺度速度特征
            use_curvature: 是否计算曲率特征
            dt: 采样间隔
        """
        # 转换为 (N, T, 3) 格式
        if position_segs.ndim == 3 and position_segs.shape[0] == 3:
            self.positions = np.transpose(position_segs, (2, 1, 0)).astype(np.float32)
            self.output_pos = np.transpose(output_segs, (2, 1, 0)).astype(np.float32)
        else:
            self.positions = position_segs.astype(np.float32)
            self.output_pos = output_segs.astype(np.float32)
        
        self.n_samples = self.positions.shape[0]
        self.normalize_config = normalize_config
        self.use_delta_target = use_delta_target
        self.use_multi_scale_vel = use_multi_scale_vel
        self.use_curvature = use_curvature
        self.dt = dt
        
        # 预处理特征（可选）
        self.input_features = self._prepare_input_features()
        self.output_targets = self._prepare_output_targets()
    
    def _prepare_input_features(self):
        """预处理所有输入特征并缓存"""
        all_features = []
        
        for i in range(self.n_samples):
            pos = self.positions[i]  # (T, 3)
            
            # 基础：原始位置
            features = [pos]
            
            # 多尺度速度
            if self.use_multi_scale_vel:
                multi_vel = compute_multi_scale_velocity(pos, self.dt, scales=[1, 2, 3])
                # 对齐长度（取前 len(pos) 行）
                multi_vel = multi_vel[:len(pos)]
                features.append(multi_vel)
            
            # 曲率
            if self.use_curvature:
                curv = compute_curvature(pos, self.dt)
                features.append(curv)
                # ✓ 新增：平面特化曲率特征（XY, YZ, XZ 平面）
                plane_curvs = compute_plane_curvatures(pos, self.dt)
                features.append(plane_curvs)
            
            # 拼接所有特征
            full_feature = np.concatenate(features, axis=1)  # (T, C) where C = 3 + 9 + 1 + 3 = 16 (if all enabled)
            all_features.append(full_feature)
        
        return all_features
    
    def _prepare_output_targets(self):
        """预处理输出目标"""
        targets = []
        
        for i in range(self.n_samples):
            if self.use_delta_target:
                # 增量位移：相对于最后一个输入位置的偏移
                # 这样做的好处：目标分布更稳定（均值接近0）
                # ✓ 修复：必须减去最后一个输入点来得到真正的增量
                last_input_pos = self.positions[i][-1]  # (3,) - 输入序列最后一点
                output_delta = self.output_pos[i] - last_input_pos  # (T_out, 3)
                targets.append(output_delta)
            else:
                targets.append(self.output_pos[i])
        
        return targets
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        inp_features = self.input_features[idx].astype(np.float32)
        out_target = self.output_targets[idx].astype(np.float32)
        
        if self.normalize_config is not None:
            # ✓ 修复：归一化所有通道，而不仅仅是位置通道
            inp_mean = self.normalize_config['input_mean']
            inp_std = self.normalize_config['input_std']
            inp_mean_all = self.normalize_config.get('input_mean_all', None)
            inp_std_all = self.normalize_config.get('input_std_all', None)
            
            # 位置通道：使用位置统计
            inp_features[:, :3] = (inp_features[:, :3] - inp_mean) / (inp_std + 1e-8)
            
            # 速度、曲率通道：使用全通道统计（防止爆炸/坍缩）
            if inp_mean_all is not None and inp_std_all is not None and len(inp_mean_all) == 16:
                for ch in range(3, 16):
                    inp_features[:, ch] = (inp_features[:, ch] - inp_mean_all[ch]) / (inp_std_all[ch] + 1e-8)
            
            # 归一化输出
            out_mean = self.normalize_config.get('output_mean', inp_mean)
            out_std = self.normalize_config.get('output_std', inp_std)
            out_target = (out_target - out_mean) / (out_std + 1e-8)
        
        return torch.from_numpy(inp_features), torch.from_numpy(out_target)


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码，增强编码器时间位置信息"""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        return x + self.pos[:, : x.size(1), :]


class EncoderSelfAttnBlock(nn.Module):
    """Transformer 风格的自注意力编码器块"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.ln1(x)
        attn_out, _ = self.mha(qkv, qkv, qkv, need_weights=False)
        x = x + self.drop1(attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.drop2(ffn_out)
        return x


class EnhancedGRUModel(nn.Module):
    """
    增强版 GRU 模型
    特点：
    - 支持多通道输入（位置 + 速度 + 曲率）
    - 内部特征融合层
    - 可选注意力机制
    """
    def __init__(self, input_size=16, hidden_dim=128, num_layers=3, 
                 dropout=0.5, output_steps=10, use_attention=False, bidirectional=True,
                 encoder_input_dim=None):
        """
        Args:
            input_size: 输入通道数（位置 3 + 多尺度速度 9 + 曲率 1 + 平面曲率 3 = 16）
            hidden_dim: 单方向隐藏单元数
            num_layers: GRU 层数
            dropout: dropout 概率
            output_steps: 输出步数
            use_attention: 是否使用注意力机制（可选）
            bidirectional: 编码器是否为双向（默认 True）
            encoder_input_dim: encoder_gru 的输入维度（feature_fusion 输出维度）；若为 None，则使用 hidden_dim
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim  # 单方向隐藏单元数
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.use_attention = use_attention

        self.num_directions = 2 if bidirectional else 1
        self.encoder_input_dim = encoder_input_dim if encoder_input_dim is not None else hidden_dim
        self.encoder_hidden_dim = hidden_dim
        self.encoder_output_dim = self.encoder_hidden_dim * self.num_directions
        self.decoder_hidden_dim = self.encoder_output_dim
        
        # 特征融合层：将多通道输入统一投影到 encoder 所需维度
        self.feature_fusion = nn.Linear(input_size, self.encoder_input_dim)
        
        # 编码器 GRU（可双向或单向）
        self.encoder_gru = nn.GRU(
            self.encoder_input_dim,
            self.encoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # 可选注意力层（编码器 refine + 解码 cross-attn）
        if use_attention:
            num_heads = 1
            # 选择可整除的多头数量，保证注意力维度均匀分配
            for candidate in (8, 4, 2, 1):
                if self.encoder_output_dim % candidate == 0:
                    num_heads = candidate
                    break

            self.pos_enc = LearnablePositionalEncoding(max_len=256, d_model=self.encoder_output_dim)
            self.enc_refiner = EncoderSelfAttnBlock(self.encoder_output_dim, num_heads, dropout=min(0.2, dropout))

            self.cross_attn = nn.MultiheadAttention(
                self.encoder_output_dim, num_heads, dropout=min(0.2, dropout), batch_first=True
            )
            self.cross_ln = nn.LayerNorm(self.encoder_output_dim)
        
        # 解码器 GRU
        self.decoder_gru = nn.GRU(
            self.decoder_hidden_dim,
            self.decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 将上一步输出映射到 decoder 所需的 hidden_dim
        self.decoder_input_proj = nn.Linear(3, self.decoder_hidden_dim)

        # 输出层（用于初始 3D guess，可与 plane heads 融合）
        self.fc = nn.Linear(self.decoder_hidden_dim, 3)

        # 三个平面头：分别预测 XY, YZ, XZ 平面中的 2D 增量
        head_dim = max(self.decoder_hidden_dim // 2, 32)
        self.plane_heads = nn.ModuleDict({
            'xy': nn.Sequential(
                nn.LayerNorm(self.decoder_hidden_dim),
                nn.Linear(self.decoder_hidden_dim, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 2),
            ),
            'yz': nn.Sequential(
                nn.LayerNorm(self.decoder_hidden_dim),
                nn.Linear(self.decoder_hidden_dim, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 2),
            ),
            'xz': nn.Sequential(
                nn.LayerNorm(self.decoder_hidden_dim),
                nn.Linear(self.decoder_hidden_dim, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 2),
            ),
        })
        self.plane_gate = nn.Linear(self.decoder_hidden_dim, 3)

    def _merge_bidirectional_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """将 encoder 双向隐藏状态拼接成 decoder 需要的形状."""
        if not self.encoder_gru.bidirectional:
            return h

        batch = h.size(1)
        h = h.view(self.num_layers, self.num_directions, batch, self.encoder_hidden_dim)
        h = h.permute(0, 2, 1, 3).reshape(self.num_layers, batch, self.decoder_hidden_dim)
        return h

    def _compute_plane_predictions(self, hidden_state: torch.Tensor):
        """Compute XY/YZ/XZ plane predictions from the decoder hidden state."""
        return {
            plane: head(hidden_state)
            for plane, head in self.plane_heads.items()
        }

    def _fuse_plane_predictions(self, plane_preds):
        """Fuse plane-specific 2D outputs into a single 3D increment."""
        xy = plane_preds['xy']  # (B, 2)
        yz = plane_preds['yz']
        xz = plane_preds['xz']

        delta_x = 0.5 * (xy[:, 0] + xz[:, 0])
        delta_y = 0.5 * (xy[:, 1] + yz[:, 0])
        delta_z = 0.5 * (yz[:, 1] + xz[:, 1])

        fused = torch.stack([delta_x, delta_y, delta_z], dim=-1)
        return fused
    
    def forward(self, x, return_plane_preds: bool = False):
        """
        Args:
            x: (batch, input_len, input_size)
        Returns:
            output: (batch, output_steps, 3)
        """
        # 特征融合
        x_fused = self.feature_fusion(x)  # (batch, input_len, encoder_hidden_dim)
        
        # 编码
        enc_out, h = self.encoder_gru(x_fused)  # enc_out: (B, T, encoder_output_dim)
        h = self._merge_bidirectional_hidden(h)

        # 可选注意力：在编码器输出上做 Transformer 风格 refine
        if self.use_attention:
            enc_out = self.pos_enc(enc_out)
            enc_out = self.enc_refiner(enc_out)

        # 自回归解码
        predictions = []
        plane_buffers = {plane: [] for plane in self.plane_heads.keys()}
        h_t = h

        if self.use_attention:
            # 使用 cross-attn 获取更稳定的初始上下文
            q0 = enc_out[:, -1:, :]  # (B,1,H)
            ctx0, _ = self.cross_attn(q0, enc_out, enc_out, need_weights=False)
            ctx0 = self.cross_ln(q0 + ctx0).squeeze(1)  # (B,H)
            prev_output = self.fc(ctx0)  # (B,3)
        else:
            last_output = enc_out[:, -1, :]  # (batch, encoder_output_dim)
            prev_output = self.fc(last_output)  # (batch, 3)

        for _ in range(self.output_steps):
            decoder_input = self.decoder_input_proj(prev_output).unsqueeze(1)  # (B,1,H)
            _, h_t = self.decoder_gru(decoder_input, h_t)

            h_last = h_t[-1]  # (B,H)

            if self.use_attention:
                q = h_last.unsqueeze(1)  # (B,1,H)
                ctx, _ = self.cross_attn(q, enc_out, enc_out, need_weights=False)
                ctx = self.cross_ln(q + ctx).squeeze(1)  # (B,H)
                base_output = self.fc(ctx)
                plane_source = ctx
            else:
                base_output = self.fc(h_last)
                plane_source = h_last

            plane_preds_step = self._compute_plane_predictions(plane_source)
            for plane, tensor in plane_preds_step.items():
                plane_buffers[plane].append(tensor.unsqueeze(1))

            plane_fused = self._fuse_plane_predictions(plane_preds_step)
            gate = torch.sigmoid(self.plane_gate(h_last))
            y_t = plane_fused * gate + base_output * (1.0 - gate)

            predictions.append(y_t.unsqueeze(1))
            prev_output = y_t.detach()
        
        output = torch.cat(predictions, dim=1)  # (batch, output_steps, 3)

        if return_plane_preds:
            plane_outputs = {
                plane: torch.cat(buffers, dim=1)
                for plane, buffers in plane_buffers.items()
            }
            return output, plane_outputs

        return output


def model_forward_with_adaptive_tf(model, input_seq, target_seq, teacher_forcing_ratio=0.5,
                                   return_plane_preds=False):
    """
    改进的 Teacher Forcing 前向传播
    - 自适应调整：可选按置信度调整 TF 概率
    """
    device = input_seq.device

    # 编码
    x_fused = model.feature_fusion(input_seq)
    enc_out, h = model.encoder_gru(x_fused)
    h = model._merge_bidirectional_hidden(h)

    # 与 forward 对齐的注意力逻辑
    if getattr(model, "use_attention", False):
        enc_out = model.pos_enc(enc_out)
        enc_out = model.enc_refiner(enc_out)

    # Ensure target_seq dtype/device matches model parameters to avoid AMP dtype mismatches
    model_dtype = next(model.parameters()).dtype
    if target_seq is not None:
        target_seq = target_seq.to(device=device, dtype=model_dtype)

    # 自回归解码（带自适应 TF）——逐步解码并在每步使用 teacher forcing 决策
    predictions = []
    plane_buffers = {plane: [] for plane in model.plane_heads.keys()}
    h_t = h

    if getattr(model, "use_attention", False):
        q0 = enc_out[:, -1:, :]  # (B,1,H)
        ctx0, _ = model.cross_attn(q0, enc_out, enc_out, need_weights=False)
        ctx0 = model.cross_ln(q0 + ctx0).squeeze(1)  # (B,H)
        prev_output = model.fc(ctx0).to(dtype=model_dtype, device=device)
    else:
        last_output = enc_out[:, -1, :]  # (batch, encoder_output_dim)
        prev_output = model.fc(last_output).to(dtype=model_dtype, device=device)  # (batch, 3)

    for t in range(model.output_steps):
        decoder_input = model.decoder_input_proj(prev_output).unsqueeze(1)  # (B,1,decoder_hidden_dim)
        _, h_t = model.decoder_gru(decoder_input, h_t)

        h_last = h_t[-1]  # (batch, decoder_hidden_dim)

        if getattr(model, "use_attention", False):
            q = h_last.unsqueeze(1)  # (B,1,H)
            ctx, _ = model.cross_attn(q, enc_out, enc_out, need_weights=False)
            ctx = model.cross_ln(q + ctx).squeeze(1)  # (B,H)
            base_output = model.fc(ctx)
            plane_source = ctx
        else:
            base_output = model.fc(h_last)
            plane_source = h_last

        plane_preds_step = model._compute_plane_predictions(plane_source)
        for plane, tensor in plane_preds_step.items():
            plane_buffers[plane].append(tensor.unsqueeze(1))

        plane_fused = model._fuse_plane_predictions(plane_preds_step)
        gate = torch.sigmoid(model.plane_gate(h_last))
        y_t = plane_fused * gate + base_output * (1.0 - gate)

        y_t = y_t.to(dtype=model_dtype, device=device)

        predictions.append(y_t.unsqueeze(1))

        # 计算自适应 teacher forcing 比率（随步数线性衰减）
        adaptive_ratio = teacher_forcing_ratio * (1.0 - float(t) / max(1, model.output_steps))

        use_tf = False
        if adaptive_ratio > 0 and target_seq is not None:
            if torch.rand(1, device=device).item() < adaptive_ratio and t < target_seq.size(1):
                use_tf = True

        if use_tf:
            # 取 target 的前三个通道（位置增量），并确保 dtype/device
            prev_output = target_seq[:, t, :3].to(device=device, dtype=model_dtype)
        else:
            prev_output = y_t.detach()

    output = torch.cat(predictions, dim=1)
    if return_plane_preds:
        plane_outputs = {
            plane: torch.cat(buffers, dim=1)
            for plane, buffers in plane_buffers.items()
        }
        return output, plane_outputs
    return output


class MultiObjectiveLoss(nn.Module):
    """
    多目标损失函数
    包括：
    1. 位置预测损失（MSE）
    2. 加速度平滑性损失（约束二阶导数）
    3. 速度连续性损失（可选）
    
    ✓ 新增：每轴加权损失，可单独调整 X/Y/Z 轴的学习重点
    """
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, axis_weights=None):
        """
        Args:
            alpha: 位置损失权重
            beta: 加速度损失权重
            gamma: 速度连续性损失权重
            axis_weights: [wx, wy, wz] 三个轴的损失权重（默认 [1.0, 1.5, 2.0] 强化 Y/Z 轴）
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # 默认权重：强化 Y 和 Z 轴的学习（XY/YZ/XZ 平面都有 Y/Z）
        if axis_weights is None:
            axis_weights = [1.0, 1.5, 2.0]  # X, Y, Z
        self.register_buffer('axis_weights', torch.tensor(axis_weights, dtype=torch.float32))
    
    def forward(self, pred, target, plane_preds=None):
        """
        Args:
            pred: (batch, output_steps, 3) 预测位置增量
            target: (batch, output_steps, 3) 真实位置增量
            plane_preds: 可选，平面预测（为兼容性保留）
        Returns:
            total_loss: 标量
        """
        diff = (pred - target) ** 2  # (B, T, 3)
        axis_w = self.axis_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, 3)
        weighted_pos_loss = torch.mean(diff * axis_w)

        # 加速度平滑性损失（也按轴加权）
        if pred.shape[1] >= 3:
            pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
            target_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
            acc_diff = (pred_acc - target_acc) ** 2
            weighted_acc_loss = torch.mean(acc_diff * axis_w)
        else:
            weighted_acc_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 速度连续性损失（按轴加权）
        if pred.shape[1] >= 2:
            pred_vel = pred[:, 1:] - pred[:, :-1]
            target_vel = target[:, 1:] - target[:, :-1]
            vel_diff = (pred_vel - target_vel) ** 2
            weighted_vel_loss = torch.mean(vel_diff * axis_w)
        else:
            weighted_vel_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 总损失
        total_loss = (self.alpha * weighted_pos_loss +
                      self.beta * weighted_acc_loss +
                      self.gamma * weighted_vel_loss)

        return total_loss


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0,
                    scaler=None, use_amp=False, teacher_forcing_ratio=0.5):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    count = 0
    
    for inp, out in loader:
        inp = inp.to(device, non_blocking=True)
        out = out.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                pred, plane_preds = model_forward_with_adaptive_tf(
                    model, inp, out, teacher_forcing_ratio, return_plane_preds=True
                )
                loss = criterion(pred, out, plane_preds=plane_preds)
            
            # ✓ NaN 检测
            if torch.isnan(loss):
                logger.warning(f"检测到 NaN loss! pred range: [{pred.min():.6f}, {pred.max():.6f}], "
                             f"out range: [{out.min():.6f}, {out.max():.6f}]")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, plane_preds = model_forward_with_adaptive_tf(
                model, inp, out, teacher_forcing_ratio, return_plane_preds=True
            )
            loss = criterion(pred, out, plane_preds=plane_preds)
            
            # ✓ NaN 检测
            if torch.isnan(loss):
                logger.warning(f"检测到 NaN loss! pred range: [{pred.min():.6f}, {pred.max():.6f}], "
                             f"out range: [{out.min():.6f}, {out.max():.6f}]")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item() * inp.size(0)
        count += inp.size(0)
    
    return total_loss / count if count > 0 else 0.0


def eval_one_epoch(model, loader, criterion, device):
    """评估一个 epoch"""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for inp, out in loader:
            inp = inp.to(device, non_blocking=True)
            out = out.to(device, non_blocking=True)
            
            pred, plane_preds = model(inp, return_plane_preds=True)
            loss = criterion(pred, out, plane_preds=plane_preds)
            
            total_loss += loss.item() * inp.size(0)
            count += inp.size(0)
    
    return total_loss / count if count > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train enhanced GRU trajectory model')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--stats_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--model_name', type=str, default='enhanced_gru_model')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.6)
    parser.add_argument('--tf_decay', type=float, default=0.005)
    parser.add_argument('--loss_alpha', type=float, default=0.7, help='位置损失权重')
    parser.add_argument('--loss_beta', type=float, default=0.2, help='加速度损失权重')
    parser.add_argument('--loss_gamma', type=float, default=0.1, help='速度损失权重')
    parser.add_argument('--loss_lambda_curv', type=float, default=0.0, help='曲率匹配损失权重')
    parser.add_argument('--loss_lambda_plane_consistency', type=float, default=0.0,
                       help='平面头一致性损失权重')
    parser.add_argument('--loss_lambda_plane_supervision', type=float, default=0.2,
                       help='平面头直接监督损失权重 (XY/YZ/XZ)')
    parser.add_argument('--axis_weights', type=str, default=None, 
                       help='X,Y,Z 轴的损失权重 (默认 1.0,1.5,2.0 强化 Y/Z) 格式: "1.0,1.5,2.0"')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers，Windows 下建议 0 以避免多进程拷贝大数组导致 MemoryError')
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_attention', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    logger.info(f"\n加载数据: {args.data_path}")
    data = np.load(args.data_path, allow_pickle=True)
    
    input_segments = data['input_segments']
    output_segments = data['output_segments']
    
    logger.info(f"输入形状: {input_segments.shape}")
    logger.info(f"输出形状: {output_segments.shape}")
    
    # 加载或计算统计量
    input_mean_all = None
    input_std_all = None
    if args.stats_path and os.path.exists(args.stats_path):
        stats = np.load(args.stats_path)
        input_mean = stats['input_mean']
        input_std = stats['input_std']
        output_mean = stats.get('output_mean', input_mean)
        output_std = stats.get('output_std', input_std)
        input_mean_all = stats.get('input_mean_all', None)
        input_std_all = stats.get('input_std_all', None)
    else:
        # 把 segments 转换为 (N, T, 3) 格式以便统计
        if input_segments.ndim == 3 and input_segments.shape[0] == 3:
            inp_arr = np.transpose(input_segments, (2, 1, 0))  # (N, Tin, 3)
            out_arr = np.transpose(output_segments, (2, 1, 0))  # (N, Tout, 3)
        else:
            inp_arr = input_segments
            out_arr = output_segments

        # 输入位置的均值/方差（按通道）
        input_mean = np.mean(inp_arr.reshape(-1, 3), axis=0)
        input_std = np.std(inp_arr.reshape(-1, 3), axis=0)

        # ✓ 新增：为了获得完整的16维统计量，需要先计算所有样本的特征
        # 这会在数据集创建时再次计算，但我们需要提前获得统计量
        logger.info("计算16维特征的统计量（这可能耗时较长）...")
        all_features_16d = []
        for i in range(inp_arr.shape[0]):
            pos = inp_arr[i]
            feats = [pos]
            vel = compute_multi_scale_velocity(pos, dt=0.1, scales=[1, 2, 3])
            vel = vel[: len(pos)]
            feats.append(vel)
            curv = compute_curvature(pos, dt=0.1)
            feats.append(curv)
            plane_curvs = compute_plane_curvatures(pos, dt=0.1)
            feats.append(plane_curvs)
            all_features_16d.append(np.concatenate(feats, axis=1))

        # 合并所有特征：(sum(Tin), 16)
        all_features_concat = np.vstack(all_features_16d)
        input_mean_all = np.mean(all_features_concat, axis=0)
        input_std_all = np.std(all_features_concat, axis=0)

        # ✓ 修复：若训练目标是 delta（相对位移），则基于增量统计 output_mean/output_std
        # 计算每个样本的输出增量 = out - last_input
        last_inputs = inp_arr[:, -1, :]  # (N, 3)
        # 广播计算 deltas： out_arr shape (N, Tout, 3)
        deltas = out_arr - last_inputs[:, None, :]  # (N, Tout, 3)
        output_mean = np.mean(deltas.reshape(-1, 3), axis=0)
        output_std = np.std(deltas.reshape(-1, 3), axis=0)

        # ✓ 防止标准差为 0 导致 NaN
        output_std = np.where(output_std < 1e-8, 1.0, output_std)
        input_std = np.where(input_std < 1e-8, 1.0, input_std)
        input_std_all = np.where(input_std_all < 1e-8, 1.0, input_std_all)

        logger.info(f"计算统计量:")
        logger.info(f"  input_mean (position): {input_mean}, input_std: {input_std}")
        logger.info(f"  input_mean_all (16D): {input_mean_all}")
        logger.info(f"  input_std_all (16D): {input_std_all}")
        logger.info(f"  output_mean (delta): {output_mean}, output_std (delta): {output_std}")

        # ✓ 保存统计量以便推理复用，保持与训练一致
        stats_save_path = os.path.join(args.output_dir, f"{args.model_name}_norm_stats.npz")
        np.savez(
            stats_save_path,
            input_mean=input_mean,
            input_std=input_std,
            input_mean_all=input_mean_all,
            input_std_all=input_std_all,
            output_mean=output_mean,
            output_std=output_std,
        )
        logger.info(f"  已保存统计量到: {stats_save_path}")
    if input_segments.ndim == 3 and input_segments.shape[0] == 3:
        num_samples = input_segments.shape[2]
    else:
        num_samples = input_segments.shape[0]
    
    logger.info(f"总样本数: {num_samples}")
    
    # 分割 train/val
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    num_val = int(num_samples * args.val_split)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    # 准备数据集
    if input_segments.ndim == 3 and input_segments.shape[0] == 3:
        train_inp = input_segments[:, :, train_indices]
        train_out = output_segments[:, :, train_indices]
        val_inp = input_segments[:, :, val_indices]
        val_out = output_segments[:, :, val_indices]
    else:
        train_inp = input_segments[train_indices]
        train_out = output_segments[train_indices]
        val_inp = input_segments[val_indices]
        val_out = output_segments[val_indices]
    
    normalize_config = {
        'input_mean': input_mean,
        'input_std': input_std,
        'input_mean_all': input_mean_all if 'input_mean_all' in locals() else None,
        'input_std_all': input_std_all if 'input_std_all' in locals() else None,
        'output_mean': output_mean,
        'output_std': output_std,
    }
    
    train_dataset = EnhancedTrajectoryDataset(
        train_inp, train_out, normalize_config,
        use_delta_target=True, use_multi_scale_vel=True, use_curvature=True
    )
    val_dataset = EnhancedTrajectoryDataset(
        val_inp, val_out, normalize_config,
        use_delta_target=True, use_multi_scale_vel=True, use_curvature=True
    )
    
    persistent_flag = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=persistent_flag
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=persistent_flag
    )
    
    # 创建模型
    input_size = 16  # 3 (pos) + 9 (multi-scale vel) + 1 (curvature) + 3 (plane curvatures)
    model = EnhancedGRUModel(
        input_size=input_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_steps=10,
        use_attention=args.use_attention
    )
    model.to(device)
    
    logger.info(f"\n模型配置:")
    logger.info(f"  输入通道: {input_size}")
    logger.info(f"  隐藏维度: {args.hidden_dim}")
    logger.info(f"  层数: {args.num_layers}")
    logger.info(f"  参数数: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  使用注意力: {args.use_attention}")
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 解析每轴损失权重（默认强化 Y/Z 轴）
    axis_weights = None
    if args.axis_weights:
        try:
            axis_weights = [float(x.strip()) for x in args.axis_weights.split(',')]
            if len(axis_weights) != 3:
                logger.warning(f"axis_weights 必须有 3 个值, 使用默认值")
                axis_weights = None
        except ValueError:
            logger.warning(f"无法解析 axis_weights: {args.axis_weights}, 使用默认值")
            axis_weights = None
    
    # 创建多目标损失函数
    criterion = MultiObjectiveLoss(
        args.loss_alpha, args.loss_beta, args.loss_gamma,
        axis_weights=axis_weights
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    # ✓ 新增：完整的训练历史记录
    training_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'teacher_forcing_ratio': [],
        'epoch_time': []
    }
    
    # ✓ 新增：保存训练配置，便于复现
    config_dict = {
        'timestamp': datetime.now().isoformat(),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'patience': args.patience,
        'seed': args.seed,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'val_split': args.val_split,
        'teacher_forcing_ratio': args.teacher_forcing_ratio,
        'tf_decay': args.tf_decay,
        'loss_alpha': args.loss_alpha,
        'loss_beta': args.loss_beta,
        'loss_gamma': args.loss_gamma,
        'axis_weights': args.axis_weights if args.axis_weights else "default(1.0,1.5,2.0)",
        'use_amp': args.use_amp,
        'use_attention': args.use_attention,
        'data_path': args.data_path,
        'input_mean': input_mean.tolist(),
        'input_std': input_std.tolist(),
        'output_mean': output_mean.tolist(),
        'output_std': output_std.tolist(),
        'num_train_samples': len(train_dataset),
        'num_val_samples': len(val_dataset),
        'total_params': sum(p.numel() for p in model.parameters()),
    }
    
    # 保存配置文件
    config_path = os.path.join(args.output_dir, f'{args.model_name}_training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 已保存训练配置到: {config_path}")
    
    # ✓ 新增：文件句柄用于日志记录
    log_file = os.path.join(args.output_dir, f'{args.model_name}_training.log')
    csv_file = os.path.join(args.output_dir, f'{args.model_name}_history.csv')
    
    logger.info(f"\n开始训练 ({args.epochs} epochs)")
    logger.info(f"  训练日志: {log_file}")
    logger.info(f"  历史记录: {csv_file}")
    print("=" * 100)
    print(f"{'Epoch':<8} {'Train Loss':<16} {'Val Loss':<16} {'LR':<12} {'TF Ratio':<12} {'ETA':<12} {'Status':<15}")
    print("=" * 100)
    
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # 自适应 TF 衰减
        tf_current = max(0.0, args.teacher_forcing_ratio - args.tf_decay * (epoch - 1))
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                    args.grad_clip, scaler, args.use_amp, tf_current)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_time = np.mean(epoch_times[-10:])
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        eta_seconds = (args.epochs - epoch) * avg_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # ✓ 新增：记录到历史
        training_history['epoch'].append(epoch)
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate'].append(current_lr)
        training_history['teacher_forcing_ratio'].append(tf_current)
        training_history['epoch_time'].append(epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            status = "✓ BEST"
            
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.model_name}_best_model.pth'))
            np.savez(os.path.join(args.output_dir, f'{args.model_name}_norm_stats.npz'),
                    input_mean=input_mean, input_std=input_std,
                    input_mean_all=input_mean_all, input_std_all=input_std_all,
                    output_mean=output_mean, output_std=output_std)
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{args.patience}"
        
        print(f"{epoch:<8} {train_loss:<16.6f} {val_loss:<16.6f} {current_lr:<12.2e} {tf_current:<12.4f} {eta_str:<12} {status:<15}")
        
        # ✓ 新增：实时保存历史到CSV（每个epoch都更新，便于中途查看）
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate', 'Teacher Forcing Ratio', 'Epoch Time (s)'])
            for i in range(len(training_history['epoch'])):
                writer.writerow([
                    training_history['epoch'][i],
                    f"{training_history['train_loss'][i]:.6f}",
                    f"{training_history['val_loss'][i]:.6f}",
                    f"{training_history['learning_rate'][i]:.6e}",
                    f"{training_history['teacher_forcing_ratio'][i]:.4f}",
                    f"{training_history['epoch_time'][i]:.2f}"
                ])
        
        if patience_counter >= args.patience:
            logger.info(f"早停! (patience={args.patience})")
            break
    
    total_time = time.time() - start_time
    logger.info(f"\n✓ 训练完成!")
    logger.info(f"  总耗时: {str(timedelta(seconds=int(total_time)))}")
    logger.info(f"  最优验证损失: {best_val_loss:.6f}")
    logger.info(f"  最优模型: {os.path.join(args.output_dir, f'{args.model_name}_best_model.pth')}")
    logger.info(f"  训练历史: {csv_file}")
    
    # ✓ 新增：保存最终统计信息
    final_stats = {
        'best_val_loss': float(best_val_loss),
        'total_epochs_trained': epoch,
        'total_time_seconds': total_time,
        'avg_epoch_time': np.mean(epoch_times),
        'final_train_loss': training_history['train_loss'][-1],
        'final_val_loss': training_history['val_loss'][-1],
    }
    final_stats_path = os.path.join(args.output_dir, f'{args.model_name}_final_stats.json')
    with open(final_stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
    logger.info(f"  最终统计: {final_stats_path}")


if __name__ == '__main__':
    main()
