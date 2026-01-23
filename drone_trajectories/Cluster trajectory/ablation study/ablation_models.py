#!/usr/bin/env python3
"""
消融实验模型定义
尽量复用经过验证的v版本模型类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 导入经过验证的v版本模型类
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_swarm_model_v3_with_gnn import (
    DynamicsAwareSwarmGRUModel_with_GNN,
    MultiHeadGraphAttention,
    build_adjacency_from_positions
)
from train_swarm_model_v2_dynamics_aware import (
    DynamicsAwareSwarmGRUModel,
    DynamicsAwareLoss,
    compute_acceleration_decomposition
)


# ====================================================================
# 实验1：基线模型（需要新实现，因为v版本都使用BiGRU）
# ====================================================================

class BaselineGRUModel(nn.Module):
    """
    基线模型：单向GRU，无BiGRU，无Cross Attention
    用于实验1：× × × (16D)
    
    注意：这是唯一需要新实现的模型，因为v版本都使用BiGRU
    """
    
    def __init__(self, input_size=16, hidden_size=128, num_layers=2,
                 output_size=3, dropout=0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # 特征融合层
        self.feature_fusion = nn.Linear(input_size, hidden_size)
        
        # 单向GRU编码器（无BiGRU）
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # 单向GRU
        )
        
        # 解码器（无Cross Attention）
        self.decoder_input_proj = nn.Linear(output_size, hidden_size)
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc_position = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, x_orig, y=None, y_velocity=None, y_accel=None,
                teacher_forcing_ratio=0.5):
        """
        Args:
            x: 特征 (batch*agents, seq_in, input_size)
            x_orig: 原始位置 (batch*agents, seq_in, 3)
            y: 位置增量目标
            y_velocity: 速度目标（不使用）
            y_accel: 加速度目标（不使用）
            teacher_forcing_ratio: TF比例
        """
        # 处理4D输入
        batch_size_orig = None
        if x.dim() == 4:
            batch_size_orig, seq_in, num_agents_orig, feat_dim = x.shape
            x_reshaped = x.reshape(batch_size_orig * num_agents_orig, seq_in, feat_dim)
            x_orig_reshaped = x_orig.reshape(batch_size_orig * num_agents_orig, seq_in, 3)
            batch_size = batch_size_orig * num_agents_orig
            if y is not None:
                y_reshaped = y.reshape(batch_size, -1, 3)
            else:
                y_reshaped = None
        else:
            x_reshaped = x
            x_orig_reshaped = x_orig
            y_reshaped = y
            batch_size = x.shape[0]
        
        # 特征融合和编码
        x_fused = self.dropout(torch.relu(self.feature_fusion(x_reshaped)))
        enc_out, h = self.encoder(x_fused)  # h: (num_layers, batch, hidden_size)
        
        # 解码（无Cross Attention，直接使用最后隐藏状态）
        seq_out = y_reshaped.shape[1] if y_reshaped is not None else 10
        
        predictions_position = []
        h_t = h
        prev_output = torch.zeros(batch_size, self.output_size, device=x.device)
        
        for t in range(seq_out):
            decoder_input = self.decoder_input_proj(prev_output).unsqueeze(1)
            decoder_out, h_t = self.decoder(decoder_input, h_t)
            decoder_out = decoder_out.squeeze(1)
            
            pred_pos = self.fc_position(decoder_out)
            predictions_position.append(pred_pos.unsqueeze(1))
            
            if y_reshaped is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_output = y_reshaped[:, t, :]
            else:
                prev_output = pred_pos.detach()
        
        output_position = torch.cat(predictions_position, dim=1)
        
        # 返回兼容格式（速度、加速度设为None）
        output_velocity = torch.zeros_like(output_position)
        output_accel = torch.zeros(batch_size, seq_out, 2, device=x.device)
        
        # 重塑回4D
        if batch_size_orig is not None:
            output_position = output_position.reshape(batch_size_orig, seq_out, num_agents_orig, self.output_size)
            output_velocity = output_velocity.reshape(batch_size_orig, seq_out, num_agents_orig, 3)
            output_accel = output_accel.reshape(batch_size_orig, seq_out, num_agents_orig, 2)
        
        return output_position, output_velocity, output_accel


# ====================================================================
# 实验2：特征增强 + BiGRU + Cross Attention（直接使用v2模型）
# ====================================================================

# 实验2直接使用DynamicsAwareSwarmGRUModel，只需设置input_size=32
# 不需要创建新类，在训练脚本中直接使用即可


# ====================================================================
# 实验3：GAT + BiGRU + Cross Attention（直接使用v3模型）
# ====================================================================

# 实验3直接使用DynamicsAwareSwarmGRUModel_with_GNN，只需设置input_size=16
# 不需要创建新类，在训练脚本中直接使用即可


# ====================================================================
# 实验4：GAT + 特征增强（无BiGRU+Cross Attention）
# ====================================================================

class GNNFeatureModel(DynamicsAwareSwarmGRUModel_with_GNN):
    """
    GAT + 特征增强（32D，无BiGRU+Cross Attention）
    用于实验4：√ √ × (32D)
    
    继承v3模型，但禁用BiGRU和Cross Attention
    通过覆盖encoder和forward方法实现
    """
    
    def __init__(self, input_size=32, hidden_size=128, num_layers=2,
                 output_size=3, dropout=0.3, use_attention=False,
                 gnn_hidden=64, num_gnn_heads=4, edge_threshold=5.0,
                 fusion_mode='concat'):
        # 先调用父类初始化（会创建GNN等组件）
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            use_attention=False,  # 禁用Cross Attention
            gnn_hidden=gnn_hidden,
            num_gnn_heads=num_gnn_heads,
            edge_threshold=edge_threshold,
            fusion_mode=fusion_mode
        )
        
        # 覆盖encoder为单向GRU（无BiGRU）
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # 单向GRU
        )
        
        # 覆盖解码器为单向GRU（无Cross Attention）
        self.decoder_input_proj = nn.Linear(output_size, hidden_size)
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 覆盖输出层（使用hidden_size而非hidden_size*2）
        self.fc_position = nn.Linear(hidden_size, output_size)
        self.fc_velocity_dir = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)
        )
        self.fc_velocity_mag = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        self.fc_accel_tangent = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        self.fc_accel_normal = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x, x_orig, y=None, y_velocity=None, y_accel=None,
                teacher_forcing_ratio=0.5):
        """覆盖forward方法，使用单向GRU，无Cross Attention"""
        # 处理4D输入
        batch_size_orig = None
        if x.dim() == 4:
            batch_size_orig, seq_in, num_agents_orig, feat_dim = x.shape
            x_reshaped = x.reshape(batch_size_orig * num_agents_orig, seq_in, feat_dim)
            x_orig_reshaped = x_orig.reshape(batch_size_orig * num_agents_orig, seq_in, 3)
            batch_size = batch_size_orig * num_agents_orig
            if y is not None:
                y_reshaped = y.reshape(batch_size, -1, 3)
            else:
                y_reshaped = None
            if y_velocity is not None:
                y_velocity_reshaped = y_velocity.reshape(batch_size, -1, 3)
            else:
                y_velocity_reshaped = None
            if y_accel is not None:
                y_accel_reshaped = y_accel.reshape(batch_size, -1, 2)
            else:
                y_accel_reshaped = None
        else:
            x_reshaped = x
            x_orig_reshaped = x_orig
            y_reshaped = y
            y_velocity_reshaped = y_velocity
            y_accel_reshaped = y_accel
            batch_size = x.shape[0]
        
        # 应用GNN（复用父类方法）
        gnn_features = self._apply_gnn_to_sequence(x_reshaped, x_orig_reshaped)
        
        # 特征融合（复用父类逻辑）
        if self.fusion_mode == 'concat':
            fused = torch.cat([x_reshaped, gnn_features], dim=-1)
            x_fused = self.fusion_fc(fused)
        elif self.fusion_mode == 'gate':
            fused = torch.cat([x_reshaped, gnn_features], dim=-1)
            gate_logits = self.gate_fc(fused)
            gate = torch.sigmoid(gate_logits.mean(dim=-1, keepdim=True))
            gnn_padded = F.pad(gnn_features, (0, self.input_size - self.gnn_hidden))
            fused_gated = gate * x_reshaped + (1 - gate) * gnn_padded
            x_fused = self.fusion_fc(fused_gated)
        elif self.fusion_mode == 'add':
            gnn_proj = self.gnn_projection(gnn_features)
            x_fused = self.fusion_fc(x_reshaped + gnn_proj)
        
        # 单向GRU编码（无BiGRU）
        x_fused = self.dropout(torch.relu(self.feature_fusion_v2(x_fused)))
        enc_out, h = self.encoder(x_fused)  # h: (num_layers, batch, hidden_size)
        
        # 无Cross Attention，直接使用最后隐藏状态
        
        # 解码（无Cross Attention）
        seq_out = y_reshaped.shape[1] if y_reshaped is not None else 10
        
        predictions_position = []
        predictions_velocity = []
        predictions_accel = []
        
        h_t = h
        prev_output = torch.zeros(batch_size, self.output_size, device=x.device)
        
        for t in range(seq_out):
            decoder_input = self.decoder_input_proj(prev_output).unsqueeze(1)
            decoder_out, h_t = self.decoder(decoder_input, h_t)
            decoder_out = decoder_out.squeeze(1)
            
            # 无Cross Attention，直接使用解码器输出
            state = decoder_out
            
            pred_pos = self.fc_position(state)
            pred_vel_dir = self.fc_velocity_dir(state)
            pred_vel_mag = torch.relu(self.fc_velocity_mag(state))
            pred_vel = pred_vel_dir * pred_vel_mag
            pred_accel_tan = self.fc_accel_tangent(state)
            pred_accel_nor = torch.relu(self.fc_accel_normal(state))
            pred_accel = torch.cat([pred_accel_tan, pred_accel_nor], dim=1)
            
            predictions_position.append(pred_pos.unsqueeze(1))
            predictions_velocity.append(pred_vel.unsqueeze(1))
            predictions_accel.append(pred_accel.unsqueeze(1))
            
            adaptive_ratio = teacher_forcing_ratio * (1.0 - float(t) / max(1, seq_out))
            if y_reshaped is not None and torch.rand(1).item() < adaptive_ratio:
                prev_output = y_reshaped[:, t, :]
            else:
                prev_output = pred_pos.detach()
        
        output_position = torch.cat(predictions_position, dim=1)
        output_velocity = torch.cat(predictions_velocity, dim=1)
        output_accel = torch.cat(predictions_accel, dim=1)
        
        # 重塑回4D
        if batch_size_orig is not None:
            output_position = output_position.reshape(batch_size_orig, seq_out, num_agents_orig, self.output_size)
            output_velocity = output_velocity.reshape(batch_size_orig, seq_out, num_agents_orig, 3)
            output_accel = output_accel.reshape(batch_size_orig, seq_out, num_agents_orig, 2)
        
        return output_position, output_velocity, output_accel


# ====================================================================
# 实验5：完整模型（直接使用v3模型，input_size=32）
# ====================================================================

# 实验5直接使用DynamicsAwareSwarmGRUModel_with_GNN，只需设置input_size=32
# 不需要创建新类，在训练脚本中直接使用即可（就像v4那样）
