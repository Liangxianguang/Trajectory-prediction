#!/usr/bin/env python3
"""
集群轨迹模型 v3 - 动力学感知 + GNN 增强版本
================================================

相比 v2 的改进：
✅ 加入 GAT（图注意力网络）显式建模代理间交互
✅ 基于位置距离动态构建邻接矩阵
✅ 保留原有 24D 特征、BiGRU、多任务损失框架
✅ 完全兼容现有训练脚本（可开关 --use_gnn）

架构：
    输入位置 → [序列级 GNN + 24D 特征融合] → BiGRU 编码 → 多分支解码

特性：
    1. 距离阈值邻接矩阵（物理意义清晰）
    2. 多头 GAT（自适应权重）
    3. 融合策略灵活（拼接/加权/Gate）
    4. 支持可变代理数（通过重塑保证兼容性）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 导入 v2 的基础组件
from train_swarm_model_v2_dynamics_aware import (
    compute_features_enhanced_24d,
    compute_velocity_direction,
    compute_acceleration_decomposition,
    DynamicsAwareLoss
)


# ====================================================================
# GNN 模块 - 基于 PyTorch 手工实现（不依赖 torch_geometric）
# ====================================================================

class GraphAttentionHead(nn.Module):
    """
    单头图注意力层
    
    输入：
        x: (num_nodes, in_channels)
        adjacency: (num_nodes, num_nodes) 稀疏/稠密矩阵
    
    输出：
        out: (num_nodes, out_channels)
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.3, concat=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat
        
        # 线性投影层
        self.fc = nn.Linear(in_channels, out_channels)
        
        # 注意力计算
        self.attn_fc = nn.Linear(2 * out_channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x, adjacency):
        """
        Args:
            x: (num_nodes, in_channels)
            adjacency: (num_nodes, num_nodes)，值为 0/1
        
        Returns:
            out: (num_nodes, out_channels)
        """
        # 线性投影：(num_nodes, out_channels)
        h = self.fc(x)
        
        # 计算注意力系数
        # self.attn_fc 期望输入 (num_edges, 2*out_channels)
        # 构造所有可能边对的特征拼接
        num_nodes = h.shape[0]
        
        # (num_nodes, 1, out_channels) 与 (1, num_nodes, out_channels) 扩展
        h_i = h.unsqueeze(1).expand(-1, num_nodes, -1)  # (num_nodes, num_nodes, out_channels)
        h_j = h.unsqueeze(0).expand(num_nodes, -1, -1)  # (num_nodes, num_nodes, out_channels)
        
        # 拼接：(num_nodes, num_nodes, 2*out_channels)
        h_pair = torch.cat([h_i, h_j], dim=-1)
        
        # 计算注意力分数：(num_nodes, num_nodes, 1) -> (num_nodes, num_nodes)
        attn_logits = self.attn_fc(h_pair).squeeze(-1)
        attn_logits = self.leaky_relu(attn_logits)
        
        # 应用 mask：仅在有边的地方计算注意力
        # adjacency: (num_nodes, num_nodes) 的 0/1 矩阵
        attn_logits = attn_logits.masked_fill((adjacency == 0).unsqueeze(-1).squeeze(-1), float('-inf'))
        
        # softmax 归一化
        attn_weights = torch.softmax(attn_logits, dim=1)  # (num_nodes, num_nodes)
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        
        attn_weights = self.dropout(attn_weights)
        
        # 聚合邻域特征：(num_nodes, num_nodes) x (num_nodes, out_channels) 
        #            = (num_nodes, out_channels)
        out = torch.bmm(
            attn_weights.unsqueeze(0),
            h.unsqueeze(0)
        ).squeeze(0)
        # 更正：使用矩阵乘法
        out = torch.matmul(attn_weights, h)
        
        return out, attn_weights


class MultiHeadGraphAttention(nn.Module):
    """
    多头图注意力层（融合多个注意力头的输出）
    
    Args:
        in_channels: 输入特征维度
        out_channels: 每个头的输出维度
        num_heads: 注意力头数
        dropout: Dropout 比例
        concat: 是否拼接多头（True）还是平均（False）
    """
    
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.3, concat=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        
        self.heads = nn.ModuleList([
            GraphAttentionHead(in_channels, out_channels, dropout, concat=True)
            for _ in range(num_heads)
        ])
        
        if concat:
            self.fc_out = nn.Linear(num_heads * out_channels, out_channels)
        else:
            self.fc_out = None
    
    def forward(self, x, adjacency):
        """
        Args:
            x: (num_nodes, in_channels)
            adjacency: (num_nodes, num_nodes)
        
        Returns:
            out: (num_nodes, out_channels)
        """
        head_outputs = []
        
        for head in self.heads:
            out, _ = head(x, adjacency)
            head_outputs.append(out)
        
        # 拼接所有头的输出
        out = torch.cat(head_outputs, dim=-1)  # (num_nodes, num_heads*out_channels)
        
        # 可选：线性投影回固定维度
        if self.fc_out is not None:
            out = self.fc_out(out)
        
        return out


def build_adjacency_from_positions(positions, threshold=5.0, add_self_loops=True):
    """
    从位置信息构建邻接矩阵（距离阈值法）
    
    Args:
        positions: (batch, agents, 3) 或 (agents, 3)
        threshold: 邻接距离阈值（米）
        add_self_loops: 是否添加自环
    
    Returns:
        adjacency: (batch, agents, agents) 或 (agents, agents)，值为 0/1
    """
    if positions.dim() == 2:
        # (agents, 3) -> (1, agents, 3)
        positions = positions.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, num_agents, _ = positions.shape
    device = positions.device
    
    # 计算距离矩阵
    # (batch, agents, 1, 3) - (batch, 1, agents, 3) = (batch, agents, agents, 3)
    pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    
    # 欧氏距离
    dist = torch.norm(pos_diff, dim=-1)  # (batch, agents, agents)
    
    # 距离阈值判断（使用 < 而不是 <=，确保阈值处不连接）
    adjacency = (dist < threshold).float()
    
    # 添加自环
    if add_self_loops:
        eye = torch.eye(num_agents, device=device, dtype=adjacency.dtype).unsqueeze(0)
        adjacency = torch.clamp(adjacency + eye, 0, 1)
    
    if squeeze_output:
        adjacency = adjacency.squeeze(0)
    
    return adjacency


# ====================================================================
# 改进的模型架构 - v3：GNN + BiGRU
# ====================================================================

class DynamicsAwareSwarmGRUModel_with_GNN(nn.Module):
    """
    动力学感知的集群 GRU 模型，增强了 GNN 支持
    
    架构：
        位置序列 → [构建邻接矩阵]
                 ↓
        24D 特征 → [序列级 GNN 处理] → GNN 特征
                 ↓
        [融合 24D + GNN 特征] → BiGRU 编码 → 多分支解码
    """
    
    def __init__(self, input_size=24, hidden_size=128, num_layers=2,
                 output_size=3, dropout=0.3, use_attention=True,
                 gnn_hidden=64, num_gnn_heads=4, edge_threshold=5.0,
                 fusion_mode='concat'):
        """
        Args:
            input_size: 输入特征维度（24D）
            hidden_size: GRU 隐层维度
            num_layers: GRU 层数
            output_size: 输出维度（3D 位置）
            dropout: Dropout 比例
            use_attention: 是否使用 BiGRU 输出的注意力
            gnn_hidden: GNN 隐层维度
            num_gnn_heads: GAT 多头数
            edge_threshold: 邻接矩阵距离阈值
            fusion_mode: 特征融合模式 ('concat', 'add', 'gate')
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_attention = use_attention
        self.gnn_hidden = gnn_hidden
        self.edge_threshold = edge_threshold
        self.fusion_mode = fusion_mode
        
        # ========== GNN 模块 ==========
        self.gnn = MultiHeadGraphAttention(
            in_channels=input_size,
            out_channels=gnn_hidden,
            num_heads=num_gnn_heads,
            dropout=dropout,
            concat=True
        )
        
        # ========== 特征融合 ==========
        fused_size = input_size + gnn_hidden
        
        if fusion_mode == 'concat':
            self.fusion_fc = nn.Linear(fused_size, hidden_size)
        elif fusion_mode == 'gate':
            self.gate_fc = nn.Linear(fused_size, fused_size)
            self.fusion_fc = nn.Linear(input_size, hidden_size)  # 输入是 24D
        elif fusion_mode == 'add':
            # 投影 GNN 特征到 24D 维度
            self.gnn_projection = nn.Linear(gnn_hidden, input_size)
            self.fusion_fc = nn.Linear(input_size, hidden_size)
        
        # ========== BiGRU 编码器 ==========
        self.feature_fusion_v2 = nn.Linear(hidden_size, hidden_size)
        
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # ========== 注意力机制（与 v2 相同） ==========
        if use_attention:
            self.pos_enc = nn.Parameter(torch.randn(1, 256, hidden_size * 2))
            self.enc_refiner = nn.MultiheadAttention(
                hidden_size * 2, num_heads=4, dropout=dropout, batch_first=True
            )
            self.decoder_attn = nn.MultiheadAttention(
                hidden_size * 2, num_heads=4, dropout=dropout, batch_first=True
            )
            self.decoder_attn_ln = nn.LayerNorm(hidden_size * 2)
        
        # ========== 解码器（与 v2 相同） ==========
        self.decoder_input_proj = nn.Linear(output_size, hidden_size * 2)
        self.decoder = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc_position = nn.Linear(hidden_size * 2, output_size)
        self.fc_velocity_dir = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)
        )
        self.fc_velocity_mag = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        self.fc_accel_tangent = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        self.fc_accel_normal = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _apply_gnn_to_sequence(self, x, x_orig):
        """
        对输入序列的每一步应用 GNN
        
        Args:
            x: (batch*agents, seq_in, 24) 或 (batch, seq_in, agents, 24)
            x_orig: (batch*agents, seq_in, 3) 或 (batch, seq_in, agents, 3)
        
        Returns:
            gnn_features: (batch*agents, seq_in, gnn_hidden)
        """
        # 处理 4D 输入
        if x.dim() == 4:
            batch_size, seq_in, num_agents, feat_dim = x.shape
            x_reshaped = x.reshape(batch_size * num_agents, seq_in, feat_dim)
            x_orig_reshaped = x_orig.reshape(batch_size * num_agents, seq_in, 3)
        else:
            batch_size = x.shape[0]
            seq_in = x.shape[1]
            num_agents = 1
            x_reshaped = x
            x_orig_reshaped = x_orig
        
        gnn_features = []
        
        for t in range(seq_in):
            feat_t = x_reshaped[:, t, :]  # (batch*agents, 24)
            pos_t = x_orig_reshaped[:, t, :]  # (batch*agents, 3)
            
            # 需要重塑回 (batch, agents, ...) 来正确计算邻接矩阵
            if num_agents > 1:
                feat_t_batch = feat_t.reshape(batch_size, num_agents, -1)
                pos_t_batch = pos_t.reshape(batch_size, num_agents, -1)
                
                # 对每个 batch 独立运行 GNN
                gnn_out_list = []
                for b in range(batch_size):
                    # 构建邻接矩阵
                    adj_b = build_adjacency_from_positions(
                        pos_t_batch[b:b+1], 
                        threshold=self.edge_threshold,
                        add_self_loops=True
                    )  # (1, agents, agents)
                    adj_b = adj_b.squeeze(0)  # (agents, agents)
                    
                    # 运行 GNN
                    feat_b = feat_t_batch[b]  # (agents, 24)
                    gnn_out_b = self.gnn(feat_b, adj_b)  # (agents, gnn_hidden)
                    gnn_out_list.append(gnn_out_b)
                
                gnn_t = torch.cat(gnn_out_list, dim=0)  # (batch*agents, gnn_hidden)
            else:
                # 单个 agent，创建自循环邻接矩阵
                adj = torch.ones(1, 1, device=x.device)
                gnn_t = self.gnn(feat_t, adj)  # (batch, gnn_hidden)
            
            gnn_features.append(gnn_t)
        
        # 堆叠：(batch*agents, seq_in, gnn_hidden)
        gnn_features = torch.stack(gnn_features, dim=1)
        
        return gnn_features
    
    def _merge_bidirectional_hidden(self, h):
        """合并双向隐状态"""
        num_directions = 2
        batch_size = h.size(1)
        h = h.view(self.num_layers, num_directions, batch_size, self.hidden_size)
        h = h.transpose(1, 2).contiguous()
        h = h.reshape(self.num_layers, batch_size, -1)
        return h
    
    def forward(self, x, x_orig, y=None, y_velocity=None, y_accel=None,
                teacher_forcing_ratio=0.5):
        """
        Args:
            x: 特征 (batch*agents, seq_in, 24) 或 (batch, seq_in, agents, 24)
            x_orig: 原始位置 (batch*agents, seq_in, 3) 或 (batch, seq_in, agents, 3)
            y: 位置增量目标
            y_velocity: 速度目标
            y_accel: 加速度目标
            teacher_forcing_ratio: TF 比例
        
        Returns:
            pred_position, pred_velocity, pred_accel
        """
        # 处理 4D 输入
        batch_size_orig = None
        num_agents_orig = None
        if x.dim() == 4:
            batch_size_orig, seq_in, num_agents_orig, feat_dim = x.shape
            x_reshaped = x.reshape(batch_size_orig * num_agents_orig, seq_in, feat_dim)
            x_orig_reshaped = x_orig.reshape(batch_size_orig * num_agents_orig, seq_in, 3)
            batch_size = batch_size_orig * num_agents_orig
            if y is not None:
                y_reshaped = y.reshape(batch_size_orig * num_agents_orig, -1, 3)
            else:
                y_reshaped = None
            if y_velocity is not None:
                y_velocity_reshaped = y_velocity.reshape(batch_size_orig * num_agents_orig, -1, 3)
            else:
                y_velocity_reshaped = None
            if y_accel is not None:
                y_accel_reshaped = y_accel.reshape(batch_size_orig * num_agents_orig, -1, 2)
            else:
                y_accel_reshaped = None
        else:
            x_reshaped = x
            x_orig_reshaped = x_orig
            y_reshaped = y
            y_velocity_reshaped = y_velocity
            y_accel_reshaped = y_accel
            batch_size = x.shape[0]
        
        # ========== 应用序列级 GNN ==========
        gnn_features = self._apply_gnn_to_sequence(x_reshaped, x_orig_reshaped)  # (batch, seq_in, gnn_hidden)
        
        # ========== 特征融合 ==========
        if self.fusion_mode == 'concat':
            fused = torch.cat([x_reshaped, gnn_features], dim=-1)  # (batch, seq_in, 24+gnn_h)
            x_fused = self.fusion_fc(fused)
        elif self.fusion_mode == 'gate':
            fused = torch.cat([x_reshaped, gnn_features], dim=-1)  # (batch, seq_in, 24+gnn_h)
            gate_logits = self.gate_fc(fused)  # (batch, seq_in, 24+gnn_h)
            # 对最后一维求均值以得到标量 gate，然后扩展回特征维度
            gate = torch.sigmoid(gate_logits.mean(dim=-1, keepdim=True))  # (batch, seq_in, 1)
            # 确保两边维度相同再做融合
            gnn_padded = F.pad(gnn_features, (0, self.input_size - self.gnn_hidden))  # (batch, seq_in, 24)
            fused_gated = gate * x_reshaped + (1 - gate) * gnn_padded  # (batch, seq_in, 24)
            x_fused = self.fusion_fc(fused_gated)
        elif self.fusion_mode == 'add':
            gnn_proj = self.gnn_projection(gnn_features)  # (batch, seq_in, 24)
            x_fused = self.fusion_fc(x_reshaped + gnn_proj)  # (batch, seq_in, 24)
        
        # ========== BiGRU 编码（与 v2 相同，仅输入维度有变化）==========
        x_fused = self.dropout(torch.relu(self.feature_fusion_v2(x_fused)))
        enc_out, h = self.encoder(x_fused)
        h = self._merge_bidirectional_hidden(h)
        
        # 可选注意力
        if self.use_attention:
            enc_out = enc_out + self.pos_enc[:, :enc_out.size(1), :]
            enc_out, _ = self.enc_refiner(enc_out, enc_out, enc_out)
        
        # ========== 解码（与 v2 相同） ==========
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
            
            if self.use_attention:
                attn_out, _ = self.decoder_attn(
                    decoder_out.unsqueeze(1),
                    enc_out,
                    enc_out
                )
                state = self.decoder_attn_ln(decoder_out + attn_out.squeeze(1))
            else:
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
            
            if y_reshaped is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_output = y_reshaped[:, t, :]
            else:
                prev_output = pred_pos
        
        output_position = torch.cat(predictions_position, dim=1)
        output_velocity = torch.cat(predictions_velocity, dim=1)
        output_accel = torch.cat(predictions_accel, dim=1)
        
        # 重塑回 4D
        if batch_size_orig is not None:
            output_position = output_position.reshape(batch_size_orig, seq_out, num_agents_orig, self.output_size)
            output_velocity = output_velocity.reshape(batch_size_orig, seq_out, num_agents_orig, 3)
            output_accel = output_accel.reshape(batch_size_orig, seq_out, num_agents_orig, 2)
        
        return output_position, output_velocity, output_accel


logger.info("✅ 模型 v3 (with GNN) 定义完成")
logger.info(f"特征维度: 24D + GNN 特征")
logger.info(f"GNN 架构: 多头图注意力 (GAT)")
logger.info(f"邻接矩阵: 距离阈值法 (可配置)")
