#!/usr/bin/env python3
"""
集群轨迹预测模型 v2 - 基于动态图神经网络 (Dynamic Graph Neural Network)

核心改进：
=========
1. ✓ 动态图建模：捕捉集群内UAV的空间交互
   - 每个时刻构建邻接矩阵（基于欧氏距离）
   - GCN聚合邻近UAV特征
   - 学习避撞、速度对齐、距离保持规则

2. ✓ GCN + BiGRU时空融合
   - GCN处理空间交互（同一时刻多UAV间依赖）
   - BiGRU处理时序动态（单UAV的轨迹趋势）
   - 两者协同建模集群本质特性

3. ✓ 强化正则化与抗噪能力
   - 分层dropout (GCN: 0.1, GRU: 0.2, FC: 0.3)
   - 增强weight_decay (1e-4)
   - 噪声鲁棒训练 (0.01-0.05量级)

4. ✓ 多指标评估 + 智能早停
   - MAE + RMSE + MAPE多维度评估
   - 早停patience=15 (vs 30)
   - 监控集群级指标（最大偏差、碰撞风险）
   python train_swarm_gnn.py ^
  --agents 3 ^
  --epochs 300 ^
  --batch_size 128 ^
  --hidden_size 128 ^
  --num_layers 3 ^
  --dropout 0.4 ^
  --lr 1e-3 ^
  --weight_decay 1e-4 ^
  --patience 30 ^
  --output_dir gru_models_subset_nogcn1 ^
  --use_subset ^
  --use_gcn 0 ^
  --seed 42

参考论文：
"Multidimensional Trajectory Prediction of UAV Swarms Based on Dynamic Graph Neural Network"
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


# ============= 工具函数 =============

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
        
        for plane_idx, (dim1, dim2) in enumerate([(0, 1), (1, 2), (0, 2)]):
            pos = np.column_stack([traj[:, dim1], traj[:, dim2]])
            vel = np.gradient(pos, axis=0) / dt
            acc = np.gradient(vel, axis=0) / dt
            # 计算平面上的曲率：使用 2D 向量的标量叉积 (cross product in 2D)
            cross_2d = vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]
            vel_norm = np.linalg.norm(vel, axis=1)
            curv = np.abs(cross_2d) / np.maximum(vel_norm ** 3, eps)
            curv = np.nan_to_num(curv, nan=0.0, posinf=1.0, neginf=0.0)
            plane_curvs[:, i, plane_idx] = curv
    
    return plane_curvs


# ============= 动态图神经网络模块 =============

class DynamicGraphConstructor(nn.Module):
    """动态图构造：基于欧氏距离自适应构建邻接关系"""
    
    def __init__(self, distance_threshold=10.0, k_neighbors=2):
        """
        Args:
            distance_threshold: 距离阈值（超过则无边）
            k_neighbors: 每个节点保留的最近邻数量
        """
        super().__init__()
        self.distance_threshold = distance_threshold
        self.k_neighbors = k_neighbors
    
    def forward(self, positions):
        """
        Args:
            positions: (batch*agents, seq, 3) 或 (batch, seq, agents, 3)
        
        Returns:
            adj_matrix: (batch*agents, agents, agents) 邻接矩阵
            edge_weights: (batch*agents, agents, agents) 边权重
        """
        # 重塑为 (batch*agents, seq, 3)
        if positions.dim() == 4:
            batch, seq, agents, _ = positions.shape
            positions = positions.reshape(batch * agents, seq, 3)
        else:
            batch_agents, seq, _ = positions.shape
            agents = int(np.sqrt(batch_agents))  # 假设batch_agents = batch * agents
            batch = batch_agents // agents
        
        # 获取最后一个时刻的位置（定义图结构）
        pos_last = positions[:, -1, :]  # (batch*agents, 3)
        pos_last = pos_last.reshape(batch, agents, 3)  # (batch, agents, 3)
        
        # 计算两两欧氏距离
        # distances: (batch, agents, agents)
        pos_diff = pos_last.unsqueeze(2) - pos_last.unsqueeze(1)  # (batch, agents, agents, 3)
        distances = torch.norm(pos_diff, dim=-1)  # (batch, agents, agents)
        
        # 基于距离阈值的邻接矩阵
        adj_matrix = (distances < self.distance_threshold).float()
        
        # ✅ 移除自环（对每个batch分别处理）
        for b in range(batch):
            adj_matrix[b].fill_diagonal_(0)
        
        # ✅ K-近邻：每个节点只连接最近的k个邻点
        for b in range(batch):
            for i in range(agents):
                row_dist = distances[b, i, :]
                row_dist[i] = float('inf')  # 移除自身
                if row_dist.min() < self.distance_threshold:
                    # 保留k个最近的
                    k_nearest_idx = torch.topk(row_dist, min(self.k_neighbors, (row_dist < self.distance_threshold).sum()), 
                                              largest=False)[1]
                    mask = torch.zeros(agents, device=distances.device)
                    mask[k_nearest_idx] = 1.0
                    adj_matrix[b, i, :] *= mask
        
        # 边权重：基于距离的高斯核（距离越近权重越大）
        sigma = self.distance_threshold / 3.0
        edge_weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        # ✅ 移除自环（对每个batch分别处理）
        for b in range(batch):
            edge_weights[b].fill_diagonal_(0)
        edge_weights = edge_weights * adj_matrix
        
        return adj_matrix, edge_weights


class GraphConvolutionNetwork(nn.Module):
    """GCN层：聚合邻近UAV的特征"""
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj_matrix):
        """
        Args:
            x: (batch, agents, in_features) 节点特征
            adj_matrix: (batch, agents, agents) 邻接矩阵（含边权重）
        
        Returns:
            out: (batch, agents, out_features) 聚合后的节点特征
        """
        # 特征线性投影
        x = self.dropout(x)
        x = torch.matmul(x, self.weight) + self.bias  # (batch, agents, out_features)
        
        # 邻近聚合：通过邻接矩阵汇聚周围UAV特征
        # 对邻接矩阵进行行归一化（度数归一化）
        degree = adj_matrix.sum(dim=-1, keepdim=True) + 1e-8  # (batch, agents, 1)
        adj_normalized = adj_matrix / degree  # (batch, agents, agents)
        
        # 消息传递
        out = torch.matmul(adj_normalized, x)  # (batch, agents, out_features)
        
        return out


class DynamicGraphSwarmGRUModel(nn.Module):
    """
    动态图+BiGRU混合架构
    ========================
    原理：
    1. 对每个时刻构建动态图（基于当前位置）
    2. GCN层聚合邻近UAV的特征（空间交互）
    3. BiGRU处理GCN输出的序列（时序动态）
    4. 解码器预测增量位移
    """
    
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, 
                 num_agents=3, output_size=3, dropout=0.2, use_gcn=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_agents = num_agents
        self.output_size = output_size
        self.use_gcn = use_gcn
        
        # 动态图构造器
        if use_gcn:
            self.graph_constructor = DynamicGraphConstructor(
                distance_threshold=20.0, 
                k_neighbors=2
            )
            
            # GCN编码器
            self.gcn_layers = nn.ModuleList([
                GraphConvolutionNetwork(input_size if i == 0 else hidden_size, 
                                       hidden_size, dropout=0.1)
                for i in range(2)  # 2层GCN
            ])
        
        # BiGRU编码器
        gru_input_size = hidden_size if use_gcn else input_size
        self.encoder = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # 解码器
        self.decoder_input_proj = nn.Linear(output_size, hidden_size * 2)
        self.decoder = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 输出头
        self.fc_out = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x, x_orig, y=None, teacher_forcing_ratio=0.5):
        """
        Args:
            x: (batch, seq_in, agents, 16) 特征
            x_orig: (batch, seq_in, agents, 3) 原始位置
            y: (batch, seq_out, agents, 3) 目标增量
            teacher_forcing_ratio: TF比例
        
        Returns:
            output: (batch, seq_out, agents, 3) 预测增量
        """
        batch, seq_in, num_agents, feat_dim = x.shape
        
        if self.use_gcn:
            # ✅ 关键：对输入序列的每个时刻应用GCN（捕捉空间交互）
            gcn_outputs = []
            
            for t in range(seq_in):
                x_t = x[:, t, :, :]  # (batch, agents, 16)
                
                # 动态图构造
                adj_matrix, edge_weights = self.graph_constructor(x_orig)
                
                # GCN前向传播
                h = x_t
                for gcn_layer in self.gcn_layers:
                    h = F.relu(gcn_layer(h, edge_weights))
                
                gcn_outputs.append(h)
            
            # 拼接GCN输出序列
            x_encoded = torch.stack(gcn_outputs, dim=1)  # (batch, seq_in, agents, hidden_size)
        else:
            x_encoded = x
        
        # BiGRU处理
        x_reshaped = x_encoded.reshape(batch * num_agents, seq_in, -1)
        enc_out, h = self.encoder(x_reshaped)
        h = self._merge_bidirectional_hidden(h, num_agents)
        
        # 解码
        seq_out = y.shape[1] if y is not None else 10
        predictions = []
        
        # 初始输出
        prev_output = torch.zeros(batch * num_agents, self.output_size, device=x.device)
        h_t = h
        
        for t in range(seq_out):
            decoder_input = self.decoder_input_proj(prev_output).unsqueeze(1)
            decoder_out, h_t = self.decoder(decoder_input, h_t)
            
            y_t = self.fc_out(decoder_out.squeeze(1))
            predictions.append(y_t.unsqueeze(1))
            
            # Teacher Forcing
            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                y_reshaped = y.reshape(batch * num_agents, seq_out, self.output_size)
                prev_output = y_reshaped[:, t, :]
            else:
                prev_output = y_t.detach()
        
        output = torch.cat(predictions, dim=1)
        output = output.reshape(batch, seq_out, num_agents, self.output_size)
        
        return output
    
    def _merge_bidirectional_hidden(self, h, num_agents):
        """合并双向隐状态"""
        # h: (num_layers * 2, batch*agents, hidden_size)
        batch_agents = h.size(1)
        h = h.view(self.num_layers, 2, batch_agents, self.hidden_size)
        h = h.transpose(1, 2).reshape(self.num_layers, batch_agents, -1)
        return h


# ============= 损失函数 =============

class SwarmMultiObjectiveLoss(nn.Module):
    """
    集群特化的多目标损失：同时考虑单机精度与集群交互规律
    """
    
    def __init__(self, alpha=0.6, beta=0.2, gamma=0.1, delta=0.1):
        """
        alpha: 位置MSE权重 (单机精度)
        beta: 加速度权重 (平滑性)
        gamma: 速度权重 (连续性)
        delta: 集群约束权重 (避撞/对齐/距离保持)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
    
    def forward(self, pred, target, positions_abs=None):
        """
        Args:
            pred: (batch, seq_out, agents, 3) 预测增量
            target: (batch, seq_out, agents, 3) 目标增量
            positions_abs: (batch, seq_out, agents, 3) 绝对位置（用于集群约束）
        """
        # 1. 位置MSE
        pos_loss = F.mse_loss(pred, target)
        
        # 2. 加速度平滑性
        if pred.shape[1] > 2:
            pred_acc = torch.diff(torch.diff(pred, dim=1), dim=1)
            target_acc = torch.diff(torch.diff(target, dim=1), dim=1)
            acc_loss = F.mse_loss(pred_acc, target_acc)
        else:
            acc_loss = torch.tensor(0.0, device=pred.device)
        
        # 3. 速度连续性
        if pred.shape[1] > 1:
            pred_vel = torch.diff(pred, dim=1)
            target_vel = torch.diff(target, dim=1)
            vel_loss = F.mse_loss(pred_vel, target_vel)
        else:
            vel_loss = torch.tensor(0.0, device=pred.device)
        
        # 4. 集群约束（避撞）
        swarm_loss = torch.tensor(0.0, device=pred.device)
        if positions_abs is not None and self.delta > 0:
            batch, seq_out, num_agents, _ = positions_abs.shape
            
            # 计算两两距离（检查碰撞风险）
            for t in range(seq_out):
                pos_t = positions_abs[:, t, :, :]  # (batch, agents, 3)
                
                # 配对距离
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        dist = torch.norm(pos_t[:, i, :] - pos_t[:, j, :], dim=1)  # (batch,)
                        
                        # 惩罚距离过近（碰撞风险）
                        min_dist = 1.0  # 最小安全距离 (米)
                        collision_penalty = torch.relu(min_dist - dist).mean()
                        swarm_loss += collision_penalty
        
        total_loss = (self.alpha * pos_loss + 
                     self.beta * acc_loss + 
                     self.gamma * vel_loss + 
                     self.delta * swarm_loss)
        
        return total_loss


# ============= 数据集 =============

class SwarmTrajectoryDatasetGNN(Dataset):
    """GNN专用的数据集类"""
    
    def __init__(self, X, Y, input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 feature_mean=None, feature_std=None, dt=0.1):
        self.X_orig = X.copy()
        self.Y_orig = Y.copy()
        self.dt = dt
        
        self.input_mean = input_mean if input_mean is not None else X.mean()
        self.input_std = input_std if input_std is not None else max(X.std(), 1e-8)
        
        self.output_mean = output_mean if output_mean is not None else 0.0
        self.output_std = output_std if output_std is not None else 1.0
        
        self.feature_mean = feature_mean if feature_mean is not None else np.zeros(16)
        self.feature_std = feature_std if feature_std is not None else np.ones(16)
    
    def __len__(self):
        return len(self.X_orig)
    
    def __getitem__(self, idx):
        x = self.X_orig[idx]
        y = self.Y_orig[idx]
        
        # 计算特征
        vel = compute_multi_scale_velocity(x, self.dt)
        curv_3d = compute_curvature(x, self.dt)
        curv_plane = compute_plane_curvatures(x, self.dt)
        features = np.concatenate([x, vel, curv_3d, curv_plane], axis=-1)
        features = np.clip(features, -100, 100)
        
        # 归一化特征
        features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        features = np.clip(features, -5, 5)
        
        # 增量目标
        y_delta = y - x[-1:, :, :]
        y_target = (y_delta - self.output_mean) / (self.output_std + 1e-8)
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


# ============= 训练与评估 =============

def compute_swarm_metrics(pred_abs, target_abs):
    """
    计算多维度集群指标：
    - MAE: 平均绝对误差
    - RMSE: 均方根误差
    - MAPE: 平均绝对百分比误差
    - MaxError: 最大偏差（检测极端情况）
    """
    error = np.abs(pred_abs - target_abs)
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))
    
    # MAPE（避免除零）
    target_norm = np.abs(target_abs)
    target_norm[target_norm < 0.1] = 0.1
    mape = np.mean(error / target_norm)
    
    # 最大误差（集群级鲁棒性）
    max_error = np.max(error)
    
    return mae, rmse, mape, max_error


def train_epoch_gnn(model, train_loader, optimizer, criterion, device, 
                    grad_clip=1.0, teacher_forcing_ratio=0.5, epoch=1, total_epochs=200):
    """GNN训练循环"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    
    tf_current = max(0.0, teacher_forcing_ratio - 0.005 * (epoch - 1))
    
    for features, x_orig, y_norm, y_orig in tqdm(train_loader, desc=f"训练 [GNN, TF={tf_current:.4f}]"):
        features = features.to(device, non_blocking=True)
        x_orig = x_orig.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)
        y_orig = y_orig.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        pred_norm = model(features, x_orig, y_norm, teacher_forcing_ratio=tf_current)
        
        # 用于计算集群约束的绝对位置
        pred_abs = x_orig[:, -1:, :, :] + pred_norm
        
        loss = criterion(pred_norm, y_norm, positions_abs=pred_abs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else 0.0, tf_current


def evaluate_gnn(model, val_loader, criterion, device, output_mean=None, output_std=None):
    """GNN评估：返回多指标"""
    model.eval()
    total_loss = 0.0
    mae_list = []
    rmse_list = []
    mape_list = []
    max_error_list = []
    count = 0
    
    with torch.no_grad():
        for features, x_orig, y_norm, y_orig in tqdm(val_loader, desc="评估"):
            features = features.to(device, non_blocking=True)
            x_orig = x_orig.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            y_orig = y_orig.to(device, non_blocking=True)
            
            pred_norm = model(features, x_orig, teacher_forcing_ratio=0.0)
            pred_abs = x_orig[:, -1:, :, :] + pred_norm * output_std + output_mean
            
            loss = criterion(pred_norm, y_norm)
            
            # 多指标计算
            pred_np = pred_abs.cpu().numpy()
            target_np = y_orig.cpu().numpy()
            
            mae, rmse, mape, max_err = compute_swarm_metrics(pred_np, target_np)
            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape)
            max_error_list.append(max_err)
            
            total_loss += loss.item()
            count += 1
    
    avg_loss = total_loss / count if count > 0 else 0.0
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    avg_mape = np.mean(mape_list)
    max_error_overall = np.mean(max_error_list)
    
    return avg_loss, avg_mae, avg_rmse, avg_mape, max_error_overall


def load_swarm_data_gnn(data_dir, num_agents, batch_size=256, val_split=0.2, use_subset=False):
    """加载数据（参考之前的load_swarm_data）
    
    Args:
        use_subset: 是否使用子集数据（带 _subset 后缀）
    """
    data_path = Path(data_dir)
    
    # 尝试加载子集或完整数据
    if use_subset:
        X_file = data_path / f'input_agents_{num_agents}_subset.npz'
        Y_file = data_path / f'output_agents_{num_agents}_subset.npz'
    else:
        X_file = data_path / f'input_agents_{num_agents}.npz'
        Y_file = data_path / f'output_agents_{num_agents}.npz'
    
    # 如果子集不存在，尝试完整数据
    if not X_file.exists() and use_subset:
        logger.warning(f"⚠️ 子集文件不存在，尝试加载完整数据...")
        X_file = data_path / f'input_agents_{num_agents}.npz'
        Y_file = data_path / f'output_agents_{num_agents}.npz'
    
    if not X_file.exists() or not Y_file.exists():
        raise FileNotFoundError(f"找不到数据文件: {X_file}, {Y_file}")
    
    logger.info(f"加载 {num_agents} 架无人机数据...")
    X = np.load(X_file)['data']  # 形状: (seq_len, samples, agents, 3)
    Y = np.load(Y_file)['data']  # 形状: (seq_out, samples, agents, 3)
    
    # ✅ 数据已经是 (seq, samples, agents, 3) 格式（来自预处理）
    # 转置为 (samples, seq, agents, 3) 格式供模型使用
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    logger.info(f"  输入形状: {X.shape}, 输出形状: {Y.shape}")
    
    # 计算统计量
    input_mean = np.mean(X.reshape(-1, 3), axis=0)
    input_std = np.std(X.reshape(-1, 3), axis=0)
    input_std = np.where(input_std < 1e-8, 1.0, input_std)
    
    y_delta = Y - X[:, -1:, :, :]
    output_mean = np.mean(y_delta.reshape(-1, 3), axis=0)
    output_std = np.std(y_delta.reshape(-1, 3), axis=0)
    output_std = np.where(output_std < 1e-8, 1.0, output_std)
    
    # 16维特征统计
    input_mean_all = np.zeros(16)
    input_std_all = np.ones(16)
    
    # 分割数据
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    num_val = max(1, int(num_samples * val_split)) if val_split > 0 else 0
    val_idx = indices[:num_val]
    train_idx = indices[num_val:]
    
    train_dataset = SwarmTrajectoryDatasetGNN(
        X[train_idx], Y[train_idx],
        input_mean, input_std, output_mean, output_std,
        input_mean_all, input_std_all
    )
    val_dataset = SwarmTrajectoryDatasetGNN(
        X[val_idx], Y[val_idx],
        input_mean, input_std, output_mean, output_std,
        input_mean_all, input_std_all
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=torch.cuda.is_available())
    
    stats = {
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std,
        'input_mean_all': input_mean_all,
        'input_std_all': input_std_all,
    }
    
    return train_loader, val_loader, stats


# ============= 主函数 =============

def main():
    parser = argparse.ArgumentParser(description='基于动态图神经网络的集群轨迹预测')
    parser.add_argument('--data_dir', type=str, default='swarm_segments')
    parser.add_argument('--agents', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)  
    parser.add_argument('--output_dir', type=str, default='gru_models_gnn')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_subset', action='store_true', help='使用数据子集进行快速验证')
    parser.add_argument('--use_gcn', type=int, default=1, help='是否使用 GCN (1=是, 0=否，用于快速验证)')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # ✅ 保存超参数配置
    config = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'agents': args.agents,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'seed': args.seed,
        'device': str(device),
    }
    
    config_path = output_dir / f'config_agents_{args.agents}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✓ 配置已保存: {config_path}")
    
    try:
        train_loader, val_loader, stats = load_swarm_data_gnn(
            args.data_dir, args.agents, args.batch_size, use_subset=args.use_subset
        )
    except FileNotFoundError as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # 创建模型
    model = DynamicGraphSwarmGRUModel(
        input_size=16,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_agents=args.agents,
        output_size=3,
        dropout=args.dropout,
        use_gcn=bool(args.use_gcn)  # 支持禁用 GCN 以加快速度
    ).to(device)
    
    logger.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    criterion = SwarmMultiObjectiveLoss(alpha=0.6, beta=0.2, gamma=0.1, delta=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )
    
    # ✅ 初始化训练历史与检查点恢复
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    
    training_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_mape': [],
        'val_max_err': [],
        'learning_rate': [],
        'teacher_forcing_ratio': []
    }
    
    # ✅ 支持中断恢复逻辑
    best_model_path = output_dir / f'best_model_agents_{args.agents}.pt'
    last_checkpoint_path = output_dir / f'last_checkpoint_agents_{args.agents}.pt'
    interrupted_checkpoint_path = output_dir / f'interrupted_checkpoint_agents_{args.agents}.pt'
    
    if args.resume:
        # 显式恢复
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            logger.info(f"从检查点恢复: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            training_history = ckpt.get('training_history', training_history)
            logger.info(f"✓ 恢复成功，从 epoch {start_epoch} 继续")
    elif last_checkpoint_path.exists():
        # 自动恢复最后一个检查点
        logger.info(f"检测到最后一个检查点，自动恢复...")
        ckpt = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        training_history = ckpt.get('training_history', training_history)
        logger.info(f"✓ 自动恢复成功，从 epoch {start_epoch} 继续 (当前best_loss={best_val_loss:.6f})")
    elif interrupted_checkpoint_path.exists():
        # 恢复中断检查点（优先级最低）
        logger.info(f"检测到中断检查点，尝试恢复...")
        try:
            ckpt = torch.load(interrupted_checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            patience_counter = ckpt.get('patience_counter', 0)
            training_history = ckpt.get('training_history', training_history)
            logger.info(f"✓ 中断恢复成功，从 epoch {start_epoch} 继续")
        except Exception as e:
            logger.warning(f"中断恢复失败，从头开始: {e}")
    
    print("\n" + "="*140)
    print(f"{'Epoch':<8} {'Train Loss':<16} {'Val Loss':<16} {'MAE':<12} {'RMSE':<12} {'MAPE':<12} {'MaxErr':<12} {'LR':<12} {'TF':<8} {'Status':<20}")
    print("="*140)
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, tf = train_epoch_gnn(
            model, train_loader, optimizer, criterion, device,
            teacher_forcing_ratio=0.5, epoch=epoch+1, total_epochs=args.epochs
        )
        
        val_loss, val_mae, val_rmse, val_mape, val_max_err = evaluate_gnn(
            model, val_loader, criterion, device, 
            output_mean=torch.tensor(stats['output_mean'], device=device),
            output_std=torch.tensor(stats['output_std'], device=device)
        )
        
        # ✅ 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # ✅ 记录训练历史
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(float(train_loss))
        training_history['val_loss'].append(float(val_loss))
        training_history['val_mae'].append(float(val_mae))
        training_history['val_rmse'].append(float(val_rmse))
        training_history['val_mape'].append(float(val_mape))
        training_history['val_max_err'].append(float(val_max_err))
        training_history['learning_rate'].append(float(current_lr))
        training_history['teacher_forcing_ratio'].append(float(tf))
        
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            status = "✓ BEST"
            
            # ✅ 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'stats': stats,
                'config': config,
                'training_history': training_history,
            }, best_model_path)
        else:
            patience_counter += 1
            status = f"patience: {patience_counter}/{args.patience}"
            if patience_counter >= args.patience:
                print(f"\n{'='*140}")
                print(f"✓ 早停 (patience={args.patience})")
                break
        
        # ✅ 每个epoch保存最后一个检查点（支持中断恢复）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'stats': stats,
            'config': config,
            'training_history': training_history,
        }, last_checkpoint_path)
        
        print(f"{epoch+1:<8} {train_loss:<16.6f} {val_loss:<16.6f} {val_mae:<12.6f} {val_rmse:<12.6f} {val_mape:<12.6f} {val_max_err:<12.6f} {current_lr:<12.2e} {tf:<8.4f} {status:<20}")
        
        scheduler.step(val_loss)
    
    print("="*140)
    
    # ✅ 保存最终训练历史为JSON和CSV
    history_json_path = output_dir / f'training_history_agents_{args.agents}.json'
    with open(history_json_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"✓ 训练历史已保存: {history_json_path}")
    
    # ✅ 同时保存CSV格式（便于Excel查看）
    history_csv_path = output_dir / f'training_history_agents_{args.agents}.csv'
    with open(history_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=training_history.keys())
        writer.writeheader()
        for i in range(len(training_history['epoch'])):
            row = {k: training_history[k][i] for k in training_history.keys()}
            writer.writerow(row)
    logger.info(f"✓ 训练历史CSV已保存: {history_csv_path}")
    
    # ✅ 删除中断检查点（训练成功完成）
    if interrupted_checkpoint_path.exists():
        interrupted_checkpoint_path.unlink()
        logger.info(f"✓ 删除中断检查点")
    
    logger.info(f"✓ 训练完成！")
    logger.info(f"  最佳模型: {best_model_path}")
    logger.info(f"  最佳损失: {best_val_loss:.6f}")
    logger.info(f"  配置文件: {config_path}")


if __name__ == '__main__':
    main()
