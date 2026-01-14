# 集群轨迹预测 - GNN 集成方案设计文档

## 📋 概述

当前系统基于 **24D 动力学感知特征 + BiGRU + Cross Attention** 的 v2 模型进行单无人机轨迹预测。为了增强集群轨迹预测能力，你的想法是**引入 GNN（图神经网络）来显式建模代理间交互**。

该文档提供了完整的集成方案，包括：
1. **图构建策略**（邻近度、距离阈值、k-NN）
2. **GNN 架构选择**（GAT、GCN、GraphSAGE、GIN 等）
3. **数据流改动**（数据集、模型输入/输出）
4. **模型融合方式**（GNN → BiGRU，GNN + BiGRU 并行，迭代融合）
5. **训练适配**（损失、评估、检查点）

---

## 🎯 核心设计决策

### 1. 图构建策略

**选项 A: 距离阈值（推荐，易实现）**
```
给定集群在时间步 t 的位置 (N, 3)：
- 计算所有代理间的欧氏距离
- 若距离 < threshold，则添加边
- 自环连接：agent_i 连接到自己（保留个体特征）
```
**优点**：物理意义清晰，对应 Couzin 模型中的排斥/吸引距离  
**参数**：`edge_threshold=5.0` （米）

**选项 B: k-NN（替代方案）**
```
- 对每个代理，连接距离最近的 k 个邻近体
- 对称化：若 i→j，则 j→i
```
**优点**：图的稠密度恒定，避免孤立节点  
**参数**：`k_neighbors=3~4`

**选项 C: 全连接（基准）**
```
- 所有节点两两相连
- 相当于 vanilla attention
```
**优点**：模型表达力最强  
**缺点**：计算复杂度 O(N²)，N 大时低效

---

### 2. GNN 层选择

| GNN 类型 | 公式 | 优点 | 缺点 | 推荐度 |
|---------|------|------|------|--------|
| **GAT** | $h_i' = \sigma(\sum_j \alpha_{ij} W h_j)$，其中 $\alpha_{ij}$ 由 attention 学习 | 自适应权重，可解释性强 | 参数多，训练需要较多数据 | ⭐⭐⭐⭐⭐ |
| **GCN** | $h_i' = \sigma(W \sum_j \frac{1}{\sqrt{d_i d_j}} h_j)$ | 简洁高效，全局感受野 | 固定邻域权重，可能欠拟合 | ⭐⭐⭐⭐ |
| **GraphSAGE** | 采样-聚合-更新，支持可变节点数 | 高效可扩展，归纳学习 | 超参数多（采样率、聚合函数） | ⭐⭐⭐⭐ |
| **GIN** | $h_i' = W(h_i + \sum_j h_j)$ | 强大的图同构能力 | 可能不如 GAT 对复杂交互敏感 | ⭐⭐⭐ |

**推荐方案**：**GAT（图注意力网络）**
- 天然支持集群场景中的"注意谁"问题
- 与现有 BiGRU 中的 attention 机制和谐
- 参数量适中，效果通常最好

---

### 3. 数据流改动

#### 当前 v2 流程
```
输入位置 (batch, seq_in, agents, 3)
  ↓
计算 24D 特征 (batch, seq_in, agents, 24)
  ↓
reshape 为 (batch*agents, seq_in, 24)
  ↓
BiGRU 编码 → 解码
  ↓
输出位置增量 (batch*agents, seq_out, 3)
```

#### 改进后流程（GNN + BiGRU）
```
输入位置 (batch, seq_in, agents, 3)
  ↓
计算 24D 特征 (batch, seq_in, agents, 24)
  ↓
[循环每个时间步 t ∈ [0, seq_in)]：
   ├─ 构建邻接矩阵 A_t from pos(t) 
   │   shape: (batch, agents, agents)
   ├─ 运行 GNN 层，聚合邻近信息
   │   feat_gnn_t = GNN(feat[t], A_t)
   │   shape: (batch, agents, hidden_gnn)
   └─ 将 feat_gnn_t 与原 24D 特征融合
  ↓
融合特征 (batch, seq_in, agents, hidden_gnn + 24)
  ↓
BiGRU 编码 → 解码（与原相同，仅输入维度增加）
  ↓
输出位置增量 (batch, seq_out, agents, 3)
```

---

### 4. 模型架构选择

#### **方案 A: 序列级 GNN（推荐）**
```
对输入序列的每一步，都独立运行 GNN 得到代理间交互特征，
再送入 BiGRU 做时间建模
```

优点：
- 清晰的因果关系：空间交互 → 时间演化
- GNN 负责"现在谁与谁相互作用"，BiGRU 负责"如何演化"
- 易于调试和理解

实现：
```python
class GNNSwarmGRUModel(nn.Module):
    def __init__(self, input_size=24, hidden_size=128, gnn_hidden=64, 
                 num_gru_layers=2, num_gnn_heads=4, edge_threshold=5.0):
        super().__init__()
        
        # GNN 层（GAT）
        self.gnn = MultiHeadGAT(
            in_channels=input_size,
            out_channels=gnn_hidden,
            heads=num_gnn_heads,
            dropout=0.3
        )
        
        # 融合层：24D + GNN_hidden → 192
        fused_size = input_size + gnn_hidden
        self.feature_fusion = nn.Linear(fused_size, hidden_size)
        
        # BiGRU（与 v2 相同）
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.3 if num_gru_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # 解码器（与 v2 相同）
        # ...
    
    def forward(self, x, x_orig, ...):
        """
        x: (batch, seq_in, agents, 24)
        x_orig: (batch, seq_in, agents, 3)
        """
        batch_size, seq_in, num_agents, feat_dim = x.shape
        
        # GNN 处理序列
        gnn_features = []
        for t in range(seq_in):
            pos_t = x_orig[:, t, :, :]  # (batch, agents, 3)
            feat_t = x[:, t, :, :]      # (batch, agents, 24)
            
            # 构建邻接矩阵
            A_t = self._build_adjacency_matrix(pos_t)  # (batch, agents, agents)
            
            # 运行 GNN
            # 输入形状需要转换为 (batch*agents, feat_dim)
            feat_t_flat = feat_t.reshape(batch_size * num_agents, feat_dim)
            # A_t 需要转换为各个批次的独立图...
            # （这里可以用 batch_gnn 或循环）
            
            feat_gnn_t = self.gnn(feat_t_flat, A_t)  # (batch*agents, gnn_hidden)
            gnn_features.append(feat_gnn_t.reshape(batch_size, num_agents, -1))
        
        # 堆叠 GNN 特征
        gnn_features = torch.stack(gnn_features, dim=1)  # (batch, seq_in, agents, gnn_hidden)
        
        # 融合原特征和 GNN 特征
        fused = torch.cat([x, gnn_features], dim=-1)  # (batch, seq_in, agents, 24+gnn_hidden)
        
        # 后续流程与 v2 相同...
        # reshape, BiGRU 编码, 解码, 多任务损失
```

#### **方案 B: 时空联合（复杂）**
```
GNN 在时间维度上也传播信息，类似 ST-GCN 或 ST-ResNet
```
缺点：计算复杂，超参数多，不推荐初期使用

---

### 5. 关键实现细节

#### **邻接矩阵构建**
```python
def _build_adjacency_matrix(self, positions, threshold=5.0):
    """
    Args:
        positions: (batch, agents, 3)
        threshold: 距离阈值
    
    Returns:
        adjacency: (batch, agents, agents) 二进制稀疏矩阵
    """
    # 计算距离矩阵
    batch_size, num_agents, _ = positions.shape
    
    # (batch, agents, 1, 3) - (batch, 1, agents, 3) = (batch, agents, agents, 3)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    
    # (batch, agents, agents)
    dist = torch.norm(diff, dim=-1)
    
    # 距离 < threshold 为 1，否则 0
    adjacency = (dist < threshold).float()
    
    # 自环（可选，取决于 GNN 实现）
    # adjacency = adjacency + torch.eye(num_agents, device=positions.device).unsqueeze(0)
    
    return adjacency
```

#### **GAT 实现（使用 PyTorch Geometric）**
```python
import torch_geometric.nn as gnn

class MultiHeadGAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        self.gat = gnn.GATv2Conv(
            in_channels, 
            out_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True,  # 拼接多头，得到 out_channels
            add_self_loops=True
        )
    
    def forward(self, x, edge_index):
        """
        x: (num_nodes, in_channels)
        edge_index: (2, num_edges)
        """
        return self.gat(x, edge_index)
```

#### **代理间特征融合**
```python
# 选项 1：简单拼接（推荐）
fused = torch.cat([feat_24d, feat_gnn], dim=-1)  # (batch, agents, 24+gnn_h)

# 选项 2：加权融合
fused = self.fusion_weight * feat_24d + (1 - self.fusion_weight) * feat_gnn_projected

# 选项 3：Gating 融合
gate = torch.sigmoid(self.gate_fc(torch.cat([feat_24d, feat_gnn], dim=-1)))
fused = gate * feat_24d + (1 - gate) * feat_gnn
```

---

## 📝 修改清单

### 文件 1：`train_swarm_model_v2_dynamics_aware.py`

**新增内容**：

1. **GAT 模块**
   ```python
   class MultiHeadGAT(nn.Module):
       # 如上所示
   ```

2. **邻接矩阵构建函数**
   ```python
   def build_adjacency_from_positions(positions, threshold=5.0):
       # 如上所示
   ```

3. **新模型类：DynamicsAwareSwarmGRUModel_with_GNN**
   ```python
   class DynamicsAwareSwarmGRUModel_with_GNN(DynamicsAwareSwarmGRUModel):
       def __init__(self, ..., gnn_hidden=64, num_gnn_heads=4, edge_threshold=5.0):
           super().__init__(...)
           self.gnn = MultiHeadGAT(input_size, gnn_hidden, heads=num_gnn_heads)
           self.edge_threshold = edge_threshold
           # 融合层
       
       def forward(self, x, x_orig, ...):
           # 序列级 GNN 处理
           # ...
   ```

### 文件 2：`train_swarm_v2_complete.py`

**修改内容**：

1. **导入新模型**
   ```python
   from train_swarm_model_v2_dynamics_aware import (
       ...,
       DynamicsAwareSwarmGRUModel_with_GNN
   )
   ```

2. **命令行参数**
   ```python
   parser.add_argument('--use_gnn', action='store_true', 
                       help='启用 GNN 进行代理间交互建模')
   parser.add_argument('--gnn_hidden', type=int, default=64,
                       help='GNN 隐层维度')
   parser.add_argument('--gnn_heads', type=int, default=4,
                       help='GAT 多头数')
   parser.add_argument('--edge_threshold', type=float, default=5.0,
                       help='邻接矩阵构建的距离阈值（米）')
   ```

3. **模型实例化**
   ```python
   if args.use_gnn:
       model = DynamicsAwareSwarmGRUModel_with_GNN(
           input_size=24,
           hidden_size=args.hidden_size,
           gnn_hidden=args.gnn_hidden,
           num_gnn_heads=args.gnn_heads,
           edge_threshold=args.edge_threshold,
           ...
       )
   else:
       model = DynamicsAwareSwarmGRUModel(...)
   ```

### 文件 3（可选）：`precompute_features_v3.py`

**不需要改动** — 特征预计算与 GNN 正交

---

## 🔬 测试与验证步骤

1. **单元测试**：
   ```python
   # 验证邻接矩阵构建
   pos = torch.randn(2, 5, 3)  # 2个batch，5个代理
   A = build_adjacency_from_positions(pos, threshold=5.0)
   assert A.shape == (2, 5, 5)
   assert (A >= 0).all() and (A <= 1).all()
   assert (A.diagonal(dim1=1, dim2=2) > 0).all()  # 检查自环
   ```

2. **前向传播测试**：
   ```python
   model = DynamicsAwareSwarmGRUModel_with_GNN(...)
   x = torch.randn(8, 20, 4, 24)  # batch=8, seq_in=20, agents=4, feat=24
   x_orig = torch.randn(8, 20, 4, 3)
   pos_out, vel_out, accel_out = model(x, x_orig)
   assert pos_out.shape == (8, 10, 4, 3)  # seq_out=10
   ```

3. **梯度流测试**：
   ```python
   loss = pos_out.sum()
   loss.backward()
   # 检查 GNN 层参数的梯度是否非零
   assert model.gnn.gat.lin_l.weight.grad is not None
   ```

4. **端到端训练**：
   ```bash
   python train_swarm_v2_complete.py \
       --agents 3 \
       --epochs 5 \
       --use_gnn \
       --gnn_hidden 64 \
       --gnn_heads 4 \
       --edge_threshold 5.0 \
       --batch_size 64
   ```

---

## 📊 预期改进

| 指标 | v2（无 GNN） | v2+GNN | 提升 |
|------|------------|--------|------|
| 集群 MAE (m) | ~0.45 | ~0.30-0.35 | 25-35% ↓ |
| 速度预测准度 | ~0.15 m/s | ~0.10 m/s | 33% ↓ |
| 模型参数量 | ~450K | ~550-600K | +15-20% |
| 单 epoch 耗时 | ~45s | ~55-60s | +20-30% |
| 收敛轮数 | ~150 | ~120-130 | -15-20% |

**注**：提升幅度取决于数据中集群交互的强度和 Couzin 模型参数。

---

## 🚀 后续优化方向

1. **动态图**：使用 k-NN 替代阈值，适应不同密度集群
2. **边特征**：添加相对速度、相对加速度作为边特征
3. **多层 GNN**：堆叠 2-3 层 GAT，增强感受野
4. **时空 GNN**：在 GRU 解码阶段也添加 GNN，形成反馈
5. **Graph Pooling**：添加全局集群特征（集群质心、方差等）

---

## 💡 常见问题

**Q1: 为什么选 GAT 而不是 GCN？**  
A: GAT 的注意力权重是学习的，能更好地捕捉"谁对谁重要"这一动态关系。在无人机集群中，同一时刻距离相同的两个邻近体对当前体的影响可能不同（取决于速度、加速度等）。

**Q2: 邻接矩阵是否应该在解码阶段也使用？**  
A: 初期推荐仅在编码阶段使用（历史观察）。解码阶段的真实位置未知，只能用预测位置构建邻接矩阵，这会引入累积误差。可后续尝试迭代构建。

**Q3: GNN 会不会过拟合集群拓扑？**  
A: 风险存在，但可通过：(a) Dropout，(b) 在训练中随机化边阈值，(c) 多个 agent 数目的数据混合训练来缓解。

