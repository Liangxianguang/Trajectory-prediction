# MRGTraj 用于无人机集群轨迹预测的适配分析

## 📊 问题分析

### 1. 数据格式差异

#### 你的数据格式
```
CSV格式: timestamp, agent_0_x, agent_0_y, agent_0_z, agent_1_x, ..., agent_n_z
形状: (T,) → 相对于 T 个时间步，每个无人机 3 坐标（XYZ）
存储: NPZ 格式 (seq_len, samples, num_agents, 3)
```

#### MRGTraj 原始数据格式
```
设计用途: 行人轨迹预测（2D 平面）
输入形状: [batch_size, src_len, 2]  ← 仅 2D (X, Y)
输出形状: [batch_size, tgt_len, 4]  ← (X, Y, X_rel, Y_rel)
```

### 2. 核心差异

| 维度 | 你的数据 | MRGTraj 原始 | 兼容性 |
|------|--------|----------|--------|
| **坐标维度** | 3D (X, Y, Z) | 2D (X, Y) | ⚠️ 需要修改 |
| **多智能体** | 3-6 个无人机 | 单个行人 | ❌ 需要改进 |
| **社交互作** | 集群协作 | 人群避碰 | ✓ 概念相同 |
| **时间长度** | 20→10 步 | 8→12 步 | ✓ 可配置 |

---

## ✅ 可以使用的原因

1. **非自回归架构**: MRGTraj 使用映射-优化-生成框架，不依赖逐步预测，适合集群预测
2. **社交交互建模**: 模型中的 `SocialLatentGenerator` 可以捕捉无人机间的协作关系
3. **多样性预测**: 通过多次采样生成多个可能的轨迹（你的数据支持这种需求）

---

## ❌ 需要修改的原因

1. **维度不匹配**: 
   - 原始模型: `past_encoder` 输入维度硬编码为 2
   - 你的数据: 需要处理 3 维坐标 + 多个无人机

2. **多智能体处理**:
   - 原始模型: 处理单个行人 + 社交上下文
   - 你的数据: 同时处理多个无人机的轨迹

3. **批处理方式**:
   - 原始模型: batch_mask 用于标记有效行人
   - 你的数据: 所有无人机都有效（固定编队）

---

## 🔧 修改方案 (3 个难度级别)

### 方案 A: 最小化修改 (简单) ⭐⭐

**思路**: 将多无人机轨迹展平为单行人轨迹

```python
# 输入: (seq_len, samples, num_agents, 3)
# 转换: (seq_len*num_agents, samples, 2)  # 堆叠所有无人机，忽略 Z 坐标
# 缺点: 丧失无人机间关系

改动文件: train.py, val.py
修改行数: ~20 行
预计时间: 30 分钟
```

**优点**:
- 快速实现
- 最少代码改动
- 可验证原理可行性

**缺点**:
- 无法建模无人机间协作
- 忽略 Z 坐标信息
- 预测精度可能不理想

---

### 方案 B: 推荐方案 (中等) ⭐⭐⭐

**思路**: 修改模型支持 3D 多智能体轨迹

```python
class MRGTrajSwarm(nn.Module):
    def __init__(self, args):
        # 关键改动:
        # 1. Encoder 输入维度: 2 → num_agents * 3
        # 2. SocialLatentGenerator 输入维度: 4 → num_agents * 3
        # 3. 增加 MultiHeadAttention 捕捉无人机间交互
        
        self.past_encoder = Encoder(
            num_agents * 3,  # ← 改动: 支持多智能体 3D
            d_model, n_layers, n_heads
        )
        
        self.social_latent_generator = SocialLatentGenerator(
            dim_in=num_agents * 3,  # ← 改动: 捕捉集群社交
            ...
        )
```

改动文件: model.py, train.py, val.py
修改行数: ~100 行
预计时间: 2-3 小时

**优点**:
- 完整保留 XYZ 信息
- 建模无人机交互
- 精度更高

**缺点**:
- 需要重新训练
- 超参数需要调优

---

### 方案 C: 最优方案 (复杂) ⭐⭐⭐⭐⭐

**思路**: Graph Neural Network (GNN) + MRGTraj

```python
class MRGTrajGNNSwarm(nn.Module):
    def __init__(self, args):
        # 1. GNN 层: 建模无人机间的图关系
        self.gnn_encoder = GraphNetworkModule(...)
        
        # 2. MRGTraj 解码器: 生成轨迹
        self.mrgraj_decoder = MRGTrajDecoder(...)
        
    def forward(self, traj, adjacency_matrix):
        # 第一步: GNN 提取无人机间的交互特征
        graph_features = self.gnn_encoder(traj, adjacency_matrix)
        
        # 第二步: MRGTraj 预测未来轨迹
        predictions = self.mrgraj_decoder(graph_features)
        
        return predictions
```

改动文件: model.py (新增), train.py, val.py, 数据加载器
修改行数: ~300 行
预计时间: 5-7 小时

**优点**:
- 最先进的架构
- 充分利用集群结构
- 最高精度

**缺点**:
- 实现复杂
- 训练时间长
- 超参数多

---

## 🎯 我的建议

**推荐使用方案 B (中等)**

### 原因:
1. ✅ 平衡复杂度和精度
2. ✅ 充分利用你现有的数据格式
3. ✅ 时间投入合理 (2-3 小时)
4. ✅ 代码可维护性好

### 实现步骤:

```bash
第 1 步: 备份原始文件
  cp MRGTraj-main/model.py MRGTraj-main/model_backup.py

第 2 步: 创建新模型文件
  创建 MRGTrajSwarm.py (支持多智能体 3D)

第 3 步: 修改数据加载
  更新 data_loader.py 处理 (seq, samples, agents, 3) 格式

第 4 步: 训练验证
  python train_swarm.py --num_agents 3 --data_file input_agents_3_subset.npz

第 5 步: 推理测试
  python predict_swarm.py --checkpoint best_model.pth --num_agents 3
```

---

## 📝 数据流对比

### 原始 MRGTraj 流程
```
输入轨迹 (batch, 8, 2)
    ↓
Encoder (处理 2D)
    ↓
SocialLatentGenerator (处理行人社交)
    ↓
TemporalMapper + 噪声采样
    ↓
SocialRefiner
    ↓
预测轨迹 (batch, 12, 2)
```

### 修改后流程 (方案 B)
```
输入轨迹 (batch, 20, num_agents, 3)
    ↓
展平: (batch, 20, num_agents*3)
    ↓
Encoder (处理 3D 多智能体特征)
    ↓
SocialLatentGenerator (捕捉无人机协作)
    ↓
TemporalMapper + 噪声采样
    ↓
SocialRefiner + 多头注意力
    ↓
预测轨迹 (batch, 10, num_agents, 3)
```

---

## 📦 所需文件

需要创建或修改的文件:

| 文件 | 操作 | 说明 |
|------|------|------|
| `model_swarm.py` | 新建 | 改进的 MRGTrajSwarm 模型 |
| `train_swarm.py` | 新建 | 集群训练脚本 |
| `predict_swarm.py` | 新建 | 推理脚本 |
| `data_loader_swarm.py` | 新建 | 无人机数据加载器 |
| `utils_swarm.py` | 修改/新建 | 集群特定的工具函数 |

---

## 🚀 下一步行动

**如果你想继续，请告诉我**:

1. ✅ 是否想要我为你创建方案 B 的完整实现？
2. ✅ 对数据预处理有什么特殊需求？
3. ✅ 是否需要考虑 Z 坐标的特殊处理（如高度约束）？
4. ✅ 训练资源情况（GPU 数量、显存）？

---

## 📚 参考资源

- **原始论文**: MRGTraj (Peng et al., 2023)
- **相关工作**: 无人机集群控制 + 轨迹预测
- **类似研究**: GNNTraj, TrajectoryCascade

