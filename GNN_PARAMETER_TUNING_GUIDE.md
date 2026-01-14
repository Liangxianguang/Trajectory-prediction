# GNN 参数调优指南

## 概述

v3 模型通过图注意力网络 (GAT) 显式建模代理间的交互关系。本指南基于 **Couzin 集群模型** 的物理约束，提供参数推荐。

---

## 1. 核心参数解析

### 1.1 `edge_threshold` - 邻接阈值

**物理含义**：GNN 在构建图时，将距离小于阈值的代理连接为邻边。

**与 Couzin 模型的关系**：
```
Couzin 模型的关键距离：
├─ Repulsion distance    = 2.0 m   (碰撞规避)
├─ Orientation distance  = 5.0 m   (速度对齐)
└─ Attraction distance   = 10.0 m  (内聚力)
```

**推荐值**：
- **5.0-5.5 m** ⭐ **默认推荐**
  - 捕获 Couzin 的速度对齐 (orientation) 和部分碰撞规避规则
  - 适合 3-6 个代理的小规模群体
  - 平衡表达能力和计算效率
  
- **4.5 m** - 更紧密的局部结构
  - 仅关注碰撞规避和部分对齐
  - 适合高密度群体
  
- **6.0-7.0 m** - 更宽泛的交互
  - 包含内聚力影响
  - 适合稀疏群体或远距离协调

**实验建议**：
```bash
# 测试三个候选值
for threshold in 4.5 5.5 7.0; do
  python train_swarm_v3_complete.py \
    --agents 3 --epochs 50 \
    --use_gnn --edge_threshold $threshold \
    --gnn_fusion_mode concat
done
```

---

### 1.2 `gnn_hidden` - GNN 隐藏维度

**物理含义**：每个 GAT head 的内部表征空间。

**计算关系**：
- 单 head 输出维度 = `gnn_hidden`
- 总输出 = `gnn_hidden × gnn_heads`

**推荐值**：
```
数据集规模          推荐值        备注
小 (<1K样本)       32-64        快速迭代、防过拟合
中 (1K-10K)        64 ⭐ 推荐    通用平衡点
大 (>10K)          128-256      更复杂的交互建模
```

**对训练的影响**：
| 维度 | 速度 | 表达能力 | 内存 | 推荐场景 |
|------|------|--------|------|--------|
| 32 | 快 | 弱 | 低 | 快速原型 |
| 64 | 中 | 中 | 中 | **通用** |
| 128 | 慢 | 强 | 高 | 大规模数据 |
| 256+ | 很慢 | 很强 | 很高 | 生产环保 |

**实验建议**：
```bash
python train_swarm_v3_complete.py \
  --agents 3 --epochs 20 --gnn_hidden 64 \
  --use_gnn --edge_threshold 5.5
```

---

### 1.3 `gnn_heads` - 注意力头数

**物理含义**：从多个角度学习邻接重要性。

**计算关系**：
- 总处理维度 = `feature_dim + (gnn_hidden × gnn_heads)`
- 必须满足：`feature_dim % gnn_heads == 0`（某些配置）

**推荐值**：
```
代理数量    推荐 heads   理由
3-4        2-4         小群体，多角度可能冗余
5-6        4-8         中等群体，多视角有益
8+         8-16        大群体，复杂交互需要多头
```

**特殊案例**：
- **heads=1** 退化为单注意力机制，速度快但表达有限
- **heads=4** ⭐ **标准选择**，在速度和效果间达成平衡
- **heads=8** 更好的多角度表示，但对小数据集可能过度参数化

**实验建议**：
```bash
for heads in 2 4 8; do
  python train_swarm_v3_complete.py \
    --agents 3 --epochs 30 \
    --use_gnn --gnn_heads $heads \
    --gnn_hidden 64
done
```

---

### 1.4 `gnn_fusion_mode` - 特征融合方式

**三种融合策略对比**：

#### concat 模式 (连接) ⭐ **推荐**
```python
# 输入：features (24D) + gnn_output (gnn_hidden×heads)
# 操作：直接拼接
output = [features, gnn_features]  # 维度 → 24 + 256 = 280D (示例)
```
- **优点**：最直接，保留所有信息，实现最简单
- **缺点**：输出维度增加，GRU 输入变大
- **适用**：默认选择，特别是当计算资源充足时

#### gate 模式 (门控加权)
```python
# 操作：学习标量权重混合两个特征
gate = sigmoid(FC(gnn_features))  # 标量，范围 [0,1]
output = features * gate + gnn_features * (1 - gate)
```
- **优点**：参数最少，融合更灵活，保持原始维度 (24D)
- **缺点**：可能损失信息，需要学习更复杂的权重函数
- **适用**：内存受限、想要保持维度时

#### add 模式 (投影求和)
```python
# 操作：投影后相加
gnn_proj = FC(gnn_features)  # → 24D
output = features + gnn_proj
```
- **优点**：维度保持，参数适中，几何意义清晰
- **缺点**：过度参数化（投影矩阵较大），融合信息有限
- **适用**：需要保持维度、有足够内存时

**实验对比**：
```bash
for mode in concat gate add; do
  python train_swarm_v3_complete.py \
    --agents 3 --epochs 50 \
    --use_gnn --gnn_fusion_mode $mode \
    --gnn_hidden 64 --gnn_heads 4
done
```

**融合模式选择矩阵**：
| 场景 | concat | gate | add |
|------|--------|------|-----|
| 速度优先 | ✓ | ⭐ 最快 | - |
| 精度优先 | ⭐ 最好 | 一般 | 一般 |
| 内存受限 | - | ⭐ 推荐 | 一般 |
| 初次尝试 | ⭐ 推荐 | - | - |

---

## 2. 根据问题规模的推荐配置

### 2.1 快速原型 (5-10 epoch 验证)
```bash
python train_swarm_v3_complete.py \
  --agents 3 --epochs 10 --batch_size 64 \
  --use_gnn --gnn_hidden 32 --gnn_heads 2 \
  --edge_threshold 5.5 --gnn_fusion_mode concat \
  --use_subset  # 仅用1000样本
```

### 2.2 标准训练 (生产就绪)
```bash
python train_swarm_v3_complete.py \
  --agents 3 --epochs 150 --batch_size 256 \
  --use_gnn --gnn_hidden 64 --gnn_heads 4 \
  --edge_threshold 5.5 --gnn_fusion_mode concat \
  --lr 2e-4 --seed 42 --use_amp
```
**预期性能**：
- 训练时间：~10-15 分钟/epoch (GPU)
- MAE 收敛到：0.10-0.15 m

### 2.3 高精度训练 (需要最优结果)
```bash
python train_swarm_v3_complete.py \
  --agents 3 --epochs 200 --batch_size 128 \
  --use_gnn --gnn_hidden 128 --gnn_heads 8 \
  --edge_threshold 5.5 --gnn_fusion_mode gate \
  --lr 1e-4 --weight_decay 1e-4 --use_amp
```
**预期性能**：
- 训练时间：~15-20 分钟/epoch
- MAE 收敛到：0.08-0.12 m

---

## 3. 诊断和调优策略

### 3.1 问题：训练损失停滞

**可能原因和解决方案**：
| 症状 | 原因 | 解决方案 |
|------|------|--------|
| 损失在早期就高原 | GNN 融合不当 | 尝试 `gate` 或 `add` 模式 |
| 梯度消失 | 网络过深 | 减少 `--num_layers` 或 `--gnn_hidden` |
| 损失振荡 | 学习率太高 | 降低 `--lr`（默认 2e-4） |
| GNN 无贡献 | edge_threshold 不适当 | 调整 threshold ±1.0 m |

### 3.2 问题：验证集过拟合

**症状**：训练损失继续下降，验证损失反弹。

**解决方案**：
```bash
# 增加正则化
python train_swarm_v3_complete.py \
  --weight_decay 1e-4  # 原默认 5e-5，翻倍
  
# 或减小模型复杂度
--gnn_hidden 32  # 从 64 → 32
--gnn_heads 2    # 从 4 → 2
```

### 3.3 问题：训练速度过慢

**诊断**：
```bash
# 比较 v2 (无 GNN) vs v3 (有 GNN) 的速度
python train_swarm_v3_complete.py --agents 3 --epochs 3 --no_gnn
python train_swarm_v3_complete.py --agents 3 --epochs 3 --use_gnn
```

**加速方案**：
1. 减小 GNN 复杂度：
   ```bash
   --gnn_hidden 32 --gnn_heads 2  # 而非 64, 4
   ```
2. 启用混合精度：
   ```bash
   --use_amp  # 通常可加速 30-50%
   ```
3. 预计算特征（已自动）：
   - 脚本会自动搜索 `features_24d/` 目录的预计算特征
   - 预计算可节省 20-30% 训练时间

---

## 4. 不同代理数量的配置

### 4.1 代理数=3 (Couzin 文献标准)
```bash
python train_swarm_v3_complete.py \
  --agents 3 \
  --gnn_hidden 64 --gnn_heads 4 \
  --edge_threshold 5.5 \
  --gnn_fusion_mode concat
```

### 4.2 代理数=4-6 (小群)
```bash
python train_swarm_v3_complete.py \
  --agents 4 \
  --gnn_hidden 64 --gnn_heads 4 \
  --edge_threshold 6.0  # 稍宽泛的邻接
  --gnn_fusion_mode concat
```

### 4.3 代理数=8+ (中等群)
```bash
python train_swarm_v3_complete.py \
  --agents 8 \
  --gnn_hidden 128 --gnn_heads 8  # 更复杂的交互
  --edge_threshold 6.5 \
  --gnn_fusion_mode gate  # 节省内存
```

---

## 5. 完整调优工作流

### 第1步：快速扫描 (1-2小时)
找出最佳的 `edge_threshold` 和 `gnn_fusion_mode`。

```bash
for threshold in 4.5 5.5 6.5; do
  for mode in concat gate add; do
    python train_swarm_v3_complete.py \
      --agents 3 --epochs 20 \
      --edge_threshold $threshold \
      --gnn_fusion_mode $mode \
      --use_subset  # 快速验证
  done
done
```

### 第2步：精细调优 (4-8小时)
基于第1步的最佳组合，微调 `gnn_hidden` 和 `gnn_heads`。

```bash
for hidden in 32 64 128; do
  for heads in 2 4 8; do
    python train_swarm_v3_complete.py \
      --agents 3 --epochs 50 \
      --gnn_hidden $hidden \
      --gnn_heads $heads \
      --edge_threshold 5.5 \
      --gnn_fusion_mode concat  # 使用第1步的最佳值
  done
done
```

### 第3步：长期训练 (12-24小时)
用最优参数进行完整训练。

```bash
python train_swarm_v3_complete.py \
  --agents 3 --epochs 150 \
  --gnn_hidden 64 --gnn_heads 4 \
  --edge_threshold 5.5 \
  --gnn_fusion_mode concat \
  --use_amp --seed 42
```

---

## 6. 验证和保存

训练脚本会在 `gru_models_v3_agents_3_v3_gnn_concat/` 目录中自动保存：

```
├── training_history_agents_3_v3_gnn_concat.csv  # 每个epoch的性能指标
├── config_agents_3_v3_gnn_concat.json           # 超参数配置（可复现）
├── best_model_0042.pt                           # 最佳验证损失的模型
├── last_checkpoint_0149.pt                      # 最后一个epoch的完整状态
└── interrupted_checkpoint_0095.pt               # (如果被中断) 断点恢复
```

### 读取训练历史
```python
import pandas as pd
df = pd.read_csv('gru_models_v3_agents_3_v3_gnn_concat/training_history_agents_3_v3_gnn_concat.csv')
print(df[['Epoch', 'Val Loss', 'Val MAE (m)']].tail(10))
```

### 加载最佳模型进行推理
```python
import torch
ckpt = torch.load('gru_models_v3_agents_3_v3_gnn_concat/best_model_0042.pt')
model.load_state_dict(ckpt['model_state_dict'])
# 现在可以用 model 进行推理
```

---

## 7. 故障排查

| 问题 | 日志信息 | 解决方案 |
|------|--------|--------|
| CUDA 内存不足 | `RuntimeError: CUDA out of memory` | 减小 `--batch_size`，或使用 `--gnn_fusion_mode gate` |
| 数据找不到 | `FileNotFoundError: 未找到数据文件` | 检查 `--data_dir`，默认 `swarm_segments` |
| 预计算特征失配 | 输出形状不匹配 | 确保预计算特征与数据同长度，脚本自动处理 |
| 模型参数数太多 | `模型参数数：5,234,560` | 减小 `--hidden_size` 或 `--gnn_hidden` |

---

## 推荐起点

**对于首次使用 v3**，直接运行：

```bash
python train_swarm_v3_complete.py \
  --agents 3 --epochs 150 --batch_size 256 \
  --use_gnn --gnn_hidden 64 --gnn_heads 4 \
  --edge_threshold 5.5 --gnn_fusion_mode concat \
  --use_amp --seed 42
```

这个配置基于：
- ✅ Couzin 模型的物理约束 (edge_threshold=5.5m)
- ✅ 小群体的通用均衡点 (gnn_hidden=64, heads=4, concat)
- ✅ 已验证的训练稳定性和效果

**预期结果** (3个代理，150个epoch)：
- 最终 MAE：0.10-0.15 m
- 训练时间：~2-3 小时 (NVIDIA V100 或更好)
- 模型大小：~15-20 MB
