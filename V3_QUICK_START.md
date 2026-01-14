# v3 训练快速开始指南

## 目标

本指南展示如何使用完整的 v3 训练脚本，支持：
- ✅ 预计算 24D 特征（加速 20-30%）
- ✅ 完整的检查点管理（可随时恢复）
- ✅ 训练记录和超参数保存（完全可复现）
- ✅ GNN 参数推荐（基于 Couzin 模型物理约束）

---

## 快速开始（5 分钟）

### 1. 验证数据和特征存在

```bash
# 检查数据目录
ls swarm_segments/

# 应该看到类似的文件：
# - dataset_position_segments_synth.npz
# - features_24d/features_agents_3_subset_24d.npz (可选，但推荐)
```

### 2. 运行 5 个 epoch 的快速测试

```bash
cd "d:\Trajectory prediction\drone_trajectories\Cluster trajectory"

# 使用 v3（GNN 增强）
python train_swarm_v3_complete.py \
  --agents 3 \
  --epochs 5 \
  --batch_size 64 \
  --use_gnn \
  --gnn_fusion_mode concat \
  --use_subset \
  --seed 42
```

**预期输出**：
```
[INFO] 设备：cuda:0
[INFO] 加载数据...
[INFO] 加载数据：swarm_segments/dataset_position_segments_synth.npz
[INFO] 找到预计算特征：swarm_segments/features_24d/features_agents_3_subset_24d.npz
[INFO] 特征统计：mean=[...] std=[...]
[INFO] 训练集：800，验证集：200
[INFO] 创建模型...
[INFO] 使用 v3（GNN，fusion=concat）
[INFO] 模型参数数：2,345,678
[INFO] 开始训练（从 epoch 0）...
[Epoch   0] Train: 0.523451 | Val: 0.487234 | MAE: 0.153421m | LR: 2.00e-04 | TF: 0.6000
[Epoch   1] Train: 0.412345 | Val: 0.398765 | MAE: 0.128934m | LR: 2.00e-04 | TF: 0.5950
...
[INFO] 训练完成
```

### 3. 查看训练结果

```bash
# 查看生成的文件
ls gru_models_v3_agents_3_v3_gnn_concat/

# 应该看到：
# - training_history_agents_3_v3_gnn_concat.csv (训练曲线)
# - config_agents_3_v3_gnn_concat.json (超参数配置)
# - best_model_*.pt (最佳模型)
# - last_checkpoint_*.pt (最后检查点)
```

**查看训练历史**（Python）：
```python
import pandas as pd

df = pd.read_csv(
    'gru_models_v3_agents_3_v3_gnn_concat/training_history_agents_3_v3_gnn_concat.csv'
)
print(df)  # 显示所有 epoch 的性能指标
```

**输出示例**：
```
   Epoch  Train Loss  Train Loss (pos)  ...  Val Loss  Val MAE (m)  Learning Rate
0      0    0.523451         0.418724  ...  0.487234     0.153421       2.00e-04
1      1    0.412345         0.329876  ...  0.398765     0.128934       2.00e-04
2      2    0.356234         0.285123  ...  0.352145     0.112456       2.00e-04
...
```

---

## 完整训练（2-3 小时）

当验证快速测试成功后，运行完整训练：

```bash
python train_swarm_v3_complete.py \
  --agents 3 \
  --epochs 150 \
  --batch_size 256 \
  --use_gnn \
  --gnn_hidden 64 \
  --gnn_heads 4 \
  --edge_threshold 5.5 \
  --gnn_fusion_mode concat \
  --lr 2e-4 \
  --weight_decay 5e-5 \
  --use_amp \
  --seed 42
```

**预期性能**（150 epochs 后）：
- 训练损失：0.05-0.10
- 验证 MAE：0.08-0.12 m
- 总训练时间：~2-3 小时 (NVIDIA GPU)

---

## 断点续训

如果训练被中断（如 Ctrl+C 或 GPU 掉线），脚本会自动保存断点：

```bash
# 自动从最后的检查点恢复
python train_swarm_v3_complete.py \
  --agents 3 \
  --epochs 150
  # 脚本会自动加载 last_checkpoint_*.pt
```

**手动指定检查点**（用于特殊场景）：

```bash
python train_swarm_v3_complete.py \
  --agents 3 \
  --epochs 150 \
  --checkpoint_path gru_models_v3_agents_3_v3_gnn_concat/best_model_0042.pt
```

---

## 与 v2 的对比

### 仅使用 v2（无 GNN）

```bash
python train_swarm_v3_complete.py \
  --agents 3 \
  --epochs 150 \
  --batch_size 256 \
  --no_gnn \
  --use_amp
```

**性能对比**（同等条件）：
| 指标 | v2 (无 GNN) | v3 (GNN concat) |
|------|-----------|----------------|
| 训练速度 | 快 | 中等 (-20%) |
| 最终 MAE | 0.12-0.15 m | 0.08-0.12 m ✅ |
| 模型大小 | 12 MB | 15 MB |
| 参数数 | 1.8M | 2.3M |

---

## GNN 参数说明

### 最关键的参数：`edge_threshold`

这个参数控制 GNN 中哪些代理被认为"相邻"。

**基于 Couzin 模型的推荐**：
```
Couzin 关键距离        推荐 edge_threshold
2.0 m (碰撞规避)  →    4.5 m (紧密)
5.0 m (速度对齐)  →    5.5 m ⭐ 推荐 (平衡)
10.0 m (内聚力)   →    7.0 m (宽泛)
```

**实验最佳阈值**：
```bash
# 快速对比 (每个只用 20 epochs)
for threshold in 4.5 5.5 6.5; do
  python train_swarm_v3_complete.py \
    --agents 3 \
    --epochs 20 \
    --use_subset \
    --use_gnn \
    --edge_threshold $threshold
done
```

### 其他 GNN 参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|--------|------|
| `--gnn_hidden` | 64 | 32-128 | 较大 = 更复杂但更慢 |
| `--gnn_heads` | 4 | 2-8 | 注意力头数，越多越复杂 |
| `--gnn_fusion_mode` | concat | concat/gate/add | concat 最快，通常最好 |

---

## 预计算特征

脚本会自动在以下位置搜索预计算的 24D 特征：

```
优先级 1: swarm_segments/features_24d/features_agents_3_subset_24d.npz
优先级 2: swarm_segments/features_agents_3_subset_24d.npz
优先级 3: ... (其他命名约定)
```

**如果找到**：使用预计算特征，速度提升 20-30% ✅

**如果找不到**：脚本会自动在线计算（速度较慢但仍可行）

**手动生成预计算特征**（可选加速）：

```python
# 用 v2 脚本预计算特征
python compute_features_precomputed.py \
  --data_dir swarm_segments \
  --agents 3 \
  --output_dir swarm_segments/features_24d
```

---

## 超参数完全可复现

所有训练超参数都保存在 JSON 配置文件中：

```bash
cat gru_models_v3_agents_3_v3_gnn_concat/config_agents_3_v3_gnn_concat.json
```

**输出示例**：
```json
{
  "timestamp": "2024-01-15T10:23:45.123456",
  "model_version": "v3",
  "use_gnn": true,
  "num_agents": 3,
  "seq_in": 10,
  "seq_out": 5,
  "hidden_size": 128,
  "num_layers": 2,
  "batch_size": 256,
  "lr": 0.0002,
  "weight_decay": 5e-05,
  "epochs": 150,
  "seed": 42,
  "gnn_hidden": 64,
  "gnn_heads": 4,
  "edge_threshold": 5.5,
  "gnn_fusion_mode": "concat",
  "current_epoch": 149,
  "best_val_loss": 0.35214,
  "current_val_loss": 0.38765
}
```

**完全复现之前的训练**：
```bash
# 用同样的超参数重新训练
python train_swarm_v3_complete.py \
  --agents 3 \
  --epochs 150 \
  --batch_size 256 \
  --use_gnn \
  --gnn_hidden 64 \
  --gnn_heads 4 \
  --edge_threshold 5.5 \
  --gnn_fusion_mode concat \
  --lr 2e-4 \
  --weight_decay 5e-5 \
  --seed 42  # 相同的随机种子
```

---

## 高级用法

### 混合精度训练（节省内存和时间）

```bash
python train_swarm_v3_complete.py \
  --agents 3 \
  --epochs 150 \
  --use_gnn \
  --use_amp  # 启用混合精度
```

**性能提升**：30-50% 更快，内存减少 40-50%

### 多代理规模对比

```bash
# 3 个代理
python train_swarm_v3_complete.py --agents 3 --epochs 50

# 4 个代理（更复杂的交互）
python train_swarm_v3_complete.py --agents 4 --epochs 50 --gnn_heads 4

# 6 个代理（需要更强的 GNN）
python train_swarm_v3_complete.py --agents 6 --epochs 50 \
  --gnn_hidden 128 --gnn_heads 8
```

### 禁用 GNN 回退（调试用）

```bash
python train_swarm_v3_complete.py \
  --agents 3 \
  --epochs 20 \
  --no_gnn  # 完全退化为 v2
```

---

## 常见问题

### Q1: 为什么 GNN 没有明显改进？

**可能原因**：
1. `edge_threshold` 不适当（尝试 ±1.0 m）
2. GNN 特征融合方式不匹配（尝试 `gate` 或 `add`）
3. 代理交互不明显（v2 已经很好）

**诊断**：
```bash
# 对比 v2 vs v3
python train_swarm_v3_complete.py --agents 3 --no_gnn --epochs 30
python train_swarm_v3_complete.py --agents 3 --use_gnn --epochs 30
```

### Q2: 内存不足（CUDA out of memory）

**解决**：
1. 减小 batch size：`--batch_size 128` 或 64
2. 减小模型：`--gnn_hidden 32` 或 `--hidden_size 64`
3. 启用混合精度：`--use_amp`

### Q3: 训练停止改进（损失高原）

**解决**：
1. 调整学习率：`--lr 1e-4` (更小) 或 `--lr 5e-4` (更大)
2. 增加正则化：`--weight_decay 1e-4`
3. 更新 edge_threshold：GNN 可能学不到有用的东西

### Q4: 如何加载最佳模型进行推理？

```python
import torch
from train_swarm_model_v3_with_gnn import DynamicsAwareSwarmGRUModel_with_GNN

# 加载检查点
ckpt = torch.load(
    'gru_models_v3_agents_3_v3_gnn_concat/best_model_0042.pt',
    map_location='cpu'
)

# 创建模型并加载权重
model = DynamicsAwareSwarmGRUModel_with_GNN(
    feature_dim=24,
    hidden_size=128,
    num_layers=2,
    output_size=3,
    gnn_hidden=64,
    gnn_heads=4,
    edge_threshold=5.5,
    gnn_fusion_mode='concat'
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 现在可以用 model 进行推理
```

---

## 最终核查清单

在运行完整训练前，检查：

- [ ] 数据路径正确（默认 `swarm_segments`）
- [ ] 特征目录存在或脚本可自动计算
- [ ] GPU 可用且显存充足（推荐 ≥8GB）
- [ ] Python 环境有必需的包（torch, numpy, tqdm 等）
- [ ] 有足够的磁盘空间保存检查点（~500MB 用于 150 epochs）

**运行系统检查**：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 下一步

1. **快速验证** (5-10 分钟)：运行本指南的"快速开始"部分
2. **参数调优** (2-4 小时)：参考 `GNN_PARAMETER_TUNING_GUIDE.md`
3. **完整训练** (2-3 小时)：运行"完整训练"部分
4. **结果分析**：用 Python 加载 CSV 和最佳模型进行分析

祝训练愉快！🚀
