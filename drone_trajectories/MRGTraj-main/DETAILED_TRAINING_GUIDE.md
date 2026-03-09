# MRGTraj 集群版本 - 详细训练脚本使用指南

## 📊 新增功能

新的 `train_swarm_detailed.py` 脚本提供了**详细的损失函数分解和多维度评估指标**，让你能像下面这样看到完整的训练进度：

```
[Epoch 1/100] total_loss=8.234567 l2_loss=0.034076 pos_loss=0.024361 height_loss=0.027286 
              vel_loss=0.022895 acc_loss=0.009903 collision_loss=6.355933 formation_loss=6.473498 
              kl_loss=0.0024501 ade=0.118368 fde=0.141539
```

## 🎯 支持的损失函数

### 1. **基础损失**
- **`l2_loss`**: L2 重构损失 (预测值 vs 真实值)
- **`kl_loss`**: KL 散度 (变分正则化)

### 2. **位置约束损失**
- **`pos_loss`**: 位置损失 (只关注 XY 平面)
- **`height_loss`**: 高度损失 (Z 坐标)

### 3. **运动约束损失**
- **`vel_loss`**: 速度损失 (鼓励平滑速度变化)
- **`acc_loss`**: 加速度损失 (鼓励平滑加速度变化)

### 4. **集群约束损失**
- **`collision_loss`**: 碰撞约束 (防止无人机碰撞)
- **`formation_loss`**: 编队约束 (保持相对位置稳定)

### 5. **评估指标**
- **`ade`**: 平均位移误差
- **`fde`**: 最终位移误差
- **`val_best_ade`**: 验证集最佳 ADE (多样本选择)
- **`val_best_fde`**: 验证集最佳 FDE (多样本选择)

## 🚀 快速开始

### 基础训练 (默认参数)

```bash
python train_swarm_detailed.py --num_agents 3 --num_epochs 50 --batch_size 32
```

### 自定义损失权重

```bash
# 强调碰撞约束
python train_swarm_detailed.py \
    --num_agents 3 \
    --collision_weight 1.0 \
    --formation_weight 0.5 \
    --num_epochs 100

# 强调平滑运动
python train_swarm_detailed.py \
    --num_agents 3 \
    --vel_weight 0.5 \
    --acc_weight 0.3 \
    --num_epochs 100

# 平衡配置 (推荐)
python train_swarm_detailed.py \
    --num_agents 3 \
    --batch_size 64 \
    --num_epochs 100 \
    --pos_weight 0.5 \
    --height_weight 0.3 \
    --vel_weight 0.2 \
    --acc_weight 0.1 \
    --collision_weight 0.5 \
    --formation_weight 0.2 \
    --kl_weight 0.1
```

## 📊 输出日志示例

训练过程中你会看到类似如下的日志：

```
================================================================================
MRGTraj 集群版本 - 详细训练
================================================================================

配置参数:
  data_dir: swarm_segments
  num_agents: 3
  d_model: 256
  n_heads: 4
  n_layers: 2
  batch_size: 32
  num_epochs: 100
  lr: 0.001
  kl_weight: 0.1
  pos_weight: 0.5
  height_weight: 0.3
  vel_weight: 0.2
  acc_weight: 0.1
  collision_weight: 0.5
  formation_weight: 0.2

================================================================================

正在加载数据...
加载数据: swarm_segments/input_agents_3_subset.npz, swarm_segments/output_agents_3_subset.npz
  X 形状: (20, 230232, 3, 3)
  Y 形状: (10, 230232, 3, 3)
  样本数: 230232
  无人机数: 3
✓ 数据加载成功

创建模型...
✓ 模型创建成功
  总参数数: 3,456,789
  可训练参数: 3,456,789

开始训练...

Epoch [1/100]:   5%|█▌        | 1000/20000 [02:34<49:00, 6.47it/s] 
total_loss=8.234567 l2_loss=0.034076 pos_loss=0.024361 height_loss=0.027286 
vel_loss=0.022895 acc_loss=0.009903 collision_loss=6.355933 formation_loss=6.473498 
kl_loss=0.0024501 ade=0.118368 fde=0.141539

[Epoch 1/100] total_loss=7.923456 l2_loss=0.032156 pos_loss=0.023145 height_loss=0.025634
              vel_loss=0.021234 acc_loss=0.009234 collision_loss=6.234532 formation_loss=6.345234
              kl_loss=0.0023456 ade=0.117234 fde=0.140123
  ✓ 保存最佳模型 (loss: 7.923456)

[Epoch 2/100] total_loss=7.812345 l2_loss=0.031234 pos_loss=0.022456 height_loss=0.024567
              ...
```

## 📈 损失权重说明

### 默认权重配置

| 损失类型 | 默认权重 | 说明 |
|---------|---------|------|
| `kl_weight` | 0.1 | 变分正则化 (越小越多样化) |
| `pos_weight` | 0.5 | XY 平面位置约束 |
| `height_weight` | 0.3 | Z 坐标高度约束 |
| `vel_weight` | 0.2 | 速度平滑约束 |
| `acc_weight` | 0.1 | 加速度平滑约束 |
| `collision_weight` | 0.5 | 防碰撞约束 |
| `formation_weight` | 0.2 | 编队稳定性约束 |

### 调整策略

#### 如果无人机轨迹容易碰撞 ↔️ 增加 `collision_weight`
```bash
python train_swarm_detailed.py --collision_weight 1.0 --formation_weight 0.5
```

#### 如果轨迹抖动 ↔️ 增加运动约束权重
```bash
python train_swarm_detailed.py --vel_weight 0.5 --acc_weight 0.3
```

#### 如果预测多样性不足 ↔️ 减少 `kl_weight`
```bash
python train_swarm_detailed.py --kl_weight 0.01
```

#### 如果高度预测不准确 ↔️ 增加 `height_weight`
```bash
python train_swarm_detailed.py --height_weight 0.8
```

## 🔍 理解每个指标

### L2 损失 (`l2_loss`)
- 衡量预测值与真实值的总体差异
- **范围**: 0 ~ ∞
- **目标**: 尽可能小

### 位置损失 (`pos_loss`)
- 只关注 XY 平面的位置精度
- **用途**: 确保水平位置准确

### 高度损失 (`height_loss`)
- 只关注 Z 坐标的准确性
- **用途**: 确保高度预测准确

### 速度损失 (`vel_loss`)
- 相邻时间步的位移差异
- **低值** = 更平滑的运动

### 加速度损失 (`acc_loss`)
- 相邻速度的差异
- **低值** = 更自然的加速度变化

### 碰撞损失 (`collision_loss`)
- 无人机间距离低于阈值时的惩罚
- **0** = 无碰撞, **>0** = 有碰撞风险

### 编队损失 (`formation_loss`)
- 相对位置的稳定性
- **低值** = 编队保持良好

### ADE (平均位移误差)
- 预测轨迹与真实轨迹的平均误差
- **单位**: 米
- **越小越好**

### FDE (最终位移误差)
- 最后时刻的预测误差
- **单位**: 米
- **越小越好**

## 📁 输出文件结构

```
checkpoints_swarm/
└── agents_3/
    ├── best_model.pth          # 最佳模型
    ├── checkpoint_epoch_10.pth # 第 10 epoch 检查点
    ├── checkpoint_epoch_20.pth # 第 20 epoch 检查点
    ├── train_agents_3.log      # 详细训练日志
    └── logs/                   # TensorBoard 日志 (如果有 TensorBoard)
        ├── events.out.tfevents.xxx
        └── ...
```

## 💡 最佳实践

### 1. 从默认配置开始
```bash
python train_swarm_detailed.py --num_agents 3 --num_epochs 50
```

### 2. 监控日志中的关键指标
- 如果 `collision_loss` 很高 → 增加 `collision_weight`
- 如果 `ade` 没有下降 → 增加 `num_epochs` 或 `lr`
- 如果 `kl_loss` 很高 → 减少 `kl_weight`

### 3. 逐步微调权重
```bash
# 第一步：基础训练
python train_swarm_detailed.py --num_agents 3 --num_epochs 30

# 第二步：如果有特定问题，调整权重
python train_swarm_detailed.py --num_agents 3 --num_epochs 50 \
    --collision_weight 1.0  # 根据需要调整

# 第三步：完整训练
python train_swarm_detailed.py --num_agents 3 --num_epochs 100
```

### 4. 比较不同配置
```bash
# 配置 A：保守配置 (低碰撞风险)
python train_swarm_detailed.py --collision_weight 1.0 --formation_weight 0.5 \
    -o output_conservative.log

# 配置 B：进攻配置 (高多样性)
python train_swarm_detailed.py --kl_weight 0.01 --collision_weight 0.2 \
    -o output_aggressive.log
```

## 🆚 与 `train_swarm.py` 的区别

| 特性 | `train_swarm.py` | `train_swarm_detailed.py` |
|------|-----------------|--------------------------|
| **损失函数数量** | 2 (L2 + KL) | 9 (包含各种约束) |
| **评估指标** | 基础 | 完整的多维度指标 |
| **日志详细程度** | 低 | 高 (逐 batch 显示) |
| **计算量** | 低 | 中等 (多个损失计算) |
| **适用场景** | 快速验证 | 详细分析 |

## 🔧 常见问题

### Q: 所有的权重应该是多少？

**A**: 没有绝对的最优值，取决于你的数据和需求：

- **无人机协作重要** → 提高 `collision_weight`, `formation_weight`
- **平滑性重要** → 提高 `vel_weight`, `acc_weight`
- **精度重要** → 提高 `pos_weight`, `height_weight`
- **多样性重要** → 降低 `kl_weight`

### Q: 为什么 `collision_loss` 一直很高？

**A**: 可能的原因：
1. 权重太低 → 增加 `collision_weight`
2. 最小距离设置太大 → 修改代码中的 `min_distance` 参数
3. 数据本身就包含碰撞 → 检查原始数据

### Q: 如何快速看到效果？

**A**: 使用子集数据 + 较少的 epochs：
```bash
python train_swarm_detailed.py --num_agents 3 --num_epochs 5 --batch_size 64
```

### Q: 可以使用 TensorBoard 查看吗？

**A**: 可以！如果已安装 TensorBoard：
```bash
tensorboard --logdir=checkpoints_swarm/agents_3/logs --port=6006
```

然后在浏览器中访问 `http://localhost:6006`

## 📚 相关脚本

- `train_swarm.py`: 简化版训练脚本
- `predict_swarm.py`: 推理脚本
- `example_swarm.py`: 完整示例
- `data_tools.py`: 数据处理工具

---

**祝训练顺利！** 🚀
