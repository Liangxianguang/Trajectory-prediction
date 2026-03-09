# 训练脚本快速参考

## 🚀 基础命令

```bash
# 最简单的方式
python train_swarm_detailed.py --num_agents 3 --num_epochs 50

# 指定批次大小
python train_swarm_detailed.py --num_agents 3 --batch_size 64 --num_epochs 100

# 使用完整数据（不是子集）
python train_swarm_detailed.py --num_agents 3 --data_dir "path/to/full/data" --num_epochs 200
```

## 📊 查看指标含义

| 缩写 | 全称 | 说明 | 单位 | 目标 |
|------|------|------|------|------|
| `l2_loss` | L2 Loss | 基础重构损失 | - | ↓ 越小越好 |
| `pos_loss` | Position Loss | XY 平面位置 | m | ↓ 越小越好 |
| `height_loss` | Height Loss | Z 坐标高度 | m | ↓ 越小越好 |
| `vel_loss` | Velocity Loss | 速度平滑性 | m/s | ↓ 越小越好 |
| `acc_loss` | Acceleration Loss | 加速度平滑性 | m/s² | ↓ 越小越好 |
| `collision_loss` | Collision Loss | 碰撞风险 | - | ↓ 越小越好 (0 最佳) |
| `formation_loss` | Formation Loss | 编队稳定性 | - | ↓ 越小越好 |
| `kl_loss` | KL Divergence | 变分正则化 | - | - |
| `ade` | Avg Displacement Error | 平均位移误差 | m | ↓ 越小越好 |
| `fde` | Final Displacement Error | 最终位移误差 | m | ↓ 越小越好 |

## ⚙️ 关键参数

### 必需参数
```bash
--num_agents 3              # 无人机数量 (3/4/5/6)
--num_epochs 50             # 训练轮数
--batch_size 32             # 批处理大小
```

### 重要参数
```bash
--lr 1e-3                   # 学习率
--kl_weight 0.1             # KL 权重 (多样性)
--collision_weight 0.5      # 防碰撞权重
--formation_weight 0.2      # 编队稳定权重
```

### 性能调优参数
```bash
--vel_weight 0.2            # 速度平滑权重
--acc_weight 0.1            # 加速度平滑权重
--pos_weight 0.5            # 位置准确权重
--height_weight 0.3         # 高度准确权重
```

## 🎯 常见配置

### 配置 1：快速验证 (15-20 分钟)
```bash
python train_swarm_detailed.py \
    --num_agents 3 \
    --batch_size 64 \
    --num_epochs 5
```
**用途**: 测试设置是否正确

### 配置 2：标准训练 (1-2 小时)
```bash
python train_swarm_detailed.py \
    --num_agents 3 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 5e-4
```
**用途**: 获得可用的模型

### 配置 3：高质量训练 (3-5 小时)
```bash
python train_swarm_detailed.py \
    --num_agents 3 \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 1e-3 \
    --d_model 512 \
    --n_layers 3
```
**用途**: 最优性能

### 配置 4：安全飞行优先 (强制避碰)
```bash
python train_swarm_detailed.py \
    --num_agents 3 \
    --collision_weight 2.0 \
    --formation_weight 1.0 \
    --vel_weight 0.5 \
    --num_epochs 100
```
**用途**: 最小化碰撞风险

### 配置 5：多样化轨迹
```bash
python train_swarm_detailed.py \
    --num_agents 3 \
    --kl_weight 0.05 \
    --collision_weight 0.3 \
    --num_epochs 100
```
**用途**: 生成多种可能的轨迹

## 📈 实时监控日志

训练时，你会在控制台看到类似的日志：

```
Epoch [5/50]:  30%|███       | 6000/20000 [10:23<24:12, 9.65it/s]
total_loss=5.234567 l2_loss=0.023456 pos_loss=0.018234 height_loss=0.015678
vel_loss=0.012345 acc_loss=0.006789 collision_loss=4.234567 formation_loss=5.123456
kl_loss=0.001234 ade=0.087654 fde=0.098765
```

**快速诊断**:
- `ade` 值大 → 精度低，需要训练更多 epochs
- `collision_loss` 很高 → 增加 `--collision_weight`
- `formation_loss` 很高 → 增加 `--formation_weight`
- `vel_loss` 很高 → 轨迹抖动，增加 `--vel_weight`

## 💾 输出文件位置

训练完成后，模型保存在：
```
checkpoints_swarm/agents_3/
├── best_model.pth              ← 最佳模型
├── checkpoint_epoch_10.pth     ← 第 10 epoch
├── checkpoint_epoch_20.pth     ← 第 20 epoch
└── train_agents_3.log          ← 完整日志
```

## 🔍 查看训练日志

```bash
# 查看最后 50 行
tail -50 checkpoints_swarm/agents_3/train_agents_3.log

# 搜索特定 epoch
grep "Epoch 10" checkpoints_swarm/agents_3/train_agents_3.log

# 查看所有最优模型保存事件
grep "保存最佳模型" checkpoints_swarm/agents_3/train_agents_3.log
```

## 🧪 测试模型

训练完成后进行推理：

```bash
python predict_swarm.py \
    --checkpoint checkpoints_swarm/agents_3/best_model.pth \
    --input_file "Cluster trajectory/swarm_segments/input_agents_3_subset.npz" \
    --output_file predictions.csv \
    --num_samples 20 \
    --save_plot trajectory.png
```

## 🆘 问题排查

| 问题 | 症状 | 解决方案 |
|------|------|--------|
| 显存不足 | OOM error | `--batch_size 16` 或 `--d_model 128` |
| 训练太慢 | 每 batch 时间 > 1s | `--batch_size 64` 或用子集数据 |
| ADE 没有下降 | ADE 一直 > 0.5 | 增加 `--num_epochs 100` 或 `--lr 5e-4` |
| 碰撞风险高 | collision_loss 很高 | `--collision_weight 1.0` |
| 轨迹抖动 | vel_loss 很高 | `--vel_weight 0.5 --acc_weight 0.3` |

## 📚 完整文档

- **详细指南**: `DETAILED_TRAINING_GUIDE.md`
- **快速开始**: `SWARM_QUICKSTART.md`
- **技术报告**: `MRGTraj_UAV_Swarm_Adaptation_Report.md`
- **完整指南**: `MRGTraj_Swarm_Complete_Guide.md`

---

**持续改进中...** ✨
