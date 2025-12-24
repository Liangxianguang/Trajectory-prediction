# 🚀 训练快速开始指南

## 简单 MSE 损失函数版本

你的模型现在使用简单的 **Mean Squared Error (MSE)** 损失函数：

$$\text{MSE Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

## 快速开始

### 基础命令

```bash
cd d:\Trajectory\ prediction\drone_trajectories

# 训练 100 个 epoch（快速验证）
python tool/train_model_enhanced.py \
    --data_path combined_segments.npz \
    --output_dir gru_models_mse \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3

# 训练 300 个 epoch（完整训练）
python tool/train_model_enhanced.py \
    --data_path combined_segments.npz \
    --output_dir gru_models_mse \
    --epochs 300 \
    --batch_size 64 \
    --lr 1e-3

# 使用 AMP 加速（快 1.5-2 倍）
python tool/train_model_enhanced.py \
    --data_path combined_segments.npz \
    --output_dir gru_models_mse \
    --epochs 300 \
    --batch_size 64 \
    --lr 1e-3 \
    --use_amp
```

## 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 120 | 训练轮数 |
| `--batch_size` | 64 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--weight_decay` | 1e-5 | L2 正则化 |
| `--grad_clip` | 1.0 | 梯度裁剪 |
| `--patience` | 25 | 早停耐心值 |
| `--use_amp` | False | 自动混合精度 |
| `--teacher_forcing_ratio` | 0.6 | 教师强制初始比例 |

## 输出结果

训练完成后，在 `gru_models_mse/` 目录下会生成：

```
gru_models_mse/
├── enhanced_gru_best.pt              # 最佳模型权重
├── enhanced_gru_norm_stats.npz       # 归一化统计量
├── training.log                       # 训练日志
└── training_curves.png               # 损失曲线图（如果使用 visualize_prediction.py）
```

## 训练日志查看

```bash
# 查看实时日志
tail -f gru_models_mse/training.log

# 查看完整日志
type gru_models_mse/training.log
```

## 模型评估

训练完成后，使用推理脚本评估模型性能：

```bash
python tool/infer_enhanced.py \
    --checkpoint gru_models_mse/enhanced_gru_best.pt \
    --data_path combined_segments.npz \
    --output_dir evaluation_results_mse \
    --reconstruction_method physics_constrained
```

## 常见问题

### Q: 如何加速训练？

```bash
# 方案 1：使用 AMP（自动混合精度）
--use_amp

# 方案 2：增加 batch size（需要更多显存）
--batch_size 128

# 方案 3：减少 epoch（如果已经收敛）
--epochs 200 --patience 20
```

### Q: 训练不稳定？

```bash
# 降低学习率
--lr 5e-4

# 增加权重衰减
--weight_decay 1e-4

# 增加梯度裁剪值
--grad_clip 2.0
```

### Q: 如何看出模型是否收敛？

查看 `training.log` 中的验证损失（Val Loss）：
- ✅ 验证损失逐步下降 → 模型在改进
- ⚠️ 验证损失停止下降 20+ epoch → 可以早停
- ❌ 验证损失上升 → 可能过拟合，考虑增加 dropout 或早停

### Q: 显存不足？

```bash
# 减小 batch size
--batch_size 32

# 减小隐藏维度
--hidden_dim 64

# 关闭 AMP
# (移除 --use_amp)
```

## 推荐配置

### 配置 1：快速测试（5 分钟）
```bash
python train_model_enhanced.py \
    --data_path combined_segments.npz \
    --output_dir test_run \
    --epochs 20 \
    --batch_size 128
```

### 配置 2：标准训练（30 分钟）
```bash
python train_model_enhanced.py \
    --data_path combined_segments.npz \
    --output_dir gru_models_mse \
    --epochs 300 \
    --batch_size 64 \
    --lr 1e-3 \
    --use_amp
```

### 配置 3：高精度训练（1 小时）
```bash
python train_model_enhanced.py \
    --data_path combined_segments.npz \
    --output_dir gru_models_mse_final \
    --epochs 400 \
    --batch_size 32 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --use_amp
```

## 损失函数说明

详见 `LOSS_FUNCTION_EXPLANATION.md`

**核心思想**：通过最小化预测和真实轨迹之间的平方误差，使模型学习准确的轨迹预测。

---

**准备好训练了吗？🚀**
