# 🚨 核心问题诊断报告

## 概述
你观察到的"预测每次走几十米、MAE 很大"的现象由 **2 个关键问题** 引起：

---

## ⭐ 问题 1：反归一化用错了统计量（最严重）

### 位置
`validate_enhanced_model.py` 第 177-180 行（已修复）

### 原错误代码
```python
out_mean = self.input_mean.cpu().numpy()  # ❌ 这是 input 的统计量
out_std = self.input_std.cpu().numpy()
delta_denorm = delta_norm * (out_std + 1e-8) + out_mean
```

### 问题原因
- **训练时**：输出目标是 `delta = output_pos - last_input_pos`（增量），用 `output_mean/output_std` 归一化
- **推理时**：应该用同样的 `output_mean/output_std` 来反归一化
- **但代码错了**：用了 `input_mean/input_std`（≈[0.45, 0.57, 11.95] 和 ≈[22.95, 22.89, 4.44]）

### 放大倍数
```
错误 output_std = input_std ≈ 22.9
正确 output_std = delta_std ≈ 0.1-0.5  (取决于实际数据)
错误倍数 = 22.9 / 0.3 ≈ 76 倍 ！
```

这解释了为什么你看到每步预测走几十米：
- 模型在归一化空间预测很小的值（~0.1）
- 乘以错误的 `output_std` ≈ 22.9 → ~2.3（已经太大）
- 再加上可能的累积 cumsum → 最终 MAE ~ 100+ 米

### 修复方案（已完成）
```python
# ✓ 正确：从 stats.npz 加载 output_mean/output_std
self.output_mean = torch.tensor(stats.get('output_mean', stats['input_mean']), ...)
self.output_std = torch.tensor(stats.get('output_std', stats['input_std']), ...)

# 然后在 infer_enhanced 中使用
out_mean = self.output_mean.cpu().numpy()
out_std = self.output_std.cpu().numpy()
delta_denorm = delta_norm * (out_std + 1e-8) + out_mean
```

---

## ⭐ 问题 2：stats.npz 中的 output_mean/output_std 可能是旧的

### 情况
虽然 `train_model_enhanced.py` 已修复为正确计算 delta 统计量，但：
- **旧 checkpoint**（修复前生成）对应的 `stats.npz` 中 `output_mean/output_std` 等于 `input_mean/input_std`
- 这是因为旧代码有 bug：`output_mean = stats.get('output_mean', input_mean)`

### 检测方法
运行修复后的调试脚本：
```bash
cd /d "D:\Trajectory prediction"
python tool_debug_infer_inspect.py
```

查看输出中的关键行：
```
⚠️  关键检查：output_mean/output_std 是否正确？
  stats output_mean: [...]
  stats output_std: [...]
  dataset delta_mean: [...]
  dataset delta_std: [...]
  ❌ 错误！output stats 等于 input stats，说明 stats.npz 是旧的
```

### 修复方案
**必须重新训练模型**（短跑训练即可验证）：
```bash
cd /d "D:\Trajectory prediction\drone_trajectories"
python tool\train_model_enhanced.py ^
  --epochs 5 ^
  --batch_size 32 ^
  --data_path dataset_position_segments_synth.npz ^
  --output_dir tool\gru_models_enhanced ^
  --model_name enhanced_gru_model ^
  --hidden_dim 128 ^
  --num_layers 3
```

这会生成新的 `enhanced_gru_model_best_model.pth` 和 `enhanced_gru_model_norm_stats.npz`（正确的统计量）。

---

## ✅ 问题 3：特征对齐（已修复）

### 位置
`validate_enhanced_model.py` 第 340-348 行

### 原错误
```python
inp_pos = trajectory[-30:-10, :]         # 取倒数 30-10，共 20 个
features = self.compute_input_features(trajectory[-30:], input_length=20)
# compute_input_features 会取 trajectory[-30:] 的最后 20 个
# 即 trajectory[-20:]，与 inp_pos 不同步！
```

### 修复后
```python
inp_pos = trajectory[-30:-10, :]
true_future = trajectory[-10:, :]
# 直接用 inp_pos + true_future 拼成的 30 个点来计算特征
features = self.compute_input_features(np.vstack([inp_pos, true_future]), input_length=20)
```

---

## 🔍 验证步骤（按顺序）

### 步骤 1：运行诊断脚本
```bash
cd /d "D:\Trajectory prediction"
python tool_debug_infer_inspect.py
```

**关键检查项**：
- ✓ `output_std` 是否等于 `input_std`（如果是，说明 stats.npz 是旧的）
- ✓ 反归一化后每步增量是否 < 0.5 m（如果是，问题解决）
- ✓ 最后一行 MAE 是否显著下降

### 步骤 2：如果诊断显示 output_stats 错误，重新训练
```bash
cd /d "D:\Trajectory prediction\drone_trajectories"
python tool\train_model_enhanced.py --epochs 5 --batch_size 32 --data_path dataset_position_segments_synth.npz --output_dir tool\gru_models_enhanced --model_name enhanced_gru_model --hidden_dim 128 --num_layers 3
```

### 步骤 3：运行修复后的验证
```bash
cd /d "D:\Trajectory prediction\drone_trajectories\tool"
python validate_enhanced_model.py --num_trajectories 3 --device cuda
```

**预期结果**：
- MAE 应该下降到 **< 1 m**（每步误差可接受）
- 或至少相比之前的 100+ m 有明显改善

---

## 📊 影响分析

| 问题 | 症状 | 影响 | 修复后 |
|------|------|------|--------|
| 反归一化错误 | 每步预测 10-100 m | 关键 | 每步 < 0.5 m |
| stats.npz 旧值 | 依赖上一个问题 | 关键 | 需重训 |
| 特征对齐 | 输入特征偶发不对齐 | 中等 | 特征一致 |

---

## 📝 代码改动汇总

### 文件：`validate_enhanced_model.py`

#### 改动 1：加载 output stats
```python
# 第 49 行
self.output_mean = torch.tensor(stats.get('output_mean', stats['input_mean']), ...)
self.output_std = torch.tensor(stats.get('output_std', stats['input_std']), ...)
```

#### 改动 2：使用 output stats 反归一化
```python
# 第 176-180 行
out_mean = self.output_mean.cpu().numpy()  # ✓ 改为 output_mean
out_std = self.output_std.cpu().numpy()    # ✓ 改为 output_std
delta_denorm = delta_norm * (out_std + 1e-8) + out_mean
```

#### 改动 3：特征对齐
```python
# 第 348 行
features = self.compute_input_features(np.vstack([inp_pos, true_future]), input_length=20)
```

### 文件：`tool_debug_infer_inspect.py`

增加了详细的诊断输出，帮助判断 stats.npz 是否正确。

---

## ⏭️ 后续行动

1. **立即**：运行修复后的诊断脚本，确认问题
2. **如果需要**：重新训练（5 epochs smoke-run 即可）
3. **验证**：再运行 validate_enhanced_model.py 检查 MAE 改善

预期时间：
- 诊断：2-3 分钟
- 重训：5-10 分钟（5 epochs）
- 验证：3-5 分钟

---

**问题根源**：训练/推理流水线中的归一化/反归一化不一致 + 旧的统计量缓存
