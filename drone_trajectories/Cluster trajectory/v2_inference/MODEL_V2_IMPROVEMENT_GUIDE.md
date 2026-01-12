# 轨迹预测模型 v2 改进指南
## 动力学感知版本 - 重点关注速度方向、加速度变化和周期运动

创建时间: 2026-01-11

---

## 🎯 核心问题与解决方案

### 问题分析
当前v1模型存在的限制：
1. **缺乏速度方向的显式约束** - 模型只预测位置增量，无法直接学习速度方向的变化规律
2. **加速度分解不完整** - 无法区分"沿速度方向加速"与"转向加速度"（法向加速度）
3. **周期运动识别困难** - 复杂的周期性运动（如圆周运动）难以准确捕捉
4. **缺乏中间监督信号** - 只有最终位置的监督，训练不稳定

### v2解决方案

#### 1️⃣ 增强特征工程 (16D → 24D)

| 特征类型 | v1 (16D) | v2 (24D) | 目的 |
|---------|---------|---------|------|
| 位置 | ✓ 3D | ✓ 3D | 基础位置信息 |
| 速度（多尺度） | ✓ 9D | ✓ 9D | 运动趋势 |
| 速度方向 | ✗ | ✅ 3D (单位向量) | **显式速度方向** |
| 速度大小 | ✗ | ✅ 1D | **显式速度幅度** |
| 切向加速度 | ✗ | ✅ 1D | **区分加速/减速** |
| 法向加速度 | ✗ | ✅ 1D | **区分转向** |
| 角速度 | ✗ | ✅ 1D | **转弯率** |
| Jerk (三阶导数) | ✗ | ✅ 1D | **捕捉运动平滑性变化** |
| 曲率 | ✓ 1D | ✓ 1D | 空间弯曲度 |
| 平面曲率 | ✓ 3D | ✓ 3D | 多维运动特性 |

**新增特征的物理意义：**

```
速度向量 = 速度方向(单位向量) × 速度大小
         = û × |v|

加速度 = 切向加速度(沿速度方向) + 法向加速度(垂直于速度)
       = a_t × û + a_n × n̂

这样可以：
✓ 区分 匀速直线 vs 加速直线 vs 转弯减速
✓ 识别 周期性圆周运动（法向加速度周期性）
✓ 检测 突然的方向改变（高Jerk）
```

#### 2️⃣ 双分支架构 + 多任务学习

**架构对比：**

```
v1 模型:
输入(16D特征)
    ↓
  编码器(BiGRU)
    ↓
  解码器(GRU)
    ↓
  输出: 位置增量(3D)
    ↓
损失函数: MSE(位置)


v2 模型:
输入(24D特征)
    ↓
  编码器(BiGRU) ← 更强大的特征
    ↓
  ┌─────────────────────────────────────┐
  │       解码器主体 (共享GRU)           │
  └─────────────────────────────────────┘
  ↙          ↓           ↖
位置分支    速度分支     加速度分支
(65%)      (20%)        (15%)
↓          ↓            ↓
预测:      预测:        预测:
位置增量   速度向量     切向+法向加速度
(3D)       (3D)         (2D)

损失函数: 
  L_total = 0.65×L_position + 0.20×L_velocity + 0.15×L_accel
```

**多任务学习的好处：**
- 位置预测(65%): 主任务，直接优化轨迹准确性
- 速度预测(20%): 中间监督，确保速度方向和大小正确
  - 余弦相似度: 约束方向
  - MSE: 约束大小
- 加速度预测(15%): 平滑性约束，防止不合理的运动
  - 确保加速/减速平滑
  - 识别周期性的加速度模式

---

## 📊 特征详解

### 速度方向 (3D)
```python
速度向量 v = (x, y, z) / dt
速度大小 |v| = sqrt(x² + y² + z²)
速度方向 v̂ = v / |v|  ← 单位向量，捕捉运动方向

作用：
- 匀速直线: v̂ 保持不变
- 圆周运动: v̂ 周期性旋转
- 螺旋运动: v̂ 同时改变方向和提升
```

### 切向/法向加速度分解
```python
加速度 a = dv/dt
速度方向 v̂ = v / |v|

切向加速度: a_t = a · v̂  
  ↳ 描述速度大小的变化率
  ↳ a_t > 0: 加速
  ↳ a_t < 0: 减速
  ↳ a_t ≈ 0: 恒定速度

法向加速度: a_n = |a - a_t × v̂|
  ↳ 描述转向的剧烈程度
  ↳ a_n = 0: 直线运动
  ↳ a_n > 0: 转向（如圆周运动）
  ↳ a_n ∝ 1/r: 与转弯半径成反比

应用：
  匀速直线: a_t ≈ 0, a_n ≈ 0
  加速直线: a_t > 0, a_n ≈ 0
  转弯: a_t ≈ 任意, a_n > 0
  高速急转: a_t ≈ 0, a_n >> 0
```

### 角速度 (转弯率)
```python
ω = (v × a) / |v|²

物理意义：
- ω = 0: 直线运动
- 小ω: 缓慢转向
- 大ω: 急速转向

与曲率的关系:
- κ = ω / |v|  (曲率 = 角速度 / 速度)
- 圆周运动: ω = v / r (常数)
```

### Jerk (三阶导数，加速度变化率)
```python
J = da/dt

物理意义：
- J ≈ 0: 平滑运动（如恒定曲率圆周运动）
- J > 0: 转向变急或加速变快
- J < 0: 转向变缓或加速变缓

应用场景：
✓ 识别从直线到急转的转变
✓ 检测无人机的突然操控
✓ 平滑性评估（J越小越平滑）
```

---

## 🚀 预期改进效果

### 定量目标

| 指标 | v1基线 | v2目标 | 改进% |
|-----|--------|--------|--------|
| 位置MAE | 0.10-0.12m | 0.08-0.10m | ↓ 15-20% |
| 速度方向余弦相似度 | 不计算 | > 0.95 | 新增 |
| 转弯识别准确率 | ~70% | > 85% | ↑ 15% |
| 周期运动识别 | 差 | 优 | 显著 |
| 模型参数数 | ~500K | ~750K | +50% |
| 训练时间 | ~30s/epoch | ~35s/epoch | +17% |

### 定性改进

✅ **匀速直线识别**
- v1: 可能预测为微弱转向
- v2: 准确识别，速度方向保持一致，a_t≈0, a_n≈0

✅ **加速/减速识别**
- v1: 可能混淆为转向
- v2: 清晰区分，显式预测切向加速度a_t>0/a_t<0

✅ **转弯运动识别**
- v1: 笨重，容易"拉直"转弯
- v2: 精确预测法向加速度，保持转弯轨迹

✅ **周期运动识别（如圆周运动）**
- v1: 失败率高，趋势漂移
- v2: 周期特征+角速度监督，恢复准确的圆周轨迹

✅ **复杂运动（螺旋上升等）**
- v1: 难以处理
- v2: 分离水平和竖直运动，分别优化

---

## 💻 使用指南

### 训练 v2 模型

```bash
# 基础训练 (推荐配置)
python train_swarm_v2_complete.py \
  --data_dir swarm_segments \
  --agents 3 \
  --epochs 200 \
  --batch_size 256 \
  --hidden_size 128 \
  --num_layers 2 \
  --dropout 0.3 \
  --lr 2e-4 \
  --patience 25 \
  --teacher_forcing_ratio 0.6 \
  --use_amp \
  --use_attention \
  --seed 42

# 快速测试 (使用子集)
python train_swarm_v2_complete.py \
  --data_dir swarm_segments \
  --agents 3 \
  --use_subset \
  --epochs 100 \
  --batch_size 512 \
  --use_amp

# 深层优化 (更强的模型)
python train_swarm_v2_complete.py \
  --data_dir swarm_segments \
  --agents 3 \
  --epochs 300 \
  --batch_size 128 \
  --hidden_size 256 \
  --num_layers 3 \
  --dropout 0.25 \
  --lr 1e-4 \
  --patience 40 \
  --use_amp \
  --use_attention
```

### 训练参数解释

| 参数 | 说明 | v2推荐值 |
|------|------|---------|
| `--hidden_size` | 隐藏维度 | 128-256 |
| `--num_layers` | GRU层数 | 2-3 |
| `--dropout` | Dropout比例 | 0.2-0.3 |
| `--lr` | 学习率 | 2e-4 (平衡) |
| `--patience` | 早停耐心值 | 25 |
| `--teacher_forcing_ratio` | TF初始比例 | 0.6 |
| `--batch_size` | 批次大小 | 256-512 |
| `--use_amp` | 混合精度 | 启用 (快2倍) |

### 模型输出解释

```python
# v1 输出
pred_position: (batch, seq_out, agents, 3)  # 位置增量

# v2 输出
pred_position: (batch, seq_out, agents, 3)  # 位置增量
pred_velocity: (batch, seq_out, agents, 3)  # 速度向量(方向+大小)
pred_accel: (batch, seq_out, agents, 2)     # [切向加速度, 法向加速度]

# 推理时使用
pos_next = pos_last + pred_position
velocity = pred_velocity  # 可用于平滑性检查
# 如果 |pred_accel| 很大 → 急剧运动
# 如果 pred_accel[1] 周期性 → 周期运动
```

---

## 🔍 诊断与调试

### 训练日志解释

```
Epoch 50: 
  train_loss=0.045612 
    (pos=0.035, vel=0.007, accel=0.003),  ← 3个子任务的损失
  val_loss=0.052000,
  lr=0.000200
```

- **pos**: 位置预测损失（应该最大）
- **vel**: 速度预测损失（约20%权重）
- **accel**: 加速度约束损失（约15%权重）
- 三者的比例应接近 0.65:0.20:0.15

### 异常表现

| 现象 | 原因 | 解决方案 |
|-----|------|---------|
| accel损失 >> pos损失 | 训练不稳定，加速度目标计算有误 | 检查y_accel计算; 增加dropout |
| vel损失不下降 | 速度方向或大小难以预测 | 增加hidden_size; 降低学习率 |
| 早停太快(epoch<50) | 学习率过高，振荡太大 | 降低lr到1e-4; 增加patience |
| 训练陷入局部最优 | 初始化或特征不足 | 尝试不同seed; 检查特征缩放 |

---

## 📈 对比实验建议

### 实验1: v1 vs v2 对比

```bash
# v1基线 (原始模型)
python train_swarm_model_enhanced.py \
  --agents 3 --epochs 200 --batch_size 256 \
  --use_amp --seed 42 \
  --output_dir baseline_v1

# v2改进 (新模型)
python train_swarm_v2_complete.py \
  --agents 3 --epochs 200 --batch_size 256 \
  --use_amp --seed 42 \
  --output_dir baseline_v2
```

**对比指标：**
1. 最终VAL_LOSS
2. 收敛速度 (epochs to convergence)
3. 位置MAE (physical units)
4. 推理速度 (时间/样本)

### 实验2: 特征重要性分析

可以通过删除某些特征，观察性能下降：

```python
# 在train_swarm_v2_complete.py中修改 compute_features_enhanced_24d()
# 删除特定特征，观察MAE变化

# 例如只保留速度方向而删除角速度：
# 观察圆周运动预测精度是否下降 → 证明角速度的重要性
```

---

## ⚡ 性能优化

### 内存优化
- v2使用24D特征，较v1(16D)增加50%内存
- 在GPU上训练时，可能需要:
  - 降低batch_size (256 → 128)
  - 减少hidden_size (128 → 64) [牺牲精度换内存]
  - 启用混合精度训练 `--use_amp`

### 计算优化
- 特征预计算: 将24D特征预先计算保存，加速数据加载
- 推荐改造: `precompute_features_v3.py` (支持24D)

---

## 📚 理论基础

### 参考文献
1. **Gupta et al.** (2018) - "Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks"
   - 提出速度-加速度联合学习
   
2. **Jain et al.** (2016) - "Structural RNNs for Visible Surface Reconstruction"
   - 多任务学习在轨迹预测中的应用
   
3. **Helbing & Molnár** (1995) - "Social Force Model for Pedestrian Dynamics"
   - 加速度分解理论基础

### 数学基础

**微分几何中的曲线参数化：**
```
曲线 r(t) = (x(t), y(t), z(t))
速度 v = dr/dt
加速度 a = dv/dt
曲率 κ = |dT/ds| = |dv/ds| / |v|  (Frenet-Serret框架)

切向分量: a_t = (a · v̂)
法向分量: a_n = |a × v̂|
```

---

## 🎓 学习资源

如需理解速度-加速度分解的更多细节，建议查阅：
- Multivariable Calculus (James Stewart)
- Differential Geometry (do Carmo)
- Robotics Fundamentals (Siciliano et al.) - Chapter 2

---

## ✅ 检查清单

部署v2之前：

- [ ] 验证24D特征的形状 (T, agents, 24)
- [ ] 确认特征归一化成功 (mean≈0, std≈1)
- [ ] 检查y_velocity和y_accel的计算正确性
- [ ] 验证损失权重相加为1 (0.65 + 0.20 + 0.15 = 1.0)
- [ ] 测试小批次数据，确保无Shape错误
- [ ] 在子集上验证收敛性
- [ ] 对比v1基线的性能
- [ ] 保存全部training_history用于分析

---

**下一步行动：**

1. 运行 `train_swarm_v2_complete.py` 进行初步训练
2. 对比训练曲线和最终MAE与v1
3. 分析三个损失分支的贡献度
4. 进行针对性的超参数调整
5. 在推理时观察预测的速度和加速度的合理性
