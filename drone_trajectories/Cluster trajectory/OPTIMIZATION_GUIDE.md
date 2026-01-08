# 集群轨迹模型优化指南

## 问题诊断总结

根据详细诊断分析，模型遇到的瓶颈：

### 1. **X轴学习瓶颈（核心问题）**
- X轴 MAE: **0.1215m**（比平均高 **35.8%**）
- Y轴 MAE: 0.0924m（较好）
- Z轴 MAE: 0.0544m（最好，是X轴的2.2倍）
- **轴向平衡分数: 0.6930**（一般，需要 >0.85 为优秀）

### 2. **无人机学习均衡性**
| 无人机 | 3D MAE | X轴差异 | 学习均衡度 |
|------|--------|--------|---------|
| Agent 0 | 0.0938m | X:0.133m | 中等 |
| Agent 1 | 0.0790m | X:0.109m | **最好** |
| Agent 2 | 0.0956m | X:0.122m | 中等 |

### 3. **集群协作能力（强）**
- 无人机间距离预测相关系数：0.95~0.98（非常好）
- 说明：模型理解集群几何关系，问题不在协作机制

---

## 优化方案

### 方案 1：轴向损失权重调整（立竿见影 ⭐⭐⭐⭐⭐）

**原理**：X轴学习差，需要在损失函数中加强其权重

**实现**：修改 `train_swarm_model_enhanced.py` 中的 `MultiObjectiveLoss`

```python
class MultiObjectiveLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, axis_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # ✅ 改进：轴向权重 - 强化X轴
        if axis_weights is None:
            # 原始: [1.0, 1.1, 1.2]
            # 改进: X轴权重提高到 1.5, 让X学习获得更多梯度
            axis_weights = [1.5, 1.2, 1.0]  # X, Y, Z
            
        self.register_buffer('axis_weights', torch.tensor(axis_weights, dtype=torch.float32))
```

**关键参数**：
- `axis_weights[0]` (X轴) 从 1.0 → **1.5** （+50%）
- `axis_weights[1]` (Y轴) 从 1.1 → 1.2 （维持较高）
- `axis_weights[2]` (Z轴) 从 1.2 → 1.0 （降低，已学得好）

**期望效果**：X轴 MAE 从 0.121m → 0.09m（目标）

**命令**：
```bash
python train_swarm_model_enhanced.py \
  --agents 3 \
  --epochs 300 \
  --batch_size 1024 \
  --hidden_size 128 \
  --num_layers 3 \
  --dropout 0.5 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --patience 50 \
  --output_dir optimized_axis_weights \
  --use_amp \
  --use_attention
```

---

### 方案 2：特征工程增强（中期优化 ⭐⭐⭐⭐）

**问题根源**：X轴可能在特征中表现不足

**改进点**：在 `compute_features_for_inference` 中为X轴添加更多特征

```python
def compute_features_for_inference(trajectory, input_mean_all=None, input_std_all=None, 
                                   input_mean=None, input_std=None, dt=0.1):
    vel = compute_multi_scale_velocity(trajectory, dt=dt)  # (seq_in, agents, 9)
    curv_3d = compute_curvature(trajectory, dt=dt)  # (seq_in, agents, 1)
    curv_plane = compute_plane_curvatures(trajectory, dt=dt)  # (seq_in, agents, 3)
    
    # ✅ 新增：X轴专项特征
    x_vel = trajectory[:, :, 0]  # (seq_in, agents)
    x_accel = np.gradient(np.gradient(trajectory[:, :, 0], axis=0), axis=0) / (dt ** 2)
    x_jerk = np.gradient(x_accel, axis=0) / dt
    
    # 将新特征追加（现在从16维扩展到19维）
    features = np.concatenate([trajectory, vel, curv_3d, curv_plane, 
                               x_vel[..., np.newaxis], 
                               x_accel[..., np.newaxis],
                               x_jerk[..., np.newaxis]], axis=-1)  # (seq_in, agents, 19)
    
    # ... 后续归一化逻辑保持不变
```

**代价**：需要调整模型输入维度从 16 → 19（略微增加参数）

**期望效果**：X轴特征表达更丰富，学习更深入

---

### 方案 3：数据增强与正则化（长期稳定 ⭐⭐⭐⭐）

#### 3.1 数据增强（预处理阶段）

在 `preprocess_swarm_trajectories.py` 或数据加载时添加：

```python
def augment_trajectory_xyz(trajectory, aug_prob=0.5):
    """
    数据增强：沿X/Y/Z轴随机扰动，模拟传感器噪声与变差
    """
    if np.random.random() > aug_prob:
        return trajectory
    
    # 随机选择轴向
    if np.random.random() < 0.4:  # 40% 概率增强X轴
        # X轴添加小的高斯噪声
        noise_scale = 0.02 * np.std(trajectory[..., 0])
        trajectory[..., 0] += np.random.normal(0, noise_scale, trajectory[..., 0].shape)
    
    # 轻微旋转（保持物理约束）
    angle = np.random.uniform(-5, 5) * np.pi / 180  # ±5 度
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xy = trajectory[..., :2].copy()
    trajectory[..., 0] = cos_a * xy[..., 0] - sin_a * xy[..., 1]
    trajectory[..., 1] = sin_a * xy[..., 0] + cos_a * xy[..., 1]
    
    return trajectory
```

在 Dataset 中集成：
```python
def __getitem__(self, idx):
    x = self.X_orig[idx]
    y = self.Y_orig[idx]
    
    # ✅ 应用数据增强
    x = augment_trajectory_xyz(x, aug_prob=0.5)
    y = augment_trajectory_xyz(y, aug_prob=0.3)  # 输出增强概率略低
    
    # ... 后续特征计算
```

#### 3.2 增强正则化

```python
# 在训练循环中添加额外的 L2 正则化
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=2e-4  # 从 1e-4 → 2e-4，加倍
)

# 添加梯度正则化
def train_epoch(...):
    for ...:
        # ... 计算 loss
        loss.backward()
        
        # ✅ 梯度范数剪切（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # ✅ 梯度平滑（可选，实验性）
        for param in model.parameters():
            if param.grad is not None:
                param.grad *= 0.99  # 轻微衰减
        
        optimizer.step()
```

---

### 方案 4：学习率调度优化（渐进式降低 ⭐⭐⭐⭐⭐）

**问题**：当前 lr 固定，plateau 时无法进一步优化

**改进**：使用阶梯式学习率下降

```python
# 在 main() 中配置
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,  # 每次降低到 50%
    patience=8,  # 更激进：8 epoch 无改进就降
    verbose=True,
    min_lr=1e-6
)

# 在每个 epoch 后调用
val_loss, val_mae = evaluate(...)
scheduler.step(val_loss)

# ✅ 同时加入 CosineAnnealingLR 作为主调度器
main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # 100 epoch 内从初始 lr 降到 0
    eta_min=1e-6
)

# 每个 epoch 调用（在 ReduceLROnPlateau 之前）
main_scheduler.step()
```

---

### 方案 5：模型架构微调（可选 ⭐⭐⭐）

**增加针对X轴的专项头**：

```python
class EnhancedSwarmGRUModel(nn.Module):
    def __init__(self, ...):
        # ... 现有代码
        
        # ✅ 新增：X轴特化头
        self.x_axis_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 1),  # 仅预测X坐标
        )
    
    def forward(self, x, x_orig, y=None, teacher_forcing_ratio=0.5):
        # ... 现有解码逻辑得到 base_output (batch, 3)
        
        # ✅ X轴特化预测
        x_pred_special = self.x_axis_head(h[-1])  # (batch, 1)
        
        # 融合：用特化头预测的X代替基础输出的X
        alpha = 0.3  # 权重
        base_output[:, 0] = (1 - alpha) * base_output[:, 0] + alpha * x_pred_special[:, 0]
        
        return base_output
```

---

## 实验方案对比

| 方案 | 优先级 | 实施时间 | 期望改进 | 复杂度 | 推荐 |
|------|------|--------|--------|--------|------|
| 1. 轴向权重 | 🔴 立即 | 5分钟 | X轴 -25% | 极低 | ⭐⭐⭐⭐⭐ |
| 2. 特征增强 | 🔴 立即 | 15分钟 | 全体 -10% | 低 | ⭐⭐⭐⭐ |
| 3a. 数据增强 | 🟡 短期 | 30分钟 | 泛化 +5% | 低 | ⭐⭐⭐⭐ |
| 3b. 正则化 | 🟡 短期 | 10分钟 | 稳定性 +10% | 低 | ⭐⭐⭐⭐ |
| 4. LR调度 | 🟡 短期 | 20分钟 | 全体 -15% | 中 | ⭐⭐⭐⭐⭐ |
| 5. 专项头 | 🟢 长期 | 45分钟 | X轴 -20% | 高 | ⭐⭐⭐ |

---

## 推荐执行流程

### 第一阶段（今天，预计改进：MAE 0.089→0.08m）
1. ✅ 修改 `axis_weights = [1.5, 1.2, 1.0]`
2. ✅ 修改 `ReduceLROnPlateau` 的 `patience=8`
3. ✅ 增加 `weight_decay` 到 `2e-4`

**命令**：
```bash
python train_swarm_model_enhanced.py \
  --agents 3 \
  --epochs 200 \
  --batch_size 1024 \
  --lr 1e-4 \
  --weight_decay 2e-4 \
  --patience 50 \
  --dropout 0.5 \
  --output_dir optimized_v1 \
  --use_amp \
  --use_attention
```

### 第二阶段（明天，预计改进：MAE 0.08→0.075m）
1. ✅ 增加 X/Y/Z 特征维度（添加加速度、抖动）
2. ✅ 加入数据增强（旋转、噪声）
3. ✅ 集成 CosineAnnealingLR

### 第三阶段（可选，预计改进：MAE 0.075→0.07m）
1. ✅ 添加 X轴特化头
2. ✅ 微调集群协作权重

---

## 监控指标

训练时关注：
```
✓ 全局 MAE（目标：0.089→0.08）
✓ X轴 MAE（目标：0.121→0.09）  ← 关键指标
✓ 轴向平衡分数（目标：0.69→0.85）
✓ Train/Val MAE Gap（应 <10%）
```

---

## 快速对比参考

使用诊断脚本查看改进效果：
```bash
python analyze_per_agent_predictions.py \
  --predictions inference_results/predictions_agents_3.npz \
  --output_dir agent_analysis_v2 \
  --num_samples -1
```

对比 `analysis_report.txt` 中的 X轴 MAE 和平衡分数变化。
