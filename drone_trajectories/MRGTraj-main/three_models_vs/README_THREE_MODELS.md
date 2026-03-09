# 三模型对比框架

对比三个高性能轨迹预测模型的性能：

## 模型简介

| 模型 | 出处 | 关键特性 | 预期 ADE |
|------|------|---------|----------|
| **LBEBM3D** | 3DMoTraj (基准) | VAE + Subgoal + Physics | ~1.0m |
| **Exp5 (DG32-BCAT)** | 带GNN的改进版 | 32D特征 + Graph Attention | ~0.4m |
| **MRGTraj-LBEBM3D** | 本项目 | 多智能体 + LBEBM3D启发 | ~0.28m ✨ |

## 快速开始

### 1. 准备模型文件

确保以下文件存在：

```
3DMoTraj/
  └─ saved_models/checkpoints_accfix/epoch_010.pt

Cluster trajectory/ablation study/
  └─ ablation_results_agents_3_exp5_full/
      ├─ config_agents_3_exp5_full.json
      ├─ stats_agents_3_exp5_full.npz
      └─ best_model_agents_3_exp5_full.pt

MRGTraj-main/
  └─ checkpoints_lbebm3d/
      └─ agents_3_lbebm3d_inspired/
          └─ best_model.pth
```

### 2. 运行对比

#### 方式 A: 使用批处理脚本 (Windows)

```batch
cd d:\Trajectory prediction\drone_trajectories\Cluster trajectory\ablation study\LBEBM3DvsGNN
run_compare_three_models.bat
```

#### 方式 B: 直接运行 Python 脚本

```bash
python compare_three_models.py \
  --data_dir ../../swarm_segments \
  --agents 3 \
  --use_subset \
  --lbebm_model ../../../3DMoTraj/saved_models/checkpoints_accfix/epoch_010.pt \
  --exp5_dir ../ablation_results_agents_3_exp5_full \
  --mrgraj_model ../../../drone_trajectories/MRGTraj-main/checkpoints_lbebm3d/agents_3_lbebm3d_inspired/best_model.pth \
  --features_32d_dir ../../features_32d \
  --output_dir comparison_results_three_models \
  --num_samples 15 \
  --seed 42
```

### 3. 查看结果

输出文件存储在 `comparison_results_three_models/` 目录：

- **comparison_summary.json** - 汇总统计数据
- **sample_*.png** - 每个样本的可视化对比图表

## 参数说明

### 数据参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 轨迹数据目录 | 必需 |
| `--agents` | 无人机数量 | 3 |
| `--use_subset` | 使用子集数据 | False |

### 模型路径

| 参数 | 说明 | 必需 |
|------|------|------|
| `--lbebm_model` | LBEBM3D 模型路径 | ✓ |
| `--exp5_dir` | Exp5 结果目录 | ✓ |
| `--mrgraj_model` | MRGTraj 最佳模型路径 | ✓ |
| `--features_32d_dir` | 32D 特征目录 | features_32d |

### LBEBM3D 推理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_scale` | 数据缩放因子 | 1.0 |
| `--e_init_sig` | 初始能量信号 | 2.0 |
| `--e_prior_sig` | 先验能量信号 | 2.0 |
| `--e_l_steps` | 能量优化步数 | 20 |
| `--e_l_step_size` | 能量优化步长 | 0.4 |
| `--e_l_with_noise` | 带噪声的能量优化 | False |

### 采样参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample_indices` | 逗号分隔的样本索引 | None (随机) |
| `--num_samples` | 随机样本数 | 10 |
| `--seed` | 随机种子 | 42 |
| `--use_val_split` | 使用验证集划分 | False |
| `--val_split` | 验证集比例 | 0.2 |

### 输出参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--no_visualize` | 禁用可视化 | False |
| `--output_dir` | 输出目录 | comparison_three_models |

## 可视化输出

每个样本的对比图表包含 8 个子图：

### 上排 (4 个)
1. **3D 轨迹** - 三维空间中三个模型的预测轨迹
2. **XY 平面** - X-Y 平面投影
3. **XZ 平面** - X-Z 平面投影
4. **YZ 平面** - Y-Z 平面投影

### 下排 (4 个)
5. **逐步误差** - 预测步长处的位置误差曲线
6. **ADE/FDE 对比** - 三模型的平均和最终位移误差柱状图
7. **单智能体误差** - 每个无人机的平均误差
8. **统计汇总** - 文本格式的关键指标

## 性能指标

### 关键指标

- **ADE (Average Displacement Error)** - 预测轨迹与真实轨迹的平均欧氏距离
- **FDE (Final Displacement Error)** - 最后一帧的位置误差
- **RMSE (Root Mean Square Error)** - 均方根误差

### 统计分析

对于每个指标，脚本输出：
- **均值** - 所有样本的平均值
- **标准差** - 所有样本的标准差
- **改进 %** - 相对改进百分比

## 示例输出

```
===  Epoch 300/300]: 100%|█████████| 450/450 [00:41<00:00, 10.84it/s, ade=0.2807, fde=0.4295, loss=1.7303]

============================================
三模型对比: LBEBM3D vs Exp5 vs MRGTraj
============================================

✓ LBEBM3D 统计:
  ADE: 1.0234 ± 0.2145m
  FDE: 1.5678 ± 0.3421m
  RMSE: 1.1890 ± 0.2456m

✓ Exp5 (DG32-BCAT) 统计:
  ADE: 0.4123 ± 0.1234m
  FDE: 0.6789 ± 0.1890m
  RMSE: 0.5234 ± 0.1456m

✓ MRGTraj-LBEBM3D 统计:
  ADE: 0.2807 ± 0.0890m
  FDE: 0.4295 ± 0.1234m
  RMSE: 0.3456 ± 0.1123m

=== 性能改进 ===
Exp5 vs LBEBM3D (ADE):      -59.75%  (↓ 改进)
MRGTraj vs LBEBM3D (ADE):   -72.56%  (↓ 改进)
MRGTraj vs Exp5 (ADE):      -31.90%  (↓ 改进)
```

## 故障排除

### 问题 1: "LBEBM3D 不可用"

**解决方案**:
1. 检查 `3DMoTraj/tool/infer_lbebm3d_baseline.py` 存在
2. 检查 Python 路径是否正确添加

### 问题 2: "特征文件未找到"

**解决方案**:
1. 确认 `features_32d/` 目录存在
2. 检查特征文件命名是否匹配 `features_agents_3_32d.npz`

### 问题 3: "MRGTraj 不可用"

**解决方案**:
1. 检查 `MRGTraj-main/model_swarm.py` 存在
2. 运行 `python -c "from model_swarm import MRGTrajSwarm"` 验证导入

## 参考

- **LBEBM3D**: https://github.com/PredictionSystem/3DMoTraj
- **MRGTraj**: 多智能体融合 LBEBM3D 经验的优化版本
- **Exp5**: 带 GNN 的改进基线 (DG32-BCAT)

## 许可证

同主项目许可证
