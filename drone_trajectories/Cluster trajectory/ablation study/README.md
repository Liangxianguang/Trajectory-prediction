# 消融实验说明

## 📋 5个消融实验配置

| 实验编号 | GAT | 特征增强 | BiGRU+Cross Attention | 特征维度 | 训练脚本 | 模型类 |
|---------|-----|---------|----------------------|---------|---------|--------|
| 1 | × | × | × | 16D | `train_ablation_exp1_baseline.py` | `BaselineGRUModel` |
| 2 | × | √ | √ | 32D | `train_ablation_exp2_feat_bigru.py` | `FeatureEnhancedBiGRUModel` |
| 3 | √ | × | √ | 16D | `train_ablation_exp3_gnn_bigru.py` | `GNNBiGRUModel` |
| 4 | √ | √ | × | 32D | `train_ablation_exp4_gnn_feat.py` | `GNNFeatureModel` |
| 5 | √ | √ | √ | 32D | `train_ablation_exp5_full.py` | `DynamicsAwareSwarmGRUModel_with_GNN` |

## 📁 文件结构

```
ablation study/
├── README.md                          # 本文件
├── ablation_models.py                 # 消融实验模型定义
├── ablation_train_utils.py            # 训练工具函数
├── train_ablation_exp1_baseline.py    # 实验1训练脚本
├── train_ablation_exp2_feat_bigru.py # 实验2训练脚本
├── train_ablation_exp3_gnn_bigru.py  # 实验3训练脚本
├── train_ablation_exp4_gnn_feat.py   # 实验4训练脚本
└── train_ablation_exp5_full.py       # 实验5训练脚本
```

## 🚀 快速开始

### 前置准备

1. **确保数据文件存在**：
   - `../swarm_segments/input_agents_3.npz`
   - `../swarm_segments/output_agents_3.npz`

2. **预计算特征（可选但推荐）**：
   - 16D特征：使用 `precompute_features.py` 或实时计算
   - 32D特征：使用 `precompute_features_v4.py` 生成到 `../features_32d/`

### 实验1：基线模型（16D，无GAT，无特征增强，无BiGRU+CA）
```bash
cd "drone_trajectories/Cluster trajectory/ablation study"
python train_ablation_exp1_baseline.py --agents 3 --epochs 150 --batch_size 256 --use_amp
```

### 实验2：特征增强 + BiGRU+CA（32D，无GAT）
```bash
python train_ablation_exp2_feat_bigru.py --agents 3 --epochs 150 --batch_size 256 --features_dir ../features_32d --use_amp
```

### 实验3：GAT + BiGRU+CA（16D，无特征增强）
```bash
python train_ablation_exp3_gnn_bigru.py --agents 3 --epochs 150 --batch_size 256 --use_amp
```

### 实验4：GAT + 特征增强（32D，无BiGRU+CA）
```bash
python train_ablation_exp4_gnn_feat.py --agents 3 --epochs 150 --batch_size 256 --features_dir ../features_32d --use_amp
```

### 实验5：完整模型 DG32-BCAT（32D，GAT + 特征增强 + BiGRU+CA）
```bash
python train_ablation_exp5_full.py --agents 3 --epochs 150 --batch_size 256 --features_dir ../features_32d --use_amp
```

## 📊 实验详细说明

### 实验1：基线模型
- **模型架构**：单向GRU编码器 + 单向GRU解码器
- **特征**：16D（位置3 + 多尺度速度9 + 曲率1 + 平面曲率3）
- **特点**：最简单的基线，用于对比其他改进的效果

### 实验2：特征增强 + BiGRU+CA
- **模型架构**：BiGRU编码器 + Cross Attention解码器
- **特征**：32D（24D基础 + 8D曲率增强）
- **特点**：评估特征增强和BiGRU+CA的独立贡献

### 实验3：GAT + BiGRU+CA
- **模型架构**：GAT + BiGRU编码器 + Cross Attention解码器
- **特征**：16D（无特征增强）
- **特点**：评估GAT的独立贡献（在16D特征上）

### 实验4：GAT + 特征增强
- **模型架构**：GAT + 单向GRU编码器 + 单向GRU解码器（无Cross Attention）
- **特征**：32D
- **特点**：评估GAT和特征增强的组合效果（无BiGRU+CA）

### 实验5：完整模型
- **模型架构**：GAT + BiGRU编码器 + Cross Attention解码器
- **特征**：32D
- **特点**：完整模型，包含所有改进

## 🔧 通用参数说明

所有训练脚本支持以下参数：

- `--data_dir`: 数据目录（默认：`../swarm_segments`）
- `--agents`: 无人机数量（默认：3）
- `--use_subset`: 使用子集数据（用于快速测试）
- `--features_dir`: 预计算特征目录（实验1和3可选，实验2/4/5推荐）
- `--hidden_size`: 隐藏层维度（默认：128）
- `--num_layers`: GRU层数（默认：2）
- `--epochs`: 训练轮数（默认：150）
- `--batch_size`: 批次大小（默认：256）
- `--lr`: 学习率（默认：2e-4）
- `--weight_decay`: 权重衰减（默认：5e-5）
- `--use_amp`: 使用混合精度训练（推荐）
- `--seed`: 随机种子（默认：42）
- `--device`: 设备（默认：cuda:0）

## 📈 输出文件

每个实验会在 `ablation_results_agents_{N}_exp{N}_*` 目录下生成：

- `best_model_*.pt`: 最佳模型检查点
- `checkpoint_*.pt`: 定期保存的检查点（每10个epoch）
- `training_history_*.csv`: 训练历史记录
- `config_*.json`: 实验配置信息

## 📊 实验对比

运行所有实验后，可以：

1. **查看训练历史**：比较各实验的 `training_history_*.csv`
2. **对比最佳性能**：查看各实验的 `best_model_*.pt` 对应的验证损失和MAE
3. **分析组件贡献**：通过对比实验1-5，评估每个组件的独立贡献

## ⚠️ 注意事项

1. **特征维度匹配**：确保使用的特征维度与实验配置一致
   - 实验1和3：16D特征
   - 实验2、4、5：32D特征

2. **预计算特征**：如果使用预计算特征，确保特征文件存在且维度正确

3. **内存使用**：32D特征模型需要更多内存，可以适当减小batch_size

4. **训练时间**：实验5（完整模型）训练时间最长，建议使用GPU和混合精度训练
