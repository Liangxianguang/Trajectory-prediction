# 位置和速度模型训练命令清单

## 步骤 1: 生成速度数据集（仅需一次）

如果还没有 `dataset_velocity_segments_synth.npz`，先生成它：

```cmd
cd /d "D:\Trajectory prediction"
python drone_trajectories\process_multiple_trajectories.py "D:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories" 20 10 --compute_velocity --save_path "D:\Trajectory prediction\drone_trajectories\dataset_velocity_segments_synth.npz"
```

## 位置模型 - 三个大小

### Short (推荐先训练)
```cmd
cd /d "D:\Trajectory prediction"
python drone_trajectories\train_model.py --data_path drone_trajectories\dataset_position_segments_synth.npz --stats_path drone_trajectories\transformed_trajectories\pos_stats.npz --output_dir drone_trajectories\gru_models_synth --model_name position_synth_short --hidden_dim 64 --num_layers 2 --epochs 120 --batch_size 64 --lr 1e-3 --patience 15
```
**预计耗时**: ~30分钟 | **参数量**: ~38K | **推理速度**: 最快

### Mix (平衡)
```cmd
cd /d "D:\Trajectory prediction"
python drone_trajectories\train_model.py --data_path drone_trajectories\dataset_position_segments_synth.npz --stats_path drone_trajectories\transformed_trajectories\pos_stats.npz --output_dir drone_trajectories\gru_models_synth --model_name position_synth_mix --hidden_dim 128 --num_layers 3 --epochs 120 --batch_size 64 --lr 1e-3 --patience 15
```
**预计耗时**: ~60分钟 | **参数量**: ~128K | **推理速度**: 中等

### Long (最强)
```cmd
cd /d "D:\Trajectory prediction"
python drone_trajectories\train_model.py --data_path drone_trajectories\dataset_position_segments_synth.npz --stats_path drone_trajectories\transformed_trajectories\pos_stats.npz --output_dir drone_trajectories\gru_models_synth --model_name position_synth_long --hidden_dim 256 --num_layers 5 --epochs 120 --batch_size 32 --lr 1e-3 --patience 15
```
**预计耗时**: ~120分钟 | **参数量**: ~429K | **推理速度**: 较慢

---

## 速度模型 - 三个大小

### Short
```cmd
cd /d "D:\Trajectory prediction"
python drone_trajectories\train_model.py --data_path drone_trajectories\dataset_velocity_segments_synth.npz --stats_path drone_trajectories\transformed_trajectories\vel_stats.npz --output_dir drone_trajectories\gru_models_synth --model_name velocity_synth_short --hidden_dim 64 --num_layers 2 --epochs 120 --batch_size 64 --lr 1e-3 --patience 15
```

### Mix
```cmd
cd /d "D:\Trajectory prediction"
python drone_trajectories\train_model.py --data_path drone_trajectories\dataset_velocity_segments_synth.npz --stats_path drone_trajectories\transformed_trajectories\vel_stats.npz --output_dir drone_trajectories\gru_models_synth --model_name velocity_synth_mix --hidden_dim 128 --num_layers 3 --epochs 120 --batch_size 64 --lr 1e-3 --patience 15
```

### Long
```cmd
cd /d "D:\Trajectory prediction"
python drone_trajectories\train_model.py --data_path drone_trajectories\dataset_velocity_segments_synth.npz --stats_path drone_trajectories\transformed_trajectories\vel_stats.npz --output_dir drone_trajectories\gru_models_synth --model_name velocity_synth_long --hidden_dim 256 --num_layers 5 --epochs 120 --batch_size 32 --lr 1e-3 --patience 15
```

---

## 执行选项

### 选项 A: 全自动（推荐）
双击运行 `train_all_models.bat` 一键训练6个模型（总耗时 ~5-6小时）

### 选项 B: 分别执行
依次复制上述命令到 CMD 窗口，按回车执行

### 选项 C: 指定训练
如只想训练位置模型的三个大小，只复制位置模型的三个命令

---

## 监控指标说明

训练输出示例：
```
Epoch 1/120 - Loss: 0.1234 - Val Loss: 0.0987
Epoch 2/120 - Loss: 0.0987 - Val Loss: 0.0876
...
Epoch 120/120 - Loss: 0.0012 - Val Loss: 0.0018
```

- **Loss**: 训练集损失函数（MSE）
- **Val Loss**: 验证集损失（目标：最小化）
- **Early Stopping**: 若验证损失连续15个epoch无改进，自动停止

---

## 输出文件位置

训练完成后，模型和统计量保存在：
```
D:\Trajectory prediction\drone_trajectories\gru_models_synth\
├── position_synth_short.pt
├── position_synth_short_best.pt
├── position_synth_mix.pt
├── position_synth_mix_best.pt
├── position_synth_long.pt
├── position_synth_long_best.pt
├── velocity_synth_short.pt
├── velocity_synth_short_best.pt
├── velocity_synth_mix.pt
├── velocity_synth_mix_best.pt
├── velocity_synth_long.pt
└── velocity_synth_long_best.pt

D:\Trajectory prediction\drone_trajectories\transformed_trajectories\
├── pos_stats.npz (位置归一化统计量)
└── vel_stats.npz (速度归一化统计量)
```

---

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| `CUDA out of memory` | 减小 `--batch_size` (e.g., 32→16) |
| `Module not found` | 检查虚拟环境激活: `python -m venv traj_pred_prep\Scripts\activate` |
| 训练卡住 | Ctrl+C 中断，检查数据路径是否正确 |
| `pos_stats.npz not found` | 确认已运行 `compute_stats.py` |

