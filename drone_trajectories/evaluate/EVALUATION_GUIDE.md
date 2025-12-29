# 统一评估脚本使用指南

## 功能说明

`evaluate_all_models.py` 是一个全面的模型评估工具，支持：

✓ **多模型对比**：同时评估多个模型（单向/双向 GRU，带/不带 attention）
✓ **完整指标计算**：MAE、MSE、RMSE、MAPE（全局 + 分轴统计）
✓ **批量数据处理**：自动遍历测试集目录的所有 CSV 轨迹文件
✓ **详细报告生成**：CSV 汇总表 + JSON 配置 + 排名对比

## 快速开始

### 方式 1：使用批处理脚本（推荐）

```bash
cd /d "D:\Trajectory prediction\drone_trajectories\evaluate"
.\run_evaluation_fixed.bat
```

脚本会自动：
1. 加载 `eval_config_example.json` 中定义的所有模型
2. 在 `..\..\Synthetic-UAV-Flight-Trajectories` 目录中测试所有轨迹
3. 输出结果到 `evaluation_results` 目录

### 方式 2：命令行手动调用

```bash
cd /d "D:\Trajectory prediction\drone_trajectories\evaluate"

python evaluate_all_models.py ^
  --models eval_config_example.json ^
  --test_dir ..\..\Synthetic-UAV-Flight-Trajectories ^
  --output_dir evaluation_results ^
  --input_length 20 ^
  --method physics_constrained ^
  --max_samples 50
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--models` | str | 必需 | 模型配置 JSON 文件路径 |
| `--test_dir` | str | 必需 | 测试集目录（包含 CSV 轨迹文件）|
| `--output_dir` | str | `./evaluation_results` | 结果输出目录 |
| `--input_length` | int | 20 | 输入序列长度 |
| `--method` | str | `physics_constrained` | 重建方法（simple/physics_constrained/smoothed）|
| `--max_samples` | int | None | 最多评估样本数（用于快速测试） |
| `--device` | str | `cuda` | 计算设备（cuda/cpu）|

## 模型配置文件格式

编辑 `eval_config_example.json` 来添加/修改要评估的模型：

```json
[
  {
    "name": "短轨迹_单向GRU_64h",
    "model_path": "../tool/combined_short_gru_models_enhanced/short_enhanced_gru_model_best_model.pth",
    "stats_path": "../tool/combined_short_gru_models_enhanced/short_enhanced_gru_model_norm_stats.npz",
    "hidden_dim": 64,
    "num_layers": 2,
    "bidirectional": false
  },
  ...
]
```

**参数说明：**
- `name`: 模型显示名称（用于报告和排名）
- `model_path`: 模型权重文件路径（相对于 evaluate 目录，用 `../` 返回到 drone_trajectories）
- `stats_path`: 归一化统计量文件路径
- `hidden_dim`: 隐藏层单元数（**必需**）
  - 单向 GRU: 使用实际 hidden_dim（如 64, 128, 256）
  - 双向 GRU: 使用单方向的 hidden_dim（系统会自动处理双向扩展）
- `num_layers`: GRU 层数（**必需**）
- `bidirectional`: 是否为双向 GRU（可选，默认 false，**仅用于标识**，推理器自动检测）

**获取正确参数的方法：**
查看模型训练时的配置文件或日志，找到 `hidden_dim` 和 `num_layers` 的值。

## 输出文件说明

评估完成后，`evaluation_results` 目录会生成：

```
evaluation_results/
├── models_comparison.csv          # 所有模型的对比汇总（最重要！）
├── models_comparison.json         # JSON 版本的对比结果
├── 模型1名称_detailed_results.csv  # 模型 1 在每条轨迹上的详细指标
├── 模型2名称_detailed_results.csv  # 模型 2 在每条轨迹上的详细指标
└── ...
```

### models_comparison.csv 示例

| model_name | num_samples | num_errors | avg_MAE | avg_MSE | avg_RMSE | avg_MAPE |
|-----------|------------|-----------|---------|---------|----------|----------|
| 短轨迹_双向GRU_64h | 150 | 2 | 0.001234 | 0.000003 | 0.001856 | 1.23% |
| 短轨迹_单向GRU_64h | 150 | 2 | 0.001456 | 0.000004 | 0.002012 | 1.45% |
| 标准GRU_128h | 150 | 3 | 0.001678 | 0.000005 | 0.002234 | 1.67% |

## 关键指标解释

- **MAE（平均绝对误差）**：平均误差距离，单位通常为米
- **MSE（均方误差）**：平方误差的均值，对大误差更敏感
- **RMSE（均方根误差）**：MSE 的平方根，与原数据单位一致
- **MAPE（平均百分比误差）**：相对误差百分比（%），便于跨尺度对比
- **分轴指标（_x/_y/_z）**：分别统计 X/Y/Z 三个轴方向的误差

## 快速测试（快速模式）

若要快速验证脚本功能或调试，可限制样本数：

```bash
python tool\evaluate_all_models.py ^
  --models tool/eval_config_example.json ^
  --test_dir ..\..\Synthetic-UAV-Flight-Trajectories ^
  --output_dir evaluation_results_quick ^
  --max_samples 10
```

这样只会评估前 10 个测试文件，耗时约 1-2 分钟。

## 常见问题

**Q：评估耗时太长怎么办？**
A：使用 `--max_samples N` 限制样本数，或改用 CPU 加快（改 `--device cpu`）。

**Q：找不到模型文件怎么办？**
A：确保模型文件路径相对于 `drone_trajectories` 目录正确，或改为绝对路径。

**Q：如何添加新模型进行对比？**
A：
1. 确保新模型的权重文件和统计量文件已生成
2. 在 `eval_config_example.json` 中添加新条目（参考现有条目）
3. 重新运行评估脚本

**Q：为什么有些轨迹显示为失败？**
A：常见原因：
- 轨迹长度不足（需 >= 20）
- CSV 列名不匹配（需要 tx/ty/tz 或 x/y/z）
- 模型权重或统计量文件不完整

## 高级用法

### 使用物理约束方法（推荐）

```bash
python evaluate_all_models.py ^
  --models eval_config_example.json ^
  --test_dir ..\..\Synthetic-UAV-Flight-Trajectories ^
  --method physics_constrained
```

### 使用简单积分方法

```bash
python evaluate_all_models.py ^
  --models eval_config_example.json ^
  --test_dir ..\..\Synthetic-UAV-Flight-Trajectories ^
  --method simple
```

### 使用轨迹平滑方法

```bash
python evaluate_all_models.py ^
  --models eval_config_example.json ^
  --test_dir ..\..\Synthetic-UAV-Flight-Trajectories ^
  --method smoothed
```

## 下一步建议

评估完成后，你可以：

1. **对比性能差异**：检查 `models_comparison.csv`，找出最佳模型
2. **分析失败原因**：查看每个模型的详细结果，理解在哪些轨迹上表现较差
3. **调优参数**：根据分轴指标（X/Y/Z），针对性地改进模型（如加强 Z 轴约束）
4. **持续迭代**：每次改进后重新评估，跟踪性能提升

---

**作者**：自动生成  
**日期**：2025-12-29
