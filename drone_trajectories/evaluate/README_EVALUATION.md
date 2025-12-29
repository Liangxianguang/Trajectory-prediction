# 轨迹预测模型评估指南

## 概述

`evaluate_all_models.py` 是一个统一的评估脚本，用于在多个测试集上评估轨迹预测模型，支持：

- ✅ **自动模型发现**：自动扫描 `tool/` 下的所有 checkpoint（.pth 文件）
- ✅ **多格式支持**：加载 CSV 和 TXT 格式的轨迹文件
- ✅ **多目录评估**：同时在多个测试目录上评估
- ✅ **自动参数推断**：从 checkpoint 自动推断 `hidden_dim`、`num_layers`、`bidirectional` 等参数
- ✅ **综合指标**：MAE、MSE、RMSE、MAPE、逐轴统计
- ✅ **对比报告**：生成 CSV + JSON 格式的模型对比报告
- ✅ **可视化**：可选为部分样本生成预测结果的 3D 可视化（需要 plotly）

## 快速开始

### 1. 快速测试（推荐首先运行）

在 `evaluate/` 目录下运行：

```cmd
run_quick_test.bat
```

或直接执行：

```cmd
cd /d "D:\Trajectory prediction\drone_trajectories\evaluate"
python evaluate_all_models.py ^
  --auto_models ^
  --tool_dir "..\..\drone_trajectories\tool" ^
  --test_dir "..\..\Synthetic-UAV-Flight-Trajectories" ^
  --max_samples 5 ^
  --output_dir "evaluation_results_test"
```

**预期**：
- 耗时 1-2 分钟
- 加载第一个模型、评估 5 个轨迹样本
- 输出汇总统计和结果文件

### 2. 完整评估（所有模型 + 所有测试集）

在 `evaluate/` 目录下运行：

```cmd
run_full_evaluation.bat
```

或直接执行：

```cmd
cd /d "D:\Trajectory prediction\drone_trajectories\evaluate"
python evaluate_all_models.py ^
  --auto_models ^
  --tool_dir "..\..\drone_trajectories\tool" ^
  --test_dir "..\..\Synthetic-UAV-Flight-Trajectories,..\..\drone_trajectories\random_traj_100ms,..\..\drone_trajectories\new_random_traj_100ms" ^
  --output_dir "evaluation_results_full" ^
  --method physics_constrained ^
  --device cuda
```

**预期**：
- 耗时 1-2 小时（取决于模型数量和硬件）
- 自动发现 tool/ 下的所有模型（通常 20+ 个）
- 在三个测试目录评估
- 生成综合对比报告

## 命令行参数

### 必需参数

```
--test_dir TEST_DIR
```

测试集目录（支持多个，逗号分隔）：

```cmd
# 单个目录
--test_dir "..\..\Synthetic-UAV-Flight-Trajectories"

# 多个目录
--test_dir "..\..\Synthetic-UAV-Flight-Trajectories,..\..\drone_trajectories\random_traj_100ms"
```

### 模型加载选项

#### 选项 A：自动扫描（推荐）

```
--auto_models                      # 启用自动扫描
--tool_dir TOOL_DIR               # tool 目录路径（默认自动推断）
```

自动扫描会：
1. 递归查找 `tool_dir` 下的所有 `*.pth` checkpoint
2. 自动匹配对应的 `*_norm_stats.npz` 统计量文件
3. 跳过缺少 stats 的 checkpoint

#### 选项 B：使用配置文件

```
--models CONFIG.json              # JSON 配置文件（如 eval_config_example.json）
```

配置文件格式：

```json
[
  {
    "name": "模型名称",
    "model_path": "path/to/checkpoint.pth",
    "stats_path": "path/to/stats.npz"
  }
]
```

### 评估选项

```
--method {simple,physics_constrained,smoothed}
                                  # 重建方法（默认 physics_constrained）

--input_length INPUT_LENGTH       # 输入序列长度（默认 20）

--max_samples MAX_SAMPLES         # 最多评估样本数（默认全部）
                                  # 用于快速测试时指定较小值

--output_dir OUTPUT_DIR           # 输出目录（默认 ./evaluation_results）
```

### 可视化选项（可选）

```
--visualize                        # 启用可视化（需要 matplotlib/plotly）

--visual_samples N                # 每个模型生成的可视化样本数（默认 0）

--visual_output_dir DIR           # 可视化输出目录（默认使用 --output_dir）
```

### 硬件选项

```
--device {cuda,cpu}               # 计算设备（默认 cuda，无 GPU 时自动降级）
```

## 输出文件

评估完成后，在指定的 `--output_dir` 中会生成：

### 主要输出

```
evaluation_results/
├── models_comparison.csv          # 所有模型的对比汇总表
├── models_comparison.json         # JSON 格式的对比结果
├── 模型A_详细结果.csv           # 模型 A 在所有样本上的详细指标
├── 模型B_详细结果.csv
└── ...
```

### CSV 格式说明

**models_comparison.csv** 包含列：
- `model_name`: 模型名称
- `num_samples`: 有效评估样本数
- `num_errors`: 失败样本数
- `avg_MAE`, `avg_MSE`, `avg_RMSE`, `avg_MAPE`: 全局指标平均值
- `avg_MAE_x`, `avg_MAE_y`, `avg_MAE_z`: 逐轴 MAE
- `avg_RMSE_x`, `avg_RMSE_y`, `avg_RMSE_z`: 逐轴 RMSE
- `max_error`, `min_error`, `std_error`: 误差统计

**模型_详细结果.csv** 包含每个样本的详细指标（同上）

## 示例用法

### 示例 1：快速验证单个目录

```cmd
python evaluate_all_models.py ^
  --auto_models ^
  --test_dir "..\..\Synthetic-UAV-Flight-Trajectories" ^
  --max_samples 100 ^
  --output_dir "results_quick"
```

### 示例 2：完整评估（所有模型、所有目录）

```cmd
python evaluate_all_models.py ^
  --auto_models ^
  --tool_dir "..\..\drone_trajectories\tool" ^
  --test_dir "..\..\Synthetic-UAV-Flight-Trajectories,..\..\drone_trajectories\random_traj_100ms,..\..\drone_trajectories\new_random_traj_100ms" ^
  --output_dir "results_full" ^
  --device cuda
```

### 示例 3：可视化前 5 个样本

```cmd
python evaluate_all_models.py ^
  --auto_models ^
  --test_dir "..\..\Synthetic-UAV-Flight-Trajectories" ^
  --max_samples 10 ^
  --visualize ^
  --visual_samples 5 ^
  --output_dir "results_with_visual"
```

### 示例 4：使用自定义配置文件

```cmd
python evaluate_all_models.py ^
  --models eval_config_example.json ^
  --test_dir "..\..\Synthetic-UAV-Flight-Trajectories" ^
  --output_dir "results_custom"
```

## 故障排除

### 问题 1：未发现模型

**症状**：`自动发现 0 个模型 checkpoint`

**解决**：
1. 检查 `--tool_dir` 路径是否正确
2. 确保 checkpoint 文件名包含 `best_model` 或其他 `*.pth`
3. 查看是否有对应的 `*_norm_stats.npz` 文件（通常名字相同）

```cmd
# 调试：列出 tool 目录下的所有 .pth 文件
dir /s "..\..\drone_trajectories\tool\*.pth"
```

### 问题 2：无法加载轨迹文件

**症状**：`加载轨迹失败` 或 `未找到坐标列`

**解决**：
- 对于 CSV：确保有 `tx`, `ty`, `tz` 或 `x`, `y`, `z` 等标准列名
- 对于 TXT：确保是空白分隔格式，前 3 列为 X/Y/Z 坐标
- 脚本会自动尝试检测，但手动检查数据格式可加快排查

### 问题 3：内存不足

**症状**：`RuntimeError: CUDA out of memory`

**解决**：
1. 使用 `--max_samples` 限制评估样本数
2. 切换到 CPU: `--device cpu`
3. 对多个目录分别评估，而非一次性全部评估

### 问题 4：可视化失败

**症状**：`警告: 可视化失败`

**解决**：
1. 检查 plotly 是否安装：`pip install plotly`
2. 不必须：可以禁用 `--visualize` 继续评估（仅输出指标）

## 性能预期

| 配置 | 耗时 | 备注 |
|------|------|------|
| 单模型 + 5 样本 | ~30 秒 | 快速验证 |
| 单模型 + 100 样本 | ~2 分钟 | 快速评估 |
| 5 模型 + 100 样本 | ~10 分钟 | 中等规模 |
| 24 模型 + 5000+ 样本 | 1-2 小时 | 完整评估 |

*基于 GPU (CUDA) 计算，CPU 会慢 5-10 倍*

## 扩展和定制

### 修改评估指标

编辑 `compute_metrics()` 方法添加自定义指标（如 MAPE、相对误差等）

### 修改模型加载逻辑

如需从其他来源加载模型，修改 `discover_models_in_tool()` 或添加新的模型加载函数

### 集成其他评估数据集

在 `--test_dir` 中添加新目录路径（逗号分隔）：

```cmd
--test_dir "dir1,dir2,dir3,dir4,..."
```

## 常见问题（FAQ）

**Q: 脚本支持哪些模型格式？**
A: 目前支持 PyTorch checkpoint（`.pth`），自动从 checkpoint 推断模型参数（hidden_dim、num_layers、是否双向等）

**Q: 可以评估非 GRU 模型吗？**
A: 需要确保模型兼容 `EnhancedGRUModel` 接口。如需支持其他架构，需修改 `EnhancedInference` 类

**Q: 如何并行评估多个模型加速？**
A: 目前脚本顺序评估。若要并行，可修改主循环使用 `multiprocessing` 或 `joblib`

**Q: 评估结果中哪个指标最重要？**
A: 建议关注 RMSE（均方根误差）和 MAE（平均绝对误差），它们最直观反映预测误差。MAPE（百分比误差）适合相对比较

## 联系与反馈

如有问题或建议，请检查日志输出或参考脚本内的注释。

---

**最后更新**：2025-12-29
