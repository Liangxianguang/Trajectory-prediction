# 消融实验推理与可视化 - 快速指南

## 概述

本文件夹包含完整的消融实验推理、可视化和论文级图表生成流程。

**三个关键脚本：**

1. **infer_and_visualize_ablation.py** - 推理 + 边绘制个别样本对比图
2. **generate_paper_figures.py** - 生成论文级总结图表
3. **run_inference_and_paper_figures.bat** - 一键执行所有步骤

## 使用方式

### 方式1：Windows批处理（最简单 ⭐）

```bash
run_inference_and_paper_figures.bat
```

这将：
- 推理50个随机样本，生成50张对比图
- 生成5种论文级总结图表
- 所有结果保存到 `ablation_results_final/`

### 方式2：分步执行

**Step 1：推理和可视化**

```bash
python infer_and_visualize_ablation.py \
    --data_dir ../swarm_segments \
    --ablation_dir . \
    --output_dir my_results \
    --num_samples 30 \
    --seed 42
```

**Step 2：生成论文图表**

```bash
python generate_paper_figures.py \
    --ablation_dir . \
    --inference_results my_results/summary.json \
    --output_dir paper_figures
```

### 方式3：自定义指定样本

```bash
# 指定特定的样本索引
python infer_and_visualize_ablation.py \
    --data_dir ../swarm_segments \
    --sample_indices "100,500,1000,2000,5000" \
    --output_dir custom_samples
```

## 输出文件说明

### 推理阶段输出

```
my_results/
├── sample_000100_comparison.png    # 样本#100的5模型对比图
├── sample_000500_comparison.png    # 样本#500的5模型对比图
├── sample_001000_comparison.png    # ...
├── ...
└── summary.json                    # 汇总统计
```

**每张对比图包含：**
- 3D轨迹对比（5个模型 + 真实轨迹）
- XY/XZ平面投影
- 时间步误差曲线
- 总体MAE柱状图对比
- 性能指标表

### 论文图表输出

```
paper_figures/
├── training_curves_comparison.png  # 4个子图：train loss, val loss, val MAE, LR
├── best_metrics_summary.png        # 最佳指标对比表
├── improvement_analysis.png        # 绝对MAE + 相对改进%
└── training_vs_inference.png       # 训练vs推理性能对比
```

## 参数说明

### infer_and_visualize_ablation.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `../swarm_segments` | 验证数据目录 |
| `--ablation_dir` | `.` | 消融实验结果目录（包含5个exp输出目录） |
| `--output_dir` | `ablation_viz_results` | 输出目录 |
| `--num_samples` | `10` | 随机抽取样本数 |
| `--sample_indices` | `None` | 指定样本索引（逗号分隔，如"100,500"） |
| `--batch_size` | `256` | 推理批次大小 |
| `--seed` | `42` | 随机种子 |
| `--device` | `cuda:0` | GPU设备 |

### generate_paper_figures.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ablation_dir` | `.` | 消融实验结果目录 |
| `--inference_results` | `None` | 推理结果summary.json文件路径 |
| `--output_dir` | `paper_figures` | 输出目录 |

## 常见用法

### 用法1：快速验证（推荐第一次运行）

```bash
# 仅推理10个样本（快速）
python infer_and_visualize_ablation.py --num_samples 10

# 生成论文图表
python generate_paper_figures.py --inference_results ablation_viz_results/summary.json
```

### 用法2：详细分析（推论文）

```bash
# 推理100个样本进行详细分析
python infer_and_visualize_ablation.py --num_samples 100

# 生成论文级图表
python generate_paper_figures.py --inference_results ablation_viz_results/summary.json
```

### 用法3：特定样本分析

```bash
# 分析特定的样本（如异常样本）
python infer_and_visualize_ablation.py --sample_indices "2504,17995,33018"

# 生成论文图表
python generate_paper_figures.py --inference_results ablation_viz_results/summary.json
```

### 用法4：论文制作（完整流程）

```bash
# 1. 一键运行所有步骤
run_inference_and_paper_figures.bat

# 2. 查看生成的图表
cd ablation_results_final/paper_figures
# 使用以下图表用于论文：
#   - training_curves_comparison.png
#   - best_metrics_summary.png
#   - improvement_analysis.png
#   - training_vs_inference.png

# 3. 可选：查看个别样本对比
cd ../samples_comparison
# 查看任意sample_*.png文件
```

## 图表说明

### 1. training_curves_comparison.png

4个子图：
- **左上**：训练损失 - 展示5个模型的训练过程
- **右上**：验证损失 - 关键指标，越低越好
- **左下**：验证MAE - 直接的预测误差
- **右下**：学习率调度 - 学习率变化过程

**用途**：论文"实验设置"章节

### 2. best_metrics_summary.png

表格形式展示：
- 各模型的最佳验证损失和对应epoch
- 最佳MAE和对应epoch
- 最终的训练/验证损失

**用途**：论文"实验结果"表格

### 3. improvement_analysis.png

两个对比图：
- **左图**：各模型的绝对MAE值（柱状图）
- **右图**：各模型相对于基线(E1)的改进百分比

**用途**：论文"消融分析"章节，直观展示各创新点的贡献

### 4. training_vs_inference.png

对比图：
- 蓝色柱：训练阶段最佳验证MAE
- 橙色柱：推理阶段平均MAE

**用途**：论文"泛化性能"验证

### 5. sample_*_comparison.png（个别样本）

每个样本的完整分析：
- 3D轨迹和各平面投影
- 5个模型的预测对比
- 时间步误差分析
- 性能指标对比

**用途**：论文附录或汇报演示

## 数据流

```
验证数据集 (46046 samples)
    ↓
随机抽取 50 个样本
    ↓
加载5个消融实验模型
    ↓
对每个样本推理 5 个模型
    ↓
计算MAE/RMSE等指标
    ↓
绘制3D对比图 (sample_*.png)
    ↓
汇总统计结果 (summary.json)
    ↓
生成论文级总结图表 (paper_figures/*.png)
```

## 常见问题

**Q: 推理慢吗？**

A: 50个样本约需30-60秒（GPU上）。如果需要更快，可降低`--num_samples`。

**Q: 可以推理全部数据吗？**

A: 可以，但46K样本推理会很慢。建议：
- 快速预览：10-50个样本
- 详细分析：100-500个样本
- 完整推理：1000+个样本（需要时间）

**Q: 图表质量如何？**

A: 所有图表使用300 DPI生成，可直接用于论文/报告。

**Q: 如何修改图表样式？**

A: 编辑 `generate_paper_figures.py` 中的COLORS和LABELS变量，或修改matplotlib设置。

**Q: 可以导出为PDF吗？**

A: 可以。在matplotlib中添加：
```python
plt.savefig('figure.pdf', format='pdf', dpi=300, bbox_inches='tight')
```

## 故障排查

### 错误：KeyError: 'y_orig'

✗ **原因**：数据加载格式不匹配

✓ **解决**：确保使用 `ablation_train_utils.py` 的数据加载函数

### 错误：模型加载失败

✗ **原因**：模型文件路径不正确

✓ **解决**：检查 `ablation_results_agents_3_*/best_model_*.pt` 文件是否存在

### 错误：GPU内存不足

✗ **原因**：批次大小太大

✓ **解决**：降低 `--batch_size` 参数，例如 `--batch_size 64`

### 图表空白

✗ **原因**：训练历史文件未找到

✓ **解决**：确保 `ablation_results_agents_3_**/training_history_*.csv` 存在

## 后续步骤

1. **验证结果**
   - 查看individual sample对比图确保模型预测合理
   - 检查improvement分析确认消融实验的有效性

2. **论文撰写**
   - 使用paper_figures中的图表
   - 参考best_metrics_summary生成表格

3. **性能优化（可选）**
   - 基于improvement_analysis确定最有效的创新点
   - 考虑进一步的模型改进

4. **汇报演示（可选）**
   - 使用sample对比图做演示
   - 强调improvement_analysis中的改进百分比

---

更新时间：2026-01-28

有问题请检查脚本输出的错误信息，或参考父目录的README文件。
