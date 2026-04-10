# 4×4 场景对比图生成指南

## 概述

本指南说明如何生成一个 **4×4 的发布级对比图**，展示4个特定场景各自的4个观测维度。

```
┌─────────────────────────────────────────────────────────────┐
│                    4×4 布局结构                              │
├──────────┬──────────────┬──────────────┬──────────────┤
│ 场景     │ 3D 全景      │ XY 平面      │ XZ 平面      │ YZ 平面
├──────────┼──────────────┼──────────────┼──────────────┤
│ 行 1: S1 │ 轨迹交叉     │   -          │   -          │   -
│ 场景分类 │ (Sample      │              │              │
│ (S1)     │  20280)      │              │              │
├──────────┼──────────────┼──────────────┼──────────────┤
│ 行 2: S2 │ 急转弯       │   -          │   -          │   -
│ 场景分类 │ (Sample      │              │              │
│ (S2)     │  173142)     │              │              │
├──────────┼──────────────┼──────────────┼──────────────┤
│ 行 3: S3 │ 垂直爬升     │   -          │   -          │   -
│ 场景分类 │ (Sample      │              │              │
│ (S3)     │  212515)     │              │              │
├──────────┼──────────────┼──────────────┼──────────────┤
│ 行 4: S4 │ S形轨迹      │   -          │   -          │   -
│ 场景分类 │ (Sample 33)  │              │              │
└──────────┴──────────────┴──────────────┴──────────────┘
```

## 场景定义

### S1: 复杂交互场景中的空间建模能力 (Sample #20280)
- **特点**: 多个智能体轨迹在3D空间中交叉
- **评估维度**: 模型是否准确预测轨迹的空间分布和相对位置
- **展示维度**:
  - 3D View: 全景轨迹交叉情况
  - XY Plane: 水平平面的交互轨迹
  - XZ Plane: 侧视图的高度变化
  - YZ Plane: 前视图的侧向变化

### S2: 高曲率机动中的物理一致性 (Sample #173142)
- **特点**: 三个智能体协同执行急转弯机动
- **评估维度**: 模型是否遵守物理约束并预测一致的转向轨迹
- **展示维度**:
  - 3D View: 转弯的三维形状
  - XY Plane: 转弯的水平投影
  - XZ Plane: 高度维度的变化
  - YZ Plane: 侧向转向的投影

### S3: 三维机动场景中的高度预测能力 (Sample #212515)
- **特点**: 智能体执行快速垂直爬升机动
- **评估维度**: 模型是否准确预测Z轴（高度）的动态变化
- **展示维度**:
  - 3D View: 垂直爬升的立体效果
  - XY Plane: 水平位移（应较小）
  - XZ Plane: **关键** - 垂直爬升的侧视图
  - YZ Plane: 高度随Y变化的投影

### S4: 复杂周期机动中的时序建模能力 (Sample #33)
- **特点**: 智能体执行S形（蛇形）周期性轨迹
- **评估维度**: 模型是否学会了时间序列中的周期模式
- **展示维度**:
  - 3D View: S形的螺旋结构
  - XY Plane: S形的水平投影
  - XZ Plane: S形的高度变化
  - YZ Plane: S形的侧向投影

## 快速开始

### 方式 1: 使用批处理脚本（推荐）

```batch
cd D:\Trajectory prediction\drone_trajectories\GRUTrajectoryPredictor
generate_4x4_scenarios.bat
```

此脚本将自动：
1. **步骤 1**: 运行 `compare_four_models_image.py` 处理 4 个关键样本
   - 生成 4 个单个样本的对比图（可选）
   - 保存所有预测数据到 JSON 文件

2. **步骤 2**: 运行 `generate_4x4_comparison.py` 生成最终大图
   - 从 JSON 中提取 4 个场景的轨迹数据
   - 组合成 4×4 布局图
   - 输出高质量 PNG 图像

### 方式 2: 手动分步运行

#### 步骤 1: 生成预测结果和 JSON 数据

```bash
python compare_four_models_image.py ^
    --data_dir "D:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments" ^
    --agents 3 ^
    --use_subset ^
    --lbebm_model "D:\Trajectory prediction\drone_trajectories\3DMoTraj\saved_models\checkpoints_accfix\epoch_030.pt" ^
    --exp5_dir "D:\Trajectory prediction\drone_trajectories\Cluster trajectory\ablation study\ablation_results_agents_3_exp5_full" ^
    --mrgraj_model "D:\Trajectory prediction\drone_trajectories\MRGTraj-main\checkpoints_lbebm3d\agents_3_lbebm3d_inspired\best_model.pth" ^
    --gru_model "D:\Trajectory prediction\drone_trajectories\GRUTrajectoryPredictor\checkpoints\agents_3_20260309_141203\epoch_190.pth" ^
    --features_32d_dir "D:\Trajectory prediction\drone_trajectories\Cluster trajectory\features_32d" ^
    --output_dir "comparison_four_models" ^
    --sample_indices "20280,173142,212515,33" ^
    --no_visualize
```

**关键参数:**
- `--sample_indices "20280,173142,212515,33"` - 指定 4 个关键场景
- `--no_visualize` - 只生成数据和 JSON，不生成单个样本图（节省时间）

**输出:**
- `comparison_four_models/comparison_summary.json` - 包含所有轨迹数据
- `comparison_four_models/comparison_summary.csv` - 数值汇总

#### 步骤 2: 生成 4×4 对比图

```bash
python generate_4x4_comparison.py ^
    --json_results "comparison_four_models/comparison_summary.json" ^
    --output_dir "comparison_4x4_scenarios" ^
    --output_name "4x4_scenario_comparison.png"
```

**输出:**
- `comparison_4x4_scenarios/4x4_scenario_comparison.png` - 最终 4×4 大图 (150 DPI)

## 输出文件

### 单个样本对比图（可选）
```
comparison_four_models/
├── sample_020280_comparison_publication.png  (S1 场景)
├── sample_173142_comparison_publication.png  (S2 场景)
├── sample_212515_comparison_publication.png  (S3 场景)
└── sample_000033_comparison_publication.png  (S4 场景)
```

### 4×4 对比图
```
comparison_4x4_scenarios/
└── 4x4_scenario_comparison.png  (最终发布级大图)
```

## 图像质量

- **分辨率**: 44" × 36" 逻辑尺寸 (DPI: 150)
- **最终尺寸**: ~6600 × 5400 像素（可根据需求调整 DPI）
- **颜色方案**:
  - 黑色（#000000）: Ground Truth
  - 蓝色（#0078FF）: Ours (SwarmGRU) - **实线**
  - 橙色（#FF9500）: MRGTraj - **虚线**
  - 绿色（#53F50E）: 3DMoTraj - **虚线**
  - 紫色（#9933CC）: VECTOR - **虚线**
  - 灰色（#F75A5A）: History Trajectory

- **线条样式**:
  - 实线: GT、Ours
  - 虚线: 所有基线方法

## 自定义选项

### 修改场景定义

编辑 `generate_4x4_comparison.py` 中的 `SCENARIO_CONFIG`:

```python
SCENARIO_CONFIG = {
    'S1': {'sample_idx': 20280, 'title': 'S1: 场景标题', 'desc': 'Description'},
    'S2': {'sample_idx': 173142, 'title': 'S2: 场景标题', 'desc': 'Description'},
    # ... 等等
}
```

### 调整图像尺寸

编辑 `generate_4x4_comparison.py` 的 `create_4x4_layout` 函数:

```python
fig = plt.figure(figsize=(44, 36), facecolor='white', dpi=100)  # 修改 figsize 或 dpi
```

### 调整布局间距

编辑 `GridSpec` 参数:

```python
gs = fig.add_gridspec(4, 4, left=0.06, right=0.96, top=0.94, bottom=0.06,
                     hspace=0.4, wspace=0.35)  # 调整 hspace 和 wspace
```

## 故障排除

### 问题 1: "JSON 文件不存在"
**原因**: 步骤 1 没有成功完成
**解决**: 
1. 检查所有模型路径是否正确
2. 检查是否有足够的磁盘空间
3. 查看完整的错误日志

### 问题 2: 某个场景的轨迹数据缺失
**原因**: 该样本在数据集中不存在或模型预测失败
**解决**:
1. 验证样本索引是否正确
2. 尝试运行 `compare_four_models_image.py` 处理所有样本，找到有效的场景

### 问题 3: 图像输出模糊或颜色不对
**原因**: DPI 设置或 colormap 问题
**解决**:
1. 增加 `dpi` 参数
2. 检查图形保存时是否有警告信息

## 配置参数详解

### compare_four_models_image.py

| 参数 | 类型 | 说明 |
|------|------|------|
| `--sample_indices` | str | 逗号分隔的样本索引，用于 4×4 图 |
| `--no_visualize` | flag | 跳过单个样本可视化，仅生成数据 |
| `--disable_insets` | flag | 禁用缩放区域 |
| 其他 | - | 同原脚本文档 |

### generate_4x4_comparison.py

| 参数 | 类型 | 说明 |
|------|------|------|
| `--json_results` | str | **必须** - JSON 结果文件路径 |
| `--output_dir` | str | 输出目录（默认: comparison_4x4_scenarios） |
| `--output_name` | str | 输出文件名（默认: 4x4_scenario_comparison.png） |

## 预期输出示例

最终的 4×4 图应该显示：

- **S1 行**: 轨迹交叉场景的 4 个视角，清晰显示 3 个智能体的交叉点
- **S2 行**: 急转弯场景的 4 个视角，XY 平面显示明显的 U 形转向
- **S3 行**: 垂直爬升场景的 4 个视角，**XZ 平面显示明显的垂直上升**
- **S4 行**: S 形轨迹场景的 4 个视角，所有平面显示周期性的 S 形图案

每个子图中：
- **黑线**: Ground Truth（参考轨迹）
- **蓝线**: Ours（我们的模型预测）
- **虚线**: 基线方法（MRGTraj、3DMoTraj、VECTOR）

## 论文发布建议

1. **导出高分辨率版本**:
   ```python
   # 修改 dpi=300 以获得更高分辨率
   plt.savefig(..., dpi=300, ...)
   ```

2. **调整为出版商要求**:
   - 检查图像宽度是否符合期刊要求（通常 3 列宽 = ~7-8 英寸）
   - 调整字体大小以保持可读性

3. **添加子图标签**:
   ```python
   # 在每个子图的左上角添加 (a), (b), (c), (d) 等标签
   ```

4. **补充信息**:
   - 在图下方添加数值统计（ADE/FDE）
   - 在图例中添加模型的论文引用

---

## 相关文件

- `compare_four_models_image.py` - 主对比脚本（已修改以保存 JSON）
- `generate_4x4_comparison.py` - 4×4 图生成脚本
- `generate_4x4_scenarios.bat` - 完整流程批处理脚本
- `run_compare_4x4.bat` - 单个四模型对比脚本

## 更新日志

### v1.0 (2026-03-17)
- ✓ 创建 4×4 对比图的完整脚本
- ✓ 支持 4 个关键场景和 4 个观测维度
- ✓ 自动从 JSON 结果提取轨迹数据
- ✓ 发布级质量输出 (150 DPI)
