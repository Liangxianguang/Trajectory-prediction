# 🎨 论文级图表绘制指南

## 📊 你的现状

✅ **已完成：**
- ✅ 2000个样本的评估完成
- ✅ 详细指标数据已保存：`comparison_summary.json`
- ✅ 单个样本的可视化已生成（1000+张PNG）

## 🚀 接下来如何操作

### 步骤 1️⃣ - 运行论文级图表生成脚本

打开命令行，进入项目目录：

```cmd
cd "d:\Trajectory prediction\drone_trajectories\Cluster trajectory"
```

运行图表生成脚本：

```cmd
python plot_lbebm_vs_gnn_comparison.py ^
    --summary_json comparison_results_1w_lbebm_vs_gnn/comparison_summary.json ^
    --output_dir comparison_results_1w_lbebm_vs_gnn/paper_figures
```

或者使用批处理文件（一行命令）：

```cmd
cd "d:\Trajectory prediction"
generate_paper_figures.bat
```

**预计耗时：** 2-3 分钟

### 步骤 2️⃣ - 查看生成的图表

生成的文件将保存在：
```
comparison_results_1w_lbebm_vs_gnn/paper_figures/
```

生成的图表包括：
- ✅ `overall_comparison.png` - **整体性能对比（最重要！论文主图）**
- ✅ `axis_mae_comparison.png` - 轴向误差对比（X/Y/Z）
- ✅ `mae_boxplot_comparison.png` - 误差分布箱线图
- ✅ `per_agent_mae_comparison.png` - 单体性能对比（3个Agent）
- ✅ `error_trend_comparison.png` - 误差随时间步的趋势
- ✅ `comparison_table.txt` - 详细指标表（用于论文表格）

### 步骤 3️⃣ - 查看汇总统计信息

也可以直接查看统计摘要：

```cmd
python view_comparison_summary.py
```

输出示例：
```
📊 评估样本数: 2000

指标              LBEBM3D              GNN+BiGRU            改进
----------------------------------------------------------------------
ADE             0.117238             0.115093             ↓   1.83%
FDE             0.212343             0.252238             ↑  18.79%
RMSE            0.130422             0.141226             ↑   8.28%
```

---

## 📋 图表详细说明

### 1. **overall_comparison.png** ⭐⭐⭐
**这是你论文的主图表！**

- 展示：ADE、FDE、RMSE、MAPE 四个指标
- 红色 = LBEBM3D
- 蓝色 = GNN+BiGRU
- 误差条 = 标准差
- 推荐位置：论文 Results 章节
- 标题建议：
  ```
  Figure X: Overall Performance Comparison (LBEBM3D vs GNN+BiGRU, n=2000)
  ```

### 2. **axis_mae_comparison.png**
- 展示：各轴向（X/Y/Z）的预测精度
- 帮助理解模型在不同维度的表现差异
- 推荐位置：论文补充材料

### 3. **mae_boxplot_comparison.png**
- 展示：2000个样本的误差分布
- 菱形 = 均值
- 方框 = 四分位数
- 反映模型的稳定性和离群值
- 推荐位置：论文 Discussion 章节

### 4. **per_agent_mae_comparison.png**
- 展示：3个无人机（Agent）的性能
- 检验模型是否存在Agent偏差
- 推荐位置：论文补充材料

### 5. **error_trend_comparison.png**
- 展示：误差如何随预测步长增长
- 判断长期预测能力
- 推荐位置：论文 Analysis 章节

### 6. **comparison_table.txt**
- 详细的指标表格（均值 ± 标准差）
- 复制内容到 LaTeX 表格
- 推荐位置：论文 Results 表格

---

## 💻 使用已有脚本的命令

### 快速查看统计摘要
```cmd
python "d:\Trajectory prediction\view_comparison_summary.py"
```

### 生成论文图表（自定义路径）
```cmd
python plot_lbebm_vs_gnn_comparison.py ^
    --summary_json "d:\Trajectory prediction\drone_trajectories\Cluster trajectory\comparison_results_1w_lbebm_vs_gnn\comparison_summary.json" ^
    --output_dir "d:\Trajectory prediction\drone_trajectories\Cluster trajectory\comparison_results_1w_lbebm_vs_gnn\paper_figures"
```

### 生成更高分辨率的图表（用于印刷）
```cmd
python plot_lbebm_vs_gnn_comparison.py ^
    --summary_json comparison_results_1w_lbebm_vs_gnn/comparison_summary.json ^
    --output_dir comparison_results_1w_lbebm_vs_gnn/paper_figures_hires ^
    --dpi 600
```

---

## 📝 在论文中使用图表

### LaTeX 嵌入示例

```latex
\subsection{Comparative Results}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/overall_comparison.png}
  \caption{Performance comparison between LBEBM3D and GNN+BiGRU 
  on 2,000 trajectory samples. Error bars represent standard deviation.}
  \label{fig:overall}
\end{figure}

The results show that LBEBM3D achieves an ADE of 0.1172 meters 
compared to GNN+BiGRU's 0.1151 meters. While GNN+BiGRU shows 1.83\% 
improvement in ADE, LBEBM3D is significantly better in FDE (18.79\% 
improvement) and RMSE (8.28\% improvement).

\begin{table}[htbp]
  \caption{Detailed Metrics Comparison (n=2000)}
  \label{tab:metrics}
  \centering
  \input{tables/comparison_table}
\end{table}
```

---

## ✅ 检查清单

运行完脚本后，检查以下文件是否都生成了：

```
□ comparison_results_1w_lbebm_vs_gnn/paper_figures/
  ├─ overall_comparison.png          ✅
  ├─ overall_comparison.pdf          ✅
  ├─ axis_mae_comparison.png         ✅
  ├─ axis_mae_comparison.pdf         ✅
  ├─ mae_boxplot_comparison.png      ✅
  ├─ mae_boxplot_comparison.pdf      ✅
  ├─ per_agent_mae_comparison.png    ✅
  ├─ per_agent_mae_comparison.pdf    ✅
  ├─ error_trend_comparison.png      ✅
  ├─ error_trend_comparison.pdf      ✅
  └─ comparison_table.txt            ✅
```

---

## 🎨 配色说明

- **LBEBM3D**: #E74C3C (深红) - 传统统计方法
- **GNN+BiGRU**: #E67E22 (橙色) - 现代深度学习方法

这两种颜色：
- ✅ 对色盲友好
- ✅ 印刷友好（黑白能分清）
- ✅ 视觉对比清晰

---

## 🔧 如果出现问题

### 问题 1：找不到 JSON 文件
```
❌ 错误: 找不到 comparison_summary.json
```

**解决方案：**
```cmd
# 检查文件是否存在
dir "d:\Trajectory prediction\drone_trajectories\Cluster trajectory\comparison_results_1w_lbebm_vs_gnn"

# 确认文件名是否正确（不要用旧的名字）
```

### 问题 2：Python 找不到模块
```
ModuleNotFoundError: No module named 'matplotlib'
```

**解决方案：**
```cmd
pip install matplotlib seaborn numpy
```

### 问题 3：权限不足无法保存
```
PermissionError: [Errno 13] Permission denied
```

**解决方案：**
- 关闭任何打开该文件夹的程序
- 或者指定其他输出目录

---

## 📞 更多帮助

所有脚本都在这些位置：

- 图表生成: `d:\Trajectory prediction\drone_trajectories\Cluster trajectory\plot_lbebm_vs_gnn_comparison.py`
- 统计查看: `d:\Trajectory prediction\view_comparison_summary.py`
- 批处理脚本: `d:\Trajectory prediction\generate_paper_figures.bat`

祝论文写作顺利！ 🎓
