#!/usr/bin/env python3
"""推理 + 可视化脚本 - 显示预测轨迹和误差分析"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent / 'drone_path_predictor_ros-main'))

from drone_path_predictor_ros.trajectory_predictor_gru import PredictorGRU

# 文件路径
model_path = r"D:\Trajectory prediction\drone_trajectories\gru_models_synth\position_synth_short_best_model.pth"
stats_path = r"D:\Trajectory prediction\drone_trajectories\gru_models_synth\position_synth_short_norm_stats.npz"
data_path = r"D:\Trajectory prediction\drone_trajectories\random_traj_100ms\circle_2.txt"

print("="*70)
print("GRU 轨迹预测推理 + 可视化")
print("="*70)

# 加载轨迹
print(f"\n1. 加载轨迹数据: {data_path}")
data = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=float)
if data.ndim == 1:
    data = data.reshape(1, -1)
trajectory = data[:, 1:4]  # 取 tx, ty, tz
print(f"   轨迹形状: {trajectory.shape}")

# 创建预测器
print(f"\n2. 创建预测器")
predictor = PredictorGRU(
    position_model_path=model_path,
    position_stats_file=stats_path,
    velocity_model_path=model_path,
    velocity_stats_file=stats_path,
    pos_hidden_dim=64,
    pos_num_layers=2,
    pos_dropout=0.5
)

# 进行预测
print(f"\n3. 执行预测")
predictions = predictor.predict_positions(trajectory, input_length=20)
print(f"   预测形状: {predictions.shape}")

# 准备对齐数据
actual_future = trajectory[-10:]
alignment_point = actual_future[0]
pred_aligned = predictions + (alignment_point - predictions[0])

# 计算误差
errors = np.linalg.norm(pred_aligned - actual_future, axis=1)
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))

print(f"\n4. 误差统计")
print(f"   MAE: {mae:.6f} m")
print(f"   RMSE: {rmse:.6f} m")
print(f"   每步误差: {[f'{e:.6f}' for e in errors]}")

# ============ 开始绘图 ============
print(f"\n5. 生成可视化图表...")

fig = plt.figure(figsize=(18, 12))

# ========== 子图 1: 完整轨迹 3D ==========
ax1 = fig.add_subplot(2, 3, 1, projection='3d')

# 绘制历史轨迹 (前100个点)
history_len = min(100, len(trajectory) - 10)
history_traj = trajectory[:history_len]
ax1.plot(history_traj[:, 0], history_traj[:, 1], history_traj[:, 2], 
         'b-', linewidth=2, label='History', alpha=0.7)

# 绘制末尾实际轨迹 (最后10个点)
ax1.plot(actual_future[:, 0], actual_future[:, 1], actual_future[:, 2], 
         'g-o', linewidth=3, markersize=8, label='Actual Future (10 steps)', alpha=0.9)

# 绘制预测轨迹
ax1.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2], 
         'r--s', linewidth=3, markersize=8, label='Predicted Future (10 steps)', alpha=0.9)

# 标记起点和终点
ax1.scatter(*alignment_point, color='black', s=200, marker='o', label='Start Point', zorder=10)
ax1.scatter(*actual_future[-1], color='green', s=200, marker='s', label='Actual End', zorder=10)
ax1.scatter(*pred_aligned[-1], color='red', s=200, marker='^', label='Predicted End', zorder=10)

ax1.set_xlabel('X (m)', fontweight='bold')
ax1.set_ylabel('Y (m)', fontweight='bold')
ax1.set_zlabel('Z (m)', fontweight='bold')
ax1.set_title('3D 轨迹对比\n(历史 + 实际未来 + 预测未来)', fontweight='bold', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ========== 子图 2: X 坐标对比 ==========
ax2 = fig.add_subplot(2, 3, 2)
time_steps = np.arange(1, 11)
ax2.plot(time_steps, actual_future[:, 0], 'g-o', linewidth=2.5, markersize=8, label='Actual', alpha=0.8)
ax2.plot(time_steps, pred_aligned[:, 0], 'r--s', linewidth=2.5, markersize=8, label='Predicted', alpha=0.8)
ax2.fill_between(time_steps, actual_future[:, 0], pred_aligned[:, 0], alpha=0.2, color='orange')
ax2.set_xlabel('Prediction Step', fontweight='bold')
ax2.set_ylabel('X (m)', fontweight='bold')
ax2.set_title('X 坐标预测对比', fontweight='bold', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=9)

# ========== 子图 3: Y 坐标对比 ==========
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(time_steps, actual_future[:, 1], 'g-o', linewidth=2.5, markersize=8, label='Actual', alpha=0.8)
ax3.plot(time_steps, pred_aligned[:, 1], 'r--s', linewidth=2.5, markersize=8, label='Predicted', alpha=0.8)
ax3.fill_between(time_steps, actual_future[:, 1], pred_aligned[:, 1], alpha=0.2, color='orange')
ax3.set_xlabel('Prediction Step', fontweight='bold')
ax3.set_ylabel('Y (m)', fontweight='bold')
ax3.set_title('Y 坐标预测对比', fontweight='bold', fontsize=11)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=9)

# ========== 子图 4: Z 坐标对比 ==========
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(time_steps, actual_future[:, 2], 'g-o', linewidth=2.5, markersize=8, label='Actual', alpha=0.8)
ax4.plot(time_steps, pred_aligned[:, 2], 'r--s', linewidth=2.5, markersize=8, label='Predicted', alpha=0.8)
ax4.fill_between(time_steps, actual_future[:, 2], pred_aligned[:, 2], alpha=0.2, color='orange')
ax4.set_xlabel('Prediction Step', fontweight='bold')
ax4.set_ylabel('Z (m)', fontweight='bold')
ax4.set_title('Z 坐标预测对比', fontweight='bold', fontsize=11)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(fontsize=9)

# ========== 子图 5: 点对点误差 ==========
ax5 = fig.add_subplot(2, 3, 5)
ax5.bar(time_steps, errors, color='coral', alpha=0.8, edgecolor='darkred', linewidth=2)
ax5.axhline(y=mae, color='blue', linestyle='--', linewidth=2, label=f'MAE={mae:.6f}m')
ax5.set_xlabel('Prediction Step', fontweight='bold')
ax5.set_ylabel('Error (m)', fontweight='bold')
ax5.set_title('每步预测误差', fontweight='bold', fontsize=11)
ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
ax5.legend(fontsize=9)

# ========== 子图 6: 误差统计信息 ==========
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

stats_text = f"""
预测性能统计

模型配置:
  • 隐藏层维度: 64
  • GRU 层数: 2
  • Dropout: 0.5
  • 输入序列长度: 20
  • 预测步数: 10

误差指标:
  • MAE (平均误差): {mae:.6f} m
  • RMSE: {rmse:.6f} m
  • 最大误差: {np.max(errors):.6f} m
  • 最小误差: {np.min(errors):.6f} m

轨迹数据:
  • 轨迹总长度: {len(trajectory)} 点
  • 输入片段: {trajectory[-20:-10]}
  • 实际未来: {actual_future}
  • 预测未来: {pred_aligned}
"""

ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(r'D:\Trajectory prediction\prediction_visualization.png', dpi=150, bbox_inches='tight')
print("   保存图表: D:\\Trajectory prediction\\prediction_visualization.png")

plt.show()

# ============ 第二个图：更详细的分析 ============
fig2 = plt.figure(figsize=(16, 10))

# ========== 子图 1: 累积误差 ==========
ax1 = fig2.add_subplot(2, 3, 1)
cumsum_errors = np.cumsum(errors)
ax1.plot(time_steps, cumsum_errors, 'ro-', linewidth=2.5, markersize=8)
ax1.fill_between(time_steps, 0, cumsum_errors, alpha=0.3, color='red')
ax1.set_xlabel('Prediction Step', fontweight='bold')
ax1.set_ylabel('Cumulative Error (m)', fontweight='bold')
ax1.set_title('累积预测误差', fontweight='bold', fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')

# ========== 子图 2: 欧氏距离分解 ==========
ax2 = fig2.add_subplot(2, 3, 2)
errors_x = np.abs(actual_future[:, 0] - pred_aligned[:, 0])
errors_y = np.abs(actual_future[:, 1] - pred_aligned[:, 1])
errors_z = np.abs(actual_future[:, 2] - pred_aligned[:, 2])

width = 0.25
ax2.bar(time_steps - width, errors_x, width, label='X error', alpha=0.8, color='red')
ax2.bar(time_steps, errors_y, width, label='Y error', alpha=0.8, color='green')
ax2.bar(time_steps + width, errors_z, width, label='Z error', alpha=0.8, color='blue')
ax2.set_xlabel('Prediction Step', fontweight='bold')
ax2.set_ylabel('Error (m)', fontweight='bold')
ax2.set_title('各轴向预测误差', fontweight='bold', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# ========== 子图 3: XY 平面投影 ==========
ax3 = fig2.add_subplot(2, 3, 3)
ax3.plot(actual_future[:, 0], actual_future[:, 1], 'g-o', linewidth=2.5, markersize=8, label='Actual', alpha=0.8)
ax3.plot(pred_aligned[:, 0], pred_aligned[:, 1], 'r--s', linewidth=2.5, markersize=8, label='Predicted', alpha=0.8)

# 连接对应的点
for i in range(len(actual_future)):
    ax3.plot([actual_future[i, 0], pred_aligned[i, 0]], 
            [actual_future[i, 1], pred_aligned[i, 1]], 
            'k--', alpha=0.3, linewidth=1)

ax3.set_xlabel('X (m)', fontweight='bold')
ax3.set_ylabel('Y (m)', fontweight='bold')
ax3.set_title('XY 平面投影', fontweight='bold', fontsize=11)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=9)
ax3.set_aspect('equal', adjustable='box')

# ========== 子图 4: XZ 平面投影 ==========
ax4 = fig2.add_subplot(2, 3, 4)
ax4.plot(actual_future[:, 0], actual_future[:, 2], 'g-o', linewidth=2.5, markersize=8, label='Actual', alpha=0.8)
ax4.plot(pred_aligned[:, 0], pred_aligned[:, 2], 'r--s', linewidth=2.5, markersize=8, label='Predicted', alpha=0.8)

for i in range(len(actual_future)):
    ax4.plot([actual_future[i, 0], pred_aligned[i, 0]], 
            [actual_future[i, 2], pred_aligned[i, 2]], 
            'k--', alpha=0.3, linewidth=1)

ax4.set_xlabel('X (m)', fontweight='bold')
ax4.set_ylabel('Z (m)', fontweight='bold')
ax4.set_title('XZ 平面投影', fontweight='bold', fontsize=11)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(fontsize=9)
ax4.set_aspect('equal', adjustable='box')

# ========== 子图 5: YZ 平面投影 ==========
ax5 = fig2.add_subplot(2, 3, 5)
ax5.plot(actual_future[:, 1], actual_future[:, 2], 'g-o', linewidth=2.5, markersize=8, label='Actual', alpha=0.8)
ax5.plot(pred_aligned[:, 1], pred_aligned[:, 2], 'r--s', linewidth=2.5, markersize=8, label='Predicted', alpha=0.8)

for i in range(len(actual_future)):
    ax5.plot([actual_future[i, 1], pred_aligned[i, 1]], 
            [actual_future[i, 2], pred_aligned[i, 2]], 
            'k--', alpha=0.3, linewidth=1)

ax5.set_xlabel('Y (m)', fontweight='bold')
ax5.set_ylabel('Z (m)', fontweight='bold')
ax5.set_title('YZ 平面投影', fontweight='bold', fontsize=11)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.legend(fontsize=9)
ax5.set_aspect('equal', adjustable='box')

# ========== 子图 6: 误差分布直方图 ==========
ax6 = fig2.add_subplot(2, 3, 6)
ax6.hist(errors, bins=5, color='skyblue', edgecolor='black', alpha=0.7)
ax6.axvline(x=mae, color='red', linestyle='--', linewidth=2, label=f'MAE={mae:.6f}m')
ax6.axvline(x=rmse, color='orange', linestyle='--', linewidth=2, label=f'RMSE={rmse:.6f}m')
ax6.set_xlabel('Error (m)', fontweight='bold')
ax6.set_ylabel('频率', fontweight='bold')
ax6.set_title('误差分布', fontweight='bold', fontsize=11)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig(r'D:\Trajectory prediction\prediction_analysis.png', dpi=150, bbox_inches='tight')
print("   保存图表: D:\\Trajectory prediction\\prediction_analysis.png")

plt.show()

print("\n" + "="*70)
print("推理 + 可视化完成！")
print("="*70)
print("\n生成的图表:")
print("  1. prediction_visualization.png - 主要对比图")
print("  2. prediction_analysis.png - 详细分析图")
