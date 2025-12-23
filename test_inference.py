#!/usr/bin/env python3
"""测试推理脚本 - 直接调用新的 GRU 预测器"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'drone_path_predictor_ros-main'))

from drone_path_predictor_ros.trajectory_predictor_gru import PredictorGRU

# 文件路径
model_path = r"D:\Trajectory prediction\drone_trajectories\gru_models\position_short_best_model.pth"
stats_path = r"D:\Trajectory prediction\drone_trajectories\gru_models\position_short_norm_stats.npz"
data_path = r"D:\Trajectory prediction\drone_trajectories\random_traj_100ms\line_1.txt"

print("="*70)
print("GRU 轨迹预测推理测试")
print("="*70)

# 验证文件存在
print(f"\n1. 验证文件存在")
print(f"   模型文件: {Path(model_path).exists()} - {model_path}")
print(f"   统计量文件: {Path(stats_path).exists()} - {stats_path}")
print(f"   轨迹文件: {Path(data_path).exists()} - {data_path}")

# 加载轨迹
print(f"\n2. 加载轨迹数据")
data = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=float)
if data.ndim == 1:
    data = data.reshape(1, -1)
trajectory = data[:, 1:4]  # 取 tx, ty, tz
print(f"   轨迹形状: {trajectory.shape}")
print(f"   前3个点: {trajectory[:3]}")
print(f"   后3个点: {trajectory[-3:]}")

# 创建预测器
print(f"\n3. 创建预测器")
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
print(f"\n4. 执行预测")
predictions = predictor.predict_positions(trajectory, input_length=20)
print(f"   预测形状: {predictions.shape}")
print(f"\n   预测的未来10步:")
for i, p in enumerate(predictions):
    print(f"   Step {i+1:2d}: ({p[0]:8.3f}, {p[1]:8.3f}, {p[2]:8.3f})")

# 计算误差 (用末尾数据模拟真实未来)
print(f"\n5. 误差评估 (用末尾10个点作为真实未来)")
actual_future = trajectory[-10:]
alignment_point = actual_future[0]
pred_aligned = predictions + (alignment_point - predictions[0])
errors = np.linalg.norm(pred_aligned - actual_future, axis=1)

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))

print(f"   MAE: {mae:.6f} m")
print(f"   RMSE: {rmse:.6f} m")
print(f"   每步误差: {[f'{e:.6f}' for e in errors]}")

print("\n" + "="*70)
print("推理测试完成！")
print("="*70)
