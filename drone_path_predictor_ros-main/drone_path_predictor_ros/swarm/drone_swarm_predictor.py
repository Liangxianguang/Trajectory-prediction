#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# 确保能找到模块
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 尝试导入新的 GRU 预测器，回退到旧的
try:
    from trajectory_predictor_gru import Predictor
    print("使用新的 GRU 预测器 (trajectory_predictor_gru)")
except ImportError:
    from trajectory_predictor import Predictor
    print("使用旧的预测器 (trajectory_predictor)")

from pose_buffer import PoseBuffer


class DroneSwarmPredictor:
    """
    无人机群轨迹预测器，支持批量预测多架无人机的轨迹
    """
    
    def __init__(self, position_model_path, velocity_model_path, 
                 position_stats_file, velocity_stats_file,
                 pos_hidden_dim=64, pos_num_layers=2, pos_dropout=0.5,
                 vel_hidden_dim=64, vel_num_layers=2, vel_dropout=0.5,
                 use_velocity_prediction=False, use_whitening=True,
                 buffer_duration=2.0, dt=0.1,
                 prediction_smoothing_window=1, actual_smoothing_window=0):
        """
        初始化无人机群预测器
        
        Args:
            position_model_path: 位置模型路径
            velocity_model_path: 速度模型路径
            position_stats_file: 位置统计文件路径
            velocity_stats_file: 速度统计文件路径
            pos_hidden_dim: 位置模型隐藏层维度
            pos_num_layers: 位置模型层数
            pos_dropout: 位置模型dropout率
            vel_hidden_dim: 速度模型隐藏层维度
            vel_num_layers: 速度模型层数
            vel_dropout: 速度模型dropout率
            use_velocity_prediction: 是否使用速度预测
            use_whitening: 是否使用白化处理
            buffer_duration: 缓冲持续时间(秒)
            dt: 采样时间间隔(秒)
        """
        self.predictor = Predictor(
            position_model_path, velocity_model_path,
            position_stats_file, velocity_stats_file,
            pos_hidden_dim=pos_hidden_dim, pos_num_layers=pos_num_layers, 
            pos_dropout=pos_dropout,
            vel_hidden_dim=vel_hidden_dim, vel_num_layers=vel_num_layers, 
            vel_dropout=vel_dropout,
            use_whitening=use_whitening
        )
        self.use_velocity_prediction = use_velocity_prediction
        self.buffer_duration = buffer_duration
        self.dt = dt
        self.drone_buffers = {}  # 存储每架无人机的缓冲
        self.prediction_smoothing_window = max(1, prediction_smoothing_window)
        self.actual_smoothing_window = max(0, actual_smoothing_window)
        
    def add_drone(self, drone_id):
        """添加无人机到预测系统"""
        if drone_id not in self.drone_buffers:
            self.drone_buffers[drone_id] = PoseBuffer(
                buffer_duration=self.buffer_duration, 
                dt=self.dt
            )
            
    def update_drone_position(self, drone_id, position, timestamp):
        """
        更新无人机位置
        
        Args:
            drone_id: 无人机ID
            position: 位置 (x, y, z)
            timestamp: 时间戳
            
        Returns:
            bool: 更新成功
        """
        self.add_drone(drone_id)
        return self.drone_buffers[drone_id].update_buffer(position, timestamp)
    
    def predict_drone_trajectory(self, drone_id, input_length=None):
        """
        预测单架无人机的轨迹
        
        Args:
            drone_id: 无人机ID
            input_length: 输入序列长度，为None时使用缓冲区所有数据
            
        Returns:
            np.ndarray: 预测轨迹，形状为 (10, 3)，失败返回None
        """
        if drone_id not in self.drone_buffers:
            return None
            
        buffer = self.drone_buffers[drone_id]
        positions = buffer.get_regularly_sampled_positions()
        
        if input_length is None:
            input_length = len(positions)
        
        if len(positions) < input_length:
            # 如果缓冲区的数据少于请求的输入长度，降级为使用缓冲区全部数据并给出警告
            # 但至少需要 2 个点来进行速度计算/预测
            if len(positions) < 2:
                # 不足以进行预测
                return None
            print(f"Warning: drone {drone_id} buffer has {len(positions)} points < requested input_length {input_length}. Using {len(positions)} instead.")
            input_length = len(positions)

        # 获取最后input_length个位置作为输入
        input_seq = np.array(positions[-input_length:])
        
        try:
            if self.use_velocity_prediction:
                prediction = self.predictor.predict_positions_from_velocity(input_seq, self.dt)
            else:
                prediction = self.predictor.predict_positions(input_seq)

            prediction = self._smooth_prediction(prediction)
            aligned_prediction = self._align_prediction(prediction, input_seq[-1])
            return aligned_prediction
        except Exception as e:
            print(f"预测无人机 {drone_id} 轨迹失败: {e}")
            return None

    def _smooth_prediction(self, prediction):
        if self.prediction_smoothing_window <= 1 or len(prediction) < self.prediction_smoothing_window:
            return prediction
        kernel = np.ones(self.prediction_smoothing_window) / self.prediction_smoothing_window
        smoothed = np.empty_like(prediction)
        for dim in range(prediction.shape[1]):
            smoothed[:, dim] = np.convolve(prediction[:, dim], kernel, mode='same')
        return smoothed

    def _align_prediction(self, prediction, anchor_point):
        if prediction.size == 0:
            return prediction
        offset = anchor_point - prediction[0]
        return prediction + offset

    def _smooth_sequence(self, sequence, window):
        if window <= 1 or len(sequence) == 0:
            return sequence
        kernel = np.ones(window) / window
        smoothed = np.empty_like(sequence)
        for dim in range(sequence.shape[1]):
            smoothed[:, dim] = np.convolve(sequence[:, dim], kernel, mode='same')
        return smoothed

    def get_future_segment_from_trajectory(self, trajectory, future_steps):
        if future_steps <= 0 or len(trajectory) == 0:
            return None
        return trajectory[-future_steps:]

    def evaluate_prediction_quality(self, predictions, trajectories, future_steps=10):
        metrics = {}
        for drone_id, prediction in predictions.items():
            traj = trajectories.get(drone_id)
            if traj is None or len(traj) == 0:
                continue
            actual_future = self.get_future_segment_from_trajectory(traj, future_steps)
            if actual_future is None or len(actual_future) == 0:
                continue
            actual_smoothed = self._smooth_sequence(actual_future, self.actual_smoothing_window)
            alignment_point = actual_smoothed[0]
            aligned_pred = self._align_prediction(prediction, alignment_point)
            steps = min(len(actual_future), len(aligned_pred))
            actual = actual_smoothed[:steps]
            pred = aligned_pred[:steps]
            errors = np.linalg.norm(actual - pred, axis=1)
            metrics[drone_id] = {
                'mae': float(np.mean(np.abs(errors))),
                'rmse': float(np.sqrt(np.mean(errors**2))),
                'errors': errors.tolist(),
                'future_steps': steps
            }
        return metrics
    
    def predict_all_trajectories(self, input_length=None):
        """
        预测所有无人机的轨迹
        
        Args:
            input_length: 输入序列长度
            
        Returns:
            dict: {drone_id: prediction_array}
        """
        predictions = {}
        for drone_id in self.drone_buffers.keys():
            pred = self.predict_drone_trajectory(drone_id, input_length)
            if pred is not None:
                predictions[drone_id] = pred
        return predictions
    
    def compute_inter_drone_distances(self, predictions, timestamps=None):
        """
        计算无人机之间的距离时间序列
        
        Args:
            predictions: 预测字典 {drone_id: prediction_array}
            timestamps: 时间戳列表
            
        Returns:
            dict: {(drone_i, drone_j): distance_array}
        """
        distances = {}
        drone_ids = list(predictions.keys())
        
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                drone_i, drone_j = drone_ids[i], drone_ids[j]
                pred_i = predictions[drone_i]
                pred_j = predictions[drone_j]
                
                # 计算对应时间步的距离
                dists = np.linalg.norm(pred_i - pred_j, axis=1)
                distances[(drone_i, drone_j)] = dists
        
        return distances
    
    def detect_collisions(self, predictions, collision_threshold=1.0):
        """
        检测无人机碰撞
        
        Args:
            predictions: 预测字典 {drone_id: prediction_array}
            collision_threshold: 碰撞阈值(距离)
            
        Returns:
            list: [(drone_i, drone_j, min_distance, time_step), ...]
        """
        collisions = []
        distances = self.compute_inter_drone_distances(predictions)
        
        for (drone_i, drone_j), dists in distances.items():
            min_dist = np.min(dists)
            if min_dist < collision_threshold:
                time_step = np.argmin(dists)
                collisions.append((drone_i, drone_j, min_dist, time_step))
        
        return collisions
    
    def compute_prediction_metrics(self, drone_id):
        """
        获取无人机的预测评估指标
        
        Args:
            drone_id: 无人机ID
            
        Returns:
            tuple: (mse, rmse) 或 None
        """
        if drone_id not in self.drone_buffers:
            return None
        
        buffer = self.drone_buffers[drone_id]
        result = buffer.evaluate_trajectory()
        return result
    
    def set_evaluation_trajectory(self, drone_id, future_positions):
        """
        设置无人机的真实未来轨迹用于评估
        
        Args:
            drone_id: 无人机ID
            future_positions: 未来位置列表 [(x,y,z), ...]
        """
        if drone_id not in self.drone_buffers:
            self.add_drone(drone_id)
        
        buffer = self.drone_buffers[drone_id]
        buffer.trajectory_to_evaluate = future_positions
        # 生成时间戳
        buffer.trajectory_to_evaluate_timestamps = [
            i * self.dt for i in range(len(future_positions))
        ]
    
    def load_swarm_data(self, data_path):
        """
        加载群轨迹数据
        
        Args:
            data_path: 数据文件路径(.npz 或 .txt CSV 格式)
            
        Returns:
            tuple: (trajectories_dict, timestamps_dict)
                   trajectories_dict[drone_id] = trajectory_array (seq_len, 3)
                   timestamps_dict[drone_id] = timestamps (seq_len,)
        """
        trajectories = {}
        timestamps = {}
        
        # 检查文件格式
        if data_path.endswith('.txt') or data_path.endswith('.csv'):
            # 加载 CSV/TXT 文件
            try:
                data = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=float)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                # CSV 格式: timestamp, tx, ty, tz
                timestamps_list = data[:, 0]
                positions = data[:, 1:4]  # 取 tx, ty, tz 三列
                
                # 所有点都作为单个 drone (ID=0)
                trajectories[0] = positions
                timestamps[0] = timestamps_list
                
            except Exception as e:
                print(f"Error loading CSV file {data_path}: {e}")
                return trajectories, timestamps
        else:
            # 加载 .npz 文件
            data = np.load(data_path, allow_pickle=True)
            
            # 加载轨迹数据
            if 'trajectories' in data:
                trajs_data = data['trajectories']
                # 检查是否是字典格式（0-d object array）
                if hasattr(trajs_data, 'item'):
                    try:
                        trajs = trajs_data.item()
                        if isinstance(trajs, dict):
                            trajectories = trajs
                        else:
                            # 如果是数组，按索引存储
                            for i, traj in enumerate(trajs):
                                trajectories[i] = traj
                    except (ValueError, TypeError):
                        # 如果转换失败，直接作为数组处理
                        for i, traj in enumerate(trajs_data):
                            trajectories[i] = traj
                else:
                    for i, traj in enumerate(trajs_data):
                        trajectories[i] = traj
            
            # 加载时间戳数据
            if 'timestamps' in data:
                ts_data = data['timestamps']
                # 检查是否是字典格式（0-d object array）
                if hasattr(ts_data, 'item'):
                    try:
                        ts = ts_data.item()
                        if isinstance(ts, dict):
                            timestamps = ts
                        else:
                            # 如果是数组，按索引存储
                            for i, t in enumerate(ts):
                                timestamps[i] = t
                    except (ValueError, TypeError):
                        # 如果转换失败，为每个轨迹生成时间戳
                        for drone_id in trajectories.keys():
                            timestamps[drone_id] = np.arange(len(trajectories[drone_id])) * self.dt
                else:
                    for i, t in enumerate(ts_data):
                        timestamps[i] = t
            
            # 如果没有时间戳或时间戳不完整，生成默认时间戳
            for drone_id in trajectories.keys():
                if drone_id not in timestamps:
                    timestamps[drone_id] = np.arange(len(trajectories[drone_id])) * self.dt
        
        return trajectories, timestamps
    
    def visualize_swarm(self, trajectories, predictions=None, collisions=None, 
                       title='Drone Swarm Trajectories'):
        """
        可视化无人机群轨迹 - 3D展示实际轨迹和预测轨迹
        
        Args:
            trajectories: dict {drone_id: trajectory_array}
            predictions: dict {drone_id: prediction_array}
            collisions: list of collisions
            title: 图标题
        """
        fig = plt.figure(figsize=(16, 10))
        
        # 子图1: 完整轨迹（实际+预测）
        ax1 = fig.add_subplot(121, projection='3d')
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        
        for idx, (drone_id, traj) in enumerate(trajectories.items()):
            color = colors[idx % len(colors)]
            
            # 绘制完整实际轨迹
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    color=color, linestyle='-', linewidth=2.5, 
                    label=f'Drone {drone_id} (Actual)', alpha=0.9)
            
            # 绘制起始点
            ax1.scatter(*traj[0], color=color, s=120, marker='o', alpha=0.9, edgecolors='black', linewidth=1)
            
            # 绘制终止点
            ax1.scatter(*traj[-1], color=color, s=120, marker='s', alpha=0.9, edgecolors='black', linewidth=1)
            
            # 绘制预测轨迹（从最后一个实际位置开始）
            if predictions and drone_id in predictions:
                pred = predictions[drone_id]
                pred_start = traj[-1]  # 预测从最后一个实际位置开始
                pred_shifted = pred + (pred_start - pred[0])
                ax1.plot(pred_shifted[:, 0], pred_shifted[:, 1], pred_shifted[:, 2],
                        color=color, linestyle='--', linewidth=2.5, 
                        label=f'Drone {drone_id} (Predicted)', alpha=0.7)
                
                # 标记预测轨迹的终点
                ax1.scatter(*pred_shifted[-1], color=color, s=120, marker='^', 
                           alpha=0.7, edgecolors='black', linewidth=1)
        
        # 标记碰撞
        if collisions:
            for drone_i, drone_j, dist, _ in collisions:
                ax1.scatter([], [], [], color='red', marker='x', s=200, 
                           label=f'Collision: {drone_i}-{drone_j} (d={dist:.2f}m)')
        
        ax1.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax1.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
        ax1.set_title('3D Trajectory Overview\n(Solid=Actual, Dashed=Predicted)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 预测误差对比（仅显示轨迹终点附近）
        ax2 = fig.add_subplot(122, projection='3d')
        
        for idx, (drone_id, traj) in enumerate(trajectories.items()):
            color = colors[idx % len(colors)]
            
            # 仅显示末尾部分的实际轨迹
            tail_length = min(30, len(traj))
            tail_traj = traj[-tail_length:]
            ax2.plot(tail_traj[:, 0], tail_traj[:, 1], tail_traj[:, 2], 
                    color=color, linestyle='-', linewidth=3, 
                    label=f'Drone {drone_id} (Last {tail_length} points)', alpha=0.9)
            
            # 绘制预测轨迹
            if predictions and drone_id in predictions:
                pred = predictions[drone_id]
                pred_start = traj[-1]  # 预测从最后一个实际位置开始
                pred_shifted = pred + (pred_start - pred[0])
                ax2.plot(pred_shifted[:, 0], pred_shifted[:, 1], pred_shifted[:, 2],
                        color=color, linestyle='--', linewidth=3, 
                        label=f'Drone {drone_id} (Predicted)', alpha=0.7)
                
                # 连接实际轨迹末端和预测轨迹起点
                ax2.plot([traj[-1, 0], pred_shifted[0, 0]], 
                        [traj[-1, 1], pred_shifted[0, 1]], 
                        [traj[-1, 2], pred_shifted[0, 2]],
                        color=color, linestyle=':', linewidth=2, alpha=0.5)
                
                # 绘制转接点
                ax2.scatter(*traj[-1], color=color, s=150, marker='o', 
                           alpha=0.9, edgecolors='black', linewidth=2, zorder=5)
        
        ax2.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax2.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
        ax2.set_title('Prediction Detail\n(末尾实际 → 预测对比)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_collision_analysis(self, predictions, collision_threshold=1.0):
        """
        可视化碰撞分析和预测精度
        
        Args:
            predictions: dict {drone_id: prediction_array}
            collision_threshold: 碰撞阈值
        """
        distances = self.compute_inter_drone_distances(predictions)
        num_pairs = len(distances)
        
        # 如果没有多个无人机对，跳过此可视化
        if num_pairs == 0:
            print("  Skipping collision analysis: only 1 drone detected (need at least 2 drones for distance analysis)")
            return
        
        # 计算需要的子图数
        num_subplots = min(4, num_pairs)
        if num_pairs <= 4:
            rows, cols = (num_pairs + 1) // 2, 2
        else:
            rows, cols = 2, 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        fig.suptitle('Inter-Drone Distance Analysis During Prediction Period', fontsize=16, fontweight='bold')
        
        if num_subplots == 1:
            axes = [[axes]]
        elif num_subplots <= 2:
            axes = [[axes[0], axes[1]]] if num_subplots == 2 else [[axes, None]]
        else:
            axes = [list(axes[i]) for i in range(rows)]
        
        ax_idx = 0
        for (drone_i, drone_j), dists in list(distances.items())[:num_subplots]:
            row = ax_idx // 2
            col = ax_idx % 2
            
            if num_subplots <= 2:
                ax = axes[0][col]
            else:
                ax = axes[row][col]
            
            time_steps = np.arange(len(dists))
            ax.plot(time_steps, dists, 'b-', linewidth=2.5, label='Distance', marker='o', markersize=4)
            ax.axhline(y=collision_threshold, color='r', linestyle='--', 
                      linewidth=2.5, label=f'Collision Threshold ({collision_threshold}m)')
            
            # 标记最小距离
            min_dist_idx = np.argmin(dists)
            min_dist_val = dists[min_dist_idx]
            ax.scatter(min_dist_idx, min_dist_val, color='r', s=150, 
                      zorder=5, label=f'Min: {min_dist_val:.2f}m @ step {min_dist_idx}', edgecolors='darkred', linewidth=2)
            
            # 标记最大距离
            max_dist_idx = np.argmax(dists)
            max_dist_val = dists[max_dist_idx]
            ax.scatter(max_dist_idx, max_dist_val, color='g', s=150, 
                      zorder=5, label=f'Max: {max_dist_val:.2f}m @ step {max_dist_idx}', edgecolors='darkgreen', linewidth=2)
            
            # 填充危险区域
            if np.any(dists < collision_threshold):
                ax.fill_between(time_steps, 0, collision_threshold, alpha=0.2, color='red', label='Danger Zone')
            
            ax.set_xlabel('Prediction Time Step', fontsize=11, fontweight='bold')
            ax.set_ylabel('Distance (m)', fontsize=11, fontweight='bold')
            ax.set_title(f'Drone {drone_i} ↔ Drone {drone_j}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, loc='best')
            ax.set_ylim(bottom=0)
            
            ax_idx += 1
        
        # 隐藏多余的子图
        for idx in range(num_subplots, rows * cols):
            row = idx // 2
            col = idx % 2
            if num_subplots <= 2:
                if axes[0][col] is not None:
                    axes[0][col].set_visible(False)
            else:
                axes[row][col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_prediction_comparison(self, trajectories, predictions):
        """
        可视化预测与真实轨迹的对比 - 显示预测的10个点 vs 真实的10个点
        
        Args:
            trajectories: dict {drone_id: trajectory_array}
            predictions: dict {drone_id: prediction_array (10, 3)}
        """
        num_drones = len(predictions)
        
        # 第一个可视化：3D对比
        fig = plt.figure(figsize=(16, 5*((num_drones+1)//2)))
        
        for drone_idx, (drone_id, traj) in enumerate(trajectories.items()):
            if drone_id not in predictions:
                continue
            
            pred = predictions[drone_id]  # shape: (10, 3)
            
            # 获取真实的未来10个点（在缓冲区末尾之后）
            # 由于我们的轨迹已经完整，取倒数20-30点作为"未来"进行验证
            future_start_idx = max(0, len(traj) - 20)  # 取末尾20点中的后10点作为真实未来
            actual_future = traj[future_start_idx:future_start_idx+10]
            
            # 如果不足10个点，进行补全
            if len(actual_future) < 10:
                actual_future = traj[-10:] if len(traj) >= 10 else traj
            
            # 重要：两个轨迹的起始点必须相同！
            # 将两个轨迹都对齐到真实轨迹的第一个点
            alignment_point = actual_future[0]
            actual_future_aligned = actual_future
            pred_aligned = pred + (alignment_point - pred[0])  # 将预测轨迹对齐到真实起点
            
            # 3D子图
            ax = fig.add_subplot(((num_drones+1)//2), 2, drone_idx+1, projection='3d')
            
            # 绘制起始点（两条轨迹的共同起点）
            ax.scatter(*alignment_point, color='black', s=200, marker='o', label='Start Point', 
                      edgecolors='white', linewidth=2, zorder=10)
            
            # 绘制真实的未来轨迹（10个点）
            ax.plot(actual_future_aligned[:, 0], actual_future_aligned[:, 1], actual_future_aligned[:, 2], 
                   color='blue', linestyle='-', linewidth=3, marker='o', markersize=6,
                   label='Actual Future (10 steps)', alpha=0.8, zorder=5)
            
            # 标记真实轨迹终点
            ax.scatter(*actual_future_aligned[-1], color='blue', s=200, marker='s', 
                      edgecolors='white', linewidth=2, zorder=10)
            
            # 绘制预测的未来轨迹（10个点）
            ax.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2], 
                   color='red', linestyle='--', linewidth=3, marker='s', markersize=6,
                   label='Predicted Future (10 steps)', alpha=0.8, zorder=5)
            
            # 标记预测轨迹终点
            ax.scatter(*pred_aligned[-1], color='red', s=200, marker='^', 
                      edgecolors='white', linewidth=2, zorder=10)
            
            # 计算误差
            errors = np.linalg.norm(actual_future_aligned - pred_aligned, axis=1)
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            
            ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
            ax.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
            ax.set_title(f'Drone {drone_id} - Prediction Accuracy\nMean Error: {mean_error:.3f}m | Max Error: {max_error:.3f}m', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 第二个可视化：坐标分量对比 + 误差
        fig, axes = plt.subplots(3, num_drones, figsize=(16, 12))
        
        if num_drones == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Prediction vs Actual Comparison - 10 Future Steps Analysis', 
                     fontsize=14, fontweight='bold')
        
        for drone_idx, (drone_id, traj) in enumerate(trajectories.items()):
            if drone_id not in predictions:
                continue
            
            pred = predictions[drone_id]  # shape: (10, 3)
            
            # 获取真实的未来10个点
            future_start_idx = max(0, len(traj) - 20)
            actual_future = traj[future_start_idx:future_start_idx+10]
            
            if len(actual_future) < 10:
                actual_future = traj[-10:] if len(traj) >= 10 else traj
            
            # 重要：两个轨迹的起始点必须相同！
            alignment_point = actual_future[0]
            actual_future_aligned = actual_future
            pred_aligned = pred + (alignment_point - pred[0])
            
            # 时间步
            time_steps = np.arange(1, 11)
            
            # 计算误差
            errors = np.linalg.norm(actual_future_aligned - pred_aligned, axis=1)
            mean_error = np.mean(errors)
            rmse = np.sqrt(np.mean(errors**2))
            
            # X 坐标
            ax_x = axes[0, drone_idx]
            ax_x.plot(time_steps, actual_future_aligned[:, 0], 'b-o', linewidth=2.5, markersize=8, label='Actual', alpha=0.8)
            ax_x.plot(time_steps, pred_aligned[:, 0], 'r--s', linewidth=2.5, markersize=8, label='Predicted', alpha=0.8)
            ax_x.fill_between(time_steps, actual_future_aligned[:, 0], pred_aligned[:, 0], alpha=0.2, color='orange')
            ax_x.set_ylabel('X (m)', fontsize=10, fontweight='bold')
            ax_x.set_title(f'Drone {drone_id} - X Coordinate', fontsize=11, fontweight='bold')
            ax_x.grid(True, alpha=0.3, linestyle='--')
            ax_x.legend(fontsize=9)
            
            # Y 坐标
            ax_y = axes[1, drone_idx]
            ax_y.plot(time_steps, actual_future_aligned[:, 1], 'b-o', linewidth=2.5, markersize=8, label='Actual', alpha=0.8)
            ax_y.plot(time_steps, pred_aligned[:, 1], 'r--s', linewidth=2.5, markersize=8, label='Predicted', alpha=0.8)
            ax_y.fill_between(time_steps, actual_future_aligned[:, 1], pred_aligned[:, 1], alpha=0.2, color='orange')
            ax_y.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
            ax_y.set_title(f'Drone {drone_id} - Y Coordinate', fontsize=11, fontweight='bold')
            ax_y.grid(True, alpha=0.3, linestyle='--')
            ax_y.legend(fontsize=9)
            
            # Z 坐标 + 误差信息
            ax_z = axes[2, drone_idx]
            ax_z.plot(time_steps, actual_future_aligned[:, 2], 'b-o', linewidth=2.5, markersize=8, label='Actual', alpha=0.8)
            ax_z.plot(time_steps, pred_aligned[:, 2], 'r--s', linewidth=2.5, markersize=8, label='Predicted', alpha=0.8)
            ax_z.fill_between(time_steps, actual_future_aligned[:, 2], pred_aligned[:, 2], alpha=0.2, color='orange')
            ax_z.set_ylabel('Z (m)', fontsize=10, fontweight='bold')
            ax_z.set_xlabel('Prediction Step', fontsize=10, fontweight='bold')
            
            # 在Z坐标图上添加误差统计
            ax_z_info = ax_z.twinx()
            ax_z_info.plot(time_steps, errors, 'g-^', linewidth=2, markersize=6, label='Error', alpha=0.6)
            ax_z_info.set_ylabel('Point-wise Error (m)', fontsize=10, fontweight='bold', color='green')
            ax_z_info.tick_params(axis='y', labelcolor='green')
            
            title_text = f'Drone {drone_id} - Z Coordinate\nMean Error: {mean_error:.3f}m | RMSE: {rmse:.3f}m'
            ax_z.set_title(title_text, fontsize=11, fontweight='bold')
            ax_z.grid(True, alpha=0.3, linestyle='--')
            ax_z.legend(fontsize=9, loc='upper left')
            ax_z_info.legend(fontsize=9, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # 第三个可视化：误差统计总结
        print("\n" + "="*70)
        print("PREDICTION ERROR ANALYSIS (与真实未来10步的对比)")
        print("="*70)
        for drone_id, traj in trajectories.items():
            if drone_id not in predictions:
                continue
            
            pred = predictions[drone_id]
            future_start_idx = max(0, len(traj) - 20)
            actual_future = traj[future_start_idx:future_start_idx+10]
            
            if len(actual_future) < 10:
                actual_future = traj[-10:] if len(traj) >= 10 else traj
            
            # 重要：两个轨迹的起始点必须相同！
            alignment_point = actual_future[0]
            actual_future_aligned = actual_future
            pred_aligned = pred + (alignment_point - pred[0])
            
            errors = np.linalg.norm(actual_future_aligned - pred_aligned, axis=1)
            mean_error = np.mean(errors)
            rmse = np.sqrt(np.mean(errors**2))
            max_error = np.max(errors)
            min_error = np.min(errors)
            
            print(f"\nDrone {drone_id}:")
            print(f"  共同起始点: ({alignment_point[0]:.2f}, {alignment_point[1]:.2f}, {alignment_point[2]:.2f})")
            print(f"  Mean Error (MAE):  {mean_error:.4f} m")
            print(f"  RMSE:              {rmse:.4f} m")
            print(f"  Max Error:         {max_error:.4f} m (at step {np.argmax(errors)+1})")
            print(f"  Min Error:         {min_error:.4f} m (at step {np.argmin(errors)+1})")
            print(f"  Error per step:    {[f'{e:.4f}' for e in errors]}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Drone Swarm Trajectory Prediction')
    parser.add_argument('--position_model_path', type=str, required=True,
                       help='Path to position model (.pth)')
    parser.add_argument('--velocity_model_path', type=str, required=True,
                       help='Path to velocity model (.pth)')
    parser.add_argument('--position_stats_file', type=str, required=True,
                       help='Path to position stats (.npz)')
    parser.add_argument('--velocity_stats_file', type=str, required=True,
                       help='Path to velocity stats (.npz)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to swarm data (.npz)')
    parser.add_argument('--collision_threshold', type=float, default=1.0,
                       help='Collision detection threshold (meters)')
    parser.add_argument('--use_velocity_prediction', action='store_true',
                       help='Use velocity prediction mode')
    parser.add_argument('--use_whitening', action='store_true',
                       help='Use whitening normalization')
    parser.add_argument('--buffer_duration', type=float, default=2.0,
                       help='Buffer duration (seconds)')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='Sampling time interval (seconds)')
    parser.add_argument('--input_length', type=int, default=30,
                       help='Input sequence length')
    parser.add_argument('--pos_hidden_dim', type=int, default=64,
                       help='Position model hidden dimension')
    parser.add_argument('--pos_num_layers', type=int, default=2,
                       help='Position model number of layers')
    parser.add_argument('--pos_dropout', type=float, default=0.5,
                       help='Position model dropout rate')
    parser.add_argument('--vel_hidden_dim', type=int, default=64,
                       help='Velocity model hidden dimension')
    parser.add_argument('--vel_num_layers', type=int, default=2,
                       help='Velocity model number of layers')
    parser.add_argument('--vel_dropout', type=float, default=0.5,
                       help='Velocity model dropout rate')
    parser.add_argument('--future_steps', type=int, default=5,
                       help='Number of future steps to evaluate/predict')
    parser.add_argument('--prediction_smoothing_window', type=int, default=1,
                       help='Moving-average window applied to raw predictions before alignment')
    parser.add_argument('--actual_smoothing_window', type=int, default=3,
                       help='Moving-average window applied to the actual future segment before evaluation (0=off)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    
    args = parser.parse_args()
    
    # 创建预测器
    print("Initializing swarm predictor...")
    swarm_predictor = DroneSwarmPredictor(
        args.position_model_path, args.velocity_model_path,
        args.position_stats_file, args.velocity_stats_file,
        pos_hidden_dim=args.pos_hidden_dim,
        pos_num_layers=args.pos_num_layers,
        pos_dropout=args.pos_dropout,
        vel_hidden_dim=args.vel_hidden_dim,
        vel_num_layers=args.vel_num_layers,
        vel_dropout=args.vel_dropout,
        use_velocity_prediction=args.use_velocity_prediction,
        use_whitening=args.use_whitening,
        buffer_duration=args.buffer_duration,
        dt=args.dt,
        prediction_smoothing_window=args.prediction_smoothing_window,
        actual_smoothing_window=args.actual_smoothing_window
    )
    
    # 加载数据
    print(f"Loading swarm data from {args.data_path}...")
    trajectories, timestamps = swarm_predictor.load_swarm_data(args.data_path)
    print(f"Loaded trajectories for {len(trajectories)} drones")
    
    # 更新无人机位置到缓冲区
    print("Processing trajectories through buffers...")
    for drone_id, trajectory in trajectories.items():
        print(f"  Processing drone {drone_id}: {len(trajectory)} points")
        for pos, t in zip(trajectory, timestamps[drone_id]):
            swarm_predictor.update_drone_position(drone_id, tuple(pos), t)
        buffer_positions = swarm_predictor.drone_buffers[drone_id].get_regularly_sampled_positions()
        print(f"    -> Buffer size after update: {len(buffer_positions)} points")
    
    # 进行预测
    print("\nPredicting trajectories...")
    for drone_id in swarm_predictor.drone_buffers.keys():
        buffer_positions = swarm_predictor.drone_buffers[drone_id].get_regularly_sampled_positions()
        print(f"  Drone {drone_id}: buffer has {len(buffer_positions)} points, input_length={args.input_length}")
    
    predictions = swarm_predictor.predict_all_trajectories(args.input_length)
    print(f"Predictions generated for {len(predictions)} drones")
    if len(predictions) == 0:
        print("  WARNING: No predictions generated. Checking buffer status...")
    
    # 检测碰撞
    print("Detecting collisions...")
    collisions = swarm_predictor.detect_collisions(predictions, args.collision_threshold)
    print(f"Detected {len(collisions)} potential collisions")
    
    for drone_i, drone_j, min_dist, time_step in collisions:
        print(f"  - Drone {drone_i} <-> Drone {drone_j}: "
              f"min_distance={min_dist:.3f}m at time_step={time_step}")
    
    # 分析无人机间距离
    distances = swarm_predictor.compute_inter_drone_distances(predictions)
    print("\nInter-drone distance statistics:")
    all_min_distances = []
    for (drone_i, drone_j), dists in distances.items():
        min_dist = np.min(dists)
        max_dist = np.max(dists)
        mean_dist = np.mean(dists)
        all_min_distances.append(min_dist)
        print(f"  - Drone {drone_i} <-> Drone {drone_j}: "
              f"min={min_dist:.3f}m, max={max_dist:.3f}m, mean={mean_dist:.3f}m")
    
    # 总体统计
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total drones: {len(predictions)}")
    print(f"Total drone pairs: {len(distances)}")
    print(f"Collision threshold: {args.collision_threshold}m")
    print(f"Collisions detected: {len(collisions)}")
    if all_min_distances:
        print(f"Overall minimum distance: {np.min(all_min_distances):.3f}m")
        print(f"Overall maximum distance: {np.max(all_min_distances):.3f}m")
        print(f"Overall mean distance: {np.mean(all_min_distances):.3f}m")
    print("="*60)
    
    print("\nEvaluating prediction quality against the final segments:")
    prediction_metrics = swarm_predictor.evaluate_prediction_quality(predictions, trajectories, args.future_steps)
    if prediction_metrics:
        for drone_id, summary in prediction_metrics.items():
            errors = summary['errors']
            print(f"  Drone {drone_id}: MAE={summary['mae']:.3f}m, RMSE={summary['rmse']:.3f}m, "
                  f"steps={summary['future_steps']}")
            print(f"    Errors per step: {[f'{e:.3f}' for e in errors]}")
    else:
        print("  Unable to compute prediction error metrics (not enough trajectory data).")

    # 可视化
    if args.visualize:
        print("\nVisualizing results...")
        print("  - 3D Swarm Trajectories (Left: Overview, Right: Detail)...")
        swarm_predictor.visualize_swarm(trajectories, predictions, collisions)
        
        print("  - Inter-Drone Distance Analysis...")
        swarm_predictor.visualize_collision_analysis(predictions, args.collision_threshold)
        
        print("  - Prediction vs Actual Comparison (X, Y, Z coordinates)...")
        swarm_predictor.visualize_prediction_comparison(trajectories, predictions)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
