#!/usr/bin/env python3
"""
简化评估脚本：只展示输入序列 + 真实后续轨迹 + 预测轨迹
重点：3D 轨迹对比 + XYZ 坐标时间序列
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import torch
import argparse
import logging
import subprocess
import platform

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / 'drone_trajectories'))
sys.path.insert(0, str(Path(__file__).parent / 'drone_path_predictor_ros-main'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class TrajectoryPredictor:
    """轨迹预测器（位置 + 速度）"""
    def __init__(self, model_path, stats_path, hidden_dim=64, num_layers=2, device='cuda'):
        from train_model import GRUModel
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 加载统计量
        stats = np.load(stats_path)
        self.input_mean = torch.tensor(stats['input_mean'], dtype=torch.float32, device=self.device)
        self.input_std = torch.tensor(stats['input_std'], dtype=torch.float32, device=self.device)
        
        if 'output_mean' in stats:
            self.output_mean = torch.tensor(stats['output_mean'], dtype=torch.float32, device=self.device)
        else:
            self.output_mean = self.input_mean.clone()
        
        if 'output_std' in stats:
            self.output_std = torch.tensor(stats['output_std'], dtype=torch.float32, device=self.device)
        else:
            self.output_std = self.input_std.clone()
        
        # 加载模型
        state_dict = torch.load(model_path, map_location=self.device)
        self.model = GRUModel(input_size=3, hidden_dim=hidden_dim, num_layers=num_layers,
                             dropout=0.5, output_steps=10)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"✓ 模型已加载: {Path(model_path).name}")
    
    def normalize(self, x):
        """归一化"""
        return (x - self.input_mean) / (self.input_std + 1e-8)
    
    def denormalize(self, x):
        """反归一化"""
        return x * (self.output_std + 1e-8) + self.output_mean
    
    def predict(self, trajectory, inp_len=20):
        """预测未来轨迹"""
        if len(trajectory) < inp_len:
            raise ValueError(f"轨迹长度 {len(trajectory)} < 输入长度 {inp_len}")
        
        inp = trajectory[-inp_len:, :].astype(np.float32)
        inp_tensor = torch.tensor(inp, device=self.device).unsqueeze(0)
        inp_norm = self.normalize(inp_tensor)
        
        with torch.no_grad():
            pred_norm = self.model(inp_norm)
        
        pred = self.denormalize(pred_norm).squeeze(0).cpu().numpy()
        return pred


def load_trajectory(csv_path):
    """加载轨迹 CSV"""
    df = pd.read_csv(csv_path)
    traj = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    return traj


def compute_velocity(trajectory, dt=0.1):
    """计算速度"""
    velocity = np.diff(trajectory, axis=0) / dt
    return velocity


def integrate_velocity(start_pos, velocity, dt=0.1):
    """积分速度得到位置"""
    positions = [start_pos]
    pos = start_pos.copy()
    for v in velocity:
        pos = pos + v * dt
        positions.append(pos.copy())
    return np.array(positions)


def align_predictions(pred, actual):
    """按首点对齐"""
    if pred.shape != actual.shape:
        raise ValueError(f"形状不匹配")
    shift = actual[0] - pred[0]
    return pred + shift


def compute_metrics(pred_aligned, actual):
    """计算误差"""
    errors = np.linalg.norm(pred_aligned - actual, axis=1)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': float(np.max(errors)),
        'errors': errors
    }


def plot_position_models(inp_pos, actual_future, pos_pred_dict, output_dir):
    """
    位置模型专用可视化：3D 轨迹 + XY 平面 + XYZ 时间序列
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors_pos = {'position_synth_short': '#FF6B6B', 'position_synth_mix': '#4ECDC4', 'position_synth_long': '#850742'}
    
    fig = plt.figure(figsize=(22, 14))
    
    # ========== 第一行：3D 轨迹对比 ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(inp_pos[:, 0], inp_pos[:, 1], inp_pos[:, 2], 'b-o', linewidth=3, markersize=6, label='观测输入 (Input)', alpha=0.9)
    ax1.plot(actual_future[:, 0], actual_future[:, 1], actual_future[:, 2], 'g-s', linewidth=4, markersize=10, 
            label='真实后续 (Actual)', alpha=1.0, zorder=15)
    
    for model_name, data in pos_pred_dict.items():
        pred = data['pred']
        ax1.plot(pred[:, 0], pred[:, 1], pred[:, 2], linestyle='--', marker='^', linewidth=2.5, markersize=7, 
                color=colors_pos.get(model_name, 'gray'), label=f'预测 {model_name}', alpha=0.85)
    
    ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax1.set_title('3D 轨迹对比 - 位置预测模型', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.4)
    
    # XY 平面投影
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(inp_pos[:, 0], inp_pos[:, 1], 'b-o', linewidth=3, markersize=6, label='观测输入', alpha=0.9)
    ax2.plot(actual_future[:, 0], actual_future[:, 1], 'g-s', linewidth=4, markersize=10, 
            label='真实后续', alpha=1.0, zorder=15)
    
    for model_name, data in pos_pred_dict.items():
        pred = data['pred']
        ax2.plot(pred[:, 0], pred[:, 1], linestyle='--', marker='^', linewidth=2.5, markersize=7,
                color=colors_pos.get(model_name, 'gray'), label=f'{model_name}', alpha=0.85)
    
    ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax2.set_title('XY 平面投影 - 位置预测模型', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4)
    ax2.axis('equal')
    
    # XZ 平面投影
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(inp_pos[:, 0], inp_pos[:, 2], 'b-o', linewidth=3, markersize=6, label='观测输入', alpha=0.9)
    ax3.plot(actual_future[:, 0], actual_future[:, 2], 'g-s', linewidth=4, markersize=10, 
            label='真实后续', alpha=1.0, zorder=15)
    
    for model_name, data in pos_pred_dict.items():
        pred = data['pred']
        ax3.plot(pred[:, 0], pred[:, 2], linestyle='--', marker='^', linewidth=2.5, markersize=7,
                color=colors_pos.get(model_name, 'gray'), label=f'{model_name}', alpha=0.85)
    
    ax3.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
    ax3.set_title('XZ 平面投影 - 位置预测模型', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.4)
    ax3.axis('equal')
    
    # ========== 第二行：坐标时间序列 ==========
    time_steps = np.arange(1, 11)
    
    for idx, coord_name in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(2, 3, 4 + idx)
        
        ax.plot(time_steps, actual_future[:, idx], 'g-s', linewidth=4, markersize=12, 
               label='真实 (Actual)', alpha=1.0, zorder=15)
        
        for model_name, data in pos_pred_dict.items():
            pred = data['pred']
            metrics = data['metrics']
            ax.plot(time_steps, pred[:, idx], linestyle='-', marker='o', linewidth=2.5, markersize=8,
                   color=colors_pos.get(model_name, 'gray'), 
                   label=f'{model_name}\nMAE={metrics["mae"]:.4f}m RMSE={metrics["rmse"]:.4f}m', 
                   alpha=0.85)
        
        ax.set_xlabel('预测步数 (Prediction Step)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{coord_name} 坐标 (m)', fontsize=12, fontweight='bold')
        ax.set_title(f'{coord_name} 坐标时间序列 - 位置预测模型', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=10, loc='best')
        ax.set_xticks(time_steps)
    
    plt.tight_layout()
    
    png_path = output_dir / 'position_models_evaluation.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    logging.info(f"✓ 位置模型图像已保存: {png_path}")
    
    return png_path


def plot_velocity_models(inp_pos, actual_future, vel_pred_dict, output_dir):
    """
    速度模型专用可视化：3D 轨迹 + XY 平面 + XYZ 时间序列（积分后的位置）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors_vel = {'velocity_synth_short': '#FF9E9E', 'velocity_synth_mix': '#7FE5DD', 'velocity_synth_long': '#C73E4A'}
    
    fig = plt.figure(figsize=(22, 14))
    
    # ========== 第一行：3D 轨迹对比 ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(inp_pos[:, 0], inp_pos[:, 1], inp_pos[:, 2], 'b-o', linewidth=3, markersize=6, label='观测输入 (Input)', alpha=0.9)
    ax1.plot(actual_future[:, 0], actual_future[:, 1], actual_future[:, 2], 'g-s', linewidth=4, markersize=10, 
            label='真实后续 (Actual)', alpha=1.0, zorder=15)
    
    for model_name, data in vel_pred_dict.items():
        pred = data['pred']
        ax1.plot(pred[:, 0], pred[:, 1], pred[:, 2], linestyle='--', marker='^', linewidth=2.5, markersize=7, 
                color=colors_vel.get(model_name, 'gray'), label=f'预测 {model_name}', alpha=0.85)
    
    ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax1.set_title('3D 轨迹对比 - 速度预测模型（积分后）', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.4)
    
    # XY 平面投影
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(inp_pos[:, 0], inp_pos[:, 1], 'b-o', linewidth=3, markersize=6, label='观测输入', alpha=0.9)
    ax2.plot(actual_future[:, 0], actual_future[:, 1], 'g-s', linewidth=4, markersize=10, 
            label='真实后续', alpha=1.0, zorder=15)
    
    for model_name, data in vel_pred_dict.items():
        pred = data['pred']
        ax2.plot(pred[:, 0], pred[:, 1], linestyle='--', marker='^', linewidth=2.5, markersize=7,
                color=colors_vel.get(model_name, 'gray'), label=f'{model_name}', alpha=0.85)
    
    ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax2.set_title('XY 平面投影 - 速度预测模型（积分后）', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4)
    ax2.axis('equal')
    
    # XZ 平面投影
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(inp_pos[:, 0], inp_pos[:, 2], 'b-o', linewidth=3, markersize=6, label='观测输入', alpha=0.9)
    ax3.plot(actual_future[:, 0], actual_future[:, 2], 'g-s', linewidth=4, markersize=10, 
            label='真实后续', alpha=1.0, zorder=15)
    
    for model_name, data in vel_pred_dict.items():
        pred = data['pred']
        ax3.plot(pred[:, 0], pred[:, 2], linestyle='--', marker='^', linewidth=2.5, markersize=7,
                color=colors_vel.get(model_name, 'gray'), label=f'{model_name}', alpha=0.85)
    
    ax3.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
    ax3.set_title('XZ 平面投影 - 速度预测模型（积分后）', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.4)
    ax3.axis('equal')
    
    # ========== 第二行：坐标时间序列 ==========
    time_steps = np.arange(1, 11)
    
    for idx, coord_name in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(2, 3, 4 + idx)
        
        ax.plot(time_steps, actual_future[:, idx], 'g-s', linewidth=4, markersize=12, 
               label='真实 (Actual)', alpha=1.0, zorder=15)
        
        for model_name, data in vel_pred_dict.items():
            pred = data['pred']
            metrics = data['metrics']
            ax.plot(time_steps, pred[:, idx], linestyle='-', marker='o', linewidth=2.5, markersize=8,
                   color=colors_vel.get(model_name, 'gray'), 
                   label=f'{model_name}\nMAE={metrics["mae"]:.4f}m RMSE={metrics["rmse"]:.4f}m', 
                   alpha=0.85)
        
        ax.set_xlabel('预测步数 (Prediction Step)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{coord_name} 坐标 (m)', fontsize=12, fontweight='bold')
        ax.set_title(f'{coord_name} 坐标时间序列 - 速度预测模型（积分后）', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=10, loc='best')
        ax.set_xticks(time_steps)
    
    plt.tight_layout()
    
    png_path = output_dir / 'velocity_models_evaluation.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    logging.info(f"✓ 速度模型图像已保存: {png_path}")
    
    return png_path


def main():
    parser = argparse.ArgumentParser(description='简化评估：输入 + 真实 + 预测')
    parser.add_argument('--trajectory', '-t',
                       #default=r'D:\Trajectory prediction\drone_trajectories\random_traj_100ms\circle_92.txt',
                       default=r'D:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory3D-2_1785.csv',
                       help='轨迹 CSV 文件路径')
    parser.add_argument('--models-dir', '-m',
                       default=r'D:\Trajectory prediction\drone_trajectories\gru_models_enhanced',
                       help='训练好的模型目录')
    parser.add_argument('--stats-dir', '-s',
                       default=r'D:\Trajectory prediction\drone_trajectories\transformed_trajectories',
                       help='统计量文件目录')
    parser.add_argument('--output-dir', '-o',
                       default=r'D:\Trajectory prediction\evaluation_results',
                       help='输出目录')
    parser.add_argument('--show', action='store_true', help='显示图像')
    parser.add_argument('--inp-len', type=int, default=20, help='输入序列长度')
    parser.add_argument('--out-len', type=int, default=10, help='输出序列长度')
    parser.add_argument('--dt', type=float, default=0.1, help='采样间隔（秒）')
    
    args = parser.parse_args()
    
    trajectory_path = Path(args.trajectory)
    models_dir = Path(args.models_dir)
    stats_dir = Path(args.stats_dir)
    output_dir = Path(args.output_dir)
    
    # 加载轨迹
    logging.info(f"加载轨迹: {trajectory_path.name}")
    trajectory = load_trajectory(trajectory_path)
    logging.info(f"轨迹形状: {trajectory.shape}")
    
    if len(trajectory) < args.inp_len + args.out_len:
        logging.error(f"轨迹太短")
        return
    
    # 分割：输入 + 真实后续
    inp_pos = trajectory[-args.inp_len - args.out_len:-args.out_len, :]
    actual_future = trajectory[-args.out_len:, :]
    
    logging.info(f"输入位置序列: {inp_pos.shape}")
    logging.info(f"真实后续序列: {actual_future.shape}")
    
    # 计算速度
    actual_velocity = compute_velocity(trajectory, dt=args.dt)
    
    # ========== 位置模型预测 ==========
    logging.info("\n" + "="*80)
    logging.info("位置模型预测")
    logging.info("="*80)
    
    pos_pred_dict = {}
    model_configs = [
        ('position_synth_short', 64, 2),
        ('position_synth_mix', 128, 3),
        ('position_synth_long', 256, 5),
    ]
    
    for model_name, hidden_dim, num_layers in model_configs:
        try:
            model_path = models_dir / f'{model_name}_best_model.pth'
            stats_path = stats_dir / 'pos_stats.npz'
            
            if not model_path.exists():
                logging.warning(f"模型不存在: {model_name}")
                continue
            
            predictor = TrajectoryPredictor(str(model_path), str(stats_path),
                                           hidden_dim=hidden_dim, num_layers=num_layers)
            pred = predictor.predict(trajectory, inp_len=args.inp_len)
            
            # 对齐
            shift = actual_future[0] - pred[0]
            pred_aligned = pred + shift
            
            metrics = compute_metrics(pred_aligned, actual_future)
            pos_pred_dict[model_name] = {'pred': pred_aligned, 'metrics': metrics}
            
            logging.info(f"{model_name:25} → MAE: {metrics['mae']:.4f}m  RMSE: {metrics['rmse']:.4f}m")
            
        except Exception as e:
            logging.error(f"处理 {model_name} 失败: {e}")
    
    # ========== 速度模型预测 ==========
    logging.info("\n" + "="*80)
    logging.info("速度模型预测（积分为位置）")
    logging.info("="*80)
    
    vel_pred_dict = {}
    vel_configs = [
        ('velocity_synth_short', 64, 2),
        ('velocity_synth_mix', 128, 3),
        ('velocity_synth_long', 256, 5),
    ]
    
    # 统一速度输入
    vel_input = actual_velocity[-args.inp_len:, :]
    start_pos = trajectory[-args.out_len, :]
    
    for model_name, hidden_dim, num_layers in vel_configs:
        try:
            model_path = models_dir / f'{model_name}_best_model.pth'
            stats_path = stats_dir / 'vel_stats.npz'
            
            if not model_path.exists():
                logging.warning(f"模型不存在: {model_name}")
                continue
            
            predictor = TrajectoryPredictor(str(model_path), str(stats_path),
                                           hidden_dim=hidden_dim, num_layers=num_layers)
            pred_vel = predictor.predict(vel_input, inp_len=args.inp_len)
            
            # 积分
            pred_pos = integrate_velocity(start_pos, pred_vel, dt=args.dt)[1:, :]
            
            # 对齐
            shift = actual_future[0] - pred_pos[0]
            pred_aligned = pred_pos + shift
            
            metrics = compute_metrics(pred_aligned, actual_future)
            vel_pred_dict[model_name] = {'pred': pred_aligned, 'metrics': metrics}
            
            logging.info(f"{model_name:25} → MAE: {metrics['mae']:.4f}m  RMSE: {metrics['rmse']:.4f}m")
            
        except Exception as e:
            logging.error(f"处理 {model_name} 失败: {e}")
    
    if not pos_pred_dict and not vel_pred_dict:
        logging.error("没有成功加载模型")
        return
    
    # ========== 可视化 ==========
    logging.info("\n生成可视化...")
    
    # 绘制位置模型图
    if pos_pred_dict:
        pos_png_path = plot_position_models(inp_pos, actual_future, pos_pred_dict, output_dir)
    else:
        pos_png_path = None
        logging.warning("没有位置模型预测，跳过位置模型图")
    
    # 绘制速度模型图
    if vel_pred_dict:
        vel_png_path = plot_velocity_models(inp_pos, actual_future, vel_pred_dict, output_dir)
    else:
        vel_png_path = None
        logging.warning("没有速度模型预测，跳过速度模型图")
    
    # ========== 显示 ==========
    if args.show:
        plt.show()
    else:
        try:
            if pos_png_path:
                if platform.system() == 'Windows':
                    os.startfile(pos_png_path)
                elif platform.system() == 'Darwin':
                    subprocess.Popen(['open', str(pos_png_path)])
                else:
                    subprocess.Popen(['xdg-open', str(pos_png_path)])
                logging.info(f"✓ 位置模型图已用默认查看器打开")
            
            if vel_png_path:
                if platform.system() == 'Windows':
                    os.startfile(vel_png_path)
                elif platform.system() == 'Darwin':
                    subprocess.Popen(['open', str(vel_png_path)])
                else:
                    subprocess.Popen(['xdg-open', str(vel_png_path)])
                logging.info(f"✓ 速度模型图已用默认查看器打开")
        except:
            if pos_png_path:
                logging.info(f"✓ 请手动打开位置模型图: {pos_png_path}")
            if vel_png_path:
                logging.info(f"✓ 请手动打开速度模型图: {vel_png_path}")
    
    logging.info(f"\n✓ 评估完成！结果已保存到: {output_dir}")
    logging.info(f"  位置模型图: {pos_png_path if pos_png_path else '(无)'}")
    logging.info(f"  速度模型图: {vel_png_path if vel_png_path else '(无)'}")


if __name__ == '__main__':
    main()
