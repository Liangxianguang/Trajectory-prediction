#!/usr/bin/env python3
"""
增强评估脚本：用位置 + 速度模型联合预测
- 先用速度模型预测未来速度序列
- 用预测的速度积分得到位置（物理约束）
- 与直接位置预测对比
- 生成详细可视化与指标对比
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
            logging.warning("stats 缺少 output_mean，使用 input_mean")
        
        if 'output_std' in stats:
            self.output_std = torch.tensor(stats['output_std'], dtype=torch.float32, device=self.device)
        else:
            self.output_std = self.input_std.clone()
            logging.warning("stats 缺少 output_std，使用 input_std")
        
        # 加载模型
        state_dict = torch.load(model_path, map_location=self.device)
        self.model = GRUModel(input_size=3, hidden_dim=hidden_dim, num_layers=num_layers,
                             dropout=0.5, output_steps=10)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"模型已加载: {Path(model_path).name} | 隐藏维度: {hidden_dim} | 层数: {num_layers}")
    
    def normalize(self, x):
        """归一化"""
        return (x - self.input_mean) / (self.input_std + 1e-8)
    
    def denormalize(self, x):
        """反归一化"""
        return x * (self.output_std + 1e-8) + self.output_mean
    
    def predict(self, trajectory, inp_len=20):
        """
        预测未来 10 步轨迹
        trajectory: (N, 3) 轨迹数组
        inp_len: 输入长度（默认 20）
        """
        if len(trajectory) < inp_len:
            raise ValueError(f"轨迹长度 {len(trajectory)} < 输入长度 {inp_len}")
        
        # 取最后 inp_len 步作为输入
        inp = trajectory[-inp_len:, :].astype(np.float32)  # (inp_len, 3)
        
        # 转为张量并归一化
        inp_tensor = torch.tensor(inp, device=self.device).unsqueeze(0)  # (1, inp_len, 3)
        inp_norm = self.normalize(inp_tensor)
        
        # 预测
        with torch.no_grad():
            pred_norm = self.model(inp_norm)  # (1, 10, 3)
        
        # 反归一化
        pred = self.denormalize(pred_norm).squeeze(0).cpu().numpy()  # (10, 3)
        
        return pred


def load_trajectory(csv_path):
    """加载轨迹 CSV"""
    df = pd.read_csv(csv_path)
    traj = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    return traj


def compute_velocity(trajectory, dt=0.1):
    """
    从轨迹计算速度
    trajectory: (N, 3)
    dt: 采样间隔
    返回: (N-1, 3) 速度序列
    """
    velocity = np.diff(trajectory, axis=0) / dt
    return velocity


def integrate_velocity(start_pos, velocity, dt=0.1):
    """
    从速度积分得到位置
    start_pos: (3,) 起始位置
    velocity: (T, 3) 速度序列
    dt: 采样间隔
    返回: (T, 3) 位置序列
    """
    positions = [start_pos]
    pos = start_pos.copy()
    
    for v in velocity:
        pos = pos + v * dt
        positions.append(pos.copy())
    
    return np.array(positions)


def align_predictions(pred, actual):
    """按首点对齐预测与实际值"""
    if pred.shape != actual.shape:
        raise ValueError(f"形状不匹配: pred {pred.shape} vs actual {actual.shape}")
    shift = actual[0] - pred[0]
    return pred + shift


def compute_metrics(pred_aligned, actual):
    """计算误差指标"""
    errors = np.linalg.norm(pred_aligned - actual, axis=1)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_err = float(np.max(errors))
    min_err = float(np.min(errors))
    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_err,
        'min_error': min_err,
        'errors': errors
    }


def plot_results(trajectory, actual_future, predictions_dict, output_dir, actual_velocity=None, velocity_predictions_dict=None):
    """
    可视化：3D 轨迹 + 坐标对比 + 误差分析 + 速度对比
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = {
        'position_synth_short': '#FF6B6B',
        'position_synth_mix': '#4ECDC4',
        'position_synth_long': '#850742',
    }
    
    fig = plt.figure(figsize=(22, 16))
    
    # ========== 3D 轨迹对比 ==========
    ax_3d = fig.add_subplot(3, 4, 1, projection='3d')
    
    history_len = min(100, len(trajectory) - 10)
    history = trajectory[-history_len-10:-10, :]
    ax_3d.plot(history[:, 0], history[:, 1], history[:, 2], 'b-', linewidth=2.5, label='History', alpha=0.7)
    ax_3d.plot(actual_future[:, 0], actual_future[:, 1], actual_future[:, 2], 'g-o', 
              linewidth=3, markersize=8, label='Actual Future', alpha=0.9, zorder=5)
    
    for i, (model_name, data) in enumerate(predictions_dict.items()):
        pred = data['pred']
        ax_3d.plot(pred[:, 0], pred[:, 1], pred[:, 2], linestyle='--', marker='s', 
                  linewidth=2, markersize=6, color=colors.get(model_name, f'C{i}'),
                  label=f'{model_name}', alpha=0.8)
    
    ax_3d.set_xlabel('X (m)', fontsize=9)
    ax_3d.set_ylabel('Y (m)', fontsize=9)
    ax_3d.set_zlabel('Z (m)', fontsize=9)
    ax_3d.set_title('3D 轨迹对比（位置模型）', fontsize=11, fontweight='bold')
    ax_3d.legend(fontsize=8)
    ax_3d.grid(True, alpha=0.3)
    
    # ========== X, Y, Z 坐标对比 ==========
    time_steps = np.arange(1, 11)
    
    for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(3, 4, coord_idx + 2)
        ax.plot(time_steps, actual_future[:, coord_idx], 'g-o', linewidth=3, 
               markersize=8, label='Actual', alpha=0.9, zorder=5)
        
        for model_name, data in predictions_dict.items():
            pred = data['pred']
            ax.plot(time_steps, pred[:, coord_idx], linestyle='--', marker='s',
                   linewidth=2, markersize=6, color=colors.get(model_name, f'C{list(predictions_dict.keys()).index(model_name)}'),
                   label=model_name, alpha=0.8)
        
        ax.set_xlabel('Time Step', fontsize=9)
        ax.set_ylabel(f'{coord_name} (m)', fontsize=9)
        ax.set_title(f'{coord_name} 坐标预测对比', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # ========== 速度对比（若有速度预测）==========
    if velocity_predictions_dict and actual_velocity is not None:
        ax_3d_vel = fig.add_subplot(3, 4, 5, projection='3d')
        
        actual_vel_future = actual_velocity[-10:, :]  # 最后 10 步速度
        ax_3d_vel.plot(actual_vel_future[:, 0], actual_vel_future[:, 1], actual_vel_future[:, 2], 
                      'g-o', linewidth=3, markersize=8, label='Actual Velocity', alpha=0.9, zorder=5)
        
        for model_name, data in velocity_predictions_dict.items():
            pred_vel = data['pred']
            ax_3d_vel.plot(pred_vel[:, 0], pred_vel[:, 1], pred_vel[:, 2], linestyle='--', marker='s',
                          linewidth=2, markersize=6, color=colors.get(model_name, 'C0'),
                          label=f'{model_name}', alpha=0.8)
        
        ax_3d_vel.set_xlabel('Vx (m/s)', fontsize=9)
        ax_3d_vel.set_ylabel('Vy (m/s)', fontsize=9)
        ax_3d_vel.set_zlabel('Vz (m/s)', fontsize=9)
        ax_3d_vel.set_title('3D 速度对比（速度模型）', fontsize=11, fontweight='bold')
        ax_3d_vel.legend(fontsize=8)
        ax_3d_vel.grid(True, alpha=0.3)
    
    # ========== 速度分量对比 ==========
    if velocity_predictions_dict and actual_velocity is not None:
        actual_vel_future = actual_velocity[-10:, :]
        for coord_idx, coord_name in enumerate(['Vx', 'Vy', 'Vz']):
            ax = fig.add_subplot(3, 4, coord_idx + 6)
            ax.plot(time_steps, actual_vel_future[:, coord_idx], 'g-o', linewidth=3,
                   markersize=8, label='Actual', alpha=0.9, zorder=5)
            
            for model_name, data in velocity_predictions_dict.items():
                pred_vel = data['pred']
                ax.plot(time_steps, pred_vel[:, coord_idx], linestyle='--', marker='s',
                       linewidth=2, markersize=6, color=colors.get(model_name, 'C0'),
                       label=model_name, alpha=0.8)
            
            ax.set_xlabel('Time Step', fontsize=9)
            ax.set_ylabel(f'{coord_name} (m/s)', fontsize=9)
            ax.set_title(f'{coord_name} 预测对比', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    
    # ========== 误差对比 ==========
    for i, (model_name, data) in enumerate(predictions_dict.items()):
        ax_err = fig.add_subplot(3, 4, 10 + i)
        errors = data['metrics']['errors']
        mae = data['metrics']['mae']
        rmse = data['metrics']['rmse']
        
        bars = ax_err.bar(time_steps, errors, color=colors.get(model_name, f'C{i}'), alpha=0.7, edgecolor='black')
        ax_err.axhline(y=mae, color='r', linestyle='--', linewidth=2, label=f'MAE: {mae:.4f}m')
        ax_err.set_xlabel('Time Step', fontsize=9)
        ax_err.set_ylabel('Error (m)', fontsize=9)
        ax_err.set_title(f'{model_name.replace("position_synth_", "")}\nRMSE: {rmse:.4f}m', fontsize=9, fontweight='bold')
        ax_err.legend(fontsize=8)
        ax_err.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    png_path = output_dir / 'evaluation_result.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    logging.info(f"图像已保存: {png_path}")
    
    return png_path


def main():
    parser = argparse.ArgumentParser(description='评估轨迹预测效果（位置 + 速度联合）')
    parser.add_argument('--trajectory', '-t', 
                       default=r'D:\Trajectory prediction\drone_trajectories\random_traj_100ms\line_10.txt',
                       help='轨迹 CSV 文件路径')
    parser.add_argument('--models-dir', '-m',
                       default=r'D:\Trajectory prediction\drone_trajectories\gru_models_synth',
                       help='训练好的模型目录')
    parser.add_argument('--stats-dir', '-s',
                       default=r'D:\Trajectory prediction\drone_trajectories\transformed_trajectories',
                       help='统计量文件目录')
    parser.add_argument('--output-dir', '-o',
                       default=r'D:\Trajectory prediction\evaluation_results',
                       help='输出目录')
    parser.add_argument('--show', action='store_true', help='显示交互式图像')
    parser.add_argument('--inp-len', type=int, default=20, help='输入序列长度')
    parser.add_argument('--out-len', type=int, default=10, help='输出序列长度')
    parser.add_argument('--dt', type=float, default=0.1, help='采样间隔（秒）')
    
    args = parser.parse_args()
    
    trajectory_path = Path(args.trajectory)
    models_dir = Path(args.models_dir)
    stats_dir = Path(args.stats_dir)
    output_dir = Path(args.output_dir)
    
    # ========== 加载轨迹 ==========
    logging.info(f"加载轨迹: {trajectory_path}")
    trajectory = load_trajectory(trajectory_path)
    logging.info(f"轨迹形状: {trajectory.shape}")
    
    if len(trajectory) < args.inp_len + args.out_len:
        logging.error(f"轨迹太短，至少需要 {args.inp_len + args.out_len} 点")
        return
    
    # 实际未来 10 步
    actual_future = trajectory[-args.out_len:, :]
    
    # 计算速度序列
    actual_velocity = compute_velocity(trajectory, dt=args.dt)
    logging.info(f"速度序列形状: {actual_velocity.shape}")
    
    # ========== 加载位置模型并预测 ==========
    model_configs = [
        ('position_synth_short', 64, 2),
        ('position_synth_mix', 128, 3),
        ('position_synth_long', 256, 5),
    ]
    
    predictions_dict = {}
    velocity_predictions_dict = {}
    
    for model_name, hidden_dim, num_layers in model_configs:
        try:
            model_path = models_dir / f'{model_name}_best_model.pth'
            stats_path = stats_dir / 'pos_stats.npz'
            
            if not model_path.exists():
                logging.warning(f"模型不存在: {model_path}")
                continue
            
            logging.info(f"\n处理位置模型: {model_name}")
            
            # 位置预测
            predictor = TrajectoryPredictor(str(model_path), str(stats_path),
                                           hidden_dim=hidden_dim, num_layers=num_layers)
            pred = predictor.predict(trajectory, inp_len=args.inp_len)
            pred_aligned = align_predictions(pred, actual_future)
            metrics = compute_metrics(pred_aligned, actual_future)
            
            predictions_dict[model_name] = {'pred': pred_aligned, 'metrics': metrics}
            
            logging.info(f"  位置 MAE: {metrics['mae']:.6f} m, RMSE: {metrics['rmse']:.6f} m")
            
        except Exception as e:
            logging.error(f"处理位置模型 {model_name} 失败: {e}")
            continue
    
    # ========== 加载速度模型并预测 ==========
    logging.info(f"\n" + "="*80)
    logging.info("加载速度模型...")
    
    velocity_model_configs = [
        ('velocity_synth_short', 64, 2),
        ('velocity_synth_mix', 128, 3),
        ('velocity_synth_long', 256, 5),
    ]
    
    # 准备统一的速度输入（所有模型用相同输入）
    vel_input = actual_velocity[-args.inp_len:, :]
    start_pos = trajectory[-args.out_len, :]
    
    for model_name, hidden_dim, num_layers in velocity_model_configs:
        try:
            model_path = models_dir / f'{model_name}_best_model.pth'
            stats_path = stats_dir / 'vel_stats.npz'
            
            if not model_path.exists():
                logging.warning(f"速度模型不存在: {model_path}")
                continue
            
            logging.info(f"\n处理速度模型: {model_name}")
            
            # 速度预测
            predictor = TrajectoryPredictor(str(model_path), str(stats_path),
                                           hidden_dim=hidden_dim, num_layers=num_layers)
            
            # 用统一的速度输入（保证所有模型输入相同）
            pred_velocity = predictor.predict(vel_input, inp_len=args.inp_len)
            
            # 从预测速度积分得到位置（用统一的起始位置）
            pred_pos_from_vel = integrate_velocity(start_pos, pred_velocity, dt=args.dt)
            # 取积分后除第一点外的部分（对应 10 步预测）
            pred_pos_from_vel = pred_pos_from_vel[1:, :]
            
            # 对齐
            pred_aligned = align_predictions(pred_pos_from_vel, actual_future)
            metrics = compute_metrics(pred_aligned, actual_future)
            
            velocity_predictions_dict[model_name] = {'pred': pred_velocity, 'metrics': metrics}
            
            logging.info(f"  速度 MAE: {metrics['mae']:.6f} m, RMSE: {metrics['rmse']:.6f} m (via integration)")
            logging.info(f"  速度序列首点: {pred_velocity[0, :]}")
            
        except Exception as e:
            logging.error(f"处理速度模型 {model_name} 失败: {e}")
            continue
    
    if not predictions_dict and not velocity_predictions_dict:
        logging.error("没有成功加载任何模型")
        return
    
    # ========== 可视化 ==========
    logging.info(f"\n生成可视化...")
    png_path = plot_results(trajectory, actual_future, predictions_dict, output_dir,
                           actual_velocity=actual_velocity, 
                           velocity_predictions_dict=velocity_predictions_dict)
    
    # ========== 保存预测结果 ==========
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, data in predictions_dict.items():
        pred_df = pd.DataFrame(data['pred'], columns=['tx', 'ty', 'tz'])
        csv_path = output_dir / f'{model_name}_predictions.csv'
        pred_df.to_csv(csv_path, index=False)
        logging.info(f"预测结果已保存: {csv_path}")
    
    # ========== 保存评估摘要 ==========
    summary_path = output_dir / 'evaluation_summary_with_velocity.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("轨迹评估报告（位置 + 速度联合预测）\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"轨迹文件: {trajectory_path.name}\n")
        f.write(f"轨迹长度: {len(trajectory)} 点\n")
        f.write(f"采样间隔: {args.dt} s\n")
        f.write(f"输入长度: {args.inp_len}\n")
        f.write(f"输出长度: {args.out_len}\n\n")
        
        f.write("-" * 100 + "\n")
        f.write("位置模型评估结果\n")
        f.write("-" * 100 + "\n\n")
        
        for model_name, data in predictions_dict.items():
            metrics = data['metrics']
            f.write(f"模型: {model_name}\n")
            f.write(f"  MAE:         {metrics['mae']:.6f} m\n")
            f.write(f"  RMSE:        {metrics['rmse']:.6f} m\n")
            f.write(f"  Max Error:   {metrics['max_error']:.6f} m\n\n")
        
        f.write("-" * 100 + "\n")
        f.write("速度模型评估结果（位置积分后）\n")
        f.write("-" * 100 + "\n\n")
        
        for model_name, data in velocity_predictions_dict.items():
            metrics = data['metrics']
            f.write(f"模型: {model_name}\n")
            f.write(f"  MAE (integrated):   {metrics['mae']:.6f} m\n")
            f.write(f"  RMSE (integrated):  {metrics['rmse']:.6f} m\n")
            f.write(f"  Max Error:          {metrics['max_error']:.6f} m\n\n")
    
    logging.info(f"评估摘要已保存: {summary_path}")
    
    # ========== 显示图像 ==========
    if args.show:
        logging.info("显示图像...")
        plt.show()
    else:
        try:
            if platform.system() == 'Windows':
                os.startfile(png_path)
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', str(png_path)])
            else:
                subprocess.Popen(['xdg-open', str(png_path)])
            logging.info(f"已用默认查看器打开: {png_path}")
        except Exception as e:
            logging.warning(f"无法自动打开图像: {e}，请手动打开 {png_path}")
    
    logging.info(f"\n✓ 评估完成！所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
