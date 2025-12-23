#!/usr/bin/env python3
"""
评估脚本：用训练好的模型预测 gazebo_trajectory_1.csv 并可视化结果
- 加载位置模型（short/mix/long）进行预测
- 计算 MAE/RMSE 等指标
- 生成 3D 轨迹 + 坐标对比 + 误差分析图
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

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / 'drone_trajectories'))
sys.path.insert(0, str(Path(__file__).parent / 'drone_path_predictor_ros-main'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class TrajectoryPredictor:
    """轨迹预测器（位置）"""
    def __init__(self, model_path, stats_path, hidden_dim=64, num_layers=2, device='cuda'):
        from train_model import GRUModel
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 加载统计量（处理缺失的 output_mean/output_std）
        stats = np.load(stats_path)
        self.input_mean = torch.tensor(stats['input_mean'], dtype=torch.float32, device=self.device)
        self.input_std = torch.tensor(stats['input_std'], dtype=torch.float32, device=self.device)
        
        # 若缺少 output_mean/output_std，使用 input_mean/input_std 作为默认值
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
    # 提取 tx, ty, tz 列
    traj = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    return traj


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


def plot_results(trajectory, actual_future, predictions_dict, out_dir):
    """
    可视化：3D 轨迹 + 坐标对比 + 误差分析
    predictions_dict: {'model_name': {'pred': pred_aligned, 'metrics': metrics}, ...}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 颜色配置
    colors = {
        'position_synth_short': '#FF6B6B',
        'position_synth_mix': '#4ECDC4',
        'position_synth_long': "#850742",
    }
    
    fig = plt.figure(figsize=(20, 14))
    
    # ========== 第一个子图: 3D 轨迹对比 ==========
    ax_3d = fig.add_subplot(3, 3, 1, projection='3d')
    
    # 历史轨迹（最近 100 步）
    history_len = min(100, len(trajectory) - 10)
    history = trajectory[-history_len-10:-10, :]
    ax_3d.plot(history[:, 0], history[:, 1], history[:, 2], 'b-', linewidth=2.5, label='History (latest 100)', alpha=0.7)
    
    # 实际未来
    ax_3d.plot(actual_future[:, 0], actual_future[:, 1], actual_future[:, 2], 'g-o', 
              linewidth=3, markersize=8, label='Actual Future', alpha=0.9, zorder=5)
    
    # 预测结果
    for i, (model_name, data) in enumerate(predictions_dict.items()):
        pred = data['pred']
        ax_3d.plot(pred[:, 0], pred[:, 1], pred[:, 2], linestyle='--', marker='s', 
                  linewidth=2, markersize=6, color=colors.get(model_name, f'C{i}'),
                  label=f'Predicted ({model_name})', alpha=0.8)
    
    ax_3d.set_xlabel('X (m)', fontsize=10)
    ax_3d.set_ylabel('Y (m)', fontsize=10)
    ax_3d.set_zlabel('Z (m)', fontsize=10)
    ax_3d.set_title('3D 轨迹对比', fontsize=12, fontweight='bold')
    ax_3d.legend(fontsize=9, loc='upper left')
    ax_3d.grid(True, alpha=0.3)
    
    # ========== X, Y, Z 坐标对比 ==========
    time_steps = np.arange(1, 11)
    
    for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(3, 3, coord_idx + 2)
        ax.plot(time_steps, actual_future[:, coord_idx], 'g-o', linewidth=3, 
               markersize=8, label='Actual', alpha=0.9, zorder=5)
        
        for model_name, data in predictions_dict.items():
            pred = data['pred']
            ax.plot(time_steps, pred[:, coord_idx], linestyle='--', marker='s',
                   linewidth=2, markersize=6, color=colors.get(model_name, f'C{list(predictions_dict.keys()).index(model_name)}'),
                   label=model_name, alpha=0.8)
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel(f'{coord_name} (m)', fontsize=10)
        ax.set_title(f'{coord_name} 坐标预测对比', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    # ========== 误差分析 ==========
    for i, (model_name, data) in enumerate(predictions_dict.items()):
        ax_err = fig.add_subplot(3, 3, 6 + i)
        errors = data['metrics']['errors']
        mae = data['metrics']['mae']
        rmse = data['metrics']['rmse']
        
        bars = ax_err.bar(time_steps, errors, color=colors.get(model_name, f'C{i}'), alpha=0.7, edgecolor='black')
        ax_err.axhline(y=mae, color='r', linestyle='--', linewidth=2, label=f'MAE: {mae:.4f}m')
        ax_err.set_xlabel('Time Step', fontsize=10)
        ax_err.set_ylabel('Error (m)', fontsize=10)
        ax_err.set_title(f'{model_name}\nRMSE: {rmse:.4f}m', fontsize=10, fontweight='bold')
        ax_err.legend(fontsize=9)
        ax_err.grid(True, alpha=0.3, axis='y')
    
    # ========== 指标汇总表 ==========
    ax_table = fig.add_subplot(3, 3, 9)
    ax_table.axis('off')
    
    table_data = []
    table_data.append(['Model', 'MAE (m)', 'RMSE (m)', 'Max Error (m)'])
    for model_name, data in predictions_dict.items():
        metrics = data['metrics']
        table_data.append([
            model_name.replace('position_synth_', ''),
            f"{metrics['mae']:.6f}",
            f"{metrics['rmse']:.6f}",
            f"{metrics['max_error']:.6f}"
        ])
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 表头着色
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax_table.set_title('评估指标汇总', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图像
    png_path = out_dir / 'evaluation_result.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    logging.info(f"图像已保存: {png_path}")
    
    return png_path


def main():
    parser = argparse.ArgumentParser(description='评估 gazebo_trajectory_1.csv 的预测效果')
    parser.add_argument('--trajectory', '-t', 
                       default=r'D:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory2D-2_266.csv',
                       help='轨迹 CSV 文件路径')
    parser.add_argument('--models-dir', '-m',
                       default=r'D:\Trajectory prediction\drone_trajectories\gru_models_synth',
                       help='训练好的模型目录')
    parser.add_argument('--stats-dir', '-s',
                       default=r'D:\Trajectory prediction\drone_trajectories\transformed_trajectories',
                       help='统计量文件目录')
    parser.add_argument('--output-dir', '-o',
                       default=r'D:\Trajectory prediction\evaluation_results',
                       help='输出目录（保存图像、CSV、统计量）')
    parser.add_argument('--show', action='store_true', help='显示交互式图像')
    parser.add_argument('--inp-len', type=int, default=20, help='输入序列长度')
    parser.add_argument('--out-len', type=int, default=10, help='输出序列长度')
    
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
    
    # ========== 加载模型并预测 ==========
    model_configs = [
        ('position_synth_short', 64, 2),
        ('position_synth_mix', 128, 3),
        ('position_synth_long', 256, 5),
    ]
    
    predictions_dict = {}
    
    for model_name, hidden_dim, num_layers in model_configs:
        try:
            model_path = models_dir / f'{model_name}_best_model.pth'
            stats_path = stats_dir / 'pos_stats.npz'
            
            if not model_path.exists():
                logging.warning(f"模型不存在: {model_path}")
                continue
            
            logging.info(f"\n处理模型: {model_name}")
            
            # 创建预测器
            predictor = TrajectoryPredictor(
                str(model_path),
                str(stats_path),
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
            
            # 预测
            pred = predictor.predict(trajectory, inp_len=args.inp_len)
            
            # 对齐
            pred_aligned = align_predictions(pred, actual_future)
            
            # 计算指标
            metrics = compute_metrics(pred_aligned, actual_future)
            
            predictions_dict[model_name] = {
                'pred': pred_aligned,
                'metrics': metrics
            }
            
            logging.info(f"  MAE: {metrics['mae']:.6f} m")
            logging.info(f"  RMSE: {metrics['rmse']:.6f} m")
            logging.info(f"  Max Error: {metrics['max_error']:.6f} m")
            
        except Exception as e:
            logging.error(f"处理 {model_name} 失败: {e}")
            continue
    
    if not predictions_dict:
        logging.error("没有成功加载任何模型")
        return
    
    # ========== 可视化 ==========
    logging.info(f"\n生成可视化...")
    png_path = plot_results(trajectory, actual_future, predictions_dict, output_dir)
    
    # ========== 保存预测结果为 CSV ==========
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, data in predictions_dict.items():
        pred_df = pd.DataFrame(data['pred'], columns=['tx', 'ty', 'tz'])
        csv_path = output_dir / f'{model_name}_predictions.csv'
        pred_df.to_csv(csv_path, index=False)
        logging.info(f"预测结果已保存: {csv_path}")
    
    # ========== 保存评估摘要 ==========
    summary_path = output_dir / 'evaluation_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"轨迹评估报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"轨迹文件: {trajectory_path.name}\n")
        f.write(f"轨迹长度: {len(trajectory)} 点\n")
        f.write(f"输入长度: {args.inp_len}\n")
        f.write(f"输出长度: {args.out_len}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("模型评估结果\n")
        f.write("-" * 80 + "\n\n")
        
        for model_name, data in predictions_dict.items():
            metrics = data['metrics']
            f.write(f"模型: {model_name}\n")
            f.write(f"  MAE:         {metrics['mae']:.6f} m\n")
            f.write(f"  RMSE:        {metrics['rmse']:.6f} m\n")
            f.write(f"  Max Error:   {metrics['max_error']:.6f} m\n")
            f.write(f"  Min Error:   {metrics['min_error']:.6f} m\n")
            f.write("\n")
    
    logging.info(f"评估摘要已保存: {summary_path}")
    
    # ========== 显示图像 ==========
    if args.show:
        logging.info("显示图像...")
        plt.show()
    else:
        # 若未指定 --show，自动用系统默认图像查看器打开 PNG
        import subprocess
        import platform
        try:
            if platform.system() == 'Windows':
                os.startfile(png_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['open', str(png_path)])
            else:  # Linux
                subprocess.Popen(['xdg-open', str(png_path)])
            logging.info(f"已用默认查看器打开: {png_path}")
        except Exception as e:
            logging.warning(f"无法自动打开图像: {e}，请手动打开 {png_path}")
    
    logging.info(f"\n✓ 评估完成！所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
