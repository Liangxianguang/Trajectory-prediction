"""
MRGTraj 集群版本推理脚本
========================
使用训练好的模型进行轨迹预测

使用方法:
  python predict_swarm.py --checkpoint checkpoints_swarm/agents_3/best_model.pth \\
                          --num_agents 3 \\
                          --input_file test_past_traj.npy \\
                          --output_file predictions.npy
"""

import argparse
import os
import logging
import numpy as np
import torch
from pathlib import Path

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("Matplotlib 未安装，无法进行可视化")

from model_swarm import MRGTrajSwarm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """轨迹预测可视化"""
    
    @staticmethod
    def plot_3d_trajectories(past_traj, pred_traj, future_traj=None, 
                            num_samples=1, agent_ids=None):
        """
        绘制 3D 轨迹
        
        Args:
            past_traj: (batch_size, obs_len, num_agents, 3)
            pred_traj: (num_samples, batch_size, pred_len, num_agents, 3) 或 (batch_size, pred_len, num_agents, 3)
            future_traj: (batch_size, pred_len, num_agents, 3) 或 None
            num_samples: 预测样本数
            agent_ids: 代理 ID 列表
        """
        batch_size = past_traj.shape[0]
        num_agents = past_traj.shape[2]
        
        if agent_ids is None:
            agent_ids = list(range(num_agents))
        
        # 处理预测形状
        if pred_traj.dim() == 5:  # (num_samples, batch_size, pred_len, num_agents, 3)
            # 只取第一个样本
            pred_traj = pred_traj[0]  # (batch_size, pred_len, num_agents, 3)
        
        # 只绘制第一个批次
        past = past_traj[0].cpu().numpy() if isinstance(past_traj, torch.Tensor) else past_traj[0]
        pred = pred_traj[0].cpu().numpy() if isinstance(pred_traj, torch.Tensor) else pred_traj[0]
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D 轨迹
        ax1 = fig.add_subplot(131, projection='3d')
        colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))
        
        for agent_id in range(num_agents):
            # 过去轨迹
            ax1.plot(past[:, agent_id, 0], past[:, agent_id, 1], past[:, agent_id, 2],
                    'o-', color=colors[agent_id], label=f'Agent {agent_ids[agent_id]} (past)',
                    linewidth=2, markersize=4)
            
            # 预测轨迹
            ax1.plot(pred[:, agent_id, 0], pred[:, agent_id, 1], pred[:, agent_id, 2],
                    's--', color=colors[agent_id], label=f'Agent {agent_ids[agent_id]} (pred)',
                    linewidth=2, markersize=4)
            
            # 如果有真实未来轨迹
            if future_traj is not None:
                future = future_traj[0].cpu().numpy() if isinstance(future_traj, torch.Tensor) else future_traj[0]
                ax1.plot(future[:, agent_id, 0], future[:, agent_id, 1], future[:, agent_id, 2],
                        '^:', color=colors[agent_id], label=f'Agent {agent_ids[agent_id]} (true)',
                        linewidth=1, markersize=3, alpha=0.7)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory Prediction')
        ax1.legend(fontsize=8, loc='upper left')
        ax1.grid(True)
        
        # XY 平面
        ax2 = fig.add_subplot(132)
        for agent_id in range(num_agents):
            ax2.plot(past[:, agent_id, 0], past[:, agent_id, 1], 'o-', 
                    color=colors[agent_id], label=f'Agent {agent_ids[agent_id]} (past)')
            ax2.plot(pred[:, agent_id, 0], pred[:, agent_id, 1], 's--',
                    color=colors[agent_id], label=f'Agent {agent_ids[agent_id]} (pred)')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Plane')
        ax2.legend(fontsize=8)
        ax2.grid(True)
        ax2.axis('equal')
        
        # 高度随时间变化
        ax3 = fig.add_subplot(133)
        time_past = np.arange(past.shape[0])
        time_pred = np.arange(past.shape[0], past.shape[0] + pred.shape[0])
        
        for agent_id in range(num_agents):
            ax3.plot(time_past, past[:, agent_id, 2], 'o-',
                    color=colors[agent_id], label=f'Agent {agent_ids[agent_id]} (past)')
            ax3.plot(time_pred, pred[:, agent_id, 2], 's--',
                    color=colors[agent_id], label=f'Agent {agent_ids[agent_id]} (pred)')
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Height over Time')
        ax3.legend(fontsize=8)
        ax3.grid(True)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_trajectories_csv(past_traj, pred_traj, output_file, num_agents=None):
        """保存为 CSV 格式"""
        if isinstance(past_traj, torch.Tensor):
            past_traj = past_traj.cpu().numpy()
        if isinstance(pred_traj, torch.Tensor):
            pred_traj = pred_traj.cpu().numpy()
        
        # 处理预测形状
        if pred_traj.ndim == 5:  # (num_samples, batch_size, pred_len, num_agents, 3)
            pred_traj = pred_traj[0, 0]  # 取第一个样本的第一个批次
        elif pred_traj.ndim == 4:  # (batch_size, pred_len, num_agents, 3)
            pred_traj = pred_traj[0]
        
        past_traj = past_traj[0]  # 取第一个批次
        
        obs_len = past_traj.shape[0]
        pred_len = pred_traj.shape[0]
        num_agents = past_traj.shape[1]
        
        # 创建完整轨迹
        full_traj = np.concatenate([past_traj, pred_traj], axis=0)  # (obs+pred, num_agents, 3)
        
        # 写入 CSV
        with open(output_file, 'w') as f:
            # 写入头部
            header = "timestamp"
            for agent_id in range(num_agents):
                header += f",agent_{agent_id}_x,agent_{agent_id}_y,agent_{agent_id}_z"
            f.write(header + "\n")
            
            # 写入数据
            dt = 0.1  # 时间步长 (100ms)
            for t, traj_step in enumerate(full_traj):
                timestamp = t * dt
                row = str(timestamp)
                for agent_id in range(num_agents):
                    x, y, z = traj_step[agent_id]
                    row += f",{x},{y},{z}"
                f.write(row + "\n")
        
        logger.info(f"✓ 轨迹已保存到: {output_file}")


def load_checkpoint(checkpoint_path, device='cuda'):
    """加载检查点"""
    logger.info(f"加载检查点: {checkpoint_path}")
    
    # 处理 PyTorch 2.6+ 的安全加载
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        if "Namespace" in str(e):
            torch.serialization.add_safe_globals([argparse.Namespace])
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            raise
    
    args = checkpoint['args']
    
    # 创建模型
    model = MRGTrajSwarm(args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"✓ 模型加载成功")
    logger.info(f"  无人机数量: {args.num_agents}")
    logger.info(f"  观察长度: {args.obs_len}")
    logger.info(f"  预测长度: {args.pred_len}")
    
    return model, args


def predict_from_npz(model, args, input_file, device='cuda', num_samples=10):
    """从 NPZ 文件读取数据并进行预测"""
    logger.info(f"加载输入数据: {input_file}")
    
    data = np.load(input_file)['data']  # (obs_len, num_samples, num_agents, 3)
    logger.info(f"  数据形状: {data.shape}")
    
    obs_len, num_samples, num_agents, agent_dim = data.shape
    
    # 转换为 (batch_size, obs_len, num_agents, 3)
    past_traj = data.transpose(1, 0, 2, 3)
    past_traj = torch.from_numpy(past_traj).float().to(device)
    
    logger.info(f"生成预测 (num_samples={num_samples})...")
    
    with torch.no_grad():
        # 生成多个样本
        predictions = model.inference(past_traj, num_samples=num_samples)
        # predictions: (num_samples, batch_size, pred_len, num_agents, 3)
    
    logger.info(f"✓ 预测完成")
    logger.info(f"  预测形状: {predictions.shape}")
    
    return past_traj, predictions


def predict_from_file(model, args, input_file, device='cuda', num_samples=10):
    """从 CSV 文件读取数据并进行预测"""
    logger.info(f"加载 CSV 数据: {input_file}")
    
    # 读取 CSV
    data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    
    # 提取坐标信息
    # 格式: timestamp, agent_0_x, agent_0_y, agent_0_z, agent_1_x, ...
    coords = data[:, 1:]  # 去掉时间戳
    
    # 确定无人机数量
    num_agents = coords.shape[1] // 3
    
    logger.info(f"  总时间步: {coords.shape[0]}")
    logger.info(f"  无人机数量: {num_agents}")
    
    # 重塑为 (num_timesteps, num_agents, 3)
    trajectory = coords.reshape(coords.shape[0], num_agents, 3)
    
    # 提取过去轨迹 (前 obs_len 步)
    past_traj = trajectory[:args.obs_len]  # (obs_len, num_agents, 3)
    
    # 添加 batch 维度: (1, obs_len, num_agents, 3)
    past_traj = torch.from_numpy(past_traj).float().unsqueeze(0).to(device)
    
    logger.info(f"生成预测...")
    
    with torch.no_grad():
        predictions = model.inference(past_traj, num_samples=num_samples)
        # predictions: (num_samples, 1, pred_len, num_agents, 3)
    
    logger.info(f"✓ 预测完成")
    logger.info(f"  预测形状: {predictions.shape}")
    
    return past_traj, predictions, trajectory


def main():
    parser = argparse.ArgumentParser(description="MRGTraj 集群版本推理脚本")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点文件路径")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="无人机数量")
    parser.add_argument("--input_file", type=str,
                        help="输入文件 (NPZ 或 CSV)")
    parser.add_argument("--output_file", type=str, default="predictions.csv",
                        help="输出预测文件")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="预测样本数")
    parser.add_argument("--visualize", action="store_true",
                        help="是否绘制可视化图表")
    parser.add_argument("--save_plot", type=str,
                        help="保存绘图的文件路径")
    parser.add_argument("--gpu_num", type=str, default="0",
                        help="GPU 编号")
    
    args = parser.parse_args()
    
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model, model_args = load_checkpoint(args.checkpoint, device)
    
    # 进行预测
    if args.input_file.endswith('.npz'):
        past_traj, predictions = predict_from_npz(
            model, model_args, args.input_file, device, args.num_samples
        )
        future_traj = None
    elif args.input_file.endswith('.csv'):
        past_traj, predictions, full_trajectory = predict_from_file(
            model, model_args, args.input_file, device, args.num_samples
        )
        # 如果有足够的数据，提取真实未来轨迹
        if full_trajectory.shape[0] >= model_args.obs_len + model_args.pred_len:
            future_traj = full_trajectory[model_args.obs_len:model_args.obs_len + model_args.pred_len]
            future_traj = torch.from_numpy(future_traj).float().unsqueeze(0)
        else:
            future_traj = None
    else:
        raise ValueError("输入文件必须是 .npz 或 .csv 格式")
    
    # 保存预测结果
    logger.info(f"保存预测结果: {args.output_file}")
    PredictionVisualizer.save_trajectories_csv(
        past_traj, predictions, args.output_file, model_args.num_agents
    )
    
    # 可视化
    if args.visualize or args.save_plot:
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib 未安装，跳过可视化")
        else:
            logger.info("生成可视化...")
            fig = PredictionVisualizer.plot_3d_trajectories(
                past_traj, predictions, future_traj, args.num_samples
            )
            
            if args.save_plot:
                fig.savefig(args.save_plot, dpi=150, bbox_inches='tight')
                logger.info(f"✓ 图表已保存: {args.save_plot}")
            
            if args.visualize:
                plt.show()
    
    logger.info("✓ 推理完成！")


if __name__ == "__main__":
    main()
