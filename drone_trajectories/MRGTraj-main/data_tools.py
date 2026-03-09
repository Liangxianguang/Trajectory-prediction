"""
数据兼容性检查和转换工具
========================

用于检查和转换无人机集群轨迹数据格式

功能:
  1. 验证 NPZ 数据格式
  2. 验证 CSV 数据格式
  3. NPZ <-> CSV 相互转换
  4. 数据统计和可视化
"""

import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证工具"""
    
    @staticmethod
    def validate_npz(npz_file):
        """验证 NPZ 文件格式"""
        logger.info(f"验证 NPZ 文件: {npz_file}")
        
        try:
            data = np.load(npz_file)['data']
            
            logger.info(f"✓ 文件格式有效")
            logger.info(f"  形状: {data.shape}")
            logger.info(f"  数据类型: {data.dtype}")
            logger.info(f"  内存占用: {data.nbytes / 1024 / 1024:.2f} MB")
            
            # 检查数据统计
            logger.info(f"  数据统计:")
            logger.info(f"    Min: {data.min():.4f}")
            logger.info(f"    Max: {data.max():.4f}")
            logger.info(f"    Mean: {data.mean():.4f}")
            logger.info(f"    Std: {data.std():.4f}")
            
            # 检查 NaN/Inf
            num_nan = np.isnan(data).sum()
            num_inf = np.isinf(data).sum()
            
            if num_nan > 0:
                logger.warning(f"  ⚠️ 发现 {num_nan} 个 NaN 值")
            if num_inf > 0:
                logger.warning(f"  ⚠️ 发现 {num_inf} 个 Inf 值")
            
            if num_nan == 0 and num_inf == 0:
                logger.info(f"  ✓ 数据质量良好")
            
            return True, data
        
        except Exception as e:
            logger.error(f"✗ 文件格式错误: {e}")
            return False, None
    
    @staticmethod
    def validate_csv(csv_file):
        """验证 CSV 文件格式"""
        logger.info(f"验证 CSV 文件: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            
            logger.info(f"✓ 文件格式有效")
            logger.info(f"  行数: {len(df)}")
            logger.info(f"  列数: {len(df.columns)}")
            logger.info(f"  列名: {list(df.columns[:5])}...")
            
            # 计算无人机数量
            # 格式: timestamp, agent_0_x, agent_0_y, agent_0_z, ...
            num_coords = len(df.columns) - 1
            if num_coords % 3 != 0:
                logger.warning(f"  ⚠️ 坐标列数不是 3 的倍数")
                return False, None
            
            num_agents = num_coords // 3
            logger.info(f"  无人机数量: {num_agents}")
            
            # 检查数据统计
            coords_df = df.iloc[:, 1:]
            logger.info(f"  坐标统计:")
            logger.info(f"    Min: {coords_df.values.min():.4f}")
            logger.info(f"    Max: {coords_df.values.max():.4f}")
            logger.info(f"    Mean: {coords_df.values.mean():.4f}")
            
            # 检查缺失值
            num_missing = df.isnull().sum().sum()
            if num_missing > 0:
                logger.warning(f"  ⚠️ 发现 {num_missing} 个缺失值")
            else:
                logger.info(f"  ✓ 无缺失值")
            
            return True, df
        
        except Exception as e:
            logger.error(f"✗ 文件格式错误: {e}")
            return False, None


class DataConverter:
    """数据转换工具"""
    
    @staticmethod
    def npz_to_csv(npz_file, output_csv, dt=0.1):
        """将 NPZ 转换为 CSV"""
        logger.info(f"将 NPZ 转换为 CSV...")
        
        valid, data = DataValidator.validate_npz(npz_file)
        if not valid:
            return False
        
        # 数据形状: (seq_len, num_samples, num_agents, 3)
        seq_len, num_samples, num_agents, _ = data.shape
        
        # 只转换第一个样本
        traj = data[:, 0, :, :]  # (seq_len, num_agents, 3)
        
        logger.info(f"转换第一个样本: {traj.shape}")
        
        # 创建 DataFrame
        rows = []
        for t, traj_step in enumerate(traj):
            row = {'timestamp': t * dt}
            for agent_id in range(num_agents):
                x, y, z = traj_step[agent_id]
                row[f'agent_{agent_id}_x'] = x
                row[f'agent_{agent_id}_y'] = y
                row[f'agent_{agent_id}_z'] = z
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        
        logger.info(f"✓ 已保存到: {output_csv}")
        return True
    
    @staticmethod
    def csv_to_npz(csv_file, output_npz):
        """将 CSV 转换为 NPZ"""
        logger.info(f"将 CSV 转换为 NPZ...")
        
        valid, df = DataValidator.validate_csv(csv_file)
        if not valid:
            return False
        
        # 提取坐标
        coords = df.iloc[:, 1:].values  # (seq_len, num_coords)
        
        num_coords = coords.shape[1]
        num_agents = num_coords // 3
        seq_len = coords.shape[0]
        
        # 重塑为 (seq_len, num_agents, 3)
        traj = coords.reshape(seq_len, num_agents, 3)
        
        # 添加样本维度: (seq_len, 1, num_agents, 3)
        data = traj[:, np.newaxis, :, :]
        
        # 保存
        np.savez_compressed(output_npz, data=data)
        
        logger.info(f"✓ 已保存到: {output_npz}")
        logger.info(f"  形状: {data.shape}")
        
        return True


class DataVisualizer:
    """数据可视化工具"""
    
    @staticmethod
    def plot_trajectory_3d(data, sample_idx=0, title=None):
        """绘制 3D 轨迹"""
        if isinstance(data, str):
            # 从文件加载
            if data.endswith('.npz'):
                npz_data = np.load(data)['data']
                traj = npz_data[:, sample_idx, :, :]
            elif data.endswith('.csv'):
                df = pd.read_csv(data)
                coords = df.iloc[:, 1:].values
                num_agents = coords.shape[1] // 3
                traj = coords.reshape(-1, num_agents, 3)
        else:
            traj = data
        
        seq_len, num_agents, _ = traj.shape
        
        fig = plt.figure(figsize=(12, 4))
        
        # 3D 轨迹
        ax1 = fig.add_subplot(131, projection='3d')
        colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))
        
        for agent_id in range(num_agents):
            ax1.plot(traj[:, agent_id, 0], traj[:, agent_id, 1], traj[:, agent_id, 2],
                    'o-', color=colors[agent_id], label=f'Agent {agent_id}')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(title or '3D Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # XY 平面
        ax2 = fig.add_subplot(132)
        for agent_id in range(num_agents):
            ax2.plot(traj[:, agent_id, 0], traj[:, agent_id, 1], 'o-',
                    color=colors[agent_id], label=f'Agent {agent_id}')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Plane')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # 高度随时间
        ax3 = fig.add_subplot(133)
        time = np.arange(seq_len)
        for agent_id in range(num_agents):
            ax3.plot(time, traj[:, agent_id, 2], 'o-',
                    color=colors[agent_id], label=f'Agent {agent_id}')
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Height over Time')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_statistics(npz_file):
        """绘制数据统计"""
        logger.info(f"绘制数据统计...")
        
        data = np.load(npz_file)['data']  # (seq_len, num_samples, num_agents, 3)
        
        seq_len, num_samples, num_agents, _ = data.shape
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # X 坐标分布
        axes[0, 0].hist(data[:, :, :, 0].flatten(), bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_title('X Coordinate Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Y 坐标分布
        axes[0, 1].hist(data[:, :, :, 1].flatten(), bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Y (m)')
        axes[0, 1].set_title('Y Coordinate Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Z 坐标分布
        axes[0, 2].hist(data[:, :, :, 2].flatten(), bins=50, edgecolor='black')
        axes[0, 2].set_xlabel('Z (m)')
        axes[0, 2].set_title('Z Coordinate Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 样本大小统计
        axes[1, 0].bar(range(num_agents), [num_samples] * num_agents)
        axes[1, 0].set_xlabel('Agent ID')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Samples per Agent')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 速度统计 (相邻时间步的位移)
        velocity = np.sqrt(np.sum(np.diff(data, axis=0) ** 2, axis=-1))
        axes[1, 1].hist(velocity.flatten(), bins=50, edgecolor='black')
        axes[1, 1].set_xlabel('Velocity (m/step)')
        axes[1, 1].set_title('Velocity Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 代理间距离
        distances = []
        for sample_idx in range(min(100, num_samples)):  # 只计算前 100 个样本
            for agent_i in range(num_agents):
                for agent_j in range(agent_i + 1, num_agents):
                    dist = np.linalg.norm(
                        data[:, sample_idx, agent_i, :] - data[:, sample_idx, agent_j, :],
                        axis=-1
                    )
                    distances.extend(dist)
        
        axes[1, 2].hist(distances, bins=50, edgecolor='black')
        axes[1, 2].set_xlabel('Distance (m)')
        axes[1, 2].set_title('Inter-agent Distance Distribution')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def main():
    parser = argparse.ArgumentParser(description="数据兼容性检查和转换工具")
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证数据文件')
    validate_parser.add_argument('file', help='要验证的文件 (NPZ 或 CSV)')
    
    # 转换命令
    convert_parser = subparsers.add_parser('convert', help='转换数据格式')
    convert_parser.add_argument('input_file', help='输入文件')
    convert_parser.add_argument('output_file', help='输出文件')
    
    # 可视化命令
    plot_parser = subparsers.add_parser('plot', help='绘制数据')
    plot_parser.add_argument('file', help='数据文件')
    plot_parser.add_argument('--type', choices=['traj', 'stats'], default='traj',
                            help='绘图类型')
    plot_parser.add_argument('--sample', type=int, default=0,
                            help='样本索引 (仅用于轨迹绘图)')
    plot_parser.add_argument('--save', help='保存图表到文件')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("数据兼容性检查和转换工具")
    logger.info("=" * 80)
    
    if args.command == 'validate':
        file_ext = Path(args.file).suffix.lower()
        if file_ext == '.npz':
            DataValidator.validate_npz(args.file)
        elif file_ext == '.csv':
            DataValidator.validate_csv(args.file)
        else:
            logger.error("不支持的文件类型")
    
    elif args.command == 'convert':
        input_ext = Path(args.input_file).suffix.lower()
        output_ext = Path(args.output_file).suffix.lower()
        
        if input_ext == '.npz' and output_ext == '.csv':
            DataConverter.npz_to_csv(args.input_file, args.output_file)
        elif input_ext == '.csv' and output_ext == '.npz':
            DataConverter.csv_to_npz(args.input_file, args.output_file)
        else:
            logger.error("不支持的转换类型")
    
    elif args.command == 'plot':
        if args.type == 'traj':
            fig = DataVisualizer.plot_trajectory_3d(args.file, args.sample)
        else:
            fig = DataVisualizer.plot_statistics(args.file)
        
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            logger.info(f"✓ 图表已保存: {args.save}")
        else:
            plt.show()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
