#!/usr/bin/env python3
"""
集群轨迹数据集处理器
将 NPZ 格式的集群轨迹转换为 [B,L,N,D] 的深度学习训练格式

数据格式：
  输入: X shape = [B, L, N, D]
    B: 批量大小（轨迹段数）
    L: 时间步长（输入序列长度）
    N: 目标数量
    D: 特征维度（位置3 + 速度3 + 加速度3 = 9，或其他）
  
  输出: Y shape = [B, K, N, D]
    K: 预测时间步数
    其他同上

  掩码: mask shape = [B, L, N]
    用于标记有效的目标（1=有效, 0=无效/丢失）

使用方法：
    python swarm_dataset_processor.py \
        --input_dir swarm_trajectories \
        --output_dir swarm_dataset \
        --input_length 20 \
        --output_length 10 \
        --features position,velocity,acceleration
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Tuple, List, Dict, Optional
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SwarmDatasetProcessor:
    """集群轨迹数据处理器"""
    
    def __init__(self,
                 input_length: int = 20,
                 output_length: int = 10,
                 features: List[str] = None,
                 normalize: bool = True,
                 stride: int = 1):
        """
        初始化处理器
        
        Args:
            input_length: 输入序列长度
            output_length: 输出序列长度
            features: 特征列表 ['position', 'velocity', 'acceleration']
            normalize: 是否进行归一化
            stride: 滑动窗口步长
        """
        self.input_length = input_length
        self.output_length = output_length
        self.features = features or ['position', 'velocity']
        self.normalize = normalize
        self.stride = stride
        
        # 统计数据（用于归一化）
        self.stats = {
            'position_mean': None,
            'position_std': None,
            'velocity_mean': None,
            'velocity_std': None,
            'acceleration_mean': None,
            'acceleration_std': None,
        }
    
    def _compute_velocity(self, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        计算速度（差分）
        
        Args:
            trajectory: 轨迹数组 (T, N, 3)
            dt: 时间步长
            
        Returns:
            速度数组 (T, N, 3)
        """
        vel = np.zeros_like(trajectory)
        vel[1:] = (trajectory[1:] - trajectory[:-1]) / dt
        vel[0] = vel[1]  # 第一帧填充为第二帧的值
        return vel
    
    def _compute_acceleration(self, velocity: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        计算加速度
        
        Args:
            velocity: 速度数组 (T, N, 3)
            dt: 时间步长
            
        Returns:
            加速度数组 (T, N, 3)
        """
        acc = np.zeros_like(velocity)
        acc[1:] = (velocity[1:] - velocity[:-1]) / dt
        acc[0] = acc[1]
        return acc
    
    def _extract_features(self, trajectory: np.ndarray, 
                         dt: float = 0.1) -> np.ndarray:
        """
        提取特征
        
        Args:
            trajectory: 轨迹 (T, N, 3)
            dt: 时间步长
            
        Returns:
            特征数组 (T, N, D) 其中 D 取决于选择的特征
        """
        features_list = []
        
        if 'position' in self.features:
            features_list.append(trajectory)
        
        if 'velocity' in self.features:
            vel = self._compute_velocity(trajectory, dt)
            features_list.append(vel)
        
        if 'acceleration' in self.features:
            vel = self._compute_velocity(trajectory, dt)
            acc = self._compute_acceleration(vel, dt)
            features_list.append(acc)
        
        # 按特征维度拼接
        features = np.concatenate(features_list, axis=-1)  # (T, N, D)
        return features.astype(np.float32)
    
    def _compute_statistics(self, features: np.ndarray):
        """
        计算特征统计量（用于归一化）
        
        Args:
            features: 特征数组 (T, N, D)
        """
        # 展平为 (T*N, D)
        features_flat = features.reshape(-1, features.shape[-1])
        
        self.stats['position_mean'] = np.mean(features_flat[:, :3], axis=0, keepdims=True)
        self.stats['position_std'] = np.std(features_flat[:, :3], axis=0, keepdims=True) + 1e-6
        
        idx = 3
        if 'velocity' in self.features:
            self.stats['velocity_mean'] = np.mean(features_flat[:, idx:idx+3], axis=0, keepdims=True)
            self.stats['velocity_std'] = np.std(features_flat[:, idx:idx+3], axis=0, keepdims=True) + 1e-6
            idx += 3
        
        if 'acceleration' in self.features:
            self.stats['acceleration_mean'] = np.mean(features_flat[:, idx:idx+3], axis=0, keepdims=True)
            self.stats['acceleration_std'] = np.std(features_flat[:, idx:idx+3], axis=0, keepdims=True) + 1e-6
        
        logger.info(f"统计信息计算完成")
        logger.info(f"  位置均值: {self.stats['position_mean'].flatten()}")
        logger.info(f"  位置标准差: {self.stats['position_std'].flatten()}")
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        特征归一化
        
        Args:
            features: 特征数组 (*, D)
            
        Returns:
            归一化后的特征
        """
        if not self.normalize or self.stats['position_mean'] is None:
            return features
        
        features_norm = features.copy()
        
        # 位置归一化
        features_norm[..., :3] = (features[..., :3] - self.stats['position_mean']) / self.stats['position_std']
        
        # 速度归一化
        idx = 3
        if 'velocity' in self.features:
            features_norm[..., idx:idx+3] = (features[..., idx:idx+3] - self.stats['velocity_mean']) / self.stats['velocity_std']
            idx += 3
        
        # 加速度归一化
        if 'acceleration' in self.features:
            features_norm[..., idx:idx+3] = (features[..., idx:idx+3] - self.stats['acceleration_mean']) / self.stats['acceleration_std']
        
        return features_norm
    
    def process_trajectory(self, swarm_trajectory: np.ndarray,
                          dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        处理单条集群轨迹，生成输入输出对
        
        Args:
            swarm_trajectory: 集群轨迹 (T, N, 3)
            dt: 时间步长
            
        Returns:
            (X, Y, mask): 
            - X: 输入特征 (B, L, N, D)
            - Y: 输出特征 (B, K, N, D)
            - mask: 有效性掩码 (B, L, N)
        """
        T, N, _ = swarm_trajectory.shape
        
        # 提取特征
        features = self._extract_features(swarm_trajectory, dt)  # (T, N, D)
        
        # 生成输入输出对
        X_list = []
        Y_list = []
        mask_list = []
        
        for start_idx in range(0, T - self.input_length - self.output_length + 1, self.stride):
            end_idx = start_idx + self.input_length
            output_end = end_idx + self.output_length
            
            if output_end > T:
                break
            
            # 输入段 (L, N, D)
            x_seg = features[start_idx:end_idx]
            
            # 输出段 (K, N, D)
            y_seg = features[end_idx:output_end]
            
            X_list.append(x_seg)
            Y_list.append(y_seg)
            
            # 全有效掩码
            mask_seg = np.ones((self.input_length, N), dtype=np.uint8)
            mask_list.append(mask_seg)
        
        if len(X_list) == 0:
            logger.warning(f"轨迹长度不足 (需要 >= {self.input_length + self.output_length})")
            return np.array([]), np.array([]), np.array([])
        
        # 沿批次维度堆叠
        X = np.stack(X_list, axis=0)  # (B, L, N, D)
        Y = np.stack(Y_list, axis=0)  # (B, K, N, D)
        mask = np.stack(mask_list, axis=0)  # (B, L, N)
        
        return X, Y, mask
    
    def process_dataset(self, input_dir: str, output_dir: str) -> Dict:
        """
        处理整个数据集
        
        注意：支持可变的无人机数量（3-6架）
        
        Args:
            input_dir: 输入集群轨迹 NPZ 目录
            output_dir: 输出目录
            
        Returns:
            统计信息字典
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 列出所有 NPZ 文件
        npz_files = sorted(list(input_path.glob('*.npz')))
        logger.info(f"找到 {len(npz_files)} 个集群轨迹文件")
        
        all_X = []
        all_Y = []
        all_mask = []
        all_features_for_stats = []
        
        # 第一遍：计算统计量（采样方式，避免内存溢出）
        logger.info("\n第一遍：计算统计量...")
        sample_size = min(200, len(npz_files))
        sample_indices = np.linspace(0, len(npz_files)-1, sample_size, dtype=int)
        
        for sample_idx, file_idx in enumerate(sample_indices, 1):
            try:
                npz_file = npz_files[file_idx]
                data = np.load(npz_file)
                swarm_traj = data['swarm_trajectory']  # (T, N, 3)
                
                # 提取特征用于统计
                features = self._extract_features(swarm_traj)  # (T, N, D)
                
                # 展平为 (T*N, D) 用于统计
                features_flat = features.reshape(-1, features.shape[-1])
                all_features_for_stats.append(features_flat)
                
                if sample_idx % max(1, sample_size // 10) == 0:
                    logger.info(f"  采样 {sample_idx}/{sample_size} 个文件")
            except Exception as e:
                logger.error(f"处理 {npz_file.name} 失败: {e}")
                continue
        
        # 计算统计量（从采样数据）
        if all_features_for_stats:
            features_all = np.concatenate(all_features_for_stats, axis=0)  # (T*N*sample, D)
            self._compute_statistics(features_all)
            logger.info(f"  统计样本数: {len(features_all)}")
        
        # 第二遍：处理所有文件并保存
        logger.info("\n第二遍：生成训练数据...")
        total_samples = 0
        processed_count = 0
        failed_count = 0
        
        # 按无人机数分组处理
        trajectories_by_num_agents = {}
        
        for idx, npz_file in enumerate(npz_files, 1):
            try:
                data = np.load(npz_file)
                swarm_traj = data['swarm_trajectory']  # (T, N, 3)
                num_agents = int(data['num_agents'])
                
                # 处理轨迹
                X, Y, mask = self.process_trajectory(swarm_traj)
                
                if len(X) == 0:
                    continue
                
                # 归一化
                X = self._normalize_features(X)
                Y = self._normalize_features(Y)
                
                # 保存个别文件
                output_file = output_path / f"{npz_file.stem}_processed.npz"
                np.savez(
                    output_file,
                    X=X,  # (B, L, N, D) - 可变 N
                    Y=Y,  # (B, K, N, D) - 可变 N
                    mask=mask,  # (B, L, N)
                    num_agents=num_agents,
                    source_file=str(npz_file.name)
                )
                
                # 按无人机数分组（用于后续分析）
                if num_agents not in trajectories_by_num_agents:
                    trajectories_by_num_agents[num_agents] = []
                trajectories_by_num_agents[num_agents].append((X, Y, mask))
                
                total_samples += len(X)
                processed_count += 1
                
                if idx % max(1, len(npz_files) // 20) == 0:
                    logger.info(f"  已处理 {idx}/{len(npz_files)} 个文件 (成功: {processed_count}, 失败: {failed_count})")
            
            except Exception as e:
                logger.error(f"处理 {npz_file.name} 失败: {e}")
                failed_count += 1
                continue
        
        logger.info(f"\n  总计: {processed_count} 成功，{failed_count} 失败")
        logger.info(f"\n按无人机数分布:")
        for n_agents in sorted(trajectories_by_num_agents.keys()):
            count = len(trajectories_by_num_agents[n_agents])
            logger.info(f"  {n_agents} 架无人机: {count} 个轨迹")
        
        # 为每个无人机数量级别创建合并数据集
        logger.info("\n生成分组数据集...")
        for num_agents in sorted(trajectories_by_num_agents.keys()):
            traj_list = trajectories_by_num_agents[num_agents]
            X_list, Y_list, mask_list = zip(*traj_list)
            
            try:
                # 拼接同一组的数据（维度相同）
                X_group = np.concatenate(X_list, axis=0)  # (B_group, L, N, D)
                Y_group = np.concatenate(Y_list, axis=0)  # (B_group, K, N, D)
                mask_group = np.concatenate(mask_list, axis=0)  # (B_group, L, N)
                
                # 保存分组数据集
                group_file = output_path / f'swarm_dataset_agents_{num_agents}.npz'
                np.savez(
                    group_file,
                    X=X_group,
                    Y=Y_group,
                    mask=mask_group,
                    num_agents=num_agents,
                    input_length=self.input_length,
                    output_length=self.output_length,
                    features=self.features,
                    num_samples=len(X_group)
                )
                
                logger.info(f"  {num_agents} 架无人机: X{X_group.shape} Y{Y_group.shape} "
                           f"({len(X_group)} 样本) -> {group_file.name}")
            except Exception as e:
                logger.error(f"处理 {num_agents} 架无人机的数据失败: {e}")
                continue
        
        # 保存统计量
        stats_file = output_path / 'dataset_stats.npz'
        np.savez(stats_file, **{k: v for k, v in self.stats.items() if v is not None})
        logger.info(f"✓ 统计量保存到: {stats_file.name}")
        
        # 保存配置
        config = {
            'input_length': self.input_length,
            'output_length': self.output_length,
            'features': self.features,
            'normalize': self.normalize,
            'stride': self.stride,
            'total_files': len(npz_files),
            'processed_files': processed_count,
            'failed_files': failed_count,
            'total_samples': total_samples,
            'num_agents_distribution': {k: len(v) for k, v in trajectories_by_num_agents.items()}
        }
        config_file = output_path / 'dataset_config.pkl'
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        logger.info(f"✓ 配置保存到: {config_file.name}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✓ 数据集处理完成!")
        logger.info(f"{'='*70}")
        logger.info(f"✓ 已处理: {processed_count} 个文件")
        logger.info(f"✗ 失败: {failed_count} 个文件")
        logger.info(f"✓ 总样本数: {total_samples}")
        logger.info(f"✓ 输入长度: {self.input_length}")
        logger.info(f"✓ 输出长度: {self.output_length}")
        logger.info(f"✓ 特征: {', '.join(self.features)}")
        logger.info(f"✓ 输出目录: {output_path}")
        logger.info(f"{'='*70}")
        
        return config


def main():
    parser = argparse.ArgumentParser(
        description='集群轨迹数据集处理 - 生成 [B,L,N,D] 格式的训练数据'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入集群轨迹 NPZ 目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出处理后的数据集目录')
    parser.add_argument('--input_length', type=int, default=20,
                        help='输入序列长度 (默认: 20)')
    parser.add_argument('--output_length', type=int, default=10,
                        help='输出序列长度 (默认: 10)')
    parser.add_argument('--features', type=str, default='position,velocity,acceleration',
                        help='特征列表，逗号分隔 (默认: position,velocity,acceleration)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='是否归一化 (默认: 是)')
    parser.add_argument('--stride', type=int, default=1,
                        help='滑动窗口步长 (默认: 1)')
    
    args = parser.parse_args()
    
    # 解析特征列表
    features = [f.strip() for f in args.features.split(',')]
    
    # 创建处理器并处理数据集
    processor = SwarmDatasetProcessor(
        input_length=args.input_length,
        output_length=args.output_length,
        features=features,
        normalize=args.normalize,
        stride=args.stride
    )
    
    processor.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
