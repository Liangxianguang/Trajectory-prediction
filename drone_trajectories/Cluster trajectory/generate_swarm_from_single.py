#!/usr/bin/env python3
"""
基于单机轨迹生成集群轨迹数据集
实现 Couzin 模型：排斥 + 定向 + 吸引规则

使用方法：
    python generate_swarm_from_single.py \
        --input_dir ../random_traj_100ms \
        --output_dir swarm_trajectories \
        --num_agents 5 \
        --num_copies 100 \
        --dt 0.1
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Tuple, List, Dict
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CouzinSwarmGenerator:
    """
    基于 Couzin 模型生成集群轨迹
    
    公式：r_i(t+1) = r_leader(t) + Δ_i(t) + ζ_i(t)
    其中：
    - r_leader(t): 领航者位置
    - Δ_i(t): 相对于其他目标的相对位移（基于排斥、定向、吸引）
    - ζ_i(t): 随机扰动
    """
    
    def __init__(self,
                 num_agents: int = 5,
                 repulsion_distance: float = 2.0,
                 orientation_distance: float = 5.0,
                 attraction_distance: float = 10.0,
                 repulsion_weight: float = 1.0,
                 orientation_weight: float = 0.5,
                 attraction_weight: float = 0.3,
                 noise_scale: float = 0.1,
                 dt: float = 0.1):
        """
        初始化 Couzin 模型参数
        
        Args:
            num_agents: 集群中的目标数量
            repulsion_distance: 排斥规则的作用距离（米）
            orientation_distance: 定向规则的作用距离（米）
            attraction_distance: 吸引规则的作用距离（米）
            repulsion_weight: 排斥规则的权重
            orientation_weight: 定向规则的权重
            attraction_weight: 吸引规则的权重
            noise_scale: 随机扰动的标准差
            dt: 时间步长（秒）
        """
        self.num_agents = num_agents
        self.rep_dist = repulsion_distance
        self.ori_dist = orientation_distance
        self.att_dist = attraction_distance
        self.rep_weight = repulsion_weight
        self.ori_weight = orientation_weight
        self.att_weight = attraction_weight
        self.noise_scale = noise_scale
        self.dt = dt
    
    def _compute_repulsion_force(self, pos_i: np.ndarray, pos_others: np.ndarray) -> np.ndarray:
        """
        排斥力：与邻近的目标保持距离
        
        Args:
            pos_i: 目标 i 的位置 (3,)
            pos_others: 其他目标的位置 (N, 3)
            
        Returns:
            排斥加速度 (3,)
        """
        if len(pos_others) == 0:
            return np.zeros(3, dtype=np.float32)
        
        # 计算与所有其他目标的距离和方向
        diff = pos_i - pos_others  # (N, 3)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)  # (N, 1)
        
        # 只考虑在排斥距离内的目标
        mask = (dist < self.rep_dist) & (dist > 1e-6)  # 避免距离为零
        masked_diff = np.where(mask, diff, 0)  # (N, 3)
        
        # 累积排斥力（距离越近，力越大）
        weight = np.where(mask, (self.rep_dist - dist) / (self.rep_dist + 1e-6), 0)  # (N, 1)
        repulsion_force = np.sum(weight * masked_diff, axis=0)  # (3,)
        
        return repulsion_force.astype(np.float32)
    
    def _compute_orientation_force(self, vel_i: np.ndarray, vel_others: np.ndarray, 
                                    pos_i: np.ndarray, pos_others: np.ndarray) -> np.ndarray:
        """
        定向力：与邻近的目标速度对齐
        
        Args:
            vel_i: 目标 i 的速度 (3,)
            vel_others: 其他目标的速度 (N, 3)
            pos_i: 目标 i 的位置 (3,)
            pos_others: 其他目标的位置 (N, 3)
            
        Returns:
            定向加速度 (3,)
        """
        if len(vel_others) == 0:
            return np.zeros(3, dtype=np.float32)
        
        # 计算与其他目标的距离
        diff = pos_i - pos_others  # (N, 3)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)  # (N, 1)
        
        # 只考虑在定向距离内的目标
        mask = (dist < self.ori_dist) & (dist > 1e-6)
        
        # 计算平均速度方向（加权）
        weight = np.where(mask, 1.0, 0)
        total_weight = np.sum(weight) + 1e-6
        mean_vel = np.sum(vel_others * weight, axis=0) / total_weight  # (3,)
        
        # 定向加速度：朝向平均速度方向
        orientation_force = mean_vel - vel_i
        
        return orientation_force.astype(np.float32)
    
    def _compute_attraction_force(self, pos_i: np.ndarray, pos_others: np.ndarray) -> np.ndarray:
        """
        吸引力：朝向邻近的目标（集群凝聚）
        
        Args:
            pos_i: 目标 i 的位置 (3,)
            pos_others: 其他目标的位置 (N, 3)
            
        Returns:
            吸引加速度 (3,)
        """
        if len(pos_others) == 0:
            return np.zeros(3, dtype=np.float32)
        
        # 计算与其他目标的距离和方向
        diff = pos_others - pos_i  # (N, 3)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)  # (N, 1)
        
        # 只考虑在吸引距离内的目标
        mask = (dist < self.att_dist) & (dist > 1e-6)
        
        # 加权平均方向（距离越远，权重越低）
        weight = np.where(mask, (self.att_dist - dist) / (self.att_dist + 1e-6), 0)
        total_weight = np.sum(weight) + 1e-6
        mean_direction = np.sum(diff * weight, axis=0) / total_weight  # (3,)
        
        # 规范化为单位向量
        norm = np.linalg.norm(mean_direction) + 1e-6
        attraction_force = mean_direction / norm
        
        return attraction_force.astype(np.float32)
    
    def generate_swarm_trajectory(self, leader_trajectory: np.ndarray, 
                                 init_position_noise: float = 1.0) -> np.ndarray:
        """
        基于领航者轨迹生成集群轨迹
        
        Args:
            leader_trajectory: 领航者轨迹 (T, 3)
            init_position_noise: 初始位置的噪声范围（米）
            
        Returns:
            集群轨迹 (T, N, 3) 其中 T=轨迹长度，N=目标数，3=x,y,z坐标
        """
        T = len(leader_trajectory)
        swarm_trajectory = np.zeros((T, self.num_agents, 3), dtype=np.float32)
        
        # 初始化位置：围绕领航者的初始位置随机分布
        leader_start = leader_trajectory[0]
        for i in range(self.num_agents):
            swarm_trajectory[0, i] = leader_start + np.random.uniform(
                -init_position_noise, init_position_noise, 3
            )
        
        # 初始化速度
        velocities = np.zeros((self.num_agents, 3), dtype=np.float32)
        if T > 1:
            leader_vel = leader_trajectory[1] - leader_trajectory[0]
            for i in range(self.num_agents):
                velocities[i] = leader_vel + np.random.randn(3) * self.noise_scale
        
        # 迭代更新轨迹
        for t in range(1, T):
            leader_pos = leader_trajectory[t]
            leader_vel = leader_trajectory[t] - leader_trajectory[t-1]
            
            # 对每个目标应用 Couzin 规则
            for i in range(self.num_agents):
                pos_i = swarm_trajectory[t-1, i]
                vel_i = velocities[i]
                
                # 其他目标的位置和速度
                pos_others = np.delete(swarm_trajectory[t-1], i, axis=0)  # (N-1, 3)
                vel_others = np.delete(velocities, i, axis=0)  # (N-1, 3)
                
                # 计算三种力
                rep_force = self._compute_repulsion_force(pos_i, pos_others)
                ori_force = self._compute_orientation_force(vel_i, vel_others, pos_i, pos_others)
                att_force = self._compute_attraction_force(pos_i, pos_others)
                
                # 合并力：基于领航者 + Couzin 规则 + 噪声
                combined_force = (
                    leader_vel +  # 跟随领航者
                    self.rep_weight * rep_force +
                    self.ori_weight * ori_force +
                    self.att_weight * att_force +
                    np.random.randn(3) * self.noise_scale
                )
                
                # 更新速度和位置
                new_vel = vel_i + combined_force * self.dt
                new_pos = pos_i + new_vel * self.dt
                
                swarm_trajectory[t, i] = new_pos
                velocities[i] = new_vel
        
        return swarm_trajectory


class SingleToSwarmConverter:
    """将单机轨迹扩展为集群轨迹数据集"""
    
    def __init__(self,
                 num_agents: int = 5,
                 couzin_params: Dict = None):
        """
        初始化转换器
        
        Args:
            num_agents: 集群中的目标数量
            couzin_params: Couzin 模型参数字典
        """
        self.num_agents = num_agents
        self.couzin_params = couzin_params or {}
        self.generator = CouzinSwarmGenerator(
            num_agents=num_agents,
            **self.couzin_params
        )
    
    def load_single_trajectory(self, file_path: str) -> np.ndarray:
        """
        加载单机轨迹 CSV/TXT 文件
        
        Args:
            file_path: 轨迹文件路径
            
        Returns:
            轨迹数组 (T, 3)
        """
        df = pd.read_csv(file_path)
        
        # 尝试多种列名组合
        if all(col in df.columns for col in ['tx', 'ty', 'tz']):
            trajectory = df[['tx', 'ty', 'tz']].values.astype(np.float32)
        elif all(col in df.columns for col in ['x', 'y', 'z']):
            trajectory = df[['x', 'y', 'z']].values.astype(np.float32)
        elif all(col in df.columns for col in ['px', 'py', 'pz']):
            trajectory = df[['px', 'py', 'pz']].values.astype(np.float32)
        else:
            # 自动检测：取前 3 个数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                trajectory = df[numeric_cols[:3]].values.astype(np.float32)
            else:
                raise ValueError(f"无法从 {file_path} 中提取坐标信息")
        
        return trajectory
    
    def save_swarm_trajectory_as_csv(self, swarm_trajectory: np.ndarray, 
                                     output_file: str, dt: float = 0.1) -> bool:
        """
        将集群轨迹保存为 CSV 格式，包含时间戳
        
        Args:
            swarm_trajectory: 集群轨迹 (T, N, 3)
            output_file: 输出 CSV 文件路径
            dt: 时间步长（秒，默认 0.1）
            
        Returns:
            是否保存成功
        """
        try:
            T, N, _ = swarm_trajectory.shape
            
            # 创建时间戳列，并格式化为单位小数
            timestamps = np.round(np.arange(T) * dt, 1)
            
            # 创建列名：timestamp, agent_0_x, agent_0_y, agent_0_z, agent_1_x, ...
            columns = ['timestamp']
            for agent_id in range(N):
                columns.extend([f'agent_{agent_id}_x', f'agent_{agent_id}_y', f'agent_{agent_id}_z'])
            
            # 将轨迹数据展平为 (T, N*3)
            reshaped_data = swarm_trajectory.reshape(T, N * 3)
            
            # 合并时间戳和轨迹数据
            data_with_time = np.column_stack([timestamps, reshaped_data])
            
            # 创建 DataFrame 并保存
            df = pd.DataFrame(data_with_time, columns=columns)
            # 设置时间戳列的格式，避免浮点数精度问题
            df['timestamp'] = df['timestamp'].apply(lambda x: f"{x:.1f}")
            df.to_csv(output_file, index=False)
            
            return True
        except Exception as e:
            logger.error(f"保存 CSV 失败: {e}")
            return False
    
    def generate_swarm_dataset(self, input_dir: str, output_dir: str, 
                               num_copies: int = 1, dt: float = 0.1):
        """
        批量生成集群轨迹数据集
        
        Args:
            input_dir: 输入单机轨迹目录
            output_dir: 输出集群轨迹目录
            num_copies: 每条轨迹扩展的集群副本数
            dt: 时间步长（秒）
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 列出所有轨迹文件
        trajectory_files = list(input_path.glob('*.csv')) + list(input_path.glob('*.txt'))
        logger.info(f"找到 {len(trajectory_files)} 条单机轨迹文件")
        
        total_swarm_trajectories = 0
        
        for idx, traj_file in enumerate(trajectory_files, 1):
            try:
                # 加载单机轨迹
                leader_trajectory = self.load_single_trajectory(str(traj_file))
                logger.info(f"[{idx}/{len(trajectory_files)}] 处理: {traj_file.name} (长度={len(leader_trajectory)})")
                
                if len(leader_trajectory) < 10:
                    logger.warning(f"  轨迹过短 (< 10), 跳过")
                    continue
                
                # 为该轨迹生成多个集群副本
                for copy_idx in range(num_copies):
                    swarm_traj = self.generator.generate_swarm_trajectory(leader_trajectory)
                    
                    # 保存为 CSV 格式
                    output_file = output_path / f"{traj_file.stem}_swarm_{copy_idx:03d}.csv"
                    self.save_swarm_trajectory_as_csv(swarm_traj, str(output_file), dt)
                    
                    total_swarm_trajectories += 1
                    
                    if (copy_idx + 1) % max(1, num_copies // 5) == 0 or copy_idx == 0:
                        logger.debug(f"  生成副本 {copy_idx+1}/{num_copies}")
                
                logger.info(f"  ✓ 完成，生成了 {num_copies} 个集群副本")
                
            except Exception as e:
                logger.error(f"处理 {traj_file.name} 失败: {e}")
                continue
        
        logger.info(f"\n数据集生成完成!")
        logger.info(f"  总轨迹条数: {len(trajectory_files)}")
        logger.info(f"  集群副本总数: {total_swarm_trajectories}")
        logger.info(f"  集群中的目标数: {self.num_agents}")
        logger.info(f"  输出目录: {output_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='基于单机轨迹生成集群轨迹数据集 (Couzin 模型)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 新模式（推荐）- 每种无人机数量（3-6架）各生成3个副本，保存为CSV，按数量分文件夹
  python generate_swarm_from_single.py \\
    --input_dirs "D:/Trajectory prediction/drone_trajectories/new_random_traj_100ms" \\
                 "D:/Trajectory prediction/Synthetic-UAV-Flight-Trajectories" \\
    --output_dir swarm_trajectories \\
    --agent_range 3 6 \\
    --copies_per_agent 3

  # 旧模式（向后兼容）- 随机 3-6 架无人机，每条轨迹 10 个副本
  python generate_swarm_from_single.py \\
    --input_dir ../random_traj_100ms \\
    --output_dir swarm_trajectories \\
    --num_copies 10 \\
    --num_agents 5
        """
    )
    parser.add_argument('--input_dir', type=str, default=None,
                        help='单个输入轨迹目录（与 --input_dirs 互斥）')
    parser.add_argument('--input_dirs', type=str, nargs='+', default=None,
                        help='多个输入轨迹目录，用空格分隔。支持 TXT 和 CSV 混合')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出集群轨迹目录')
    parser.add_argument('--num_agents', type=int, default=None,
                        help='固定的集群目标数量 (默认: None，使用随机)')
    parser.add_argument('--random_agents', type=int, nargs=2, default=[3, 6],
                        help='随机目标数范围：min max (默认: 3 6)。仅在 --agent_range 未指定时使用')
    parser.add_argument('--agent_range', type=int, nargs=2, default=None,
                        help='无人机数量范围：min max (如: 3 6)。指定此参数时使用新模式')
    parser.add_argument('--copies_per_agent', type=int, default=None,
                        help='每种无人机数量的副本数 (如: 3)。指定此参数时使用新模式')
    parser.add_argument('--num_copies', type=int, default=10,
                        help='每条单机轨迹扩展的集群副本数 (默认: 10)。旧模式用')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='时间步长 (秒, 默认: 0.1)')
    parser.add_argument('--rep_dist', type=float, default=2.0,
                        help='排斥规则作用距离 (米, 默认: 2.0)')
    parser.add_argument('--ori_dist', type=float, default=5.0,
                        help='定向规则作用距离 (米, 默认: 5.0)')
    parser.add_argument('--att_dist', type=float, default=10.0,
                        help='吸引规则作用距离 (米, 默认: 10.0)')
    parser.add_argument('--noise_scale', type=float, default=0.1,
                        help='随机扰动标准差 (默认: 0.1)')
    
    args = parser.parse_args()
    
    # 验证输入参数
    if args.input_dir is None and args.input_dirs is None:
        parser.error("需要指定 --input_dir 或 --input_dirs")
    
    if args.input_dir is not None and args.input_dirs is not None:
        parser.error("--input_dir 和 --input_dirs 互斥，只能指定一个")
    
    # 判断运行模式
    use_new_mode = args.agent_range is not None and args.copies_per_agent is not None
    
    # 收集所有输入目录
    input_dirs = []
    if args.input_dir:
        input_dirs = [args.input_dir]
    else:
        input_dirs = args.input_dirs
    
    # 验证所有目录存在
    for input_dir in input_dirs:
        if not Path(input_dir).exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return
    
    logger.info(f"\n输入目录数: {len(input_dirs)}")
    for i, d in enumerate(input_dirs, 1):
        logger.info(f"  {i}. {d}")
    
    # 设置 Couzin 参数
    couzin_params = {
        'repulsion_distance': args.rep_dist,
        'orientation_distance': args.ori_dist,
        'attraction_distance': args.att_dist,
        'noise_scale': args.noise_scale,
        'dt': args.dt
    }
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n配置:")
    logger.info(f"  输出目录: {args.output_dir}")
    
    if use_new_mode:
        logger.info(f"  运行模式: 新模式 (分无人机数量保存)")
        logger.info(f"  无人机数量范围: {args.agent_range[0]} - {args.agent_range[1]}")
        logger.info(f"  每种无人机数量的副本数: {args.copies_per_agent}")
        logger.info(f"  保存格式: CSV (按无人机数量分文件夹)")
        agent_numbers = list(range(args.agent_range[0], args.agent_range[1] + 1))
        num_copies = args.copies_per_agent
    else:
        logger.info(f"  运行模式: 旧模式 (随机无人机数量)")
        logger.info(f"  每条轨迹副本数: {args.num_copies}")
        if args.num_agents is None:
            logger.info(f"  目标数（随机）: {args.random_agents[0]} - {args.random_agents[1]}")
        else:
            logger.info(f"  目标数（固定）: {args.num_agents}")
        agent_numbers = None
        num_copies = args.num_copies
    
    # 处理每个输入目录
    total_swarm_count = 0
    total_trajectories = 0
    failed_count = 0
    stats_by_agents = {agent_num: 0 for agent_num in (agent_numbers if agent_numbers else [])}
    
    for input_dir in input_dirs:
        logger.info(f"\n处理目录: {input_dir}")
        
        input_path = Path(input_dir)
        trajectory_files = sorted(
            list(input_path.glob('*.csv')) + list(input_path.glob('*.txt'))
        )
        logger.info(f"  找到 {len(trajectory_files)} 条轨迹文件")
        
        if len(trajectory_files) == 0:
            logger.warning(f"  未找到 CSV 或 TXT 文件，跳过此目录")
            continue
        
        for file_idx, traj_file in enumerate(trajectory_files, 1):
            try:
                # 加载单机轨迹（临时转换器，仅用来读取）
                converter_temp = SingleToSwarmConverter(num_agents=5)
                leader_trajectory = converter_temp.load_single_trajectory(str(traj_file))
                
                if leader_trajectory is None:
                    failed_count += 1
                    continue
                
                if len(leader_trajectory) < 10:
                    logger.debug(f"  {traj_file.name}: 轨迹过短 ({len(leader_trajectory)} < 10)，跳过")
                    continue
                
                total_trajectories += 1
                
                if use_new_mode:
                    # 新模式：为每个无人机数量生成指定数量的副本
                    for num_agents in agent_numbers:
                        for copy_idx in range(num_copies):
                            # 创建对应的生成器
                            converter = SingleToSwarmConverter(
                                num_agents=num_agents,
                                couzin_params=couzin_params
                            )
                            
                            # 生成集群轨迹
                            swarm_traj = converter.generator.generate_swarm_trajectory(leader_trajectory)
                            
                            # 创建按无人机数量分的文件夹
                            agent_folder = output_path / f"swarm_{num_agents}_agents"
                            agent_folder.mkdir(parents=True, exist_ok=True)
                            
                            # 保存为 CSV 格式（传入 dt 参数）
                            output_file = agent_folder / f"{traj_file.stem}_copy_{copy_idx:03d}.csv"
                            converter.save_swarm_trajectory_as_csv(swarm_traj, str(output_file), args.dt)
                            
                            total_swarm_count += 1
                            stats_by_agents[num_agents] += 1
                else:
                    # 旧模式：为该轨迹生成多个集群副本（随机或固定无人机数量）
                    for copy_idx in range(num_copies):
                        # 随机生成目标数（如果 num_agents 为 None）
                        if args.num_agents is None:
                            num_agents = np.random.randint(args.random_agents[0], args.random_agents[1] + 1)
                        else:
                            num_agents = args.num_agents
                        
                        # 创建对应的生成器
                        converter = SingleToSwarmConverter(
                            num_agents=num_agents,
                            couzin_params=couzin_params
                        )
                        
                        # 生成集群轨迹
                        swarm_traj = converter.generator.generate_swarm_trajectory(leader_trajectory)
                        
                        # 保存为 CSV 格式（传入 dt 参数）
                        output_file = output_path / f"{traj_file.stem}_swarm_{copy_idx:03d}.csv"
                        converter.save_swarm_trajectory_as_csv(swarm_traj, str(output_file), args.dt)
                        
                        total_swarm_count += 1
                
                if file_idx % max(1, len(trajectory_files) // 10 + 1) == 0 or file_idx == 1:
                    logger.info(f"  进度: {file_idx}/{len(trajectory_files)} ({traj_file.name})")
                
            except Exception as e:
                logger.error(f"  处理 {traj_file.name} 失败: {e}")
                failed_count += 1
                import traceback
                traceback.print_exc()
                continue
    
    # 输出统计信息
    logger.info(f"\n{'='*70}")
    logger.info(f"数据集生成完成!")
    logger.info(f"{'='*70}")
    logger.info(f"✓ 成功处理轨迹: {total_trajectories}")
    logger.info(f"✗ 失败轨迹: {failed_count}")
    logger.info(f"✓ 生成集群轨迹总数: {total_swarm_count}")
    
    if use_new_mode:
        logger.info(f"✓ 各无人机数量的轨迹数:")
        for agent_num in agent_numbers:
            logger.info(f"    {agent_num} 架无人机: {stats_by_agents[agent_num]} 条轨迹")
        logger.info(f"✓ 每种无人机数量的副本数: {args.copies_per_agent}")
    else:
        logger.info(f"✓ 每条轨迹副本数: {args.num_copies}")
    
    logger.info(f"✓ 输出目录: {output_path}")
    logger.info(f"✓ 保存格式: CSV")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
