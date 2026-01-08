#!/usr/bin/env python3
"""
集群轨迹数据预处理脚本
将 CSV 格式的集群轨迹转换为模型训练所需的 NPZ 格式

特点：
1. ✓ 支持按无人机数量分类处理 (swarm_3_agents, swarm_4_agents 等)
2. ✓ 自动提取多个无人机的位置数据 (T, agents, 3) 格式
3. ✓ 生成序列化数据对 (input_segments, output_segments)
4. ✓ 输出为 (seq_len, num_samples, agents, 3) 的 NPZ 格式，与模型兼容
5. ✓ 支持多个输入目录聚合

使用示例：
python preprocess_swarm_trajectories.py ^
  --input_dir swarm_trajectories ^
  --output_dir swarm_segments ^
  --seq_in 20^
  --seq_out 10 ^
  --stride 1

输出文件结构：
    swarm_segments/
    ├── input_agents_3.npz   (20, N, 3, 3)
    ├── output_agents_3.npz  (10, N, 3, 3)
    ├── input_agents_4.npz   (20, N, 4, 3)
    ├── output_agents_4.npz  (10, N, 4, 3)
    ├── stats.json           (全局统计信息)
    └── dataset_info.txt     (数据集汇总信息)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_swarm_trajectory_from_csv(csv_file: str) -> Optional[np.ndarray]:
    """
    从 CSV 文件加载集群轨迹
    
    预期 CSV 格式：
        timestamp,agent_0_x,agent_0_y,agent_0_z,agent_1_x,agent_1_y,agent_1_z,...
        0.0,-45.08,-4.54,30.96,-45.44,-4.17,31.10,...
        0.1,-45.11,-4.39,30.98,-45.48,-4.02,31.13,...
        ...
    
    Args:
        csv_file: CSV 文件路径
        
    Returns:
        trajectory: (T, agents, 3) 形状的轨迹数组，或 None 表示加载失败
    """
    try:
        df = pd.read_csv(csv_file)
        
        # 移除 timestamp 列，获取坐标数据
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        
        # 获取代理数量
        num_agents = len(df.columns) // 3
        T = len(df)
        
        if num_agents == 0 or T < 2:
            logger.warning(f"  ⚠ {csv_file} 数据无效: agents={num_agents}, T={T}")
            return None
        
        # 重塑为 (T, agents, 3)
        data = df.values.reshape(T, num_agents, 3)
        
        return data.astype(np.float32)
    except Exception as e:
        logger.error(f"加载 {csv_file} 失败: {e}")
        return None


def create_sequences(trajectory: np.ndarray, 
                    seq_in: int = 20, 
                    seq_out: int = 10,
                    stride: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从完整轨迹创建输入输出序列对
    
    支持两种模式：
    1. stride=seq_out（推荐）：避免样本重叠，提高数据多样性
    2. stride=1（覆盖）：最大化样本数量，但存在时间相关性
    
    Args:
        trajectory: (T, agents, 3) 的完整轨迹
        seq_in: 输入序列长度（过去时间步）
        seq_out: 输出序列长度（未来时间步）
        stride: 滑动窗口步长（默认=seq_out，避免重叠）
        
    Returns:
        X: (samples, seq_in, agents, 3) - 输入位置序列
        Y: (samples, seq_out, agents, 3) - 输出位置序列
    """
    if stride is None:
        stride = seq_out  # 默认不重叠
    
    T, agents, _ = trajectory.shape
    total_len = seq_in + seq_out
    
    if T < total_len:
        return None, None
    
    X_list = []
    Y_list = []
    
    # 滑动窗口生成序列对
    for t in range(0, T - total_len + 1, stride):
        X_list.append(trajectory[t:t+seq_in])  # (seq_in, agents, 3)
        Y_list.append(trajectory[t+seq_in:t+seq_in+seq_out])  # (seq_out, agents, 3)
    
    if len(X_list) == 0:
        logger.warning(f"  ⚠ 轨迹过短，无法创建序列 (T={T}, seq_in+seq_out={total_len})")
        return None, None
    
    X = np.stack(X_list, axis=0)  # (samples, seq_in, agents, 3)
    Y = np.stack(Y_list, axis=0)  # (samples, seq_out, agents, 3)
    
    return X, Y


def preprocess_swarm_data(input_dir: str, 
                         output_dir: str, 
                         seq_in: int = 20, 
                         seq_out: int = 10,
                         stride: Optional[int] = None) -> Dict:
    """
    预处理所有集群轨迹数据
    
    处理流程：
    1. 按无人机数量分类检测子目录
    2. 加载每个子目录下的所有 CSV 轨迹
    3. 创建序列对
    4. 合并并保存为 NPZ 格式
    5. 统计全局信息
    
    Args:
        input_dir: 输入目录 (包含 swarm_*_agents 子文件夹)
        output_dir: 输出目录
        seq_in: 输入序列长度
        seq_out: 输出序列长度
        stride: 滑动窗口步长（默认=seq_out）
        
    Returns:
        stats: 统计信息字典
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 检测所有 swarm_*_agents 子目录
    agent_dirs = sorted(input_path.glob('swarm_*_agents'))
    
    if not agent_dirs:
        logger.error(f"❌ 未找到 swarm_*_agents 子目录，请检查输入目录: {input_dir}")
        logger.info(f"   当前目录内容: {list(input_path.iterdir())[:5]}")
        return {}
    
    logger.info(f"\n{'='*70}")
    logger.info(f"发现 {len(agent_dirs)} 个无人机分类目录")
    logger.info(f"{'='*70}")
    
    # 全局统计信息
    global_stats = {
        'agents_count': {},
        'samples_per_agent': {},
        'total_samples': 0,
        'seq_in': seq_in,
        'seq_out': seq_out,
        'stride': stride if stride is not None else seq_out,
    }
    
    # 处理每个无人机数量的目录
    for agent_dir in agent_dirs:
        # 提取代理数量
        dir_name = agent_dir.name  # e.g., "swarm_3_agents"
        try:
            num_agents = int(dir_name.split('_')[1])
        except (IndexError, ValueError):
            logger.warning(f"⚠ 无法从目录名解析代理数量: {dir_name}，跳过")
            continue
        
        logger.info(f"\n[{num_agents} 架无人机] 处理目录: {agent_dir.name}")
        
        # 查找所有 CSV 文件
        csv_files = sorted(agent_dir.glob('*.csv'))
        logger.info(f"  找到 {len(csv_files)} 个 CSV 文件")
        
        if len(csv_files) == 0:
            logger.warning(f"  ⚠ 目录中无 CSV 文件，跳过")
            continue
        
        all_X = []
        all_Y = []
        valid_files = 0
        skipped_files = 0
        
        # 加载所有轨迹文件
        for csv_file in tqdm(csv_files, desc=f"  加载 CSV", unit="file"):
            traj = load_swarm_trajectory_from_csv(str(csv_file))
            
            if traj is None:
                skipped_files += 1
                continue
            
            # 检查代理数量是否匹配
            if traj.shape[1] != num_agents:
                logger.warning(f"  ⚠ {csv_file.name}: 代理数不匹配 (期望{num_agents}, 实际{traj.shape[1]})")
                skipped_files += 1
                continue
            
            X, Y = create_sequences(traj, seq_in, seq_out, stride)
            
            if X is not None and Y is not None:
                all_X.append(X)
                all_Y.append(Y)
                valid_files += 1
            else:
                skipped_files += 1
        
        logger.info(f"  ✓ 成功加载: {valid_files}, 跳过: {skipped_files}")
        
        if len(all_X) == 0:
            logger.warning(f"  ⚠ 无有效数据用于 {num_agents} 架无人机，跳过保存")
            continue
        
        # 拼接所有数据
        X_all = np.concatenate(all_X, axis=0)  # (N, seq_in, agents, 3)
        Y_all = np.concatenate(all_Y, axis=0)  # (N, seq_out, agents, 3)
        
        logger.info(f"  数据形状统计:")
        logger.info(f"    输入 (X): {X_all.shape} = (samples, seq_in, agents, 3)")
        logger.info(f"    输出 (Y): {Y_all.shape}")
        
        # ✅ 关键：转置为 (seq_in, N, agents, 3) 格式（与模型兼容）
        X_all_transposed = np.transpose(X_all, (1, 0, 2, 3))  # (seq_in, N, agents, 3)
        Y_all_transposed = np.transpose(Y_all, (1, 0, 2, 3))  # (seq_out, N, agents, 3)
        
        logger.info(f"  转置后形状:")
        logger.info(f"    输入 (X): {X_all_transposed.shape} = (seq_in, samples, agents, 3)")
        logger.info(f"    输出 (Y): {Y_all_transposed.shape}")
        
        # 保存为 NPZ 格式
        X_file = output_path / f'input_agents_{num_agents}.npz'
        Y_file = output_path / f'output_agents_{num_agents}.npz'
        
        np.savez_compressed(X_file, data=X_all_transposed)
        np.savez_compressed(Y_file, data=Y_all_transposed)
        
        logger.info(f"  ✓ 保存输入数据: {X_file.name} ({X_file.stat().st_size / 1024 / 1024:.2f} MB)")
        logger.info(f"  ✓ 保存输出数据: {Y_file.name} ({Y_file.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # 更新全局统计
        num_samples = X_all.shape[0]
        global_stats['agents_count'][num_agents] = {
            'files': valid_files,
            'samples': num_samples,
            'input_shape': tuple(X_all_transposed.shape),
            'output_shape': tuple(Y_all_transposed.shape),
        }
        global_stats['total_samples'] += num_samples
    
    # ========== 保存统计信息 ==========
    
    # 保存 JSON 统计信息
    stats_file = output_path / 'stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(global_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✓ 保存统计信息: {stats_file}")
    
    # 保存易读的文本汇总
    info_file = output_path / 'dataset_info.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("集群轨迹数据集预处理信息\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"处理配置：\n")
        f.write(f"  输入序列长度: {seq_in} (过去 {seq_in*0.1:.1f} 秒)\n")
        f.write(f"  输出序列长度: {seq_out} (预测 {seq_out*0.1:.1f} 秒)\n")
        f.write(f"  滑动窗口步长: {stride if stride is not None else seq_out}\n\n")
        
        f.write(f"数据统计：\n")
        f.write(f"  总样本数: {global_stats['total_samples']}\n")
        f.write(f"  无人机分类: {len(global_stats['agents_count'])}\n\n")
        
        f.write("各类别详情：\n")
        for num_agents in sorted(global_stats['agents_count'].keys()):
            info = global_stats['agents_count'][num_agents]
            f.write(f"\n  {num_agents} 架无人机：\n")
            f.write(f"    文件数: {info['files']}\n")
            f.write(f"    样本数: {info['samples']}\n")
            f.write(f"    输入形状: {info['input_shape']}\n")
            f.write(f"    输出形状: {info['output_shape']}\n")
    
    logger.info(f"✓ 保存信息文件: {info_file}\n")
    
    # ========== 输出最终统计 ==========
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ 数据预处理完成！")
    logger.info(f"{'='*70}\n")
    logger.info(f"输出目录: {output_path}")
    logger.info(f"总样本数: {global_stats['total_samples']}")
    logger.info(f"无人机分类: {list(sorted(global_stats['agents_count'].keys()))}")
    logger.info(f"\n生成的文件:")
    
    for num_agents in sorted(global_stats['agents_count'].keys()):
        logger.info(f"  • input_agents_{num_agents}.npz")
        logger.info(f"  • output_agents_{num_agents}.npz")
    
    logger.info(f"  • stats.json (统计信息)")
    logger.info(f"  • dataset_info.txt (人类可读信息)")
    
    return global_stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='预处理集群轨迹数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本使用
  python preprocess_swarm_trajectories.py \\
    --input_dir swarm_trajectories \\
    --output_dir swarm_segments \\
    --seq_in 20 \\
    --seq_out 10

  # 自定义步长（增加样本数）
  python preprocess_swarm_trajectories.py \\
    --input_dir swarm_trajectories \\
    --output_dir swarm_segments \\
    --seq_in 20 \\
    --seq_out 10 \\
    --stride 1

  # 处理其他数据源
  python preprocess_swarm_trajectories.py \\
    --input_dir "D:/Trajectory prediction/Synthetic-UAV-Flight-Trajectories" \\
    --output_dir swarm_segments \\
    --seq_in 20 \\
    --seq_out 10
        """
    )
    
    parser.add_argument('--input_dir', type=str, default='swarm_trajectories',
                        help='输入目录（包含 swarm_*_agents 子文件夹），默认: swarm_trajectories')
    parser.add_argument('--output_dir', type=str, default='swarm_segments',
                        help='输出目录，默认: swarm_segments')
    parser.add_argument('--seq_in', type=int, default=20,
                        help='输入序列长度（过去时间步），默认: 20 (2.0秒)')
    parser.add_argument('--seq_out', type=int, default=10,
                        help='输出序列长度（预测时间步），默认: 10 (1.0秒)')
    parser.add_argument('--stride', type=int, default=None,
                        help='滑动窗口步长（默认=seq_out，不重叠）。设为1以最大化样本数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，默认: 42')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 验证输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 运行预处理
    stats = preprocess_swarm_data(
        args.input_dir,
        args.output_dir,
        seq_in=args.seq_in,
        seq_out=args.seq_out,
        stride=args.stride
    )
    
    if not stats:
        logger.error("❌ 预处理失败，未生成任何数据")
        return
    
    logger.info("✅ 预处理成功完成！")
    logger.info(f"\n下一步：训练模型")
    logger.info(f"  python train_swarm_model_enhanced.py --agents 3 --epochs 200 --batch_size 256")
    logger.info(f"  或")
    logger.info(f"  python train_swarm_model_enhanced.py --agents all --epochs 200 --batch_size 256")


if __name__ == '__main__':
    main()
