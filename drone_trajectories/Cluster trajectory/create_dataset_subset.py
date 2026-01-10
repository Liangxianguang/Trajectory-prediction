#!/usr/bin/env python3
"""
创建数据子集以加速训练验证
=============================
原始数据: 2,302,320 样本（stride=1，最大化样本数）
        → 275 小时训练时间（300 epochs）

子集生成策略:
  采样率: 保留每 10 个样本中的 1 个
  子集大小: ~230,000 样本
  预计时间: ~27.5 小时训练（300 epochs）
  用途: 快速验证 GNN 模型效果 ✓

使用场景:
  1. 验证 GNN 模型能否正常训练（不平原）
  2. 对比 GNN vs BiGRU 性能
  3. 快速超参数调优
  4. 生成对比图表
  
执行:
  python create_dataset_subset.py --sample_ratio 0.1 --output_suffix "_subset"
  
输出:
  swarm_segments/
  ├── input_agents_3_subset.npz   (230k, 20, 3, 3)
  ├── output_agents_3_subset.npz  (230k, 10, 3, 3)
  ├── input_agents_4_subset.npz
  ├── output_agents_4_subset.npz
  ... (以此类推)
"""

import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def create_subset(input_npz, output_npz, output_input, output_output, 
                  sample_ratio=0.1, seed=42):
    """
    从原始 NPZ 数据创建子集
    
    Args:
        input_npz: 原始输入 NPZ 文件路径
        output_npz: 原始输出 NPZ 文件路径
        output_input: 子集输出文件（输入）
        output_output: 子集输出文件（输出）
        sample_ratio: 采样比率（0.1 = 保留 10% 的样本）
        seed: 随机种子
    """
    logger.info(f"加载原始数据...")
    X = np.load(input_npz)['data']  # (seq_len, samples, agents, 3)
    Y = np.load(output_npz)['data']  # (seq_out, samples, agents, 3)
    
    logger.info(f"  原始 X 形状: {X.shape}")
    logger.info(f"  原始 Y 形状: {Y.shape}")
    
    # 获取样本维度（第二维）
    seq_in, num_samples, num_agents, coords = X.shape
    seq_out = Y.shape[0]
    
    # 生成随机采样索引
    np.random.seed(seed)
    num_subset = max(1, int(num_samples * sample_ratio))
    subset_indices = np.random.choice(num_samples, size=num_subset, replace=False)
    subset_indices = np.sort(subset_indices)  # 保持顺序
    
    logger.info(f"采样参数:")
    logger.info(f"  采样率: {sample_ratio * 100:.1f}%")
    logger.info(f"  原始样本数: {num_samples:,}")
    logger.info(f"  子集样本数: {num_subset:,}")
    logger.info(f"  节省比例: {(1 - sample_ratio) * 100:.1f}%")
    
    # 提取子集
    logger.info(f"提取子集（可能耗时...）")
    X_subset = X[:, subset_indices, :, :]  # (seq_in, num_subset, agents, 3)
    Y_subset = Y[:, subset_indices, :, :]  # (seq_out, num_subset, agents, 3)
    
    logger.info(f"  X_subset 形状: {X_subset.shape}")
    logger.info(f"  Y_subset 形状: {Y_subset.shape}")
    
    # 保存子集
    logger.info(f"保存子集...")
    np.savez_compressed(output_input, data=X_subset)
    np.savez_compressed(output_output, data=Y_subset)
    
    logger.info(f"✓ 子集已保存:")
    logger.info(f"  {output_input}")
    logger.info(f"  {output_output}")
    
    # 返回统计信息
    return {
        'original_samples': num_samples,
        'subset_samples': num_subset,
        'sample_ratio': sample_ratio,
        'seq_in': seq_in,
        'seq_out': seq_out,
        'agents': num_agents,
    }


def main():
    parser = argparse.ArgumentParser(description='创建数据子集以加速训练验证')
    parser.add_argument('--data_dir', type=str, default='swarm_segments',
                        help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='swarm_segments',
                        help='输出目录（默认=输入目录）')
    parser.add_argument('--sample_ratio', type=float, default=0.1,
                        help='采样比率 (默认=0.1，即 10%%)')
    parser.add_argument('--output_suffix', type=str, default='_subset',
                        help='输出文件后缀 (默认="_subset")')
    parser.add_argument('--agents', type=str, default='3,4,5,6',
                        help='无人机数量列表，逗号分隔 (默认="3,4,5,6")')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    agents_list = [int(x.strip()) for x in args.agents.split(',')]
    
    logger.info(f"="*80)
    logger.info(f"数据子集生成工具")
    logger.info(f"="*80)
    logger.info(f"配置:")
    logger.info(f"  数据目录: {data_dir}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  采样比率: {args.sample_ratio * 100:.1f}%")
    logger.info(f"  输出后缀: {args.output_suffix}")
    logger.info(f"  无人机数量: {agents_list}")
    logger.info(f"="*80)
    
    all_stats = {}
    
    for num_agents in agents_list:
        logger.info(f"\n处理 {num_agents} 架无人机...")
        logger.info(f"-" * 80)
        
        input_file = data_dir / f'input_agents_{num_agents}.npz'
        output_file = data_dir / f'output_agents_{num_agents}.npz'
        
        if not input_file.exists() or not output_file.exists():
            logger.warning(f"⚠️  找不到数据文件: {input_file}, {output_file}")
            logger.warning(f"   跳过 {num_agents} 架无人机")
            continue
        
        output_input = output_dir / f'input_agents_{num_agents}{args.output_suffix}.npz'
        output_output = output_dir / f'output_agents_{num_agents}{args.output_suffix}.npz'
        
        stats = create_subset(
            input_file, output_file,
            output_input, output_output,
            sample_ratio=args.sample_ratio,
            seed=args.seed
        )
        
        all_stats[num_agents] = stats
    
    logger.info(f"\n" + "="*80)
    logger.info(f"✓ 子集生成完成！")
    logger.info(f"="*80)
    
    logger.info(f"\n统计摘要:")
    for num_agents, stats in all_stats.items():
        logger.info(f"\n{num_agents} 架无人机:")
        logger.info(f"  原始样本: {stats['original_samples']:,}")
        logger.info(f"  子集样本: {stats['subset_samples']:,}")
        logger.info(f"  节省比例: {(1 - stats['sample_ratio']) * 100:.1f}%")
        
        # 估计训练时间
        # 假设原始 2,302,320 样本需要 275 小时
        # 子集样本需要 275 * sample_ratio 小时
        original_time_hours = 275  # 300 epochs on full data (2.3M samples)
        subset_time_hours = original_time_hours * stats['sample_ratio']
        logger.info(f"  估计训练时间 (300 epochs): {subset_time_hours:.1f} 小时")
    
    logger.info(f"\n下一步:")
    logger.info(f"  使用子集训练 GNN 模型:")
    logger.info(f"  python train_swarm_gnn.py --data_dir {output_dir} --agents 3 --epochs 50")
    logger.info(f"\n  对比模型:")
    logger.info(f"  python compare_models.py --subset_suffix {args.output_suffix}")


if __name__ == '__main__':
    main()
