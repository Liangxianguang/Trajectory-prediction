#!/usr/bin/env python3
"""
支持多无人机（行人）的轨迹预测可视化
从保存的预测结果和原始数据集加载，展示所有无人机的轨迹对比

使用示例：
python visualize_predictions_with_peds.py \
  --predictions_file ./output/lbebm3D/2026-01-10-17-50-33/predictions.pickle \
  --dataset_folder ../dataset \
  --dataset_name swarm \
  --output_dir ../visualization_output_multi_uav \
  --num_samples 10
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# 添加 tool 目录到路径，以便导入 utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='多无人机轨迹预测可视化')
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='预测结果pickle文件路径')
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help='数据集文件夹')
    parser.add_argument('--dataset_name', type=str, default='swarm',
                        help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='../visualization_output_multi_uav',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='可视化样本数')
    parser.add_argument('--dpi', type=int, default=150,
                        help='图表DPI')
    parser.add_argument('--obs', type=int, default=20,
                        help='观测时间步长')
    parser.add_argument('--preds', type=int, default=10,
                        help='预测时间步长')
    parser.add_argument('--delim', type=str, default='\t',
                        help='数据分隔符')
    
    return parser.parse_args()


def load_dataset_with_peds(dataset_folder, dataset_name, obs_len, pred_len):
    """加载数据集并提取行人/无人机 ID 信息"""
    print(f"\n加载数据集: {dataset_name}")
    
    test_dataset, _ = utils.create_dataset(
        dataset_folder, dataset_name, 0, obs_len, pred_len,
        delim='\t', train=False, eval=True, verbose=False
    )
    
    # 从数据集中提取peds信息
    if hasattr(test_dataset, 'data') and 'peds' in test_dataset.data:
        peds = test_dataset.data['peds']
        # 转换为整数类型（如果是浮点数）
        peds = np.asarray(peds, dtype=int)
        unique_peds = np.unique(peds)
        print(f"  加载了 {len(unique_peds)} 个无人机")
        # 显示每个无人机的样本数
        counts = np.bincount(peds)
        print(f"  每个无人机的样本数: {counts[:min(5, len(counts))]}...")
        return peds
    else:
        print("  警告: 无法从数据集中提取peds信息")
        return None


def load_predictions(predictions_file):
    """加载预测结果"""
    print(f"加载预测结果: {predictions_file}")
    
    with open(predictions_file, 'rb') as f:
        data = pickle.load(f)
    
    input_history = data['input_history']      # (N, 20, 3)
    ground_truth = data['ground_truth']        # (N, 10, 3)
    predictions = data['predictions']          # (N, 10, 3)
    ade = data.get('ade', None)
    fde = data.get('fde', None)
    
    print(f"  输入历史: {input_history.shape}")
    print(f"  真实未来: {ground_truth.shape}")
    print(f"  预测结果: {predictions.shape}")
    if ade:
        print(f"  全局 ADE: {ade:.6f} m")
    if fde:
        print(f"  全局 FDE: {fde:.6f} m")
    
    return input_history, ground_truth, predictions, ade, fde


def group_by_scene(peds, num_uavs=3, samples_per_uav=None):
    """
    将样本按场景分组。
    
    数据结构: 每个无人机的所有样本连续存储
    例如: 无人机0的18767个样本, 然后是无人机1的18767个样本, etc.
    
    Args:
        peds: 无人机ID数组
        num_uavs: 无人机数量
        samples_per_uav: 每个无人机的样本数（如果为None则自动计算）
    """
    if peds is None:
        return None
    
    peds = np.array(peds)
    num_samples = len(peds)
    
    if samples_per_uav is None:
        # 自动计算每个无人机的样本数
        samples_per_uav = num_samples // num_uavs
    
    print(f"\n场景分组信息:")
    print(f"  总样本数: {num_samples}")
    print(f"  无人机数: {num_uavs}")
    print(f"  每个无人机样本数: {samples_per_uav}")
    
    # 生成场景: 第i个场景包含所有无人机的第i个样本
    # 例如: 场景0 = [样本0(UAV0), 样本18767(UAV1), 样本37534(UAV2)]
    scenes = []
    
    for scene_idx in range(samples_per_uav):
        scene_samples = []
        for uav_idx in range(num_uavs):
            sample_idx = uav_idx * samples_per_uav + scene_idx
            if sample_idx < num_samples:
                scene_samples.append(sample_idx)
        
        if len(scene_samples) == num_uavs:  # 只保留完整的场景
            scenes.append(scene_samples)
    
    print(f"  生成场景数: {len(scenes)}")
    
    return scenes


def visualize_single_uav(hist, true_fut, pred_fut, ped_id, scene_idx, uav_idx, output_dir, dpi=150):
    """
    绘制单个无人机的轨迹（3D + XY + XZ）
    """
    # 颜色列表
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    color = colors[uav_idx % len(colors)]
    
    # 创建图表
    fig = plt.figure(figsize=(18, 6))
    
    # 3D 视图
    ax1 = fig.add_subplot(131, projection='3d')
    
    # 历史轨迹
    ax1.plot(hist[:, 0], hist[:, 1], hist[:, 2],
            color=color, marker='o', linestyle='-', linewidth=2.5,
            label='历史轨迹', markersize=5, alpha=0.8)
    
    # 真实未来轨迹
    start_point = hist[-1]
    true_display = np.vstack([start_point, true_fut])
    ax1.plot(true_display[:, 0], true_display[:, 1], true_display[:, 2],
            color=color, marker='s', linestyle='-', linewidth=2.5,
            label='真实未来', markersize=5, alpha=0.8)
    
    # 预测未来轨迹
    pred_display = np.vstack([start_point, pred_fut])
    ax1.plot(pred_display[:, 0], pred_display[:, 1], pred_display[:, 2],
            color=color, marker='^', linestyle='--', linewidth=2,
            label='预测未来', markersize=5, alpha=0.7)
    
    # 标记关键点
    ax1.scatter(*hist[0], color=color, s=120, marker='*', zorder=5, label='起点')
    ax1.scatter(*hist[-1], color=color, s=100, marker='D', zorder=5, label='历史结束')
    ax1.scatter(*true_fut[-1], color=color, s=120, marker='s', zorder=5, alpha=0.7)
    ax1.scatter(*pred_fut[-1], color=color, s=120, marker='^', zorder=5, alpha=0.7)
    
    ax1.set_xlabel('X (m)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Y (m)', fontweight='bold', fontsize=11)
    ax1.set_zlabel('Z (m)', fontweight='bold', fontsize=11)
    ax1.set_title('3D 轨迹对比', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # XY 平面
    ax2 = fig.add_subplot(132)
    ax2.plot(hist[:, 0], hist[:, 1], 'o-', color=color,
            label='历史轨迹', markersize=4, linewidth=2, alpha=0.8)
    ax2.plot(true_display[:, 0], true_display[:, 1], 's-', color=color,
            label='真实未来', markersize=4, linewidth=2.5, alpha=0.8)
    ax2.plot(pred_display[:, 0], pred_display[:, 1], '^--', color=color,
            label='预测未来', markersize=4, linewidth=2, alpha=0.7)
    
    ax2.scatter(hist[0, 0], hist[0, 1], color=color, s=100, marker='*', zorder=5)
    ax2.scatter(hist[-1, 0], hist[-1, 1], color=color, s=80, marker='D', zorder=5)
    
    ax2.set_xlabel('X (m)', fontweight='bold')
    ax2.set_ylabel('Y (m)', fontweight='bold')
    ax2.set_title('XY 平面视图', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # XZ 平面
    ax3 = fig.add_subplot(133)
    ax3.plot(hist[:, 0], hist[:, 2], 'o-', color=color,
            label='历史轨迹', markersize=4, linewidth=2, alpha=0.8)
    ax3.plot(true_display[:, 0], true_display[:, 2], 's-', color=color,
            label='真实未来', markersize=4, linewidth=2.5, alpha=0.8)
    ax3.plot(pred_display[:, 0], pred_display[:, 2], '^--', color=color,
            label='预测未来', markersize=4, linewidth=2, alpha=0.7)
    
    ax3.scatter(hist[0, 0], hist[0, 2], color=color, s=100, marker='*', zorder=5)
    ax3.scatter(hist[-1, 0], hist[-1, 2], color=color, s=80, marker='D', zorder=5)
    
    ax3.set_xlabel('X (m)', fontweight='bold')
    ax3.set_ylabel('Z (m)', fontweight='bold')
    ax3.set_title('XZ 平面视图', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.suptitle(f'无人机轨迹预测对比 - 场景 #{scene_idx:03d} / UAV{ped_id}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, f'scene_{scene_idx:03d}_uav_{uav_idx}.png')
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_file


def visualize_multi_uav_sample(scene_samples, input_history, ground_truth, predictions, 
                               peds, scene_idx, output_dir, dpi=150):
    """
    为一个场景中的每个无人机生成独立的轨迹图
    
    scene_samples: 这个场景包含的样本索引列表
    """
    if not scene_samples or len(scene_samples) < 1:
        return []
    
    output_files = []
    
    # 为每个无人机生成单独的图表
    for plot_idx, sample_idx in enumerate(scene_samples):
        hist = input_history[sample_idx]           # (20, 3)
        true_fut = ground_truth[sample_idx]        # (10, 3)
        pred_fut = predictions[sample_idx]         # (10, 3)
        ped_id = peds[sample_idx] if peds is not None else plot_idx
        
        output_file = visualize_single_uav(
            hist, true_fut, pred_fut, ped_id, scene_idx, plot_idx, output_dir, dpi
        )
        output_files.append(output_file)
    
    return output_files


def calculate_scene_error(scene_samples, ground_truth, predictions):
    """计算场景中所有无人机的平均误差"""
    errors = []
    for sample_idx in scene_samples:
        error = np.linalg.norm(predictions[sample_idx] - ground_truth[sample_idx])
        errors.append(error)
    
    ade = np.mean(errors)
    return ade


def main():
    args = parse_args()
    
    # 规范化路径
    args.predictions_file = os.path.abspath(args.predictions_file)
    args.dataset_folder = os.path.abspath(args.dataset_folder)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("多无人机轨迹预测可视化（包含行人/无人机信息）")
    print("=" * 70)
    
    try:
        # 加载数据集并获取peds信息
        peds = load_dataset_with_peds(args.dataset_folder, args.dataset_name, 
                                       args.obs, args.preds)
        
        # 加载预测结果
        input_history, ground_truth, predictions, overall_ade, overall_fde = \
            load_predictions(args.predictions_file)
        
        # 按场景分组
        print(f"\n按场景分组样本...")
        # 检测无人机数量
        unique_peds = np.unique(peds)
        num_uavs = len(unique_peds)
        samples_per_uav = len(input_history) // num_uavs
        
        scenes = group_by_scene(peds, num_uavs=num_uavs, samples_per_uav=samples_per_uav)
        
        print(f"  检测到 {len(scenes)} 个场景")
        
        # 生成可视化
        print(f"\n生成可视化 ({min(args.num_samples, len(scenes))} 个场景)...")
        num_scenes = min(args.num_samples, len(scenes))
        
        scene_errors = []
        successful = 0
        total_figures = 0
        
        for scene_idx in range(num_scenes):
            try:
                scene_samples = scenes[scene_idx]
                
                # 计算场景误差
                scene_ade = calculate_scene_error(scene_samples, ground_truth, predictions)
                scene_errors.append(scene_ade)
                
                # 生成可视化（每个无人机一个图）
                output_files = visualize_multi_uav_sample(
                    scene_samples, input_history, ground_truth, predictions,
                    peds, scene_idx, args.output_dir, dpi=args.dpi
                )
                
                successful += 1
                total_figures += len(output_files)
                print(f"  ✓ 场景 #{scene_idx:03d}: 无人机数={len(scene_samples)}, ADE={scene_ade:.6f}m, 生成 {len(output_files)} 个图表")
            
            except Exception as e:
                print(f"  ✗ 场景 #{scene_idx:03d} 失败: {e}")
        
        # 统计信息
        print()
        print("=" * 70)
        print(f"✓ 可视化完成！共生成 {successful} 个场景，{total_figures} 个图表")
        print()
        print("全局指标:")
        if overall_ade:
            print(f"  整体 ADE: {overall_ade:.6f} m")
        if overall_fde:
            print(f"  整体 FDE: {overall_fde:.6f} m")
        
        if scene_errors:
            print()
            print("场景指标统计:")
            print(f"  平均场景 ADE: {np.mean(scene_errors):.6f} m")
            print(f"  中位数 ADE: {np.median(scene_errors):.6f} m")
            print(f"  最小 ADE: {np.min(scene_errors):.6f} m")
            print(f"  最大 ADE: {np.max(scene_errors):.6f} m")
        
        print()
        print(f"输出目录: {args.output_dir}")
        print("=" * 70)
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
