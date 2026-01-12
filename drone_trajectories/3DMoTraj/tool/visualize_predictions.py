#!/usr/bin/env python3
"""
从保存的预测结果文件读取数据并进行可视化
支持显示输入历史 + 真实未来 + 预测未来的对比

使用示例：
python visualize_predictions.py \
  --predictions_file ../output/lbebm3D/xxx/predictions.pickle \
  --output_dir ../visualization_output \
  --num_samples 20
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='轨迹预测结果可视化')
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='预测结果pickle文件路径')
    parser.add_argument('--output_dir', type=str, default='../visualization_output',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='可视化样本数')
    parser.add_argument('--dpi', type=int, default=100,
                        help='图表DPI')
    parser.add_argument('--num_agents', type=int, default=3,
                        help='场景中无人机/智能体数量')
    
    return parser.parse_args()


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
    print(f"  ADE: {ade:.6f}" if ade else "")
    print(f"  FDE: {fde:.6f}" if fde else "")
    
    return input_history, ground_truth, predictions, ade, fde


def visualize_sample(hist, true_future, pred_future, sample_idx, output_dir, num_agents=3, dpi=100):
    """绘制单个样本：历史 + 真实未来 + 预测未来（支持多个智能体）"""
    
    # 分离多个智能体的轨迹
    # hist: (20, 3*num_agents) -> reshape to (20, num_agents, 3)
    # true_future, pred_future 同理
    
    hist_reshaped = hist.reshape(hist.shape[0], num_agents, 3)
    true_reshaped = true_future.reshape(true_future.shape[0], num_agents, 3)
    pred_reshaped = pred_future.reshape(pred_future.shape[0], num_agents, 3)
    
    # 颜色和标记
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    markers_hist = ['o', 's', '^', 'D', 'v', 'p']
    markers_true = ['s', 's', 's', 's', 's', 's']
    markers_pred = ['^', '^', '^', '^', '^', '^']
    
    # 创建图表
    fig = plt.figure(figsize=(20, 6))
    
    # 3D 视图
    ax1 = fig.add_subplot(131, projection='3d')
    
    for agent_id in range(num_agents):
        hist_agent = hist_reshaped[:, agent_id, :]
        true_agent = true_reshaped[:, agent_id, :]
        pred_agent = pred_reshaped[:, agent_id, :]
        
        color = colors[agent_id % len(colors)]
        
        # 历史轨迹
        ax1.plot(hist_agent[:, 0], hist_agent[:, 1], hist_agent[:, 2], 
                 color=color, marker=markers_hist[agent_id], linestyle='-',
                 label=f'无人机{agent_id} 历史', markersize=4, linewidth=2, alpha=0.7)
        
        # 真实未来
        start_point = hist_agent[-1]
        true_display = np.vstack([start_point, true_agent])
        ax1.plot(true_display[:, 0], true_display[:, 1], true_display[:, 2],
                 color=color, marker=markers_true[agent_id], linestyle='-',
                 label=f'无人机{agent_id} 真实', markersize=4, linewidth=2.5, alpha=0.8)
        
        # 预测未来
        pred_display = np.vstack([start_point, pred_agent])
        ax1.plot(pred_display[:, 0], pred_display[:, 1], pred_display[:, 2],
                 color=color, marker=markers_pred[agent_id], linestyle='--',
                 label=f'无人机{agent_id} 预测', markersize=4, linewidth=2, alpha=0.6)
        
        # 标记关键点
        ax1.scatter(*hist_agent[0], color=color, s=100, marker='*', zorder=5)
        ax1.scatter(*hist_agent[-1], color=color, s=80, marker='D', zorder=5)
    
    ax1.set_xlabel('X (m)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Y (m)', fontweight='bold', fontsize=11)
    ax1.set_zlabel('Z (m)', fontweight='bold', fontsize=11)
    ax1.set_title('3D 轨迹对比', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # XY 平面
    ax2 = fig.add_subplot(132)
    for agent_id in range(num_agents):
        hist_agent = hist_reshaped[:, agent_id, :]
        true_agent = true_reshaped[:, agent_id, :]
        pred_agent = pred_reshaped[:, agent_id, :]
        
        color = colors[agent_id % len(colors)]
        start_point = hist_agent[-1]
        true_display = np.vstack([start_point, true_agent])
        pred_display = np.vstack([start_point, pred_agent])
        
        ax2.plot(hist_agent[:, 0], hist_agent[:, 1], 'o-', color=color, 
                 label=f'UAV{agent_id} 历史', markersize=3, linewidth=1.5, alpha=0.7)
        ax2.plot(true_display[:, 0], true_display[:, 1], 's-', color=color,
                 label=f'UAV{agent_id} 真实', markersize=3, linewidth=2, alpha=0.8)
        ax2.plot(pred_display[:, 0], pred_display[:, 1], '^--', color=color,
                 label=f'UAV{agent_id} 预测', markersize=3, linewidth=1.5, alpha=0.6)
    
    ax2.set_xlabel('X (m)', fontweight='bold')
    ax2.set_ylabel('Y (m)', fontweight='bold')
    ax2.set_title('XY 平面视图', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # XZ 平面
    ax3 = fig.add_subplot(133)
    for agent_id in range(num_agents):
        hist_agent = hist_reshaped[:, agent_id, :]
        true_agent = true_reshaped[:, agent_id, :]
        pred_agent = pred_reshaped[:, agent_id, :]
        
        color = colors[agent_id % len(colors)]
        start_point = hist_agent[-1]
        true_display = np.vstack([start_point, true_agent])
        pred_display = np.vstack([start_point, pred_agent])
        
        ax3.plot(hist_agent[:, 0], hist_agent[:, 2], 'o-', color=color,
                 label=f'UAV{agent_id} 历史', markersize=3, linewidth=1.5, alpha=0.7)
        ax3.plot(true_display[:, 0], true_display[:, 2], 's-', color=color,
                 label=f'UAV{agent_id} 真实', markersize=3, linewidth=2, alpha=0.8)
        ax3.plot(pred_display[:, 0], pred_display[:, 2], '^--', color=color,
                 label=f'UAV{agent_id} 预测', markersize=3, linewidth=1.5, alpha=0.6)
    
    ax3.set_xlabel('X (m)', fontweight='bold')
    ax3.set_ylabel('Z (m)', fontweight='bold')
    ax3.set_title('XZ 平面视图', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.suptitle(f'多无人机轨迹预测对比 - 样本 #{sample_idx:03d}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, f'prediction_{sample_idx:03d}.png')
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_file


def calculate_sample_error(true_future, pred_future):
    """计算单个样本的误差"""
    error_per_step = np.linalg.norm(pred_future - true_future, axis=1)
    ade = np.mean(error_per_step)
    fde = np.linalg.norm(pred_future[-1] - true_future[-1])
    return ade, fde


def main():
    args = parse_args()
    
    # 规范化路径
    args.predictions_file = os.path.abspath(args.predictions_file)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("轨迹预测结果可视化")
    print("=" * 70)
    print()
    
    try:
        # 加载预测结果
        input_history, ground_truth, predictions, overall_ade, overall_fde = load_predictions(
            args.predictions_file
        )
        
        # 检测无人机数量
        num_agents = args.num_agents
        if input_history.shape[-1] % 3 == 0:
            detected_agents = input_history.shape[-1] // 3
            if detected_agents != num_agents:
                print(f"\n⚠ 警告: 数据维度表明有 {detected_agents} 个无人机，但指定了 {num_agents} 个")
                num_agents = detected_agents
                print(f"使用检测到的无人机数量: {num_agents}")
        
        print(f"\n无人机数量: {num_agents}")
        print(f"生成可视化样本 ({args.num_samples} 个)...")
        num_samples = min(args.num_samples, len(input_history))
        
        sample_errors = []
        
        for idx in range(num_samples):
            try:
                hist = input_history[idx]
                true_fut = ground_truth[idx]
                pred_fut = predictions[idx]
                
                # 计算误差（支持多无人机）
                if num_agents > 1:
                    # 对每个无人机计算误差
                    hist_reshaped = hist.reshape(hist.shape[0], num_agents, 3)
                    true_reshaped = true_fut.reshape(true_fut.shape[0], num_agents, 3)
                    pred_reshaped = pred_fut.reshape(pred_fut.shape[0], num_agents, 3)
                    
                    # 计算整体误差（所有无人机平均）
                    sample_ade, sample_fde = calculate_sample_error(
                        true_reshaped.reshape(-1, 3), pred_reshaped.reshape(-1, 3)
                    )
                else:
                    sample_ade, sample_fde = calculate_sample_error(true_fut, pred_fut)
                
                sample_errors.append((sample_ade, sample_fde))
                
                # 生成可视化
                output_file = visualize_sample(
                    hist, true_fut, pred_fut, idx, args.output_dir, 
                    num_agents=num_agents, dpi=args.dpi
                )
                
                print(f"  ✓ 样本 #{idx:03d}: ADE={sample_ade:.6f}m, FDE={sample_fde:.6f}m")
            
            except Exception as e:
                print(f"  ✗ 样本 #{idx:03d} 失败: {e}")
        
        # 统计信息
        print()
        print("=" * 70)
        print(f"✓ 可视化完成！共生成 {num_samples} 个样本")
        print()
        print("全局指标:")
        print(f"  整体 ADE: {overall_ade:.6f} m")
        print(f"  整体 FDE: {overall_fde:.6f} m")
        
        if sample_errors:
            sample_ades = [e[0] for e in sample_errors]
            sample_fdes = [e[1] for e in sample_errors]
            print()
            print("样本指标统计:")
            print(f"  平均样本 ADE: {np.mean(sample_ades):.6f} m")
            print(f"  中位数 ADE: {np.median(sample_ades):.6f} m")
            print(f"  平均样本 FDE: {np.mean(sample_fdes):.6f} m")
            print(f"  中位数 FDE: {np.median(sample_fdes):.6f} m")
        
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
