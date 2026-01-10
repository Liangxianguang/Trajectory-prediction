#!/usr/bin/env python3
"""
简化版可视化脚本 - 直接调用 lbebm3D.py 进行推理和可视化
无需复杂的类导入，直接使用已训练的模型
"""
import subprocess
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def run_inference_and_save_results(args):
    """运行模型推理并保存结果"""
    
    # 构建推理命令
    cmd = [
        sys.executable, 
        'lbebm3D.py',
        '--test_mode',
        '--dataset_name', args.dataset_name,
        '--dataset_folder', args.dataset_folder,
        '--model_path', args.model_path,
        '--device', str(args.device),
        '--obs', str(args.obs),
        '--preds', str(args.preds),
        '--past_length', str(args.past_length),
        '--future_length', str(args.future_length),
        '--batch_size', str(args.batch_size),
        '--n_values', str(args.num_samples),
    ]
    
    print("运行模型推理...")
    print(f"命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print("模型推理失败！")
        return False
    
    return True

def generate_sample_visualizations(args, output_dir, num_samples=10):
    """生成可视化样本"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[2/2] 生成可视化样本 ({num_samples} 个)...")
    
    # 加载实际的测试数据来生成可视化
    import pickle
    dataset_path = os.path.join(
        args.dataset_folder, 
        args.dataset_name, 
        'test', 
        'saved_data.pickle'
    )
    
    if not os.path.exists(dataset_path):
        print(f"警告：找不到数据文件 {dataset_path}")
        return
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # 数据结构: src (N, 20, 6), trg (N, 10, 6)
    src_data = data['src']  # (N, past_length, 6)
    trg_data = data['trg']  # (N, future_length, 6)
    
    num_samples = min(num_samples, len(src_data))
    
    for sample_idx in range(num_samples):
        try:
            # 获取数据 - 只取前3个维度 (x, y, z)
            hist = src_data[sample_idx, :, :3]  # (20, 3)
            future_true = trg_data[sample_idx, :, :3]  # (10, 3)
            
            # 创建 3D 图表
            fig = plt.figure(figsize=(15, 5))
            
            # 3D 视图
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot(hist[:, 0], hist[:, 1], hist[:, 2], 'b-o', label='历史轨迹', markersize=5, linewidth=2)
            ax1.plot(future_true[:, 0], future_true[:, 1], future_true[:, 2], 'g-s', label='真实未来', markersize=5, linewidth=2)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title(f'3D 轨迹对比 (样本 #{sample_idx})')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # XY 平面
            ax2 = fig.add_subplot(132)
            ax2.plot(hist[:, 0], hist[:, 1], 'b-o', label='历史轨迹', markersize=5, linewidth=2)
            ax2.plot(future_true[:, 0], future_true[:, 1], 'g-s', label='真实未来', markersize=5, linewidth=2)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('XY 平面视图')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal', adjustable='box')
            
            # XZ 平面
            ax3 = fig.add_subplot(133)
            ax3.plot(hist[:, 0], hist[:, 2], 'b-o', label='历史轨迹', markersize=5, linewidth=2)
            ax3.plot(future_true[:, 0], future_true[:, 2], 'g-s', label='真实未来', markersize=5, linewidth=2)
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Z (m)')
            ax3.set_title('XZ 平面视图')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_aspect('equal', adjustable='box')
            
            plt.suptitle(f'轨迹预测可视化 - 样本 #{sample_idx:03d}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # 保存图表
            output_file = os.path.join(output_dir, f'sample_{sample_idx:03d}.png')
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 样本 #{sample_idx:03d} 已保存")
        
        except Exception as e:
            print(f"  ⚠ 样本 {sample_idx} 生成失败: {e}")
    
    print(f"\n✓ 可视化样本已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='LBEBM3D 轨迹预测可视化')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--dataset_folder', type=str, default='../dataset', help='数据集文件夹')
    parser.add_argument('--dataset_name', type=str, default='swarm', help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='../validation_results', help='输出目录')
    parser.add_argument('--obs', type=int, default=20, help='观测步长')
    parser.add_argument('--preds', type=int, default=10, help='预测步长')
    parser.add_argument('--past_length', type=int, default=20, help='模型 past_length')
    parser.add_argument('--future_length', type=int, default=10, help='模型 future_length')
    parser.add_argument('--batch_size', type=int, default=70, help='批大小')
    parser.add_argument('--device', type=int, default=0, help='GPU 设备号')
    parser.add_argument('--num_samples', type=int, default=20, help='可视化样本数')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LBEBM3D 轨迹预测可视化")
    print("=" * 70)
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.dataset_name}")
    print(f"观测长度: {args.obs}, 预测长度: {args.preds}")
    print()
    
    # 规范化路径
    args.model_path = os.path.abspath(args.model_path)
    args.dataset_folder = os.path.abspath(args.dataset_folder)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # 运行推理
    print("[1/2] 运行模型推理...")
    success = run_inference_and_save_results(args)
    
    if not success:
        print("推理失败！")
        return
    
    print("[2/2] ✓ 推理完成")
    print()
    
    # 生成可视化
    generate_sample_visualizations(args, args.output_dir, args.num_samples)
    
    print("\n" + "=" * 70)
    print("✓ 可视化完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()