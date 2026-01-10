#!/usr/bin/env python3
"""
LBEBM3D 模型评估 + 可视化脚本
通过调用 lbebm3D.py 进行推理，然后生成可视化结果

使用示例：
python eval_and_visualize.py \
  --model_path ./saved_models/lbebm3D_scene1.pt \
  --dataset_name swarm \
  --output_dir validation_results_lbebm3d \
  --num_samples 50
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import subprocess
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(__file__))

import utils


def parse_args():
    parser = argparse.ArgumentParser(description="LBEBM3D 轨迹预测评估 + 可视化")
    parser.add_argument('--model_path', type=str, required=True,
                        help='已训练模型的路径')
    parser.add_argument('--dataset_folder', type=str, default='dataset',
                        help='数据集文件夹路径')
    parser.add_argument('--dataset_name', type=str, default='swarm',
                        help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='validation_results_lbebm3d',
                        help='输出结果保存目录')
    parser.add_argument('--obs', type=int, default=20,
                        help='观测步长')
    parser.add_argument('--preds', type=int, default=10,
                        help='预测步长')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='要评估的样本数量')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批处理大小')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU 设备 ID')
    parser.add_argument('--past_length', type=int, default=20,
                        help='过去步数长度')
    parser.add_argument('--future_length', type=int, default=10,
                        help='未来步数长度')
    return parser.parse_args()


class EvalVisualizer:
    def __init__(self, model_path, dataset_folder, dataset_name, obs, preds, 
                 past_length, future_length, device=0):
        """初始化评估可视化器"""
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.obs = obs
        self.preds = preds
        self.past_length = past_length
        self.future_length = future_length
        self.model_path = model_path
        
        # 处理相对路径：如果 dataset_folder 是相对路径，转换为相对于 tool 目录的父目录
        if not os.path.isabs(dataset_folder):
            # 从 tool 目录返回到项目根目录，然后进入 dataset
            tool_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(tool_dir)
            self.dataset_folder = os.path.join(parent_dir, dataset_folder)
        else:
            self.dataset_folder = dataset_folder
        
        self.dataset_name = dataset_name
        
        print(f"设备: {self.device}")
        print(f"观测长度: {obs}, 预测长度: {preds}")
        print(f"模型参数: past_length={past_length}, future_length={future_length}")
        print(f"数据集路径: {self.dataset_folder}")
    
    def load_dataset(self, split='test', batch_size=128):
        """加载数据集"""
        print(f"\n[1/2] 加载 {split} 数据集...")
        test_dataset, _ = utils.create_dataset(
            self.dataset_folder,
            self.dataset_name,
            0,
            self.obs,
            self.preds,
            train=(split == 'train'),
            eval=(split == 'test'),
            verbose=True
        )
        
        dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"✓ 数据集加载完成，共 {len(test_dataset)} 个样本")
        return dataloader, test_dataset
    
    def run_model_evaluation(self):
        """
        运行 lbebm3D.py 进行模型评估
        返回 ADE, FDE 结果
        """
        print(f"\n[2/2] 运行模型评估...")
        
        # 获取当前 tool 目录
        tool_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 规范化路径（处理 ../ 等相对路径，保持在当前目录相对）
        if not os.path.isabs(self.model_path):
            # 相对于 tool 目录计算绝对路径
            abs_model_path = os.path.abspath(os.path.join(tool_dir, self.model_path))
        else:
            abs_model_path = self.model_path
        
        # 数据集路径已经在 __init__ 中处理过了
        abs_dataset_folder = self.dataset_folder
        
        print(f"  规范化模型路径: {abs_model_path}")
        print(f"  规范化数据集路径: {abs_dataset_folder}")
        
        # 验证文件是否存在
        if not os.path.exists(abs_model_path):
            print(f"  ❌ 模型文件不存在: {abs_model_path}")
            return None, None, f"模型文件不存在: {abs_model_path}"
        
        cmd = [
            'python', 'lbebm3D.py',
            '--test_mode',
            '--dataset_name', self.dataset_name,
            '--dataset_folder', abs_dataset_folder,
            '--model_path', abs_model_path,
            '--device', str(self.device.index if hasattr(self.device, 'index') else 0),
            '--obs', str(self.obs),
            '--preds', str(self.preds),
            '--past_length', str(self.past_length),
            '--future_length', str(self.future_length),
        ]
        
        print(f"  执行命令: {' '.join(cmd)}")
        print(f"  工作目录: {tool_dir}")
        
        try:
            # 在 tool 目录中运行命令
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=tool_dir, timeout=300)
            output = result.stdout + result.stderr
            
            print(f"\n--- 模型评估输出 ---")
            print(output)
            print(f"--- 输出结束 ---\n")
            
            # 从输出中提取 ADE 和 FDE
            lines = output.split('\n')
            ade_value = None
            fde_value = None
            
            for line in lines:
                print(f"  检查行: {line}")
                if 'test ADE' in line:
                    try:
                        parts = line.split()
                        ade_value = float(parts[-1])
                        print(f"  ✓ 提取 ADE: {ade_value}")
                    except Exception as e:
                        print(f"  ✗ 提取 ADE 失败: {e}")
                elif 'test FDE' in line:
                    try:
                        parts = line.split()
                        fde_value = float(parts[-1])
                        print(f"  ✓ 提取 FDE: {fde_value}")
                    except Exception as e:
                        print(f"  ✗ 提取 FDE 失败: {e}")
            
            return ade_value, fde_value, output
        
        except subprocess.TimeoutExpired:
            print(f"❌ 模型评估超时（>5分钟）")
            return None, None, "超时"
        except Exception as e:
            print(f"❌ 模型评估失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, str(e)
    
    def calculate_metrics(self, pred_traj, true_traj):
        """计算预测误差指标"""
        # 重塑为 (batch, steps, 3)
        pred_reshaped = pred_traj.reshape(-1, self.preds, 3)
        true_reshaped = true_traj.reshape(-1, self.preds, 3)
        
        # ADE (Average Displacement Error)
        errors = np.linalg.norm(pred_reshaped - true_reshaped, axis=2)
        ade = np.mean(errors, axis=1)
        
        # FDE (Final Displacement Error)
        fde = np.linalg.norm(pred_reshaped[:, -1] - true_reshaped[:, -1], axis=1)
        
        return ade, fde
    
    def plot_single_prediction(self, history, true_future, sample_id, output_dir):
        """绘制单个样本的预测结果对比"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 重塑数据
        history = history.reshape(-1, 3)
        true_future = true_future.reshape(-1, 3)
        
        start_point = history[-1]
        true_display = np.vstack([start_point, true_future])
        
        # 创建图表
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # 3D 轨迹图
        ax3d = fig.add_subplot(gs[0:2, 0], projection='3d')
        ax3d.plot(history[:, 0], history[:, 1], history[:, 2], 'b-o', 
                 label='输入历史', linewidth=2.5, markersize=6)
        ax3d.plot(true_display[:, 0], true_display[:, 1], true_display[:, 2], 'g-s', 
                 label='真实未来', linewidth=2.5, markersize=7)
        ax3d.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax3d.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax3d.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
        ax3d.set_title(f'3D 轨迹对比 (样本 #{sample_id})', fontsize=13, fontweight='bold')
        ax3d.legend(fontsize=10, loc='upper left')
        ax3d.grid(True, alpha=0.3)
        
        # XY 平面
        ax_xy = fig.add_subplot(gs[0, 1])
        ax_xy.plot(history[:, 0], history[:, 1], 'b-o', label='历史', linewidth=2)
        ax_xy.plot(true_display[:, 0], true_display[:, 1], 'g-s', label='真实', linewidth=2.5)
        ax_xy.set_xlabel('X (m)', fontsize=10)
        ax_xy.set_ylabel('Y (m)', fontsize=10)
        ax_xy.set_title('XY 平面视图')
        ax_xy.legend(fontsize=8)
        ax_xy.grid(True, alpha=0.3)
        
        # XZ 平面
        ax_xz = fig.add_subplot(gs[1, 1])
        ax_xz.plot(history[:, 0], history[:, 2], 'b-o', linewidth=2)
        ax_xz.plot(true_display[:, 0], true_display[:, 2], 'g-s', linewidth=2.5)
        ax_xz.set_xlabel('X (m)', fontsize=10)
        ax_xz.set_ylabel('Z (m)', fontsize=10)
        ax_xz.set_title('XZ 平面视图')
        ax_xz.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        png_path = output_dir / f'sample_{sample_id:03d}.png'
        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return png_path
    
    def generate_summary_report(self, ade_value, fde_value, output_dir):
        """生成摘要报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("LBEBM3D 轨迹预测模型 - 评估报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"数据集: {self.dataset_name}\n")
            f.write(f"观测步长: {self.obs}, 预测步长: {self.preds}\n")
            f.write(f"模型参数: past_length={self.past_length}, future_length={self.future_length}\n\n")
            
            f.write("性能指标\n")
            f.write("-" * 70 + "\n")
            
            if ade_value is not None:
                f.write(f"平均 ADE (Average Displacement Error): {ade_value:.6f} m\n")
            else:
                f.write(f"ADE: 未获取\n")
            
            if fde_value is not None:
                f.write(f"平均 FDE (Final Displacement Error): {fde_value:.6f} m\n")
            else:
                f.write(f"FDE: 未获取\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"✓ 评估报告已保存到: {report_path}")
        return report_path
    
    def run(self, num_samples=50, batch_size=128, output_dir='validation_results_lbebm3d'):
        """运行完整的评估流程"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("LBEBM3D 轨迹预测模型评估")
        print("=" * 70)
        
        # 加载数据集
        val_dataloader, val_dataset = self.load_dataset(split='test', batch_size=batch_size)
        
        # 运行模型评估
        ade_value, fde_value, eval_output = self.run_model_evaluation()
        
        print(f"\n📊 评估结果:")
        if ade_value is not None:
            print(f"   ADE: {ade_value:.6f} m")
        if fde_value is not None:
            print(f"   FDE: {fde_value:.6f} m")
        
        # 生成可视化样本
        print(f"\n[3/3] 生成可视化样本...")
        sample_count = 0
        for batch_idx, batch in enumerate(val_dataloader):
            if sample_count >= num_samples:
                break
            
            past_traj = batch['src'][:, :, :3].cpu().numpy()
            true_future = batch['trg'][:, :, :3].cpu().numpy()
            
            for i in range(len(past_traj)):
                if sample_count >= num_samples:
                    break
                
                history = past_traj[i]
                true_fut = true_future[i]
                
                self.plot_single_prediction(history, true_fut, sample_count, output_dir)
                sample_count += 1
        
        # 生成摘要报告
        print(f"\n[4/4] 生成摘要报告...")
        self.generate_summary_report(ade_value, fde_value, output_dir)
        
        # 打印最终统计
        print("\n" + "=" * 70)
        print("评估完成!")
        print(f"结果保存到: {output_dir}")
        print(f"已生成 {sample_count} 个可视化样本")
        print("=" * 70 + "\n")


def main():
    args = parse_args()
    
    visualizer = EvalVisualizer(
        model_path=args.model_path,
        dataset_folder=args.dataset_folder,
        dataset_name=args.dataset_name,
        obs=args.obs,
        preds=args.preds,
        past_length=args.past_length,
        future_length=args.future_length,
        device=args.device
    )
    
    visualizer.run(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
