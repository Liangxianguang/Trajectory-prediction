#!/usr/bin/env python3
"""
LBEBM3D 模型推理 + 可视化脚本
用于评估训练好的模型效果，生成轨迹预测可视化

使用示例：
python visualize_lbebm3d.py \
  --model saved_models/lbebm3D_scene1.pt \
  --dataset_folder dataset \
  --dataset_name swarm \
  --output_dir validation_results_lbebm3d \
  --num_samples 50 \
  --interactive
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed, interactive HTML plots will be skipped")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(__file__))

import utils
from trajectory_augmenter import TrajectoryAugmenter


def parse_args():
    parser = argparse.ArgumentParser(description="LBEBM3D 轨迹预测可视化")
    parser.add_argument('--model', type=str, required=True,
                        help='已训练模型的路径 (e.g., saved_models/lbebm3D_scene1.pt)')
    parser.add_argument('--dataset_folder', type=str, default='dataset',
                        help='数据集文件夹路径')
    parser.add_argument('--dataset_name', type=str, default='swarm',
                        help='数据集名称 (e.g., swarm, eth)')
    parser.add_argument('--output_dir', type=str, default='validation_results_lbebm3d',
                        help='输出结果保存目录')
    parser.add_argument('--obs', type=int, default=20,
                        help='观测步长 (default: 20)')
    parser.add_argument('--preds', type=int, default=10,
                        help='预测步长 (default: 10)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='要可视化的样本数量 (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='推理时的批处理大小')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU 设备 ID (default: 0)')
    parser.add_argument('--interactive', action='store_true',
                        help='显示交互式 Matplotlib 窗口')
    parser.add_argument('--use_plotly', action='store_true',
                        help='生成 Plotly 交互式 HTML (需要 plotly 包)')
    return parser.parse_args()


class LBEBM3DVisualizer:
    def __init__(self, model_path, dataset_folder, dataset_name, obs, preds, device=0):
        """初始化可视化器"""
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.obs = obs
        self.preds = preds
        self.model_path = model_path
        self.dataset_folder = dataset_folder
        self.dataset_name = dataset_name
        
        print(f"设备: {self.device}")
        print(f"观测长度: {obs}, 预测长度: {preds}")
        
    def _infer_model_params_from_checkpoint(self, state_dict):
        """
        从checkpoint的state_dict智能推断模型参数
        MLP(input_dim, output_dim, hidden_size)：
          hidden_size是所有隐藏层的输出维度列表
          例: hidden_size=[512, 256]表示输入→512→256→输出
        """
        print(f"\n  🔍 从checkpoint自动识别模型参数...")
        
        params = {}
        
        def extract_hidden_sizes(module_name):
            """提取某个模块的隐藏层维度"""
            hidden = []
            i = 0
            while f'{module_name}.layers.{i}.weight' in state_dict:
                w = state_dict[f'{module_name}.layers.{i}.weight']
                hidden.append(w.shape[0])  # 输出维度
                i += 1
            # 去掉最后一个元素（那是输出层维度，由output_dim参数确定）
            if len(hidden) > 1:
                return hidden[:-1]
            return hidden
        
        # 推断 enc_past_size
        enc_past = extract_hidden_sizes('encoder_past')
        if enc_past:
            params['enc_past_size'] = enc_past
            print(f"    ✓ enc_past_size: {params['enc_past_size']}")
        
        # 推断 enc_dest_size
        enc_dest = extract_hidden_sizes('encoder_dest')
        if enc_dest:
            params['enc_dest_size'] = enc_dest
            print(f"    ✓ enc_dest_size: {params['enc_dest_size']}")
        
        # 推断 enc_latent_size
        enc_latent = extract_hidden_sizes('encoder_latent')
        if enc_latent:
            params['enc_latent_size'] = enc_latent
            print(f"    ✓ enc_latent_size: {params['enc_latent_size']}")
        
        # 推断 dec_size（decoder的所有隐藏层）
        # decoder_z、decoder_x、decoder_y结构相同
        dec = extract_hidden_sizes('decoder_z')
        if dec:
            params['dec_size'] = dec
            print(f"    ✓ dec_size: {params['dec_size']}")
        
        # 推断 predictor_size
        pred = extract_hidden_sizes('predictor_x')
        if pred:
            params['predictor_size'] = pred
            print(f"    ✓ predictor_size: {params['predictor_size']}")
        
        # 推断 fdim：从encoder_past的最后一层输出维度
        encoder_past_max = 0
        while f'encoder_past.layers.{encoder_past_max}.weight' in state_dict:
            encoder_past_max += 1
        encoder_past_max -= 1
        if f'encoder_past.layers.{encoder_past_max}.weight' in state_dict:
            w = state_dict[f'encoder_past.layers.{encoder_past_max}.weight']
            params['fdim'] = w.shape[0]
            print(f"    ✓ fdim (从encoder_past最后层推断): {params['fdim']}")
        
        # 推断 zdim：从encoder_latent的最后一层输出维度（是2*zdim）
        encoder_latent_max = 0
        while f'encoder_latent.layers.{encoder_latent_max}.weight' in state_dict:
            encoder_latent_max += 1
        encoder_latent_max -= 1
        if f'encoder_latent.layers.{encoder_latent_max}.weight' in state_dict:
            w = state_dict[f'encoder_latent.layers.{encoder_latent_max}.weight']
            params['zdim'] = w.shape[0] // 2
            print(f"    ✓ zdim (从encoder_latent最后层推断): {params['zdim']}")
        
        # 设置默认值
        params.setdefault('sigma', 1.3)
        
        return params
    
    def load_model(self):
        """加载模型 - 使用智能参数识别"""
        print(f"\n{'='*60}")
        print(f"📦 加载模型: {self.model_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 直接导入模块级别的 LBEBM3D 类
        from lbebm3D import LBEBM3D
        
        # 加载 checkpoint
        print(f"\n  [1/3] 加载checkpoint文件...")
        checkpoint = torch.load(self.model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # 从state_dict智能推断模型参数
        print(f"  [2/3] 智能识别模型参数...")
        inferred_params = self._infer_model_params_from_checkpoint(state_dict)
        
        # 创建模型实例 (使用推断的参数)
        print(f"\n  [3/3] 创建模型实例...")
        model = LBEBM3D(
            enc_past_size=inferred_params.get('enc_past_size', [512, 256]),
            enc_dest_size=inferred_params.get('enc_dest_size', [256, 128]),
            enc_latent_size=inferred_params.get('enc_latent_size', [256, 512]),
            dec_size=inferred_params.get('dec_size', [1024, 512, 1024]),
            predictor_size=inferred_params.get('predictor_size', [1024, 512, 256]),
            fdim=inferred_params.get('fdim', 16),
            zdim=inferred_params.get('zdim', 16),
            sigma=inferred_params.get('sigma', 1.3),
            past_length=self.obs,
            future_length=self.preds,
            sub_goal_indexes=[2, 5, 7, 9]
        )
        
        # 转为 Double 精度（与训练时一致）并加载权重
        model = model.double()
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        print(f"\n  ✓ 模型加载成功")
        print(f"  ✓ 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  ✓ 模型数据类型: {next(model.parameters()).dtype}")
        
        return model
    
    def _load_model_class_dynamically(self):
        """备用：动态加载（如果直接导入失败）"""
        import importlib.util
        
        lbebm3d_path = os.path.join(os.path.dirname(__file__), 'lbebm3D.py')
        spec = importlib.util.spec_from_file_location("lbebm3d_temp", lbebm3d_path)
        module = importlib.util.module_from_spec(spec)
        
        # 设置必要的全局变量供模块使用
        import random
        import shutil
        import logging
        import datetime
        module.torch = torch
        module.nn = nn
        module.F = F
        module.random = random
        module.np = np
        module.pd = pd
        module.os = os
        module.sys = sys
        module.shutil = shutil
        module.logging = logging
        module.datetime = datetime
        module.Variable = Variable
        module.TrajectoryAugmenter = TrajectoryAugmenter
        module.utils = utils
        
        try:
            spec.loader.exec_module(module)
            if hasattr(module, 'LBEBM3D'):
                print("✓ 从动态加载获取 LBEBM3D 类")
                return module.LBEBM3D
        except Exception as e:
            print(f"动态加载失败: {e}")
        
        raise ImportError("无法加载 LBEBM3D 类")
    
    def load_dataset(self, split='val', batch_size=128):
        """加载数据集"""
        val_dataset, _ = utils.create_dataset(
            self.dataset_folder,
            self.dataset_name,
            0,
            self.obs,
            self.preds,
            train=(split == 'train'),
            verbose=True
        )
        
        dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return dataloader, val_dataset
    
    def predict_batch(self, model, past_traj, sub_goal_indexes=[2, 5, 7, 9]):
        """
        对一个批次进行完整推理预测
        
        推理流程:
        1. 编码过去轨迹 → ftraj (batch, fdim)
        2. 通过EBM采样潜变量 z (batch, zdim)
        3. 通过解码器生成子目标位置
        4. 通过预测器生成完整轨迹
        
        Args:
            model: LBEBM3D 模型
            past_traj: 过去的轨迹 (batch, obs*3)
            sub_goal_indexes: 子目标索引
        
        Returns:
            predicted_future: 预测的未来轨迹 (batch, preds*3)
        """
        # 确保输入数据类型与模型一致（转为 Double，与模型权重匹配）
        if past_traj.dtype != torch.float64:
            past_traj = past_traj.double()
        
        batch_size = past_traj.shape[0]
        
        # 1. 编码过去轨迹
        with torch.no_grad():
            ftraj = model.encoder_past(past_traj)  # (batch, fdim)
        
        # 2. 通过EBM采样潜变量 z
        # 初始化随机z
        z_init = torch.randn(batch_size, model.zdim, dtype=torch.float64, device=self.device) * 2.0
        
        # Langevin采样参数 (来自训练配置的默认值)
        e_l_steps = 20  # Langevin步数
        e_l_step_size = 0.4  # 步长
        e_l_with_noise = True  # 是否添加噪声
        e_prior_sig = 2.0  # 先验方差
        
        # 执行Langevin采样
        z = z_init.clone().detach()
        
        for step in range(e_l_steps):
            # 为当前步骤创建需要梯度的张量
            z_opt = z.clone().detach().requires_grad_(True)
            
            # 计算能量（不再使用no_grad）
            z_c = torch.cat((z_opt, ftraj.detach()), dim=1)  # (batch, zdim+fdim)
            neg_energy = model.EBM(z_c)  # (batch, ny)
            energy = -neg_energy.logsumexp(dim=1)  # (batch,)
            
            # 计算梯度
            z_grad = torch.autograd.grad(energy.sum(), z_opt, create_graph=False)[0]
            
            # Langevin更新（在detach的数据上进行）
            z = z - 0.5 * e_l_step_size * e_l_step_size * (
                z_grad.detach() + 1.0 / (e_prior_sig * e_prior_sig) * z
            )
            if e_l_with_noise:
                z = z + e_l_step_size * torch.randn_like(z)
        
        # 3. 通过解码器生成子目标位置
        with torch.no_grad():
            z_concat = torch.cat((ftraj, z.detach()), dim=1)  # (batch, fdim+zdim)
            dest_x = model.decoder_x(z_concat)  # (batch, num_subgoals)
            dest_y = model.decoder_y(z_concat)  # (batch, num_subgoals)
            dest_z = model.decoder_z(z_concat)  # (batch, num_subgoals)
            
            # 4. 重组为坐标形式 (batch, num_subgoals*3)
            num_subgoals = dest_x.shape[1]
            generated_dest = torch.zeros(batch_size, num_subgoals*3, dtype=torch.float64, device=self.device)
            for i in range(num_subgoals):
                generated_dest[:, i*3] = dest_x[:, i]
                generated_dest[:, i*3+1] = dest_y[:, i]
                generated_dest[:, i*3+2] = dest_z[:, i]
            
            # 5. 进行轨迹预测
            interpolated_future = model.predict(past_traj, generated_dest)
            predicted_future = interpolated_future.cpu().numpy()
        
        return predicted_future
    
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
    
    def plot_single_prediction(self, history, true_future, pred_future, sample_id, output_dir):
        """绘制单个样本的预测结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 重塑数据
        history = history.reshape(-1, 3)
        true_future = true_future.reshape(-1, 3)
        pred_future = pred_future.reshape(-1, 3)
        
        start_point = history[-1]
        true_display = np.vstack([start_point, true_future])
        pred_display = np.vstack([start_point, pred_future])
        
        # 计算误差
        errors = np.linalg.norm(pred_future - true_future, axis=1)
        ade = np.mean(errors)
        fde = np.linalg.norm(pred_future[-1] - true_future[-1])
        
        # 创建图表
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 3D 轨迹图
        ax3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        ax3d.plot(history[:, 0], history[:, 1], history[:, 2], 'b-o', 
                 label='输入历史', linewidth=2.5, markersize=6)
        ax3d.plot(true_display[:, 0], true_display[:, 1], true_display[:, 2], 'g-s', 
                 label='真实未来', linewidth=2.5, markersize=7)
        ax3d.plot(pred_display[:, 0], pred_display[:, 1], pred_display[:, 2], 'r--^', 
                 label='预测未来', linewidth=2, markersize=7)
        ax3d.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax3d.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax3d.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
        ax3d.set_title(f'3D 轨迹对比 (样本 #{sample_id})', fontsize=13, fontweight='bold')
        ax3d.legend(fontsize=10, loc='upper left')
        ax3d.grid(True, alpha=0.3)
        
        # XY 平面
        ax_xy = fig.add_subplot(gs[0, 2])
        ax_xy.plot(history[:, 0], history[:, 1], 'b-o', label='历史', linewidth=2)
        ax_xy.plot(true_display[:, 0], true_display[:, 1], 'g-s', label='真实', linewidth=2.5)
        ax_xy.plot(pred_display[:, 0], pred_display[:, 1], 'r--^', label='预测', linewidth=2)
        ax_xy.set_xlabel('X (m)', fontsize=10)
        ax_xy.set_ylabel('Y (m)', fontsize=10)
        ax_xy.set_title('XY 平面视图')
        ax_xy.legend(fontsize=8)
        ax_xy.grid(True, alpha=0.3)
        
        # XZ 平面
        ax_xz = fig.add_subplot(gs[1, 2])
        ax_xz.plot(history[:, 0], history[:, 2], 'b-o', linewidth=2)
        ax_xz.plot(true_display[:, 0], true_display[:, 2], 'g-s', linewidth=2.5)
        ax_xz.plot(pred_display[:, 0], pred_display[:, 2], 'r--^', linewidth=2)
        ax_xz.set_xlabel('X (m)', fontsize=10)
        ax_xz.set_ylabel('Z (m)', fontsize=10)
        ax_xz.set_title('XZ 平面视图')
        ax_xz.grid(True, alpha=0.3)
        
        # 逐步误差
        ax_error_steps = fig.add_subplot(gs[2, 0:2])
        steps = np.arange(len(true_future))
        error_x = np.abs(pred_future[:, 0] - true_future[:, 0])
        error_y = np.abs(pred_future[:, 1] - true_future[:, 1])
        error_z = np.abs(pred_future[:, 2] - true_future[:, 2])
        
        ax_error_steps.plot(steps, error_x, 'r-s', label='X 轴误差', linewidth=2.5, markersize=6)
        ax_error_steps.plot(steps, error_y, 'b-o', label='Y 轴误差', linewidth=2.5, markersize=6)
        ax_error_steps.plot(steps, error_z, 'g-^', label='Z 轴误差', linewidth=2.5, markersize=6)
        ax_error_steps.set_xlabel('预测步数', fontsize=11, fontweight='bold')
        ax_error_steps.set_ylabel('绝对误差 (m)', fontsize=11, fontweight='bold')
        ax_error_steps.set_title('各轴逐步误差', fontsize=12, fontweight='bold')
        ax_error_steps.grid(True, alpha=0.3)
        ax_error_steps.legend(fontsize=9)
        
        # 误差分布柱状图
        ax_error_dist = fig.add_subplot(gs[2, 2])
        ax_error_dist.bar(steps, errors, color='tab:red', alpha=0.7, edgecolor='darkred')
        ax_error_dist.set_xlabel('步数', fontsize=10)
        ax_error_dist.set_ylabel('位置误差 (m)', fontsize=10)
        ax_error_dist.set_title(f'每步误差\n(ADE={ade:.4f}m, FDE={fde:.4f}m)', fontsize=11, fontweight='bold')
        ax_error_dist.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        png_path = output_dir / f'prediction_sample_{sample_id:03d}.png'
        fig.savefig(png_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        return png_path, ade, fde
    
    def generate_summary_report(self, all_ades, all_fdes, output_dir):
        """生成摘要报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'summary_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("LBEBM3D 轨迹预测模型 - 评估报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"数据集: {self.dataset_name}\n")
            f.write(f"观测步长: {self.obs}, 预测步长: {self.preds}\n\n")
            
            f.write("性能指标\n")
            f.write("-" * 70 + "\n")
            f.write(f"平均 ADE (Average Displacement Error): {np.mean(all_ades):.6f} m\n")
            f.write(f"中位数 ADE: {np.median(all_ades):.6f} m\n")
            f.write(f"最小 ADE: {np.min(all_ades):.6f} m\n")
            f.write(f"最大 ADE: {np.max(all_ades):.6f} m\n")
            f.write(f"ADE 标准差: {np.std(all_ades):.6f} m\n\n")
            
            f.write(f"平均 FDE (Final Displacement Error): {np.mean(all_fdes):.6f} m\n")
            f.write(f"中位数 FDE: {np.median(all_fdes):.6f} m\n")
            f.write(f"最小 FDE: {np.min(all_fdes):.6f} m\n")
            f.write(f"最大 FDE: {np.max(all_fdes):.6f} m\n")
            f.write(f"FDE 标准差: {np.std(all_fdes):.6f} m\n\n")
            
            f.write("样本数量\n")
            f.write("-" * 70 + "\n")
            f.write(f"已评估样本数: {len(all_ades)}\n")
            f.write("=" * 70 + "\n")
        
        print(f"\n✓ 摘要报告已保存到: {report_path}")
        return report_path
    
    def run(self, num_samples=50, batch_size=128, output_dir='validation_results_lbebm3d',
            interactive=False, use_plotly=False):
        """运行完整的评估流程"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("LBEBM3D 轨迹预测可视化评估")
        print("=" * 70)
        
        # 加载模型
        print("\n[1/4] 加载模型...")
        model = self.load_model()
        
        # 加载数据集
        print("\n[2/4] 加载验证数据集...")
        val_dataloader, val_dataset = self.load_dataset(split='val', batch_size=batch_size)
        
        print(f"✓ 验证集大小: {len(val_dataset)} 个样本")
        
        # 收集预测结果
        print(f"\n[3/4] 进行推理预测 (处理 {num_samples} 个样本)...")
        all_ades = []
        all_fdes = []
        
        sample_count = 0
        for batch_idx, batch in enumerate(val_dataloader):
            if sample_count >= num_samples:
                break
            
            # 确保数据转为 Double 类型（与模型权重匹配）
            past_traj = batch['src'][:, :, :3].to(self.device)
            if past_traj.dtype != torch.float64:
                past_traj = past_traj.double()
            
            true_future = batch['trg'][:, :, :3].double().cpu().numpy()
            
            # 重塑为模型输入格式
            past_traj_flat = past_traj.reshape(past_traj.shape[0], -1)
            
            # 预测
            pred_future = self.predict_batch(model, past_traj_flat)
            
            # 计算指标
            ade, fde = self.calculate_metrics(pred_future, true_future)
            
            # 保存单个样本的可视化
            for i in range(len(ade)):
                if sample_count >= num_samples:
                    break
                
                history = past_traj[i].cpu().numpy()
                true_fut = true_future[i]
                pred_fut = pred_future[i]
                
                png_path, sample_ade, sample_fde = self.plot_single_prediction(
                    history, true_fut, pred_fut, sample_count, output_dir
                )
                
                all_ades.append(sample_ade)
                all_fdes.append(sample_fde)
                
                print(f"  样本 #{sample_count:03d}: ADE={sample_ade:.6f}m, FDE={sample_fde:.6f}m")
                
                sample_count += 1
        
        # 生成摘要报告
        print(f"\n[4/4] 生成摘要报告...")
        self.generate_summary_report(all_ades, all_fdes, output_dir)
        
        # 打印最终统计
        print("\n" + "=" * 70)
        print("最终评估结果")
        print("=" * 70)
        print(f"平均 ADE: {np.mean(all_ades):.6f} m")
        print(f"平均 FDE: {np.mean(all_fdes):.6f} m")
        print(f"已处理样本: {len(all_ades)}")
        print(f"结果保存到: {output_dir}")
        print("=" * 70 + "\n")
        
        if interactive:
            print("提示: 所有 PNG 图表已保存到输出目录，可用图像查看器打开")


def main():
    args = parse_args()
    
    visualizer = LBEBM3DVisualizer(
        model_path=args.model,
        dataset_folder=args.dataset_folder,
        dataset_name=args.dataset_name,
        obs=args.obs,
        preds=args.preds,
        device=args.device
    )
    
    visualizer.run(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        interactive=args.interactive,
        use_plotly=args.use_plotly
    )


if __name__ == '__main__':
    main()
