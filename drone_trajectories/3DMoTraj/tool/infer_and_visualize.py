#!/usr/bin/env python3
"""
从 CSV 文件读取轨迹数据进行推理和可视化
支持随机选择指定个数的轨迹样本

使用示例：
python infer_and_visualize.py \
  --model_path ../saved_models/lbebm3D_scene1.pt \
  --csv_dir ../Synthetic-UAV-Flight-Trajectories \
  --num_samples 10 \
  --output_dir ../validation_results_csv
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
import random
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='从CSV读取轨迹进行推理和可视化')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--csv_dir', type=str, required=True, help='CSV文件目录')
    parser.add_argument('--output_dir', type=str, default='../validation_results_csv', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10, help='样本数量')
    parser.add_argument('--obs_length', type=int, default=20, help='观测步长')
    parser.add_argument('--pred_length', type=int, default=10, help='预测步长')
    parser.add_argument('--device', type=int, default=0, help='GPU设备号')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()


def load_trajectory_from_csv(csv_file, obs_length, pred_length):
    """从CSV文件加载轨迹数据"""
    try:
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        
        if len(data) < obs_length + pred_length:
            return None
        
        # 提取 x, y, z 坐标（假设格式为 x, y, z, ... 或其他格式）
        # 根据实际CSV格式调整列索引
        if data.shape[1] >= 3:
            # 取前3列作为 x, y, z
            trajectory = data[:, :3]
        else:
            return None
        
        # 生成多个样本（滑动窗口）
        samples = []
        for start_idx in range(len(trajectory) - obs_length - pred_length + 1):
            hist = trajectory[start_idx:start_idx + obs_length]
            future = trajectory[start_idx + obs_length:start_idx + obs_length + pred_length]
            samples.append((hist, future))
        
        return samples
    
    except Exception as e:
        print(f"  加载CSV失败 {csv_file}: {e}")
        return None


def load_all_trajectories(csv_dir, obs_length, pred_length, num_samples):
    """从目录中加载所有CSV文件并提取样本"""
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    all_samples = []
    
    for csv_file in csv_files:
        samples = load_trajectory_from_csv(csv_file, obs_length, pred_length)
        if samples:
            all_samples.extend(samples)
            print(f"  ✓ {os.path.basename(csv_file)}: 提取 {len(samples)} 个样本")
    
    print(f"\n总共提取 {len(all_samples)} 个样本")
    
    # 随机选择指定个数的样本
    if len(all_samples) > num_samples:
        selected_samples = random.sample(all_samples, num_samples)
        print(f"随机选择 {num_samples} 个样本进行推理")
    else:
        selected_samples = all_samples
        print(f"可用样本少于 {num_samples}，使用全部 {len(all_samples)} 个样本")
    
    return selected_samples


def infer_with_model(model, past_traj, device, e_l_steps=20, e_l_step_size=0.4, e_prior_sig=2.0):
    """使用模型进行推理"""
    if past_traj.dtype != torch.float64:
        past_traj = torch.from_numpy(past_traj).double().to(device)
    else:
        past_traj = torch.from_numpy(past_traj).to(device)
    
    # 添加batch维度
    if past_traj.dim() == 2:
        past_traj = past_traj.unsqueeze(0)
    
    batch_size = past_traj.shape[0]
    
    with torch.no_grad():
        ftraj = model.encoder_past(past_traj)
    
    # 初始化随机z
    z_init = torch.randn(batch_size, model.zdim, dtype=torch.float64, device=device) * 2.0
    z = z_init.clone().detach()
    
    # Langevin采样
    for step in range(e_l_steps):
        z_opt = z.clone().detach().requires_grad_(True)
        z_c = torch.cat((z_opt, ftraj.detach()), dim=1)
        neg_energy = model.EBM(z_c)
        energy = -neg_energy.logsumexp(dim=1)
        
        z_grad = torch.autograd.grad(energy.sum(), z_opt, create_graph=False)[0]
        
        z = z - 0.5 * e_l_step_size * e_l_step_size * (
            z_grad.detach() + 1.0 / (e_prior_sig * e_prior_sig) * z
        )
        z = z + e_l_step_size * torch.randn_like(z)
    
    # 生成子目标位置
    with torch.no_grad():
        z_concat = torch.cat((ftraj, z.detach()), dim=1)
        dest_x = model.decoder_x(z_concat)
        dest_y = model.decoder_y(z_concat)
        dest_z = model.decoder_z(z_concat)
        
        num_subgoals = dest_x.shape[1]
        generated_dest = torch.zeros(batch_size, num_subgoals*3, dtype=torch.float64, device=device)
        for i in range(num_subgoals):
            generated_dest[:, i*3] = dest_x[:, i]
            generated_dest[:, i*3+1] = dest_y[:, i]
            generated_dest[:, i*3+2] = dest_z[:, i]
        
        # 进行轨迹预测
        predicted_future = model.predict(past_traj, generated_dest)
        predicted_future = predicted_future.cpu().numpy()
    
    return predicted_future


def visualize_trajectory(hist, future_true, pred_future, sample_idx, output_dir):
    """绘制单个样本的轨迹可视化（包含预测）"""
    
    # 只取前3个维度 (x, y, z)
    if hist.shape[1] > 3:
        hist = hist[:, :3]
    if future_true.shape[1] > 3:
        future_true = future_true[:, :3]
    if pred_future.shape[1] > 3:
        pred_future = pred_future.reshape(-1, 3)[:, :3]
    else:
        pred_future = pred_future.reshape(-1, 3)
    
    # 创建图表
    fig = plt.figure(figsize=(18, 6))
    
    # 3D 视图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(hist[:, 0], hist[:, 1], hist[:, 2], 'b-o', label='历史轨迹', 
             markersize=6, linewidth=2.5, alpha=0.8)
    ax1.plot(future_true[:, 0], future_true[:, 1], future_true[:, 2], 'g-s', 
             label='真实未来', markersize=6, linewidth=2.5, alpha=0.8)
    ax1.plot(pred_future[:, 0], pred_future[:, 1], pred_future[:, 2], 'r--^', 
             label='预测未来', markersize=6, linewidth=2, alpha=0.8)
    
    # 标记关键点
    ax1.scatter(hist[0, 0], hist[0, 1], hist[0, 2], color='blue', s=100, marker='*', zorder=5)
    ax1.scatter(hist[-1, 0], hist[-1, 1], hist[-1, 2], color='cyan', s=100, marker='D', zorder=5)
    ax1.scatter(future_true[-1, 0], future_true[-1, 1], future_true[-1, 2], color='green', s=100, marker='*', zorder=5)
    ax1.scatter(pred_future[-1, 0], pred_future[-1, 1], pred_future[-1, 2], color='red', s=100, marker='*', zorder=5)
    
    ax1.set_xlabel('X (m)', fontweight='bold')
    ax1.set_ylabel('Y (m)', fontweight='bold')
    ax1.set_zlabel('Z (m)', fontweight='bold')
    ax1.set_title('3D 轨迹对比', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # XY 平面
    ax2 = fig.add_subplot(132)
    ax2.plot(hist[:, 0], hist[:, 1], 'b-o', label='历史轨迹', markersize=5, linewidth=2.5)
    ax2.plot(future_true[:, 0], future_true[:, 1], 'g-s', label='真实未来', markersize=5, linewidth=2.5)
    ax2.plot(pred_future[:, 0], pred_future[:, 1], 'r--^', label='预测未来', markersize=5, linewidth=2)
    ax2.scatter(hist[0, 0], hist[0, 1], color='blue', s=80, marker='*', zorder=5)
    ax2.scatter(hist[-1, 0], hist[-1, 1], color='cyan', s=80, marker='D', zorder=5)
    ax2.scatter(future_true[-1, 0], future_true[-1, 1], color='green', s=80, marker='*', zorder=5)
    ax2.scatter(pred_future[-1, 0], pred_future[-1, 1], color='red', s=80, marker='*', zorder=5)
    ax2.set_xlabel('X (m)', fontweight='bold')
    ax2.set_ylabel('Y (m)', fontweight='bold')
    ax2.set_title('XY 平面视图', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # XZ 平面
    ax3 = fig.add_subplot(133)
    ax3.plot(hist[:, 0], hist[:, 2], 'b-o', label='历史轨迹', markersize=5, linewidth=2.5)
    ax3.plot(future_true[:, 0], future_true[:, 2], 'g-s', label='真实未来', markersize=5, linewidth=2.5)
    ax3.plot(pred_future[:, 0], pred_future[:, 2], 'r--^', label='预测未来', markersize=5, linewidth=2)
    ax3.scatter(hist[0, 0], hist[0, 2], color='blue', s=80, marker='*', zorder=5)
    ax3.scatter(hist[-1, 0], hist[-1, 2], color='cyan', s=80, marker='D', zorder=5)
    ax3.scatter(future_true[-1, 0], future_true[-1, 2], color='green', s=80, marker='*', zorder=5)
    ax3.scatter(pred_future[-1, 0], pred_future[-1, 2], color='red', s=80, marker='*', zorder=5)
    ax3.set_xlabel('X (m)', fontweight='bold')
    ax3.set_ylabel('Z (m)', fontweight='bold')
    ax3.set_title('XZ 平面视图', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 计算误差指标
    errors = np.linalg.norm(pred_future - future_true, axis=1)
    ade = np.mean(errors)
    fde = np.linalg.norm(pred_future[-1] - future_true[-1])
    
    plt.suptitle(f'轨迹推理可视化 - 样本 #{sample_idx:03d} | ADE={ade:.4f}m | FDE={fde:.4f}m', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, f'sample_{sample_idx:03d}.png')
    fig.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return ade, fde


def load_model(model_path, device):
    """加载已训练的模型"""
    print(f"\n加载模型: {model_path}")
    
    # 导入lbebm3D模块
    sys.path.insert(0, os.path.dirname(model_path))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lbebm3D_module",
        os.path.join(os.path.dirname(model_path), 'lbebm3D.py')
    )
    lbebm3D_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lbebm3D_module)
    
    # 加载checkpoint获取模型参数
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # 从state_dict推断模型参数
    def extract_hidden_sizes(module_name):
        hidden = []
        i = 0
        while f'{module_name}.layers.{i}.weight' in state_dict:
            w = state_dict[f'{module_name}.layers.{i}.weight']
            hidden.append(w.shape[0])
            i += 1
        if len(hidden) > 1:
            return hidden[:-1]
        return hidden
    
    enc_past = extract_hidden_sizes('encoder_past') or [512, 256]
    enc_dest = extract_hidden_sizes('encoder_dest') or [256, 128]
    enc_latent = extract_hidden_sizes('encoder_latent') or [256, 512]
    dec = extract_hidden_sizes('decoder_z') or [1024, 512, 1024]
    pred = extract_hidden_sizes('predictor_x') or [1024, 512, 256]
    
    # 推断fdim和zdim
    fdim = state_dict[f'encoder_past.layers.{len(enc_past)}.weight'].shape[0]
    zdim = state_dict[f'encoder_latent.layers.{len(enc_latent)}.weight'].shape[0] // 2
    
    # 创建模型
    model = lbebm3D_module.LBEBM3D(
        enc_past_size=enc_past,
        enc_dest_size=enc_dest,
        enc_latent_size=enc_latent,
        dec_size=dec,
        predictor_size=pred,
        fdim=fdim,
        zdim=zdim,
        sigma=1.3,
        past_length=20,
        future_length=10
    )
    
    model = model.double()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功")
    return model


def main():
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 规范化路径
    args.model_path = os.path.abspath(args.model_path)
    args.csv_dir = os.path.abspath(args.csv_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("从CSV文件进行轨迹推理和可视化")
    print("=" * 70)
    print(f"CSV目录: {args.csv_dir}")
    print(f"样本数: {args.num_samples}")
    print(f"设备: {device}")
    print()
    
    # [1/3] 加载CSV轨迹
    print("[1/3] 加载CSV轨迹数据...")
    samples = load_all_trajectories(args.csv_dir, args.obs_length, args.pred_length, args.num_samples)
    
    if not samples:
        print("❌ 没有找到有效的轨迹样本！")
        return
    
    # [2/3] 加载模型
    print("\n[2/3] 加载模型...")
    model = load_model(args.model_path, device)
    
    # [3/3] 推理和可视化
    print("\n[3/3] 进行推理和可视化...")
    all_ades = []
    all_fdes = []
    
    for idx, (hist, future_true) in enumerate(samples):
        try:
            # 将历史轨迹展平为模型输入格式
            hist_flat = hist.reshape(-1)
            
            # 推理
            pred_future = infer_with_model(model, hist, device)
            
            # 可视化
            ade, fde = visualize_trajectory(hist, future_true, pred_future, idx, args.output_dir)
            all_ades.append(ade)
            all_fdes.append(fde)
            
            print(f"  ✓ 样本 #{idx:03d}: ADE={ade:.6f}m, FDE={fde:.6f}m")
        
        except Exception as e:
            print(f"  ✗ 样本 #{idx:03d} 失败: {e}")
    
    # 打印统计信息
    if all_ades:
        print("\n" + "=" * 70)
        print("推理和可视化完成!")
        print("=" * 70)
        print(f"平均 ADE: {np.mean(all_ades):.6f} m")
        print(f"平均 FDE: {np.mean(all_fdes):.6f} m")
        print(f"已处理样本: {len(all_ades)}")
        print(f"结果保存到: {args.output_dir}")
        print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
