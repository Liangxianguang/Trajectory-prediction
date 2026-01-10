#!/usr/bin/env python3
"""
诊断推理脚本 - 逐点输出位置信息，找出问题所在
"""

import numpy as np
import torch
from pathlib import Path
import argparse
from train_swarm_gnn import (
    DynamicGraphSwarmGRUModel,
    compute_multi_scale_velocity,
    compute_curvature,
    compute_plane_curvatures,
)

def diagnose_single_sample():
    """诊断单个样本的推理过程"""
    
    # 参数设置
    model_path = 'gru_models_subset_nogcn1/best_model_agents_3.pt'
    data_dir = 'swarm_segments'
    num_agents = 3
    use_gcn = 0
    sample_idx = 0  # 只看第一个样本
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    # ============= 加载数据 =============
    print("="*100)
    print("阶段 1: 加载数据")
    print("="*100)
    
    data_path = Path(data_dir)
    X = np.load(data_path / f'input_agents_{num_agents}.npz')['data']
    Y = np.load(data_path / f'output_agents_{num_agents}.npz')['data']
    
    # 转置为 (samples, seq, agents, 3)
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    print(f"数据加载完成:")
    print(f"  X 形状: {X.shape}")
    print(f"  Y 形状: {Y.shape}")
    
    x_sample = X[sample_idx]  # (20, 3, 3)
    y_sample = Y[sample_idx]  # (10, 3, 3)
    
    print(f"\n样本 {sample_idx}:")
    print(f"  输入序列形状: {x_sample.shape}")
    print(f"  输出序列形状: {y_sample.shape}")
    
    # ============= 输出输入序列的每个点 =============
    print("\n" + "="*100)
    print("阶段 2: 输入序列位置详情")
    print("="*100)
    print("\n【输入序列 X (20 步)】")
    print("时步 | 无人机0位置          | 无人机1位置          | 无人机2位置")
    print("-" * 100)
    
    for t in range(x_sample.shape[0]):
        pos0 = x_sample[t, 0, :]
        pos1 = x_sample[t, 1, :]
        pos2 = x_sample[t, 2, :]
        print(f"{t:2d}   | [{pos0[0]:7.3f}, {pos0[1]:7.3f}, {pos0[2]:7.3f}] | "
              f"[{pos1[0]:7.3f}, {pos1[1]:7.3f}, {pos1[2]:7.3f}] | "
              f"[{pos2[0]:7.3f}, {pos2[1]:7.3f}, {pos2[2]:7.3f}]")
    
    # 输入序列的最后一个点
    print(f"\n✓ 输入序列最后一个点 (x[-1]):")
    print(f"  无人机0: {x_sample[-1, 0, :]}")
    print(f"  无人机1: {x_sample[-1, 1, :]}")
    print(f"  无人机2: {x_sample[-1, 2, :]}")
    
    # ============= 输出真实输出序列的每个点 =============
    print("\n" + "="*100)
    print("阶段 3: 真实输出序列位置详情")
    print("="*100)
    print("\n【真实输出序列 Y (10 步)】")
    print("时步 | 无人机0位置          | 无人机1位置          | 无人机2位置")
    print("-" * 100)
    
    for t in range(y_sample.shape[0]):
        pos0 = y_sample[t, 0, :]
        pos1 = y_sample[t, 1, :]
        pos2 = y_sample[t, 2, :]
        print(f"{t:2d}   | [{pos0[0]:7.3f}, {pos0[1]:7.3f}, {pos0[2]:7.3f}] | "
              f"[{pos1[0]:7.3f}, {pos1[1]:7.3f}, {pos1[2]:7.3f}] | "
              f"[{pos2[0]:7.3f}, {pos2[1]:7.3f}, {pos2[2]:7.3f}]")
    
    # 计算增量
    print(f"\n✓ 真实增量 (Y 相对于 X[-1]) - 第一步:")
    print(f"  无人机0: Y[0] - X[-1] = {y_sample[0, 0, :] - x_sample[-1, 0, :]}")
    print(f"  无人机1: Y[0] - X[-1] = {y_sample[0, 1, :] - x_sample[-1, 1, :]}")
    print(f"  无人机2: Y[0] - X[-1] = {y_sample[0, 2, :] - x_sample[-1, 2, :]}")
    
    # ============= 加载模型 =============
    print("\n" + "="*100)
    print("阶段 4: 加载模型并推理")
    print("="*100)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    stats = checkpoint.get('stats', {})
    
    print(f"模型配置:")
    print(f"  hidden_size: {config.get('hidden_size')}")
    print(f"  num_layers: {config.get('num_layers')}")
    print(f"  dropout: {config.get('dropout')}")
    print(f"  use_gcn: {use_gcn}")
    
    # 反归一化参数
    output_mean = np.array(stats.get('output_mean', 0.0))
    output_std = np.array(stats.get('output_std', 1.0))
    input_mean_all = stats.get('input_mean_all')
    input_std_all = stats.get('input_std_all')
    input_mean = stats.get('input_mean')
    input_std = stats.get('input_std')
    
    print(f"\n统计量:")
    print(f"  output_mean: {output_mean}")
    print(f"  output_std: {output_std}")
    print(f"  input_mean: {input_mean}")
    print(f"  input_std: {input_std}")
    
    model = DynamicGraphSwarmGRUModel(
        input_size=16,
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        num_agents=num_agents,
        output_size=3,
        dropout=config.get('dropout', 0.2),
        use_gcn=bool(use_gcn)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载完成")
    
    # ============= 计算特征 =============
    print("\n" + "="*100)
    print("阶段 5: 特征计算")
    print("="*100)
    
    vel = compute_multi_scale_velocity(x_sample, dt=0.1)
    curv_3d = compute_curvature(x_sample, dt=0.1)
    curv_plane = compute_plane_curvatures(x_sample, dt=0.1)
    
    features = np.concatenate([x_sample, vel, curv_3d, curv_plane], axis=-1)
    print(f"特征维度: {features.shape}")
    print(f"特征: {features.shape} = 位置(3) + 多尺度速度(9) + 3D曲率(1) + 平面曲率(3)")
    
    # 归一化特征
    features = (features - input_mean_all) / (input_std_all + 1e-8)
    features = np.clip(features, -5, 5)
    
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    x_tensor = torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)
    
    print(f"✓ 特征已准备: shape={features_tensor.shape}")
    
    # ============= 模型推理 =============
    print("\n" + "="*100)
    print("阶段 6: 模型推理")
    print("="*100)
    
    with torch.no_grad():
        pred_norm = model(features_tensor, x_tensor, teacher_forcing_ratio=0.0)
    
    pred_norm = pred_norm.cpu().numpy()  # (1, 10, 3, 3)
    print(f"模型输出（归一化增量）:")
    print(f"  形状: {pred_norm.shape}")
    print(f"  取值范围: [{pred_norm.min():.6f}, {pred_norm.max():.6f}]")
    
    # ============= 反归一化过程详解 =============
    print("\n" + "="*100)
    print("阶段 7: 反归一化过程详解")
    print("="*100)
    
    print(f"\n反归一化公式:")
    print(f"  y_norm (训练时): (y_delta - output_mean) / output_std")
    print(f"  y_delta (反解):  y_norm * output_std + output_mean")
    print(f"  y_abs (绝对):    y_delta + x[-1]")
    
    print(f"\n数值:")
    print(f"  output_mean: {output_mean}")
    print(f"  output_std:  {output_std}")
    
    # 反归一化第一步的无人机0
    agent_id = 0
    step = 0
    
    print(f"\n【示例：第{step}步，无人机{agent_id}】")
    
    y_norm_val = pred_norm[0, step, agent_id, :]
    print(f"  y_norm (模型输出): {y_norm_val}")
    
    y_delta = y_norm_val * output_std + output_mean
    print(f"  y_delta (反归一化): {y_norm_val} * {output_std} + {output_mean}")
    print(f"              = {y_delta}")
    
    x_last = x_sample[-1, agent_id, :]
    print(f"  x[-1] (输入最后位置): {x_last}")
    
    y_pred_abs = x_last + y_delta
    print(f"  y_pred (绝对位置): {x_last} + {y_delta}")
    print(f"                = {y_pred_abs}")
    
    y_true_abs = y_sample[step, agent_id, :]
    print(f"  y_true (真实位置): {y_true_abs}")
    
    error = np.abs(y_pred_abs - y_true_abs)
    print(f"  误差: {error}")
    print(f"  MAE: {np.mean(error):.6f}")
    
    # ============= 完整输出 =============
    print("\n" + "="*100)
    print("阶段 8: 完整预测序列 vs 真实序列")
    print("="*100)
    
    pred_delta = pred_norm[0] * output_std + output_mean  # (10, 3, 3)
    pred_abs = x_sample[-1:, :, :] + pred_delta  # (10, 3, 3)
    
    print("\n【对比表：预测 vs 真实】")
    print("\n无人机 0:")
    print("时步 | 预测位置                  | 真实位置                  | 误差                    ")
    print("-" * 110)
    
    for t in range(y_sample.shape[0]):
        pred = pred_abs[t, 0, :]
        true = y_sample[t, 0, :]
        err = np.abs(pred - true)
        print(f"{t:2d}   | [{pred[0]:7.3f}, {pred[1]:7.3f}, {pred[2]:7.3f}] | "
              f"[{true[0]:7.3f}, {true[1]:7.3f}, {true[2]:7.3f}] | "
              f"[{err[0]:6.4f}, {err[1]:6.4f}, {err[2]:6.4f}]")
    
    print("\n无人机 1:")
    print("时步 | 预测位置                  | 真实位置                  | 误差                    ")
    print("-" * 110)
    
    for t in range(y_sample.shape[0]):
        pred = pred_abs[t, 1, :]
        true = y_sample[t, 1, :]
        err = np.abs(pred - true)
        print(f"{t:2d}   | [{pred[0]:7.3f}, {pred[1]:7.3f}, {pred[2]:7.3f}] | "
              f"[{true[0]:7.3f}, {true[1]:7.3f}, {true[2]:7.3f}] | "
              f"[{err[0]:6.4f}, {err[1]:6.4f}, {err[2]:6.4f}]")
    
    print("\n无人机 2:")
    print("时步 | 预测位置                  | 真实位置                  | 误差                    ")
    print("-" * 110)
    
    for t in range(y_sample.shape[0]):
        pred = pred_abs[t, 2, :]
        true = y_sample[t, 2, :]
        err = np.abs(pred - true)
        print(f"{t:2d}   | [{pred[0]:7.3f}, {pred[1]:7.3f}, {pred[2]:7.3f}] | "
              f"[{true[0]:7.3f}, {true[1]:7.3f}, {true[2]:7.3f}] | "
              f"[{err[0]:6.4f}, {err[1]:6.4f}, {err[2]:6.4f}]")
    
    # ============= 关键检查 =============
    print("\n" + "="*100)
    print("阶段 9: 关键检查")
    print("="*100)
    
    # 检查第一步是否从输入最后一个点开始
    first_step_pred = pred_abs[0, :, :]
    last_step_input = x_sample[-1, :, :]
    
    print(f"\n✓ 第一步预测是否从输入最后一个点开始?")
    print(f"  预测第0步位置: {first_step_pred}")
    print(f"  输入第19步位置: {last_step_input}")
    print(f"  差距: {np.abs(first_step_pred - last_step_input)}")
    
    if np.allclose(first_step_pred, last_step_input, atol=0.01):
        print(f"  ✓✓✓ 正确! 预测的第一步从输入序列最后一个点开始")
    else:
        print(f"  ✗✗✗ 错误! 预测的第一步与输入序列最后一个点不匹配")
        print(f"       可能原因:")
        print(f"       1. 模型输出的是完整位置而非增量")
        print(f"       2. 反归一化公式错误")
        print(f"       3. output_mean/output_std 错误")


if __name__ == '__main__':
    diagnose_single_sample()
