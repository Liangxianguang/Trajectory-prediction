#!/usr/bin/env python3
"""
详细诊断脚本 - 逐步追踪模型输出
"""

import numpy as np
import torch
from pathlib import Path
from train_swarm_gnn import (
    DynamicGraphSwarmGRUModel,
    compute_multi_scale_velocity,
    compute_curvature,
    compute_plane_curvatures,
)

def detailed_diagnosis():
    """详细诊断推理过程"""
    
    model_path = 'gru_models_subset_nogcn1/best_model_agents_3.pt'
    data_dir = 'swarm_segments'
    num_agents = 3
    use_gcn = 0
    sample_idx = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    # ============= 加载数据 =============
    data_path = Path(data_dir)
    X = np.load(data_path / f'input_agents_{num_agents}.npz')['data']
    Y = np.load(data_path / f'output_agents_{num_agents}.npz')['data']
    
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    x_sample = X[sample_idx]  # (20, 3, 3)
    y_sample = Y[sample_idx]  # (10, 3, 3)
    
    print(f"样本 {sample_idx}:")
    print(f"  输入序列 X[{sample_idx}]: shape={x_sample.shape}")
    print(f"  输出序列 Y[{sample_idx}]: shape={y_sample.shape}")
    
    # ============= 加载模型 =============
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    stats = checkpoint.get('stats', {})
    
    output_mean = np.array(stats.get('output_mean', 0.0))
    output_std = np.array(stats.get('output_std', 1.0))
    input_mean_all = stats.get('input_mean_all')
    input_std_all = stats.get('input_std_all')
    
    print(f"\n统计量:")
    print(f"  output_mean: {output_mean}")
    print(f"  output_std:  {output_std}")
    
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
    
    # ============= 计算特征和推理 =============
    vel = compute_multi_scale_velocity(x_sample, dt=0.1)
    curv_3d = compute_curvature(x_sample, dt=0.1)
    curv_plane = compute_plane_curvatures(x_sample, dt=0.1)
    
    features = np.concatenate([x_sample, vel, curv_3d, curv_plane], axis=-1)
    features = (features - input_mean_all) / (input_std_all + 1e-8)
    features = np.clip(features, -5, 5)
    
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    x_tensor = torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)
    
    # ============= 推理 =============
    with torch.no_grad():
        pred_norm = model(features_tensor, x_tensor, teacher_forcing_ratio=0.0)
    
    pred_norm = pred_norm.cpu().numpy()  # (1, 10, 3, 3)
    
    print(f"\n" + "="*100)
    print("模型输出分析")
    print("="*100)
    
    print(f"\n模型输出 (pred_norm) 的统计信息:")
    print(f"  形状: {pred_norm.shape}")
    print(f"  均值: {pred_norm.mean():.6f}")
    print(f"  标准差: {pred_norm.std():.6f}")
    print(f"  最小值: {pred_norm.min():.6f}")
    print(f"  最大值: {pred_norm.max():.6f}")
    
    # 看看第一步的模型输出
    print(f"\n【第0步的模型原始输出 (pred_norm)】")
    print(f"  无人机0: {pred_norm[0, 0, 0, :]}")
    print(f"  无人机1: {pred_norm[0, 0, 1, :]}")
    print(f"  无人机2: {pred_norm[0, 0, 2, :]}")
    
    # ============= 各种反归一化尝试 =============
    print(f"\n" + "="*100)
    print("反归一化尝试")
    print("="*100)
    
    x_last = x_sample[-1, :, :]  # (3, 3)
    
    print(f"\n输入序列最后位置 x[-1]:")
    print(f"  无人机0: {x_last[0, :]}")
    print(f"  无人机1: {x_last[1, :]}")
    print(f"  无人机2: {x_last[2, :]}")
    
    # 方案1：当前方式
    print(f"\n【方案1：当前反归一化方式】")
    print(f"  y_delta = pred_norm * output_std + output_mean")
    pred_delta_1 = pred_norm[0] * output_std + output_mean
    pred_abs_1 = x_last + pred_delta_1
    
    print(f"  第0步反归一化后:")
    print(f"    pred_norm[0,0]: {pred_norm[0, 0, 0, :]}")
    print(f"    pred_delta[0,0]: {pred_delta_1[0, 0, :]}")
    print(f"    pred_abs[0,0]: {pred_abs_1[0, 0, :]}")
    print(f"    真实[0]: {y_sample[0, 0, :]}")
    print(f"    误差: {np.abs(pred_abs_1[0, 0, :] - y_sample[0, 0, :])}")
    
    # 方案2：不加output_mean
    print(f"\n【方案2：不加 output_mean】")
    print(f"  y_delta = pred_norm * output_std")
    pred_delta_2 = pred_norm[0] * output_std
    pred_abs_2 = x_last + pred_delta_2
    
    print(f"  第0步反归一化后:")
    print(f"    pred_norm[0,0]: {pred_norm[0, 0, 0, :]}")
    print(f"    pred_delta[0,0]: {pred_delta_2[0, 0, :]}")
    print(f"    pred_abs[0,0]: {pred_abs_2[0, 0, :]}")
    print(f"    真实[0]: {y_sample[0, 0, :]}")
    print(f"    误差: {np.abs(pred_abs_2[0, 0, :] - y_sample[0, 0, :])}")
    
    # 方案3：直接用pred_norm作为位置
    print(f"\n【方案3：pred_norm 直接作为绝对位置】")
    print(f"  y = pred_norm")
    pred_abs_3 = pred_norm[0]
    
    print(f"  第0步:")
    print(f"    pred_abs[0,0]: {pred_abs_3[0, 0, :]}")
    print(f"    真实[0]: {y_sample[0, 0, :]}")
    print(f"    误差: {np.abs(pred_abs_3[0, 0, :] - y_sample[0, 0, :])}")
    
    # 方案4：pred_norm 反归一化后直接用作位置（不加x_last）
    print(f"\n【方案4：反归一化后直接用作位置（不加x_last）】")
    print(f"  y = pred_norm * output_std + output_mean")
    pred_abs_4 = pred_norm[0] * output_std + output_mean
    
    print(f"  第0步:")
    print(f"    pred_norm[0,0]: {pred_norm[0, 0, 0, :]}")
    print(f"    pred_abs[0,0]: {pred_abs_4[0, 0, :]}")
    print(f"    真实[0]: {y_sample[0, 0, :]}")
    print(f"    误差: {np.abs(pred_abs_4[0, 0, :] - y_sample[0, 0, :])}")
    
    # ============= 训练时的真实标签 =============
    print(f"\n" + "="*100)
    print("查看训练时的真实标签是如何定义的")
    print("="*100)
    
    y_delta_true = y_sample - x_last  # (10, 3, 3)
    print(f"\n真实增量 (Y - X[-1]):")
    print(f"  第0步增量 (无人机0): {y_delta_true[0, 0, :]}")
    print(f"  第0步增量 (无人机1): {y_delta_true[0, 1, :]}")
    print(f"  第0步增量 (无人机2): {y_delta_true[0, 2, :]}")
    
    # 看看增量的统计
    print(f"\n增量的统计信息:")
    print(f"  均值: {y_delta_true.mean():.6f}")
    print(f"  标准差: {y_delta_true.std():.6f}")
    print(f"  最小值: {y_delta_true.min():.6f}")
    print(f"  最大值: {y_delta_true.max():.6f}")
    
    # 对增量进行归一化
    y_target_true = (y_delta_true - output_mean) / (output_std + 1e-8)
    print(f"\n归一化后的真实目标 ((Y-X[-1] - output_mean) / output_std):")
    print(f"  第0步 (无人机0): {y_target_true[0, 0, :]}")
    print(f"  第0步 (无人机1): {y_target_true[0, 1, :]}")
    print(f"  第0步 (无人机2): {y_target_true[0, 2, :]}")
    
    print(f"\n对比模型输出:")
    print(f"  模型输出第0步 (无人机0): {pred_norm[0, 0, 0, :]}")
    print(f"  模型输出第0步 (无人机1): {pred_norm[0, 0, 1, :]}")
    print(f"  模型输出第0步 (无人机2): {pred_norm[0, 0, 2, :]}")
    
    mae_against_target = np.mean(np.abs(pred_norm[0] - y_target_true))
    print(f"\n  MAE (pred_norm vs y_target_true): {mae_against_target:.6f}")
    
    # ============= 最佳方案选择 =============
    print(f"\n" + "="*100)
    print("总结：哪个方案最好？")
    print("="*100)
    
    mae_1 = np.mean(np.abs(pred_abs_1 - y_sample))
    mae_2 = np.mean(np.abs(pred_abs_2 - y_sample))
    mae_3 = np.mean(np.abs(pred_abs_3 - y_sample))
    mae_4 = np.mean(np.abs(pred_abs_4 - y_sample))
    
    print(f"\n方案1 (当前): y = x[-1] + pred_norm*std + mean, MAE = {mae_1:.6f}")
    print(f"方案2: y = x[-1] + pred_norm*std, MAE = {mae_2:.6f}")
    print(f"方案3: y = pred_norm, MAE = {mae_3:.6f}")
    print(f"方案4: y = pred_norm*std + mean, MAE = {mae_4:.6f}")
    
    best_mae = min(mae_1, mae_2, mae_3, mae_4)
    best_scheme = None
    
    if best_mae == mae_1:
        best_scheme = 1
        print(f"\n✓ 最佳方案：方案1 (当前实现)")
    elif best_mae == mae_2:
        best_scheme = 2
        print(f"\n✓ 最佳方案：方案2 (移除 output_mean)")
    elif best_mae == mae_3:
        best_scheme = 3
        print(f"\n✓ 最佳方案：方案3 (直接用 pred_norm 作为位置)")
    else:
        best_scheme = 4
        print(f"\n✓ 最佳方案：方案4 (反归一化但不加 x_last)")


if __name__ == '__main__':
    detailed_diagnosis()
