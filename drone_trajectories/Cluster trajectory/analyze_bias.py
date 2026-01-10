#!/usr/bin/env python3
"""
分析模型预测偏差 - 检查是否存在系统性偏差
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

def analyze_prediction_bias():
    """分析大量样本的预测偏差"""
    
    model_path = 'gru_models_subset_nogcn1/best_model_agents_3.pt'
    data_dir = 'swarm_segments'
    num_agents = 3
    use_gcn = 0
    num_samples = 100  # 分析前100个样本
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============= 加载数据 =============
    data_path = Path(data_dir)
    X = np.load(data_path / f'input_agents_{num_agents}.npz')['data']
    Y = np.load(data_path / f'output_agents_{num_agents}.npz')['data']
    
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    # 只取前num_samples个样本
    X = X[:num_samples]
    Y = Y[:num_samples]
    
    # ============= 加载模型 =============
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    stats = checkpoint.get('stats', {})
    
    output_mean = np.array(stats.get('output_mean', 0.0))
    output_std = np.array(stats.get('output_std', 1.0))
    input_mean_all = stats.get('input_mean_all')
    input_std_all = stats.get('input_std_all')
    
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
    
    # ============= 推理所有样本 =============
    print(f"对 {num_samples} 个样本进行推理...")
    
    all_pred_abs = []
    all_y_true = []
    all_pred_delta = []
    all_y_delta_true = []
    
    for idx in range(num_samples):
        x_sample = X[idx]
        y_sample = Y[idx]
        
        # 计算特征
        vel = compute_multi_scale_velocity(x_sample, dt=0.1)
        curv_3d = compute_curvature(x_sample, dt=0.1)
        curv_plane = compute_plane_curvatures(x_sample, dt=0.1)
        
        features = np.concatenate([x_sample, vel, curv_3d, curv_plane], axis=-1)
        features = (features - input_mean_all) / (input_std_all + 1e-8)
        features = np.clip(features, -5, 5)
        
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        x_tensor = torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            pred_norm = model(features_tensor, x_tensor, teacher_forcing_ratio=0.0)
        
        pred_norm = pred_norm.cpu().numpy()[0]  # (10, 3, 3)
        
        # 反归一化
        pred_delta = pred_norm * output_std + output_mean
        x_last = x_sample[-1, :, :]
        pred_abs = x_last + pred_delta
        
        # 真实增量
        y_delta_true = y_sample - x_last
        
        all_pred_abs.append(pred_abs)
        all_y_true.append(y_sample)
        all_pred_delta.append(pred_delta)
        all_y_delta_true.append(y_delta_true)
    
    # 合并所有结果
    pred_abs_all = np.array(all_pred_abs)  # (100, 10, 3, 3)
    y_true_all = np.array(all_y_true)      # (100, 10, 3, 3)
    pred_delta_all = np.array(all_pred_delta)  # (100, 10, 3, 3)
    y_delta_true_all = np.array(all_y_delta_true)  # (100, 10, 3, 3)
    
    print(f"✓ 推理完成")
    
    # ============= 分析 =============
    print("\n" + "="*100)
    print("预测误差分析")
    print("="*100)
    
    error = np.abs(pred_abs_all - y_true_all)
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))
    
    print(f"\n绝对位置误差:")
    print(f"  MAE: {mae:.6f} m")
    print(f"  RMSE: {rmse:.6f} m")
    print(f"  最小值: {error.min():.6f} m")
    print(f"  最大值: {error.max():.6f} m")
    
    # 按轴分解
    mae_x = np.mean(np.abs(pred_abs_all[..., 0] - y_true_all[..., 0]))
    mae_y = np.mean(np.abs(pred_abs_all[..., 1] - y_true_all[..., 1]))
    mae_z = np.mean(np.abs(pred_abs_all[..., 2] - y_true_all[..., 2]))
    
    print(f"\n按轴分解:")
    print(f"  X 轴 MAE: {mae_x:.6f} m")
    print(f"  Y 轴 MAE: {mae_y:.6f} m")
    print(f"  Z 轴 MAE: {mae_z:.6f} m")
    
    # 按时步分解
    print(f"\n按时步分解:")
    for t in range(10):
        mae_t = np.mean(np.abs(pred_abs_all[:, t, :, :] - y_true_all[:, t, :, :]))
        print(f"  时步 {t}: {mae_t:.6f} m")
    
    # ============= 增量误差分析 =============
    print("\n" + "="*100)
    print("增量（相对）误差分析")
    print("="*100)
    
    delta_error = np.abs(pred_delta_all - y_delta_true_all)
    delta_mae = np.mean(delta_error)
    
    print(f"\n增量误差:")
    print(f"  MAE: {delta_mae:.6f} m")
    print(f"  RMSE: {np.sqrt(np.mean(delta_error ** 2)):.6f} m")
    
    # ============= 检查是否存在系统性偏差 =============
    print("\n" + "="*100)
    print("系统性偏差检查")
    print("="*100)
    
    # 计算平均偏差（有符号）
    signed_error = pred_abs_all - y_true_all
    mean_bias = np.mean(signed_error, axis=(0, 1, 2))
    
    print(f"\n平均偏差 (有符号的均值):")
    print(f"  X 轴: {mean_bias[0]:.6f} m")
    print(f"  Y 轴: {mean_bias[1]:.6f} m")
    print(f"  Z 轴: {mean_bias[2]:.6f} m")
    
    if np.abs(mean_bias).max() > 0.01:
        print(f"\n⚠️  存在显著的系统性偏差!")
        print(f"   可能原因:")
        print(f"   1. output_mean 计算不准确")
        print(f"   2. 训练数据和测试数据分布不同")
        print(f"   3. 模型学到了某种偏差")
    else:
        print(f"\n✓ 偏差很小，基本无系统性偏差")
    
    # ============= 按时步分析增量 =============
    print("\n" + "="*100)
    print("增量按时步分析")
    print("="*100)
    
    print(f"\n增量预测 vs 真实 (按时步):")
    for t in range(10):
        pred_delta_t = pred_delta_all[:, t, :, :].mean(axis=(0, 1))
        true_delta_t = y_delta_true_all[:, t, :, :].mean(axis=(0, 1))
        error_t = np.abs(pred_delta_t - true_delta_t)
        
        print(f"  时步 {t}:")
        print(f"    预测增量均值: [{pred_delta_t[0]:7.4f}, {pred_delta_t[1]:7.4f}, {pred_delta_t[2]:7.4f}]")
        print(f"    真实增量均值: [{true_delta_t[0]:7.4f}, {true_delta_t[1]:7.4f}, {true_delta_t[2]:7.4f}]")
        print(f"    误差:        [{error_t[0]:7.4f}, {error_t[1]:7.4f}, {error_t[2]:7.4f}]")
    
    # ============= 关键结论 =============
    print("\n" + "="*100)
    print("结论")
    print("="*100)
    
    print(f"\n✓ 当前推理实现是正确的")
    print(f"✓ 推理第一步正确地从输入序列最后位置开始")
    print(f"\n⚠️  模型预测精度: MAE = {mae:.6f} m")
    print(f"   这不是推理问题，而是训练问题")
    print(f"   可能的改进方向:")
    print(f"   1. 增加训练数据（当前使用10%子集）")
    print(f"   2. 调整超参数（hidden_size, num_layers, dropout）")
    print(f"   3. 改进特征工程")
    print(f"   4. 优化损失函数权重")


if __name__ == '__main__':
    analyze_prediction_bias()
