#!/usr/bin/env python3
"""
验证预测轨迹的连续性问题
===============================

问题诊断：
- 预测的起始位置不在输入序列的最后一个点
- 预测值和真实值的起始点存在跳跃

本脚本用于：
1. 精确诊断位置连续性问题
2. 对比三种重建方法的效果
3. 验证数学公式是否正确
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from train_swarm_gnn import (
    DynamicGraphSwarmGRUModel,
    SwarmTrajectoryDatasetGNN,
)
from infer_gnn_model import (
    load_test_data,
    reconstruct_positions_simple,
    reconstruct_positions_physics_constrained,
    reconstruct_positions_trajectory_smoothing,
    infer_batch,
)


def diagnose_continuity_issue(model, features, x_orig, y_orig, output_std, output_mean, device):
    """
    诊断单个样本的位置连续性问题
    
    Args:
        model: 模型
        features: (batch, seq_in, agents, 16)
        x_orig: (batch, seq_in, agents, 3) 输入序列
        y_orig: (batch, seq_out, agents, 3) 真实输出
        output_std: 反归一化因子
        output_mean: 反归一化均值
        device: 设备
    
    Returns:
        诊断报告字典
    """
    batch_size = x_orig.shape[0]
    
    # 获取最后一个输入位置
    last_input_pos = x_orig[0, -1, :, :]  # (agents, 3)
    
    # 获取第一个真实输出位置
    first_target_pos = y_orig[0, 0, :, :]  # (agents, 3)
    
    # 用三种方法推理
    results = {}
    
    model.eval()
    with torch.no_grad():
        features_t = torch.tensor(features, device=device, dtype=torch.float32)
        x_orig_t = torch.tensor(x_orig, device=device, dtype=torch.float32)
        
        # 推理得到归一化增量
        pred_norm = model(features_t, x_orig_t, teacher_forcing_ratio=0.0)
        
        output_std_t = torch.tensor(output_std, dtype=torch.float32, device=device)
        output_mean_t = torch.tensor(output_mean, dtype=torch.float32, device=device)
        
        pred_delta = pred_norm * output_std_t + output_mean_t
        pred_delta_np = pred_delta.cpu().numpy()  # (batch, seq_out, agents, 3)
        
        # 方法1: 直接方法
        x_orig_np = x_orig_t.cpu().numpy()
        last_pos = x_orig_np[:, -1:, :, :]
        pred_direct = last_pos + pred_delta_np
        results['direct'] = pred_direct[0, 0, :, :]  # 第一时步的预测
        
        # 方法2: 简单积分 + 速度约束
        pred_simple = reconstruct_positions_simple(pred_delta_np, x_orig_np, dt=0.1)
        results['simple'] = pred_simple[0, 0, :, :]
        
        # 方法3: 物理约束
        pred_physics = reconstruct_positions_physics_constrained(pred_delta_np, x_orig_np, dt=0.1)
        pred_physics_smooth = reconstruct_positions_trajectory_smoothing(pred_physics, window_size=3)
        results['physics'] = pred_physics_smooth[0, 0, :, :]
    
    # 计算连续性偏差 (相对于最后一个输入位置)
    report = {
        'last_input_pos': last_input_pos,
        'first_target_pos': first_target_pos,
        'pred_results': results,
        'continuity_errors': {},
        'target_error': {},
    }
    
    for method, pred_pos in results.items():
        error = np.linalg.norm(pred_pos - last_input_pos, axis=1)  # 每个agent的误差
        report['continuity_errors'][method] = {
            'per_agent': error.tolist(),
            'mean': float(np.mean(error)),
            'max': float(np.max(error)),
        }
    
    # 相对于真实值的误差
    for method, pred_pos in results.items():
        error = np.linalg.norm(pred_pos - first_target_pos, axis=1)
        report['target_error'][method] = {
            'per_agent': error.tolist(),
            'mean': float(np.mean(error)),
        }
    
    return report


def run_comprehensive_test(model_path, data_dir, num_agents, num_samples=10):
    """
    运行完整的连续性测试
    
    Args:
        model_path: 模型路径
        data_dir: 数据目录
        num_agents: 无人机数量
        num_samples: 测试样本数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    stats = checkpoint.get('stats', {})
    
    model = DynamicGraphSwarmGRUModel(
        input_size=16,
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        num_agents=num_agents,
        output_size=3,
        dropout=config.get('dropout', 0.2),
        use_gcn=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    output_std = np.array(stats.get('output_std', 1.0))
    output_mean = np.array(stats.get('output_mean', 0.0))
    
    logger.info(f"✓ 模型加载完成")
    
    # 加载数据
    logger.info(f"加载数据...")
    X_all, Y_all = load_test_data(data_dir, num_agents, num_samples=num_samples)
    
    logger.info(f"✓ 数据加载完成: {len(X_all)} 样本")
    
    # 创建特征
    dataset = SwarmTrajectoryDatasetGNN(
        X_all, Y_all,
        input_mean=stats.get('input_mean'),
        input_std=stats.get('input_std'),
        output_mean=stats.get('output_mean'),
        output_std=stats.get('output_std'),
        feature_mean=stats.get('input_mean_all'),
        feature_std=stats.get('input_std_all'),
    )
    
    # 诊断每个样本
    all_reports = []
    
    logger.info(f"\n开始诊断 {num_samples} 个样本的连续性...")
    logger.info("="*80)
    
    for idx in tqdm(range(len(dataset)), desc="诊断进度"):
        features, x_orig, y_norm, y_orig = dataset[idx]
        
        # 转换为 numpy 并添加 batch 维度
        features = features.unsqueeze(0).numpy()
        x_orig = x_orig.unsqueeze(0).numpy()
        y_orig = y_orig.unsqueeze(0).numpy()
        
        report = diagnose_continuity_issue(
            model, features, x_orig, y_orig, 
            output_std, output_mean, device
        )
        report['sample_idx'] = idx
        all_reports.append(report)
    
    # 汇总统计
    logger.info("\n" + "="*80)
    logger.info("诊断结果汇总")
    logger.info("="*80)
    
    summary = {
        'num_samples': num_samples,
        'num_agents': num_agents,
        'continuity_stats': {
            'direct': {'mean': [], 'max': []},
            'simple': {'mean': [], 'max': []},
            'physics': {'mean': [], 'max': []},
        },
        'target_error_stats': {
            'direct': {'mean': []},
            'simple': {'mean': []},
            'physics': {'mean': []},
        },
    }
    
    # 收集统计数据
    for report in all_reports:
        for method in ['direct', 'simple', 'physics']:
            if method in report['continuity_errors']:
                summary['continuity_stats'][method]['mean'].append(
                    report['continuity_errors'][method]['mean']
                )
                summary['continuity_stats'][method]['max'].append(
                    report['continuity_errors'][method]['max']
                )
            if method in report['target_error']:
                summary['target_error_stats'][method]['mean'].append(
                    report['target_error'][method]['mean']
                )
    
    # 计算平均值
    report_text = "\n【位置连续性错误】 (预测起始点与输入最后一点的偏差)\n"
    report_text += "="*80 + "\n"
    
    for method in ['direct', 'simple', 'physics']:
        means = summary['continuity_stats'][method]['mean']
        maxs = summary['continuity_stats'][method]['max']
        
        if means:
            avg_mean = np.mean(means)
            avg_max = np.mean(maxs)
            min_mean = np.min(means)
            max_mean = np.max(means)
            
            report_text += f"\n{method.upper()} 方法:\n"
            report_text += f"  平均偏差 (均值):      {avg_mean:.6f} m\n"
            report_text += f"  平均偏差 (最大值):    {avg_max:.6f} m\n"
            report_text += f"  偏差范围:            {min_mean:.6f} - {max_mean:.6f} m\n"
    
    # 相对于真实值的误差
    report_text += "\n【相对于真实值的误差】(预测与真实的起始点差距)\n"
    report_text += "="*80 + "\n"
    
    for method in ['direct', 'simple', 'physics']:
        means = summary['target_error_stats'][method]['mean']
        
        if means:
            avg_mean = np.mean(means)
            min_mean = np.min(means)
            max_mean = np.max(means)
            
            report_text += f"\n{method.upper()} 方法:\n"
            report_text += f"  平均误差:            {avg_mean:.6f} m\n"
            report_text += f"  误差范围:            {min_mean:.6f} - {max_mean:.6f} m\n"
    
    # 详细样本报告
    report_text += "\n\n【详细样本分析】 (前5个样本)\n"
    report_text += "="*80 + "\n"
    
    for report in all_reports[:5]:
        sample_idx = report['sample_idx']
        report_text += f"\n样本 {sample_idx}:\n"
        report_text += f"  输入最后位置:        {report['last_input_pos'][0]}\n"
        report_text += f"  真实起始位置:        {report['first_target_pos'][0]}\n"
        report_text += f"\n  预测结果对比:\n"
        
        for method in ['direct', 'simple', 'physics']:
            pred_pos = report['pred_results'][method]
            cont_err = report['continuity_errors'][method]
            tgt_err = report['target_error'][method]
            
            report_text += f"    {method.upper()}:\n"
            report_text += f"      预测位置:          {pred_pos[0]}\n"
            report_text += f"      连续性偏差:        {cont_err['mean']:.6f} m\n"
            report_text += f"      相对真实值误差:    {tgt_err['mean']:.6f} m\n"
    
    logger.info(report_text)
    
    # 保存报告
    output_dir = Path('continuity_diagnosis')
    output_dir.mkdir(exist_ok=True)
    
    # 文本报告
    report_file = output_dir / 'continuity_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"\n✓ 文本报告已保存: {report_file}")
    
    # JSON 报告 - 定义序列化函数
    def _make_serializable(obj):
        """递归将 numpy 类型转为 Python 原生类型"""
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_make_serializable(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64, np.int_)):
            return int(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        return obj
    
    json_file = output_dir / 'continuity_diagnosis.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        serializable_obj = {
            'summary': {
                'num_samples': summary['num_samples'],
                'num_agents': summary['num_agents'],
                'continuity_stats': {
                    method: {
                        'avg_mean_error': float(np.mean(summary['continuity_stats'][method]['mean'])) if summary['continuity_stats'][method]['mean'] else 0,
                        'avg_max_error': float(np.mean(summary['continuity_stats'][method]['max'])) if summary['continuity_stats'][method]['max'] else 0,
                    }
                    for method in ['direct', 'simple', 'physics']
                },
            },
            'detailed_reports': all_reports,
        }
        json.dump(_make_serializable(serializable_obj), f, indent=2, ensure_ascii=False)
    logger.info(f"✓ JSON 报告已保存: {json_file}")
    
    return summary, all_reports


def main():
    import argparse
    parser = argparse.ArgumentParser(description='验证预测轨迹连续性')
    parser.add_argument('--model', type=str, 
                       default='gru_models_subset_nogcn1/best_model_agents_3.pt',
                       help='模型路径')
    parser.add_argument('--data_dir', type=str, default='swarm_segments',
                       help='数据目录')
    parser.add_argument('--agents', type=int, default=3,
                       help='无人机数量')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='诊断样本数')
    
    args = parser.parse_args()
    
    summary, reports = run_comprehensive_test(
        args.model, args.data_dir, args.agents, args.num_samples
    )


if __name__ == '__main__':
    main()
