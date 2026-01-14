#!/usr/bin/env python3
"""
集群轨迹模型 v2 vs v3（含 GNN）训练脚本集成指南

这个文件展示了如何修改 train_swarm_v2_complete.py 来支持 GNN
建议将这些片段 merge 到原脚本中

改动点：
1. 新的命令行参数（--use_gnn, --gnn_hidden, --gnn_heads, --edge_threshold）
2. 模型实例化逻辑
3. 训练循环保持不变
"""

import argparse
import logging
from pathlib import Path
import torch
import numpy as np

# 导入 v2 与 v3 模型
from train_swarm_model_v2_dynamics_aware import (
    DynamicsAwareSwarmGRUModel,
    DynamicsAwareLoss,
    compute_features_enhanced_24d,
)
from train_swarm_model_v3_with_gnn import (
    DynamicsAwareSwarmGRUModel_with_GNN,
    build_adjacency_from_positions
)

logger = logging.getLogger(__name__)


# ====================================================================
# 修改点 1：扩展命令行参数
# ====================================================================

def build_parser_with_gnn():
    """
    扩展原有参数解析器，添加 GNN 相关参数
    
    使用方法：
        # v2 原有模式（无 GNN）
        python train_swarm_v2_complete.py --agents 3 --epochs 200 --batch_size 256
        
        # v3 新模式（含 GNN）
        python train_swarm_v2_complete.py \\
            --agents 3 \\
            --epochs 200 \\
            --batch_size 256 \\
            --use_gnn \\
            --gnn_hidden 64 \\
            --gnn_heads 4 \\
            --edge_threshold 5.0 \\
            --gnn_fusion concat
    """
    parser = argparse.ArgumentParser(description='训练轨迹模型（v2 或 v3+GNN）')
    
    # 原有参数（保持不变）
    parser.add_argument('--data_dir', type=str, default='swarm_segments',
                        help='数据目录')
    parser.add_argument('--agents', type=str, default='3',
                        help='无人机数量')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='GRU 隐层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='GRU 层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 比例')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=25,
                        help='早停耐心值')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.6,
                        help='TF 初始比例')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--output_dir', type=str, default='swarm_models_v2',
                        help='输出目录')
    parser.add_argument('--features_dir', type=str, default=None,
                        help='预计算特征目录')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='使用注意力')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader 工作数')
    parser.add_argument('--use_subset', action='store_true',
                        help='使用子集数据')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定检查点恢复训练')
    parser.add_argument('--no_resume', action='store_true',
                        help='跳过自动恢复')
    
    # ========== 新增 GNN 参数 ==========
    parser.add_argument('--use_gnn', action='store_true',
                        help='启用 GNN 进行代理间交互建模（v3 模型）')
    parser.add_argument('--gnn_hidden', type=int, default=64,
                        help='GNN 隐层维度（仅在 --use_gnn 时生效）')
    parser.add_argument('--gnn_heads', type=int, default=4,
                        help='GAT 多头数（仅在 --use_gnn 时生效）')
    parser.add_argument('--edge_threshold', type=float, default=5.0,
                        help='邻接矩阵距离阈值，单位米（仅在 --use_gnn 时生效）')
    parser.add_argument('--gnn_fusion', type=str, default='concat',
                        choices=['concat', 'gate', 'add'],
                        help='GNN 特征融合模式：concat/gate/add')
    
    return parser


# ====================================================================
# 修改点 2：模型实例化逻辑
# ====================================================================

def create_model(args, device):
    """
    根据参数创建 v2 或 v3 模型
    
    Args:
        args: 命令行参数
        device: PyTorch 设备
    
    Returns:
        model: 初始化的模型实例
    """
    if args.use_gnn:
        logger.info("=" * 70)
        logger.info("创建 v3 模型（含 GNN）")
        logger.info("=" * 70)
        
        model = DynamicsAwareSwarmGRUModel_with_GNN(
            input_size=24,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=3,
            dropout=args.dropout,
            use_attention=args.use_attention,
            gnn_hidden=args.gnn_hidden,
            num_gnn_heads=args.gnn_heads,
            edge_threshold=args.edge_threshold,
            fusion_mode=args.gnn_fusion
        ).to(device)
        
        logger.info(f"✓ GNN 配置：")
        logger.info(f"  ├─ 隐层维度: {args.gnn_hidden}")
        logger.info(f"  ├─ 多头数: {args.gnn_heads}")
        logger.info(f"  ├─ 邻接距离阈值: {args.edge_threshold} m")
        logger.info(f"  └─ 融合模式: {args.gnn_fusion}")
        
    else:
        logger.info("=" * 70)
        logger.info("创建 v2 模型（仅 BiGRU + Attention）")
        logger.info("=" * 70)
        
        model = DynamicsAwareSwarmGRUModel(
            input_size=24,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=3,
            dropout=args.dropout,
            use_attention=args.use_attention
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数量: {num_params:,}")
    
    return model


# ====================================================================
# 修改点 3：检查点命名（支持 v2/v3 区分）
# ====================================================================

def get_checkpoint_paths(output_dir, num_agents, use_gnn=False):
    """
    生成版本相关的检查点路径
    
    Args:
        output_dir: 输出目录
        num_agents: 无人机数量
        use_gnn: 是否使用 GNN
    
    Returns:
        dict: 包含各种检查点路径
    """
    output_path = Path(output_dir)
    
    if use_gnn:
        suffix = f"_agents_{num_agents}_v3"
    else:
        suffix = f"_agents_{num_agents}_v2"
    
    return {
        'best_model': output_path / f"best_model{suffix}.pt",
        'last_checkpoint': output_path / f"last_checkpoint{suffix}.pt",
        'interrupted_checkpoint': output_path / f"interrupted_checkpoint{suffix}.pt",
        'training_history': output_path / f"training_history{suffix}.csv",
        'config': output_path / f"training_config{suffix}.json",
    }


# ====================================================================
# 修改点 4：配置文件保存（记录 GNN 参数）
# ====================================================================

def save_training_config(args, checkpoint_paths, model):
    """
    保存训练配置文件，包含 GNN 参数
    """
    import json
    
    config = {
        'model_version': 'v3_with_gnn' if args.use_gnn else 'v2',
        'architecture': {
            'input_size': 24,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'output_size': 3,
            'dropout': args.dropout,
            'use_attention': args.use_attention,
        },
        'training': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'teacher_forcing_ratio': args.teacher_forcing_ratio,
            'patience': args.patience,
        },
        'data': {
            'agents': args.agents,
            'val_split': args.val_split,
            'use_subset': args.use_subset,
            'use_amp': args.use_amp,
        }
    }
    
    # 添加 GNN 配置
    if args.use_gnn:
        config['gnn'] = {
            'hidden_dim': args.gnn_hidden,
            'num_heads': args.gnn_heads,
            'edge_threshold': args.edge_threshold,
            'fusion_mode': args.gnn_fusion,
        }
    
    config_path = checkpoint_paths['config']
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"配置已保存: {config_path}")


# ====================================================================
# 修改点 5：评估函数（同时支持 v2/v3）
# ====================================================================

def evaluate_model(model, val_loader, criterion, device, stats, use_gnn=False):
    """
    评估模型（v2 和 v3 通用）
    
    Args:
        model: 模型实例
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: PyTorch 设备
        stats: 统计量字典
        use_gnn: 是否使用 GNN（用于日志）
    
    Returns:
        avg_loss, avg_mae
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            features, x_orig, y_delta, y_vel, y_accel, y_orig = batch
            features = features.to(device)
            x_orig = x_orig.to(device)
            y_delta = y_delta.to(device)
            y_vel = y_vel.to(device)
            y_accel = y_accel.to(device)
            y_orig = y_orig.to(device)
            
            # 前向传播（v2/v3 接口相同）
            pred_pos, pred_vel, pred_accel = model(
                features, x_orig, y=y_delta, y_velocity=y_vel,
                y_accel=y_accel, teacher_forcing_ratio=0.0
            )
            
            # 计算损失
            loss, loss_pos, loss_vel, loss_accel = criterion(
                pred_pos, y_delta,
                pred_velocity=pred_vel, target_velocity=y_vel,
                pred_accel=pred_accel, target_accel=y_accel
            )
            
            # 计算 MAE（未归一化）
            pred_pos_denorm = pred_pos * torch.tensor(
                stats['output_std'], device=device
            ) + torch.tensor(stats['output_mean'], device=device)
            y_denorm = y_orig
            mae = torch.abs(pred_pos_denorm - y_denorm).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            count += 1
    
    return total_loss / max(1, count), total_mae / max(1, count)


# ====================================================================
# 修改点 6：打印信息更新（显示 v2/v3 区分）
# ====================================================================

def print_epoch_header(use_gnn=False):
    """打印训练进度表头"""
    model_name = "v3+GNN" if use_gnn else "v2"
    print("=" * 140)
    print(f"{'Epoch':<8} {'Train Loss':<16} {'Val Loss':<16} {'MAE (m)':<16} {'LR':<14} {'TF Ratio':<12} {f'[{model_name}]':<20}")
    print("=" * 140)


# ====================================================================
# 修改点 7：集成到主函数（示例）
# ====================================================================

def main_with_gnn_support():
    """
    主函数示例（展示如何使用新参数）
    
    实际集成时，需要替换原脚本的对应部分
    """
    parser = build_parser_with_gnn()
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = create_model(args, device)
    
    # 获取检查点路径
    checkpoint_paths = get_checkpoint_paths(args.output_dir, args.agents, args.use_gnn)
    
    # 保存配置
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_training_config(args, checkpoint_paths, model)
    
    # 打印表头
    print_epoch_header(use_gnn=args.use_gnn)
    
    # 后续训练循环保持不变...
    logger.info("✅ 模型创建完成，可以开始训练")


# ====================================================================
# 使用说明
# ====================================================================

if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    GNN 集成快速指南                                    ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  1. 将本文件内容集成到 train_swarm_v2_complete.py                    ║
║                                                                        ║
║  2. 导入新模型和函数：                                                ║
║     from train_swarm_model_v3_with_gnn import (                       ║
║         DynamicsAwareSwarmGRUModel_with_GNN,                          ║
║         build_adjacency_from_positions                                ║
║     )                                                                 ║
║                                                                        ║
║  3. 使用 v3 模型训练：                                                ║
║     python train_swarm_v2_complete.py \\                              ║
║         --agents 3 \\                                                 ║
║         --epochs 200 \\                                               ║
║         --use_gnn \\                                                  ║
║         --gnn_hidden 64 \\                                            ║
║         --gnn_heads 4 \\                                              ║
║         --edge_threshold 5.0 \\                                       ║
║         --gnn_fusion concat \\                                        ║
║         --batch_size 256 \\                                           ║
║         --use_amp \\                                                  ║
║         --seed 42                                                     ║
║                                                                        ║
║  4. 保留 v2 模型训练（对比实验）：                                    ║
║     python train_swarm_v2_complete.py \\                              ║
║         --agents 3 \\                                                 ║
║         --epochs 200 \\                                               ║
║         --batch_size 256 \\                                           ║
║         --use_amp \\                                                  ║
║         --seed 42                                                     ║
║                                                                        ║
║  5. 监控训练输出：                                                     ║
║     - v2 模型：输出 "best_model_agents_3_v2.pt" 等                   ║
║     - v3 模型：输出 "best_model_agents_3_v3.pt" 等                   ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
