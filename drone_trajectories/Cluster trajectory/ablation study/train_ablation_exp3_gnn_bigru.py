#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验3：GAT + BiGRU + Cross Attention
配置：√ × √ (16D)
- GAT
- 无特征增强（16D特征）
- BiGRU+Cross Attention
"""

import sys
import io
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train_ablation_exp1_baseline import train_epoch, evaluate
from ablation_train_utils import load_ablation_data
# ✅ 直接使用经过验证的v3模型类
from train_swarm_model_v3_with_gnn import DynamicsAwareSwarmGRUModel_with_GNN
from train_swarm_model_v2_dynamics_aware import DynamicsAwareLoss

import numpy as np
import torch
import argparse
import logging
from datetime import datetime
import json
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='消融实验3：GAT+BiGRU+CA (16D, 无特征增强)')
    
    parser.add_argument('--data_dir', type=str, default='../swarm_segments', help='数据目录')
    parser.add_argument('--agents', type=int, default=3, help='代理数量')
    parser.add_argument('--use_subset', action='store_true', help='使用数据子集')
    parser.add_argument('--features_dir', type=str, default='../swarm_features', help='16D特征目录（默认: ../swarm_features）')
    
    parser.add_argument('--hidden_size', type=int, default=128, help='BiGRU隐藏维度')
    parser.add_argument('--num_layers', type=int, default=3, help='BiGRU层数')
    parser.add_argument('--gnn_hidden', type=int, default=64, help='GNN隐藏维度')
    parser.add_argument('--gnn_heads', type=int, default=4, help='GNN注意力头数')
    parser.add_argument('--edge_threshold', type=float, default=5.0, help='邻接阈值')
    
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='权重衰减')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--no_resume', action='store_true', help='不从最后的检查点恢复')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='指定检查点路径')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    logger.info("加载数据和16D特征...")
    data_info = load_ablation_data(
        args.data_dir, args.agents, feature_dim=16,
        batch_size=args.batch_size, val_split=0.2, num_workers=0,
        use_subset=args.use_subset, features_dir=args.features_dir
    )
    
    train_loader = data_info['train_loader']
    val_loader = data_info['val_loader']
    
    logger.info(f"✓ 数据加载完成")
    
    logger.info("创建模型...")
    # ✅ 直接使用经过验证的v3模型类，只需设置input_size=16
    model = DynamicsAwareSwarmGRUModel_with_GNN(
        input_size=16,  # 16D特征（v3原本是24D）
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=3,
        dropout=0.3,
        use_attention=True,  # BiGRU + Cross Attention
        gnn_hidden=args.gnn_hidden,
        num_gnn_heads=args.gnn_heads,
        edge_threshold=args.edge_threshold,
        fusion_mode='concat'
    )
    model = model.to(device)
    logger.info(f"✓ 模型创建: GAT+BiGRU+Cross Attention（16D特征）")
    logger.info(f"  ✅ 使用经过验证的v3模型类 (DynamicsAwareSwarmGRUModel_with_GNN)")
    logger.info(f"  参数数: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_fn = DynamicsAwareLoss(weight_position=0.65, weight_velocity=0.25, weight_accel=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    suffix = f"agents_{args.agents}_exp3_gnn_bigru"
    ckpt_dir = Path(f"ablation_results_{suffix}")
    ckpt_dir.mkdir(exist_ok=True)

    csv_file = ckpt_dir / f"training_history_{suffix}.csv"
    config_file = ckpt_dir / f"config_{suffix}.json"
    
    config = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp3_gnn_bigru',
        'description': 'GAT+BiGRU+Cross Attention (16D, 无特征增强)',
        'input_features': 16,
        'num_agents': args.agents,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'gnn_hidden': args.gnn_hidden,
        'gnn_heads': args.gnn_heads,
        'edge_threshold': args.edge_threshold,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'seed': args.seed,
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)  # ensure_ascii=False 保持中文可读
    
    logger.info(f"\n开始训练消融实验3...")
    print("\n" + "="*130)
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Pos':<10} {'Vel':<10} {'Accel':<10} {'Val Loss':<14} {'MAE (m)':<12} {'LR':<12}")
    print("="*130)
    
    # ===== v3 风格：断点续训 + 逐 epoch 保存 =====
    start_epoch = 0
    best_val_loss = float('inf')
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_loss_pos': [],
        'train_loss_vel': [],
        'train_loss_accel': [],
        'val_loss': [],
        'val_mae': [],
        'lr': [],
        'tf_ratio': [],
    }

    if not args.no_resume and args.checkpoint_path is None:
        ckpt_candidates = sorted(ckpt_dir.glob('checkpoint_*.pt'))
        if ckpt_candidates:
            ckpt_path = ckpt_candidates[-1]
            logger.info(f"从检查点恢复：{ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            training_history = ckpt.get('training_history', training_history)
    elif args.checkpoint_path:
        logger.info(f"加载检查点：{args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        training_history = ckpt.get('training_history', training_history)
    
    try:
        for epoch in range(start_epoch, args.epochs):
            tf_ratio = max(0.0, 0.6 - 0.005 * epoch)
            
            train_loss, train_pos, train_vel, train_accel = train_epoch(
                model, train_loader, optimizer, loss_fn, device,
                use_amp=args.use_amp, tf_ratio=tf_ratio, flatten_agents=False
            )
            
            val_loss, val_mae = evaluate(
                model, val_loader, loss_fn, device,
                data_info['output_mean'], data_info['output_std']
            )
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            training_history['epoch'].append(epoch)
            training_history['train_loss'].append(train_loss)
            training_history['train_loss_pos'].append(train_pos)
            training_history['train_loss_vel'].append(train_vel)
            training_history['train_loss_accel'].append(train_accel)
            training_history['val_loss'].append(val_loss)
            training_history['val_mae'].append(val_mae)
            training_history['lr'].append(current_lr)
            training_history['tf_ratio'].append(tf_ratio)

            with open(csv_file, 'a' if epoch > start_epoch else 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'Epoch', 'Train Loss', 'Train Loss (pos)', 'Train Loss (vel)',
                        'Train Loss (accel)', 'Val Loss', 'Val MAE (m)',
                        'Learning Rate', 'Teacher Forcing Ratio'
                    ]
                )
                if epoch == start_epoch:
                    writer.writeheader()
                writer.writerow({
                    'Epoch': epoch,
                    'Train Loss': f'{train_loss:.6f}',
                    'Train Loss (pos)': f'{train_pos:.6f}',
                    'Train Loss (vel)': f'{train_vel:.6f}',
                    'Train Loss (accel)': f'{train_accel:.6f}',
                    'Val Loss': f'{val_loss:.6f}',
                    'Val MAE (m)': f'{val_mae:.6f}',
                    'Learning Rate': f'{current_lr:.2e}',
                    'Teacher Forcing Ratio': f'{tf_ratio:.4f}',
                })

            config['current_epoch'] = epoch
            config['best_val_loss'] = best_val_loss
            config['current_val_loss'] = val_loss
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            status = "[BEST]" if val_loss < best_val_loss else ""
            print(
                f"{epoch:<8} {train_loss:<14.6f} {train_pos:<10.6f} {train_vel:<10.6f} "
                f"{train_accel:<10.6f} {val_loss:<14.6f} {val_mae:<12.6f} {current_lr:<12.2e} {tf_ratio:<10.4f} {status}"
            )

            # checkpoint 每 10 轮保存一个
            if (epoch + 1) % 10 == 0:
                ckpt_path = ckpt_dir / f'checkpoint_{epoch:04d}.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'training_history': training_history,
                    'input_mean': data_info['input_mean'],
                    'input_std': data_info['input_std'],
                    'output_mean': data_info['output_mean'],
                    'output_std': data_info['output_std'],
                    'feature_mean': data_info['feature_mean'],
                    'feature_std': data_info['feature_std'],
                    'config': config,
                }, ckpt_path)
                logger.info(f"✓ 定期检查点保存: {ckpt_path.name}")

            # best_model：只保留一个全局最优
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = ckpt_dir / f'best_model_{suffix}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_mae': val_mae,
                    'config': config,
                    'input_mean': data_info['input_mean'],
                    'input_std': data_info['input_std'],
                    'output_mean': data_info['output_mean'],
                    'output_std': data_info['output_std'],
                    'feature_mean': data_info['feature_mean'],
                    'feature_std': data_info['feature_std'],
                }, best_model_path)
                logger.info(f"✓ 最佳模型已更新: {best_model_path.name} (VAL_LOSS={val_loss:.6f}, MAE={val_mae:.6f}m)")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 训练被中断")
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss, 'training_history': training_history,
            'input_mean': data_info['input_mean'], 'input_std': data_info['input_std'],
            'output_mean': data_info['output_mean'], 'output_std': data_info['output_std'],
            'feature_mean': data_info['feature_mean'], 'feature_std': data_info['feature_std'],
            'config': config,
        }, ckpt_dir / f"interrupted_checkpoint_{epoch:04d}.pt")
    
    print("="*130)
    
    # 保存统计信息文件（与v4格式一致）
    stats_file = ckpt_dir / f"stats_{suffix}.npz"
    np.savez(stats_file,
             input_mean=data_info['input_mean'],
             input_std=data_info['input_std'],
             output_mean=data_info['output_mean'],
             output_std=data_info['output_std'],
             feature_mean=data_info['feature_mean'] if data_info['feature_mean'] is not None else np.zeros(16),
             feature_std=data_info['feature_std'] if data_info['feature_std'] is not None else np.ones(16))
    logger.info(f"✓ 统计信息已保存: {stats_file}")
    
    # 保存训练历史（最终再落一份）
    df = pd.DataFrame(training_history)
    df.to_csv(csv_file, index=False)
    logger.info(f"✓ 训练历史已保存: {csv_file}")
    
    logger.info(f"\n✓ 消融实验3训练完成!")
    logger.info(f"  ├─ 最佳验证损失: {best_val_loss:.6f}")
    best_mae = min(training_history['val_mae']) if training_history['val_mae'] else float('inf')
    logger.info(f"  ├─ 最佳验证MAE: {best_mae:.6f}m")
    logger.info(f"  ├─ 输出目录: {ckpt_dir}")
    logger.info(f"  ├─ 配置文件: {config_file}")
    logger.info(f"  ├─ 训练历史: {csv_file}")
    logger.info(f"  ├─ 统计信息: {stats_file}")
    logger.info(f"  └─ 最佳模型: best_model_{suffix}.pt")


if __name__ == '__main__':
    main()
