#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验1：基线模型
配置：× × × (16D)
- 无GAT
- 无特征增强（16D特征）
- 无BiGRU+Cross Attention（单向GRU，无Cross Attention）
"""
import sys
import io
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
from datetime import datetime
import json
import pandas as pd

# 导入消融实验模块
from ablation_models import BaselineGRUModel
from ablation_train_utils import load_ablation_data
from train_swarm_model_v2_dynamics_aware import DynamicsAwareLoss


def _flatten_batch_agents(t):
    """(B, T, A, C) -> (B*A, T, C)"""
    if t is None:
        return None
    if t.dim() != 4:
        return t
    b, tlen, a, c = t.shape
    return t.reshape(b * a, tlen, c)


def train_epoch(model, train_loader, optimizer, loss_fn, device, use_amp=False, tf_ratio=0.6, flatten_agents=True):
    """训练单个epoch

    flatten_agents:
      - True: (B,T,A,C) -> (B*A,T,C)（适用于非GNN模型）
      - False: 保持4D输入（适用于GNN模型，模型内部会处理4D）
    """
    model.train()
    total_loss = 0
    loss_pos = 0
    loss_vel = 0
    loss_accel = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(train_loader, desc="训练")
    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)
        x_orig = batch['x_orig'].to(device)
        features = batch['features'].to(device)
        y_target = batch['y_delta'].to(device)
        y_vel_target = batch['y_velocity'].to(device)
        y_accel_target = batch['y_accel'].to(device)

        if flatten_agents:
            features_in = _flatten_batch_agents(features)
            x_orig_in = _flatten_batch_agents(x_orig)
            y_target_in = _flatten_batch_agents(y_target)
            y_vel_target_in = _flatten_batch_agents(y_vel_target)
            y_accel_target_in = _flatten_batch_agents(y_accel_target)
        else:
            features_in = features
            x_orig_in = x_orig
            y_target_in = y_target
            y_vel_target_in = y_vel_target
            y_accel_target_in = y_accel_target
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                output_pos, output_vel, output_accel = model(
                    features_in,
                    x_orig_in,
                    y_target_in,
                    y_vel_target_in,
                    y_accel_target_in,
                    teacher_forcing_ratio=tf_ratio
                )
                
                loss_result = loss_fn(
                    _flatten_batch_agents(output_pos), _flatten_batch_agents(y_target_in),
                    _flatten_batch_agents(output_vel), _flatten_batch_agents(y_vel_target_in),
                    _flatten_batch_agents(output_accel), _flatten_batch_agents(y_accel_target_in)
                )
                
                loss = loss_result[0]
                l_pos = loss_result[1]
                l_vel = loss_result[2]
                l_accel = loss_result[3]
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output_pos, output_vel, output_accel = model(
                features_in,
                x_orig_in,
                y_target_in,
                y_vel_target_in,
                y_accel_target_in,
                teacher_forcing_ratio=tf_ratio
            )
            
            loss_result = loss_fn(
                _flatten_batch_agents(output_pos), _flatten_batch_agents(y_target_in),
                _flatten_batch_agents(output_vel), _flatten_batch_agents(y_vel_target_in),
                _flatten_batch_agents(output_accel), _flatten_batch_agents(y_accel_target_in)
            )
            
            loss = loss_result[0]
            l_pos = loss_result[1]
            l_vel = loss_result[2]
            l_accel = loss_result[3]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        with torch.no_grad():
            total_loss += float(loss.item())
            loss_pos += float(l_pos.item())
            loss_vel += float(l_vel.item())
            loss_accel += float(l_accel.item())
        
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.6f}',
            'pos': f'{loss_pos / (batch_idx + 1):.6f}',
            'vel': f'{loss_vel / (batch_idx + 1):.6f}',
            'accel': f'{loss_accel / (batch_idx + 1):.6f}',
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_pos = loss_pos / len(train_loader)
    avg_vel = loss_vel / len(train_loader)
    avg_accel = loss_accel / len(train_loader)
    
    return avg_loss, avg_pos, avg_vel, avg_accel


def evaluate(model, val_loader, loss_fn, device, output_mean, output_std):
    """验证模型（默认使用teacher_forcing_ratio=0.0）"""
    model.eval()
    total_loss = 0
    total_mae = 0
    
    output_mean_tensor = torch.from_numpy(output_mean).float().to(device)
    output_std_tensor = torch.from_numpy(output_std).float().to(device)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证", leave=False):
            x = batch['x'].to(device)
            x_orig = batch['x_orig'].to(device)
            features = batch['features'].to(device)
            y_target = batch['y_delta'].to(device)
            y_vel_target = batch['y_velocity'].to(device)
            y_accel_target = batch['y_accel'].to(device)

            # 这里不做teacher forcing
            output_pos, output_vel, output_accel = model(
                features,
                x_orig,
                y=None,
                teacher_forcing_ratio=0.0
            )

            # 统一在loss/metric处展平 (B,T,A,C)->(B*A,T,C)
            y_target_flat = _flatten_batch_agents(y_target)
            y_vel_flat = _flatten_batch_agents(y_vel_target)
            y_accel_flat = _flatten_batch_agents(y_accel_target)
            out_pos_flat = _flatten_batch_agents(output_pos)
            out_vel_flat = _flatten_batch_agents(output_vel)
            out_accel_flat = _flatten_batch_agents(output_accel)

            loss_result = loss_fn(
                out_pos_flat, y_target_flat,
                out_vel_flat, y_vel_flat,
                out_accel_flat, y_accel_flat
            )
            
            loss = loss_result[0]
            
            mean_expanded = output_mean_tensor.view(1, 1, 3)
            std_expanded = output_std_tensor.view(1, 1, 3)
            
            output_pos_denorm = out_pos_flat * std_expanded + mean_expanded
            y_target_denorm = y_target_flat * std_expanded + mean_expanded
            
            mae = torch.mean(torch.abs(output_pos_denorm - y_target_denorm)).item()
            
            total_loss += loss.item()
            total_mae += mae
    
    avg_loss = total_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)
    
    return avg_loss, avg_mae


def main():
    parser = argparse.ArgumentParser(description='消融实验1：基线模型 (16D, 无GAT, 无BiGRU+CA)')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='../swarm_segments', help='数据目录')
    parser.add_argument('--agents', type=int, default=3, help='代理数量')
    parser.add_argument('--use_subset', action='store_true', help='使用数据子集')
    parser.add_argument('--features_dir', type=str, default='../swarm_features', help='16D特征目录（默认: ../swarm_features）')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=128, help='GRU隐藏维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GRU层数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='权重衰减')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    
    # 杂项
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--no_resume', action='store_true', help='不从最后的检查点恢复')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='指定检查点路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据（16D特征）
    print("加载数据和16D特征...")
    data_info = load_ablation_data(
        args.data_dir,
        args.agents,
        feature_dim=16,
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=0,
        use_subset=args.use_subset,
        features_dir=args.features_dir
    )
    
    train_loader = data_info['train_loader']
    val_loader = data_info['val_loader']
    
    print(f"✓ 数据加载完成: 训练={len(data_info['train_dataset'])}, 验证={len(data_info['val_dataset'])}")
    
    # 创建模型（基线：单向GRU，无Cross Attention）
    print("创建基线模型...")
    model = BaselineGRUModel(
        input_size=16,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=3,
        dropout=0.3
    )
    model = model.to(device)
    print(f"✓ 模型创建: 基线模型（单向GRU，无Cross Attention）")
    print(f"  参数数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  输入维度: 16D")
    
    # Loss函数（使用多任务损失，但基线模型可能不输出速度和加速度）
    loss_fn = DynamicsAwareLoss(weight_position=0.8, weight_velocity=0.1, weight_accel=0.1)
    print(f"  Loss权重: position=1.0, velocity=0.0, accel=0.0")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 检查点目录
    suffix = f"agents_{args.agents}_exp1_baseline"
    ckpt_dir = Path(f"ablation_results_{suffix}")
    ckpt_dir.mkdir(exist_ok=True)
    
    csv_file = ckpt_dir / f"training_history_{suffix}.csv"
    config_file = ckpt_dir / f"config_{suffix}.json"
    
    # 配置信息
    config = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp1_baseline',
        'description': '基线模型 (16D, 无GAT, 无特征增强, 无BiGRU+CA)',
        'input_features': 16,
        'num_agents': args.agents,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'seed': args.seed,
        'use_amp': args.use_amp,
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)  # ensure_ascii=False 保持中文可读
    
    print(f"配置已保存: {config_file}")
    
    # 训练循环
    print(f"\n开始训练消融实验1（基线模型）...")
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
            print(f"从检查点恢复：{ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            training_history = ckpt.get('training_history', training_history)
    elif args.checkpoint_path:
        print(f"加载检查点：{args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        training_history = ckpt.get('training_history', training_history)
    
    try:
        for epoch in range(start_epoch, args.epochs):
            # 教师强制比率衰减
            tf_ratio = max(0.0, 0.6 - 0.005 * epoch)
            
            # 训练
            train_loss, train_pos, train_vel, train_accel = train_epoch(
                model, train_loader, optimizer, loss_fn, device,
                use_amp=args.use_amp, tf_ratio=tf_ratio
            )
            
            # 验证
            val_loss, val_mae = evaluate(
                model, val_loader, loss_fn, device,
                data_info['output_mean'], data_info['output_std']
            )
            
            # 调度器
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

            # CSV 逐 epoch 追加（v3 风格）
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

            # checkpoint 每 10 轮保存一个（减少文件数量）
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
                print(f"✓ 定期检查点保存: {ckpt_path.name}")

            # best_model：只保留一个全局最优（不断覆盖）
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
                print(f"✓ 最佳模型已更新: {best_model_path.name} (val_loss={val_loss:.6f}, mae={val_mae:.6f}m)")
    
    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断，正在保存断点...")
        interrupted_ckpt = ckpt_dir / f"interrupted_checkpoint_{epoch:04d}.pt"
        torch.save({
            'epoch': epoch,
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
        }, interrupted_ckpt)
        print(f"✓ 断点已保存: {interrupted_ckpt}")
    
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
    print(f"✓ 统计信息已保存: {stats_file}")
    
    # 保存训练历史（最终再落一份）
    df = pd.DataFrame(training_history)
    df.to_csv(csv_file, index=False)
    print(f"✓ 训练历史已保存: {csv_file}")
    
    print(f"\n✓ 消融实验1训练完成!")
    print(f"  ├─ 最佳验证损失: {best_val_loss:.6f}")
    best_mae = min(training_history['val_mae']) if training_history['val_mae'] else float('inf')
    print(f"  ├─ 最佳验证MAE: {best_mae:.6f}m")
    print(f"  ├─ 输出目录: {ckpt_dir}")
    print(f"  ├─ 配置文件: {config_file}")
    print(f"  ├─ 训练历史: {csv_file}")
    print(f"  ├─ 统计信息: {stats_file}")
    print(f"  └─ 最佳模型: best_model_{suffix}.pt")


if __name__ == '__main__':
    main()
