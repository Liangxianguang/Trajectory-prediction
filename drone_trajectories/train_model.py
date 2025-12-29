#!/usr/bin/env python3
"""
GRU 轨迹预测模型训练脚本
支持位置/速度预测，可配置模型架构和训练参数
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from datetime import datetime, timedelta
from torch.cuda.amp import autocast, GradScaler

class TrajectoryDataset(Dataset):
    """轨迹数据集，支持 (3,T,N) 或 (N,T,3) 格式"""
    def __init__(self, input_segs, output_segs, normalize=None):
        # 转换为 (N, T, 3) 格式
        if input_segs.ndim == 3 and input_segs.shape[0] == 3:
            self.input = np.transpose(input_segs, (2, 1, 0)).astype(np.float32)
            self.output = np.transpose(output_segs, (2, 1, 0)).astype(np.float32)
        else:
            self.input = input_segs.astype(np.float32)
            self.output = output_segs.astype(np.float32)
        
        self.normalize = normalize
        self.n_samples = self.input.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        inp = self.input[idx]
        out = self.output[idx]
        
        if self.normalize is not None:
            inp_mean = self.normalize['input_mean']
            inp_std = self.normalize['input_std']
            out_mean = self.normalize['output_mean']
            out_std = self.normalize['output_std']
            
            inp = (inp - inp_mean) / (inp_std + 1e-8)
            out = (out - out_mean) / (out_std + 1e-8)
        
        return torch.from_numpy(inp).float(), torch.from_numpy(out).float()


class GRUModel(nn.Module):
    """GRU 轨迹预测模型"""
    def __init__(self, input_size=3, hidden_dim=64, num_layers=2, 
                 dropout=0.5, output_steps=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # 编码器 GRU
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, input_size)
    
    def forward(self, x):
        """
        x: (batch, input_len, 3)
        output: (batch, output_steps, 3)
        """
        # 编码
        _, h = self.gru(x)  # h: (num_layers, batch, hidden)
        
        # 自回归解码
        predictions = []
        h_t = h
        
        for _ in range(self.output_steps):
            # 取最后一个隐藏状态
            h_last = h_t[-1:]  # (1, batch, hidden)
            
            # 预测
            y_t = self.fc(h_last.squeeze(0))  # (batch, 3)
            predictions.append(y_t.unsqueeze(1))
            
            # 更新隐藏状态（重新输入GRU）
            y_t_in = y_t.unsqueeze(1)  # (batch, 1, 3)
            _, h_t = self.gru(y_t_in, h_t)
        
        output = torch.cat(predictions, dim=1)  # (batch, output_steps, 3)
        return output

def model_forward_with_tf(model, input_seq, target_seq, teacher_forcing_ratio=0.5):
    """
    前向传播（带 Teacher Forcing）

    Args:
        model: GRUModel 实例（带 .gru 和 .fc）
        input_seq: (batch, input_len, 3)
        target_seq: (batch, output_len, 3)
        teacher_forcing_ratio: 在每一步使用真实目标作为下一个输入的概率

    Returns:
        predictions: (batch, output_steps, 3)
    """
    batch_size = input_seq.size(0)
    device = input_seq.device

    # 编码
    _, h = model.gru(input_seq)  # h: (num_layers, batch, hidden)

    # 自回归解码（带 TF）
    predictions = []
    h_t = h

    for t in range(model.output_steps):
        h_last = h_t[-1:]
        y_t = model.fc(h_last.squeeze(0))  # (batch, 3)
        predictions.append(y_t.unsqueeze(1))

        # Teacher Forcing：按概率使用真实目标作为下一个输入
        use_tf = False
        if teacher_forcing_ratio is not None and target_seq is not None:
            if np.random.rand() < teacher_forcing_ratio and t < target_seq.size(1):
                use_tf = True

        if use_tf:
            y_t_in = target_seq[:, t, :].unsqueeze(1)  # 使用真值
        else:
            y_t_in = y_t.unsqueeze(1)  # 使用模型预测

        _, h_t = model.gru(y_t_in, h_t)

    output = torch.cat(predictions, dim=1)
    return output


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    count = 0
    batch_count = 0
    
    for inp, out in loader:
        inp = inp.to(device, non_blocking=True)
        out = out.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 前向 + 反向（支持 AMP）
        if use_amp and scaler is not None:
            with autocast():
                pred = model(inp)
                loss = criterion(pred, out)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(inp)
            loss = criterion(pred, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item() * inp.size(0)
        count += inp.size(0)
        batch_count += 1
    
    return total_loss / count if count > 0 else 0.0, batch_count


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for inp, out in loader:
            inp = inp.to(device, non_blocking=True)
            out = out.to(device, non_blocking=True)
            
            pred = model(inp)
            loss = criterion(pred, out)
            
            total_loss += loss.item() * inp.size(0)
            count += inp.size(0)
    
    return total_loss / count if count > 0 else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0,
                    scaler=None, use_amp=False, teacher_forcing_ratio=0.5):
    """训练一个 epoch，支持混合精度和 Teacher Forcing"""
    model.train()
    total_loss = 0.0
    count = 0
    batch_count = 0

    for inp, out in loader:
        inp = inp.to(device, non_blocking=True)
        out = out.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 前向 + 反向（支持 AMP）
        if use_amp and scaler is not None:
            with autocast():
                pred = model_forward_with_tf(model, inp, out, teacher_forcing_ratio=teacher_forcing_ratio)
                loss = criterion(pred, out)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model_forward_with_tf(model, inp, out, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(pred, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * inp.size(0)
        count += inp.size(0)
        batch_count += 1

    return total_loss / count if count > 0 else 0.0, batch_count


def main():
    parser = argparse.ArgumentParser(description='Train GRU trajectory model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--stats_path', type=str, default=None,
                       help='外部统计量文件路径 (.npz)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='训练数据路径 (.npz)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='模型与统计量保存目录')
    parser.add_argument('--model_name', type=str, default='gru_model',
                        help='模型名称前缀')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='隐藏单元数')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='GRU 层数')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout 概率')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='初始 teacher forcing 比例（每步使用真值的概率，0-1）')
    parser.add_argument('--tf_decay', type=float, default=0.0,
                        help='每个 epoch 递减的 teacher forcing 比例（线性衰减）')
    
    # 优化参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader 工作线程数')
    parser.add_argument('--pin_memory', action='store_true',
                       help='启用 pin_memory 加速 GPU 传输')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度 (AMP) 训练')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=True,
                       help='启用 cudnn.benchmark')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 优化 CUDA 设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        print(f"cudnn.benchmark: {args.cudnn_benchmark}")
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print(f"\n加载数据: {args.data_path}")
    data = np.load(args.data_path, allow_pickle=True)
    
    input_segments = data['input_segments']
    output_segments = data['output_segments']
    
    print(f"输入段形状: {input_segments.shape}")
    print(f"输出段形状: {output_segments.shape}")
    
    # 统计量：优先从外部加载，否则从数据文件读取
    if args.stats_path is not None:
        print(f"\n从外部加载统计量: {args.stats_path}")
        stats = np.load(args.stats_path)
        input_mean = stats['input_mean']
        input_std = stats['input_std']
        output_mean = stats.get('output_mean', input_mean)
        output_std = stats.get('output_std', input_std)
    else:
        # 有些 .npz（如合成样本）未保存 mean/std，做健壮处理：优先读取字段，否则从数据中计算
        if 'input_mean' in data and 'input_std' in data:
            input_mean = data['input_mean']
            input_std = data['input_std']
            output_mean = data.get('output_mean', input_mean)
            output_std = data.get('output_std', input_std)
        else:
            print("\n统计量未在数据文件中找到，正在从样本计算 mean/std（可用于 smoke-run）...")
            # 将可能的 (3, T, N) 转换为 (N, T, 3)
            if input_segments.ndim == 3 and input_segments.shape[0] == 3:
                inp_arr = np.transpose(input_segments, (2, 1, 0))
                out_arr = np.transpose(output_segments, (2, 1, 0))
            else:
                inp_arr = input_segments
                out_arr = output_segments

            # 计算全局的通道均值/标准差（shape: (3,)），可在 __getitem__ 中广播到 (T,3)
            input_mean = np.mean(inp_arr, axis=(0, 1))
            input_std = np.std(inp_arr, axis=(0, 1))
            output_mean = np.mean(out_arr, axis=(0, 1))
            output_std = np.std(out_arr, axis=(0, 1))
            print(f"  计算得到 input_mean: {input_mean}, input_std: {input_std}")
    
    print(f"\n统计量:")
    print(f"  输入 mean: {input_mean}")
    print(f"  输入 std: {input_std}")
    print(f"  输出 mean: {output_mean}")
    print(f"  输出 std: {output_std}")
    
    # 确定样本数
    if input_segments.ndim == 3 and input_segments.shape[0] == 3:
        num_samples = input_segments.shape[2]
    else:
        num_samples = input_segments.shape[0]
    
    print(f"\n总样本数: {num_samples}")
    
    # 分割 train/val
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    num_val = int(num_samples * args.val_split)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    print(f"训练集: {len(train_indices)}, 验证集: {len(val_indices)}")
    
    # 准备数据集
    if input_segments.ndim == 3 and input_segments.shape[0] == 3:
        train_inp = input_segments[:, :, train_indices]
        train_out = output_segments[:, :, train_indices]
        val_inp = input_segments[:, :, val_indices]
        val_out = output_segments[:, :, val_indices]
    else:
        train_inp = input_segments[train_indices]
        train_out = output_segments[train_indices]
        val_inp = input_segments[val_indices]
        val_out = output_segments[val_indices]
    
    normalize_stats = {
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std,
    }
    
    train_dataset = TrajectoryDataset(train_inp, train_out, normalize_stats)
    val_dataset = TrajectoryDataset(val_inp, val_out, normalize_stats)
    
    # 优化的 DataLoader（启用多进程 + pin_memory + persistent_workers）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # 获取输出步数
    if output_segments.ndim == 3 and output_segments.shape[0] == 3:
        output_steps = output_segments.shape[1]
    else:
        output_steps = output_segments.shape[1]
    
    print(f"输出步数: {output_steps}")
    
    # 创建模型
    model = GRUModel(input_size=3, hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers, dropout=args.dropout,
                    output_steps=output_steps)
    model.to(device)
    
    print(f"\n模型配置:")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  层数: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  参数数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  DataLoader workers: {args.num_workers}")
    print(f"  Pin memory: {args.pin_memory}")
    print(f"  使用 AMP: {args.use_amp}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    # 混合精度缩放器
    scaler = GradScaler(enabled=args.use_amp)
    
    # 训练
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n开始训练 ({args.epochs} epochs)...")
    print("=" * 90)
    print(f"{'Epoch':<8} {'Batch':<12} {'Train Loss':<14} {'Val Loss':<14} {'LR':<10} {'ETA':<12} {'Status':<15}")
    print("=" * 90)
    
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # 计算本 epoch 的 teacher forcing 比例（线性衰减）
        tf_current = max(0.0, args.teacher_forcing_ratio - args.tf_decay * (epoch - 1))
        train_loss, num_batches = train_one_epoch(model, train_loader, optimizer, criterion,
                                    device, args.grad_clip, scaler, args.use_amp,
                                    teacher_forcing_ratio=tf_current)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times[-10:])  # 最近10个epoch的平均时间
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算 ETA
        remaining_epochs = args.epochs - epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # 检查最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            status = "✓ BEST"
            
            # 保存最优模型
            model_path = os.path.join(args.output_dir, f'{args.model_name}_best_model.pth')
            torch.save(model.state_dict(), model_path)
            
            # 保存统计量
            stats_path = os.path.join(args.output_dir, f'{args.model_name}_norm_stats.npz')
            np.savez(stats_path, input_mean=input_mean, input_std=input_std,
                    output_mean=output_mean, output_std=output_std)
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{args.patience}"
        
        # 打印进度
        print(f"{epoch:<8} {num_batches:<12} {train_loss:<14.6f} {val_loss:<14.6f} {current_lr:<10.2e} {eta_str:<12} {status:<15}")
        
        # 早停
        if patience_counter >= args.patience:
            print("=" * 90)
            print(f"✓ 早停! (patience={args.patience})")
            break
    
    total_time = time.time() - start_time
    print("=" * 90)
    print(f"\n✓ 训练完成!")
    print(f"{'总耗时':<20} {str(timedelta(seconds=int(total_time)))}")
    print(f"{'最优验证损失':<20} {best_val_loss:.6f}")
    print(f"{'最优模型路径':<20} {os.path.join(args.output_dir, f'{args.model_name}_best_model.pth')}")
    print(f"{'统计量保存路径':<20} {os.path.join(args.output_dir, f'{args.model_name}_norm_stats.npz')}")
    print(f"{'训练轮次':<20} {epoch} / {args.epochs}")
    print()


if __name__ == '__main__':
    main()
