#!/usr/bin/env python3
"""
MRGTraj Swarm - 优化版训练脚本（ADE/FDE加速下降）
==================================================

关键优化:
1. ADE/FDE 权重逐步增加（Early Stopping防止过拟合）
2. 移除竞争性损失（collision/formation），简化目标
3. 梯度焦点：直接优化ADE/FDE而不是L2重建
4. 学习率warm-up + 动态调整
5. 轨迹平滑正则化（不是碰撞避免）
"""

import argparse
import os
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys

from model_swarm import MRGTrajSwarm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except Exception:
    HAS_TENSORBOARD = False
    SummaryWriter = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """详细指标跟踪器"""
    
    def __init__(self):
        self.metrics = {}
    
    def reset(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(float(val))
    
    def get_averages(self):
        averages = {}
        for key, vals in self.metrics.items():
            if vals:
                averages[key] = np.mean(vals)
        return averages
    
    def format_metrics(self):
        averages = self.get_averages()
        parts = []
        for key in sorted(averages.keys()):
            val = averages[key]
            parts.append(f"{key}={val:.6f}")
        return " ".join(parts)


class SwarmDataLoader:
    """无人机集群数据加载器"""
    
    def __init__(self, npz_input_file, npz_output_file, obs_len, pred_len, batch_size, shuffle=True):
        logger.info(f"加载数据: {npz_input_file}, {npz_output_file}")
        
        self.X = np.load(npz_input_file)['data']  # (obs_len, num_samples, num_agents, 3)
        self.Y = np.load(npz_output_file)['data']  # (pred_len, num_samples, num_agents, 3)
        
        logger.info(f"  X 形状: {self.X.shape}")
        logger.info(f"  Y 形状: {self.Y.shape}")
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.num_samples = self.X.shape[1]
        self.num_agents = self.X.shape[2]
        
        self.indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(self.indices)
        
        logger.info(f"  总样本数: {self.num_samples}")
        logger.info(f"  无人机数: {self.num_agents}")
    
    def __iter__(self):
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            X_batch = self.X[:, batch_indices, :, :]
            Y_batch = self.Y[:, batch_indices, :, :]
            
            X_batch = X_batch.transpose(1, 0, 2, 3)
            Y_batch = Y_batch.transpose(1, 0, 2, 3)
            
            X_batch = torch.from_numpy(X_batch).float()
            Y_batch = torch.from_numpy(Y_batch).float()
            
            yield X_batch, Y_batch
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def create_data_loaders(args):
    """创建数据加载器"""
    data_dir = Path(args.data_dir)
    
    if not data_dir.is_absolute():
        possible_paths = [
            Path(__file__).parent.parent / data_dir,
            Path(__file__).parent / data_dir,
            data_dir
        ]
        for possible_path in possible_paths:
            if possible_path.exists():
                data_dir = possible_path
                break

    train_input_file = data_dir / f'input_agents_{args.num_agents}_subset.npz'
    train_output_file = data_dir / f'output_agents_{args.num_agents}_subset.npz'
    
    if not train_input_file.exists():
        train_input_file = data_dir / f'input_agents_{args.num_agents}.npz'
        train_output_file = data_dir / f'output_agents_{args.num_agents}.npz'
    
    if not train_input_file.exists():
        logger.error("数据文件不存在")
        raise FileNotFoundError(f"数据文件不存在: {train_input_file}")
    
    logger.info("使用数据文件:")
    logger.info(f"  输入: {train_input_file}")
    logger.info(f"  输出: {train_output_file}")
    
    train_loader = SwarmDataLoader(
        train_input_file,
        train_output_file,
        args.obs_len,
        args.pred_len,
        args.batch_size,
        shuffle=True
    )
    
    return train_loader


class OptimizedLossFunctions:
    """优化的损失函数集合"""
    
    @staticmethod
    def ade_loss(pred, target):
        """ADE 损失 - 平均位移误差"""
        diff = torch.norm(pred - target, dim=-1)  # (batch, pred_len, num_agents)
        return diff.mean()
    
    @staticmethod
    def fde_loss(pred, target):
        """FDE 损失 - 最终位移误差（权重更高）"""
        diff = torch.norm(pred[:, -1, :, :] - target[:, -1, :, :], dim=-1)  # (batch, num_agents)
        return diff.mean()
    
    @staticmethod
    def weighted_ade_loss(pred, target):
        """加权 ADE - 近期帧权重更高"""
        diff = torch.norm(pred - target, dim=-1)  # (batch, pred_len, num_agents)
        seq_len = diff.shape[1]
        
        # 权重：早期 0.5，后期 1.5
        weights = torch.linspace(0.5, 1.5, seq_len, device=diff.device)
        weights = weights.view(1, -1, 1)
        
        weighted_diff = (diff * weights).mean()
        return weighted_diff
    
    @staticmethod
    def smoothness_loss(pred):
        """轨迹平滑损失 - 鼓励连贯运动"""
        # 计算二阶导数（加速度）
        if pred.shape[1] < 3:
            return torch.tensor(0.0, device=pred.device)
        
        vel1 = torch.diff(pred, dim=1)
        acc = torch.diff(vel1, dim=1)
        
        return (acc ** 2).mean()
    
    @staticmethod
    def l2_loss(pred, target):
        """L2 重建损失"""
        return ((pred - target) ** 2).mean()
    
    @staticmethod
    def velocity_consistency_loss(pred, target):
        """速度一致性损失"""
        pred_vel = torch.diff(pred, dim=1)
        target_vel = torch.diff(target, dim=1)
        return ((pred_vel - target_vel) ** 2).mean()


class ADE_FDE_Calculator:
    """ADE/FDE 计算器"""
    
    @staticmethod
    def ade(pred, target):
        diff = torch.norm(pred - target, dim=-1)
        return diff.mean().item()
    
    @staticmethod
    def fde(pred, target):
        diff = torch.norm(pred[:, -1, :, :] - target[:, -1, :, :], dim=-1)
        return diff.mean().item()
    
    @staticmethod
    def ade_per_agent(pred, target):
        diff = torch.norm(pred - target, dim=-1)
        return diff.mean(dim=(0, 1)).cpu().numpy()


def get_loss_weight_schedule(epoch, total_epochs, base_ade=2.0, base_fde=1.0):
    """获得动态损失权重
    
    策略：
    - 前期：ADE/FDE 权重较低，让模型稳定学习
    - 中期：逐步提升 ADE/FDE 权重
    - 后期：最大化 ADE/FDE 优化
    """
    progress = epoch / total_epochs
    
    # ADE 权重从 1.0 增到 3.0
    ade_weight = 1.0 + 2.0 * (progress ** 1.5)
    
    # FDE 权重从 0.5 增到 2.0
    fde_weight = 0.5 + 1.5 * (progress ** 1.5)
    
    # L2 权重从 1.0 衰减到 0.3
    l2_weight = 1.0 * (1 - 0.7 * progress)
    
    # 平滑损失权重从 0.2 增到 0.5
    smooth_weight = 0.2 + 0.3 * progress
    
    return {
        'ade': ade_weight,
        'fde': fde_weight,
        'l2': l2_weight,
        'smooth': smooth_weight,
        'vel': 0.1
    }


def get_learning_rate_schedule(epoch, base_lr, total_epochs, warmup_epochs=10):
    """学习率调度
    
    策略：
    1. Warmup: 线性增长
    2. 衰减: Cosine 衰减
    """
    if epoch < warmup_epochs:
        # Warmup 阶段
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine 衰减
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def train_epoch(model, train_loader, optimizer, epoch, args, metrics_tracker, total_epochs):
    """训练一个 epoch"""
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", ncols=100)
    
    loss_fn = OptimizedLossFunctions()
    calculator = ADE_FDE_Calculator()
    
    # 获取动态权重
    loss_weights = get_loss_weight_schedule(epoch, total_epochs)
    
    for batch_idx, (past_traj, future_traj) in enumerate(pbar):
        past_traj = past_traj.cuda()
        future_traj = future_traj.cuda()
        
        # 前向传播
        pred_traj, mu, log_var = model(past_traj, future_traj)
        
        # 计算各个损失
        ade_loss_val = loss_fn.ade_loss(pred_traj, future_traj)
        fde_loss_val = loss_fn.fde_loss(pred_traj, future_traj)
        weighted_ade_loss_val = loss_fn.weighted_ade_loss(pred_traj, future_traj)
        smoothness_loss_val = loss_fn.smoothness_loss(pred_traj)
        l2_loss_val = loss_fn.l2_loss(pred_traj, future_traj)
        vel_consistency_loss_val = loss_fn.velocity_consistency_loss(pred_traj, future_traj)
        
        # KL 散度（变分正则化）
        kl_loss_val = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]
        
        # 组合损失 - 直接优化 ADE/FDE
        total_loss = (
            loss_weights['ade'] * ade_loss_val +
            loss_weights['fde'] * fde_loss_val +
            loss_weights['l2'] * l2_loss_val +
            loss_weights['smooth'] * smoothness_loss_val +
            loss_weights['vel'] * vel_consistency_loss_val +
            0.01 * kl_loss_val  # KL 权重很低
        )
        
        # 计算 ADE/FDE 指标
        ade_val = calculator.ade(pred_traj, future_traj)
        fde_val = calculator.fde(pred_traj, future_traj)
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 记录指标
        metrics_tracker.update(
            total_loss=total_loss.item(),
            ade_loss=ade_loss_val.item(),
            fde_loss=fde_loss_val.item(),
            weighted_ade=weighted_ade_loss_val.item(),
            l2_loss=l2_loss_val.item(),
            smooth_loss=smoothness_loss_val.item(),
            vel_loss=vel_consistency_loss_val.item(),
            kl_loss=kl_loss_val.item(),
            ade=ade_val,
            fde=fde_val
        )
        
        if batch_idx % 5 == 0:
            pbar.set_postfix_str(metrics_tracker.format_metrics())
    
    return metrics_tracker.get_averages()


def main(args):
    """主训练函数"""
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    # 创建检查点目录
    checkpoint_dir = Path(args.checkpoint_dir) / f"agents_{args.num_agents}_optimized"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = checkpoint_dir / f"train_optimized.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("=" * 100)
    logger.info("MRGTraj Swarm - 优化版训练（ADE/FDE 加速）")
    logger.info("=" * 100)
    
    # 打印配置
    logger.info("配置参数:")
    for key, val in vars(args).items():
        logger.info(f"  {key}: {val}")
    logger.info("=" * 100)
    
    # 加载数据
    logger.info("\n加载数据...")
    train_loader = create_data_loaders(args)
    logger.info(f"[OK] 数据加载成功")
    logger.info(f"  总样本数: {train_loader.num_samples}")
    logger.info(f"  批次大小: {args.batch_size}")
    logger.info(f"  每个 epoch 的批次数: {len(train_loader)}")
    
    # 创建模型
    logger.info("\n创建模型...")
    model = MRGTrajSwarm(args)
    model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[OK] 模型创建成功")
    logger.info(f"  模型架构: MRGTrajSwarm")
    logger.info(f"  总参数: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    
    # 优化器 - 更高的初始学习率以加速收敛
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f"\n优化器配置:")
    logger.info(f"  类型: Adam")
    logger.info(f"  初始学习率: {args.lr}")
    logger.info(f"  权重衰减: {args.weight_decay}")
    
    # TensorBoard
    if HAS_TENSORBOARD:
        log_dir = checkpoint_dir / "logs"
        writer = SummaryWriter(log_dir)
        logger.info(f"\nTensorBoard 已启用")
        logger.info(f"  日志目录: {log_dir}")
    else:
        writer = None
        logger.warning("\nTensorBoard 不可用")
    
    logger.info("\n" + "=" * 100)
    logger.info("开始训练...")
    logger.info("=" * 100 + "\n")
    
    best_ade = float('inf')
    best_epoch = 0
    no_improve_count = 0
    patience = 30
    
    for epoch in range(args.num_epochs):
        # 更新学习率
        new_lr = get_learning_rate_schedule(epoch, args.lr, args.num_epochs, warmup_epochs=10)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # 训练
        metrics_tracker = MetricsTracker()
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, args, metrics_tracker, args.num_epochs)
        
        # 日志
        log_msg = f"[Epoch {epoch+1}/{args.num_epochs}] "
        metrics_parts = []
        for key in sorted(train_metrics.keys()):
            v = train_metrics[key]
            metrics_parts.append(f"{key}={v:.6f}")
        
        log_msg += " | ".join(metrics_parts)
        log_msg += f" | LR={new_lr:.2e}"
        logger.info(log_msg)
        
        # TensorBoard
        if writer:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            writer.add_scalar("train/learning_rate", new_lr, epoch)
        
        # 检查 ADE 改进
        current_ade = train_metrics.get('ade', float('inf'))
        if current_ade < best_ade:
            best_ade = current_ade
            best_epoch = epoch
            no_improve_count = 0
            
            checkpoint_path = checkpoint_dir / f"best_model_ade_{best_ade:.4f}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ade': best_ade,
                'args': args
            }, checkpoint_path)
            logger.info(f"  ✓ [BEST] 保存最佳模型 - ADE={best_ade:.6f}m")
        else:
            no_improve_count += 1
        
        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ade': current_ade,
                'args': args
            }, checkpoint_path)
            logger.info(f"  ✓ [CHECKPOINT] 保存检查点 (epoch {epoch+1})")
        
        # 早停
        if no_improve_count >= patience:
            logger.info(f"\n⏹ 早停触发！在 epoch {best_epoch+1} 达到最佳 ADE={best_ade:.6f}m")
            break
    
    if writer:
        writer.close()
    
    logger.info("\n" + "=" * 100)
    logger.info("[完成] 训练结束!")
    logger.info(f"  最佳 ADE: {best_ade:.6f}m")
    logger.info(f"  最佳 epoch: {best_epoch+1}")
    logger.info(f"  检查点目录: {checkpoint_dir}")
    logger.info("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRGTraj Swarm - 优化版训练脚本")
    
    # 数据相关
    parser.add_argument("--data_dir", type=str, default="../Cluster trajectory/swarm_segments",
                        help="数据目录路径")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="无人机数量")
    
    # 模型参数
    parser.add_argument("--d_model", type=int, default=256,
                        help="模型维度")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="注意力头数")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Transformer 层数")
    parser.add_argument("--noise_dim", type=int, default=64,
                        help="噪声维度")
    parser.add_argument("--agent_dim", type=int, default=3,
                        help="每个智能体的特征维度")
    
    # 序列参数
    parser.add_argument("--obs_len", type=int, default=20,
                        help="观察序列长度")
    parser.add_argument("--pred_len", type=int, default=10,
                        help="预测序列长度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=512,
                        help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="初始学习率（加速收敛）")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--gpu_num", type=str, default="0",
                        help="GPU 设备号")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_optimized",
                        help="检查点保存目录")
    
    args = parser.parse_args()
    
    main(args)
