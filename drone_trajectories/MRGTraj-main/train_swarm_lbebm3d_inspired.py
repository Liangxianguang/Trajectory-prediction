#!/usr/bin/env python3
"""
MRGTraj 融合 LBEBM3D 经验 - 优化版
====================================

关键改进点（从 LBEBM3D 汲取）:

1. **双LSTM层级架构**
   - LBEBM3D使用双LSTM（社交+物理）
   - 改进: 添加Agent-Level LSTM + Swarm-Level LSTM
   
2. **显式的多尺度建模**
   - 局部邻域 + 全局集群
   - 改进: 分层注意力 (Local Social + Global Motion)
   
3. **损失函数设计**
   - 直接ADE/FDE优化而不是重建损失
   - 改进: ADE焦点 > L2重建 > KL分布
   
4. **运动约束**
   - LBEBM3D显式建模物理约束
   - 改进: 速度一致性 + 平滑性 + 方向连续性
   
5. **学习率调度**
   - 保守的decay策略
   - 改进: Warmup + 分阶段学习率调整
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import random
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from model_swarm import MRGTrajSwarm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics = {}
    
    def reset(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.metrics[key].append(val)
    
    def get_averages(self):
        """获取平均值"""
        averages = {}
        for key, vals in self.metrics.items():
            averages[key] = np.mean(vals) if vals else 0.0
        return averages


class SwarmDataLoader:
    """无人机集群数据加载器"""
    
    def __init__(self, npz_input_file, npz_output_file, obs_len, pred_len, batch_size, shuffle=True):
        logger.info(f"加载数据: {npz_input_file}, {npz_output_file}")
        
        self.X = np.load(npz_input_file)['data']  # (obs_len, num_samples, num_agents, 3)
        self.Y = np.load(npz_output_file)['data']  # (pred_len, num_samples, num_agents, 3)
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.num_samples = self.X.shape[1]
        self.num_agents = self.X.shape[2]
        
        logger.info(f"  数据形状 X: {self.X.shape}, Y: {self.Y.shape}")
        logger.info(f"  总样本数: {self.num_samples}, 无人机数: {self.num_agents}")
        
        self.indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            past_traj = torch.from_numpy(self.X[:, batch_indices, :, :]).float()
            future_traj = torch.from_numpy(self.Y[:, batch_indices, :, :]).float()

            # 转换为 batch-first: (batch, obs_len, num_agents, 3)
            past_traj = past_traj.permute(1, 0, 2, 3).contiguous()
            future_traj = future_traj.permute(1, 0, 2, 3).contiguous()
            
            yield past_traj, future_traj
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class LBEBMLikeConstrainedLosses:
    """
    受LBEBM3D启发的损失函数
    
    LBEBM3D核心思想:
    - Local: 智能体之间的相互影响
    - Physical: 物理运动约束
    - 显式的多尺度建模
    """
    
    @staticmethod
    def ade_loss(pred, target):
        """ADE 损失 - 轨迹平均偏差"""
        # pred, target: (batch, pred_len, num_agents, 3)
        euclidean_dist = torch.norm(pred - target, dim=-1)  # (batch, pred_len, num_agents)
        return torch.mean(euclidean_dist)
    
    @staticmethod
    def fde_loss(pred, target):
        """FDE 损失 - 最终位置偏差"""
        # 只看最后一帧: (batch, num_agents)
        final_dist = torch.norm(pred[:, -1, :, :] - target[:, -1, :, :], dim=-1)
        return torch.mean(final_dist)
    
    @staticmethod
    def velocity_consistency_loss(pred, target):
        """速度一致性损失（从LBEBM3D）
        
        约束预测轨迹的速度变化与目标一致
        """
        # 计算速度 (相邻帧的差): (batch, pred_len-1, num_agents, 3)
        pred_vel = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_vel = target[:, 1:, :, :] - target[:, :-1, :, :]
        
        vel_dist = torch.norm(pred_vel - target_vel, dim=-1)  # (batch, pred_len-1, num_agents)
        return torch.mean(vel_dist)
    
    @staticmethod
    def smoothness_loss(pred):
        """平滑性损失（防止抖动）
        
        约束相邻帧的加速度变化平缓
        """
        # 加速度 = 速度的变化: (batch, pred_len-2, num_agents, 3)
        vel = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        acc = vel[:, 1:, :, :] - vel[:, :-1, :, :]
        
        acc_magnitude = torch.norm(acc, dim=-1)  # (batch, pred_len-2, num_agents)
        return torch.mean(acc_magnitude)
    
    @staticmethod
    def collision_avoidance_loss(pred, min_distance=0.5):
        """碰撞避免损失（软约束）
        
        从LBEBM3D: 鼓励无人机之间保持最小距离
        """
        # pred: (batch, pred_len, num_agents, 3)
        batch, pred_len, num_agents, _ = pred.shape
        
        loss = 0
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # 计算智能体i和j之间的距离: (batch, pred_len)
                dist = torch.norm(pred[:, :, i, :] - pred[:, :, j, :], dim=-1)
                # 如果距离小于阈值，施加惩罚
                penalty = torch.clamp(min_distance - dist, min=0.0)
                loss = loss + torch.mean(penalty)
        
        return loss / (num_agents * (num_agents - 1) / 2) if num_agents > 1 else torch.tensor(0.0, device=pred.device)
    
    @staticmethod
    def formation_preservation_loss(pred, target):
        """编队保持损失（从LBEBM3D）
        
        约束多无人机的相对位置保持
        """
        # 计算质心: (batch, pred_len, 1, 3)
        pred_centroid = torch.mean(pred, dim=2, keepdim=True)
        target_centroid = torch.mean(target, dim=2, keepdim=True)
        
        # 相对于质心的位置: (batch, pred_len, num_agents, 3)
        pred_relative = pred - pred_centroid
        target_relative = target - target_centroid
        
        # 编队应该保持相对位置
        formation_dist = torch.norm(pred_relative - target_relative, dim=-1)
        return torch.mean(formation_dist)


class LBEBM3D_Inspired_Scheduler:
    """
    从LBEBM3D汲取的学习率调度
    
    策略: 保守的衰减，避免过快的学习率下降
    """
    
    @staticmethod
    def get_learning_rate(epoch, total_epochs, base_lr=0.001):
        """
        分阶段学习率:
        - 0-20%: Warmup (线性增长)
        - 20%-70%: 高速学习 (基础学习率)
        - 70%-100%: 精细调整 (指数衰减)
        """
        if epoch < 0.2 * total_epochs:
            # Warmup 阶段
            return base_lr * (epoch / (0.2 * total_epochs))
        elif epoch < 0.7 * total_epochs:
            # 高速学习阶段
            return base_lr
        else:
            # 衰减阶段
            decay_progress = (epoch - 0.7 * total_epochs) / (0.3 * total_epochs)
            return base_lr * np.exp(-3.0 * decay_progress)


class ADE_FDE_Calculator:
    """ADE/FDE 计算器"""
    
    @staticmethod
    def ade(pred, target):
        """计算平均位移误差 (batch-first format)"""
        # pred, target: (batch, pred_len, num_agents, 3)
        euclidean_dist = torch.norm(pred - target, dim=-1)
        return torch.mean(euclidean_dist).item()
    
    @staticmethod
    def fde(pred, target):
        """计算最终位置误差 (batch-first format)"""
        # Only penalize the final timestep
        final_dist = torch.norm(pred[:, -1, :, :] - target[:, -1, :, :], dim=-1)
        return torch.mean(final_dist).item()


def get_loss_weights_schedule(epoch, total_epochs):
    """
    从LBEBM3D启发的损失权重调度
    
    策略: 逐步从多目标转向ADE/FDE焦点
    """
    progress = epoch / total_epochs
    
    # ADE: 从 1.0 增到 3.0（主要优化目标）
    ade_w = 1.0 + 2.0 * progress
    
    # FDE: 从 0.5 增到 2.0（末端精度）
    fde_w = 0.5 + 1.5 * progress
    
    # Velocity: 保持稳定 0.1
    vel_w = 0.1
    
    # Smoothness: 从 0.1 增到 0.3（防止抖动）
    smooth_w = 0.1 + 0.2 * progress
    
    # Collision: 从 0.2 衰减到 0.05（软约束）
    collision_w = 0.2 * (1 - 0.75 * progress)
    
    # Formation: 从 0.1 衰减到 0.02（编队维护）
    formation_w = 0.1 * (1 - 0.8 * progress)
    
    return {
        'ade': ade_w,
        'fde': fde_w,
        'vel': vel_w,
        'smooth': smooth_w,
        'collision': collision_w,
        'formation': formation_w
    }


def train_epoch(model, train_loader, optimizer, epoch, args, total_epochs):
    """训练一个 epoch（融合LBEBM3D思想）"""
    model.train()
    metrics = MetricsTracker()
    
    loss_fn = LBEBMLikeConstrainedLosses()
    calculator = ADE_FDE_Calculator()
    loss_weights = get_loss_weights_schedule(epoch, total_epochs)
    
    # 设置动态学习率
    lr = LBEBM3D_Inspired_Scheduler.get_learning_rate(epoch, total_epochs, args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", ncols=100)
    
    for batch_idx, (past_traj, future_traj) in enumerate(pbar):
        past_traj = past_traj.cuda()      # (batch, obs_len, num_agents, 3)
        future_traj = future_traj.cuda()  # (batch, pred_len, num_agents, 3)
        
        # 前向传播
        optimizer.zero_grad()
        pred, mu, log_var = model(past_traj, future_traj)  # (batch, pred_len, num_agents, 3)
        
        # 计算损失（融合LBEBM3D多目标）
        ade_loss = loss_fn.ade_loss(pred, future_traj)
        fde_loss = loss_fn.fde_loss(pred, future_traj)
        vel_loss = loss_fn.velocity_consistency_loss(pred, future_traj)
        smooth_loss = loss_fn.smoothness_loss(pred)
        collision_loss = loss_fn.collision_avoidance_loss(pred, min_distance=0.8)
        formation_loss = loss_fn.formation_preservation_loss(pred, future_traj)
        
        # 加权总损失
        total_loss = (
            loss_weights['ade'] * ade_loss +
            loss_weights['fde'] * fde_loss +
            loss_weights['vel'] * vel_loss +
            loss_weights['smooth'] * smooth_loss +
            loss_weights['collision'] * collision_loss +
            loss_weights['formation'] * formation_loss
        )
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 记录指标
        metrics.update(
            ade_loss=ade_loss,
            fde_loss=fde_loss,
            vel_loss=vel_loss,
            smooth_loss=smooth_loss,
            collision_loss=collision_loss,
            formation_loss=formation_loss,
            total_loss=total_loss
        )
        
        # 计算实际ADE/FDE
        ade = calculator.ade(pred.detach(), future_traj)
        fde = calculator.fde(pred.detach(), future_traj)
        metrics.update(ade=ade, fde=fde)
        
        # 更新进度条
        avg_metrics = metrics.get_averages()
        pbar.set_postfix({
            'ade': f"{avg_metrics.get('ade', 0):.4f}",
            'fde': f"{avg_metrics.get('fde', 0):.4f}",
            'loss': f"{avg_metrics.get('total_loss', 0):.4f}"
        })
    
    return metrics.get_averages()


def create_data_loaders(args):
    """创建数据加载器"""
    data_dir = Path(args.data_dir)
    
    if not data_dir.is_absolute():
        possible_paths = [
            Path(__file__).parent.parent / data_dir,
            Path(__file__).parent / data_dir,
            data_dir
        ]
        for p in possible_paths:
            if p.exists():
                data_dir = p
                break
    
    input_file = data_dir / f'input_agents_{args.num_agents}_subset.npz'
    output_file = data_dir / f'output_agents_{args.num_agents}_subset.npz'
    
    if not input_file.exists():
        input_file = data_dir / f'input_agents_{args.num_agents}.npz'
        output_file = data_dir / f'output_agents_{args.num_agents}.npz'
    
    if not input_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {input_file}")
    
    logger.info(f"使用数据: {input_file}, {output_file}")
    
    train_loader = SwarmDataLoader(
        input_file, output_file,
        args.obs_len, args.pred_len,
        args.batch_size, shuffle=True
    )
    
    return train_loader


def main(args):
    """主训练函数"""
    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    # 检查点目录
    checkpoint_dir = Path(args.checkpoint_dir) / f"agents_{args.num_agents}_lbebm3d_inspired"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = checkpoint_dir / "train.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("=" * 100)
    logger.info("MRGTraj - 融合LBEBM3D经验的优化版")
    logger.info("=" * 100)
    
    # 打印配置
    logger.info("\n配置参数:")
    for key, val in vars(args).items():
        logger.info(f"  {key}: {val}")
    logger.info("=" * 100)
    
    # 加载数据
    logger.info("\n加载数据...")
    train_loader = create_data_loaders(args)
    
    # 创建模型
    logger.info("\n创建模型...")
    model = MRGTrajSwarm(args)
    model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数: {total_params:,}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    logger.info(f"\n优化器: Adam, lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info("\n" + "=" * 100)
    logger.info("开始训练...")
    logger.info("=" * 100 + "\n")
    
    best_ade = float('inf')
    best_epoch = 0
    no_improve_count = 0
    
    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, args, args.num_epochs)
        
        # 日志
        logger.info(f"[Epoch {epoch+1}/{args.num_epochs}]")
        logger.info(f"  ADE: {train_metrics.get('ade', 0):.6f}m")
        logger.info(f"  FDE: {train_metrics.get('fde', 0):.6f}m")
        logger.info(f"  Loss: {train_metrics.get('total_loss', 0):.6f}")
        
        # 检查改进
        if train_metrics.get('ade', float('inf')) < best_ade:
            best_ade = train_metrics.get('ade', 0)
            best_epoch = epoch
            no_improve_count = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ade': best_ade
            }, checkpoint_dir / 'best_model.pth')
            
            logger.info(f"  ✓ [BEST] ADE下降到 {best_ade:.6f}m")
        else:
            no_improve_count += 1
        
        # 早停
        if no_improve_count >= args.patience:
            logger.info(f"\n早停触发！最佳ADE: {best_ade:.6f}m (Epoch {best_epoch+1})")
            break
    
    logger.info("\n" + "=" * 100)
    logger.info("[完成] 训练结束!")
    logger.info(f"  最佳 ADE: {best_ade:.6f}m")
    logger.info(f"  最佳 Epoch: {best_epoch+1}")
    logger.info(f"  检查点: {checkpoint_dir}")
    logger.info("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRGTraj - 融合LBEBM3D经验")
    
    # 数据
    parser.add_argument("--data_dir", type=str, default="../Cluster trajectory/swarm_segments",
                       help="数据目录")
    parser.add_argument("--num_agents", type=int, default=3, help="无人机数量")
    
    # 模型
    parser.add_argument("--d_model", type=int, default=256, help="模型维度")
    parser.add_argument("--n_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--n_layers", type=int, default=2, help="Transformer层数")
    parser.add_argument("--noise_dim", type=int, default=64, help="噪声维度")
    parser.add_argument("--agent_dim", type=int, default=3, help="智能体特征维度")
    
    # 序列
    parser.add_argument("--obs_len", type=int, default=20, help="观察长度")
    parser.add_argument("--pred_len", type=int, default=10, help="预测长度")
    
    # 训练
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gpu_num", type=str, default="0", help="GPU号")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_lbebm3d",
                       help="检查点目录")
    
    args = parser.parse_args()
    main(args)
