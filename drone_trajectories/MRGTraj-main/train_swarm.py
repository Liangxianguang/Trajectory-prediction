"""
MRGTraj 集群版本训练脚本
========================
用于训练改进的 MRGTrajSwarm 模型

使用方法:
  python train_swarm.py --num_agents 3 --data_dir swarm_segments --batch_size 32 --num_epochs 100
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

# 导入模型
from model_swarm import MRGTrajSwarm

# 尝试导入 TensorBoard，如果失败则跳过
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except Exception as e:
    logging.warning(f"无法导入 TensorBoard: {e}")
    HAS_TENSORBOARD = False
    SummaryWriter = None

# 简单的日志记录工具
class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SwarmDataLoader:
    """无人机集群数据加载器"""
    
    def __init__(self, npz_input_file, npz_output_file, obs_len, pred_len, batch_size, shuffle=True):
        """
        Args:
            npz_input_file: 输入 NPZ 文件路径
            npz_output_file: 输出 NPZ 文件路径
            obs_len: 观察长度
            pred_len: 预测长度
            batch_size: 批处理大小
            shuffle: 是否打乱数据
        """
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
        
        # 生成索引
        self.indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(self.indices)
        
        logger.info(f"  样本数: {self.num_samples}")
        logger.info(f"  无人机数: {self.num_agents}")
    
    def __iter__(self):
        """迭代器"""
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            # 获取批数据
            X_batch = self.X[:, batch_indices, :, :]  # (obs_len, batch_size, num_agents, 3)
            Y_batch = self.Y[:, batch_indices, :, :]  # (pred_len, batch_size, num_agents, 3)
            
            # 转置为 (batch_size, seq_len, num_agents, 3)
            X_batch = X_batch.transpose(1, 0, 2, 3)  # (batch_size, obs_len, num_agents, 3)
            Y_batch = Y_batch.transpose(1, 0, 2, 3)  # (batch_size, pred_len, num_agents, 3)
            
            # 转换为 torch tensor
            X_batch = torch.from_numpy(X_batch).float()
            Y_batch = torch.from_numpy(Y_batch).float()
            
            yield X_batch, Y_batch
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def create_data_loaders(args):
    """创建数据加载器"""
    data_dir = Path(args.data_dir)
    
    # 构建文件路径
    train_input_file = data_dir / f'input_agents_{args.num_agents}_subset.npz'
    train_output_file = data_dir / f'output_agents_{args.num_agents}_subset.npz'
    
    if not train_input_file.exists():
        train_input_file = data_dir / f'input_agents_{args.num_agents}.npz'
        train_output_file = data_dir / f'output_agents_{args.num_agents}.npz'
    
    if not train_input_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {train_input_file}")
    
    logger.info(f"使用数据文件:")
    logger.info(f"  输入: {train_input_file}")
    logger.info(f"  输出: {train_output_file}")
    
    # 创建加载器
    train_loader = SwarmDataLoader(
        train_input_file,
        train_output_file,
        args.obs_len,
        args.pred_len,
        args.batch_size,
        shuffle=True
    )
    
    return train_loader


def l2_loss(pred, target):
    """L2 损失"""
    return ((pred - target) ** 2).mean()


def kl_divergence_loss(mu, log_var):
    """KL 散度损失"""
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kld / mu.shape[0]


def train_epoch(model, train_loader, optimizer, epoch, writer, args):
    """训练一个 epoch"""
    model.train()
    losses = AverageMeter("Loss", ":.6f")
    l2_losses = AverageMeter("L2Loss", ":.6f")
    kl_losses = AverageMeter("KLLoss", ":.6f")
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    
    for batch_idx, (past_traj, future_traj) in enumerate(pbar):
        # 移到 GPU
        past_traj = past_traj.cuda()
        future_traj = future_traj.cuda()
        
        # 前向传播
        pred_traj, mu, log_var = model(past_traj, future_traj)
        
        # 计算损失
        l2_loss_val = l2_loss(pred_traj, future_traj)
        kl_loss_val = kl_divergence_loss(mu, log_var)
        
        # 加权组合
        total_loss = l2_loss_val + args.kl_weight * kl_loss_val
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 记录
        losses.update(total_loss.item())
        l2_losses.update(l2_loss_val.item())
        kl_losses.update(kl_loss_val.item())
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.6f}',
            'l2': f'{l2_losses.avg:.6f}',
            'kl': f'{kl_losses.avg:.6f}'
        })
    
    # 写入 tensorboard
    if writer is not None:
        writer.add_scalar("train/loss", losses.avg, epoch)
        writer.add_scalar("train/l2_loss", l2_losses.avg, epoch)
        writer.add_scalar("train/kl_loss", kl_losses.avg, epoch)
    
    logger.info(f"Epoch {epoch+1} - Loss: {losses.avg:.6f}, L2: {l2_losses.avg:.6f}, KL: {kl_losses.avg:.6f}")
    
    return losses.avg


def main(args):
    """主函数"""
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    # 创建检查点目录
    checkpoint_dir = Path(args.checkpoint_dir) / f"agents_{args.num_agents}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = checkpoint_dir / f"train_agents_{args.num_agents}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )
    
    logger.info("=" * 80)
    logger.info("MRGTraj 集群版本训练")
    logger.info("=" * 80)
    logger.info(f"配置:")
    logger.info(f"  无人机数量: {args.num_agents}")
    logger.info(f"  观察长度: {args.obs_len}")
    logger.info(f"  预测长度: {args.pred_len}")
    logger.info(f"  批处理大小: {args.batch_size}")
    logger.info(f"  训练 epochs: {args.num_epochs}")
    logger.info(f"  学习率: {args.lr}")
    logger.info(f"  KL 权重: {args.kl_weight}")
    logger.info("=" * 80)
    
    # 创建数据加载器
    train_loader = create_data_loaders(args)
    
    # 创建模型
    logger.info("创建模型...")
    model = MRGTrajSwarm(args)
    model.cuda()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数:")
    logger.info(f"  总参数数: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # TensorBoard (可选)
    writer = None
    if HAS_TENSORBOARD:
        log_dir = checkpoint_dir / "logs"
        writer = SummaryWriter(str(log_dir))
        logger.info(f"TensorBoard 已启用，日志目录: {log_dir}")
    else:
        logger.info("TensorBoard 已禁用（protobuf 版本问题），仅保存文本日志")
    
    # 训练循环
    logger.info("开始训练...")
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        loss = train_epoch(model, train_loader, optimizer, epoch, writer, args)
        scheduler.step()
        
        # 保存最佳模型
        if loss < best_loss:
            best_loss = loss
            checkpoint_path = checkpoint_dir / f"best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'args': args
            }, checkpoint_path)
            logger.info(f"✓ 保存最佳模型: {checkpoint_path} (loss: {loss:.6f})")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'args': args
            }, checkpoint_path)
    
    if writer is not None:
        writer.close()
    
    logger.info("✓ 训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRGTraj 集群版本训练脚本")
    
    # 数据相关
    parser.add_argument("--data_dir", type=str, default="swarm_segments",
                        help="数据目录")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="无人机数量 (3-6)")
    
    # 模型参数
    parser.add_argument("--d_model", type=int, default=256,
                        help="模型维度")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="注意力头数")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Transformer 层数")
    parser.add_argument("--noise_dim", type=int, default=64,
                        help="噪声/隐式编码维度")
    parser.add_argument("--agent_dim", type=int, default=3,
                        help="单个智能体维度 (XYZ)")
    
    # 序列参数
    parser.add_argument("--obs_len", type=int, default=20,
                        help="观察序列长度")
    parser.add_argument("--pred_len", type=int, default=10,
                        help="预测序列长度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批处理大小")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="训练 epochs 数")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="KL 散度损失权重")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每 N 个 epoch 保存一次检查点")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--gpu_num", type=str, default="0",
                        help="GPU 编号")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_swarm",
                        help="检查点保存目录")
    
    args = parser.parse_args()
    
    main(args)
