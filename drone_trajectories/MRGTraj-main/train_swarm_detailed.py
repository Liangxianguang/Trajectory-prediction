"""
MRGTraj Swarm Version - Enhanced Training Script
================================================
Detailed loss decomposition and evaluation metrics

Usage:
  python train_swarm_detailed.py --num_agents 3 --batch_size 32 --num_epochs 100
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

# Import model
from model_swarm import MRGTrajSwarm

# Try to import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except Exception as e:
    logging.warning(f"TensorBoard import failed: {e}")
    HAS_TENSORBOARD = False
    SummaryWriter = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Detailed metrics tracker"""
    
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
        """Get average values for all metrics"""
        averages = {}
        for key, vals in self.metrics.items():
            if vals:
                averages[key] = np.mean(vals)
        return averages
    
    def format_metrics(self):
        """Format metrics as string"""
        averages = self.get_averages()
        parts = []
        for key in sorted(averages.keys()):
            val = averages[key]
            parts.append(f"{key}={val:.6f}")
        return " ".join(parts)


class SwarmDataLoader:
    """UAV swarm data loader"""
    
    def __init__(self, npz_input_file, npz_output_file, obs_len, pred_len, batch_size, shuffle=True):
        logger.info(f"Loading data: {npz_input_file}, {npz_output_file}")
        
        self.X = np.load(npz_input_file)['data']  # (obs_len, num_samples, num_agents, 3)
        self.Y = np.load(npz_output_file)['data']  # (pred_len, num_samples, num_agents, 3)
        
        logger.info(f"  X shape: {self.X.shape}")
        logger.info(f"  Y shape: {self.Y.shape}")
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.num_samples = self.X.shape[1]
        self.num_agents = self.X.shape[2]
        
        # Generate indices
        self.indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(self.indices)
        
        logger.info(f"  Total samples: {self.num_samples}")
        logger.info(f"  Number of agents: {self.num_agents}")
    
    def __iter__(self):
        """Iterator"""
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
    """Create data loaders"""
    data_dir = Path(args.data_dir)
    
    # Resolve data directory path
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
        logger.error("Data files not found. Attempted paths:")
        logger.error(f"  {data_dir / f'input_agents_{args.num_agents}_subset.npz'}")
        logger.error(f"  {data_dir / f'input_agents_{args.num_agents}.npz'}")
        raise FileNotFoundError(f"Data files not found: {train_input_file}")
    
    logger.info("Using data files:")
    logger.info(f"  Input: {train_input_file}")
    logger.info(f"  Output: {train_output_file}")
    
    train_loader = SwarmDataLoader(
        train_input_file,
        train_output_file,
        args.obs_len,
        args.pred_len,
        args.batch_size,
        shuffle=True
    )
    
    return train_loader


class LossFunctionSet:
    """Loss functions set"""
    
    @staticmethod
    def l2_loss(pred, target, reduction='mean'):
        """L2 reconstruction loss"""
        loss = ((pred - target) ** 2)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss
    
    @staticmethod
    def ade_loss(pred, target):
        """Average Displacement Error as loss (direct optimization)"""
        # pred: (batch, pred_len, num_agents, 3)
        # target: (batch, pred_len, num_agents, 3)
        diff = torch.norm(pred - target, dim=-1)  # (batch, pred_len, num_agents)
        return diff.mean()  # Mean across all timesteps and agents
    
    @staticmethod
    def fde_loss(pred, target):
        """Final Displacement Error as loss (direct optimization)"""
        # Only penalize the final timestep
        diff = torch.norm(pred[:, -1, :, :] - target[:, -1, :, :], dim=-1)  # (batch, num_agents)
        return diff.mean()
    
    @staticmethod
    def position_loss(pred, target):
        """Position loss (XY plane only)"""
        pred_xy = pred[..., :2]
        target_xy = target[..., :2]
        return ((pred_xy - target_xy) ** 2).mean()
    
    @staticmethod
    def height_loss(pred, target):
        """Height loss (Z coordinate)"""
        pred_z = pred[..., 2:3]
        target_z = target[..., 2:3]
        return ((pred_z - target_z) ** 2).mean()
    
    @staticmethod
    def velocity_loss(pred, target):
        """Velocity loss (smooth motion)"""
        pred_vel = torch.diff(pred, dim=1)  # (batch, pred_len-1, agents, 3)
        target_vel = torch.diff(target, dim=1)
        return ((pred_vel - target_vel) ** 2).mean()
    
    @staticmethod
    def acceleration_loss(pred, target):
        """Acceleration loss (smoother trajectories)"""
        if pred.shape[1] < 3:
            return torch.tensor(0.0, device=pred.device)
        
        pred_acc = torch.diff(torch.diff(pred, dim=1), dim=1)
        target_acc = torch.diff(torch.diff(target, dim=1), dim=1)
        return ((pred_acc - target_acc) ** 2).mean()
    
    @staticmethod
    def collision_loss(pred):
        """Collision avoidance (min distance penalty)"""
        # pred: (batch, pred_len, num_agents, 3)
        batch_size, pred_len, num_agents, _ = pred.shape
        
        if num_agents < 2:
            return torch.tensor(0.0, device=pred.device)
        
        min_distance = 0.5  # Minimum allowed distance (meters)
        loss = 0
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = torch.norm(pred[:, :, i, :] - pred[:, :, j, :], dim=-1)
                violation = torch.clamp(min_distance - dist, min=0)
                loss += violation.mean()
        
        return loss / (num_agents * (num_agents - 1) / 2)
    
    @staticmethod
    def kl_divergence_loss(mu, log_var):
        """KL divergence loss (variational regularization)"""
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kld / mu.shape[0]
    
    @staticmethod
    def formation_loss(pred):
        """Formation loss (maintain relative positions)"""
        # Calculate variance of relative positions
        batch_size, pred_len, num_agents, _ = pred.shape
        
        if num_agents < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute center of all agents
        center = pred.mean(dim=2, keepdim=True)
        
        # Relative position
        relative_pos = pred - center  # (batch, pred_len, num_agents, 3)
        
        # Change in relative position
        relative_vel = torch.diff(relative_pos, dim=1)
        
        # Formation loss: encourage stable relative positions
        loss = (relative_vel ** 2).mean()
        
        return loss


class ADE_FDE_Calculator:
    """ADE and FDE calculator"""
    
    @staticmethod
    def ade(pred, target):
        """Average displacement error"""
        # pred: (batch, pred_len, num_agents, 3)
        diff = torch.norm(pred - target, dim=-1)  # (batch, pred_len, num_agents)
        return diff.mean().item()
    
    @staticmethod
    def fde(pred, target):
        """Final displacement error"""
        diff = torch.norm(pred[:, -1, :, :] - target[:, -1, :, :], dim=-1)  # (batch, num_agents)
        return diff.mean().item()
    
    @staticmethod
    def ade_per_agent(pred, target):
        """ADE per agent"""
        diff = torch.norm(pred - target, dim=-1)  # (batch, pred_len, num_agents)
        return diff.mean(dim=(0, 1)).cpu().numpy()  # (num_agents,)
    
    @staticmethod
    def best_ade_fde(predictions, target, num_samples=10):
        """Best ADE/FDE from multi-sample predictions"""
        if predictions.dim() == 4:  # (batch, pred_len, num_agents, 3)
            predictions = predictions.unsqueeze(0)  # (1, batch, pred_len, num_agents, 3)
        
        # predictions: (num_samples, batch, pred_len, num_agents, 3)
        ades = []
        fdes = []
        
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            ade = ADE_FDE_Calculator.ade(pred, target)
            fde = ADE_FDE_Calculator.fde(pred, target)
            ades.append(ade)
            fdes.append(fde)
        
        return min(ades), min(fdes)


def train_epoch(model, train_loader, optimizer, epoch, args, metrics_tracker):
    """Train one epoch"""
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", ncols=100)
    
    loss_fn = LossFunctionSet()
    calculator = ADE_FDE_Calculator()
    
    for batch_idx, (past_traj, future_traj) in enumerate(pbar):
        past_traj = past_traj.cuda()
        future_traj = future_traj.cuda()
        
        # Forward pass
        pred_traj, mu, log_var = model(past_traj, future_traj)
        
        # Compute loss components
        l2_loss_val = loss_fn.l2_loss(pred_traj, future_traj)
        ade_loss_val = loss_fn.ade_loss(pred_traj, future_traj)  # NEW: Direct ADE optimization
        fde_loss_val = loss_fn.fde_loss(pred_traj, future_traj)  # NEW: Direct FDE optimization
        position_loss_val = loss_fn.position_loss(pred_traj, future_traj)
        height_loss_val = loss_fn.height_loss(pred_traj, future_traj)
        velocity_loss_val = loss_fn.velocity_loss(pred_traj, future_traj)
        acceleration_loss_val = loss_fn.acceleration_loss(pred_traj, future_traj)
        collision_loss_val = loss_fn.collision_loss(pred_traj)
        formation_loss_val = loss_fn.formation_loss(pred_traj)
        kl_loss_val = loss_fn.kl_divergence_loss(mu, log_var)
        
        # Weighted combination (prioritize ADE/FDE optimization)
        total_loss = (
            args.ade_weight * ade_loss_val +
            args.fde_weight * fde_loss_val +
            l2_loss_val +
            args.pos_weight * position_loss_val +
            args.height_weight * height_loss_val +
            args.vel_weight * velocity_loss_val +
            args.acc_weight * acceleration_loss_val +
            args.collision_weight * collision_loss_val +
            args.formation_weight * formation_loss_val +
            args.kl_weight * kl_loss_val
        )
        
        # Compute ADE/FDE
        ade_val = calculator.ade(pred_traj, future_traj)
        fde_val = calculator.fde(pred_traj, future_traj)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        metrics_tracker.update(
            total_loss=total_loss.item(),
            ade_loss=ade_loss_val.item(),
            fde_loss=fde_loss_val.item(),
            l2_loss=l2_loss_val.item(),
            pos_loss=position_loss_val.item(),
            height_loss=height_loss_val.item(),
            vel_loss=velocity_loss_val.item(),
            acc_loss=acceleration_loss_val.item(),
            collision_loss=collision_loss_val.item(),
            formation_loss=formation_loss_val.item(),
            kl_loss=kl_loss_val.item(),
            ade=ade_val,
            fde=fde_val
        )
        
        # Update progress bar
        if batch_idx % 5 == 0:
            pbar.set_postfix_str(metrics_tracker.format_metrics())
    
    return metrics_tracker.get_averages()


def validate(model, val_loader, args):
    """Validation"""
    model.eval()
    
    loss_fn = LossFunctionSet()
    calculator = ADE_FDE_Calculator()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for past_traj, future_traj in val_loader:
            past_traj = past_traj.cuda()
            future_traj = future_traj.cuda()
            
            # Forward pass
            pred_traj, mu, log_var = model(past_traj, future_traj)
            
            # Compute loss
            l2_loss_val = loss_fn.l2_loss(pred_traj, future_traj)
            kl_loss_val = loss_fn.kl_divergence_loss(mu, log_var)
            total_loss = l2_loss_val + args.kl_weight * kl_loss_val
            
            # Compute best ADE/FDE from multi-sample predictions
            with torch.no_grad():
                predictions_list = []
                for _ in range(10):  # Generate 10 samples
                    pred = model.inference(past_traj, num_samples=1)
                    predictions_list.append(pred.squeeze(0))
                
                predictions = torch.stack(predictions_list, dim=0)
                best_ade, best_fde = calculator.best_ade_fde(predictions, future_traj)
            
            ade_val = calculator.ade(pred_traj, future_traj)
            fde_val = calculator.fde(pred_traj, future_traj)
            
            metrics_tracker.update(
                val_loss=total_loss.item(),
                val_ade=ade_val,
                val_fde=fde_val,
                val_best_ade=best_ade,
                val_best_fde=best_fde
            )
    
    return metrics_tracker.get_averages()


def main(args):
    """Main training function"""
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / f"agents_{args.num_agents}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging file
    log_file = checkpoint_dir / f"train_agents_{args.num_agents}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("=" * 100)
    logger.info("MRGTraj Swarm Version - Detailed Training")
    logger.info("=" * 100)
    
    # Print configuration
    logger.info("Configuration Parameters:")
    for key, val in vars(args).items():
        logger.info(f"  {key}: {val}")
    logger.info("=" * 100)
    
    # Create data loaders
    logger.info("\nLoading data...")
    train_loader = create_data_loaders(args)
    logger.info(f"[OK] Data loaded successfully")
    logger.info(f"  Total samples: {train_loader.num_samples}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Total batches per epoch: {len(train_loader)}")
    
    # Create model
    logger.info("\nCreating model...")
    model = MRGTrajSwarm(args)
    model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[OK] Model created successfully")
    logger.info(f"  Model architecture: MRGTrajSwarm")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    logger.info(f"\nOptimizer configuration:")
    logger.info(f"  Type: Adam")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Scheduler: CosineAnnealing (T_max={args.num_epochs})")
    
    # TensorBoard
    if HAS_TENSORBOARD:
        log_dir = checkpoint_dir / "logs"
        writer = SummaryWriter(log_dir)
        logger.info(f"\nTensorBoard enabled")
        logger.info(f"  Log directory: {log_dir}")
    else:
        writer = None
        logger.warning("\nTensorBoard not available - metrics will only be saved to file")
    
    logger.info("\n" + "=" * 100)
    logger.info("Starting training...")
    logger.info("=" * 100 + "\n")
    
    best_loss = float('inf')
    best_ade = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train
        metrics_tracker = MetricsTracker()
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, args, metrics_tracker)
        scheduler.step()
        
        # Log training metrics
        log_msg = f"[Epoch {epoch+1}/{args.num_epochs}] "
        
        # Detailed metrics
        metrics_parts = []
        for key in sorted(train_metrics.keys()):
            v = train_metrics[key]
            metrics_parts.append(f"{key}={v:.6f}")
        
        log_msg += " | ".join(metrics_parts)
        logger.info(log_msg)
        
        if writer:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if train_metrics.get('total_loss', float('inf')) < best_loss:
            best_loss = train_metrics['total_loss']
            checkpoint_path = checkpoint_dir / f"best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': args
            }, checkpoint_path)
            logger.info(f"  [BEST] Saved best model - loss={best_loss:.6f}")
        
        # Save checkpoints every N epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_metrics.get('total_loss', 0),
                'args': args
            }, checkpoint_path)
            logger.info(f"  [CHECKPOINT] Saved model at epoch {epoch+1}")
    
    if writer:
        writer.close()
    
    logger.info("\n" + "=" * 100)
    logger.info("[COMPLETE] Training finished!")
    logger.info(f"  Best loss: {best_loss:.6f}")
    logger.info(f"  Checkpoint directory: {checkpoint_dir}")
    logger.info("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRGTraj Swarm Version - Detailed Training Script")
    
    # Data related
    parser.add_argument("--data_dir", type=str, default="../Cluster trajectory/swarm_segments",
                        help="Data directory path")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Number of UAVs")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of Transformer layers")
    parser.add_argument("--noise_dim", type=int, default=64,
                        help="Noise dimension")
    parser.add_argument("--agent_dim", type=int, default=3,
                        help="Per-agent feature dimension")
    
    # Sequence parameters
    parser.add_argument("--obs_len", type=int, default=20,
                        help="Observation sequence length")
    parser.add_argument("--pred_len", type=int, default=10,
                        help="Prediction sequence length")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    
    # Loss weights
    parser.add_argument("--ade_weight", type=float, default=1.0,
                        help="ADE loss weight (direct optimization)")
    parser.add_argument("--fde_weight", type=float, default=0.5,
                        help="FDE loss weight (final position focus)")
    parser.add_argument("--kl_weight", type=float, default=0.05,
                        help="KL divergence weight")
    parser.add_argument("--pos_weight", type=float, default=0.3,
                        help="Position loss weight (XY plane)")
    parser.add_argument("--height_weight", type=float, default=0.2,
                        help="Height loss weight (Z coordinate)")
    parser.add_argument("--vel_weight", type=float, default=0.1,
                        help="Velocity loss weight")
    parser.add_argument("--acc_weight", type=float, default=0.05,
                        help="Acceleration loss weight")
    parser.add_argument("--collision_weight", type=float, default=0.3,
                        help="Collision avoidance weight")
    parser.add_argument("--formation_weight", type=float, default=0.1,
                        help="Formation constraint weight")
    
    # Others
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu_num", type=str, default="0",
                        help="GPU device number")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_swarm100",
                        help="Checkpoint save directory")
    
    args = parser.parse_args()
    
    main(args)
