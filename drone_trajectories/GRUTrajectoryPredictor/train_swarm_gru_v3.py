"""
Swarm GRU Trajectory Predictor - Enhanced Training Script (v3)
============================================================
Enhanced version with comprehensive metrics and checkpoint management.

Metrics included:
  - MSE, RMSE, MAE (Error metrics)
  - MAPE (Mean Absolute Percentage Error)
  - ADE (Average Displacement Error)
  - FDE (Final Displacement Error)
  - R² Score

Usage:
    python train_swarm_gru_v3.py --num_agents 3 --num_epochs 50 --batch_size 64 --use_subset
    python train_swarm_gru_v3.py --num_agents 3 --num_epochs 100 --batch_size 32 --save_every 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import csv
import shutil

# Configure paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path(r"d:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments")
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Create directories
for d in [MODELS_DIR, RESULTS_DIR, LOG_DIR, CHECKPOINTS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============ Dataset Class ============
class SwarmTrajectoryDataset(Dataset):
    """Multi-agent trajectory dataset"""
    
    def __init__(self, X, Y, normalize=True, input_mean=None, input_std=None,
                 output_mean=None, output_std=None):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.normalize = normalize
        self.samples = len(X)
        
        if input_mean is None:
            self.input_mean = np.mean(self.X.reshape(-1, 3), axis=0)
            self.input_std = np.std(self.X.reshape(-1, 3), axis=0)
        else:
            self.input_mean = np.array(input_mean, dtype=np.float32)
            self.input_std = np.array(input_std, dtype=np.float32)
        
        self.input_std = np.where(self.input_std < 1e-8, 1.0, self.input_std)
        
        if output_mean is None:
            y_delta = self.Y - self.X[:, -1:, :, :]
            self.output_mean = np.mean(y_delta.reshape(-1, 3), axis=0)
            self.output_std = np.std(y_delta.reshape(-1, 3), axis=0)
        else:
            self.output_mean = np.array(output_mean, dtype=np.float32)
            self.output_std = np.array(output_std, dtype=np.float32)
        
        self.output_std = np.where(self.output_std < 1e-8, 1.0, self.output_std)
        
        logger.info(f"Dataset: {self.samples} samples")
        logger.info(f"  Input:  mean={self.input_mean}, std={self.input_std}")
        logger.info(f"  Output: mean={self.output_mean}, std={self.output_std}")
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx].copy()
        
        if self.normalize:
            x_norm = (x - self.input_mean) / self.input_std
        else:
            x_norm = x
        
        y_delta = y - x[-1:, :, :]
        
        if self.normalize:
            y_delta_norm = (y_delta - self.output_mean) / self.output_std
        else:
            y_delta_norm = y_delta
        
        return {
            'x': torch.from_numpy(x_norm).float(),
            'y': torch.from_numpy(y_delta_norm).float(),
            'x_last': torch.from_numpy(x[-1:, :, :]).float()
        }


# ============ Model ============
class SwarmGRUModel(nn.Module):
    """GRU encoder-decoder for trajectory prediction"""
    
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3,
                 num_layers=2, dropout=0.3, num_agents=3, seq_out=10):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_agents = num_agents
        self.seq_out = seq_out
        
        self.encoder = nn.GRU(
            input_size=input_dim * num_agents,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, input_dim * num_agents)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_in = x.size(1)
        
        x_flat = x.view(batch_size, seq_in, -1)
        _, h_n = self.encoder(x_flat)
        
        decoder_in = torch.zeros(batch_size, self.seq_out, self.hidden_dim,
                                device=x.device, dtype=x.dtype)
        
        decoder_out, _ = self.decoder(decoder_in, h_n)
        y_flat = self.fc(decoder_out)
        y = y_flat.view(batch_size, self.seq_out, self.num_agents, self.output_dim)
        
        return y


# ============ Loss Function ============
class CombinedLoss(nn.Module):
    """Position + velocity loss"""
    
    def __init__(self, pos_weight=1.0, vel_weight=0.5, dt=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.dt = dt
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        loss_pos = self.mse(y_pred, y_true)
        
        if y_pred.size(1) > 1:
            v_pred = (y_pred[:, 1:] - y_pred[:, :-1]) / self.dt
            v_true = (y_true[:, 1:] - y_true[:, :-1]) / self.dt
            loss_vel = self.mse(v_pred, v_true)
        else:
            loss_vel = torch.tensor(0.0, device=y_pred.device)
        
        loss = self.pos_weight * loss_pos + self.vel_weight * loss_vel
        return loss, loss_pos, loss_vel


# ============ Metrics Computation ============
def compute_comprehensive_metrics(y_pred, y_true):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_pred: (batch, seq_out, agents, 3) or reshaped predictions
        y_true: (batch, seq_out, agents, 3) or reshaped ground truth
    
    Returns:
        dict with all metrics
    """
    # Flatten to (N, 3)
    y_pred_flat = y_pred.reshape(-1, 3)
    y_true_flat = y_true.reshape(-1, 3)
    
    # Basic error metrics
    mse = np.mean((y_pred_flat - y_true_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_flat - y_true_flat))
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    denominator = np.abs(y_true_flat) + 1e-10
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / denominator)) * 100
    
    # R² Score
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat, axis=0, keepdims=True)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # ADE (Average Displacement Error) - L2 distance at each timestep
    # Reshape back to (batch, seq_out, agents, 3)
    if len(y_pred.shape) == 4:
        diff = y_pred - y_true  # (batch, seq_out, agents, 3)
        displacements = np.sqrt(np.sum(diff ** 2, axis=-1))  # (batch, seq_out, agents)
        ade = np.mean(displacements)
    else:
        diff = y_pred_flat - y_true_flat
        ade = np.mean(np.sqrt(np.sum(diff ** 2, axis=-1)))
    
    # FDE (Final Displacement Error) - L2 distance at final timestep
    if len(y_pred.shape) == 4:
        fde = np.mean(np.sqrt(np.sum((y_pred[:, -1] - y_true[:, -1]) ** 2, axis=-1)))
    else:
        # Assume last timestep is at indices seq_out-1
        fde = ade  # fallback
    
    # Per-coordinate metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'ade': ade,
        'fde': fde,
    }
    
    # Per-coordinate
    for i, coord in enumerate(['x', 'y', 'z']):
        metrics[f'mse_{coord}'] = np.mean((y_pred_flat[:, i] - y_true_flat[:, i]) ** 2)
        metrics[f'mae_{coord}'] = np.mean(np.abs(y_pred_flat[:, i] - y_true_flat[:, i]))
    
    return metrics


# ============ Training Functions ============
def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_pos = 0.0
    total_vel = 0.0
    all_preds = []
    all_trues = []
    count = 0
    
    for batch in tqdm(loader, desc='Train', leave=False):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        y_pred = model(x)
        loss, loss_pos, loss_vel = loss_fn(y_pred, y)
        
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        
        total_loss += loss.item()
        total_pos += loss_pos.item()
        total_vel += loss_vel.item()
        
        all_preds.append(y_pred.detach().cpu().numpy())
        all_trues.append(y.detach().cpu().numpy())
        count += 1
    
    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    metrics = compute_comprehensive_metrics(all_preds, all_trues)
    
    return {
        'loss': total_loss / count,
        'loss_pos': total_pos / count,
        'loss_vel': total_vel / count,
        **metrics
    }


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_pos = 0.0
    total_vel = 0.0
    all_preds = []
    all_trues = []
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Val', leave=False):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            y_pred = model(x)
            loss, loss_pos, loss_vel = loss_fn(y_pred, y)
            
            total_loss += loss.item()
            total_pos += loss_pos.item()
            total_vel += loss_vel.item()
            
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
            count += 1
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    metrics = compute_comprehensive_metrics(all_preds, all_trues)
    
    return {
        'loss': total_loss / count,
        'loss_pos': total_pos / count,
        'loss_vel': total_vel / count,
        **metrics
    }


# ============ Checkpoint Management ============
def save_checkpoint(state, is_best, checkpoint_dir, filename):
    """Save checkpoint"""
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    if is_best:
        best_filepath = checkpoint_dir / 'best_model.pth'
        shutil.copy(filepath, best_filepath)
        logger.info(f"  ✓ Best model updated")


def load_checkpoint(filepath, model, optimizer=None):
    """Load checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


# ============ Main Training ============
def main(args):
    logger.info("=" * 80)
    logger.info("Swarm GRU Training (Enhanced v3)")
    logger.info("=" * 80)
    logger.info(f"Arguments: {vars(args)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Device: {device}")
    
    # Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    logger.info(f"Loading data ({args.num_agents} agents)...")
    input_file = DATA_DIR / f"input_agents_{args.num_agents}_subset.npz"
    output_file = DATA_DIR / f"output_agents_{args.num_agents}_subset.npz"
    
    if not input_file.exists() or not output_file.exists():
        input_file = DATA_DIR / f"input_agents_{args.num_agents}.npz"
        output_file = DATA_DIR / f"output_agents_{args.num_agents}.npz"
    
    if not input_file.exists():
        logger.error(f"Data not found: {input_file}")
        return
    
    X = np.load(input_file)['data']
    Y = np.load(output_file)['data']
    
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    logger.info(f"Data shape: X={X.shape}, Y={Y.shape}")
    
    if args.use_subset and len(X) > 10000:
        X = X[:10000]
        Y = Y[:10000]
        logger.info(f"Using subset: X={X.shape}, Y={Y.shape}")
    
    # Dataset
    dataset = SwarmTrajectoryDataset(X, Y, normalize=True)
    
    # Split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Train: {len(train_set)}, Val: {len(val_set)}")
    
    # Model
    logger.info("Creating model...")
    model = SwarmGRUModel(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        output_dim=3,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_agents=args.num_agents,
        seq_out=Y.shape[1]
    )
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")
    
    # Optimizer, scheduler, loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps,
                                         gamma=args.lr_decay)
    loss_fn = CombinedLoss(pos_weight=1.0, vel_weight=0.5, dt=0.1)
    
    # Create checkpoint directory for this run
    checkpoint_dir = CHECKPOINTS_DIR / f"agents_{args.num_agents}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV logger for metrics
    csv_path = RESULTS_DIR / f"metrics_agents_{args.num_agents}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write header
    header = ['epoch', 'train_loss', 'train_mse', 'train_mae', 'train_mape', 'train_ade', 'train_fde',
              'val_loss', 'val_mse', 'val_mae', 'val_mape', 'val_ade', 'val_fde']
    csv_writer.writerow(header)
    csv_file.flush()
    
    # Training
    logger.info("Starting training...")
    best_val_loss = float('inf')
    patience_count = 0
    
    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Log metrics
        log_msg = (f"Epoch {epoch+1:3d}/{args.num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.6f} "
                  f"MAE: {train_metrics['mae']:.6f} "
                  f"ADE: {train_metrics['ade']:.6f} "
                  f"FDE: {train_metrics['fde']:.6f} | "
                  f"Val Loss: {val_metrics['loss']:.6f} "
                  f"MAE: {val_metrics['mae']:.6f} "
                  f"ADE: {val_metrics['ade']:.6f} "
                  f"FDE: {val_metrics['fde']:.6f}")
        logger.info(log_msg)
        
        # Write to CSV
        csv_row = [epoch+1, 
                  train_metrics['loss'], train_metrics['mse'], train_metrics['mae'], 
                  train_metrics['mape'], train_metrics['ade'], train_metrics['fde'],
                  val_metrics['loss'], val_metrics['mse'], val_metrics['mae'], 
                  val_metrics['mape'], val_metrics['ade'], val_metrics['fde']]
        csv_writer.writerow(csv_row)
        csv_file.flush()
        
        scheduler.step()
        
        # Early stopping + checkpointing
        is_best = val_metrics['loss'] < best_val_loss
        
        if is_best:
            best_val_loss = val_metrics['loss']
            patience_count = 0
            
            # Save best model
            best_model_path = MODELS_DIR / f"swarm_gru_agents_{args.num_agents}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
                'dataset_stats': {
                    'input_mean': dataset.input_mean.tolist(),
                    'input_std': dataset.input_std.tolist(),
                    'output_mean': dataset.output_mean.tolist(),
                    'output_std': dataset.output_std.tolist(),
                },
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, best_model_path)
            logger.info(f"  [BEST] Model saved: {best_model_path}")
        else:
            patience_count += 1
        
        # Periodic checkpoint save
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch+1:03d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
                'dataset_stats': {
                    'input_mean': dataset.input_mean.tolist(),
                    'input_std': dataset.input_std.tolist(),
                    'output_mean': dataset.output_mean.tolist(),
                    'output_std': dataset.output_std.tolist(),
                },
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            logger.info(f"  [CHECKPOINT] Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if patience_count >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {patience_count} epochs without improvement")
            break
    
    csv_file.close()
    logger.info(f"[SUCCESS] Metrics CSV saved: {csv_path}")
    
    # Summary
    logger.info("=" * 80)
    logger.info("Training Summary")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {best_val_loss:.8f}")
    logger.info(f"Best model saved: {MODELS_DIR / f'swarm_gru_agents_{args.num_agents}_best.pth'}")
    logger.info(f"Metrics CSV: {csv_path}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--use_subset', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_decay_steps', type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=0, 
                       help='Save checkpoint every N epochs (0 to disable)')
    
    args = parser.parse_args()
    main(args)
