"""
Swarm GRU Trajectory Predictor - Training Script (v2)
=====================================================
Trains a GRU-based model to predict multi-agent drone swarm trajectories.

Fixed and improved version with working encoder-decoder architecture.

Usage:
    python train_swarm_gru_v2.py --num_agents 3 --epochs 50 --batch_size 64 --use_subset
    python train_swarm_gru_v2.py --num_agents 3 --epochs 100 --batch_size 32
    python train_swarm_gru_v2.py --help
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

# Configure paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path(r"d:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments")
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
for d in [MODELS_DIR, RESULTS_DIR, LOG_DIR]:
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
        """
        Args:
            X: (samples, seq_in, agents, 3) - input trajectories
            Y: (samples, seq_out, agents, 3) - output trajectories
        """
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.normalize = normalize
        self.samples = len(X)
        
        # Compute statistics if not provided
        if input_mean is None:
            self.input_mean = np.mean(self.X.reshape(-1, 3), axis=0)
            self.input_std = np.std(self.X.reshape(-1, 3), axis=0)
        else:
            self.input_mean = np.array(input_mean, dtype=np.float32)
            self.input_std = np.array(input_std, dtype=np.float32)
        
        self.input_std = np.where(self.input_std < 1e-8, 1.0, self.input_std)
        
        # Output relative changes
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
        
        # Normalize input
        if self.normalize:
            x_norm = (x - self.input_mean) / self.input_std
        else:
            x_norm = x
        
        # Output as relative displacement from last input frame
        y_delta = y - x[-1:, :, :]
        
        if self.normalize:
            y_delta_norm = (y_delta - self.output_mean) / self.output_std
        else:
            y_delta_norm = y_delta
        
        return {
            'x': torch.from_numpy(x_norm),
            'y': torch.from_numpy(y_delta_norm),
            'x_last': torch.from_numpy(x[-1:, :, :])  # Last frame of input
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
        
        # Encoder: process all agent coordinates concatenated
        self.encoder = nn.GRU(
            input_size=input_dim * num_agents,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Decoder: generate sequence
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, input_dim * num_agents)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_in, agents, 3)
        Returns:
            y: (batch, seq_out, agents, 3)
        """
        batch_size = x.size(0)
        seq_in = x.size(1)
        
        # Flatten agents: (batch, seq_in, agents*3)
        x_flat = x.view(batch_size, seq_in, -1)
        
        # Encode
        _, h_n = self.encoder(x_flat)  # h_n: (num_layers, batch, hidden_dim)
        
        # Decode: create decoder input (zeros)
        decoder_in = torch.zeros(batch_size, self.seq_out, self.hidden_dim,
                                device=x.device, dtype=x.dtype)
        
        decoder_out, _ = self.decoder(decoder_in, h_n)  # (batch, seq_out, hidden_dim)
        
        # Project to agent coordinates
        y_flat = self.fc(decoder_out)  # (batch, seq_out, agents*3)
        
        # Reshape back
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
        """
        Args:
            y_pred: (batch, seq_out, agents, 3)
            y_true: (batch, seq_out, agents, 3)
        """
        # Position loss
        loss_pos = self.mse(y_pred, y_true)
        
        # Velocity loss (only if seq_out > 1)
        if y_pred.size(1) > 1:
            v_pred = (y_pred[:, 1:] - y_pred[:, :-1]) / self.dt
            v_true = (y_true[:, 1:] - y_true[:, :-1]) / self.dt
            loss_vel = self.mse(v_pred, v_true)
        else:
            loss_vel = torch.tensor(0.0, device=y_pred.device)
        
        loss = self.pos_weight * loss_pos + self.vel_weight * loss_vel
        return loss, loss_pos, loss_vel


# ============ Training Functions ============
def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_pos = 0.0
    total_vel = 0.0
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
        count += 1
    
    return total_loss / count, total_pos / count, total_vel / count


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_pos = 0.0
    total_vel = 0.0
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
            count += 1
    
    return total_loss / count, total_pos / count, total_vel / count


# ============ Main Training ============
def main(args):
    logger.info("=" * 80)
    logger.info("Swarm GRU Training")
    logger.info("=" * 80)
    
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
    
    X = np.load(input_file)['data']  # (seq_in, samples, agents, 3)
    Y = np.load(output_file)['data']  # (seq_out, samples, agents, 3)
    
    # Transpose: (samples, seq_in/out, agents, 3)
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    logger.info(f"Data shape: X={X.shape}, Y={Y.shape}")
    
    # Use subset if requested
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
        seq_out=Y.shape[1]  # Use actual output seq length
    )
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")
    
    # Optimizer, scheduler, loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps,
                                         gamma=args.lr_decay)
    loss_fn = CombinedLoss(pos_weight=1.0, vel_weight=0.5, dt=0.1)
    
    # Training
    logger.info("Starting training...")
    history = {
        'train_loss': [],
        'train_pos': [],
        'train_vel': [],
        'val_loss': [],
        'val_pos': [],
        'val_vel': [],
    }
    
    best_val_loss = float('inf')
    patience_count = 0
    
    for epoch in range(args.num_epochs):
        train_loss, train_pos, train_vel = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_pos, val_vel = validate(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['train_pos'].append(train_pos)
        history['train_vel'].append(train_vel)
        history['val_loss'].append(val_loss)
        history['val_pos'].append(val_pos)
        history['val_vel'].append(val_vel)
        
        logger.info(f"Epoch {epoch+1:3d} | "
                   f"Train: {train_loss:.6f} (pos:{train_pos:.6f}, vel:{train_vel:.6f}) | "
                   f"Val: {val_loss:.6f} (pos:{val_pos:.6f}, vel:{val_vel:.6f})")
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            
            # Save
            model_path = MODELS_DIR / f"swarm_gru_agents_{args.num_agents}_best.pth"
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
                }
            }, model_path)
            logger.info(f"  -> Model saved")
        else:
            patience_count += 1
            if patience_count >= args.early_stopping_patience:
                logger.info(f"Early stopping after {patience_count} epochs")
                break
    
    # Save history
    hist_file = RESULTS_DIR / f"history_agents_{args.num_agents}.json"
    with open(hist_file, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved: {hist_file}")
    
    # Plot
    plot_training(history, args.num_agents)
    
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


def plot_training(history, num_agents):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], 'o-', label='Train')
    axes[0].plot(history['val_loss'], 's-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Components
    axes[1].plot(history['train_pos'], 'o-', label='Train Pos')
    axes[1].plot(history['train_vel'], '^-', label='Train Vel')
    axes[1].plot(history['val_pos'], 's-', label='Val Pos')
    axes[1].plot(history['val_vel'], 'd-', label='Val Vel')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    out = RESULTS_DIR / f"training_agents_{num_agents}.png"
    plt.savefig(out, dpi=150)
    logger.info(f"Plot saved: {out}")
    plt.close()


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
    
    args = parser.parse_args()
    main(args)
