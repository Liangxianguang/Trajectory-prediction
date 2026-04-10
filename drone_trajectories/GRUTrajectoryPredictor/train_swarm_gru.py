"""
Swarm GRU Trajectory Predictor - Training Script
=================================================
Trains a GRU-based model to predict multi-agent drone swarm trajectories.

Features:
  - GRU encoder-decoder architecture
  - Multi-agent trajectory handling (agents x, y, z coordinates)
  - Velocity-aware loss function
  - Early stopping and learning rate scheduling
  - Model checkpointing
  - Comprehensive evaluation metrics

Usage:
    python train_swarm_gru.py --num_agents 3 --epochs 100 --batch_size 64
    python train_swarm_gru.py --num_agents 3 --epochs 50 --batch_size 128 --use_subset
    python train_swarm_gru.py --help  # Show all options
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

from config import TRAINING_CONFIG, MODELS_DIR, RESULTS_DIR, DATA_DIR, LOG_CONFIG

# ============ Logging Setup ============
logging.basicConfig(
    level=LOG_CONFIG['log_level'],
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_CONFIG['log_dir'] / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============ Dataset Class ============
class SwarmTrajectoryDataset(Dataset):
    """Multi-agent trajectory dataset with normalization"""
    
    def __init__(self, X, Y, normalize=True, input_mean=None, input_std=None,
                 output_mean=None, output_std=None):
        """
        Args:
            X: (samples, seq_in, agents, 3) - input trajectories
            Y: (samples, seq_out, agents, 3) - output trajectories
            normalize: whether to normalize data
            input_mean/std: statistics for input data
            output_mean/std: statistics for output data (deltas)
        """
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.normalize = normalize
        self.samples = len(X)
        
        # Compute statistics if not provided
        if input_mean is None:
            self.input_mean = np.mean(X.reshape(-1, 3), axis=0)
            self.input_std = np.std(X.reshape(-1, 3), axis=0)
        else:
            self.input_mean = np.array(input_mean, dtype=np.float32)
            self.input_std = np.array(input_std, dtype=np.float32)
        
        # Prevent division by zero
        self.input_std = np.where(self.input_std < 1e-8, 1.0, self.input_std)
        
        # Output deltas statistics
        if output_mean is None:
            y_delta = Y - X[:, -1:, :, :]
            self.output_mean = np.mean(y_delta.reshape(-1, 3), axis=0)
            self.output_std = np.std(y_delta.reshape(-1, 3), axis=0)
        else:
            self.output_mean = np.array(output_mean, dtype=np.float32)
            self.output_std = np.array(output_std, dtype=np.float32)
        
        self.output_std = np.where(self.output_std < 1e-8, 1.0, self.output_std)
        
        logger.info(f"Dataset initialized: {self.samples} samples")
        logger.info(f"  Input mean: {self.input_mean}, std: {self.input_std}")
        logger.info(f"  Output mean: {self.output_mean}, std: {self.output_std}")
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (seq_in, agents, 3)
        y = self.Y[idx].copy()  # (seq_out, agents, 3)
        
        # Normalize input
        if self.normalize:
            x_norm = (x - self.input_mean) / self.input_std
        else:
            x_norm = x
        
        # Compute output delta (relative to last input frame)
        y_delta = y - x[-1:, :, :]
        
        if self.normalize:
            y_delta_norm = (y_delta - self.output_mean) / self.output_std
        else:
            y_delta_norm = y_delta
        
        return {
            'x': torch.from_numpy(x_norm),
            'x_orig': torch.from_numpy(x),
            'y_delta': torch.from_numpy(y_delta_norm),
            'y_orig': torch.from_numpy(y),
        }


# ============ Model Definition ============
class SwarmGRUPredictor(nn.Module):
    """
    Encoder-decoder GRU model for multi-agent trajectory prediction.
    
    Architecture:
    - Encoder: GRU that processes input trajectory sequence
    - Decoder: GRU that generates output sequence from encoder hidden state
    - Output: Fully connected layer projecting to (agents, 3) coordinates
    """
    
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, 
                 num_layers=2, dropout=0.3, num_agents=3, seq_out=10):
        """
        Args:
            input_dim: Input feature dimension (3 for x,y,z)
            hidden_dim: GRU hidden size
            output_dim: Output feature dimension (3 for x,y,z)
            num_layers: Number of GRU layers
            dropout: Dropout rate
            num_agents: Number of agents in swarm
            seq_out: Output sequence length
        """
        super(SwarmGRUPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_agents = num_agents
        self.seq_out = seq_out
        
        # Reshape input: (batch, seq_in, agents, 3) -> (batch, seq_in, agents*3)
        self.encoder_input_size = input_dim * num_agents
        
        # Encoder: processes concatenated agent coordinates
        self.encoder = nn.GRU(
            input_size=self.encoder_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder: generates output sequence
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection: map hidden state to agent coordinates
        self.fc_out = nn.Linear(hidden_dim, self.encoder_input_size)
        
        # Optional: attention or additional layers
        self.fc_refine = nn.Sequential(
            nn.Linear(self.encoder_input_size, self.encoder_input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder_input_size, self.encoder_input_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_in, agents, 3)
        
        Returns:
            y_pred: (batch_size, seq_out, agents, 3)
        """
        batch_size = x.size(0)
        seq_in = x.size(1)
        
        # Reshape for encoder: (batch, seq_in, agents*3)
        x_flat = x.view(batch_size, seq_in, -1)
        
        # Encode
        encoder_out, h_n = self.encoder(x_flat)
        
        # Decode: generate output sequence
        decoder_input = torch.zeros(
            batch_size, self.seq_out, self.hidden_dim,
            device=x.device, dtype=x.dtype
        )
        decoder_out, _ = self.decoder(decoder_input, h_n)
        
        # Project to agent coordinates
        y_pred_flat = self.fc_out(decoder_out)  # (batch, seq_out, agents*3)
        y_pred_flat = self.fc_refine(y_pred_flat)
        
        # Reshape back: (batch, seq_out, agents, 3)
        y_pred = y_pred_flat.view(batch_size, self.seq_out, self.num_agents, self.output_dim)
        
        return y_pred


# ============ Loss Functions ============
class MultiTaskLoss(nn.Module):
    """Combined loss for position and velocity"""
    
    def __init__(self, weight_position=1.0, weight_velocity=0.5, dt=0.1):
        super().__init__()
        self.weight_position = weight_position
        self.weight_velocity = weight_velocity
        self.dt = dt
        self.mse_loss = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (batch, seq_out, agents, 3)
            y_true: (batch, seq_out, agents, 3)
        
        Returns:
            total_loss: weighted combination of position and velocity losses
        """
        # Position loss
        loss_pos = self.mse_loss(y_pred, y_true)
        
        # Velocity loss (predict velocity changes)
        if y_pred.size(1) > 1:
            vel_pred = (y_pred[:, 1:] - y_pred[:, :-1]) / self.dt
            vel_true = (y_true[:, 1:] - y_true[:, :-1]) / self.dt
            loss_vel = self.mse_loss(vel_pred, vel_true)
        else:
            loss_vel = 0.0
        
        total_loss = self.weight_position * loss_pos + self.weight_velocity * loss_vel
        
        return total_loss, loss_pos, loss_vel


# ============ Training Functions ============
def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_loss_pos = 0.0
    total_loss_vel = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch in pbar:
        x = batch['x'].to(device)
        y_delta = batch['y_delta'].to(device)
        
        # Forward pass
        y_pred = model(x)
        loss, loss_pos, loss_vel = loss_fn(y_pred, y_delta)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_pos += loss_pos.item()
        total_loss_vel += loss_vel.item() if isinstance(loss_vel, torch.Tensor) else 0.0
        num_batches += 1
        
        pbar.update(1)
    
    return {
        'total': total_loss / num_batches,
        'position': total_loss_pos / num_batches,
        'velocity': total_loss_vel / num_batches,
    }


def validate(model, val_loader, loss_fn, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_loss_pos = 0.0
    total_loss_vel = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', leave=False):
            x = batch['x'].to(device)
            y_delta = batch['y_delta'].to(device)
            
            y_pred = model(x)
            loss, loss_pos, loss_vel = loss_fn(y_pred, y_delta)
            
            total_loss += loss.item()
            total_loss_pos += loss_pos.item()
            total_loss_vel += loss_vel.item() if isinstance(loss_vel, torch.Tensor) else 0.0
            num_batches += 1
    
    return {
        'total': total_loss / num_batches,
        'position': total_loss_pos / num_batches,
        'velocity': total_loss_vel / num_batches,
    }


# ============ Metrics ============
def compute_metrics(y_pred, y_true):
    """Compute MSE, RMSE, MAE metrics"""
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }


# ============ Main Training Function ============
def train_model(args):
    """Main training function"""
    logger.info("=" * 80)
    logger.info("Swarm GRU Trajectory Predictor - Training")
    logger.info("=" * 80)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    logger.info(f"Loading data for {args.num_agents} agents...")
    input_file = DATA_DIR / f"input_agents_{args.num_agents}.npz"
    output_file = DATA_DIR / f"output_agents_{args.num_agents}.npz"
    
    if not input_file.exists() or not output_file.exists():
        # Try with _subset suffix
        input_file = DATA_DIR / f"input_agents_{args.num_agents}_subset.npz"
        output_file = DATA_DIR / f"output_agents_{args.num_agents}_subset.npz"
    
    if not input_file.exists() or not output_file.exists():
        logger.error(f"Data files not found: {input_file}, {output_file}")
        raise FileNotFoundError(f"Data files not found")
    
    X = np.load(input_file)['data']  # (seq_in, samples, agents, 3)
    Y = np.load(output_file)['data']  # (seq_out, samples, agents, 3)
    
    # Transpose to (samples, seq_in/out, agents, 3)
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    logger.info(f"Data loaded: X {X.shape}, Y {Y.shape}")
    
    # Limit to subset if requested
    if args.use_subset:
        subset_size = 10000
        X = X[:subset_size]
        Y = Y[:subset_size]
        logger.info(f"Using subset: X {X.shape}, Y {Y.shape}")
    
    # Create dataset
    dataset = SwarmTrajectoryDataset(X, Y, normalize=True)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = SwarmGRUPredictor(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        output_dim=3,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_agents=args.num_agents,
        seq_out=20  # Fixed output sequence length
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.lr_decay_steps, 
        gamma=args.lr_decay
    )
    loss_fn = MultiTaskLoss(
        weight_position=args.loss_weight_pos,
        weight_velocity=args.loss_weight_vel,
        dt=TRAINING_CONFIG['dt']
    )
    
    # Training loop
    logger.info("Starting training...")
    history = {
        'train_loss': [],
        'train_loss_pos': [],
        'train_loss_vel': [],
        'val_loss': [],
        'val_loss_pos': [],
        'val_loss_vel': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        val_losses = validate(model, val_loader, loss_fn, device)
        
        # Log
        logger.info(f"  Train Loss: {train_losses['total']:.6f} "
                   f"(pos: {train_losses['position']:.6f}, vel: {train_losses['velocity']:.6f})")
        logger.info(f"  Val Loss:   {val_losses['total']:.6f} "
                   f"(pos: {val_losses['position']:.6f}, vel: {val_losses['velocity']:.6f})")
        
        history['train_loss'].append(train_losses['total'])
        history['train_loss_pos'].append(train_losses['position'])
        history['train_loss_vel'].append(train_losses['velocity'])
        history['val_loss'].append(val_losses['total'])
        history['val_loss_pos'].append(val_losses['position'])
        history['val_loss_vel'].append(val_losses['velocity'])
        
        # Learning rate scheduler
        scheduler.step()
        
        # Early stopping
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            
            # Save checkpoint
            model_path = MODELS_DIR / f"swarm_gru_agents_{args.num_agents}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(args),
                'dataset_stats': {
                    'input_mean': dataset.input_mean.tolist(),
                    'input_std': dataset.input_std.tolist(),
                    'output_mean': dataset.output_mean.tolist(),
                    'output_std': dataset.output_std.tolist(),
                }
            }, model_path)
            logger.info(f"  Model saved: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {args.early_stopping_patience} epochs")
                break
    
    # Save training history
    history_path = RESULTS_DIR / f"training_history_agents_{args.num_agents}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved: {history_path}")
    
    # Plot training curves
    plot_training_history(history, args.num_agents)
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    
    return model, history


def plot_training_history(history, num_agents):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Total loss
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss components
    axes[1].plot(history['train_loss_pos'], label='Train Pos', marker='o')
    axes[1].plot(history['train_loss_vel'], label='Train Vel', marker='^')
    axes[1].plot(history['val_loss_pos'], label='Val Pos', marker='s')
    axes[1].plot(history['val_loss_vel'], label='Val Vel', marker='d')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"training_history_agents_{num_agents}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training plot saved: {plot_path}")
    plt.close()


# ============ Main Entry Point ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Swarm GRU Trajectory Predictor"
    )
    
    # Data
    parser.add_argument('--num_agents', type=int, default=3,
                       help='Number of agents in swarm')
    parser.add_argument('--use_subset', action='store_true',
                       help='Use subset of data (10k samples) for quick testing')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='GRU hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                       help='Learning rate decay factor')
    parser.add_argument('--lr_decay_steps', type=int, default=10,
                       help='Learning rate decay steps')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    
    # Loss
    parser.add_argument('--loss_weight_pos', type=float, default=1.0,
                       help='Weight for position loss')
    parser.add_argument('--loss_weight_vel', type=float, default=0.5,
                       help='Weight for velocity loss')
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Train
    model, history = train_model(args)
