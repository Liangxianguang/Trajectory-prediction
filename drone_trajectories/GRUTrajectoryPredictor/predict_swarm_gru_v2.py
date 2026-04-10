"""
Swarm GRU Trajectory Predictor - Inference Script (v2)
====================================================
Performs inference on trained model.

Usage:
    python predict_swarm_gru_v2.py --num_agents 3 --use_subset
    python predict_swarm_gru_v2.py --num_agents 3 --visualize --save_results
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime

# Configure paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path(r"d:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments")
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"
LOG_DIR = PROJECT_ROOT / "logs"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


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


# ============ Load Model ============
def load_checkpoint(model_path, device):
    logger.info(f"Loading model: {model_path}")
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return None, None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    args = checkpoint['args']
    model = SwarmGRUModel(
        input_dim=3,
        hidden_dim=args['hidden_dim'],
        output_dim=3,
        num_layers=args['num_layers'],
        dropout=args['dropout'],
        num_agents=args['num_agents'],
        seq_out=checkpoint['model_state_dict']['fc.weight'].shape[1] // (3 * args['num_agents'])
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    stats = checkpoint['dataset_stats']
    logger.info(f"Model loaded (best val loss: {checkpoint['best_val_loss']:.6f})")
    
    return model, args, stats


# ============ Metrics ============
def compute_metrics(y_pred, y_true):
    """Compute evaluation metrics"""
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }


# ============ Inference ============
def run_inference(model, X, Y, dataset_stats, num_agents, batch_size, device):
    """Run inference on data"""
    logger.info("Running inference...")
    
    X = np.transpose(X, (1, 0, 2, 3))  # (samples, seq_in, agents, 3)
    Y = np.transpose(Y, (1, 0, 2, 3))  # (samples, seq_out, agents, 3)
    
    # Normalize input
    input_mean = np.array(dataset_stats['input_mean'])
    input_std = np.array(dataset_stats['input_std'])
    
    X_norm = (X - input_mean) / input_std
    
    # Predict
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_norm), batch_size), desc='Inference'):
            batch_X = X_norm[i:i+batch_size]
            batch_X = torch.from_numpy(batch_X).float().to(device)
            
            y_pred = model(batch_X)
            predictions.append(y_pred.cpu().numpy())
    
    y_pred_norm = np.concatenate(predictions, axis=0)
    
    # Denormalize
    output_mean = np.array(dataset_stats['output_mean'])
    output_std = np.array(dataset_stats['output_std'])
    
    y_pred_delta = y_pred_norm * output_std + output_mean
    
    # Add to last input frame to get absolute coordinates
    X_last = X[:, -1:, :, :]  # (samples, 1, agents, 3)
    y_pred = y_pred_delta + X_last
    
    return y_pred, Y


# ============ Visualization ============
def plot_sample_trajectories(X, y_true, y_pred, num_agents, save_dir):
    """Plot sample predictions"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Plotting {min(5, len(X))} samples...")
    
    for sample_id in range(min(5, len(X))):
        for agent_id in range(min(3, num_agents)):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            x_traj = X[sample_id, :, agent_id, :]
            y_true_traj = y_true[sample_id, :, agent_id, :]
            y_pred_traj = y_pred[sample_id, :, agent_id, :]
            
            ax.plot(x_traj[:, 0], x_traj[:, 1], x_traj[:, 2],
                   'b-o', label='Input', linewidth=2, markersize=4)
            ax.plot(y_true_traj[:, 0], y_true_traj[:, 1], y_true_traj[:, 2],
                   'g-s', label='True', linewidth=2, markersize=4)
            ax.plot(y_pred_traj[:, 0], y_pred_traj[:, 1], y_pred_traj[:, 2],
                   'r--^', label='Predicted', linewidth=2, markersize=4)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Sample {sample_id}, Agent {agent_id}')
            ax.legend()
            ax.grid(alpha=0.3)
            
            out_file = save_dir / f"traj_sample{sample_id}_agent{agent_id}.png"
            plt.savefig(out_file, dpi=100, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Plots saved to {save_dir}")


def plot_metrics(y_true, y_pred, num_agents, save_dir):
    """Plot error metrics"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Per-agent metrics
    agent_mse = []
    for agent_id in range(num_agents):
        y_true_agent = y_true[:, :, agent_id, :]
        y_pred_agent = y_pred[:, :, agent_id, :]
        mse = np.mean((y_true_agent - y_pred_agent) ** 2)
        agent_mse.append(mse)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # MSE per agent
    axes[0].bar(range(num_agents), agent_mse)
    axes[0].set_xlabel('Agent ID')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE per Agent')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Error over time steps
    error_over_time = []
    for t in range(y_pred.shape[1]):
        error = np.mean((y_true[:, t] - y_pred[:, t]) ** 2)
        error_over_time.append(error)
    
    axes[1].plot(error_over_time, 'o-')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('MSE over Time Steps')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    out_file = save_dir / "metrics.png"
    plt.savefig(out_file, dpi=100)
    logger.info(f"Metrics plot saved: {out_file}")
    plt.close()


# ============ Main ============
def main(args):
    logger.info("=" * 80)
    logger.info("Swarm GRU Inference")
    logger.info("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load model
    model_path = MODELS_DIR / f"swarm_gru_agents_{args.num_agents}_best.pth"
    model, model_args, stats = load_checkpoint(model_path, device)
    
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Load data
    logger.info(f"Loading data ({args.num_agents} agents)...")
    input_file = DATA_DIR / f"input_agents_{args.num_agents}_subset.npz"
    output_file = DATA_DIR / f"output_agents_{args.num_agents}_subset.npz"
    
    if not input_file.exists():
        input_file = DATA_DIR / f"input_agents_{args.num_agents}.npz"
        output_file = DATA_DIR / f"output_agents_{args.num_agents}.npz"
    
    if not input_file.exists():
        logger.error(f"Data not found")
        return
    
    X = np.load(input_file)['data']
    Y = np.load(output_file)['data']
    logger.info(f"Data loaded: X={X.shape}, Y={Y.shape}")
    
    # Use subset if requested
    if args.use_subset and X.shape[1] > 1000:
        X = X[:, :1000, :, :]
        Y = Y[:, :1000, :, :]
        logger.info(f"Using subset: X={X.shape}, Y={Y.shape}")
    
    # Inference
    y_pred, y_true = run_inference(model, X, Y, stats, args.num_agents,
                                   args.batch_size, device)
    
    # Metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(y_pred, y_true)
    
    logger.info("=" * 80)
    logger.info("Results:")
    logger.info(f"  MSE:  {metrics['mse']:.8f}")
    logger.info(f"  RMSE: {metrics['rmse']:.8f}")
    logger.info(f"  MAE:  {metrics['mae']:.8f}")
    logger.info(f"  R²:   {metrics['r2']:.6f}")
    logger.info("=" * 80)
    
    # Save metrics
    metrics_file = RESULTS_DIR / f"metrics_agents_{args.num_agents}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_file}")
    
    # Visualize
    if args.visualize:
        X_orig = np.transpose(X, (1, 0, 2, 3))
        plot_sample_trajectories(X_orig, y_true, y_pred, args.num_agents,
                                RESULTS_DIR / "visualizations")
        plot_metrics(y_true, y_pred, args.num_agents, RESULTS_DIR / "metrics")
    
    # Save results
    if args.save_results:
        results = {
            'y_pred': y_pred,
            'y_true': y_true,
            'metrics': metrics,
        }
        out_file = RESULTS_DIR / f"predictions_agents_{args.num_agents}.npz"
        np.savez(out_file, **results)
        logger.info(f"Results saved: {out_file}")
    
    logger.info("Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--use_subset', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    main(args)
