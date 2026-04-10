"""
Swarm GRU Trajectory Predictor - Enhanced Inference Script (v3)
============================================================
Enhanced inference with comprehensive metrics.

Usage:
    python predict_swarm_gru_v3.py --num_agents 3 --use_subset --visualize
    python predict_swarm_gru_v3.py --num_agents 3 --save_results
    python train_swarm_gru_v3.py --num_agents 3 --num_epochs 300 --save_every 10
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
import csv
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


# ============ Metrics Computation ============
def compute_comprehensive_metrics(y_pred, y_true):
    """Compute comprehensive metrics"""
    
    y_pred_flat = y_pred.reshape(-1, 3)
    y_true_flat = y_true.reshape(-1, 3)
    
    mse = np.mean((y_pred_flat - y_true_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_flat - y_true_flat))
    
    denominator = np.abs(y_true_flat) + 1e-10
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / denominator)) * 100
    
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat, axis=0, keepdims=True)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # ADE & FDE
    if len(y_pred.shape) == 4:
        diff = y_pred - y_true
        displacements = np.sqrt(np.sum(diff ** 2, axis=-1))
        ade = np.mean(displacements)
        fde = np.mean(np.sqrt(np.sum((y_pred[:, -1] - y_true[:, -1]) ** 2, axis=-1)))
    else:
        ade = np.mean(np.sqrt(np.sum((y_pred_flat - y_true_flat) ** 2, axis=-1)))
        fde = ade
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'ade': ade,
        'fde': fde,
    }
    
    for i, coord in enumerate(['x', 'y', 'z']):
        metrics[f'mse_{coord}'] = np.mean((y_pred_flat[:, i] - y_true_flat[:, i]) ** 2)
        metrics[f'mae_{coord}'] = np.mean(np.abs(y_pred_flat[:, i] - y_true_flat[:, i]))
    
    return metrics


# ============ Load Model ============
def load_checkpoint(model_path, device):
    logger.info(f"Loading model: {model_path}")
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return None, None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    args = checkpoint['args']
    # For swarm trajectory prediction with 3D coords, seq_out is always 10
    # (This matches the output_agents_*.npz files)
    seq_out = 10
    logger.info(f"  Creating model with seq_out={seq_out}")
    
    model = SwarmGRUModel(
        input_dim=3,
        hidden_dim=args['hidden_dim'],
        output_dim=3,
        num_layers=args['num_layers'],
        dropout=args['dropout'],
        num_agents=args['num_agents'],
        seq_out=seq_out
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    stats = checkpoint['dataset_stats']
    logger.info(f"Model loaded (best val loss: {checkpoint['best_val_loss']:.6f})")
    
    return model, args, stats


# ============ Inference ============
def run_inference(model, X, Y, dataset_stats, num_agents, batch_size, device):
    """Run inference on data"""
    logger.info("Running inference...")
    
    # X shape: (seq_in, samples, agents, 3) -> (samples, seq_in, agents, 3)
    # Y shape: (seq_out, samples, agents, 3) -> (samples, seq_out, agents, 3)
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    input_mean = np.array(dataset_stats['input_mean'])
    input_std = np.array(dataset_stats['input_std'])
    
    X_norm = (X - input_mean) / input_std
    
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_norm), batch_size), desc='Inference'):
            batch_X = X_norm[i:i+batch_size]
            batch_X = torch.from_numpy(batch_X).float().to(device)
            
            y_pred = model(batch_X)
            predictions.append(y_pred.cpu().numpy())
    
    # y_pred_norm shape: (samples, seq_out, agents, 3)
    y_pred_norm = np.concatenate(predictions, axis=0)
    
    # Denormalize: y_pred_norm is the delta, so add it to last input position
    output_mean = np.array(dataset_stats['output_mean'])
    output_std = np.array(dataset_stats['output_std'])
    
    y_pred_delta = y_pred_norm * output_std + output_mean
    X_last = X[:y_pred_norm.shape[0], -1:, :, :]  # Take last frame of input (shape: (samples, 1, agents, 3))
    y_pred = y_pred_delta + X_last
    
    # Trim Y to match the number of predictions
    Y_trimmed = Y[:y_pred_norm.shape[0], :, :, :]
    
    return y_pred, Y_trimmed


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
    
    logger.info(f"[SUCCESS] Plots saved to {save_dir}")


def plot_metrics(y_true, y_pred, num_agents, save_dir):
    """Plot error metrics"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Per-agent metrics
    agent_mse = []
    agent_mae = []
    agent_ade = []
    
    for agent_id in range(num_agents):
        y_true_agent = y_true[:, :, agent_id, :]
        y_pred_agent = y_pred[:, :, agent_id, :]
        
        mse = np.mean((y_true_agent - y_pred_agent) ** 2)
        mae = np.mean(np.abs(y_true_agent - y_pred_agent))
        diff = y_true_agent - y_pred_agent
        ade = np.mean(np.sqrt(np.sum(diff ** 2, axis=-1)))
        
        agent_mse.append(mse)
        agent_mae.append(mae)
        agent_ade.append(ade)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # MSE per agent
    axes[0, 0].bar(range(num_agents), agent_mse)
    axes[0, 0].set_xlabel('Agent ID')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('MSE per Agent')
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # MAE per agent
    axes[0, 1].bar(range(num_agents), agent_mae, color='orange')
    axes[0, 1].set_xlabel('Agent ID')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('MAE per Agent')
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # ADE per agent
    axes[1, 0].bar(range(num_agents), agent_ade, color='green')
    axes[1, 0].set_xlabel('Agent ID')
    axes[1, 0].set_ylabel('ADE')
    axes[1, 0].set_title('ADE per Agent')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Error over time steps
    error_over_time = []
    for t in range(y_pred.shape[1]):
        error = np.mean((y_true[:, t] - y_pred[:, t]) ** 2)
        error_over_time.append(error)
    
    axes[1, 1].plot(error_over_time, 'o-', color='purple')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('MSE over Time Steps')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    out_file = save_dir / "metrics_summary.png"
    plt.savefig(out_file, dpi=100)
    logger.info(f"[SUCCESS] Metrics plot saved: {out_file}")
    plt.close()


# ============ Main ============
def main(args):
    logger.info("=" * 80)
    logger.info("Swarm GRU Inference (Enhanced v3)")
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
        logger.error("Data not found")
        return
    
    X = np.load(input_file)['data']
    Y = np.load(output_file)['data']
    logger.info(f"Data loaded: X={X.shape}, Y={Y.shape}")
    
    if args.use_subset and X.shape[1] > 1000:
        X = X[:, :1000, :, :]
        Y = Y[:, :1000, :, :]
        logger.info(f"Using subset: X={X.shape}, Y={Y.shape}")
    
    # Inference
    y_pred, y_true = run_inference(model, X, Y, stats, args.num_agents,
                                   args.batch_size, device)
    
    # Debug: Log shapes
    logger.info(f"After inference - y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
    
    # Compute metrics
    logger.info("Computing comprehensive metrics...")
    metrics = compute_comprehensive_metrics(y_pred, y_true)
    
    # Save metrics to JSON
    metrics_json_path = RESULTS_DIR / f"metrics_agents_{args.num_agents}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save metrics to CSV
    metrics_csv_path = RESULTS_DIR / f"inference_metrics_agents_{args.num_agents}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())
    
    # Print results
    logger.info("=" * 80)
    logger.info("Inference Results:")
    logger.info("=" * 80)
    logger.info(f"  MSE:  {metrics['mse']:.8f}")
    logger.info(f"  RMSE: {metrics['rmse']:.8f}")
    logger.info(f"  MAE:  {metrics['mae']:.8f}")
    logger.info(f"  MAPE: {metrics['mape']:.4f} %")
    logger.info(f"  ADE:  {metrics['ade']:.8f}")
    logger.info(f"  FDE:  {metrics['fde']:.8f}")
    logger.info(f"  R²:   {metrics['r2']:.6f}")
    logger.info(f"\n  Per-Coordinate:")
    logger.info(f"    MSE_X: {metrics['mse_x']:.8f}")
    logger.info(f"    MSE_Y: {metrics['mse_y']:.8f}")
    logger.info(f"    MSE_Z: {metrics['mse_z']:.8f}")
    logger.info("=" * 80)
    
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
        out_file = RESULTS_DIR / f"predictions_agents_{args.num_agents}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez(out_file, **results)
        logger.info(f"[SUCCESS] Results saved: {out_file}")
    
    logger.info(f"[SUCCESS] Metrics JSON: {metrics_json_path}")
    logger.info(f"[SUCCESS] Metrics CSV: {metrics_csv_path}")
    logger.info("=" * 80)


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
