"""
Swarm GRU Trajectory Predictor - Inference Script
==================================================
Performs inference using trained model to predict drone swarm trajectories.

Features:
  - Load trained model and statistics
  - Batch inference on test data
  - Denormalization of predictions
  - Comprehensive evaluation metrics
  - 3D visualization of predictions
  - Export results to multiple formats

Usage:
    # Inference on test data
    python predict_swarm_gru.py --num_agents 3 --model_path Models/swarm_gru_agents_3_best.pth

    # Inference with visualization
    python predict_swarm_gru.py --num_agents 3 --model_path Models/swarm_gru_agents_3_best.pth --visualize

    # Inference and save results
    python predict_swarm_gru.py --num_agents 3 --model_path Models/swarm_gru_agents_3_best.pth --save_results
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

from config import MODELS_DIR, RESULTS_DIR, DATA_DIR
from train_swarm_gru import SwarmGRUPredictor, SwarmTrajectoryDataset, compute_metrics

# ============ Logging Setup ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ============ Prediction Functions ============
def load_checkpoint(model_path, device):
    """Load trained model and metadata"""
    logger.info(f"Loading checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']
    model = SwarmGRUPredictor(
        input_dim=3,
        hidden_dim=config['hidden_dim'],
        output_dim=3,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_agents=config['num_agents'],
        seq_out=20
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    dataset_stats = checkpoint['dataset_stats']
    
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    return model, config, dataset_stats


def predict_batch(model, x, device):
    """
    Predict trajectory for a batch of inputs.
    
    Args:
        model: trained SwarmGRUPredictor
        x: (batch_size, seq_in, agents, 3) input trajectories
        device: torch device
    
    Returns:
        y_pred: (batch_size, seq_out, agents, 3) predicted trajectories
    """
    x_tensor = torch.from_numpy(x).to(device)
    
    with torch.no_grad():
        y_pred = model(x_tensor)
    
    return y_pred.cpu().numpy()


def denormalize_predictions(y_pred, y_true, dataset_stats, x_last):
    """
    Convert normalized predictions back to original coordinates.
    
    Args:
        y_pred: (samples, seq_out, agents, 3) normalized predictions
        y_true: (samples, seq_out, agents, 3) normalized true values
        dataset_stats: dictionary with normalization statistics
        x_last: (samples, agents, 3) last frame of input sequence
    
    Returns:
        y_pred_denorm: denormalized predictions
        y_true_denorm: denormalized true values
    """
    output_mean = np.array(dataset_stats['output_mean'])
    output_std = np.array(dataset_stats['output_std'])
    
    # Denormalize: multiply by std and add mean
    y_pred_denorm = y_pred * output_std + output_mean
    y_true_denorm = y_true * output_std + output_mean
    
    # Convert from delta to absolute coordinates
    y_pred_denorm = y_pred_denorm + x_last[:, np.newaxis, :, :]
    y_true_denorm = y_true_denorm + x_last[:, np.newaxis, :, :]
    
    return y_pred_denorm, y_true_denorm


def evaluate_predictions(y_pred, y_true, x_input):
    """
    Compute evaluation metrics.
    
    Args:
        y_pred: (samples, seq_out, agents, 3)
        y_true: (samples, seq_out, agents, 3)
        x_input: (samples, seq_in, agents, 3)
    
    Returns:
        metrics: dictionary with evaluation metrics
    """
    # Flatten to compute metrics
    y_pred_flat = y_pred.reshape(-1, 3)
    y_true_flat = y_true.reshape(-1, 3)
    
    metrics = compute_metrics(y_pred_flat, y_true_flat)
    
    # Per-coordinate metrics
    for i, coord in enumerate(['x', 'y', 'z']):
        metrics[f'mse_{coord}'] = np.mean((y_pred_flat[:, i] - y_true_flat[:, i]) ** 2)
        metrics[f'mae_{coord}'] = np.mean(np.abs(y_pred_flat[:, i] - y_true_flat[:, i]))
    
    # Per-agent average
    y_pred_per_agent = y_pred.mean(axis=1)  # (samples, seq_out, agents, 3) -> (samples, agents, 3)
    y_true_per_agent = y_true.mean(axis=1)
    
    agent_mse = []
    for agent in range(y_pred.shape[2]):
        agent_metrics = compute_metrics(
            y_pred_per_agent[:, agent, :],
            y_true_per_agent[:, agent, :]
        )
        agent_mse.append(agent_metrics['mse'])
    
    metrics['agent_mse'] = agent_mse
    metrics['avg_agent_mse'] = np.mean(agent_mse)
    
    return metrics


def visualize_predictions(x_input, y_true, y_pred, agent_id=0, sample_id=0, 
                         save_path=None):
    """
    Visualize 3D trajectory prediction.
    
    Args:
        x_input: (samples, seq_in, agents, 3)
        y_true: (samples, seq_out, agents, 3)
        y_pred: (samples, seq_out, agents, 3)
        agent_id: which agent to visualize
        sample_id: which sample to visualize
        save_path: where to save the figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get data for specific agent and sample
    x_traj = x_input[sample_id, :, agent_id, :]
    y_traj_true = y_true[sample_id, :, agent_id, :]
    y_traj_pred = y_pred[sample_id, :, agent_id, :]
    
    # Plot
    ax.plot(x_traj[:, 0], x_traj[:, 1], x_traj[:, 2], 
            'b-o', label='Input Trajectory', linewidth=2, markersize=4)
    ax.plot(y_traj_true[:, 0], y_traj_true[:, 1], y_traj_true[:, 2], 
            'g-s', label='True Future', linewidth=2, markersize=4)
    ax.plot(y_traj_pred[:, 0], y_traj_pred[:, 1], y_traj_pred[:, 2], 
            'r--^', label='Predicted Future', linewidth=2, markersize=4)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Trajectory Prediction (Agent {agent_id}, Sample {sample_id})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved: {save_path}")
    
    plt.close()


def save_results_csv(y_pred, y_true, x_input, save_dir, num_agents):
    """Save predictions to CSV files"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for agent_id in range(num_agents):
        # Prepare data
        data_list = []
        for sample_id in range(len(y_pred)):
            for step_id in range(y_pred.shape[1]):
                data_list.append({
                    'sample_id': sample_id,
                    'agent_id': agent_id,
                    'step': step_id,
                    'pred_x': y_pred[sample_id, step_id, agent_id, 0],
                    'pred_y': y_pred[sample_id, step_id, agent_id, 1],
                    'pred_z': y_pred[sample_id, step_id, agent_id, 2],
                    'true_x': y_true[sample_id, step_id, agent_id, 0],
                    'true_y': y_true[sample_id, step_id, agent_id, 1],
                    'true_z': y_true[sample_id, step_id, agent_id, 2],
                })
        
        df = pd.DataFrame(data_list)
        csv_path = save_dir / f"predictions_agent_{agent_id}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved: {csv_path}")


def save_results_npz(y_pred, y_true, x_input, dataset_stats, save_path):
    """Save predictions to NPZ file"""
    np.savez(
        save_path,
        y_pred=y_pred,
        y_true=y_true,
        x_input=x_input,
        dataset_stats=dataset_stats
    )
    logger.info(f"Results saved: {save_path}")


# ============ Main Inference Function ============
def run_inference(args):
    """Main inference function"""
    logger.info("=" * 80)
    logger.info("Swarm GRU Trajectory Predictor - Inference")
    logger.info("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    model, config, dataset_stats = load_checkpoint(args.model_path, device)
    
    # Load data
    logger.info(f"Loading test data for {args.num_agents} agents...")
    input_file = DATA_DIR / f"input_agents_{args.num_agents}.npz"
    output_file = DATA_DIR / f"output_agents_{args.num_agents}.npz"
    
    if not input_file.exists() or not output_file.exists():
        input_file = DATA_DIR / f"input_agents_{args.num_agents}_subset.npz"
        output_file = DATA_DIR / f"output_agents_{args.num_agents}_subset.npz"
    
    if not input_file.exists() or not output_file.exists():
        logger.error(f"Data files not found")
        raise FileNotFoundError("Data files not found")
    
    X = np.load(input_file)['data']  # (seq_in, samples, agents, 3)
    Y = np.load(output_file)['data']  # (seq_out, samples, agents, 3)
    
    # Transpose to (samples, seq_in/out, agents, 3)
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.transpose(Y, (1, 0, 2, 3))
    
    logger.info(f"Data loaded: X {X.shape}, Y {Y.shape}")
    
    # Create dataset for normalization statistics
    dataset = SwarmTrajectoryDataset(
        X, Y, 
        normalize=True,
        input_mean=dataset_stats['input_mean'],
        input_std=dataset_stats['input_std'],
        output_mean=dataset_stats['output_mean'],
        output_std=dataset_stats['output_std'],
    )
    
    # Use subset for faster inference if requested
    if args.use_subset:
        subset_size = 1000
        indices = np.random.choice(len(dataset), size=min(subset_size, len(dataset)), replace=False)
        X_test = X[indices]
        Y_test = Y[indices]
        logger.info(f"Using subset: {len(indices)} samples")
    else:
        X_test = X
        Y_test = Y
    
    # Inference
    logger.info("Running inference...")
    y_pred_list = []
    y_true_list = []
    
    batch_size = args.batch_size
    for i in tqdm(range(0, len(X_test), batch_size), desc='Inference'):
        batch_end = min(i + batch_size, len(X_test))
        
        # Normalize input
        X_batch = X_test[i:batch_end]
        Y_batch = Y_test[i:batch_end]
        
        X_batch_norm = (X_batch - dataset.input_mean) / dataset.input_std
        Y_batch_norm = (Y_batch - X_batch[:, -1:, :, :] - dataset.output_mean) / dataset.output_std
        
        # Predict
        y_pred_batch = predict_batch(model, X_batch_norm, device)
        
        y_pred_list.append(y_pred_batch)
        y_true_list.append(Y_batch_norm)
    
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    
    logger.info(f"Predictions shape: {y_pred.shape}")
    
    # Denormalize
    logger.info("Denormalizing predictions...")
    X_test_last = X_test[:, -1, :, :]  # (samples, agents, 3)
    y_pred_denorm, y_true_denorm = denormalize_predictions(
        y_pred, y_true, dataset_stats, X_test_last
    )
    
    # Evaluate
    logger.info("Computing metrics...")
    metrics = evaluate_predictions(y_pred_denorm, y_true_denorm, X_test)
    
    logger.info("=" * 80)
    logger.info("Evaluation Metrics:")
    logger.info("=" * 80)
    logger.info(f"MSE:  {metrics['mse']:.8f}")
    logger.info(f"RMSE: {metrics['rmse']:.8f}")
    logger.info(f"MAE:  {metrics['mae']:.8f}")
    logger.info(f"R²:   {metrics['r2']:.6f}")
    
    for coord in ['x', 'y', 'z']:
        logger.info(f"  {coord.upper()}: MSE={metrics[f'mse_{coord}']:.8f}, "
                   f"MAE={metrics[f'mae_{coord}']:.8f}")
    
    logger.info(f"\nPer-Agent MSE:")
    for agent_id, mse in enumerate(metrics['agent_mse']):
        logger.info(f"  Agent {agent_id}: {mse:.8f}")
    logger.info(f"  Average: {metrics['avg_agent_mse']:.8f}")
    
    # Save metrics
    metrics_path = RESULTS_DIR / f"inference_metrics_agents_{args.num_agents}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved: {metrics_path}")
    
    # Visualize
    if args.visualize:
        logger.info("Generating visualizations...")
        viz_dir = RESULTS_DIR / "visualizations"
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Visualize a few samples
        for sample_id in range(min(5, len(X_test))):
            for agent_id in range(min(3, args.num_agents)):
                save_path = viz_dir / f"trajectory_sample_{sample_id}_agent_{agent_id}.png"
                visualize_predictions(X_test, y_true_denorm, y_pred_denorm, 
                                     agent_id=agent_id, sample_id=sample_id, 
                                     save_path=save_path)
    
    # Save results
    if args.save_results:
        logger.info("Saving results...")
        
        if args.output_format == 'npz':
            npz_path = RESULTS_DIR / f"predictions_agents_{args.num_agents}.npz"
            save_results_npz(y_pred_denorm, y_true_denorm, X_test, dataset_stats, npz_path)
        
        elif args.output_format == 'csv':
            csv_dir = RESULTS_DIR / f"predictions_agents_{args.num_agents}"
            save_results_csv(y_pred_denorm, y_true_denorm, X_test, csv_dir, args.num_agents)
    
    logger.info("=" * 80)
    logger.info("Inference completed!")
    logger.info("=" * 80)
    
    return y_pred_denorm, y_true_denorm, metrics


# ============ Main Entry Point ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with Swarm GRU Trajectory Predictor"
    )
    
    # Model
    parser.add_argument('--model_path', type=str, 
                       default=str(MODELS_DIR / 'swarm_gru_agents_3_best.pth'),
                       help='Path to trained model checkpoint')
    parser.add_argument('--num_agents', type=int, default=3,
                       help='Number of agents')
    
    # Data
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for inference')
    parser.add_argument('--use_subset', action='store_true',
                       help='Use subset of data (1000 samples)')
    
    # Output
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save_results', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--output_format', type=str, default='npz',
                       choices=['npz', 'csv'],
                       help='Output format for results')
    
    args = parser.parse_args()
    
    # Run inference
    y_pred, y_true, metrics = run_inference(args)
