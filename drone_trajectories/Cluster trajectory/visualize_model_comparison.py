#!/usr/bin/env python3
"""
Multi-Model Comparison Visualization Script
===========================================

Compare predictions from v1, v2, v3, v4 models on the same samples.
Perfect for paper figures showing model evolution.

Usage:
    python visualize_model_comparison.py \
        --data_dir swarm_segments \
        --sample_indices "2504,17995,33018" \
        --output_dir comparison_figures \
        --v1_model path/to/v1.pt \
        --v2_model path/to/v2.pt \
        --v3_model path/to/v3.pt \
        --v4_model path/to/v4.pt
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
import sys

# Import inference functions from different versions
sys.path.insert(0, str(Path(__file__).parent))

# Try to import v1 inference (EnhancedSwarmGRUModel, 16D features)
try:
    from infer_swarm_model import (
        load_data_robust as load_data_v1,
        infer_batch as infer_batch_v1,
        compute_features_for_inference as compute_features_v1,
    )
    from train_swarm_model_enhanced import EnhancedSwarmGRUModel as V1Model
    V1_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    V1_AVAILABLE = False
    logging.warning(f"v1 inference not available: {e}")

# Try to import v2 inference
try:
    # Import directly from the file, avoiding __init__.py issues
    v2_file = Path(__file__).parent / 'v2_inference' / 'infer_swarm_model_v2.py'
    if v2_file.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("infer_swarm_model_v2", v2_file)
        v2_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2_module)
        
        load_data_v2 = v2_module.load_data_robust
        infer_batch_v2 = v2_module.infer_batch
        compute_features_v2 = v2_module.compute_features_for_inference
        estimate_stats_v2 = v2_module.estimate_feature_stats_from_data
        V2_AVAILABLE = True
    else:
        V2_AVAILABLE = False
        logging.warning("v2 inference file not found")
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    V2_AVAILABLE = False
    logging.warning(f"v2 inference not available: {e}")

# Try to import v3 inference
try:
    from infer_swarm_model_v3_gnn import (
        load_data_robust as load_data_v3,
        infer_batch_v3,
        compute_features_for_inference as compute_features_v3,
        estimate_feature_stats_from_data as estimate_stats_v3,
        DynamicsAwareSwarmGRUModel,
        DynamicsAwareSwarmGRUModel_with_GNN,
        detect_model_version as detect_v3
    )
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False
    logging.warning("v3 inference not available")

# Try to import v4 inference
try:
    from infer_swarm_model_v4_enhanced_tail import (
        load_data_robust as load_data_v4,
        infer_batch_v4_enhanced,
        load_all_32d_features,
        compute_feature_statistics,
        normalize_features,
        EnhancedTailDynamicsAnalyzer
    )
    # Import model classes - v4 uses the same model classes as v3
    # Note: v4 uses DynamicsAwareSwarmGRUModel_with_GNN with input_size=32
    from train_swarm_model_v3_with_gnn import (
        DynamicsAwareSwarmGRUModel_with_GNN
    )
    # v4 also needs the base model (without GNN)
    from train_swarm_model_v2_dynamics_aware import (
        DynamicsAwareSwarmGRUModel
    )
    V4_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    V4_AVAILABLE = False
    logging.warning(f"v4 inference not available: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set font for better paper quality
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_all_24d_features(features_dir, num_agents, use_subset=False):
    """
    Load all 24D features into memory at once (for v2 and v3)
    """
    features_dir = Path(features_dir)
    subset_suffix = '_subset' if use_subset else ''
    
    feature_candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_24d.npz',
        features_dir / f'features_agents_{num_agents}_24d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
    ]
    
    for feat_path in feature_candidates:
        if feat_path.exists():
            try:
                logger.info(f"Preloading 24D feature file: {feat_path} ...")
                data = np.load(feat_path)
                features_all = np.asarray(data['features'])
                logger.info(f"✓ 24D feature preloading complete: {features_all.shape}")
                return features_all
            except Exception as e:
                logger.warning(f"Failed to preload 24D feature file {feat_path}: {e}")
    return None


def compute_detailed_metrics(true_future, pred_future):
    """
    Compute detailed metrics for a single model prediction
    
    Args:
        true_future: (seq_out, num_agents, 3) ground truth
        pred_future: (seq_out, num_agents, 3) prediction
    
    Returns:
        dict with metrics: MAE, RMSE, MAPE, MAE_X/Y/Z, MAE per step, MAE per agent
    """
    # Overall 3D position error
    errors = np.linalg.norm(pred_future - true_future, axis=2)  # (seq_out, num_agents)
    
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # MAPE: Mean Absolute Percentage Error
    # Calculate distance from origin for true positions to avoid division by zero
    true_distances = np.linalg.norm(true_future, axis=2)  # (seq_out, num_agents)
    # Avoid division by zero: use a small epsilon or skip zero distances
    epsilon = 1e-6
    valid_mask = true_distances > epsilon
    if np.any(valid_mask):
        mape = float(np.mean(np.abs(errors[valid_mask] / true_distances[valid_mask]) * 100.0))
    else:
        mape = 0.0  # If all distances are too small, MAPE is undefined
    
    # Per-axis MAE
    mae_x = float(np.mean(np.abs(pred_future[..., 0] - true_future[..., 0])))
    mae_y = float(np.mean(np.abs(pred_future[..., 1] - true_future[..., 1])))
    mae_z = float(np.mean(np.abs(pred_future[..., 2] - true_future[..., 2])))
    
    # Per-step MAE (average across agents)
    mae_per_step = np.mean(errors, axis=1).tolist()
    
    # Per-agent MAE (average across steps)
    mae_per_agent = np.mean(errors, axis=0).tolist()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_z': mae_z,
        'mae_per_step': mae_per_step,
        'mae_per_agent': mae_per_agent
    }


def plot_model_comparison(history, truth, predictions_dict, sample_idx, agents, output_dir):
    """
    Plot comparison of multiple model predictions
    
    Args:
        history: (seq_in, agents, 3) input history
        truth: (seq_out, agents, 3) ground truth future
        predictions_dict: dict with keys like 'v1', 'v2', 'v3', 'v4' 
                         and values (seq_out, agents, 3) predictions
        sample_idx: sample index for title
        agents: number of agents
        output_dir: output directory
    """
    fig = plt.figure(figsize=(24, 16))
    
    # Define colors for each model - using vibrant colors similar to v3 style
    model_colors = {
        'v1': '#E74C3C',  # Bright Red
        'v2': '#3498DB',  # Bright Blue
        'v3': '#9B59B6',  # Purple
        'v4': '#E67E22',  # Orange
        'truth': '#27AE60'  # Bright Green (similar to v3's 'gs-')
    }
    
    model_markers = {
        'v1': 'o',
        'v2': 's',
        'v3': '^',
        'v4': 'D',
        'truth': 's'  # Square marker like v3
    }
    
    model_linestyles = {
        'v1': '--',
        'v2': '-.',
        'v3': ':',
        'v4': '--',
        'truth': '-'  # Solid line like v3
    }
    
    # 1. 3D Trajectory Comparison
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    
    for agent_id in range(agents):
        last_point = history[-1, agent_id, :]
        
        # History trajectory (blue, like v3)
        if agent_id == 0:
            ax3d.plot(history[:, agent_id, 0], history[:, agent_id, 1], history[:, agent_id, 2],
                     'b-o', linewidth=2.5, markersize=5, alpha=0.8,
                     label='History', zorder=1)
        else:
            ax3d.plot(history[:, agent_id, 0], history[:, agent_id, 1], history[:, agent_id, 2],
                     'bo', linewidth=2.5, markersize=5, alpha=0.8, zorder=1)
        
        # Ground truth (green, like v3)
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        ax3d.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
                 's-', color=model_colors['truth'],
                 linewidth=2.8, markersize=7, alpha=0.9,
                 label='Ground Truth' if agent_id == 0 else '', zorder=10)
        
        # Model predictions (vibrant colors, higher alpha)
        for model_name, pred in predictions_dict.items():
            if pred is not None:
                pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
                ax3d.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
                         f'{model_markers[model_name]}{model_linestyles[model_name]}',
                         color=model_colors[model_name], linewidth=2.8, 
                         markersize=6, alpha=0.85,
                         label=f'{model_name.upper()}' if agent_id == 0 else '',
                         zorder=5)
    
    ax3d.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax3d.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax3d.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax3d.set_title(f'Sample {sample_idx}: 3D Trajectory Comparison\nAll {agents} Agents',
                  fontsize=13, fontweight='bold')
    ax3d.legend(fontsize=10, loc='upper left', ncol=2)
    ax3d.grid(True, alpha=0.3)
    
    # 2. XY Plane Projection
    ax_xy = fig.add_subplot(2, 3, 2)
    for agent_id in range(agents):
        last_point = history[-1, agent_id, :]
        
        # History (blue, like v3)
        if agent_id == 0:
            ax_xy.plot(history[:, agent_id, 0], history[:, agent_id, 1],
                      'b-o', linewidth=2.5, markersize=5, alpha=0.8, label='History', zorder=1)
        else:
            ax_xy.plot(history[:, agent_id, 0], history[:, agent_id, 1],
                      'bo', linewidth=2.5, markersize=5, alpha=0.8, zorder=1)
        
        # Ground truth (green, like v3)
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        ax_xy.plot(true_traj[:, 0], true_traj[:, 1],
                  's-', color=model_colors['truth'],
                  linewidth=2.8, markersize=7, alpha=0.9, label='Ground Truth' if agent_id == 0 else '',
                  zorder=10)
        
        # Model predictions (vibrant colors, higher alpha)
        for model_name, pred in predictions_dict.items():
            if pred is not None:
                pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
                ax_xy.plot(pred_traj[:, 0], pred_traj[:, 1],
                          f'{model_markers[model_name]}{model_linestyles[model_name]}',
                          color=model_colors[model_name], linewidth=2.8, markersize=6,
                          alpha=0.85, label=f'{model_name.upper()}' if agent_id == 0 else '',
                          zorder=5)
    
    ax_xy.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax_xy.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax_xy.set_title('XY Plane Projection', fontsize=12, fontweight='bold')
    ax_xy.legend(fontsize=10, loc='best', ncol=2)
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect('equal', adjustable='box')
    
    # 3. XZ Plane Projection
    ax_xz = fig.add_subplot(2, 3, 3)
    for agent_id in range(agents):
        last_point = history[-1, agent_id, :]
        
        # History (blue, like v3)
        if agent_id == 0:
            ax_xz.plot(history[:, agent_id, 0], history[:, agent_id, 2],
                      'b-o', linewidth=2.5, markersize=5, alpha=0.8, zorder=1)
        else:
            ax_xz.plot(history[:, agent_id, 0], history[:, agent_id, 2],
                      'bo', linewidth=2.5, markersize=5, alpha=0.8, zorder=1)
        
        # Ground truth (green, like v3)
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        ax_xz.plot(true_traj[:, 0], true_traj[:, 2],
                  's-', color=model_colors['truth'],
                  linewidth=2.8, markersize=7, alpha=0.9, zorder=10)
        
        # Model predictions (vibrant colors, higher alpha)
        for model_name, pred in predictions_dict.items():
            if pred is not None:
                pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
                ax_xz.plot(pred_traj[:, 0], pred_traj[:, 2],
                          f'{model_markers[model_name]}{model_linestyles[model_name]}',
                          color=model_colors[model_name], linewidth=2.8, markersize=6,
                          alpha=0.85, zorder=5)
    
    ax_xz.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax_xz.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
    ax_xz.set_title('XZ Plane Projection', fontsize=12, fontweight='bold')
    ax_xz.grid(True, alpha=0.3)
    ax_xz.set_aspect('equal', adjustable='box')
    
    # 4. YZ Plane Projection
    ax_yz = fig.add_subplot(2, 3, 4)
    for agent_id in range(agents):
        last_point = history[-1, agent_id, :]
        
        # History (blue, like v3)
        if agent_id == 0:
            ax_yz.plot(history[:, agent_id, 1], history[:, agent_id, 2],
                      'b-o', linewidth=2.5, markersize=5, alpha=0.8, zorder=1)
        else:
            ax_yz.plot(history[:, agent_id, 1], history[:, agent_id, 2],
                      'bo', linewidth=2.5, markersize=5, alpha=0.8, zorder=1)
        
        # Ground truth (green, like v3)
        true_traj = np.vstack([last_point, truth[:, agent_id, :]])
        ax_yz.plot(true_traj[:, 1], true_traj[:, 2],
                  's-', color=model_colors['truth'],
                  linewidth=2.8, markersize=7, alpha=0.9, zorder=10)
        
        # Model predictions (vibrant colors, higher alpha)
        for model_name, pred in predictions_dict.items():
            if pred is not None:
                pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
                ax_yz.plot(pred_traj[:, 1], pred_traj[:, 2],
                          f'{model_markers[model_name]}{model_linestyles[model_name]}',
                          color=model_colors[model_name], linewidth=2.8, markersize=6,
                          alpha=0.85, zorder=5)
    
    ax_yz.set_xlabel('Y (m)', fontsize=12, fontweight='bold')
    ax_yz.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
    ax_yz.set_title('YZ Plane Projection', fontsize=12, fontweight='bold')
    ax_yz.grid(True, alpha=0.3)
    ax_yz.set_aspect('equal', adjustable='box')
    
    # 5. Per-Step Error Comparison
    ax_error = fig.add_subplot(2, 3, 5)
    steps = np.arange(len(truth))
    
    for model_name, pred in predictions_dict.items():
        if pred is not None:
            # Calculate per-step error for all agents
            errors = np.linalg.norm(pred - truth, axis=2)  # (steps, agents)
            mean_error = np.mean(errors, axis=1)  # (steps,)
            ax_error.plot(steps, mean_error, 
                         f'{model_markers[model_name]}-', 
                         color=model_colors[model_name],
                         linewidth=2.8, markersize=7, alpha=0.9, label=f'{model_name.upper()}')
    
    ax_error.set_xlabel('Prediction Step', fontsize=12, fontweight='bold')
    ax_error.set_ylabel('Mean Position Error (m)', fontsize=12, fontweight='bold')
    ax_error.set_title('Per-Step Error Comparison (All Agents Avg)', fontsize=12, fontweight='bold')
    ax_error.legend(fontsize=10, loc='best')
    ax_error.grid(True, alpha=0.3)
    ax_error.set_xticks(steps)
    
    # 6. Overall MAE Comparison (Bar Chart)
    ax_mae = fig.add_subplot(2, 3, 6)
    model_names = []
    mae_values = []
    rmse_values = []
    
    for model_name, pred in predictions_dict.items():
        if pred is not None:
            errors = np.linalg.norm(pred - truth, axis=2)
            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            model_names.append(model_name.upper())
            mae_values.append(mae)
            rmse_values.append(rmse)
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax_mae.bar(x_pos - width/2, mae_values, width,
                      color=[model_colors[m.lower()] for m in model_names],
                      alpha=0.85, edgecolor='black', linewidth=2.0, label='MAE')
    
    bars2 = ax_mae.bar(x_pos + width/2, rmse_values, width,
                      color=[model_colors[m.lower()] for m in model_names],
                      alpha=0.6, edgecolor='black', linewidth=2.0, label='RMSE', hatch='///')
    
    # Add value labels on bars
    for bar, mae in zip(bars1, mae_values):
        height = bar.get_height()
        ax_mae.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mae:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, rmse in zip(bars2, rmse_values):
        height = bar.get_height()
        ax_mae.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rmse:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax_mae.set_ylabel('Error (m)', fontsize=12, fontweight='bold')
    ax_mae.set_title('Overall MAE & RMSE Comparison', fontsize=12, fontweight='bold')
    ax_mae.set_xticks(x_pos)
    ax_mae.set_xticklabels(model_names)
    ax_mae.legend(fontsize=10, loc='upper left')
    ax_mae.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    png_file = output_path / f'comparison_sample_{sample_idx:06d}.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Comparison figure saved: {png_file}")
    plt.close()
    
    # Compute detailed metrics for each model
    metrics_dict = {}
    for model_name, pred in predictions_dict.items():
        if pred is not None:
            metrics_dict[model_name.upper()] = compute_detailed_metrics(truth, pred)
    
    return {
        'sample_idx': int(sample_idx),
        'metrics': metrics_dict
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Comparison Visualization')
    
    parser.add_argument('--data_dir', required=True, help='Data directory')
    parser.add_argument('--features_24d_dir', default='features_24d', help='24D features directory (for v2 and v3)')
    parser.add_argument('--features_32d_dir', default='features_32d', help='32D features directory (for v4)')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--sample_indices', type=str, default=None,
                       help='Comma-separated sample indices (e.g., "2504,17995,33018"). If not provided, will use random sampling.')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to randomly select (used when sample_indices not provided)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    parser.add_argument('--output_dir', default='comparison_figures', help='Output directory')
    
    # Model paths
    parser.add_argument('--v1_model', help='v1 model path (optional)')
    parser.add_argument('--v2_model', help='v2 model path')
    parser.add_argument('--v3_model', help='v3 model path')
    parser.add_argument('--v4_model', help='v4 model path')
    
    # v4 specific
    parser.add_argument('--use_subset', action='store_true', help='Use subset data')
    parser.add_argument('--edge_threshold', type=float, default=5.0, help='GNN edge threshold')
    parser.add_argument('--no_gnn', action='store_true', help='Disable GNN for v4')
    parser.add_argument('--no_tail_enhancement', action='store_true', help='Disable tail enhancement for v4')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Parse sample indices or randomly select
    if args.sample_indices:
        # Use specified sample indices
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(',')]
        logger.info(f"Using specified sample indices: {sample_indices}")
    else:
        # Will randomly select after loading data
        sample_indices = None
        logger.info(f"Will randomly select {args.num_samples} samples with seed={args.seed}")
    
    # Load data
    logger.info(f"Loading data from: {args.data_dir}")
    if V4_AVAILABLE:
        X_all, Y_all = load_data_v4(args.data_dir, args.agents, use_subset=args.use_subset)
    elif V3_AVAILABLE:
        X_all, Y_all = load_data_v3(args.data_dir, args.agents, use_subset=args.use_subset)
    elif V2_AVAILABLE:
        X_all, Y_all = load_data_v2(args.data_dir, args.agents)
    else:
        raise RuntimeError("No inference module available")
    
    logger.info(f"Loaded data: X_all shape={X_all.shape}, Y_all shape={Y_all.shape}")
    
    # Randomly select samples if not specified
    total_samples = len(X_all)
    if sample_indices is None:
        # Randomly select samples from entire dataset
        num_samples = min(args.num_samples, total_samples)
        sample_indices = np.random.choice(total_samples, num_samples, replace=False)
        sample_indices = sample_indices.tolist()
        logger.info(f"Randomly selected {len(sample_indices)} samples (from {total_samples} total) with seed={args.seed}")
        logger.info(f"Selected sample indices: {sample_indices}")
    
    # Validate sample indices
    invalid = [idx for idx in sample_indices if idx < 0 or idx >= total_samples]
    if invalid:
        raise ValueError(f"Invalid sample indices (out of range [0, {total_samples-1}]): {invalid}")
    
    # Load models and prepare inference functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    models = {}
    inference_funcs = {}
    
    # Load v1 model (EnhancedSwarmGRUModel, 16D features computed on-the-fly)
    if args.v1_model and V1_AVAILABLE:
        logger.info(f"Loading v1 model: {args.v1_model}")
        try:
            checkpoint = torch.load(args.v1_model, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.v1_model, map_location='cpu')
        
        config = checkpoint.get('config', {})
        
        model = V1Model(
            input_size=16,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            output_size=3,
            dropout=config.get("dropout", 0.0),
            use_attention=config.get("use_attention", False),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        models['v1'] = model
        
        # Statistics for de-normalization + optional feature normalization
        if "output_mean" not in checkpoint or "output_std" not in checkpoint:
            raise ValueError("v1 checkpoint missing output_mean/output_std")
        output_mean = np.array(checkpoint["output_mean"], dtype=np.float32)
        output_std = np.array(checkpoint["output_std"], dtype=np.float32)
        
        input_mean_all = checkpoint.get("input_mean_all", None)
        input_std_all = checkpoint.get("input_std_all", None)
        input_mean = checkpoint.get("input_mean", None)
        input_std = checkpoint.get("input_std", None)
        
        if input_mean_all is not None and input_std_all is not None:
            input_mean_all = np.array(input_mean_all, dtype=np.float32)
            input_std_all = np.array(input_std_all, dtype=np.float32)
        else:
            input_mean_all = np.zeros(16, dtype=np.float32)
            input_std_all = np.ones(16, dtype=np.float32)
        
        if input_mean is None or input_std is None:
            input_mean = np.zeros(3, dtype=np.float32)
            input_std = np.ones(3, dtype=np.float32)
        else:
            input_mean = np.array(input_mean, dtype=np.float32)
            input_std = np.array(input_std, dtype=np.float32)
        
        def infer_v1(X_batch, indices=None):
            # v1 features are 16D and computed per-sample (no precomputed feature file)
            features_batch = np.stack(
                [
                    compute_features_v1(
                        x,
                        input_mean_all=input_mean_all,
                        input_std_all=input_std_all,
                        input_mean=input_mean,
                        input_std=input_std,
                        dt=0.1,
                    )
                    for x in X_batch
                ],
                axis=0,
            )
            return infer_batch_v1(
                models["v1"],
                features_batch,
                X_batch,
                device,
                output_mean,
                output_std,
                debug=False,
            )
        
        inference_funcs['v1'] = infer_v1
    
    # Load v4 model
    if args.v4_model and V4_AVAILABLE:
        logger.info(f"Loading v4 model: {args.v4_model}")
        try:
            checkpoint = torch.load(args.v4_model, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.v4_model, map_location='cpu')
        config = checkpoint.get('config', {})
        use_gnn = not args.no_gnn and ('gnn' in str(args.v4_model).lower() or config.get('use_gnn', False))
        
        input_size = config.get('input_size', config.get('input_features', 32))  # v4 uses 32D features
        
        # Import model classes from training scripts
        from train_swarm_model_v3_with_gnn import DynamicsAwareSwarmGRUModel_with_GNN
        from train_swarm_model_v2_dynamics_aware import DynamicsAwareSwarmGRUModel
        
        if use_gnn:
            model = DynamicsAwareSwarmGRUModel_with_GNN(
                input_size=input_size, 
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2), 
                num_gnn_heads=config.get('num_gnn_heads', config.get('gnn_heads', 4)),
                output_size=3, 
                dropout=config.get('dropout', 0.1),
                use_attention=config.get('use_attention', True),
                edge_threshold=config.get('edge_threshold', args.edge_threshold),
                gnn_hidden=config.get('gnn_hidden', 64),
                fusion_mode=config.get('gnn_fusion_mode', 'concat')
            )
        else:
            model = DynamicsAwareSwarmGRUModel(
                input_size=input_size, 
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2), 
                output_size=3,
                dropout=config.get('dropout', 0.1),
                use_attention=config.get('use_attention', True)
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models['v4'] = model
        
        # Load features and statistics for v4 (32D features)
        features_all = load_all_32d_features(args.features_32d_dir, args.agents, use_subset=args.use_subset)
        feature_mean, feature_std = compute_feature_statistics(features_all, num_samples=1000)
        output_mean = np.array(checkpoint['output_mean'], dtype=np.float32)
        output_std = np.array(checkpoint['output_std'], dtype=np.float32)
        
        tail_analyzer = None if args.no_tail_enhancement else EnhancedTailDynamicsAnalyzer(
            short_window=3, medium_window=5, long_window=8
        )
        
        def infer_v4(X_batch, features_batch):
            features_norm = normalize_features(features_batch, feature_mean, feature_std)
            return infer_batch_v4_enhanced(
                models['v4'], features_norm, X_batch, device,
                output_mean, output_std,
                tail_analyzer=tail_analyzer,
                edge_threshold=args.edge_threshold,
                use_gnn=use_gnn,
                use_tail_enhancement=not args.no_tail_enhancement,
                debug=False
            )
        inference_funcs['v4'] = infer_v4
        inference_funcs['v4_features'] = features_all
    
    # Load v3 model
    if args.v3_model and V3_AVAILABLE:
        logger.info(f"Loading v3 model: {args.v3_model}")
        try:
            checkpoint = torch.load(args.v3_model, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.v3_model, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Import v3 model class from training script
        from train_swarm_model_v3_with_gnn import DynamicsAwareSwarmGRUModel_with_GNN
        
        model = DynamicsAwareSwarmGRUModel_with_GNN(
            input_size=24, hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2), num_gnn_heads=config.get('num_gnn_heads', config.get('gnn_heads', 4)),
            output_size=3, dropout=config.get('dropout', 0.1),
            use_attention=config.get('use_attention', True),
            edge_threshold=config.get('edge_threshold', args.edge_threshold),
            gnn_hidden=config.get('gnn_hidden', 64),
            fusion_mode=config.get('gnn_fusion_mode', 'concat')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models['v3'] = model
        
        # Load precomputed 24D features for v3
        features_24d_all = load_all_24d_features(args.features_24d_dir, args.agents, use_subset=args.use_subset)
        if features_24d_all is not None:
            # Get statistics from checkpoint or compute from features
            if 'feature_mean' in checkpoint and 'feature_std' in checkpoint:
                feature_mean_all = np.array(checkpoint['feature_mean'], dtype=np.float32)
                feature_std_all = np.array(checkpoint['feature_std'], dtype=np.float32)
            else:
                # Compute from precomputed features
                subset_for_stats = features_24d_all[:min(1000, len(features_24d_all))].reshape(-1, 24)
                feature_mean_all = np.mean(subset_for_stats, axis=0)
                feature_std_all = np.std(subset_for_stats, axis=0)
                feature_std_all = np.where(feature_std_all < 1e-8, 1.0, feature_std_all)
            logger.info(f"Using precomputed 24D features for v3: shape={features_24d_all.shape}")
        else:
            # Fallback to computing features on the fly
            logger.warning("Precomputed 24D features not found, computing on the fly for v3")
            stats_sample_count = min(200, len(X_all))
            feature_mean_all, feature_std_all = estimate_stats_v3(
                X_all, dt=0.1, num_samples=stats_sample_count, seed=42
            )
            features_24d_all = None
        
        output_mean = np.array(checkpoint['output_mean'], dtype=np.float32)
        output_std = np.array(checkpoint['output_std'], dtype=np.float32)
        
        def infer_v3(X_batch, indices=None):
            if features_24d_all is not None and indices is not None:
                # Use precomputed features
                features_batch = features_24d_all[indices].astype(np.float32)
                # Normalize features
                mean_vec = feature_mean_all.reshape(1, 1, 1, 24)
                std_vec = feature_std_all.reshape(1, 1, 1, 24)
                features_batch = (features_batch - mean_vec) / (std_vec + 1e-8)
                features_batch = np.clip(features_batch, -5.0, 5.0)
            else:
                # Compute features on the fly
                features_batch = np.stack([
                    compute_features_v3(x, feature_mean_all, feature_std_all) for x in X_batch
                ], axis=0)
            return infer_batch_v3(
                models['v3'], features_batch, X_batch, device,
                output_mean, output_std,
                edge_threshold=args.edge_threshold,
                debug=False
            )
        inference_funcs['v3'] = infer_v3
        inference_funcs['v3_features'] = features_24d_all
    
    # Load v2 model
    if args.v2_model and V2_AVAILABLE:
        logger.info(f"Loading v2 model: {args.v2_model}")
        try:
            checkpoint = torch.load(args.v2_model, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.v2_model, map_location='cpu')
        
        # v2 model structure - use model class from training script
        from train_swarm_model_v2_dynamics_aware import DynamicsAwareSwarmGRUModel as V2Model
        config = checkpoint.get('config', {})
        input_size = config.get('input_size', 24)  # v2 uses 24D features
        
        model = V2Model(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            output_size=3,
            dropout=config.get('dropout', 0.3),
            use_attention=config.get('use_attention', True)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models['v2'] = model
        
        # Load precomputed 24D features for v2
        features_24d_all = load_all_24d_features(args.features_24d_dir, args.agents, use_subset=args.use_subset)
        if features_24d_all is not None:
            # Get statistics from checkpoint or compute from features
            if 'feature_mean' in checkpoint and 'feature_std' in checkpoint:
                feature_mean_all = np.array(checkpoint['feature_mean'], dtype=np.float32)
                feature_std_all = np.array(checkpoint['feature_std'], dtype=np.float32)
            else:
                # Compute from precomputed features
                subset_for_stats = features_24d_all[:min(1000, len(features_24d_all))].reshape(-1, 24)
                feature_mean_all = np.mean(subset_for_stats, axis=0)
                feature_std_all = np.std(subset_for_stats, axis=0)
                feature_std_all = np.where(feature_std_all < 1e-8, 1.0, feature_std_all)
            logger.info(f"Using precomputed 24D features for v2: shape={features_24d_all.shape}")
        else:
            # Fallback to computing features on the fly
            logger.warning("Precomputed 24D features not found, computing on the fly for v2")
            stats_sample_count = min(200, len(X_all))
            if V2_AVAILABLE and 'estimate_stats_v2' in globals():
                feature_mean_all, feature_std_all = estimate_stats_v2(
                    X_all, dt=0.1, num_samples=stats_sample_count, seed=42
                )
            else:
                # Manual computation if estimate_stats_v2 not available
                sample_indices = np.random.choice(len(X_all), stats_sample_count, replace=False)
                feature_chunks = []
                for idx in sample_indices:
                    features = compute_features_v2(X_all[idx], feature_mean_all=None, feature_std_all=None)
                    feature_chunks.append(features.reshape(-1, 24))
                all_features = np.concatenate(feature_chunks, axis=0)
                feature_mean_all = np.mean(all_features, axis=0)
                feature_std_all = np.std(all_features, axis=0)
                feature_std_all = np.where(feature_std_all < 1e-8, 1.0, feature_std_all)
            features_24d_all = None
        
        output_mean = np.array(checkpoint['output_mean'], dtype=np.float32)
        output_std = np.array(checkpoint['output_std'], dtype=np.float32)
        
        def infer_v2(X_batch, indices=None):
            if features_24d_all is not None and indices is not None:
                # Use precomputed features
                features_batch = features_24d_all[indices].astype(np.float32)
                # Normalize features
                mean_vec = feature_mean_all.reshape(1, 1, 1, 24)
                std_vec = feature_std_all.reshape(1, 1, 1, 24)
                features_batch = (features_batch - mean_vec) / (std_vec + 1e-8)
                features_batch = np.clip(features_batch, -5.0, 5.0)
            else:
                # Compute features on the fly
                features_batch = np.stack([
                    compute_features_v2(x, feature_mean_all, feature_std_all) for x in X_batch
                ], axis=0)
            return infer_batch_v2(
                models['v2'], features_batch, X_batch, device,
                output_mean, output_std,
                debug=False
            )
        inference_funcs['v2'] = infer_v2
        inference_funcs['v2_features'] = features_24d_all
    
    # Process each sample
    all_metrics = []
    for i, sample_idx in enumerate(sample_indices):
        logger.info(f"\nProcessing sample {sample_idx} (index {i}/{len(sample_indices)-1})...")
        
        # Extract sample data
        X_sample = X_all[sample_idx:sample_idx+1]  # (1, seq_in, agents, 3)
        Y_sample = Y_all[sample_idx:sample_idx+1]  # (1, seq_out, agents, 3)
        
        history = X_sample[0]  # (seq_in, agents, 3)
        truth = Y_sample[0]    # (seq_out, agents, 3)
        
        # Get predictions from each model
        predictions_dict = {}

        # v1 prediction (16D features computed on-the-fly)
        if 'v1' in inference_funcs:
            try:
                pred_v1 = inference_funcs['v1'](X_sample, indices=None)[0]
                predictions_dict['v1'] = pred_v1
            except Exception as e:
                logger.warning(f"v1 prediction failed for sample {sample_idx}: {e}")
        
        # v4 prediction (uses 32D features)
        if 'v4' in inference_funcs:
            if 'v4_features' in inference_funcs and inference_funcs['v4_features'] is not None:
                features_sample = inference_funcs['v4_features'][sample_idx:sample_idx+1]
                pred_v4 = inference_funcs['v4'](X_sample, features_sample)[0]
                predictions_dict['v4'] = pred_v4
            else:
                logger.warning("v4 features not available")
        
        # v3 prediction (uses 24D features)
        if 'v3' in inference_funcs:
            try:
                if 'v3_features' in inference_funcs and inference_funcs['v3_features'] is not None:
                    # Use precomputed features with index
                    pred_v3 = inference_funcs['v3'](X_sample, indices=np.array([sample_idx]))[0]
                else:
                    # Compute features on the fly
                    pred_v3 = inference_funcs['v3'](X_sample, indices=None)[0]
                predictions_dict['v3'] = pred_v3
            except Exception as e:
                logger.warning(f"v3 prediction failed for sample {sample_idx}: {e}")
        
        # v2 prediction (uses 24D features)
        if 'v2' in inference_funcs:
            try:
                if 'v2_features' in inference_funcs and inference_funcs['v2_features'] is not None:
                    # Use precomputed features with index
                    pred_v2 = inference_funcs['v2'](X_sample, indices=np.array([sample_idx]))[0]
                else:
                    # Compute features on the fly
                    pred_v2 = inference_funcs['v2'](X_sample, indices=None)[0]
                predictions_dict['v2'] = pred_v2
            except Exception as e:
                logger.warning(f"v2 prediction failed for sample {sample_idx}: {e}")
        
        # Plot comparison
        metrics = plot_model_comparison(history, truth, predictions_dict, sample_idx, 
                                       args.agents, args.output_dir)
        all_metrics.append(metrics)
        
        # Print metrics for this sample
        logger.info(f"  Sample {sample_idx} Metrics:")
        for model_name, model_metrics in metrics['metrics'].items():
            logger.info(f"    {model_name}: MAE={model_metrics['mae']:.6f}m, "
                       f"RMSE={model_metrics['rmse']:.6f}m, "
                       f"MAPE={model_metrics['mape']:.4f}%, "
                       f"MAE_X={model_metrics['mae_x']:.6f}m, "
                       f"MAE_Y={model_metrics['mae_y']:.6f}m, "
                       f"MAE_Z={model_metrics['mae_z']:.6f}m")
    
    # Compute aggregate statistics
    output_path = Path(args.output_dir)
    
    # Aggregate metrics across all samples
    aggregate_metrics = {}
    model_list = list(predictions_dict.keys())
    
    for model_name in model_list:
        model_key = model_name.upper()
        metrics_list = [m['metrics'][model_key] for m in all_metrics if model_key in m['metrics']]
        
        if metrics_list:
            aggregate_metrics[model_key] = {
                'avg_mae': float(np.mean([m['mae'] for m in metrics_list])),
                'std_mae': float(np.std([m['mae'] for m in metrics_list])),
                'avg_rmse': float(np.mean([m['rmse'] for m in metrics_list])),
                'std_rmse': float(np.std([m['rmse'] for m in metrics_list])),
                'avg_mape': float(np.mean([m['mape'] for m in metrics_list])),
                'std_mape': float(np.std([m['mape'] for m in metrics_list])),
                'avg_mae_x': float(np.mean([m['mae_x'] for m in metrics_list])),
                'avg_mae_y': float(np.mean([m['mae_y'] for m in metrics_list])),
                'avg_mae_z': float(np.mean([m['mae_z'] for m in metrics_list])),
                'min_mae': float(np.min([m['mae'] for m in metrics_list])),
                'max_mae': float(np.max([m['mae'] for m in metrics_list])),
                'min_mape': float(np.min([m['mape'] for m in metrics_list])),
                'max_mape': float(np.max([m['mape'] for m in metrics_list]))
            }
    
    # Save summary report
    report = {
        'timestamp': datetime.now().isoformat(),
        'sample_indices': [int(idx) for idx in sample_indices],
        'num_samples': len(sample_indices),
        'models_compared': [m.upper() for m in model_list],
        'aggregate_metrics': aggregate_metrics,
        'samples': all_metrics
    }
    
    report_file = output_path / f'comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nComparison report saved: {report_file}")
    
    # Print aggregate statistics
    logger.info("\n" + "="*80)
    logger.info("AGGREGATE METRICS SUMMARY")
    logger.info("="*80)
    for model_name, agg_metrics in aggregate_metrics.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  MAE: {agg_metrics['avg_mae']:.6f} ± {agg_metrics['std_mae']:.6f} m "
                   f"(min: {agg_metrics['min_mae']:.6f}, max: {agg_metrics['max_mae']:.6f})")
        logger.info(f"  RMSE: {agg_metrics['avg_rmse']:.6f} ± {agg_metrics['std_rmse']:.6f} m")
        logger.info(f"  MAPE: {agg_metrics['avg_mape']:.4f} ± {agg_metrics['std_mape']:.4f}% "
                   f"(min: {agg_metrics['min_mape']:.4f}, max: {agg_metrics['max_mape']:.4f})")
        logger.info(f"  MAE per axis: X={agg_metrics['avg_mae_x']:.6f}, "
                   f"Y={agg_metrics['avg_mae_y']:.6f}, Z={agg_metrics['avg_mae_z']:.6f} m")
    logger.info("="*80)
    
    logger.info("\n✓ All comparison figures generated successfully!")


if __name__ == '__main__':
    main()
