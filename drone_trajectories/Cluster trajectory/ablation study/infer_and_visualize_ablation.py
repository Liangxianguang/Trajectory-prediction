#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study Inference and Visualization Script (Enhanced v4-style)
======================================================================

Real-time inference and visualization with comprehensive evaluation:
  1. Random or specified sample selection
  2. Inference all 5 ablation models on each sample
  3. Real-time 3D comparison figure plotting
  4. Physical constraints for smooth trajectory reconstruction
  5. Compute and save detailed evaluation metrics
  6. Generate publication-ready summary figures
  7. Per-sample and aggregate statistics

✓ Enhancements (v4-style):
  - Comprehensive feature management (16D and 32D)
  - Automatic feature normalization
  - Physical constraints with acceleration smoothing
  - Multi-dimensional metric computation (per-step, per-agent, per-axis)
  - Detailed visualization with metrics tables
  - JSON export for results processing

Usage examples:
    # Randomly select 50 samples for inference and visualization
    python infer_and_visualize_ablation.py \
        --data_dir ../swarm_segments \
        --num_samples 50 \
        --output_dir ablation_viz_results \
        --features_32d_dir ../features_32d \
        --features_16d_dir ../features_16d \
        --use_subset

    # Specific sample indices with custom feature directories
    python infer_and_visualize_ablation.py \
        --data_dir ../swarm_segments \
        --sample_indices "100,500,1000,2000" \
        --output_dir ablation_viz_results \
        --features_16d_dir ../features_16d \
        --features_32d_dir ../features_32d \
        --use_subset \
        --visualize

    # Full evaluation with smoothing parameters
    python infer_and_visualize_ablation.py \
        --data_dir ../swarm_segments \
        --num_samples 100 \
        --output_dir ablation_viz_results \
        --smoothing_weight 0.3 \
        --batch_size 64
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import logging
import json
import csv
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加父路径到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ablation_models import BaselineGRUModel, GNNFeatureModel
from ablation_train_utils import load_ablation_data
from train_swarm_model_v2_dynamics_aware import (
    DynamicsAwareSwarmGRUModel,
    DynamicsAwareLoss,
)
from train_swarm_model_v3_with_gnn import DynamicsAwareSwarmGRUModel_with_GNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class AblationModelEnsemble:
    """Ensemble of ablation study models"""
    
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.stats = {}
        self.configs = {}
        
    def load_experiment(self, exp_id, exp_name, ablation_results_dir):
        """Load single ablation experiment"""
        ckpt_dir = Path(ablation_results_dir) / f"ablation_results_agents_3_{exp_name}"
        
        # Load config
        config_file = ckpt_dir / f"config_agents_3_{exp_name}.json"
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.configs[exp_id] = config
        
        # Load statistics
        stats_file = ckpt_dir / f"stats_agents_3_{exp_name}.npz"
        stats = np.load(stats_file)
        self.stats[exp_id] = {
            'input_mean': stats['input_mean'],
            'input_std': stats['input_std'],
            'output_mean': stats['output_mean'],
            'output_std': stats['output_std'],
        }
        
        # Load best model
        best_model_file = ckpt_dir / f"best_model_agents_3_{exp_name}.pt"
        checkpoint = torch.load(best_model_file, map_location=self.device, weights_only=False)
        
        # Create model
        if exp_id == 1:
            model = BaselineGRUModel(
                input_size=16,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=3,
                dropout=0.3
            )
        elif exp_id == 2:
            model = DynamicsAwareSwarmGRUModel(
                input_size=32,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=3,
                dropout=0.3,
                use_attention=True
            )
        elif exp_id == 3:
            model = DynamicsAwareSwarmGRUModel_with_GNN(
                input_size=16,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=3,
                dropout=0.3,
                use_attention=True,
                gnn_hidden=config.get('gnn_hidden', 64),
                num_gnn_heads=config.get('gnn_heads', 4),
                edge_threshold=config.get('edge_threshold', 5.0),
                fusion_mode='concat'
            )
        elif exp_id == 4:
            model = GNNFeatureModel(
                input_size=32,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=3,
                dropout=0.3,
                use_attention=False,
                gnn_hidden=config.get('gnn_hidden', 64),
                num_gnn_heads=config.get('gnn_heads', 4),
                edge_threshold=config.get('edge_threshold', 5.0),
                fusion_mode='concat'
            )
        elif exp_id == 5:
            model = DynamicsAwareSwarmGRUModel_with_GNN(
                input_size=32,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=3,
                dropout=0.3,
                use_attention=True,
                gnn_hidden=config.get('gnn_hidden', 64),
                num_gnn_heads=config.get('gnn_heads', 4),
                edge_threshold=config.get('edge_threshold', 5.0),
                fusion_mode='concat'
            )
        else:
            raise ValueError(f"Unknown experiment ID: {exp_id}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.models[exp_id] = model
        
        print(f"✓ Loaded Experiment {exp_id}: {config['description']}")
        return config
        
    def infer_batch(self, exp_id, features, x_orig):
        """Inference for single experiment"""
        model = self.models[exp_id]
        
        with torch.no_grad():
            result = model(features, x_orig, y=None, teacher_forcing_ratio=0.0)
            
            if isinstance(result, tuple):
                output_pos, output_vel, output_accel = result
            elif isinstance(result, dict):
                output_pos = result.get('output', result.get('pred_pos', result.get('y_hat')))
                output_vel = result.get('vel', None)
                output_accel = result.get('accel', None)
            else:
                output_pos = result
                output_vel = None
                output_accel = None
        
        return output_pos, output_vel, output_accel


def load_all_32d_features(features_dir, num_agents, use_subset=False):
    """Load all 32D features into memory at once (v4-style pre-loading)"""
    features_dir = Path(features_dir)
    subset_suffix = '_subset' if use_subset else ''
    
    feature_candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_32d.npz',
        features_dir / f'features_agents_{num_agents}_32d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
    ]
    
    for feat_path in feature_candidates:
        if feat_path.exists():
            try:
                logger.info(f"Pre-loading 32D features: {feat_path}")
                data = np.load(feat_path)
                features_all = np.asarray(data['features'])
                logger.info(f"  ✓ 32D features pre-loaded: {features_all.shape}")
                return features_all
            except Exception as e:
                logger.warning(f"Failed to pre-load features from {feat_path}: {e}")
    return None


def load_features_16d(features_16d_dir, num_agents, use_subset=False):
    """Load all 16D features into memory at once (v4-style pre-loading)"""
    features_dir = Path(features_16d_dir)
    subset_suffix = '_subset' if use_subset else ''
    
    # Try different file naming patterns
    feature_candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_16d.npz',
        features_dir / f'features_agents_{num_agents}_16d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}.npz',
    ]
    
    for feat_path in feature_candidates:
        if feat_path.exists():
            try:
                logger.info(f"Pre-loading 16D features: {feat_path}")
                features_file = np.load(feat_path)
                features_16d = features_file['features']  # (N, seq, agents, 16)
                logger.info(f"  ✓ 16D features pre-loaded: {features_16d.shape}")
                return features_16d
            except Exception as e:
                logger.warning(f"Failed to pre-load 16D features: {e}")
    
    logger.warning(f"16D features not found in {features_dir}")
    return None


def normalize_features(features, feature_mean=None, feature_std=None):
    """
    Normalize features using Z-score (v4-style, consistent with training)
    
    Args:
        features: (N, seq, agents, D) or (seq, agents, D) features
        feature_mean: (D,) feature mean
        feature_std: (D,) feature std
    
    Returns:
        normalized_features: Same shape as input, normalized
    """
    if feature_mean is None or feature_std is None:
        logger.warning("Feature statistics not provided, returning unnormalized features")
        return features
    
    # Z-score normalization
    safe_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    normalized = (features - feature_mean[np.newaxis, ...]) / safe_std[np.newaxis, ...]
    
    # Clip outliers (beyond +/-5 sigma)
    normalized = np.clip(normalized, -5.0, 5.0)
    
    return normalized.astype(np.float32)


def apply_physical_constraints(history, pred_delta, dt=0.1, smoothing_weight=0.2, constraint_relaxation=1.0):
    """
    Apply physical constraints with enhanced velocity-aware reconstruction (v4-enhanced).
    
    ✓ Key improvements for better velocity prediction:
    1. **Separate velocity direction and magnitude**: Extract direction from model, magnitude from input
    2. **Tangential acceleration analysis**: Distinguish speed changes from direction changes
    3. **Velocity magnitude scaling**: Match input sequence velocity characteristics
    4. **Progressive velocity constraints**: Allow reasonable speed variations (1.2x-1.5x range)
    5. **Multi-component acceleration blending**: Recent trend + global trend + model guidance
    
    This approach solves the velocity mismatch problem by:
    - Preserving model's intended direction changes
    - Maintaining realistic speed transitions from input sequence
    - Allowing natural acceleration/deceleration patterns
    - Preventing unrealistic velocity jumps
    
    Args:
        history: (seq_in, agents, 3) Input position history
        pred_delta: (seq_out, agents, 3) Predicted displacement 
        dt: Time step
        smoothing_weight: Acceleration smoothing weight (0-1), default 0.3
        constraint_relaxation: Multiplier (>1 loosens velocity/acc limits)
    
    Returns:
        reconstructed: (seq_out, agents, 3) Smooth reconstructed absolute positions
        
    Algorithm Overview:
        For each prediction step:
        1. Extract velocity direction from model's displacement prediction
        2. Compute velocity magnitude based on input sequence + acceleration trends
        3. Combine direction (model) + magnitude (physics) for desired velocity
        4. Compute required acceleration with multi-component blending
        5. Apply progressive velocity constraints (comfort zone: 1.2x, max: 1.5x)
        6. Update position with constrained velocity
    """
    history = np.array(history, dtype=np.float32)
    seq_in, num_agents, _ = history.shape
    
    if seq_in < 2:
        # Not enough history, just use raw displacement
        return history[-1:, :, :] + pred_delta
    
    # ✓ Compute velocity from history (key: use last steps for current state)
    history_vel = np.diff(history, axis=0) / dt  # (seq_in-1, agents, 3)
    
    if history_vel.shape[0] == 0:
        # No velocity, use raw displacement
        return history[-1:, :, :] + pred_delta
    
    # ✓ v4-style: Use last 5 steps for stable velocity estimate
    # This captures the actual motion state at input sequence end
    if history_vel.shape[0] >= 5:
        last_vel = np.mean(history_vel[-5:, :, :], axis=0)  # (agents, 3) - average of last 5 steps
    else:
        last_vel = np.mean(history_vel, axis=0)  # (agents, 3)
    
    # ✓ Compute acceleration from history
    # Important: This tells us if motion is accelerating or decelerating
    history_acc = np.diff(history_vel, axis=0) / dt if history_vel.shape[0] > 1 else np.zeros((1, num_agents, 3), dtype=np.float32)
    
    # ✓ Overall acceleration trend (for global smoothing)
    avg_acc = history_acc.mean(axis=0) if history_acc.shape[0] > 0 else np.zeros((num_agents, 3), dtype=np.float32)
    
    # ✓ Recent acceleration trend (last few steps - current acceleration state)
    if history_acc.shape[0] >= 3:
        recent_acc = history_acc[-3:].mean(axis=0)  # (agents, 3) - trend of recent acceleration
    elif history_acc.shape[0] > 0:
        recent_acc = history_acc[-1, :, :]  # (agents, 3)
    else:
        recent_acc = np.zeros((num_agents, 3), dtype=np.float32)
    
    # ✓ Compute maximum velocity and acceleration from history for constraints
    # These define the physical envelope of motion
    vel_norms = np.linalg.norm(history_vel, axis=2)  # (seq_in-1, agents)
    max_vel = np.maximum(np.max(vel_norms, axis=0), 1e-3)  # (agents,) - historical speed limit
    
    acc_norms = np.linalg.norm(history_acc, axis=2)  # (seq_in-2, agents)
    max_acc = np.maximum(np.max(acc_norms, axis=0), 1e-3)  # (agents,) - historical acceleration limit

    relaxation = constraint_relaxation if constraint_relaxation > 0 else 1.0
    max_vel = max_vel * relaxation
    max_acc = max_acc * relaxation
    
    # Start from last position with actual last velocity
    current_pos = history[-1:, :, :].copy()  # (1, agents, 3)
    current_vel = last_vel.copy()  # (agents, 3)
    seq_out = pred_delta.shape[0]
    
    reconstructed = np.zeros((seq_out, num_agents, 3), dtype=np.float32)
    
    # ✓ Enhanced velocity-aware reconstruction (v4-style velocity matching)
    # Key innovation: Separate velocity direction (from model) and magnitude (from input sequence)
    # This preserves both model flexibility and input sequence velocity characteristics
    
    # Compute initial velocity magnitude reference from input sequence
    last_vel_mag = np.linalg.norm(last_vel, axis=1, keepdims=True)  # (agents, 1)
    last_vel_mag = np.maximum(last_vel_mag, 1e-3)  # Avoid division by zero
    
    # Get the most recent velocity for direction reference
    if history_vel.shape[0] >= 1:
        last_vel_recent = history_vel[-1, :, :]  # (agents, 3) - most recent velocity
    else:
        last_vel_recent = last_vel
    
    for step in range(seq_out):
        # ========================================================================
        # Step 1: Extract velocity direction from model prediction
        # ========================================================================
        if step == 0:
            # First step: use the first predicted delta
            step_delta = pred_delta[step, :, :]  # (agents, 3)
        else:
            # Subsequent steps: compute incremental delta
            step_delta = pred_delta[step, :, :] - pred_delta[step-1, :, :]  # (agents, 3)
        
        # Desired velocity direction from model prediction
        step_delta_norm = np.linalg.norm(step_delta, axis=1, keepdims=True) + 1e-8
        desired_vel_dir = step_delta / step_delta_norm  # (agents, 3) - unit direction vector
        
        # ========================================================================
        # Step 2: Compute velocity magnitude based on acceleration trends
        # ========================================================================
        # Project recent acceleration onto current velocity direction to get tangential component
        last_vel_norm = np.linalg.norm(last_vel, axis=1, keepdims=True) + 1e-8
        last_vel_unit = last_vel / last_vel_norm  # (agents, 3) - unit velocity vector
        
        # Tangential acceleration: how much acceleration affects speed (not direction)
        accel_tangent = np.sum(recent_acc * last_vel_unit, axis=1, keepdims=True)  # (agents, 1)
        
        # Adjust velocity magnitude based on acceleration trend with smooth transition
        # Positive tangential acceleration -> speed up, negative -> slow down
        accel_factor = 1.0 + np.tanh(accel_tangent * dt * 2.0) * 0.25  # Scale range: 0.75 to 1.25
        target_vel_mag = last_vel_mag * accel_factor  # (agents, 1)
        
        # ========================================================================
        # Step 3: Combine direction from model with magnitude from input sequence
        # ========================================================================
        desired_vel = desired_vel_dir * target_vel_mag  # (agents, 3)
        
        # ========================================================================
        # Step 4: Compute and constrain acceleration
        # ========================================================================
        raw_accel = (desired_vel - current_vel) / dt  # (agents, 3)
        
        # Blend with recent acceleration trend for smooth transitions
        # Use separate weights for magnitude and direction components
        accel_weight = 0.4  # Weight for recent acceleration influence
        constrained_accel = (
            (1 - smoothing_weight) * raw_accel +                                    # 70% model guidance
            smoothing_weight * (1 - accel_weight) * avg_acc +                      # 18% global trend
            smoothing_weight * accel_weight * recent_acc                           # 12% recent trend
        )  # (agents, 3)
        
        # Limit acceleration magnitude to historical bounds
        accel_norm = np.linalg.norm(constrained_accel, axis=1, keepdims=True)  # (agents, 1)
        accel_scale = np.minimum(1.0, max_acc[:, np.newaxis] / (accel_norm + 1e-8))
        constrained_accel = constrained_accel * accel_scale
        
        # ========================================================================
        # Step 5: Update velocity with enhanced constraints
        # ========================================================================
        new_vel = current_vel + constrained_accel * dt  # (agents, 3)
        
        # Enhanced velocity constraint: allow reasonable speed variations
        vel_norm = np.linalg.norm(new_vel, axis=1, keepdims=True)  # (agents, 1)
        
        # Allow up to 1.5x max velocity but prefer staying within 1.2x range
        # This preserves input sequence velocity characteristics while allowing model flexibility
        max_allowed_vel = max_vel[:, np.newaxis] * 1.5  # Maximum allowed speed
        comfort_vel = max_vel[:, np.newaxis] * 1.2      # Preferred maximum speed
        
        # Apply smooth scaling: no scaling if within comfort zone, gradual scaling beyond
        vel_scale = np.where(
            vel_norm <= comfort_vel, 
            1.0,  # No scaling within comfort zone
            np.minimum(1.0, max_allowed_vel / (vel_norm + 1e-8))  # Scale if beyond comfort zone
        )
        
        current_vel = new_vel * vel_scale
        
        # ========================================================================
        # Step 6: Update position
        # ========================================================================
        current_pos = current_pos + current_vel[np.newaxis, :, :] * dt  # (1, agents, 3)
        reconstructed[step] = current_pos[0, :, :]  # (agents, 3)
    
    return reconstructed


def compute_metrics(true_pos, pred_pos):
    """
    Compute comprehensive evaluation metrics for paper-quality analysis
    
    Args:
        true_pos: (seq_out, agents, 3) ground truth trajectory
        pred_pos: (seq_out, agents, 3) predicted trajectory
    
    Returns:
        dict with comprehensive metrics including:
        - Overall: MAE, RMSE, ADE, FDE, MAPE
        - Per-axis: MAE_X, MAE_Y, MAE_Z 
        - Per-step: MAE_per_step, RMSE_per_step, FDE_per_step
        - Per-agent: MAE_per_agent, RMSE_per_agent, FDE_per_agent
        - Statistics: Min, Max, Std, Median, P25, P75, P90, P95, P99
        - Trajectory: Initial_error, Final_error, Max_error_step
    """
    # Basic position errors: (seq_out, agents)
    errors = np.linalg.norm(pred_pos - true_pos, axis=-1)
    errors_flat = errors.flatten()
    
    # Per-axis errors: (seq_out, agents)
    errors_x = np.abs(pred_pos[..., 0] - true_pos[..., 0])
    errors_y = np.abs(pred_pos[..., 1] - true_pos[..., 1])  
    errors_z = np.abs(pred_pos[..., 2] - true_pos[..., 2])
    
    # Overall trajectory metrics
    mae = float(np.mean(errors_flat))
    rmse = float(np.sqrt(np.mean(errors_flat ** 2)))
    ade = mae  # Average Displacement Error = MAE
    fde = float(np.mean(errors[-1, :]))  # Final Displacement Error (last step)
    
    # MAPE: Mean Absolute Percentage Error
    true_distances = np.linalg.norm(true_pos, axis=2)  # Distance from origin
    epsilon = 1e-6
    valid_mask = true_distances > epsilon
    if np.any(valid_mask):
        mape = float(np.mean((errors[valid_mask] / true_distances[valid_mask]) * 100))
    else:
        mape = 0.0
    
    # Per-axis MAE
    mae_x = float(np.mean(errors_x))
    mae_y = float(np.mean(errors_y))
    mae_z = float(np.mean(errors_z))
    
    # Per-step metrics: (seq_out,)
    mae_per_step = [float(np.mean(errors[t, :])) for t in range(errors.shape[0])]
    rmse_per_step = [float(np.sqrt(np.mean(errors[t, :] ** 2))) for t in range(errors.shape[0])]
    fde_per_step = mae_per_step  # FDE per step = MAE per step
    
    # Per-agent metrics: (agents,)
    mae_per_agent = [float(np.mean(errors[:, a])) for a in range(errors.shape[1])]
    rmse_per_agent = [float(np.sqrt(np.mean(errors[:, a] ** 2))) for a in range(errors.shape[1])]
    fde_per_agent = [float(errors[-1, a]) for a in range(errors.shape[1])]  # Final error for each agent
    
    # Statistical distributions
    percentiles = [25, 50, 75, 90, 95, 99]
    error_percentiles = {f'P{p}': float(np.percentile(errors_flat, p)) for p in percentiles}
    
    # Trajectory analysis
    initial_error = float(np.mean(errors[0, :]))  # First step error
    final_error = fde  # Last step error
    max_error_step = int(np.argmax(np.mean(errors, axis=1)))  # Step with highest average error
    max_error_value = float(np.max(np.mean(errors, axis=1)))  # Highest average error across steps
    
    # Error growth analysis
    error_trend = np.mean(errors, axis=1)  # Average error per step
    error_growth_rate = float((error_trend[-1] - error_trend[0]) / len(error_trend)) if len(error_trend) > 1 else 0.0
    
    # Velocity error analysis (if more than 1 step)
    if pred_pos.shape[0] > 1:
        # Compute velocities
        true_vel = np.diff(true_pos, axis=0)  # (seq_out-1, agents, 3)
        pred_vel = np.diff(pred_pos, axis=0)  # (seq_out-1, agents, 3)
        
        vel_errors = np.linalg.norm(pred_vel - true_vel, axis=-1)  # (seq_out-1, agents)
        vel_mae = float(np.mean(vel_errors))
        vel_rmse = float(np.sqrt(np.mean(vel_errors ** 2)))
        
        # Speed error (magnitude only)
        true_speed = np.linalg.norm(true_vel, axis=-1)  # (seq_out-1, agents)
        pred_speed = np.linalg.norm(pred_vel, axis=-1)  # (seq_out-1, agents)
        speed_error_mae = float(np.mean(np.abs(pred_speed - true_speed)))
    else:
        vel_mae = 0.0
        vel_rmse = 0.0
        speed_error_mae = 0.0
    
    # Compile comprehensive metrics dictionary
    metrics = {
        # ================== Overall Metrics ==================
        'MAE': mae,
        'RMSE': rmse,
        'ADE': ade,  # Average Displacement Error
        'FDE': fde,  # Final Displacement Error
        'MAPE': mape,  # Mean Absolute Percentage Error
        
        # ================== Per-Axis Metrics ==================
        'MAE_X': mae_x,
        'MAE_Y': mae_y,
        'MAE_Z': mae_z,
        
        # ================== Per-Step Metrics ==================
        'MAE_per_step': mae_per_step,
        'RMSE_per_step': rmse_per_step,
        'FDE_per_step': fde_per_step,
        
        # ================== Per-Agent Metrics ==================
        'MAE_per_agent': mae_per_agent,
        'RMSE_per_agent': rmse_per_agent,
        'FDE_per_agent': fde_per_agent,
        
        # ================== Statistical Distribution ==================
        'Min': float(np.min(errors_flat)),
        'Max': float(np.max(errors_flat)),
        'Std': float(np.std(errors_flat)),
        'Median': float(np.median(errors_flat)),
        **error_percentiles,  # P25, P50, P75, P90, P95, P99
        
        # ================== Trajectory Analysis ==================
        'Initial_error': initial_error,
        'Final_error': final_error,
        'Max_error_step': max_error_step,
        'Max_error_value': max_error_value,
        'Error_growth_rate': error_growth_rate,
        
        # ================== Velocity & Speed Metrics ==================
        'Velocity_MAE': vel_mae,
        'Velocity_RMSE': vel_rmse,
        'Speed_error_MAE': speed_error_mae,
        
        # ================== Legacy Metrics (for compatibility) ==================
        'Max_legacy': float(np.max(errors_flat)),  # Same as Max, kept for compatibility
        'Min_legacy': float(np.min(errors_flat)),  # Same as Min, kept for compatibility
        'Std_legacy': float(np.std(errors_flat)),  # Same as Std, kept for compatibility
    }
    
    return metrics


def plot_sample_comparison(sample_data, output_dir, sample_idx):
    """Plot comparison of 5 ablation models for a single sample (v4-enhanced visualization)"""
    
    # Configuration - Professional color scheme inspired by visualize_model_comparison.py
    colors = {
        1: '#E74C3C',  # Bright Red - Baseline
        2: '#3498DB',  # Bright Blue - Features+BiCA  
        3: '#9B59B6',  # Purple - GNN+BiCA
        4: '#27AE60',  # Bright Green - GNN+Features
        5: '#E67E22',  # Orange - Full Model
        'truth': '#2C3E50',  # Dark Blue-Gray - Ground Truth
        'history': '#34495E'  # Slate Gray - History
    }
    
    labels = {
        1: 'Exp1: Baseline GRU (16D)',
        2: 'Exp2: Dynamics+BiCA (32D)', 
        3: 'Exp3: GNN+BiCA (16D)',
        4: 'Exp4: GNN+Features (32D)',
        5: 'Exp5: Full Model (32D)',
    }
    
    # Model markers and line styles for distinction
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    linestyles = {1: '--', 2: '-.', 3: ':', 4: '--', 5: '-'}
    
    true_pos = sample_data['true_pos']  # (seq_out, A, 3)
    history_traj = sample_data['history_traj']  # (seq_in, A, 3)
    num_agents = true_pos.shape[1]
    
    # Get input history dimensions
    seq_in = history_traj.shape[0]
    seq_out = true_pos.shape[0]
    
    # Create high-resolution figure with professional styling
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(24, 16))  # Larger figure for better quality
    fig.suptitle(f'Ablation Study Sample #{sample_idx} - 5 Models Comparison\n'
                f'{num_agents} Agents × {seq_in} History + {seq_out} Future Steps', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 1. 3D Trajectory Comparison (All Agents)
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    
    for agent_id in range(num_agents):
        # History trajectory (blue-gray, prominent)
        ax3d.plot(history_traj[:, agent_id, 0], history_traj[:, agent_id, 1], history_traj[:, agent_id, 2],
                 'o-', color=colors['history'], linewidth=3.0, markersize=5, alpha=0.8,
                 label='History' if agent_id == 0 else '', zorder=1)
        
        # Get connection point (last position of history)
        last_point = history_traj[-1, agent_id, :]
        
        # Ground truth trajectory (dark, prominent)
        true_traj = np.vstack([last_point, true_pos[:, agent_id, :]])
        ax3d.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
                 's-', color=colors['truth'], linewidth=3.5, markersize=8, alpha=0.9,
                 label='Ground Truth' if agent_id == 0 else '', zorder=10)
        
        # Model predictions (vibrant colors with distinct styles)
        for exp_id in range(1, 6):
            pred = sample_data[f'exp{exp_id}_pred']  # (seq_out, A, 3)
            pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
            ax3d.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
                     f'{markers[exp_id]}{linestyles[exp_id]}',
                     color=colors[exp_id], linewidth=3.0, markersize=6, alpha=0.85,
                     label=labels[exp_id] if agent_id == 0 else '', zorder=5)
    
    ax3d.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax3d.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax3d.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax3d.set_title(f'3D Trajectory Comparison\nAll {num_agents} Agents', fontsize=13, fontweight='bold')
    ax3d.legend(fontsize=10, loc='upper left', ncol=2)
    ax3d.grid(True, alpha=0.3)
    
    # 2. XY Plane Projection
    ax_xy = fig.add_subplot(2, 3, 2)
    for agent_id in range(num_agents):
        # History trajectory (blue-gray)
        ax_xy.plot(history_traj[:, agent_id, 0], history_traj[:, agent_id, 1],
                  'o-', color=colors['history'], linewidth=3.0, markersize=5, alpha=0.8,
                  label='History' if agent_id == 0 else '', zorder=1)
        
        # Get connection point
        last_point = history_traj[-1, agent_id, :]
        
        # Ground truth (prominent dark line)
        true_traj = np.vstack([last_point, true_pos[:, agent_id, :]])
        ax_xy.plot(true_traj[:, 0], true_traj[:, 1],
                  's-', color=colors['truth'], linewidth=3.5, markersize=8, alpha=0.9,
                  label='Ground Truth' if agent_id == 0 else '', zorder=10)
        
        # Model predictions
        for exp_id in range(1, 6):
            pred = sample_data[f'exp{exp_id}_pred']
            pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
            ax_xy.plot(pred_traj[:, 0], pred_traj[:, 1],
                      f'{markers[exp_id]}{linestyles[exp_id]}',
                      color=colors[exp_id], linewidth=3.0, markersize=6, alpha=0.85,
                      label=labels[exp_id] if agent_id == 0 else '', zorder=5)
    
    ax_xy.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax_xy.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax_xy.set_title('XY Plane Projection', fontsize=12, fontweight='bold')
    ax_xy.legend(fontsize=10, loc='best', ncol=2)
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect('equal', adjustable='box')
    
    # 3. XZ Plane Projection  
    ax_xz = fig.add_subplot(2, 3, 3)
    for agent_id in range(num_agents):
        # History trajectory (blue-gray)
        ax_xz.plot(history_traj[:, agent_id, 0], history_traj[:, agent_id, 2],
                  'o-', color=colors['history'], linewidth=3.0, markersize=5, alpha=0.8,
                  label='History' if agent_id == 0 else '', zorder=1)
        
        # Get connection point
        last_point = history_traj[-1, agent_id, :]
        
        # Ground truth (prominent dark line)
        true_traj = np.vstack([last_point, true_pos[:, agent_id, :]])
        ax_xz.plot(true_traj[:, 0], true_traj[:, 2],
                  's-', color=colors['truth'], linewidth=3.5, markersize=8, alpha=0.9, zorder=10)
        
        # Model predictions
        for exp_id in range(1, 6):
            pred = sample_data[f'exp{exp_id}_pred']
            pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
            ax_xz.plot(pred_traj[:, 0], pred_traj[:, 2],
                      f'{markers[exp_id]}{linestyles[exp_id]}',
                      color=colors[exp_id], linewidth=3.0, markersize=6, alpha=0.85, zorder=5)
    
    ax_xz.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax_xz.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
    ax_xz.set_title('XZ Plane Projection', fontsize=12, fontweight='bold')
    ax_xz.grid(True, alpha=0.3)
    ax_xz.set_aspect('equal', adjustable='box')
    
    # 4. YZ Plane Projection
    ax_yz = fig.add_subplot(2, 3, 4)
    for agent_id in range(num_agents):
        # History trajectory (blue-gray)
        ax_yz.plot(history_traj[:, agent_id, 1], history_traj[:, agent_id, 2],
                  'o-', color=colors['history'], linewidth=3.0, markersize=5, alpha=0.8,
                  label='History' if agent_id == 0 else '', zorder=1)
        
        # Get connection point
        last_point = history_traj[-1, agent_id, :]
        
        # Ground truth (prominent dark line)
        true_traj = np.vstack([last_point, true_pos[:, agent_id, :]])
        ax_yz.plot(true_traj[:, 1], true_traj[:, 2],
                  's-', color=colors['truth'], linewidth=3.5, markersize=8, alpha=0.9, zorder=10)
        
        # Model predictions
        for exp_id in range(1, 6):
            pred = sample_data[f'exp{exp_id}_pred']
            pred_traj = np.vstack([last_point, pred[:, agent_id, :]])
            ax_yz.plot(pred_traj[:, 1], pred_traj[:, 2],
                      f'{markers[exp_id]}{linestyles[exp_id]}',
                      color=colors[exp_id], linewidth=3.0, markersize=6, alpha=0.85, zorder=5)
    
    ax_yz.set_xlabel('Y (m)', fontsize=12, fontweight='bold')
    ax_yz.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
    ax_yz.set_title('YZ Plane Projection', fontsize=12, fontweight='bold')
    ax_yz.grid(True, alpha=0.3)
    ax_yz.set_aspect('equal', adjustable='box')
    
    # 5. Per-Step Error Comparison (Enhanced)
    ax_error = fig.add_subplot(2, 3, 5)
    steps = np.arange(true_pos.shape[0])
    
    for exp_id in range(1, 6):
        pred = sample_data[f'exp{exp_id}_pred']
        errors = np.linalg.norm(pred - true_pos, axis=-1)  # (seq_out, A)
        mean_error = np.mean(errors, axis=1)  # Average across agents
        ax_error.plot(steps, mean_error, f'{markers[exp_id]}-',
                     color=colors[exp_id], label=f'Exp{exp_id}', 
                     linewidth=3.0, markersize=7, alpha=0.9)
    
    ax_error.set_xlabel('Prediction Step', fontsize=12, fontweight='bold')
    ax_error.set_ylabel('Mean Position Error (m)', fontsize=12, fontweight='bold')
    ax_error.set_title('Per-Step Error Comparison\n(All Agents Average)', fontsize=12, fontweight='bold')
    ax_error.legend(fontsize=10, loc='best')
    ax_error.grid(True, alpha=0.3)
    ax_error.set_xticks(steps)
    
    # 6. Overall MAE & RMSE Comparison (Enhanced Bar Chart)
    ax_mae = fig.add_subplot(2, 3, 6)
    exp_ids = list(range(1, 6))
    mae_values = [sample_data.get(f'metrics_exp{exp_id}', {}).get('MAE', 0) for exp_id in exp_ids]
    rmse_values = [sample_data.get(f'metrics_exp{exp_id}', {}).get('RMSE', 0) for exp_id in exp_ids]
    
    x_pos = np.arange(len(exp_ids))
    width = 0.35
    
    # MAE bars
    bars1 = ax_mae.bar(x_pos - width/2, mae_values, width,
                      color=[colors[i] for i in exp_ids], alpha=0.85,
                      edgecolor='black', linewidth=2.0, label='MAE')
    
    # RMSE bars (with pattern)
    bars2 = ax_mae.bar(x_pos + width/2, rmse_values, width,
                      color=[colors[i] for i in exp_ids], alpha=0.6,
                      edgecolor='black', linewidth=2.0, label='RMSE', hatch='///')
    
    # Add value labels on bars
    for bar, mae in zip(bars1, mae_values):
        height = bar.get_height()
        ax_mae.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mae:.4f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    for bar, rmse in zip(bars2, rmse_values):
        height = bar.get_height()  
        ax_mae.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rmse:.4f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    ax_mae.set_ylabel('Error (m)', fontsize=12, fontweight='bold')
    ax_mae.set_title('Overall MAE & RMSE Comparison', fontsize=12, fontweight='bold')
    ax_mae.set_xticks(x_pos)
    ax_mae.set_xticklabels([f'E{i}' for i in exp_ids])
    ax_mae.legend(fontsize=10, loc='upper left')
    ax_mae.grid(True, alpha=0.3, axis='y')
    
    # Final layout and save
    plt.tight_layout()
    output_file = Path(output_dir) / f'sample_{sample_idx:06d}_comparison_enhanced.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"  ✓ Enhanced comparison figure saved: {output_file.name}")
    plt.close()
    
    # Reset matplotlib parameters
    plt.rcdefaults()


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Inference and Visualization (Enhanced v4-style)')
    parser.add_argument('--data_dir', type=str, default='../swarm_segments', help='Data directory')
    parser.add_argument('--ablation_dir', type=str, default='.', help='Ablation results directory')
    parser.add_argument('--output_dir', type=str, default='ablation_viz_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples')
    parser.add_argument('--sample_indices', type=str, default=None, help='Specific sample indices (comma-separated)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_subset', action='store_true', help='Use subset data')
    parser.add_argument('--features_16d_dir', type=str, default='../features_16d', help='16D features directory')
    parser.add_argument('--features_32d_dir', type=str, default='../features_32d', help='32D features directory')
    
    # Physical constraints parameters (v4-style)
    parser.add_argument('--smoothing_weight', type=float, default=0.3, help='Acceleration smoothing weight (0-1)')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--exp2_constraint_relaxation', type=float, default=1, help='Relaxation factor (>1 loosens Exp2 constraints)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization figures')
    parser.add_argument('--save_metrics', action='store_true', help='Save detailed metrics')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("Ablation Study - Real-time Inference and Visualization")
    print("="*80 + "\n")
    
    # Log configuration
    print(f"Configuration:")
    print(f"  ├─ Smoothing weight: {args.smoothing_weight}")
    print(f"  ├─ Time step (dt): {args.dt}")
    print(f"  ├─ Device: {args.device}")
    print(f"  ├─ Batch size: {args.batch_size}")
    print(f"  └─ Output: {args.output_dir}\n")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load models
    print("Loading ablation study models...\n")
    ensemble = AblationModelEnsemble(device=args.device)
    
    experiments = [
        (1, 'exp1_baseline'),
        (2, 'exp2_feat_bigru'),
        (3, 'exp3_gnn_bigru'),
        (4, 'exp4_gnn_feat'),
        (5, 'exp5_full'),
    ]
    
    for exp_id, exp_name in experiments:
        ensemble.load_experiment(exp_id, exp_name, args.ablation_dir)
    
    # Load data
    print("Loading validation dataset...\n")
    data_info = load_ablation_data(
        args.data_dir, num_agents=3, feature_dim=32, batch_size=args.batch_size,
        val_split=0.2, num_workers=0, use_subset=args.use_subset, features_dir=args.features_32d_dir
    )
    
    val_loader = data_info['val_loader']
    val_dataset = data_info['val_dataset']
    
    print(f"✓ Data loaded: {len(val_dataset)} samples\n")
    
    # ✓ v4-style: Pre-load all features into memory to avoid repeated disk reads
    print("Pre-loading all features into memory...\n")
    features_32d_all = load_all_32d_features(args.features_32d_dir, num_agents=3, use_subset=args.use_subset)
    features_16d_all = load_features_16d(args.features_16d_dir, num_agents=3, use_subset=args.use_subset)
    
    # ✓ Validate 16D features alignment
    if features_16d_all is not None:
        if len(features_16d_all) != len(val_dataset):
            logger.warning(f"⚠️  16D features size ({len(features_16d_all)}) != val_dataset size ({len(val_dataset)})")
            logger.warning(f"   Detected full dataset features (training split)")
            
            # ✓ Solution: Extract validation subset from full dataset features
            val_size = len(val_dataset)
            logger.info(f"   Extracting validation subset: features[0:{val_size}]")
            features_16d_all = features_16d_all[:val_size]  # Take first val_size samples
            logger.info(f"✓ Successfully extracted 16D validation subset ({len(features_16d_all)} samples)")
        else:
            logger.info(f"✓ 16D features size matches validation set ({len(features_16d_all)} samples)\n")
    
    # Determine sample indices
    if args.sample_indices:
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(',')]
    else:
        total_samples = len(val_dataset)
        sample_indices = np.random.choice(total_samples, min(args.num_samples, total_samples), replace=False)
        sample_indices = sorted(sample_indices.tolist())
    
    print(f"Processing {len(sample_indices)} samples:\n")
    
    # Inference and visualization
    all_metrics = {exp_id: [] for exp_id in range(1, 6)}
    
    for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Inference+Visualization")):
        # Get sample
        sample = val_dataset[sample_idx]
        
        # Move to device
        x = sample['x'].unsqueeze(0).to(ensemble.device)
        x_orig = sample['x_orig'].unsqueeze(0).to(ensemble.device)  # (batch=1, seq_in=20, A=3, coords=3)
        y_delta = sample['y_delta'].unsqueeze(0).to(ensemble.device)  # (batch=1, seq_out=10, A=3, coords=3)
        features = sample['features'].unsqueeze(0).to(ensemble.device)  # (batch=1, seq_in=20, A=3, D=32)
        
        y_velocity = sample['y_velocity'].unsqueeze(0).to(ensemble.device)
        
        # Extract features for different experiments
        # Exp1, Exp3: use true 16D features if available, otherwise truncate from 32D
        # Exp2, Exp4, Exp5: use all 32D
        features_full = features  # (1, 20, 3, 32)
        
        # Use true 16D features if available
        if features_16d_all is not None:
            features_16d_sample = torch.from_numpy(features_16d_all[sample_idx]).unsqueeze(0).float().to(ensemble.device)
            features_16d = features_16d_sample  # (1, 20, 3, 16)
        else:
            # Fallback: truncate 32D features to 16D
            features_16d = features[:, :, :, :16]  # (1, 20, 3, 16)
        
        # Get last frame of input sequence (reference point for absolute position reconstruction)
        # x_orig shape: (batch=1, seq_in=20, agents=3, coords=3)
        last_pos = x_orig[:, -1, :, :].cpu().numpy()  # (1, A=3, 3) - last frame of input
        
        # Ground truth: y_delta is normalized delta (Y_t - X_last in normalized space)
        # y_delta shape: (batch=1, seq_out=10, A=3, coords=3)
        true_delta_norm = y_delta.squeeze(0).cpu().numpy()  # (seq_out=10, A=3, 3)
        
        # Use Exp1 stats as reference for denormalization (same for all experiments)
        output_mean = ensemble.stats[1]['output_mean']  # (3,)
        output_std = ensemble.stats[1]['output_std']    # (3,)
        
        # Denormalize ground truth delta: delta_physical = delta_norm * std + mean
        true_delta_phys = true_delta_norm * output_std[np.newaxis, np.newaxis, :] + output_mean[np.newaxis, np.newaxis, :]
        
        # Reconstruct absolute positions: Y_abs = X_last + delta_physical
        true_pos = last_pos[0, np.newaxis, :, :] + true_delta_phys  # (seq_out=10, A=3, 3)
        
        # Extract history trajectory for visualization
        history_traj = x_orig.squeeze(0).cpu().numpy()  # (seq_in=20, A=3, 3)
        
        # Inference
        sample_data = {
            'true_pos': true_pos,
            'history_traj': history_traj  # Add history trajectory
        }
        
        for exp_id in range(1, 6):
            # Select appropriate feature dimension
            if exp_id in [1, 3]:  # Exp1 and Exp3 use 16D
                feat = features_16d
            else:  # Exp2, Exp4, Exp5 use 32D
                feat = features_full
            
            # Model inference returns normalized delta
            pred_delta_norm, _, _ = ensemble.infer_batch(exp_id, feat, x_orig)
            # pred_delta_norm shape: (1, seq_out=10, A=3, 3)
            pred_delta_norm_np = pred_delta_norm.squeeze(0).cpu().numpy()  # (seq_out=10, A=3, 3)
            
            # Get statistics for this experiment
            out_mean = ensemble.stats[exp_id]['output_mean']  # (3,)
            out_std = ensemble.stats[exp_id]['output_std']    # (3,)
            
            # Denormalize prediction: delta_physical = delta_norm * std + mean
            pred_delta_phys = pred_delta_norm_np * out_std[np.newaxis, np.newaxis, :] + out_mean[np.newaxis, np.newaxis, :]
            
            # ✓ Apply experiment-specific physical constraints
            x_orig_np = x_orig.squeeze(0).cpu().numpy()  # (seq_in=20, A=3, 3)
            
            if exp_id == 2:  # Exp2: Dynamics-aware model - lighter constraints
                pred_pos = apply_physical_constraints(
                    history=x_orig_np,
                    pred_delta=pred_delta_phys,
                    dt=args.dt,
                    smoothing_weight=args.smoothing_weight,
                    constraint_relaxation=args.exp2_constraint_relaxation
                )
            else:  # Other experiments - standard constraints
                pred_pos = apply_physical_constraints(
                    history=x_orig_np,
                    pred_delta=pred_delta_phys,
                    dt=args.dt,
                    smoothing_weight=args.smoothing_weight
                )  # (seq_out=10, A=3, 3)
            
            sample_data[f'exp{exp_id}_pred'] = pred_pos
            
            # Compute metrics
            metrics = compute_metrics(true_pos, pred_pos)
            sample_data[f'metrics_exp{exp_id}'] = metrics
            all_metrics[exp_id].append(metrics)  # Store complete metrics dict instead of just MAE
        
        # Plot comparison
        plot_sample_comparison(sample_data, output_dir, sample_idx)
    
    # Summary statistics (v4-style comprehensive metrics)
    print("\n" + "="*80)
    print("Inference Complete - Comprehensive Statistics Summary")
    print("="*80 + "\n")
    
    # Experiment names for display
    exp_names = {
        1: "Exp1: Baseline GRU (16D)",
        2: "Exp2: Dynamics-Aware + BiCA (32D)",
        3: "Exp3: GNN + BiCA (16D)",
        4: "Exp4: GNN + Features (32D)",
        5: "Exp5: Full Model (32D)",
    }
    
    # Extract comprehensive statistics from all metrics
    def extract_aggregate_stats(metrics_list):
        """Extract key aggregate statistics for plotting (simplified version)"""
        if not metrics_list:
            return {}
        
        # Define key metrics for plotting
        key_metrics = ['MAE', 'RMSE', 'ADE', 'FDE', 'MAPE', 'MAE_X', 'MAE_Y', 'MAE_Z',
                      'Velocity_MAE', 'Velocity_RMSE', 'Speed_error_MAE']
        
        stats = {}
        
        # Aggregate key metrics (mean, std, min, max for error bars and box plots)
        for metric in key_metrics:
            values = [m[metric] for m in metrics_list if metric in m]
            if values:
                stats[f'{metric}_mean'] = float(np.mean(values))
                stats[f'{metric}_std'] = float(np.std(values))
                stats[f'{metric}_min'] = float(np.min(values))
                stats[f'{metric}_max'] = float(np.max(values))
        
        # Add per-step MAE for trend plots
        mae_per_step_all = [m['MAE_per_step'] for m in metrics_list if 'MAE_per_step' in m and m['MAE_per_step']]
        if mae_per_step_all:
            try:
                values_array = np.array(mae_per_step_all)
                stats['MAE_per_step_mean'] = values_array.mean(axis=0).tolist()
                stats['MAE_per_step_std'] = values_array.std(axis=0).tolist()
            except:
                pass  # Skip if inconsistent dimensions
        
        # Add per-agent MAE for multi-agent analysis
        mae_per_agent_all = [m['MAE_per_agent'] for m in metrics_list if 'MAE_per_agent' in m and m['MAE_per_agent']]
        if mae_per_agent_all:
            try:
                values_array = np.array(mae_per_agent_all)
                stats['MAE_per_agent_mean'] = values_array.mean(axis=0).tolist()
                stats['MAE_per_agent_std'] = values_array.std(axis=0).tolist()
            except:
                pass
        
        return stats
    
    # Prepare streamlined summary (focused on key plotting data)
    summary = {
        'num_samples': len(sample_indices),
        'configuration': {
            'smoothing_weight': args.smoothing_weight,
            'dt': args.dt,
            'use_subset': args.use_subset,
        },
        'experiments': {}
        # Removed detailed config and raw metrics to reduce file size
    }
    
    # Process each experiment
    for exp_id in range(1, 6):
        metrics_list = all_metrics[exp_id]
        aggregate_stats = extract_aggregate_stats(metrics_list)
        
        summary['experiments'][f'exp{exp_id}'] = {
            'experiment_id': exp_id,
            'name': exp_names[exp_id],
            'num_samples': len(metrics_list),
            'aggregate_stats': aggregate_stats
            # Removed 'all_metrics' to reduce file size - detailed data in separate file if needed
        }
    
    # Print comprehensive results
    print("📊 PERFORMANCE SUMMARY\n")
    
    # Overall performance comparison table
    print("="*90)
    print(f"{'Experiment':<30} {'MAE':<10} {'RMSE':<10} {'FDE':<10} {'MAPE':<10} {'Vel_MAE':<10}")
    print("="*90)
    
    for exp_id in range(1, 6):
        stats = summary['experiments'][f'exp{exp_id}']['aggregate_stats']
        name = exp_names[exp_id][:28]  # Truncate long names
        
        mae = stats.get('MAE_mean', 0)
        rmse = stats.get('RMSE_mean', 0)
        fde = stats.get('FDE_mean', 0)
        mape = stats.get('MAPE_mean', 0)
        vel_mae = stats.get('Velocity_MAE_mean', 0)
        
        print(f"{name:<30} {mae:<10.4f} {rmse:<10.4f} {fde:<10.4f} {mape:<10.4f} {vel_mae:<10.4f}")
    
    print("="*90)
    
    # Performance ranking
    experiments_ranked = sorted(range(1, 6), 
                               key=lambda x: summary['experiments'][f'exp{x}']['aggregate_stats'].get('MAE_mean', float('inf')))
    
    best_exp = experiments_ranked[0]
    best_mae = summary['experiments'][f'exp{best_exp}']['aggregate_stats'].get('MAE_mean', 0)
    
    print(f"\n🏆 Best model: {exp_names[best_exp]} (MAE: {best_mae:.4f}m)")
    
    # Show improvements over baseline (Exp1)
    if best_exp != 1:
        exp1_mae = summary['experiments']['exp1']['aggregate_stats'].get('MAE_mean', 0)
        if exp1_mae > 0:
            improvement = ((exp1_mae - best_mae) / exp1_mae) * 100
            print(f"💡 Improvement over baseline: {improvement:.1f}% better")
    
    print()
    
    # Save streamlined summary as JSON (optimized for plotting)
    summary_file = output_dir / 'ablation_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Summary saved: {summary_file}")
    
    # Save CSV format for easy analysis
    csv_file = output_dir / 'ablation_results.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['experiment_id', 'experiment_name', 'MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std', 
                     'ADE_mean', 'ADE_std', 'FDE_mean', 'FDE_std', 'MAPE_mean', 'MAPE_std',
                     'MAE_X_mean', 'MAE_Y_mean', 'MAE_Z_mean', 'Velocity_MAE_mean', 'Speed_error_MAE_mean',
                     'Min_mean', 'Median_mean', 'Max_mean', 'P95_mean']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for exp_id in range(1, 6):
            stats = summary['experiments'][f'exp{exp_id}']['aggregate_stats']
            row = {
                'experiment_id': exp_id,
                'experiment_name': exp_names[exp_id],
                'MAE_mean': stats.get('MAE_mean', 0),
                'MAE_std': stats.get('MAE_std', 0),
                'RMSE_mean': stats.get('RMSE_mean', 0),
                'RMSE_std': stats.get('RMSE_std', 0),
                'ADE_mean': stats.get('ADE_mean', 0),
                'ADE_std': stats.get('ADE_std', 0),
                'FDE_mean': stats.get('FDE_mean', 0),
                'FDE_std': stats.get('FDE_std', 0),
                'MAPE_mean': stats.get('MAPE_mean', 0),
                'MAPE_std': stats.get('MAPE_std', 0),
                'MAE_X_mean': stats.get('MAE_X_mean', 0),
                'MAE_Y_mean': stats.get('MAE_Y_mean', 0),
                'MAE_Z_mean': stats.get('MAE_Z_mean', 0),
                'Velocity_MAE_mean': stats.get('Velocity_MAE_mean', 0),
                'Speed_error_MAE_mean': stats.get('Speed_error_MAE_mean', 0),
                'Min_mean': stats.get('Min_mean', 0),
                'Median_mean': stats.get('Median_mean', 0),
                'Max_mean': stats.get('Max_mean', 0),
                'P95_mean': stats.get('P95_mean', 0),
            }
            writer.writerow(row)
    
    print(f"✅ CSV results saved: {csv_file}")
    
    # Save per-sample detailed results (v4-style)
    if args.save_metrics:
        detailed_file = output_dir / 'detailed_per_sample_metrics.json'
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({'detailed_metrics': all_metrics, 'sample_indices': sample_indices}, 
                     f, indent=2, ensure_ascii=False)
        print(f"✅ Detailed per-sample metrics saved: {detailed_file}")
    
    print(f"\n🎉 ABLATION STUDY COMPLETE! ({len(sample_indices)} samples)")
    print(f"📁 Results: {output_dir}")
    print(f"🏆 Best: {exp_names[experiments_ranked[0]]} (MAE: {best_mae:.4f}m)")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
