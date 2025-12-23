#!/usr/bin/env python3
"""
Compare predicted vs actual trajectories with accuracy metrics.

This script:
    1. Loads trajectory data
    2. Runs swarm predictions
    3. Compares predictions with actual future trajectories
    4. Calculates error metrics (MAE, RMSE, etc.)
    5. Visualizes comparison plots
"""

import sys
import logging
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

# Setup matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from drone_path_predictor_ros.swarm import (
    SwarmDataLoader, DroneSwarmPredictor
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for trajectory prediction."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    max_error: float  # Maximum error
    mean_error: float  # Mean error (signed)
    trajectory_length: int  # Number of predicted steps


def calculate_accuracy_metrics(predicted: np.ndarray, actual: np.ndarray) -> AccuracyMetrics:
    """
    Calculate accuracy metrics between predicted and actual trajectories.
    
    Args:
        predicted: (N, 3) array of predicted positions
        actual: (N, 3) array of actual positions
        
    Returns:
        AccuracyMetrics object
    """
    pred_array = np.asarray(predicted)
    actual_array = np.asarray(actual)
    
    # Ensure same length
    min_len = min(len(pred_array), len(actual_array))
    pred_array = pred_array[:min_len]
    actual_array = actual_array[:min_len]
    
    # Calculate point-wise distances
    distances = np.linalg.norm(pred_array - actual_array, axis=1)
    
    # Calculate metrics
    mae = float(np.mean(distances))
    rmse = float(np.sqrt(np.mean(distances ** 2)))
    max_error = float(np.max(distances))
    mean_error = float(np.mean(pred_array - actual_array))
    
    return AccuracyMetrics(
        mae=mae,
        rmse=rmse,
        max_error=max_error,
        mean_error=mean_error,
        trajectory_length=min_len
    )


def plot_trajectory_comparison_3d(predicted: np.ndarray, actual: np.ndarray, 
                                  drone_id: str, title: str = "",
                                  save_path: Path = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot predicted vs actual trajectory in 3D.
    
    Args:
        predicted: (N, 3) predicted trajectory
        actual: (N, 3) actual trajectory
        drone_id: Drone identifier
        title: Plot title
        save_path: Optional save path
        
    Returns:
        Figure and axes
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    pred_array = np.asarray(predicted)
    actual_array = np.asarray(actual)
    min_len = min(len(pred_array), len(actual_array))
    
    # Plot actual trajectory
    ax.plot(actual_array[:min_len, 0], actual_array[:min_len, 1], actual_array[:min_len, 2],
           'g-o', linewidth=2.5, markersize=5, label='Actual', alpha=0.8)
    
    # Plot predicted trajectory
    ax.plot(pred_array[:min_len, 0], pred_array[:min_len, 1], pred_array[:min_len, 2],
           'r--s', linewidth=2.5, markersize=5, label='Predicted', alpha=0.8)
    
    # Mark start point
    ax.scatter(*actual_array[0], color='green', s=150, marker='o', 
              edgecolors='black', linewidth=2, label='Start (Actual)', zorder=5)
    ax.scatter(*pred_array[0], color='red', s=150, marker='s',
              edgecolors='black', linewidth=2, label='Start (Predicted)', zorder=5)
    
    # Mark end point
    ax.scatter(*actual_array[min_len-1], color='green', s=150, marker='^',
              edgecolors='black', linewidth=2, label='End (Actual)', zorder=5)
    ax.scatter(*pred_array[min_len-1], color='red', s=150, marker='v',
              edgecolors='black', linewidth=2, label='End (Predicted)', zorder=5)
    
    # Add connection lines between predicted and actual at each step (error visualization)
    for i in range(0, min_len, max(1, min_len // 10)):  # Show 10 error lines
        ax.plot([actual_array[i, 0], pred_array[i, 0]],
               [actual_array[i, 1], pred_array[i, 1]],
               [actual_array[i, 2], pred_array[i, 2]],
               'k:', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    else:
        ax.set_title(f"Trajectory Comparison - {drone_id}", fontsize=13, fontweight='bold')
    
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    return fig, ax


def plot_error_over_time(predicted: np.ndarray, actual: np.ndarray,
                         drone_id: str, title: str = "",
                         save_path: Path = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot prediction error over time.
    
    Args:
        predicted: (N, 3) predicted trajectory
        actual: (N, 3) actual trajectory
        drone_id: Drone identifier
        title: Plot title
        save_path: Optional save path
        
    Returns:
        Figure and axes array
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    pred_array = np.asarray(predicted)
    actual_array = np.asarray(actual)
    min_len = min(len(pred_array), len(actual_array))
    
    time_steps = np.arange(min_len)
    
    # Point-wise errors
    errors = np.linalg.norm(pred_array[:min_len] - actual_array[:min_len], axis=1)
    
    # Individual coordinate errors
    error_x = np.abs(pred_array[:min_len, 0] - actual_array[:min_len, 0])
    error_y = np.abs(pred_array[:min_len, 1] - actual_array[:min_len, 1])
    error_z = np.abs(pred_array[:min_len, 2] - actual_array[:min_len, 2])
    
    # Plot 1: Total error over time
    axes[0, 0].plot(time_steps, errors, 'r-o', linewidth=2, markersize=6)
    axes[0, 0].fill_between(time_steps, 0, errors, alpha=0.3, color='red')
    axes[0, 0].set_xlabel('Time Step', fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('Distance Error (m)', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('Total Position Error Over Time', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Coordinate-wise errors
    axes[0, 1].plot(time_steps, error_x, 'b-o', label='X error', linewidth=2, markersize=5)
    axes[0, 1].plot(time_steps, error_y, 'g-s', label='Y error', linewidth=2, markersize=5)
    axes[0, 1].plot(time_steps, error_z, 'r-^', label='Z error', linewidth=2, markersize=5)
    axes[0, 1].set_xlabel('Time Step', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('Coordinate Error (m)', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('Coordinate-wise Errors', fontsize=11, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Actual trajectory coordinates
    axes[1, 0].plot(time_steps, actual_array[:min_len, 0], 'b-o', label='X', linewidth=2, markersize=5)
    axes[1, 0].plot(time_steps, actual_array[:min_len, 1], 'g-s', label='Y', linewidth=2, markersize=5)
    axes[1, 0].plot(time_steps, actual_array[:min_len, 2], 'r-^', label='Z', linewidth=2, markersize=5)
    axes[1, 0].set_xlabel('Time Step', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Position (m)', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('Actual Trajectory Coordinates', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Predicted trajectory coordinates
    axes[1, 1].plot(time_steps, pred_array[:min_len, 0], 'b--o', label='X', linewidth=2, markersize=5)
    axes[1, 1].plot(time_steps, pred_array[:min_len, 1], 'g--s', label='Y', linewidth=2, markersize=5)
    axes[1, 1].plot(time_steps, pred_array[:min_len, 2], 'r--^', label='Z', linewidth=2, markersize=5)
    axes[1, 1].set_xlabel('Time Step', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Position (m)', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('Predicted Trajectory Coordinates', fontsize=11, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f"Error Analysis - {drone_id if not title else title}", 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved error plot to {save_path}")
    
    return fig, axes


def plot_accuracy_metrics_comparison(metrics_dict: Dict[str, AccuracyMetrics],
                                     title: str = "Accuracy Metrics Comparison",
                                     save_path: Path = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Compare accuracy metrics across all drones.
    
    Args:
        metrics_dict: Dict[drone_id] -> AccuracyMetrics
        title: Plot title
        save_path: Optional save path
        
    Returns:
        Figure and axes array
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    drone_ids = list(metrics_dict.keys())
    mae_values = [metrics_dict[d].mae for d in drone_ids]
    rmse_values = [metrics_dict[d].rmse for d in drone_ids]
    max_errors = [metrics_dict[d].max_error for d in drone_ids]
    
    x_pos = np.arange(len(drone_ids))
    
    # MAE comparison
    bars1 = axes[0].bar(x_pos, mae_values, color='skyblue', edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('MAE (m)', fontsize=11, fontweight='bold')
    axes[0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(drone_ids, rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # RMSE comparison
    bars2 = axes[1].bar(x_pos, rmse_values, color='lightcoral', edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('RMSE (m)', fontsize=11, fontweight='bold')
    axes[1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(drone_ids, rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Max error comparison
    bars3 = axes[2].bar(x_pos, max_errors, color='lightgreen', edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Max Error (m)', fontsize=11, fontweight='bold')
    axes[2].set_title('Maximum Error', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(drone_ids, rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved metrics comparison to {save_path}")
    
    return fig, axes


def main():
    """Main function."""
    
    logger.info("="*70)
    logger.info("PREDICTION ACCURACY ANALYSIS")
    logger.info("="*70)
    
    # Setup paths
    workspace_root = Path(__file__).parent.parent.parent.parent
    traj_dir = workspace_root / "drone_trajectories" / "random_traj_100ms"
    config_dir = workspace_root / "drone_path_predictor_ros-main" / "config" / "mixed_dataset"
    output_dir = workspace_root / "drone_path_predictor_ros-main" / "accuracy_analysis"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not traj_dir.exists():
        logger.error(f"Trajectory directory not found: {traj_dir}")
        return False
    
    try:
        # Load trajectories
        logger.info("\n[1] Loading trajectories...")
        loader = SwarmDataLoader()
        swarm_data = loader.load_directory(str(traj_dir), pattern="*.txt", max_drones=5)
        logger.info(f"    Loaded {swarm_data.num_drones} drones")
        
        # Setup model paths
        pos_model = str(config_dir / "mix_pos_max_norm_64_2_0p5.pth")
        vel_model = str(config_dir / "mix_vel_max_norm_64_2_0p5.pth")
        pos_stats = str(config_dir / "pos_stats.npz")
        vel_stats = str(config_dir / "vel_stats.npz")
        
        # Create predictor
        logger.info("\n[2] Creating predictor...")
        predictor = DroneSwarmPredictor(
            position_model_path=pos_model,
            velocity_model_path=vel_model,
            position_stats_file=pos_stats,
            velocity_stats_file=vel_stats,
            pos_hidden_dim=64,
            pos_num_layers=2,
            pos_dropout=0.5,
            vel_hidden_dim=64,
            vel_num_layers=2,
            vel_dropout=0.5,
            collision_threshold=2.0,
            prediction_horizon=10
        )
        logger.info(f"    Predictor created")
        
        # Run predictions
        logger.info("\n[3] Running predictions...")
        predictions = predictor.predict_swarm(swarm_data, use_threading=True)
        logger.info(f"    Predicted {len(predictions)} drones")
        
        # Calculate accuracy metrics
        logger.info("\n[4] Calculating accuracy metrics...")
        metrics_dict = {}
        
        for drone_id in swarm_data.drone_ids:
            # Get actual trajectory positions (without time column)
            actual_full = swarm_data.trajectories[drone_id][:, 1:4]  # Shape: (N, 3) [x, y, z]
            
            # Model input length is 21 steps
            history_len = 21
            
            # Get actual future trajectory for comparison
            # Use 10 steps (same as prediction_horizon)
            if len(actual_full) >= history_len + 10:
                actual_future = actual_full[history_len:history_len+10]  # Next 10 steps after history
            else:
                # If trajectory is shorter, use what's available
                actual_future = actual_full[history_len:]
                if len(actual_future) < 1:
                    logger.warning(f"  {drone_id}: Trajectory too short ({len(actual_full)} points), skipping")
                    continue
            
            # Get prediction
            predicted = predictions[drone_id]
            
            # Ensure both have the same length for comparison
            min_len = min(len(predicted), len(actual_future))
            if min_len < 1:
                logger.warning(f"  {drone_id}: No valid prediction/actual data")
                continue
            
            # Truncate to same length
            predicted_trunc = predicted[:min_len]
            actual_future_trunc = actual_future[:min_len]
            
            # Calculate metrics
            metrics = calculate_accuracy_metrics(predicted_trunc, actual_future_trunc)
            metrics_dict[drone_id] = metrics
            
            logger.info(f"    {drone_id}:")
            logger.info(f"      - Points compared: {min_len}")
            logger.info(f"      - MAE: {metrics.mae:.4f}m")
            logger.info(f"      - RMSE: {metrics.rmse:.4f}m")
            logger.info(f"      - Max Error: {metrics.max_error:.4f}m")
        
        # Generate comparison plots
        logger.info("\n[5] Generating comparison visualizations...")
        
        for i, drone_id in enumerate(swarm_data.drone_ids):
            if drone_id not in metrics_dict:
                logger.warning(f"    Skipping {drone_id} (no valid metrics)")
                continue
            
            actual_full = swarm_data.trajectories[drone_id][:, 1:4]
            history_len = 21
            
            # Ensure we have enough data
            if len(actual_full) >= history_len + 10:
                actual_future = actual_full[history_len:history_len+10]
            else:
                actual_future = actual_full[history_len:]
            
            predicted = predictions[drone_id]
            
            # Align lengths
            min_len = min(len(predicted), len(actual_future))
            if min_len < 1:
                continue
            
            predicted = predicted[:min_len]
            actual_future = actual_future[:min_len]
            
            # 3D comparison plot
            fig1, ax1 = plot_trajectory_comparison_3d(
                predicted, actual_future, drone_id,
                save_path=output_dir / f"{i+1}_3d_comparison_{drone_id}.png"
            )
            plt.close(fig1)
            
            # Error over time plot
            fig2, axes2 = plot_error_over_time(
                predicted, actual_future, drone_id,
                save_path=output_dir / f"{i+1}_error_analysis_{drone_id}.png"
            )
            plt.close(fig2)
        
        # Overall metrics comparison
        fig3, axes3 = plot_accuracy_metrics_comparison(
            metrics_dict,
            save_path=output_dir / "00_metrics_comparison.png"
        )
        plt.close(fig3)
        
        logger.info(f"    All visualizations saved to {output_dir}")
        
        # Print summary statistics
        logger.info("\n" + "="*70)
        logger.info("ACCURACY SUMMARY")
        logger.info("="*70)
        
        if not metrics_dict:
            logger.error("No metrics calculated!")
            return False
        
        all_mae = [m.mae for m in metrics_dict.values()]
        all_rmse = [m.rmse for m in metrics_dict.values()]
        all_max = [m.max_error for m in metrics_dict.values()]
        
        logger.info(f"\nPrediction Configuration:")
        logger.info(f"  History length: 21 steps (2.1s)")
        logger.info(f"  Prediction horizon: 10 steps (1.0s)")
        logger.info(f"  Time step: 0.1s")
        
        logger.info(f"\nGlobal Statistics:")
        logger.info(f"  Drones analyzed: {len(metrics_dict)}")
        logger.info(f"  Mean MAE across all drones: {np.mean(all_mae):.4f}m")
        logger.info(f"  Mean RMSE across all drones: {np.mean(all_rmse):.4f}m")
        logger.info(f"  Mean Max Error across all drones: {np.mean(all_max):.4f}m")
        
        logger.info(f"\n  Best performing drone (lowest MAE):")
        best_drone = min(metrics_dict.items(), key=lambda x: x[1].mae)
        logger.info(f"    {best_drone[0]}: MAE={best_drone[1].mae:.4f}m, RMSE={best_drone[1].rmse:.4f}m")
        
        logger.info(f"\n  Worst performing drone (highest MAE):")
        worst_drone = max(metrics_dict.items(), key=lambda x: x[1].mae)
        logger.info(f"    {worst_drone[0]}: MAE={worst_drone[1].mae:.4f}m, RMSE={worst_drone[1].rmse:.4f}m")
        
        logger.info(f"\nDetailed Results per Drone:")
        for drone_id in sorted(metrics_dict.keys()):
            m = metrics_dict[drone_id]
            logger.info(f"  {drone_id:15s}: MAE={m.mae:7.4f}m, RMSE={m.rmse:7.4f}m, Max={m.max_error:7.4f}m")
        
        logger.info(f"\nOutput directory: {output_dir}")
        logger.info("\nShowing plots (close windows to continue)...")
        plt.show()
        
        logger.info("="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
