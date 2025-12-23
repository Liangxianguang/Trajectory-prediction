#!/usr/bin/env python3
"""
Interactive Trajectory Prediction Visualizer
=============================================

A professional visualization system for drone swarm trajectory prediction.
Shows:
- 3D trajectories with history (green) and predictions (red)
- Per-drone error metrics
- Swarm-level statistics
- Interactive viewing angle controls
"""

import sys
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Setup matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from drone_path_predictor_ros.swarm import (
    SwarmDataLoader, DroneSwarmPredictor
)

logging.basicConfig(level=logging.WARNING)  # Suppress excess logging for cleaner UI
logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Interactive 3D trajectory prediction visualizer."""
    
    def __init__(self, swarm_data, predictions, predictor=None, history_len=21):
        """
        Initialize visualizer.
        
        Args:
            swarm_data: SwarmTrajectoryData object
            predictions: Dict[drone_id] -> predicted positions (N, 3)
            predictor: DroneSwarmPredictor instance (REQUIRED for proper history resampling)
            history_len: Number of historical steps to show (for reference only)
            
        Note: We now use predictor's _prepare_history_sequence() to ensure consistency
              between the history used for prediction and the history displayed.
              This is critical - if the visualizer uses a different history, predictions won't match!
        """
        self.swarm_data = swarm_data
        self.history_len = history_len
        self.predictor = predictor
        
        # Prepare per-drone data
        self.drone_ids = swarm_data.drone_ids
        self.histories = {}
        self.pred_starts = {}
        self.actual_positions = {}
        self.predictions = {}

        # First pass: Extract actual positions and prepare history using predictor's method
        for drone_id in self.drone_ids:
            actual = np.asarray(swarm_data.get_positions(drone_id))
            self.actual_positions[drone_id] = actual

            # CRITICAL FIX: Use predictor's history preparation method for consistency
            if predictor is not None:
                # Get timestamps for interpolation
                timestamps = np.asarray(swarm_data.get_timestamps(drone_id))
                # Use the same resampling method as the predictor
                history = predictor._prepare_history_sequence(actual, timestamps)
                logger.info(f"Drone {drone_id}: Using predictor's resampled history (shape={history.shape})")
            else:
                # Fallback: Select history from the end of trajectory
                logger.warning(f"Drone {drone_id}: No predictor provided, using fallback history selection")
                if len(actual) >= self.history_len:
                    history = actual[-self.history_len:]
                else:
                    history = actual.copy()
            
            self.histories[drone_id] = history

            # Set prediction start point (last point of selected history)
            if len(history) > 0:
                self.pred_starts[drone_id] = history[-1].copy()
            elif len(actual) > 0:
                self.pred_starts[drone_id] = actual[-1].copy()
            else:
                self.pred_starts[drone_id] = np.zeros((3,), dtype=np.float32)

        # Second pass: Align predictions to history endpoints
        for drone_id in self.drone_ids:
            raw_prediction = np.asarray(predictions.get(drone_id, np.zeros((0, 3))))
            self.predictions[drone_id] = self._align_prediction_to_history(drone_id, raw_prediction)
        
        # Colors for drones
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.drone_ids)))
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Drone Swarm Trajectory Prediction Visualizer', 
                         fontsize=16, fontweight='bold', y=0.98)
        
        # Main 3D plot
        self.ax_3d = self.fig.add_subplot(2, 2, 1, projection='3d')
        
        # Metrics plots
        self.ax_errors = self.fig.add_subplot(2, 2, 2)
        self.ax_metrics = self.fig.add_subplot(2, 2, 3)
        self.ax_info = self.fig.add_subplot(2, 2, 4)
        self.ax_info.axis('off')
        # Initialize plots
        self.plot_lines = {}
        self.pred_lines = {}
        self.compare_fig = None
        self.compare_axes = []
        self.detail_fig = None
        self.detail_axes = []
        self.draw_initial_plots()
        self.create_comparison_axes()
        self.draw_individual_comparisons()
        self.create_drone_detail_axes()
        self.draw_per_drone_3d()

        # Setup interactivity
        self.setup_controls()
    
    def draw_initial_plots(self):
        """Draw initial trajectory plots."""
        
        # Plot trajectories
        for i, drone_id in enumerate(self.drone_ids):
            color = self.colors[i]
            
            # History (green solid line)
            if len(self.histories[drone_id]) > 0:
                hist = self.histories[drone_id]
                line, = self.ax_3d.plot(hist[:, 0], hist[:, 1], hist[:, 2],
                                       'o-', color=color, linewidth=2.5, 
                                       markersize=4, label=f'{drone_id} (history)',
                                       alpha=0.7)
                self.plot_lines[drone_id] = line
            
            # Prediction (red dashed line)
            if drone_id in self.predictions:
                pred = self.predictions[drone_id]
                if len(pred) > 0:
                    line, = self.ax_3d.plot(pred[:, 0], pred[:, 1], pred[:, 2],
                                           '--s', color=color, linewidth=2.5,
                                           markersize=4, label=f'{drone_id} (pred)',
                                           alpha=0.7)
                    self.pred_lines[drone_id] = line
            
            # Start point marker (large square)
            start = self.histories[drone_id][-1] if len(self.histories[drone_id]) > 0 else self.pred_starts[drone_id]
            self.ax_3d.scatter(*start, s=200, marker='s', color=color,
                             edgecolors='black', linewidth=2, zorder=5)
            
            # End point marker (large triangle)
            if drone_id in self.predictions and len(self.predictions[drone_id]) > 0:
                end = self.predictions[drone_id][-1]
                self.ax_3d.scatter(*end, s=200, marker='^', color=color,
                                 edgecolors='black', linewidth=2, zorder=5)
        
        # Setup 3D plot
        self.ax_3d.set_xlabel('X (m)', fontweight='bold')
        self.ax_3d.set_ylabel('Y (m)', fontweight='bold')
        self.ax_3d.set_zlabel('Z (m)', fontweight='bold')
        self.ax_3d.set_title('3D Trajectory Prediction', fontweight='bold', pad=10)
        self.ax_3d.legend(loc='upper left', fontsize=8, ncol=2)
        self.ax_3d.grid(True, alpha=0.3)
        self.ax_3d.view_init(elev=20, azim=45)
        
        # Plot error metrics
        self.draw_error_metrics()
        
        # Plot swarm metrics
        self.draw_swarm_metrics()
        
        # Plot info text
        self.draw_info_text()
    
    def draw_error_metrics(self):
        """Draw per-drone error metrics."""
        self.ax_errors.clear()
        
        errors = []
        labels = []
        
        for drone_id in self.drone_ids:
            if drone_id in self.predictions:
                pred = self.predictions[drone_id]
                actual_future = self._get_actual_future(drone_id, len(pred))

                min_len = min(len(actual_future), len(pred))
                if min_len > 0:
                    distances = np.linalg.norm(pred[:min_len] - actual_future[:min_len], axis=1)
                    mae = np.mean(distances)
                    errors.append(mae)
                    labels.append(drone_id)
        
        if errors:
            colors_bar = [self.colors[self.drone_ids.index(label)] for label in labels]
            bars = self.ax_errors.bar(range(len(errors)), errors, color=colors_bar, 
                                     edgecolor='black', linewidth=1.5, alpha=0.7)
            
            # Add value labels on bars
            for bar, err in zip(bars, errors):
                height = bar.get_height()
                self.ax_errors.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{err:.2f}m', ha='center', va='bottom', fontsize=9)
            
            self.ax_errors.set_xticks(range(len(labels)))
            self.ax_errors.set_xticklabels(labels, rotation=45)
            self.ax_errors.set_ylabel('Mean Absolute Error (m)', fontweight='bold')
            self.ax_errors.set_title('Per-Drone Prediction Error', fontweight='bold')
            self.ax_errors.grid(True, alpha=0.3, axis='y')
    
    def draw_swarm_metrics(self):
        """Draw swarm-level metrics."""
        self.ax_metrics.clear()
        
        # Get final positions
        final_positions = []
        for drone_id in self.drone_ids:
            if drone_id in self.predictions and len(self.predictions[drone_id]) > 0:
                final_positions.append(self.predictions[drone_id][-1])
        
        if final_positions:
            final_positions = np.array(final_positions)
            
            # Calculate metrics
            com = np.mean(final_positions, axis=0)
            compactness = np.mean(np.linalg.norm(final_positions - com, axis=1))
            
            pairwise_dists = []
            for i in range(len(final_positions)):
                for j in range(i+1, len(final_positions)):
                    dist = np.linalg.norm(final_positions[i] - final_positions[j])
                    pairwise_dists.append(dist)
            
            diameter = max(pairwise_dists) if pairwise_dists else 0
            min_dist = min(pairwise_dists) if pairwise_dists else 0
            
            metrics_text = f"""
SWARM METRICS (Final Prediction Step)
{'='*40}

Formation Statistics:
  • Center of Mass: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}) m
  • Swarm Compactness: {compactness:.2f} m
  • Swarm Diameter: {diameter:.2f} m
  • Min Inter-drone Distance: {min_dist:.2f} m

Drone Count: {len(self.drone_ids)}
Prediction Horizon: 1.0 s (10 steps)
"""
            self.ax_metrics.text(0.05, 0.95, metrics_text, transform=self.ax_metrics.transAxes,
                               fontfamily='monospace', fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Swarm Metrics', fontweight='bold', loc='left')
    
    def draw_info_text(self):
        """Draw information text."""
        info_text = f"""
VISUALIZATION INFORMATION
{'='*40}

Data Configuration:
    • Drones: {len(self.drone_ids)}
    • History: {self.history_len} steps = 2.1 s
    • Prediction window: Next 10 steps = 1.0 s
    • Prediction start: Last observed (history) position
    • Time step: 0.1 s

Color Legend:
    • Solid line (○): Historical trajectory
    • Dashed line (□): Predicted trajectory
    • □ marker: Prediction start (last history point)
    • △ marker: Ending position

Comparison Panels:
    • XY subplots compare the historical trajectory, your prediction, and the true future path
    • 3D subplots show each drone with history/prediction/ground truth for the next 1s

Controls:
  • Rotate: Left mouse drag in 3D plot
  • Zoom: Scroll wheel in 3D plot
  • Elevation: Use right mouse drag

File locations:
  • History: From loaded trajectories
  • Predictions: From GRU model output
"""
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontfamily='monospace', fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        self.ax_info.set_title('Information Panel', fontweight='bold', loc='left')

    def create_comparison_axes(self):
        """Prepare a separate figure with per-drone comparison subplots."""
        if not self.drone_ids:
            return

        # Create one subplot per drone to compare XY trajectories
        fig, axes_grid = plt.subplots(nrows=len(self.drone_ids), figsize=(10, 3 * len(self.drone_ids)), squeeze=False)
        axes = [ax for row in axes_grid for ax in row]
        self.compare_fig = fig
        self.compare_axes = axes

        for ax, drone_id in zip(self.compare_axes, self.drone_ids):
            ax.set_title(f'{drone_id} XY Trajectory Comparison')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True, alpha=0.3)

    def draw_individual_comparisons(self):
        """Plot history, predicted, and true future XY paths per drone."""
        if not self.compare_axes:
            return

        for ax, drone_id in zip(self.compare_axes, self.drone_ids):
            ax.clear()
            history = np.asarray(self.histories.get(drone_id, np.zeros((0, 3))))
            pred = self.predictions.get(drone_id, np.zeros((0, 3)))
            actual_positions = self.actual_positions.get(drone_id, np.zeros((0, 3)))

            # Plot history
            if history.size:
                ax.plot(history[:, 0], history[:, 1], 'o-', color='tab:green', label='History', alpha=0.8)

            # Plot predicted future
            if pred.size:
                ax.plot(pred[:, 0], pred[:, 1], '--', color='tab:blue', label='Predicted (next 1s)')

            # Plot true future
            actual_future = self._get_actual_future(drone_id, len(pred))
            if actual_future.size:
                ax.plot(actual_future[:, 0], actual_future[:, 1], ':', color='tab:red', label='Ground truth (next 1s)')

            # Debug prints to help verify alignment between predicted and true future trajectories
            try:
                print('\n--- Drone:', drone_id, '---')
                print('History last point (prediction start):', history[-1] if history.size else 'N/A')
                print('Pred start stored:', self.pred_starts.get(drone_id))
                print('Predicted points (len={}):'.format(len(pred)))
                if pred.size:
                    np.set_printoptions(precision=3, suppress=True)
                    print(pred)
                else:
                    print('  <no prediction>')

                print('Actual future (len={}):'.format(len(actual_future)))
                if actual_future.size:
                    print(actual_future)
                else:
                    print('  <no actual future data>')
            except Exception as _e:
                print('Debug print failed for', drone_id, _e)

            ax.set_title(f'{drone_id} XY Trajectory Comparison')
            ax.legend(fontsize=8)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True, alpha=0.3)

    def _align_prediction_to_history(self, drone_id: str, prediction: np.ndarray) -> np.ndarray:
        """
        Ensure the first predicted point matches the selected history end.
        
        This method translates the entire prediction trajectory so that its first point
        aligns exactly with the last point of the displayed history. This ensures
        visual continuity between history and prediction, and makes error metrics meaningful.
        
        Args:
            drone_id: Drone identifier
            prediction: Raw prediction array (N, 3)
            
        Returns:
            Aligned prediction array (N, 3) with prediction[0] == history[-1]
        """
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.size == 0:
            logger.debug(f"Drone '{drone_id}': Empty prediction, returning as-is")
            return prediction

        start_point = self.pred_starts.get(drone_id)
        if start_point is None:
            logger.warning(f"Drone '{drone_id}': No prediction start point found")
            return prediction
        
        start_point = np.asarray(start_point, dtype=np.float32)
        if start_point.size == 0:
            logger.warning(f"Drone '{drone_id}': Empty start point")
            return prediction

        # Calculate shift needed to align first prediction point with history end
        shift = start_point - prediction[0]
        shift_magnitude = np.linalg.norm(shift)
        
        # Only apply shift if it's significant (> 1 micron)
        if shift_magnitude > 1e-6:
            prediction = prediction + shift
            logger.debug(f"Drone '{drone_id}': Aligned prediction with shift magnitude {shift_magnitude:.6f}m")
        else:
            logger.debug(f"Drone '{drone_id}': Prediction already aligned (shift={shift_magnitude:.9f}m)")
        
        return prediction

    def _get_actual_future(self, drone_id, pred_len):
        """
        Return the actual future trajectory aligned with prediction.
        
        Both prediction and actual trajectory must start from the SAME time point
        to be comparable. The prediction starts at the last point of interpolated history.
        
        Strategy:
        1. Find where prediction starts (last history point)
        2. Find the closest matching point in original trajectory
        3. Return points STARTING FROM THAT MATCH POINT, including that point itself
        
        This ensures: prediction[0] aligns with actual_future[0] in space and time
        
        Args:
            drone_id: Drone identifier
            pred_len: Number of future steps to return
            
        Returns:
            Actual trajectory segment (pred_len points) aligned with prediction start
        """
        if pred_len <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        actual = np.asarray(self.actual_positions.get(drone_id, np.zeros((0, 3))))
        history = np.asarray(self.histories.get(drone_id, np.zeros((0, 3))))
        
        if actual.size == 0 or history.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # The anchor point is where prediction starts (last point of interpolated history)
        pred_start_point = history[-1]
        
        # Find the closest point in the actual trajectory
        distances = np.linalg.norm(actual - pred_start_point, axis=1)
        start_idx = np.argmin(distances)
        match_distance = distances[start_idx]
        
        # Debug output
        logger.debug(f"Drone {drone_id}: pred_start={pred_start_point}, "
                    f"closest_actual={actual[start_idx]}, distance={match_distance:.6f}m")
        
        if match_distance > 1.0:
            logger.warning(f"Large alignment error for {drone_id}: {match_distance:.3f}m")
        
        # Return pred_len points STARTING FROM start_idx (inclusive)
        # The first point will be the match point itself
        # Subsequent points will be the true future
        end_idx = min(start_idx + pred_len, len(actual))
        
        if start_idx >= len(actual):
            return np.zeros((0, 3), dtype=np.float32)

        result = actual[start_idx:end_idx].astype(np.float32)
        logger.debug(f"Returning {len(result)} actual future points for {drone_id}")
        return result

    def create_drone_detail_axes(self):
        """Create one 3D axis per drone for detailed comparison."""
        if not self.drone_ids:
            return

        cols = 2
        rows = math.ceil(len(self.drone_ids) / cols)
        fig = plt.figure(figsize=(12, 4 * rows))
        axes = []
        for idx in range(len(self.drone_ids)):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            axes.append(ax)
        self.detail_fig = fig
        self.detail_axes = axes

    def draw_per_drone_3d(self):
        """Render history, predicted, and truth trajectories on dedicated 3D axes."""
        if not self.detail_axes:
            return

        for ax, drone_id in zip(self.detail_axes, self.drone_ids):
            ax.clear()
            history = np.asarray(self.histories.get(drone_id, np.zeros((0, 3))))
            pred = self.predictions.get(drone_id, np.zeros((0, 3)))
            actual = self.actual_positions.get(drone_id, np.zeros((0, 3)))
            actual_future = self._get_actual_future(drone_id, len(pred))

            if actual.size:
                ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color='lightgray', alpha=0.4, label='Full actual trajectory')
            if history.size:
                ax.plot(history[:, 0], history[:, 1], history[:, 2], 'o-', color='tab:green', label='History', alpha=0.8)
            if pred.size:
                start_marker = history[-1] if history.size else (actual[0] if actual.size else np.zeros(3))
                ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], '--', color='tab:blue', label='Prediction (next 1s)')
                ax.scatter(*start_marker, s=70, marker='s', color='tab:blue', edgecolors='black', label='Prediction start')
            if actual_future.size:
                ax.plot(actual_future[:, 0], actual_future[:, 1], actual_future[:, 2], ':', color='tab:red', label='Ground truth (next 1s)')

            ax.set_title(f'{drone_id} 3D Comparison')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    
    def setup_controls(self):
        """Setup interactive controls."""
        
        # Slider for viewing angle
        ax_slider = plt.axes([0.2, 0.05, 0.3, 0.03])
        self.slider_azim = Slider(ax_slider, 'Azimuth', 0, 360, valinit=45, valstep=5)
        self.slider_azim.on_changed(self.update_view)
        
        # Reset button
        ax_reset = plt.axes([0.6, 0.05, 0.1, 0.04])
        btn_reset = Button(ax_reset, 'Reset View')
        btn_reset.on_clicked(self.reset_view)
        
        plt.subplots_adjust(bottom=0.15)
    
    def update_view(self, val):
        """Update 3D view angle."""
        azim = self.slider_azim.val
        self.ax_3d.view_init(elev=20, azim=azim)
        self.fig.canvas.draw_idle()
    
    def reset_view(self, event):
        """Reset view to default."""
        self.slider_azim.set_val(45)
        self.ax_3d.view_init(elev=20, azim=45)
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the visualizer."""
        self.fig.tight_layout()
        if self.compare_fig:
            self.compare_fig.tight_layout()
        if self.detail_fig:
            self.detail_fig.tight_layout()
        plt.show()


def main():
    """Main function."""
    
    print("\n" + "="*80)
    print("INTERACTIVE TRAJECTORY PREDICTION VISUALIZER")
    print("="*80)
    
    # Setup paths
    workspace_root = Path(__file__).parent.parent.parent.parent
    traj_dir = workspace_root / "drone_trajectories" / "random_traj_100ms"
    # Use 5000_gz_dataset which contains the trained models with 256/5 architecture
    config_dir = workspace_root / "drone_path_predictor_ros-main" / "config" / "5000_gz_dataset"
    
    if not traj_dir.exists():
        print(f"Error: Trajectory directory not found: {traj_dir}")
        return False
    
    try:
        # Load trajectories
        print("\n[1] Loading trajectory data...")
        loader = SwarmDataLoader()
        swarm_data = loader.load_directory(str(traj_dir), pattern="*.txt", max_drones=5)
        print(f"    Loaded {swarm_data.num_drones} drones, {swarm_data.duration:.2f}s duration")
        
        # Setup model paths
        # Using 5000_gz_dataset models with correct architecture: 256 hidden, 5 layers, whitening
        pos_model = str(config_dir / "5000_pos_white_norm_256_5_0p5.pth")
        vel_model = str(config_dir / "5000_vel_white_norm_256_5_0p5.pth")
        pos_stats = str(config_dir / "pos_stats.npz")
        vel_stats = str(config_dir / "vel_stats.npz")
        
        # Create predictor
        print("\n[2] Creating predictor...")
        # CRITICAL: Match the configuration file parameters exactly!
        # From trajectory_predictor.yaml:
        #   - pos_hidden_dim: 256, pos_num_layers: 5
        #   - vel_hidden_dim: 256, vel_num_layers: 5
        #   - use_whitening: true (models are white_norm)
        #   - pos_input_length: 20, pos_output_length: 10
        predictor = DroneSwarmPredictor(
            position_model_path=pos_model,
            velocity_model_path=vel_model,
            position_stats_file=pos_stats,
            velocity_stats_file=vel_stats,
            pos_hidden_dim=256,      # ✅ Changed from 64 to 256 (match 5000_gz_dataset)
            pos_num_layers=5,        # ✅ Changed from 2 to 5 (match 5000_gz_dataset)
            pos_dropout=0.5,
            vel_hidden_dim=256,      # ✅ Changed from 64 to 256 (match 5000_gz_dataset)
            vel_num_layers=5,        # ✅ Changed from 2 to 5 (match 5000_gz_dataset)
            vel_dropout=0.5,
            use_whitening=True,      # ✅ Changed from False to True (models use white_norm)
            collision_threshold=2.0,
            prediction_horizon=10,
            dt=0.1,
            history_window=20,       # ✅ Matches pos_input_length from config
            prediction_mode='position'  # Options: 'position', 'velocity', 'blend'
        )
        print(f"    Predictor ready (256 hidden, 5 layers, whitening enabled)")
        
        # Run predictions
        print("\n[3] Predicting trajectories...")
        predictions = predictor.predict_swarm(swarm_data, use_threading=True)
        print(f"    Predicted {len(predictions)} drones")
        
        # Create visualizer with predictor for proper history consistency
        print("\n[4] Initializing visualizer...")
        # CRITICAL: Pass the predictor so visualizer uses the same history resampling method
        visualizer = TrajectoryVisualizer(swarm_data, predictions, predictor=predictor, history_len=21)
        print(f"    Visualizer ready")
        
        # Show
        print("\n[5] Displaying interactive visualization...")
        print("    Tip: Use mouse to rotate, scroll to zoom, R to reset view")
        print("    Close the window to exit\n")
        visualizer.show()
        
        print("\n" + "="*80)
        print("VISUALIZATION CLOSED")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
