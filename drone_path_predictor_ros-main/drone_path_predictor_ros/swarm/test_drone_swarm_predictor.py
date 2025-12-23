#!/usr/bin/env python3
"""
Quick test of DroneSwarmPredictor - parallel prediction and collision detection
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from drone_path_predictor_ros.swarm import (
    SwarmDataLoader, DroneSwarmPredictor, predict_and_analyze
)

def test_swarm_predictor():
    """Test swarm prediction and collision detection."""
    
    workspace_root = Path(__file__).parent.parent.parent.parent
    traj_dir = workspace_root / "drone_trajectories" / "random_traj_100ms"
    
    print("\n" + "="*70)
    print("DRONE SWARM PREDICTOR TEST")
    print("="*70)
    
    if not traj_dir.exists():
        print(f"Error: Trajectory directory not found: {traj_dir}")
        return False
    
    try:
        # Load trajectories
        print("\n[1] Loading trajectories...")
        loader = SwarmDataLoader()
        swarm_data = loader.load_directory(str(traj_dir), pattern="*.txt", max_drones=5)
        print(f"OK - Loaded {swarm_data.num_drones} drones")
        
        # Setup model paths
        config_dir = workspace_root / "drone_path_predictor_ros-main" / "config" / "mixed_dataset"
        pos_model = str(config_dir / "mix_pos_max_norm_64_2_0p5.pth")
        vel_model = str(config_dir / "mix_vel_max_norm_64_2_0p5.pth")
        pos_stats = str(config_dir / "pos_stats.npz")
        vel_stats = str(config_dir / "vel_stats.npz")
        
        print(f"\n[2] Model paths:")
        print(f"   Position model: {Path(pos_model).name}")
        print(f"   Velocity model: {Path(vel_model).name}")
        print(f"   Position stats: {Path(pos_stats).name}")
        print(f"   Velocity stats: {Path(vel_stats).name}")
        
        # Create predictor
        print("\n[3] Creating swarm predictor...")
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
        print(f"OK - Predictor created (threshold={predictor.collision_threshold}m, "
              f"horizon={predictor.prediction_horizon})")
        
        # Single drone prediction
        print("\n[4] Testing single drone prediction...")
        first_drone = swarm_data.drone_ids[0]
        positions = swarm_data.get_positions(first_drone)
        pred_single = predictor.predict_single_drone(first_drone, positions)
        print(f"OK - Predicted {len(pred_single)} steps for drone '{first_drone}'")
        print(f"    Prediction shape: {pred_single.shape}")
        print(f"    Sample prediction:\n{pred_single[:3]}")
        
        # Swarm prediction (parallel)
        print("\n[5] Predicting entire swarm (parallel)...")
        predictions = predictor.predict_swarm(swarm_data, use_threading=True)
        print(f"OK - Predicted {len(predictions)} drones")
        for drone_id, pred in list(predictions.items())[:3]:
            print(f"    {drone_id}: {len(pred)} steps predicted")
        
        # Collision detection
        print("\n[6] Detecting collisions...")
        collisions = predictor.detect_collisions(predictions)
        print(f"OK - Found {len(collisions)} collision alerts")
        if collisions:
            for alert in collisions[:3]:
                print(f"    - {alert}")
        
        # Swarm metrics
        print("\n[7] Computing swarm metrics...")
        metrics = predictor.compute_swarm_metrics(predictions, time_index=-1)
        print(f"OK - Swarm metrics at final step:")
        print(f"    - Center of mass: {metrics.center_of_mass}")
        print(f"    - Compactness: {metrics.swarm_compactness:.2f}m")
        print(f"    - Diameter: {metrics.swarm_diameter:.2f}m")
        print(f"    - Min pairwise distance: {metrics.min_pairwise_distance:.2f}m")
        print(f"    - Formation stability: {metrics.formation_stability:.2f}")
        
        # Swarm trajectory (CoM)
        print("\n[8] Computing center of mass trajectory...")
        com_trajectory = predictor.get_swarm_trajectory(predictions)
        print(f"OK - CoM trajectory: {com_trajectory.shape}")
        print(f"    Starting CoM: {com_trajectory[0]}")
        print(f"    Final CoM: {com_trajectory[-1]}")
        
        # Evolution analysis
        print("\n[9] Analyzing swarm evolution...")
        evolution = predictor.analyze_swarm_evolution(predictions)
        print(f"OK - Evolution analysis complete")
        print(f"    Compactness range: {min(evolution['compactness_over_time']):.2f}m "
              f"to {max(evolution['compactness_over_time']):.2f}m")
        print(f"    Diameter range: {min(evolution['diameter_over_time']):.2f}m "
              f"to {max(evolution['diameter_over_time']):.2f}m")
        
        # Convenience function
        print("\n[10] Testing convenience function predict_and_analyze()...")
        preds, colls, evo = predict_and_analyze(swarm_data, collision_threshold=2.0,
                                                position_model_path=pos_model,
                                                velocity_model_path=vel_model,
                                                position_stats_file=pos_stats,
                                                velocity_stats_file=vel_stats)
        print(f"OK - Convenience function works")
        print(f"    Predictions: {len(preds)} drones")
        print(f"    Collisions: {len(colls)} alerts")
        
        # Summary
        print("\n[11] Generating summary...")
        summary = predictor.summary(predictions, collisions)
        print(summary)
        
        print("\n" + "="*70)
        print("SUCCESS - ALL TESTS PASSED")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_swarm_predictor()
    sys.exit(0 if success else 1)
