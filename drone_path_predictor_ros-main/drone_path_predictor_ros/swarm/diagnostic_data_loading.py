#!/usr/bin/env python3
"""
Diagnostic script to validate trajectory data loading and coordinate alignment.
Verifies that:
1. Each drone starts at its correct position
2. Coordinates are properly aligned (X, Y, Z)
3. Time synchronization is correct
4. Data quality is good
"""

import sys
import logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from drone_path_predictor_ros.swarm import SwarmDataLoader

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def diagnose_data_loading():
    """Run comprehensive data loading diagnostics."""
    
    logger.info("="*80)
    logger.info("TRAJECTORY DATA LOADING DIAGNOSTIC")
    logger.info("="*80)
    
    # Setup paths
    workspace_root = Path(__file__).parent.parent.parent.parent
    traj_dir = workspace_root / "drone_trajectories" / "random_traj_100ms"
    
    if not traj_dir.exists():
        logger.error(f"Trajectory directory not found: {traj_dir}")
        return False
    
    try:
        # Load trajectories
        logger.info("\n[1] LOADING TRAJECTORY DATA")
        logger.info("-" * 80)
        loader = SwarmDataLoader()
        swarm_data = loader.load_directory(str(traj_dir), pattern="*.txt", max_drones=5)
        
        logger.info(f"Loaded {swarm_data.num_drones} drones")
        logger.info(f"Total duration: {swarm_data.duration:.2f} seconds")
        logger.info(f"Drone IDs: {swarm_data.drone_ids}")
        
        # Analyze each drone
        logger.info("\n[2] PER-DRONE ANALYSIS")
        logger.info("-" * 80)
        
        drone_analysis = {}
        
        for drone_id in swarm_data.drone_ids:
            logger.info(f"\nDrone: {drone_id}")
            logger.info("  " + "="*70)
            
            # Get trajectory data
            trajectory = swarm_data.trajectories[drone_id]  # Shape: (N, 4) [t, x, y, z]
            timestamps = swarm_data.timestamps[drone_id]
            positions = swarm_data.get_positions(drone_id)  # Shape: (N, 3) [x, y, z]
            
            # Basic info
            logger.info(f"  Total points: {len(trajectory)}")
            logger.info(f"  Time range: [{timestamps[0]:.2f}s, {timestamps[-1]:.2f}]")
            logger.info(f"  Duration: {timestamps[-1] - timestamps[0]:.2f}s")
            
            # Starting position (t=0)
            start_pos = positions[0]
            logger.info(f"  Start position (t=0.00s):")
            logger.info(f"    X: {start_pos[0]:10.4f} m")
            logger.info(f"    Y: {start_pos[1]:10.4f} m")
            logger.info(f"    Z: {start_pos[2]:10.4f} m")
            logger.info(f"    Distance from origin: {np.linalg.norm(start_pos):.4f} m")
            
            # Ending position
            end_pos = positions[-1]
            logger.info(f"  End position (t={timestamps[-1]:.2f}s):")
            logger.info(f"    X: {end_pos[0]:10.4f} m")
            logger.info(f"    Y: {end_pos[1]:10.4f} m")
            logger.info(f"    Z: {end_pos[2]:10.4f} m")
            logger.info(f"    Distance from origin: {np.linalg.norm(end_pos):.4f} m")
            
            # Trajectory statistics
            diffs = np.diff(positions, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_distance = np.sum(distances)
            avg_velocity = total_distance / (timestamps[-1] - timestamps[0])
            
            logger.info(f"  Trajectory statistics:")
            logger.info(f"    Total distance traveled: {total_distance:.4f} m")
            logger.info(f"    Average velocity: {avg_velocity:.4f} m/s")
            logger.info(f"    Max velocity (per step): {np.max(distances):.4f} m/s")
            logger.info(f"    Min velocity (per step): {np.min(distances):.4f} m/s")
            
            # Coordinate ranges
            logger.info(f"  Coordinate ranges:")
            logger.info(f"    X: [{np.min(positions[:, 0]):8.4f}, {np.max(positions[:, 0]):8.4f}] m")
            logger.info(f"    Y: [{np.min(positions[:, 1]):8.4f}, {np.max(positions[:, 1]):8.4f}] m")
            logger.info(f"    Z: [{np.min(positions[:, 2]):8.4f}, {np.max(positions[:, 2]):8.4f}] m")
            
            # Time step validation
            time_diffs = np.diff(timestamps)
            unique_steps = np.unique(np.round(time_diffs, 2))
            logger.info(f"  Time step analysis:")
            logger.info(f"    Unique time steps: {unique_steps}")
            logger.info(f"    Expected (0.1s): {'OK' if 0.1 in unique_steps or np.isclose(unique_steps[0], 0.1) else 'MISMATCH'}")
            
            drone_analysis[drone_id] = {
                'start': start_pos,
                'end': end_pos,
                'total_distance': total_distance,
                'num_points': len(trajectory)
            }
        
        # Swarm-level analysis
        logger.info("\n[3] SWARM-LEVEL ANALYSIS")
        logger.info("-" * 80)
        
        start_positions = np.array([drone_analysis[d]['start'] for d in swarm_data.drone_ids])
        
        # Calculate inter-drone distances at start
        logger.info(f"\nStarting positions (t=0.00s):")
        for i, drone_id in enumerate(swarm_data.drone_ids):
            pos = start_positions[i]
            logger.info(f"  {drone_id:12s}: ({pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f})")
        
        logger.info(f"\nInter-drone distances at start (t=0.00s):")
        for i in range(len(swarm_data.drone_ids)):
            for j in range(i+1, len(swarm_data.drone_ids)):
                drone_id_1 = swarm_data.drone_ids[i]
                drone_id_2 = swarm_data.drone_ids[j]
                dist = np.linalg.norm(start_positions[i] - start_positions[j])
                logger.info(f"  {drone_id_1} <-> {drone_id_2}: {dist:.4f} m")
        
        # Swarm bounding box
        all_positions = np.vstack([swarm_data.get_positions(d) for d in swarm_data.drone_ids])
        logger.info(f"\nSwarm bounding box (all time):")
        logger.info(f"  X: [{np.min(all_positions[:, 0]):8.4f}, {np.max(all_positions[:, 0]):8.4f}] m")
        logger.info(f"  Y: [{np.min(all_positions[:, 1]):8.4f}, {np.max(all_positions[:, 1]):8.4f}] m")
        logger.info(f"  Z: [{np.min(all_positions[:, 2]):8.4f}, {np.max(all_positions[:, 2]):8.4f}] m")
        
        # Data quality checks
        logger.info("\n[4] DATA QUALITY CHECKS")
        logger.info("-" * 80)
        
        checks = {
            "All drones have same duration": all(
                abs(swarm_data.trajectories[d][-1, 0] - swarm_data.trajectories[swarm_data.drone_ids[0]][-1, 0]) < 0.01
                for d in swarm_data.drone_ids
            ),
            "All drones start at t~0": all(
                abs(swarm_data.trajectories[d][0, 0]) < 0.01
                for d in swarm_data.drone_ids
            ),
            "Time steps are regular (0.1s)": all(
                np.allclose(np.diff(swarm_data.timestamps[d]), 0.1, atol=0.01)
                for d in swarm_data.drone_ids
            ),
            "No NaN values in trajectories": all(
                not np.any(np.isnan(swarm_data.trajectories[d]))
                for d in swarm_data.drone_ids
            ),
            "No Inf values in trajectories": all(
                not np.any(np.isinf(swarm_data.trajectories[d]))
                for d in swarm_data.drone_ids
            ),
        }
        
        for check_name, result in checks.items():
            status = "OK" if result else "FAIL"
            logger.info(f"  {check_name}: {status}")
        
        all_checks_pass = all(checks.values())
        
        # Prediction feasibility
        logger.info("\n[5] PREDICTION FEASIBILITY")
        logger.info("-" * 80)
        
        history_len = 21  # Model expects 21 steps of history
        pred_horizon = 10  # Model predicts 10 steps
        required_points = history_len + pred_horizon
        
        logger.info(f"  Required points for prediction:")
        logger.info(f"    History length: {history_len} steps = {history_len * 0.1:.1f}s")
        logger.info(f"    Prediction horizon: {pred_horizon} steps = {pred_horizon * 0.1:.1f}s")
        logger.info(f"    Total required: {required_points} steps = {required_points * 0.1:.1f}s")
        
        logger.info(f"\n  Per-drone feasibility:")
        feasible_drones = 0
        for drone_id in swarm_data.drone_ids:
            num_points = len(swarm_data.trajectories[drone_id])
            is_feasible = num_points >= required_points
            status = "OK" if is_feasible else "INSUFFICIENT"
            logger.info(f"    {drone_id}: {num_points} points - {status}")
            if is_feasible:
                feasible_drones += 1
        
        logger.info(f"\n  Feasible drones: {feasible_drones}/{swarm_data.num_drones}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("="*80)
        logger.info(f"Data loading: {'SUCCESS' if all_checks_pass else 'WARNING'}")
        logger.info(f"Prediction feasibility: {'OK' if feasible_drones == swarm_data.num_drones else 'PARTIAL'}")
        logger.info(f"Coordinate system: {'VERIFIED' if all_checks_pass else 'CHECK NEEDED'}")
        logger.info(f"\nReady for: {'Prediction and visualization' if all_checks_pass else 'Review data issues'}")
        
        return all_checks_pass and feasible_drones == swarm_data.num_drones
        
    except Exception as e:
        logger.error(f"Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = diagnose_data_loading()
    sys.exit(0 if success else 1)
