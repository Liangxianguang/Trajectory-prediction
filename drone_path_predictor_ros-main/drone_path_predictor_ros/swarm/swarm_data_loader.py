#!/usr/bin/env python3
"""
Swarm Data Loader - Load multiple UAV trajectory files

Loads drone trajectory data from CSV/TXT files and provides structured 
access to multi-drone swarm trajectories.

Author: Trajectory Prediction Team
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SwarmTrajectoryData:
    """Container for swarm trajectory data."""
    drone_ids: List[str]
    trajectories: Dict[str, np.ndarray]  # drone_id -> trajectory array (N, 4) [t, x, y, z]
    timestamps: Dict[str, np.ndarray]    # drone_id -> timestamps
    duration: float                       # Total duration in seconds
    num_drones: int                       # Number of drones
    
    def get_positions(self, drone_id: str) -> np.ndarray:
        """Get positions (without time) for a drone."""
        return self.trajectories[drone_id][:, 1:4]
    
    def get_timestamps(self, drone_id: str) -> np.ndarray:
        """Get timestamps for a drone."""
        return self.timestamps[drone_id]
    
    def __repr__(self):
        return f"SwarmTrajectoryData(drones={self.num_drones}, duration={self.duration:.2f}s)"


class SwarmDataLoader:
    """
    Load and manage swarm trajectory data from files.
    
    Supports:
    - Single and multiple trajectory files
    - CSV and TXT formats
    - Automatic synchronization of trajectories
    - Data validation and filtering
    
    Example:
        >>> loader = SwarmDataLoader()
        >>> data = loader.load_directory('path/to/trajectories')
        >>> print(data.num_drones)
        >>> positions = data.get_positions('drone_0')
    """
    
    def __init__(self):
        """Initialize the loader."""
        logger.info("SwarmDataLoader initialized")
    
    def load_file(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Load a single trajectory file.
        
        Expected format: timestamp tx ty tz (CSV or TXT with space/comma delimiter)
        
        Args:
            file_path: Path to trajectory file
            
        Returns:
            Tuple of (trajectory_data, drone_id)
            trajectory_data shape: (N, 4) [timestamp, x, y, z]
            drone_id: extracted from filename or generic
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")
        
        try:
            # Read file with flexible parsing (handles headers and spacing)
            data = None
            parse_attempts = [
                {'delimiter': ',', 'header': 0, 'skipinitialspace': True},
                {'delimiter': r'\s+', 'header': 0, 'engine': 'python'},
                {'delimiter': r'\s+', 'header': None, 'engine': 'python'},
            ]

            for opts in parse_attempts:
                try:
                    data = pd.read_csv(file_path, comment='#', **opts)
                    if not data.empty:
                        break
                except Exception:
                    data = None
                    continue

            if data is None or data.empty:
                raise ValueError("Could not parse file with any delimiter")

            if data.shape[1] < 4:
                raise ValueError(f"Expected at least 4 columns (t, x, y, z), got {data.shape[1]}")

            # Extract columns and convert to float
            trajectory = data.iloc[:, :4].astype(float).values
            
            # Extract drone ID from filename
            drone_id = file_path.stem
            
            logger.info(f"Loaded trajectory from {file_path.name}: {len(trajectory)} points, drone_id={drone_id}")
            
            return trajectory, drone_id
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def load_directory(self, directory_path: str, pattern: str = "*.txt",
                      max_drones: Optional[int] = None) -> SwarmTrajectoryData:
        """
        Load all trajectory files from a directory.
        
        Args:
            directory_path: Path to directory containing trajectory files
            pattern: File pattern to match (e.g., "*.txt", "*.csv")
            max_drones: Maximum number of drones to load (None = load all)
            
        Returns:
            SwarmTrajectoryData object with all trajectories
            
        Raises:
            FileNotFoundError: If directory does not exist
            ValueError: If no trajectory files found
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find trajectory files
        trajectory_files = list(directory.glob(pattern))
        
        if not trajectory_files:
            raise ValueError(f"No trajectory files found matching pattern '{pattern}' in {directory_path}")
        
        if max_drones:
            trajectory_files = trajectory_files[:max_drones]
        
        logger.info(f"Found {len(trajectory_files)} trajectory files in {directory_path}")
        
        # Load all trajectories
        trajectories = {}
        timestamps = {}
        drone_ids = []
        
        for file_path in sorted(trajectory_files):
            try:
                trajectory, drone_id = self.load_file(file_path)
                trajectories[drone_id] = trajectory
                timestamps[drone_id] = trajectory[:, 0]  # First column is timestamp
                drone_ids.append(drone_id)
            except Exception as e:
                logger.warning(f"Skipped file {file_path.name}: {e}")
                continue
        
        if not trajectories:
            raise ValueError("No valid trajectory files could be loaded")
        
        # Synchronize trajectories to common time range
        self._synchronize_trajectories(trajectories, timestamps)
        
        # Calculate total duration
        min_time = min(ts[0] for ts in timestamps.values())
        max_time = max(ts[-1] for ts in timestamps.values())
        duration = max_time - min_time
        
        swarm_data = SwarmTrajectoryData(
            drone_ids=drone_ids,
            trajectories=trajectories,
            timestamps=timestamps,
            duration=duration,
            num_drones=len(drone_ids)
        )
        
        logger.info(f"Loaded swarm data: {swarm_data}")
        return swarm_data
    
    def _synchronize_trajectories(self, trajectories: Dict[str, np.ndarray],
                                 timestamps: Dict[str, np.ndarray]) -> None:
        """
        Synchronize trajectories to a common time range (in-place).
        
        Removes data outside the common time range for all drones.
        
        Args:
            trajectories: Dictionary of drone trajectories
            timestamps: Dictionary of drone timestamps
        """
        # Find common time range
        all_start_times = [ts[0] for ts in timestamps.values()]
        all_end_times = [ts[-1] for ts in timestamps.values()]
        
        common_start = max(all_start_times)
        common_end = min(all_end_times)
        
        if common_start >= common_end:
            logger.warning("No overlapping time range found in trajectories!")
            return
        
        # Filter trajectories to common range
        for drone_id in trajectories.keys():
            traj = trajectories[drone_id]
            ts = timestamps[drone_id]
            
            mask = (ts >= common_start) & (ts <= common_end)
            trajectories[drone_id] = traj[mask]
            timestamps[drone_id] = ts[mask]
        
        logger.info(f"Synchronized trajectories to common time range: "
                   f"[{common_start:.2f}, {common_end:.2f}]")
    
    def resample_trajectory(self, trajectory: np.ndarray, 
                           old_timestamps: np.ndarray,
                           new_timestamps: np.ndarray) -> np.ndarray:
        """
        Resample a trajectory to new timestamps using linear interpolation.
        
        Args:
            trajectory: Original trajectory (N, 3) [x, y, z]
            old_timestamps: Original timestamps (N,)
            new_timestamps: Target timestamps (M,)
            
        Returns:
            Resampled trajectory (M, 3)
        """
        from scipy.interpolate import interp1d
        
        try:
            # Interpolate each dimension
            f_x = interp1d(old_timestamps, trajectory[:, 0], kind='linear', 
                          fill_value='extrapolate')
            f_y = interp1d(old_timestamps, trajectory[:, 1], kind='linear',
                          fill_value='extrapolate')
            f_z = interp1d(old_timestamps, trajectory[:, 2], kind='linear',
                          fill_value='extrapolate')
            
            new_trajectory = np.column_stack([
                f_x(new_timestamps),
                f_y(new_timestamps),
                f_z(new_timestamps)
            ])
            
            return new_trajectory
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            raise
    
    def compute_swarm_statistics(self, swarm_data: SwarmTrajectoryData) -> Dict:
        """
        Compute basic statistics for the swarm.
        
        Args:
            swarm_data: Swarm trajectory data
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_drones': swarm_data.num_drones,
            'duration': swarm_data.duration,
            'avg_trajectory_length': 0.0,
            'min_trajectory_length': float('inf'),
            'max_trajectory_length': 0.0,
            'avg_velocity': {},
            'swarm_spread': {},  # Per drone
        }
        
        all_lengths = []
        
        for drone_id in swarm_data.drone_ids:
            traj = swarm_data.get_positions(drone_id)
            ts = swarm_data.get_timestamps(drone_id)
            
            # Trajectory length (total distance)
            diffs = np.diff(traj, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_length = np.sum(distances)
            all_lengths.append(total_length)
            
            stats['min_trajectory_length'] = min(stats['min_trajectory_length'], total_length)
            stats['max_trajectory_length'] = max(stats['max_trajectory_length'], total_length)
            
            # Average velocity
            avg_vel = total_length / swarm_data.duration if swarm_data.duration > 0 else 0
            stats['avg_velocity'][drone_id] = avg_vel
            
            # Swarm spread (distance from center)
            center = np.mean(traj, axis=0)
            distances_from_center = np.linalg.norm(traj - center, axis=1)
            stats['swarm_spread'][drone_id] = {
                'mean': np.mean(distances_from_center),
                'max': np.max(distances_from_center),
                'min': np.min(distances_from_center),
            }
        
        stats['avg_trajectory_length'] = np.mean(all_lengths)
        
        return stats
    
    def filter_trajectories(self, swarm_data: SwarmTrajectoryData,
                          min_length: float = 0.0,
                          max_length: Optional[float] = None) -> SwarmTrajectoryData:
        """
        Filter out drones with trajectories outside specified range.
        
        Args:
            swarm_data: Original swarm data
            min_length: Minimum trajectory length
            max_length: Maximum trajectory length (None = no limit)
            
        Returns:
            Filtered SwarmTrajectoryData
        """
        filtered_ids = []
        filtered_trajs = {}
        filtered_ts = {}
        
        for drone_id in swarm_data.drone_ids:
            traj = swarm_data.get_positions(drone_id)
            diffs = np.diff(traj, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_length = np.sum(distances)
            
            if total_length >= min_length:
                if max_length is None or total_length <= max_length:
                    filtered_ids.append(drone_id)
                    filtered_trajs[drone_id] = swarm_data.trajectories[drone_id]
                    filtered_ts[drone_id] = swarm_data.timestamps[drone_id]
        
        logger.info(f"Filtered trajectories: {len(swarm_data.drone_ids)} -> {len(filtered_ids)} drones")
        
        return SwarmTrajectoryData(
            drone_ids=filtered_ids,
            trajectories=filtered_trajs,
            timestamps=filtered_ts,
            duration=swarm_data.duration,
            num_drones=len(filtered_ids)
        )


# ============================================================================
# Convenience functions
# ============================================================================

def load_swarm_trajectories(directory_path: str, **kwargs) -> SwarmTrajectoryData:
    """
    Convenience function to load swarm trajectories.
    
    Args:
        directory_path: Path to directory with trajectory files
        **kwargs: Additional arguments passed to loader
        
    Returns:
        SwarmTrajectoryData
    """
    loader = SwarmDataLoader()
    return loader.load_directory(directory_path, **kwargs)
