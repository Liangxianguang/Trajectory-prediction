#!/usr/bin/env python3
"""
Streaming Swarm Predictor - Continuous trajectory prediction with sliding window

Implements sliding window buffer management for continuous multi-drone prediction:
- Maintains observation history (buffer) for each drone
- Makes predictions using current buffer state
- Updates buffer with new observations
- Supports rolling prediction across entire trajectory

This enables realistic prediction scenarios where:
  1. You observe first N steps
  2. Predict next M steps
  3. Receive actual step N+1
  4. Add N+1 to buffer, remove oldest
  5. Predict next M steps from new buffer state
  6. Repeat for entire trajectory duration

Author: Trajectory Prediction Team
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class StreamingPredictionResult:
    """Result of a single streaming prediction step."""
    timestamp: float                    # Current observation timestamp
    buffer_positions: Dict[str, np.ndarray]  # Current buffer for each drone (N, 3)
    predictions: Dict[str, np.ndarray]       # Predictions (horizon, 3)
    actual_positions: Optional[Dict[str, np.ndarray]] = None  # Next actual positions (for validation)
    prediction_error: Optional[Dict[str, float]] = None       # RMSE per drone


class StreamingDroneBuffer:
    """Manages observation history for a single drone."""
    
    def __init__(self, drone_id: str, buffer_size: int = 10):
        """
        Initialize buffer for a drone.
        
        Args:
            drone_id: Unique drone identifier
            buffer_size: Maximum buffer size (positions kept in history)
        """
        self.drone_id = drone_id
        self.buffer_size = buffer_size
        self.position_buffer = deque(maxlen=buffer_size)  # Keeps last N positions
        self.timestamp_buffer = deque(maxlen=buffer_size)  # Corresponding timestamps
        
    def add_observation(self, position: np.ndarray, timestamp: float) -> None:
        """
        Add new observation to buffer.
        
        Args:
            position: Position (3,) [x, y, z]
            timestamp: Observation timestamp
        """
        if position.shape != (3,):
            raise ValueError(f"Expected position shape (3,), got {position.shape}")
        self.position_buffer.append(position.copy())
        self.timestamp_buffer.append(timestamp)
    
    def get_buffer(self) -> np.ndarray:
        """Get current buffer as (N, 3) array."""
        if not self.position_buffer:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(list(self.position_buffer), dtype=np.float32)
    
    def get_timestamps(self) -> np.ndarray:
        """Get current timestamps."""
        return np.array(list(self.timestamp_buffer), dtype=np.float32)
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.position_buffer) == self.buffer_size
    
    def is_ready(self) -> bool:
        """Check if buffer has at least some observations."""
        return len(self.position_buffer) > 0
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.position_buffer)
    
    def reset(self) -> None:
        """Clear buffer."""
        self.position_buffer.clear()
        self.timestamp_buffer.clear()
    
    def __repr__(self):
        return f"StreamingDroneBuffer({self.drone_id}, size={self.size()}/{self.buffer_size})"


class StreamingSwarmPredictor:
    """
    Continuous multi-drone trajectory prediction with sliding window.
    
    Workflow:
        1. Initialize with trajectory data for all drones
        2. Fill buffers with first N observations
        3. For each future timestep:
           a. Make predictions using current buffers
           b. Get actual next observation (from data or sensor)
           c. Update buffers (add observation, drop oldest)
           d. Store predictions and observations
        4. Analyze prediction errors and trajectory quality
    
    Example:
        >>> from drone_path_predictor_ros.swarm import SwarmDataLoader, StreamingSwarmPredictor
        >>> loader = SwarmDataLoader()
        >>> swarm_data = loader.load_directory('path/to/trajectories')
        >>> 
        >>> streaming = StreamingSwarmPredictor(
        ...     swarm_data=swarm_data,
        ...     buffer_size=10,
        ...     prediction_horizon=10,
        ...     config_file='path/to/config.yaml'
        ... )
        >>> streaming.warmup(num_initial_steps=10)  # Fill buffers
        >>> 
        >>> # Predict and update iteratively
        >>> for i in range(streaming.max_steps):
        ...     result = streaming.predict_and_update(step=i)
        ...     print(f"Step {result.timestamp}: Predicted {len(result.predictions)} drones")
        >>> 
        >>> # Get full results
        >>> trajectory_predictions = streaming.get_all_predictions()
        >>> errors = streaming.compute_prediction_errors()
    """
    
    def __init__(self,
                 swarm_data,
                 buffer_size: int = 10,
                 prediction_horizon: int = 10,
                 config_file: Optional[str] = None):
        """
        Initialize streaming predictor.
        
        Args:
            swarm_data: SwarmTrajectoryData object from SwarmDataLoader
            buffer_size: Number of past observations to maintain
            prediction_horizon: Number of steps to predict into future
            config_file: Path to predictor config (or None to use default)
        """
        self.swarm_data = swarm_data
        self.buffer_size = buffer_size
        self.prediction_horizon = prediction_horizon
        self.config_file = config_file
        
        # Initialize buffers for each drone
        self.buffers: Dict[str, StreamingDroneBuffer] = {
            drone_id: StreamingDroneBuffer(drone_id, buffer_size)
            for drone_id in swarm_data.drone_ids
        }
        
        # Storage for predictions and tracking
        self.prediction_history: List[StreamingPredictionResult] = []
        self.current_step: int = 0
        
        # Lazy load the predictor
        self._predictor = None
        self._predictor_initialized = False
        
        logger.info(f"Initialized StreamingSwarmPredictor: "
                   f"buffers={buffer_size}, horizon={prediction_horizon}, "
                   f"drones={len(swarm_data.drone_ids)}")
    
    @property
    def predictor(self):
        """Lazy load predictor on first access."""
        if not self._predictor_initialized:
            from drone_path_predictor_ros.swarm import DroneSwarmPredictor
            self._predictor = DroneSwarmPredictor(
                config_file=self.config_file,
                prediction_horizon=self.prediction_horizon
            )
            self._predictor_initialized = True
        return self._predictor
    
    @property
    def max_steps(self) -> int:
        """Maximum number of steps available in data."""
        if not self.swarm_data.trajectories:
            return 0
        return max(len(traj) for traj in self.swarm_data.trajectories.values())
    
    def warmup(self, num_initial_steps: int) -> None:
        """
        Fill buffers with initial observations.
        
        Args:
            num_initial_steps: Number of initial steps to load into buffers
        """
        if num_initial_steps > self.max_steps:
            logger.warning(f"Requested {num_initial_steps} steps but only {self.max_steps} available")
            num_initial_steps = self.max_steps
        
        for step in range(num_initial_steps):
            for drone_id in self.swarm_data.drone_ids:
                trajectory = self.swarm_data.trajectories[drone_id]
                position = trajectory[step, 1:4]  # Extract x, y, z
                timestamp = trajectory[step, 0]    # Extract time
                self.buffers[drone_id].add_observation(position, timestamp)
        
        logger.info(f"Warmed up buffers with {num_initial_steps} initial steps")
    
    def predict_and_update(self, step: int, 
                          actual_next_position: Optional[Dict[str, np.ndarray]] = None) -> StreamingPredictionResult:
        """
        Make prediction using current buffers, optionally update with actual next observation.
        
        Args:
            step: Current step index
            actual_next_position: Optional actual next positions for validation
                                (drone_id -> position (3,))
        
        Returns:
            StreamingPredictionResult with predictions and tracking info
        """
        self.current_step = step
        
        # Extract current buffer state
        current_buffers = {
            drone_id: self.buffers[drone_id].get_buffer()
            for drone_id in self.swarm_data.drone_ids
        }
        current_timestamps = {
            drone_id: self.buffers[drone_id].get_timestamps()
            for drone_id in self.swarm_data.drone_ids
        }
        
        # Get current timestamp (latest in buffer)
        current_timestamp = max(ts[-1] if len(ts) > 0 else 0.0 
                               for ts in current_timestamps.values())
        
        # Make predictions
        predictions = {}
        for drone_id in self.swarm_data.drone_ids:
            buf = current_buffers[drone_id]
            if len(buf) == 0:
                predictions[drone_id] = np.zeros((self.prediction_horizon, 3))
                continue
            
            try:
                # Use the existing predictor which handles rebasing correctly
                pred = self.predictor.predict_single_drone(
                    drone_id,
                    positions=buf,
                    last_position=buf[-1]  # Pass last position for rebasing
                )
                predictions[drone_id] = pred
            except Exception as e:
                logger.error(f"Prediction failed for {drone_id}: {e}")
                predictions[drone_id] = np.zeros((self.prediction_horizon, 3))
        
        # Compute prediction errors if actual positions provided
        prediction_errors = None
        if actual_next_position:
            prediction_errors = {}
            for drone_id in self.swarm_data.drone_ids:
                if drone_id in actual_next_position:
                    # Compare first predicted step with actual next position
                    if len(predictions[drone_id]) > 0:
                        predicted_next = predictions[drone_id][0]
                        actual_next = actual_next_position[drone_id]
                        error = np.linalg.norm(predicted_next - actual_next)
                        prediction_errors[drone_id] = float(error)
        
        # Create result
        result = StreamingPredictionResult(
            timestamp=current_timestamp,
            buffer_positions=current_buffers,
            predictions=predictions,
            actual_positions=actual_next_position,
            prediction_error=prediction_errors
        )
        
        self.prediction_history.append(result)
        logger.debug(f"Step {step}: Made predictions (buffers: {len(current_buffers)} drones)")
        
        # Update buffers with next observation if provided
        if actual_next_position:
            for drone_id in self.swarm_data.drone_ids:
                if drone_id in actual_next_position:
                    trajectory = self.swarm_data.trajectories[drone_id]
                    next_idx = self.buffer_size + step + 1
                    if next_idx < len(trajectory):
                        next_obs = trajectory[next_idx]
                        position = next_obs[1:4]
                        timestamp = next_obs[0]
                        self.buffers[drone_id].add_observation(position, timestamp)
        
        return result
    
    def run_streaming_prediction(self, 
                                num_steps: Optional[int] = None,
                                validate: bool = True) -> List[StreamingPredictionResult]:
        """
        Run full streaming prediction on available data.
        
        Args:
            num_steps: Number of steps to predict (None = all available)
            validate: If True, compare predictions with actual future positions
        
        Returns:
            List of StreamingPredictionResult for each step
        """
        # Initialize buffers
        self.warmup(self.buffer_size)
        
        # Determine how many steps we can predict
        if num_steps is None:
            num_steps = max(0, self.max_steps - self.buffer_size - self.prediction_horizon)
        else:
            num_steps = min(num_steps, max(0, self.max_steps - self.buffer_size - self.prediction_horizon))
        
        logger.info(f"Running streaming prediction for {num_steps} steps "
                   f"(buffer={self.buffer_size}, horizon={self.prediction_horizon})")
        
        results = []
        for step in range(num_steps):
            # Get actual next observation for validation
            actual_next = None
            if validate:
                actual_next = {}
                for drone_id in self.swarm_data.drone_ids:
                    trajectory = self.swarm_data.trajectories[drone_id]
                    next_idx = self.buffer_size + step
                    if next_idx < len(trajectory):
                        actual_next[drone_id] = trajectory[next_idx, 1:4]
            
            # Predict and update
            result = self.predict_and_update(step, actual_next_position=actual_next)
            results.append(result)
            
            # Periodic logging
            if (step + 1) % max(1, num_steps // 10) == 0:
                logger.info(f"Progress: {step + 1}/{num_steps} steps completed")
        
        return results
    
    def get_all_predictions(self) -> Dict[str, List[np.ndarray]]:
        """
        Get all predictions across all steps.
        
        Returns:
            Dict mapping drone_id -> list of prediction arrays
        """
        all_preds = {drone_id: [] for drone_id in self.swarm_data.drone_ids}
        for result in self.prediction_history:
            for drone_id, pred in result.predictions.items():
                all_preds[drone_id].append(pred.copy())
        return all_preds
    
    def compute_prediction_errors(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Compute prediction error statistics.
        
        Returns:
            Dict mapping drone_id -> (mean_error, std_error, max_error)
        """
        errors_per_drone = {drone_id: [] for drone_id in self.swarm_data.drone_ids}
        
        for result in self.prediction_history:
            if result.prediction_error:
                for drone_id, error in result.prediction_error.items():
                    errors_per_drone[drone_id].append(error)
        
        stats = {}
        for drone_id, errors in errors_per_drone.items():
            if errors:
                stats[drone_id] = (
                    float(np.mean(errors)),
                    float(np.std(errors)),
                    float(np.max(errors))
                )
            else:
                stats[drone_id] = (0.0, 0.0, 0.0)
        
        return stats
    
    def reset(self) -> None:
        """Reset all buffers and history."""
        for buffer in self.buffers.values():
            buffer.reset()
        self.prediction_history.clear()
        self.current_step = 0
        logger.info("Streaming predictor reset")


__all__ = [
    'StreamingDroneBuffer',
    'StreamingSwarmPredictor',
    'StreamingPredictionResult',
]
