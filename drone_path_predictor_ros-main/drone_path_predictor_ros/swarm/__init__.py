"""
Drone Swarm Prediction Module

Sub-modules:
- swarm_data_loader: Load multiple drone trajectories
- drone_swarm_predictor: Multi-drone trajectory prediction with collision detection
- swarm_visualizer: 3D visualization of swarm behavior
- swarm_demo: End-to-end demonstration

Author: Trajectory Prediction Team
"""

from .swarm_data_loader import (
    SwarmDataLoader,
    SwarmTrajectoryData,
    load_swarm_trajectories
)

from .drone_swarm_predictor import (
    DroneSwarmPredictor,
)
from .swarm_visualizer import (
    plot_swarm_static,
    plot_swarm_animation,
    plot_swarm_plotly,
    visualize_evolution,
    plot_swarm_comparison,
    plot_swarm_overlay
)

from .swarm_streaming_predictor import (
    StreamingSwarmPredictor,
    StreamingDroneBuffer,
    StreamingPredictionResult
)

from .swarm_demo import main as run_swarm_demo

__all__ = [
    'SwarmDataLoader',
    'SwarmTrajectoryData',
    'load_swarm_trajectories',
    'DroneSwarmPredictor',
    'plot_swarm_static',
    'plot_swarm_animation',
    'plot_swarm_plotly',
    'visualize_evolution',
    'plot_swarm_comparison',
    'plot_swarm_overlay',
    'StreamingSwarmPredictor',
    'StreamingDroneBuffer',
    'StreamingPredictionResult',
    'run_swarm_demo',
]

__version__ = '1.0.0'
