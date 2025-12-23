"""Visualization helpers for the swarm prediction module."""

import importlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ColorMap = List[str]

_plt_cache: Dict[str, Any] = {}


def _load_matplotlib() -> Tuple[Any, Any, Any]:
    if _plt_cache:
        return _plt_cache['plt'], _plt_cache['animation'], _plt_cache['Axes']
    try:
        plt = importlib.import_module('matplotlib.pyplot')
        animation = importlib.import_module('matplotlib.animation')
        axes_mod = importlib.import_module('matplotlib.axes')
        Axes = getattr(axes_mod, 'Axes')
        _plt_cache.update({'plt': plt, 'animation': animation, 'Axes': Axes})
        return plt, animation, Axes
    except (ImportError, ModuleNotFoundError) as e:
        logger.error("Failed to import matplotlib modules: %s", e)
        raise


def _validate_predictions(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not predictions:
        raise ValueError("Predictions dictionary cannot be empty")
    min_len = min(len(trajectory) for trajectory in predictions.values())
    if min_len == 0:
        raise ValueError("All prediction trajectories must contain at least one point")
    return {
        drone_id: trajectory[:min_len]
        for drone_id, trajectory in predictions.items()
    }


def _extract_positions(predictions: Dict[str, np.ndarray], step: int,
                       normalized: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    normalized = normalized or _validate_predictions(predictions)
    if step < 0:
        step = next(iter(normalized.values())).shape[0] + step
    step = max(0, min(step, next(iter(normalized.values())).shape[0] - 1))
    return np.array([trajectory[step] for trajectory in normalized.values()])


def _compute_bounds(data: np.ndarray) -> Dict[str, float]:
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    span = (max_vals - min_vals) * 0.2
    return {
        'x_min': min_vals[0] - span[0],
        'x_max': max_vals[0] + span[0],
        'y_min': min_vals[1] - (span[1] if span[1] else 1.0),
        'y_max': max_vals[1] + (span[1] if span[1] else 1.0),
        'z_min': min_vals[2] - (span[2] if span[2] else 1.0),
        'z_max': max_vals[2] + (span[2] if span[2] else 1.0),
    }


def plot_swarm_static(predictions: Dict[str, np.ndarray],
                      step: int = -1,
                      ax: Optional[Any] = None,
                      colormap: Optional[ColorMap] = None,
                      show: bool = True) -> Any:
    """Plot a single timestep of the predicted swarm trajectories using Matplotlib."""
    plt, _, _ = _load_matplotlib()
    normalized = _validate_predictions(predictions)
    positions = _extract_positions(predictions, step, normalized)
    drone_ids = list(normalized.keys())

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if colormap:
        colors = colormap
    else:
        colors = plt.cm.get_cmap("tab10")(range(len(drone_ids)))

    for idx, (drone_id, pos) in enumerate(zip(drone_ids, positions)):
        ax.scatter(pos[0], pos[1], pos[2], s=50, color=colors[idx % len(colors)], label=drone_id)
        ax.text(pos[0], pos[1], pos[2], drone_id, size=8)

    bounds = _compute_bounds(positions)
    ax.set_xlim(bounds['x_min'], bounds['x_max'])
    ax.set_ylim(bounds['y_min'], bounds['y_max'])
    ax.set_zlim(bounds['z_min'], bounds['z_max'])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"Swarm positions at step {step if step >= 0 else 'final'}")
    ax.legend(loc='upper right', fontsize='small')

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_swarm_animation(predictions: Dict[str, np.ndarray],
                         interval: int = 250,
                         trail: int = 5,
                         axis_limits: Optional[Dict[str, float]] = None) -> Any:
    """Animate predicted swarm trajectories using Matplotlib."""
    plt, animation, _ = _load_matplotlib()
    normalized = _validate_predictions(predictions)
    drone_ids = list(normalized.keys())
    trajectories = np.stack(list(normalized.values()))  # (N_drones, horizon, 3)
    horizon = trajectories.shape[1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.get_cmap("tab10")(range(len(drone_ids)))

    first_positions = trajectories[:, 0]
    scatters = ax.scatter(first_positions[:, 0], first_positions[:, 1], first_positions[:, 2], s=40)
    lines = [ax.plot([], [], [], color=colors[idx % len(colors)], alpha=0.4)[0]
             for idx in range(len(drone_ids))]

    bounds = axis_limits or _compute_bounds(trajectories.reshape(-1, 3))
    ax.set_xlim(bounds['x_min'], bounds['x_max'])
    ax.set_ylim(bounds['y_min'], bounds['y_max'])
    ax.set_zlim(bounds['z_min'], bounds['z_max'])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Swarm trajectory animation')

    def _update(frame: int):
        positions = trajectories[:, frame]
        scatters._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        for idx, line in enumerate(lines):
            start = max(0, frame - trail)
            trail_positions = trajectories[idx, start:frame + 1]
            if trail_positions.size:
                line.set_data(trail_positions[:, 0], trail_positions[:, 1])
                line.set_3d_properties(trail_positions[:, 2])
        return [scatters] + lines

    anim = animation.FuncAnimation(fig, _update, frames=horizon, interval=interval, blit=False)
    return anim


def plot_swarm_plotly(predictions: Dict[str, np.ndarray],
                      title: str = "Swarm prediction",
                      show: bool = True) -> Optional[Any]:
    """Create an interactive Plotly 3D trace of the swarm predictions."""
    try:
        go = importlib.import_module('plotly.graph_objects')
    except ModuleNotFoundError:
        logger.warning('Plotly is not installed; install plotly to use this helper')
        return None
    normalized = _validate_predictions(predictions)
    fig = go.Figure()

    for drone_id, trajectory in normalized.items():
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines+markers',
                name=drone_id,
                marker=dict(size=3),
                line=dict(width=2)
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
        ),
        legend=dict(itemsizing='trace')
    )

    if show:
        fig.show()
    return fig


def visualize_evolution(predictions: Dict[str, np.ndarray],
                        evolution: Optional[Dict[str, List[float]]] = None,
                        show: bool = True) -> Any:
    """Visualize evolution statistics across the prediction horizon."""
    plt, _, _ = _load_matplotlib()
    normalized = _validate_predictions(predictions)
    time_steps = next(iter(normalized.values())).shape[0]
    com = np.array([np.mean([trajectory[t] for trajectory in normalized.values()], axis=0)
                    for t in range(time_steps)])

    if evolution is None:
        unique_positions = np.stack(list(normalized.values()))  # (N, horizon, 3)
        compactness = []
        diameter = []
        min_distance = []
        for t in range(time_steps):
            frame = unique_positions[:, t, :]
            center = np.mean(frame, axis=0)
            distances = np.linalg.norm(frame - center, axis=1)
            compactness.append(np.mean(distances))
            if frame.shape[0] < 2:
                diameter.append(0.0)
                min_distance.append(0.0)
                continue
            pairwise = frame[:, None, :] - frame[None, :, :]
            pairwise_dist = np.linalg.norm(pairwise, axis=2)
            np.fill_diagonal(pairwise_dist, np.inf)
            diameter.append(np.max(pairwise_dist))
            min_distance.append(np.min(pairwise_dist))
        evolution = {
            'compactness_over_time': compactness,
            'diameter_over_time': diameter,
            'min_distance_over_time': min_distance,
        }

    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes_grid.flatten()

    axes[0].plot(com[:, 0], label='CoM X')
    axes[0].plot(com[:, 1], label='CoM Y')
    axes[0].plot(com[:, 2], label='CoM Z')
    axes[0].set_title('Center-of-mass evolution')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Position (m)')
    axes[0].legend()

    axes[1].plot(evolution['compactness_over_time'], color='tab:green')
    axes[1].set_title('Swarm compactness over time')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Average distance from CoM (m)')

    axes[2].plot(evolution['diameter_over_time'], color='tab:orange', label='Diameter')
    axes[2].plot(evolution['min_distance_over_time'], color='tab:red', label='Min pairwise')
    axes[2].set_title('Pairwise distances')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Distance (m)')
    axes[2].legend()

    axes[3].axis('off')
    axes[3].text(0.5, 0.5, 'Swarm evolution overview', ha='center', va='center', fontsize=12)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_swarm_comparison(historical: Dict[str, np.ndarray],
                          predictions: Dict[str, np.ndarray],
                          ax: Optional[Any] = None,
                          colormap: Optional[ColorMap] = None,
                          show: bool = True) -> Tuple[Any, Any]:
    """Compare historical and predicted trajectories side-by-side in two 3D plots."""
    plt, _, _ = _load_matplotlib()
    
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    
    drone_ids = list(historical.keys())
    if colormap:
        colors = colormap
    else:
        colors = plt.cm.get_cmap("tab10")(range(len(drone_ids)))
    
    # Compute bounds for both plots
    all_data = np.vstack(list(historical.values()) + list(predictions.values()))
    bounds = _compute_bounds(all_data)
    
    # Plot 1: Historical trajectories
    for idx, drone_id in enumerate(drone_ids):
        hist = historical[drone_id]
        ax1.plot(hist[:, 0], hist[:, 1], hist[:, 2], 
                color=colors[idx % len(colors)], alpha=0.7, linewidth=2.5, label=drone_id)
        ax1.scatter(hist[-1, 0], hist[-1, 1], hist[-1, 2], 
                   color=colors[idx % len(colors)], s=120, marker='o', edgecolors='black', linewidth=1.5)
    
    ax1.set_xlim(bounds['x_min'], bounds['x_max'])
    ax1.set_ylim(bounds['y_min'], bounds['y_max'])
    ax1.set_zlim(bounds['z_min'], bounds['z_max'])
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Historical Trajectories')
    ax1.legend(loc='upper left', fontsize='small')
    
    # Plot 2: Predicted trajectories
    for idx, drone_id in enumerate(drone_ids):
        if drone_id in predictions:
            pred = predictions[drone_id]
            ax2.plot(pred[:, 0], pred[:, 1], pred[:, 2], 
                    color=colors[idx % len(colors)], alpha=0.7, linewidth=2.5, 
                    linestyle='--', label=f'{drone_id} (pred)')
            ax2.scatter(pred[-1, 0], pred[-1, 1], pred[-1, 2], 
                       color=colors[idx % len(colors)], s=120, marker='^', edgecolors='black', linewidth=1.5)
    
    ax2.set_xlim(bounds['x_min'], bounds['x_max'])
    ax2.set_ylim(bounds['y_min'], bounds['y_max'])
    ax2.set_zlim(bounds['z_min'], bounds['z_max'])
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Predicted Trajectories')
    ax2.legend(loc='upper left', fontsize='small')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax1, ax2


def plot_swarm_overlay(historical: Dict[str, np.ndarray],
                       predictions: Dict[str, np.ndarray],
                       ax: Optional[Any] = None,
                       colormap: Optional[ColorMap] = None,
                       show: bool = True) -> Any:
    """Overlay predicted trajectories on top of historical ones in a single 3D plot."""
    plt, _, _ = _load_matplotlib()
    
    if ax is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
    
    drone_ids = list(historical.keys())
    if colormap:
        colors = colormap
    else:
        colors = plt.cm.get_cmap("tab10")(range(len(drone_ids)))
    
    all_data = np.vstack(list(historical.values()) + list(predictions.values()))
    bounds = _compute_bounds(all_data)
    ax.set_xlim(bounds['x_min'], bounds['x_max'])
    ax.set_ylim(bounds['y_min'], bounds['y_max'])
    ax.set_zlim(bounds['z_min'], bounds['z_max'])
    
    # Historical trajectories (solid lines)
    for idx, drone_id in enumerate(drone_ids):
        hist = historical[drone_id]
        ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], 
               color=colors[idx % len(colors)], alpha=0.8, linewidth=2.5, 
               label=f'{drone_id} (history)')
        ax.scatter(hist[-1, 0], hist[-1, 1], hist[-1, 2], 
                  color=colors[idx % len(colors)], s=120, marker='o', edgecolors='black', linewidth=1.5)
    
    # Predicted trajectories (dashed lines)
    for idx, drone_id in enumerate(drone_ids):
        if drone_id in predictions:
            pred = predictions[drone_id]
            ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], 
                   color=colors[idx % len(colors)], alpha=0.5, linewidth=2, 
                   linestyle='--', label=f'{drone_id} (predicted)')
            ax.scatter(pred[-1, 0], pred[-1, 1], pred[-1, 2], 
                      color=colors[idx % len(colors)], s=120, marker='^', edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Historical vs Predicted Trajectories (Overlay)')
    ax.legend(loc='upper left', fontsize='small', ncol=2)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


__all__ = [
    'plot_swarm_static',
    'plot_swarm_animation',
    'plot_swarm_plotly',
    'visualize_evolution',
    'plot_swarm_comparison',
    'plot_swarm_overlay',
]
