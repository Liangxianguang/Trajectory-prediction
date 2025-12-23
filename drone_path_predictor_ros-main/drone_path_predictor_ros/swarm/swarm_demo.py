"""End-to-end demonstration of the swarm prediction workflow."""

import argparse
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

from drone_path_predictor_ros.swarm import (
    SwarmDataLoader,
    DroneSwarmPredictor,
    plot_swarm_animation,
    plot_swarm_static,
    plot_swarm_plotly,
    visualize_evolution,
)

logger = logging.getLogger(__name__)


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "drone_trajectories" / "random_traj_100ms"


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "trajectory_predictor.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick swarm prediction demo using saved trajectory samples."
    )
    parser.add_argument("-d", "--data-dir", type=Path, default=_default_data_dir(),
                        help="Directory containing trajectory files.")
    parser.add_argument("-c", "--config-file", type=Path, default=_default_config_path(),
                        help="Config file that points to the trained models.")
    parser.add_argument("-p", "--pattern", default="*.txt",
                        help="Glob pattern used to discover trajectory files.")
    parser.add_argument("-m", "--max-drones", type=int, default=5,
                        help="Limit the number of drones to load for prediction.")
    parser.add_argument("-t", "--collision-threshold", type=float, default=2.0,
                        help="Collision distance threshold (meters).")
    parser.add_argument("--prediction-mode", choices=("delta", "absolute"), default="delta",
                        help="How to interpret the GRU output (relative vs absolute).")
    parser.add_argument("--min-severity", choices=("info", "warning", "critical"), default="info",
                        help="Minimum severity to log when reporting collisions.")
    parser.add_argument("--device", default=None, help="Override the device used by the predictor")
    parser.add_argument("--no-threading", action="store_true",
                        help="Disable multi-threaded prediction (useful for debugging).")
    parser.add_argument("--visualize", action="store_true",
                        help="Display a static Matplotlib scatter of the final step.")
    parser.add_argument("--animate", action="store_true",
                        help="Animate the predicted trajectories in Matplotlib.")
    parser.add_argument("--plotly", action="store_true",
                        help="Render an interactive Plotly visualization (requires plotly installed).")
    parser.add_argument("--animation-interval", type=int, default=200,
                        help="Interval (ms) between animation frames.")
    parser.add_argument("--animation-trail", type=int, default=8,
                        help="Number of past steps to draw as trailing lines in the animation.")
    parser.add_argument("--plot-step", type=int, default=-1,
                        help="Which predicted step to visualize for the static snapshot.")
    return parser.parse_args()


def _load_swarms(loader: SwarmDataLoader, data_dir: Path, pattern: str, max_drones: int):
    logger.info("Loading swarm trajectories from %s", data_dir)
    swarm_data = loader.load_directory(str(data_dir), pattern=pattern, max_drones=max_drones)
    logger.info("Loaded %d drones, %.2fs duration", swarm_data.num_drones, swarm_data.duration)
    return swarm_data


def _log_statistics(loader: SwarmDataLoader, swarm_data):
    stats = loader.compute_swarm_statistics(swarm_data)
    logger.info("Swarm statistics:")
    logger.info("  Avg length: %.2fm", stats['avg_trajectory_length'])
    logger.info("  Min length: %.2fm", stats['min_trajectory_length'])
    logger.info("  Max length: %.2fm", stats['max_trajectory_length'])
    logger.info("  Avg velocity: %s", stats['avg_velocity'])


def _report_collisions(collisions: List) -> None:
    if not collisions:
        logger.info("No collision alerts detected.")
        return
    logger.warning("Collision alerts (%d):", len(collisions))
    for alert in collisions:
        logger.warning("  %s", alert)


def _main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    loader = SwarmDataLoader()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {args.data_dir}")

    swarm_data = _load_swarms(loader, args.data_dir, args.pattern, args.max_drones)
    _log_statistics(loader, swarm_data)

    predictor = DroneSwarmPredictor(
        config_file=str(args.config_file),
        collision_threshold=args.collision_threshold,
        prediction_mode=args.prediction_mode,
        device=args.device,
    )
    predictions = predictor.predict_swarm(swarm_data, use_threading=not args.no_threading)
    collisions = predictor.detect_collisions(predictions, min_severity=args.min_severity)
    evolution = predictor.analyze_swarm_evolution(predictions)

    summary = predictor.summary(predictions, collisions)
    print(summary)
    _report_collisions(collisions)

    # Visualization (with graceful fallback on matplotlib issues)
    if args.visualize:
        try:
            plot_swarm_static(predictions, step=args.plot_step)
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                "Could not display static visualization: %s\n"
                "Try: pip install --upgrade matplotlib", e
            )
    
    if args.animate:
        try:
            anim = plot_swarm_animation(
                predictions,
                interval=args.animation_interval,
                trail=args.animation_trail
            )
            if anim is not None:
                try:
                    plt = importlib.import_module('matplotlib.pyplot')
                    plt.show()
                except (ImportError, ModuleNotFoundError) as e:
                    logger.warning("Could not show animation: %s", e)
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning("Could not create animation: %s", e)
    
    if args.plotly:
        try:
            plot_swarm_plotly(predictions, title="Swarm prediction demo")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                "Could not display Plotly visualization: %s\n"
                "Try: pip install plotly", e
            )


def main() -> None:
    try:
        _main()
    except Exception as e:
        logger.exception("Swarm demo failed: %s", e)
        raise


if __name__ == "__main__":
    main()
