#!/usr/bin/env python3
"""Simple linear regression test harness for Predictor."""

import argparse
import sys
from pathlib import Path

import numpy as np

# ensure parent modules importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from trajectory_predictor import Predictor


def generate_linear_traj(start, velocity, num_points, dt):
    t = np.arange(num_points) * dt
    traj = np.array([start + velocity * ti for ti in t])
    return traj


def evaluate_prediction(input_seq, prediction, actual_future):
    last_known = input_seq[-1]
    pred_direct = prediction.copy()
    pred_cumsum = np.cumsum(prediction, axis=0) + last_known
    length = min(len(pred_direct), len(actual_future))
    pred_direct = pred_direct[:length]
    pred_cumsum = pred_cumsum[:length]
    actual = actual_future[:length]
    err_direct = np.linalg.norm(actual - pred_direct, axis=1)
    err_cumsum = np.linalg.norm(actual - pred_cumsum, axis=1)
    return {
        "direct": {
            "per_step": err_direct,
            "mae": np.mean(err_direct),
            "rmse": np.sqrt(np.mean(err_direct**2)),
        },
        "cumsum": {
            "per_step": err_cumsum,
            "mae": np.mean(err_cumsum),
            "rmse": np.sqrt(np.mean(err_cumsum**2)),
        },
        "pred_direct": pred_direct,
        "pred_cumsum": pred_cumsum,
    }


def main():
    parser = argparse.ArgumentParser(description="Linear regression sanity test for Predictor")
    parser.add_argument("--position_model_path", type=str, required=True)
    parser.add_argument("--velocity_model_path", type=str, required=True)
    parser.add_argument("--position_stats_file", type=str, required=True)
    parser.add_argument("--velocity_stats_file", type=str, required=True)
    parser.add_argument("--use_whitening", action="store_true")
    parser.add_argument("--use_velocity_prediction", action="store_true")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--input_length", type=int, default=21)
    parser.add_argument("--future_steps", type=int, default=10)
    parser.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--velocity", type=float, nargs=3, default=[0.5, 0.2, 0.0])
    parser.add_argument("--num_points", type=int, default=200)
    args = parser.parse_args()

    start = np.array(args.start, dtype=float)
    velocity = np.array(args.velocity, dtype=float)
    traj = generate_linear_traj(start, velocity, args.num_points, args.dt)

    input_seq = traj[-(args.input_length + args.future_steps):-args.future_steps]
    actual_future = traj[-args.future_steps:]

    print("Trajectory setup:")
    print(f"  start={start}, velocity={velocity}, dt={args.dt}")
    print(f"  input_length={len(input_seq)}, future_steps={len(actual_future)}")
    print(f"  last_known={input_seq[-1]}\n")

    predictor = Predictor(
        args.position_model_path,
        args.velocity_model_path,
        args.position_stats_file,
        args.velocity_stats_file,
        pos_hidden_dim=256,
        pos_num_layers=5,
        pos_dropout=0.5,
        vel_hidden_dim=256,
        vel_num_layers=5,
        vel_dropout=0.5,
        use_whitening=args.use_whitening,
    )

    if args.use_velocity_prediction:
        prediction = predictor.predict_positions_from_velocity(input_seq, args.dt)
    else:
        prediction = predictor.predict_positions(input_seq)

    metrics = evaluate_prediction(input_seq, prediction, actual_future)

    print("Prediction sample (first 3 steps):")
    for idx in range(min(3, len(actual_future))):
        print(f"  step {idx+1}: actual={actual_future[idx]}, direct={metrics['pred_direct'][idx]}, cumsum={metrics['pred_cumsum'][idx]}")

    print("\nDirect errors:")
    print(f"  MAE={metrics['direct']['mae']:.4f}, RMSE={metrics['direct']['rmse']:.4f}")
    print(f"  per step: {[float(e) for e in metrics['direct']['per_step']]}\n")

    print("Cumsum (relative) errors:")
    print(f"  MAE={metrics['cumsum']['mae']:.4f}, RMSE={metrics['cumsum']['rmse']:.4f}")
    print(f"  per step: {[float(e) for e in metrics['cumsum']['per_step']]}\n")


if __name__ == "__main__":
    main()
