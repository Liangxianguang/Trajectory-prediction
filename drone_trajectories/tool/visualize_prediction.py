#!/usr/bin/env python3
"""推理 + 可视化验证：在 validation_results 下输出图表与预测数据
python visualize_prediction.py ^
  --model "gru_models_enhanced/enhanced_gru_model_best_model.pth" ^
  --stats "gru_models_enhanced/enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory2D-2_266.csv" ^
  --method physics_constrained ^
  --interactive ^
  --smoothing-weight 0.1,0.1,0.1

python visualize_prediction.py ^
  --model "new_gru_models_enhanced/new_enhanced_gru_model_best_model.pth" ^
  --stats "new_gru_models_enhanced/new_enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\drone_trajectories\random_traj_100ms\line_25.txt" ^
  --method physics_constrained ^
  --interactive ^
  --smoothing-weight 0.1,0.1,0.1

python visualize_prediction.py ^
  --model "new_gru_models_enhanced/new_enhanced_gru_model_best_model.pth" ^
  --stats "new_gru_models_enhanced/new_enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\drone_trajectories\random_traj_100ms\circle_23.txt" ^
  --method physics_constrained ^
  --interactive ^
  --smoothing-weight 0.01,0.01,0.01

python visualize_prediction.py ^
  --model "long_gru_models_enhanced/long_enhanced_gru_model_best_model.pth" ^
  --stats "long_gru_models_enhanced/long_enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory3D-2_141.csv" ^
  --method physics_constrained ^
  --interactive ^
  --smoothing-weight 0.1,0.1,0.1

python visualize_prediction.py ^
  --model "new_mid_gru_models_enhanced/mid_enhanced_gru_model_best_model.pth" ^
  --stats "new_mid_gru_models_enhanced/mid_enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory3D-2_41.csv" ^
  --method simple ^
  --interactive ^
  --smoothing-weight 0.1,0.1,0.1

python visualize_prediction.py ^
  --model "new_long_gru_models_plane_supervised/long_enhanced_gru_model_best_model.pth" ^
  --stats "new_long_gru_models_plane_supervised/long_enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory3D-2_41.csv" ^
  --method physics_constrained ^
  --interactive ^
  --smoothing-weight 0.1,0.1,0.1

python visualize_prediction.py ^
  --model "combined_mid_gru_models_enhanced/mid_enhanced_gru_model_best_model.pth" ^
  --stats "combined_mid_gru_models_enhanced/mid_enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory3D-2_41.csv" ^
  --method physics_constrained ^
  --interactive ^
  --smoothing-weight 0.1,0.1,0.1

python visualize_prediction.py ^
  --model "new_long_gru_models_plane_supervised/long_enhanced_gru_model_best_model.pth" ^
  --stats "new_long_gru_models_plane_supervised/long_enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory2D-2_211.csv" ^
  --method physics_constrained ^
  --interactive ^
  --smoothing-weight 0.1,0.1,0.1

  python visualize_prediction.py ^
  --model "newdata1_short_gru_models/short_enhanced_gru_model_best_model.pth" ^
  --stats "newdata1_short_gru_models/short_enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\drone_trajectories\random_traj_100ms\circle_96.txt" ^
  --method physics_constrained ^
  --interactive ^
  --smoothing-weight 0.1,0.1,0.1 ^
  --use_attention

  python visualize_prediction.py ^
  --model "bigru_long_gru_models/long_enhanced_gru_model_best_model.pth" ^
  --stats "bigru_long_gru_models/long_enhanced_gru_model_norm_stats.npz" ^
  --trajectory "..\..\drone_trajectories\random_traj_100ms\circle_96.txt" ^
  --method physics_constrained ^
  --interactive ^
  --smoothing-weight 0.1,0.1,0.1 ^
  --use_attention

  
  D:\Trajectory prediction\drone_trajectories\random_traj_100ms\circle_66.txt
  D:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory3D-2_41.csv
  D:\Trajectory prediction\drone_trajectories\tool\new_gru_models_enhanced\new_enhanced_gru_model_best_model.pth
  drone_trajectories\tool\combined_long_gru_models_enhanced\long_enhanced_gru_model_best_model.pth
  combined_long_gru_models_enhanced\long_enhanced_gru_model_norm_stats.npz
  new_long_gru_models_plane_supervised
  new_mid_gru_models_speed\mid_enhanced_gru_model_best_model.pth
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib
try:
    plotly_spec = importlib.util.find_spec("plotly.graph_objects")
    if plotly_spec is not None:
        go = importlib.import_module("plotly.graph_objects")
        PLOTLY_AVAILABLE = True
    else:
        go = None
        PLOTLY_AVAILABLE = False
except Exception:
    go = None
    PLOTLY_AVAILABLE = False

from infer_enhanced import EnhancedInference

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

#VALIDATION_ROOT = Path(__file__).resolve().parents[2] / "validation_results"
VALIDATION_ROOT = Path(__file__).resolve().parents[2] / "validation_results_newloss_NEWdata"
VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)


def parse_smoothing_weight(value: str):
    if "," in value:
        parts = [float(part.strip()) for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("--smoothing-weight must be a scalar or three comma-separated numbers")
        return parts
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--smoothing-weight must be a number or three comma-separated numbers")


def plot_prediction(history, true_future, pred_future, traj_name, output_dir, interactive=False):
    fig = plt.figure(figsize=(16, 10))
    start_point = history[-1]
    true_display = np.vstack([start_point, true_future])
    pred_display = np.vstack([start_point, pred_future])

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax3d.plot(history[:, 0], history[:, 1], history[:, 2], "b-o", label="输入历史", linewidth=2, markersize=4)
    ax3d.plot(true_display[:, 0], true_display[:, 1], true_display[:, 2], "g-s", label="真实未来", linewidth=2.5, markersize=6)
    ax3d.plot(pred_display[:, 0], pred_display[:, 1], pred_display[:, 2], "r--^", label="预测未来", linewidth=2, markersize=6)
    ax3d.set_title("3D轨迹对比 (可用鼠标拖拽旋转)")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(fontsize=9)
    ax3d.grid(True, alpha=0.3)

    ax_xy = fig.add_subplot(2, 3, 2)
    ax_xy.plot(history[:, 0], history[:, 1], "b-o", label="历史", linewidth=2)
    ax_xy.plot(true_display[:, 0], true_display[:, 1], "g-s", label="真实", linewidth=2.5)
    ax_xy.plot(pred_display[:, 0], pred_display[:, 1], "r--^", label="预测", linewidth=2)
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_title("XY 平面")
    ax_xy.legend(fontsize=8)
    ax_xy.grid(True, alpha=0.3)

    ax_xz = fig.add_subplot(2, 3, 3)
    ax_xz.plot(history[:, 0], history[:, 2], "b-o", linewidth=2)
    ax_xz.plot(true_display[:, 0], true_display[:, 2], "g-s", linewidth=2.5)
    ax_xz.plot(pred_display[:, 0], pred_display[:, 2], "r--^", linewidth=2)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_title("XZ 平面")
    ax_xz.grid(True, alpha=0.3)

    ax_yz = fig.add_subplot(2, 3, 4)
    ax_yz.plot(history[:, 1], history[:, 2], "b-o", linewidth=2)
    ax_yz.plot(true_display[:, 1], true_display[:, 2], "g-s", linewidth=2.5)
    ax_yz.plot(pred_display[:, 1], pred_display[:, 2], "r--^", linewidth=2)
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.set_title("YZ 平面")
    ax_yz.grid(True, alpha=0.3)

    ax_ts = fig.add_subplot(2, 3, 5)
    steps = np.arange(len(true_future))
    # 计算各轴误差（分米级，便于观察）
    error_x = np.abs(pred_future[:, 0] - true_future[:, 0])
    error_y = np.abs(pred_future[:, 1] - true_future[:, 1])
    error_z = np.abs(pred_future[:, 2] - true_future[:, 2])
    
    ax_ts.plot(steps, error_x, "r-s", label="X 轴误差", linewidth=2.5, markersize=6)
    ax_ts.plot(steps, error_y, "b-o", label="Y 轴误差", linewidth=2.5, markersize=6)
    ax_ts.plot(steps, error_z, "g-^", label="Z 轴误差", linewidth=2.5, markersize=6)
    ax_ts.set_xlabel("预测步数", fontsize=11, fontweight='bold')
    ax_ts.set_ylabel("绝对误差 (m)", fontsize=11, fontweight='bold')
    ax_ts.set_title("各轴逐步误差对比", fontsize=13, fontweight='bold')
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(fontsize=9)

    ax_err = fig.add_subplot(2, 3, 6)
    errors = np.linalg.norm(pred_future - true_future, axis=1)
    ax_err.bar(steps, errors, color="tab:red", alpha=0.7)
    ax_err.set_xlabel("步数")
    ax_err.set_ylabel("位置误差 (m)")
    ax_err.set_title("每步误差")
    ax_err.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    png_path = output_dir / f"prediction_visual_{traj_name}.png"
    html_path = None
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    if interactive:
        plt.show()
    plt.close(fig)

    scene = {"xaxis": {"title": "X (m)"}, "yaxis": {"title": "Y (m)"}, "zaxis": {"title": "Z (m)"}}
    if PLOTLY_AVAILABLE:
        plotly_fig = go.Figure()
        plotly_fig.add_trace(go.Scatter3d(x=history[:, 0], y=history[:, 1], z=history[:, 2], mode="markers+lines",
                                          name="输入历史", marker=dict(size=4, color="blue")))
        plotly_fig.add_trace(go.Scatter3d(x=true_display[:, 0], y=true_display[:, 1], z=true_display[:, 2], mode="markers+lines",
                                          name="真实未来", marker=dict(size=5, color="green")))
        plotly_fig.add_trace(go.Scatter3d(x=pred_display[:, 0], y=pred_display[:, 1], z=pred_display[:, 2], mode="markers+lines",
                                          name="预测未来", marker=dict(size=5, color="red")))
        plotly_fig.update_layout(title=f"交互式轨迹 - {traj_name}", scene=scene, width=1000, height=600)
        html_path = output_dir / f"prediction_visual_interactive_{traj_name}.html"
        plotly_fig.write_html(html_path, include_plotlyjs="cdn")

    return png_path, html_path


def main():
    parser = argparse.ArgumentParser(description="增强模型推理+可视化")
    parser.add_argument("--model", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--method", default="physics_constrained",
                        choices=["simple", "physics_constrained", "smoothed"])
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--smoothing-weight", type=parse_smoothing_weight, default="0.3",
                        help="加速度平滑权重，可选 float（统一）或三个逗号分隔值 (X,Y,Z)")
    parser.add_argument("--interactive", action="store_true", help="显示 Matplotlib 窗口以便鼠标旋转观察")
    parser.add_argument("--use_attention", action="store_true", help="使用注意力机制（与模型训练配置一致）")
    parser.add_argument("--input_length", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=str(VALIDATION_ROOT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infer = EnhancedInference(args.model, args.stats, hidden_dim=128, num_layers=3, 
                             use_attention=args.use_attention)

    trail = pd.read_csv(args.trajectory)[["tx", "ty", "tz"]].values.astype(np.float32)
    if len(trail) < args.input_length + 10:
        raise ValueError("轨迹长度不足")

    history = trail[-(args.input_length + 10):-10]
    true_future = trail[-10:]

    if args.method == "simple":
        pred = infer.reconstruct_positions_simple(history, args.dt, args.input_length)
    elif args.method == "physics_constrained":
        pred = infer.reconstruct_positions_physics_constrained(
            history, args.dt, args.input_length, smoothing_weight=args.smoothing_weight
        )
    else:
        pred = infer.reconstruct_positions_trajectory_smoothing(
            history, args.dt, args.input_length, smoothing_weight=args.smoothing_weight
        )

    pred_path = output_dir / f"prediction_{Path(args.trajectory).stem}_{args.method}.csv"
    pd.DataFrame(pred, columns=["x", "y", "z"]).to_csv(pred_path, index=False)


    errors = np.linalg.norm(pred - true_future, axis=1)
    metrics = {
        "mae": np.mean(errors),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "steps": len(errors)
    }

    comparison_path = output_dir / f"trajectory_comparison_{Path(args.trajectory).stem}_{args.method}.csv"
    comparison_df = pd.DataFrame({
        "step": np.arange(len(true_future)),
        "true_x": true_future[:, 0],
        "true_y": true_future[:, 1],
        "true_z": true_future[:, 2],
        "pred_x": pred[:, 0],
        "pred_y": pred[:, 1],
        "pred_z": pred[:, 2],
    })
    comparison_df.to_csv(comparison_path, index=False)

    png_path, html_path = plot_prediction(history, true_future, pred, Path(args.trajectory).stem, output_dir, interactive=args.interactive)

    summary_path = output_dir / f"prediction_summary_{Path(args.trajectory).stem}.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"method: {args.method}\n")
        fh.write(f"trajectory: {args.trajectory}\n")
        fh.write(f"mae: {metrics['mae']:.4f} m\n")
        fh.write(f"rmse: {metrics['rmse']:.4f} m\n")
        fh.write(f"steps: {metrics['steps']}\n")
        fh.write(f"predictions: {pred_path}\n")
        fh.write(f"visual_png: {png_path}\n")
        fh.write(f"comparison_csv: {comparison_path}\n")
        fh.write(f"smoothing_weight: {args.smoothing_weight}\n")
        if html_path:
            fh.write(f"visual_html: {html_path}\n")

    print(f"预测保存: {pred_path}")
    print(f"可视化静态图: {png_path}")
    print(f"真实/预测对比 CSV: {comparison_path}")
    print(f"加速度平滑权重: {args.smoothing_weight}")
    if html_path:
        print(f"可视化交互 HTML: {html_path}（浏览器打开即可拖拽旋转）")
    else:
        print("可视化交互 HTML: plotly 未安装，跳过生成")
    print(f"指标: MAE={metrics['mae']:.4f}m RMSE={metrics['rmse']:.4f}m")


if __name__ == "__main__":
    main()
