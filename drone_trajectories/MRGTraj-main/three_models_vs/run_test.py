#!/usr/bin/env python3
"""快速测试脚本"""
import subprocess
import sys

cmd = [
    sys.executable,
    "compare_three_models.py",
    "--data_dir", r"D:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments",
    "--agents", "3",
    "--use_subset",
    "--lbebm_model", r"D:\Trajectory prediction\drone_trajectories\3DMoTraj\saved_models\checkpoints_accfix\epoch_010.pt",
    "--exp5_dir", r"D:\Trajectory prediction\drone_trajectories\Cluster trajectory\ablation study\ablation_results_agents_3_exp5_full",
    "--mrgraj_model", r"D:\Trajectory prediction\drone_trajectories\MRGTraj-main\checkpoints_lbebm3d\agents_3_lbebm3d_inspired\best_model.pth",
    "--features_32d_dir", r"D:\Trajectory prediction\drone_trajectories\Cluster trajectory\features_32d",
    "--num_samples", "3",
    "--seed", "42",
    "--output_dir", r"D:\Trajectory prediction\drone_trajectories\MRGTraj-main\three_models_vs\comparison_results",
]

print(f"运行命令: {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=r"D:\Trajectory prediction\drone_trajectories\MRGTraj-main\three_models_vs")
sys.exit(result.returncode)
