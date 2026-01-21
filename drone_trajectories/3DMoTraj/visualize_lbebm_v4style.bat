@echo off
REM ============================================================================
REM LBEBM3D v4-style visualization (Windows Batch)
REM ============================================================================

setlocal enabledelayedexpansion

set WORKSPACE=d:\Trajectory prediction
set TOOL_DIR=%WORKSPACE%\drone_trajectories\3DMoTraj\tool
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments

set MODEL_PATH=%WORKSPACE%\drone_trajectories\3DMoTraj\saved_models\checkpoints_accfix\best.pt
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\3DMoTraj\lbebm_visualization_output

if not exist "%MODEL_PATH%" (
  echo [ERROR] Model not found: %MODEL_PATH%
  pause
  exit /b 1
)

cd /d "%TOOL_DIR%"

python visualize_lbebm_prediction_v4style.py ^
  --model_path "%MODEL_PATH%" ^
  --data_dir "%DATA_DIR%" ^
  --agents 3 ^
  --use_subset ^
  --output_dir "%OUTPUT_DIR%" ^
  --num_samples 10 ^
  --seed 42 ^
  --device cuda:0 ^
  --data_scale 1.0 ^
  --e_init_sig 2.0 ^
  --e_prior_sig 2.0 ^
  --e_l_steps 20 ^
  --e_l_step_size 0.4 ^
  --e_l_with_noise

if errorlevel 1 (
  echo [ERROR] Visualization failed.
  pause
  exit /b 1
)

echo.
echo Done. Check PNGs in: %OUTPUT_DIR%
pause

