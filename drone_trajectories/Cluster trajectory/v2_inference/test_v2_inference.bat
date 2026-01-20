@echo off
REM ============================================================================
REM v2 Inference Script (Windows Batch)
REM ============================================================================

setlocal enabledelayedexpansion

REM Set paths
set WORKSPACE=d:\Trajectory prediction
set MODEL_PATH=%WORKSPACE%\drone_trajectories\Cluster trajectory\24dmodel\swarm_segments_subset\best_model_agents_3_v2.pt
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\visualization_output

REM Check model file
if not exist "%MODEL_PATH%" (
    echo.
    echo [ERROR] Model file does not exist: %MODEL_PATH%
    echo.
    echo Please ensure you have trained the model using:
    echo   python train_swarm_v2_complete.py --num_agents 3
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo v2 Inference
echo ============================================================================
echo.
echo Model path:    %MODEL_PATH%
echo Data directory:    %DATA_DIR%
echo Output directory:    %OUTPUT_DIR%
echo.

REM Change to Python path
cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory\v2_inference"

REM Run inference
echo [1/3] Running inference...
python infer_swarm_model_v2.py ^
    --model_path "%MODEL_PATH%" ^
    --num_agents 3 ^
    --data_dir "%DATA_DIR%" ^
    --num_samples 5 ^
    --seed 42

if errorlevel 1 (
    echo.
    echo [ERROR] Inference failed!
    pause
    exit /b 1
)

echo.
echo [?] Inference complete
echo.
pause
