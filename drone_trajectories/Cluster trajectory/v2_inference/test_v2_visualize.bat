@echo off
REM ============================================================================
REM v2 Visualization Script (Windows Batch)
REM ============================================================================

setlocal enabledelayedexpansion

REM Set paths
set WORKSPACE=d:\Trajectory prediction
set MODEL_PATH=%WORKSPACE%\drone_trajectories\Cluster trajectory\24dmodel\swarm_segments_subset_feature\best_model_agents_3_v2.pt
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\v2_inference\visualization_output

REM Check model file
if not exist "%MODEL_PATH%" (
    echo.
    echo [ERROR] Model file does not exist: %MODEL_PATH%
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo v2 Visualization
echo ============================================================================
echo.
echo Model path:    %MODEL_PATH%
echo Data directory:    %DATA_DIR%
echo Output directory:    %OUTPUT_DIR%
echo.

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Change to Python path
cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory\v2_inference"

REM Run visualization
echo [1/1] Generating visualization...
python visualize_swarm_prediction_v2.py ^
    --model_path "%MODEL_PATH%" ^
    --agents 3 ^
    --data_dir "%DATA_DIR%" ^
    --num_samples 300 ^
    --output_dir "%OUTPUT_DIR%" ^
    --seed 42 ^
    --use_subset ^
    --fast

if errorlevel 1 (
    echo.
    echo [ERROR] Visualization failed!
    pause
    exit /b 1
)

echo.
echo [?] Visualization complete
echo.
echo Output file location: %OUTPUT_DIR%
echo.
pause
