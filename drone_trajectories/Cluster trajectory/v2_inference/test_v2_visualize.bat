@echo off
REM ============================================================================
REM v2 ??????????? (Windows Batch)
REM ============================================================================

setlocal enabledelayedexpansion

REM ????¡¤??
set WORKSPACE=d:\Trajectory prediction
set MODEL_PATH=%WORKSPACE%\drone_trajectories\Cluster trajectory\24dmodel\swarm_segments_subset_feature\best_model_agents_3_v2.pt
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\v2_inference\visualization_output

REM ?????????
if not exist "%MODEL_PATH%" (
    echo.
    echo [????] ????????????: %MODEL_PATH%
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo v2 ?????????
echo ============================================================================
echo.
echo ???¡¤??:    %MODEL_PATH%
echo ??????:    %DATA_DIR%
echo ?????:    %OUTPUT_DIR%
echo.

REM ?????????
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM ???? Python ¡¤??
cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory\v2_inference"

REM ???§á????
echo [1/1] ????????...
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
    echo [????] ????????????!
    pause
    exit /b 1
)

echo.
echo [?] ????????
echo.
echo ??????¦Ë??: %OUTPUT_DIR%
echo.
pause
