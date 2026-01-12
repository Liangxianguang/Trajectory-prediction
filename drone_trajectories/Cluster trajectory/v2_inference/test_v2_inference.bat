@echo off
REM ============================================================================
REM v2 推理测试脚本 (Windows Batch)
REM ============================================================================

setlocal enabledelayedexpansion

REM 设置路径
set WORKSPACE=d:\Trajectory prediction
set MODEL_PATH=%WORKSPACE%\drone_trajectories\Cluster trajectory\24dmodel\swarm_segments_subset\best_model_agents_3_v2.pt
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\visualization_output

REM 检查模型文件
if not exist "%MODEL_PATH%" (
    echo.
    echo [错误] 模型文件不存在: %MODEL_PATH%
    echo.
    echo 请确保已运行训练脚本生成模型:
    echo   python train_swarm_v2_complete.py --num_agents 3
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo v2 推理测试
echo ============================================================================
echo.
echo 模型路径:    %MODEL_PATH%
echo 数据目录:    %DATA_DIR%
echo 输出目录:    %OUTPUT_DIR%
echo.

REM 设置 Python 路径
cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory\v2_inference"

REM 运行推理
echo [1/3] 运行推理...
python infer_swarm_model_v2.py ^
    --model_path "%MODEL_PATH%" ^
    --num_agents 3 ^
    --data_dir "%DATA_DIR%" ^
    --num_samples 5 ^
    --seed 42

if errorlevel 1 (
    echo.
    echo [错误] 推理失败!
    pause
    exit /b 1
)

echo.
echo [?] 推理完成
echo.
pause
