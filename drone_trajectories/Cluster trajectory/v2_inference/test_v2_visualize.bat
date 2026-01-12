@echo off
REM ============================================================================
REM v2 可视化测试脚本 (Windows Batch)
REM ============================================================================

setlocal enabledelayedexpansion

REM 设置路径
set WORKSPACE=d:\Trajectory prediction
set MODEL_PATH=%WORKSPACE%\drone_trajectories\Cluster trajectory\24dmodel\swarm_segments_subset_feature\best_model_agents_3_v2.pt
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\v2_inference\visualization_output

REM 检查模型文件
if not exist "%MODEL_PATH%" (
    echo.
    echo [错误] 模型文件不存在: %MODEL_PATH%
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo v2 可视化测试
echo ============================================================================
echo.
echo 模型路径:    %MODEL_PATH%
echo 数据目录:    %DATA_DIR%
echo 输出目录:    %OUTPUT_DIR%
echo.

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 设置 Python 路径
cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory\v2_inference"

REM 运行可视化
echo [1/1] 生成可视化...
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
    echo [错误] 可视化生成失败!
    pause
    exit /b 1
)

echo.
echo [?] 可视化完成
echo.
echo 输出文件位置: %OUTPUT_DIR%
echo.
pause
