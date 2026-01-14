@echo off
REM ============================================================================
REM v3 快速可视化脚本 (Windows Batch)
REM ============================================================================

setlocal enabledelayedexpansion

REM 设置路径
set WORKSPACE=d:\Trajectory prediction
set MODEL_PATH=%WORKSPACE%\drone_trajectories\Cluster trajectory\gru_models_v3_agents_3_v3_gnn_concat\last_checkpoint_0199.pt
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\visualization_output_v3

REM 检查模型文件
if not exist "%MODEL_PATH%" (
    echo.
    echo [错误] 模型文件不存在: %MODEL_PATH%
    echo.
    echo 请确保已训练 v3 模型。
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo v3 快速可视化脚本
echo ============================================================================
echo.
echo 模型路径:    %MODEL_PATH%
echo 数据目录:    %DATA_DIR%
echo 输出目录:    %OUTPUT_DIR%
echo.

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 切换到脚本目录
cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory"

REM 快速可视化 (使用合成数据进行测试)
echo [1/1] 执行快速可视化...
python visualize_swarm_prediction_v3.py ^
    --model_path "%MODEL_PATH%" ^
    --agents 3 ^
    --data_dir "%DATA_DIR%" ^
    --num_samples 3 ^
    --output_dir "%OUTPUT_DIR%" ^
    --seed 42 ^
    --use_subset ^
    --fast

if errorlevel 1 (
    echo.
    echo [错误] 可视化失败!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo 快速可视化完成！
echo ============================================================================
echo.
echo 输出文件位置:
echo   - PNG 图表: %OUTPUT_DIR%\swarm_prediction_v3_sample_*.png
echo   - 评估报告: %OUTPUT_DIR%\evaluation_report_v3_*.json
echo.
echo 您现在可以在 %OUTPUT_DIR% 目录中查看生成的图表和报告。
echo.
pause
