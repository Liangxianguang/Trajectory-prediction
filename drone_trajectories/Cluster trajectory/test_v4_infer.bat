@echo off
REM ============================================================================
REM v4 推断脚本 (Windows Batch - 32D特征版本)
REM ============================================================================

setlocal enabledelayedexpansion

REM 设置路径
set WORKSPACE=d:\Trajectory prediction
set MODEL_PATH=%WORKSPACE%\drone_trajectories\Cluster trajectory\gru_models_v4_fixed_agents_3\best_model_v4_agents_3.pt
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments
set FEATURES_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\features_32d
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\infer_results_v4

REM 检查模型文件
if not exist "%MODEL_PATH%" (
    echo.
    echo [错误] 模型文件不存在: %MODEL_PATH%
    echo.
    echo 请确保已训练 v4 模型。
    echo.
    pause
    exit /b 1
)

REM 检查特征目录
if not exist "%FEATURES_DIR%" (
    echo.
    echo [错误] 特征目录不存在: %FEATURES_DIR%
    echo.
    echo 请确保已生成 32D 预计算特征。
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo v4 推断脚本 (32D特征版本)
echo ============================================================================
echo.
echo 模型路径:    %MODEL_PATH%
echo 数据目录:    %DATA_DIR%
echo 特征目录:    %FEATURES_DIR%
echo 输出目录:    %OUTPUT_DIR%
echo.

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 切换到脚本目录
cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory"

REM 快速推断测试 (10个样本)
echo [1/1] 执行推断 (10 样本)...
python infer_swarm_model_v4.py ^
    --model "%MODEL_PATH%" ^
    --agents 3 ^
    --data_dir "%DATA_DIR%" ^
    --features_dir "%FEATURES_DIR%" ^
    --num_samples 10 ^
    --output_dir "%OUTPUT_DIR%" ^
    --seed 42 ^
    --use_subset

if errorlevel 1 (
    echo.
    echo [错误] 推断失败!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo 推断完成！
echo ============================================================================
echo.
echo 输出文件位置:
echo   - 预测结果: %OUTPUT_DIR%\predictions_agents_3_v4.npz
echo   - 评估报告: %OUTPUT_DIR%\evaluation_report_agents_3_v4.txt
echo.
echo 您现在可以在 %OUTPUT_DIR% 目录中查看推断结果。
echo.
pause
