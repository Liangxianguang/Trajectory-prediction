@echo off
REM ============================================================================
REM v4 增强末端推断 + 可视化脚本
REM ============================================================================
REM
REM 功能：
REM   ? 执行 v4 增强末端动力学推断
REM   ? 自动生成可视化对比图
REM   ? 生成详细评估报告
REM   ? 显示末端动力学特征
REM
REM 输出：
REM   - infer_results_v4_enhanced/ : 推断结果
REM   - visualization_v4_enhanced/ : 可视化图表
REM

setlocal enabledelayedexpansion

REM ============================================================================
REM 配置区
REM ============================================================================

set WORKSPACE=d:\Trajectory prediction
set MODEL_PATH=%WORKSPACE%\drone_trajectories\Cluster trajectory\gru_models_v4_fixed_agents_3_v4_fixed_gnn\best_model_agents_3_v4_fixed_gnn.pt
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments
set FEATURES_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\features_32d
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\infer_results_v4_enhanced
set VIS_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\visualization_v4_enhanced_2000
set AGENTS=3
set NUM_SAMPLES=2000
set TAIL_WINDOW=8

REM ============================================================================
REM 验证阶段
REM ============================================================================

echo.
echo ============================================================================
echo v4 增强末端推断 + 可视化 (Enhanced Tail Dynamics)
echo ============================================================================
echo.

REM 检查模型文件
if not exist "%MODEL_PATH%" (
    echo [错误] 模型文件不存在!
    echo 路径: %MODEL_PATH%
    pause
    exit /b 1
)
echo [?] 模型文件已找到

REM 检查数据目录
if not exist "%DATA_DIR%" (
    echo [错误] 数据目录不存在!
    echo 路径: %DATA_DIR%
    pause
    exit /b 1
)
echo [?] 数据目录已找到

REM 检查特征目录
if not exist "%FEATURES_DIR%" (
    echo [错误] 特征目录不存在!
    echo 路径: %FEATURES_DIR%
    pause
    exit /b 1
)
echo [?] 特征目录已找到

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%VIS_DIR%" mkdir "%VIS_DIR%"
echo [?] 输出目录已创建

REM ============================================================================
REM 第 1 步：增强末端推断
REM ============================================================================

echo.
echo [第 1/2 步] 执行 v4 增强末端动力学推断...
echo.

cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory"

python infer_swarm_model_v4_enhanced_tail.py ^
    --model "%MODEL_PATH%" ^
    --data_dir "%DATA_DIR%" ^
    --features_dir "%FEATURES_DIR%" ^
    --agents %AGENTS% ^
    --num_samples %NUM_SAMPLES% ^
    --random_sample ^
    --output_dir "%OUTPUT_DIR%" ^
    --tail_window %TAIL_WINDOW% ^
    --use_multi_scale ^
    --use_subset ^
    --seed 42

if errorlevel 1 (
    echo.
    echo [错误] 推断失败!
    pause
    exit /b 1
)

echo.
echo [?] 推断完成
echo.

REM ============================================================================
REM 第 2 步：生成可视化对比图
REM ============================================================================

echo [第 2/2 步] 生成可视化对比图...
echo.

python visualize_v3_new_inference.py ^
    --result_file "%OUTPUT_DIR%\predictions_agents_%AGENTS%_v4_enhanced.npz" ^
    --num_samples %NUM_SAMPLES% ^
    --output_dir "%VIS_DIR%" ^
    --tail_window %TAIL_WINDOW%

if errorlevel 1 (
    echo.
    echo [警告] 可视化生成失败，但推断结果已保存
    pause
    exit /b 0
)

echo.
echo [?] 可视化完成
echo.

REM ============================================================================
REM 完成
REM ============================================================================

echo.
echo ============================================================================
echo ? 完整流程已完成!
echo ============================================================================
echo.
echo 输出目录：
echo   推断结果: %OUTPUT_DIR%
echo   可视化图: %VIS_DIR%
echo.
echo 推断结果文件：
echo   - %OUTPUT_DIR%\predictions_agents_%AGENTS%_v4_enhanced.npz
echo   - %OUTPUT_DIR%\evaluation_report_agents_%AGENTS%_v4_enhanced.txt
echo.
echo 可视化文件：
echo   - PNG 图表: %VIS_DIR%\prediction_sample_*.png
echo   - 评估报告: %VIS_DIR%\visualization_report_*.json
echo.
echo 下一步：
echo   1. 查看 PNG 图表了解预测质量
echo   2. 检查 evaluation_report_*.txt 了解详细指标
echo   3. 对比末端动力学特征（第6个子图）
echo   4. 查看行为分类和增强效果
echo.

pause
