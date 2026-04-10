@echo off
REM ====================================================
REM 生成 4×4 场景对比图的完整流程
REM 
REM 步骤:
REM   1. 首先运行 compare_four_models_image.py 生成单个样本的对比图和 JSON 结果
REM   2. 然后运行 generate_4x4_comparison.py 从 JSON 结果中提取 4 个特定场景
REM   3. 生成最终的 4×4 大布局图
REM ====================================================

setlocal enabledelayedexpansion

REM === 配置路径 ===
set DATA_DIR=D:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments
set FEATURES_32D_DIR=D:\Trajectory prediction\drone_trajectories\Cluster trajectory\features_32d
set OUTPUT_DIR_COMPARE=D:\Trajectory prediction\drone_trajectories\GRUTrajectoryPredictor\comparison_four_models
set OUTPUT_DIR_4X4=D:\Trajectory prediction\drone_trajectories\GRUTrajectoryPredictor\comparison_4x4_scenarios

REM === 模型路径 ===
set LBEBM_MODEL=D:\Trajectory prediction\drone_trajectories\3DMoTraj\saved_models\checkpoints_accfix\epoch_030.pt
set EXP5_DIR=D:\Trajectory prediction\drone_trajectories\Cluster trajectory\ablation study\ablation_results_agents_3_exp5_full
set MRGRAJ_MODEL=D:\Trajectory prediction\drone_trajectories\MRGTraj-main\checkpoints_lbebm3d\agents_3_lbebm3d_inspired\best_model.pth
set GRU_MODEL=D:\Trajectory prediction\drone_trajectories\GRUTrajectoryPredictor\checkpoints\agents_3_20260309_141203\epoch_190.pth

REM === 参数 ===
REM 这四个样本是关键场景
set SAMPLE_INDICES=20280,173142,212515,33

set SEED=42

echo.
echo ================================================================
echo  4×4 场景对比图生成流程
echo ================================================================
echo.
echo 关键场景定义:
echo   1. 样本 20280  - 复杂交互场景中的空间建模能力（轨迹交叉）
echo   2. 样本 173142 - 高曲率机动中的物理一致性（协同急转弯）
echo   3. 样本 212515 - 三维机动场景中的高度预测能力（垂直爬升）
echo   4. 样本 33     - 复杂周期机动中的时序建模能力（S形轨迹）
echo.
echo ================================================================

REM 验证模型文件
echo 验证模型文件...
if not exist "%LBEBM_MODEL%" (
    echo [ERROR] 3DMoTraj 模型不存在: %LBEBM_MODEL%
    pause
    exit /b 1
)

if not exist "%EXP5_DIR%\best_model_agents_3_exp5_full.pt" (
    echo [ERROR] VECTOR 模型不存在: %EXP5_DIR%
    pause
    exit /b 1
)

if not exist "%MRGRAJ_MODEL%" (
    echo [ERROR] MRGTraj 模型不存在: %MRGRAJ_MODEL%
    pause
    exit /b 1
)

if not exist "%GRU_MODEL%" (
    echo [ERROR] Ours 模型不存在: %GRU_MODEL%
    pause
    exit /b 1
)

echo [OK] 所有模型文件已验证
echo.

REM 创建输出目录
if not exist "%OUTPUT_DIR_COMPARE%" mkdir "%OUTPUT_DIR_COMPARE%"
if not exist "%OUTPUT_DIR_4X4%" mkdir "%OUTPUT_DIR_4X4%"

echo ================================================================
echo [步骤 1/2] 运行四模型对比脚本
echo           生成关键场景的预测结果和单个样本对比图
echo ================================================================
echo.

set CMD_COMPARE=python compare_four_models_image.py ^
    --data_dir "%DATA_DIR%" ^
    --agents 3 ^
    --use_subset ^
    --lbebm_model "%LBEBM_MODEL%" ^
    --exp5_dir "%EXP5_DIR%" ^
    --mrgraj_model "%MRGRAJ_MODEL%" ^
    --gru_model "%GRU_MODEL%" ^
    --features_32d_dir "%FEATURES_32D_DIR%" ^
    --output_dir "%OUTPUT_DIR_COMPARE%" ^
    --sample_indices "%SAMPLE_INDICES%" ^
    --seed %SEED% ^
    --no_visualize

echo 执行: !CMD_COMPARE!
echo.

!CMD_COMPARE!

if !ERRORLEVEL! neq 0 (
    echo [ERROR] 步骤 1 失败
    pause
    exit /b 1
)

echo.
echo [OK] 步骤 1 完成
echo     结果已保存到: %OUTPUT_DIR_COMPARE%
echo.

echo ================================================================
echo [步骤 2/2] 生成 4×4 场景对比图
echo           从 JSON 结果中提取 4 个关键场景，组合成大布局图
echo ================================================================
echo.

set JSON_RESULTS=%OUTPUT_DIR_COMPARE%\comparison_summary.json

if not exist "%JSON_RESULTS%" (
    echo [ERROR] JSON 结果文件不存在: %JSON_RESULTS%
    echo 请检查步骤 1 是否成功完成
    pause
    exit /b 1
)

set CMD_4X4=python generate_4x4_comparison.py ^
    --json_results "%JSON_RESULTS%" ^
    --output_dir "%OUTPUT_DIR_4X4%" ^
    --output_name "4x4_scenario_comparison.png"

echo 执行: !CMD_4X4!
echo.

!CMD_4X4!

if !ERRORLEVEL! neq 0 (
    echo [ERROR] 步骤 2 失败
    pause
    exit /b 1
)

echo.
echo [OK] 步骤 2 完成
echo.

echo ================================================================
echo  ? 完成!
echo ================================================================
echo.
echo 生成的文件:
echo   单个样本对比图:
echo     - %OUTPUT_DIR_COMPARE%\sample_*_comparison_publication.png
echo.
echo   4×4 场景对比图:
echo     - %OUTPUT_DIR_4X4%\4x4_scenario_comparison.png
echo.
echo 关键特性:
echo   ? 4 个行 = 4 个特定场景（各展示不同预测能力）
echo   ? 4 个列 = 4 个观测维度（3D + XY + XZ + YZ）
echo   ? 发布级质量 (150 DPI，清晰布局)
echo.
echo ================================================================

pause
exit /b 0
