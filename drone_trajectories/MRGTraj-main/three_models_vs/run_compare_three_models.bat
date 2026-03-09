@echo off
REM ====================================================
REM 三模型对比脚本
REM 3DMoTraj (LBEBM3D) vs DG32-BCAT (Exp5) vs MRGTraj-LBEBM3D
REM ====================================================

setlocal enabledelayedexpansion

REM === 数据路径 ===
set DATA_DIR=D:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments
set FEATURES_32D_DIR=D:\Trajectory prediction\drone_trajectories\Cluster trajectory\features_32d
set OUTPUT_DIR=D:\Trajectory prediction\drone_trajectories\MRGTraj-main\three_models_vs\2knewimage_comparison_results_three_models

REM === 模型路径 ===
set LBEBM_MODEL=D:\Trajectory prediction\drone_trajectories\3DMoTraj\saved_models\checkpoints_accfix\epoch_020.pt
set EXP5_DIR=D:\Trajectory prediction\drone_trajectories\Cluster trajectory\ablation study\ablation_results_agents_3_exp5_full
set MRGRAJ_MODEL=D:\Trajectory prediction\drone_trajectories\MRGTraj-main\checkpoints_lbebm3d\agents_3_lbebm3d_inspired\best_model.pth

REM === LBEBM3D 参数 ===
set DATA_SCALE=1.0
set E_INIT_SIG=2.0
set E_PRIOR_SIG=2.0
set E_L_STEPS=20
set E_L_STEP_SIZE=0.4

REM === 样本选择 ===
REM 选项1: 指定索引
REM set SAMPLE_INDICES=100,500,1000,2000,5000

REM 选项2: 随机采样
set NUM_SAMPLES=2000
set SEED=42

REM === 物理约束参数 ===
REM Exp5 物理约束
set ENABLE_EXP5_PHYSICS=1
set PC_DT=0.1
set PC_SMOOTHING_WEIGHT=0.3
set PC_CONSTRAINT_RELAXATION=1.0

REM MRGTraj 物理约束
set ENABLE_MRGRAJ_PHYSICS=1
set MRGRAJ_PC_DT=0.1
set MRGRAJ_PC_SMOOTHING_WEIGHT=0.1
set MRGRAJ_PC_CONSTRAINT_RELAXATION=0.6

REM === 验证集划分 ===
set USE_VAL_SPLIT=1
set VAL_SPLIT=0.2

REM === 可视化 ===
REM 1 = 生成每个样本的图表，0 = 跳过以加快速度
set ENABLE_VISUALIZE=1

REM ====================================================

echo ============================================
echo 三模型对比: LBEBM3D vs Exp5 vs MRGTraj
echo ============================================
echo.
echo 数据目录: %DATA_DIR%
echo LBEBM3D 模型: %LBEBM_MODEL%
echo Exp5 目录: %EXP5_DIR%
echo MRGTraj 模型: %MRGRAJ_MODEL%
echo 输出目录: %OUTPUT_DIR%
echo.

REM 检查模型文件
if not exist "%LBEBM_MODEL%" (
    echo [ERROR] LBEBM3D 模型未找到: %LBEBM_MODEL%
    pause
    exit /b 1
)

if not exist "%EXP5_DIR%\best_model_agents_3_exp5_full.pt" (
    echo [ERROR] Exp5 模型未找到: %EXP5_DIR%
    pause
    exit /b 1
)

if not exist "%MRGRAJ_MODEL%" (
    echo [ERROR] MRGTraj 模型未找到: %MRGRAJ_MODEL%
    pause
    exit /b 1
)

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 构建命令
set CMD=python compare_three_models.py ^
    --data_dir "%DATA_DIR%" ^
    --agents 3 ^
    --use_subset ^
    --lbebm_model "%LBEBM_MODEL%" ^
    --exp5_dir "%EXP5_DIR%" ^
    --mrgraj_model "%MRGRAJ_MODEL%" ^
    --features_32d_dir "%FEATURES_32D_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --data_scale %DATA_SCALE% ^
    --e_init_sig %E_INIT_SIG% ^
    --e_prior_sig %E_PRIOR_SIG% ^
    --e_l_steps %E_L_STEPS% ^
    --e_l_step_size %E_L_STEP_SIZE% ^
    --seed %SEED%

REM 添加样本选择标志
if defined SAMPLE_INDICES (
    set CMD=!CMD! --sample_indices "%SAMPLE_INDICES%"
) else (
    set CMD=!CMD! --num_samples %NUM_SAMPLES%
)

REM 添加验证集划分标志
if "%USE_VAL_SPLIT%"=="1" (
    set CMD=!CMD! --use_val_split --val_split %VAL_SPLIT%
)

REM 添加物理约束标志
if "%ENABLE_EXP5_PHYSICS%"=="0" (
    set CMD=!CMD! --no_physical_constraints
)
set CMD=!CMD! --pc_dt %PC_DT% --pc_smoothing_weight %PC_SMOOTHING_WEIGHT% --pc_constraint_relaxation %PC_CONSTRAINT_RELAXATION%

if "%ENABLE_MRGRAJ_PHYSICS%"=="0" (
    set CMD=!CMD! --no_mrgraj_physical_constraints
)
set CMD=!CMD! --mrgraj_pc_dt %MRGRAJ_PC_DT% --mrgraj_pc_smoothing_weight %MRGRAJ_PC_SMOOTHING_WEIGHT% --mrgraj_pc_constraint_relaxation %MRGRAJ_PC_CONSTRAINT_RELAXATION%

REM 添加可视化标志
if "%ENABLE_VISUALIZE%"=="0" (
    set CMD=!CMD! --no_visualize
)

echo 运行对比...
echo 命令: !CMD!
echo.

REM 执行命令
!CMD!

set EXIT_CODE=%ERRORLEVEL%

echo.
if !EXIT_CODE! equ 0 (
    echo ============================================
    echo 对比完成！
    echo 结果保存至: %OUTPUT_DIR%
    echo ============================================
    echo.
    echo 输出文件:
    echo   - comparison_summary.json (汇总数据)
    echo   - sample_*_comparison.png  (可视化图表)
) else (
    echo ============================================
    echo [ERROR] 对比失败，退出码: !EXIT_CODE!
    echo ============================================
)

pause
exit /b !EXIT_CODE!
