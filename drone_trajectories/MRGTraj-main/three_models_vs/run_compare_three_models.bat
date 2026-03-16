@echo off
REM ====================================================
REM Three-model comparison script
REM 3DMoTraj (LBEBM3D) vs DG32-BCAT (Exp5) vs MRGTraj-LBEBM3D
REM ====================================================

setlocal enabledelayedexpansion

REM === Data paths ===
set DATA_DIR=D:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments
set FEATURES_32D_DIR=D:\Trajectory prediction\drone_trajectories\Cluster trajectory\features_32d
set OUTPUT_DIR=D:\Trajectory prediction\drone_trajectories\MRGTraj-main\three_models_vs\2knewimage_comparison_results_three_models

REM === Model paths ===
set LBEBM_MODEL=D:\Trajectory prediction\drone_trajectories\3DMoTraj\saved_models\checkpoints_accfix\epoch_020.pt
set EXP5_DIR=D:\Trajectory prediction\drone_trajectories\Cluster trajectory\ablation study\ablation_results_agents_3_exp5_full
set MRGRAJ_MODEL=D:\Trajectory prediction\drone_trajectories\MRGTraj-main\checkpoints_lbebm3d\agents_3_lbebm3d_inspired\best_model.pth

REM === LBEBM3D parameters ===
set DATA_SCALE=1.0
set E_INIT_SIG=2.0
set E_PRIOR_SIG=2.0
set E_L_STEPS=20
set E_L_STEP_SIZE=0.4

REM === Sampling options ===
REM Option 1: specify sample indices
REM set SAMPLE_INDICES=100,500,1000,2000,5000

REM Option 2: random samples
set NUM_SAMPLES=2000
set SEED=42

REM === Physical constraints ===
REM Exp5 constraints
set ENABLE_EXP5_PHYSICS=1
set PC_DT=0.1
set PC_SMOOTHING_WEIGHT=0.3
set PC_CONSTRAINT_RELAXATION=1.0

REM MRGTraj constraints
set ENABLE_MRGRAJ_PHYSICS=1
set MRGRAJ_PC_DT=0.1
set MRGRAJ_PC_SMOOTHING_WEIGHT=0.1
set MRGRAJ_PC_CONSTRAINT_RELAXATION=0.6

REM === Validation split ===
set USE_VAL_SPLIT=1
set VAL_SPLIT=0.2

REM === Visualization ===
REM 1 = enable per-sample plots, 0 = disable for speed
set ENABLE_VISUALIZE=1

REM ====================================================

echo ============================================
echo Three-model comparison: LBEBM3D vs Exp5 vs MRGTraj
echo ============================================
echo.
echo Data dir: %DATA_DIR%
echo LBEBM3D model: %LBEBM_MODEL%
echo Exp5 dir: %EXP5_DIR%
echo MRGTraj model: %MRGRAJ_MODEL%
echo Output dir: %OUTPUT_DIR%
echo.

REM Validate model files
if not exist "%LBEBM_MODEL%" (
    echo [ERROR] LBEBM3D model not found: %LBEBM_MODEL%
    pause
    exit /b 1
)

if not exist "%EXP5_DIR%\best_model_agents_3_exp5_full.pt" (
    echo [ERROR] Exp5 model not found: %EXP5_DIR%
    pause
    exit /b 1
)

if not exist "%MRGRAJ_MODEL%" (
    echo [ERROR] MRGTraj model not found: %MRGRAJ_MODEL%
    pause
    exit /b 1
)

REM Create output dir
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Build command
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

REM Sample selection flags
if defined SAMPLE_INDICES (
    set CMD=!CMD! --sample_indices "%SAMPLE_INDICES%"
) else (
    set CMD=!CMD! --num_samples %NUM_SAMPLES%
)

REM Validation flags
if "%USE_VAL_SPLIT%"=="1" (
    set CMD=!CMD! --use_val_split --val_split %VAL_SPLIT%
)

REM Physical constraint flags
if "%ENABLE_EXP5_PHYSICS%"=="0" (
    set CMD=!CMD! --no_physical_constraints
)
set CMD=!CMD! --pc_dt %PC_DT% --pc_smoothing_weight %PC_SMOOTHING_WEIGHT% --pc_constraint_relaxation %PC_CONSTRAINT_RELAXATION%

if "%ENABLE_MRGRAJ_PHYSICS%"=="0" (
    set CMD=!CMD! --no_mrgraj_physical_constraints
)
set CMD=!CMD! --mrgraj_pc_dt %MRGRAJ_PC_DT% --mrgraj_pc_smoothing_weight %MRGRAJ_PC_SMOOTHING_WEIGHT% --mrgraj_pc_constraint_relaxation %MRGRAJ_PC_CONSTRAINT_RELAXATION%

REM Visualization flag
if "%ENABLE_VISUALIZE%"=="0" (
    set CMD=!CMD! --no_visualize
)

echo Running comparison...
echo Command: !CMD!
echo.

REM Execute
!CMD!

set EXIT_CODE=%ERRORLEVEL%

echo.
if !EXIT_CODE! equ 0 (
    echo ============================================
    echo Comparison completed.
    echo Results saved to: %OUTPUT_DIR%
    echo ============================================
    echo.
    echo Output files:
    echo   - comparison_summary.json (summary metrics)
    echo   - sample_*_comparison.png  (visualizations)
) else (
    echo ============================================
    echo [ERROR] Comparison failed. Exit code: !EXIT_CODE!
    echo ============================================
)

pause
exit /b !EXIT_CODE!
