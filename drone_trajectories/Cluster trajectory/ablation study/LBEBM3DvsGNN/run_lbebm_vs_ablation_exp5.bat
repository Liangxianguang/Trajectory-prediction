@echo off
REM ====================================================
REM LBEBM3D vs Ablation Exp5 (DG32-BCAT) Comparison Script
REM ====================================================

setlocal enabledelayedexpansion

REM === Data Paths ===
set DATA_DIR=..\..\swarm_segments
set FEATURES_32D_DIR=..\..\features_32d
set OUTPUT_DIR=comparison_results_exp5_vs_lbebm_2000new1_allimages

REM === Model Paths ===
set LBEBM_MODEL=..\..\..\3DMoTraj\saved_models\checkpoints_accfix\epoch_010.pt
set EXP5_DIR=..\ablation_results_agents_3_exp5_full

REM === LBEBM Parameters ===
set DATA_SCALE=1.0
set E_INIT_SIG=2.0
set E_PRIOR_SIG=2.0
set E_L_STEPS=20
set E_L_STEP_SIZE=0.4

REM === Sample Selection ===
REM Option 1: Specify indices
REM set SAMPLE_INDICES=2504,17995,33018

REM Option 2: Random sampling
set NUM_SAMPLES=20
set SEED=45

REM === Validation Split Sampling (match infer_and_visualize_ablation.py) ===
set USE_VAL_SPLIT=1
set VAL_SPLIT=0.2

REM === Visualization ===
REM 1 = generate per-sample figures, 0 = skip for faster runs
set ENABLE_VISUALIZE=1

REM === Outlier Filter Parameters ===
REM Remove the worst X%% samples according to Exp5 metric (MAE or FDE)
set REMOVE_EXP5_OUTLIERS=1
set EXP5_OUTLIER_METRIC=MAE
set EXP5_OUTLIER_PERCENT=0

REM ====================================================

echo ============================================
echo LBEBM3D vs Ablation Exp5 (DG32-BCAT)
echo ============================================
echo.
echo Data Directory: %DATA_DIR%
echo LBEBM Model: %LBEBM_MODEL%
echo Exp5 Dir: %EXP5_DIR%
echo Output Directory: %OUTPUT_DIR%
echo.

REM Check if models exist
if not exist "%LBEBM_MODEL%" (
    echo [ERROR] LBEBM model not found: %LBEBM_MODEL%
    echo Please ensure the model path is correct.
    pause
    exit /b 1
)

if not exist "%EXP5_DIR%\best_model_agents_3_exp5_full.pt" (
    echo [ERROR] Exp5 model not found in: %EXP5_DIR%
    echo Please ensure the ablation results directory is correct.
    pause
    exit /b 1
)

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Construct command
set CMD=python compare_lbebm_vs_ablation_exp5.py ^
    --data_dir "%DATA_DIR%" ^
    --agents 3 ^
    --use_subset ^
    --lbebm_model "%LBEBM_MODEL%" ^
    --exp5_dir "%EXP5_DIR%" ^
    --features_32d_dir "%FEATURES_32D_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --data_scale %DATA_SCALE% ^
    --e_init_sig %E_INIT_SIG% ^
    --e_prior_sig %E_PRIOR_SIG% ^
    --e_l_steps %E_L_STEPS% ^
    --e_l_step_size %E_L_STEP_SIZE% ^
    --seed %SEED%

REM Add sample selection flags
if defined SAMPLE_INDICES (
    set CMD=!CMD! --sample_indices "%SAMPLE_INDICES%"
) else (
    set CMD=!CMD! --num_samples %NUM_SAMPLES%
)

REM Add validation split flags
if "%USE_VAL_SPLIT%"=="1" (
    set CMD=!CMD! --use_val_split --val_split %VAL_SPLIT%
)

REM Add outlier filtering flags
if "%REMOVE_EXP5_OUTLIERS%"=="1" (
    set CMD=!CMD! --remove_exp5_outliers --exp5_outlier_metric %EXP5_OUTLIER_METRIC% --exp5_outlier_percent %EXP5_OUTLIER_PERCENT%
)

REM Add visualization flag
if "%ENABLE_VISUALIZE%"=="0" (
    set CMD=!CMD! --no_visualize
)

echo Running comparison...
echo Command: !CMD!
echo.

REM Execute command
!CMD!

set EXIT_CODE=%ERRORLEVEL%

echo.
if !EXIT_CODE! equ 0 (
    echo ============================================
    echo Comparison completed successfully!
    echo Results saved to: %OUTPUT_DIR%
    echo ============================================
) else (
    echo ============================================
    echo [ERROR] Comparison failed with exit code !EXIT_CODE!
    echo ============================================
)

pause
exit /b !EXIT_CODE!
