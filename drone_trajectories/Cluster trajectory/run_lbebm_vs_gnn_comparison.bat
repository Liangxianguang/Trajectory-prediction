@echo off
REM ====================================================
REM LBEBM3D vs GNN+BiGRU Comparison Script
REM ====================================================

setlocal enabledelayedexpansion

REM === Data Paths ===
set DATA_DIR=swarm_segments
set FEATURES_32D_DIR=features_32d
set OUTPUT_DIR=comparison_results_1w_lbebm_vs_gnn

REM === Model Paths ===
set LBEBM_MODEL=..\3DMoTraj\saved_models\checkpoints_accfix\best.pt
set GNN_MODEL=gru_models_v4_fixed_agents_3_v4_fixed_gnn\best_model_agents_3_v4_fixed_gnn.pt

REM === LBEBM Parameters ===
set DATA_SCALE=1.0
set E_INIT_SIG=2.0
set E_PRIOR_SIG=2.0
set E_L_STEPS=20
set E_L_STEP_SIZE=0.4

REM === Sample Selection ===
REM Option 1: Specify indices
REM set SAMPLE_INDICES=2504,17995,33018

REM Option 2: Random sampling (2000 samples for full evaluation)
set NUM_SAMPLES=2000
set SEED=42

REM === GNN Parameters ===
set EDGE_THRESHOLD=5.0

REM ====================================================

echo ============================================
echo LBEBM3D vs GNN+BiGRU Comparison
echo ============================================
echo.
echo Data Directory: %DATA_DIR%
echo LBEBM Model: %LBEBM_MODEL%
echo GNN Model: %GNN_MODEL%
echo Output Directory: %OUTPUT_DIR%
echo.

REM Check if models exist
if not exist "%LBEBM_MODEL%" (
    echo [ERROR] LBEBM model not found: %LBEBM_MODEL%
    echo Please ensure the model path is correct.
    pause
    exit /b 1
)

if not exist "%GNN_MODEL%" (
    echo [ERROR] GNN model not found: %GNN_MODEL%
    echo Please ensure the model path is correct.
    pause
    exit /b 1
)

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Construct command
set CMD=python compare_lbebm_vs_gnn.py ^
    --data_dir "%DATA_DIR%" ^
    --agents 3 ^
    --use_subset ^
    --lbebm_model "%LBEBM_MODEL%" ^
    --gnn_model "%GNN_MODEL%" ^
    --data_scale %DATA_SCALE% ^
    --e_init_sig %E_INIT_SIG% ^
    --e_prior_sig %E_PRIOR_SIG% ^
    --e_l_steps %E_L_STEPS% ^
    --e_l_step_size %E_L_STEP_SIZE% ^
    --features_32d_dir "%FEATURES_32D_DIR%" ^
    --edge_threshold %EDGE_THRESHOLD% ^
    --output_dir "%OUTPUT_DIR%" ^
    --seed %SEED%

REM Add sample selection flags
if defined SAMPLE_INDICES (
    set CMD=!CMD! --sample_indices "%SAMPLE_INDICES%"
) else (
    set CMD=!CMD! --num_samples %NUM_SAMPLES%
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
