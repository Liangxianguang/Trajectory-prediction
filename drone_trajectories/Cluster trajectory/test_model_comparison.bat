@echo off
REM ============================================================================
REM Multi-Model Comparison Visualization Script
REM ============================================================================
REM
REM Compare predictions from v2, v3, v4 models on the same samples
REM Perfect for paper figures
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM ============================================================================
REM Configuration
REM ============================================================================

set WORKSPACE=d:\Trajectory prediction
set DATA_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\swarm_segments
set FEATURES_24D_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\features_24d
set FEATURES_32D_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\features_32d
set OUTPUT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\comparison_figures_allmodels
set AGENTS=3

REM Model paths
set V1_MODEL=%WORKSPACE%\drone_trajectories\Cluster trajectory\exp_balanced_v1\best_model_agents_3.pt
set V2_MODEL=%WORKSPACE%\drone_trajectories\Cluster trajectory\24dmodel\swarm_segments_subset_feature\best_model_agents_3_v2.pt
set V3_MODEL=%WORKSPACE%\drone_trajectories\Cluster trajectory\gru_models_v3_agents_3_v3_gnn_concat\last_checkpoint_0199.pt
set V4_MODEL=%WORKSPACE%\drone_trajectories\Cluster trajectory\gru_models_v4_fixed_agents_3_v4_fixed_gnn\best_model_agents_3_v4_fixed_gnn.pt

REM Random sampling configuration
REM If you want to specify exact samples, uncomment SAMPLE_INDICES and comment out NUM_SAMPLES
REM set SAMPLE_INDICES=2504,17995,33018
set NUM_SAMPLES=2000
set SEED=42

REM ============================================================================
REM Validation
REM ============================================================================

echo.
echo ============================================================================
echo Multi-Model Comparison Visualization
echo ============================================================================
echo.

REM Check data directory
if not exist "%DATA_DIR%" (
    echo [ERROR] Data directory does not exist: %DATA_DIR%
    pause
    exit /b 1
)
echo [?] Data directory found

REM Check feature directories
if not exist "%FEATURES_24D_DIR%" (
    echo [WARNING] 24D feature directory does not exist: %FEATURES_24D_DIR%
    echo v2 and v3 models require 24D features
) else (
    echo [?] 24D feature directory found
)

if not exist "%FEATURES_32D_DIR%" (
    echo [WARNING] 32D feature directory does not exist: %FEATURES_32D_DIR%
    echo v4 model requires 32D features
) else (
    echo [?] 32D feature directory found
)

REM Check model files
if not exist "%V1_MODEL%" (
    echo [WARNING] v1 model file does not exist: %V1_MODEL%
    echo v1 comparison will be skipped
) else (
    echo [?] v1 model file found
)

if not exist "%V2_MODEL%" (
    echo [WARNING] v2 model file does not exist: %V2_MODEL%
    echo v2 comparison will be skipped
) else (
    echo [?] v2 model file found
)

if not exist "%V3_MODEL%" (
    echo [WARNING] v3 model file does not exist: %V3_MODEL%
    echo v3 comparison will be skipped
) else (
    echo [?] v3 model file found
)

if not exist "%V4_MODEL%" (
    echo [ERROR] v4 model file does not exist: %V4_MODEL%
    pause
    exit /b 1
)
echo [?] v4 model file found

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
echo [?] Output directory created

REM ============================================================================
REM Run Comparison Visualization
REM ============================================================================

echo.
echo [Execute] Generating multi-model comparison figures...
echo.

cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory"

python visualize_model_comparison.py ^
    --data_dir "%DATA_DIR%" ^
    --features_24d_dir "%FEATURES_24D_DIR%" ^
    --features_32d_dir "%FEATURES_32D_DIR%" ^
    --agents %AGENTS% ^
    --num_samples %NUM_SAMPLES% ^
    --seed %SEED% ^
    --output_dir "%OUTPUT_DIR%" ^
    --v1_model "%V1_MODEL%" ^
    --v2_model "%V2_MODEL%" ^
    --v3_model "%V3_MODEL%" ^
    --v4_model "%V4_MODEL%" ^
    --use_subset ^
    --edge_threshold 5.0

if errorlevel 1 (
    echo.
    echo [ERROR] Comparison visualization failed!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo ? Comparison visualization complete!
echo ============================================================================
echo.
echo Output directory: %OUTPUT_DIR%
echo.
echo Generated files:
echo   - PNG figures: comparison_sample_*.png
echo   - Evaluation report: comparison_report_*.json
echo.
echo Random sampling: %NUM_SAMPLES% samples (seed=%SEED%)
echo.
echo Next steps:
echo   1. View PNG figures to understand model prediction comparison
echo   2. Check JSON report for detailed metrics
echo   3. Use these figures for your paper
echo.

pause
