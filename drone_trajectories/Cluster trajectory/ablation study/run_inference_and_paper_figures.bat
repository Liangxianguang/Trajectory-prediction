@echo off
REM No UTF-8 encoding for batch files - keep ASCII only

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo   Ablation Study - Complete Inference and Visualization Workflow
echo ============================================================================
echo.

set PYTHON=python
set DATA_DIR=..\swarm_segments
set ABLATION_DIR=.
set OUTPUT_DIR=ablation_results_final

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM ============================================================================
REM Step 1: Inference and Visualization (sample subset)
REM ============================================================================

echo [1/2] Running inference and visualization on 50 random samples...
echo.

%PYTHON% infer_and_visualize_ablation.py ^
    --data_dir "%DATA_DIR%" ^
    --ablation_dir "%ABLATION_DIR%" ^
    --output_dir "%OUTPUT_DIR%/3kNEWsamples_comparison" ^
    --num_samples 3000 ^
    --batch_size 256 ^
    --seed 45 ^
    --use_subset

if errorlevel 1 (
    echo.
    echo [FAILED] Inference and visualization step failed!
    pause
    exit /b 1
)

echo.
echo [OK] Inference and visualization complete! Generated 50 sample comparison figures.
echo.

REM ============================================================================
REM Step 2: Generate paper-ready summary figures
REM ============================================================================

echo [2/2] Generating paper-ready summary figures...
echo.

%PYTHON% generate_paper_figures.py ^
    --ablation_dir "%ABLATION_DIR%" ^
    --inference_results "%OUTPUT_DIR%/samples_comparison/summary.json" ^
    --output_dir "%OUTPUT_DIR%/paper_figures"

if errorlevel 1 (
    echo.
    echo [FAILED] Paper figures generation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo   [OK] Complete workflow executed successfully!
echo ============================================================================
echo.
echo Output directory structure:
echo   %OUTPUT_DIR%/
echo   +-- samples_comparison/          (50 sample comparison figures)
echo   |   +-- sample_000001_comparison.png
echo   |   +-- sample_000002_comparison.png
echo   |   +-- ... (50 total)
echo   |   +-- summary.json
echo   +-- paper_figures/               (publication-ready summary figures)
echo       +-- training_curves_comparison.png
echo       +-- best_metrics_summary.png
echo       +-- improvement_analysis.png
echo       +-- training_vs_inference.png
echo.
echo Next steps:
echo   1. View sample_*_comparison.png files to check model predictions
echo   2. Use paper_figures/* for thesis/presentation
echo   3. Modify parameters and re-run as needed
echo.

pause
