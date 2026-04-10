@echo off
REM Quick Start for Swarm GRU Trajectory Predictor
REM ============================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo  Swarm GRU Trajectory Predictor - Quick Start
echo ============================================================
echo.

REM Get Python
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo Python version: %PYTHON_VER%
echo.

REM Menu
echo Choose an option:
echo.
echo  1. Run Quick Test (REQUIRED - run first!)
echo  2. Train Model (quick - subset, ~5 min)
echo  3. Train Model (full - all data, ~1-2 hours)
echo  4. Inference (quick - subset)
echo  5. Inference (full - all data)
echo  6. Both (quick training + inference)
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Running quick test...
    echo.
    python test_quick.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Training on subset (10k samples)...
    echo.
    python train_swarm_gru_v2.py --num_agents 3 --epochs 50 --batch_size 64 --use_subset
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Training on full dataset...
    echo.
    python train_swarm_gru_v2.py --num_agents 3 --epochs 100 --batch_size 32
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Inference on subset (1000 samples)...
    echo.
    python predict_swarm_gru_v2.py --num_agents 3 --use_subset --visualize --save_results
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Inference on full dataset...
    echo.
    python predict_swarm_gru_v2.py --num_agents 3 --visualize --save_results
    goto end
)

if "%choice%"=="6" (
    echo.
    echo Quick training on subset...
    echo.
    python train_swarm_gru_v2.py --num_agents 3 --epochs 30 --batch_size 64 --use_subset
    
    if errorlevel 1 (
        echo Training failed!
        goto end
    )
    
    echo.
    echo Training completed. Now running inference...
    echo.
    python predict_swarm_gru_v2.py --num_agents 3 --use_subset --visualize --save_results
    goto end
)

echo Invalid choice!

:end
echo.
echo ============================================================
echo  Done!
echo ============================================================
echo.
pause
