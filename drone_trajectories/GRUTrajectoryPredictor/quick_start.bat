@echo off
REM Quick start script for Swarm GRU Trajectory Predictor
REM =====================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  Swarm GRU Trajectory Predictor - Quick Start
echo ============================================================
echo.

REM Get Python executable
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON=%%i

if "%PYTHON%"=="" (
    echo ERROR: Python not found
    exit /b 1
)

echo Using Python: %PYTHON%
echo.

REM Training
echo [1/2] Starting training...
echo.
%PYTHON% train_swarm_gru.py --num_agents 3 --epochs 50 --batch_size 64 --use_subset --early_stopping_patience 15
if errorlevel 1 (
    echo ERROR: Training failed
    exit /b 1
)

echo.
echo [2/2] Starting inference...
echo.
%PYTHON% predict_swarm_gru.py --num_agents 3 --model_path Models/swarm_gru_agents_3_best.pth --use_subset --visualize --save_results

echo.
echo ============================================================
echo  Quick Start Completed!
echo ============================================================
echo.
echo Results saved to:
echo   - Models/swarm_gru_agents_3_best.pth (trained model)
echo   - Results/ (metrics, plots, predictions)
echo.
pause
