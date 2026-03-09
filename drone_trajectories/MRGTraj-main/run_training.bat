@echo off
REM Quick training script for MRGTraj Swarm model
REM Usage: run_training.bat [agents] [epochs]

setlocal enabledelayedexpansion

REM Default values
set AGENTS=3
set EPOCHS=100
set BATCH_SIZE=32

REM Parse command line arguments
if not "%1"=="" set AGENTS=%1
if not "%2"=="" set EPOCHS=%2

echo.
echo ============================================================
echo MRGTraj Swarm Training - Quick Launcher
echo ============================================================
echo Configuration:
echo   - Number of agents: %AGENTS%
echo   - Number of epochs: %EPOCHS%
echo   - Batch size: %BATCH_SIZE%
echo ============================================================
echo.

python train_swarm_detailed.py ^
  --num_agents %AGENTS% ^
  --batch_size %BATCH_SIZE% ^
  --num_epochs %EPOCHS% ^
  --collision_weight 0.5 ^
  --formation_weight 0.2 ^
  --kl_weight 0.1

echo.
echo Training completed! Check checkpoints_swarm100/agents_%AGENTS%/ for results
echo.
pause
