@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Run full ablation experiments (Exp1~Exp5) on Windows
REM Location: drone_trajectories\Cluster trajectory\ablation study\
REM
REM Usage:
REM   - Double click this .bat, or run in cmd:
REM       run_all_ablation_experiments.bat
REM
REM Notes:
REM   - This will run experiments sequentially.
REM   - Logs are saved under: ablation_logs\YYYYMMDD_HHMMSS\
REM   - If any experiment fails (errorlevel != 0), script stops.
REM ============================================================

REM ---- Console encoding (avoid GBK issues in some terminals) ----
chcp 65001 >nul

REM ---- Force Python UTF-8 I/O (important when redirecting output to log files) ----
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ---- [EDIT HERE] Global settings ----
set AGENTS=3
set EPOCHS=150
set USE_SUBSET=1

REM Device: cuda:0 or cpu
set DEVICE=cuda:0

REM Resume behavior:
REM   RESUME=1 -> default (auto resume from last_checkpoint_*.pt if exists)
REM   RESUME=0 -> pass --no_resume to each script
set RESUME=1

REM Optional: override from command line:
REM   run_all_ablation_experiments.bat [EPOCHS] [DEVICE] [USE_SUBSET] [AGENTS] [RESUME]
if not "%~1"=="" set EPOCHS=%~1
if not "%~2"=="" set DEVICE=%~2
if not "%~3"=="" set USE_SUBSET=%~3
if not "%~4"=="" set AGENTS=%~4
if not "%~5"=="" set RESUME=%~5

REM Batch sizes (GNN experiments are much heavier)
set BS_EXP1=512
set BS_EXP2=512
set BS_EXP3=128
set BS_EXP4=128
set BS_EXP5=128

REM Feature directories (relative to this .bat directory)
set FEAT_16D=..\swarm_features
set FEAT_32D=..\features_32d

REM Optional: reduce fragmentation / help OOM in some cases
REM (PYTORCH_CUDA_ALLOC_CONF is deprecated; use PYTORCH_ALLOC_CONF)
set PYTORCH_ALLOC_CONF=expandable_segments:True

REM ---- Resolve paths ----
set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%"

echo ======================================================================
echo Full ablation run
echo   Dir     : %ROOT_DIR%
echo   Agents  : %AGENTS%
echo   Epochs  : %EPOCHS%
echo   Device  : %DEVICE%
echo   Subset  : %USE_SUBSET%
echo   Resume  : %RESUME%
echo ======================================================================
echo.

REM ---- Helper: build subset flag ----
set SUBSET_FLAG=
if "%USE_SUBSET%"=="1" set SUBSET_FLAG=--use_subset

REM ---- Helper: build resume flag ----
set RESUME_FLAG=
if "%RESUME%"=="0" set RESUME_FLAG=--no_resume

REM ---- Helper: run a single experiment and tee logs ----
REM We use cmd redirection to log file. (No interactive tee in plain cmd)
REM Each experiment has its own log file.

call :run_exp "exp1_baseline" "train_ablation_exp1_baseline.py" "%BS_EXP1%" "%FEAT_16D%"
call :run_exp "exp2_feat_bigru" "train_ablation_exp2_feat_bigru.py" "%BS_EXP2%" "%FEAT_32D%"
call :run_exp "exp3_gnn_bigru" "train_ablation_exp3_gnn_bigru.py" "%BS_EXP3%" "%FEAT_16D%"
call :run_exp "exp4_gnn_feat"  "train_ablation_exp4_gnn_feat.py"  "%BS_EXP4%" "%FEAT_32D%"
call :run_exp "exp5_full"      "train_ablation_exp5_full.py"      "%BS_EXP5%" "%FEAT_32D%"

echo.
echo ======================================================================
echo [SUCCESS] All ablation experiments finished.
echo ======================================================================
exit /b 0


:run_exp
set EXP_NAME=%~1
set SCRIPT=%~2
set BS=%~3
set FEAT_DIR=%~4

echo ----------------------------------------------------------------------
echo Running %EXP_NAME%
echo   Script    : %SCRIPT%
echo   BatchSize : %BS%
echo   Features  : %FEAT_DIR%
echo ----------------------------------------------------------------------

REM Run (real-time console output; no log files)
python "%SCRIPT%" --agents %AGENTS% --epochs %EPOCHS% --batch_size %BS% %SUBSET_FLAG% %RESUME_FLAG% --features_dir "%FEAT_DIR%" --device %DEVICE%
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
  echo.
  echo [FAILED] %EXP_NAME% failed with errorlevel=%RC%
  exit /b %RC%
)

echo [OK] %EXP_NAME% finished.
echo.
exit /b 0

