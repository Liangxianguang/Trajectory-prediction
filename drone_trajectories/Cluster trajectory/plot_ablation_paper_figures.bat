@echo off
REM ============================================================================
REM Generate paper-ready ablation figures (V1-V4)
REM ============================================================================
setlocal enabledelayedexpansion

set WORKSPACE=d:\Trajectory prediction
set REPORT_DIR=%WORKSPACE%\drone_trajectories\Cluster trajectory\comparison_figures_allmodels
set OUTPUT_DIR=%REPORT_DIR%

echo [INFO] Output dir: %OUTPUT_DIR%

cd /d "%WORKSPACE%\drone_trajectories\Cluster trajectory"

REM If you want to force a specific report file, set REPORT_JSON here:
set REPORT_JSON=%REPORT_DIR%\comparison_report_20260119_182931.json

if defined REPORT_JSON (
    python plot_ablation_paper_figures.py --output_dir "%OUTPUT_DIR%" --order "V1,V2,V3,V4" --report_json "%REPORT_JSON%"
) else (
    python plot_ablation_paper_figures.py --output_dir "%OUTPUT_DIR%" --order "V1,V2,V3,V4" --reports_dir "%REPORT_DIR%"
)

if errorlevel 1 (
    echo [ERROR] Failed to generate figures.
    pause
    exit /b 1
)

echo [OK] Done.
pause

