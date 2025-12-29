@echo off
REM 快速测试脚本：用小样本验证评估流程
REM 
REM 用途：快速检查评估脚本是否正常工作
REM 评估 5 个样本，预期耗时 1-2 分钟
REM

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ================================================
echo 快速测试：轨迹预测模型评估（小样本）
echo ================================================
echo.
echo 这将仅评估 5 个样本文件，用于验证脚本功能
echo.

python evaluate_all_models.py ^
  --auto_models ^
  --tool_dir "..\..\drone_trajectories\tool" ^
  --test_dir "..\..\Synthetic-UAV-Flight-Trajectories" ^
  --output_dir "evaluation_results_quick_test" ^
  --method physics_constrained ^
  --max_samples 5 ^
  --device cuda

echo.
echo ================================================
echo 快速测试完成
echo ================================================
echo 结果已保存到: evaluation_results_quick_test
echo.
pause
