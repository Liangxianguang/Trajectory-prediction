@echo off
REM 完整评估脚本：评估所有三个测试目录下的所有轨迹文件
REM 
REM 功能：
REM 1. 自动扫描 tool/ 下的所有模型 checkpoint
REM 2. 在三个测试目录中评估（CSV + TXT 格式支持）
REM 3. 生成综合对比报告
REM

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ================================================
echo 完整轨迹预测模型评估
echo ================================================
echo.
echo 评估范围：
echo   - 模型来源: drone_trajectories\tool (自动扫描)
echo   - 测试集1: ..\..\Synthetic-UAV-Flight-Trajectories (CSV)
echo   - 测试集2: ..\..\drone_trajectories\random_traj_100ms (TXT)
echo   - 测试集3: ..\..\drone_trajectories\new_random_traj_100ms (TXT)
echo.

REM 设置输出目录为当前日期时间命名
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set output_dir=evaluation_results_%mydate%_%mytime%

echo 输出目录: %output_dir%
echo.

REM 调用评估脚本，支持多个测试目录（逗号分隔）
python evaluate_all_models.py ^
  --auto_models ^
  --tool_dir "..\..\drone_trajectories\tool" ^
  --test_dir "..\..\Synthetic-UAV-Flight-Trajectories,..\..\drone_trajectories\random_traj_100ms,..\..\drone_trajectories\new_random_traj_100ms" ^
  --output_dir "%output_dir%" ^
  --method physics_constrained ^
  --device cuda

REM 检查结果
if exist "%output_dir%\models_comparison.csv" (
    echo.
    echo ================================================
    echo 评估完成！
    echo ================================================
    echo 结果已保存到: %output_dir%
    echo.
    echo 生成的文件:
    echo   - models_comparison.csv: 模型对比汇总表
    echo   - models_comparison.json: 模型对比 JSON 格式
    echo   - *_detailed_results.csv: 每个模型的详细结果
    echo.
    pause
) else (
    echo.
    echo 警告：评估可能失败，未找到结果文件
    echo 请检查日志输出
    pause
)
