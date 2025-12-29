@echo off
REM 统一评估脚本启动

cd /d "D:\Trajectory prediction\drone_trajectories\evaluate"

echo ====================================================
echo 统一模型评估脚本
echo ====================================================

REM 检查测试集目录
if not exist "..\..\Synthetic-UAV-Flight-Trajectories" (
    echo 错误: 找不到测试集目录 ..\..\Synthetic-UAV-Flight-Trajectories
    pause
    exit /b 1
)

echo.
echo 开始评估...
echo.

python evaluate_all_models.py ^
  --models eval_config_example.json ^
  --test_dir ..\..\Synthetic-UAV-Flight-Trajectories ^
  --output_dir evaluation_results ^
  --input_length 20 ^
  --method physics_constrained ^
  --device cuda

echo.
echo ====================================================
echo 评估完成! 结果已保存到 evaluation_results 目录
echo ====================================================
pause
