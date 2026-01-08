@echo off
REM 快速评估脚本

chcp 65001 > nul
setlocal enabledelayedexpansion

echo.
echo ======================================================================
echo 增强版集群轨迹预测模型 - 快速评估
echo ======================================================================
echo.

REM 评估 3 架无人机模型
echo 【评估】3 架无人机模型
echo 样本数: 100，批次大小: 32
echo.

python infer_swarm_model.py ^
    --model newdata1_swarm_models_enhanced/best_model_agents_3.pt ^
    --agents 3 ^
    --num_samples 100 ^
    --batch_size 32 ^
    --output_dir inference_results

if errorlevel 1 (
    echo.
    echo [错误] 评估失败！
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo ✓ 评估完成！
echo ======================================================================
echo.
echo 结果已保存到: inference_results/
echo.
pause
