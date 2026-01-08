@echo off
REM 快速测试可视化脚本（只生成 PNG，无需交互窗口）
REM 推荐用这个命令进行快速检验

echo ========================================
echo 集群轨迹可视化 - 快速模式
echo ========================================
echo.
echo 模式：PNG 只输出（不显示交互窗口）
echo 样本数：1 个
echo 预计时间：5-10 秒
echo.

python visualize_swarm_prediction.py ^
    --model_path newdata1_swarm_models_enhanced/best_model_agents_3.pt ^
    --agents 3 ^
    --num_samples 1 ^
    --fast ^
    --output_dir visualization_results_fast

echo.
echo ✓ 完成！
echo 输出目录：visualization_results_fast
pause
