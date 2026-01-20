@echo off
REM 交互模式：生成可旋转的 3D 图表
REM 这会显示一个交互窗口，可用鼠标拖拽旋转 3D 轨迹

echo ========================================
echo 集群轨迹可视化 - 交互模式
echo ========================================
echo.
echo 模式：生成 PNG + 交互式 3D 图表（可直接旋转）
echo 样本数：1 个
echo.
echo 操作方式：
echo   - 左键拖拽：旋转 3D 视图
echo   - 右键拖拽：缩放
echo   - 滚轮：放大/缩小
echo.

python visualize_swarm_prediction.py ^
    --model_path exp_balanced_v1/best_model_agents_3.pt ^
    --agents 3 ^
    --num_samples 200 ^
    --fast ^
    --output_dir visualization_results_exp_balanced_v1

echo.
echo ✓ 完成！
echo 输出目录：visualization_results_exp_balanced_v1
pause
