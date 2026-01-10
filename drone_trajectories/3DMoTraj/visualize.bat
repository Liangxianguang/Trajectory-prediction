@echo off
set "PYTHON_PATH=python"
set "SCRIPT_PATH=%~dp0tool\visualize_lbebm3d.py"

echo ========================================
echo LBEBM3D 轨迹预测可视化
echo ========================================
echo.

%PYTHON_PATH% "%SCRIPT_PATH%" ^
    --model saved_models/lbebm3D_scene1.pt ^
    --dataset_folder dataset ^
    --dataset_name swarm ^
    --output_dir validation_results_lbebm3d ^
    --obs 20 ^
    --preds 10 ^
    --num_samples 50 ^
    --batch_size 128 ^
    --device 0 ^
    --interactive

echo.
echo ========================================
echo 可视化完成！
echo 结果保存在: validation_results_lbebm3d
echo ========================================
pause
