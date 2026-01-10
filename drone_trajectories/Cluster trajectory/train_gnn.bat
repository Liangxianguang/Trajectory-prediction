@echo off
REM 快速启动 GNN 训练脚本
REM ========================

echo.
echo 【集群轨迹预测 - GNN改进模型】
echo ================================
echo.
echo 本脚本将使用动态图神经网络训练集群轨迹预测模型
echo.
echo 关键改进：
echo   ✓ 动态图建模 (捕捉集群内UAV交互)
echo   ✓ GCN + BiGRU 时空融合
echo   ✓ 多指标评估 (MAE, RMSE, MAPE, MaxErr)
echo   ✓ 智能早停 (patience=15, 防止过拟合)
echo.
echo 预期效果：
echo   ✗ 旧模型: val_loss在Epoch 11后停滞，MAE≈0.089m
echo   ✓ 新模型: val_loss持续下降至Epoch 100+，MAE预计<0.080m
echo.
echo ================================
echo.

REM 检查数据文件
if not exist "swarm_segments\input_agents_3.npz" (
    echo 错误: 找不到数据文件 swarm_segments/input_agents_3.npz
    echo 请确保已运行数据预处理步骤
    exit /b 1
)

REM 创建输出目录
if not exist "gru_models_gnn" mkdir gru_models_gnn

echo [1/3] 启动 GNN 训练...
python train_swarm_gnn.py ^
    --agents 3 ^
    --epochs 300 ^
    --batch_size 256 ^
    --hidden_size 128 ^
    --num_layers 2 ^
    --dropout 0.2 ^
    --lr 1e-3 ^
    --weight_decay 1e-4 ^
    --patience 25 ^
    --output_dir gru_models_gnn ^
    --seed 42

if %ERRORLEVEL% NEQ 0 (
    echo 错误: GNN 训练失败
    exit /b 1
)

echo.
echo [2/3] 训练完成！模型已保存到 gru_models_gnn/
echo.
echo [3/3] 可选：运行诊断分析
echo  python analyze_per_agent_predictions.py --predictions inference_results/predictions_agents_3.npz
echo.
echo ================================
echo ✓ GNN 训练完成！
echo.
echo 下一步：
echo   1. 查看 gru_models_gnn/training_history_agents_3.json 中的训练曲线
echo   2. 运行诊断脚本 analyze_per_agent_predictions.py 对比改进效果
echo   3. 使用 compare_models.py 生成对比图表
echo.
pause
