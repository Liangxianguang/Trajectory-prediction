@echo off
REM 快速训练模式 - 推荐配置
REM 不使用 Attention 以获得最大速度

echo 启动增强版集群轨迹模型快速训练...
echo.

REM 基础快速训练 (推荐首次运行)
REM 预计时间: 3 agents, 50 epochs, batch_size=256 ≈ 2-3 小时
python train_swarm_model_enhanced.py ^
    --agents 3 ^
    --epochs 50 ^
    --batch_size 256 ^
    --hidden_size 128 ^
    --num_layers 3 ^
    --dropout 0.5 ^
    --lr 1e-3 ^
    --output_dir newdata3_swarm_models_enhanced ^
    --use_amp

echo.
echo 训练完成！检查 swarm_models_enhanced/ 文件夹中的最佳模型
pause
