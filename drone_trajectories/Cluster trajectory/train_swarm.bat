@echo off
REM 集群轨迹模型训练 - 快速命令脚本

cd /d "d:\Trajectory prediction\drone_trajectories\Cluster trajectory"

echo.
echo ========================================
echo   集群轨迹模型训练
echo ========================================
echo.
echo 选择训练模式:
echo.
echo 1. 快速测试 (3架, 20轮, 128批次)
echo 2. 标准训练 (3架, 100轮, 64批次)
echo 3. 深度训练 (3架, 150轮, 32批次, hidden=256)
echo 4. 批量训练 (全部规模 3-6架)
echo 5. 使用 AMP 加速训练 (3架, 混合精度)
echo 6. 从检查点恢复 (3架)
echo 7. 自定义参数
echo.

set /p choice="请输入选择 (1-7): "

if "%choice%"=="1" (
    echo 启动快速测试...
    python train_swarm_model.py ^
        --agents 3 ^
        --epochs 20 ^
        --batch_size 128 ^
        --hidden_size 64 ^
        --num_layers 2 ^
        --dropout 0.3 ^
        --lr 0.001
    
) else if "%choice%"=="2" (
    echo 启动标准训练...
    python train_swarm_model.py ^
        --agents 3 ^
        --epochs 100 ^
        --batch_size 64 ^
        --hidden_size 128 ^
        --num_layers 2 ^
        --dropout 0.3 ^
        --lr 0.001
    
) else if "%choice%"=="3" (
    echo 启动深度训练...
    python train_swarm_model.py ^
        --agents 3 ^
        --epochs 150 ^
        --batch_size 32 ^
        --hidden_size 256 ^
        --num_layers 3 ^
        --dropout 0.4 ^
        --lr 0.0005 ^
        --patience 30
    
) else if "%choice%"=="4" (
    echo 启动批量训练 (全部规模)...
    python train_swarm_model.py ^
        --agents all ^
        --epochs 100 ^
        --batch_size 64 ^
        --hidden_size 128 ^
        --num_layers 2 ^
        --dropout 0.3 ^
        --lr 0.001
    
) else if "%choice%"=="5" (
    echo 启动 AMP 加速训练...
    python train_swarm_model.py ^
        --agents 3 ^
        --epochs 100 ^
        --batch_size 128 ^
        --hidden_size 128 ^
        --num_layers 2 ^
        --dropout 0.3 ^
        --lr 0.001 ^
        --use_amp
    
) else if "%choice%"=="6" (
    echo 启动检查点恢复训练...
    python train_swarm_model.py ^
        --agents 3 ^
        --epochs 200 ^
        --resume best_model_agents_3.pt
    
) else if "%choice%"=="7" (
    echo.
    echo 自定义参数训练
    echo 保留默认值，按回车键跳过
    echo.
    
    set /p agents="无人机数量 [3]: "
    if "!agents!"=="" set agents=3
    
    set /p epochs="训练轮数 [100]: "
    if "!epochs!"=="" set epochs=100
    
    set /p batch_size="批次大小 [64]: "
    if "!batch_size!"=="" set batch_size=64
    
    set /p hidden_size="隐藏层大小 [128]: "
    if "!hidden_size!"=="" set hidden_size=128
    
    set /p lr="学习率 [0.001]: "
    if "!lr!"=="" set lr=0.001
    
    echo.
    echo 启动自定义训练...
    python train_swarm_model.py ^
        --agents !agents! ^
        --epochs !epochs! ^
        --batch_size !batch_size! ^
        --hidden_size !hidden_size! ^
        --lr !lr!
    
) else (
    echo 无效选择，退出。
    exit /b 1
)

echo.
echo ========================================
echo   训练完成！
echo ========================================
echo.
echo 模型保存在: swarm_models\
echo.
pause
