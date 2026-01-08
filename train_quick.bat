@echo off
REM ================================================================
REM 集群轨迹模型快速训练脚本 (Windows 批处理)
REM ================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ================================ 🚀 模型训练脚本 ================================
echo.
echo 1. 快速验证修复 (50 epoch, ~40 分钟)
echo 2. 完整训练 3 架 (200 epoch, ~3-4 小时)
echo 3. 多架训练 3-6 (200 epoch, ~12-16 小时)
echo 4. 自定义训练
echo 5. 从检查点恢复
echo 6. 退出
echo.

set /p choice="请选择 [1-6]: "

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto full_train_3
if "%choice%"=="3" goto full_train_all
if "%choice%"=="4" goto custom_train
if "%choice%"=="5" goto resume_train
if "%choice%"=="6" goto end

echo 无效选择，请重试
goto start_menu

:quick_test
echo.
echo ================================ 快速验证修复 ================================
echo 配置: 3 架无人机, 50 epoch, 自动检测 GPU
echo 训练时间: ~40 分钟
echo.

python train_swarm_model_enhanced.py ^
  --agents 3 ^
  --epochs 50 ^
  --batch_size 256 ^
  --use_amp

goto end

:full_train_3
echo.
echo ================================ 完整训练 3 架模型 ================================
echo 配置: 3 架无人机, 200 epoch, 启用所有优化
echo 训练时间: ~3-4 小时
echo.

python train_swarm_model_enhanced.py ^
  --agents 3 ^
  --data_dir swarm_segments ^
  --features_dir swarm_features ^
  --epochs 200 ^
  --batch_size 512 ^
  --hidden_size 64 ^
  --num_layers 2 ^
  --dropout 0.6 ^
  --lr 5e-4 ^
  --weight_decay 1e-4 ^
  --val_split 0.1 ^
  --output_dir swarm_models_fixed ^
  --use_amp ^
  --use_attention

goto end

:full_train_all
echo.
echo ================================ 多架训练 (3-6 架) ================================
echo 配置: 3, 4, 5, 6 架无人机，各 200 epoch
echo 训练时间: ~12-16 小时
echo.

python train_swarm_model_enhanced.py ^
  --agents all ^
  --data_dir swarm_segments ^
  --features_dir swarm_features ^
  --epochs 200 ^
  --batch_size 512 ^
  --hidden_size 64 ^
  --num_layers 2 ^
  --dropout 0.6 ^
  --lr 5e-4 ^
  --weight_decay 1e-4 ^
  --val_split 0.1 ^
  --output_dir swarm_models_fixed ^
  --use_amp ^
  --use_attention

goto end

:custom_train
echo.
echo ================================ 自定义训练参数 ================================
echo.

set /p agents="无人机数量 (3/4/5/6/all) [默认: 3]: "
if "!agents!"=="" set agents=3

set /p epochs="训练 epoch 数 [默认: 200]: "
if "!epochs!"=="" set epochs=200

set /p batch_size="批次大小 [默认: 512]: "
if "!batch_size!"=="" set batch_size=512

set /p hidden_size="隐藏维度 [默认: 64]: "
if "!hidden_size!"=="" set hidden_size=64

set /p lr="学习率 [默认: 5e-4]: "
if "!lr!"=="" set lr=5e-4

set /p use_amp="使用混合精度 (y/n) [默认: y]: "
if "!use_amp!"=="" set use_amp=y

echo.
echo 即将开始训练，参数:
echo   - agents: !agents!
echo   - epochs: !epochs!
echo   - batch_size: !batch_size!
echo   - hidden_size: !hidden_size!
echo   - learning_rate: !lr!
echo   - use_amp: !use_amp!
echo.
pause

if "!use_amp!"=="y" (
  python train_swarm_model_enhanced.py ^
    --agents !agents! ^
    --epochs !epochs! ^
    --batch_size !batch_size! ^
    --hidden_size !hidden_size! ^
    --lr !lr! ^
    --output_dir swarm_models_fixed ^
    --use_amp ^
    --use_attention
) else (
  python train_swarm_model_enhanced.py ^
    --agents !agents! ^
    --epochs !epochs! ^
    --batch_size !batch_size! ^
    --hidden_size !hidden_size! ^
    --lr !lr! ^
    --output_dir swarm_models_fixed ^
    --use_attention
)

goto end

:resume_train
echo.
echo ================================ 从检查点恢复训练 ================================
echo.

set /p agents="无人机数量 (3/4/5/6) [默认: 3]: "
if "!agents!"=="" set agents=3

echo 正在恢复训练...
echo （代码会自动检测最后的检查点）
echo.

python train_swarm_model_enhanced.py ^
  --agents !agents! ^
  --data_dir swarm_segments ^
  --features_dir swarm_features ^
  --epochs 200 ^
  --batch_size 512 ^
  --hidden_size 64 ^
  --num_layers 2 ^
  --dropout 0.6 ^
  --lr 5e-4 ^
  --weight_decay 1e-4 ^
  --val_split 0.1 ^
  --output_dir swarm_models_fixed ^
  --use_amp ^
  --use_attention

goto end

:end
echo.
echo ================================ 训练完成 ================================
echo.
if exist "swarm_models_fixed\training_history_agents_3.csv" (
  echo 训练历史已保存到: swarm_models_fixed\training_history_agents_*.csv
  echo.
  echo 查看训练结果:
  echo   type swarm_models_fixed\training_history_agents_3.csv
  echo   type swarm_models_fixed\training_config_agents_3.json
)
echo.
pause
