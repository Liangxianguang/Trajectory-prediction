@echo off
REM ============================================================================
REM BiGRU + Cross-Attention 完整对比实验脚本
REM ============================================================================
REM 此脚本训练不同配置的BiGRU模型，用于消融和对比实验
REM ============================================================================

cd /d "%~dp0"
cd drone_trajectories

if not exist "dataset_position_segments_synth.npz" (
    echo 错误：找不到数据文件
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo BiGRU 消融和对比实验
echo ============================================================================
echo.

REM 基础参数
set DATA_PATH=dataset_position_segments_synth.npz
set BASE_OUTPUT=tool\gru_models_bigru_improved
set EPOCHS=120
set PATIENCE=30
set NUM_WORKERS=0

REM ============================================================================
REM 配置 1：基础 BiGRU (128维, 3层)
REM ============================================================================
echo.
echo [1/3] 训练基础 BiGRU (hidden_dim=128, num_layers=3)...
echo.

python tool\train_model_bigru_improved.py ^
  --data_path %DATA_PATH% ^
  --output_dir %BASE_OUTPUT%_128_3 ^
  --model_name bigru_128_3 ^
  --epochs %EPOCHS% ^
  --batch_size 256 ^
  --hidden_dim 128 ^
  --num_layers 3 ^
  --lr 0.001 ^
  --dropout 0.3 ^
  --patience %PATIENCE% ^
  --teacher_forcing_ratio 0.6 ^
  --tf_decay 0.005 ^
  --use_amp ^
  --num_workers %NUM_WORKERS%

if errorlevel 1 (
    echo 配置 1 训练失败！
    pause
    exit /b 1
)

REM ============================================================================
REM 配置 2：较大 BiGRU (256维, 5层) - 更深更宽
REM ============================================================================
echo.
echo [2/3] 训练较大 BiGRU (hidden_dim=256, num_layers=5)...
echo.

python tool\train_model_bigru_improved.py ^
  --data_path %DATA_PATH% ^
  --output_dir %BASE_OUTPUT%_256_5 ^
  --model_name bigru_256_5 ^
  --epochs %EPOCHS% ^
  --batch_size 128 ^
  --hidden_dim 256 ^
  --num_layers 5 ^
  --lr 0.0005 ^
  --dropout 0.4 ^
  --patience %PATIENCE% ^
  --teacher_forcing_ratio 0.6 ^
  --tf_decay 0.004 ^
  --use_amp ^
  --num_workers %NUM_WORKERS%

if errorlevel 1 (
    echo 配置 2 训练失败！
    pause
    exit /b 1
)

REM ============================================================================
REM 配置 3：轻量级 BiGRU (64维, 2层) - 更小更快
REM ============================================================================
echo.
echo [3/3] 训练轻量级 BiGRU (hidden_dim=64, num_layers=2)...
echo.

python tool\train_model_bigru_improved.py ^
  --data_path %DATA_PATH% ^
  --output_dir %BASE_OUTPUT%_64_2 ^
  --model_name bigru_64_2 ^
  --epochs %EPOCHS% ^
  --batch_size 512 ^
  --hidden_dim 64 ^
  --num_layers 2 ^
  --lr 0.001 ^
  --dropout 0.2 ^
  --patience %PATIENCE% ^
  --teacher_forcing_ratio 0.7 ^
  --tf_decay 0.006 ^
  --use_amp ^
  --num_workers %NUM_WORKERS%

if errorlevel 1 (
    echo 配置 3 训练失败！
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo 所有对比实验完成！
echo ============================================================================
echo.
echo 训练结果：
echo   [1] 基础模型：%BASE_OUTPUT%_128_3\bigru_128_3_best_model.pth
echo   [2] 较大模型：%BASE_OUTPUT%_256_5\bigru_256_5_best_model.pth
echo   [3] 轻量模型：%BASE_OUTPUT%_64_2\bigru_64_2_best_model.pth
echo.
echo 下一步：
echo 1. 使用 evaluate_all_models.py 评估所有模型
echo 2. 比较三个模型的性能差异
echo 3. 选择最优配置用于最终训练
echo.
pause
