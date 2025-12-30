@echo off
REM ============================================================================
REM BiGRU + Cross-Attention 轨迹预测模型训练脚本
REM ============================================================================
REM 特点：
REM - BiGRU编码器充分保留双向信息（前向+反向）
REM - Cross-Attention解码器让每个预测步都能访问编码器全部位置
REM - 充分利用双向编码的优势，避免信息浪费
REM ============================================================================

cd /d "%~dp0"
cd drone_trajectories

echo.
echo ============================================================================
echo BiGRU + Cross-Attention 模型训练
echo ============================================================================
echo.

REM 检查数据文件
if not exist "dataset_position_segments_synth.npz" (
    echo 错误：找不到 dataset_position_segments_synth.npz
    echo 请确保在 drone_trajectories 目录下
    pause
    exit /b 1
)

REM 设置参数
set DATA_PATH=dataset_position_segments_synth.npz
set OUTPUT_DIR=tool\gru_models_bigru_improved
set MODEL_NAME=bigru_improved_model
set EPOCHS=120
set BATCH_SIZE=256
set HIDDEN_DIM=128
set NUM_LAYERS=3
set LR=0.001
set DROPOUT=0.3
set PATIENCE=30
set TF_RATIO=0.6
set TF_DECAY=0.005

echo.
echo 模型配置：
echo   数据文件：%DATA_PATH%
echo   输出目录：%OUTPUT_DIR%
echo   模型名称：%MODEL_NAME%
echo   隐藏维度：%HIDDEN_DIM%
echo   层数：%NUM_LAYERS%
echo   Epochs：%EPOCHS%
echo   批大小：%BATCH_SIZE%
echo   学习率：%LR%
echo   Dropout：%DROPOUT%
echo   耐心度：%PATIENCE%
echo   Teacher Forcing 比率：%TF_RATIO%
echo   TF衰减：%TF_DECAY%
echo.

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 运行训练
echo 开始训练...
echo.

python tool\train_model_bigru_improved.py ^
  --data_path %DATA_PATH% ^
  --output_dir %OUTPUT_DIR% ^
  --model_name %MODEL_NAME% ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --hidden_dim %HIDDEN_DIM% ^
  --num_layers %NUM_LAYERS% ^
  --lr %LR% ^
  --dropout %DROPOUT% ^
  --patience %PATIENCE% ^
  --teacher_forcing_ratio %TF_RATIO% ^
  --tf_decay %TF_DECAY% ^
  --use_amp ^
  --num_workers 0

if errorlevel 1 (
    echo.
    echo 训练失败！
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo 训练完成！
echo ============================================================================
echo.
echo 最优模型已保存到: %OUTPUT_DIR%\%MODEL_NAME%_best_model.pth
echo 训练历史已保存到: %OUTPUT_DIR%\%MODEL_NAME%_history.csv
echo 训练配置已保存到: %OUTPUT_DIR%\%MODEL_NAME%_training_config.json
echo.
echo 下一步建议：
echo 1. 查看训练历史：%OUTPUT_DIR%\%MODEL_NAME%_history.csv
echo 2. 使用评估脚本评估模型
echo 3. 与其他版本进行对比
echo.
pause
