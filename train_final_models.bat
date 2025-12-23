@echo off
REM Final training pipeline (Windows CMD)
REM 用途：训练位置 + 速度 模型（short/mix/long 三种配置），启用 Teacher Forcing 线性衰减
REM 使用方法：在 cmd 中运行或双击本文件

SETLOCAL ENABLEDELAYEDEXPANSION

REM 根目录
cd /d "D:\Trajectory prediction"

REM Python 可执行
SET PY=python

REM 公共路径
SET DATA_POS=drone_trajectories\dataset_position_segments_synth.npz
SET DATA_VEL=drone_trajectories\dataset_velocity_segments_synth.npz
SET STATS_POS=drone_trajectories\transformed_trajectories\pos_stats.npz
SET STATS_VEL=drone_trajectories\transformed_trajectories\vel_stats.npz
SET OUT_DIR=drone_trajectories\gru_models_final

REM 训练超参
SET EPOCHS=120
SET TF_START=0.5
SET TF_DECAY=0.00333
SET LR=1e-3
SET WEIGHT_DECAY=1e-5
SET GRAD_CLIP=1.0
SET DROPOUT=0.5
SET BATCH_SIZE_SMALL=128
SET BATCH_SIZE_LARGE=64
SET PATIENCE=15
SET NUM_WORKERS=8

REM Data prep（如已生成可注释掉）
echo.
echo ===============================================
echo Step 0: (可选) 生成速度/位置数据集（如果尚未生成）
echo ===============================================
if not exist %DATA_VEL% (
  echo 生成速度数据集...
  cd drone_trajectories
  %PY% process_multiple_trajectories.py "D:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories" 20 10 --compute_velocity --save_path "dataset_velocity_segments_synth.npz"
  cd ..
) else (
  echo 数据集已存在： %DATA_VEL%
)

REM 创建输出目录
mkdir %OUT_DIR% 2>nul

echo.
echo ===============================================
echo 位置模型训练（short / mix / long）
echo ===============================================

REM position_synth_short (hidden=64, layers=2)
echo.
echo [1/6] Training position_short (hidden=64, layers=2)...
%PY% drone_trajectories/train_model.py --data_path %DATA_POS% --stats_path %STATS_POS% --output_dir %OUT_DIR% --model_name position_synth_short --hidden_dim 64 --num_layers 2 --dropout %DROPOUT% --epochs %EPOCHS% --batch_size 128 --lr %LR% --weight_decay %WEIGHT_DECAY% --teacher_forcing_ratio %TF_START% --tf_decay %TF_DECAY% --patience 15 --num_workers %NUM_WORKERS% --pin_memory --use_amp
if errorlevel 1 (
    echo ERROR: position_synth_short 训练失败
    pause
    exit /b 1
)

REM position_synth_mix (hidden=128, layers=3)
echo.
echo [2/6] Training position_mix (hidden=128, layers=3)...
%PY% drone_trajectories/train_model.py --data_path %DATA_POS% --stats_path %STATS_POS% --output_dir %OUT_DIR% --model_name position_synth_mix --hidden_dim 128 --num_layers 3 --dropout %DROPOUT% --epochs %EPOCHS% --batch_size 64 --lr %LR% --weight_decay %WEIGHT_DECAY% --teacher_forcing_ratio %TF_START% --tf_decay %TF_DECAY% --patience 15 --num_workers %NUM_WORKERS% --pin_memory --use_amp
if errorlevel 1 (
    echo ERROR: position_synth_mix 训练失败
    pause
    exit /b 1
)

REM position_synth_long (hidden=256, layers=5)
echo.
echo [3/6] Training position_long (hidden=256, layers=5)...
%PY% drone_trajectories/train_model.py --data_path %DATA_POS% --stats_path %STATS_POS% --output_dir %OUT_DIR% --model_name position_synth_long --hidden_dim 256 --num_layers 5 --dropout %DROPOUT% --epochs %EPOCHS% --batch_size 64 --lr %LR% --weight_decay %WEIGHT_DECAY% --teacher_forcing_ratio %TF_START% --tf_decay %TF_DECAY% --patience 15 --num_workers %NUM_WORKERS% --pin_memory --use_amp
if errorlevel 1 (
    echo ERROR: position_synth_long 训练失败
    pause
    exit /b 1
)

echo.
echo ===============================================
echo 速度模型训练（short / mix / long）
echo ===============================================

REM velocity_synth_short (hidden=64, layers=2)
echo.
echo [4/6] Training velocity_short (hidden=64, layers=2)...
%PY% drone_trajectories/train_model.py --data_path %DATA_VEL% --stats_path %STATS_VEL% --output_dir %OUT_DIR% --model_name velocity_synth_short --hidden_dim 64 --num_layers 2 --dropout %DROPOUT% --epochs %EPOCHS% --batch_size 128 --lr %LR% --weight_decay %WEIGHT_DECAY% --teacher_forcing_ratio %TF_START% --tf_decay %TF_DECAY% --patience 15 --num_workers %NUM_WORKERS% --pin_memory --use_amp
if errorlevel 1 (
    echo ERROR: velocity_synth_short 训练失败
    pause
    exit /b 1
)

REM velocity_synth_mix (hidden=128, layers=3)
echo.
echo [5/6] Training velocity_mix (hidden=128, layers=3)...
%PY% drone_trajectories/train_model.py --data_path %DATA_VEL% --stats_path %STATS_VEL% --output_dir %OUT_DIR% --model_name velocity_synth_mix --hidden_dim 128 --num_layers 3 --dropout %DROPOUT% --epochs %EPOCHS% --batch_size 64 --lr %LR% --weight_decay %WEIGHT_DECAY% --teacher_forcing_ratio %TF_START% --tf_decay %TF_DECAY% --patience 15 --num_workers %NUM_WORKERS% --pin_memory --use_amp
if errorlevel 1 (
    echo ERROR: velocity_synth_mix 训练失败
    pause
    exit /b 1
)

REM velocity_synth_long (hidden=256, layers=5)
echo.
echo [6/6] Training velocity_long (hidden=256, layers=5)...
%PY% drone_trajectories/train_model.py --data_path %DATA_VEL% --stats_path %STATS_VEL% --output_dir %OUT_DIR% --model_name velocity_synth_long --hidden_dim 256 --num_layers 5 --dropout %DROPOUT% --epochs %EPOCHS% --batch_size 64 --lr %LR% --weight_decay %WEIGHT_DECAY% --teacher_forcing_ratio %TF_START% --tf_decay %TF_DECAY% --patience 15 --num_workers %NUM_WORKERS% --pin_memory --use_amp
if errorlevel 1 (
    echo ERROR: velocity_synth_long 训练失败
    pause
    exit /b 1
)

echo.
echo ===============================================
echo 所有训练完成！
echo 模型位置: %CD%\%OUT_DIR%\
echo ===============================================
echo.
echo 生成的文件列表：
dir %OUT_DIR% /B
echo.
pause
ENDLOCAL
