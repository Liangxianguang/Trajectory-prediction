@echo off
REM 完整训练脚本：位置+速度模型，三个大小配置（Windows CMD）
REM 运行方式：双击运行此文件，或在 cmd 中输入: train_all_models.bat

cd /d "D:\Trajectory prediction"

echo.
echo ===============================================
echo Step 1: 生成速度数据集（仅需一次）
echo ===============================================
echo.
cd drone_trajectories
python process_multiple_trajectories.py "D:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories" 20 10 --compute_velocity --save_path "D:\Trajectory prediction\drone_trajectories\dataset_velocity_segments_synth.npz"
cd ..

echo.
echo ===============================================
echo 位置模型训练（Synthetic-UAV 数据）
echo ===============================================

echo.
echo [1/6] Training position_synth_short (hidden=64, layers=2)...
python drone_trajectories\train_model.py ^
  --data_path drone_trajectories\dataset_position_segments_synth.npz ^
  --stats_path drone_trajectories\transformed_trajectories\pos_stats.npz ^
  --output_dir drone_trajectories\gru_models_synth ^
  --model_name position_synth_short ^
  --hidden_dim 64 --num_layers 2 ^
  --epochs 120 --batch_size 128 --lr 1e-3 --patience 15 ^
  --num_workers 8 --pin_memory --use_amp

if errorlevel 1 (
    echo ERROR: position_synth_short 训练失败
    pause
    exit /b 1
)

echo.
echo [2/6] Training position_synth_mix (hidden=128, layers=3)...
python drone_trajectories\train_model.py ^
  --data_path drone_trajectories\dataset_position_segments_synth.npz ^
  --stats_path drone_trajectories\transformed_trajectories\pos_stats.npz ^
  --output_dir drone_trajectories\gru_models_synth ^
  --model_name position_synth_mix ^
  --hidden_dim 128 --num_layers 3 ^
  --epochs 120 --batch_size 64 --lr 1e-3 --patience 15 ^
  --num_workers 8 --pin_memory --use_amp

if errorlevel 1 (
    echo ERROR: position_synth_mix 训练失败
    pause
    exit /b 1
)

echo.
echo [3/6] Training position_synth_long (hidden=256, layers=5)...
python drone_trajectories\train_model.py ^
  --data_path drone_trajectories\dataset_position_segments_synth.npz ^
  --stats_path drone_trajectories\transformed_trajectories\pos_stats.npz ^
  --output_dir drone_trajectories\gru_models_synth ^
  --model_name position_synth_long ^
  --hidden_dim 256 --num_layers 5 ^
  --epochs 120 --batch_size 64 --lr 1e-3 --patience 15 ^
  --num_workers 8 --pin_memory --use_amp

if errorlevel 1 (
    echo ERROR: position_synth_long 训练失败
    pause
    exit /b 1
)

echo.
echo ===============================================
echo 速度模型训练（Synthetic-UAV 数据）
echo ===============================================

echo.
echo [4/6] Training velocity_synth_short (hidden=64, layers=2)...
python drone_trajectories\train_model.py ^
  --data_path drone_trajectories\dataset_velocity_segments_synth.npz ^
  --stats_path drone_trajectories\transformed_trajectories\vel_stats.npz ^
  --output_dir drone_trajectories\gru_models_synth ^
  --model_name velocity_synth_short ^
  --hidden_dim 64 --num_layers 2 ^
  --epochs 120 --batch_size 128 --lr 1e-3 --patience 15 ^
  --num_workers 8 --pin_memory --use_amp

if errorlevel 1 (
    echo ERROR: velocity_synth_short 训练失败
    pause
    exit /b 1
)

echo.
echo [5/6] Training velocity_synth_mix (hidden=128, layers=3)...
python drone_trajectories\train_model.py ^
  --data_path drone_trajectories\dataset_velocity_segments_synth.npz ^
  --stats_path drone_trajectories\transformed_trajectories\vel_stats.npz ^
  --output_dir drone_trajectories\gru_models_synth ^
  --model_name velocity_synth_mix ^
  --hidden_dim 128 --num_layers 3 ^
  --epochs 120 --batch_size 64 --lr 1e-3 --patience 15 ^
  --num_workers 8 --pin_memory --use_amp

if errorlevel 1 (
    echo ERROR: velocity_synth_mix 训练失败
    pause
    exit /b 1
)

echo.
echo [6/6] Training velocity_synth_long (hidden=256, layers=5)...
python drone_trajectories\train_model.py ^
  --data_path drone_trajectories\dataset_velocity_segments_synth.npz ^
  --stats_path drone_trajectories\transformed_trajectories\vel_stats.npz ^
  --output_dir drone_trajectories\gru_models_synth ^
  --model_name velocity_synth_long ^
  --hidden_dim 256 --num_layers 5 ^
  --epochs 120 --batch_size 64 --lr 1e-3 --patience 15 ^
  --num_workers 8 --pin_memory --use_amp

if errorlevel 1 (
    echo ERROR: velocity_synth_long 训练失败
    pause
    exit /b 1
)

echo.
echo ===============================================
echo 所有训练完成！
echo ===============================================
echo 模型位置: D:\Trajectory prediction\drone_trajectories\gru_models_synth\
echo.
pause
