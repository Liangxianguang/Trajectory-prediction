@echo off
REM ============================================================================
REM BiGRU + Cross-Attention 完整对比实验脚本
REM ============================================================================
REM 此脚本训练不同配置的BiGRU模型，用于消融和对比实验
REM ============================================================================

echo.
echo ========================================
echo 开始训练短模型 (Short Model: 2x64)
echo ========================================
python tool\train_model_bigru_improved.py ^
  --data_path merged_segments.npz ^
  --output_dir tool\bigru_Cross-Attention_short_gru_models ^
  --model_name short_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 2048 ^
  --hidden_dim 64 ^
  --num_layers 2 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --loss_alpha 0.7 ^
  --loss_beta 0.2 ^
  --loss_gamma 0.1 ^
  --axis_weights "1.0,1.1,1.2" ^
  --num_workers 0

if errorlevel 1 (
    echo 短模型训练失败！
    pause
    exit /b 1
echo.
echo ========================================
echo 开始训练中模型 (Mid Model: 3x128)
echo ========================================
python tool\train_model_bigru_improved.py ^
  --data_path merged_segments.npz ^
  --output_dir tool\bigru_Cross-Attention_mid_gru_models ^
  --model_name mid_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 2048 ^
  --hidden_dim 128 ^
  --num_layers 3 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --loss_alpha 0.7 ^
  --loss_beta 0.2 ^
  --loss_gamma 0.1 ^
  --axis_weights "1.0,1.1,1.2" ^
  --num_workers 0

if errorlevel 1 (
    echo 中模型训练失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 开始训练长模型 (Long Model: 5x256)
echo ========================================
python tool\train_model_bigru_improved.py ^
  --data_path merged_segments.npz ^
  --output_dir tool\bigru_Cross-Attention_long_gru_models ^
  --model_name long_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 2048 ^
  --hidden_dim 256 ^
  --num_layers 5 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --loss_alpha 0.7 ^
  --loss_beta 0.2 ^
  --loss_gamma 0.1 ^
  --axis_weights "1.0,1.1,1.2" ^
  --num_workers 0

if errorlevel 1 (
    echo 长模型训练失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 所有模型训练完成！
echo ========================================
pause