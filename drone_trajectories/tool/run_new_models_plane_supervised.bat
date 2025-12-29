@echo off
REM Enhanced plane-supervised training for short/mid/long models
cd /d "%~dp0\..\"

REM Short model (2 layers, 64 hidden)
python tool\train_model_enhanced.py ^
  --data_path combined_segments.npz ^
  --output_dir tool\new_short_gru_models_speed ^
  --model_name short_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 2048 ^
  --hidden_dim 64 ^
  --num_layers 2 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --num_workers 0 ^
  --loss_lambda_curv 0.02 ^
  --loss_lambda_plane_consistency 0.05 ^
  --loss_lambda_plane_supervision 0.2 ^
  --axis_weights 1.0,1.1,1.2
  

REM Mid model (3 layers, 128 hidden)
python tool\train_model_enhanced.py ^
  --data_path combined_segments.npz ^
  --output_dir tool\new_mid_gru_models_speed ^
  --model_name mid_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 2048 ^
  --hidden_dim 128 ^
  --num_layers 3 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --num_workers 0 ^
  --loss_lambda_curv 0.02 ^
  --loss_lambda_plane_consistency 0.05 ^
  --loss_lambda_plane_supervision 0.2 ^
  --axis_weights 1.0,1.1,1.2

REM Long model (5 layers, 256 hidden)
python tool\train_model_enhanced.py ^
  --data_path combined_segments.npz ^
  --output_dir tool\new_long_gru_models_speed ^
  --model_name long_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 2048 ^
  --hidden_dim 256 ^
  --num_layers 5 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --num_workers 0 ^
  --loss_lambda_curv 0.02 ^
  --loss_lambda_plane_consistency 0.05 ^
  --loss_lambda_plane_supervision 0.2 ^
  --axis_weights 1.0,1.1,1.2