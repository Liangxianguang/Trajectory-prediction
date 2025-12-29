@echo off
cd /d "%~dp0\..\"
python tool\train_model_enhanced.py ^
  --data_path combined_segments.npz ^
  --output_dir tool\new_short_gru_models_enhanced ^
  --model_name short_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 4096 ^
  --hidden_dim 64 ^
  --num_layers 2 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --loss_lambda_curv 0.02 ^
  --axis_weights 1.0,1.2,1.5

python tool\train_model_enhanced.py ^
  --data_path combined_segments.npz ^
  --output_dir tool\new_mid_gru_models_enhanced ^
  --model_name mid_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 4096 ^
  --hidden_dim 128 ^
  --num_layers 3 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --loss_lambda_curv 0.02 ^
  --axis_weights 1.0,1.2,1.5

python tool\train_model_enhanced.py ^
  --data_path combined_segments.npz ^
  --output_dir tool\new_long_gru_models_enhanced ^
  --model_name long_enhanced_gru_model ^
  --epochs 300 ^
  --batch_size 4096 ^
  --hidden_dim 256 ^
  --num_layers 5 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --use_amp ^
  --use_attention ^
  --loss_lambda_curv 0.02 ^
  --axis_weights 1.0,1.2,1.5