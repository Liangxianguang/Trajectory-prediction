@echo off
set "PYTHON_PATH=python"
set "SCRIPT_PATH=%~dp0tool\lbebm3D.py"

echo Starting training for SWARM dataset...
%PYTHON_PATH% "%SCRIPT_PATH%" ^
    --dataset_name swarm ^
    --dataset_folder dataset ^
    --obs 20 ^
    --preds 10 ^
    --past_length 20 ^
    --future_length 10 ^
    --learning_rate 0.00005 ^
    --num_epochs 50 ^
    --batch_size 512 ^
    --num_workers 0 ^
    --device 0 ^
    --data_scale 1 ^
    --e_l_steps 5

pause