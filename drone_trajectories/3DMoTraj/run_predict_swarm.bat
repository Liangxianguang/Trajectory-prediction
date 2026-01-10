@echo off
set "PYTHON_PATH=python"
set "SCRIPT_PATH=%~dp0tool\lbebm3D.py"
set "MODEL_PATH=%~dp0saved_models\lbebm3D_swarm_best.pt"

echo Running prediction/testing for SWARM dataset...
%PYTHON_PATH% "%SCRIPT_PATH%" ^
    --test_mode ^
    --dataset_name swarm ^
    --dataset_folder dataset ^
    --model_path "%MODEL_PATH%" ^
    --obs 8 ^
    --preds 12 ^
    --vis True

pause
