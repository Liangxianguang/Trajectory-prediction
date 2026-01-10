@echo off
set "PYTHON_PATH=python"
set "SCRIPT_PATH=%~dp0tool\convert_swarm_data.py"
set "INPUT_PATH=%~dp0swarm_trajectories"
set "OUTPUT_PATH=%~dp0dataset"

echo Converting swarm data...
%PYTHON_PATH% "%SCRIPT_PATH%" --input_path "%INPUT_PATH%" --output_path "%OUTPUT_PATH%" --dataset_name swarm
pause
