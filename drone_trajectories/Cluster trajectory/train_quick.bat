@echo off
REM ============================================================================
REM 集群无人机轨迹预测模型 - 快速训练脚本
REM ============================================================================
REM 用法：双击运行此脚本，或在终端执行 train_quick.bat

setlocal enabledelayedexpansion

REM 设置颜色和样式
color 0A
title 无人机轨迹预测 - 模型训练

echo.
echo ============================================================================
echo  集群无人机轨迹预测模型训练脚本
echo ============================================================================
echo.
echo 预置选项：
echo  [1] 快速测试 (5 epochs, 3架无人机, ~5-10分钟)
echo  [2] 标准训练 (200 epochs, 3架无人机, ~1-2小时)
echo  [3] 完整训练 (200 epochs, 全部配置 3-6架, ~4-8小时)
echo  [4] 高性能训练 (300 epochs, 全部配置, ~6-12小时)
echo  [5] 自定义参数
echo  [6] 查看训练进度
echo.

set /p choice="请选择训练模式 (1-6): "

if "%choice%"=="1" (
    echo.
    echo [快速测试模式] 启动...
    python train_swarm_model_enhanced.py ^
      --agents 3 ^
      --data_dir swarm_segments ^
      --features_dir swarm_features ^
      --epochs 5 ^
      --batch_size 256 ^
      --output_dir test_swarm_models ^
      --use_amp
    
) else if "%choice%"=="2" (
    echo.
    echo [标准训练模式] 启动...
    python train_swarm_model_enhanced.py ^
      --agents 3 ^
      --data_dir swarm_segments ^
      --features_dir swarm_features ^
      --epochs 200 ^
      --batch_size 256 ^
      --output_dir newloss_swarm_models_enhanced ^
      --use_amp ^
      --use_attention
    
) else if "%choice%"=="3" (
    echo.
    echo [完整训练模式] 启动（3-6架无人机）...
    python train_swarm_model_enhanced.py ^
      --agents all ^
      --data_dir swarm_segments ^
      --features_dir swarm_features ^
      --epochs 200 ^
      --batch_size 256 ^
      --output_dir newloss_swarm_models_enhanced ^
      --use_amp ^
      --use_attention
    
) else if "%choice%"=="4" (
    echo.
    echo [高性能训练模式] 启动...
    python train_swarm_model_enhanced.py ^
      --agents all ^
      --data_dir swarm_segments ^
      --features_dir swarm_features ^
      --epochs 300 ^
      --batch_size 512 ^
      --hidden_size 256 ^
      --num_layers 3 ^
      --output_dir newloss_swarm_models_enhanced ^
      --use_amp ^
      --use_attention
    
) else if "%choice%"=="5" (
    echo.
    echo [自定义参数模式]
    echo.
    set /p agents="输入无人机架数 (3/4/5/6/all, 默认all): " || set agents=all
    set /p epochs="输入训练轮数 (默认200): " || set epochs=200
    set /p batch_size="输入批次大小 (默认256): " || set batch_size=256
    set /p hidden_size="输入隐藏层大小 (默认128): " || set hidden_size=128
    set /p output_dir="输入输出目录 (默认newloss_swarm_models_enhanced): " || set output_dir=newloss_swarm_models_enhanced
    
    echo.
    echo [自定义模式] 启动...
    echo   架数: !agents!
    echo   轮数: !epochs!
    echo   批次: !batch_size!
    echo   隐藏层: !hidden_size!
    echo   输出: !output_dir!
    echo.
    
    python train_swarm_model_enhanced.py ^
      --agents !agents! ^
      --data_dir swarm_segments ^
      --features_dir swarm_features ^
      --epochs !epochs! ^
      --batch_size !batch_size! ^
      --hidden_size !hidden_size! ^
      --output_dir !output_dir! ^
      --use_amp ^
      --use_attention
    
) else if "%choice%"=="6" (
    echo.
    echo [查看训练进度]
    echo.
    set /p agent_num="输入要查看的架数 (3/4/5/6, 默认3): " || set agent_num=3
    
    if exist "newloss_swarm_models_enhanced\training_history_agents_!agent_num!.csv" (
        echo.
        echo 最近的训练历史 (前10行):
        echo.
        for /f "tokens=*" %%a in ('type "newloss_swarm_models_enhanced\training_history_agents_!agent_num!.csv" ^| findstr /R ".*"') do (
            set count=0
            !count! equ 10 goto :done
            echo %%a
            set /a count+=1
        )
        :done
        echo.
        echo 完整文件位置:
        echo newloss_swarm_models_enhanced\training_history_agents_!agent_num!.csv
        echo.
        echo 提示: 用 Excel/Google Sheets 打开 CSV 文件查看完整图表
    ) else (
        echo 未找到训练历史文件
        echo newloss_swarm_models_enhanced\training_history_agents_!agent_num!.csv
    )
    
    pause
    goto :EOF
    
) else (
    echo 无效选择，退出
    pause
    goto :EOF
)

echo.
echo ============================================================================
echo 训练启动完成！
echo ============================================================================
echo.
echo 查看进度:
echo   - 实时日志显示在上方
echo   - 训练历史保存在: newloss_swarm_models_enhanced\training_history_agents_*.csv
echo   - 训练配置保存在: newloss_swarm_models_enhanced\training_config_agents_*.json
echo.
echo 提示:
echo   - 按 Ctrl+C 可暂停训练（会自动保存检查点）
echo   - 再次运行相同命令会自动恢复训练
echo   - 最佳模型保存在: newloss_swarm_models_enhanced\best_model_agents_*.pt
echo.

pause
