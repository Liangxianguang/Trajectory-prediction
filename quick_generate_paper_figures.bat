@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
cls

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║     LBEBM3D vs GNN+BiGRU - 论文级对比图表生成                    ║
echo ║                  Paper-ready Visualization Generator              ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.

REM 设置路径
set REPO_ROOT=d:\Trajectory prediction
set TRAJ_DIR=%REPO_ROOT%\drone_trajectories\Cluster trajectory
set RESULTS_DIR=%TRAJ_DIR%\comparison_results_1w_lbebm_vs_gnn
set SCRIPT_PATH=%TRAJ_DIR%\plot_lbebm_vs_gnn_comparison.py
set SUMMARY_JSON=%RESULTS_DIR%\comparison_summary.json
set OUTPUT_DIR=%RESULTS_DIR%\paper_figures

echo 📋 配置信息:
echo   脚本位置: %SCRIPT_PATH%
echo   输入数据: %SUMMARY_JSON%
echo   输出目录: %OUTPUT_DIR%
echo.

REM 检查依赖文件
echo 🔍 正在检查依赖文件...
if not exist "%SUMMARY_JSON%" (
    echo.
    echo ❌ 错误: 找不到汇总数据文件
    echo    位置: %SUMMARY_JSON%
    echo.
    echo 💡 请确保已运行过 run_lbebm_vs_gnn_comparison.bat
    echo.
    pause
    exit /b 1
)
echo ✅ 汇总数据文件存在

if not exist "%SCRIPT_PATH%" (
    echo.
    echo ❌ 错误: 找不到生成脚本
    echo    位置: %SCRIPT_PATH%
    echo.
    pause
    exit /b 1
)
echo ✅ 生成脚本存在
echo.

REM 创建输出目录
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
    echo ✅ 创建输出目录: %OUTPUT_DIR%
)

REM 运行 Python 脚本
echo.
echo 📊 开始生成论文级图表...
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

python "%SCRIPT_PATH%" ^
    --summary_json "%SUMMARY_JSON%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --dpi 300

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    echo.
    echo ✅ 图表生成成功！
    echo.
    echo 📁 输出位置: %OUTPUT_DIR%
    echo.
    echo 📊 生成的图表:
    echo   ✓ overall_comparison.png (整体性能对比 - 主图！)
    echo   ✓ overall_comparison.pdf (矢量格式)
    echo   ✓ axis_mae_comparison.png (轴向误差对比)
    echo   ✓ axis_mae_comparison.pdf
    echo   ✓ mae_boxplot_comparison.png (样本分布箱线图)
    echo   ✓ mae_boxplot_comparison.pdf
    echo   ✓ per_agent_mae_comparison.png (单体性能对比)
    echo   ✓ per_agent_mae_comparison.pdf
    echo   ✓ error_trend_comparison.png (误差趋势)
    echo   ✓ error_trend_comparison.pdf
    echo   ✓ comparison_table.txt (详细指标表 - 用于论文表格)
    echo.
    echo 📝 下一步建议:
    echo   1. 打开图表文件夹查看生成的PNG/PDF文件
    echo   2. 选择 overall_comparison.png 作为论文主图
    echo   3. 复制 comparison_table.txt 中的数据到论文表格
    echo.
    echo 🎨 配色方案:
    echo   红色 #E74C3C = LBEBM3D
    echo   橙色 #E67E22 = GNN+BiGRU
    echo   (对色盲友好，适合印刷)
    echo.
    echo 📖 使用建议:
    echo   - PNG 格式: 用于在线论文、演示稿
    echo   - PDF 格式: 用于印刷版论文（高分辨率矢量）
    echo.
) else (
    echo.
    echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    echo.
    echo ❌ 图表生成失败！
    echo.
    echo 💡 排查建议:
    echo   1. 检查 Python 环境是否正确配置
    echo   2. 检查是否安装了 matplotlib 和 seaborn
    echo      运行: pip install matplotlib seaborn numpy
    echo   3. 检查输入数据文件是否完整
    echo.
)

echo.
pause
