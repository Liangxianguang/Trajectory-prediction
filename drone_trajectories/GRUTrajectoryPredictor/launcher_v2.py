#!/usr/bin/env python
"""
Swarm GRU 训练启动器 - 完整版本
==============================
提供简单的命令来训练和推理无人机群轨迹预测模型

使用示例：
    # 运行快速测试
    python launcher_v2.py test
    
    # 快速训练 (5 分钟)
    python launcher_v2.py train quick
    
    # 完整训练 (1 小时)
    python launcher_v2.py train full
    
    # 快速推理
    python launcher_v2.py infer quick
    
    # 完整推理
    python launcher_v2.py infer full
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def train_quick():
    """快速训练 - 5分钟"""
    print_header("快速训练 - 5k 样本, 10 epochs")
    print("预计时间: ~5 分钟\n")
    
    cmd = [
        "python", "train_swarm_gru_v3.py",
        "--num_agents", "3",
        "--num_epochs", "10",
        "--batch_size", "64",
        "--use_subset",
        "--early_stopping_patience", "10",
        "--save_every", "5"
    ]
    
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def train_medium():
    """中等训练 - 20 分钟"""
    print_header("中等训练 - 50k 样本, 30 epochs")
    print("预计时间: ~20 分钟\n")
    
    cmd = [
        "python", "train_swarm_gru_v3.py",
        "--num_agents", "3",
        "--num_epochs", "30",
        "--batch_size", "64",
        "--early_stopping_patience", "15",
        "--save_every", "5"
    ]
    
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def train_full():
    """完整训练 - 1-2小时"""
    print_header("完整训练 - 230k 样本, 100 epochs")
    print("预计时间: ~1-2 小时 (取决于 GPU)\n")
    
    cmd = [
        "python", "train_swarm_gru_v3.py",
        "--num_agents", "3",
        "--num_epochs", "100",
        "--batch_size", "64",
        "--early_stopping_patience", "20",
        "--save_every", "10"
    ]
    
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def train_custom(args):
    """自定义训练"""
    print_header("自定义训练")
    
    cmd = ["python", "train_swarm_gru_v3.py"] + args
    
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def infer_quick():
    """快速推理"""
    print_header("快速推理 - 1000 样本")
    
    cmd = [
        "python", "predict_swarm_gru_v3.py",
        "--num_agents", "3",
        "--batch_size", "128",
        "--use_subset",
        "--visualize",
        "--save_results"
    ]
    
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def infer_full():
    """完整推理"""
    print_header("完整推理 - 所有测试样本")
    
    cmd = [
        "python", "predict_swarm_gru_v3.py",
        "--num_agents", "3",
        "--batch_size", "128",
        "--visualize",
        "--save_results"
    ]
    
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def run_test():
    """运行快速测试"""
    print_header("快速测试")
    
    cmd = ["python", "test_quick.py"]
    
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def show_help():
    """显示帮助信息"""
    help_text = """
╔════════════════════════════════════════════════════════════════════════╗
║         Swarm GRU Trajectory Predictor - Enhanced Launcher v2          ║
╚════════════════════════════════════════════════════════════════════════╝

用法：
    python launcher_v2.py <command> [options]

训练命令：
    launcher_v2.py train quick        快速训练 (5k 样本, 10 epochs, ~5 分钟)
    launcher_v2.py train medium       中等训练 (50k 样本, 30 epochs, ~20 分钟)
    launcher_v2.py train full         完整训练 (230k 样本, 100 epochs, ~1 小时)
    launcher_v2.py train custom ...   自定义训练参数

推理命令：
    launcher_v2.py infer quick        快速推理 (1000 样本)
    launcher_v2.py infer full         完整推理 (所有测试样本)

其他命令：
    launcher_v2.py test               运行快速测试（验证环境）
    launcher_v2.py help               显示此帮助信息

示例：
    # 完整工作流
    python launcher_v2.py test
    python launcher_v2.py train quick
    python launcher_v2.py infer quick
    
    # 自定义训练参数
    python launcher_v2.py train custom --num_epochs 50 --batch_size 32 --save_every 5
    
    # 完整训练和推理
    python launcher_v2.py train full
    python launcher_v2.py infer full

新增功能 (v3)：
    ✓ 详细的指标记录 (MSE, MAE, MAPE, ADE, FDE, R²)
    ✓ CSV 格式的训练日志
    ✓ 定期模型检查点保存
    ✓ 更详细的推理结果
    ✓ 按 Agent 和坐标的分解指标

输出文件位置：
    Models/swarm_gru_agents_3_best.pth       最佳模型
    checkpoints/agents_3_<timestamp>/        模型检查点
    Results/metrics_agents_3_<timestamp>.csv 训练指标 CSV
    Results/inference_metrics_*.json          推理指标 JSON
    Results/visualizations/                   3D轨迹图
    Results/metrics/                          误差度量图

关键改进：
    - 训练时实时计算 ADE, FDE 等指标
    - 自动保存最佳模型和周期检查点
    - CSV 格式便于 Excel 分析
    - 推理时包含详细的误差分析
"""
    print(help_text)


def main():
    if len(sys.argv) < 2:
        show_help()
        return 0
    
    command = sys.argv[1].lower()
    
    if command == "help":
        show_help()
        return 0
    
    elif command == "test":
        return run_test()
    
    elif command == "train":
        if len(sys.argv) < 3:
            print("用法: python launcher_v2.py train <quick|medium|full|custom> [args...]")
            return 1
        
        mode = sys.argv[2].lower()
        
        if mode == "quick":
            return train_quick()
        elif mode == "medium":
            return train_medium()
        elif mode == "full":
            return train_full()
        elif mode == "custom":
            return train_custom(sys.argv[3:])
        else:
            print(f"未知训练模式: {mode}")
            print("使用: quick, medium, full, 或 custom")
            return 1
    
    elif command == "infer":
        if len(sys.argv) < 3:
            print("用法: python launcher_v2.py infer <quick|full>")
            return 1
        
        mode = sys.argv[2].lower()
        
        if mode == "quick":
            return infer_quick()
        elif mode == "full":
            return infer_full()
        else:
            print(f"未知推理模式: {mode}")
            print("使用: quick 或 full")
            return 1
    
    else:
        print(f"未知命令: {command}")
        print("使用: test, train, infer, 或 help")
        return 1


if __name__ == "__main__":
    sys.exit(main())
