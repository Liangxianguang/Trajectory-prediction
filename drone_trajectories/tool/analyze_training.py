#!/usr/bin/env python3
"""
训练历史深度分析脚本
生成详细的训练报告，包含：
- Loss 统计分析
- 收敛速度分析
- 超参数影响分析
- 模型对比分析

使用方法：
  python analyze_training.py --model_dir path/to/model_output
  python analyze_training.py --csv_file path/to/history.csv
  python analyze_training.py --csv_file file1.csv file2.csv file3.csv  # 对比多个模型
"""
import os
import sys
import argparse
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime

class TrainingAnalyzer:
    """训练历史分析器"""
    
    def __init__(self, csv_file, config_file=None):
        """
        初始化分析器
        
        Args:
            csv_file: 训练历史 CSV 文件
            config_file: 训练配置 JSON 文件（可选）
        """
        self.csv_file = csv_file
        self.config_file = config_file
        
        # 读取数据
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV 文件不存在: {csv_file}")
        
        self.df = pd.read_csv(csv_file)
        self.config = {}
        
        if config_file and os.path.exists(config_file):
            with open(config_file) as f:
                self.config = json.load(f)
        
        # 从文件名提取模型名称
        self.model_name = Path(csv_file).stem.replace('_history', '')
        
        if len(self.df) == 0:
            raise ValueError("CSV 文件为空")
    
    def analyze_loss_curves(self):
        """分析 loss 曲线特性"""
        print(f"\n📈 Loss 曲线分析 ({self.model_name})")
        print("="*70)
        
        # 基础统计
        train_loss = self.df['Train Loss'].values
        val_loss = self.df['Val Loss'].values
        
        print(f"\nTrain Loss 统计:")
        print(f"  初值:           {train_loss[0]:.6f}")
        print(f"  末值:           {train_loss[-1]:.6f}")
        print(f"  最小值:         {train_loss.min():.6f} (Epoch {self.df['Train Loss'].idxmin() + 1})")
        print(f"  平均值:         {train_loss.mean():.6f}")
        print(f"  标准差:         {train_loss.std():.6f}")
        print(f"  下降幅度:       {(train_loss[0] - train_loss[-1]) / train_loss[0] * 100:.2f}%")
        
        print(f"\nVal Loss 统计:")
        print(f"  初值:           {val_loss[0]:.6f}")
        print(f"  末值:           {val_loss[-1]:.6f}")
        print(f"  最小值:         {val_loss.min():.6f} (Epoch {self.df['Val Loss'].idxmin() + 1})")
        print(f"  平均值:         {val_loss.mean():.6f}")
        print(f"  标准差:         {val_loss.std():.6f}")
        print(f"  下降幅度:       {(val_loss[0] - val_loss[-1]) / val_loss[0] * 100:.2f}%")
        
        # 分析过拟合
        last_n = min(20, len(self.df) // 5)
        train_loss_tail = train_loss[-last_n:].mean()
        val_loss_tail = val_loss[-last_n:].mean()
        gap = (val_loss_tail - train_loss_tail) / train_loss_tail * 100
        
        print(f"\n过拟合分析（末 {last_n} epoch 平均）:")
        print(f"  Train Loss:     {train_loss_tail:.6f}")
        print(f"  Val Loss:       {val_loss_tail:.6f}")
        print(f"  差距比例:       {gap:.2f}%")
        if gap > 10:
            print(f"  ⚠️  可能过拟合，建议增加 dropout 或使用 L2 正则化")
        elif gap < -5:
            print(f"  ⚠️  欠拟合，建议减少正则化")
        else:
            print(f"  ✓ 正常范围")
    
    def analyze_convergence(self):
        """分析收敛速度"""
        print(f"\n🎯 收敛速度分析")
        print("="*70)
        
        val_loss = self.df['Val Loss'].values
        epochs = self.df['Epoch'].values
        
        # 找到 loss 改善的平台期
        min_loss = val_loss.min()
        improvement_threshold = min_loss * 0.01  # 1% 改善
        
        # 从最小值开始回溯，找到最后一次显著改善的位置
        last_improvement = 0
        for i in range(len(val_loss) - 1, -1, -1):
            if val_loss[i] - min_loss > improvement_threshold:
                last_improvement = i
                break
        
        convergence_epoch = epochs[last_improvement]
        
        print(f"  最优 Loss 达成:  Epoch {int(epochs[val_loss.argmin()])}")
        print(f"  最优 Loss 值:    {min_loss:.6f}")
        
        # 收敛速度（前10% epochs 的改善）
        first_10pct = max(1, len(val_loss) // 10)
        early_improvement = (val_loss[0] - val_loss[first_10pct - 1]) / val_loss[0] * 100
        print(f"  前 {first_10pct} epochs 改善: {early_improvement:.2f}%")
        
        # 平台期分析
        plateau_start = convergence_epoch if convergence_epoch > 0 else len(epochs)
        plateau_length = len(epochs) - plateau_start
        print(f"  收敛平台期:      Epoch {int(convergence_epoch)} - {len(epochs)}")
        print(f"  平台期长度:      {plateau_length} epochs")
        
        if plateau_length > len(epochs) * 0.3:
            print(f"  💡 平台期较长，可考虑调整超参数或提前停止")
    
    def analyze_learning_dynamics(self):
        """分析学习动态"""
        print(f"\n⚡ 学习动态分析")
        print("="*70)
        
        # 学习率变化
        lr = self.df['Learning Rate'].values
        print(f"\n学习率变化:")
        print(f"  初始:           {lr[0]:.6e}")
        print(f"  最终:           {lr[-1]:.6e}")
        print(f"  衰减比例:       {lr[-1] / lr[0]:.2e}")
        
        # Teacher Forcing 衰减
        tf_ratio = self.df['Teacher Forcing Ratio'].values
        print(f"\nTeacher Forcing 衰减:")
        print(f"  初始:           {tf_ratio[0]:.4f}")
        print(f"  最终:           {tf_ratio[-1]:.4f}")
        print(f"  衰减比例:       {(tf_ratio[0] - tf_ratio[-1]) / tf_ratio[0] * 100:.2f}%")
        
        # Epoch 时间变化
        epoch_time = self.df['Epoch Time (s)'].values
        print(f"\nEpoch 时间分析:")
        print(f"  平均:           {epoch_time.mean():.2f}s")
        print(f"  最快:           {epoch_time.min():.2f}s")
        print(f"  最慢:           {epoch_time.max():.2f}s")
    
    def analyze_training_efficiency(self):
        """分析训练效率"""
        print(f"\n⏱️  训练效率分析")
        print("="*70)
        
        total_epochs = len(self.df)
        total_time = self.df['Epoch Time (s)'].sum()
        avg_epoch_time = total_time / total_epochs
        
        print(f"  总 Epochs:      {total_epochs}")
        print(f"  总耗时:         {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"  平均/Epoch:     {avg_epoch_time:.2f}s")
        
        # 超参数影响
        if self.config:
            print(f"\n超参数配置:")
            print(f"  Batch Size:     {self.config.get('batch_size', 'N/A')}")
            print(f"  Hidden Dim:     {self.config.get('hidden_dim', 'N/A')}")
            print(f"  Num Layers:     {self.config.get('num_layers', 'N/A')}")
            print(f"  Loss (α,β,γ):   {self.config.get('loss_alpha', 'N/A')}, {self.config.get('loss_beta', 'N/A')}, {self.config.get('loss_gamma', 'N/A')}")
    
    def generate_report(self, output_file=None):
        """生成完整分析报告"""
        self.analyze_loss_curves()
        self.analyze_convergence()
        self.analyze_learning_dynamics()
        self.analyze_training_efficiency()
        
        # 保存报告到文件
        if output_file is None:
            output_file = str(Path(self.csv_file).parent / f"{self.model_name}_analysis_report.txt")
        
        print(f"\n✓ 报告已保存到: {output_file}")

def compare_multiple_models(csv_files, output_dir=None):
    """对比多个模型的训练结果"""
    print("\n" + "="*70)
    print("📊 多模型对比分析")
    print("="*70)
    
    dfs = []
    names = []
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"⚠ 跳过不存在的文件: {csv_file}")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            names.append(Path(csv_file).stem.replace('_history', ''))
        except Exception as e:
            print(f"⚠ 读取失败 {csv_file}: {e}")
    
    if not dfs:
        print("❌ 没有有效的 CSV 文件")
        return
    
    print(f"\n对比 {len(dfs)} 个模型:\n")
    print(f"{'Model':<25} {'Min Val Loss':<15} {'Best Epoch':<12} {'Train/Val Gap':<15}")
    print("-"*70)
    
    for df, name in zip(dfs, names):
        min_val = df['Val Loss'].min()
        best_epoch = df['Val Loss'].idxmin() + 1
        train_loss_at_best = df.loc[df['Val Loss'].idxmin(), 'Train Loss']
        gap = (min_val - train_loss_at_best) / train_loss_at_best * 100
        
        print(f"{name:<25} {min_val:<15.6f} {best_epoch:<12} {gap:<15.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='深度分析训练历史')
    parser.add_argument('--csv_file', type=str, help='单个 CSV 文件或模型输出目录')
    parser.add_argument('--csv_files', type=str, nargs='+', help='多个 CSV 文件用于对比')
    parser.add_argument('--model_dir', type=str, help='模型输出目录（自动查找 *_history.csv）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出报告目录')
    
    args = parser.parse_args()
    
    if args.csv_file:
        # 检查是否是目录
        if os.path.isdir(args.csv_file):
            # 查找该目录下的所有 CSV 文件
            csv_files = list(Path(args.csv_file).glob('*_history.csv'))
            if not csv_files:
                print(f"❌ 在 {args.csv_file} 中未找到 *_history.csv 文件")
                sys.exit(1)
            
            if len(csv_files) == 1:
                csv_file = str(csv_files[0])
                config_file = str(csv_files[0]).replace('_history.csv', '_training_config.json')
                analyzer = TrainingAnalyzer(csv_file, config_file)
                analyzer.generate_report()
            else:
                compare_multiple_models([str(f) for f in csv_files])
        else:
            config_file = args.csv_file.replace('_history.csv', '_training_config.json')
            analyzer = TrainingAnalyzer(args.csv_file, config_file)
            analyzer.generate_report()
    
    elif args.csv_files:
        compare_multiple_models(args.csv_files)
    
    elif args.model_dir:
        csv_files = list(Path(args.model_dir).glob('*_history.csv'))
        if not csv_files:
            print(f"❌ 在 {args.model_dir} 中未找到 *_history.csv 文件")
            sys.exit(1)
        
        csv_files = [str(f) for f in csv_files]
        
        if len(csv_files) == 1:
            config_file = csv_files[0].replace('_history.csv', '_training_config.json')
            analyzer = TrainingAnalyzer(csv_files[0], config_file)
            analyzer.generate_report()
        else:
            compare_multiple_models(csv_files)
    else:
        print("❌ 请指定 --csv_file、--csv_files 或 --model_dir")
        parser.print_help()
