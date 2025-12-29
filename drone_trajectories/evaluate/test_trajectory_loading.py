#!/usr/bin/env python3
"""快速测试轨迹加载"""
import pandas as pd
import numpy as np
from pathlib import Path

test_dir = Path(r"D:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories")
csv_files = list(test_dir.glob("*.csv"))[:3]

print(f"找到 {len(csv_files)} 个测试文件\n")

for csv_file in csv_files:
    print(f"[{csv_file.name}]")
    df = pd.read_csv(csv_file)
    print(f"  原始列名: {df.columns.tolist()}")
    
    # 规范化列名
    df.columns = [col.strip().lower() for col in df.columns]
    print(f"  规范化后: {df.columns.tolist()}")
    
    # 尝试检测
    if all(col in df.columns for col in ['tx', 'ty', 'tz']):
        print(f"  ✓ 匹配 tx/ty/tz")
        trajectory = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    elif all(col in df.columns for col in ['x', 'y', 'z']):
        print(f"  ✓ 匹配 x/y/z")
        trajectory = df[['x', 'y', 'z']].values.astype(np.float32)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'timestamp']
        print(f"  数值列: {numeric_cols}")
        if len(numeric_cols) >= 3:
            print(f"  ✓ 自动检测前 3 列: {numeric_cols[:3]}")
            trajectory = df[numeric_cols[:3]].values.astype(np.float32)
        else:
            print(f"  ✗ 无法检测坐标列")
            continue
    
    print(f"  轨迹形状: {trajectory.shape}")
    print(f"  首行数据: {trajectory[0]}")
    print()
