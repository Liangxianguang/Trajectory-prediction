#!/usr/bin/env python3
"""快速测试导入"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("测试从 lbebm3D 导入 LBEBM3D")
print("=" * 70)

try:
    from lbebm3D import LBEBM3D, MLP, SC_LSTM, ReplayMemory
    print("✓ 成功导入所有类!")
    print(f"  - LBEBM3D: {LBEBM3D}")
    print(f"  - MLP: {MLP}")
    print(f"  - SC_LSTM: {SC_LSTM}")
    print(f"  - ReplayMemory: {ReplayMemory}")
    
    print("\n[√] 可以安全运行 visualize_lbebm3d.py")
    
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
