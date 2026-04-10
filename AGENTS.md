# AGENTS.md - AI 编码代理指南

> **注意**：请始终使用简体中文进行回复，除非我明确要求使用其他语言。
> **注意**：涉及编程术语时，请保留英文原文并（在必要时）提供中文解释。

本文件为此代码库中的 AI 编码代理提供编码指南和约定。

## 项目概述

这是一个基于 PyTorch 深度学习模型（GRU）的**无人机轨迹预测**项目。
- 主包：`drone_path_predictor_ros-main`（ROS 2 包）
- 依赖项：NumPy、PyTorch、matplotlib、pandas、pytest

## 构建和测试命令

### 运行测试
```bash
# 运行所有测试
cd drone_path_predictor_ros-main && pytest

# 运行单个测试文件
cd drone_path_predictor_ros-main && pytest test/test_flake8.py

# 运行单个测试
cd drone_path_predictor_ros-main && pytest test/test_flake8.py::test_flake8 -v

# 使用详细输出运行
cd drone_path_predictor_ros-main && pytest -v
```

### 代码检查
```bash
# 运行所有检查器（flake8、pep257、copyright）
cd drone_path_predictor_ros-main && pytest test/

# 仅运行 flake8
cd drone_path_predictor_ros-main && pytest test/test_flake8.py

# 仅运行 pep257（文档字符串风格）
cd drone_path_predictor_ros-main && pytest test/test_pep257.py

# 手动运行 ament_flake8
python -m ament_flake8.main -- .

# 手动运行 ament_pep257
python -m ament_pep257.main -- .
```

## 代码风格指南

### 一般规则
- 使用 **Python 3**（shebang：`#!/usr/bin/env python3`）
- 遵循 **PEP 8** 代码风格指南
- 遵循 **PEP 257** 文档字符串约定
- 使用 4 个空格进行缩进（不使用 Tab）
- 最大行长度：120 个字符

### 导入顺序
- 标准库导入放在最前面
- 第三方导入其次（numpy、torch 等）
- 本地导入放在最后
- 按类型分组导入，组之间用空行分隔
- 示例：
```python
import sys
import time

import numpy as np
import torch
import torch.nn as nn
```

### 命名约定
- **类名**：`CamelCase`（例如 `PositionPredictor`、`Predictor`）
- **函数/变量**：`snake_case`（例如 `load_normalization_parameters`、`predicted_output`）
- **常量**：`UPPER_SNAKE_CASE`（例如 `MAX_SEQUENCE_LENGTH`）
- **私有方法**：使用 `_` 前缀（例如 `_internal_method`）
- 避免使用单字母变量名，循环中除外（`i`、`j`、`k`）

### 类型提示
- 为函数参数和返回值使用类型提示（当有益时）
- 示例：
```python
def load_normalization_parameters(npz_file_path: str) -> tuple:
    """从 npz 文件加载归一化参数。"""
    ...
```

### 文档字符串
- 使用 **PEP 257** 风格（Google 或 NumPy 风格也可接受）
- 包含：描述、参数、返回值、异常
- 示例：
```python
def predict_trajectory(model, sequence, device):
    """
    使用训练好的模型预测轨迹。

    Args:
        model: 训练好的 PyTorch 模型。
        sequence: 输入序列张量。
        device: 计算用的 torch.device。

    Returns:
        numpy.ndarray: 预测的轨迹序列。
    """
```

### 错误处理
- 使用具体的异常类型
- 包含有意义的错误消息
- 示例：
```python
try:
    data = np.load(filename, allow_pickle=True)
except FileNotFoundError:
    raise FileNotFoundError(f"统计文件未找到: {filename}")
```

### PyTorch 最佳实践
- 推理时使用 `torch.no_grad()`
- 推理前调用 `model.eval()`
- 将模型/张量移动到设备：`.to(device)`
- 使用 `torch.tensor(..., dtype=torch.float32)` 创建张量

### 代码组织
- 将相关函数放在一起
- 将工具函数分组到模块级别
- 将类定义分组
- 谨慎使用空行（顶级定义之间最多 2 个空行）

### 日志和打印
- 使用 `print()` 进行简单调试输出
- 生产代码考虑使用 logging
- 避免过多的 print 语句

### ROS 2 特定
- 包名：`drone_path_predictor_ros`
- 节点入口点：`trajectory_predictor_node`
- 使用 ROS 2 启动文件运行节点

## 文件结构

```
drone_path_predictor_ros-main/
├── drone_path_predictor_ros/
│   ├── __init__.py
│   ├── trajectory_predictor.py     # 核心模型和工具函数
│   ├── trajectory_predictor_node.py # ROS 节点
│   ├── pose_buffer.py
│   └── swarm/                      # 蜂群预测子包
├── test/                           # 检查器测试
│   ├── test_flake8.py
│   ├── test_pep257.py
│   └── test_copyright.py
├── launch/                         # ROS 启动文件
├── config/                        # ROS 配置文件
└── setup.py                       # 包设置
```

## 常见模式

### 加载 PyTorch 模型
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(...)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
```

### 归一化/反归一化
```python
def normalize_sequence(sequence, mean, std):
    return (sequence - mean) / std

def denormalize_sequence(sequence, mean, std):
    return (sequence * std) + mean
```

### 运行推理
```python
def predict_trajectory(model, sequence, device):
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_output = model(sequence_tensor)
    return predicted_output.squeeze(0).cpu().numpy()
```

## 版权头

在新 Python 文件中包含以下头：
```python
# Copyright 2017 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
```

## 代理注意事项

- 这主要是轨迹预测的研究/原型代码库
- 主要开发在 `drone_path_predictor_ros-main/` 中进行
- `test/` 中的测试主要是检查器，不是单元测试
- 模型训练脚本通常单独运行（不属于 ROS 包）
