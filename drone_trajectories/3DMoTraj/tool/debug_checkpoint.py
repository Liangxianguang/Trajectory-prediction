#!/usr/bin/env python3
import torch
import os

model_path = 'saved_models/lbebm3D_scene1.pt'

if not os.path.exists(model_path):
    print(f"❌ 模型文件不存在: {model_path}")
    exit(1)

print("=" * 80)
print("检查点结构分析")
print("=" * 80)

checkpoint = torch.load(model_path, map_location='cpu')
state_dict = checkpoint['model_state_dict']

print(f"\n总参数数: {len(state_dict)}\n")

# 按模块分组显示
modules = {}
for key in sorted(state_dict.keys()):
    module_name = key.split('.')[0]
    if module_name not in modules:
        modules[module_name] = []
    modules[module_name].append(key)

# 详细显示encoder_past, encoder_dest, predictor等关键模块
for module_name in ['encoder_past', 'encoder_dest', 'encoder_latent', 'decoder_z', 'decoder_x', 'decoder_y', 'predictor_z', 'predictor_x', 'predictor_y']:
    if module_name in modules:
        print(f"\n{module_name}:")
        for key in modules[module_name]:
            w = state_dict[key]
            print(f"  {key}: {w.shape}")
