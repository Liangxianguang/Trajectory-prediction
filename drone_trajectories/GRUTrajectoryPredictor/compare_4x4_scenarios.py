#!/usr/bin/env python3
"""
4×4 场景对比脚本：4个特定场景 × 4个观测维度
=================================================================================
行 (场景): 4个特定场景，各展示不同预测能力
列 (维度): 3D全景、XY平面、XZ平面、YZ平面

场景设定：
  Row 1 (S1): 复杂交互场景中的空间建模能力（轨迹交叉场景）              - Sample 20280
  Row 2 (S2): 高曲率机动中的物理一致性（协同急转弯场景）                - Sample 173142
  Row 3 (S3): 三维机动场景中的高度预测能力（快速垂直爬升场景）          - Sample 212515
  Row 4 (S4): 复杂周期机动中的时序建模能力（S形轨迹）                    - Sample 33

输出:
  - 4x4_scenario_comparison.png (4×4 大图)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D

# 这些配置应与 compare_four_models_image.py 保持一致
SCENARIO_SAMPLES = {
    'S1_Spatial': {
        'sample_idx': 20280,
        'title': 'S1: 轨迹交叉场景\n(空间建模能力)'
    },
    'S2_Curvature': {
        'sample_idx': 173142,
        'title': 'S2: 急转弯场景\n(物理一致性)'
    },
    'S3_Vertical': {
        'sample_idx': 212515,
        'title': 'S3: 垂直爬升场景\n(高度预测能力)'
    },
    'S4_Periodic': {
        'sample_idx': 33,
        'title': 'S4: S形轨迹\n(时序建模能力)'
    },
}

PAPER_STYLE = {
    "palette": {
        "history": "#F75A5AFF",        # 历史轨迹
        "gt": "#000000FF",             # GT: 纯黑
        "mrgraj": "#FF9500FF",         # MRGTraj: 鲜橙
        "3dmotraj": "#53F50EFF",       # 3DMoTraj: 绿色
        "exp5": "#9933CCFF",           # VECTOR: 紫色
        "swarm_gru": "#0078FFFF",      # Ours: 蓝色
    },
    "linestyles": {
        "history": "-",
        "gt": "-",
        "mrgraj": (0, (2, 2)),
        "3dmotraj": (0, (2, 2)),
        "exp5": (0, (2, 2)),
        "swarm_gru": "-",
    },
    "linewidth": {
        "gt": 2.0,
        "swarm_gru": 3.0,
        "others": 2.0,
    },
    "markersize": 8,
}


def plot_3d_view(ax, x_sample, y_sample, pred_mrgraj, pred_3dmotraj, pred_exp5, pred_swarm, colors, lw_map):
    """绘制 3D 全景视图"""
    num_agents = x_sample.shape[1]
    
    for aid in range(num_agents):
        # 历史轨迹
        ax.plot(x_sample[:, aid, 0], x_sample[:, aid, 1], x_sample[:, aid, 2],
                color=colors['history'], linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)
        
        last_pt = x_sample[-1:, aid, :]
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        
        # GT
        ax.plot(gt_full[:, 0], gt_full[:, 1], gt_full[:, 2],
                color=colors['gt'], linestyle='-', linewidth=lw_map['gt'], alpha=0.95, zorder=10)
        
        # 预测轨迹
        for pred, color_key in [(pred_mrgraj, 'mrgraj'), (pred_3dmotraj, '3dmotraj'), 
                                (pred_exp5, 'exp5'), (pred_swarm, 'swarm_gru')]:
            pred_full = np.vstack([last_pt, pred[:, aid, :]])
            ax.plot(pred_full[:, 0], pred_full[:, 1], pred_full[:, 2],
                   color=colors[color_key], linestyle='--' if color_key != 'swarm_gru' else '-',
                   linewidth=lw_map.get(color_key, 2.0), alpha=0.85, zorder=5)
    
    ax.set_xlabel('X', fontsize=9, fontweight='bold')
    ax.set_ylabel('Y', fontsize=9, fontweight='bold')
    ax.set_zlabel('Z', fontsize=9, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.view_init(elev=25, azim=-60)


def plot_xy_view(ax, x_sample, y_sample, pred_mrgraj, pred_3dmotraj, pred_exp5, pred_swarm, colors, lw_map):
    """绘制 XY 平面视图"""
    num_agents = x_sample.shape[1]
    
    for aid in range(num_agents):
        ax.plot(x_sample[:, aid, 0], x_sample[:, aid, 1],
                color=colors['history'], linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)
        
        last_pt = x_sample[-1:, aid, :]
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        
        ax.plot(gt_full[:, 0], gt_full[:, 1],
                color=colors['gt'], linestyle='-', linewidth=lw_map['gt'], alpha=0.95, zorder=10)
        
        for pred, color_key in [(pred_mrgraj, 'mrgraj'), (pred_3dmotraj, '3dmotraj'),
                                (pred_exp5, 'exp5'), (pred_swarm, 'swarm_gru')]:
            pred_full = np.vstack([last_pt, pred[:, aid, :]])
            ax.plot(pred_full[:, 0], pred_full[:, 1],
                   color=colors[color_key], linestyle='--' if color_key != 'swarm_gru' else '-',
                   linewidth=lw_map.get(color_key, 2.0), alpha=0.85, zorder=5)
    
    ax.set_xlabel('X (m)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=9, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_aspect('equal', adjustable='box')


def plot_xz_view(ax, x_sample, y_sample, pred_mrgraj, pred_3dmotraj, pred_exp5, pred_swarm, colors, lw_map):
    """绘制 XZ 平面视图"""
    num_agents = x_sample.shape[1]
    
    for aid in range(num_agents):
        ax.plot(x_sample[:, aid, 0], x_sample[:, aid, 2],
                color=colors['history'], linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)
        
        last_pt = x_sample[-1:, aid, :]
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        
        ax.plot(gt_full[:, 0], gt_full[:, 2],
                color=colors['gt'], linestyle='-', linewidth=lw_map['gt'], alpha=0.95, zorder=10)
        
        for pred, color_key in [(pred_mrgraj, 'mrgraj'), (pred_3dmotraj, '3dmotraj'),
                                (pred_exp5, 'exp5'), (pred_swarm, 'swarm_gru')]:
            pred_full = np.vstack([last_pt, pred[:, aid, :]])
            ax.plot(pred_full[:, 0], pred_full[:, 2],
                   color=colors[color_key], linestyle='--' if color_key != 'swarm_gru' else '-',
                   linewidth=lw_map.get(color_key, 2.0), alpha=0.85, zorder=5)
    
    ax.set_xlabel('X (m)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Z (m)', fontsize=9, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_aspect('equal', adjustable='box')


def plot_yz_view(ax, x_sample, y_sample, pred_mrgraj, pred_3dmotraj, pred_exp5, pred_swarm, colors, lw_map):
    """绘制 YZ 平面视图"""
    num_agents = x_sample.shape[1]
    
    for aid in range(num_agents):
        ax.plot(x_sample[:, aid, 1], x_sample[:, aid, 2],
                color=colors['history'], linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)
        
        last_pt = x_sample[-1:, aid, :]
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        
        ax.plot(gt_full[:, 1], gt_full[:, 2],
                color=colors['gt'], linestyle='-', linewidth=lw_map['gt'], alpha=0.95, zorder=10)
        
        for pred, color_key in [(pred_mrgraj, 'mrgraj'), (pred_3dmotraj, '3dmotraj'),
                                (pred_exp5, 'exp5'), (pred_swarm, 'swarm_gru')]:
            pred_full = np.vstack([last_pt, pred[:, aid, :]])
            ax.plot(pred_full[:, 1], pred_full[:, 2],
                   color=colors[color_key], linestyle='--' if color_key != 'swarm_gru' else '-',
                   linewidth=lw_map.get(color_key, 2.0), alpha=0.85, zorder=5)
    
    ax.set_xlabel('Y (m)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Z (m)', fontsize=9, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_aspect('equal', adjustable='box')


def create_4x4_comparison(scenarios_data, output_path):
    """
    创建 4×4 场景对比图
    
    scenarios_data: dict, 每个场景包含 x_sample, y_sample, pred_* 数据
    """
    colors = PAPER_STYLE["palette"]
    lw_map = PAPER_STYLE["linewidth"]
    
    fig = plt.figure(figsize=(40, 32), facecolor='white')
    
    # 创建 4×4 的子图网格
    gs = fig.add_gridspec(4, 4, left=0.05, right=0.98, top=0.96, bottom=0.04,
                         hspace=0.35, wspace=0.3)
    
    # 列标题
    col_titles = ['3D View\n(全景轨迹)', 'XY Plane\n(水平交互)', 
                  'XZ Plane\n(高度机动)', 'YZ Plane\n(侧向机动)']
    
    # 行标题（场景名称）
    row_titles = ['S1: 空间建模', 'S2: 物理一致', 'S3: 高度预测', 'S4: 时序建模']
    
    # 绘制列标题
    for col_idx, col_title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col_idx]) if col_idx == 0 else None
        if ax is None:
            ax = fig.add_subplot(gs[0, col_idx])
        ax.text(0.5, 0.5, col_title, ha='center', va='center', fontsize=14,
               fontweight='bold', transform=ax.transAxes)
        ax.axis('off')
    
    # 绘制 4×4 网格
    scenario_keys = list(scenarios_data.keys())
    
    for row_idx, scenario_key in enumerate(scenario_keys):
        scenario = scenarios_data[scenario_key]
        
        # 行标题
        ax_title = fig.add_subplot(gs[row_idx+1, 0])
        ax_title.text(0.5, 0.5, f"{row_titles[row_idx]}\n(#{scenario['sample_idx']})", 
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     transform=ax_title.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_title.axis('off')
        
        # 四个视图
        # 3D 视图
        ax = fig.add_subplot(gs[row_idx+1, 0], projection='3d')
        plot_3d_view(ax, scenario['x_sample'], scenario['y_sample'],
                    scenario['pred_mrgraj'], scenario['pred_3dmotraj'],
                    scenario['pred_exp5'], scenario['pred_swarm'], colors, lw_map)
        
        # XY 平面
        ax = fig.add_subplot(gs[row_idx+1, 1])
        plot_xy_view(ax, scenario['x_sample'], scenario['y_sample'],
                    scenario['pred_mrgraj'], scenario['pred_3dmotraj'],
                    scenario['pred_exp5'], scenario['pred_swarm'], colors, lw_map)
        
        # XZ 平面
        ax = fig.add_subplot(gs[row_idx+1, 2])
        plot_xz_view(ax, scenario['x_sample'], scenario['y_sample'],
                    scenario['pred_mrgraj'], scenario['pred_3dmotraj'],
                    scenario['pred_exp5'], scenario['pred_swarm'], colors, lw_map)
        
        # YZ 平面
        ax = fig.add_subplot(gs[row_idx+1, 3])
        plot_yz_view(ax, scenario['x_sample'], scenario['y_sample'],
                    scenario['pred_mrgraj'], scenario['pred_3dmotraj'],
                    scenario['pred_exp5'], scenario['pred_swarm'], colors, lw_map)
    
    # 添加总标题和图例
    fig.suptitle('4×4 Multi-Scenario Trajectory Prediction Comparison\n' + 
                '4 Scenarios × 4 Viewpoints | Publication-Grade Visualization',
                fontsize=18, fontweight='bold', y=0.98)
    
    # 添加底部图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors['history'], linewidth=2, label='History'),
        Line2D([0], [0], color=colors['gt'], linewidth=2.5, label='GT'),
        Line2D([0], [0], color=colors['mrgraj'], linewidth=2, linestyle='--', label='MRGTraj'),
        Line2D([0], [0], color=colors['3dmotraj'], linewidth=2, linestyle='--', label='3DMoTraj'),
        Line2D([0], [0], color=colors['exp5'], linewidth=2, linestyle='--', label='VECTOR'),
        Line2D([0], [0], color=colors['swarm_gru'], linewidth=2.5, label='Ours (SwarmGRU)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=12,
              frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ 4×4 对比图已保存: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("4×4 场景对比脚本")
    print("="*70)
    print("\n场景定义:")
    for key, cfg in SCENARIO_SAMPLES.items():
        print(f"  {key}: Sample #{cfg['sample_idx']} - {cfg['title']}")
    print("\n注意: 此脚本需要与 compare_four_models_image.py 集成")
    print("当前脚本仅展示布局结构")
