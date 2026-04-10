#!/usr/bin/env python3
"""
生成 4×4 场景对比图脚本
4 个特定场景 × 4 个观测维度 (3D + XY + XZ + YZ)

场景定义：
  S1 (行1): 复杂交互场景中的空间建模能力（轨迹交叉场景）              - Sample 20280
  S2 (行2): 高曲率机动中的物理一致性（协同急转弯场景）                - Sample 173142
  S3 (行3): 三维机动场景中的高度预测能力（快速垂直爬升场景）          - Sample 212515
  S4 (行4): 复杂周期机动中的时序建模能力（S形轨迹）                    - Sample 33
"""

import argparse
import json
import logging
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# 配置路径
current_dir = Path(__file__).resolve().parent
workspace_root = Path("D:\\Trajectory prediction")
cluster_traj_dir = workspace_root / "drone_trajectories" / "Cluster trajectory"
tool_dir = workspace_root / "drone_trajectories" / "3DMoTraj" / "tool"
mrgraj_dir = workspace_root / "drone_trajectories" / "MRGTraj-main"

sys.path.insert(0, str(cluster_traj_dir))
sys.path.insert(0, str(tool_dir))
sys.path.insert(0, str(mrgraj_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 样本定义
SCENARIO_CONFIG = {
    'S1': {'sample_idx': 20280, 'title': 'S1: 空间建模能力\n(轨迹交叉场景)', 'desc': 'Spatial Modeling in Crossing'},
    'S2': {'sample_idx': 173142, 'title': 'S2: 物理一致性\n(协同急转弯场景)', 'desc': 'Physical Consistency in Sharp Turn'},
    'S3': {'sample_idx': 212515, 'title': 'S3: 高度预测能力\n(垂直爬升场景)', 'desc': 'Altitude Prediction in Vertical Climb'},
    'S4': {'sample_idx': 33, 'title': 'S4: 时序建模能力\n(S形轨迹)', 'desc': 'Temporal Modeling in S-shape'},
}

# 颜色和样式配置
PAPER_STYLE = {
    "palette": {
        "history": "#F75A5AFF",
        "gt": "#000000FF",
        "mrgraj": "#FF9500FF",
        "3dmotraj": "#53F50EFF",
        "exp5": "#9933CCFF",
        "swarm_gru": "#0078FFFF",
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
        "gt": 1.8,
        "swarm_gru": 2.2,
        "others": 1.5,
    },
    "markersize": 6,
}


def load_predictions_from_json(json_path):
    """从 JSON 文件加载所有预测结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_scenario_data(all_results, sample_idx, num_agents=3):
    """从总结果中提取特定场景的数据"""
    scenario_data = {}
    
    # 查找该样本的预测结果
    for result in all_results.get('samples', []):
        if result['sample_idx'] == sample_idx:
            # 从各个模型提取预测
            scenario_data['x_sample'] = np.array(result.get('x_sample', []))
            scenario_data['y_sample'] = np.array(result.get('y_sample', []))
            scenario_data['pred_mrgraj'] = np.array(result.get('pred_mrgraj', []))
            scenario_data['pred_3dmotraj'] = np.array(result.get('pred_lbebm', []))
            scenario_data['pred_exp5'] = np.array(result.get('pred_exp5', []))
            scenario_data['pred_swarm_gru'] = np.array(result.get('pred_swarm_gru', []))
            return scenario_data
    
    logger.warning(f"样本 {sample_idx} 未找到")
    return None


def plot_3d_view(ax, x_sample, y_sample, pred_dict, colors, lw_map):
    """绘制 3D 全景视图"""
    num_agents = min(3, x_sample.shape[1])
    
    # 调整视角和背景
    ax.view_init(elev=22, azim=-65)
    ax.set_box_aspect([1, 1, 0.8])  # 调整 Z 轴显示比例
    
    for aid in range(num_agents):
        # 1. 过去真实轨迹 (History)
        ax.plot(x_sample[:, aid, 0], x_sample[:, aid, 1], x_sample[:, aid, 2],
                color=colors['history'], linestyle='-', linewidth=1.5, alpha=0.6, zorder=1)
        
        # 获取最后一个历史点作为起点
        last_pt = x_sample[-1:, aid, :]
        
        # 2. 未来真实轨迹 (Ground Truth)
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        ax.plot(gt_full[:, 0], gt_full[:, 1], gt_full[:, 2],
                color=colors['gt'], linestyle='-', linewidth=lw_map['gt'], 
                alpha=0.9, zorder=10)
        
        # 3. 各模型预测轨迹
        for model_key, color_key in [('pred_mrgraj', 'mrgraj'), 
                                      ('pred_3dmotraj', '3dmotraj'),
                                      ('pred_exp5', 'exp5'), 
                                      ('pred_swarm_gru', 'swarm_gru')]:
            if model_key in pred_dict and pred_dict[model_key] is not None:
                pred = pred_dict[model_key]
                if len(pred) > 0:
                    pred_full = np.vstack([last_pt, pred[:, aid, :]])
                    linestyle = '-' if color_key == 'swarm_gru' else '--'
                    ax.plot(pred_full[:, 0], pred_full[:, 1], pred_full[:, 2],
                           color=colors[color_key], linestyle=linestyle,
                           linewidth=lw_map.get(color_key, lw_map['others']),
                           alpha=0.85, zorder=5)
    
    ax.set_xlabel('X (m)', fontsize=8, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=8, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.view_init(elev=25, azim=-60)
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_fontsize(7)


def plot_2d_view(ax, x_sample, y_sample, pred_dict, colors, lw_map, plane='xy'):
    """绘制 2D 平面视图 (xy, xz, yz)"""
    num_agents = min(3, x_sample.shape[1])
    
    # 确定坐标映射
    if plane == 'xy':
        x_idx, y_idx = 0, 1
    elif plane == 'xz':
        x_idx, y_idx = 0, 2
    else:  # yz
        x_idx, y_idx = 1, 2
    
    for aid in range(num_agents):
        # 1. 过去真实轨迹 (History)
        ax.plot(x_sample[:, aid, x_idx], x_sample[:, aid, y_idx],
                color=colors['history'], linestyle='-', linewidth=1.5, alpha=0.6, zorder=1)
        
        # 获取最后一个历史点作为起点
        last_pt = x_sample[-1:, aid, :]
        
        # 2. 未来真实轨迹 (Ground Truth)
        gt_full = np.vstack([last_pt, y_sample[:, aid, :]])
        ax.plot(gt_full[:, x_idx], gt_full[:, y_idx],
                color=colors['gt'], linestyle='-', linewidth=lw_map['gt'],
                alpha=0.9, zorder=10)
        
        # 3. 各模型预测轨迹
        for model_key, color_key in [('pred_mrgraj', 'mrgraj'),
                                      ('pred_3dmotraj', '3dmotraj'),
                                      ('pred_exp5', 'exp5'),
                                      ('pred_swarm_gru', 'swarm_gru')]:
            if model_key in pred_dict and pred_dict[model_key] is not None:
                pred = pred_dict[model_key]
                if len(pred) > 0:
                    pred_full = np.vstack([last_pt, pred[:, aid, :]])
                    linestyle = '-' if color_key == 'swarm_gru' else '--'
                    ax.plot(pred_full[:, x_idx], pred_full[:, y_idx],
                           color=colors[color_key], linestyle=linestyle,
                           linewidth=lw_map.get(color_key, lw_map['others']),
                           alpha=0.85, zorder=5)
    
    # 标签
    labels = {'xy': ('X (m)', 'Y (m)'), 'xz': ('X (m)', 'Z (m)'), 'yz': ('Y (m)', 'Z (m)')}
    ax.set_xlabel(labels[plane][0], fontsize=8, fontweight='bold')
    ax.set_ylabel(labels[plane][1], fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=7)


def create_4x4_layout(scenarios_dict, output_path):
    """创建 4×4 对比布局"""
    colors = PAPER_STYLE["palette"]
    lw_map = PAPER_STYLE["linewidth"]
    
    fig = plt.figure(figsize=(44, 36), facecolor='white', dpi=100)
    
    # GridSpec: 4 行 4 列，每行左侧加行标题
    gs = fig.add_gridspec(4, 4, left=0.06, right=0.96, top=0.94, bottom=0.06,
                         hspace=0.4, wspace=0.35)
    
    # 列标题
    col_titles = ['3D View', 'XY Plane', 'XZ Plane', 'YZ Plane']
    
    # 绘制每个场景的 4 个视图
    for row_idx, (scenario_key, scenario_cfg) in enumerate(SCENARIO_CONFIG.items()):
        scenario_data = scenarios_dict.get(scenario_key)
        
        if scenario_data is None:
            logger.warning(f"场景 {scenario_key} 数据缺失")
            continue
        
        # 3D 视图
        ax_3d = fig.add_subplot(gs[row_idx, 0], projection='3d')
        plot_3d_view(ax_3d, scenario_data['x_sample'], scenario_data['y_sample'],
                    {k: scenario_data.get(k) for k in ['pred_mrgraj', 'pred_3dmotraj', 
                                                         'pred_exp5', 'pred_swarm_gru']},
                    colors, lw_map)
        ax_3d.set_title(scenario_cfg['title'], fontsize=11, fontweight='bold', pad=10)
        
        # XY 平面
        ax_xy = fig.add_subplot(gs[row_idx, 1])
        plot_2d_view(ax_xy, scenario_data['x_sample'], scenario_data['y_sample'],
                    {k: scenario_data.get(k) for k in ['pred_mrgraj', 'pred_3dmotraj',
                                                         'pred_exp5', 'pred_swarm_gru']},
                    colors, lw_map, plane='xy')
        
        # XZ 平面
        ax_xz = fig.add_subplot(gs[row_idx, 2])
        plot_2d_view(ax_xz, scenario_data['x_sample'], scenario_data['y_sample'],
                    {k: scenario_data.get(k) for k in ['pred_mrgraj', 'pred_3dmotraj',
                                                         'pred_exp5', 'pred_swarm_gru']},
                    colors, lw_map, plane='xz')
        
        # YZ 平面
        ax_yz = fig.add_subplot(gs[row_idx, 3])
        plot_2d_view(ax_yz, scenario_data['x_sample'], scenario_data['y_sample'],
                    {k: scenario_data.get(k) for k in ['pred_mrgraj', 'pred_3dmotraj',
                                                         'pred_exp5', 'pred_swarm_gru']},
                    colors, lw_map, plane='yz')
        
        # 仅在第一行添加列标题
        if row_idx == 0:
            for col_idx, col_title in enumerate(col_titles):
                ax = fig.add_axes([0.06 + col_idx * 0.225, 0.96, 0.18, 0.02])
                ax.text(0.5, 0.5, col_title, ha='center', va='center',
                       fontsize=11, fontweight='bold', transform=ax.transAxes)
                ax.axis('off')
    
    # 总标题
    fig.suptitle('4×4 Multi-Scenario Trajectory Prediction Comparison\n' +
                '4 Scenarios × 4 Viewpoints (3D + XY + XZ + YZ Projections)',
                fontsize=16, fontweight='bold', y=0.985)
    
    # 图例（底部）
    legend_elements = [
        Line2D([0], [0], color=colors['history'], linewidth=1.5, label='History Trajectory'),
        Line2D([0], [0], color=colors['gt'], linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color=colors['mrgraj'], linewidth=1.8, linestyle='--', label='MRGTraj'),
        Line2D([0], [0], color=colors['3dmotraj'], linewidth=1.8, linestyle='--', label='3DMoTraj'),
        Line2D([0], [0], color=colors['exp5'], linewidth=1.8, linestyle='--', label='VECTOR'),
        Line2D([0], [0], color=colors['swarm_gru'], linewidth=2.2, label='Ours (SwarmGRU)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=11,
              frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.01))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    logger.info(f"✓ 4×4 对比图已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='生成 4×4 场景对比图')
    
    parser.add_argument('--json_results', required=True, 
                       help='汇总的JSON结果文件路径')
    parser.add_argument('--output_dir', default='comparison_4x4_scenarios',
                       help='输出目录')
    parser.add_argument('--output_name', default='4x4_scenario_comparison.png',
                       help='输出文件名')
    
    args = parser.parse_args()
    
    # 加载 JSON 结果
    json_path = Path(args.json_results)
    if not json_path.exists():
        logger.error(f"JSON 文件不存在: {json_path}")
        sys.exit(1)
    
    logger.info(f"加载结果文件: {json_path}")
    all_results = load_predictions_from_json(json_path)
    
    # 提取各场景数据
    scenarios_dict = {}
    for scenario_key, scenario_cfg in SCENARIO_CONFIG.items():
        logger.info(f"提取场景 {scenario_key} (样本 #{scenario_cfg['sample_idx']})...")
        scenario_data = extract_scenario_data(all_results, scenario_cfg['sample_idx'])
        if scenario_data is not None:
            scenarios_dict[scenario_key] = scenario_data
        else:
            logger.warning(f"无法提取场景 {scenario_key} 的数据")
    
    if len(scenarios_dict) < 4:
        logger.error("无法获取足够的场景数据")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成 4×4 布局图
    output_path = output_dir / args.output_name
    logger.info("生成 4×4 对比图...")
    create_4x4_layout(scenarios_dict, str(output_path))
    
    logger.info("="*70)
    logger.info("✓ 完成! 4×4 场景对比图已保存")
    logger.info(f"  输出路径: {output_path}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
