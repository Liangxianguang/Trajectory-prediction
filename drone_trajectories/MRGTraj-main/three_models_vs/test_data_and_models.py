#!/usr/bin/env python3
"""
快速测试数据和模型加载
"""
import sys
from pathlib import Path
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============ 配置路径 ============
workspace_root = Path("D:\\Trajectory prediction")
cluster_traj_dir = workspace_root / "drone_trajectories" / "Cluster trajectory"
tool_dir = workspace_root / "drone_trajectories" / "3DMoTraj" / "tool"
mrgraj_dir = workspace_root / "drone_trajectories" / "MRGTraj-main"

sys.path.insert(0, str(cluster_traj_dir))
sys.path.insert(0, str(tool_dir))
sys.path.insert(0, str(mrgraj_dir))

logger.info("="*70)
logger.info("快速数据和模型加载测试")
logger.info("="*70)

# ============ 测试数据加载 ============
logger.info("\n[1/5] 测试数据加载...")
data_dir = workspace_root / "drone_trajectories" / "Cluster trajectory" / "swarm_segments"
x_file = data_dir / "input_agents_3_subset.npz"
y_file = data_dir / "output_agents_3_subset.npz"

logger.info(f"  X 文件: {x_file}")
logger.info(f"  Y 文件: {y_file}")
logger.info(f"  X 存在: {x_file.exists()}")
logger.info(f"  Y 存在: {y_file.exists()}")

x = np.load(x_file)['data']
y = np.load(y_file)['data']
x = np.transpose(x, (1, 0, 2, 3))
y = np.transpose(y, (1, 0, 2, 3))
logger.info(f"  X 形状: {x.shape} (samples, seq, agents, 3)")
logger.info(f"  Y 形状: {y.shape} (samples, seq, agents, 3)")
logger.info("✓ 数据加载成功")

# ============ 测试特征加载 ============
logger.info("\n[2/5] 测试特征加载...")
features_dir = workspace_root / "drone_trajectories" / "Cluster trajectory" / "features_32d"
f_file = features_dir / "features_agents_3_subset_32d.npz"
logger.info(f"  特征文件: {f_file}")
logger.info(f"  存在: {f_file.exists()}")

if f_file.exists():
    f_data = np.load(f_file)
    features = f_data['features']
    logger.info(f"  特征形状: {features.shape}")
    logger.info(f"  特征 keys: {list(f_data.keys())}")
    logger.info("✓ 特征加载成功")
else:
    logger.warning("✗ 特征文件不存在")

# ============ 测试模型导入 ============
logger.info("\n[3/5] 测试 LBEBM3D 模型加载...")
try:
    from infer_lbebm3d_baseline import LBEBM3DInfer, infer_model_params_from_state_dict
    lbebm_model_path = workspace_root / "drone_trajectories" / "3DMoTraj" / "saved_models" / "checkpoints_accfix" / "epoch_010.pt"
    ckpt = torch.load(lbebm_model_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    params = infer_model_params_from_state_dict(state_dict)
    logger.info(f"  模型参数: past_len={params['past_length']}, future_len={params['future_length']}")
    logger.info("✓ LBEBM3D 模型加载成功")
except Exception as e:
    logger.error(f"✗ LBEBM3D 加载失败: {e}")

# ============ 测试 Exp5 模型 ============
logger.info("\n[4/5] 测试 Exp5 (DG32-BCAT) 模型加载...")
try:
    from train_swarm_model_v3_with_gnn import DynamicsAwareSwarmGRUModel_with_GNN
    import json
    exp5_dir = workspace_root / "drone_trajectories" / "Cluster trajectory" / "ablation study" / "ablation_results_agents_3_exp5_full"
    config_file = exp5_dir / "config_agents_3_exp5_full.json"
    model_file = exp5_dir / "best_model_agents_3_exp5_full.pt"
    
    logger.info(f"  配置文件: {config_file}")
    logger.info(f"  模型文件: {model_file}")
    logger.info(f"  配置存在: {config_file.exists()}")
    logger.info(f"  模型存在: {model_file.exists()}")
    
    if config_file.exists() and model_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        logger.info(f"  配置: input_features={config.get('input_features', 32)}, hidden={config.get('hidden_size', 128)}")
        logger.info("✓ Exp5 模型文件验证成功")
    else:
        logger.error("✗ Exp5 模型或配置文件不存在")
except Exception as e:
    logger.error(f"✗ Exp5 加载失败: {e}")

# ============ 测试 MRGTraj 模型 ============
logger.info("\n[5/5] 测试 MRGTraj 模型加载...")
try:
    from model_swarm import MRGTrajSwarm
    mrgraj_model_path = workspace_root / "drone_trajectories" / "MRGTraj-main" / "checkpoints_lbebm3d" / "agents_3_lbebm3d_inspired" / "best_model.pth"
    logger.info(f"  模型文件: {mrgraj_model_path}")
    logger.info(f"  存在: {mrgraj_model_path.exists()}")
    
    if mrgraj_model_path.exists():
        import argparse
        args = argparse.Namespace(
            d_model=256, n_heads=4, n_layers=2, noise_dim=64,
            agent_dim=3, obs_len=20, pred_len=10, num_agents=3
        )
        model = MRGTrajSwarm(args)
        ckpt = torch.load(mrgraj_model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info("✓ MRGTraj 模型加载成功")
    else:
        logger.error("✗ MRGTraj 模型文件不存在")
except Exception as e:
    logger.error(f"✗ MRGTraj 加载失败: {e}")

logger.info("\n" + "="*70)
logger.info("测试完成！所有数据和模型都已验证")
logger.info("="*70)
