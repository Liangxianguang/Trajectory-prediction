#!/usr/bin/env python3
"""
测试三模型导入
"""
import sys
import logging
from pathlib import Path

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

logger.info("Python 路径已配置:")
for i, p in enumerate(sys.path[:5]):
    logger.info(f"  [{i}] {p}")

# ============ 测试导入 LBEBM3D ============
logger.info("\n[1/3] 测试 LBEBM3D 导入...")
try:
    from infer_lbebm3d_baseline import LBEBM3DInfer, infer_model_params_from_state_dict
    logger.info("✓ LBEBM3D 导入成功")
except Exception as e:
    logger.error(f"✗ LBEBM3D 导入失败: {type(e).__name__}: {e}")

# ============ 测试导入 Exp5 ============
logger.info("\n[2/3] 测试 Exp5 (DG32-BCAT) 导入...")
try:
    from train_swarm_model_v3_with_gnn import DynamicsAwareSwarmGRUModel_with_GNN
    logger.info("✓ Exp5 导入成功")
except Exception as e:
    logger.error(f"✗ Exp5 导入失败: {type(e).__name__}: {e}")
    logger.info("  提示: v3 依赖 v2 的工具函数，需确保两者都在 sys.path 中")

# ============ 测试导入 MRGTraj ============
logger.info("\n[3/3] 测试 MRGTraj 导入...")
try:
    from model_swarm import MRGTrajSwarm
    logger.info("✓ MRGTraj 导入成功")
except Exception as e:
    logger.error(f"✗ MRGTraj 导入失败: {type(e).__name__}: {e}")

logger.info("\n" + "="*60)
logger.info("导入测试完成！")
logger.info("="*60)
