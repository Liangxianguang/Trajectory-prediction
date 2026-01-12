"""
v2 推理和可视化工具包

提供完整的推理和可视化功能用于 24D 动力学感知模型
"""

from .infer_swarm_model_v2 import (
    SwarmPredictorV2,
    infer_batch,
    compute_features_for_inference,
)

from .visualize_swarm_prediction_v2 import (
    SwarmVisualizerV2,
)

__version__ = '1.0.0'
__all__ = [
    'SwarmPredictorV2',
    'SwarmVisualizerV2',
    'infer_batch',
    'compute_features_for_inference',
]
