import logging
from pathlib import Path
from infer_enhanced import EnhancedInference

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

MODELS = [
    ("newdata1_short_gru_models/short_enhanced_gru_model_best_model.pth",
     "newdata1_short_gru_models/short_enhanced_gru_model_norm_stats.npz",
     True),
    ("bigru_long_gru_models/long_enhanced_gru_model_best_model.pth",
     "bigru_long_gru_models/long_enhanced_gru_model_norm_stats.npz",
     True),
]

for model_path, stats_path, use_attention in MODELS:
    print(f"\n=== 加载 {model_path} ===")
    infer = EnhancedInference(model_path, stats_path, use_attention=use_attention)
    print("构建成功 ->", infer.model)
