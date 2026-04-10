"""
Configuration file for Swarm GRU Trajectory Predictor
======================================================
Defines all hyperparameters and paths for training and inference
"""

import os
from pathlib import Path

# ============ Paths ============
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path(r"d:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments")
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Data files
INPUT_FILE_PATTERN = "input_agents_{agents}.npz"
OUTPUT_FILE_PATTERN = "output_agents_{agents}.npz"

# ============ Training Parameters ============
TRAINING_CONFIG = {
    # Model architecture
    'num_agents': 3,              # Number of agents in swarm
    'seq_in': 20,                 # Input sequence length
    'seq_out': 10,                # Output sequence length
    'input_dim': 3,               # Position dimensions (x, y, z)
    'output_dim': 3,              # Output dimensions (x, y, z)
    'hidden_dim': 64,             # GRU hidden size
    'num_layers': 2,              # Number of GRU layers
    'dropout': 0.3,               # Dropout rate
    
    # Training
    'batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'lr_decay': 0.9,              # Learning rate decay factor
    'lr_decay_steps': 10,         # Decay every N epochs
    'early_stopping_patience': 20, # Early stopping patience
    'val_split': 0.1,             # Validation split ratio
    
    # Device
    'device': 'cuda',             # 'cuda' or 'cpu'
    
    # Data normalization
    'normalize': True,
    'dt': 0.1,                    # Time step in seconds
    
    # Loss function
    'loss_weight_position': 1.0,  # Weight for position loss
    'loss_weight_velocity': 0.5,  # Weight for velocity loss
    
    # Misc
    'seed': 42,
    'num_workers': 0,             # DataLoader workers
}

# ============ Model Architectures ============
MODEL_TYPES = {
    'gru': 'GRU-based encoder-decoder',
    'gru_bidirectional': 'Bidirectional GRU',
    'gru_attention': 'GRU with attention mechanism',
}

# ============ Prediction Parameters ============
PREDICTION_CONFIG = {
    'model_type': 'gru',
    'batch_size': 32,
    'visualization': True,
    'save_results': True,
    'output_format': 'npz',  # 'npz' or 'csv'
}

# ============ Logging ============
LOG_CONFIG = {
    'log_dir': PROJECT_ROOT / "logs",
    'log_level': 'INFO',
    'save_metrics': True,
}

LOG_CONFIG['log_dir'].mkdir(exist_ok=True, parents=True)
