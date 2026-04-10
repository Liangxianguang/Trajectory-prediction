"""
Quick Test Script for Swarm GRU Trajectory Predictor
====================================================
Tests if training and inference work correctly with minimal data.

Run this first to verify everything is working!

Usage:
    python test_quick.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_imports():
    """Test if all required packages are available"""
    print_section("Test 1: Checking Imports")
    
    packages = {
        'torch': torch,
        'numpy': np,
    }
    
    for name, module in packages.items():
        version = getattr(module, '__version__', 'unknown')
        logger.info(f"✓ {name:15} version: {version}")
    
    return True


def test_device():
    """Test if GPU is available"""
    print_section("Test 2: Checking Device")
    
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Device count: {torch.cuda.device_count()}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
    else:
        logger.warning("! CUDA not available, will use CPU")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"✓ Using device: {device}")
    
    return device


def test_data():
    """Test if data files exist"""
    print_section("Test 3: Checking Data Files")
    
    data_dir = Path(r"d:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments")
    
    files_to_check = [
        'input_agents_3_subset.npz',
        'output_agents_3_subset.npz',
    ]
    
    all_exist = True
    for fname in files_to_check:
        fpath = data_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size / (1024**2)
            logger.info(f"✓ {fname:30} ({size:.1f} MB)")
        else:
            logger.error(f"✗ {fname:30} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        logger.warning("Some data files not found. Trying alternative names...")
        alt_files = [
            'input_agents_3.npz',
            'output_agents_3.npz',
        ]
        for fname in alt_files:
            fpath = data_dir / fname
            if fpath.exists():
                size = fpath.stat().st_size / (1024**2)
                logger.info(f"✓ {fname:30} ({size:.1f} MB)")
                all_exist = True
    
    if not all_exist:
        logger.error("Data files not found!")
        return None, None
    
    # Load data
    logger.info("\n  Loading data...")
    try:
        input_file = None
        output_file = None
        
        for fname in files_to_check + ['input_agents_3.npz', 'output_agents_3.npz']:
            fpath = data_dir / fname
            if fpath.exists():
                if 'input' in fname:
                    input_file = fpath
                else:
                    output_file = fpath
        
        X = np.load(input_file)['data']
        Y = np.load(output_file)['data']
        
        logger.info(f"✓ Data loaded: X{X.shape}, Y{Y.shape}")
        return X, Y
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None


def test_model(device):
    """Test if model can be created and run"""
    print_section("Test 4: Creating and Testing Model")
    
    # Create dummy data
    batch_size = 4
    seq_in = 20
    seq_out = 10
    num_agents = 3
    
    x_dummy = torch.randn(batch_size, seq_in, num_agents, 3).to(device)
    
    logger.info(f"  Input shape: {x_dummy.shape}")
    
    # Create simple model
    class SimpleGRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_agents, seq_out):
            super().__init__()
            self.encoder = nn.GRU(input_dim * num_agents, hidden_dim, 2, batch_first=True)
            self.decoder = nn.GRU(hidden_dim, hidden_dim, 2, batch_first=True)
            self.fc = nn.Linear(hidden_dim, input_dim * num_agents)
            self.seq_out = seq_out
            self.num_agents = num_agents
            
        def forward(self, x):
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1, 3 * self.num_agents)
            _, h = self.encoder(x_flat)
            dec_in = torch.zeros(batch_size, self.seq_out, h.shape[-1], device=x.device)
            out, _ = self.decoder(dec_in, h)
            y_flat = self.fc(out)
            y = y_flat.view(batch_size, self.seq_out, self.num_agents, 3)
            return y
    
    try:
        model = SimpleGRU(3, 64, 3, seq_out).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model created: {n_params:,} parameters")
        
        # Forward pass
        with torch.no_grad():
            y = model(x_dummy)
        logger.info(f"✓ Forward pass successful: output shape {y.shape}")
        
        return model
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return None


def test_training_step(model, device):
    """Test one training step"""
    print_section("Test 5: Training Step")
    
    try:
        batch_size = 4
        seq_in = 20
        seq_out = 10
        num_agents = 3
        
        x = torch.randn(batch_size, seq_in, num_agents, 3).to(device)
        y = torch.randn(batch_size, seq_out, num_agents, 3).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"✓ Training step successful")
        logger.info(f"  Loss: {loss.item():.6f}")
        
        return True
    except Exception as e:
        logger.error(f"Training step failed: {e}")
        return False


def test_inference_step(model, X, Y, device):
    """Test inference with real data"""
    print_section("Test 6: Inference with Real Data")
    
    try:
        # Use subset
        X_test = X[:, :100, :, :]  # (seq_in, 100 samples, agents, 3)
        Y_test = Y[:, :100, :, :]
        
        # Transpose to (samples, seq, agents, 3)
        X_test = np.transpose(X_test, (1, 0, 2, 3))
        Y_test = np.transpose(Y_test, (1, 0, 2, 3))
        
        logger.info(f"  Test data: X{X_test.shape}, Y{Y_test.shape}")
        
        # Normalize
        x_mean = X_test.reshape(-1, 3).mean(axis=0)
        x_std = X_test.reshape(-1, 3).std(axis=0) + 1e-8
        X_norm = (X_test - x_mean) / x_std
        
        logger.info(f"  Normalized: mean={x_mean}, std={x_std}")
        
        # Inference
        model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X_norm).to(device)
            y_pred = model(x_tensor)
        
        logger.info(f"✓ Inference successful")
        logger.info(f"  Output shape: {y_pred.shape}")
        logger.info(f"  Output range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        
        return True
    except Exception as e:
        logger.error(f"Inference step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_existing_files():
    """Check if training/prediction scripts exist"""
    print_section("Test 7: Checking Script Files")
    
    project_root = Path(__file__).parent
    scripts = {
        'train_swarm_gru_v2.py': 'Training script',
        'predict_swarm_gru_v2.py': 'Prediction script',
        'config.py': 'Configuration',
    }
    
    all_exist = True
    for script, desc in scripts.items():
        path = project_root / script
        if path.exists():
            logger.info(f"✓ {script:30} - {desc}")
        else:
            logger.error(f"✗ {script:30} - NOT FOUND")
            all_exist = False
    
    return all_exist


# ============ Main ============
def main():
    print("\n" + "="*70)
    print("  SWARM GRU TRAJECTORY PREDICTOR - QUICK TEST")
    print("="*70)
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: Device
    device = test_device()
    results['device'] = device is not None
    
    # Test 3: Data
    X, Y = test_data()
    results['data'] = X is not None and Y is not None
    
    # Test 4: Model creation
    model = test_model(device)
    results['model'] = model is not None
    
    # Test 5: Training step
    if model is not None:
        results['training'] = test_training_step(model, device)
    
    # Test 6: Inference
    if model is not None and X is not None:
        results['inference'] = test_inference_step(model, X, Y, device)
    
    # Test 7: Files
    results['files'] = test_existing_files()
    
    # Summary
    print_section("TEST SUMMARY")
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:15} : {status}")
        all_passed = all_passed and result
    
    print()
    if all_passed:
        logger.info("✓ All tests passed! You can now run training and inference.")
        logger.info("\n  To train a model:")
        logger.info("    python train_swarm_gru_v2.py --num_agents 3 --epochs 50 --batch_size 64 --use_subset")
        logger.info("\n  To run inference:")
        logger.info("    python predict_swarm_gru_v2.py --num_agents 3 --use_subset --visualize")
    else:
        logger.error("✗ Some tests failed. Please fix the issues above.")
    
    print(f"\n{'='*70}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
