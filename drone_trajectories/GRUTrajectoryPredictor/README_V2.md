# GRU Swarm Trajectory Predictor

Predicts multi-agent (drone swarm) 3D trajectories using GRU encoder-decoder networks.

## 📁 Directory Structure

```
GRUTrajectoryPredictor/
├── train_swarm_gru_v2.py       # Main training script (FIXED VERSION)
├── predict_swarm_gru_v2.py     # Inference/prediction script
├── test_quick.py               # Quick validation test
├── config.py                   # Configuration parameters
├── Models/                     # Trained model checkpoints
├── Results/                    # Training results and predictions
├── logs/                       # Training logs
└── README.md                   # This file
```

## ⚡ Quick Start

### 1. Verify Installation (REQUIRED - Run First!)

```bash
python test_quick.py
```

This will check:
- ✓ Python packages (torch, numpy)
- ✓ GPU/CUDA availability
- ✓ Data files exist
- ✓ Model can be created and trained
- ✓ Inference works correctly

### 2. Train Model

**Quick training (subset, ~5 minutes):**
```bash
python train_swarm_gru_v2.py --num_agents 3 --epochs 50 --batch_size 64 --use_subset
```

**Full training (all data, ~1-2 hours):**
```bash
python train_swarm_gru_v2.py --num_agents 3 --epochs 100 --batch_size 32
```

### 3. Run Inference

**Quick inference:**
```bash
python predict_swarm_gru_v2.py --num_agents 3 --use_subset --visualize
```

**Full inference with results saved:**
```bash
python predict_swarm_gru_v2.py --num_agents 3 --visualize --save_results
```

## 📊 Training Parameters

```
--num_agents           Number of agents in swarm (default: 3)
--num_epochs           Number of training epochs (default: 50)
--batch_size           Batch size (default: 64)
--learning_rate        Learning rate (default: 0.001)
--hidden_dim           GRU hidden dimension (default: 64)
--num_layers           Number of GRU layers (default: 2)
--dropout              Dropout rate (default: 0.3)
--early_stopping_patience   Early stopping patience (default: 15)
--val_split            Validation split ratio (default: 0.1)
--use_subset           Use only 10k samples for quick testing
--device               'cuda' or 'cpu' (default: cuda)
--seed                 Random seed (default: 42)
```

## 📈 Inference Parameters

```
--num_agents           Number of agents (default: 3)
--batch_size           Batch size for inference (default: 128)
--use_subset           Use only first 1000 samples
--visualize            Generate 3D trajectory plots
--save_results         Save predictions to NPZ file
--device               'cuda' or 'cpu'
```

## 📁 Data Format

Data files are located in: `d:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments\`

Required files (for 3 agents):
- `input_agents_3_subset.npz` - Input trajectories (20 timesteps, ~230k samples)
- `output_agents_3_subset.npz` - Target trajectories (10 timesteps, ~230k samples)

Or full dataset:
- `input_agents_3.npz` - Full input trajectories
- `output_agents_3.npz` - Full output trajectories

Data format:
- Inputs: `(seq_in=20, samples, agents=3, coords=3)`
- Outputs: `(seq_out=10, samples, agents=3, coords=3)`

## 🏗️ Model Architecture

**GRU Encoder-Decoder:**

```
Input (batch, 20 timesteps, 3 agents, 3 coords)
    ↓
[Reshape] → (batch, 20, 9)
    ↓
[Encoder GRU] → Hidden state h
    ↓
[Decoder GRU] ← h, decoder_input (10, 64)
    ↓
[FC Layer] → (batch, 10, 9)
    ↓
[Reshape] → Output (batch, 10 timesteps, 3 agents, 3 coords)
```

**Loss Function:**
- Position loss (MSE)
- Velocity loss (MSE of derivatives) 
- Combined: `loss = pos_loss + 0.5 * vel_loss`

## 📊 Output Files

After training:
- `Models/swarm_gru_agents_3_best.pth` - Best model checkpoint
- `Results/training_history_agents_3.json` - Training history
- `Results/training_agents_3.png` - Training curves

After inference:
- `Results/metrics_agents_3.json` - Evaluation metrics
- `Results/predictions_agents_3.npz` - Predictions (if --save_results)
- `Results/visualizations/` - 3D trajectory plots (if --visualize)
- `Results/metrics/` - Error metrics plots (if --visualize)

## 📊 Evaluation Metrics

- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R²** - Coefficient of Determination

## 🐛 Troubleshooting

### CUDA out of memory
- Reduce `--batch_size` (e.g., 32 or 16)
- Use `--device cpu` to run on CPU

### Data not found
- Check path: `d:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments\`
- Run `test_quick.py` to verify

### Training too slow
- Use `--use_subset` to train on 10k samples only
- Reduce `--num_epochs`
- Increase `--batch_size`

### Out of memory on CPU
- Use smaller batch size: `--batch_size 8`
- Use subset: `--use_subset`

## 📝 Example Usage

```bash
# Test everything first
python test_quick.py

# Quick training (5 min)
python train_swarm_gru_v2.py --num_agents 3 --epochs 30 --batch_size 64 --use_subset

# Inference with visualization
python predict_swarm_gru_v2.py --num_agents 3 --visualize --save_results

# Full training (1-2 hours)
python train_swarm_gru_v2.py --num_agents 3 --epochs 100 --batch_size 32

# Full inference
python predict_swarm_gru_v2.py --num_agents 3 --visualize --save_results
```

## 📋 Expected Results

After training on subset for 50 epochs:
- Training Loss: ~0.001
- Validation Loss: ~0.002
- MSE: ~0.00001-0.0001
- RMSE: ~0.003-0.01
- R²: ~0.9-0.99

## 🔗 Related Files

- Configuration: `config.py`
- Data directory: `d:\Trajectory prediction\drone_trajectories\Cluster trajectory\swarm_segments\`
- Raw data processing: `d:\Trajectory prediction\drone_trajectories\compute_*.py`

## 📚 References

This implementation is based on:
- **VECTOR**: Velocity-Enhanced GRU Neural Network for Real-Time 3D UAV Trajectory Prediction
- GRU encoder-decoder architecture for sequence-to-sequence prediction
- Multi-agent trajectory forecasting

## ✅ Verification Checklist

Before running training:
- [ ] Run `test_quick.py` and all tests pass
- [ ] Data files exist in the specified directory
- [ ] GPU memory available (or willing to use CPU)
- [ ] Python 3.8+
- [ ] PyTorch 1.10+

## 📞 Support

If you encounter issues:
1. Run `test_quick.py` to identify the problem
2. Check logs in `logs/` directory
3. Verify data files exist
4. Try with `--device cpu` if GPU issues
5. Reduce batch size if out of memory
