# Training Script Improvements

## Overview
Enhanced `train_swarm_detailed.py` with better logging, reduced Chinese characters (fixing encoding issues), and improved checkpoint management.

## Key Changes

### 1. **Language Conversion**
- **Before**: All comments and log messages in Chinese (causing encoding issues)
- **After**: All comments and log messages in English for better compatibility
- **Result**: No more garbled characters in log files

### 2. **Enhanced Logging**
- **More detailed initialization logging**: Shows data loading stats, model parameters, optimizer config
- **Better structured epoch logs**: Uses `|` separator for clear metric sections
- **Status indicators**: `[OK]`, `[BEST]`, `[CHECKPOINT]`, `[COMPLETE]` for quick visual scanning
- **Training progress visibility**: Logs total samples, batch size, batches per epoch

#### Example Log Output:
```
[Epoch 1/100] total_loss=0.234567 | l2_loss=0.123456 | pos_loss=0.045678 | height_loss=0.012345 | 
              vel_loss=0.008901 | acc_loss=0.002345 | collision_loss=0.001234 | 
              formation_loss=0.000567 | kl_loss=0.034567 | ade=0.456789 | fde=0.654321
  [BEST] Saved best model - loss=0.234567
```

### 3. **Checkpoint Management**
- **Automatic save frequency**: Models now save **every 10 epochs** (not configurable via `save_every`)
- **Clear filenames**: `checkpoint_epoch_0010.pth`, `checkpoint_epoch_0020.pth` (zero-padded for easy sorting)
- **Best model tracking**: Still saves `best_model.pth` when loss improves
- **Checkpoint info logging**: Shows when checkpoints are saved

### 4. **Improved File Encoding**
- **UTF-8 encoding** for log files: Prevents encoding errors on all platforms
- **Cross-platform compatibility**: Works correctly on Windows, Linux, macOS

## Configuration

### Checkpoint Schedule
Models are automatically saved every 10 epochs:
```
Epoch 10  → checkpoint_epoch_0010.pth
Epoch 20  → checkpoint_epoch_0020.pth
Epoch 30  → checkpoint_epoch_0030.pth
...
```

### Loss Weight Parameters
All configurable via command line:
```bash
python train_swarm_detailed.py \
  --num_agents 3 \
  --batch_size 32 \
  --num_epochs 100 \
  --collision_weight 0.5 \
  --formation_weight 0.2 \
  --kl_weight 0.1
```

## Log File Examples

### Initialization Section:
```
2026-03-05 15:05:40 - INFO - Configuration Parameters:
  data_dir: ../Cluster trajectory/swarm_segments
  num_agents: 3
  batch_size: 32
  num_epochs: 100
  ...

[OK] Data loaded successfully
  Total samples: 12500
  Batch size: 32
  Total batches per epoch: 391

[OK] Model created successfully
  Model architecture: MRGTrajSwarm
  Total parameters: 2,456,789
  Trainable parameters: 2,456,789

Optimizer configuration:
  Type: Adam
  Learning rate: 0.001
  Weight decay: 0.00001
  Scheduler: CosineAnnealing (T_max=100)
```

### Training Section:
```
[Epoch 1/100] total_loss=1.234567 | l2_loss=0.945678 | pos_loss=0.145678 | height_loss=0.045678 | 
              vel_loss=0.025678 | acc_loss=0.008901 | collision_loss=0.045678 | 
              formation_loss=0.015678 | kl_loss=0.134567 | ade=1.234567 | fde=2.345678
  [BEST] Saved best model - loss=1.234567

[Epoch 2/100] total_loss=1.189234 | l2_loss=0.901234 | ... 

[Epoch 10/100] total_loss=0.956789 | ...
  [CHECKPOINT] Saved model at epoch 10

[Epoch 20/100] total_loss=0.812345 | ...
  [CHECKPOINT] Saved model at epoch 20
```

### Completion Section:
```
[COMPLETE] Training finished!
  Best loss: 0.123456
  Checkpoint directory: checkpoints_swarm100/agents_3
```

## Metrics Tracked

Each epoch logs all 11 metrics:
1. `total_loss` - Weighted sum of all losses
2. `l2_loss` - L2 reconstruction error
3. `pos_loss` - XY plane position error
4. `height_loss` - Z coordinate error
5. `vel_loss` - Velocity smoothness
6. `acc_loss` - Acceleration smoothness
7. `collision_loss` - Collision avoidance penalty
8. `formation_loss` - Formation stability
9. `kl_loss` - KL divergence regularization
10. `ade` - Average displacement error
11. `fde` - Final displacement error

## Benefits

✅ **No more encoding issues** - All text in ASCII-safe English  
✅ **Better readability** - Clear structure and status indicators  
✅ **Easier monitoring** - More detailed progress information  
✅ **Automatic checkpointing** - Every 10 epochs without configuration  
✅ **Better debugging** - Rich logs for troubleshooting  
✅ **Professional output** - Clean, organized log files  

## Usage Examples

### Basic training:
```bash
python train_swarm_detailed.py --num_agents 3 --batch_size 32 --num_epochs 100
```

### With custom loss weights:
```bash
python train_swarm_detailed.py \
  --num_agents 3 \
  --batch_size 32 \
  --num_epochs 100 \
  --collision_weight 1.0 \
  --formation_weight 0.5 \
  --kl_weight 0.05
```

### High-quality training:
```bash
python train_swarm_detailed.py \
  --num_agents 3 \
  --batch_size 32 \
  --num_epochs 200 \
  --d_model 512 \
  --n_layers 3 \
  --lr 0.0005
```

## Files Monitored

- `checkpoints_swarm100/agents_3/best_model.pth` - Best model based on loss
- `checkpoints_swarm100/agents_3/checkpoint_epoch_*.pth` - Periodic checkpoints (every 10 epochs)
- `checkpoints_swarm100/agents_3/train_agents_3.log` - Training log (English, UTF-8 encoded)
- `checkpoints_swarm100/agents_3/logs/` - TensorBoard event files (if available)
