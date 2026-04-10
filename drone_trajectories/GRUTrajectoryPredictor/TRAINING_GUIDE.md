"""
Swarm GRU 训练说明
==================

关于数据量的说明：

1. 可用的数据文件：
   - input_agents_3_subset.npz    (144.9 MB, 230,232 样本)
   - output_agents_3_subset.npz   (72.5 MB,  230,232 样本)
   
   这两个文件是经过采样的 "子集"，但实际包含 230k+ 样本！

2. --use_subset 参数的作用：
   - --use_subset: 从 230k 数据中只取前 10,000 个样本（用于快速测试）
   - 不加该参数: 使用全部 230,232 个样本（完整训练）

3. 训练命令对比：

   【快速测试】- 仅 10k 样本，~2-3 分钟
   python train_swarm_gru_v2.py --num_agents 3 --num_epochs 10 --batch_size 64 --use_subset
   
   【完整训练】- 230k 样本，~30-60 分钟（取决于 GPU）
   python train_swarm_gru_v2.py --num_agents 3 --num_epochs 50 --batch_size 64
   
   【大规模训练】- 230k 样本，更多 epoch，~1-2 小时
   python train_swarm_gru_v2.py --num_agents 3 --num_epochs 100 --batch_size 32

4. 建议的训练策略：

   第1步：快速测试（验证代码是否工作）
   python train_swarm_gru_v2.py --num_agents 3 --num_epochs 5 --batch_size 64 --use_subset
   
   第2步：完整训练（生成最终模型）
   python train_swarm_gru_v2.py --num_agents 3 --num_epochs 100 --batch_size 64

5. 内存考虑：

   如果你的 GPU 显存充足（>8GB）：
   - 使用 --batch_size 64 或 128，更快收敛
   
   如果显存有限（<8GB）：
   - 使用 --batch_size 32 或 16
   - python train_swarm_gru_v2.py --num_agents 3 --num_epochs 100 --batch_size 16

6. 训练过程中的输出说明：

   Epoch   1 | Train: 0.123456 (pos:0.100000, vel:0.023456) | Val: 0.125000 (pos:0.102000, vel:0.023000)
   ├─ Train Loss: 总损失 (位置损失 + 速度损失)
   └─ Val Loss:   验证损失（用于早停）
   
   最佳模型会自动保存到: Models/swarm_gru_agents_3_best.pth
"""
