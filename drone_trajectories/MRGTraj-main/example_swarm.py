#!/usr/bin/env python3
"""
MRGTraj 无人机集群版本 - 完整示例
===================================

这个脚本演示了如何使用改进的 MRGTrajSwarm 模型进行训练和预测

使用方法:
  python example_swarm.py --mode train --num_agents 3 --num_epochs 10
  python example_swarm.py --mode predict --checkpoint best_model.pth
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_train(num_agents=3, num_epochs=5, batch_size=32, d_model=256, 
                  n_heads=4, n_layers=2, noise_dim=64, lr=1e-3, 
                  kl_weight=0.1, data_dir="../Cluster trajectory/swarm_segments",
                  gpu_num="0"):
    """
    完整的训练示例
    
    Args:
        num_agents: 无人机数量
        num_epochs: 训练 epochs 数
        batch_size: 批处理大小
        d_model: Transformer 维度
        n_heads: 注意力头数
        n_layers: Transformer 层数
        noise_dim: 噪声维度
        lr: 学习率
        kl_weight: KL 散度权重
        data_dir: 数据目录
        gpu_num: GPU 编号
    """
    print("\n" + "="*80)
    print("MRGTrajSwarm 训练示例")
    print("="*80 + "\n")
    
    # 导入所需模块
    from train_swarm import create_data_loaders
    from model_swarm import MRGTrajSwarm
    import torch.optim as optim
    
    # 创建 args 对象 (使用 Namespace 便于序列化)
    args = argparse.Namespace(
        num_agents=num_agents,
        obs_len=20,
        pred_len=10,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        noise_dim=noise_dim,
        agent_dim=3,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=1e-5,
        kl_weight=kl_weight,
        data_dir=data_dir,
        seed=42,
        gpu_num=gpu_num,
        checkpoint_dir="checkpoints_swarm",
        save_every=max(1, num_epochs // 5)  # 每 1/5 保存一次
    )
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"✓ 设备: {device}")
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"✓ 配置参数:")
    print(f"  - 无人机数量: {args.num_agents}")
    print(f"  - 观察长度: {args.obs_len}")
    print(f"  - 预测长度: {args.pred_len}")
    print(f"  - 批处理大小: {args.batch_size}")
    print(f"  - 训练 epochs: {args.num_epochs}")
    
    try:
        # 创建数据加载器
        print(f"\n正在加载数据...")
        train_loader = create_data_loaders(args)
        print(f"✓ 数据加载成功")
        
        # 创建模型
        print(f"\n创建模型...")
        model = MRGTrajSwarm(args).to(device)
        print(f"✓ 模型创建成功")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - 总参数数: {total_params:,}")
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # 简单训练循环
        print(f"\n开始训练...\n")
        
        model.train()
        for epoch in range(args.num_epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (past_traj, future_traj) in enumerate(train_loader):
                # 移到设备
                past_traj = past_traj.to(device)
                future_traj = future_traj.to(device)
                
                # 前向传播
                pred_traj, mu, log_var = model(past_traj, future_traj)
                
                # 计算损失
                l2_loss_val = ((pred_traj - future_traj) ** 2).mean()
                kl_loss_val = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]
                total_loss = l2_loss_val + args.kl_weight * kl_loss_val
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                if batch_idx % 5 == 0:
                    print(f"  Epoch {epoch+1}/{args.num_epochs}, "
                          f"Batch {batch_idx}, Loss: {total_loss.item():.6f}")
            
            avg_loss = epoch_loss / batch_count
            print(f"✓ Epoch {epoch+1} 完成 - 平均损失: {avg_loss:.6f}\n")
        
        print("✓ 训练完成！")
        
        # 保存模型
        checkpoint_dir = Path(args.checkpoint_dir) / f"agents_{args.num_agents}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "example_model.pth"
        
        torch.save({
            'args': args,
            'model_state_dict': model.state_dict(),
        }, checkpoint_path)
        
        print(f"✓ 模型已保存: {checkpoint_path}")
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def example_predict():
    """完整的推理示例"""
    print("\n" + "="*80)
    print("MRGTrajSwarm 推理示例")
    print("="*80 + "\n")
    
    from model_swarm import MRGTrajSwarm
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 设备: {device}")
    
    # 加载模型
    checkpoint_path = Path("checkpoints_swarm/agents_3/example_model.pth")
    
    if not checkpoint_path.exists():
        print(f"✗ 模型文件不存在: {checkpoint_path}")
        print(f"  请先运行训练: python example_swarm.py --mode train")
        return False
    
    print(f"\n加载模型: {checkpoint_path}")
    
    # 处理 PyTorch 2.6+ 的安全加载
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        if "Namespace" in str(e):
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            raise
    
    args = checkpoint['args']
    
    # 创建模型
    model = MRGTrajSwarm(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功")
    
    # 生成随机过去轨迹用于演示
    print(f"\n生成示例数据...")
    batch_size = 2
    past_traj = torch.randn(batch_size, args.obs_len, args.num_agents, 3).to(device)
    
    print(f"✓ 输入形状: {past_traj.shape}")
    print(f"  - 批处理大小: {batch_size}")
    print(f"  - 观察长度: {args.obs_len}")
    print(f"  - 无人机数量: {args.num_agents}")
    print(f"  - 坐标维度: 3 (XYZ)")
    
    # 进行推理
    print(f"\n进行推理 (生成 10 个样本)...")
    
    with torch.no_grad():
        predictions = model.inference(past_traj, num_samples=10)
    
    print(f"✓ 推理完成")
    print(f"  - 输出形状: {predictions.shape}")
    print(f"  - 维度说明: (num_samples, batch_size, pred_len, num_agents, agent_dim)")
    print(f"  - 维度说明: ({predictions.shape[0]}, {predictions.shape[1]}, {predictions.shape[2]}, {predictions.shape[3]}, {predictions.shape[4]})")
    
    # 分析输出
    print(f"\n推理结果统计:")
    predictions_np = predictions.cpu().numpy()
    print(f"  - 最小值: {predictions_np.min():.4f}")
    print(f"  - 最大值: {predictions_np.max():.4f}")
    print(f"  - 均值: {predictions_np.mean():.4f}")
    print(f"  - 标准差: {predictions_np.std():.4f}")
    
    # 展示第一个样本的第一个预测
    print(f"\n第一个样本的第一个预测 (前 3 个时间步):")
    first_pred = predictions_np[0, 0, :3, :, :]  # (3 time steps, num_agents, 3 coords)
    for t in range(3):
        print(f"  时间步 {t}:")
        for agent_id in range(args.num_agents):
            x, y, z = first_pred[t, agent_id]
            print(f"    Agent {agent_id}: ({x:.4f}, {y:.4f}, {z:.4f})")
    
    print(f"\n✓ 推理示例完成！")
    return True


def example_data_conversion():
    """数据转换示例"""
    print("\n" + "="*80)
    print("数据转换示例")
    print("="*80 + "\n")
    
    import pandas as pd
    from data_tools import DataConverter, DataValidator
    
    # 创建示例数据
    print("创建示例数据...")
    
    # 创建一个简单的 NPZ 文件
    seq_len = 20
    num_samples = 10
    num_agents = 3
    agent_dim = 3
    
    # 生成随机轨迹数据
    data = np.random.randn(seq_len, num_samples, num_agents, agent_dim)
    
    npz_file = "example_trajectory.npz"
    np.savez_compressed(npz_file, data=data)
    
    print(f"✓ 创建 NPZ 文件: {npz_file}")
    print(f"  - 形状: {data.shape}")
    
    # 验证 NPZ 文件
    print(f"\n验证 NPZ 文件...")
    valid, _ = DataValidator.validate_npz(npz_file)
    
    if not valid:
        print(f"✗ 验证失败")
        return False
    
    # 转换为 CSV
    csv_file = "example_trajectory.csv"
    print(f"\n转换为 CSV: {csv_file}")
    
    try:
        DataConverter.npz_to_csv(npz_file, csv_file)
        print(f"✓ 转换成功")
        
        # 读取 CSV 文件显示内容
        df = pd.read_csv(csv_file)
        print(f"\nCSV 文件预览:")
        print(df.head(3))
        
        # 转换回 NPZ
        npz_file_2 = "example_trajectory_2.npz"
        print(f"\n转换回 NPZ: {npz_file_2}")
        DataConverter.csv_to_npz(csv_file, npz_file_2)
        print(f"✓ 转换成功")
        
        # 验证转换后的数据
        data_2 = np.load(npz_file_2)['data']
        print(f"  - 原始形状: {data.shape}")
        print(f"  - 转换后形状: {data_2.shape}")
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="MRGTraj 无人机集群版本 - 完整示例")
    
    parser.add_argument("--mode", choices=['train', 'predict', 'data'], default='train',
                        help="运行模式")
    
    # 训练参数
    parser.add_argument("--num_agents", type=int, default=3,
                        help="无人机数量")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="训练 epochs 数")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="批处理大小")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer 维度")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="注意力头数")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Transformer 层数")
    parser.add_argument("--noise_dim", type=int, default=64,
                        help="噪声维度")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="KL 散度权重")
    parser.add_argument("--data_dir", type=str, default="../Cluster trajectory/swarm_segments",
                        help="数据目录")
    parser.add_argument("--gpu_num", type=str, default="0",
                        help="GPU 编号")
    
    # 推理参数
    parser.add_argument("--checkpoint", type=str,
                        help="模型检查点路径 (仅用于推理)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("欢迎使用 MRGTraj 无人机集群版本")
    print("="*80)
    
    if args.mode == 'train':
        success = example_train(
            num_agents=args.num_agents,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            noise_dim=args.noise_dim,
            lr=args.lr,
            kl_weight=args.kl_weight,
            data_dir=args.data_dir,
            gpu_num=args.gpu_num
        )
    elif args.mode == 'predict':
        success = example_predict()
    elif args.mode == 'data':
        try:
            import pandas as pd
            success = example_data_conversion()
        except ImportError:
            print("✗ 需要安装 pandas 库")
            print("  pip install pandas")
            success = False
    else:
        success = False
    
    if success:
        print("\n" + "="*80)
        print("✓ 示例完成！")
        print("="*80)
        
        print("\n下一步:")
        if args.mode == 'train':
            print(f"1. 查看已保存的模型: checkpoints_swarm/agents_{args.num_agents}/")
            print(f"2. 进行推理: python example_swarm.py --mode predict")

        elif args.mode == 'predict':
            print("1. 查看推理脚本: predict_swarm.py")
            print("2. 使用你的数据: python predict_swarm.py --input_file your_data.csv")
        
        print("\n完整文档:")
        print("- SWARM_QUICKSTART.md: 快速开始指南")
        print("- MRGTraj_Swarm_Complete_Guide.md: 完整使用指南")
        print("- MRGTraj_UAV_Swarm_Adaptation_Report.md: 技术细节")
    else:
        print("\n✗ 示例执行失败")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
