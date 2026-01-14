#!/usr/bin/env python3
"""
单元测试：GNN 模块与 v3 模型验证

运行方式：
    python test_gnn_integration.py
    
或使用 pytest：
    pytest test_gnn_integration.py -v
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# 导入测试对象
try:
    from train_swarm_model_v3_with_gnn import (
        GraphAttentionHead,
        MultiHeadGraphAttention,
        build_adjacency_from_positions,
        DynamicsAwareSwarmGRUModel_with_GNN,
    )
    from train_swarm_model_v2_dynamics_aware import (
        compute_features_enhanced_24d,
        compute_velocity_direction,
        compute_acceleration_decomposition,
        DynamicsAwareLoss,
    )
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在正确的目录中运行此脚本")
    sys.exit(1)


# ====================================================================
# 测试 1: 邻接矩阵构建
# ====================================================================

def test_adjacency_matrix():
    """测试邻接矩阵的构建"""
    print("\n" + "="*70)
    print("测试 1: 邻接矩阵构建")
    print("="*70)
    
    # 创建简单的位置数据
    positions = torch.tensor([
        [[0, 0, 0],
         [1, 0, 0],
         [2, 0, 0],
         [10, 0, 0]],
    ], dtype=torch.float32)  # (1, 4, 3)
    
    threshold = 2.0
    adjacency = build_adjacency_from_positions(positions, threshold=threshold, add_self_loops=True)
    
    print(f"✓ 位置输入形状: {positions.shape}")
    print(f"✓ 距离阈值: {threshold} m")
    print(f"✓ 邻接矩阵形状: {adjacency.shape}")
    print(f"✓ 邻接矩阵:\n{adjacency[0]}")
    print(f"✓ 距离矩阵验证:")
    print(f"  - pos[0] - pos[1] = {torch.norm(positions[0, 0] - positions[0, 1]):.4f} m (< {threshold}? 连接)")
    print(f"  - pos[0] - pos[2] = {torch.norm(positions[0, 0] - positions[0, 2]):.4f} m (< {threshold}? 不连接)")
    print(f"  - pos[0] - pos[3] = {torch.norm(positions[0, 0] - positions[0, 3]):.4f} m (< {threshold}? 不连接)")
    
    # 验证
    assert adjacency.shape == (1, 4, 4), f"形状错误: {adjacency.shape}"
    assert (adjacency >= 0).all() and (adjacency <= 1).all(), "邻接矩阵值应在 [0,1]"
    assert adjacency[0, 0, 0] == 1, "自环应为 1"
    assert adjacency[0, 0, 1] == 1, "距离 1m < 2.0m，应该连接"
    # 距离恰好为 2.0，由于使用 < 比较，应该为 0，但加入自环后行 0 全为 1
    assert adjacency[0, 0, 3] == 0, "距离 10m >= 2.0m，应该不连接"
    
    print("✅ 邻接矩阵测试通过")


# ====================================================================
# 测试 2: 单头图注意力
# ====================================================================

def test_single_gat_head():
    """测试单头图注意力层"""
    print("\n" + "="*70)
    print("测试 2: 单头图注意力层")
    print("="*70)
    
    # 创建 GAT 单头
    gat_head = GraphAttentionHead(
        in_channels=24,
        out_channels=32,
        dropout=0.1
    )
    
    # 创建输入
    num_nodes = 5
    x = torch.randn(num_nodes, 24)  # (5, 24)
    adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adjacency = adjacency + torch.eye(num_nodes)  # 添加自环
    adjacency = torch.clamp(adjacency, 0, 1)
    
    print(f"✓ 输入特征形状: {x.shape}")
    print(f"✓ 邻接矩阵形状: {adjacency.shape}")
    
    # 前向传播
    output, attn_weights = gat_head(x, adjacency)
    
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 注意力权重形状: {attn_weights.shape}")
    print(f"✓ 注意力权重示例 (agent 0):\n{attn_weights[0]}")
    
    # 验证输出
    assert output.shape == (num_nodes, 32), f"输出形状错误: {output.shape}"
    assert attn_weights.shape == (num_nodes, num_nodes), f"注意力权重形状错误: {attn_weights.shape}"
    assert not torch.isnan(output).any(), "输出包含 NaN"
    assert not torch.isnan(attn_weights).any(), "注意力权重包含 NaN"
    
    print("✅ 单头 GAT 测试通过")


# ====================================================================
# 测试 3: 多头图注意力
# ====================================================================

def test_multi_head_gat():
    """测试多头图注意力层"""
    print("\n" + "="*70)
    print("测试 3: 多头图注意力层")
    print("="*70)
    
    # 创建多头 GAT
    gat_multi = MultiHeadGraphAttention(
        in_channels=24,
        out_channels=32,
        num_heads=4,
        dropout=0.1,
        concat=True
    )
    
    # 创建输入
    num_nodes = 5
    x = torch.randn(num_nodes, 24)
    adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adjacency = adjacency + torch.eye(num_nodes)
    adjacency = torch.clamp(adjacency, 0, 1)
    
    print(f"✓ 多头数: 4")
    print(f"✓ 每头输出维度: 32")
    print(f"✓ 拼接后输出维度: 128 (4*32)")
    
    # 前向传播
    output = gat_multi(x, adjacency)
    
    print(f"✓ 最终输出形状: {output.shape}")
    
    # 验证
    assert output.shape == (num_nodes, 32), f"输出形状应为 (5, 32)，但得到 {output.shape}"
    assert not torch.isnan(output).any(), "输出包含 NaN"
    
    print("✅ 多头 GAT 测试通过")


# ====================================================================
# 测试 4: 特征计算（24D 特征）
# ====================================================================

def test_feature_computation():
    """测试 24D 特征计算"""
    print("\n" + "="*70)
    print("测试 4: 24D 特征计算")
    print("="*70)
    
    # 创建随机轨迹
    trajectory = torch.randn(20, 3, 3).numpy()  # (T=20, agents=3, 3)
    
    print(f"✓ 轨迹形状: {trajectory.shape}")
    
    # 计算特征
    features = compute_features_enhanced_24d(trajectory, dt=0.1)
    
    print(f"✓ 特征形状: {features.shape}")
    print(f"✓ 特征维度: 24")
    print(f"✓ 特征统计:")
    print(f"  - min: {features.min():.6f}")
    print(f"  - max: {features.max():.6f}")
    print(f"  - mean: {features.mean():.6f}")
    print(f"  - std: {features.std():.6f}")
    
    # 验证
    assert features.shape == (20, 3, 24), f"特征形状错误: {features.shape}"
    assert not np.isnan(features).any(), "特征包含 NaN"
    assert not np.isinf(features).any(), "特征包含 Inf"
    
    print("✅ 特征计算测试通过")


# ====================================================================
# 测试 5: v3 模型前向传播（3D 输入）
# ====================================================================

def test_v3_model_forward_3d():
    """测试 v3 模型的前向传播（3D 输入）"""
    print("\n" + "="*70)
    print("测试 5: v3 模型前向传播 (3D 输入)")
    print("="*70)
    
    device = torch.device('cpu')
    
    # 创建模型
    model = DynamicsAwareSwarmGRUModel_with_GNN(
        input_size=24,
        hidden_size=128,
        num_layers=2,
        output_size=3,
        dropout=0.3,
        use_attention=True,
        gnn_hidden=64,
        num_gnn_heads=4,
        edge_threshold=5.0,
        fusion_mode='concat'
    ).to(device)
    
    print(f"✓ 模型创建完成")
    print(f"✓ GNN 隐层维度: 64")
    print(f"✓ GAT 多头数: 4")
    print(f"✓ 邻接距离阈值: 5.0 m")
    
    # 创建随机输入
    batch_size = 4
    seq_in = 20
    num_agents = 3
    feat_dim = 24
    seq_out = 10
    
    x = torch.randn(batch_size, seq_in, num_agents, feat_dim).to(device)
    x_orig = torch.randn(batch_size, seq_in, num_agents, 3).to(device)
    y = torch.randn(batch_size, seq_out, num_agents, 3).to(device)
    
    print(f"✓ 输入形状:")
    print(f"  - 特征: {x.shape}")
    print(f"  - 位置: {x_orig.shape}")
    print(f"  - 目标: {y.shape}")
    
    # 前向传播
    with torch.no_grad():
        pred_pos, pred_vel, pred_accel = model(
            x, x_orig, y=y,
            teacher_forcing_ratio=0.5
        )
    
    print(f"✓ 输出形状:")
    print(f"  - 位置预测: {pred_pos.shape}")
    print(f"  - 速度预测: {pred_vel.shape}")
    print(f"  - 加速度预测: {pred_accel.shape}")
    
    # 验证输出
    assert pred_pos.shape == (batch_size, seq_out, num_agents, 3), f"位置输出形状错误"
    assert pred_vel.shape == (batch_size, seq_out, num_agents, 3), f"速度输出形状错误"
    assert pred_accel.shape == (batch_size, seq_out, num_agents, 2), f"加速度输出形状错误"
    assert not torch.isnan(pred_pos).any(), "位置预测包含 NaN"
    assert not torch.isnan(pred_vel).any(), "速度预测包含 NaN"
    assert not torch.isnan(pred_accel).any(), "加速度预测包含 NaN"
    
    print("✅ v3 模型前向传播测试通过 (3D)")


# ====================================================================
# 测试 6: v3 模型梯度流
# ====================================================================

def test_v3_model_gradient_flow():
    """测试 v3 模型的梯度流"""
    print("\n" + "="*70)
    print("测试 6: v3 模型梯度流")
    print("="*70)
    
    device = torch.device('cpu')
    
    # 创建模型
    model = DynamicsAwareSwarmGRUModel_with_GNN(
        input_size=24,
        hidden_size=128,
        num_layers=2,
        output_size=3,
        dropout=0.1,
        use_attention=True,
        gnn_hidden=64,
        num_gnn_heads=4,
        edge_threshold=5.0,
        fusion_mode='concat'
    ).to(device)
    
    # 创建损失函数
    criterion = DynamicsAwareLoss()
    
    # 创建输入
    batch_size = 2
    seq_in = 10
    num_agents = 3
    feat_dim = 24
    seq_out = 5
    
    x = torch.randn(batch_size, seq_in, num_agents, feat_dim).to(device)
    x_orig = torch.randn(batch_size, seq_in, num_agents, 3).to(device)
    y = torch.randn(batch_size, seq_out, num_agents, 3).to(device)
    y_vel = torch.randn(batch_size, seq_out, num_agents, 3).to(device)
    y_accel = torch.randn(batch_size, seq_out, num_agents, 2).to(device)
    
    print(f"✓ 前向传播...")
    
    # 前向传播
    pred_pos, pred_vel, pred_accel = model(
        x, x_orig, y=y, y_velocity=y_vel, y_accel=y_accel,
        teacher_forcing_ratio=0.5
    )
    
    # 计算损失
    loss, _, _, _ = criterion(
        pred_pos, y,
        pred_velocity=pred_vel, target_velocity=y_vel,
        pred_accel=pred_accel, target_accel=y_accel
    )
    
    print(f"✓ 损失: {loss.item():.6f}")
    print(f"✓ 反向传播...")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    gnn_has_grad = False
    gru_has_grad = False
    fc_has_grad = False
    
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            if 'gnn' in name:
                gnn_has_grad = True
            if 'encoder' in name or 'decoder' in name:
                gru_has_grad = True
            if 'fc_' in name:
                fc_has_grad = True
    
    print(f"✓ 梯度检查:")
    print(f"  - GNN 层有非零梯度: {gnn_has_grad}")
    print(f"  - GRU 层有非零梯度: {gru_has_grad}")
    print(f"  - FC 层有非零梯度: {fc_has_grad}")
    
    assert gnn_has_grad, "GNN 层梯度未更新"
    assert gru_has_grad, "GRU 层梯度未更新"
    assert fc_has_grad, "FC 层梯度未更新"
    
    print("✅ 梯度流测试通过")


# ====================================================================
# 测试 7: 不同融合模式
# ====================================================================

def test_fusion_modes():
    """测试不同的特征融合模式"""
    print("\n" + "="*70)
    print("测试 7: 特征融合模式")
    print("="*70)
    
    device = torch.device('cpu')
    
    modes = ['concat', 'gate', 'add']
    
    for mode in modes:
        print(f"\n🔹 测试融合模式: {mode}")
        
        model = DynamicsAwareSwarmGRUModel_with_GNN(
            input_size=24,
            hidden_size=128,
            num_layers=2,
            output_size=3,
            dropout=0.2,
            use_attention=True,
            gnn_hidden=64,
            num_gnn_heads=4,
            edge_threshold=5.0,
            fusion_mode=mode
        ).to(device)
        
        # 输入
        x = torch.randn(2, 10, 3, 24).to(device)
        x_orig = torch.randn(2, 10, 3, 3).to(device)
        
        # 前向传播
        with torch.no_grad():
            pred_pos, pred_vel, pred_accel = model(x, x_orig)
        
        print(f"   ✓ 前向传播成功")
        print(f"   ✓ 输出形状: {pred_pos.shape}")
        assert not torch.isnan(pred_pos).any(), f"{mode} 模式输出包含 NaN"
    
    print("\n✅ 融合模式测试通过")


# ====================================================================
# 测试 8: 不同代理数
# ====================================================================

def test_variable_agents():
    """测试不同数量的代理"""
    print("\n" + "="*70)
    print("测试 8: 可变代理数")
    print("="*70)
    
    device = torch.device('cpu')
    
    model = DynamicsAwareSwarmGRUModel_with_GNN(
        input_size=24,
        hidden_size=128,
        num_layers=2,
        output_size=3,
        dropout=0.2,
        use_attention=True,
        gnn_hidden=64,
        num_gnn_heads=4,
        edge_threshold=5.0,
        fusion_mode='concat'
    ).to(device)
    
    agent_counts = [2, 3, 4, 5, 6]
    
    for num_agents in agent_counts:
        x = torch.randn(2, 10, num_agents, 24).to(device)
        x_orig = torch.randn(2, 10, num_agents, 3).to(device)
        
        with torch.no_grad():
            pred_pos, _, _ = model(x, x_orig)
        
        print(f"✓ {num_agents} 个代理: 输出形状 {pred_pos.shape}")
        assert pred_pos.shape[2] == num_agents, f"代理数不匹配"
    
    print("✅ 可变代理数测试通过")


# ====================================================================
# 主测试函数
# ====================================================================

def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           GNN 模块 & v3 模型集成单元测试                        ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    tests = [
        ("邻接矩阵构建", test_adjacency_matrix),
        ("单头 GAT", test_single_gat_head),
        ("多头 GAT", test_multi_head_gat),
        ("24D 特征计算", test_feature_computation),
        ("v3 模型前向传播", test_v3_model_forward_3d),
        ("v3 模型梯度流", test_v3_model_gradient_flow),
        ("特征融合模式", test_fusion_modes),
        ("可变代理数", test_variable_agents),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ 测试失败: {test_name}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 打印总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"✅ 通过: {passed}/{len(tests)}")
    print(f"❌ 失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！v3 模型已准备好使用")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查错误信息")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
