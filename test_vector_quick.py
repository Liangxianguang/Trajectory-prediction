#!/usr/bin/env python3
"""
快速测试：验证 VECTOR 增强预测器是否工作正常
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / 'drone_path_predictor_ros-main' / 'drone_path_predictor_ros'))

def test_predictor():
    """测试预测器初始化和基本预测"""
    print("=" * 80)
    print("VECTOR 增强预测器快速测试")
    print("=" * 80)
    
    try:
        from vector_predictor_enhanced import EnhancedPredictorGRU
        print("\n✓ 成功导入 EnhancedPredictorGRU")
    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        return False
    
    # 路径配置
    models_dir = Path(r'D:\Trajectory prediction\drone_path_predictor_ros-main\config\mixed_dataset')
    pos_model = models_dir / 'mix_pos_max_norm_128_3_0p5.pth'
    pos_stats = models_dir / 'pos_stats.npz'
    vel_model = models_dir / 'mix_vel_max_norm_128_3_0p5.pth'
    vel_stats = models_dir / 'vel_stats.npz'
    
    # 检查文件存在性
    print("\n检查模型文件...")
    files = {
        '位置模型': pos_model,
        '位置统计': pos_stats,
        '速度模型': vel_model,
        '速度统计': vel_stats
    }
    
    for name, path in files.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name}: {path.name}")
    
    # 加载轨迹
    print("\n加载测试轨迹...")
    traj_path = Path(r'D:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories\gazebo_trajectory_1.csv')
    if not traj_path.exists():
        print(f"✗ 轨迹文件不存在: {traj_path}")
        return False
    
    df = pd.read_csv(traj_path)
    trajectory = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    print(f"✓ 加载轨迹: {trajectory.shape}")
    
    # 测试 1：初始化预测器（直接位置预测）
    print("\n" + "-" * 80)
    print("【测试 1】直接位置预测器")
    print("-" * 80)
    try:
        predictor_direct = EnhancedPredictorGRU(
            str(pos_model), str(pos_stats),
            use_velocity_integration=False
        )
        print("✓ 初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 测试预测
    try:
        pred = predictor_direct.predict_enhanced(trajectory, method='position')
        print(f"✓ 预测成功: {pred.shape}")
        print(f"  预测值范围: X={pred[:, 0].min():.3f}~{pred[:, 0].max():.3f} m")
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 2：初始化预测器（速度积分预测）
    print("\n" + "-" * 80)
    print("【测试 2】VECTOR 速度积分预测器")
    print("-" * 80)
    try:
        predictor_velocity = EnhancedPredictorGRU(
            str(pos_model), str(pos_stats),
            str(vel_model), str(vel_stats),
            use_velocity_integration=True,
            enforce_first_step_continuity=True
        )
        print("✓ 初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试预测
    try:
        pred = predictor_velocity.predict_enhanced(trajectory, method='velocity')
        print(f"✓ 预测成功: {pred.shape}")
        print(f"  预测值范围: X={pred[:, 0].min():.3f}~{pred[:, 0].max():.3f} m")
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 3：评估质量
    print("\n" + "-" * 80)
    print("【测试 3】评估指标计算")
    print("-" * 80)
    
    # 生成虚拟真实值
    actual = trajectory[-10:].copy()
    
    try:
        metrics = EnhancedPredictorGRU.evaluate_prediction_quality(actual, pred)
        print("✓ 评估成功:")
        print(f"  MAE: {metrics['mae']:.4f} m")
        print(f"  RMSE: {metrics['rmse']:.4f} m")
        print(f"  R²: {metrics['r_squared']:.4f}")
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        return False
    
    # 测试 4：对比两种方法
    print("\n" + "-" * 80)
    print("【测试 4】方法对比")
    print("-" * 80)
    
    try:
        pred_direct = predictor_direct.predict_enhanced(trajectory, method='position')
        pred_velocity = predictor_velocity.predict_enhanced(trajectory, method='velocity')
        
        metrics_direct = EnhancedPredictorGRU.evaluate_prediction_quality(actual, pred_direct)
        metrics_velocity = EnhancedPredictorGRU.evaluate_prediction_quality(actual, pred_velocity)
        
        print(f"\n直接位置预测:")
        print(f"  MAE: {metrics_direct['mae']:.4f} m")
        print(f"  RMSE: {metrics_direct['rmse']:.4f} m")
        
        print(f"\nVECTOR 速度积分:")
        print(f"  MAE: {metrics_velocity['mae']:.4f} m")
        print(f"  RMSE: {metrics_velocity['rmse']:.4f} m")
        
        mae_improve = (metrics_direct['mae'] - metrics_velocity['mae']) / metrics_direct['mae'] * 100
        rmse_improve = (metrics_direct['rmse'] - metrics_velocity['rmse']) / metrics_direct['rmse'] * 100
        
        print(f"\n改进:")
        print(f"  MAE: {mae_improve:+.2f}%")
        print(f"  RMSE: {rmse_improve:+.2f}%")
        
    except Exception as e:
        print(f"✗ 对比失败: {e}")
        return False
    
    # 测试 5：实时预测器
    print("\n" + "-" * 80)
    print("【测试 5】实时预测器")
    print("-" * 80)
    
    try:
        from vector_predictor_enhanced import RealTimePredictor
        rt_predictor = RealTimePredictor(
            str(pos_model), str(pos_stats),
            str(vel_model), str(vel_stats)
        )
        print("✓ 初始化成功")
        
        # 添加位置数据
        for pos in trajectory[-20:]:
            rt_predictor.add_position(pos)
        
        # 实时预测
        pred = rt_predictor.real_time_predict()
        status = rt_predictor.get_buffer_status()
        
        print(f"✓ 实时预测成功: {pred.shape}")
        print(f"  缓冲区大小: {status['buffer_size']}")
        print(f"  速度大小: {status['velocity_magnitude']:.2f} m/s")
        
    except Exception as e:
        print(f"✗ 实时预测失败: {e}")
        import traceback
        traceback.print_exc()
        # 不返回 False，因为这是可选功能
    
    print("\n" + "=" * 80)
    print("✓ 所有关键测试通过！VECTOR 增强预测器工作正常")
    print("=" * 80)
    return True


if __name__ == '__main__':
    success = test_predictor()
    sys.exit(0 if success else 1)
