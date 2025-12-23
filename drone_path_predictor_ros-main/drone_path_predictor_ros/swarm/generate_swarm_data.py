#!/usr/bin/env python3
"""
生成测试用的无人机群轨迹数据
"""

import numpy as np
import argparse


def generate_circular_trajectory(center, radius, num_points, height_variation=0):
    """生成圆形轨迹"""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    z = center[2] + height_variation * np.sin(2 * angles)
    trajectory = np.column_stack([x, y, z])
    return trajectory


def generate_linear_trajectory(start, end, num_points):
    """生成线性轨迹"""
    trajectory = np.linspace(start, end, num_points)
    return trajectory


def generate_spiral_trajectory(center, radius, num_points, height_start=0, height_end=50):
    """生成螺旋轨迹"""
    angles = np.linspace(0, 4 * np.pi, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    z = np.linspace(height_start, height_end, num_points)
    trajectory = np.column_stack([x, y, z])
    return trajectory


def generate_swarm_data(num_drones=5, num_points=200, dt=0.1, output_path='swarm_data.npz'):
    """
    生成多无人机轨迹数据
    
    Args:
        num_drones: 无人机数量
        num_points: 每架无人机的轨迹点数
        dt: 采样时间间隔
        output_path: 输出文件路径
    """
    trajectories = {}
    timestamps = {}
    
    print(f"生成 {num_drones} 架无人机的轨迹数据...")
    
    # 生成不同类型的轨迹
    for drone_id in range(num_drones):
        if drone_id == 0:
            # 第一架无人机：圆形轨迹
            trajectory = generate_circular_trajectory(
                center=[0, 0, 50],
                radius=30,
                num_points=num_points,
                height_variation=10
            )
            print(f"  Drone {drone_id}: 圆形轨迹 (center=[0,0,50], radius=30m)")
            
        elif drone_id == 1:
            # 第二架无人机：螺旋轨迹
            trajectory = generate_spiral_trajectory(
                center=[40, 40, 20],
                radius=25,
                num_points=num_points,
                height_start=20,
                height_end=80
            )
            print(f"  Drone {drone_id}: 螺旋轨迹 (center=[40,40,20], height: 20-80m)")
            
        elif drone_id == 2:
            # 第三架无人机：线性轨迹
            trajectory = generate_linear_trajectory(
                start=[0, 50, 30],
                end=[100, 50, 30],
                num_points=num_points
            )
            print(f"  Drone {drone_id}: 线性轨迹 ([0,50,30] -> [100,50,30])")
            
        elif drone_id == 3:
            # 第四架无人机：圆形轨迹（不同方向）
            trajectory = generate_circular_trajectory(
                center=[-40, -40, 50],
                radius=35,
                num_points=num_points,
                height_variation=15
            )
            print(f"  Drone {drone_id}: 圆形轨迹 (center=[-40,-40,50], radius=35m)")
            
        else:
            # 其他无人机：螺旋轨迹变体
            trajectory = generate_spiral_trajectory(
                center=[-50, 50, 30],
                radius=20,
                num_points=num_points,
                height_start=30,
                height_end=70
            )
            print(f"  Drone {drone_id}: 螺旋轨迹 (center=[-50,50,30])")
        
        # 添加少量噪声使数据更逼真
        noise = np.random.normal(0, 0.5, trajectory.shape)
        trajectory = trajectory + noise
        
        # 存储轨迹和时间戳
        trajectories[drone_id] = trajectory
        timestamps[drone_id] = np.arange(num_points) * dt
    
    # 保存数据
    print(f"\n保存数据到 {output_path}...")
    np.savez(
        output_path,
        trajectories=trajectories,
        timestamps=timestamps
    )
    
    print(f"✓ 成功生成数据")
    print(f"  - 无人机数量: {num_drones}")
    print(f"  - 每架轨迹点数: {num_points}")
    print(f"  - 采样时间间隔: {dt}s")
    print(f"  - 总轨迹时长: {num_points * dt:.1f}s")
    print(f"  - 输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='生成无人机群轨迹测试数据')
    parser.add_argument('--num_drones', type=int, default=5,
                       help='无人机数量 (默认: 5)')
    parser.add_argument('--num_points', type=int, default=200,
                       help='每架无人机的轨迹点数 (默认: 200)')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='采样时间间隔(秒) (默认: 0.1)')
    parser.add_argument('--output', type=str, default='swarm_data.npz',
                       help='输出文件路径 (默认: swarm_data.npz)')
    
    args = parser.parse_args()
    
    generate_swarm_data(
        num_drones=args.num_drones,
        num_points=args.num_points,
        dt=args.dt,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
