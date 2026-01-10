
import os
import pandas as pd
import numpy as np
import shutil
import glob
import argparse
import random

def convert_swarm_to_standard(input_root, output_root, dataset_name="swarm"):
    """
    将宽格式的集群数据转换为标准的长格式 (frame, ped_id, x, y, z)
    并按比例划分为 train/val/test
    """
    
    # 查找所有CSV文件
    # 假设数据在 input_root 下的子文件夹中，例如 swarm_4_agents/*.csv
    search_path = os.path.join(input_root, "**", "*.csv")
    all_files = glob.glob(search_path, recursive=True)
    
    if not all_files:
        print(f"在 {input_root} 下未找到CSV文件")
        return
    
    files = all_files
    print(f"找到 {len(files)} 个文件。开始转换...")

    # 创建输出目录结构
    base_out = os.path.join(output_root, dataset_name)
    subsets = ['train', 'val', 'test']
    for s in subsets:
        os.makedirs(os.path.join(base_out, s), exist_ok=True)

    # 随机打乱筛选后的文件列表
    random.seed(42)
    random.shuffle(files) 
    
    total_files = len(files)
    train_split = int(total_files * 0.7)
    val_split = int(total_files * 0.85)

    global_ped_id_offset = 0 # 如果需要跨文件保持ID唯一性，可以用这个，但在utils.py中通常每个文件单独处理

    for idx, file_path in enumerate(files):
        try:
            df = pd.read_csv(file_path)
            
            # 解析列名，找出有多少个agent
            # 列名通常是 timestamp, agent_0_x, agent_0_y, ...
            # 我们需要重构数据框
            
            standard_rows = []
            
            # 获取时间戳列，假设第一列是 timestamp 或者 0.0, 0.1 等
            # 用户数据示例: timestamp,agent_0_x...
            
            if 'timestamp' not in df.columns:
                 # 尝试自动检测，或者假设第一列是时间
                 time_col = df.columns[0]
            else:
                 time_col = 'timestamp'

            # 识别所有 agent
            # 查找以 _x 结尾的列
            x_cols = [c for c in df.columns if c.endswith('_x')]
            agent_prefixes = [c[:-2] for c in x_cols] # 移除 _x
            
            # 生成 frame ID (简单用行索引 * 10 作为一个近似，或者直接用行索引)
            # utils.py 通常将 frame 视为整数帧号
            # 我们使用行号作为 frame
            
            for frame_idx, row in df.iterrows():
                frame_val = frame_idx * 10 # 假设采样间隔比较密，乘10作为帧号
                
                for agent_idx, prefix in enumerate(agent_prefixes):
                    x_col = f"{prefix}_x"
                    y_col = f"{prefix}_y"
                    z_col = f"{prefix}_z"
                    
                    if x_col in row and y_col in row and z_col in row:
                        x = row[x_col]
                        y = row[y_col]
                        z = row[z_col]
                        
                        # ped ID: 在当前文件中唯一即可
                        ped_id = agent_idx 
                        
                        standard_rows.append([frame_val, ped_id, x, y, z])
            
            # 创建新的 DataFrame
            new_df = pd.DataFrame(standard_rows, columns=['frame', 'ped', 'x', 'y', 'z'])
            
            # 确定保存路径
            if idx < train_split:
                subset = 'train'
            elif idx < val_split:
                subset = 'val'
            else:
                subset = 'test'
            
            # 为了避免文件名冲突，加上父文件夹名
            parent_dir = os.path.basename(os.path.dirname(file_path))
            filename = os.path.basename(file_path)
            out_name = f"{parent_dir}_{filename}"
            
            out_path = os.path.join(base_out, subset, out_name)
            
            # 保存，使用 tab 分隔符，不带 header (utils.py 读取时没有 header，或者你可以修改 utils.py)
            # 根据 utils.py: raw_data = pd.read_csv(..., delimiter=delim, names=["frame", "ped", "x", "y", "z"]...)
            # 这意味着文件不应该包含 header，且默认 tab 分隔
            new_df.to_csv(out_path, sep='\t', header=False, index=False)
            
        except Exception as e:
            print(f"处理文件 {file_path} 出错: {e}")

    print(f"转换完成。数据保存在 {base_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=r"D:\Trajectory prediction\drone_trajectories\3DMoTraj\swarm_trajectories\swarm_3_agents")
    parser.add_argument("--output_path", type=str, default=r"D:\Trajectory prediction\drone_trajectories\3DMoTraj\dataset")
    parser.add_argument("--dataset_name", type=str, default="swarm")
    # parser.add_argument("--num_agents", type=int, default=3) # 不再需要强制筛选数量
    
    args = parser.parse_args()
    
    convert_swarm_to_standard(args.input_path, args.output_path, args.dataset_name)
