"""
python process_multiple_trajectories.py ^
  "d:\Trajectory prediction\drone_trajectories\new_random_traj_100ms" ^
  "d:\Trajectory prediction\Synthetic-UAV-Flight-Trajectories" ^
   20 10 ^
  --save_path merged_segments.npz
"""
import pandas as pd
import argparse
import numpy as np
import os
from scipy.interpolate import splrep
import sys
from tqdm import tqdm

def read_trajectory(file_path):
    df = pd.read_csv(file_path)
    return df

def process_trajectory(df, inp_seg_len, out_seg_len, compute_velocity=False, precomputed_velocity=False):
    input_segments = []
    output_segments = []
    knots_input = []
    coeffs_input = []
    knots_output = []
    coeffs_output = []
    
    if compute_velocity and not precomputed_velocity:
        # Compute velocity (difference in position / difference in time)
        df_velocity = df[['tx', 'ty', 'tz']].diff() / df['timestamp'].diff().values[:, None]
        df = df_velocity.dropna()
    
    for i in range(0, len(df) - inp_seg_len - out_seg_len + 1):
        if precomputed_velocity:
            input_segment = df.iloc[i:i+inp_seg_len][['vx', 'vy', 'vz']].values.T
            output_segment = df.iloc[i+inp_seg_len:i+inp_seg_len+out_seg_len][['vx', 'vy', 'vz']].values.T
        else:
            input_segment = df.iloc[i:i+inp_seg_len][['tx', 'ty', 'tz']].values.T
            output_segment = df.iloc[i+inp_seg_len:i+inp_seg_len+out_seg_len][['tx', 'ty', 'tz']].values.T
        
        input_segments.append(input_segment)
        output_segments.append(output_segment)
        
        # Compute spline representation for the input segment
        tck_x_input = splrep(np.arange(inp_seg_len), input_segment[0], k=3)
        knots_input.append(tck_x_input[0])
        coeffs_input.append(tck_x_input[1])
        
        # Compute spline representation for the output segment
        tck_x_output = splrep(np.arange(out_seg_len), output_segment[0], k=3)
        knots_output.append(tck_x_output[0])
        coeffs_output.append(tck_x_output[1])
        
    input_segments = np.array(input_segments).transpose(1, 2, 0)
    output_segments = np.array(output_segments).transpose(1, 2, 0)
    
    return input_segments, output_segments, knots_input, coeffs_input, knots_output, coeffs_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process drone trajectory data.')
    parser.add_argument('dir_paths', nargs='+', type=str, help='Path(s) to directory(ies) containing the drone trajectory files (you can pass multiple)')
    parser.add_argument('inp_seg_len', type=int, help='Length of input segments')
    parser.add_argument('out_seg_len', type=int, help='Length of output segments')
    parser.add_argument('--compute_velocity', action='store_true', help='Compute velocity trajectory instead of position')
    parser.add_argument('--precomputed_velocity', action='store_true', help='Use precomputed velocity instead of computing it')
    parser.add_argument('--save_path', type=str, default='segments.npz', help='Path to save the .npz file')
    parser.add_argument('--test', action='store_true', help='Test mode: only process first N files')
    
    args = parser.parse_args()
    
    dir_paths = args.dir_paths
    inp_seg_len = args.inp_seg_len
    out_seg_len = args.out_seg_len
    compute_velocity = args.compute_velocity
    precomputed_velocity = args.precomputed_velocity
    save_path = args.save_path
    
    all_input_segments = []
    all_output_segments = []
    all_knots_input = []
    all_coeffs_input = []
    all_knots_output = []
    all_coeffs_output = []
    
    # 列出所有 CSV/TXT 文件（支持多个路径）
    file_paths = []
    for d in dir_paths:
        if not os.path.isdir(d):
            print(f"Warning: {d} is not a directory, skipping")
            continue
        files = [f for f in os.listdir(d) if f.endswith('.csv') or f.endswith('.txt')]
        file_paths.extend([os.path.join(d, f) for f in files])

    print(f"Found {len(file_paths)} trajectory files in {len(dir_paths)} directories")
    
    skipped = 0
    processed = 0
    
    for file_path in tqdm(file_paths, desc="Processing trajectories"):
        try:
            filename = os.path.basename(file_path)
            df = read_trajectory(file_path)
            
            # 检查必要列
            cols = set(df.columns)
            if compute_velocity and not precomputed_velocity:
                required = {'timestamp', 'tx', 'ty', 'tz'}
            elif precomputed_velocity:
                required = {'vx', 'vy', 'vz'}
            else:
                required = {'tx', 'ty', 'tz'}
            
            if not required.issubset(cols):
                print(f"\nWarning: {file_path} missing columns {required - cols}, skipping")
                skipped += 1
                continue
            
            # 检查数据行数
            if len(df) < inp_seg_len + out_seg_len:
                skipped += 1
                continue
            
            input_segments, output_segments, knots_input, coeffs_input, knots_output, coeffs_output = process_trajectory(df, inp_seg_len, out_seg_len, compute_velocity, precomputed_velocity)
            
            all_input_segments.append(input_segments)
            all_output_segments.append(output_segments)
            all_knots_input.extend(knots_input)
            all_coeffs_input.extend(coeffs_input)
            all_knots_output.extend(knots_output)
            all_coeffs_output.extend(coeffs_output)
            
            processed += 1
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessed: {processed}, Skipped: {skipped}")
    
    all_input_segments = np.concatenate(all_input_segments, axis=2)
    all_output_segments = np.concatenate(all_output_segments, axis=2)
    
    num_input_segments = all_input_segments.shape[2]
    num_output_segments = all_output_segments.shape[2]
    
    print(f"Total number of input segments: {num_input_segments}")
    print(f"Total number of output segments: {num_output_segments}")
    
    # Save segments, spline data, and additional information to .npz file
    npz_data = {
        'input_segments': all_input_segments,
        'output_segments': all_output_segments,
        'num_input_segments': num_input_segments,
        'num_output_segments': num_output_segments,
        'inp_seg_len': inp_seg_len,
        'out_seg_len': out_seg_len
    }
    
    for i in range(num_input_segments):
        npz_data[f'knots_input_{i}'] = all_knots_input[i]
        npz_data[f'coeffs_input_{i}'] = all_coeffs_input[i]
        npz_data[f'spline_data_input_{i}'] = np.concatenate([all_knots_input[i], all_coeffs_input[i]])
        
    for i in range(num_output_segments):
        npz_data[f'knots_output_{i}'] = all_knots_output[i]
        npz_data[f'coeffs_output_{i}'] = all_coeffs_output[i]
        npz_data[f'spline_data_output_{i}'] = np.concatenate([all_knots_output[i], all_coeffs_output[i]])
    
    np.savez(save_path, **npz_data)
