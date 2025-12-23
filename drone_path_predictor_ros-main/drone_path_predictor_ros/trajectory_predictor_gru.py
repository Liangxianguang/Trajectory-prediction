#!/usr/bin/env python3
"""
新的轨迹预测器 - 兼容 train_model.py 训练出的 GRU 模型
"""
import numpy as np
import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """与 train_model.py 保持一致的 GRU 模型"""
    def __init__(self, input_size=3, hidden_dim=64, num_layers=2, 
                 dropout=0.5, output_steps=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, input_size)
    
    def forward(self, x):
        """
        x: (batch, input_len, 3)
        output: (batch, output_steps, 3)
        """
        _, h = self.gru(x)
        predictions = []
        h_t = h
        
        for _ in range(self.output_steps):
            h_last = h_t[-1:]
            y_t = self.fc(h_last.squeeze(0))
            predictions.append(y_t.unsqueeze(1))
            y_t_in = y_t.unsqueeze(1)
            _, h_t = self.gru(y_t_in, h_t)
        
        output = torch.cat(predictions, dim=1)
        return output


class PredictorGRU:
    """
    新版轨迹预测器 - 支持位置和速度预测
    兼容 train_model.py 训练出的模型
    """
    def __init__(self, position_model_path, position_stats_file,
                 velocity_model_path=None, velocity_stats_file=None,
                 pos_hidden_dim=64, pos_num_layers=2, pos_dropout=0.5,
                 vel_hidden_dim=64, vel_num_layers=2, vel_dropout=0.5,
                 use_whitening=False, device=None):
        """
        Args:
            position_model_path: 位置模型 .pth 文件路径
            position_stats_file: 位置统计量 .npz 文件路径
            velocity_model_path: 速度模型 .pth 文件路径 (可选)
            velocity_stats_file: 速度统计量 .npz 文件路径 (可选)
            pos_hidden_dim: 位置模型隐藏层维度
            pos_num_layers: 位置模型 GRU 层数
            pos_dropout: 位置模型 dropout
            vel_hidden_dim: 速度模型隐藏层维度
            vel_num_layers: 速度模型 GRU 层数
            vel_dropout: 速度模型 dropout
            use_whitening: 是否使用白化处理 (暂未实现)
            device: PyTorch 设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_whitening = use_whitening
        
        # 加载位置模型
        print(f"加载位置模型: {position_model_path}")
        self.position_model = GRUModel(
            input_size=3,
            hidden_dim=pos_hidden_dim,
            num_layers=pos_num_layers,
            dropout=pos_dropout,
            output_steps=10
        )
        self.position_model.load_state_dict(torch.load(position_model_path, map_location='cpu'))
        self.position_model.to(self.device).eval()
        
        # 加载位置统计量
        print(f"加载位置统计量: {position_stats_file}")
        pos_stats = np.load(position_stats_file)
        self.pos_input_mean = pos_stats['input_mean']
        self.pos_input_std = pos_stats['input_std']
        self.pos_output_mean = pos_stats.get('output_mean', self.pos_input_mean)
        self.pos_output_std = pos_stats.get('output_std', self.pos_input_std)
        
        # 加载速度模型 (可选)
        self.velocity_model = None
        self.vel_input_mean = None
        self.vel_input_std = None
        self.vel_output_mean = None
        self.vel_output_std = None
        
        if velocity_model_path and velocity_stats_file:
            print(f"加载速度模型: {velocity_model_path}")
            self.velocity_model = GRUModel(
                input_size=3,
                hidden_dim=vel_hidden_dim,
                num_layers=vel_num_layers,
                dropout=vel_dropout,
                output_steps=10
            )
            self.velocity_model.load_state_dict(torch.load(velocity_model_path, map_location='cpu'))
            self.velocity_model.to(self.device).eval()
            
            print(f"加载速度统计量: {velocity_stats_file}")
            vel_stats = np.load(velocity_stats_file)
            self.vel_input_mean = vel_stats['input_mean']
            self.vel_input_std = vel_stats['input_std']
            self.vel_output_mean = vel_stats.get('output_mean', self.vel_input_mean)
            self.vel_output_std = vel_stats.get('output_std', self.vel_input_std)
        
        # ★ 重要：所有统计量加载完成后再 collapse（确保速度统计量也被处理）
        self._collapse_stats()
    
    def _collapse_stats(self):
        """处理时变统计量 (T, 3) -> (3,)"""
        def collapse(arr):
            arr = np.array(arr)
            if arr.ndim == 2:
                return arr.mean(axis=0)
            return arr
        
        self.pos_input_mean = collapse(self.pos_input_mean)
        self.pos_input_std = collapse(self.pos_input_std)
        self.pos_output_mean = collapse(self.pos_output_mean)
        self.pos_output_std = collapse(self.pos_output_std)
        
        if self.vel_input_mean is not None:
            self.vel_input_mean = collapse(self.vel_input_mean)
            self.vel_input_std = collapse(self.vel_input_std)
            self.vel_output_mean = collapse(self.vel_output_mean)
            self.vel_output_std = collapse(self.vel_output_std)
    
    def predict_positions(self, input_positions, input_length=None):
        """
        预测未来位置序列
        
        Args:
            input_positions: 输入位置序列 (N, 3) 或 (3, N)
            input_length: 使用的输入长度 (若为 None 使用全部)
            
        Returns:
            predictions: 预测位置 (10, 3) 在物理空间
        """
        # 确保输入为 (N, 3) 格式
        input_pos = np.array(input_positions, dtype=np.float32)
        if input_pos.ndim == 2 and input_pos.shape[0] == 3 and input_pos.shape[1] != 3:
            input_pos = input_pos.T  # (3, N) -> (N, 3)
        
        # 取最后 input_length 个点
        if input_length is not None and len(input_pos) > input_length:
            input_pos = input_pos[-input_length:]
        
        # 归一化
        input_norm = (input_pos - self.pos_input_mean) / (self.pos_input_std + 1e-8)
        input_tensor = torch.from_numpy(input_norm.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, N, 3)
        
        # 推理
        with torch.no_grad():
            pred_norm = self.position_model(input_tensor)  # (1, 10, 3)
        
        # 反归一化
        pred = pred_norm[0].cpu().numpy()
        pred = pred * self.pos_output_std + self.pos_output_mean
        
        return pred  # (10, 3)
    
    def predict_velocities(self, input_velocities, input_length=None):
        """
        预测未来速度序列
        
        Args:
            input_velocities: 输入速度序列 (N, 3)
            input_length: 使用的输入长度
            
        Returns:
            predictions: 预测速度 (10, 3)
        """
        if self.velocity_model is None:
            raise ValueError("速度模型未加载")
        
        input_vel = np.array(input_velocities, dtype=np.float32)
        if input_vel.ndim == 2 and input_vel.shape[0] == 3 and input_vel.shape[1] != 3:
            input_vel = input_vel.T
        
        if input_length is not None and len(input_vel) > input_length:
            input_vel = input_vel[-input_length:]
        
        # 归一化
        input_norm = (input_vel - self.vel_input_mean) / (self.vel_input_std + 1e-8)
        input_tensor = torch.from_numpy(input_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            pred_norm = self.velocity_model(input_tensor)
        
        # 反归一化
        pred = pred_norm[0].cpu().numpy()
        pred = pred * self.vel_output_std + self.vel_output_mean
        
        return pred  # (10, 3)
    
    def predict_positions_from_velocity(self, input_positions, dt=0.1):
        """
        基于速度预测位置序列
        
        Args:
            input_positions: 输入位置序列 (N, 3)
            dt: 时间间隔
            
        Returns:
            predictions: 预测位置 (10, 3)
        """
        if self.velocity_model is None:
            # 如果没有速度模型，直接用位置模型
            return self.predict_positions(input_positions)
        
        # 计算输入速度
        input_pos = np.array(input_positions, dtype=np.float32)
        if input_pos.ndim == 2 and input_pos.shape[0] == 3:
            input_pos = input_pos.T
        
        input_vel = np.diff(input_pos, axis=0) / dt
        
        # 预测速度
        pred_vel = self.predict_velocities(input_vel)

        # ★ 强制首步连续性：将第一步预测速度替换为最后观测速度，减少积分时的跳变
        try:
            if len(input_vel) > 0:
                pred_vel[0] = input_vel[-1]
        except Exception:
            # 如果 shape 不匹配则忽略此强制步骤
            pass

        # 积分得到位置
        last_pos = input_pos[-1]
        pred_pos = last_pos + np.cumsum(pred_vel * dt, axis=0)

        return pred_pos


# 向后兼容：保持原有接口名称
class Predictor(PredictorGRU):
    """向后兼容的别名"""
    pass
