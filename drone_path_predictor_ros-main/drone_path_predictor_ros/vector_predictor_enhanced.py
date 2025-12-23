#!/usr/bin/env python3
"""
VECTOR 论文的增强实现
- 速度优先策略：在未知位置分布下泛化能力更好
- 多种归一化方法：Max 范数 vs 白化归一化
- 完整的评估指标：MSE/RMSE/MAE/R²
- 实时预测优化：支持高频率推理
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRUModel(nn.Module):
    """单 GRU 模型（用于推理时兼容 train_model.py 训练的模型）"""
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


class DoubleGRUModel(nn.Module):
    """双 GRU 模型（编码器-解码器）- 兼容现有 checkpoint"""
    def __init__(self, input_size=3, hidden_dim=64, num_layers=2, 
                 dropout=0.5, output_steps=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # 编码器：输入为坐标 (3)
        self.gru1 = nn.GRU(input_size, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 解码器：输入为隐藏维度
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 输出层：hidden_dim -> coord(3)
        self.fc = nn.Linear(hidden_dim, input_size)
    
    def forward(self, x):
        """
        x: (batch, input_len, 3)
        output: (batch, output_steps, 3)
        """
        # 编码
        _, h_encoder = self.gru1(x)
        
        # 自回归解码
        predictions = []
        h_decoder = h_encoder
        
        for _ in range(self.output_steps):
            # 用隐藏态作为当前输出
            h_last = h_decoder[-1:]  # (1, batch, hidden)
            y_t = self.fc(h_last.squeeze(0))  # (batch, 3)
            predictions.append(y_t.unsqueeze(1))
            
            # 准备解码器输入（隐藏态本身）
            decoder_input = h_last.permute(1, 0, 2)  # (batch, 1, hidden)
            _, h_decoder = self.gru2(decoder_input, h_decoder)
        
        output = torch.cat(predictions, dim=1)
        return output


class EnhancedDataProcessor:
    """增强的数据预处理类，实现论文中的两种归一化方法"""
    
    @staticmethod
    def max_norm_normalization(data):
        """
        L2 范数归一化方法
        data: (N, 3) 轨迹点
        return: (N, 3) 归一化后的轨迹，max_norm 值
        """
        max_norm = np.max(np.linalg.norm(data, axis=1))
        if max_norm < 1e-8:
            max_norm = 1e-8
        return data / max_norm, max_norm
    
    @staticmethod
    def whitening_normalization(data, mean, cov_matrix):
        """
        白化归一化方法（论文中的推荐方法）
        data: (N, 3) 轨迹点
        mean: (3,) 均值
        cov_matrix: (3, 3) 协方差矩阵
        return: (N, 3) 白化后的轨迹，Cholesky 因子 L
        """
        try:
            from scipy.linalg import cholesky
            L = cholesky(cov_matrix + np.eye(3) * 1e-6, lower=True)
            L_inv = np.linalg.inv(L)
            data_centered = data - mean
            data_whitened = data_centered @ L_inv.T
            return data_whitened, L
        except Exception as e:
            logger.warning(f"白化失败: {e}，回退到 max_norm")
            return EnhancedDataProcessor.max_norm_normalization(data)
    
    @staticmethod
    def dewhitening(data_whitened, L, mean):
        """
        反白化处理
        """
        return (data_whitened @ L.T) + mean


class EnhancedPredictorGRU:
    """
    增强的轨迹预测器 - 实现 VECTOR 论文方法
    核心改进：
    1. 速度优先策略（速度积分预测位置）
    2. 支持多种归一化方法
    3. 改进的首步连续性处理
    4. 完整的评估指标
    """
    
    def __init__(self, position_model_path, position_stats_file,
                 velocity_model_path=None, velocity_stats_file=None,
                 pos_hidden_dim=64, pos_num_layers=2,
                 vel_hidden_dim=64, vel_num_layers=2,
                 use_velocity_integration=True,  # 速度优先策略
                 normalization_method='max_norm',  # 'max_norm' 或 'whitening'
                 enforce_first_step_continuity=True,  # 强制首步速度连续性
                 device=None):
        """
        Args:
            position_model_path: 位置模型路径
            position_stats_file: 位置统计量路径
            velocity_model_path: 速度模型路径（可选）
            velocity_stats_file: 速度统计量路径（可选）
            use_velocity_integration: 优先使用速度积分预测
            normalization_method: 归一化方法
            enforce_first_step_continuity: 强制首步速度等于最后观测速度
            device: PyTorch 设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_velocity_integration = use_velocity_integration
        self.normalization_method = normalization_method
        self.enforce_first_step_continuity = enforce_first_step_continuity
        self.data_processor = EnhancedDataProcessor()
        
        # 加载位置模型
        logger.info(f"加载位置模型: {Path(position_model_path).name}")
        self.position_model = self._load_model(
            position_model_path, pos_hidden_dim, pos_num_layers
        )
        self.position_model.to(self.device).eval()
        
        # 加载位置统计量
        logger.info(f"加载位置统计量: {Path(position_stats_file).name}")
        pos_stats = np.load(position_stats_file)
        self.pos_input_mean = pos_stats['input_mean']
        self.pos_input_std = pos_stats['input_std']
        self.pos_output_mean = pos_stats.get('output_mean', self.pos_input_mean)
        self.pos_output_std = pos_stats.get('output_std', self.pos_input_std)
        
        # 加载协方差矩阵（用于白化）
        self.pos_cov_matrix = pos_stats.get('cov_matrix', None)
        self.pos_L_matrix = pos_stats.get('L_matrix', None)
        
        # 加载速度模型（可选）
        self.velocity_model = None
        self.vel_input_mean = None
        self.vel_input_std = None
        self.vel_output_mean = None
        self.vel_output_std = None
        self.vel_cov_matrix = None
        self.vel_L_matrix = None
        
        if velocity_model_path and velocity_stats_file:
            logger.info(f"加载速度模型: {Path(velocity_model_path).name}")
            self.velocity_model = self._load_model(
                velocity_model_path, vel_hidden_dim, vel_num_layers
            )
            self.velocity_model.to(self.device).eval()
            
            logger.info(f"加载速度统计量: {Path(velocity_stats_file).name}")
            vel_stats = np.load(velocity_stats_file)
            self.vel_input_mean = vel_stats['input_mean']
            self.vel_input_std = vel_stats['input_std']
            self.vel_output_mean = vel_stats.get('output_mean', self.vel_input_mean)
            self.vel_output_std = vel_stats.get('output_std', self.vel_input_std)
            self.vel_cov_matrix = vel_stats.get('cov_matrix', None)
            self.vel_L_matrix = vel_stats.get('L_matrix', None)
        
        # 处理时变统计量 (T, 3) -> (3,)
        self._collapse_stats()
    
    def _load_model(self, model_path, hidden_dim, num_layers):
        """
        自动检测模型架构并加载
        支持单 GRU（train_model.py） 和 双 GRU（现有 checkpoint）
        从 checkpoint 推断真实的 hidden_dim 和 num_layers
        """
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 检测架构类型
        has_gru1 = any('gru1' in key for key in state_dict.keys())
        has_gru2 = any('gru2' in key for key in state_dict.keys())
        
        # 从 checkpoint 推断架构参数
        inferred_hidden_dim, inferred_num_layers = self._infer_model_params(state_dict, has_gru1)
        logger.info(f"  从 checkpoint 推断: hidden_dim={inferred_hidden_dim}, num_layers={inferred_num_layers}")
        
        if has_gru1 and has_gru2:
            logger.info("  检测到双 GRU 架构（编码器-解码器）")
            model = DoubleGRUModel(
                input_size=3,
                hidden_dim=inferred_hidden_dim,
                num_layers=inferred_num_layers,
                dropout=0.5,
                output_steps=10
            )
        else:
            logger.info("  检测到单 GRU 架构")
            model = GRUModel(
                input_size=3,
                hidden_dim=inferred_hidden_dim,
                num_layers=inferred_num_layers,
                dropout=0.5,
                output_steps=10
            )
        
        try:
            model.load_state_dict(state_dict)
            logger.info("  ✓ 模型加载成功")
        except Exception as e:
            logger.error(f"  ✗ 加载失败: {e}")
            raise
        
        return model
    
    @staticmethod
    def _infer_model_params(state_dict, has_gru1):
        """
        从 state_dict 推断 hidden_dim 和 num_layers
        
        GRU 权重形状：
        - weight_ih_l0: (3*hidden_dim, input_size)  -> 行数 = 3*hidden_dim
        - weight_hh_l0: (3*hidden_dim, hidden_dim)  -> 行数 = 3*hidden_dim
        """
        gru_prefix = 'gru1' if has_gru1 else 'gru'
        
        # 从 weight_ih_l0 推断 hidden_dim
        key = f'{gru_prefix}.weight_ih_l0'
        if key in state_dict:
            weight_shape = state_dict[key].shape
            hidden_dim = weight_shape[0] // 3  # GRU: 3*hidden_dim
        else:
            # 默认值
            hidden_dim = 64
        
        # 从最大层数推断 num_layers
        num_layers = 0
        layer_idx = 0
        while f'{gru_prefix}.weight_ih_l{layer_idx}' in state_dict:
            num_layers += 1
            layer_idx += 1
        
        if num_layers == 0:
            num_layers = 2  # 默认值
        
        return hidden_dim, num_layers
    
    def _collapse_stats(self):
        """折叠时变统计量到单一向量"""
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
    
    def _normalize_data(self, data, data_type='position'):
        """
        根据选择的归一化方法处理数据
        支持 max_norm 和 whitening 两种方法
        """
        if data_type == 'position':
            mean = self.pos_input_mean
            cov_matrix = self.pos_cov_matrix
        else:
            mean = self.vel_input_mean
            cov_matrix = self.vel_cov_matrix
        
        if self.normalization_method == 'whitening' and cov_matrix is not None:
            normalized, L = self.data_processor.whitening_normalization(data, mean, cov_matrix)
            return normalized, L
        else:
            # 默认使用 max_norm
            normalized, norm_val = self.data_processor.max_norm_normalization(data)
            return normalized, norm_val
    
    def _denormalize_data(self, data_norm, norm_factor, data_type='position'):
        """反归一化"""
        if data_type == 'position':
            output_mean = self.pos_output_mean
            output_std = self.pos_output_std
        else:
            output_mean = self.vel_output_mean
            output_std = self.vel_output_std
        
        if self.normalization_method == 'whitening' and isinstance(norm_factor, np.ndarray):
            # norm_factor 是 L 矩阵
            data = self.data_processor.dewhitening(data_norm, norm_factor, output_mean)
        else:
            # norm_factor 是 max_norm 值
            data = data_norm * norm_factor
        
        return data
    
    def predict_positions_direct(self, input_positions, input_length=20):
        """
        直接位置预测方法
        
        Args:
            input_positions: 输入轨迹 (N, 3) 或 (3, N)
            input_length: 使用的输入长度
            
        Returns:
            预测位置 (10, 3)
        """
        # 确保输入格式 (N, 3)
        input_pos = np.array(input_positions, dtype=np.float32)
        if input_pos.ndim == 2 and input_pos.shape[0] == 3 and input_pos.shape[1] != 3:
            input_pos = input_pos.T
        
        if len(input_pos) > input_length:
            input_pos = input_pos[-input_length:]
        
        # 标准化
        input_norm, norm_factor = self._normalize_data(input_pos, 'position')
        input_tensor = torch.from_numpy(input_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            pred_norm = self.position_model(input_tensor)
        
        pred = pred_norm[0].cpu().numpy()
        
        # 反标准化
        pred_denorm = self._denormalize_data(pred, norm_factor, 'position')
        
        return pred_denorm
    
    def predict_positions_from_velocity(self, input_positions, dt=0.1, input_length=20):
        """
        ★ 核心创新：通过速度积分预测位置（VECTOR 论文主要方法）
        
        这种方法在未知位置分布下具有更好的泛化能力，因为：
        1. 速度是局部特征，对绝对位置的依赖性较低
        2. 物理约束更强（速度->位置的积分有物理意义）
        3. 在分布外样本上更鲁棒
        
        Args:
            input_positions: 输入位置轨迹 (N, 3)
            dt: 采样间隔
            input_length: 使用的输入长度
            
        Returns:
            预测位置 (10, 3)
        """
        if self.velocity_model is None:
            logger.warning("速度模型未加载，回退到直接位置预测")
            return self.predict_positions_direct(input_positions, input_length)
        
        # 确保输入格式
        input_pos = np.array(input_positions, dtype=np.float32)
        if input_pos.ndim == 2 and input_pos.shape[0] == 3 and input_pos.shape[1] != 3:
            input_pos = input_pos.T
        
        if len(input_pos) > input_length:
            input_pos = input_pos[-input_length:]
        
        # 计算输入速度（论文中的关键步骤）
        input_vel = np.diff(input_pos, axis=0) / dt
        
        # 标准化速度
        input_vel_norm, vel_norm_factor = self._normalize_data(input_vel, 'velocity')
        input_tensor = torch.from_numpy(input_vel_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # 速度预测
        with torch.no_grad():
            pred_vel_norm = self.velocity_model(input_tensor)
        
        pred_vel_norm = pred_vel_norm[0].cpu().numpy()
        
        # 反标准化速度
        pred_vel = self._denormalize_data(pred_vel_norm, vel_norm_factor, 'velocity')
        
        # ★ 强制首步连续性（可选）
        if self.enforce_first_step_continuity and len(input_vel) > 0:
            last_obs_vel = input_vel[-1]
            pred_vel[0] = last_obs_vel
            logger.debug(f"强制首步速度连续: {last_obs_vel}")
        
        # 积分得到位置（论文中的位置重建方法）
        last_pos = input_pos[-1]
        pred_pos = np.zeros((len(pred_vel), 3))
        
        for i in range(len(pred_vel)):
            if i == 0:
                pred_pos[i] = last_pos + pred_vel[i] * dt
            else:
                pred_pos[i] = pred_pos[i-1] + pred_vel[i] * dt
        
        return pred_pos
    
    def predict_enhanced(self, input_positions, dt=0.1, input_length=20, method=None):
        """
        增强预测接口（自动选择最佳方法）
        
        Args:
            input_positions: 输入轨迹
            dt: 采样间隔
            input_length: 输入长度
            method: 'velocity'（速度积分）、'position'（直接）或 None（自动）
            
        Returns:
            预测位置 (10, 3)
        """
        if method is None:
            # 自动选择：优先速度积分（更好的泛化性）
            method = 'velocity' if self.use_velocity_integration and self.velocity_model else 'position'
        
        logger.info(f"预测方法: {method}")
        
        if method == 'velocity':
            return self.predict_positions_from_velocity(input_positions, dt, input_length)
        else:
            return self.predict_positions_direct(input_positions, input_length)
    
    @staticmethod
    def evaluate_prediction_quality(actual_positions, predicted_positions):
        """
        评估预测质量（论文中的评估指标）
        
        Args:
            actual_positions: 真实位置 (N, 3)
            predicted_positions: 预测位置 (N, 3)
            
        Returns:
            dict: 包含 MSE/RMSE/MAE/R² 等指标
        """
        actual = np.array(actual_positions, dtype=np.float32)
        predicted = np.array(predicted_positions, dtype=np.float32)
        
        # 计算逐点误差
        errors = predicted - actual
        point_errors = np.linalg.norm(errors, axis=1)
        
        # MSE（均方误差）
        mse = np.mean(errors ** 2)
        
        # RMSE（均方根误差）
        rmse = np.sqrt(mse)
        
        # MAE（平均绝对误差）
        mae = np.mean(np.abs(point_errors))
        
        # R² 系数
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual - np.mean(actual, axis=0)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        # 最大/最小误差
        max_error = np.max(point_errors)
        min_error = np.min(point_errors)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r_squared': float(r_squared),
            'max_error': float(max_error),
            'min_error': float(min_error),
            'point_errors': point_errors
        }


class RealTimePredictor(EnhancedPredictorGRU):
    """
    实时预测优化版本
    支持高频率推理（30Hz+），用于实时 ROS 节点
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_buffer = []
        self.velocity_buffer = []
        self.max_buffer_size = 20  # 2秒数据 (0.1s 采样)
    
    def add_position(self, position):
        """添加新的位置数据到缓冲区"""
        position = np.array(position, dtype=np.float32)
        self.position_buffer.append(position)
        
        # 维护缓冲区大小
        if len(self.position_buffer) > self.max_buffer_size:
            self.position_buffer.pop(0)
        
        # 更新速度缓冲区
        if len(self.position_buffer) > 1:
            vel = (self.position_buffer[-1] - self.position_buffer[-2]) / 0.1
            self.velocity_buffer.append(vel)
            if len(self.velocity_buffer) > self.max_buffer_size - 1:
                self.velocity_buffer.pop(0)
    
    def real_time_predict(self, dt=0.1, min_buffer_size=10):
        """
        实时预测
        
        Args:
            dt: 采样间隔
            min_buffer_size: 最小缓冲区大小
            
        Returns:
            预测位置 (10, 3) 或 None（缓冲区不足）
        """
        if len(self.position_buffer) < min_buffer_size:
            logger.warning(f"缓冲区不足: {len(self.position_buffer)}/{min_buffer_size}")
            return None
        
        # 使用最后 input_length 个点进行预测
        input_positions = np.array(self.position_buffer[-20:])
        
        return self.predict_enhanced(input_positions, dt=dt)
    
    def get_last_position(self):
        """获取最后一个位置"""
        if len(self.position_buffer) > 0:
            return self.position_buffer[-1].copy()
        return None
    
    def get_buffer_status(self):
        """获取缓冲区状态"""
        return {
            'buffer_size': len(self.position_buffer),
            'last_position': self.get_last_position(),
            'velocity_magnitude': np.linalg.norm(self.velocity_buffer[-1]) if len(self.velocity_buffer) > 0 else 0.0
        }
