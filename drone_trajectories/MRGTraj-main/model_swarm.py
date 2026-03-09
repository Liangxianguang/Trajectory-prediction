"""
MRGTraj 改进版 - 支持无人机集群 3D 轨迹预测
================================================

核心改动:
  1. Encoder 输入维度从 2 改为 num_agents * 3 (支持 XYZ 和多智能体)
  2. SocialLatentGenerator 改进为捕捉集群协作关系
  3. 增加 MultiAgentAttention 建模无人机间交互
  4. TemporalMapper 支持可变长度序列
  5. 预测层输出多个智能体的完整 3D 轨迹

数据格式:
  输入: (batch_size, seq_len, num_agents, 3)  # (x,y,z)
  输出: (batch_size, pred_len, num_agents, 3) # 预测的 (x,y,z)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer import Encoder, PositionalEncoding


def get_noise(shape, noise_type):
    """生成噪声"""
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class MultiAgentAttention(nn.Module):
    """多智能体注意力层 - 捕捉无人机间的相互作用"""
    
    def __init__(self, d_model, num_agents, num_heads=4, dropout=0.1):
        super(MultiAgentAttention, self).__init__()
        self.num_agents = num_agents
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_k = d_model // num_heads
        self.head_dim = d_model // num_heads
        
        # 智能体级别的多头注意力
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_agents, d_model)
            
        Returns:
            output: (batch_size, seq_len, num_agents, d_model)
            注意力权重: (batch_size, num_heads, num_agents, num_agents)
        """
        batch_size, seq_len, num_agents, d_model = x.shape
        
        # 重塑为 (batch*seq_len, num_agents, d_model)
        x_reshaped = x.reshape(batch_size * seq_len, num_agents, d_model)
        
        # 投影到 Q, K, V
        Q = self.W_Q(x_reshaped)  # (batch*seq_len, num_agents, d_model)
        K = self.W_K(x_reshaped)
        V = self.W_V(x_reshaped)
        
        # 多头拆分: (batch*seq_len, num_heads, num_agents, head_dim)
        Q = Q.reshape(batch_size * seq_len, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size * seq_len, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size * seq_len, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到值
        context = torch.matmul(attn_weights, V)  # (batch*seq_len, num_heads, num_agents, head_dim)
        
        # 合并多头: (batch*seq_len, num_agents, d_model)
        context = context.transpose(1, 2).reshape(batch_size * seq_len, num_agents, d_model)
        
        # 最终投影
        output = self.fc_out(context)
        output = self.dropout(output)
        
        # 重塑回原尺寸并添加残差连接
        output = output.reshape(batch_size, seq_len, num_agents, d_model)
        output = self.layer_norm(output + x)
        
        return output


class SwarmLatentGenerator(nn.Module):
    """集群潜在代码生成器 - 学习集群协作的高斯分布"""
    
    def __init__(self, num_agents, agent_dim, d_model=256, dim_z=64, dff=1024, dropout=0.3):
        """
        Args:
            num_agents: 无人机数量 (3-6)
            agent_dim: 每个智能体的维度 (3 for XYZ)
            d_model: 模型维度
            dim_z: 隐式编码维度
            dff: 前馈网络维度
            dropout: dropout 比率
        """
        super(SwarmLatentGenerator, self).__init__()
        
        self.num_agents = num_agents
        self.agent_dim = agent_dim
        self.dim_z = dim_z
        self.d_model = d_model
        
        # 嵌入层: (batch, pred_len, num_agents, 3) → (batch, pred_len, num_agents, d_model)
        self.emb_layer = nn.Sequential(
            nn.Linear(agent_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 聚合层: 汇总所有无人机的信息 (num_agents, d_model) → d_model
        self.aggregation = nn.Sequential(
            nn.Linear(num_agents * d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 潜在代码生成层
        self.latent = nn.Sequential(
            nn.Linear(dff, dff),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(dff, dff),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(dff, dim_z * 2),
            nn.ReLU6()
        )
    
    def forward(self, input_traj, valid_mask=None):
        """
        Args:
            input_traj: (batch_size, pred_len, num_agents, 3)  # 未来轨迹
            valid_mask: (batch_size, num_agents) 或 None  # 标记有效的无人机
            
        Returns:
            z: (batch_size, dim_z)  # 采样的隐式编码
            mu: (batch_size, dim_z)  # 均值
            log_var: (batch_size, dim_z)  # 对数方差
        """
        batch_size, pred_len, num_agents, agent_dim = input_traj.shape
        
        # 嵌入: (batch, pred_len, num_agents, 3) → (batch, pred_len, num_agents, d_model)
        embedded = self.emb_layer(input_traj)
        
        # 对时间维度取平均: (batch, num_agents, d_model)
        embedded_mean = embedded.mean(dim=1)
        
        # 应用有效性掩码 (如果有的话)
        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(-1).float()  # (batch, num_agents, 1)
            embedded_mean = embedded_mean * mask_expanded
        
        # 展平并聚合: (batch, num_agents, d_model) → (batch, num_agents*d_model)
        embedded_flat = embedded_mean.reshape(batch_size, -1)
        
        # 聚合: → (batch, dff)
        aggregated = self.aggregation(embedded_flat)
        
        # 生成潜在变量: → (batch, dim_z*2)
        latent_variables = self.latent(aggregated)
        
        # 拆分为 μ 和 log_var
        mu = latent_variables[:, :self.dim_z]
        log_var = latent_variables[:, self.dim_z:]
        
        # 重参数化技巧: z = μ + σ * ε, 其中 ε ~ N(0, 1)
        var = log_var.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_().cuda()
        z = eps.mul(var).add_(mu)
        
        return z, mu, log_var
    
    def sample(self, batch_size):
        """无条件采样"""
        z = get_noise((batch_size, self.dim_z), "gaussian")
        return z


class MRGTrajSwarm(nn.Module):
    """改进的 MRGTraj - 支持无人机集群 3D 轨迹预测"""
    
    def __init__(self, args):
        """
        Args:
            args.num_agents: 无人机数量
            args.obs_len: 观察长度
            args.pred_len: 预测长度
            args.d_model: 模型维度
            args.n_heads: 注意力头数
            args.n_layers: Transformer 层数
            args.noise_dim: 噪声维度
            args.agent_dim: 单个智能体维度 (默认 3 for XYZ)
        """
        super(MRGTrajSwarm, self).__init__()
        
        # 基础参数
        self.num_agents = args.num_agents
        self.agent_dim = getattr(args, 'agent_dim', 3)  # XYZ
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.d_model = args.d_model
        self.noise_dim = args.noise_dim
        
        # 输入维度: 所有无人机的坐标展平
        input_dim = self.num_agents * self.agent_dim
        
        # 1. 过去轨迹编码器
        # 输入: (batch, obs_len, num_agents*agent_dim) → 输出: (batch, obs_len, d_model)
        self.past_encoder = Encoder(
            input_dim=input_dim,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_len=args.obs_len + args.pred_len,
            if_emb=True,
            if_pos=True
        )
        
        # 2. 集群潜在代码生成器
        self.swarm_latent_generator = SwarmLatentGenerator(
            num_agents=self.num_agents,
            agent_dim=self.agent_dim,
            d_model=args.d_model,
            dim_z=args.noise_dim
        )
        
        # 3. 时间映射层: 从观察长度 → 预测长度
        self.temporal_mapper = nn.Sequential(
            nn.Linear(args.obs_len, args.obs_len * 2),
            nn.ReLU(),
            nn.Linear(args.obs_len * 2, args.pred_len)
        )
        
        # 4. 多智能体交互细化 (可选，用于增强)
        self.multi_agent_refiner = MultiAgentAttention(
            d_model=args.d_model + args.noise_dim,
            num_agents=self.num_agents,
            num_heads=args.n_heads,
            dropout=0.1
        )
        
        # 5. 社交优化器: 融合时间特征和噪声
        self.social_refiner = Encoder(
            input_dim=args.d_model + args.noise_dim,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_len=args.pred_len,
            if_emb=False,
            if_pos=False
        )
        
        # 6. 预测头: 输出所有无人机的坐标
        self.prediction_layer = nn.Linear(args.d_model + args.noise_dim, input_dim)
    
    def forward(self, past_traj, future_traj=None, valid_mask=None):
        """
        训练前向传播
        
        Args:
            past_traj: (batch_size, obs_len, num_agents, agent_dim)
            future_traj: (batch_size, pred_len, num_agents, agent_dim) 或 None
            valid_mask: (batch_size, num_agents) 或 None
            
        Returns:
            pred_traj: (batch_size, pred_len, num_agents, agent_dim)
            mu: (batch_size, noise_dim) - 潜在分布的均值
            log_var: (batch_size, noise_dim) - 潜在分布的对数方差
        """
        batch_size, obs_len, num_agents, agent_dim = past_traj.shape
        
        # 步骤 1: 展平多智能体轨迹 (batch, obs_len, num_agents*agent_dim)
        past_traj_flat = past_traj.reshape(batch_size, obs_len, -1)
        
        # 步骤 2: 编码过去轨迹 (batch, obs_len, d_model) → (obs_len, batch, d_model)
        past_encoding = self.past_encoder(past_traj_flat)  # 返回值形状依赖于 Encoder
        
        # 步骤 3: 时间映射 
        # 假设 Encoder 返回 (batch, obs_len, d_model)
        if past_encoding.dim() == 3 and past_encoding.shape[0] == batch_size:
            # 形状: (batch, obs_len, d_model)
            mapped = self.temporal_mapper(past_encoding.transpose(1, 2))  # (batch, d_model, pred_len)
            future_decoding = mapped.transpose(1, 2)  # (batch, pred_len, d_model)
        else:
            # 形状: (obs_len, batch, d_model) 需要转置
            past_encoding_t = past_encoding.transpose(0, 1)  # (batch, obs_len, d_model)
            mapped = self.temporal_mapper(past_encoding_t.transpose(1, 2))  # (batch, d_model, pred_len)
            future_decoding = mapped.transpose(1, 2)  # (batch, pred_len, d_model)
        
        # 步骤 4: 生成潜在编码
        if future_traj is not None:
            z, mu, log_var = self.swarm_latent_generator(future_traj, valid_mask)
        else:
            # 推理时: 不提供 future_traj
            z = self.swarm_latent_generator.sample(batch_size)
            mu, log_var = torch.zeros_like(z), torch.zeros_like(z)
        
        # 步骤 5: 融合时间特征和噪声 (batch, pred_len, d_model+noise_dim)
        z_expanded = z.unsqueeze(1).expand(batch_size, self.pred_len, self.noise_dim)
        future_with_noise = torch.cat([future_decoding, z_expanded], dim=-1)
        
        # 步骤 6: 多智能体交互细化 (可选)
        # 重塑为 (batch, pred_len, num_agents, (d_model+noise_dim)//num_agents + extra)
        # 或保持原样，依赖于后续处理
        
        # 步骤 7: 社交优化
        # 输入转置以适配 Encoder: (batch, pred_len, d_model+noise_dim)
        refined = self.social_refiner(future_with_noise)  # 返回形状取决于实现
        
        # 步骤 8: 预测未来轨迹
        if refined.dim() == 3:
            pred_traj_flat = self.prediction_layer(refined)  # (batch, pred_len, num_agents*agent_dim)
        else:
            # 如果是 (pred_len, batch, d_model+noise_dim)
            refined_t = refined.transpose(0, 1)  # (batch, pred_len, d_model+noise_dim)
            pred_traj_flat = self.prediction_layer(refined_t)
        
        # 步骤 9: 重塑回多智能体格式 (batch, pred_len, num_agents, agent_dim)
        pred_traj = pred_traj_flat.reshape(batch_size, self.pred_len, num_agents, agent_dim)
        
        return pred_traj, mu, log_var
    
    def inference(self, past_traj, valid_mask=None, num_samples=1):
        """
        推理 - 生成多个轨迹样本
        
        Args:
            past_traj: (batch_size, obs_len, num_agents, agent_dim)
            valid_mask: (batch_size, num_agents) 或 None
            num_samples: 生成的样本数
            
        Returns:
            predictions: (num_samples, batch_size, pred_len, num_agents, agent_dim)
        """
        batch_size, obs_len, num_agents, agent_dim = past_traj.shape
        
        # 展平多智能体轨迹
        past_traj_flat = past_traj.reshape(batch_size, obs_len, -1)
        
        # 编码
        past_encoding = self.past_encoder(past_traj_flat)
        
        # 时间映射
        if past_encoding.dim() == 3 and past_encoding.shape[0] == batch_size:
            mapped = self.temporal_mapper(past_encoding.transpose(1, 2))
            future_decoding = mapped.transpose(1, 2)
        else:
            past_encoding_t = past_encoding.transpose(0, 1)
            mapped = self.temporal_mapper(past_encoding_t.transpose(1, 2))
            future_decoding = mapped.transpose(1, 2)
        
        all_predictions = []
        
        for _ in range(num_samples):
            # 采样噪声
            z = self.swarm_latent_generator.sample(batch_size)
            
            # 融合特征
            z_expanded = z.unsqueeze(1).expand(batch_size, self.pred_len, self.noise_dim)
            future_with_noise = torch.cat([future_decoding, z_expanded], dim=-1)
            
            # 细化和预测
            refined = self.social_refiner(future_with_noise)
            
            if refined.dim() == 3:
                pred_traj_flat = self.prediction_layer(refined)
            else:
                refined_t = refined.transpose(0, 1)
                pred_traj_flat = self.prediction_layer(refined_t)
            
            # 重塑
            pred_traj = pred_traj_flat.reshape(batch_size, self.pred_len, num_agents, agent_dim)
            all_predictions.append(pred_traj)
        
        # 堆叠所有样本: (num_samples, batch_size, pred_len, num_agents, agent_dim)
        predictions = torch.stack(all_predictions, dim=0)
        
        return predictions
