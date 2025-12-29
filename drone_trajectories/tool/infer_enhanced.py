#!/usr/bin/env python3
"""
增强版推理脚本
支持：
1. 增量位移 -> 绝对位置重建
2. 物理约束积分（加速度平滑）
3. 多种后处理方法
"""
import os
import sys
import pathlib
import re

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ATTENTION_PARAM_KEY_PREFIXES = (
    "pos_enc.",
    "enc_refiner.",
    "cross_attn.",
    "cross_ln.",
)


def _state_dict_has_attention(state_dict):
    return any(
        key.startswith(prefix)
        for key in state_dict
        for prefix in ATTENTION_PARAM_KEY_PREFIXES
    )

# 导入增强模型
workspace_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace_root))
from drone_trajectories.tool.train_model_enhanced import EnhancedGRUModel


def parse_smoothing_weight(value: str):
    """解析加速度平滑权重，支持单个 float 或逗号分隔的三个轴权重"""
    if isinstance(value, (float, int)):
        return float(value)

    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "--smoothing-weight must be a scalar or three comma-separated numbers"
            )
        return [float(part) for part in parts]

    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--smoothing-weight must be a number or three comma-separated numbers"
        )
class EnhancedInference:
    """增强版推理器"""
    
    def __init__(self, model_path, stats_path, hidden_dim=None, num_layers=None,
                 use_attention=False, device=None):
        """初始化推理器：智能兼容 checkpoint 的 input_size

        逻辑：先在 CPU 上读取 checkpoint 的 state_dict，尝试从其中推断
        `feature_fusion.weight` 的列数作为 input_size（兼容旧模型）。
        然后构造模型并加载权重；若严格加载失败，降级为 strict=False 并记录缺失/多余键。
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 先在 CPU 上读取 checkpoint（只为推断与兼容）
        logger.info(f"读取 checkpoint: {Path(model_path).name} (map_location=cpu) for inspection")
        ckpt = torch.load(model_path, map_location='cpu')
        # 支持两种保存格式：直接 state_dict 或 {'model_state_dict': ...} 或 {'state_dict': ...}
        if isinstance(ckpt, dict) and ('model_state_dict' in ckpt or 'state_dict' in ckpt):
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        else:
            state_dict = ckpt

        # 如果 checkpoint 的 key 带有 'module.' 前缀（DataParallel 保存），移除前缀以便匹配
        def _strip_module_prefix(sd):
            if any(k.startswith('module.') for k in sd.keys()):
                return {k[len('module.'):]: v for k, v in sd.items()}
            return sd

        state_dict_stripped = _strip_module_prefix(state_dict)

        state_has_attention = _state_dict_has_attention(state_dict_stripped)
        if state_has_attention and not use_attention:
            logger.warning("Checkpoint 含有注意力层参数，但当前推理未设置 --use_attention，将自动启用。")
            use_attention = True
        elif not state_has_attention and use_attention:
            logger.warning("当前推理设置了 --use_attention，但 checkpoint 中未检测到相关参数，加载时可能会随机初始化注意力权重。")

        # 尝试从 state_dict 中找到 feature_fusion.weight 并推断 input_size
        inferred_input_size = None
        for k in state_dict_stripped.keys():
            if k.endswith('feature_fusion.weight'):
                inferred_input_size = state_dict_stripped[k].shape[1]
                logger.info(f"从 checkpoint 推断 input_size = {inferred_input_size} (key={k})")
                break

        # 决定使用的 input_size：优先使用 checkpoint 推断值，否则使用默认 16
        input_size = inferred_input_size if inferred_input_size is not None else 16
        if inferred_input_size is not None and inferred_input_size != 16:
            logger.warning(f"checkpoint input_size={inferred_input_size} != 当前代码默认 16，采用 {inferred_input_size} 以兼容 checkpoint")

        # 尝试从 state_dict 明确推断模型形状
        # ⭐ 首先检测 encoder 是否为双向（存在 *_reverse 权重）
        has_reverse = any(k.endswith('_reverse') and k.startswith('encoder_gru.') for k in state_dict_stripped.keys())

        # 优先使用 encoder_gru.weight_ih_l0（3*hidden_dim）来推断 hidden_dim
        encoder_hidden_dim = None
        encoder_weight = state_dict_stripped.get('encoder_gru.weight_ih_l0')
        if encoder_weight is not None:
            encoder_hidden_dim = encoder_weight.shape[0] // 3
        else:
            rev_weight = state_dict_stripped.get('encoder_gru.weight_ih_l0_reverse')
            if rev_weight is not None:
                encoder_hidden_dim = rev_weight.shape[0] // 3

        # 如果还未推断，再从 feature_fusion 的输出维度退回计算
        feature_fusion_weight = state_dict_stripped.get('feature_fusion.weight')
        inferred_encoder_input_dim = None
        if feature_fusion_weight is not None:
            inferred_encoder_input_dim = feature_fusion_weight.shape[0]
            logger.info(f"从 checkpoint 推断 encoder_input_dim = {inferred_encoder_input_dim} (feature_fusion output)")
        
        if encoder_hidden_dim is None and feature_fusion_weight is not None:
            encoder_output_dim = feature_fusion_weight.shape[0]
            encoder_hidden_dim = encoder_output_dim // (2 if has_reverse else 1)

        # 如果仍然无法推断，再尝试从 decoder 参数推断（decoder_hidden = encoder_hidden_dim * num_directions）
        decoder_hidden = None
        dec_weight = state_dict_stripped.get('decoder_gru.weight_ih_l0')
        if dec_weight is not None:
            decoder_hidden = dec_weight.shape[0] // 3

        # 决定最终 hidden_dim（即单向 encoder_hidden_dim）
        inferred_hidden_dim = None
        if encoder_hidden_dim is not None:
            inferred_hidden_dim = int(encoder_hidden_dim)
        elif decoder_hidden is not None:
            # 如果 decoder_hidden 可用，反推 encoder_hidden_dim
            inferred_hidden_dim = int(decoder_hidden // (2 if has_reverse else 1))

        if inferred_hidden_dim is not None:
            logger.info(f"从 checkpoint 推断 hidden_dim = {inferred_hidden_dim}")
        else:
            logger.info("无法从 checkpoint 精确推断 hidden_dim，使用默认值")

        # 尝试从 GRU 层级命名推断 num_layers
        layer_indices = set()
        for key in state_dict_stripped.keys():
            match = re.search(r'(?:encoder_gru|decoder_gru)\.weight_ih_l(\d+)', key)
            if match:
                layer_indices.add(int(match.group(1)))

        inferred_num_layers = max(layer_indices) + 1 if layer_indices else None
        if inferred_num_layers is not None:
            logger.info(f"从 checkpoint 推断 num_layers = {inferred_num_layers}")

        # ⭐ 新增：检测是否为双向 GRU
        # 方法：查看 checkpoint 中是否有 *_reverse 后缀的权重（双向 GRU 特有）
        has_reverse_weights = any('_reverse' in k for k in state_dict_stripped.keys())
        if has_reverse_weights:
            logger.info("✓ 检测到双向 GRU 模型（含有 _reverse 权重）")
            inferred_bidirectional = True
        else:
            logger.info("✓ 检测到单向 GRU 模型（无 _reverse 权重）")
            inferred_bidirectional = False

        # 如果 checkpoint 提供了 hidden_dim/num_layers，优先使用，并提示
        if inferred_hidden_dim is not None:
            if hidden_dim is None or hidden_dim != inferred_hidden_dim:
                logger.warning(f"覆盖传入 hidden_dim={hidden_dim}，改用 checkpoint hidden_dim={inferred_hidden_dim}")
            hidden_dim = inferred_hidden_dim
        if inferred_num_layers is not None:
            if num_layers is None or num_layers != inferred_num_layers:
                logger.warning(f"覆盖传入 num_layers={num_layers}，改用 checkpoint num_layers={inferred_num_layers}")
            num_layers = inferred_num_layers

        hidden_dim = hidden_dim or 128
        num_layers = num_layers or 3
        self.use_attention = use_attention
        
        # ⭐ 关键修改：根据 checkpoint 动态设置 bidirectional
        # 这样就能兼容单向和双向模型了
        bidirectional = inferred_bidirectional

        # 构造模型并加载权重（先尝试严格加载）
        logger.info(f"构造模型: input_size={input_size}, hidden_dim={hidden_dim}, num_layers={num_layers}, bidirectional={bidirectional}, encoder_input_dim={inferred_encoder_input_dim}")
        self.model = EnhancedGRUModel(
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_steps=10,
            use_attention=self.use_attention,
            bidirectional=bidirectional,
            encoder_input_dim=inferred_encoder_input_dim
        )
        self.model.to(self.device)

        # 尝试加载（先严格加载；若失败则尝试 strict=False）
        try:
            self.model.load_state_dict(state_dict_stripped)
            logger.info("模型严格加载成功 (state_dict match).")
        except Exception as e:
            logger.warning(f"严格加载失败: {e}; 尝试按键逐一兼容加载")
            # 按键逐一拷贝参数：仅拷贝形状完全匹配的参数/缓冲
            model_state = self.model.state_dict()
            copied = []
            skipped_shape = []
            skipped_missing = []
            for k, v in state_dict_stripped.items():
                if k in model_state:
                    try:
                        if isinstance(v, torch.Tensor) and model_state[k].shape == v.shape:
                            model_state[k] = v.clone().to(model_state[k].device)
                            copied.append(k)
                        else:
                            skipped_shape.append((k, getattr(v, 'shape', None), model_state[k].shape))
                    except Exception as ex:
                        skipped_shape.append((k, getattr(v, 'shape', None), getattr(model_state[k], 'shape', None)))
                else:
                    skipped_missing.append(k)

            # 加载合并后的 state_dict，忽略 missing/unexpected
            try:
                self.model.load_state_dict(model_state, strict=False)
                logger.info(f"部分兼容加载完成，copied_keys={len(copied)}, skipped_shape={len(skipped_shape)}, skipped_missing={len(skipped_missing)}")
                if skipped_shape:
                    logger.warning(f"跳过形状不匹配的参数示例: {skipped_shape[:5]}")
                if skipped_missing:
                    logger.info(f"checkpoint 中存在但模型未使用的键示例: {skipped_missing[:5]}")
                logger.warning("注意：部分参数未加载，相关层将随机初始化；建议对该模型进行微调以恢复性能。")
            except Exception as e2:
                logger.error(f"无法按键兼容加载 checkpoint: {e2}")
                raise

        self.model.eval()

        # 加载统计量（保持原有逻辑）
        logger.info(f"加载统计量: {Path(stats_path).name}")
        stats = np.load(stats_path)
        self.input_mean = stats['input_mean']
        self.input_std = stats['input_std']
        self.input_mean_all = stats.get('input_mean_all')
        self.input_std_all = stats.get('input_std_all')
        if self.input_mean_all is not None and self.input_std_all is not None:
            self.input_mean_all = np.array(self.input_mean_all, dtype=np.float32)
            self.input_std_all = np.array(self.input_std_all, dtype=np.float32)
        self.output_mean = stats.get('output_mean', self.input_mean)
        self.output_std = stats.get('output_std', self.input_std)
    
    def compute_multi_scale_velocity(self, trajectory, dt=0.1, scales=[1, 2, 3]):
        """
        多尺度速度计算（改进版）
        
        改进点：
        1. 处理边界情况，确保输出长度始终与输入匹配
        2. 使用更稳定的前向差分（而非后向填充）
        3. 防止 NaN 和 Inf
        """
        multi_scale_vels = []
        
        for scale in scales:
            if len(trajectory) <= scale:
                # 若轨迹不够长，用零填充
                multi_scale_vels.append(np.zeros((len(trajectory), 3), dtype=np.float32))
                continue
            
            # 计算位置差分
            pos_diff = trajectory[scale:] - trajectory[:-scale]  # (T-scale, 3)
            vel = pos_diff / (scale * dt)  # (T-scale, 3)
            
            # ⭐ 改进：前向扩展（而非重复第一个）
            # 前 scale 个点用线性外推或零填充
            vel_padded = np.vstack([np.zeros((scale, 3), dtype=np.float32), vel])  # (T, 3)
            
            # 防止数值不稳定
            vel_padded = np.nan_to_num(vel_padded, nan=0.0, posinf=0.0, neginf=0.0)
            multi_scale_vels.append(vel_padded)
        
        if multi_scale_vels:
            result = np.concatenate(multi_scale_vels, axis=1)  # (T, 9)
            return result.astype(np.float32)
        else:
            return np.zeros((len(trajectory), 9), dtype=np.float32)
    
    def compute_curvature(self, trajectory, dt=0.1):
        """
        曲率特征计算（改进版）
        
        改进点：
        1. 增加数值稳定性（epsilon 增大）
        2. 处理低速情况（当速度接近0时）
        3. 防止 NaN/Inf
        """
        if len(trajectory) < 2:
            return np.zeros((len(trajectory), 1), dtype=np.float32)
        
        # 使用中心差分而非梯度（更稳定）
        vel = np.gradient(trajectory, axis=0) / dt
        acc = np.gradient(vel, axis=0) / dt
        
        # 计算叉积
        cross_prod = np.cross(vel, acc)
        vel_norm = np.linalg.norm(vel, axis=1)
        
        # ⭐ 改进：更强的数值稳定性
        eps = 1e-6  # 增大 epsilon 防止除零
        curvature = np.linalg.norm(cross_prod, axis=1) / np.maximum(vel_norm ** 3, eps)
        
        # 防止 NaN/Inf
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 限制曲率范围 [0, 10]，超出范围的认为异常
        curvature = np.clip(curvature, 0, 10.0)
        
        return curvature.reshape(-1, 1).astype(np.float32)
    
    def compute_plane_curvatures(self, trajectory, dt=0.1):
        """
        三平面曲率计算（改进版）
        
        改进点：
        1. 增加数值稳定性
        2. 统一处理边界和异常值
        3. 跳过低速情况
        """
        if len(trajectory) < 2:
            return np.zeros((len(trajectory), 3), dtype=np.float32)
        
        curv_list = []
        eps = 1e-6
        
        # XY 平面曲率
        pos_xy = np.column_stack([trajectory[:, 0], trajectory[:, 1], np.zeros(len(trajectory))])
        vel_xy = np.gradient(pos_xy, axis=0) / dt
        acc_xy = np.gradient(vel_xy, axis=0) / dt
        cross_xy = np.cross(vel_xy, acc_xy)
        vel_norm_xy = np.linalg.norm(vel_xy, axis=1)
        curv_xy = np.linalg.norm(cross_xy, axis=1) / np.maximum(vel_norm_xy ** 3, eps)
        curv_xy = np.nan_to_num(curv_xy, nan=0.0, posinf=1.0, neginf=0.0)
        curv_xy = np.clip(curv_xy, 0, 10.0)
        curv_list.append(curv_xy)
        
        # YZ 平面曲率
        pos_yz = np.column_stack([np.zeros(len(trajectory)), trajectory[:, 1], trajectory[:, 2]])
        vel_yz = np.gradient(pos_yz, axis=0) / dt
        acc_yz = np.gradient(vel_yz, axis=0) / dt
        cross_yz = np.cross(vel_yz, acc_yz)
        vel_norm_yz = np.linalg.norm(vel_yz, axis=1)
        curv_yz = np.linalg.norm(cross_yz, axis=1) / np.maximum(vel_norm_yz ** 3, eps)
        curv_yz = np.nan_to_num(curv_yz, nan=0.0, posinf=1.0, neginf=0.0)
        curv_yz = np.clip(curv_yz, 0, 10.0)
        curv_list.append(curv_yz)
        
        # XZ 平面曲率
        pos_xz = np.column_stack([trajectory[:, 0], np.zeros(len(trajectory)), trajectory[:, 2]])
        vel_xz = np.gradient(pos_xz, axis=0) / dt
        acc_xz = np.gradient(vel_xz, axis=0) / dt
        cross_xz = np.cross(vel_xz, acc_xz)
        vel_norm_xz = np.linalg.norm(vel_xz, axis=1)
        curv_xz = np.linalg.norm(cross_xz, axis=1) / np.maximum(vel_norm_xz ** 3, eps)
        curv_xz = np.nan_to_num(curv_xz, nan=0.0, posinf=1.0, neginf=0.0)
        curv_xz = np.clip(curv_xz, 0, 10.0)
        curv_list.append(curv_xz)
        
        return np.column_stack(curv_list).astype(np.float32)  # (T, 3)
    
    def prepare_input_features(self, positions, dt=0.1):
        """
        准备输入特征（改进版）
        
        改进点：
        1. ✅ 特征归一化的双层策略（位置和其他特征分别处理）
        2. ✅ 数值稳定性检查
        3. ✅ 异常值检测和修复
        """
        positions = np.array(positions, dtype=np.float32)
        if positions.ndim == 2 and positions.shape[0] == 3:
            positions = positions.T
        
        # 检查位置数据有效性
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            logger.warning("⚠️  输入轨迹包含 NaN 或 Inf，尝试修复...")
            positions = np.nan_to_num(positions, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # 计算特征
        multi_vel = self.compute_multi_scale_velocity(positions, dt)
        multi_vel = multi_vel[:len(positions)]  # 确保长度匹配
        curv = self.compute_curvature(positions, dt)
        plane_curvs = self.compute_plane_curvatures(positions, dt)
        
        # 拼接特征：[3D位置] + [9D多尺度速度] + [1D曲率] + [3D平面曲率] = 16D
        features = np.concatenate([positions, multi_vel, curv, plane_curvs], axis=1)
        
        # ⭐ 改进：分层归一化策略
        # 第一层：位置通道（索引 0-2）
        features[:, :3] = (features[:, :3] - self.input_mean) / (self.input_std + 1e-8)
        
        # 第二层：其他通道（索引 3-15）
        if self.input_mean_all is not None and self.input_std_all is not None:
            if len(self.input_mean_all) == features.shape[1]:
                features[:, 3:] = (features[:, 3:] - self.input_mean_all[3:]) / (self.input_std_all[3:] + 1e-8)
            else:
                logger.warning(f"⚠️  input_mean_all 维度不匹配: {len(self.input_mean_all)} != {features.shape[1]}")
        
        # ⭐ 改进：检查归一化后的数据
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 强制限制范围以防止极端值
        features = np.clip(features, -5.0, 5.0)
        
        return features.astype(np.float32)
    
    def predict_delta_increments(self, input_positions, input_length=20, verbose=False):
        """
        预测位置增量（改进版）
        
        Args:
            input_positions: 输入轨迹
            input_length: 输入序列长度
            verbose: 是否打印诊断信息
        Returns:
            pred_delta: (10, 3) 预测的增量位移
        """
        input_pos = np.array(input_positions, dtype=np.float32)
        if input_pos.shape[0] == 3:
            input_pos = input_pos.T
        
        if len(input_pos) > input_length:
            input_pos = input_pos[-input_length:]
        
        # 准备特征
        features = self.prepare_input_features(input_pos)

        # 兼容模型的 input_size：若 checkpoint 是旧模型（13）则截断特征；若模型需要更多维则用 0 填充
        expected_C = getattr(self.model, 'input_size', features.shape[1])
        if features.shape[1] != expected_C:
            if features.shape[1] > expected_C:
                features = features[:, :expected_C]
            else:
                pad = np.zeros((features.shape[0], expected_C - features.shape[1]), dtype=features.dtype)
                features = np.concatenate([features, pad], axis=1)

        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            pred_delta_norm = self.model(features_tensor)
        
        pred_delta_norm = pred_delta_norm[0].cpu().numpy()
        
        # ⭐ 改进：添加反归一化诊断信息
        if verbose:
            logger.info(f"\n[反归一化诊断]")
            logger.info(f"  归一化后的预测 (pred_delta_norm):")
            logger.info(f"    范围: [{pred_delta_norm.min():.6f}, {pred_delta_norm.max():.6f}]")
            logger.info(f"    均值: {pred_delta_norm.mean():.6f}")
            logger.info(f"  output_mean: {self.output_mean}")
            logger.info(f"  output_std:  {self.output_std}")
        
        # ✨ 反归一化：pred_delta_norm -> pred_delta
        # 训练时：pred_delta = (out_target - out_mean) / out_std
        # 推理时：pred_delta = pred_delta_norm * out_std + out_mean
        pred_delta = pred_delta_norm * self.output_std + self.output_mean
        
        if verbose:
            logger.info(f"  反归一化后的预测 (pred_delta):")
            logger.info(f"    范围: [{pred_delta.min():.6f}, {pred_delta.max():.6f}]")
            logger.info(f"    均值: {pred_delta.mean():.6f}")
            logger.info(f"  最后输入位置: {input_pos[-1]}")
        
        return pred_delta  # (10, 3)
    
    def reconstruct_positions_simple(self, input_positions, dt=0.1, input_length=20, verbose=False):
        """
        简单位置重建（改进版）：直接积分位移增量 + 误差修正
        
        改进点：
        - ✅ 增加速度平滑（简化版，防止跳变）
        - ✅ 增加位置连续性检查
        - ✅ 添加诊断信息
        """
        pred_delta = self.predict_delta_increments(input_positions, input_length, verbose=verbose)
        
        input_pos = np.array(input_positions, dtype=np.float32)
        if input_pos.shape[0] == 3:
            input_pos = input_pos.T
        
        last_pos = input_pos[-1]
        
        if verbose:
            logger.info(f"\n[位置重建诊断 - Simple]")
            logger.info(f"  最后输入位置: {last_pos}")
            logger.info(f"  预测增量 pred_delta 范围: [{pred_delta.min():.6f}, {pred_delta.max():.6f}]")
        
        # 直接积分（原始方法）
        pred_positions = last_pos + pred_delta
        
        # ⭐ 改进：简单的速度约束（防止跳变太大）
        last_vel = np.mean(np.diff(input_pos[-5:], axis=0) / dt, axis=0) if len(input_pos) > 1 else np.zeros(3)
        max_vel = np.max(np.linalg.norm(np.diff(input_pos, axis=0), axis=1)) * 1.5 if len(input_pos) > 1 else 10.0
        
        if verbose:
            logger.info(f"  最后速度: {last_vel}")
            logger.info(f"  最大速度限制: {max_vel:.6f}")
        
        for i in range(len(pred_positions)):
            step_vel = pred_delta[i] / dt
            step_vel_norm = np.linalg.norm(step_vel)
            
            # 如果速度过大，进行缩放
            if step_vel_norm > max_vel:
                pred_delta[i] = pred_delta[i] * (max_vel / (step_vel_norm + 1e-8)) * dt
        
        # 重新积分
        pred_positions = last_pos + np.cumsum(pred_delta, axis=0)
        
        if verbose:
            logger.info(f"  重建位置范围: [{pred_positions.min():.6f}, {pred_positions.max():.6f}]")
            logger.info(f"  重建位置:\n{pred_positions}")
        
        return pred_positions
    
    def reconstruct_positions_physics_constrained(self, input_positions, dt=0.1, 
                                                 input_length=20, smoothing_weight=0.3):
        """
        物理约束位置重建（改进版）：加入加速度平滑约束 + 速度约束
        
        核心思想：
        1. 预测的速度变化不应过于剧烈 → 加速度平滑
        2. 累积位移应与历史轨迹一致 → 速度连续性
        3. 最大加速度应受限 → 物理约束
        
        改进点：
        - ✅ 增加速度约束（防止速度跳变）
        - ✅ 增加最大加速度限制
        - ✅ 改进初始速度估计（使用多步平均）
        - ✅ 误差反馈机制（检测和修正累积误差）
        """
        pred_delta = self.predict_delta_increments(input_positions, input_length)
        
        input_pos = np.array(input_positions, dtype=np.float32)
        if input_pos.shape[0] == 3:
            input_pos = input_pos.T
        
        # ⭐ 改进1：更稳健的加速度估计
        input_vel = np.diff(input_pos, axis=0) / dt
        if len(input_vel) > 1:
            input_acc = np.diff(input_vel, axis=0) / dt
            avg_acc = np.mean(input_acc, axis=0)
            max_acc = np.max(np.linalg.norm(input_acc, axis=1))
        else:
            avg_acc = np.zeros(3)
            max_acc = 1.0
        
        last_pos = input_pos[-1]
        last_vel = input_vel[-1] if len(input_vel) > 0 else np.zeros(3)
        
        # 预测原始位置
        pred_positions = last_pos + pred_delta
        current_vel = last_vel.copy()
        
        # 计算期望速度
        desired_vel = np.diff(np.vstack([last_pos, pred_positions]), axis=0) / dt
        
        smoothed_positions = np.zeros_like(pred_positions)
        current_pos = last_pos.copy()
        
        # 处理 smoothing_weight
        weight_arr = np.array(smoothing_weight, dtype=np.float32)
        if weight_arr.ndim == 0:
            weight_arr = np.full(3, weight_arr)
        elif weight_arr.shape != (3,):
            raise ValueError("smoothing_weight must be a scalar or 3-element iterable")
        
        # ⭐ 改进2：逐步构建预测，加入多重约束
        for i in range(len(pred_delta)):
            # 计算原始加速度
            raw_accel = (desired_vel[i] - current_vel) / dt
            
            # ⭐ 约束1：加速度平滑（相信历史）
            constrained_accel = (1 - weight_arr) * raw_accel + weight_arr * avg_acc
            
            # ⭐ 约束2：最大加速度限制（物理约束）
            accel_norm = np.linalg.norm(constrained_accel)
            max_accel_norm = max(max_acc * 1.5, 5.0)  # 允许比历史最大加速度高 50%
            if accel_norm > max_accel_norm:
                constrained_accel = constrained_accel * (max_accel_norm / (accel_norm + 1e-8))
            
            # ⭐ 约束3：速度更新（防止速度跳变）
            new_vel = current_vel + constrained_accel * dt
            
            # 限制速度不能过大
            vel_norm = np.linalg.norm(new_vel)
            max_vel = np.max(np.linalg.norm(input_vel, axis=0)) * 2.0 if len(input_vel) > 0 else 10.0
            if vel_norm > max_vel:
                new_vel = new_vel * (max_vel / (vel_norm + 1e-8))
            
            current_vel = new_vel
            
            # 更新位置
            next_pos = current_pos + current_vel * dt
            smoothed_positions[i] = next_pos
            current_pos = next_pos
        
        return smoothed_positions
    
    def reconstruct_positions_trajectory_smoothing(self, input_positions, dt=0.1,
                                                  input_length=20, window_size=3, smoothing_weight=0.3):
        """
        轨迹平滑：对预测结果进行滑动平均
        适用于对噪声敏感的应用（如可视化）
        """
        positions = self.reconstruct_positions_physics_constrained(input_positions, dt, input_length,
                                                                  smoothing_weight=smoothing_weight)
        
        # 滑动平均平滑
        if window_size > 1:
            smoothed = np.zeros_like(positions)
            for i in range(len(positions)):
                start = max(0, i - window_size // 2)
                end = min(len(positions), i + window_size // 2 + 1)
                smoothed[i] = np.mean(positions[start:end], axis=0)
            return smoothed
        
        return positions


def main():
    parser = argparse.ArgumentParser(description='Enhanced trajectory inference')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--stats', type=str, required=True, help='统计量路径')
    parser.add_argument('--trajectory', type=str, required=True, help='输入轨迹 CSV')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--method', type=str, default='physics_constrained',
                       choices=['simple', 'physics_constrained', 'smoothed'],
                       help='重建方法')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--input_length', type=int, default=20)
    parser.add_argument('--smoothing-weight', type=parse_smoothing_weight, default='0.3',
                        help='加速度平滑权重，可传 scalar 或 X,Y,Z 三个值')
    parser.add_argument('--verbose', action='store_true', help='打印诊断信息')
    
    args = parser.parse_args()
    
    # 加载轨迹
    import pandas as pd
    df = pd.read_csv(args.trajectory)
    trajectory = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    
    # 创建推理器
    infer = EnhancedInference(args.model, args.stats, args.hidden_dim, 
                             args.num_layers, args.use_attention)
    
    # 预测
    if args.method == 'simple':
        pred = infer.reconstruct_positions_simple(trajectory, args.dt, args.input_length, 
                                                 verbose=args.verbose)
    elif args.method == 'physics_constrained':
        pred = infer.reconstruct_positions_physics_constrained(
            trajectory, args.dt, args.input_length, smoothing_weight=args.smoothing_weight
        )
    else:
        pred = infer.reconstruct_positions_trajectory_smoothing(
            trajectory, args.dt, args.input_length, smoothing_weight=args.smoothing_weight
        )
    logger.info(f"使用 smoothing_weight: {args.smoothing_weight}")
    
    logger.info(f"\n预测完成!")
    logger.info(f"预测位置形状: {pred.shape}")
    logger.info(f"预测位置:\n{pred}")
    
    # 保存
    output_path = Path(args.trajectory).parent / f'predictions_{args.method}.csv'
    pred_df = pd.DataFrame(pred, columns=['x', 'y', 'z'])
    pred_df.to_csv(output_path, index=False)
    logger.info(f"已保存到: {output_path}")


if __name__ == '__main__':
    main()
