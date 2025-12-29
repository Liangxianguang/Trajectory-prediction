#!/usr/bin/env python3
"""
统一评估脚本：对比多个模型在所有测试集上的性能
支持：
1. 批量加载多个模型（单向/双向 GRU，带/不带 attention）
2. 遍历测试集目录，测试每条轨迹
3. 计算 MAE、MSE、RMSE、MAPE（逐点、逐步、全局）
4. 分轴统计（X/Y/Z）
5. 生成对比报告（CSV + 可视化）
python evaluate_all_models.py --auto_models --tool_dir "..\..\drone_trajectories\tool" --test_dir "..\..\Synthetic-UAV-Flight-Trajectories,..\..\drone_trajectories\random_traj_100ms,..\..\drone_trajectories\new_random_traj_100ms" 
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import re
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from drone_trajectories.tool.train_model_enhanced import EnhancedGRUModel
from drone_trajectories.tool.infer_enhanced import EnhancedInference
try:
    # 用于可视化（可选），如果不存在则继续不阻塞评估
    from drone_trajectories.tool.visualize_prediction import plot_prediction
    PLOT_AVAILABLE = True
except Exception:
    plot_prediction = None
    PLOT_AVAILABLE = False

import glob


def discover_models_in_tool(tool_dir: Path) -> List[Dict]:
    """扫描 tool 目录，查找模型 checkpoint (.pth) 与对应的 norm_stats (.npz)，返回模型配置列表。"""
    models = []
    tool_path = Path(tool_dir)
    if not tool_path.exists():
        logger.warning(f"指定的 tool 目录不存在: {tool_dir}")
        return models

    # 搜索所有 .pth 文件（优先匹配 *_best_model*.pth）
    pths = list(tool_path.rglob('*best_model*.pth'))
    if not pths:
        # 回退到搜索所有 pth
        pths = list(tool_path.rglob('*.pth'))

    for p in pths:
        # 在同一目录查找 *_norm_stats.npz，优先与 p.stem 中的前缀匹配
        stats_candidates = list(p.parent.glob('*_norm_stats.npz'))
        stats_path = None
        if stats_candidates:
            # 若存在多个，尝试匹配前缀
            for s in stats_candidates:
                if s.stem.startswith(p.stem.split('_best_model')[0]):
                    stats_path = s
                    break
            if stats_path is None:
                stats_path = stats_candidates[0]

        model_entry = {
            'name': p.parent.name + '_' + p.stem,
            'model_path': str(p),
            'stats_path': str(stats_path) if stats_path is not None else '',
        }
        models.append(model_entry)

    return models

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class UnifiedEvaluator:
    """统一评估器"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def load_model(self, model_path: str, stats_path: str, model_name: str,
                   hidden_dim=None, num_layers=None, use_attention=False,
                   bidirectional=False) -> Tuple[EnhancedInference, str]:
        """
        加载单个模型和统计量
        
        注意：EnhancedInference 会自动从 checkpoint 推断参数
        为了兼容性，传入的 hidden_dim/num_layers/bidirectional 会被自动推断覆盖
        """
        try:
            logger.info(f"加载模型: {model_name}")
            
            # 直接传递参数到 EnhancedInference
            # 推理代码会自动从 checkpoint 推断，传入的值如果与 checkpoint 不符会被覆盖
            infer = EnhancedInference(
                model_path, stats_path,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                use_attention=use_attention,
                device=self.device
            )
            logger.info(f"  ✓ {model_name} 加载成功")
            return infer, None
        except Exception as e:
            error_msg = f"加载模型 {model_name} 失败: {str(e)}"
            logger.error(f"  ✗ {error_msg}")
            return None, error_msg
    
    def load_trajectory(self, csv_path: str) -> Tuple[np.ndarray, str]:
        """加载单条轨迹 CSV/TXT"""
        try:
            file_path = Path(csv_path)
            
            # 根据文件扩展名选择解析方式
            if file_path.suffix.lower() == '.txt':
                # TXT 文件：可能是 CSV 格式（有列名和逗号分隔）或纯文本格式
                # 首先尝试 CSV 格式（列名 + 逗号分隔，可能有空格）
                try:
                    df = pd.read_csv(csv_path, sep=',', skipinitialspace=True)
                    logger.debug(f"TXT: 检测为 CSV 格式（逗号分隔）")
                except Exception as e1:
                    try:
                        # 回退：尝试空白分隔（无列名）
                        df = pd.read_csv(csv_path, sep=r'\s+', header=None, engine='python')
                        logger.debug(f"TXT: 检测为纯文本格式（空白分隔）")
                    except Exception as e2:
                        try:
                            # 再回退：尝试制表符分隔
                            df = pd.read_csv(csv_path, sep='\t', header=None)
                            logger.debug(f"TXT: 检测为制表符分隔")
                        except Exception as e3:
                            return None, f"TXT 解析失败（尝试了多种格式）: CSV 失败 + 空白分隔失败"
                
                # 规范化列名（去除空格、转小写）
                df.columns = [col.strip().lower() for col in df.columns]
                logger.debug(f"TXT 列名: {df.columns.tolist()}")
                
                # 尝试找坐标列
                if all(col in df.columns for col in ['tx', 'ty', 'tz']):
                    trajectory = df[['tx', 'ty', 'tz']].values.astype(np.float32)
                    logger.debug(f"TXT: 使用列 tx, ty, tz")
                elif all(col in df.columns for col in ['x', 'y', 'z']):
                    trajectory = df[['x', 'y', 'z']].values.astype(np.float32)
                    logger.debug(f"TXT: 使用列 x, y, z")
                elif df.shape[1] >= 3:
                    # 如果没有标准列名，尝试用前 3 列（可能是 timestamp 后面的 3 列）
                    # 跳过第一列（如果是 timestamp）
                    if 'timestamp' in df.columns:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) >= 3:
                            trajectory = df[numeric_cols[:3]].values.astype(np.float32)
                            logger.debug(f"TXT: 使用数值列 {numeric_cols[:3]}")
                        else:
                            return None, f"TXT 文件：无足够的数值列 (需要 >= 3，实际 {len(numeric_cols)})"
                    else:
                        # 没有 timestamp，直接用前 3 列
                        trajectory = df.iloc[:, :3].values.astype(np.float32)
                        logger.debug(f"TXT: 使用前 3 列作为坐标")
                else:
                    return None, f"TXT 文件列数不足 (需要 >= 3 列，实际 {df.shape[1]})"
            else:
                # CSV 文件：尝试使用已命名的列
                df = pd.read_csv(csv_path)
                
                # 规范化列名（去除空格、转小写）
                df.columns = [col.strip().lower() for col in df.columns]
                
                logger.debug(f"CSV 列名: {df.columns.tolist()}")
                
                # 尝试多种列名组合
                if all(col in df.columns for col in ['tx', 'ty', 'tz']):
                    trajectory = df[['tx', 'ty', 'tz']].values.astype(np.float32)
                    logger.debug(f"使用列: tx, ty, tz")
                elif all(col in df.columns for col in ['x', 'y', 'z']):
                    trajectory = df[['x', 'y', 'z']].values.astype(np.float32)
                    logger.debug(f"使用列: x, y, z")
                elif all(col in df.columns for col in ['px', 'py', 'pz']):
                    trajectory = df[['px', 'py', 'pz']].values.astype(np.float32)
                    logger.debug(f"使用列: px, py, pz")
                elif all(col in df.columns for col in ['pos_x', 'pos_y', 'pos_z']):
                    trajectory = df[['pos_x', 'pos_y', 'pos_z']].values.astype(np.float32)
                    logger.debug(f"使用列: pos_x, pos_y, pos_z")
                else:
                    # 尝试自动检测：找前 3 个数值列（排除 timestamp）
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c != 'timestamp']  # 排除 timestamp
                    logger.debug(f"检测到数值列（排除 timestamp）: {numeric_cols}")
                    
                    if len(numeric_cols) >= 3:
                        trajectory = df[numeric_cols[:3]].values.astype(np.float32)
                        logger.debug(f"自动检测使用前 3 列: {numeric_cols[:3]}")
                    else:
                        # 如果没有匹配，列出可用的列名
                        return None, f"未找到坐标列。可用数值列: {numeric_cols}，所有列: {df.columns.tolist()}"
            
            if len(trajectory) < 20:
                return None, f"轨迹长度不足 (需要 >= 20, 实际 {len(trajectory)})"
            
            return trajectory, None
        except Exception as e:
            import traceback
            return None, f"加载轨迹失败: {str(e)}\n{traceback.format_exc()}"
    
    def predict_trajectory(self, infer: EnhancedInference, trajectory: np.ndarray,
                          input_length: int = 20, method: str = 'physics_constrained') -> Tuple[np.ndarray, str]:
        """预测轨迹 - 调用 EnhancedInference 的重建方法（复用现有推理 API）"""
        try:
            if method == 'simple':
                pred = infer.reconstruct_positions_simple(trajectory, input_length=input_length)
            elif method == 'physics_constrained':
                # 有些版本的 infer 方法签名不接受 verbose，统一传入必需参数
                pred = infer.reconstruct_positions_physics_constrained(trajectory, input_length=input_length)
            else:  # smoothed
                pred = infer.reconstruct_positions_trajectory_smoothing(trajectory, input_length=input_length)

            return pred, None
        except Exception as e:
            return None, f"预测失败: {str(e)}"
    
    @staticmethod
    def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        """
        计算 MAE、MSE、RMSE、MAPE
        
        Args:
            pred: (T, 3) 预测位置
            true: (T, 3) 真实位置
        
        Returns:
            metrics_dict: 各项指标
        """
        if pred.shape != true.shape:
            return {}
        
        # 按点计算误差
        error = pred - true  # (T, 3)
        abs_error = np.abs(error)
        sqr_error = error ** 2
        
        # 全局指标
        mae_all = np.mean(abs_error)
        mse_all = np.mean(sqr_error)
        rmse_all = np.sqrt(mse_all)
        
        # MAPE（平均百分比误差）- 避免除零
        eps = 1e-8
        true_norm = np.linalg.norm(true, axis=1)
        pred_norm = np.linalg.norm(pred, axis=1)
        error_norm = np.linalg.norm(error, axis=1)
        
        mape_all = np.mean(error_norm / (np.linalg.norm(true, axis=1) + eps)) * 100  # %
        
        # 分轴指标
        metrics = {
            'MAE_all': float(mae_all),
            'MSE_all': float(mse_all),
            'RMSE_all': float(rmse_all),
            'MAPE_all': float(mape_all),
            'MAE_x': float(np.mean(abs_error[:, 0])),
            'MAE_y': float(np.mean(abs_error[:, 1])),
            'MAE_z': float(np.mean(abs_error[:, 2])),
            'MSE_x': float(np.mean(sqr_error[:, 0])),
            'MSE_y': float(np.mean(sqr_error[:, 1])),
            'MSE_z': float(np.mean(sqr_error[:, 2])),
            'RMSE_x': float(np.sqrt(np.mean(sqr_error[:, 0]))),
            'RMSE_y': float(np.sqrt(np.mean(sqr_error[:, 1]))),
            'RMSE_z': float(np.sqrt(np.mean(sqr_error[:, 2]))),
            'Max_error': float(np.max(error_norm)),
            'Min_error': float(np.min(error_norm)),
            'Std_error': float(np.std(error_norm)),
        }
        
        return metrics
    
    def evaluate_model_on_dataset(self, infer: EnhancedInference, model_name: str,
                                 test_dir: str, input_length: int = 20,
                                 method: str = 'physics_constrained',
                                 max_samples: int = None,
                                 visualize: bool = False,
                                 visual_samples: int = 0,
                                 visual_output_dir: str = None) -> Dict:
        """
        在整个测试集上评估单个模型
        
        Args:
            infer: 推理器
            model_name: 模型名称
            test_dir: 测试集目录
            input_length: 输入序列长度
            method: 重建方法
            max_samples: 最多评估样本数（None 表示全部）
        
        Returns:
            summary: 汇总统计
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"评估模型: {model_name}")
        logger.info(f"测试目录: {test_dir}")
        logger.info(f"{'='*80}")
        
        test_path = Path(test_dir)
        if not test_path.exists():
            logger.error(f"测试目录不存在: {test_dir}")
            return None
        
        # 查找所有 CSV 和 TXT 文件
        csv_files = list(test_path.glob('*.csv'))
        txt_files = list(test_path.glob('*.txt'))
        traj_files = csv_files + txt_files
        
        if not traj_files:
            logger.warning(f"测试目录中未找到 CSV 或 TXT 文件: {test_dir}")
            return None
        
        if max_samples:
            traj_files = traj_files[:max_samples]
        
        logger.info(f"找到 {len(traj_files)} 个测试文件（CSV: {len(csv_files)}, TXT: {len(txt_files)}）")
        
        all_metrics = []
        errors = []
        
        # 调试：统计各类错误
        load_errors = {}
        pred_errors = {}
        visuals_done = 0
        if visualize and not PLOT_AVAILABLE:
            logger.warning("请求可视化，但 plot_prediction 不可用（无法导入 visualize_prediction），将跳过可视化。")
        
        for i, traj_file in enumerate(traj_files, 1):
            traj_name = traj_file.stem
            
            # 加载真实轨迹
            trajectory, load_err = self.load_trajectory(str(traj_file))
            if load_err:
                logger.debug(f"  [{i}/{len(traj_files)}] {traj_name}: {load_err}")
                # 统计加载错误类型
                if load_err not in load_errors:
                    load_errors[load_err] = 0
                load_errors[load_err] += 1
                errors.append((traj_name, load_err))
                continue
            
            logger.debug(f"  [{i}/{len(traj_files)}] {traj_name}: 成功加载，形状 {trajectory.shape}")
            
            # 截断/补齐输入部分（前 input_length 点 + 后续待预测部分）
            if len(trajectory) <= input_length:
                logger.debug(f"  [{i}/{len(traj_files)}] {traj_name}: 轨迹过短 (len={len(trajectory)} <= input_length={input_length})，跳过")
                continue
            
            # 分割：前 input_length 作为输入，后续作为预测目标
            input_traj = trajectory[:input_length]
            true_future = trajectory[input_length:]
            
            # 限制预测长度（通常预测 10 步）
            max_pred_steps = min(10, len(true_future))
            true_future = true_future[:max_pred_steps]
            
            # 预测
            pred, pred_err = self.predict_trajectory(infer, input_traj, input_length, method)
            if pred_err:
                logger.debug(f"  [{i}/{len(traj_files)}] {traj_name}: {pred_err}")
                # 统计预测错误类型
                if pred_err not in pred_errors:
                    pred_errors[pred_err] = 0
                pred_errors[pred_err] += 1
                errors.append((traj_name, pred_err))
                continue
            
            # 截断预测到与真实相同长度
            pred = pred[:len(true_future)]
            
            # 计算指标
            metrics = self.compute_metrics(pred, true_future)
            if not metrics:
                logger.debug(f"  [{i}/{len(csv_files)}] {traj_name}: 指标计算失败")
                continue

            # 可选可视化（只对前 visual_samples 个样本）
            if visualize and visuals_done < visual_samples and PLOT_AVAILABLE:
                try:
                    out_dir = Path(visual_output_dir) if visual_output_dir is not None else Path.cwd()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    png_path, html_path = plot_prediction(input_traj, true_future, pred, traj_name, out_dir, interactive=False)
                    logger.info(f"  [{i}/{len(traj_files)}] 可视化已保存: {png_path} {html_path if html_path else ''}")
                    visuals_done += 1
                except Exception as e:
                    logger.warning(f"可视化失败: {e}")
            
            metrics['trajectory'] = traj_name
            metrics['input_length'] = len(input_traj)
            metrics['pred_length'] = len(pred)
            all_metrics.append(metrics)
            
            if i % 100 == 0:
                logger.info(f"  已处理 {i}/{len(traj_files)} 个文件...")
        
        logger.info(f"加载错误统计: {load_errors}")
        logger.info(f"预测错误统计: {pred_errors}")
        
        # 汇总统计
        if not all_metrics:
            logger.error(f"模型 {model_name} 未能成功评估任何样本")
            return None
        
        metrics_df = pd.DataFrame(all_metrics)
        
        summary = {
            'model_name': model_name,
            'num_samples': len(all_metrics),
            'num_errors': len(errors),
            'avg_MAE': float(metrics_df['MAE_all'].mean()),
            'avg_MSE': float(metrics_df['MSE_all'].mean()),
            'avg_RMSE': float(metrics_df['RMSE_all'].mean()),
            'avg_MAPE': float(metrics_df['MAPE_all'].mean()),
            'avg_MAE_x': float(metrics_df['MAE_x'].mean()),
            'avg_MAE_y': float(metrics_df['MAE_y'].mean()),
            'avg_MAE_z': float(metrics_df['MAE_z'].mean()),
            'avg_RMSE_x': float(metrics_df['RMSE_x'].mean()),
            'avg_RMSE_y': float(metrics_df['RMSE_y'].mean()),
            'avg_RMSE_z': float(metrics_df['RMSE_z'].mean()),
            'max_error': float(metrics_df['Max_error'].mean()),
            'min_error': float(metrics_df['Min_error'].mean()),
            'std_error': float(metrics_df['Std_error'].mean()),
        }
        
        logger.info(f"\n[{model_name} 汇总]")
        logger.info(f"  有效样本: {summary['num_samples']}")
        logger.info(f"  失败样本: {summary['num_errors']}")
        logger.info(f"  平均 MAE: {summary['avg_MAE']:.6f}")
        logger.info(f"  平均 MSE: {summary['avg_MSE']:.6f}")
        logger.info(f"  平均 RMSE: {summary['avg_RMSE']:.6f}")
        logger.info(f"  平均 MAPE: {summary['avg_MAPE']:.6f}%")
        logger.info(f"  分轴 MAE  - X: {summary['avg_MAE_x']:.6f}, Y: {summary['avg_MAE_y']:.6f}, Z: {summary['avg_MAE_z']:.6f}")
        logger.info(f"  分轴 RMSE - X: {summary['avg_RMSE_x']:.6f}, Y: {summary['avg_RMSE_y']:.6f}, Z: {summary['avg_RMSE_z']:.6f}")
        
        # 保存详细结果
        detail_path = Path(test_dir).parent / f'{model_name}_detailed_results.csv'
        metrics_df.to_csv(detail_path, index=False)
        logger.info(f"  详细结果已保存: {detail_path}")
        
        return summary, metrics_df
    
    def compare_models(self, summaries: List[Dict], output_dir: str = '.'):
        """
        对比多个模型的性能
        
        Args:
            summaries: 多个模型的汇总统计
            output_dir: 输出目录
        """
        if not summaries:
            logger.error("没有有效的模型汇总数据")
            return
        
        logger.info(f"\n{'='*100}")
        logger.info("模型性能对比")
        logger.info(f"{'='*100}")
        
        # 构建对比表格
        comparison_df = pd.DataFrame(summaries)
        
        # 打印对比表
        print("\n" + "="*120)
        print("指标对比汇总表")
        print("="*120)
        
        cols_to_show = [
            'model_name', 'num_samples', 'num_errors',
            'avg_MAE', 'avg_MSE', 'avg_RMSE', 'avg_MAPE',
            'avg_MAE_x', 'avg_MAE_y', 'avg_MAE_z',
            'avg_RMSE_x', 'avg_RMSE_y', 'avg_RMSE_z',
        ]
        
        print(comparison_df[cols_to_show].to_string(index=False))
        print("="*120)
        
        # 保存对比结果
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV 版本
        csv_path = os.path.join(output_dir, 'models_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"✓ 对比结果已保存: {csv_path}")
        
        # JSON 版本
        json_path = os.path.join(output_dir, 'models_comparison.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ JSON 结果已保存: {json_path}")
        
        # 生成排名
        logger.info(f"\n{'='*100}")
        logger.info("性能排名 (按 RMSE 升序)")
        logger.info(f"{'='*100}")
        
        ranked = comparison_df.sort_values('avg_RMSE')
        for rank, (idx, row) in enumerate(ranked.iterrows(), 1):
            logger.info(f"{rank}. {row['model_name']:<30} RMSE={row['avg_RMSE']:.6f} MAE={row['avg_MAE']:.6f} MAPE={row['avg_MAPE']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='统一评估多个模型在测试集上的性能')
    parser.add_argument('--models', type=str, required=False, default=None,
                       help='模型配置 JSON 文件（包含模型路径、名称等）；使用 --auto_models 时可省略')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='测试集目录（支持多个目录，逗号分隔；如 "dir1,dir2,dir3"）')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='输出目录')
    parser.add_argument('--input_length', type=int, default=20,
                       help='输入序列长度')
    parser.add_argument('--method', type=str, default='physics_constrained',
                       choices=['simple', 'physics_constrained', 'smoothed'],
                       help='重建方法')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最多评估样本数（用于快速测试）')
    parser.add_argument('--auto_models', action='store_true', help='自动扫描 tool 目录下的模型并评估（忽略 --models 指定的文件）')
    parser.add_argument('--tool_dir', type=str, default=str(Path(__file__).resolve().parent.parent / 'tool'),
                        help='当使用 --auto_models 时，扫描的 tool 目录')
    parser.add_argument('--visualize', action='store_true', help='为部分样本生成可视化图（需要 plotly 或 matplotlib 支持）')
    parser.add_argument('--visual_samples', type=int, default=0, help='每个模型生成的可视化样本数量')
    parser.add_argument('--visual_output_dir', type=str, default=None, help='可视化输出目录（默认使用 --output_dir）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 验证互斥参数
    if not args.auto_models and not args.models:
        parser.error("必须指定 --models 或使用 --auto_models 来加载模型配置")
    
    # 处理路径：如果是相对路径，相对于当前脚本目录
    current_dir = Path(__file__).resolve().parent
    
    # 处理多个测试目录（支持逗号分隔）
    test_dirs_raw = args.test_dir.split(',')
    test_dirs = []
    for td in test_dirs_raw:
        td = td.strip()
        test_dir = Path(td)
        if not test_dir.is_absolute():
            test_dir = current_dir / test_dir
        test_dirs.append(str(test_dir))
    
    logger.info(f"测试目录: {test_dirs}")
    
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型配置（支持自动扫描 tool 目录）
    models_config = None
    if args.auto_models:
        tool_dir = Path(args.tool_dir)
        if not tool_dir.is_absolute():
            tool_dir = current_dir / tool_dir
        logger.info(f"自动扫描 models: tool_dir={tool_dir}")
        models_config = discover_models_in_tool(tool_dir)
        logger.info(f"自动发现 {len(models_config)} 个模型 checkpoint")
    else:
        models_config_path = Path(args.models)
        if not models_config_path.is_absolute():
            models_config_path = current_dir / models_config_path
        
        logger.info(f"加载模型配置: {models_config_path}")
        try:
            with open(models_config_path, 'r', encoding='utf-8') as f:
                models_config = json.load(f)
        except Exception as e:
            logger.error(f"无法加载模型配置: {e}")
            return

        if not isinstance(models_config, list):
            models_config = [models_config]

        logger.info(f"共加载 {len(models_config)} 个模型配置")
    
    # 初始化评估器
    evaluator = UnifiedEvaluator(device=device)
    
    all_summaries = []
    
    # 逐个评估每个模型
    for model_cfg in models_config:
        model_name = model_cfg.get('name', 'unknown')
        model_path_raw = model_cfg.get('model_path')
        stats_path_raw = model_cfg.get('stats_path')
        
        # 处理模型路径：相对于当前脚本所在目录
        model_path = Path(model_path_raw)
        if not model_path.is_absolute():
            model_path = current_dir / model_path
        
        stats_path = Path(stats_path_raw)
        if not stats_path.is_absolute():
            stats_path = current_dir / stats_path
        
        # 如果自动发现但未找到对应 stats 文件，跳过该 checkpoint
        if not stats_path_raw:
            logger.warning(f"模型 {model_name} 未发现对应的 stats 文件，已跳过: model_path={model_path}")
            continue
        
        # ⭐ 关键修改：不传入 hidden_dim/num_layers
        # 让 EnhancedInference 从 checkpoint 自动推断（准确性更高）
        # 配置文件中的 hidden_dim/num_layers 只作参考，不实际使用
        use_attention = model_cfg.get('use_attention', False)  # 从 checkpoint 检测
        
        # 加载模型（hidden_dim/num_layers 设为 None，让推理代码自动推断）
        infer, load_err = evaluator.load_model(
            str(model_path), str(stats_path), model_name,
            hidden_dim=None,              # 自动推断
            num_layers=None,              # 自动推断
            use_attention=use_attention,
            bidirectional=False           # 也会从 checkpoint 自动推断
        )
        
        if load_err:
            logger.error(f"跳过模型 {model_name}")
            continue
        
        # 对每个测试目录评估模型
        for test_dir in test_dirs:
            logger.info(f"\n{'='*80}")
            logger.info(f"评估模型 '{model_name}' 在目录 '{test_dir}' 上")
            logger.info(f"{'='*80}")
            
            # 为这个测试集生成独特的模型名称（带目录标识）
            test_dir_name = Path(test_dir).name
            model_name_with_dataset = f"{model_name}_{test_dir_name}"
            
            # 评估模型
            result = evaluator.evaluate_model_on_dataset(
                infer, model_name_with_dataset, str(test_dir),
                input_length=args.input_length,
                method=args.method,
                max_samples=args.max_samples,
                visualize=args.visualize,
                visual_samples=args.visual_samples,
                visual_output_dir=(args.visual_output_dir or args.output_dir)
            )
            
            if result:
                summary, metrics_df = result
                all_summaries.append(summary)
    
    # 对比所有模型
    if all_summaries:
        evaluator.compare_models(all_summaries, args.output_dir)
        logger.info(f"\n✓ 评估完成！结果已保存到: {args.output_dir}")
    else:
        logger.error("没有成功评估任何模型")


if __name__ == '__main__':
    main()

