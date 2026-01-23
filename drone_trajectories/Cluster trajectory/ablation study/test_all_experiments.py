#!/usr/bin/env python3
"""
消融实验测试脚本
让每个实验训练1个epoch，验证是否能正常运行并保存模型
确保保存格式与v版本一致
"""

import subprocess
import sys
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 实验配置
EXPERIMENTS = [
    {
        'name': '实验1：基线模型',
        'script': 'train_ablation_exp1_baseline.py',
        'args': [
            '--agents', '3',
            '--epochs', '1',
            '--batch_size', '512',  # 小批次加快测试
            '--use_subset',  # 使用子集数据
            '--features_dir', '../swarm_features',  # 16D特征
        ],
        'expected_files': [
            'best_model_agents_3_exp1_baseline.pt',
            'config_agents_3_exp1_baseline.json',
            'training_history_agents_3_exp1_baseline.csv',
        ]
    },
    {
        'name': '实验2：特征增强+BiGRU+CA',
        'script': 'train_ablation_exp2_feat_bigru.py',
        'args': [
            '--agents', '3',
            '--epochs', '1',
            '--batch_size', '512',
            '--use_subset',
            '--features_dir', '../features_32d',  # 32D特征
        ],
        'expected_files': [
            'best_model_agents_3_exp2_feat_bigru.pt',
            'config_agents_3_exp2_feat_bigru.json',
            'training_history_agents_3_exp2_feat_bigru.csv',
        ]
    },
    {
        'name': '实验3：GAT+BiGRU+CA',
        'script': 'train_ablation_exp3_gnn_bigru.py',
        'args': [
            '--agents', '3',
            '--epochs', '1',
            '--batch_size', '128',
            '--use_subset',
            '--features_dir', '../swarm_features',  # 16D特征
        ],
        'expected_files': [
            'best_model_agents_3_exp3_gnn_bigru.pt',
            'config_agents_3_exp3_gnn_bigru.json',
            'training_history_agents_3_exp3_gnn_bigru.csv',
        ]
    },
    {
        'name': '实验4：GAT+特征增强（无BiGRU+CA）',
        'script': 'train_ablation_exp4_gnn_feat.py',
        'args': [
            '--agents', '3',
            '--epochs', '1',
            '--batch_size', '128',
            '--use_subset',
            '--features_dir', '../features_32d',  # 32D特征
        ],
        'expected_files': [
            'best_model_agents_3_exp4_gnn_feat.pt',
            'config_agents_3_exp4_gnn_feat.json',
            'training_history_agents_3_exp4_gnn_feat.csv',
        ]
    },
    {
        'name': '实验5：完整模型',
        'script': 'train_ablation_exp5_full.py',
        'args': [
            '--agents', '3',
            '--epochs', '1',
            '--batch_size', '128',
            '--use_subset',
            '--features_dir', '../features_32d',  # 32D特征
        ],
        'expected_files': [
            'best_model_agents_3_exp5_full.pt',
            'config_agents_3_exp5_full.json',
            'training_history_agents_3_exp5_full.csv',
        ]
    },
]

# 目前只先测试实验3-5（跳过实验1/2）
EXPERIMENTS = EXPERIMENTS[0:]


def check_model_save_format(checkpoint_path, experiment_name):
    """
    检查模型保存格式是否与v版本一致
    """
    import torch
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        required_keys = [
            'epoch',
            'model_state_dict',
            'best_val_loss',
            'config',
            'input_mean',
            'input_std',
            'output_mean',
            'output_std',
            'feature_mean',
            'feature_std',
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in ckpt:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"  ❌ {experiment_name}: 缺少必要的键: {missing_keys}")
            return False
        
        # 检查是否有best_val_mae（v4格式）或val_mae（v3格式）
        if 'best_val_mae' in ckpt:
            logger.info(f"     - best_val_mae: {ckpt['best_val_mae']:.6f} (v4格式)")
        elif 'val_mae' in ckpt:
            logger.info(f"     - val_mae: {ckpt['val_mae']:.6f} (v3格式)")
        else:
            logger.warning(f"  ⚠️  {experiment_name}: 缺少val_mae（可选但推荐）")
        
        logger.info(f"  ✅ {experiment_name}: 模型保存格式正确")
        logger.info(f"     - epoch: {ckpt.get('epoch', 'N/A')}")
        logger.info(f"     - best_val_loss: {ckpt.get('best_val_loss', 'N/A'):.6f}")
        logger.info(f"     - 包含所有必要的统计量（input_mean/std, output_mean/std, feature_mean/std）")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ {experiment_name}: 检查模型格式失败: {e}")
        return False


def run_experiment(exp_config):
    """
    运行单个实验
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"开始测试: {exp_config['name']}")
    logger.info(f"{'='*80}")
    
    script_path = Path(__file__).parent / exp_config['script']
    
    if not script_path.exists():
        logger.error(f"❌ 脚本不存在: {script_path}")
        return False
    
    # 构建命令
    cmd = [sys.executable, str(script_path)] + exp_config['args']
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 运行训练脚本（实时显示输出）
        logger.info("开始训练...")
        print()  # 空行分隔
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            # 不捕获输出，让训练过程的输出直接显示到终端
            timeout=600  # 10分钟超时
        )
        print()  # 空行分隔
        
        if result.returncode != 0:
            logger.error(f"❌ {exp_config['name']}: 训练失败 (返回码: {result.returncode})")
            return False
        
        logger.info(f"✅ {exp_config['name']}: 训练完成")
        
        # 检查输出文件
        output_dir = Path(__file__).parent
        all_files_exist = True
        
        # 从expected_files中提取suffix来确定目录名
        # 例如: best_model_agents_3_exp1_baseline.pt -> agents_3_exp1_baseline
        first_file = exp_config['expected_files'][0]
        if 'best_model_' in first_file:
            suffix = first_file.replace('best_model_', '').replace('.pt', '')
            result_dir = output_dir / f"ablation_results_{suffix}"
        else:
            # 回退：查找所有可能的目录
            result_dir = None
            for dir_path in output_dir.glob('ablation_results_*'):
                if dir_path.is_dir():
                    result_dir = dir_path
                    break
        
        if result_dir and result_dir.exists():
            logger.info(f"  ✓ 找到输出目录: {result_dir.name}")
        else:
            logger.warning(f"  ⚠️  未找到输出目录，尝试查找文件...")
            result_dir = None
        
        for filename in exp_config['expected_files']:
            if result_dir:
                file_path = result_dir / filename
            else:
                # 回退：在所有ablation_results_*目录中查找
                file_path = None
                for dir_path in output_dir.glob('ablation_results_*'):
                    candidate = dir_path / filename
                    if candidate.exists():
                        file_path = candidate
                        break
            
            if file_path and file_path.exists():
                logger.info(f"  ✓ 找到文件: {file_path.name}")
                
                # 如果是模型文件，检查保存格式
                if filename.endswith('.pt'):
                    check_model_save_format(file_path, exp_config['name'])
            else:
                logger.warning(f"  ⚠️  未找到文件: {filename}")
                all_files_exist = False
        
        return all_files_exist
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {exp_config['name']}: 训练超时（>10分钟）")
        return False
    except Exception as e:
        logger.error(f"❌ {exp_config['name']}: 运行出错: {e}")
        return False


def main():
    """
    主函数：运行所有实验测试
    """
    logger.info("="*80)
    logger.info("消融实验测试脚本")
    logger.info("="*80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"将测试 {len(EXPERIMENTS)} 个实验，每个训练1个epoch")
    logger.info("")
    
    results = []
    
    for i, exp_config in enumerate(EXPERIMENTS, 1):
        logger.info(f"\n[{i}/{len(EXPERIMENTS)}] {exp_config['name']}")
        
        success = run_experiment(exp_config)
        results.append({
            'name': exp_config['name'],
            'success': success
        })
        
        if not success:
            logger.warning(f"⚠️  {exp_config['name']} 测试失败，但继续测试其他实验...")
    
    # 总结
    logger.info("\n" + "="*80)
    logger.info("测试总结")
    logger.info("="*80)
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    for result in results:
        status = "✅ 通过" if result['success'] else "❌ 失败"
        logger.info(f"{status}: {result['name']}")
    
    logger.info("")
    logger.info(f"总计: {success_count}/{total_count} 个实验测试通过")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        logger.info("🎉 所有实验测试通过！")
        return 0
    else:
        logger.warning(f"⚠️  有 {total_count - success_count} 个实验测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
