# @File: test_new_env_framewise.py
# @Description: 在新的、从未见过的50个环境(250-299)上按帧评估模型性能。
#               1. 包含一个定制的DataLoader，专门加载 env_250 到 env_299 的所有轨迹。
#               2. 根据配置动态计算预测帧编号，分别计算每一帧的 NMSE, RMSE, PSNR, SSIM 指标。
#               3. 支持动态上下文配置 (context_frames, context_start_frame_idx)。
#               4. 每帧的指标基于5000个序列计算（50个环境 × 5个轨迹 × 20个tx）。
#               5. 生成按帧分组的指标结果，保存到txt文件中。

import os
import argparse
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from skimage import io
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError

# 从 train.py 导入模型定义
from train import LightningConvLSTM

# ===================================================================
# 1. 定制化的数据加载器 (Custom Data Loader for New Environments)
# ===================================================================
class NewEnvUrbanRadioVideoDataset(Dataset):
    """
    专门用于加载新环境 (250-299) 进行按帧测试的 Dataset 类。
    加载所有环境下所有轨迹的所有tx数据。
    """
    def __init__(self, data_config, transform=transforms.ToTensor()):
        self.transform = transform
        self.data_config = data_config
        self.trajectory_slices = []

        self.context_frames = data_config['context_frames']
        self.prediction_frames = data_config['prediction_frames']
        self.context_start_frame_idx = data_config.get('context_start_frame_idx', 0)
        self.sequence_length = self.context_start_frame_idx + self.context_frames + self.prediction_frames
        
        dir_gain = os.path.join(data_config['dynamic_data_root'], "gain")

        # 定义要加载的新环境范围
        self.test_env_indices = range(250, 300)  # 环境 250 到 299 (50个环境)
        self.test_traj_indices = range(0, 5)     # 轨迹 0 到 4 (5个轨迹)

        print(f"[Frame-wise New Env Test] 扫描环境 {self.test_env_indices.start}-{self.test_env_indices.stop-1} 的所有轨迹...")

        for env_idx in self.test_env_indices:
            env_folder = f"env_{env_idx:03d}"
            for traj_idx in self.test_traj_indices:
                traj_folder = f"traj_{traj_idx:02d}"
                traj_path = os.path.join(dir_gain, env_folder, traj_folder)
                if not os.path.isdir(traj_path):
                    print(f"警告: 目录不存在，跳过: {traj_path}")
                    continue

                # 遍历所有tx文件夹（不只是tx_00）
                for tx_folder in sorted(os.listdir(traj_path)):
                    tx_path = os.path.join(traj_path, tx_folder)
                    if not os.path.isdir(tx_path) or not tx_folder.startswith('tx_'): 
                        continue
                    
                    try:
                        frame_files = sorted([f for f in os.listdir(tx_path) if f.endswith('.png')], 
                                             key=lambda f: int(f.split('_')[-1].split('.')[0]))
                    except (IndexError, ValueError):
                        continue
                    
                    # 只取第一个完整序列（15帧）
                    if len(frame_files) >= self.sequence_length:
                        full_slice_paths = [os.path.join(tx_path, f) for f in frame_files[:self.sequence_length]]
                        original_filenames = frame_files[:self.sequence_length]
                        self.trajectory_slices.append({
                            "paths": full_slice_paths,
                            "original_filenames": original_filenames,
                            "env": env_folder,
                            "traj": traj_folder,
                            "tx": tx_folder
                        })
        
        print(f"[Frame-wise New Env Test] 找到 {len(self.trajectory_slices)} 个序列，预期应为 {50*5*20} = 5000 个")

    def __len__(self):
        return len(self.trajectory_slices)

    def __getitem__(self, idx):
        slice_info = self.trajectory_slices[idx]
        frame_paths = slice_info["paths"]
        original_filenames = slice_info["original_filenames"]

        frames = []
        for path in frame_paths:
            image = np.expand_dims(io.imread(path).astype(np.float32) / 255.0, axis=2)
            frames.append(self.transform(image))
        
        video_tensor = torch.stack(frames, dim=0).type(torch.float32)
        
        context_video = video_tensor[self.context_start_frame_idx : self.context_start_frame_idx + self.context_frames]
        target_video = video_tensor[self.context_start_frame_idx + self.context_frames : self.context_start_frame_idx + self.context_frames + self.prediction_frames]
        
        return context_video, target_video, {
            "filenames": original_filenames,
            "env": slice_info["env"],
            "traj": slice_info["traj"],
            "tx": slice_info["tx"]
        }

# ===================================================================
# 2. 按帧评估函数 (Frame-wise Evaluation Function)
# ===================================================================
def test_and_evaluate_new_env_framewise():
    # --- 参数与配置加载 ---
    parser = argparse.ArgumentParser(description="按帧测试模型在新环境(250-299)上的性能。")
    parser.add_argument('--checkpoint', type=str, default=None, help='模型checkpoint路径。如果不提供则自动检测。')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径。')
    args = parser.parse_args()

    # --- 从配置文件加载配置 ---
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 为测试调整一些参数
    config['data']['loader_params'] = {
        'batch_size': 16,  # 增大batch size以提高效率
        'shuffle': False,
        'num_workers': 4
    }
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # --- 自动查找 Checkpoint ---
    if args.checkpoint is None:
        checkpoint_dir = config['callbacks']['checkpoint']['dirpath']
        candidate_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('best-convlstm') and f.endswith('.ckpt')]
        if not candidate_files:
            print(f"错误: 在 '{checkpoint_dir}' 中未找到 'best-convlstm' checkpoint。")
            return
        candidate_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
        args.checkpoint = os.path.join(checkpoint_dir, candidate_files[0])
    print(f"使用 Checkpoint: {args.checkpoint}")

    # --- 加载模型和数据 ---
    model = LightningConvLSTM.load_from_checkpoint(args.checkpoint, map_location=device).model
    model.eval()

    test_dataset = NewEnvUrbanRadioVideoDataset(data_config=config['data'])
    test_loader = DataLoader(test_dataset, **config['data']['loader_params'])

    # --- 为每一帧初始化指标 ---
    num_prediction_frames = config['data']['prediction_frames']
    
    # 为每一帧创建独立的指标计算器
    frame_metrics = {}
    for frame_idx in range(num_prediction_frames):
        frame_metrics[frame_idx] = {
            'mse': MeanSquaredError().to(device),
            'psnr': PeakSignalNoiseRatio(data_range=1.0).to(device),
            'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
            'nmse_numerator': 0.0,
            'nmse_denominator': 0.0
        }

    # --- 执行测试循环 ---
    total_samples = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="按帧评估新环境测试集")
        
        for context, target, metadata in pbar:
            context, target = context.to(device), target.to(device)
            
            # 模型推理
            prediction = model(context)
            
            # 对每一帧分别计算指标
            for frame_idx in range(num_prediction_frames):
                # 提取当前帧的预测和真值
                pred_frame = prediction[:, frame_idx]  # shape: [batch_size, 1, H, W]
                target_frame = target[:, frame_idx]    # shape: [batch_size, 1, H, W]
                
                # 确保tensor是连续的，避免view操作错误
                pred_frame = pred_frame.contiguous()
                target_frame = target_frame.contiguous()
                
                # 更新当前帧的指标
                frame_metrics[frame_idx]['mse'].update(pred_frame, target_frame)
                frame_metrics[frame_idx]['psnr'].update(pred_frame, target_frame)
                frame_metrics[frame_idx]['ssim'].update(pred_frame, target_frame)
                
                # NMSE 计算
                frame_metrics[frame_idx]['nmse_numerator'] += torch.sum((pred_frame - target_frame) ** 2).item()
                frame_metrics[frame_idx]['nmse_denominator'] += torch.sum(target_frame ** 2).item()
            
            total_samples += context.shape[0]

    print(f"\n总共处理了 {total_samples} 个样本")

    # --- 计算最终结果并生成报告 ---
    context_start_idx = config['data'].get('context_start_frame_idx', 0)
    context_frames = config['data']['context_frames']
    
    results = {}
    for frame_idx in range(num_prediction_frames):
        # 根据配置动态计算预测帧的实际编号
        actual_frame_num = context_start_idx + context_frames + frame_idx
        frame_num = actual_frame_num
        
        final_mse = frame_metrics[frame_idx]['mse'].compute().item()
        final_psnr = frame_metrics[frame_idx]['psnr'].compute().item()
        final_ssim = frame_metrics[frame_idx]['ssim'].compute().item()
        final_rmse = np.sqrt(final_mse)
        final_nmse = (frame_metrics[frame_idx]['nmse_numerator'] / 
                     frame_metrics[frame_idx]['nmse_denominator'] 
                     if frame_metrics[frame_idx]['nmse_denominator'] > 0 else 0)
        
        results[frame_num] = {
            'NMSE': final_nmse,
            'RMSE': final_rmse,
            'PSNR': final_psnr,
            'SSIM': final_ssim
        }

    # --- 生成报告 ---
    report = f"""
按帧新环境模型评估报告 (Environments 250-299)
============================================================
Checkpoint: {os.path.basename(args.checkpoint)}
总样本数: {total_samples}
============================================================

各帧评估指标 (基于约{total_samples}个样本):
"""

    for frame_num in sorted(results.keys()):
        metrics = results[frame_num]
        prediction_idx = frame_num - context_start_idx - context_frames + 1  # 预测的第几帧 (1-based)
        report += f"""
------------------------------------------------------------
第 {frame_num} 帧 (预测第 {prediction_idx} 帧):
  NMSE:   {metrics['NMSE']:.6f}
  RMSE:   {metrics['RMSE']:.6f}
  PSNR:   {metrics['PSNR']:.4f} dB
  SSIM:   {metrics['SSIM']:.4f}
"""

    report += "\n============================================================\n"

    # --- 保存报告 ---
    report_path = os.path.join(os.path.dirname(args.checkpoint), 'test_evaluation_results_new_env_framewise.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print("按帧评估完成！")
    print(report)
    print(f"详细报告已保存至: {report_path}")
    print(f"{'='*60}")

    return results

if __name__ == '__main__':
    test_and_evaluate_new_env_framewise()
