# @File: test_new_env.py
# @Description: 在新的、从未见过的50个环境(250-299)上评估模型性能。
#               1. 包含一个定制的DataLoader，专门加载 env_250 到 env_299 的所有轨迹。
#               2. 在这个新的测试集上计算 NMSE, RMSE, PSNR, SSIM 指标。
#               3. 测量推理时间并生成独立的评估报告。

import os
import argparse
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from skimage import io
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError

# 从 train.py 导入模型定义，确保脚本可以独立运行
from train import LightningConvLSTM

# ===================================================================
# 1. 定制化的数据加载器 (Custom Data Loader for New Environments)
# ===================================================================
class NewEnvUrbanRadioVideoDataset(Dataset):
    """
    一个专门用于加载新环境 (250-299) 进行测试的 Dataset 类。
    它会加载指定环境下所有的轨迹 (traj_00 到 traj_04)。
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
        self.test_env_indices = range(250, 300) # 环境 250 到 299
        self.test_traj_indices = range(0, 5)   # 轨迹 0 到 4

        print(f"[New Env Test] Scanning for trajectories for environments {self.test_env_indices.start}-{self.test_env_indices.stop-1}...")

        for env_idx in self.test_env_indices:
            env_folder = f"env_{env_idx:03d}"
            for traj_idx in self.test_traj_indices:
                traj_folder = f"traj_{traj_idx:02d}"
                traj_path = os.path.join(dir_gain, env_folder, traj_folder)
                if not os.path.isdir(traj_path):
                    print(f"Warning: Directory not found, skipping: {traj_path}")
                    continue

                for tx_folder in sorted(os.listdir(traj_path)):
                    tx_path = os.path.join(traj_path, tx_folder)
                    if not os.path.isdir(tx_path): continue
                    
                    try:
                        frame_files = sorted([f for f in os.listdir(tx_path) if f.endswith('.png')], 
                                             key=lambda f: int(f.split('_')[-1].split('.')[0]))
                    except (IndexError, ValueError):
                        continue
                    
                    if len(frame_files) >= self.sequence_length:
                        for i in range(len(frame_files) - self.sequence_length + 1):
                            full_slice_paths = [os.path.join(tx_path, f) for f in frame_files[i : i + self.sequence_length]]
                            original_filenames = frame_files[i : i + self.sequence_length]
                            self.trajectory_slices.append({
                                "paths": full_slice_paths,
                                "original_filenames": original_filenames
                            })
        
        print(f"[New Env Test] Found {len(self.trajectory_slices)} sequences in the new environments.")

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
        
        return context_video, target_video, {"filenames": original_filenames}

# ===================================================================
# 2. 主评估函数 (Main Evaluation Function)
# ===================================================================
def test_and_evaluate_new_env():
    # --- 参数与配置加载 ---
    parser = argparse.ArgumentParser(description="Test model on new, unseen environments (250-299).")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint. Auto-detects if not provided.')
    parser.add_argument('--output_dir', type=str, default='./test_new_env_resu', help='Directory to save visualization images.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file.')
    args = parser.parse_args()

    # --- 从配置文件加载配置 ---
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 为测试调整一些参数
    config['data']['loader_params'] = {
        'batch_size': 1,
        'shuffle': False, # 测试时通常不打乱
        'num_workers': 1
    }
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 自动查找 Checkpoint ---
    if args.checkpoint is None:
        checkpoint_dir = config['callbacks']['checkpoint']['dirpath']
        candidate_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('best-convlstm') and f.endswith('.ckpt')]
        if not candidate_files:
            print(f"错误: 在 '{checkpoint_dir}' 中未找到 'best-convlstm' checkpoint。"); return
        candidate_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
        args.checkpoint = os.path.join(checkpoint_dir, candidate_files[0])
    print(f"使用 Checkpoint: {args.checkpoint}")

    # --- 加载模型和数据 ---
    model = LightningConvLSTM.load_from_checkpoint(args.checkpoint, map_location=device).model
    model.eval()

    test_dataset = NewEnvUrbanRadioVideoDataset(data_config=config['data'])
    test_loader = DataLoader(test_dataset, **config['data']['loader_params'])

    # --- 初始化指标和工具 ---
    mse_metric = MeanSquaredError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total_nmse_numerator, total_nmse_denominator = 0.0, 0.0
    inference_times = []
    saved_samples_count = 0
    num_samples_to_save = 50  # 50个环境 × 1个轨迹 = 50个样本 (每个环境只保存第一个轨迹的第一个tx)
    saved_env_tx = set()  # 记录已保存的环境-tx组合
    
    def extract_env_traj_tx_from_path(file_path):
        """从文件路径中提取环境、轨迹和tx信息"""
        path_parts = file_path.split('/')
        env_folder = None
        traj_folder = None
        tx_folder = None
        
        for part in path_parts:
            if part.startswith('env_'):
                env_folder = part
            elif part.startswith('traj_'):
                traj_folder = part
            elif part.startswith('tx_'):
                tx_folder = part
        
        return env_folder, traj_folder, tx_folder

    # --- 执行测试循环 ---
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="正在评估新环境测试集")
        batch_idx = 0
        for context, target, metadata in pbar:
            context, target = context.to(device), target.to(device)
            
            if device.type == 'cuda':
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
            
            prediction = model(context)
            
            if device.type == 'cuda':
                end_event.record(); torch.cuda.synchronize()
                inference_times.append(start_event.elapsed_time(end_event) / context.shape[0])

            # 指标计算
            pred_reshaped, target_reshaped = prediction.view(-1, *prediction.shape[2:]), target.view(-1, *target.shape[2:])
            mse_metric.update(pred_reshaped, target_reshaped)
            psnr_metric.update(pred_reshaped, target_reshaped)
            ssim_metric.update(pred_reshaped, target_reshaped)
            total_nmse_numerator += torch.sum((pred_reshaped - target_reshaped) ** 2).item()
            total_nmse_denominator += torch.sum(target_reshaped ** 2).item()

            # --- 可视化保存 (50个环境×5个轨迹，每个轨迹的第一个tx) ---
            if saved_samples_count < num_samples_to_save:
                filenames_per_sample = list(zip(*metadata["filenames"]))
                
                for j in range(context.shape[0]):
                    if saved_samples_count >= num_samples_to_save: break
                    
                    # 获取当前样本在数据集中的索引
                    sample_idx = batch_idx * test_loader.batch_size + j
                    if sample_idx >= len(test_dataset.trajectory_slices):
                        break
                        
                    # 从数据集中获取当前样本的路径信息
                    sample_slice = test_dataset.trajectory_slices[sample_idx]
                    sample_path = sample_slice["paths"][0]  # 取第一个路径
                    env_folder, traj_folder, tx_folder = extract_env_traj_tx_from_path(sample_path)
                    
                    # 检查是否应该保存：每个环境只保存第一个轨迹的第一个tx
                    if env_folder and traj_folder and tx_folder:
                        env_tx_key = f"{env_folder}_{tx_folder}"
                        
                        # 只保存第一个轨迹的第一个tx，且每个环境只保存一次
                        if traj_folder == 'traj_00' and tx_folder == 'tx_00' and env_tx_key not in saved_env_tx:
                            saved_env_tx.add(env_tx_key)
                            sample_filenames = filenames_per_sample[j]
                            
                            context_frames = config['data']['context_frames']
                            context_start_idx = config['data'].get('context_start_frame_idx', 0)
                            prediction_frames = config['data']['prediction_frames']
                            
                            # 保存上下文帧 (Context) - 根据配置动态确定数量
                            for t in range(context_frames):
                                filename = os.path.join(args.output_dir, f"context_{sample_filenames[context_start_idx + t]}")
                                save_image(context[j, t], filename)
                            
                            # 保存真值帧 (Ground Truth) - 紧接着上下文帧
                            for t in range(prediction_frames):
                                filename = os.path.join(args.output_dir, f"gt_{sample_filenames[context_start_idx + context_frames + t]}")
                                save_image(target[j, t], filename)
                            
                            # 保存预测帧 (Prediction) - 与真值帧对应
                            for t in range(prediction_frames):
                                filename = os.path.join(args.output_dir, f"pred_{sample_filenames[context_start_idx + context_frames + t]}")
                                save_image(prediction[j, t], filename)
                            
                            saved_samples_count += 1
            
            batch_idx += 1

    # --- 生成报告 ---
    final_mse = mse_metric.compute().item()
    final_psnr = psnr_metric.compute().item()
    final_ssim = ssim_metric.compute().item()
    final_rmse = np.sqrt(final_mse)
    final_nmse = total_nmse_numerator / total_nmse_denominator if total_nmse_denominator else 0
    avg_time_per_traj_ms = np.mean(inference_times) if inference_times else 0
    avg_time_per_frame_ms = avg_time_per_traj_ms / config['data']['prediction_frames'] if inference_times else 0

    report = f"""
新环境模型评估报告 (Environments 250-299)
============================================================
Checkpoint: {os.path.basename(args.checkpoint)}
============================================================
性能指标:
  平均推理时间 (预测5帧): {avg_time_per_traj_ms:.4f} ms/traj
  平均每帧推理时间:         {avg_time_per_frame_ms:.4f} ms/frame
------------------------------------------------------------
评估指标 (在 env 250-299 的所有轨迹上计算):
  NMSE:   {final_nmse:.6f}
  RMSE:   {final_rmse:.6f}
  PSNR:   {final_psnr:.4f} dB
  SSIM:   {final_ssim:.4f}
============================================================
"""
    # 将报告保存在checkpoint目录下，并使用特定文件名
    report_path = os.path.join(os.path.dirname(args.checkpoint), 'test_evaluation_results_new_env.txt')
    with open(report_path, 'w') as f: f.write(report)
    print("\n" + "="*60 + "\n评估完成！" + report + f"详细报告已保存至: {report_path}\n" + "="*60)

if __name__ == '__main__':
    test_and_evaluate_new_env()
