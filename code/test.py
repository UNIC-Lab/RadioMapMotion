# @File: test.py
# @Description: 最终版评估脚本 (已修复Bug)。
#               1. 正确保存10帧context, 5帧gt, 5帧pred。
#               2. 在整个测试集上计算 NMSE, RMSE, PSNR, SSIM 指标。
#               3. 测量推理时间并生成综合报告。

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError


from train import LightningConvLSTM
from data_loader import UrbanRadioVideoDataset # 确保您使用的是返回文件名的版本

def test_and_evaluate():
    # --- 参数与配置加载 ---
    parser = argparse.ArgumentParser(description="Final testing and evaluation script (Bug Fixed).")
    parser.add_argument('--config', type=str, default='/Data/hgjia/scr/RadioMotion/config.yaml', help='Path to config file.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint. Auto-detects if not provided.')
    parser.add_argument('--output_dir', type=str, default='./test_resu', help='Directory to save visualization images.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{config['trainer_config']['devices'][0]}" if torch.cuda.is_available() else "cpu")
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

    test_params = config['data']['loader_params'].copy()
    test_params['shuffle'] = False
    test_dataset = UrbanRadioVideoDataset(phase="test", data_config=config['data'])
    test_loader = DataLoader(test_dataset, **test_params)

    # --- 初始化指标和工具 ---
    mse_metric, psnr_metric, ssim_metric = MeanSquaredError().to(device), PeakSignalNoiseRatio(data_range=1.0).to(device), StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total_nmse_numerator, total_nmse_denominator = 0.0, 0.0
    inference_times = []
    saved_samples_count = 0
    num_samples_to_save = 50  # 50个环境，每个环境一个样本
    saved_env_tx = set()  # 记录已保存的环境-tx组合
    
    def extract_env_tx_from_path(file_path):
        """从文件路径中提取环境和tx信息"""
        path_parts = file_path.split('/')
        env_folder = None
        tx_folder = None
        
        for part in path_parts:
            if part.startswith('env_'):
                env_folder = part
            elif part.startswith('tx_'):
                tx_folder = part
        
        return env_folder, tx_folder

    # --- 执行测试循环 ---
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="正在评估测试集")
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
            mse_metric.update(pred_reshaped, target_reshaped); psnr_metric.update(pred_reshaped, target_reshaped); ssim_metric.update(pred_reshaped, target_reshaped)
            total_nmse_numerator += torch.sum((pred_reshaped - target_reshaped) ** 2).item()
            total_nmse_denominator += torch.sum(target_reshaped ** 2).item()

            # --- 可视化保存 (50个环境样本，每个环境的第一个tx) ---
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
                    env_folder, tx_folder = extract_env_tx_from_path(sample_path)
                    
                    # 检查是否应该保存：前50个环境且第一个tx
                    if env_folder and tx_folder:
                        # 提取环境编号
                        env_num = int(env_folder.split('_')[1]) if len(env_folder.split('_')) > 1 else 999
                        env_tx_key = f"{env_folder}_{tx_folder}"
                        
                        # 只保存前50个环境的第一个tx，且每个环境只保存一次
                        if env_num < 50 and tx_folder == 'tx_00' and env_tx_key not in saved_env_tx:
                            saved_env_tx.add(env_tx_key)
                            sample_filenames = filenames_per_sample[j]
                            
                            # 保存10个输入帧 (Context)
                            for t in range(10):
                                filename = os.path.join(args.output_dir, f"context_{sample_filenames[t]}")
                                save_image(context[j, t], filename)
                            
                            # 保存5个真值帧 (Ground Truth)
                            for t in range(5):
                                filename = os.path.join(args.output_dir, f"gt_{sample_filenames[10+t]}")
                                save_image(target[j, t], filename)
                            
                            # 保存5个预测帧 (Prediction)
                            for t in range(5):
                                filename = os.path.join(args.output_dir, f"pred_{sample_filenames[10+t]}")
                                save_image(prediction[j, t], filename)
                            
                            saved_samples_count += 1
            
            batch_idx += 1

    # --- 生成报告 ---
    final_mse, final_psnr, final_ssim = mse_metric.compute().item(), psnr_metric.compute().item(), ssim_metric.compute().item()
    final_rmse, final_nmse = np.sqrt(final_mse), total_nmse_numerator / total_nmse_denominator
    avg_time_per_traj_ms = np.mean(inference_times) if inference_times else 0
    avg_time_per_frame_ms = avg_time_per_traj_ms / config['data']['prediction_frames'] if inference_times else 0

    report = f"""
模型评估报告
============================================================
Checkpoint: {os.path.basename(args.checkpoint)}
============================================================
性能指标:
  平均推理时间 (预测5帧): {avg_time_per_traj_ms:.4f} ms/traj
  平均每帧推理时间:         {avg_time_per_frame_ms:.4f} ms/frame
------------------------------------------------------------
评估指标 (在整个测试集上计算):
  NMSE:   {final_nmse:.6f}
  RMSE:   {final_rmse:.6f}
  PSNR:   {final_psnr:.4f} dB
  SSIM:   {final_ssim:.4f}
============================================================
"""
    report_path = os.path.join(os.path.dirname(args.checkpoint), 'test_evaluation_results.txt')
    with open(report_path, 'w') as f: f.write(report)
    print("\n" + "="*60 + "\n评估完成！" + report + f"详细报告已保存至: {report_path}\n" + "="*60)

if __name__ == '__main__':
    test_and_evaluate()