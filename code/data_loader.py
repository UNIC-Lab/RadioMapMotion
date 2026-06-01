
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms

class UrbanRadioVideoDataset(Dataset):
    def __init__(self, phase="train", data_config=None, transform=transforms.ToTensor()):
        self.phase = phase
        self.transform = transform
        self.data_config = data_config
        self.trajectory_slices = []

        self.context_frames = data_config['context_frames']
        self.prediction_frames = data_config['prediction_frames']
        self.context_start_frame_idx = data_config.get('context_start_frame_idx', 0) # 新增：从配置中读取起始帧索引，默认为0

        # 序列总长度现在需要考虑起始帧索引
        self.sequence_length = self.context_start_frame_idx + self.context_frames + self.prediction_frames
        self.num_environments = data_config.get('num_environments', 250)  # 从配置中读取环境数量，默认为250

        dir_gain = os.path.join(data_config['dynamic_data_root'], "gain")

        if self.phase == "train":
            traj_ranges = [(0, 2)]
        elif self.phase == "val":
            traj_ranges = [(3, 3)]
        else:  # "test"
            traj_ranges = [(4, 4)]

        print(f"[{self.phase.upper()}] Scanning for trajectories for RadioMotionMap...")

        for env_idx in range(self.num_environments):
            env_folder = f"env_{env_idx:03d}"
            for start_traj, end_traj in traj_ranges:
                for traj_idx in range(start_traj, end_traj + 1):
                    traj_folder = f"traj_{traj_idx:02d}"
                    for tx_folder in sorted(os.listdir(os.path.join(dir_gain, env_folder, traj_folder))):
                        tx_path = os.path.join(dir_gain, env_folder, traj_folder, tx_folder)
                        if not os.path.isdir(tx_path): continue
                        
                        try:
                            frame_files = sorted([f for f in os.listdir(tx_path) if f.endswith('.png')], key=lambda f: int(f.split('_')[-1].split('.')[0]))
                        except (IndexError, ValueError): continue
                        
                        # 确保有足够的帧来提取完整的序列
                        if len(frame_files) >= self.sequence_length:
                            for i in range(len(frame_files) - self.sequence_length + 1):
                                # 保存完整路径和用于命名的原始文件名
                                full_slice_paths = [os.path.join(tx_path, f) for f in frame_files[i : i + self.sequence_length]]
                                original_filenames = frame_files[i : i + self.sequence_length]
                                self.trajectory_slices.append({
                                    "paths": full_slice_paths,
                                    "original_filenames": original_filenames
                                })
        
        print(f"[{self.phase.upper()}] Found {len(self.trajectory_slices)} sequences.")

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
        
        # 根据 context_start_frame_idx 和 context_frames 提取上下文帧
        context_video = video_tensor[self.context_start_frame_idx : self.context_start_frame_idx + self.context_frames]
        # 目标帧紧随上下文帧之后
        target_video = video_tensor[self.context_start_frame_idx + self.context_frames : self.context_start_frame_idx + self.context_frames + self.prediction_frames]
        
        # 返回数据和用于命名的元数据
        return context_video, target_video, {"filenames": original_filenames}