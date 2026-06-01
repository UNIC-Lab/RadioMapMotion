# RadioMapMotion

基于 ConvLSTM-UNet 的无线信号强度预测模型，用于预测车辆移动场景下的信号传播动态变化。

## 环境要求

```bash
# Python >= 3.8
pip install torch torchvision pytorch-lightning torchmetrics einops scikit-image tqdm pyyaml
```

## 数据集

使用 UrbanRadio-V 数据集，结构如下：

```
data/
├── gain/
│   ├── env_000/
│   │   ├── traj_00/
│   │   │   ├── tx_00/  (包含 frame_0.png ~ frame_14.png)
│   │   │   ├── tx_01/
│   │   │   └── ...
│   │   ├── traj_01/
│   │   └── ...
│   └── ...
```

数据集划分：
- **训练集**: env_000-env_249, traj_00-traj_02 (15000 序列)
- **验证集**: env_000-env_249, traj_03 (5000 序列)
- **测试集**: env_000-env_249, traj_04 (5000 序列)
- **新环境测试集**: env_250-env_299, traj_00-traj_04 (5000 序列)

每个序列包含 8 帧上下文帧和 5 帧预测帧。

## 训练

1. 修改 `config.yaml` 中的配置：
   - `data.dynamic_data_root`: 数据集路径
   - `trainer_config.devices`: 使用的 GPU 设备
   - `callbacks.checkpoint.dirpath`: 模型保存路径

2. 运行训练：

```bash
python train.py --config config.yaml
```

## 测试

### 测试集评估 (env_000-env_249, traj_04)

```bash
python test.py --config config.yaml --checkpoint path/to/checkpoint.ckpt --output_dir ./test_resu
```

### 新环境评估 (env_250-env_299)

```bash
python test_new_env.py --config config.yaml --checkpoint path/to/checkpoint.ckpt --output_dir ./new_env_test_resu
```

### 新环境按帧评估

```bash
python test_new_env_framewise.py --config config.yaml --checkpoint path/to/checkpoint.ckpt --output_dir ./new_env_framewise_resu
```

评估指标：NMSE, RMSE, PSNR, SSIM

## 模型架构

ConvLSTM-UNet：结合 ConvLSTM 时序建模能力与 U-Net 编码器-解码器结构，用于无线信号强度时空序列预测。
