# RadioMapMotion

A dataset and benchmark for proactive spatio-temporal radio environment prediction in vehicular scenarios.

## Dataset

The UrbanRadio-V dataset is available at:

[Google Drive](https://drive.google.com/file/d/1PvjnCHc9E3dBKN_34lwHv36g6cAsf4et/view?usp=sharing)

### Dataset Structure

```
data/
├── gain/
│   ├── env_000/
│   │   ├── traj_00/
│   │   │   ├── tx_00/  (frame_0.png ~ frame_14.png)
│   │   │   ├── tx_01/
│   │   │   └── ...
│   │   ├── traj_01/
│   │   └── ...
│   └── ...
```

### Dataset Split

- **Training**: env_000-env_249, traj_00-traj_02 (15000 sequences)
- **Validation**: env_000-env_249, traj_03 (5000 sequences)
- **Test**: env_000-env_249, traj_04 (5000 sequences)
- **New Environment Test**: env_250-env_299, traj_00-traj_04 (5000 sequences)

Each sequence contains 8 context frames and 5 prediction frames.

## Requirements

```bash
pip install torch torchvision pytorch-lightning torchmetrics einops scikit-image tqdm pyyaml
```

## Training

1. Update `config.yaml`:
   - `data.dynamic_data_root`: path to dataset
   - `trainer_config.devices`: GPU devices
   - `callbacks.checkpoint.dirpath`: model save path

2. Run training:

```bash
python code/train.py --config code/config.yaml
```

## Testing

### Test Set (env_000-env_249, traj_04)

```bash
python code/test.py --config code/config.yaml --checkpoint path/to/checkpoint.ckpt --output_dir ./test_resu
```

### New Environment Test (env_250-env_299)

```bash
python code/test_new_env.py --config code/config.yaml --checkpoint path/to/checkpoint.ckpt --output_dir ./new_env_test_resu
```

### New Environment Frame-wise Test

```bash
python code/test_new_env_framewise.py --config code/config.yaml --checkpoint path/to/checkpoint.ckpt --output_dir ./new_env_framewise_resu
```

Evaluation metrics: NMSE, RMSE, PSNR, SSIM.

## Citation

```bibtex
@article{cheng2026radiomapmotion,
  title={RadioMapMotion: A Dataset and Benchmark for Proactive Spatio-Temporal Radio Environment Prediction},
  author={Cheng, Nan and Jia, Haotian and Wang, Xiang and others},
  journal={IEEE Transactions on Cognitive Communications and Networking},
  year={2026}
}
```
