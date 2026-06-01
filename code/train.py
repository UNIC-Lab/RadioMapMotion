# @File: train.py
# @Description: 训练 ConvLSTM_UNet 模型的脚本。

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml
import argparse
from torchvision.utils import make_grid, save_image
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import StructuralSimilarityIndexMeasure
from einops import rearrange

from modules_convlstm_unet import ConvLSTM_UNet
from data_loader import UrbanRadioVideoDataset

class CombinedLoss(nn.Module):
    """ 组合损失函数，保持不变。 """
    def __init__(self, l1_weight=0.3, mse_weight=0.2, ssim_weight=0.5):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

    def forward(self, prediction, target):
        b, n, c, h, w = prediction.shape
        prediction_reshaped = rearrange(prediction, 'b n c h w -> (b n) c h w')
        target_reshaped = rearrange(target, 'b n c h w -> (b n) c h w')
        
        self.ssim_metric.to(prediction.device)
        l1 = self.l1_loss(prediction_reshaped, target_reshaped)
        mse = self.mse_loss(prediction_reshaped, target_reshaped)
        ssim_val = self.ssim_metric(prediction_reshaped, target_reshaped)
        
        # 1. Clamp ssim_val to prevent it from exceeding 1 due to precision issues
        ssim_val = torch.clamp(ssim_val, max=0.9999)
        
        ssim_loss = 1.0 - ssim_val

        total_loss = (self.l1_weight * l1) + (self.mse_weight * mse) + (self.ssim_weight * ssim_loss)
        # 2. Return both ssim_loss for calculation and ssim_val for logging
        return total_loss, l1, mse, ssim_loss, ssim_val

class LightningConvLSTM(pl.LightningModule):
    """ PyTorch Lightning 封装器，用于 ConvLSTM_UNet 模型。 """
    def __init__(self, model_params, training_params):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvLSTM_UNet(**model_params)
        self.training_params = training_params
        self.loss_fn = CombinedLoss(**training_params.get('loss_weights', {}))

    def forward(self, x):
        return self.model(x)

    def _log_metrics(self, stage, loss, l1, mse, ssim_loss, ssim_val):
        self.log_dict({
            f'{stage}_loss': loss, f'{stage}_l1': l1,
            f'{stage}_mse': mse, f'{stage}_ssim_loss': ssim_loss,
            f'{stage}_ssim': ssim_val
        }, on_step=(stage=='train'), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        context, target, _ = batch
        prediction = self(context)
        loss, l1, mse, ssim_loss, ssim_val = self.loss_fn(prediction, target)
        self._log_metrics('train', loss, l1, mse, ssim_loss, ssim_val)
        return loss

    def validation_step(self, batch, batch_idx):
        context, target, _ = batch
        prediction = self(context)
        loss, l1, mse, ssim_loss, ssim_val = self.loss_fn(prediction, target)
        self._log_metrics('val', loss, l1, mse, ssim_loss, ssim_val)
        if batch_idx == 0 and self.global_rank == 0:
            self.visualize_prediction(context, target, prediction)

    def test_step(self, batch, batch_idx):
        context, target, _ = batch
        prediction = self(context)
        loss, l1, mse, ssim_loss, ssim_val = self.loss_fn(prediction, target)
        self._log_metrics('test', loss, l1, mse, ssim_loss, ssim_val)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.training_params['learning_rate'])
    
    def visualize_prediction(self, context, target, prediction):
        context = context[0].cpu().detach()
        target = target[0].cpu().detach()
        prediction = prediction[0].cpu().detach()
        
        num_context_frames = context.shape[0]
        
        # 固定布局：3行×5列，前两行用于上下文（共10个位置），第三行用于预测
        total_context_positions = 10
        
        # 创建前两行的可视化内容
        if num_context_frames >= total_context_positions:
            # 上下文帧够填满前两行，取最后10帧
            context_viz = context[-total_context_positions:]
        else:
            # 上下文帧不够，前面用白色填充
            white_padding = torch.ones(total_context_positions - num_context_frames, *context.shape[1:])
            context_viz = torch.cat([white_padding, context], dim=0)

        viz_tensor = torch.cat([context_viz, prediction], dim=0)
        grid = make_grid(viz_tensor, nrow=5, padding=2, pad_value=1)
        
        if self.logger:
            save_dir = os.path.join(os.path.dirname(self.logger.log_dir), "val_png")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"epoch_{self.current_epoch}_step_{self.global_step}.png")
            save_image(grid, save_path)

def main():
    parser = argparse.ArgumentParser(description="Train ConvLSTM_UNet Model")
    parser.add_argument('--config', type=str, default='/Data/hgjia/scr/RadioMotion/config.yaml', help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    pl.seed_everything(config['seed'], workers=True)

    data_module = pl.LightningDataModule()
    data_module.train_dataloader = lambda: DataLoader(UrbanRadioVideoDataset(phase="train", data_config=config['data']), **config['data']['loader_params'])
    data_module.val_dataloader = lambda: DataLoader(UrbanRadioVideoDataset(phase="val", data_config=config['data']), **config['data']['loader_params'])
    data_module.test_dataloader = lambda: DataLoader(UrbanRadioVideoDataset(phase="test", data_config=config['data']), **config['data']['loader_params'])

    model = LightningConvLSTM(model_params=config['model'], training_params=config['training'])

    callbacks = [
        ModelCheckpoint(**config['callbacks']['checkpoint']),
        EarlyStopping(**config['callbacks']['early_stopping'])
    ]

    trainer = pl.Trainer(**config['trainer_config'], callbacks=callbacks, logger=pl.loggers.TensorBoardLogger(**config['logging']['tensorboard']))
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()