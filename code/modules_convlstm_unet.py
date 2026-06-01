# @File: modules_convlstm_unet.py
# @Description: 采用 ConvLSTM 和 U-Net 结构的视频预测模型。
#               该模型显式地对时空动态进行建模，更适合视频预测任务。

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    基础的 ConvLSTM 单元。
    在每个时间步，它接收一个输入特征图和上一时刻的隐藏/细胞状态，
    并计算输出当前时刻的隐藏/细胞状态。
    """
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # 为四个门（输入、遗忘、输出、细胞门）同时计算
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, states):
        h_prev, c_prev = states
        # 将当前输入 x 和上一时刻的隐藏状态 h_prev 在通道维度上拼接
        combined = torch.cat([x, h_prev], dim=1)
        # 一次卷积计算出所有门
        gates = self.conv(combined)
        
        # 将结果切分成四个门
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        o = torch.sigmoid(o)  # 输出门
        g = torch.tanh(g)     # 细胞门
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))

class ConvLSTM_UNet(nn.Module):
    """
    一个基于 ConvLSTM 和 U-Net 结构的编码器-解码器模型，用于视频预测。
    """
    def __init__(self, in_channels=1, out_channels=5, embed_dim=64):
        super(ConvLSTM_UNet, self).__init__()
        
        self.prediction_frames = out_channels
        
        # 编码器
        self.enc1 = ConvLSTMCell(in_channels, embed_dim, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ConvLSTMCell(embed_dim, embed_dim * 2, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码器
        self.dec2 = ConvLSTMCell(embed_dim * 2, embed_dim * 2, kernel_size=3)
        self.up2 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        
        # 解码器第一层输入是上采样结果和编码器第一层的跳跃连接，所以通道数是 embed_dim + embed_dim
        self.dec1 = ConvLSTMCell(embed_dim * 2, embed_dim, kernel_size=3)
        
        # 输出头
        self.head = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x_context):
        # x_context shape: (B, T_in, C, H, W)
        B, T_in, C, H, W = x_context.shape
        device = x_context.device
        
        # 初始化所有层的隐藏状态
        h1, c1 = self.enc1.init_hidden(B, (H, W), device)
        h2, c2 = self.enc2.init_hidden(B, (H // 2, W // 2), device)
        
        # ---- 编码器: 逐帧处理输入序列 ----
        for t in range(T_in):
            x_t = x_context[:, t, ...]
            h1, c1 = self.enc1(x_t, (h1, c1))
            p1 = self.pool1(h1)
            
            h2, c2 = self.enc2(p1, (h2, c2))
            
        # 编码器最后一帧的 h1 作为跳跃连接
        skip_connection = h1
        
        # ---- 解码器: 逐帧生成预测序列 ----
        predictions = []
        # 解码器的隐藏状态从编码器最后一帧继承
        dec_h2, dec_c2 = h2, c2 
        dec_h1, dec_c1 = h1, c1
        
        for t in range(self.prediction_frames):
            dec_h2, dec_c2 = self.dec2(dec_h2, (dec_h2, dec_c2)) # 自回归
            up_h2 = self.up2(dec_h2)
            
            # 融合跳跃连接
            dec_input1 = torch.cat([up_h2, skip_connection], dim=1)
            dec_h1, dec_c1 = self.dec1(dec_input1, (dec_h1, dec_c1))
            
            pred_frame = self.head(dec_h1)
            predictions.append(pred_frame)

        # 堆叠预测结果
        predictions = torch.stack(predictions, dim=1)  # -> (B, T_out, C, H, W)
        return torch.sigmoid(predictions)