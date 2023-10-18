import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_
import math
from configs.config_setting import setting_config
import numpy as np
import random

"""
Basenet privided for selecting
"""

"""
SimpleNet for testing the validation
"""
seed = setting_config.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class SimpleNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleNet, self).__init__()
        
        # 编码器（下采样部分）
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 解码器（上采样部分）
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

"""
U-Net structure
"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # 编码器部分
        self.conv1 = DoubleConv(in_channels, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   #64
        self.conv2 = DoubleConv(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)   #32
        self.conv3 = DoubleConv(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)   #16
        self.conv4 = DoubleConv(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)   #8
        
        # 解码器部分
        self.conv5 = DoubleConv(512, 1024)
        self.upconv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)   #16
        self.conv6 = DoubleConv(1024, 512)
        self.upconv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)    #32
        self.conv7 = DoubleConv(512, 256)
        self.upconv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)    #64
        self.conv8 = DoubleConv(256, 128)
        self.upconv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)     #128 feature map size
        self.conv9 = DoubleConv(64, 64)
        
        # 输出层
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # 编码器
        c1 = self.conv1(x)
        p1 = self.maxpool1(c1)
        c2 = self.conv2(p1)
        p2 = self.maxpool2(c2)
        c3 = self.conv3(p2)
        p3 = self.maxpool3(c3)
        c4 = self.conv4(p3)
        p4 = self.maxpool4(c4)
        
        # 解码器
        u5 = self.conv5(p4)
        u5 = self.upconv6(u5)
        u5 = torch.cat((u5, c4), dim=1)
        c6 = self.conv6(u5)
        u6 = self.upconv7(c6)
        u6 = torch.cat((u6, c3), dim=1)
        c7 = self.conv7(u6)
        u7 = self.upconv8(c7)
        u7 = torch.cat((u7, c2), dim=1)
        c8 = self.conv8(u7)
        u8 = self.upconv9(c8)
        c9 = self.conv9(u8)
        
        # 输出层
        output = self.output(c9)
        return output
    
"""
U-Net++ structure
"""
class UnetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        skip_connection = x
        x = self.pool(x)
        return x, skip_connection

# 定义Unet++的Decoder部分
class UnetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        # skip_connection = F.interpolate(skip_connection, size=x.size()[2:], mode='bilinear')
        x = torch.cat((x, skip_connection), dim=1)  # x.dimension + skip_connection = in_channels.dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# 定义Unet++模型
class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetPlusPlus, self).__init__()
        self.encoder1 = UnetEncoder(in_channels, 64)
        self.encoder2 = UnetEncoder(64, 128)
        self.encoder3 = UnetEncoder(128, 256)
        self.encoder4 = UnetEncoder(256, 512)
        
        self.middleconv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.middleconv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        
        self.decoder4 = UnetDecoder(1024, 512)
        self.decoder3 = UnetDecoder(512, 256)
        self.decoder2 = UnetDecoder(256, 128)
        self.decoder1 = UnetDecoder(128, 64)
        
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_connection1 = self.encoder1(x)
        x, skip_connection2 = self.encoder2(x)
        x, skip_connection3 = self.encoder3(x)
        x, skip_connection4 = self.encoder4(x)  # 最后一层Encoder不需要skip connection

        x = F.relu(self.middleconv1(x))
        x = F.relu(self.middleconv2(x))
        
        x = self.decoder4(x, skip_connection4)  #x:1024->512, skip_connection4:512
        x = self.decoder3(x, skip_connection3)  #x:512->256, skip_connection3:256
        x = self.decoder2(x, skip_connection2)  #x:256->128, skip_connection2:128
        x = self.decoder1(x, skip_connection1)  #x:128->64, skip_connection1:64
        
        x = self.output(x)
        # x = torch.sigmoid(x)  # 使用sigmoid激活函数输出分割结果
        
        return x
