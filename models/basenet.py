import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_
import math

"""
Basenet privided for selecting
"""

"""
SimpleNet for testing the validation
"""
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