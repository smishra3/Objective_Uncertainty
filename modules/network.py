import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models
#from modules.layers import ConvOffset2D

import math 
import numpy as np


class UNetUp_C(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class UNetUp_T(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class UNetUp16(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 16, stride=16),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class UNetUp8(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 8, stride=8),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class UNetUp4(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class UNetUp2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)



class UNetDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        layers = [
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)

class AUCU(nn.Module):

    def __init__(self, num_classes, num_channels):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(64, 128)
        self.dec3 = UNetDown(128, 256)
        self.dec4 = UNetDown(256, 512)

        self.dec4_up1 = UNetUp2(512,256)
        self.dec4_up2 = UNetUp4(512,128)
        self.dec4_up3 = UNetUp8(512,64)

        self.dec3_up1 = UNetUp2(256,128)
        self.dec3_up2 = UNetUp4(256,64)

        self.dec2_up1 = UNetUp2(128,64)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(512, 1024, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(1024)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(1024, 1024, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(1024)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(512)
        self.center3_relu = nn.ReLU(inplace=True)

        self.dec5_up1 = UNetUp16(1024,64)
        
        self.enc4_1 = UNetUp_C(1024, 512)
        self.enc4_2 = UNetUp_T(512,256)
        self.enc3_1 = UNetUp_C(512, 256)
        self.enc3_2 = UNetUp_T(256, 128)
        self.enc2_1 = UNetUp_C(256, 128)
        self.enc2_2 = UNetUp_T(128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(384, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.enc0_16 = UNetUp16(1024, 2)
        self.enc0_8 = UNetUp8(512, 2)
        self.enc0_4 = UNetUp4(256, 2) 
        self.enc0_2 = UNetUp2(128, 2)        

        self.final = nn.Conv2d(64, num_classes, 1)
        self.final2 = nn.Conv2d(2, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        #print("good!")
        dec1 = self.first(x)
        #print("good!")
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        dec4_up1 = self.dec4_up1(dec4)
        dec4_up2 = self.dec4_up2(dec4)
        dec4_up3 = self.dec4_up3(dec4)

        dec3_up1 = self.dec3_up1(dec3)
        dec3_up2 = self.dec3_up2(dec3)

        dec2_up1 = self.dec2_up1(dec2)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)

        dec5_up1 = self.dec5_up1(center7)

        enc4_1 = self.enc4_1(torch.cat([center10, dec4], 1))
        enc4_2 = self.enc4_2(enc4_1)
        enc3_1 = self.enc3_1(torch.cat([enc4_2, dec3], 1))
        enc3_2 = self.enc3_2(enc3_1)
        enc2_1 = self.enc2_1(torch.cat([enc3_2, dec2], 1))
        enc2_2 = self.enc2_2(enc2_1)
        enc1 = self.enc1(torch.cat([enc2_2, dec1, dec4_up3, dec3_up2, dec2_up1, dec5_up1], 1))
        
        enc0_16 = self.enc0_16(center7)
        enc0_8 = self.enc0_8(enc4_1)
        enc0_4 = self.enc0_4(enc3_1)
        enc0_2 = self.enc0_2(enc2_1) 


        return (self.final(enc1)+self.final2(enc0_2)+self.final2(enc0_4)+self.final2(enc0_8)+self.final2(enc0_16), self.final(enc1))



