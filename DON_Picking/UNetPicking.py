
import random
import sys
sys.path.insert(1, '..')
from collections import namedtuple,OrderedDict
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from ITRIP.Configuration import config


print ("Using U_Net new picking ")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                          DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class NormalUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, n_classes, bilinear)

#         self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        flatten = x5.reshape(x5.size(0), x5.size(1)*x5.size(2)*x5.size(3))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #         logits = self.outc(x)
        return x,flatten


class SucModel(nn.Module):

    def __init__(self,modeObsevation, num_cls=1):
        super(SucModel, self).__init__()
        self.depth_norm = nn.InstanceNorm2d(1, affine=False)

        self.maxpool = nn.MaxPool2d(2)

        #self.DON_trunk = UNet(8, 16)
        self.obs_trunk = UNet(config["inputChannel"][modeObsevation], 16)

        self.head = nn.Sequential(
            conv_block(16+8, 32),
            conv_block(32, 64),
            nn.Conv2d(64, num_cls, 1),
        )

        self.valueOut = nn.Sequential(
            nn.Linear(65536+1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)

        )

    def forward(self, des = None, obs=None,addData = 0, isEvaluation  = False):
        input_shape = obs.shape[-2:]

        #depth = self.depth_norm(depth)
        if (obs is None):
          des_feature, des_flatten = self.DON_trunk(des)
          feature = des_feature
        elif (des is None): 
          obs = nn.InstanceNorm2d(1)(obs)
          obs_feature, obs_flatten = self.obs_trunk(obs)
          feature = obs_feature
        else:
          #des_feature, des_flatten = self.DON_trunk(des)
          obs = nn.InstanceNorm2d(1)(obs)
          obs_feature, obs_flatten = self.obs_trunk(obs)
          #feature = torch.cat([des_feature, obs_feature], dim=1)
          feature = torch.cat([des, obs_feature], dim=1)

        pred = self.head(feature)
        #x1Flatten = pred.view(pred.size(0), -1)

        if input_shape != pred[-2:]:
            pred = F.interpolate(pred,
                                 size=input_shape,
                                 mode='bilinear',
                                 align_corners=True)

        value = torch.from_numpy(np.array([0])).cuda()
        if (not isEvaluation):
            addData = addData.view (addData.size(0),-1)
            
            if (des is None):
              x = torch.cat ((obs_flatten,addData),dim=1)
            elif (obs is None):
              x = torch.cat ((des_flatten,addData),dim=1)
            else:
              x = torch.cat ((obs_flatten,addData),dim=1)
            value = self.valueOut(x)

        pred = self.maxpool(pred)
        return pred.squeeze(1),value

if __name__ == "__main__":
    batch_size = 8
    des = torch.randn(batch_size,  8, 256, 256)#.cuda()
    obs = torch.randn(batch_size, 4, 256, 256)  # .cuda()
    addData = torch.randn(batch_size)
    hiNet = SucModel("D",1)
    #hiNet.cuda()
    hiNet.train()

    pytorch_total_params = sum(p.numel() for p in hiNet.parameters() if p.requires_grad)
    print (pytorch_total_params ,"params")
    predPos, value = hiNet(des,obs,addData)
    print ( predPos.shape)
    print(value.squeeze())
