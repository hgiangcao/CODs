from __future__ import print_function, division
import random
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw, ImageFont
import sys
import math
# NETWORK
import torch
import sys
import torch.nn as nn
from DON_Training.dense_correspondence_network import DenseCorrespondenceNetwork

from torchvision import transforms
from torch.autograd import Variable

from scipy import misc, ndimage
# Ignore warnings
import warnings

print(torch)
print(torch.__version__)
# import cv2
import yaml

warnings.filterwarnings("ignore")
DES_MEAN = [0.06185, -0.2538, -0.0283]
DES_STD = [0.9313, 0.6393, 0.5042]

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

unit = math.pi / 180


class DensObjectNet():

    def __init__(self, setting="RGB",pretrained=None):


        self.transformDes = transforms.Compose([transforms.ToTensor()])  # , norm_transform_des])
        norm_transform_img = transforms.Normalize(IMG_MEAN, IMG_STD)
        # self.transformImg = transforms.Compose([transforms.ToTensor(), norm_transform_img])
        self.transformImg = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            ),
                        ])
        self.transformImgTensor = transforms.Compose([
                            #transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            ),
                        ])
        self.depth_norm = nn.InstanceNorm2d(1, affine=False)
        self.setting = setting
        self.pretrained = pretrained
        self.setupNetwork()

    def resetNetwork(self):
        self.dcn = DenseCorrespondenceNetwork.from_config(setting = self.setting)
        #print ("Reset")

    def setupNetwork(self, config=None):

        self.dcn = DenseCorrespondenceNetwork.from_config(setting = self.setting)

        if (self.pretrained is not None):
            print ("Load pretrained",self.pretrained)
            self.dcn.load_state_dict(torch.load("../DON_Training/trained_models/"+ self.pretrained))
        else:
            self.dcn.load_state_dict(torch.load("../DON_Training/trained_models/"+ "RGBD_8_Multi_New/DON_15600"))

        self.dcn.cuda()
        #self.dcn.eval()

        self.dcn = self.dcn.float()
        self.dcn.share_memory()

        #print ("Done setting DON",self.pretrained)

    def getDescriptor(self, rgb,depth, mask=None):
        # if (mask is not None and depth is not None):
        #	rgb_1_tensor = self.processImage(rgb,mask,depth)

        if (not torch.is_tensor(rgb)):
            rgb = self.transformImg(rgb).float()

        if (not torch.is_tensor(depth)):
            depth = transforms.ToTensor()(depth).float()
        #rgb_1_tensor = transforms.ToTensor()(rgb).float()
        #rgb_1_tensor = Variable(img_a.cuda()
        des = self.dcn.forward_single_image_tensor(rgb,depth)#.data.cpu().numpy()

        return des
    '''
    def getBatchDescriptor(self, rgb_1_tensor, mask=None, depth=None):
        # if (mask is not None and depth is not None):

        des = self.dcn.foward_batch(rgb_1_tensor)#.data.cpu().numpy()

        return des
    '''
    def getBatchLayer(self, rgb_1_tensor,depth):
        # if (mask is not None and depth is not None):
        #print (depth.shape)

        for i in range (rgb_1_tensor.shape[0]):
          rgb_1_tensor[i] = self.transformImgTensor(rgb_1_tensor[i])

        
        #print (rgb_1_tensor.shape)
        depth = self.depth_norm(depth)
        rgb_1_tensor = torch.cat([rgb_1_tensor, depth], dim=1)
        
        #print (rgb_1_tensor.shape,depth.shape)

        #l1
        #print (self.dcn._fcn)
        #rgb_1_tensor = self.dcn._fcn.depth_norm(rgb_1_tensor)
        l1 = self.dcn._fcn.resnet34_8s.conv1(rgb_1_tensor)
        l1 = self.dcn._fcn.resnet34_8s.bn1(l1)
        l1 = self.dcn._fcn.resnet34_8s.relu(l1)
        l2 = self.dcn._fcn.resnet34_8s.maxpool(l1)
        l2 = self.dcn._fcn.resnet34_8s.layer1(l2)

        l3 =self.dcn._fcn.resnet34_8s.layer2(l2)
        l4 = self.dcn._fcn.resnet34_8s.layer3(l3)
        l5 = self.dcn._fcn.resnet34_8s.layer4(l4)
        l6 = self.dcn._fcn.resnet34_8s.fc(l5)


        return l1,l2,l3,l4,l5,l6

    def getBestMatchPointOnly(self, pixel_in_target, refDescriptor, currentDescriptor, maskSource=None):
        u, v = pixel_in_target[0], pixel_in_target[1]
        best_match_uv, best_match_diff, norm_diffs = self.dcn.find_best_match_only((u, v), refDescriptor,
                                                                                   currentDescriptor, maskSource)
        # print ("best_match_uv",best_match_uv)
        return best_match_uv, best_match_diff.item()

    def getBestMatchPoint(self, pixel_in_target, refDescriptor, currentDescriptor, maskSource=None):
        u, v = pixel_in_target[0], pixel_in_target[1]
        best_match_uv, best_match_diff, norm_diffs = self.dcn.find_best_match((u, v), refDescriptor, currentDescriptor,
                                                                              maskSource)
        # print ("best_match_uv",best_match_uv)
        return best_match_uv, best_match_diff.item()

    def getBestMatchArea(self, pixel_in_target, refDescriptor, currentDescriptor, maskSource=None, delta=2):
        u, v = pixel_in_target[0], pixel_in_target[1]
        listBestMatchPoint = self.dcn.find_area_best_match((u, v), refDescriptor, currentDescriptor, maskSource, delta)
        return np.array(listBestMatchPoint)


if __name__ == "__main__":
    batch_size = 8
    obs = torch.randn(batch_size, 1, 224, 224)  # .cuda()

    setting = "D"
    DON = DensObjectNet(setting=setting, pretrained="DON_8_D_New_Normalized_D_Resnet/DON_36000")

    output = DON.getBatchDescriptor(obs)

    print (output.shape)
