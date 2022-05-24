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

from network import DenseCorrespondenceNetwork
from torchvision import transforms
from torch.autograd import Variable
import plotting as dc_plotting
from plotting import normalize_descriptor
from scipy import misc, ndimage
# Ignore warnings
import warnings

print(torch)
print(torch.__version__)
import cv2
import yaml

warnings.filterwarnings("ignore")
DES_MEAN = [0.06185, -0.2538, -0.0283]
DES_STD = [0.9313, 0.6393, 0.5042]

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

unit = math.pi / 180


class DON():

    def __init__(self):
        self.setupNetwork()
        self.transformDes = transforms.Compose([transforms.ToTensor()])  # , norm_transform_des])
        norm_transform_img = transforms.Normalize(IMG_MEAN, IMG_STD)
        self.transformImg = transforms.Compose([transforms.ToTensor(), norm_transform_img])

    def setupNetwork(self, config=None):
        config = dict()
        config["backbone"] = dict()
        config["backbone"]["model_class"] = "Resnet"
        config["backbone"]["resnet_name"] = "Resnet34_8s"
        config['descriptor_dimension'] = 3
        nIteration = 3500
        nDimesion = config['descriptor_dimension']
        config["path_to_network_params_folder"] = ""
        # config["model_param_filename_tail"] = "../../pdc/trained_models/tutorials/ORIGINAL_3/%06d.pth"%(nIteration)
        config["model_param_filename_tail"] = "../../pdc/trained_models/tutorials/MULTIPLE_SYN__RGB3/%06d.pth" % (
            nIteration)
        # config["model_param_filename_tail"] = "../../pdc/trained_models/tutorials/SYNTHETIC_3/%06d.pth"%(nIteration)
        config['image_width'] = '640'
        config['image_height'] = '480'
        normalize = False

        self.fcn = DenseCorrespondenceNetwork.get_fcn(config)

        self.dcn = DenseCorrespondenceNetwork(self.fcn, config['descriptor_dimension'],
                                              image_width=config['image_width'],
                                              image_height=config['image_height'],
                                              normalize=normalize)

        self.dcn.load_state_dict(torch.load(config["model_param_filename_tail"]))
        self.dcn.cuda()
        self.dcn.eval()

        self.network_name = 'SYNTHETIC_' + str(nDimesion)

    def getDescriptor(self, rgb):
        rgb_1_tensor = self.transformImg(rgb)
        des = self.dcn.forward_single_image_tensor(rgb_1_tensor).data.cpu().numpy()

        return des

    def getBestMatchPoint(self, pixel_in_target, refDescriptor, currentDescriptor, maskSource=None):
        u, v = pixel_in_target[0], pixel_in_target[1]
        best_match_uv, best_match_diff, norm_diffs = self.dcn.find_best_match_only((u, v), refDescriptor,
                                                                                   currentDescriptor, maskSource)
        # print ("best_match_uv",best_match_uv)
        return best_match_uv, best_match_diff

    def getBestMatchArea(self, pixel_in_target, refDescriptor, currentDescriptor, maskSource=None, delta=2):
        u, v = pixel_in_target[0], pixel_in_target[1]
        listBestMatchPoint = self.dcn.find_area_best_match((u, v), refDescriptor, currentDescriptor, maskSource, delta)
        return np.array(listBestMatchPoint)