
from __future__ import print_function, division
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from DON_Training.DataGenerator_O2O import DataGenerator,loadAllModels
#from ITRIP.objects_new import graspnet_train
from PIL import Image, ImageDraw,ImageFont

import random
from ITRIP.Configuration import *

class DONDataset(Dataset):
    def __init__(self,):
        self.dataGenerator = DataGenerator()
        self.models, self.colors = loadAllModels(path="../DON_data/",loadFile=True)
        print (self.models[0].shape)
        self.dataGenerator.setOriginalModelObjects(self.models, self.colors)
        # self.models_ITRI, self.colors_ITRI = loadAllModels(nObject=13,path="dataDON/ITRI_",loadFile=True)
        # self.dataGenerator.setOriginalModelObjects_ITRI(self.models_ITRI, self.colors_ITRI)
        # print (self.models_ITRI[0].shape)
    def __len__(self):
        return (25*10*87*2)

    def __getitem__(self, idx):
        match_type = SINGLE_OBJECT_WITHIN_SCENE
        scene_type = np.random.choice([3,4]) # multi same/different
        augmentationType =  random.choice([0, 1, 2, 3])        
        pathData =  path="../DON_data/" # random.choice(["data2", "data"])

        imgA,depthA, imgB,depthB,rawData_A,rawData_B, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b, nMatchData,nNonMatchData = self.dataGenerator.generateRandomData(pathToScense=pathData, matchType=SINGLE_OBJECT_WITHIN_SCENE,sceneType = scene_type, augmentationType =augmentationType , debug=False)

        imgA,depthA ,imgB,depthB, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b = imgA.squeeze(0), depthA.squeeze(0), imgB.squeeze(0),  depthB.squeeze(0),matches_a.squeeze(0), matches_b.squeeze(0), masked_non_matches_a.squeeze(0), masked_non_matches_b.squeeze(0), background_non_matches_a.squeeze(0), background_non_matches_b.squeeze(0), blind_non_matches_a.squeeze(0), blind_non_matches_b.squeeze(0)

        return imgA,depthA,imgB, depthB, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b, ( nMatchData+nNonMatchData),match_type,scene_type
