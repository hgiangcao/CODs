import sys
sys.path.insert(1, '..')
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


from random import randint

import os
os.environ['OMP_NUM_THREADS'] = '1'
CUDA_LAUNCH_BLOCKING=1
import argparse
import torch
import sys
sys.path.insert(1, '..')
from tqdm import tqdm

# from process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil
from torch import nn

from DON_Training.DensObjectNet import DensObjectNet
from DON_Picking.ResUNet4 import RUNet
from ITRIP.Configuration import *
import cv2
print(torch.__version__)
print (torch.cuda.get_device_name(0))

def FlattenMask (mask):
    MaxPool = nn.MaxPool2d(2, 2)
    flatten = MaxPool(mask)
    flatten = flatten.view(mask.size(0),240*320)
    return flatten


class DON_Picking_API():
    def __init__(self):
        
        setting = "RGBD"

        #pickingPretrainedModelPath = "../DON_Picking/trained_models/Picking_ITRI_ResNet4_SRCODs/Picking_100" # no normalize
        pickingPretrainedModelPath =  "../DON_Picking/trained_models/Server_GraspNet_Picking_tutorial/Picking_50"
        self.pickingModel = RUNet("RGBD", 1)
        self.pickingModel.load_state_dict(torch.load(pickingPretrainedModelPath, map_location='cuda:0'))
        self.pickingModel.eval()
        self.pickingModel.float()
        self.pickingModel.cuda()
        print ("Done load ",pickingPretrainedModelPath)

    def getAction(self,tensorRGB,tensorDepth,tensorMask,addData = 0):

        #tensorRGB   = torch.from_numpy(np.array(rgb)).unsqueeze(0).cuda()
        #tensorRGB = tensorRGB.permute((0, 3, 1, 2))
        #tensorRGB = 
        tensorRGB = tensorRGB.float().cuda()
        tensorDepth = tensorDepth.float().cuda()
        # tensorDepth = tensorDepth.permute((0, 3, 1, 2))
        
        tensorMask  = torch.from_numpy(np.array(tensorMask)).unsqueeze(0).cuda()
        #print (tensorMask.shape)
        #tensorMask = tensorMask.permute((0, 3, 1, 2))
        
        #tensorAddData = torch.from_numpy(np.array(addData)).float().unsqueeze(0).cuda()

        #tensorDepth = tensorDepth.permute((0, 3, 1, 2))
        
        flattenMask = FlattenMask(tensorMask).cuda()
        flattenDepth = FlattenMask(tensorDepth).cuda()


        #inputModel = torch.cat((tensorRGB,tensorDepth),dim=1)
        
        with torch.no_grad():
            returnPolicy, _,des =  self.pickingModel(tensorRGB,tensorDepth, isEvaluate = True)
        returnPolicy = returnPolicy.detach()
        
        #print (torch.sum(returnPolicy))
        
        policy = F.softmax(returnPolicy.view(1,240*320),
                           dim=1) * flattenMask

        action = torch.argmax(policy).cpu().detach().item() 
        policy = policy.reshape((240*320))
        #meanPolicy = torch.mean()
        #old_m = Categorical(policy)
        #action = old_m.sample()

        x, y = action //320, action % 320
       
        x, y = x * 2, y * 2


        return x,y,des,policy

    def getPolicy(self,tensorDepth,tensorRGB):

        tensorDepth = tensorDepth.cuda()
        tensorRGB = tensorRGB.cuda()


        inputModel = torch.cat((tensorDepth,tensorRGB),dim=1)
        
        with torch.no_grad():
            returnPolicy, _ =  self.pickingModel(inputModel, isEvaluate = True)
        returnPolicy = returnPolicy.detach()
         
        return returnPolicy

    def getOutput(self,inputModel):
        print (inputModel.shape)
        inputModel = inputModel.float().cuda()

        
        with torch.no_grad():
            returnPolicy, _ =  self.pickingModel(inputModel, isEvaluate = True)
        returnPolicy = returnPolicy.detach()
        
        print (torch.sum(returnPolicy))
        
        #policy = F.softmax(returnPolicy.view(1,128,128),
        #                   dim=1) * flattenMask

        returnPolicy =nn.Upsample(scale_factor=2)(returnPolicy)


        return returnPolicy

if __name__ == "__main__":
    torch.manual_seed(1111)
    pickingAPI = DON_Picking_API()
    
    observation = np.load("23_reg_light/obs.npy" ,allow_pickle=True)
    print (observation.shape)


    mask1 = np.ones((240,320))*1.0
    mask2 = np.ones((240,320))*1.0

    idx1= 20

    rgb_raw = observation[idx1,0][80:560,80:560,:]
    depth_raw = observation[idx1,1][80:560,80:560]

    rgb_raw = cv2.resize(rgb_raw,(256,256))
    depth_raw = cv2.resize(depth_raw,(256,256))


    print (rgb_raw.shape)
    print (depth_raw.shape)


    mask = np.ones((256,256,1))*1.0
    rgb = rgb_raw
    depth = depth_raw.reshape(256,256,1)

    addData = 0/30


    x,y = pickingAPI.getAction(rgb,depth,mask,addData)


    print (x,y)

    while True:
        cv2.circle(rgb, (y,x), 5, (0,255,0), 1)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('video', rgb)

    cv2.destroyAllWindows()


