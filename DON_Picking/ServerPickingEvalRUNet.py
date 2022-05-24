import sys
import logging
from numpysocket import NumpySocket
sys.path.insert(1, '..')
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
#from mlsocket import MLSocket
from random import randint

import os

os.environ['OMP_NUM_THREADS'] = '1'
CUDA_LAUNCH_BLOCKING = 1
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

from torch.utils.tensorboard import SummaryWriter

from DON_Picking.ResUNet4 import RUNet
from DON_Training.DensObjectNet import DensObjectNet
from ITRIP.Configuration import *
import cv2

print(torch.__version__)


def get_args():
    parser = argparse.ArgumentParser(
        """PPO for CGIPacking""")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--DON_mode", type=str, default="RGBD")
    parser.add_argument("--DON_pretrained", type=str, default="DON_GraspNet_RGBD_Resnet8/DON_5001")

    parser.add_argument('--gamma', type=float, default=0.3, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.001, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    # parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=3)
    # parser.add_argument("--num_local_steps", type=int, default=128)
    parser.add_argument("--num_global_steps", type=int, default=1e6)
    # parser.add_argument("--num_env", type=int, default=3)
    parser.add_argument("--save_interval", type=int, default=10, help="Number of steps between savings")
    parser.add_argument("--log_path", type=str, default="tensorboard/Server")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--setting", type=str, default="Server_GraspNet_Picking_tutorial")
    parser.add_argument("--pretrained", type=str, default="None")
    parser.add_argument("--comment", type=str, default="comment Model configuration")
    parser.add_argument("--port", type=int, default=1299)
    args = parser.parse_args()
    return args


def FlattenMask(mask):
    MaxPool = nn.MaxPool2d(2, 2)
    flatten = MaxPool(mask)
    flatten = flatten.view(mask.size(0), config["HalfWidth"] * config["HalfWidth"])

    return flatten


def setup_tensorboard(tensorboard_log_dir, comment="no comment"):
    shutil.rmtree(tensorboard_log_dir, ignore_errors=True)
    # if not os.path.isdir(tensorboard_log_dir):
    os.makedirs(tensorboard_log_dir)
    # else

    print("Save path:", (tensorboard_log_dir))

    cmd = "tensorboard --logdir=%s" % (tensorboard_log_dir)
    # logging.info("tensorboard logger started")
    print(cmd)
    writer = SummaryWriter(tensorboard_log_dir, comment)
    return writer


def train(opt):
#    obs_trans = ObsTransformer()
    HOST = '10.8.4.104'
    PORT = opt.port
    print(PORT)
    npSocket =  NumpySocket()
    if (True):
        if torch.cuda.is_available():
            print("Use GPU")
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        targetModel = RUNet(opt.DON_mode, 1, DON_pretrained=opt.DON_pretrained)

        if torch.cuda.is_available():
            targetModel.cuda(0)
            targetModel.eval()

        if (opt.pretrained != "None"):
            directory = opt.saved_path + "/" + opt.pretrained
            print("Using Pretrained Model", directory)
            targetModel.load_state_dict(torch.load(directory))
            targetModel.eval()

        targetModel.share_memory()
        targetModel = targetModel.float()

        print("Done setup")
        print("Wating for conenction")
        npSocket.startServer(PORT)
        print ("connected from",npSocket.client_address)
        if (True):
            # recive first data for configuration
            data = npSocket.recieve(64)
            print (data)
            #npSocket.send(np.array([-1]))
            # data = np.array([128,10,16])
            num_local_steps, num_env, batch_size = data[0], data[1], data[2]
            num_local_steps += 1
            print (data.shape)

            nEpisode = 1024 * 4 * 1000 // (num_env) // num_local_steps

            for step2M in range(12):  # for each 4M timestep. Total 48M
                for epi in tqdm(range(nEpisode)):

                    for currentStep in tqdm(range(num_local_steps)):
                        data = npSocket.recieve()  # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
                        # data = np.random.rand(3, 256, 256, 6)
                        # print (data.shape)
                        lastInfo = data[:, 0, :4, -1]
                        lastReward = lastInfo[:, 0]
                        lastDone = lastInfo[:, 1]

                        listRGB = data[:, :, :, :3]
                        listDepth = data[:, :, :, 3]
                        listMask = data[:, :, :, 4]
                        listAddData = data[:, 0, 2, -1]

                        listRGB = torch.from_numpy(np.array(listRGB))
                        listDepth = torch.from_numpy(np.array(listDepth))
                        listMask = torch.from_numpy(np.array(listMask)).float()
                        flattenMask = FlattenMask(listMask).cuda(0)
                        listDepth = listDepth.unsqueeze(-1)
                        listMask = listMask.unsqueeze(-1)
                        listAddData = torch.from_numpy(np.array(listAddData))
                        # = listMask.unsqueeze(-1)
                        tensorListRGB = listRGB.permute((0, 3, 1, 2))
                        tensorListRGB = tensorListRGB.float().cuda(0)
                        tensorListDepth = listDepth.permute((0, 3, 1, 2))
                        tensorListDepth = tensorListDepth.float().cuda(0)
                        tensorListMask = listMask.permute((0, 3, 1, 2))
                        tensorListMask = tensorListMask.float().cuda()
                        tensorAddData = listAddData.float().cuda()

                        # process / concat current obs, get input picking model
                        # tensorListRGB,tensorListDepth = obs_trans.transform(tensorListRGB,tensorListDepth )
                        # tensorListDepth = torch.cat([tensorListDepth,tensorListRGB], dim=1)
                        # tensorListDepth = torch.cat([tensorListRGB,tensorListDepth], dim=1)

                        # gothough picking model
                        returnPolicy, returnValue = targetModel(tensorListRGB, tensorListDepth)
                        returnPolicy = returnPolicy.detach()
                        value = returnValue.detach()
                        # flattenMask= FlattenMask(listMask).cuda(0)
                        num_env = data.shape[0]
                        policy = F.softmax(returnPolicy.view(num_env, config["HalfWidth"] * config["HalfWidth"]),
                                           dim=1) * flattenMask

                        # process/sample action
                        old_m = Categorical(policy)
                        action = old_m.sample()
                        action = action.detach().cpu().numpy()
                        npSocket.send(action)



if __name__ == "__main__":
    opt = get_args()

    if os.path.isdir(opt.log_path + "/" + opt.setting):
        shutil.rmtree(opt.log_path + "/" + opt.setting)
    os.makedirs(opt.log_path + "/" + opt.setting)

    train(opt)
