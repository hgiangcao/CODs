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

#from CODs_Picking.VPG_net import RUNet,ObsTransformer
from CODs_Picking.ResUNet4 import RUNet
from ITRIP.Configuration import *
import cv2
from CODs_Training.DensObjectNet import DensObjectNet

print(torch.__version__)


def get_args():
    parser = argparse.ArgumentParser(
        """PPO for CGIPacking""")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--CODs_mode", type=str, default="RGBD")
    parser.add_argument("--CODs_pretrained", type=str, default="CODs_GraspNet_RGBD_Resnet8/CODs_10001")

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
    #obs_trans = ObsTransformer()
    #HOST = '10.8.4.104'
    PORT = opt.port
    print(PORT)
    npSocket =  NumpySocket()
    if (True):

        if torch.cuda.is_available():
            print("Use GPU")
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        if not os.path.isdir(opt.saved_path + "/" + opt.setting):
            os.makedirs(opt.saved_path + "/" + opt.setting)
            print("Save path:", (opt.saved_path + "/" + opt.setting))

        tensorboard = setup_tensorboard(opt.log_path + "/" + opt.setting)
        print("Log path:", (opt.log_path + "/" + opt.setting))

        # model = U_NET(inputChannel=9)

        trainModel = RUNet(opt.CODs_mode, 1,CODs_pretrained=opt.CODs_pretrained)
        targetModel = RUNet(opt.CODs_mode, 1,CODs_pretrained=opt.CODs_pretrained)
        if torch.cuda.is_available():
            targetModel.cuda(0)
            targetModel.eval()
            trainModel.cuda(0)

        if (opt.pretrained != "None"):
            directory = opt.saved_path + "/" + opt.pretrained
            print("Using Pretrained Model", directory)
            targetModel.load_state_dict(torch.load(directory))
            targetModel.eval()
            trainModel.load_state_dict(targetModel.state_dict())

        targetModel.share_memory()
        targetModel = targetModel.float()
        trainModel.share_memory()
        trainModel = trainModel.float()

        optimizer = torch.optim.Adam(trainModel.parameters(), lr=opt.lr)

        print("CODse setup")
        print("Wating for conenction")
        #s.bind((HOST, PORT))
        #s.listen()
        #conn, address = s.accept()
        #print("Connected from", address)
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
            #print("config", num_local_steps, num_env, batch_size)i
            print (data.shape)
            # last CODst use info
            # have to ensure that these value are never used
            old_log_policy = None
            lastReward = -1
            lastCODse = True
            lastState = None
            lastDes = None
            lastObs = None
            lastOld_log_policy = None
            lastFlattenMask = None
            lastValue = None

            nEpisode = 1024 * 4 * 1000 // (num_env) // num_local_steps
            num_epochs = 3
            curr_episode = 0
            totalTimeStep = 0

            for step2M in range(12):  # for each 4M timestep. Total 48M
                for epi in tqdm(range(nEpisode)):
                    if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                        print("Save model ", "{}/Picking_{}".format(opt.saved_path + "/" + opt.setting, curr_episode))
                        torch.save(trainModel.state_dict(),
                                   "{}/Picking_{}".format(opt.saved_path + "/" + opt.setting, curr_episode))
                    curr_episode += 1
                    torch.save(trainModel.state_dict(),
                               "{}/Picking_{}".format(opt.saved_path + "/" + opt.setting, "lastest_model"))
                    #        print (curr_episode)
                    # log
                    old_log_policies = []
                    actions = []
                    values = []
                    #states = []
                    obs = []
                    depths = []
                    rewards = []
                    maskes = []
                    CODses = []
                    addDatas = []

                    lastObs = None
                    #lastDes = None
                    lastAction = None
                    lastOld_log_policies = None
                    lastValue = None
                    lastAddData = None
                    lastDepth = None

                    tableinfo = np.zeros((5, 256, 256, 1))
                    tableinfo[:, 0, 2, 0] = -1  # inactive

                    for currentStep in tqdm(range(num_local_steps)):
                        data = npSocket.recieve()  # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
                        # data = np.random.rand(3, 256, 256, 6)
                        # print (data.shape)
                        lastInfo = data[:, 0, :4, -1]
                        lastReward = lastInfo[:, 0]
                        lastCODse = lastInfo[:, 1]

                        lastReward = torch.FloatTensor(lastReward).cuda(0)
                        lastCODse = torch.FloatTensor(lastCODse).cuda(0)

                        # append last state: old_log_policies || actions || values || states || obs  || rewards  || maskes || CODses

                        old_log_policies.append(lastOld_log_policies)
                        actions.append(lastAction)
                        values.append(lastValue)
                        #append(lastDes)
                        obs.append(lastObs)
                        depths.append(lastDepth)
                        rewards.append(lastReward)
                        maskes.append(lastFlattenMask)
                        CODses.append(lastCODse)
                        addDatas.append(lastAddData)

                        # print (lastInfo.shape) # should be env,1,3,1. Convert to env,3

                        # inputModel = data[:,:,:-1] # all layer, but the last one
                        # go thourh CODs
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
                        #tensorListRGB,tensorListDepth = obs_trans.transform(tensorListRGB,tensorListDepth )
                        #tensorListDepth = torch.cat([tensorListDepth,tensorListRGB], dim=1)
                        #tensorListDepth = torch.cat([tensorListRGB,tensorListDepth], dim=1)

                        # gothough picking model
                        returnPolicy, returnValue = targetModel(tensorListRGB,tensorListDepth)
                        returnPolicy = returnPolicy.detach()
                        value = returnValue.detach()
                        # flattenMask= FlattenMask(listMask).cuda(0)
                        num_env = data.shape[0]
                        policy = F.softmax(returnPolicy.view(num_env, config["HalfWidth"] * config["HalfWidth"]),
                                           dim=1) * flattenMask

                        # process/sample action
                        old_m = Categorical(policy)
                        action = old_m.sample()
                        # action = torch.cuda.IntTensor(realAction)
                        old_log_policy = old_m.log_prob(action)
                        # save last state/action
                        lastAction = action
                        lastFlattenMask = flattenMask
                        lastOld_log_policies = old_log_policy
                        lastValue = value.squeeze()
                        lastObs = tensorListRGB
                        lastDepth = tensorListDepth
                        lastAddData = tensorAddData

                        # send action
                        action = action.detach().cpu().numpy()
                        npSocket.send(action)

                    # DO TRAINING HERE
                    data = npSocket.recieve()  # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
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
                    tensorListMask = tensorListMask.float().cuda(0)
                    tensorAddData = listAddData.float().cuda(0)
                   
                    #tensorListDepth = torch.cat([tensorListDepth,tensorListRGB], dim=1)
                    #tensorListRGB,tensorListDepth = obs_trans.transform(tensorListRGB,tensorListDepth)
                    #tensorListDepth = torch.cat([tensorListRGB,tensorListDepth], dim=1)


                    returnPolicy, returnValue = targetModel(tensorListRGB,tensorListDepth)
                    returnPolicy = returnPolicy.detach()
                    value = returnValue.detach()

                    totalTimeStep += num_local_steps * num_env
                    next_value = value.squeeze()
                    # TRAINING
                    gae = 0
                    R = []

                    old_log_policies.pop(0)
                    actions.pop(0)
                    values.pop(0)
                    #states.pop(0)
                    obs.pop(0)
                    depths.pop(0)
                    rewards.pop(0)
                    maskes.pop(0)
                    CODses.pop(0)
                    addDatas.pop(0)

                    # print (len(CODses))

                    stackedData = list(zip(values, rewards, CODses))[::-1]
                    for value, reward, CODse in stackedData:
                        gae = gae * opt.gamma * opt.tau
                        gae = gae + reward + opt.gamma * next_value.detach() * (1 - CODse) - value.detach()
                        next_value = value
                        R.append(gae + value)
                        gae = torch.mul(gae, (1 - CODse))
                    R = R[::-1]
                    R = torch.cat(R).detach()

                    values = torch.cat(values)
                    next_value = next_value
                    old_log_policies = torch.cat(old_log_policies)
                    actions = torch.cat(actions)
                    #states = torch.cat(states)
                    maskes = torch.cat(maskes)
                    obs = torch.cat(obs)
                    depths = torch.cat(depths)
                    addDatas = torch.cat(addDatas)
                    advantages = R - values

                    tensorboard.add_scalar(f"Log_Debug/Value", torch.mean(values).item(), totalTimeStep)
                    tensorboard.add_scalar(f"Log_Debug/R", torch.mean(R).item(), totalTimeStep)

                    log_policy_loss = 0
                    add_scalar_loss = 0
                    log_entropy_loss = 0
                    log_total_loss = 0

                    numOfBatchSize = (num_local_steps * num_env) // batch_size
                    for i in range(num_epochs):
                        indice = torch.randperm((num_local_steps - 1) * num_env)
                        for j in range(numOfBatchSize):
                            batch_indices = indice[(j * batch_size): ((j + 1) * batch_size)]

                            logits, value = trainModel( obs[batch_indices], depths[batch_indices])#.cuda(1))
                            # print (maskes[batch_indices].cuda(1).shape)
                            #logits, value = trainModel(obs= obs[batch_indices].cuda(1),  addData = addDatas[batch_indices].cuda(1))
                            logits = F.softmax(logits.view(batch_size, config["HalfWidth"] * config["HalfWidth"]),
                                               dim=1) * \
                                     maskes[batch_indices]

                            new_m = Categorical(logits)
                            new_log_policy = new_m.log_prob(actions[batch_indices])
                            ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])

                            actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                               torch.clamp(ratio, 1.0 - opt.epsilon,
                                                                           1.0 + opt.epsilon) *
                                                               advantages[batch_indices]))

                            critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                            entropy_loss = torch.mean(new_m.entropy())
                            total_loss = actor_loss + critic_loss - opt.beta * entropy_loss

                            log_policy_loss += actor_loss.item()
                            add_scalar_loss += critic_loss.item()
                            log_entropy_loss += entropy_loss.item()
                            log_total_loss += total_loss.item()

                            optimizer.zero_grad()
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(trainModel.parameters(), 0.5)  # REMOVE BY CHGIANG
                            optimizer.step()

                    targetModel.load_state_dict(trainModel.state_dict())
                    targetModel.eval()

                    log_policy_loss /= (opt.num_epochs * numOfBatchSize)
                    add_scalar_loss /= (opt.num_epochs * numOfBatchSize)
                    log_entropy_loss /= (opt.num_epochs * numOfBatchSize)
                    log_total_loss /= (opt.num_epochs * numOfBatchSize)
                    tensorboard.add_scalar(f"Log_Loss/log_policy_loss", log_policy_loss, totalTimeStep)
                    tensorboard.add_scalar(f"Log_Loss/log_value_loss", add_scalar_loss, totalTimeStep)
                    tensorboard.add_scalar(f"Log_Loss/log_entropy_loss", log_entropy_loss, totalTimeStep)
                    tensorboard.add_scalar(f"Log_Loss/log_total_loss", log_total_loss, totalTimeStep)
                    # print("CODse to training")
                    # print("send last action")

                    # action = action.detach().cpu().numpy()

                    npSocket.send(action)
                    torch.cuda.empty_cache()
                    del old_log_policies
                    del actions
                    del values
#del states
                    del obs
                    del rewards
                    del maskes
                    del CODses
                    del logits
                    del value
                    del new_m
                    del new_log_policy
                    del addDatas
                    del actor_loss
                    del critic_loss
                    del entropy_loss
                    del total_loss


if __name__ == "__main__":
    opt = get_args()

    if os.path.isdir(opt.log_path + "/" + opt.setting):
        shutil.rmtree(opt.log_path + "/" + opt.setting)
    os.makedirs(opt.log_path + "/" + opt.setting)

    train(opt)
