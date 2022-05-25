from mlsocket import MLSocket

from sklearn import svm
import numpy as np

from sys import argv

'''
# Make an ndarray
data =np.random.rand(32,9,256,256)

# Send data
with MLSocket() as s:
    s.connect((HOST, PORT)) # Connect to the port and host
    s.send(data) # After sending the data, it will wait until it receives the reponse from the server
    reciveAction = s.recv(1024)

    #double check
    verifyAction = np.array([ data[1,2,3,4],data[4,3,2,1]])

    if (np.sum(verifyAction-reciveAction) < 0.00001):
        print ("Verified")
        print (verifyAction,reciveAction)

import random
'''
import os
import logging
#logging.basicConfig(level=logging.DEBUG)

os.environ['OMP_NUM_THREADS'] = '1'
CUDA_LAUNCH_BLOCKING = 1
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

import argparse
import torch
import sys

sys.path.insert(1, '..')
from tqdm import tqdm
from numpysocket import NumpySocket
from ITRIP.status_code import StatusCode

from CODs_Picking.MultiSimEnv import MultipleEnvironments
# from process import eval
import torch.multiprocessing as _mp

import numpy as np
import shutil
from ITRIP.Configuration import *
import cv2

from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(
        """Client PPO for Picking Cluttered Objects. Env setup""")

    parser.add_argument("--num_local_steps", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--nObject", type=int, default=5)
    parser.add_argument("--num_global_steps", type=int, default=1e6)
    parser.add_argument("--num_env", type=int, default=2) # at least 2
    parser.add_argument("--log_path", type=str, default="tensorboard/Client")
    parser.add_argument("--setting", type=str, default="Client_GraspNet_Picking_tutorial")
    parser.add_argument("--comment", type=str, default="comment Model configuration")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--host", type=str, default='10.8.4.104')
    parser.add_argument("--evaluation", type=bool, default=False)
    args = parser.parse_args()
    return args


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
    #HOST = '10.8.4.104'
    HOST = "10.5.11.243"#opt.host
    PORT = opt.port


    if (opt.num_env <2):
        print ("ERROR!","num_env should be >=2")
        return

    mp = _mp.get_context("spawn")

    envs = MultipleEnvironments(num_envs=opt.num_env, headless=True,isEvaluation=opt.evaluation)

    tensorboard = setup_tensorboard(opt.log_path + "/" + opt.setting)

    [agent_conn.send(("reset", opt.nObject)) for agent_conn in envs.agent_conns]
    [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_episode = 0

    nEpisode = 1024 * 4 * 1000 // (opt.num_env) // opt.num_local_steps
    nEpisode = 100
    totalTimeStep = 0
    tableInfor = np.zeros((opt.num_env, 256, 256, 1))
    npSocket = NumpySocket()
    npSocket.startClient(HOST, PORT)
    if (True):
        configData = np.array([opt.num_local_steps, opt.num_env, opt.batch_size])
        npSocket.send(configData)

        for step2M in range(1):  # for each 4M timestep. Total 48M
            for epi in tqdm(range(nEpisode)):
                curr_episode += 1

                # log
                countTotalObject = []
                totalReward = []
                totalStep = []
                finishRate = []
                totalObjectOut = []
                errorCode = [0] * 9
                tableinfo = np.zeros((opt.num_env, 256, 256, 1))
                tableinfo[:, 0, :3, 0] = -1  # inactive

                # should reset all?
                [agent_conn.send(("reset", opt.nObject)) for agent_conn in envs.agent_conns]
                [agent_conn.recv() for agent_conn in envs.agent_conns]

                for currentStep in range(opt.num_local_steps):

                    [agent_conn.send(("getObservation", 0)) for agent_conn in envs.agent_conns]
                    listDepth, listRGB, listMask = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
                    listDepth, listRGB, listMask = np.array(listDepth), np.array(listRGB), np.array(listMask)
                    # print (listDepth.shape, listRGB.shape, listMask.shape)

                    stackedData = np.concatenate([listDepth, listRGB, listMask, tableinfo], axis=3)
                    # print (stackedData.shape)
                    # send objservation to server
                    npSocket.send(stackedData)  # After sending the data, it will wait until it receives the reponse from the server

                    # get action back from server

                    action = npSocket.recieve(64)
                    # testing
                    # action = np.random.randint(128 * 128, size=opt.num_env)
                    # print ("recived action",action)

                    # startTime = time.time()
                    [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]
                    realAction, reward, CODse, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
                    # endTime = time.time()
                    # print ("Process Action",endTime - startTime)
                    tableinfo = np.zeros((opt.num_env, 256, 256, 1))
                    for i in range(opt.num_env):
                        tableinfo[i, 0, 0, 0] = reward[i]
                        tableinfo[i, 0, 1, 0] = CODse[i]
                        tableinfo[i, 0, 2, 0] = 1  # active

                    for i in range(opt.num_env):
                        if (CODse[i] > 0):
                            countTotalObject.append(info[i]["totalPickedObject"])
                            totalReward.append(info[i]["totalReward"])
                            totalObjectOut.append(info[i]["objectOut"])
                            totalStep.append([info[i]["totalActualStep"]])
                            errorCode[info[i]["errorCode"]] += 1
                            envs.agent_conns[i].send(("reset",  opt.nObject))
                            envs.agent_conns[i].recv()
                totalTimeStep += opt.num_local_steps * opt.num_env
                if (not opt.evaluation):
                    nTrajectory = len(countTotalObject)
                    avgStep = np.sum(totalStep) / nTrajectory
                    avgReward = np.sum(totalReward) / nTrajectory
                    avgObject = np.sum(countTotalObject) / nTrajectory
                    avgSuccefulRate = np.sum(countTotalObject) / np.sum(totalStep)
                    avgObjectOut = np.sum(totalObjectOut) / np.sum(totalStep)

                    tensorboard.add_scalar(f"Log/avgStep", avgStep, totalTimeStep)
                    tensorboard.add_scalar(f"Log/avgReward", avgReward, totalTimeStep)
                    tensorboard.add_scalar(f"Log/avgObject", avgObject, totalTimeStep)
                    tensorboard.add_scalar(f"Log/avgSuccefulRate", avgSuccefulRate, totalTimeStep)
                    tensorboard.add_scalar(f"Log/avgObjectOut", avgObjectOut, totalTimeStep)
                    tensorboard.add_scalar(f"Log/maxObject", np.max(countTotalObject), totalTimeStep)

                    tensorboard.add_scalar("DieReason/CODsE_FINISH", errorCode[StatusCode.CODsE_FINISH] / nTrajectory, totalTimeStep)
                    tensorboard.add_scalar("DieReason/IK_FAILURE", errorCode[StatusCode.IK_FAILURE] / nTrajectory, totalTimeStep)
                    tensorboard.add_scalar("DieReason/CONTROL_FAILURE", errorCode[StatusCode.CONTROL_FAILURE] / nTrajectory, totalTimeStep)
                    tensorboard.add_scalar("DieReason/CODsE_EXCEEDED_MAX_ACTION",errorCode[StatusCode.CODsE_EXCEEDED_MAX_ACTION] / nTrajectory, totalTimeStep)

                # SEND ONE MORE OBSERVATION
                [agent_conn.send(("getObservation", 0)) for agent_conn in envs.agent_conns]
                listDepth, listRGB, listMask = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
                listDepth, listRGB, listMask = np.array(listDepth), np.array(listRGB), np.array(listMask)
                # print (listDepth.shape, listRGB.shape, listMask.shape)

                stackedData = np.concatenate([listDepth, listRGB, listMask, tableinfo], axis=3)

                npSocket.send(stackedData)
                # waiting for training here
                action = npSocket.recieve(64)
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]
                realAction, reward, CODse, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])


if __name__ == "__main__":
    opt = get_args()

    if (not opt.evaluation):
        if os.path.isdir(opt.log_path + "/" + opt.setting):
            shutil.rmtree(opt.log_path + "/" + opt.setting)
        os.makedirs(opt.log_path + "/" + opt.setting)

    train(opt)


