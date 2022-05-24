import random

import os
os.environ['OMP_NUM_THREADS'] = '1'
CUDA_LAUNCH_BLOCKING=1
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
import argparse
import torch
import sys
sys.path.insert(1, '..')
from tqdm import tqdm

from DON_Picking.MultiSimEnv  import  MultipleEnvironments
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

def FlattenMask (mask):
    MaxPool = nn.MaxPool2d(2, 2)
    flatten = MaxPool(mask)
    flatten = flatten.view(mask.size(0),config["HalfWidth"]*config["HalfWidth"])

    return flatten

def get_args():
    parser = argparse.ArgumentParser(
        """Server PPO for Picking Cluttered Objects. PPO Training setup""")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--DON_mode', type=str, default="RGBD")
    parser.add_argument('--DON_pretrained', type=str, default="DON_GraspNet_RGBD_Resnet8/DON_5001")
    parser.add_argument('--gamma', type=float, default=0.5, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.001, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument("--num_local_steps", type=int, default=128)
    parser.add_argument("--num_global_steps", type=int, default=1e6)
    parser.add_argument("--num_env", type=int, default=3)
    parser.add_argument("--save_interval", type=int, default=10, help="Number of steps between savings")
    parser.add_argument("--log_path", type=str, default="tensorboard/Picking")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--setting", type=str, default="Picking_DON_GraspNet")
    parser.add_argument("--pretrained", type=str, default="None")
    parser.add_argument("--comment", type=str, default="comment Model configuration")
    args = parser.parse_args()
    return args



def setup_tensorboard(tensorboard_log_dir, comment="no comment"):
    shutil.rmtree(tensorboard_log_dir)
    # if not os.path.isdir(tensorboard_log_dir):
    os.makedirs(tensorboard_log_dir)
    # else

    print("Save path:", (tensorboard_log_dir))

    cmd = "tensorboard --logdir=%s" % (tensorboard_log_dir)
    # logging.info("tensorboard logger started")
    print(cmd)
    writer = SummaryWriter(tensorboard_log_dir, comment)
    return writer

'''
def explained_variance(y, pred):
    y_var = torch.var(y, dim=[0])
    diff_var = torch.var(y - pred, dim=[0])
    min_ = torch.Tensor([-1.0])
    return torch.max(
        min_.to(
            device=torch.device("cuda")
        ) if torch.cuda.is_available() else min_,
        1 - (diff_var / y_var)
    )

'''

def get_lr(optimizerX):
    for param_group in optimizerX.param_groups:
        return param_group['lr']

def train(opt):
    DON_Mod = opt.DON_mode
    DON = DensObjectNet(setting="D", pretrained=opt.DON_pretrained)

    if torch.cuda.is_available():
        print("Use GPU")
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if not os.path.isdir(opt.saved_path + "/" + opt.setting):
        os.makedirs(opt.saved_path + "/" + opt.setting)
        print("Save path:", (opt.saved_path + "/" + opt.setting))

    mp = _mp.get_context("spawn")

    envs = MultipleEnvironments(num_envs=opt.num_env,headless = True)

    tensorboard = setup_tensorboard(opt.log_path + "/" + opt.setting)
    print("Log path:", (opt.log_path + "/" + opt.setting))

    #models = U_NET(inputChannel=9)
    model = RUNet("RGBD",1)
    if torch.cuda.is_available():
        model.cuda()

    if (opt.pretrained!= "None"  ):
        directory = opt.saved_path + "/" + opt.pretrained
        print("Using Pretrained Model", directory)
        model.load_state_dict(torch.load(directory))

    model.share_memory()
    model = model.float()

    # process = mp.Process(target=eval, args=(opt, models,opt.HOST,opt.PORT))
    # process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    [agent_conn.send(("reset", 0)) for agent_conn in envs.agent_conns]
    [agent_conn.recv() for agent_conn in envs.agent_conns]

    curr_episode = 0
    totalTimeStep = 0# 60*3*128

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    nEpisode = 1024 * 4 * 1000 // (opt.num_env) // opt.num_local_steps

    ruleBaseForce = 1
    stepRuleBaseForce = 0.999
    minRuleBaseForce = 0.001

    #positionLog = np.zeros(config["HalfWidth"] * config["HalfWidth"])
    for step2M in range(12):  # for each 4M timestep. Total 48M

        curentLR = get_lr(optimizer)

        #print("Change LR: ", curentLR)
        for epi in tqdm(range(nEpisode)):

            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                print("Save models ", "{}/Picking_{}".format(opt.saved_path + "/" + opt.setting, curr_episode))
                torch.save(model.state_dict(),
                           "{}/Picking_{}".format(opt.saved_path + "/" + opt.setting, curr_episode))
            curr_episode += 1
            old_log_policies = []
            actions = []
            values = []
            states = []
            obs = []
            rewards = []
            maskes = []
            dones = []

            #log
            countTotalObject = []
            totalReward = []
            totalStep= []
            finishRate = []
            totalObjectOut = []
            errorCode = [0]*5

            for currentStep in range(opt.num_local_steps):

                [agent_conn.send(("getObservation", 0)) for agent_conn in envs.agent_conns]
                listDepth, listRGB, listMask = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])

                listRGB = torch.from_numpy(np.array(listRGB))
                listDepth = torch.from_numpy(np.array(listDepth))
                listMask = torch.from_numpy(np.array(listMask)).float()

                tensorListRGB = listRGB.permute((0,3,1,2))
                tensorListRGB = tensorListRGB.float().cuda()
                tensorListDepth = listDepth.permute((0, 3, 1, 2))
                tensorListDepth = tensorListDepth.float().cuda()

                #DOND
                tensorListRGB = tensorListDepth #torch.cat([tensorListRGB,tensorListDepth],dim=1)
                #print (tensorListRGB.shape)

                tensorDescription = DON.getDescriptorTensor(tensorListRGB)
                tensorDescription = tensorDescription.detach()
                #tensorDescription = tensorDescription[:,:3,:,:]

                #tensorDescription = torch.cat([tensorListRGB,tensorDescription], dim=1)
                returnPolicy , returnValue = model(tensorDescription,tensorListDepth)
                returnPolicy = returnPolicy.detach()
                value = returnValue.detach()

                flattenMask = FlattenMask(listMask).cuda()
                policy = F.softmax(returnPolicy.view(opt.num_env, config["HalfWidth"] * config["HalfWidth"]),dim=1)*flattenMask

                old_m = Categorical(policy)
                action = old_m.sample()
                #action = torch.cuda.IntTensor(realAction)
                old_log_policy = old_m.log_prob(action)
                old_log_policies.append(old_log_policy.cpu())
                actions.append(action.cpu())

                action = action.cpu().detach().numpy()

                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

                realAction, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])

                #action = torch.cuda.IntTensor(realAction)


                #positionLog[action] += 1

                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
                maskes.append(flattenMask.cpu())

                states.append(tensorDescription.cpu())
                obs.append(tensorListDepth.cpu())

                values.append(value.squeeze().cpu())


                rewards.append(reward)
                dones.append(done)

                for i in range(opt.num_env):
                    if (done[i] > 0):
                        countTotalObject.append(info[i]["totalPickedObject"])
                        totalReward.append(info[i]["totalReward"])
                        totalObjectOut.append(info[i]["objectOut"])
                        totalStep.append([info[i]["totalActualStep"]])
                        errorCode[info[i]["errorCode"]] +=1
                        envs.agent_conns[i].send(("reset", 0))
                        envs.agent_conns[i].recv()


            '''
            #heatmap
            currentBasket = (listRGB[0].detach().cpu().numpy()).reshape(config["W"], config["W"],3)
            visualOutput = ((policy[0]/torch.max(policy[0])).cpu().numpy() * 255).astype(np.uint8)
            visualOutput = visualOutput.reshape(config["HalfWidth"], config["HalfWidth"])
            heatmap120 = cv2.applyColorMap(visualOutput, cv2.COLORMAP_JET)
            cv2.imwrite("heatmap.jpg", heatmap120)
            heatmap120 = cv2.cvtColor(heatmap120, cv2.COLOR_BGR2RGB)

            des = tensorDescription[0,:3,:,:]
            '''

            nTrajectory = len(countTotalObject)
            avgStep = np.sum(totalStep) / nTrajectory
            avgReward = np.sum(totalReward) / nTrajectory
            avgObject = np.sum(countTotalObject) / nTrajectory
            avgSuccefulRate = np.sum(countTotalObject)/ np.sum(totalStep)
            avgObjectOut = np.sum(totalObjectOut) / np.sum(totalStep)

            totalTimeStep += opt.num_local_steps * opt.num_env
            tensorboard.add_scalar(f"Log/avgStep", avgStep, totalTimeStep)
            tensorboard.add_scalar(f"Log/avgReward", avgReward, totalTimeStep)
            tensorboard.add_scalar(f"Log/avgObject", avgObject, totalTimeStep)
            tensorboard.add_scalar(f"Log/avgSuccefulRate", avgSuccefulRate, totalTimeStep)
            tensorboard.add_scalar(f"Log/avgObjectOut", avgObjectOut, totalTimeStep)
            tensorboard.add_scalar(f"Log/maxObject", np.max(countTotalObject), totalTimeStep)

            tensorboard.add_scalar("DieReason/DONE_FINISH", errorCode[DONE_FINISH] / nTrajectory, totalTimeStep)
            tensorboard.add_scalar("DieReason/DONE_CRASH", errorCode[DONE_CRASH] / nTrajectory, totalTimeStep)
            tensorboard.add_scalar("DieReason/DONE_EXCEEDED_MAX_ACTION", errorCode[DONE_EXCEEDED_MAX_ACTION] / nTrajectory, totalTimeStep)
            tensorboard.add_scalar("DieReason/DONE_OUT_AREA", errorCode[DONE_OUT_AREA] / nTrajectory, totalTimeStep)

            #print (currentBasket.shape)
            #print (heatmap120.shape)
            '''
            tensorboard.add_image('LogImage/Basket', currentBasket, totalTimeStep, dataformats='HWC')
            tensorboard.add_image('LogImage/Des', des / 1.0, totalTimeStep, dataformats='CHW')
            tensorboard.add_image('LogImage/HeatMap', heatmap120 / 1.0, totalTimeStep, dataformats='HWC')

            tensorboard.add_image('LogImage/logPosition', positionLog.reshape(config["HalfWidth"], config["HalfWidth"]) / max(positionLog),
                                  totalTimeStep, dataformats='HW')
            '''
            # PROCESS LAST STEP
            [agent_conn.send(("getObservation", 0)) for agent_conn in envs.agent_conns]
            listDepth, listRGB,listMask  = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])

            listRGB = torch.from_numpy(np.array(listRGB))
            listDepth = torch.from_numpy(np.array(listDepth))
            listMask = torch.from_numpy(np.array(listMask)).float()

            tensorListRGB = listRGB.permute((0, 3, 1, 2))
            tensorListRGB = tensorListRGB.float().cuda()
            tensorListDepth = listDepth.permute((0, 3, 1, 2))
            tensorListDepth = tensorListDepth.float().cuda()

            # DOND
            tensorListRGB =tensorListDepth # torch.cat([tensorListRGB, tensorListDepth], dim=1)

            tensorDescription = DON.getDescriptorTensor(tensorListRGB)
            tensorDescription = tensorDescription.detach()
            #tensorDescription = torch.cat([tensorListRGB, tensorDescription], dim=1)
            # tensorDescription = tensorDescription[:, :3, :, :]

            returnPolicy , returnValue = model(tensorDescription,tensorListDepth)
            returnPolicy = returnPolicy.detach()
            value = returnValue.detach()

            next_value = value.squeeze().cpu()
            # TRAINING
            gae = 0
            R = []
            stackedData = list(zip(values, rewards, dones))[::-1]

            for value, reward, done in stackedData:
                '''
                delta = reward  + opt.gamma * next_value.detach() * (1 - done) - value.detach()
                gae = delta + opt.gamma * opt.tau * (1 - done) * gae
                R.append(gae + value)
                next_value = value

                '''
                gae = gae * opt.gamma * opt.tau
                gae = gae + reward  + opt.gamma * next_value.detach() * (1 - done) - value.detach()
                next_value = value
                R.append(gae + value)
                gae = torch.mul(gae, (1 - done))


            # values = torch.cat(values).detach()
            R = R[::-1]
            R = torch.cat(R).detach()

            values = torch.cat(values).detach()
            next_value = next_value.squeeze()
            old_log_policies = torch.cat(old_log_policies).detach()
            actions = torch.cat(actions)
            states = torch.cat(states)
            maskes = torch.cat(maskes)
            obs = torch.cat(obs)

            advantages = R - values

            #log_explained_variance = explained_variance(R, values)

            tensorboard.add_scalar(f"Log_Debug/Value", torch.mean(values).item(), totalTimeStep)
            tensorboard.add_scalar(f"Log_Debug/R", torch.mean(R).item(), totalTimeStep)
            #tensorboard.add_scalar(f"Log_Debug/Explained_variance", torch.mean(log_explained_variance).item(),
            #                       totalTimeStep)

            # models.train()

            log_policy_loss = 0
            add_scalar_loss = 0
            log_entropy_loss = 0
            log_total_loss = 0
            numOfBatchSize = (opt.num_local_steps * opt.num_env) // opt.batch_size

            for i in range(opt.num_epochs):
                indice = torch.randperm(opt.num_local_steps * opt.num_env)
                for j in range(numOfBatchSize):
                    batch_indices = indice[(j * opt.batch_size): ((j + 1) * opt.batch_size)]

                    logits, value = model(states[batch_indices].cuda(),obs[batch_indices].cuda())
                    logits =  F.softmax(logits.view(opt.batch_size, config["HalfWidth"] * config["HalfWidth"]),dim=1) * maskes[batch_indices].cuda()

                    new_m = Categorical(logits)
                    new_log_policy = new_m.log_prob(actions[batch_indices].cuda())
                    ratio = torch.exp(new_log_policy - old_log_policies[batch_indices].cuda())

                    actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices].cuda(),
                                                       torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                       advantages[batch_indices].cuda()))

                    critic_loss = F.smooth_l1_loss(R[batch_indices].cuda(), value.squeeze())
                    entropy_loss = torch.mean(new_m.entropy())
                    total_loss = actor_loss + critic_loss - opt.beta * entropy_loss

                    log_policy_loss += actor_loss.item()
                    add_scalar_loss += critic_loss.item()
                    log_entropy_loss += entropy_loss.item()
                    log_total_loss += total_loss.item()

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # REMOVE BY CHGIANG
                    optimizer.step()

            log_policy_loss /= (opt.num_epochs * numOfBatchSize)
            add_scalar_loss /= (opt.num_epochs * numOfBatchSize)
            log_entropy_loss /= (opt.num_epochs * numOfBatchSize)
            log_total_loss /= (opt.num_epochs * numOfBatchSize)

            tensorboard.add_scalar(f"Log_Loss/log_policy_loss", log_policy_loss, totalTimeStep)
            tensorboard.add_scalar(f"Log_Loss/log_value_loss", add_scalar_loss, totalTimeStep)
            tensorboard.add_scalar(f"Log_Loss/log_entropy_loss", log_entropy_loss, totalTimeStep)
            tensorboard.add_scalar(f"Log_Loss/log_total_loss", log_total_loss, totalTimeStep)

            #print("maxObject", max(countTotalObject),  "||", "max score:", max(totalReward))
            #print ("--------")


        # end of every 2M
        scheduler.step()

    tensorboard.close()


if __name__ == "__main__":
    opt = get_args()

    if os.path.isdir(opt.log_path + "/" + opt.setting):
        shutil.rmtree(opt.log_path + "/" + opt.setting)
    os.makedirs(opt.log_path + "/" + opt.setting)

    train(opt)


