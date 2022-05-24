# system
import numpy as np
import os
import random
import fnmatch
import gc
import logging
import time
import shutil
import subprocess
import copy
import shutil
# torch
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter
CUDA_LAUNCH_BLOCKING=1
import sys
sys.path.insert(1, '..')

from  DON_Training.loss_function.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
import DON_Training.loss_function.loss_composer as loss_composer

from DON_Training.dense_correspondence_network import DenseCorrespondenceNetwork
from DON_Training.ultils import *
from DON_Training.DONDataLoader_O2O import DONDataset
from ITRIP.Configuration import *


import DON_Training.evaluation.plotting as dc_plotting

print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
from tqdm import tqdm

def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def getDiffDist(p1, p2):
    maxDist = np.linalg.norm(np.zeros(2) - np.array([config["W"], config["W"]]))
    currDist = np.linalg.norm(np.array(p2) - np.array(p1))
    return (currDist / maxDist)

class DONTrainer(object):

    def __init__(self, config=None, dataset=None, dataset_test=None):


        self.config = config
        self.setup()

    def setup(self):
        self.tensorboard  = self.setup_tensorboard("../DON_Training/"+"tensorboard/"+self.config["setting"]+"_"+self.config["inputMode"]+"_"+config["model"]+str(config["descriptor_dimension"]), "comment")
        self.model = self.setup_network()
        self.model.cuda()

        #TEST INPUT
        input = torch.rand(2, 3, self.config["Width"], self.config["Height"]).cuda()
        depth = torch.rand(2, 1, self.config["Width"], self.config["Height"]).cuda()
        output = self.model.forward(input,depth)
        print(output.shape)

        self.optimizer = self.setup_optimizer(self.model.parameters())

        self._dataset = DONDataset()

        self._data_loader = torch.utils.data.DataLoader(self._dataset, batch_size=self.config['training']['batch_size'],
                                                        shuffle=False, num_workers=self.config['training']["num_workers"], drop_last=False,worker_init_fn=worker_init_fn)

    def run(self, nIteration=1, current_iteration=0, pretrained=None):

        if (pretrained is not None):
            self.load_network(pretrained)

        pixelwise_contrastive_loss = PixelwiseContrastiveLoss(image_shape=[self.config["Width"],self.config["Height"]],
                                                              config=self.config['loss_function'])
        pixelwise_contrastive_loss.debug = True
        loss = match_loss = non_match_loss = 0

        logging_rate = self.config['training']['logging_rate']
        save_rate = self.config['training']['save_rate']
        batch_size = self.config['training']['batch_size']

        current_iteration = 0
        totaldataPoint = 0
        logSelectedType = [0]*8
        countDataPoint = [0]*8
        epiLoss = np.zeros((8,5))
        tempSelectType =[0]*8
        for epoch in range(3):  # loop over the dataset multiple times
            #for ith in  tqdm(range(25*5*80//2)):
            for i, data in enumerate(tqdm(self._data_loader, 0)):

                current_iteration += 1
                start_iter = time.time()

                img_a,depth_a, img_b,depth_b,matches_a, matches_b,\
                masked_non_matches_a, masked_non_matches_b,\
                background_non_matches_a, background_non_matches_b,\
                blind_non_matches_a, blind_non_matches_b,totalData,match_type,scene_type = data
                totaldataPoint += totalData
                data_type = match_type
                logSelectedType[scene_type]+=1
                countDataPoint[scene_type] +=totalData
                #print ("out generate",img_a.shape)
                img_a = Variable(img_a.cuda(), requires_grad=False)
                img_b = Variable(img_b.cuda(), requires_grad=False)
                depth_a = Variable(depth_a.cuda(), requires_grad=False)
                depth_b = Variable(depth_b.cuda(), requires_grad=False)


                if (matches_a is not None):
                    matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
                    matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
                masked_non_matches_a = Variable(masked_non_matches_a.cuda().squeeze(0), requires_grad=False)
                masked_non_matches_b = Variable(masked_non_matches_b.cuda().squeeze(0), requires_grad=False)
                blind_non_matches_a = Variable(blind_non_matches_a.cuda().squeeze(0), requires_grad=False)
                blind_non_matches_b = Variable(blind_non_matches_b.cuda().squeeze(0), requires_grad=False)
                background_non_matches_a = Variable(background_non_matches_a.cuda().squeeze(0), requires_grad=False)
                background_non_matches_b = Variable(background_non_matches_b.cuda().squeeze(0), requires_grad=False)


                self.optimizer.zero_grad()
                if (epoch < 2):
                  self.adjust_learning_rate(current_iteration)
                # run both images through the network
                #print (img_a.shape)
                #print (img_b.shape)
                if (len(img_a.shape) ==3):
                    img_a = img_a.unsqueeze_(0)
                    img_b = img_b.unsqueeze_(0)
                    depth_a = depth_a.unsqueeze_(0)
                    depth_b = depth_b.unsqueeze_(0)

                #batchInput = torch.cat((img_a,img_b))
                #print (batchInput.shape)
                #batch_preds =  self.model.forward(batchInput)
                img_b_orig =torch.clone(img_b).cpu().numpy()
                image_a_pred_raw_output =  self.model.forward(img_a,depth_a) #batch_preds[0].unsqueeze_(0)# self.model.forward(img_a)
                image_a_pred = self.model.process_network_output(image_a_pred_raw_output, batch_size)
                image_b_pred_raw_output =self.model.forward(img_b,depth_b)  #batch_preds[1].unsqueeze_(0) # self.model.forward(img_b)
                image_b_pred = self.model.process_network_output(image_b_pred_raw_output, batch_size)


                # get loss
                loss, match_loss, masked_non_match_loss, \
                background_non_match_loss, blind_non_match_loss = loss_composer.get_loss(pixelwise_contrastive_loss,
                                                                                         match_type,
                                                                                         image_a_pred, image_b_pred,
                                                                                         matches_a, matches_b,
                                                                                         masked_non_matches_a,
                                                                                         masked_non_matches_b,
                                                                                         background_non_matches_a,
                                                                                         background_non_matches_b,
                                                                                         blind_non_matches_a,
                                                                                         blind_non_matches_b)

                loss.backward()
                self.optimizer.step()
                elapsed = time.time() - start_iter
                epiLoss[scene_type] += np.array([loss.item(), match_loss.item(), masked_non_match_loss.item(),  background_non_match_loss.item(), blind_non_match_loss.item()])

                def update_plots(loss, match_loss, masked_non_match_loss, background_non_match_loss,
                                 blind_non_match_loss,evaluateResulti=None):

                    self.tensorboard.add_scalar(f"Log/learning_rate", self.get_learning_rate(self.optimizer) ,current_iteration)
                    self.tensorboard.add_scalar(f"LogDataPoint/totalPoint", totaldataPoint, current_iteration)
                    #log loss
                    for b in range (3,8):
                        if (logSelectedType[b]):
                          self.tensorboard.add_scalar(f"Log/train_loss_" + sceneTypeString[b], epiLoss[b, 0] / logSelectedType[b], current_iteration)
                          self.tensorboard.add_scalar(f"LogDetail_match/train_match_loss_"+sceneTypeString[b], epiLoss[b,1]/logSelectedType[b],current_iteration)
                          self.tensorboard.add_scalar(f"LogDetail_nonMatch/train_masked_non_match_loss_"+sceneTypeString[b], epiLoss[b,2]/logSelectedType[b],current_iteration)
                          self.tensorboard.add_scalar(f"LogDetail_nonMatch/train_background_loss_" + sceneTypeString[b], epiLoss[b, 3] / logSelectedType[b], current_iteration)
                          self.tensorboard.add_scalar(f"LogDetail_nonMatch/train_blind_loss_"+sceneTypeString[b],epiLoss[b,4]/logSelectedType[b], current_iteration)
                          self.tensorboard.add_scalar(f"LogDataPoint/select_"+sceneTypeString[b], logSelectedType[b], current_iteration)
                    if (evaluateResult is not None):
                          for b in range (3,13):
                              self.tensorboard.add_scalar(f"Evaluate/ErrorDist_" + sceneTypeString[b], evaluateResult[0,b-3], current_iteration)
                              self.tensorboard.add_scalar(f"Evaluate/Accuracy_" + sceneTypeString[b], evaluateResult[1, b - 3], current_iteration)


                #print(des_b.shape)

                if (current_iteration % save_rate) == 0 or current_iteration> nIteration:
                    self.save_network(current_iteration)

                if (((current_iteration-1) % logging_rate)==0) or current_iteration> nIteration:
                    # if ((current_iteration -1) %(5*logging_rate) ==0) :
                    #   evaluateResult = self.evaluate()
                    # else:
                    evaluateResult = None
                    update_plots(loss, match_loss, masked_non_match_loss, background_non_match_loss,
                                 blind_non_match_loss,evaluateResult)
                    totalLoss = 0
                    totalSelected = 0
                    for b in range (3,8):
                        if (logSelectedType[b]):
                          totalLoss += epiLoss[b, 0]
                          totalSelected += logSelectedType[b]
                    self.tensorboard.add_scalar(f"Log/train_loss_SINGLE_OBJECT_WITHIN_SCENE", totalLoss/totalSelected, current_iteration)
                    
                    des_b = image_b_pred_raw_output.detach().cpu().numpy()
                    des = des_b[0,:3, :, :]
                    #print (img_b_orig.shape)
                    img_b = np.array(img_b_orig[0,:3, :, :]*255).astype(np.uint8)
                    #print (np.max(img_b))
                    img_b = np.transpose(img_b, (1, 2, 0))
                    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
                    des, _ = dc_plotting.normalize_descriptor_pair(des, des)

                    self.tensorboard.add_image('LogImage/Des', des / 1.0, current_iteration, dataformats='CHW')
                    self.tensorboard.add_image('LogImage/Img', img_b, current_iteration, dataformats='HWC')
                    logSelectedType = [0]*8
                    countDataPoint = [0]*8
                    epiLoss = np.zeros((8, 5))

                if (current_iteration> nIteration):
                    return

    def evaluate(self):
        nTestPoint = 100
        nTry = 10
        result = np.zeros((2, 10))
        for type_scene in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            totalPoint = 1
            for i in range (nTry):
                nMatchPoint = 0
                augmentationType = np.random.choice([0, 1, 2, 3])
                while (nMatchPoint < nTestPoint):
                    if (type_scene <8):  
                        imgA,depthA, imgB,depthB, rawData_1, rawData_2, matches_a, matches_b, _, _, _, _, _, _, nMatchPoint, nNoneMatchPoint = self._dataset.dataGenerator.generateRandomData(
                            pathToScense="", matchType=0, sceneType=type_scene,
                            augmentationType=augmentationType, debug=False, isEvaluate=True)
                    elif (type_scene == 8 or type_scene == 9):
                        if (type_scene ==8):
                          path_unseen = "LabObject_train_O2O_sim"
                        elif (type_scene ==9):
                          path_unseen = "data"

                        imgA,depthA, imgB,depthB, rawData_1, rawData_2, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b, totalData, nMatchPoint = generateRandomData(
                            path_unseen, SINGLE_OBJECT_WITHIN_SCENE, augmentationType,
                            debug=False)
                    else:  # 10,11,12
                        type_scene_shift = ((type_scene) - 9) * 2 + 1
                        imgA,depthA, imgB,depthB, rawData_1, rawData_2, matches_a, matches_b, _, _, _, _, _, _, nMatchPoint, nNoneMatchPoint  = self._dataset.dataGenerator.generateRandomData(
                            pathToScense="", matchType=0, sceneType=type_scene_shift,
                            augmentationType=augmentationType, debug=False, isEvaluate=True, isLoadUnseen=True)

                if (type_scene not in [8, 9]):
                    imageColor_1, depth_1, mask_1, indexMap_1, segmentMap_1 = tuple(rawData_1)
                    imageColor_2, depth_2, mask_2, indexMap_2, segmentMap_2 = tuple(rawData_2)
                else:
                    imageColor_1, depth_1, mask_1 = tuple(rawData_1)
                    imageColor_2, depth_2, mask_2 = tuple(rawData_2)

                v1, u1 = matches_a.squeeze().cpu().numpy() // config["W"], matches_a.squeeze().cpu().numpy() % \
                         config["W"]
                v2, u2 = matches_b.squeeze().cpu().numpy() // config["W"], matches_b.squeeze().cpu().numpy() % \
                         config["W"]

                nValidTestPoint = min(nTestPoint, nMatchPoint)
                totalPoint += nValidTestPoint

                setting = config["inputMode"]
                totalDistance = 0
                accuracy = 0

                descrtion1 = self.model.forward_single_image_tensor (imgA,depthA)
                descrtion2 = self.model.forward_single_image_tensor(imgB,depthB)
                descrtion1 = descrtion1.detach().cpu().numpy()
                descrtion2 = descrtion2.detach().cpu().numpy()

                for ith in range(nValidTestPoint):
                    px, py = u1[ith], v1[ith]
                    matchPoint, _,_ = self.model.find_best_match_only((px, py), descrtion1, descrtion2,
                                                                            mask_2)
                    p_lu2, p_lv2 = tuple(matchPoint)
                    g_lu2, g_lv2 = u2[ith], v2[ith]
                    diff = getDiffDist([p_lu2, p_lv2], [g_lu2, g_lv2])
                    totalDistance += diff

                    if (type_scene not in [8, 9]):
                        if (segmentMap_1[py, px] == segmentMap_2[p_lv2, p_lu2] and segmentMap_1[
                            py, px] != -1 and segmentMap_2[p_lv2, p_lu2] != -1):
                            accuracy += 1
                    else:
                        if (mask_1[py, px] == mask_2[p_lv2, p_lu2] and mask_1[py, px] != 0):
                            accuracy += 1

                result[0, type_scene - 3] += totalDistance
                result[1, type_scene - 3] += accuracy
            result[0, type_scene - 3] /= totalPoint
            result[1, type_scene - 3] /= totalPoint

        return result


    def setup_optimizer(self, parameters):
        optimizer = None

        learning_rate = float(self.config["training"]['learning_rate'])
        weight_decay = float(self.config["training"]['weight_decay'])
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

        return optimizer

    def setup_tensorboard(self,tensorboard_log_dir,comment):
        shutil.rmtree(tensorboard_log_dir, ignore_errors=True)
        createFolder(tensorboard_log_dir)
        print("log path:", tensorboard_log_dir)
        cmd = "tensorboard --logdir=%s" % (tensorboard_log_dir)
        print ("tensorboard cmd:",cmd)

        writer = SummaryWriter(tensorboard_log_dir,comment)
        return writer

    def setup_network(self):
        model = DenseCorrespondenceNetwork.from_config(config["inputMode"])
        save_path = "../DON_Training/"+self.config["saved_path"] + "/" + self.config["setting"]+"_"+self.config["inputMode"]+"_"+config["model"]+str(config["descriptor_dimension"])
        createFolder(save_path)
        print ("save path:",save_path)
        
        # pretrained
        return model

    def save_network(self, iteration):
        print("Save models ", "{}/DON_{}".format("../DON_Training/"+self.config["saved_path"] + "/" + self.config["setting"]+"_"+self.config["inputMode"]+"_"+config["model"]+str(config["descriptor_dimension"]), iteration))
        torch.save(self.model.state_dict(), "{}/DON_{}".format("../DON_Training/"+self.config["saved_path"] + "/" + self.config["setting"]+"_"+self.config["inputMode"]+"_"+config["model"]+str(config["descriptor_dimension"]), iteration))

    def load_network(self,pretrained):
        print ("Load pretrained ",pretrained)
        self.model.load_state_dict(torch.load(pretrained))

    def adjust_learning_rate(self, iteration):
        #print ("RE implement adjust learning rate")
        steps_between_learning_rate_decay = self.config['training']['steps_between_learning_rate_decay']
        if iteration % steps_between_learning_rate_decay == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.config["training"]["learning_rate_decay"]

            print("LR:", self.get_learning_rate(self.optimizer))

    @staticmethod
    def set_learning_rate(optimizer, learning_rate):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    @staticmethod
    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

