import numpy as np
from PIL import Image
from numpy.linalg import inv
import cv2
import time
import pandas as pd
from tqdm import tqdm
import sys

sys.path.insert(1, '..')

from CODs_Training.DataGenerator_O2O import DataGenerator,loadAllModels,convertToHeatmap,render,remove_values_from_list,processImage
from ITRIP.Configuration import sceneTypeString,config,SINGLE_OBJECT_WITHIN_SCENE
from CODs_Training.DensObjectNet import DensObjectNet

def getDiffDist(p1, p2):
    maxDist = np.linalg.norm(np.array([config["W"], config["W"]]))
    currDist = np.linalg.norm(np.array(p2) - np.array(p1))
    return (currDist / maxDist)

dataGenerator = DataGenerator()
models, colors = loadAllModels(path="../CODs_data/",loadFile=True)
dataGenerator.setOriginalModelObjects(models, colors)




def evaluate(savedModel,inputMode):
    config["inputMode"] = inputMode
    CODs = DensObjectNet(setting=config["inputMode"], pretrained=savedModel)
    nTry = 100
    nTestPoint = 1000

    testCase = [3,4]
    result = np.zeros(2,len(testCase))

    for unseen in [0,1]:
        for type_test in range(len(testCase)) :

            # totalTime = [0] * len(testCase)
            # totalMatchPoint = [0] * len(testCase)
            # totalNonMatch = [0] * len(testCase)
            # totalSelect = [0] * len(testCase)
            totalPoint = 0

            for i in  tqdm(range(nTry)):

                nMatchPoint = 0
                augmentationType =  0 #np.random.choice([0, 1, 2, 3])
                while (nMatchPoint < nTestPoint):
                    imgA,depthA, imgB,depthB, rawData_1, rawData_2, matches_a, matches_b, _, _, _, _, _, _, nMatchPoint, nNoneMatchPoint=\
                                                        dataGenerator.generateRandomData(pathToScense="../CODs_data/", matchType=0, sceneType=min(7,testCase[type_test]), augmentationType=augmentationType, debug=False, isEvaluate = True,isLoadUnseen=unseen,isLoadOrig = (testCase[type_test] == 8))

                    imageColor_1, depth_1, mask_1, indexMap_1, segmentMap_1 = tuple(rawData_1)
                    imageColor_2, depth_2, mask_2, indexMap_2, segmentMap_2 = tuple (rawData_2)

                v1,u1 = matches_a.squeeze().cpu().numpy() //config["W"],matches_a.squeeze().cpu().numpy()%config["W"]
                v2, u2 = matches_b.squeeze().cpu().numpy() // config["W"], matches_b.squeeze().cpu().numpy() % config["W"]

                nValidTestPoint = min(nTestPoint, nMatchPoint)
                totalPoint += nValidTestPoint


                setting = inputMode
                totalDistance = 0
                accuracy = 0

                #imgA = processImage(imageColor_1, depth_1, mask_1, setting)
                #imgB = processImage(imageColor_2, depth_2, mask_2, setting)

                descrtion1 = CODs.dcn.forward_single_image_tensor(imgA.clone(),depthA.clone())
                descrtion2 = CODs.dcn.forward_single_image_tensor(imgB.clone(),depthB.clone())
                descrtion1 = descrtion1.detach().cpu().numpy()
                descrtion2 = descrtion2.detach().cpu().numpy()


                for ith in range (nValidTestPoint):
                        px, py = u1[ith],v1[ith]
                        # print (px, py)
                        matchPoint, _ = CODs.getBestMatchPointOnly((px, py), descrtion1, descrtion2, mask_2)
                        p_lu2, p_lv2 = tuple(matchPoint)
                        g_lu2, g_lv2 = u2[ith],v2[ith]
                        diff = getDiffDist([p_lu2, p_lv2], [g_lu2, g_lv2])
                        totalDistance +=diff

                        if (segmentMap_1[py, px] == segmentMap_2[p_lv2, p_lu2] and segmentMap_1[py, px]!= -1 and segmentMap_2[p_lv2, p_lu2]  !=-1 ):
                        #if (diff <0.15):
                            accuracy += 1

                result[0, type_test] += totalDistance
                result[1, type_test] += accuracy


        indices = []
        print (totalPoint,"totalPoint")
        result/= totalPoint

        return result


def evaluateSinglePair(savedModel, inputMode,sceneType = None,unseen = 0):
    config["inputMode"] = inputMode
    CODs = DensObjectNet(setting=config["inputMode"], pretrained=savedModel)

    if(sceneType is None):
        sceneType =  np.random.choice([3,4])
    result = np.zeros(2)

    totalPoint = 0
    nTry = 100
    nTestPoint = 1000

    nMatchPoint = 0
    augmentationType = 0  # np.random.choice([0, 1, 2, 3])
    while (nMatchPoint < nTestPoint):
        imgA, depthA, imgB, depthB, rawData_1, rawData_2, matches_a, matches_b, _, _, _, _, _, _, nMatchPoint, nNoneMatchPoint = \
            dataGenerator.generateRandomData(pathToScense="../CODs_data/", matchType=0,
                                             sceneType= sceneType,
                                             augmentationType=augmentationType, debug=False,
                                             isEvaluate=True, isLoadUnseen=unseen,
                                             isLoadOrig=(sceneType == 8))

        imageColor_1, depth_1, mask_1, indexMap_1, segmentMap_1 = tuple(rawData_1)
        imageColor_2, depth_2, mask_2, indexMap_2, segmentMap_2 = tuple(rawData_2)

    v1, u1 = matches_a.squeeze().cpu().numpy() // config["W"], matches_a.squeeze().cpu().numpy() % config[
        "W"]
    v2, u2 = matches_b.squeeze().cpu().numpy() // config["W"], matches_b.squeeze().cpu().numpy() % config[
        "W"]

    nValidTestPoint = min(nTestPoint, nMatchPoint)
    totalPoint += nValidTestPoint

    setting = inputMode
    totalDistance = 0
    accuracy = 0

    descrtion1 = CODs.dcn.forward_single_image_tensor(imgA.clone(), depthA.clone())
    descrtion2 = CODs.dcn.forward_single_image_tensor(imgB.clone(), depthB.clone())
    descrtion1 = descrtion1.detach().cpu().numpy()
    descrtion2 = descrtion2.detach().cpu().numpy()

    for ith in range(nValidTestPoint):
        px, py = u1[ith], v1[ith]
        # print (px, py)
        matchPoint, _ = CODs.getBestMatchPointOnly((px, py), descrtion1, descrtion2, mask_2)
        p_lu2, p_lv2 = tuple(matchPoint)
        g_lu2, g_lv2 = u2[ith], v2[ith]
        diff = getDiffDist([p_lu2, p_lv2], [g_lu2, g_lv2])
        totalDistance += diff

        if (segmentMap_1[py, px] == segmentMap_2[p_lv2, p_lu2] and segmentMap_1[py, px] != -1 and
                segmentMap_2[p_lv2, p_lu2] != -1):
            # if (diff <0.15):
            accuracy += 1

    result[0] += totalDistance
    result[1] += accuracy

    print(totalPoint, "totalPoint")
    result /= totalPoint
    return result