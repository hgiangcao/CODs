import numpy as np
import itertools
from PIL import Image
import imageio
from random import randint
import random
import sys
import torch
sys.path.insert(1, '..')
import os
from transforms3d.euler import euler2mat
from sys import argv
import copy
from plyfile import PlyData, PlyElement
import cv2
from numpy.linalg import inv
from colorsys import rgb_to_hls, hls_to_rgb
from IPython.display import Image
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ITRIP.Configuration import *
from tqdm import tqdm
import time
import random
import math as m
from xml.dom import minidom
from transforms3d.quaternions import quat2mat
import pickle

import torch
import torchvision

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

intrinsicCamera  = np.loadtxt(config["intrisic_matrix_file"], delimiter=' ')
intrinsicCameraReal  = np.loadtxt(config["intrisic_matrix_real_file"], delimiter=' ')
intrinsicCamera[0, 2] = 0
intrinsicCamera[1, 2] = 0
nImage = config['n_image']
nScene = config['n_scene']
nSceneReal =config['n_scene']

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def empty_tensor():
    return torch.LongTensor([-1])


def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def lighten_color(r, g, b, factor=0.1):
    return adjust_color_lightness(r, g, b, 1 + factor)


def loadAllModels(nObject=87, path="",loadFile=False):
    originalModelObjects = [0]*nObject
    originalModelObjectsColor = [0]*nObject

    if (loadFile):
        file_name_model = path+"GraspNet_models/allModels.pkl"
        open_file_model = open(file_name_model, "rb")
        originalModelObjects = pickle.load(open_file_model)
        open_file_model.close()
        file_name_color = path+"GraspNet_models/allModelsColor.pkl"
        open_file_color = open(file_name_color, "rb")
        originalModelObjectsColor = pickle.load(open_file_color)
        open_file_color.close()
    return originalModelObjects, originalModelObjectsColor

def project(poseCamera, objPose, vertex,loadITRI = False,generateObjectDetection = False):
    #print (poseCamera.shape, objPose.shape, vertex.shape)

    projectedImage = np.matmul((intrinsicCamera),np.matmul((poseCamera[:3]), np.matmul((objPose), (vertex.T))))
    scale = 1 / projectedImage[2, :]

    if (not generateObjectDetection):
        for i in range(vertex.shape[0]):
            projectedImage[2, i] = np.linalg.norm(projectedImage[:, i])

    projectedImage[:2, :] *= scale

    projectedImage[:2, :] += config["HalfWidth"]
    return projectedImage.T


def createPLY(verties,colors,fileName):
    colors = colors.astype("int")
    strX = "ply\n format ascii 1.0\nelement vertex " + str(verties.shape[0])+\
            "\nproperty float x\nproperty float y\nproperty float z\nproperty uint8 red\nproperty uint8 green\nproperty uint8 blue\nend_header\n"

    for i in range (verties.shape[0]):
        strX += str(verties[i,0]) + " "+ str(verties[i,1]) + " " + str(verties[i,2]) + " " + str(colors[i,0]) + " "+  str(colors[i,1]) + " " + str(colors[i,2])
        if (i!= verties.shape[0]-1):
            strX+="\n"

    text_file = open(fileName, "w")
    n = text_file.write(strX)
    text_file.close()

def convertToHeatmap(img):
    MIN = np.min(img)
    MAX = np.max(img)
    img = ((img - MIN) / (MAX - MIN) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = Image.fromarray(np.uint8(img))
    return img



def render(uvCoord, colorCoord, img, renderMap, segmentMap, indexMap, mask, objID=0, renderImage=False):
    if (renderImage):
        draw = ImageDraw.Draw(img)
        minX, maxX = np.min(uvCoord[:, 0]), np.max(uvCoord[:, 0])
        minY, maxY = np.min(uvCoord[:, 1]), np.max(uvCoord[:, 1])
        minX =int ( max(0, min(config["W"], minX)))
        minY =int (  max(0, min(config["W"], minY)))
        maxX = int ( max(0, min(config["W"], maxX)))
        maxY =int (  max(0, min(config["W"], maxY)))
        lX   = abs(minX - maxX)
        lY   = abs(minY - maxY)
        area = lX * lY
        draw.rectangle([minX, minY, minX+lX, minY+lY])

    nVertext = uvCoord.shape[0]

    idxSorted = np.argsort(uvCoord[:,2])
    #uvCoord = uvCoord[idxSorted]
    #colorCoord = colorCoord[idxSorted]

    #uvCoord[:, :2] += config["HalfWidth"]
    step = 0
    for i in range(nVertext):
        u = (int)(uvCoord[idxSorted[i], 0])
        v = (int)(uvCoord[idxSorted[i], 1])

        if ( 0< u < config['W'] and  0 <  v < config['W']):
            mask[v,u] = 1

            if (uvCoord[idxSorted[i], 2] < renderMap[v, u] and (indexMap[v,u]  < idxSorted[i])):

                renderMap[v,u] = uvCoord[idxSorted[i], 2]
                segmentMap[v,u] = objID
                indexMap[v,u] = idxSorted[i]

                if (renderImage):
                    r, g, b = (tuple(colorCoord[idxSorted[i]]))
                    draw.rectangle((u, v, u + 1, v + 1), fill=(lighten_color(r, g, b, 0.1)))



    return img, renderMap, segmentMap, indexMap, mask

def processImage(img, depth, mask, setting="RGBD"):
    # grayScale =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[0], img.shape[1]
    inputX = np.zeros((H, W, config["inputChannel"][setting]))

    if (setting == "D"):
        inputX[:, :, 0] = depth
    if (setting == "RGB"):
        inputX[:, :, :3] = img
    if (setting == "RGBD"):
        inputX[:, :, :3] = img
        inputX[:, :, 3] = depth
    if (setting == "RGBM"):
        inputX[:, :, :3] = img
        inputX[:, :, 3] = mask
    if (setting == "RGBDM"):
        inputX[:, :, :3] = img
        inputX[:, :, 3] = depth
        inputX[:, :, 4] = mask

    return inputX

transform = torchvision.transforms.Compose([
    #torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def getRandomPoint():
    randPoint = random.sample(range(0, config["W"]), 2)
    randPoint = np.array(randPoint)
    v, u = (int)(randPoint[0]), (int)(randPoint[1])
    return v,u

def getRandomPointMask(mask):
    mask = mask.reshape((mask.shape[0]*mask.shape[1]))
    randPoint = np.random.choice(mask, p = mask/np.sum(mask))
    v, u = (int)(randPoint//config["W"]),(int)(randPoint%config["W"])
    return v,u


def getRandomVadlidPointObject(mask,segmentMap,objectID):
    v, u = getRandomPoint()
    while (not mask[v, u] or segmentMap[v, u] != objectID):
        v, u = getRandomPoint()
        #print("stuck ", "getRandomVadlidPointObject")
    return v,u

def getRandomValidPoint(segmentMap,excludeObjectID= -1,dist = 20, refPoint = -1):
    v, u = getRandomPoint()
    if (excludeObjectID!=-1):
        attemp  = 0
        while (segmentMap[v, u] == excludeObjectID or segmentMap[v, u] == -1):
            v, u = getRandomPoint()
            attemp +=1
            if(attemp > 100):
                break
    elif(refPoint!=-1):
        r_v,r_u = refPoint//config["W"], refPoint%config["W"]
        while (np.linalg.norm( np.array([r_v,r_u]) - np.array([v,u]) ) < dist or segmentMap[v, u] == -1):
            v, u = getRandomPoint()

    return v,u

def getRandomBackgroundPoint(mask):
    v, u = getRandomPoint()
    while (mask[v, u]):
        v, u = getRandomPoint()
        #print("stuck ", "getRandomBackgroundPoint")
    return v,u

def noRotate(x, y, W, H):
    return x, y

def rotate_90(x, y, W, H):
    return y, W - 1 - x

def rotate_180(x, y, W, H):
    return W - 1 - x, W - 1 - y

def rotate_270(x, y, W, H):
    return W - 1 - y, x

convertCoordinate = [None, None, None, None]
convertCoordinate[ROTATE_90] = rotate_90
convertCoordinate[ROTATE_180] = rotate_180
convertCoordinate[ROTATE_270] = rotate_270
convertCoordinate[NONE] = noRotate


def validPosition(u,v):
    return 0 < u < config["W"] and 0 < v < config["W"]

class DataGenerator():
    def __init__(self):
        self.modelObjects = []
        self.modelObjectsColor = []
        self.originalModelObjects = []
        self.originalModelObjectsColor = []

        self.originalModelObjects_ITRI = []
        self.originalModelObjectsColor_ITRI = []

    def setOriginalModelObjects (self,_originalModelObjects,_originalModelObjectsColor):
        self.originalModelObjects = _originalModelObjects
        self.originalModelObjectsColor = _originalModelObjectsColor

    def setOriginalModelObjects_ITRI (self,_originalModelObjects_ITRI,_originalModelObjectsColor_ITRI):
        self.originalModelObjects_ITRI = _originalModelObjects_ITRI
        self.originalModelObjectsColor_ITRI = _originalModelObjectsColor_ITRI

    def generateRandomData(self,pathToScense,matchType, sceneType, augmentationType, debug=False,isEvaluate = False,isLoadUnseen=False,isLoadOrig = False):
        #print ("Select",sceneTypeString[sceneType])
        return self.get_single_object_within_scene_data (pathToScense,matchType,sceneType, augmentationType, renderImage=debug, isEvaluate  = isEvaluate,isLoadUnseen = isLoadUnseen,isLoadOrig = isLoadOrig)

    def get_single_object_within_scene_data(self,pathToScense,matchType,sceneType, augmentationType, renderImage=False,isEvaluate = False,isLoadUnseen=False,isLoadOrig = False):
        pathObject_1, listObject_1,idx_1, pathObject_2, listObject_2,idx_2, matchObject,loadITRI = self.getScene(pathToScense,sceneType,isEvaluate = isEvaluate,isLoadUnseen=isLoadUnseen,isLoadOrig = isLoadOrig)
        #print (pathObject_1,idx_1,pathObject_2,idx_2)
        #idx_1,idx_2 = 0,0
        # if(sceneType ==SINGLE_SAME ):
        #     imageColor_1,imageColor_2,matches_a,matches_b,masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b\
        #                                                             = self.single_same_scene(pathObject_1, listObject_1,idx_1, pathObject_2, listObject_2,idx_2, matchObject,renderImage=renderImage)
        # if (sceneType == SINGLE_DIFFERENT):
        #     imageColor_1, imageColor_2,matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b \
        #         = self.single_different_scene(pathObject_1, listObject_1, idx_1, pathObject_2, listObject_2, idx_2,nNoneMatchPoint=2000)
        # #if (sceneType == SINGLE_TO_MULTI or sceneType == MULTI_SAME or sceneType == MULTI_DIFFERENT):
        if (sceneType >1):
            imageColor_1,depth_1, imageColor_2,depth_2,rawData_1,rawData_2,matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b \
                = self.getMatchPoint(pathObject_1, listObject_1, idx_1, pathObject_2, listObject_2, idx_2, matchObject,
                                     renderImage=renderImage,sceneType=sceneType,augmentationType = augmentationType, isEvaluate = isEvaluate,loadITRI = loadITRI)

        nMatchPoint = (matches_a.shape[0])
        #print (nMatchPoint)
        nNonMatchPoint = (masked_non_matches_a.shape[0])
        nBackgroundNoneMatch = (background_non_matches_a.shape[0])
        nBlindNonMatch = (blind_non_matches_a.shape[0])
        totalData = nMatchPoint + nNonMatchPoint + nBackgroundNoneMatch + nBlindNonMatch

        if (sceneType != SINGLE_DIFFERENT):
            #print (nMatchPoint,nBlindNonMatch)
            if (nMatchPoint * nNonMatchPoint * nBackgroundNoneMatch * nBlindNonMatch == 0):
                #print ("RETURN",nMatchPoint , nNonMatchPoint , nBackgroundNoneMatch , nBlindNonMatch)
                return self.get_single_object_within_scene_data (pathToScense,matchType,sceneType, augmentationType, renderImage=renderImage,isEvaluate = isEvaluate,isLoadUnseen=isLoadUnseen)

            if (renderImage ):
                rgb_1, _,_,_,_= tuple(rawData_1)
                rgb_2, _,_,_,_ = tuple(rawData_2)

                #print (renderImage)
                checkMappingImg = self.debugMapping(rgb_1, rgb_2, matches_a, matches_b, nSamplePoint=10,fileName="matchPoint_O2O.jpg")
                checkMappingImg = self.debugMapping(rgb_1, rgb_2, masked_non_matches_a, masked_non_matches_b,nSamplePoint=10, fileName="MaskNonMatch_O2O.jpg")
                checkMappingImg = self.debugMapping(rgb_1, rgb_2, background_non_matches_a,background_non_matches_b, nSamplePoint=10,fileName="BackgroundNonMatch_O2O.jpg")
                checkMappingImg = self.debugMapping(rgb_1, rgb_2, blind_non_matches_a, blind_non_matches_b,nSamplePoint=10, fileName="BlindNonMatch_O2O.jpg")

            #imgA, imgB = None,None
            #imgA = transforms.ToTensor()(imageColor_1).float()
            #imgB = transforms.ToTensor()(imageColor_2).float()

            matches_a = torch.from_numpy(matches_a).view(1, nMatchPoint).long()
            matches_b = torch.from_numpy(matches_b).view(1, nMatchPoint).long()

            masked_non_matches_a = torch.from_numpy(masked_non_matches_a).view(1, nNonMatchPoint).long()
            masked_non_matches_b = torch.from_numpy(masked_non_matches_b).view(1, nNonMatchPoint).long()

            blind_non_matches_a = torch.from_numpy(blind_non_matches_a).view(1, nBlindNonMatch).long()
            blind_non_matches_b = torch.from_numpy(blind_non_matches_b).view(1, nBlindNonMatch).long()

            background_non_matches_a = torch.from_numpy(background_non_matches_a).view(1, nBackgroundNoneMatch).long()
            background_non_matches_b = torch.from_numpy(background_non_matches_b).view(1, nBackgroundNoneMatch).long()

        return imageColor_1,depth_1, imageColor_2,depth_2,rawData_1,rawData_2,matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b, (nMatchPoint), (totalData-nMatchPoint)

    def debugMapping(self,imageColor1, imageColor2, listPointA, listPointB,nSamplePoint = 10, fileName="matchPoint.jpg"):
        H = config["Width"]
        W = config["Height"]
        #imageColor1, imageColor2 = imageColor1.permute(1,2,0).cpu().numpy(), imageColor2.permute(1,2,0).cpu().numpy()
        imageColor1 = np.array(imageColor1*255,dtype=np.uint8)
        imageColor2 = np.array(imageColor2 * 255, dtype=np.uint8)

        debugImage = np.zeros([W,W * 2, 3], dtype=np.uint8)
        debugImage.fill(255)  # or img[:] = 255
        debugImage[:, :W] = imageColor1
        debugImage[:, W:] = imageColor2
        debugImage = Image.fromarray(np.uint8(debugImage))
        draw = ImageDraw.Draw(debugImage)

        selectedPointidx = random.sample(range(0, len(listPointA)), min(len(listPointA),nSamplePoint))

        for selectedPoint in selectedPointidx:
            selectedPointA = listPointA[selectedPoint]
            selectedPointB = listPointB[selectedPoint]
            lv, lu = selectedPointA // W, selectedPointA % W
            lv2, lu2 = selectedPointB // W, selectedPointB % W + W

            draw.rectangle(((lu + 3), lv + 3, lu - 3, lv - 3), fill=(0, 255, 0))
            draw.rectangle(((lu2 + 3), lv2 + 3, lu2 - 3, lv2 - 3), fill=(0, 255, 0))
            draw.line((lu2, lv2, lu, lv), fill=(0, 0, 255), width=3)

        # debugImage = debugImage.resize((config["HalfWidth"], 256))
        cv2.imwrite(fileName , np.array(debugImage))

        return debugImage


    def get_scene_single_same(self):
        idxObject_1, idxObject_2 = -1, -2

        while ( idxObject_1 != idxObject_2):

            idxObject_1 = random.choice(range(28))
            idxObject_2 = random.choice(range(28))

        return idxObject_1, idxObject_2
    def get_scene_single_different(self):
        idxObject_1, idxObject_2 = -1, -1

        while ( idxObject_1 == idxObject_2):

            idxObject_1 = random.choice(range(28))
            idxObject_2 = random.choice(range(28))

        return idxObject_1, idxObject_2

    def getSimScene(self,isEvaluate=False,isSim2SimScene = False,loadITRI = False,isLoadUnseen = False,isLoadOrig = False):
        loadITRI = False
        rd = 1.0
        if (isLoadUnseen):
            return self.getSimSceneTest()

        if isSim2SimScene:
            if(loadITRI):# or random.random() < 0.5 ):
                idxObject = random.choice(range(12, 22))
                idxScene = randint(0, nScene - 1)
                if random.random() < rd:
                    pathScene = "ITRI_train_O2O_sm_lg/Object%d/%03d" % (idxObject, idxScene)
                else:
                    idxObject = random.choice(range(12, 22))
                    pathScene = "ITRI_train_O2O_sm_lg_orig/Object%d/%03d" % (idxObject, idxScene)
                loadITRI = True
            else: #loadGraspNet
                idxObject = 99#random.choice(range(12, 22))
                idxScene = randint(0, nSceneReal - 1)


                if (random.random() < rd):
                  pathScene = "GraspNet_train_O2O_sm_lg_rd/Object%d/%03d" % (idxObject, idxScene)
                else:
                  pathScene = "GraspNet_train_O2O_sm_lg_orig/Object%d/%03d" % (idxObject, idxScene)
        else:  # loadGraspNet
            idxObject = 99# random.choice(range(12, 22))
            idxScene = randint(0, nSceneReal - 1)


            if (random.random() < rd):
                pathScene = "GraspNet_train_O2O_sm_lg_rd/Object%d/%03d" % (idxObject, idxScene)
            else:
                pathScene = "GraspNet_train_O2O_sm_lg_orig/Object%d/%03d" % (idxObject, idxScene)

        return idxObject, idxScene, pathScene, loadITRI



    def getScene(self,path,sceneType = 0,isEvaluate = False,isLoadUnseen=False,isLoadOrig = False):  # return 2 scene that have at least 2 mutual objects
        matchObject = []

        loadITRI = False
        if (sceneType == MULTI_SAME):
            idxObject_1,idxScene_1,pathScene_1,loadITRI = self.getSimScene(isEvaluate=isEvaluate, isSim2SimScene =True)
            idxObject_2, idxScene_2, pathScene_2 = idxObject_1, idxScene_1, pathScene_1
        elif (sceneType == MULTI_DIFFERENT):
            idxObject_1, idxScene_1,pathScene_1,loadITRI = self.getSimScene(isEvaluate=isEvaluate,isSim2SimScene=True,isLoadUnseen = isLoadUnseen)
            idxObject_2, idxScene_2,pathScene_2,loadITRI = self.getSimScene(isEvaluate=isEvaluate,isSim2SimScene=loadITRI,loadITRI = loadITRI,isLoadUnseen = isLoadUnseen)

        pathObject_1 = path +  pathScene_1
        pathObject_2 = path +  pathScene_2

        listObject_1 = np.loadtxt(pathObject_1[:-3] + "listObjects.txt", delimiter=",").astype(int)
        listObject_2 = np.loadtxt(pathObject_2[:-3] + "listObjects.txt", delimiter=",").astype(int)

        #if (idxObject_2<28):
        #    listObject_2 = listObject_2.reshape((listObject_2.shape[0],1))

        set_listObject_1 = set(listObject_1[idxScene_1])
        set_listObject_2 = set(listObject_2[idxScene_2])
        if (-1 in set_listObject_1):
            set_listObject_1.remove(-1)
        if (-1 in set_listObject_2):
            set_listObject_2.remove(-1)
        matchObject = np.array(list((set_listObject_1.intersection(set_listObject_2))))

        idx_1 = randint(0, nImage - 1)
        idx_2 = randint(0, nImage - 1)

        if (isLoadOrig):
            idx_1 = 0# randint(0, 35 - 1)
            idx_2 = 0 #randint(0, 35 - 1)

        if (sceneType == SINGLE_TO_MULTI and len (matchObject) ==0):
            return self.getScene(path,sceneType,isEvaluate = isEvaluate,isLoadUnseen = isLoadUnseen)
        elif ( (sceneType == MULTI_SAME or sceneType == MULTI_DIFFERENT) and len(matchObject) < 2):
            return self.getScene(path, sceneType,isEvaluate = isEvaluate,isLoadUnseen = isLoadUnseen)

        # print (pathObject_1)
        # print (pathObject_2)

        return pathObject_1, listObject_1[idxScene_1],idx_1, pathObject_2, listObject_2[idxScene_2],idx_2, matchObject, loadITRI # selected Object


    def getImageData(self,pathScene, idx):
        poseCamera = np.loadtxt(pathScene + "/frame-%06d.MatrixPoseArm.txt" % (idx))  # 4x4 rigid transformation matrix
        poseCamera = inv(poseCamera)
        imageColor = np.array(Image.open(pathScene + "/%06d_rgb.jpg" % (idx))) 
        imageColor = cv2.cvtColor(imageColor, cv2.COLOR_BGR2RGB) 
        imageColor = imageColor /256
        # mask = np.array(Image.open(pathScene + "/%06d_mask.png" % (idx)))
        depth = cv2.imread(pathScene + "/%06d_depth.png" % (idx), -1).astype(float) / 1000.0

        indexMap = np.loadtxt(pathScene + "/frame-%06d.IndexMap.txt" % (idx), delimiter=",").astype(int)
        segmentMap = np.loadtxt(pathScene + "/frame-%06d.SegmentMap.txt" % (idx), delimiter=",").astype(int)
        mask = np.loadtxt(pathScene + "/frame-%06d.VisibleMask.txt" % (idx), delimiter=",")  # .astype(np.uint8)*255
        uvSegment = np.empty([0,0,0])# np.loadtxt(pathScene + "/frame-%06d.UVSegment.txt" % (idx), delimiter=",")  # .astype(np.uint8)*255

        arrObjectPose = np.loadtxt(pathScene + "/frame-%06d.ObjectPoses.txt" % (idx))  # 4x4 rigid transformation matrix
        arrObjectPose[:4] = 0


        return poseCamera, imageColor, depth, mask, indexMap, segmentMap,uvSegment, arrObjectPose




    def generateSegmentMap(self,listObject, poseCamera, arrObjectPose,renderImage=False,imageIDX=-1,instanceIDX=-1,generateObjectDetection = False):  # return scaleMap,sementMap, indexMap, mask
        dict_annotation = []
        img = None
        #arrObj = arrObjectPose[:4].reshape(16)

        if (renderImage):
            img = np.zeros([config["W"], config["W"], 3], dtype=np.uint8) + 128
            img = Image.fromarray(np.uint8(img))

        segmentMap = np.zeros([config["W"], config["W"]], dtype="int") -1
        renderMap = np.ones([config["W"], config["W"]], dtype="int") * 1000
        indexMap = np.zeros([config["W"], config["W"]], dtype="int") - 1
        mask = np.zeros([config["W"], config["W"]], dtype="int")

        for i in range(len(listObject)):
            modelID = (int)(listObject[i])
            selectedModel = self.originalModelObjects[modelID]
            selectedModelColor = self.originalModelObjectsColor[modelID]

            objPose = arrObjectPose[(i + 1) * 4:(i + 2) * 4]
            uvCoord = project(poseCamera, objPose, selectedModel,generateObjectDetection)

            minX, maxX = np.min(uvCoord[:, 0]), np.max(uvCoord[:, 0])
            minY, maxY = np.min(uvCoord[:, 1]), np.max(uvCoord[:, 1])
            minX = max(0, min(config["W"], minX))
            minY = max(0, min(config["W"], minY))
            maxX = max(0, min(config["W"], maxX))
            maxY = max(0, min(config["W"], maxY))
            lX = abs(minX - maxX)
            lY = abs(minY - maxY)
            area = lX * lY
            # atonation
            atonation = {
                "area": float(area),
                "iscrowd": 0,
                "image_id": imageIDX,
                "bbox": [minX, minY, lX, lY],
                "category_id": int(modelID),
                "id": instanceIDX
            }
            dict_annotation.append(atonation)
            instanceIDX+=1

            if(not generateObjectDetection or renderImage):
                img, renderMap, segmentMap, indexMap, mask = render(uvCoord, selectedModelColor, img, renderMap, segmentMap,
                                                                indexMap, mask, modelID, renderImage=renderImage)

        return img, renderMap, segmentMap, indexMap, mask,dict_annotation,instanceIDX

    def getBackgroundNonMatch(self,segmentMap_1,segmentMap_2,nNoneMatchPoint=1000):
        W = config["W"]
        blind_non_matches_a, blind_non_matches_b = [],[]
        for i in range(nNoneMatchPoint):
            randPointA = random.sample(range(0, W), 2)
            randPointA = np.array(randPointA)
            randPointB = random.sample(range(0, W), 2)
            randPointB = np.array(randPointB)

            # while (np.linalg.norm(randPointA - randPointB) < 5*MIN_DISTANCE_INSIDE):
            #while (segmentMap_1[randPointA[0], randPointA[1]]!=-1 or segmentMap_2[randPointB[0], randPointB[1]] != -1):
            while (segmentMap_1[randPointA[0], randPointA[1]] == segmentMap_2[randPointB[0], randPointB[1]]
                    or
                   (segmentMap_1[randPointA[0], randPointA[1]] == -1 and  segmentMap_2[randPointB[0], randPointB[1]] ==-1)
                   ):
                randPointA = random.sample(range(0, W), 2)
                randPointA = np.array(randPointA)
                randPointB = random.sample(range(0, W), 2)
                randPointB = np.array(randPointB)

            randPointA = randPointA[0] * W + randPointA[1]
            randPointB = randPointB[0] * W + randPointB[1]
            blind_non_matches_a.append(randPointA)
            blind_non_matches_b.append(randPointB)
        return  blind_non_matches_a,blind_non_matches_b

    def getMatchPoint(self,pathObject_1, listObject_1, idx_1, pathObject_2, listObject_2, idx_2, matchObject,nNoneMatchPoint=1000,nMatchPoint = 2000, renderImage=False,sceneType =  0,augmentationType=0,isEvaluate = False,loadITRI = False):
        H = config["Width"]
        W = config["Height"]

        if(sceneType < 5):
            poseCamera_1, imageColor_1, depth_1, mask_1, indexMap_1, segmentMap_1,uvSegment_1, arrObjectPose_1 = self.getImageData(pathObject_1, idx_1)
            poseCamera_2, imageColor_2, depth_2, mask_2, indexMap_2, segmentMap_2,uvSegment_2, arrObjectPose_2 = self.getImageData(pathObject_2, idx_2)

        #print (augmentationType)
        if (augmentationType == ROTATE_90):
            imageColor_2 = cv2.rotate(imageColor_2, cv2.ROTATE_90_CLOCKWISE)
            mask_2 = cv2.rotate(mask_2, cv2.ROTATE_90_CLOCKWISE)
            depth_2 = cv2.rotate(depth_2, cv2.ROTATE_90_CLOCKWISE)
            indexMap_2 = cv2.rotate(indexMap_2, cv2.ROTATE_90_CLOCKWISE)
            segmentMap_2 = cv2.rotate(segmentMap_2, cv2.ROTATE_90_CLOCKWISE)
        elif (augmentationType == ROTATE_180):
            imageColor_2 = cv2.rotate(imageColor_2, cv2.ROTATE_180)
            mask_2 = cv2.rotate(mask_2, cv2.ROTATE_180)
            depth_2 = cv2.rotate(depth_2, cv2.ROTATE_180)
            indexMap_2 = cv2.rotate(indexMap_2, cv2.ROTATE_180)
            segmentMap_2 = cv2.rotate(segmentMap_2, cv2.ROTATE_180)
        elif (augmentationType == ROTATE_270):
            imageColor_2 = cv2.rotate(imageColor_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask_2 = cv2.rotate(mask_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            depth_2 = cv2.rotate(depth_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            indexMap_2 = cv2.rotate(indexMap_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            segmentMap_2 = cv2.rotate(segmentMap_2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        rawData_1 = (imageColor_1, depth_1, mask_1, indexMap_1, segmentMap_1)
        rawData_2 = (imageColor_2, depth_2, mask_2, indexMap_2, segmentMap_2)


        listObject_1 = remove_values_from_list(listObject_1,-1)
        listObject_2 = remove_values_from_list(listObject_2, -1)

        lenSet_1 = len(set(listObject_1))
        lenSet_2 = len(set(listObject_2))

        #print (listObject_1,listObject_2)

        matches_a = []
        matches_b = []
        masked_non_matches_a, masked_non_matches_b, = [],[]
        background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b = [], [], [], []

        nMatchPointEachObject = nMatchPoint // max(len(listObject_1), len(listObject_2))
        #print (nMatchPointEachObject,"points")

        attemp = 0
        #print (matchObject)
        while ( len(matches_a) < nMatchPoint):
            #print ( len(matches_a))
            attemp+=1
            if (attemp>100): #map 100 objects
                break
            for modelID in matchObject:
                #print (modelID)
                # modelID = random.choice(matchObject)
                listMatchObject_1 = np.where(listObject_1 == modelID)[0]
                listMatchObject_2 = np.where(listObject_2 == modelID)[0]

                if (isEvaluate):
                    if (len(listMatchObject_1) >1 or len (listMatchObject_2)>1):
                        continue

                for idxObject_1 in listMatchObject_1:
                    for idxObject_2 in listMatchObject_2:
                        if loadITRI :
                            idx = random.sample(range(self.originalModelObjects_ITRI[modelID].shape[0]), 100)
                            selectedModel = self.originalModelObjects_ITRI[modelID][idx]
                        else:
                            idx = random.sample(range(self.originalModelObjects[modelID].shape[0]), 100)
                            selectedModel = self.originalModelObjects[modelID][idx]

                        idx_1 = idxObject_1
                        idx_2 = idxObject_2

                        objPose_1 = arrObjectPose_1[(idx_1 + 1) * 4:(idx_1 + 2) * 4]
                        objPose_2 = arrObjectPose_2[(idx_2 + 1) * 4:(idx_2 + 2) * 4]

                        if (sceneType < 5):
                            uvCoord_1 = project(poseCamera_1, objPose_1, selectedModel,loadITRI)
                            uvCoord_2 = project(poseCamera_2, objPose_2, selectedModel,loadITRI)


                        nVertex = selectedModel.shape[0]

                        tempMappping = []
                        coutPointMatchingObject = 0

                        for vertexID in range(nVertex):
                            u_1, v_1 = uvCoord_1[vertexID, 0], uvCoord_1[vertexID, 1]
                            u_2, v_2 = uvCoord_2[vertexID, 0], uvCoord_2[vertexID, 1]
                            u_1, v_1, u_2, v_2 = (int)(u_1), (int)(v_1), (int)(u_2), (int)(v_2)

                            v_2, u_2 = convertCoordinate[augmentationType](v_2, u_2, W, H)

                            if (validPosition(u_1,v_1)):
                                if (validPosition(u_2,v_2)):
                                    #print(segmentMap_1[v_1, u_1], segmentMap_2[v_2, u_2], mask_2[v_2, u_2],mask_1[v_1, u_1])
                                    if (segmentMap_1[v_1, u_1]== segmentMap_2[v_2, u_2]  and  segmentMap_2[v_2, u_2]  == modelID and mask_2[v_2, u_2] and mask_1[v_1, u_1]):

                                        if loadITRI:
                                            p3D_A = self.originalModelObjects_ITRI[modelID][indexMap_1[v_1, u_1]]
                                            p3D_B = self.originalModelObjects_ITRI[modelID][indexMap_2[v_2, u_2]]
                                        else :
                                            p3D_A = self.originalModelObjects[modelID][indexMap_1[v_1, u_1]]
                                            p3D_B = self.originalModelObjects[modelID][indexMap_2[v_2, u_2]]

                                        l = np.linalg.norm(p3D_A - p3D_B)


                                        #if (indexMap_1[v_1, u_1] == indexMap_2[v_2, u_2] or sceneType == 7 or sceneType == 5):
                                        #if (indexMap_1[v_1, u_1] == indexMap_2[ v_2, u_2]):
                                        if (l < 0.05):
                                            # MATCH POINTS
                                            pointA = v_1*W + u_1
                                            pointB = v_2 * W + u_2
                                            matches_a.append(pointA)
                                            matches_b.append(pointB)
                                            coutPointMatchingObject +=1

                                            #NONMATCH MASKED OBJECT
                                            # object to background none match
                                            for _ in range (2):
                                                t_v_1, t_u_1 = getRandomBackgroundPoint(mask_1)
                                                background_non_matches_a.append(t_v_1*W + t_u_1)
                                                background_non_matches_b.append(pointB)

                                            # object to background none match
                                            for _ in range(2):
                                                t_v_2, t_u_2 = getRandomBackgroundPoint(mask_2)
                                                background_non_matches_a.append(pointA)
                                                background_non_matches_b.append(t_v_2*W + t_u_2 )

                                            # object to object mask los
                                            if (lenSet_2 > 1):
                                                for k in range(2):
                                                    
                                                    t_v_2, t_u_2 = getRandomValidPoint(segmentMap_2,dist = 10,refPoint = pointB)
                                                    masked_non_matches_a.append(pointA)
                                                    masked_non_matches_b.append(t_v_2 * W + t_u_2)
                                                     
                                                   
                                                    t_v_2, t_u_2 = getRandomValidPoint(segmentMap_2, excludeObjectID= modelID)
                                                    masked_non_matches_a.append(pointA)
                                                    masked_non_matches_b.append(t_v_2 * W + t_u_2)

                                            if (lenSet_1>1):
                                                for _ in range(2):
                                                    
                                                    t_v_1, t_u_1 = getRandomValidPoint(segmentMap_1,dist = 10,refPoint = pointA)
                                                    masked_non_matches_a.append(t_v_1 * W + t_u_1)
                                                    masked_non_matches_b.append(pointB)
                                                    

                                                    t_v_1, t_u_1 = getRandomValidPoint(segmentMap_1,  excludeObjectID=modelID)
                                                    masked_non_matches_a.append(t_v_1 * W + t_u_1)
                                                    masked_non_matches_b.append(pointB)
                            if (coutPointMatchingObject > nMatchPointEachObject):
                                break

        matches_a = np.array(matches_a)
        matches_b = np.array(matches_b)
        # reduce mapping map
        if (matches_a.shape[0] > 0):
            idx = random.sample(range(matches_a.shape[0]), min(matches_a.shape[0], nMatchPoint))
            matches_a = matches_a[idx]
            matches_b = matches_b[idx]


        # background to background
        blind_non_matches_a,blind_non_matches_b = self.getBackgroundNonMatch(segmentMap_1,segmentMap_2)

        masked_non_matches_a = np.array(masked_non_matches_a)
        masked_non_matches_b = np.array(masked_non_matches_b)
        background_non_matches_a = np.array(background_non_matches_a)
        background_non_matches_b = np.array(background_non_matches_b)
        blind_non_matches_a = np.array(blind_non_matches_a)
        blind_non_matches_b = np.array(blind_non_matches_b)
        masked_non_matches_a = np.array(masked_non_matches_a)
        masked_non_matches_b = np.array(masked_non_matches_b)

        #should redude others??

        idx = random.sample(range(masked_non_matches_a.shape[0]), min(masked_non_matches_a.shape[0], 2000))
        masked_non_matches_a, masked_non_matches_b = masked_non_matches_a[idx], masked_non_matches_b [idx]

        idx = random.sample(range(masked_non_matches_a.shape[0]), min(masked_non_matches_a.shape[0], 5000))
        masked_non_matches_a, masked_non_matches_b = masked_non_matches_a[idx], masked_non_matches_b[idx]

        idx = random.sample(range(background_non_matches_a.shape[0]), min(background_non_matches_a.shape[0], 5000))
        background_non_matches_a, background_non_matches_b = background_non_matches_a[idx], background_non_matches_b[idx]

        idx = random.sample(range(blind_non_matches_a.shape[0]), min(blind_non_matches_a.shape[0], 1000))
        blind_non_matches_a, blind_non_matches_b = blind_non_matches_a[idx], blind_non_matches_b[idx]


        checkMappingImg = None

        #imageColor_1 = processImage(imageColor_1, depth_1, mask_1, config["inputMode"])
        #imageColor_2 = processImage(imageColor_2, depth_2, mask_2, config["inputMode"])
        imageColor_1 = transforms.ToTensor()(imageColor_1).float()
        imageColor_2 = transforms.ToTensor()(imageColor_2).float()
        imageColor_1 = transform(imageColor_1)
        imageColor_2 = transform(imageColor_2)
        depth_1 = transforms.ToTensor()(depth_1).float()
        depth_2 = transforms.ToTensor()(depth_2).float()

        '''
        if (random.choice([True, False])):
            imageColor_2[:, :, 0] = imageColor_2[:, :, 0] + np.multiply(1 - mask_2, np.random.rand(256, 256) / 10)

        if (random.choice([True, False])):
            imageColor_1[:, :, 0] = imageColor_1[:, :, 0] + np.multiply(1 - mask_1, np.random.rand(256, 256) / 10)
        '''

        return imageColor_1,depth_1,imageColor_2,depth_2,rawData_1,rawData_2, matches_a,matches_b,masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b


    def single_same_scene(self,pathObject_1, listObject_1, idx_1, pathObject_2, listObject_2, idx_2,nNoneMatchPoint=1000,nMatchPoint = 100, renderImage=False):
        H = config["Width"]
        W = config["Height"]
        poseCamera_1, imageColor_1, depth_1, mask_1, indexMap_1, segmentMap_1, uvSegment_1, arrObjectPose_1 = self.getImageData(
            pathObject_1, idx_1)
        poseCamera_2, imageColor_2, depth_2, mask_2, indexMap_2, segmentMap_2, uvSegment_2, arrObjectPose_2 = self.getImageData(
            pathObject_2, idx_2)


        matches_a,matches_b = [],[]
        masked_non_matches_a, masked_non_matches_b, = [], []
        background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b = [], [], [], []

        nMatchPointEachObject = nMatchPoint // max(len(listObject_1), len(listObject_2))

        attemp = 0

        while (len(matches_a) < nMatchPoint):
            attemp += 1
            if (attemp > 100):  # map 100 objects
                break
            modelID = listObject_1[0]

            idx = random.sample(range(self.originalModelObjects[modelID].shape[0]), nMatchPoint*5)
            selectedModel = self.originalModelObjects[modelID][idx]

            idx_1 = 0
            idx_2 = 0

            objPose_1 = arrObjectPose_1[(idx_1 + 1) * 4:(idx_1 + 2) * 4]
            uvCoord_1 = project(poseCamera_1, objPose_1, selectedModel)

            objPose_2 = arrObjectPose_2[(idx_2 + 1) * 4:(idx_2 + 2) * 4]
            uvCoord_2 = project(poseCamera_2, objPose_2, selectedModel)

            nVertex = selectedModel.shape[0]

            tempMappping = []
            coutPointMatchingObject = 0
            for vertexID in range(nVertex):
                u_1, v_1 = uvCoord_1[vertexID, 0], uvCoord_1[vertexID, 1]
                u_2, v_2 = uvCoord_2[vertexID, 0], uvCoord_2[vertexID, 1]
                u_1, v_1, u_2, v_2 = (int)(u_1), (int)(v_1), (int)(u_2), (int)(v_2)

                if (validPosition(u_1, v_1)):
                    if (validPosition(u_2, v_2)):
                        if (mask_2[v_2, u_2] and mask_1[v_1, u_1]):
                            if (indexMap_1[v_1, u_1] == indexMap_2[v_2, u_2]):
                                # MATCH POINTS
                                pointA = v_1 * W + u_1
                                pointB = v_2 * W + u_2
                                matches_a.append(pointA)
                                matches_b.append(pointB)
                                coutPointMatchingObject+=1

                                # NONMATCH MASKED OBJECT
                                # object to background none match
                                for _ in range(2):
                                    t_v_1, t_u_1 = getRandomBackgroundPoint(mask_1)
                                    background_non_matches_a.append(t_v_1 * W + t_u_1)
                                    background_non_matches_b.append(pointB)

                                # object to background none match
                                for _ in range(2):
                                    t_v_2, t_u_2 = getRandomBackgroundPoint(mask_2)
                                    background_non_matches_a.append(pointA)
                                    background_non_matches_b.append(t_v_2 * W + t_u_2)

                                # object to object mask los
                                for k in range(2):
                                    t_v_2, t_u_2 = getRandomValidPoint(segmentMap_2, dist=5, refPoint=pointB)
                                    masked_non_matches_a.append(pointA)
                                    masked_non_matches_b.append(t_v_2 * W + t_u_2)

                                for _ in range(2):
                                    t_v_1, t_u_1 = getRandomValidPoint(segmentMap_1, dist=5, refPoint=pointA)
                                    masked_non_matches_a.append(t_v_1 * W + t_u_1)
                                    masked_non_matches_b.append(pointB)
                if (coutPointMatchingObject > nMatchPoint):
                    break

        matches_a = np.array(matches_a)
        matches_b = np.array(matches_b)
        # reduce mapping map
        if (matches_a.shape[0] > 0):
            idx = random.sample(range(matches_a.shape[0]), min(matches_a.shape[0], nMatchPoint))
            matches_a = matches_a[idx]
            matches_b = matches_b[idx]

        # background to background
        blind_non_matches_a, blind_non_matches_b = self.getBackgroundNonMatch(segmentMap_1, segmentMap_2)

        masked_non_matches_a = np.array(masked_non_matches_a)
        masked_non_matches_b = np.array(masked_non_matches_b)
        background_non_matches_a = np.array(background_non_matches_a)
        background_non_matches_b = np.array(background_non_matches_b)
        blind_non_matches_a = np.array(blind_non_matches_a)
        blind_non_matches_b = np.array(blind_non_matches_b)

        # should redude others??
        '''
        idx = random.sample(range(masked_non_matches_a.shape[0]), min(masked_non_matches_a.shape[0], 2000))
        masked_non_matches_a, masked_non_matches_b = masked_non_matches_a[idx], masked_non_matches_b [idx]

        idx = random.sample(range(masked_non_matches_a.shape[0]), min(masked_non_matches_a.shape[0], 2000))
        masked_non_matches_a, masked_non_matches_b = masked_non_matches_a[idx], masked_non_matches_b[idx]

        idx = random.sample(range(background_non_matches_a.shape[0]), min(background_non_matches_a.shape[0], 2000))
        background_non_matches_a, background_non_matches_b = background_non_matches_a[idx], background_non_matches_b[idx]

        idx = random.sample(range(blind_non_matches_a.shape[0]), min(blind_non_matches_a.shape[0], 2000))
        blind_non_matches_a, blind_non_matches_b = blind_non_matches_a[idx], blind_non_matches_b[idx]
        '''

        imageColor_1 = processImage(imageColor_1, depth_1, mask_1, config["inputMode"])
        imageColor_2 = processImage(imageColor_2, depth_2, mask_2, config["inputMode"])

        if (random.choice([True, False])):
            imageColor_2[:, :, 3] = imageColor_2[:, :, 3] + np.multiply(1 - mask_2, np.random.rand(256, 256) / 10)

        if (random.choice([True, False])):
            imageColor_1[:, :, 3] = imageColor_1[:, :, 3] + np.multiply(1 - mask_1, np.random.rand(256, 256) / 10)

        return imageColor_1, imageColor_2,mask_1,mask_2, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b


    def single_different_scene (self,pathObject_1, listObject_1, idx_1, pathObject_2, listObject_2, idx_2,nNoneMatchPoint=5000):
        H = config["Width"]
        W = config["Height"]
        poseCamera_1, imageColor_1, depth_1, mask_1, indexMap_1, segmentMap_1, uvSegment_1, arrObjectPose_1 = self.getImageData( pathObject_1, idx_1)
        poseCamera_2, imageColor_2, depth_2, mask_2, indexMap_2, segmentMap_2, uvSegment_2, arrObjectPose_2 = self.getImageData( pathObject_2, idx_2)

        matches_a,matches_b = [],[]
        masked_non_matches_a, masked_non_matches_b, = [], []
        background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b = [], [], [], []

        idx_1 = 0
        idx_2 = 0
        selectedModel_1 = self.originalModelObjects[listObject_1[0] ]
        selectedModel_2 = self.originalModelObjects[listObject_2[0]]

        objPose_1 = arrObjectPose_1[(idx_1 + 1) * 4:(idx_1 + 2) * 4]
        uvCoord_1 = project(poseCamera_1, objPose_1, selectedModel_1)

        objPose_2 = arrObjectPose_2[(idx_2 + 1) * 4:(idx_2 + 2) * 4]
        uvCoord_2 = project(poseCamera_2, objPose_2, selectedModel_2)

        nVertex_1 = selectedModel_1.shape[0]
        nVertex_2 = selectedModel_2.shape[0]
        start_time = time.time()
        blind_non_matches_a, blind_non_matches_b = self.getBackgroundNonMatch(segmentMap_1, segmentMap_2,nNoneMatchPoint = nNoneMatchPoint)
        end_time = time.time()
        idx_1s = random.sample(range(nVertex_1),nNoneMatchPoint)
        idx_2s = random.sample(range(nVertex_2),nNoneMatchPoint)

        for i in range (nNoneMatchPoint):
            u_1,v_1 = int(uvCoord_1[idx_1s[i],0]),int(uvCoord_1[idx_1s[i],1])
            u_2, v_2 = int(uvCoord_2[idx_2s[i],0]),int(uvCoord_2[idx_2s[i],1])

            pointA = v_1*W + u_1
            pointB = v_2 * W + u_2
            if (validPosition(v_1,u_1) and validPosition(v_2,u_2)):
                blind_non_matches_a.append(pointA)
                blind_non_matches_b.append(pointB)

        blind_non_matches_a = np.array(blind_non_matches_b)
        blind_non_matches_b = np.array(blind_non_matches_b)

        matches_a,matches_b,masked_non_matches_a,masked_non_matches_b,background_non_matches_a,background_non_matches_b = empty_tensor(), empty_tensor(), empty_tensor(), empty_tensor(), empty_tensor(), empty_tensor()

        imageColor_1 = processImage(imageColor_1, depth_1, mask_1, config["inputMode"])
        imageColor_2 = processImage(imageColor_2, depth_2, mask_2, config["inputMode"])

        if (random.choice([True, False])):
            imageColor_2[:, :, 3] = imageColor_2[:, :, 3] + np.multiply(1 - mask_2, np.random.rand(256, 256) / 10)

        if (random.choice([True, False])):
            imageColor_1[:, :, 3] = imageColor_1[:, :, 3] + np.multiply(1 - mask_1, np.random.rand(256, 256) / 10)

        return imageColor_1, imageColor_2,mask_1,mask_2, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b
