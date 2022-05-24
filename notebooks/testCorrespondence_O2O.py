from numpy.linalg import inv
import random
from random import randint
from colorsys import rgb_to_hls, hls_to_rgb
import scipy.io
import cv2
import numpy as np
import scipy.io
from PIL import Image, ImageDraw, ImageFont
import time
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(1, '..')
from DON_Training.DataGenerator_O2O import DataGenerator,processImage, convertToHeatmap
from ITRIP.Configuration import *
import matplotlib.pyplot as plt
import DON_Training.evaluation.plotting as dc_plotting
from PIL import Image, ImageDraw, ImageFont
from DON_Training.DensObjectNet import DensObjectNet
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from DON_Training.dense_correspondence_network import DenseCorrespondenceNetwork

from numpy import linalg as LA


def getDiffDist(p1, p2):
    maxDist = np.linalg.norm(np.zeros(2) - np.array([config["W"], config["W"]]))
    currDist = np.linalg.norm(np.array(p2) - np.array(p1))
    return (currDist / maxDist)


def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()


def draw_circle(event, x, y, flags, param):
    global p, g_2_p
    global lu2, lv2
    global heatmap
    if event == cv2.EVENT_MOUSEMOVE:
        y  -= config["Width"]
        #print (x,y)
        if (x < config["W"]):
            matchPoint, _ = DON.getBestMatchPointOnly((x, y), descrtion1, descrtion2, mask2)

            lu2, lv2 = x, y-config["Width"]

            p_lu2, p_lv2 = tuple(matchPoint)
            p_lu2, p_lv2 = p_lu2 + config["Width"], p_lv2

            heatmap = -LA.norm(descrtion2 - descrtion1[y, x].reshape(1, 1, config["descriptor_dimension"]), axis=2)# * 255

            # heatmap = heatmap * 1.3

            heatmap *= mask2
            heatmap = convertToHeatmap(heatmap)
            heatmap = np.array(heatmap)

            #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(heatmap*mask2.reshape(256, 256, 1)  / 255, 0.8, imageColor2 / 255, 0.2, 0)

            dc_plotting.draw_reticle_cv2(heatmap, lu2 - 256, lv2, (0, 0, 0))

            #print (heatmap.shape)

        try:
            g_2_p = np.array([lu2, lv2, p_lu2, p_lv2])
        except:
            raise ("")

p = np.zeros(4)
g_2_p = np.zeros(4)
setting = "RGBD"
config["model"] = "Resnet"
model_path = "CODs_GraspNet_RGBD_Resnet8/DON_1001"
#"DON_Res348_cross_RGBD_Resnet_orig"
DON = DensObjectNet(setting=setting, pretrained=model_path)


FOLDER_NAME1=  "../DON_data/" + "GraspNet_train_O2O_sm_lg_rd/Object99/000"
FOLDER_NAME2= "../DON_data/" + "GraspNet_train_O2O_sm_lg_rd/Object99/002"

idx2 = 0
idx1 = 0


dataGenerator = DataGenerator()

#if 1 is SIM
#poseCamera1, imageColor1, depth1, mask1, indexMap1, segmentMap1,uvSeg_1, arrObjectPose1 = dataGenerator.getImageData(FOLDER_NAME1, idx1)
##if 1 is REAL
poseCamera1, imageColor1, depth1, mask1, indexMap1, segmentMap1,uvSeg_1, arrObjectPose1 = dataGenerator.getImageData(FOLDER_NAME1, idx1)
poseCamera2, imageColor2, depth2, mask2, indexMap2, segmentMap2,uvSeg_2, arrObjectPose2 = dataGenerator.getImageData(FOLDER_NAME2, idx2)



descrtion1 = DON.getDescriptor(imageColor1,depth1)
descrtion2 = DON.getDescriptor(imageColor2,depth2)

imageColor1 = np.array(imageColor1*255).astype(np.uint8)
imageColor2 = np.array(imageColor2*255).astype(np.uint8)

descrtion1 = descrtion1.detach().cpu().numpy()
descrtion2 = descrtion2.detach().cpu().numpy()


normDes1 = dc_plotting.normalize_descriptor(descrtion1 * mask1.reshape(256, 256, 1))
normDes2 = dc_plotting.normalize_descriptor(descrtion2 * mask2.reshape(256, 256, 1))

normDes1,normDes2= dc_plotting.normalize_descriptor_pair(descrtion1*mask1.reshape(256,256,1),descrtion2*mask2.reshape(256,256,1))

#normDes1,normDes1= dc_plotting.normalize_descriptor_pair(descrtion1*mask1.reshape(256,256,1),descrtion1*mask1.reshape(256,256,1))

normDes2 =  normDes2[:, :, :3]
normDes1 =  normDes1[:, :, :3]


cv2.namedWindow('video')
cv2.setMouseCallback('video', draw_circle)

lu2, lv2 = -1,-1
heatmap = np.zeros([256, 256, 3])
idx = 0
while True:
    img = np.zeros([config["W"], config["W"] * 3 , 3], dtype=np.uint8)
    desImg = np.zeros([config["W"], config["W"] * 2, 3], dtype=np.uint8)

    img[:, config["W"]:config["W"] * 2] = imageColor2# np.array([mask2,mask2,mask2]).transpose(1,2,0)*255
    img[:, : config["W"]] = imageColor1

    #desImg[:, config["W"]:config["W"] * 2] =  normDes2[:, :, :3] * 255* mask2.reshape(256, 256, 1)
    #desImg[:, : config["W"]] = normDes1[:, :, :3] * 255* mask1.reshape(256, 256, 1)

    desImg[:, config["W"]:config["W"] * 2] =  normDes2*mask2.reshape(256, 256, 1)*255
    desImg[:, : config["W"]] = normDes1* mask1.reshape(256, 256, 1)*255


    consistentImage = np.zeros([config["W"] * 2, config["W"] * 2, 3], dtype=np.uint8)
    consistentImage[:config["W"]] = desImg
    consistentImage[config["W"]:] = img[:, :config["W"] * 2]

    img = Image.fromarray(np.uint8(img))
    draw = ImageDraw.Draw(img)

     # lv,lu = rotate_90(lv,lu,256,256)
    p_lu2, p_lv2 = g_2_p[2], g_2_p[3]

    #lu2 = lu2 + config["Width"]
    draw.rectangle(((lu2 + 5), lv2 + 5+config["W"], lu2 - 5, lv2 - 5+config["W"]), fill=(0, 255, 0))
    draw.rectangle(((p_lu2 + 3), p_lv2 + 3, p_lu2 - 3, p_lv2 - 3), fill=(255, 0, 0))

    draw.line(( lu2, lv2+config["W"],p_lu2, p_lv2), fill=(0, 0, 255), width=3)

    img = np.array(img)
    #print (heatmap.shape)
    img[:, config["W"] * 2:] = np.array(heatmap* 255)
    dc_plotting.draw_reticle_cv2(img, (int)(p_lu2 + 255), int(p_lv2), (0, 255, 0))
    if cv2.waitKey(1) == 27:
        break



    mergedImg = np.ones([config["W"]*2, config["W"] * 3, 3], dtype=np.uint8)*255
    mergedImg[:config["W"], :config["W"] * 2]  = desImg
    mergedImg[config["W"]:config["W"]*2]= img

    #cv2.imwrite('video/matching_%06d.jpg'%(idx),mergedImg)
    #idx+=1
    #cv2.imshow('video', img)
    #cv2.imshow('des', desImg)

    if cv2.waitKey(1) == ord('f'):
        cv2.imwrite('DON_Result_des/'+model_path+'_matchingIndicate1_rd.png', img)
        cv2.imwrite('DON_Result_des/'+model_path+'_consistentClutteredDesctiptor1_rd.png', consistentImage)
        cv2.imwrite("DON_Result_des/'+model_path+'_mergedImag2e%03d_rd.jpg"%(idx),mergedImg)
        print ("Saved")
        idx+=1
    cv2.imshow("video",mergedImg)

cv2.destroyAllWindows()