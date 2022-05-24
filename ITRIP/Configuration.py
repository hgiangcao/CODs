import numpy as np
import yaml
from yaml import CLoader
import random
import time
#random.seed(time.time())
import sys
sys.path.insert(1, '..')
config= {}
config =  yaml.load(open("../ITRIP/config.yml"), Loader=CLoader)


#CONSTANT DEFINE
SINGLE_OBJECT_WITHIN_SCENE = 0
SINGLE_OBJECT_ACROSS_SCENE = 1
DIFFERENT_OBJECT = 2
MULTI_OBJECT = 3
SYNTHETIC_MULTI_OBJECT = 4

#NEW SCENE CONFIG

#get_different_object_loss
SINGLE_SAME = 0 #DONE
SINGLE_DIFFERENT = 1 #DONE

#get_within_scene_loss
SINGLE_TO_MULTI = 2
MULTI_DIFFERENT = 3
MULTI_SAME = 4
MULTI_DIFFERENT_REAL = 5
MULTI_SAME_REAL = 6
MULTI_SIM_REAL = 7
UNSEEN = 8
UNSEEN_ITRI = 9

sceneTypeString=["SINGLE_SAME","SINGLE_DIFFERENT","SINGLE_TO_MULTI","MULTI_DIFFERENT","MULTI_SAME","MULTI_DIFFERENT_REAL","MULTI_SAME_REAL","MULTI_SIM_REAL","UNSEEN","UNSEEN_ITRI","UNSEEN_GraspNet_SimSim_Random","UNSEEN_GraspNet_RealReal","UNSEEN_GraspNet_SimReal_Random"]


graspnet_splits = {'demo':[0,1,2,3],'train': [0, 2, 5, 7, 8, 9, 11, 14, 15, 17, 18, 20, 21, 22, 26, 27, 29, 30, 34, 36, 37, 38, 40, 41, 43, 44, 46, 48, 51, 52, 56, 57, 58, 60, 61, 62, 63, 66, 69, 70], 'test': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 38, 39, 41, 42, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77,
                                                                                                                                                                                                 78, 79, 80, 81, 82, 83, 84, 85, 86, 87], 'test_novel': [3, 4, 6, 17, 19, 20, 24, 26, 27, 28, 30, 31, 32, 33, 35, 45, 47, 49, 51, 55, 62, 63, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87], 'test_seen': [0, 2, 5, 7, 8, 9, 11, 14, 17, 18, 20, 21, 22, 26, 27, 29, 30, 38, 41, 48, 51, 52, 58, 60, 61, 62, 63, 66], 'test_similar': [1, 3, 4, 6, 10, 12, 13, 16, 19, 23, 25, 35, 39, 42, 50, 53, 54, 59, 64, 65, 67, 68]}
graspnet_obj_to_exclude = [9, 18, 19, 20, 26, 27, 28, 29, 30, 31, 33, 34, 43, 48, 50, 53, 54, 55, 56, 67, 70, 72, 74, 75, 77, 78, 79, 80, 81, 83, 85, 86]


graspnet_train = graspnet_splits["train"]
graspnet_train = set(graspnet_train) - set(graspnet_obj_to_exclude)
graspnet_train = list(graspnet_train)



#augementation
NONE = 0
ROTATE_90 = 1
ROTATE_180 = 2
ROTATE_270 = 3

#terminal CODE
DONE_CRASH = 1
DONE_FINISH = 2
DONE_EXCEEDED_MAX_ACTION = 3
DONE_NONE = 0
DONE_OUT_AREA = 4

#valid area
#VALID_AREA = [32, 478, 110, 400]
MIN_VALID_X = 32//2
MAX_VALID_X = 478//2
MIN_VALID_Y = 110//2
MAX_VALID_Y = 400//2


X=0
Y = 1
Z = 2
scaleCmToPx = (11.9/100 * (0.13315-0.025748) /0.13315)
print ("scaleCmToPx",scaleCmToPx)

import cv2


if __name__ == "__main__":
    print (config)
