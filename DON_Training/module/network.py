import sys, os
import numpy as np
import warnings
import logging
import resnet_dilated as resnet_dilated
import math
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


class DenseCorrespondenceNetwork(nn.Module):
    IMAGE_TO_TENSOR = valid_transform = transforms.Compose([transforms.ToTensor(), ])

    def __init__(self, fcn=None, descriptor_dimension=3, image_width=640,
                 image_height=480, normalize=False):
        """

        :param fcn:
        :type fcn:
        :param descriptor_dimension:
        :type descriptor_dimension:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :param normalize: If True normalizes the feature vectors to lie on unit ball
        :type normalize:
        """

        super(DenseCorrespondenceNetwork, self).__init__()

        self._fcn = fcn
        self._descriptor_dimension = descriptor_dimension
        self._image_width = image_width
        self._image_height = image_height

        # this defaults to the identity transform
        self._image_mean = np.zeros(3)
        self._image_std_dev = np.ones(3)

        # defaults to no image normalization, assume it is done by dataset loader instead

        self.config = dict()

        self._descriptor_image_stats = None
        self._normalize = normalize
        self._constructed_from_model_folder = False

    @property
    def fcn(self):
        return self._fcn

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @property
    def descriptor_dimension(self):
        return self._descriptor_dimension

    @property
    def image_shape(self):
        return [self._image_height, self._image_width]

    @property
    def image_mean(self):
        return self._image_mean

    @image_mean.setter
    def image_mean(self, value):
        """
        Sets the image mean used in normalizing the images before
        being passed through the network
        :param value: list of floats
        :type value:
        :return:
        :rtype:
        """
        self._image_mean = value
        self.config['image_mean'] = value
        self._update_normalize_tensor_transform()

    @property
    def image_std_dev(self):
        return self._image_std_dev

    @image_std_dev.setter
    def image_std_dev(self, value):
        """
        Sets the image std dev used in normalizing the images before
        being passed through the network
        :param value: list of floats
        :type value:
        :return:
        :rtype:
        """
        self._image_std_dev = value
        self.config['image_std_dev'] = value
        self._update_normalize_tensor_transform()

    @staticmethod
    def get_fcn(config=None):
        resnet_model = config["backbone"]["resnet_name"]
        fcn = getattr(resnet_dilated, resnet_model)(num_classes=config['descriptor_dimension'])
        return fcn

    def forward_single_image_tensor(self, img_tensor):
        """
        Simple forward pass on the network.

        Assumes the image has already been normalized (i.e. subtract mean, divide by std dev)

        Color channel should be RGB

        :param img_tensor: torch.FloatTensor with shape [3,H,W]
        :type img_tensor:
        :return: torch.FloatTensor with shape  [H, W, D]
        :rtype:
        """

        assert len(img_tensor.shape) == 3

        # transform to shape [1,3,H,W]
        img_tensor = img_tensor.unsqueeze(0)

        # make sure it's on the GPU
        img_tensor = torch.tensor(img_tensor, device=torch.device("cuda"))

        res = self.forward(img_tensor)  # shape [1,D,H,W]
        # print "res.shape 1", res.shape

        res = res.squeeze(0)  # shape [D,H,W]
        # print "res.shape 2", res.shape

        res = res.permute(1, 2, 0)  # shape [H,W,D]
        # print "res.shape 3", res.shape

        return res

    def forward(self, img_tensor):
        img_tensor = img_tensor.float()
        """
        Simple forward pass on the network.

        Does NOT normalize the image

        D = descriptor dimension
        N = batch size

        :param img_tensor: input tensor img.shape = [N, D, H , W] where
                    N is the batch size
        :type img_tensor: torch.Variable or torch.Tensor
        :return: torch.Variable with shape [N, D, H, W],
        :rtype:
        """

        res = self.fcn(img_tensor)
        if self._normalize:
            # print "normalizing descriptor norm"
            norm = torch.norm(res, 2, 1)  # [N,1,H,W]
            res = res / norm

        return res

    @staticmethod
    def find_best_match(pixel_a, res_a, res_b, imgA=None, imgB=None, delta=50, debug=False):
        """
        Compute the correspondences between the pixel_a location in image_a
        and image_b

        :param pixel_a: vector of (u,v) pixel coordinates
        :param res_a: array of dense descriptors res_a.shape = [H,W,D]
        :param res_b: array of dense descriptors
        :param pixel_b: Ground truth . . .
        :return: (best_match_uv, best_match_diff, norm_diffs)
        best_match_idx is again in (u,v) = (right, down) coordinates
        delta = defaut 10 pixel

        """

        # print (res_a.shape)
        # print (res_b.shape)
        x, y = pixel_a[1], pixel_a[0]
        # b,g,r = tuple(imgA[x,y])

        top = [max(0, x - delta), y]
        right = [x, min(639, y + delta)]
        down = [min(479, x + delta), y]
        left = [x, max(0, y - delta)]

        # print ([u,v],top,right,down,left)

        descriptor_at_pixel = res_a[x, y]

        descriptor_at_pixel_top = res_a[top[0], top[1]]
        descriptor_at_pixel_down = res_a[down[0], down[1]]
        descriptor_at_pixel_left = res_a[left[0], left[1]]
        descriptor_at_pixel_right = res_a[right[0], right[1]]

        height, width, _ = res_a.shape

        if debug:
            print("height: ", height)
            print("width: ", width)
            print("res_b.shape: ", res_b.shape)

        # non-vectorized version
        # norm_diffs = np.zeros([height, width])
        # for i in xrange(0, height):
        #     for j in xrange(0, width):
        #         norm_diffs[i,j] = np.linalg.norm(res_b[i,j] - descriptor_at_pixel)**2

        norm_diffs = np.sqrt(np.sum(np.square(res_b - descriptor_at_pixel), axis=2))
        '''
        print (norm_diffs.shape)
        topKSmallist = norm_diffs.argsort(axis=1)[:5]
        print (topKSmallist.shape)
        print (topKSmallist)
        best_match_flattened_idx = topKSmallist[0]
        print ("best_match_flattened_idx",best_match_flattened_idx)
        '''

        k = 1000
        nSample = 0
        fllaten_norm_diff = norm_diffs.reshape((1, height * width))
        fllaten_norm_diff = fllaten_norm_diff
        fllaten_norm_diff_idx = fllaten_norm_diff.argsort()
        fllaten_norm_diff_idx = fllaten_norm_diff_idx.squeeze()
        # print (fllaten_norm_diff_idx.shape)

        bestMatchDiff = 99999
        currentIDX = 0
        bestMatchIDX = 0
        '''
        for currentIDX in range (k):
            tx ,ty = np.unravel_index(fllaten_norm_diff_idx[currentIDX], norm_diffs.shape)

            maxIDX1 = np.argmax(imgA[x,y])
            maxIDX2 = np.argmax(imgB[tx,ty])
            currentDiff = abs( (int)(imgA[x,y][maxIDX1]) - (int)(imgB[tx,ty][maxIDX2]))

            #print (currentDiff,fllaten_norm_diff[:,currentIDX][0],maxIDX1,maxIDX2)
            if (maxIDX1 != maxIDX2 or abs ( currentDiff) > 100 ):
                continue

            #if (abs (r-r1) > 30  or  abs (g-g1) > 30 or  abs (b-b1) > 30) : 
            #    continue

            currentScore = ((currentDiff)/30.0 +  fllaten_norm_diff[:,currentIDX][0])

            if (currentScore < bestMatchDiff):
                #print ("currentDiff",maxIDX1,maxIDX2,currentDiff,fllaten_norm_diff[:,currentIDX][0])

                bestMatchIDX = currentIDX
                bestMatchDiff = currentScore

        print ("IDX",bestMatchIDX)
        '''
        best_match_flattened_idx = fllaten_norm_diff_idx[bestMatchIDX]  # np.argmin(norm_diffs)
        # print ("diff by flatten",fllaten_norm_diff[:,best_match_flattened_idx])
        best_match_xy = np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
        # print ("Selected: ", imgA[x,y],imgB[best_match_xy])
        best_match_diff = norm_diffs[best_match_xy]

        best_match_uv = (best_match_xy[1], best_match_xy[0])

        x1 = np.clip(best_match_xy[0], 0, height - 1)
        y1 = np.clip(best_match_xy[1], 0, width - 1)
        '''
        bestDiff  = 9999999
        bestAlpha  = 0
        stepDegree = 10 *(math.pi/180)
        nTime = int (2*math.pi // stepDegree)



        for i in range (-nTime//2,nTime//2):
            alpha = i * stepDegree

            b_top   = [x1- delta* math.cos(alpha), y1+delta * math.sin (alpha)]
            b_right = [x1- delta* math.cos(alpha+math.pi/2), y1+delta * math.sin (alpha+math.pi/2)]
            b_down  = [x1- delta* math.cos(alpha+2*math.pi/2), y1+delta * math.sin (alpha+2*math.pi/2)]
            b_left  = [x1- delta* math.cos(alpha+3*math.pi/2), y1+delta * math.sin (alpha+3*math.pi/2)]


            b_top[0] = int ( np.clip (b_top[0],0,height-1))
            b_top[1] = int ( np.clip (b_top[1],0,width-1))
            b_right[0] = int ( np.clip (b_right[0],0,height-1))
            b_right[1] =int (  np.clip (b_right[1],0,width-1))
            b_down[0] = int ( np.clip (b_down[0],0,height-1))
            b_down[1] = int ( np.clip (b_down[1],0,width-1))
            b_left[0] = int ( np.clip (b_left[0],0,height-1))
            b_left[1] = int ( np.clip (b_left[1],0,width-1))

            #print ([x1,y1],b_top,b_right,b_down,b_left)

            norm_diffs_top = np.sqrt(np.sum(np.square(res_b[b_top[0],b_top[1]] - descriptor_at_pixel_top)))
            norm_diffs_right= np.sqrt(np.sum(np.square(res_b[b_right[0],b_right[1]] - descriptor_at_pixel_right)))
            norm_diffs_down = np.sqrt(np.sum(np.square(res_b[b_down[0],b_down[1]] - descriptor_at_pixel_down)))
            norm_diffs_left = np.sqrt(np.sum(np.square(res_b[b_left[0],b_left[1]] - descriptor_at_pixel_left)))

            totalDiff = norm_diffs_top + norm_diffs_right + norm_diffs_down + norm_diffs_left

            if (totalDiff < bestDiff):
                #bestAlpha =  alpha
                bestDiff = totalDiff

        bestAlpha =  bestAlpha 

        #print ("return",best_match_diff,bestAlpha,bestDiff)
        '''
        # return best_match_uv,bestAlpha, best_match_diff, norm_diffs
        return best_match_uv, 0, 0, 0

    @staticmethod
    def find_best_match_only(pixel_a, res_a, res_b, mask_b=None):
        x, y = pixel_a[1], pixel_a[0]
        # b,g,r = tuple(imgA[x,y])

        descriptor_at_pixel = res_a[x, y]

        height, width, _ = res_a.shape

        norm_diffs = np.sqrt(np.sum(np.square(res_b - descriptor_at_pixel), axis=2))
        # if (mask_b!=None):
        mask_b = mask_b * (1) - 1
        mask_b *= (-9999)
        norm_diffs = (norm_diffs + mask_b)

        k = 1000
        nSample = 0
        best_match_xy = np.argwhere(norm_diffs == np.min(norm_diffs))[0]
        # print ("best_match_xy",best_match_xy)
        best_match_diff = norm_diffs[best_match_xy[0], best_match_xy[1]]
        '''

        fllaten_norm_diff = norm_diffs.reshape((1,height*width))
        fllaten_norm_diff = fllaten_norm_diff
        fllaten_norm_diff_idx = fllaten_norm_diff.argsort()
        fllaten_norm_diff_idx = fllaten_norm_diff_idx.squeeze()
        #print (fllaten_norm_diff_idx.shape)

        bestMatchIDX = 0

        best_match_flattened_idx = fllaten_norm_diff_idx[bestMatchIDX] #np.argmin(norm_diffs)
        #best_match_xy = [best_match_flattened_idx]      #np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
        best_match_xy = np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
        best_match_diff = norm_diffs[best_match_xy]
        '''
        best_match_uv = (best_match_xy[1], best_match_xy[0])
        # print ("best_match_uv",best_match_uv)

        x1 = np.clip(best_match_xy[0], 0, height - 1)
        y1 = np.clip(best_match_xy[1], 0, width - 1)

        return best_match_uv, best_match_diff, norm_diffs

    @staticmethod
    def find_area_best_match(pixel_a, res_a, res_b, mask_b=None, d=2):
        x, y = pixel_a[1], pixel_a[0]
        # b,g,r = tuple(imgA[x,y])
        # d = 5 # px
        delta = [[-d, 0], [+d, 0], [0, -d], [0, +d]]

        height, width, _ = res_a.shape

        listMatchPoints = []

        for i in range(4):
            descriptor_at_pixel = res_a[x, y]
            norm_diffs = np.sqrt(np.sum(np.square(res_b - descriptor_at_pixel), axis=2))
            fllaten_norm_diff = norm_diffs.reshape((1, height * width))
            fllaten_norm_diff = fllaten_norm_diff
            fllaten_norm_diff_idx = fllaten_norm_diff.argsort()
            fllaten_norm_diff_idx = fllaten_norm_diff_idx.squeeze()
            bestMatchIDX = 0
            best_match_flattened_idx = fllaten_norm_diff_idx[bestMatchIDX]  # np.argmin(norm_diffs)
            best_match_xy = np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
            best_match_diff = norm_diffs[best_match_xy]

            best_match_uv = (best_match_xy[1], best_match_xy[0])
            listMatchPoints.append(best_match_uv)

            x, y = x + delta[i % 4][0] + i % 4, y + delta[i % 4][1] + i % 4

        x1 = np.clip(best_match_xy[0], 0, height - 1)
        y1 = np.clip(best_match_xy[1], 0, width - 1)

        # print ("return",best_match_diff,bestAlpha,bestDiff)

        return listMatchPoints