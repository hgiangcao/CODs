#!/usr/bin/python

import sys, os
import numpy as np
import warnings
import logging
from DON_Training.model import Resnet34_8s,Resnet34_8s_orig,Resnet34_8s_dual
#from DON_Training.pspnet import PSPNet
#from DON_Training.UNET_New import FullUNet
#from DON_Training.UnetDON import UnetDON

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from ITRIP.Configuration import config
from numpy import linalg as LA


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
class DenseCorrespondenceNetwork(nn.Module):
    IMAGE_TO_TENSOR = valid_transform = transforms.Compose([transforms.ToTensor(), ])

    def __init__(self, fcn, descriptor_dimension, image_width=512,
                 image_height=512, normalize=False):
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

        self._fcn = fcn.float()

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

    @property
    def image_to_tensor(self):
        return self._image_to_tensor

    @image_to_tensor.setter
    def image_to_tensor(self, value):
        self._image_to_tensor = value

    @property
    def normalize_tensor_transform(self):
        return self._normalize_tensor_transform

    @property
    def path_to_network_params_folder(self):
        if not 'path_to_network_params_folder' in self.config:
            raise ValueError("DenseCorrespondenceNetwork: Config doesn't have a `path_to_network_params_folder`"
                             "entry")

        return self.config['path_to_network_params_folder']


    @property
    def constructed_from_model_folder(self):
        """
        Returns True if this models was constructed from
        :return:
        :rtype:
        """
        return self._constructed_from_model_folder

    @constructed_from_model_folder.setter
    def constructed_from_model_folder(self, value):
        self._constructed_from_model_folder = value


    def _update_normalize_tensor_transform(self):
        """
        Updates the image to tensor transform using the current image mean and
        std dev
        :return: None
        :rtype:
        """
        self._normalize_tensor_transform = transforms.Normalize(self.image_mean, self.image_std_dev)

    def forward_on_img(self, img, cuda=True):
        """
        Runs the network forward on an image
        :param img: img is an image as a numpy array in opencv format [0,255]
        :return:
        """
        img_tensor = DenseCorrespondenceNetwork.IMAGE_TO_TENSOR(img)

        if cuda:
            img_tensor.cuda()

        return self.forward(img_tensor)

    def forward_on_img_tensor(self, img):
        """
        Deprecated, use `forward` instead
        Runs the network forward on an img_tensor
        :param img: (C x H X W) in range [0.0, 1.0]
        :return:
        """
        warnings.warn("use forward method instead", DeprecationWarning)

        img = img.unsqueeze(0)
        img = torch.tensor(img, device=torch.device("cuda"))
        res = self.fcn(img)
        res = res.squeeze(0)
        res = res.permute(1, 2, 0)
        res = res.data.cpu().numpy().squeeze()

        return res

    def forward(self, img_tensor,depth_tensor):
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
        res = self.fcn(img_tensor,depth_tensor)
        if self._normalize:
            # print "normalizing descriptor norm"
            norm = torch.norm(res, 2, 1)  # [N,1,H,W]
            res = res / norm

        return res

    def forward_single_image_tensor(self, img_tensor,depth_tensor):
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
        depth_tensor = depth_tensor.unsqueeze(0)
        # make sure it's on the GPU
        img_tensor = torch.tensor(img_tensor, device=torch.device("cuda"))
        depth_tensor = torch.tensor(depth_tensor, device=torch.device("cuda"))
        res = self.forward(img_tensor,depth_tensor)  # shape [1,D,H,W]
        # print "res.shape 1", res.shape

        res = res.squeeze(0)  # shape [D,H,W]
        # print "res.shape 2", res.shape

        res = res.permute(1, 2, 0)  # shape [H,W,D]
        # print "res.shape 3", res.shape

        return res


    def foward_batch(self, img_tensor):
        #img_tensor = torch.tensor(img_tensor, device=torch.device("cuda"))
        res = self.fcn(img_tensor )  # shape [1,D,H,W]
        return res

    def process_network_output(self, image_pred, N):
        """
        Processes the network output into a new shape

        :param image_pred: output of the network img.shape = [N,descriptor_dim, H , W]
        :type image_pred: torch.Tensor
        :param N: batch size
        :type N: int
        :return: sa1me as input, new shape is [N, W*H, descriptor_dim]
        :rtype:
        """

        W = self._image_width
        H = self._image_height
        image_pred = image_pred.view(N, self.descriptor_dimension, W * H)
        image_pred = image_pred.permute(0, 2, 1)
        return image_pred

    def clip_pixel_to_image_size_and_round(self, uv):
        """
        Clips pixel to image coordinates and converts to int
        :param uv:
        :type uv:
        :return:
        :rtype:
        """
        u = min(int(round(uv[0])), self._image_width - 1)
        v = min(int(round(uv[1])), self._image_height - 1)
        return [u, v]


    @staticmethod
    def get_fcn(config,setting):
        #fcn = DensNet(5,8)
        if (config["model"] == "Resnet"):
          fcn =  Resnet34_8s(nInput=config["inputChannel"][setting],pretrained = config["usePretrained"],nDimesion=config["descriptor_dimension"])
        elif(config["model"] == "Resnet_dual"):
          fcn =  Resnet34_8s_dual(nInput=config["inputChannel"][setting],pretrained = config["usePretrained"],nDimesion=config["descriptor_dimension"])
        elif (config["model"]=="Resnet_orig"):
            print ("using Resnet_orig")
            fcn = Resnet34_8s_orig(nInput=config["inputChannel"][setting], pretrained=config["usePretrained"],
                              nDimesion=config["descriptor_dimension"])
        # elif(config["model"]=="Unet"):
        #   fcn = UnetDON(config["inputChannel"][setting], config["nDimesion"])#  FullUNet(config["inputChannel"][setting], config["nDimesion"])
        #   fcn.apply(weights_init)
        # elif(config["model"]=="PSPNet"):
        #     fcn = PSPNet(nInput = config["inputChannel"][setting], nclass = config["nDimesion"])
        else:
          print("ERROR: select Unet/Resnet")

        return fcn

    @staticmethod
    def from_config(setting):
        """
        Load a network from a configuration


        :param config: Dict specifying details of the network architecture

        :param load_stored_params: whether or not to load stored params, if so there should be
            a "path_to_network" entry in the config
        :type load_stored_params: bool

        e.g.
            path_to_network: /home/manuelli/code/dense_correspondence/recipes/trained_models/10_drill_long_3d
            parameter_file: dense_resnet_34_8s_03505.pth
            descriptor_dimensionality: 3
            image_width: 640
            image_height: 480

        :return: DenseCorrespondenceNetwork
        :rtype:
        """
        if 'normalize' in config:
            normalize = config['normalize']
        else:
            normalize = False

        fcn = DenseCorrespondenceNetwork.get_fcn(config,setting)

        dcn = DenseCorrespondenceNetwork(fcn, config['descriptor_dimension'],
                                         image_width=config['image_width'],
                                         image_height=config['image_height'],
                                         normalize=normalize)

        dcn.cuda()
        dcn.train()
        dcn.config = config
        return dcn


    @staticmethod
    def find_best_match(pixel_a, res_a, res_b, debug=False):
        """
        Compute the correspondences between the pixel_a location in image_a
        and image_b

        :param pixel_a: vector of (u,v) pixel coordinates
        :param res_a: array of dense descriptors res_a.shape = [H,W,D]
        :param res_b: array of dense descriptors
        :param pixel_b: Ground truth . . .
        :return: (best_match_uv, best_match_diff, norm_diffs)
        best_match_idx is again in (u,v) = (right, down) coordinates

        """
        descriptor_at_pixel = res_a[pixel_a[1], pixel_a[0]]
        height, width, _ = res_a.shape

        if debug:
            print
            "height: ", height
            print
            "width: ", width
            print
            "res_b.shape: ", res_b.shape

        # non-vectorized version
        # norm_diffs = np.zeros([height, width])
        # for i in xrange(0, height):
        #     for j in xrange(0, width):
        #         norm_diffs[i,j] = np.linalg.norm(res_b[i,j] - descriptor_at_pixel)**2

        norm_diffs = np.sqrt(np.sum(np.square(res_b - descriptor_at_pixel), axis=2))

        best_match_flattened_idx = np.argmin(norm_diffs)
        best_match_xy = np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
        best_match_diff = norm_diffs[best_match_xy]

        best_match_uv = (best_match_xy[1], best_match_xy[0])

        return best_match_uv, best_match_diff, norm_diffs

    @staticmethod
    def find_best_match_for_descriptor(descriptor, res):
        """
        Compute the correspondences between the given descriptor and the descriptor image
        res
        :param descriptor:
        :type descriptor:
        :param res: array of dense descriptors res = [H,W,D]
        :type res: numpy array with shape [H,W,D]
        :return: (best_match_uv, best_match_diff, norm_diffs)
        best_match_idx is again in (u,v) = (right, down) coordinates
        :rtype:
        """
        height, width, _ = res.shape

        norm_diffs = np.sqrt(np.sum(np.square(res - descriptor), axis=2))

        best_match_flattened_idx = np.argmin(norm_diffs)
        best_match_xy = np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
        best_match_diff = norm_diffs[best_match_xy]

        best_match_uv = (best_match_xy[1], best_match_xy[0])

        return best_match_uv, best_match_diff, norm_diffs

    def evaluate_descriptor_at_keypoints(self, res, keypoint_list):
        """

        :param res: result of evaluating the network
        :type res: torch.FloatTensor [D,W,H]
        :param img:
        :type img: img_tensor
        :param kp: list of cv2.KeyPoint
        :type kp:
        :return: numpy.ndarray (N,D) N = num keypoints, D = descriptor dimension
        This is the same format as sift.compute from OpenCV
        :rtype:
        """

        raise NotImplementedError("This function is currently broken")

        N = len(keypoint_list)
        D = self.descriptor_dimension
        des = np.zeros([N, D])

        for idx, kp in enumerate(keypoint_list):
            uv = self.clip_pixel_to_image_size_and_round([kp.pt[0], kp.pt[1]])
            des[idx, :] = res[uv[1], uv[0], :]

        # cast to float32, need this in order to use cv2.BFMatcher() with bf.knnMatch
        des = np.array(des, dtype=np.float32)
        return des

    @staticmethod
    def find_best_match_only(pixel_a, res_a, res_b, mask_b=None):
        x, y = pixel_a[1], pixel_a[0]
        # b,g,r = tuple(imgA[x,y])

        descriptor_at_pixel = res_a[x, y]

        height, width, _ = res_a.shape

        norm_diffs = LA.norm( (res_b-res_a[x,y].reshape(1,1,config["descriptor_dimension"])), axis =2 ) # np.sqrt(np.sum(np.square(res_b - descriptor_at_pixel), axis=2))
        #if (mask_b!=None).:
        mask_b = mask_b * (1) - 1
        mask_b = mask_b *(-9999)
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
