import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import sys
sys.path.insert(1, '..')
#from DON_Picking.UNetPicking import NormalUp, Up
import torch.utils.data as D
import torch.nn.functional as F

import copy

import torchvision
from torchvision import transforms as T

from collections import namedtuple,OrderedDict

import warnings
warnings.simplefilter("ignore", UserWarning)

def adjust_input_image_size_for_proper_feature_alignment(input_img_batch, output_stride=8):
    """Resizes the input image to allow proper feature alignment during the
    forward propagation.

    Resizes the input image to a closest multiple of `output_stride` + 1.
    This allows the proper alignment of features.
    To get more details, read here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159

    Parameters
    ----------
    input_img_batch : torch.Tensor
        Tensor containing a single input image of size (1, 3, h, w)

    output_stride : int
        Output stride of the network where the input image batch
        will be fed.

    Returns
    -------
    input_img_batch_new_size : torch.Tensor
        Resized input image batch tensor
    """

    input_spatial_dims = np.asarray( input_img_batch.shape[2:], dtype=np.float )

    # Comments about proper alignment can be found here
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159
    new_spatial_dims = np.ceil(input_spatial_dims / output_stride).astype(np.int) * output_stride + 1

    # Converting the numpy to list, torch.nn.functional.upsample_bilinear accepts
    # size in the list representation.
    new_spatial_dims = list(new_spatial_dims)

    input_img_batch_new_size = nn.functional.upsample_bilinear(input=input_img_batch,
                                                               size=new_spatial_dims)

    return input_img_batch_new_size


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Resnet34_8s(nn.Module):


    def __init__(self,nInput=3,pretrained=False, nDimesion=8):

        super(Resnet34_8s, self).__init__()
        bilinear = True
        self.nInput = nInput
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = models.resnet34(fully_conv=True,
                                       pretrained=pretrained,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.conv1 = nn.Conv2d(nInput, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #print (resnet34_8s.inplanes)
#resnet34_8s.avgpool = nn.AdaptiveAvgPool2d((32, 32))
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, nDimesion, 1)
        '''
        #resnet34_8s.fc = Identity()
        self.up1 = Up(768, 512 // 2, bilinear)
        self.up2 = Up(384, 256 // 2, bilinear)
        #self.up3 = Up(128+64, 128 // 2, bilinear)
        #self.up4 = Up(64+64, 128 // 2, bilinear)
        self.finalUp =  nn.Sequential(
                        nn.Conv2d(128,nDimesion, kernel_size=3, padding=1),
                        #nn.Conv2d(128, nDimesion, 1),
                        nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
                        )
        '''
        self.resnet34_8s = resnet34_8s

        #self._normal_initialization(self.resnet34_8s.fc)
        print ("DONE SETUP NETWORK RES-UNET")
        self.depth_norm = nn.InstanceNorm2d(1, affine=False)

    def _normal_initialization(self, layer):

        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self,rgb,depth, feature_alignment=False):
        #print (x.shape)
        if len(depth.shape) ==3:
          depth = depth.unsqueeze_(0)

        depth = self.depth_norm(depth)
        x = torch.cat([rgb, depth], dim=1)

        if (self.nInput ==1):
            x = depth
        elif (self.nInput ==3):
            x = rgb

        input_spatial_dim = x.size()[2:]

        if feature_alignment:

            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)
        
        l1 = self.resnet34_8s.conv1(x)
        l1 = self.resnet34_8s.bn1(l1)
        l1 = self.resnet34_8s.relu(l1)

        l2 = self.resnet34_8s.maxpool(l1)
        l2 = self.resnet34_8s.layer1(l2)
        l3 = self.resnet34_8s.layer2(l2)
        l4 = self.resnet34_8s.layer3(l3)
        l5 = self.resnet34_8s.layer4(l4)
        '''
#      print (l1.shape,l2.shape,l3.shape,l4.shape,l5.shape)

        u1 = self.up1(l5,l4)
        u2 = self.up2(u1,l3)
        #u3 = self.up3(u2,l1)
        #u4 = self.up4(u3,l1)
        out = self.finalUp(u2)
        
        out = self.resnet34_8s(x)
        out = nn.functional.upsample_bilinear(input=out, size=input_spatial_dim)
        '''
        #print (l5.shape)
        #l6 = self.resnet34_8s.avgpool(l5)
        #print (l6.shape)
        out = self.resnet34_8s.fc(l5)
        #print (out.shape)
        out = nn.functional.upsample_bilinear(input=out, size=input_spatial_dim)

        return out


class Resnet34_8s_dual(nn.Module):

    def __init__(self, nInput=3, pretrained=True, nDimesion=8):

        super(Resnet34_8s_dual, self).__init__()

        resnet34_8s = models.resnet34(fully_conv=True,
                                      pretrained=pretrained,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        #resnet34_8s.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, nDimesion, 1)

        self.resnet34_8s_rgb = resnet34_8s
        self.resnet34_8s_depth = models.resnet34(fully_conv=True,
                                      pretrained=False,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        #resnet34_8s.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet34_8s_depth.fc = nn.Conv2d(resnet34_8s.inplanes, nDimesion, 1)


        #self.out = nn.Conv2d(nDimesion, nDimesion, 1)
        self.final_prediction = nn.Sequential(OrderedDict([
            ('final_conv0', nn.Conv2d(nDimesion, 8, kernel_size=3, stride=1, padding=1, bias=False)),
            ('final_relu0', nn.ReLU(inplace=True)),
            ('final_conv1', nn.Conv2d(8, nDimesion, kernel_size=1, stride=1, bias=False)),
        ]))  # predict 1 # 60 x 60

        print("DONE SETUP NETWORK RES-UNET")
        self.depth_norm = nn.InstanceNorm2d(1, affine=False)

    def _normal_initialization(self, layer):

        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, rgb, depth, feature_alignment=False):
        # print (x.shape)

        if len(depth.shape) == 3:
            depth = depth.unsqueeze_(0)

        depth = self.depth_norm(depth)
        depth = torch.cat((depth,depth,depth), dim=1)
        input_spatial_dim = rgb.size()[2:]

        if feature_alignment:
            rgb = adjust_input_image_size_for_proper_feature_alignment(rgb, output_stride=8)
            depth = adjust_input_image_size_for_proper_feature_alignment(depth, output_stride=8)

        #print(rgb.shape, depth.shape)
        rgb_output = self.resnet34_8s_rgb(rgb)
        depth_output = self.resnet34_8s_depth(depth)

        add_layer = rgb_output + depth_output # torch.cat((rgb_oput,depth_oput),dim=1)

        out = self.final_prediction (add_layer)

        out = nn.functional.upsample_bilinear(input=out, size=input_spatial_dim)

        return out
class Resnet34_8s_orig(nn.Module):


    def __init__(self,nInput=3,pretrained=False, nDimesion=8):

        super(Resnet34_8s_orig, self).__init__()
        bilinear = True
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = models.resnet34(fully_conv=True,
                                       pretrained=pretrained,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.conv1 = nn.Conv2d(nInput, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #print (resnet34_8s.inplanes)
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, nDimesion, 1)

        self.resnet34_8s = resnet34_8s

        self._normal_initialization(self.resnet34_8s.fc)
        print ("DONE SETUP NETWORK RES-UNET")
        self.depth_norm = nn.InstanceNorm2d(1, affine=False)

    def _normal_initialization(self, layer):

        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x = self.depth_norm(x)
        input_spatial_dim = x.size()[2:]

        if feature_alignment:

            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)

        x = self.resnet34_8s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x

class DensNet(nn.Module):
    def __init__(self,nInput,nDimesion):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(nInput, 64, 1, 2, 3)
        self.features.classifier = nn.Conv2d(1024, nDimesion, 1)
        del preloaded

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.features(x)
        #print(x.shape)
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
#       print(x.shape)
        return x

#test model
if __name__ == "__main__":
    batch_size = 8

    ResNet = Resnet34_8s_dual(nInput=4,nDimesion=8).cuda()
    ResNet.train()
    rgb = torch.randn(batch_size, 3, 256, 256).cuda()
    depth = torch.randn(batch_size, 1, 256, 256).cuda()
    output = ResNet(rgb,depth)
    pytorch_total_params = sum(p.numel() for p in ResNet.parameters() if p.requires_grad)
    print("Resnet", pytorch_total_params, "params")
    print (output.shape)
    #print (ResNet)
