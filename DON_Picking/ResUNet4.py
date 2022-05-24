import sys
import os

os.environ['OMP_NUM_THREADS'] = '1'
CUDA_LAUNCH_BLOCKING = 1

sys.path.insert(1, '..')
import torch.nn as nn
import torchvision.models as models
from DON_Training.model import Resnet34_8s
import torch

from collections import namedtuple, OrderedDict
from DON_Picking.UNetPicking import DoubleConv, Down, Up

from DON_Training.dense_correspondence_network import DenseCorrespondenceNetwork
from ITRIP.Configuration import config
from DON_Training.DensObjectNet import DensObjectNet


class RUNet(nn.Module):

    def __init__(self, modeObsevation, n_classes, bilinear=True,
                  DON_pretrained="DON_GraspNet_O2O_full_rotation_index_plus_ITRI_insideNonMatch_rd_08_RGBD_Resnet/DON_120000"):
                  #pretrained="DOND_ResUnet_8_best_D_Resnet/DON_80000"):
        #super().__init__()
        super(RUNet, self).__init__()
        setting = modeObsevation
        self.DON = DensObjectNet(setting=setting, pretrained=DON_pretrained)
      
        self.depth_norm = nn.InstanceNorm2d(1, affine=False)
        #self.obsModel = Resnet34_8s(nInput=3,output_stride=8)
        self.obsModel = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        self.maxPool = nn.Upsample(scale_factor=2)
        self.n_channels = config["inputChannel"][modeObsevation]
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2

        self.inc = DoubleConv(self.n_channels, 32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64 * 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128 * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256 * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512 * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

        )

        self.up1 = Up(64 + 64, 256 // factor, bilinear)
        self.up2 = Up(128+64, 128 // factor, bilinear)
        self.up3 = Up(64+32, 64 // factor, bilinear)
        self.up4 = Up(32+32, 32, bilinear)
         
        self.cnnValue = nn.Sequential (
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
             nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            #nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            #nn.MaxPool2d(2,2)
          
          )
        
        self.valueOut = nn.Sequential(
            nn.Linear(1024*4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            #nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)

        )

        self.final_prediction = nn.Sequential(OrderedDict([
            ('final_conv0', nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True)),
            ('final_relu0', nn.ReLU(inplace=True)),
            ('final_conv1', nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=True)),
        ]))  # predict 1 # 60 x 60

        print("DONE SETUP NETWORK")

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self,rgb,depth):
        #depth = x[:,0,:,:].unsqueeze(1)
        
        norm_x = self.depth_norm(depth)
        depth_input = torch.cat((norm_x,norm_x,norm_x), dim=1)
#print(depth.shape)

        h1 = self.obsModel.conv1(depth_input)
        h1 = self.obsModel.bn1(h1)
        h1 = self.obsModel.relu(h1)
        h2 = self.obsModel.maxpool(h1)
        h2 = self.obsModel.layer1(h2)
        h3 = self.obsModel.layer2(h2)
        h4 = self.obsModel.layer3(h3)
        h5 = self.obsModel.layer4(h4)

        (l1, l2, l3, l4, l5,l6) = self.DON.getBatchLayer(rgb,depth)
         
        l1 = l1.detach()
        l2 = l2.detach()
        l3 = l3.detach()
        l4 = l4.detach()
        l5 = l5.detach()
        
        #print (l1.shape,l2.shape,l3.shape,l4.shape,h5.shape)
        #print (h1.shape,h2.shape,h3.shape,h4.shape,l5.shape)

        x1 = torch.cat((h1, l1), dim=1)
        x1 = self.conv1(x1)

        x2 = torch.cat((h2, l2), dim=1)
        x2 = self.conv2(x2)

        x3 = torch.cat((h3, l3), dim=1)
        x3 = self.conv3(x3)

        x4 = torch.cat((h4, l4), dim=1)
        x4 = self.conv4(x4)

        x5 = torch.cat((h5, l5), dim=1)
        x5 = self.conv5(x5)

        up1 = self.up1(x5, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        up4 = self.up4(up3, x1)
        #print (up4.shape)

        out = self.final_prediction(up4)

        # print (x1.shape,x2.shape,x3.shape,x4.shape)
        x6 = self.cnnValue(x5)
        flatten = x6.reshape(x6.size(0), x6.size(1) * x6.size(2) * x6.size(3))
        #print (flatten.shape)
        valueOut = self.valueOut(flatten)

        return out, valueOut


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from DON_Training.DataGenerator_O2O import convertToHeatmap

# test model
if __name__ == "__main__":
    pathScene = "../../dataDON/GraspNet_orig/Object29/002"
    idx = 1
    # batch_size = 8
    # input = torch.randn(batch_size,  5, 256, 256)#.cuda()
    #
    #
    # ResNet = Resnet34_8s(nInput=5,nDimesion=8)
    # ResNet.train()
    # input = torch.randn(batch_size, 5, 256, 256)
    # output = ResNet(input)
    # pytorch_total_params = sum(p.numel() for p in ResNet.parameters() if p.requires_grad)
    # print("Resnet", pytorch_total_params, "params")
    #
    # print (ResNet)

    batch_size = 8
    #depth = torch.randn(batch_size, 1, 256, 256).cuda()
    #rgb = torch.randn(batch_size, 3, 256, 256).cuda()

    orig_depth = cv2.imread(pathScene + "/%06d_depth.png" % (idx), -1).astype(float) / 1000.0
    depth = torch.from_numpy(orig_depth).float().cuda()
    depth = depth.reshape(1,1,256,256)
    print (depth.shape)

    imageColor = np.array(Image.open(pathScene + "/%06d_rgb.jpg" % (idx)))
    imageColor = cv2.cvtColor(imageColor, cv2.COLOR_BGR2RGB)
    rgb_orig = imageColor / 256
    rgb = torch.from_numpy(rgb_orig).float().cuda()
    rgb = rgb.reshape(1, 3, 256, 256)
    print (rgb.shape)
    mask = cv2.imread(pathScene + "/%06d_visible_mask.png" % (idx), -1).astype(float)
    # print (DON.dcn)

    runet = RUNet("RGBD", 1).cuda()
    out, values = runet(rgb,depth)

    pytorch_total_params = sum(p.numel() for p in runet.parameters() if p.requires_grad)
    print("Resnet", pytorch_total_params, "params")
    print(out.shape)
    print(values.shape)


    out = out.detach().cpu().numpy().squeeze().squeeze()
    print (out.shape)

    mask = cv2.resize(mask, (128, 128))
    out = out * mask
    out = convertToHeatmap(out)

    out = np.array(out)
    rgb_orig = cv2.resize(rgb_orig, (128, 128))


    heatmapMerge = cv2.addWeighted(out / 255, 0.7, rgb_orig , 0.3, 0)


    while True:
        cv2.imshow("heatmap", np.array(out))
        cv2.imshow("heatmap2", np.array(heatmapMerge))
        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit and terminate progra,
            cv2.destroyAllWindows()
            break

