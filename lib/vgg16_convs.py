# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:41:56 2019

@author: Junzhe Xu
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
#from networks.network import Network
from lib.HoughVoting.houghvoting import *


class vgg16_convs(nn.Module):
    def __init__(self, input_format, num_classes, num_units, scales, threshold_label, vote_threshold, 
                 vertex_reg_2d=False, vertex_reg_3d=False, vertex_reg_hough_in = False, pose_reg=False, 
                 adaptation=False, trainable=True, is_train=True):
        super(vgg16_convs, self).__init__()
        
        self.inputs = []
        self.input_format = input_format
        self.num_classes = num_classes
        self.num_units = num_units
        self.scale = 1.0
        self.threshold_label = threshold_label
        self.vertex_reg_2d = vertex_reg_2d
        self.vertex_reg_3d = vertex_reg_3d
        self.vertex_reg = vertex_reg_2d or vertex_reg_3d
        self.vertex_reg_hough = vertex_reg_hough_in
        self.pose_reg = pose_reg
        self.adaptation = adaptation
        self.trainable = trainable
        
        # if vote_threshold < 0, only detect single instance (default). 
        # Otherwise, multiple instances are detected if hough voting score larger than the threshold

        if is_train:
            self.is_train = 1
            self.skip_pixels = 10
            self.vote_threshold = vote_threshold
            self.vote_percentage = 0.02
        else:
            self.is_train = 0
            self.skip_pixels = 10
            self.vote_threshold = vote_threshold
            self.vote_percentage = 0.02
            

        
        # VGG-16 for feature extraction
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 1/2 of the origin image

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 1/4 of the origin image

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 1/8 of the origin image

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 1/16 of the origin image

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # If input format is RGBD we use another network
        """
        self.conv1_1_p = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2_p = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.poo1_p = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_1_p = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2_p = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2_p = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3_1_p = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2_p = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3_p = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3_p = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4_1_p = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2_p = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3_p = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4_p = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5_1_p = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2_p = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3_p = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        """
        
        
        # For combination layer
        # For semantic segmentation
        self.conv6_seman_a = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6_seman_b = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dconv6_seman_a = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=4, stride=2, padding=1, output_padding=0)
        self.dropout = nn.Dropout2d()
        self.dconv7_seman = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                                  kernel_size=16, stride=8, padding=4, output_padding=0)
        self.conv8_seman = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=3, stride=1, padding=1)
        
        
        # For Center estimation
        self.conv6_center_a = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_center_b = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dconv6_center_a = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=4, stride=2, padding=1, output_padding=0)
        self.dconv7_center = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                                  kernel_size=16, stride=8, padding=4, output_padding=0)
        self.conv8_center = nn.Conv2d(in_channels=128, out_channels=3 * self.num_classes, kernel_size=3, stride=1, padding=1)
        
        
        self.relu = nn.ReLU()
        
        self.hough_voting = HF()
        
        
    def conv_fun(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x) # 1/2 of the original image 
        
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)  # 1/4 of the original image 
        
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool3(x)  # 1/8 of the original image 
        
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        f_conv4 = self.relu(self.conv4_3(x))
        x = self.pool3(f_conv4)  # 1/16 of the original image 
        
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        f_conv5 = self.relu(self.conv5_3(x))
        x = self.pool3(f_conv5)  # 1/32 of the original image 
        return x, f_conv4, f_conv5
        
    
    def seman_net(self, f_conv4, f_conv5):
        x_a = self.dconv6_seman_a(self.conv6_seman_a(f_conv5))
        x_b = self.conv6_seman_b(f_conv4)
        x = x_a + x_b
        x = self.dconv7_seman(x)
        x = self.conv8_seman(x)
        return x
    
    
    def center_net(self, f_conv4, f_conv5):
        x_a = self.dconv6_center_a(self.conv6_center_a(f_conv5))
        x_b = self.conv6_center_b(f_conv4)
        x = x_a + x_b
        x = self.dconv7_center(x)
        x = self.conv8_center(x)
        return x
        
        
    def forward(self, x, extents, meta, gt_hough, is_train, device_cpu):
        x, f_conv4, f_conv5 = self.conv_fun(x)
        x_seman = self.seman_net(f_conv4, f_conv5)  # the output of semantic segmentation
        
        if self.vertex_reg == True:
            x_center_dir = self.center_net(f_conv4, f_conv5) # the output of the center estimation
            x_center = None
            if self.vertex_reg_hough:
                x_seman_single = torch.argmax(x_seman, dim = 1).type('torch.IntTensor')
                x_seman_single = x_seman_single.to(device_cpu)

                x_center_dir_hough = x_center_dir.permute(0,2,3,1)
                x_center_dir_hough = x_center_dir_hough.to(device_cpu)

                x_hough = self.hough_voting(x_seman_single, x_center_dir_hough, extents, meta, gt_hough, is_train)
                
            return x_seman, x_center_dir, x_hough
        else:
            return x_seman
        
    
    
    