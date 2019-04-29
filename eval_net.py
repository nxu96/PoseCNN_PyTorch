# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:04:43 2019

@author: sunhu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import average_precision_score as ap_score
from tqdm import tqdm



def cal_AP(testloader, net, criterion, device, num_obj, opt):
    '''
    Calculate Average Precision
    Evaluation for the semantic segmentation part 
    '''
    cnt = 0
    aps = []
    with torch.no_grad():
        net = net.eval()
        preds = [[] for _ in range(num_obj)]
        heatmaps = [[] for _ in range(num_obj)]
        for data in tqdm(testloader):
            if opt.vertex_reg == True:
                 # Only train the center-voting part
                 images, labels, vertex_targets, vertex_weights, extents = data
                 images = images.to(device)
                 labels = labels.type('torch.LongTensor').to(device)
                 extents = extents.to(device)
                 output_seg, _, _ = net(images, extents)
            else:
                 # Only train the segmentation part
                 images, labels = data
                 images = images.to(device)
                 labels = labels.type('torch.LongTensor').to(device)
                 output_seg = net(images)
            output = output_seg.cpu().numpy()
            for c in range(num_obj):
                preds[c].append(output[:, c].reshape(-1))
                heatmaps[c].append(labels[:, c].cpu().numpy().reshape(-1))
        
        for c in range(num_obj):
            preds[c] = np.concatenate(preds[c])
            heatmaps[c] = np.concatenate(heatmaps[c])
            if heatmaps[c].max() == 0:
                ap = float('nan')
            else:
                ap = ap_score(heatmaps[c], preds[c])
                aps.append(ap)
            print("AP = {}".format(ap))

    # print(losses / cnt)
    return aps