# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:14:11 2019

@author: Junzhe Xu
"""

import numpy as np
import torch



# FOR CENTER ESTIMATION

# Center-voting for validation
def _vote_centers_val(im_label, cls_indexes, centers, poses, num_classes, extents):
    width = im_label.shape[1]
    height = im_label.shape[0]
    vertex_targets = np.zeros((height, width, 3), dtype=np.float32)
    center = np.zeros((2, 1), dtype=np.float32)
    
    for i in range(1, num_classes):
        y, x = np.where(im_label == i)
        I = np.where(im_label == i)
        ind = np.where(cls_indexes == i)[0]
        
        if len(x) > 0 and len(ind) > 0:
            center[0] = centers[ind, 0]
            center[1] = centers[ind, 1]
            z = poses[2, 3, ind]
            R = np.tile(center, (1, len(x))) - np.vstack((x, y))
            # compute the norm
            N = np.linalg.norm(R, axis=0) + 1e-10
            # normalization
            R = np.divide(R, np.tile(N, (2,1)))
            # assignment
            vertex_targets[y, x, 0] = R[0,:]
            vertex_targets[y, x, 1] = R[1,:]
            vertex_targets[y, x, 2] = z

    return vertex_targets




# Center voting for training
def _vote_centers_train(im_label, cls_indexes, center, depth_centers, num_classes):
    height = im_label.shape[0]
    width = im_label.shape[1]
    vertex_targets = np.zeros((3*num_classes, height, width), dtype=np.float32)
    vertex_weights = np.zeros(vertex_targets.shape, dtype=np.float32)
    c = np.zeros((2, 1), dtype=np.float32)
    
    for i in range(1, num_classes):
        y, x = np.where(im_label == i)
        if len(x) > 0:
            c[0] = center[i, 0]
            c[1] = center[i, 1]
            R = np.tile(c, (1, len(x))) - np.vstack((x, y))
            # compute the norm
            N = np.linalg.norm(R, axis=0) + 1e-10
            # normalization
            R = np.divide(R, np.tile(N, (2,1)))
            # assignment
            start = 3 * i
            end = start + 3
            vertex_targets[3*i, y, x] = R[0,:]
            vertex_targets[3*i+1, y, x] = R[1,:]
            vertex_targets[3*i+2, y, x] = depth_centers[i, 0]
            vertex_weights[start:end, y, x] = 10.0

    return vertex_targets, vertex_weights



def smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights, sigma=1.0, VERTEX_W=5.0):

    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_weights * vertex_diff
    abs_diff = torch.abs(diff)
    smoothL1_sign = torch.tensor((abs_diff < 1. / sigma_2).float(), requires_grad=False)
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = torch.div(torch.sum(in_loss), torch.sum(vertex_weights) + 1e-10 )
    loss = VERTEX_W * torch.tensor(loss, requires_grad=True)

    return loss
