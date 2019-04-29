import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import lib.center_est_funcs as center_img_gt 
import matplotlib.pyplot as plt

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine, onehot, 
                 seg = False, vertex_reg = False, vertex_reg_hough = False):
#        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
#        self.objlist = [1, 2]
        self.objlist = [2]
        
        self.mode = mode
        
        self.seg_list = []
        self.list_segmentation = [] # only the 2th class have the ground truth segmentation 
        self.list_rgb = []  # save the path of the rgbd image 
        self.list_depth = []  # save the path of the depth image 
        self.list_label = []  # save the path of the label image 
        self.list_obj = []  # save the list of objlist(the folder name)  
        self.list_rank = [] # save the index of data in the folder
        self.meta = {}  # meta_file have the ground truth information
        self.pt = {}  
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine
        self.onehot = onehot
        self.seg_mode = seg
        self.vertex_reg_mode = vertex_reg
        self.vertex_reg_hough_mode = vertex_reg_hough
        
        item_count = 0
        for item in self.objlist:
            if self.seg_mode and item!=2:
                continue
            if self.mode != 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                    
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                elif self.seg_mode:
                    self.list_label.append('{0}/data/{1}/mask_all/{2}.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))
                
                self.list_obj.append(item)
                self.list_rank.append(int(input_line))
            
            
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            self.meta[item] = yaml.load(meta_file, Loader=yaml.FullLoader)
#            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043
        self.intrin_matrix = np.array([[self.cam_fx, 0 , self.cam_cx, 0], 
                                       [0, self.cam_fy, self.cam_cy, 0], 
                                       [0, 0, 1, 0]])

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.num = num  # this if the number of points
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]
        # This is the pixel value for each class, it has the same order with data in the folder
        # pixel value == 21 => the object is 01 in the folder
#        self.seg_list = [0, 21, 43, 64, 85,106,128, 149, 170, 191, 213, 234, 255]
        self.seg_list = [0, 21, 43, 106,128, 170, 191, 213, 234, 255]
        self.weight_clsss = np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.cls_indexes = [i for i in range(len(self.seg_list))]
        self.num_classes = len(self.seg_list)
        self.seg_label_to_gt_label = [0, 1, 2, 5, 6, 8, 9, 10, 11, 12]
        self.gt_label_to_seg_label = [0, 1, 2, -1, -1, 3, 4, -1, 5, 6, 7, 8, 9]
        self.extents = self.get_extents()

    def get_extents(self):
        extents = np.zeros((self.num_classes, 3))
        for i in range(1,len(self.gt_label_to_seg_label)):
            if self.gt_label_to_seg_label[i]>0:
                pt = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % i))
                model_points = pt / 1000.0
                points_arr = np.array(model_points)
                xyz_min = np.min(points_arr, axis = 0)
                xyz_max = np.max(points_arr, axis = 0)
                extents[self.gt_label_to_seg_label[i], :] = xyz_max - xyz_min
        return -np.sort(-extents, axis = 1)
    
    
    
    
    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index]))
        label_in = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index] # what the label for this index 
        rank = self.list_rank[index]      
        img = np.array(img)[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
 
       # Since the second object have more information
        if obj == 2:
            # if the object is the second  object
            if not self.vertex_reg_mode:
            # if we only need the  information of the second object 
                for i in range(0, len(self.meta[obj][rank])):
                    if self.meta[obj][rank][i]['obj_id'] == 2:
                        meta = self.meta[obj][rank][i]
                        break
            else:
            # if we need all the information for all objects
                meta = self.meta[obj][rank]
        else:
            meta = self.meta[obj][rank][0]
            
        label = None
#        unique, counts = np.unique(label_in, return_counts=True)
#        print(unique, counts)
        if self.seg_mode:
            label = np.zeros((len(self.seg_list), label_in.shape[0], label_in.shape[1]))
            for j in range(len(self.seg_list)):
                label[j, :] = label_in == self.seg_list[j]
            if not self.onehot:
                label = np.argmax(label, axis = 0)
                
#        plt.imshow(label, cmap = 'hot', interpolation = 'nearest')
#        plt.show()
#        plt.pause(100)
        
        # without vertex_reg mod, only return the data for segmentation
        if not self.vertex_reg_mode:
            return torch.from_numpy(img.astype(np.float32)), \
               torch.from_numpy(label.astype(int))
        
        # with vertex_reg mode 
        # Relate with meta
        bboxs = np.zeros((self.num_classes, 5))  # all the bounding box in the same image for different class
        extrin_matrixs = np.zeros((self.num_classes, 4, 4))
        extrin_matrixs[:,3,3] = 1
        centers = np.zeros((self.num_classes, 2))
        depth_centers = np.zeros((self.num_classes, 1))
        for sub_meta in meta:
            seg_index = self.gt_label_to_seg_label[int(sub_meta['obj_id'])]
            # preprocess the bounding box information 
            bboxs[seg_index, 0] = 1
            rmin, rmax, cmin, cmax = get_bbox(sub_meta['obj_bb'])
            bboxs[seg_index, 1:] = [cmin, rmin, cmax, rmax]
    
            # preprocess the pose information 
            extrin_matrixs[seg_index, 0:3, 0:3] = np.resize(np.array(sub_meta['cam_R_m2c']), (3, 3))
            extrin_matrixs[seg_index, 0:3, 3] = np.array(sub_meta['cam_t_m2c'])
            obj_center = np.ones((4,1))
            obj_center[0:3, 0] = extrin_matrixs[seg_index, 0:3, 3]
            center_homo = self.intrin_matrix.dot(obj_center)
            centers[seg_index, :] = center_homo[0:2].reshape(-1)/center_homo[2]
            depth_centers[seg_index, :] = extrin_matrixs[seg_index, 2, 3]
        if  self.onehot: 
            label_single_channel = np.argmax(label, axis = 0)
        else:
            label_single_channel = label
        vertex_targets, vertex_weights = center_img_gt._vote_centers_train(label_single_channel, self.cls_indexes, 
                                                                           centers, depth_centers, self.num_classes)
            
        # with vertex reg and hough voting 
        # load the point cloud and set the size of the point clout to be num_pt_mesh_small
        
        # meta data include camera intrinsic matrix 
        meta = np.zeros((48,))
        meta[0] = self.cam_fx
        meta[4] = self.cam_fy
        meta[2] = self.cam_cx
        meta[5] = self.cam_cy
        
        # gt give to hough voting information to calculate weight 
        gt_hough = np.zeros((10,1))
        
        
        
        return (torch.from_numpy(img.astype(np.float32)), 
                    torch.from_numpy(label.astype(int)), 
                    torch.from_numpy(vertex_targets.astype(np.float32)), 
                    torch.from_numpy(vertex_weights.astype(np.float32)),
                    torch.from_numpy(self.extents.astype(np.float32)),
                    torch.from_numpy(meta.astype(np.float32)),
                    torch.from_numpy(gt_hough.astype(np.float32)),
                    torch.from_numpy(extrin_matrixs.astype(np.float32)),
                    torch.from_numpy(bboxs.astype(np.float32)))
        
        """
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)])
        """
        
    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


    
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)
