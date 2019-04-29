# --------------------------------------------------------
# PoseCNN with pytorch  
# Author: university of michigan EECS442 
# --------------------------------------------------------

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from tqdm import tqdm
from eval_net import cal_AP
import shutil
import cv2
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from PIL import Image
import pdb
import copy

#from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset_posecnn import PoseDataset as PoseDataset_linemod
from datasets.YCB.dataset import Posedataset as PoseDataset_ycb

#from lib.network import PoseNet, PoseRefineNet
#from lib.loss import Loss
from lib.vgg16_convs import vgg16_convs
from lib.vgg16_convs_combine_seg_center import vgg16_convs_comb_seg_center
from lib.center_est_funcs import *

class arguments():
    def __init__(self):
        self.dataset = 'linemod'  
        self.dataset_root = '/home/ubuntu/EECS442_CourseProject/datasets/linemod/Linemod_preprocessed'
        self.num_objects = 13
        
        self.flag_pretrained_vgg = False
        self.flag_pretrained = True
        self.path_pretrained = 'trained_model/pretrained-posecnn-linemod/checkpoint.pth.tar'
#         self.num_pretrain_param = 36
        # 26 for vgg part         36 for vgg+seg    
        # 46 for vgg+seg+center   42 for vgg+seg+center(combine seg and center)
        self.num_pretrain_param_load = 46  
        self.num_pretrain_param_freeze = 0  

        self.save_model = True
        self.save_test_result = True
        self.save_train_result = True
        self.save_hough_result = True
        self.color = [(255, 255, 255), (0, 255, 0), (255, 0, 0),  (0, 0, 255), (255, 255, 0), 
                      (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), 
                      (128, 128, 0), (128, 0, 128), (0, 128, 128),(64, 0, 0), (0, 64, 0), (0, 0, 64)]
        self.arch = 'Semantic_Segmentation'
        self.gpu = True
        self.niter_print = 50
        self.nepoch_save = 1
        self.num_pretrain_param_vgg = 26
        
        self.batch_size = 4
        self.workers = 0

        self.lr = 2e-4
        
        self.iteration = 2
        self.nepoch = 5

        self.repeat_epoch = 1
        
        self.noise_trans = 0.03
        self.manualSeed = 0
        self.num_units = 10 
        self.scales = 1
        self.threshold_label = 1 
        self.vote_threshold = 1
        self.refine_start = False
        
        # FOR CENTER ESTIMATION
        self.train_single_frame = True
        self.vertex_reg = True
        self.vertex_reg_hough = True
        self.combine_seg_center = True
        self.combine_loss = True
        
opt = arguments()   

def save_image_fun(img, name):
    if img.ndim == 3:
        img = np.transpose(img, (1,2,0)).astype(np.int)
    name = os.path.join('log', name+'.jpg')
    cv2.imwrite(name, img)

def outputtoimg(output):
    if output.ndim == 2:
        return output
    else:
        img = np.argmax(output, axis = 0)
        return img

def imgscale(img, scale, offset):
    global opt
    return img*scale + offset

def save_image( images, labels, output_seg, vertex_targets, output_center, epoch, mode = 'train', i=0, index = 0):
    start_str_img = None
    start_str_class = None
    if mode == 'train':
        start_str_img = 'report_result/seg_center_dir/epoch{0}_iter{1}_image{2}'.format(epoch, i, index)
        start_str_class = 'report_result/seg_center_dir/epoch{0}_iter{1}_image{2}'.format(epoch, i, index)
    else:
        start_str_img = 'test/image{0}'.format(index)
        start_str_class = 'test/image{0}'.format(index)
        
    save_image_fun(images.cpu().numpy()[0], start_str_img)
    save_image_fun(imgscale(labels[0].cpu().detach().numpy(), 255//opt.num_objects, 0), start_str_img+'_seg_gt')
    save_image_fun(imgscale(outputtoimg(output_seg.cpu().detach().numpy()[0]), 255//opt.num_objects, 0), start_str_img+'_seg')
    if opt.vertex_reg:
        for j in range(1, opt.num_objects):
            save_image_fun(imgscale(vertex_targets[0, j*3].cpu().detach().numpy(), 100, 100), start_str_class + '_class{0}_vertex_x_gt'.format(j))
            save_image_fun(imgscale(vertex_targets[0, j*3+1].cpu().detach().numpy(), 100, 100), start_str_class + '_class{0}_vertex_y_gt'.format(j))
            save_image_fun(imgscale(vertex_targets[0, j*3+2].cpu().detach().numpy(), 0.1, 0), start_str_class + '_class{0}_vertex_depth_gt'.format(j))
        
            save_image_fun(imgscale(output_center[0, j*3].cpu().detach().numpy(), 100, 100), start_str_class + '_class{0}_vertex_x'.format(j))
            save_image_fun(imgscale(output_center[0, j*3+1].cpu().detach().numpy(), 100, 100), start_str_class + '_class{0}_vertex_y'.format(j))
            save_image_fun(imgscale(output_center[0, j*3+2].cpu().detach().numpy(), 0.1, 0), start_str_class + '_class{0}_vertex_depth'.format(j))
            # save the image for the inverse of the difference between gt and prediction
                    
                
def save_bbox_center(images, output_hough, epoch, i):
    index_bbox = 0
    output_hough = output_hough.cpu().numpy()
    images_plot = images.cpu().numpy()
    images_plot = images_plot.transpose(0,2,3,1).astype(np.uint8)
    for ii in range(images.shape[0]):
        img3 = copy.deepcopy(images_plot[ii])
        for jj in range(index_bbox, output_hough.shape[0]):
            if output_hough[jj,0] != ii:
                break
            index_bbox +=1
            width = output_hough[jj,4] - output_hough[jj,2]
            height = output_hough[jj,5] - output_hough[jj,3] 
            if width<4 or height<4:
                continue
            img2 = np.zeros((480,640,3), np.uint8)
            index_class = int(output_hough[jj,1])
            cv2.rectangle(img2, (int(output_hough[jj,2]), int(output_hough[jj,3])),
                          (int(output_hough[jj,4]), int(output_hough[jj,5])),opt.color[index_class],3)
            c_x = int(output_hough[jj,2] + 0.5*width)
            c_y = int(output_hough[jj,3] + 0.5*height)
            cv2.circle(img2, (c_x, c_y), 3, opt.color[index_class], 2)
            index = np.where(img2>0)
            img3[index] = img2[index]
        name = 'log/report_result/bbox_center/epoch{0}_iter{1}_image{2}_bboxs.png'.format(epoch, i, ii)
        cv2.imwrite(name, img3)
    
    
    
def train(trainloader, net, criterion, criterion_center, optimizer, device, device_cpu):
    global opt
    loss_his = []
    images = None
    labels = None
    vertex_targets = None 
    vertex_weights = None
    extents = None
    meta = None
    gt_hough = None
    extrin_matrixs_gt= None 
    bboxs_gt = None
    for epoch in range(opt.nepoch): #TODO decide epochs
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        net.train()
        start = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            if opt.train_single_frame:
                # train the network part by part 
                if opt.vertex_reg == True:
                  # Only train the center-voting part
                  images, labels, vertex_targets, vertex_weights, extents, meta, gt_hough, extrin_matrixs_gt, bboxs_gt = data
                  images = images.to(device)
                  labels = labels.type('torch.LongTensor').to(device)
                  vertex_targets = vertex_targets.to(device)
                  vertex_weights = vertex_weights.to(device)
                  extents = extents.to(device_cpu)
                  meta = meta.to(device_cpu)
                  gt_hough = gt_hough.to(device_cpu)
                  # change all the tensor type to CPU
                  output_seg, output_center_dir, output_hough = net(images, extents, meta, gt_hough, 1, device_cpu)
#                   print("This is the output of hough voting: ", output_hough.cpu().numpy())
                  loss_seg = criterion(output_seg, labels)
                  loss_center = criterion_center(output_center_dir, vertex_targets)
                  loss = loss_seg + loss_center
                else:
                  # Only train the segmentation part
                  images, labels = data
                  images = images.to(device)
                  labels = labels.type('torch.LongTensor').to(device)
                  output_seg = net(images)
                  loss = criterion(output_seg, labels)
            else:
                # from the begining to the end 
                print('Empty for this part')
                loss = 0
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i %opt.niter_print == opt.niter_print-1:
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / opt.niter_print, end-start))
                start = time.time()
                loss_his.append(running_loss / opt.niter_print)
                running_loss = 0.0
            if opt.save_train_result and (epoch % opt.nepoch_save == 0 or epoch == opt.nepoch-1) and i%50 == 0:
                save_image(images, labels, output_seg, vertex_targets, output_center_dir, epoch, 'train', i, i//50)
                if opt.save_hough_result:
                    save_bbox_center(images, output_hough, epoch, i)
#                     pdb.set_trace()
    return loss_his


def test(testloader, net, criterion, criterion_center, device, device_cpu):
    '''
    Function for testing.
    '''
    global opt
    losses = 0.
    cnt = 0
    cnt_image = 0
    with torch.no_grad():
        net = net.eval()
        loss = 0.0
        vertex_targets = None
        output_center = None
        for data in tqdm(testloader):
            if opt.train_single_frame:
                if opt.vertex_reg == True:
                  # Only train the center-voting part
                  images, labels, vertex_targets, vertex_weights, extents, meta, gt_hough, extrin_matrixs_gt, bboxs_gt = data
                  images = images.to(device)
                  labels = labels.type('torch.LongTensor').to(device)
                  vertex_targets = vertex_targets.to(device)
                  vertex_weights = vertex_weights.to(device)
                  extents = extents.to(device_cpu)
                  meta = meta.to(device_cpu)
                  gt_hough = gt_hough.to(device_cpu)
                  output_seg, output_center_dir, output_center= net(images, extents, meta, gt_hough, 0, device_cpu)
                  
                  loss_seg = criterion(output_seg, labels)
                  loss_center = criterion_center(output_center_dir, vertex_targets)
                  loss_temp = loss_seg + loss_center 
                  loss += loss_temp.item()
                else:
                  # Only train the segmentation part
                  images, labels = data
                  images = images.to(device)
                  labels = labels.type('torch.LongTensor').to(device)
                  output_seg = net(images)
                  loss_temp = criterion(output_seg, labels)
                  loss += loss_temp.item()
            else:
                # this part corresponding to the network is end to end
                # and only have one loss 
                print('Empty for this part')
                loss = 0
                pass 
            
            if opt.save_test_result and cnt%4 == 3:
                cnt_image+=1
                save_image(images, labels, output_seg, vertex_targets, output_center_dir, 0, 'test', i, cnt_image)
                
            cnt += 1
    print(loss / cnt)
    return (loss/cnt)



def loadpretrain(net, pretrained_dic, device, num):
    pretrained_list = list(pretrained_dic.items())
    net_dic = net.state_dict()
    net_dic_new = net_dic
    count = 0
    for k, v in net_dic.items():
        name_temp, value_pretrained = pretrained_list[count]
        net_dic_new[k] = value_pretrained
        count+=1
        if count >= num:
            break
    return net.load_state_dict(net_dic_new)


def save_checkpoint(state, is_best, filename='trained_model/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'trained_model/model_best.pth.tar')
        

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    device_cpu = torch.device('cpu')
    if opt.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')    
    else:
        device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        torch.backends.cudnn.benchmark = True

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 10
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return
    
    # check for the network mode
    if not opt.vertex_reg and opt.vertex_reg_hough:
        assert ValueError('Mode Incorrect')

    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, 
                                  opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, 
                                      opt.refine_start, False, True, opt.vertex_reg, opt.vertex_reg_hough)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, 
                                              shuffle=True, num_workers=opt.workers)
    
    
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 
                                       0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root,
                                           0.0, opt.refine_start, 
                                           False, True, opt.vertex_reg, opt.vertex_reg_hough)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                                 shuffle=False, num_workers=opt.workers)
    
    if opt.dataset == 'ycb':
        pass
    else:
        ap_data = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, 
                                      opt.refine_start, True, True, opt.vertex_reg, opt.vertex_reg_hough)
    ap_loader = torch.utils.data.DataLoader(ap_data, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
#     print(opt.sym_list)
#    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    
    # Network, optimizer and loss               
    net = vgg16_convs(None, opt.num_objects, opt.num_objects, opt.scales, opt.threshold_label, 
                      opt.vote_threshold, opt.vertex_reg, opt.vertex_reg, opt.vertex_reg_hough)
#    net = vgg16_convs_comb_seg_center(None, opt.num_objects, opt.num_objects, opt.scales, opt.threshold_label, 
#                                      opt.vote_threshold, opt.vertex_reg, opt.combine_seg_center)
    
    optimizer = optim.Adam(net.parameters(), lr = opt.lr)
    
    weight_class = torch.from_numpy(dataset.weight_clsss).type('torch.FloatTensor').to(device)
#    criterion = nn.CrossEntropyLoss(weight_class) 
    criterion = nn.CrossEntropyLoss() 
    criterion_center = nn.SmoothL1Loss()
    
    # Load pretrained model
    if opt.flag_pretrained and not opt.flag_pretrained_vgg:
        # load out model trained before as initialization to continue  
        if os.path.isfile(opt.path_pretrained):
            print("=> Loading Checkpoint '{}'".format(opt.path_pretrained))
            pre_trained = torch.load(opt.path_pretrained)
            net_dic = net.state_dict()
            net_dic_new = net_dic
            pretrained_dic = pre_trained['state_dict']
            pretrained_list = list(pretrained_dic.items())
#            net.load_state_dict()
            if opt.num_pretrain_param_load > 0:
                count = 0
                for k, v in net_dic.items():
                    if count >= opt.num_pretrain_param_load:
                        break
                    name_temp, value_pretrained = pretrained_list[count]
                    if opt.gpu:
                        net_dic_new[k] = value_pretrained
                    else:
                        net_dic_new[k] = value_pretrained.cpu()
                    count+=1
                    
                    
            net.load_state_dict(net_dic_new)
            """
            optimizer.load_state_dict(pre_trained['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.cuda.is_available:
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            """
            print("=> Loaded Checkpoint '{}'".format(opt.path_pretrained))
        else:
            assert ValueError("no pretrained_model found at {}".format(opt.path_pretrained))
        
        count = 0
        for param in net.parameters():
            if count >= opt.num_pretrain_param_freeze:
                break
            param.requires_grad = False
            count+=1
    elif not opt.flag_pretrained and opt.flag_pretrained_vgg:
        # load the pretrained weight of VGG16 net 
        # 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
        pretrained_dic = torch.load('pretrained_model\\vgg16-397923af.pth')
        pretrained_list = list(pretrained_dic.items())
        net_dic = net.state_dict()
        net_dic_new = net_dic
        count = 0
        for k, v in net_dic.items():
            name_temp, value_pretrained = pretrained_list[count]
            net_dic_new[k] = value_pretrained
            count+=1
            if count >= opt.num_pretrain_param_vgg:
                break
        net.load_state_dict(net_dic_new)
        count = 0
        for param in net.parameters():
            param.requires_grad = False
            count+=1
            if count>=opt.num_pretrain_param_vgg:
                break
    elif not opt.flag_pretrained and not opt.flag_pretrained_vgg:
        print('without laod any pretrained model')
    else:
        print('Collision with the flag of load vgg param and laod pretrain model')
    

    net.to(device)
    loss_his = []
    loss_his = train(trainloader, net, criterion, criterion_center, optimizer, device, device_cpu)
    
    print('>>>>>>>>----------Training Finished!---------<<<<<<<<')
    
    test_loss = 0
    test_loss = test(testdataloader, net, criterion, criterion_center, device, device_cpu)
    
    print('>>>>>>>>----------AP---------<<<<<<<<')
#     aps = None
#     if opt.train_single_frame:
#         aps = cal_AP(ap_loader, net, criterion, device, opt.num_objects, opt)
#         aps = np.array(aps)
#         print('Final mean AP : {}'.format(np.mean(aps)))
        
    print('>>>>>>>>----------Save the model weights!---------<<<<<<<<')
    if opt.save_model:   
        # save the trained model
        save_checkpoint({
            'epoch': opt.nepoch,
            'arch': opt.arch,
            'state_dict': net.state_dict(),
            'test_loss': test_loss,
            'aps': aps,
            'optimizer' : optimizer.state_dict(),
        }, False)
        
    print('>>>>>>>>----------Loss History---------<<<<<<<<')
    np.save('log//loss//loss', np.array(loss_his))
    plt.figure()
    plt.plot(loss_his)
    plt.show()
# <<<<<<< HEAD
    plt.savefig('/home/ubuntu/EECS442_CourseProject/log/loss/unfreeze_seg_ctr.png')
# =======
#     plt.savefig('log//loss//loss.png')
  
# >>>>>>> ce96c070c17b981e90464ae0b458ab905b1009db

if __name__ == '__main__':
    main()
