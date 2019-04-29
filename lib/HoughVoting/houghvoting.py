import torch
import torch.nn as nn
import numpy as np
import HoughVoting

# class HoughVoting(nn.Module):
#     # initialization
#     def __init__(self):
#         self.vertex_channels = 3
        
#     def forward(self, label, vertex, extents, meta_data, gt, is_train):
        ## label : (batch size, height, weight) 
        ## vertex: b, h, w, 3 * num_cls

                                                
        # # flatten 
        # v_meta_data = meta_data.view(-1)
        # v_gt = gt.view(-1)
        # v_extents = extents.view(-1)
        # # batch size
        # batch_size = label.shape[0]
        # # height
        # height = label.shape[1]
        # # width
        # width = label.shape[2]
        # # num of cls
        # num_cls = vertex.shape[3] / self.vertex_channels
        # # num of meta data
        # num_meta_data = meta_data.shape[3]
        # # num of gt
        # num_gt = gt.shape[0]
        
        # getBb3Ds(v_extents, num_cls)
        # index_meta_data = 0
        # # for each image run hough voting 
        # for n in range(batch_size):
        #     idx_label = 
        #     idx_vertex = 
        #     fx = v_meta_data(index_meta_data+0)
        #     fy = v_meta_data(index_meta_data+4)
        #     px = v_meta_data(index_meta_data+2)
        #     py = v_meta_data(index_meta_data+5)
        #     outputs = voting(idx_label, idx_vertex, label, vertex, bb3Ds, n, height, weight, num_cls, is_train, fx, fy, px ,py)
        #     index_meta_data = index_meta_data + 1
        # if (outputs.size() == 0):
        #     print("No detection")
        #     # add a dummy detection to the output? 
        #     roi = torch.empty((14,1))
        #     roi[0] = 0
        #     roi[1] = -1
        #     # add back to outputs
        #     outputs[]
# class HoughVotingFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weights, bias, old_h, old_cell):
#         outputs = HoughVoting.forward(label, vertex, extents, meta_data, gt, is_train =True)
#         # new_h, new_cell = outputs[:2]
#         # variables = outputs[1:] + [weights]
#         # ctx.save_for_backward(*variables)
#         return outputs

#     @staticmethod
#     def backward(ctx, grad_h, grad_cell):
#         outputs = lltm_cpp.backward(
#             grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
#         d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
#         return d_input, d_weights, d_bias, d_old_h, d_old_cell


class HF(torch.nn.Module):
    def __init__(self):
        super(HF, self).__init__()
        # self.input_features = input_features
        # self.state_size = state_size
        # self.weights = torch.nn.Parameter(
        #     torch.empty(3 * state_size, input_features + state_size))
        # self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        # self.reset_parameters()
        self.vertex_channels = 3

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.state_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, +stdv)

    def forward(self, label, vertex, extents, meta_data, gt, is_train):
        outputs = HoughVoting.forward(label, vertex, extents, meta_data, gt, is_train)
        return outputs

    def backward(self, label, vertex):
        label_grad = torch.zeros(label.size())
        vertex_grad = torch.zeros(vertex.size())
        return label_grad, vertex_grad
        