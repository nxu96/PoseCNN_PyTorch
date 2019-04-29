import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#######################################################################
########## Loss functions adapted from the tensorflow version #########
#######################################################################

def loss_cross_entropy_single_frame(scores, labels):
    """
    scores: a tensor [batch_size, height, width, num_classes]
    labels: a tensor [batch_size, height, width, num_classes]
    """

    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(labels * scores, reduction_indices=[3])
        loss = tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(labels)+1e-10)

    return loss


def torch_loss_cross_entropy_single_frame(scores, labels):

    """
    scores: a tensor [batch_size, height, width, num_classes]
    labels: a tensor [batch_size, height, width, num_classes]
    """

    cross_entropy = -torch.sum(labels * scores, dim=3)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)
    loss = torch.tensor(loss, requires_grad=True)

    return loss

###########################################################################

def smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):

    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = tf.multiply(vertex_weights, vertex_diff)
    abs_diff = tf.abs(diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
    in_loss = tf.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = tf.div( tf.reduce_sum(in_loss), tf.reduce_sum(vertex_weights) + 1e-10 )

    return loss

def torch_smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights, sigma=1.0, VERTEX_W=5.0):

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

###########################################################################
"""
logits should be domain_score, labels should be label_domain
"""

def torch_loss_domain(logits, labels, ADAPT_WEIGHT=0.1):

    loss = ADAPT_WEIGHT * torch.mean(F.nll_loss(F.softmax(logits), labels))
    loss = torch.tensor(loss, requires_grad=True)

    return loss

###########################################################################


