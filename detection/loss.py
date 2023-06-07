from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
from torchvision.models.detection.roi_heads import RoIHeads 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



def fastrcnn_loss1(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.
    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    #print(class_logits, labels)
    regression_targets = torch.cat(regression_targets, dim=0)
    labels = torch.cat(labels, dim =0)
    classification_loss = sigmoid_cross_entropy_loss(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

# Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/custom_fast_rcnn.py#L113  # noqa
# with slight modifications
#Detectron2
def sigmoid_cross_entropy_loss( pred_class_logits, gt_classes):
    """
    Args:
        pred_class_logits: shape (N, K+1), scores for each of the N box. Each row contains the
        scores for K object categories and 1 background class
        gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
    """
    if pred_class_logits.numel() == 0:
        return pred_class_logits.new_zeros([1])[0]

    N = pred_class_logits.shape[0]
    K = pred_class_logits.shape[1] - 1

    target = pred_class_logits.new_zeros(N, K + 1)
    target[range(len(gt_classes)), gt_classes] = 1
    #target = target[:, 1:]

    # cls_loss = F.binary_cross_entropy_with_logits(
    #     pred_class_logits[:, 1:], target, reduction="none"
    # )
    cls_loss = F.binary_cross_entropy_with_logits(
        pred_class_logits, target, reduction="none"
    )
    
    weight = 1

    loss = torch.sum(cls_loss * weight) / N
    return loss



def Focal_loss(class_logits,  labels):
    labels = torch.cat(labels, dim =0)
    if class_logits.numel() == 0:
        return class_logits.new_zeros([1])[0]

    N = class_logits.shape[0]
    K = class_logits.shape[1] - 1

    target = class_logits.new_zeros(N, K + 1)
    target[range(len(labels)), labels] = 1
    loss = torchvision.ops.focal_loss(class_logits, labels, reduction = 'mean')
    return loss 
def DIOU_loss(box_regression, regression_targets):
   return torchvision.ops.diou_loss(box_regression, regression_targets, reduction = 'sum')

#focal loss + DIOU loss

def Mix_loss(class_logits, box_regression, labels, regression_targets):
    regression_targets = torch.cat(regression_targets, dim=0)
    labels = torch.cat(labels, dim =0)
    classification_loss = Focal_loss(class_logits, labels)
    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = DIOU_loss(box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset])
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss