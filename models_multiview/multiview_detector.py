import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import cv2
from utils import transforms as T


import numpy as np
import torch 
import torchvision
from torch import nn, Tensor
from torchvision.models.detection import generalized_rcnn, faster_rcnn, roi_heads
from detection.Multi_roi_heads import Multi_roi_heads
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F
from utils.norm import get_layer, set_layer, LayerNorm2d


class Multiview_fasterrcnn(faster_rcnn.FasterRCNN):
    def __init__(self, backbone, num_classes):
        super().__init__(backbone = backbone, num_classes= num_classes)
        
    def eager_outputs(self, loss_CC, loss_MLO, detections_CC, detections_MLO):
        if self.training:
            return loss_CC, loss_MLO

        return detections_CC, detections_MLO
    
    def backbone_forward(self, images, targets):
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        return images, features, targets, original_image_sizes
    def forward(self, image_CC, image_MLO, target_CC = None, target_MLO = None):
        image_CC, feat_CC, target_CC, original_image_sizes_CC = self.backbone_forward(image_CC, target_CC)
        image_MLO,  feat_MLO, target_MLO, original_image_sizes_MLO = self.backbone_forward(image_MLO, target_MLO)
        if isinstance(feat_CC, torch.Tensor):
            feat_CC = OrderedDict([("0", feat_CC)])
            feat_MLO = OrderedDict([("0", feat_MLO)])
        proposals_CC, proposal_losses_CC = self.rpn(image_CC, feat_CC, target_CC)
        proposals_MLO, proposal_losses_MLO = self.rpn(image_MLO, feat_MLO, target_MLO)
        detections_CC, detections_MLO, detector_losses_CC, detector_losses_MLO = self.roi_heads(feat_CC, proposals_CC, image_CC.image_sizes,\
                                                                                                feat_MLO, proposals_MLO, image_MLO.image_sizes,\
                                                                                                    target_CC, target_MLO)
        detections_CC = self.transform.postprocess(detections_CC, image_CC.image_sizes, original_image_sizes_CC)  # type: ignore[operator]
        detections_MLO = self.transform.postprocess(detections_MLO, image_MLO.image_sizes, original_image_sizes_MLO)
        losses_CC = {}
        losses_CC.update(detector_losses_CC)
        losses_CC.update(proposal_losses_CC)
        losses_MLO = {}
        losses_MLO.update(detector_losses_MLO)
        losses_MLO.update(proposal_losses_MLO)
        return self.eager_outputs(losses_CC, losses_MLO, detections_CC, detections_MLO)

def create_model(num_classes, size= (1400, 1700), norm= None, pretrained=True, coco_model=False, use_self_attn = False, compute_attn= False, loss_type ='fasterrcnn1', **kwargs):
    # Load Faster RCNN pre-trained model
    weights_backbone= ResNet50_Weights.IMAGENET1K_V1
    weights_backbone = ResNet50_Weights.verify(weights_backbone)



    is_trained = weights_backbone is not None
    trainable_backbone_layers=5
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    if norm == None:
        norm_layer = misc_nn_ops.FrozenBatchNorm2d
    else:
        norm_layer = nn.BatchNorm2d
    backbone = resnet50(weights=weights_backbone, progress = True, norm_layer=norm_layer)


    if norm == 'ln' or norm =='gn':
        for name, module in backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Get current bn layer
                bn = get_layer(backbone, name)
                
                
                if norm == 'ln':
                    # Create new ln layer
                    ln = LayerNorm2d(bn.num_features)
                    # Assign mn
                    print("Swapping {} with {}".format(bn, ln))
                    set_layer(backbone, name, ln)
                elif norm =='gn':
                    # Create new gn layer
                    gn = nn.GroupNorm(1, bn.num_features)
                    # Assign mn
                    print("Swapping {} with {}".format(bn, gn))
                    set_layer(backbone, name, gn)
    print(backbone)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    
    model = Multiview_fasterrcnn(backbone = backbone, num_classes=num_classes)
    model.transform = GeneralizedRCNNTransform( size[0], size[1], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], fixed_size= size)
    out_channels = model.backbone.out_channels
    box_roi_pool=None
    box_head=None
    box_predictor=None
    box_score_thresh= 0
    box_nms_thresh=0.1
    box_detections_per_img=100
    box_fg_iou_thresh=0.5
    box_bg_iou_thresh=0.5
    box_batch_size_per_image=512
    box_positive_fraction=0.25
    bbox_reg_weights=None
    if box_roi_pool is None:
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

    if box_head is None:
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

    if box_predictor is None:
        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, num_classes)
    model.roi_heads = Multi_roi_heads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            use_self_attn,   ###use self attention in decoder
            compute_attn= compute_attn,
            loss_type = loss_type
    )
    


    return model


if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)