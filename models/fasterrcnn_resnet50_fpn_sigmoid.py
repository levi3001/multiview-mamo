import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from detection.Custom_roi_heads import Custom_roi_heads
from torchvision.ops import MultiScaleRoIAlign
from torch import nn as nn
from torch import Tensor
import torch
from typing import List, Tuple

    
    
def create_model(num_classes, pretrained=True, coco_model=False):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        #weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        rpn_fg_iou_thresh=0.5,
        rpn_bg_iou_thresh=0.3,
        box_score_thresh=0,
        box_nms_thresh=0.1,
        trainable_backbone_layers = 5,
    )
    model.transform = GeneralizedRCNNTransform(1400, 1700, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], fixed_size= (1400, 1700))
    out_channels = model.backbone.out_channels
    box_roi_pool=None
    box_head=None
    box_predictor=None
    box_score_thresh= 1e-7
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
    model.roi_heads = Custom_roi_heads(
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
    )
    if coco_model: # Return the COCO pretrained model for COCO classes.
        return model, coco_model
    
    # Get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)