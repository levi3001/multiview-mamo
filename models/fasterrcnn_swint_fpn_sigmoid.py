import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead, FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from detection.Custom_roi_heads import Custom_roi_heads
from torchvision.ops import MultiScaleRoIAlign
from torch import nn as nn
from torch import Tensor
import torch
from typing import List, Tuple
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers, BackboneWithFPN
from detection.fpn import _swin_fpn_extractor
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
import torch.nn.functional as F
from utils.norm import LayerNorm2d, get_layer, set_layer


def create_model(num_classes, size=(1400,1700), norm = None, pretrained=True, coco_model=False, loss_type ='fasterrcnn1'):
    weights_backbone= Swin_T_Weights.IMAGENET1K_V1
    weights_backbone = Swin_T_Weights.verify(weights_backbone)
    #weights_backbone = None
    
    backbone = swin_t(weights=weights_backbone, progress = True).features

    for name, module in backbone.named_children():
        print(name)
    print(backbone)
    
    backbone = _swin_fpn_extractor(backbone)
    
    model = FasterRCNN(backbone = backbone, num_classes=num_classes)
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
            loss_type = loss_type
    )
    
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)