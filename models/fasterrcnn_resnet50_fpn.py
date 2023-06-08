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
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.resnet import resnet50, ResNet50_Weights
import torch.nn.functional as F
from utils.norm import LayerNorm2d, get_layer, set_layer

    
def create_model(num_classes, size=(1400,1700), norm = None, pretrained=True, coco_model=False, loss_type ='fasterrcnn'):
    weights_backbone= ResNet50_Weights.IMAGENET1K_V1
    weights_backbone = ResNet50_Weights.verify(weights_backbone)
    #weights_backbone = None


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
    
    model = FasterRCNN(backbone = backbone,
                       num_classes=num_classes,
                           box_roi_pool=None,
                            box_head=None,
                            box_predictor=None,
                            box_score_thresh= 0,
                            box_nms_thresh=0.1,
                            box_detections_per_img=100,
                            box_fg_iou_thresh=0.5,
                            box_bg_iou_thresh=0.5,
                            box_batch_size_per_image=512,
                            box_positive_fraction=0.25,
                            bbox_reg_weights=None
                            )
    model.transform = GeneralizedRCNNTransform( size[0], size[1], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], fixed_size= size)

        # Get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)