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

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
    
def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)
    
def create_model(num_classes, size=(1400,1700), norm = None, pretrained=True, coco_model=False):
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
    
    model = FasterRCNN(backbone = backbone, num_classes=num_classes)
    model.transform = GeneralizedRCNNTransform( size[0], size[1], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], fixed_size= size)
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
    
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)