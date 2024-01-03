from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers, BackboneWithFPN
import numpy as np
import torch 
import torchvision
from torch import nn, Tensor



class BackboneWithFPNSwin(BackboneWithFPN):
    def forward(self, x):
        x= self.body(x)
        for key in x:
            #print(x[key].shape)
            x[key] =x[key].permute(0,3,1,2)
           #print(x[key].shape)
        x= self.fpn(x)
        return x

def _swin_fpn_extractor(backbone):
    #returned_layers = [1, 2, 3, 4]
    returned_layers =['1', '3', '5', '7']
    return_layers = {f"{k}": str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = 96 
    in_channels_list = [in_channels_stage2 * 2 ** (i) for i in range(len(returned_layers))]
    out_channels = 256
    return BackboneWithFPNSwin(
        backbone, return_layers, in_channels_list, out_channels
    )