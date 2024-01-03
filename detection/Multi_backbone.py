import numpy as np
import torch 
import torchvision
from torch import nn, Tensor
from torchvision.models.resnet import resnet50, ResNet50_Weights, ResNet  
from collections import OrderedDict


class Cross_transformer_backbone_inter(nn.Module):
    def __init__(self, backbone, transformer):
        self.backbone= backbone 
        self.transformer = transformer
    def forward_view(self, x):
        out = OrderedDict()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        out['1']= x
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
    def forward(self, CC, MLO):
        feat_CC = self.forward_view(CC)
        feat_MLO = self.forward_view(MLO)
        feat_CC, feat_MLO = self.transformer(feat_CC, feat_MLO)
