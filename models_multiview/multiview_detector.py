import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch 
import torchvision
from torch import nn, Tensor
from torchvision.models.detection import generalized_rcnn, faster_rcnn, roi_heads

class Multiview_fasterrcnn(faster_rcnn):
    def __init__(self):
        super().__init__()
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
        return features, targets
    def forward(self, image_CC, image_MLO, target_CC, target_MLO):
        feat_CC, target_CC = self.backbone_forward(image_CC, target_CC)
        feat_MLO, target_MLO = self.backbone_forward(image_MLO, target_MLO)
        if isinstance(feat_CC, torch.Tensor):
            feat_CC = OrderedDict([("0", feat_CC)])
            feat_MLO = OrderedDict([("0", feat_MLO)])