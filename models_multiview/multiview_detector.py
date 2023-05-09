import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch 
import torchvision
from torch import nn, Tensor
from torchvision.models.detection import generalized_rcnn, faster_rcnn, roi_heads
from detection.Multi_roi_heads import Multi_roi_heads
class Multiview_fasterrcnn(faster_rcnn):
    def __init__(self):
        super().__init__()
        self.roi_heads = Multi_roi_heads()
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
        return features, targets, original_image_sizes
    def forward(self, image_CC, image_MLO, target_CC = None, target_MLO = None):
        feat_CC, target_CC, original_image_sizes_CC = self.backbone_forward(image_CC, target_CC)
        feat_MLO, target_MLO, original_image_sizes_MLO = self.backbone_forward(image_MLO, target_MLO)
        if isinstance(feat_CC, torch.Tensor):
            feat_CC = OrderedDict([("0", feat_CC)])
            feat_MLO = OrderedDict([("0", feat_MLO)])
        proposals_CC, proposal_losses_CC = self.rpn(image_CC, feat_CC, target_CC)
        proposals_MLO, proposal_losses_MLO = self.rpn(image_MLO, target_MLO)
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
