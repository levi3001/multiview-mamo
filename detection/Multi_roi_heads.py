from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
from torchvision.models.detection.roi_heads import RoIHeads 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from detection.transformer import CrossviewTransformer
from detection.position_encoding import PositionEmbeddingSine

def pad_sequence(sequences, batch_first=True, padding_value=0):
    """
    Pad a list of variable length tensors with zeros to create a tensor with uniform shape.
    Args:
        sequences: list of PyTorch tensors
        batch_first: If True, the returned tensor will have shape (batch_size, max_sequence_length, *)
                     If False, the returned tensor will have shape (max_sequence_length, batch_size, *)
        padding_value: The value used to pad the sequences
    Returns:
        A PyTorch tensor of shape (batch_size, max_sequence_length, *) or (max_sequence_length, batch_size, *)
    """
    device= sequences[0].device
    # Find the maximum sequence length
    max_length = max([len(seq) for seq in sequences])
    # Pad the sequences with zeros
    padded_sequences = [torch.nn.functional.pad(seq, (0, 0, 0, max_length - len(seq)), value=padding_value) for seq in sequences]

    # Stack the padded sequences into a single tensor
    padded_tensor = torch.stack(padded_sequences, dim=0)
    mask_seq = [len(seq) for seq in sequences]
    mask= torch.zeros(padded_tensor.shape[:2], dtype = torch.bool)
    for i in range(len(sequences)):
        mask[i, mask_seq[i]:] = True
    # Transpose the tensor if batch_first is False
    if not batch_first:
        padded_tensor = padded_tensor.transpose(0, 1)
    padded_tensor = padded_tensor.to(device)
    mask = mask.to(device)
    return padded_tensor, mask


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

class Multi_roi_heads(RoIHeads):
    def __init__(self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # transformer
        use_self_attn,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,):
        super().__init__(box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None)
        
        
        self.crossview = CrossviewTransformer(use_self_attn=use_self_attn)
        self.pos_encode = PositionEmbeddingSine(512)
        
        
    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = torch.sigmoid(class_logits)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels
    
    def forward(
        self,
        feat_CC,  # type: Dict[str, Tensor]
        proposals_CC,  # type: List[Tensor]
        image_shapes_CC,  # type: List[Tuple[int, int]]
        feat_MLO,
        proposals_MLO,
        image_shapes_MLO,
        target_CC=None,  # type: Optional[List[Dict[str, Tensor]]]
        target_MLO=None,
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if target_CC is not None:
            for t in target_CC:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    print(t["boxes"])
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals_CC, matched_idxs_CC, labels_CC, regression_targets_CC = self.select_training_samples(proposals_CC, target_CC)
            proposals_MLO, matched_idxs_MLO, labels_MLO, regression_targets_MLO = self.select_training_samples(proposals_MLO, target_MLO)
        else:
            labels_CC = None
            labels_MLO = None
            regression_targets_CC = None
            regression_targets_MLO = None
            matched_idxs_CC = None
            matched_idxs_MLO = None
        box_features_CC = self.box_roi_pool(feat_CC, proposals_CC, image_shapes_CC)
        box_features_MLO = self.box_roi_pool(feat_MLO, proposals_MLO, image_shapes_MLO)

        box_features_CC = self.box_head(box_features_CC)
        box_features_MLO =self.box_head(box_features_MLO)
        boxes_per_image_CC = [boxes_in_image.shape[0] for boxes_in_image in proposals_CC]
        boxes_per_image_MLO = [boxes_in_image.shape[0] for boxes_in_image in proposals_MLO]
        box_features_CC = box_features_CC.split(boxes_per_image_CC, 0)
        box_features_MLO = box_features_MLO.split(boxes_per_image_MLO, 0)
        
        box_features_CC,  CC_key_padding_mask = pad_sequence(box_features_CC)
        box_features_MLO,  MLO_key_padding_mask = pad_sequence(box_features_MLO)
        
        CC_pos = self.pos_encode(proposals_CC)
        MLO_pos = self.pos_encode(proposals_MLO)
        CC_pos, _ = pad_sequence(CC_pos)
        MLO_pos, _ = pad_sequence(MLO_pos)
        
        #CC_key_padding_mask = box_features_CC == 0
        #MLO_key_padding_mask = box_features_MLO == 0
        
        box_features_CC, box_features_MLO = self.crossview(box_features_CC, box_features_MLO, CC_key_padding_mask, MLO_key_padding_mask,\
            CC_pos= CC_pos, MLO_pos= MLO_pos)
        
        box_features_CC = box_features_CC[~CC_key_padding_mask]
        box_features_MLO = box_features_MLO[~MLO_key_padding_mask]
        class_logits_CC, box_regression_CC = self.box_predictor(box_features_CC)
        class_logits_MLO, box_regression_MLO = self.box_predictor(box_features_MLO)


        result_CC: List[Dict[str, torch.Tensor]] = []
        result_MLO: List[Dict[str, torch.Tensor]] = []
        loss_CC = {}
        loss_MLO = {}
        if self.training:
            if labels_CC is None or labels_MLO is None:
                raise ValueError("labels cannot be None")
            if regression_targets_CC is None or regression_targets_MLO is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier_CC, loss_box_reg_CC = fastrcnn_loss1(class_logits_CC, box_regression_CC, labels_CC, regression_targets_CC)
            loss_classifier_MLO, loss_box_reg_MLO = fastrcnn_loss1(class_logits_MLO, box_regression_MLO, labels_MLO, regression_targets_MLO)
            loss_CC= {"loss_classifier": loss_classifier_CC, "loss_box_reg": loss_box_reg_CC }
            loss_MLO = {"loss_classifier": loss_classifier_MLO, "loss_box_reg": loss_box_reg_MLO}
        else:
            boxes_CC, scores_CC, labels_CC = self.postprocess_detections(class_logits_CC, box_regression_CC, proposals_CC, image_shapes_CC)
            boxes_MLO, scores_MLO, labels_MLO = self.postprocess_detections(class_logits_MLO, box_regression_MLO, proposals_MLO, image_shapes_MLO)
            num_images = len(boxes_CC)
            for i in range(num_images):
                result_CC.append(
                    {
                        "boxes": boxes_CC[i],
                        "labels": labels_CC[i],
                        "scores": scores_CC[i],
                    }
                )
            num_images = len(boxes_MLO)
            for i in range(num_images):
                result_MLO.append(
                    {
                        "boxes": boxes_MLO[i],
                        "labels": labels_MLO[i],
                        "scores": scores_MLO[i],
                    }
                )


        return result_CC, result_MLO, loss_CC, loss_MLO

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
    target = target[:, 1:]

    cls_loss = F.binary_cross_entropy_with_logits(
        pred_class_logits[:, 1:], target, reduction="none"
    )

    weight = 1

    loss = torch.sum(cls_loss * weight) / N
    return loss
