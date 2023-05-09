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
    # Find the maximum sequence length
    max_length = max([len(seq) for seq in sequences])

    # Pad the sequences with zeros
    padded_sequences = [torch.nn.functional.pad(seq, (0, 0, 0, max_length - len(seq)), value=padding_value) for seq in sequences]

    # Stack the padded sequences into a single tensor
    padded_tensor = torch.stack(padded_sequences, dim=0)

    # Transpose the tensor if batch_first is False
    if not batch_first:
        padded_tensor = padded_tensor.transpose(0, 1)

    return padded_tensor


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
        self.crossview = CrossviewTransformer()
        self.pos_encode = PositionEmbeddingSine(1024)
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
        
        box_features_CC = pad_sequence(box_features_CC)
        box_features_MLO = pad_sequence(box_features_MLO)
        CC_key_padding_mask = box_features_CC == 0
        MLO_key_padding_mask = box_features_MLO == 0
        CC_pos = self.pos_encode(proposals_CC)
        MLO_pos = self.pos_encode(proposals_MLO)
        box_features_CC, box_features_MLO = self.crossview(box_features_CC, box_features_MLO, CC_key_padding_mask, MLO_key_padding_mask,\
            CC_pos= CC_pos, MLO_pos= MLO_pos)
        
        class_logits, box_regression = self.box_predictor(box_features)


        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss1(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses

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