from typing import Dict, List, Optional, Tuple
import math
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
from torchvision.models.detection.roi_heads import RoIHeads 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from detection.transformer import CrossviewTransformer
from detection.position_encoding import PositionEmbeddingSine, PositionEmbeddingSine1
from detection.loss import fastrcnn_loss1, Mix_loss, Focal_loss_l1



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
        keypoint_predictor=None,
         
        # compute attention
        compute_attn= False,
        loss_type= 'fasterrcnn1'):
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
        keypoint_predictor=None,
        )
        
        
        self.crossview = CrossviewTransformer(use_self_attn=use_self_attn, compute_attn=compute_attn)
        self.pos_encode = PositionEmbeddingSine(512)
        self.loss_type =loss_type 
        if self.loss_type == 'fasterrcnn1':
                self.loss_func = fastrcnn_loss1
        elif self.loss_type == 'mix':
            self.loss_func = Mix_loss
        elif self.loss_type == 'focal':
            self.loss_func= Focal_loss_l1
        else:
            print('wrong loss')
            raise Exception()
        
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
        #CC_pos= None 
        #MLO_pos =None
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
            
            if self.loss_type == 'mix':
                box_regression_CC = self.box_coder.decode(box_regression_CC, proposals_CC)
                regression_targets_CC = torch.cat(regression_targets_CC, dim =0)
                regression_targets_CC = self.box_coder.decode(regression_targets_CC, proposals_CC).squeeze(1)
                box_regression_MLO = self.box_coder.decode(box_regression_MLO, proposals_MLO)
                regression_targets_MLO = torch.cat(regression_targets_MLO, dim=0)
                regression_targets_MLO = self.box_coder.decode(regression_targets_MLO, proposals_MLO).squeeze(1)

            
            if torch.any(torch.isnan(class_logits_CC)):
                print('finding nan')
            loss_classifier_CC, loss_box_reg_CC = self.loss_func(class_logits_CC, box_regression_CC, labels_CC, regression_targets_CC)
            loss_classifier_MLO, loss_box_reg_MLO = self.loss_func(class_logits_MLO, box_regression_MLO, labels_MLO, regression_targets_MLO)
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

class Multi_concat_roi_heads(RoIHeads):
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
        keypoint_predictor=None,
        loss_type= 'fasterrcnn1'):
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
        keypoint_predictor=None,
        )
        
        
        self.loss_type =loss_type 
    
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
        feat_CC1 = OrderedDict()
        feat_MLO1 = OrderedDict()
        for key in feat_CC:
            feat_CC1[key] = torch.cat((feat_CC[key], feat_MLO[key]), dim=1)
            feat_MLO1[key] = torch.cat((feat_MLO[key], feat_CC[key]), dim=1)
        box_features_CC = self.box_roi_pool(feat_CC1, proposals_CC, image_shapes_CC)
        box_features_MLO = self.box_roi_pool(feat_MLO1, proposals_MLO, image_shapes_MLO)

        box_features_CC = self.box_head(box_features_CC)
        box_features_MLO =self.box_head(box_features_MLO)
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
            if self.loss_type == 'fasterrcnn1':
                loss_func = fastrcnn_loss1
            elif self.loss_type == 'mix':
                loss_func = Mix_loss
                box_regression_CC = self.box_coder.decode(box_regression_CC, proposals_CC)
                regression_targets_CC = torch.cat(regression_targets_CC, dim =0)
                regression_targets_CC = self.box_coder.decode(regression_targets_CC, proposals_CC).squeeze(1)
                box_regression_MLO = self.box_coder.decode(box_regression_MLO, proposals_MLO)
                regression_targets_MLO = torch.cat(regression_targets_MLO, dim=0)
                regression_targets_MLO = self.box_coder.decode(regression_targets_MLO, proposals_MLO).squeeze(1)
            else:
                print('wrong loss')
                raise Exception()
            
            if torch.any(torch.isnan(class_logits_CC)):
                print('finding nan')
            loss_classifier_CC, loss_box_reg_CC = loss_func(class_logits_CC, box_regression_CC, labels_CC, regression_targets_CC)
            loss_classifier_MLO, loss_box_reg_MLO = loss_func(class_logits_MLO, box_regression_MLO, labels_MLO, regression_targets_MLO)
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